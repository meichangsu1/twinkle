# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os
import socket
import tempfile
import traceback
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
from transformers.utils.import_utils import is_flash_attn_2_available

from twinkle.model.transformers.models.qwen3_5 import modeling_qwen3_5 as tw_qwen35
from twinkle.model.transformers.strategy.sequence_parallel import SequenceParallel, SequenceParallelContext
from twinkle.utils import DeviceMesh

# CUDA_VISIBLE_DEVICES=0,1 \
# CUDA_LAUNCH_BLOCKING=1 \
# QWEN35_TEXTMODEL_MEMORY_BENCH=1 \
# QWEN35_TEXTMODEL_MEMORY_CASES=1x1024 \
# PYTHONPATH=src \
# python -m pytest -q -rs -s \
# tests/sequence_parallel/test_twinkle_qwen3_5_text_model.py::TestTwinkleQwen35TextModel::test_text_model_mixed_attention_memory_benchmark_across_seq_and_batch

def _build_text_config(layer_types=None) -> Qwen3_5TextConfig:
    layer_types = layer_types or ['full_attention']
    return Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        hidden_act='silu',
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        linear_conv_kernel_dim=3,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=layer_types,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def _build_memory_bench_config() -> Qwen3_5TextConfig:
    hidden_size = int(os.environ.get('QWEN35_LINEAR_ATTN_BENCH_HIDDEN_SIZE', '1024'))
    head_dim = int(os.environ.get('QWEN35_LINEAR_ATTN_BENCH_HEAD_DIM', '64'))
    num_attention_heads = hidden_size // head_dim
    return Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=1,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=max(1, num_attention_heads // 2),
        head_dim=head_dim,
        hidden_act='silu',
        max_position_embeddings=16384,
        rms_norm_eps=1e-6,
        linear_conv_kernel_dim=3,
        linear_key_head_dim=head_dim,
        linear_value_head_dim=head_dim,
        linear_num_key_heads=max(2, num_attention_heads // 2),
        linear_num_value_heads=num_attention_heads,
        layer_types=['linear_attention'],
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def _build_mixed_text_model_bench_config() -> Qwen3_5TextConfig:
    hidden_size = int(os.environ.get('QWEN35_TEXTMODEL_BENCH_HIDDEN_SIZE',
                                     os.environ.get('QWEN35_LINEAR_ATTN_BENCH_HIDDEN_SIZE', '1024')))
    head_dim = int(os.environ.get('QWEN35_TEXTMODEL_BENCH_HEAD_DIM',
                                  os.environ.get('QWEN35_LINEAR_ATTN_BENCH_HEAD_DIM', '64')))
    num_attention_heads = hidden_size // head_dim
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=2,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=max(1, num_attention_heads // 2),
        head_dim=head_dim,
        hidden_act='silu',
        max_position_embeddings=16384,
        rms_norm_eps=1e-6,
        linear_conv_kernel_dim=3,
        linear_key_head_dim=head_dim,
        linear_value_head_dim=head_dim,
        linear_num_key_heads=max(2, num_attention_heads // 2),
        linear_num_value_heads=num_attention_heads,
        layer_types=['full_attention', 'linear_attention'],
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    attn_implementation = os.environ.get('QWEN35_TEXTMODEL_BENCH_ATTN_IMPLEMENTATION', 'flash_attention_2')
    config._attn_implementation = attn_implementation
    config._attn_implementation_internal = attn_implementation
    return config


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _parse_memory_bench_cases(env_var='QWEN35_LINEAR_ATTN_MEMORY_CASES', default='1x1024,1x2048,2x2048'):
    spec = os.environ.get(env_var, default)
    cases = []
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        batch_size, seq_len = item.lower().split('x', 1)
        cases.append((int(batch_size), int(seq_len)))
    return cases


def _measure_cuda_peak_stats(run_step):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    run_step()
    torch.cuda.synchronize()
    return {
        'peak_allocated_mib': torch.cuda.max_memory_allocated() / (1024 ** 2),
        'peak_reserved_mib': torch.cuda.max_memory_reserved() / (1024 ** 2),
    }


def _run_linear_attention_memory_step(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    *,
    attention_mask: torch.Tensor,
    cu_seq_lens_q: torch.Tensor,
    sequence_parallel_context: SequenceParallelContext | None = None,
) -> dict[str, float]:

    def _step():
        module.zero_grad(set_to_none=True)
        local_hidden_states = hidden_states.detach().clone().requires_grad_(True)
        output = module(
            hidden_states=local_hidden_states,
            attention_mask=attention_mask,
            cu_seq_lens_q=cu_seq_lens_q,
            sequence_parallel_context=sequence_parallel_context,
        )
        loss = output.float().square().mean()
        loss.backward()

    return _measure_cuda_peak_stats(_step)


def _run_text_model_memory_step(
    model: torch.nn.Module,
    model_inputs: dict[str, torch.Tensor | bool],
) -> dict[str, float]:

    def _step():
        model.zero_grad(set_to_none=True)
        local_inputs = {}
        for key, value in model_inputs.items():
            if torch.is_tensor(value):
                local_inputs[key] = value.detach().clone()
            else:
                local_inputs[key] = value
        outputs = model(**local_inputs)
        loss = outputs.last_hidden_state.float().square().mean()
        loss.backward()

    return _measure_cuda_peak_stats(_step)


def _run_linear_attention_memory_worker(rank: int, world_size: int, port: int, result_path: str, cases):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_WORLD_SIZE'] = str(world_size)
    torch.cuda.set_device(rank)
    error_path = f'{result_path}.rank{rank}.err'
    try:
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=10),
        )

        device = torch.device(f'cuda:{rank}')
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        config = _build_memory_bench_config()
        results = []

        for batch_size, seq_len in cases:
            if seq_len % world_size != 0:
                raise ValueError(f'seq_len ({seq_len}) must be divisible by world_size ({world_size})')

            full_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.int64, device=device)
            full_position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
            cu_seq_lens_q = torch.arange(
                0,
                (batch_size + 1) * seq_len,
                step=seq_len,
                dtype=torch.int32,
                device=device,
            )

            baseline_module = tw_qwen35.TwinkleQwen3_5GatedDeltaNet(config, layer_idx=0).to(device=device, dtype=dtype)
            baseline_module.train()
            baseline_hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device=device, dtype=dtype)
            baseline_stats = _run_linear_attention_memory_step(
                baseline_module,
                baseline_hidden_states,
                attention_mask=full_attention_mask,
                cu_seq_lens_q=cu_seq_lens_q,
                sequence_parallel_context=None,
            )
            del baseline_module, baseline_hidden_states
            torch.cuda.empty_cache()

            local_seq_len = seq_len // world_size
            start = rank * local_seq_len
            end = start + local_seq_len
            sp_attention_mask = full_attention_mask[:, start:end].contiguous()
            sp_hidden_states = torch.randn(batch_size, local_seq_len, config.hidden_size, device=device, dtype=dtype)
            sp_context = SequenceParallelContext(
                sp_group=dist.group.WORLD,
                sp_world_size=world_size,
                rank=rank,
                world_size=world_size,
                real_position_ids=full_position_ids,
                is_packed=False,
            )
            sp_module = tw_qwen35.TwinkleQwen3_5GatedDeltaNet(config, layer_idx=0).to(device=device, dtype=dtype)
            sp_module.train()
            sp_stats = _run_linear_attention_memory_step(
                sp_module,
                sp_hidden_states,
                attention_mask=sp_attention_mask,
                cu_seq_lens_q=cu_seq_lens_q,
                sequence_parallel_context=sp_context,
            )
            del sp_module, sp_hidden_states
            torch.cuda.empty_cache()

            payload = torch.tensor([
                baseline_stats['peak_allocated_mib'],
                baseline_stats['peak_reserved_mib'],
                sp_stats['peak_allocated_mib'],
                sp_stats['peak_reserved_mib'],
            ], device=device)
            gathered = [torch.zeros_like(payload) for _ in range(world_size)]
            dist.all_gather(gathered, payload)

            if rank == 0:
                gathered_cpu = [tensor.cpu().tolist() for tensor in gathered]
                results.append({
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'baseline_peak_allocated_mib_per_rank': [row[0] for row in gathered_cpu],
                    'baseline_peak_reserved_mib_per_rank': [row[1] for row in gathered_cpu],
                    'sp_peak_allocated_mib_per_rank': [row[2] for row in gathered_cpu],
                    'sp_peak_reserved_mib_per_rank': [row[3] for row in gathered_cpu],
                    'baseline_peak_allocated_mib_max': max(row[0] for row in gathered_cpu),
                    'sp_peak_allocated_mib_max': max(row[2] for row in gathered_cpu),
                })

        if rank == 0:
            torch.save(results, result_path)
    except Exception:
        with open(error_path, 'w', encoding='utf-8') as f:
            f.write(traceback.format_exc())
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_mixed_text_model_memory_worker(rank: int, world_size: int, port: int, result_path: str, cases):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_WORLD_SIZE'] = str(world_size)
    torch.cuda.set_device(rank)
    error_path = f'{result_path}.rank{rank}.err'
    try:
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=10),
        )

        device = torch.device(f'cuda:{rank}')
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        baseline_results = []
        sp_results = []

        for batch_size, seq_len in cases:
            config = _build_mixed_text_model_bench_config()
            full_input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)
            full_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.int64, device=device)
            full_position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
            cu_seq_lens_q = torch.arange(
                0,
                (batch_size + 1) * seq_len,
                step=seq_len,
                dtype=torch.int32,
                device=device,
            )

            baseline_model = tw_qwen35.TwinkleQwen3_5TextModel(config).to(device=device, dtype=dtype)
            baseline_model.train()
            baseline_stats = _run_text_model_memory_step(
                baseline_model,
                {
                    'input_ids': full_input_ids,
                    'attention_mask': full_attention_mask,
                    'position_ids': full_position_ids,
                    'use_cache': False,
                },
            )
            baseline_results.append(baseline_stats)
            del baseline_model
            torch.cuda.empty_cache()

        device_mesh = DeviceMesh.from_sizes(
            world_size=world_size,
            dp_size=world_size,
            ulysses_size=world_size,
            device_type='cuda',
        )
        tokenizer = SimpleNamespace(pad_token_id=0)
        sp = SequenceParallel()

        for batch_size, seq_len in cases:
            if seq_len % world_size != 0:
                raise ValueError(f'seq_len ({seq_len}) must be divisible by world_size ({world_size})')

            config = _build_mixed_text_model_bench_config()
            full_input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)
            full_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.int64, device=device)
            full_position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

            sp_model = tw_qwen35.TwinkleQwen3_5TextModel(config).to(device=device, dtype=dtype)
            sp_model.train()
            sp.prepare(world_size, sp_model, tokenizer, device_mesh=device_mesh)
            sp_inputs = sp.prepare_inputs({
                'input_ids': full_input_ids,
                'attention_mask': full_attention_mask,
                'position_ids': full_position_ids,
                'use_cache': False,
            })
            sp_stats = _run_text_model_memory_step(sp_model, sp_inputs)
            sp_results.append(sp_stats)
            del sp_model
            torch.cuda.empty_cache()

        gathered_results = []
        for (batch_size, seq_len), baseline_stats, sp_stats in zip(cases, baseline_results, sp_results, strict=False):
            payload = torch.tensor([
                baseline_stats['peak_allocated_mib'],
                baseline_stats['peak_reserved_mib'],
                sp_stats['peak_allocated_mib'],
                sp_stats['peak_reserved_mib'],
            ], device=device)
            gathered = [torch.zeros_like(payload) for _ in range(world_size)]
            dist.all_gather(gathered, payload)

            if rank == 0:
                gathered_cpu = [tensor.cpu().tolist() for tensor in gathered]
                gathered_results.append({
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'attn_implementation': getattr(config, '_attn_implementation', None),
                    'baseline_peak_allocated_mib_per_rank': [row[0] for row in gathered_cpu],
                    'baseline_peak_reserved_mib_per_rank': [row[1] for row in gathered_cpu],
                    'sp_peak_allocated_mib_per_rank': [row[2] for row in gathered_cpu],
                    'sp_peak_reserved_mib_per_rank': [row[3] for row in gathered_cpu],
                    'baseline_peak_allocated_mib_max': max(row[0] for row in gathered_cpu),
                    'sp_peak_allocated_mib_max': max(row[2] for row in gathered_cpu),
                })

        if rank == 0:
            torch.save(gathered_results, result_path)
    except Exception:
        with open(error_path, 'w', encoding='utf-8') as f:
            f.write(traceback.format_exc())
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class _ContextReceiver:

    def __init__(self):
        self.context = None

    def set_sequence_parallel_context(self, context):
        self.context = context


class TestTwinkleQwen35TextModel(unittest.TestCase):

    def test_rejects_non_text_config(self):
        with self.assertRaises(TypeError):
            tw_qwen35.TwinkleQwen3_5ForCausalLM(Qwen3_5Config())

    def test_text_model_accepts_sequence_parallel_context(self):
        model = tw_qwen35.TwinkleQwen3_5TextModel(_build_text_config(['full_attention']))
        context = SequenceParallelContext(
            sp_group=None,
            sp_world_size=2,
            rank=0,
            world_size=2,
            real_position_ids=torch.tensor([[0, 1, 2]], dtype=torch.long),
            is_packed=False,
        )
        model.set_sequence_parallel_context(context)
        self.assertIs(model._sequence_parallel_context, context)

    def test_from_pretrained_loads_text_only_weights(self):
        config = _build_text_config(['full_attention'])
        hf_model = Qwen3_5ForCausalLM(config).eval()
        with tempfile.TemporaryDirectory() as temp_dir:
            hf_model.save_pretrained(temp_dir)
            tw_model = tw_qwen35.TwinkleQwen3_5ForCausalLM.from_pretrained(temp_dir).eval()

            input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                hf_outputs = hf_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    return_dict=True,
                )
                tw_outputs = tw_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    return_dict=True,
                )

            torch.testing.assert_close(tw_outputs.logits, hf_outputs.logits, rtol=0, atol=0)

    def test_sequence_parallel_prepare_inputs_injects_cu_seq_lens(self):
        sp = SequenceParallel()
        sp.world_size = 2
        sp.sp_world_size = 2
        sp.requires_cu_seq_lens_q = True
        receiver = _ContextReceiver()
        sp._bound_llm_model = receiver
        inputs = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long),
            'position_ids': torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long),
        }

        outputs = sp.prepare_inputs(inputs)

        self.assertIn('cu_seq_lens_q', outputs)
        self.assertTrue(torch.equal(outputs['cu_seq_lens_q'], torch.tensor([0, 5, 6], dtype=torch.int32)))
        self.assertIsNotNone(receiver.context)
        self.assertFalse(receiver.context.is_packed)
        self.assertTrue(torch.equal(receiver.context.real_position_ids, inputs['position_ids']))

    def test_sequence_parallel_prepare_inputs_injects_flattened_cu_seq_lens_for_batched_rows(self):
        sp = SequenceParallel()
        sp.world_size = 2
        sp.sp_world_size = 2
        sp.requires_cu_seq_lens_q = True
        receiver = _ContextReceiver()
        sp._bound_llm_model = receiver
        inputs = {
            'input_ids': torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long),
            'position_ids': torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long),
        }

        outputs = sp.prepare_inputs(inputs)

        self.assertIn('cu_seq_lens_q', outputs)
        self.assertTrue(torch.equal(outputs['cu_seq_lens_q'], torch.tensor([0, 4, 8], dtype=torch.int32)))

    def test_linear_attention_requires_fast_path_dependencies(self):
        with patch.object(tw_qwen35, '_FLA_CAUSAL_CONV1D_FN', None), \
                patch.object(tw_qwen35, '_FLA_CAUSAL_CONV1D_UPDATE', None), \
                patch.object(tw_qwen35, '_FLA_CHUNK_GATED_DELTA_RULE', None), \
                patch.object(tw_qwen35, '_FLA_FUSED_RECURRENT_GATED_DELTA_RULE', None), \
                patch.object(tw_qwen35, '_HAS_CAUSAL_CONV1D', False):
            with self.assertRaises(ImportError):
                tw_qwen35.TwinkleQwen3_5TextModel(_build_text_config(['linear_attention']))

    def test_linear_attention_sp_uses_cu_seq_lens_and_keeps_z_local(self):
        captured = {
            'cu_seqlens': None,
            'seq_to_head_calls': 0,
            'head_to_seq_calls': 0,
            'norm_z_shape': None,
        }

        def fake_conv(x, weight, bias, activation, seq_idx=None, backend=None, cu_seqlens=None):
            del weight, bias, activation, seq_idx, backend
            captured['cu_seqlens'] = cu_seqlens.clone() if cu_seqlens is not None else None
            return x

        def fake_chunk_rule(query, key, value, g, beta, initial_state=None, output_final_state=False,
                            use_qk_l2norm_in_kernel=False, cu_seqlens=None):
            del query, key, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel
            captured['cu_seqlens'] = cu_seqlens.clone() if cu_seqlens is not None else None
            return value, None

        def fake_recurrent_rule(query, key, value, g, beta, initial_state=None, output_final_state=False,
                                use_qk_l2norm_in_kernel=False):
            del query, key, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel
            return value, None

        def fake_seq_to_head(tensor, context):
            sp_world_size = context.sp_world_size
            rank = context.rank
            captured['seq_to_head_calls'] += 1
            if tensor.dim() == 4:
                local_heads = tensor.shape[2] // sp_world_size
                start = rank * local_heads
                end = start + local_heads
                return tensor[:, :, start:end, :].contiguous()
            if tensor.dim() == 3:
                local_heads = tensor.shape[2] // sp_world_size
                start = rank * local_heads
                end = start + local_heads
                return tensor[:, :, start:end].contiguous()
            return tensor

        def fake_head_to_seq(tensor, context):
            captured['head_to_seq_calls'] += 1
            return tensor.repeat_interleave(context.sp_world_size, dim=2)

        class DummyNorm(torch.nn.Module):

            def forward(self, x, z):
                captured['norm_z_shape'] = tuple(z.shape)
                return x + z

        with patch.object(tw_qwen35, '_FLA_CAUSAL_CONV1D_FN', fake_conv), \
                patch.object(tw_qwen35, '_FLA_CAUSAL_CONV1D_UPDATE', lambda *args, **kwargs: args[0]), \
                patch.object(tw_qwen35, '_FLA_CHUNK_GATED_DELTA_RULE', fake_chunk_rule), \
                patch.object(tw_qwen35, '_FLA_FUSED_RECURRENT_GATED_DELTA_RULE', fake_recurrent_rule), \
                patch.object(tw_qwen35, '_FLA_FUSED_RMS_NORM_GATED', None), \
                patch.object(tw_qwen35, '_HAS_CAUSAL_CONV1D', True), \
                patch.object(tw_qwen35, '_seq_to_head_shard', side_effect=fake_seq_to_head), \
                patch.object(tw_qwen35, '_head_to_seq_shard', side_effect=fake_head_to_seq):
            config = _build_text_config(['linear_attention'])
            module = tw_qwen35.TwinkleQwen3_5GatedDeltaNet(config, layer_idx=0)
            module.norm = DummyNorm()
            hidden_states = torch.randn(1, 2, config.hidden_size)
            attention_mask = torch.ones(1, 2, dtype=torch.int64)
            cu_seq_lens_q = torch.tensor([0, 2], dtype=torch.int32)
            context = SequenceParallelContext(
                sp_group='dummy_group',
                sp_world_size=2,
                rank=0,
                world_size=2,
                real_position_ids=torch.tensor([[0, 1]], dtype=torch.long),
                is_packed=False,
            )

            output = module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                cu_seq_lens_q=cu_seq_lens_q,
                sequence_parallel_context=context,
            )

        self.assertEqual(captured['seq_to_head_calls'], 5)
        self.assertEqual(captured['head_to_seq_calls'], 1)
        self.assertTrue(torch.equal(captured['cu_seqlens'], cu_seq_lens_q))
        self.assertEqual(captured['norm_z_shape'], (hidden_states.shape[0] * hidden_states.shape[1] * config.linear_num_value_heads, config.linear_value_head_dim))
        self.assertEqual(tuple(output.shape), (1, 2, config.hidden_size))

    def test_linear_attention_sp_flattens_batched_varlen_inputs(self):
        captured = {
            'query_shape': None,
            'cu_seqlens': None,
        }

        def fake_conv(x, weight, bias, activation, seq_idx=None, backend=None, cu_seqlens=None):
            del weight, bias, activation, seq_idx, backend
            captured['cu_seqlens'] = cu_seqlens.clone() if cu_seqlens is not None else None
            return x

        def fake_chunk_rule(query, key, value, g, beta, initial_state=None, output_final_state=False,
                            use_qk_l2norm_in_kernel=False, cu_seqlens=None):
            del key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel
            captured['query_shape'] = tuple(query.shape)
            captured['cu_seqlens'] = cu_seqlens.clone() if cu_seqlens is not None else None
            return query.new_zeros(query.shape[0], query.shape[1], 4, 4), None

        def fake_recurrent_rule(query, key, value, g, beta, initial_state=None, output_final_state=False,
                                use_qk_l2norm_in_kernel=False):
            del query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel
            raise AssertionError('recurrent path should not be used')

        class DummyNorm(torch.nn.Module):

            def forward(self, x, z):
                return x + z

        with patch.object(tw_qwen35, '_FLA_CAUSAL_CONV1D_FN', fake_conv), \
                patch.object(tw_qwen35, '_FLA_CAUSAL_CONV1D_UPDATE', lambda *args, **kwargs: args[0]), \
                patch.object(tw_qwen35, '_FLA_CHUNK_GATED_DELTA_RULE', fake_chunk_rule), \
                patch.object(tw_qwen35, '_FLA_FUSED_RECURRENT_GATED_DELTA_RULE', fake_recurrent_rule), \
                patch.object(tw_qwen35, '_FLA_FUSED_RMS_NORM_GATED', None), \
                patch.object(tw_qwen35, '_HAS_CAUSAL_CONV1D', True):
            config = _build_text_config(['linear_attention'])
            module = tw_qwen35.TwinkleQwen3_5GatedDeltaNet(config, layer_idx=0)
            module.norm = DummyNorm()
            hidden_states = torch.randn(2, 3, config.hidden_size)
            attention_mask = torch.ones(2, 3, dtype=torch.int64)
            cu_seq_lens_q = torch.tensor([0, 3, 6], dtype=torch.int32)

            _ = module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                cu_seq_lens_q=cu_seq_lens_q,
            )

        self.assertEqual(captured['query_shape'], (1, 6, 4, 4))
        self.assertTrue(torch.equal(captured['cu_seqlens'], cu_seq_lens_q))

    def test_linear_attention_sp_uses_local_attention_mask(self):
        captured = {'mask': None}

        def fake_conv(x, weight, bias, activation, seq_idx=None, backend=None, cu_seqlens=None):
            del weight, bias, activation, seq_idx, backend, cu_seqlens
            return x

        def fake_chunk_rule(query, key, value, g, beta, initial_state=None, output_final_state=False,
                            use_qk_l2norm_in_kernel=False, cu_seqlens=None):
            del query, key, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel, cu_seqlens
            return value, None

        def fake_recurrent_rule(query, key, value, g, beta, initial_state=None, output_final_state=False,
                                use_qk_l2norm_in_kernel=False):
            del query, key, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel
            return value, None

        with patch.object(tw_qwen35, '_FLA_CAUSAL_CONV1D_FN', fake_conv), \
                patch.object(tw_qwen35, '_FLA_CAUSAL_CONV1D_UPDATE', lambda *args, **kwargs: args[0]), \
                patch.object(tw_qwen35, '_FLA_CHUNK_GATED_DELTA_RULE', fake_chunk_rule), \
                patch.object(tw_qwen35, '_FLA_FUSED_RECURRENT_GATED_DELTA_RULE', fake_recurrent_rule), \
                patch.object(tw_qwen35, '_HAS_CAUSAL_CONV1D', True):
            config = _build_text_config(['linear_attention'])
            model = tw_qwen35.TwinkleQwen3_5TextModel(config)
            model.set_sequence_parallel_context(
                SequenceParallelContext(
                    sp_group='dummy_group',
                    sp_world_size=2,
                    rank=0,
                    world_size=2,
                    real_position_ids=torch.tensor([[0, 1, 2, -1]], dtype=torch.long),
                    is_packed=False,
                ))

            def fake_linear_forward(hidden_states, cache_params=None, cache_position=None, attention_mask=None,
                                    cu_seq_lens_q=None, sequence_parallel_context=None):
                del hidden_states, cache_params, cache_position, cu_seq_lens_q, sequence_parallel_context
                captured['mask'] = attention_mask.clone() if attention_mask is not None else None
                return torch.zeros(1, 2, config.hidden_size)

            with patch.object(model.layers[0].linear_attn, 'forward', side_effect=fake_linear_forward):
                _ = model(
                    input_ids=torch.tensor([[1, 2]], dtype=torch.long),
                    attention_mask=torch.tensor([[1, 1, 1, 0]], dtype=torch.int64),
                    position_ids=torch.tensor([[0, -1]], dtype=torch.long),
                    cache_position=torch.tensor([0, 1], dtype=torch.long),
                    cu_seq_lens_q=torch.tensor([0, 2], dtype=torch.int32),
                    use_cache=False,
                )

        self.assertIsNotNone(captured['mask'])
        self.assertTrue(torch.equal(captured['mask'], torch.tensor([[1, 0]], dtype=torch.int64)))

    def test_linear_attention_memory_benchmark_across_seq_and_batch(self):
        if os.environ.get('QWEN35_LINEAR_ATTN_MEMORY_BENCH') != '1':
            self.skipTest('Set QWEN35_LINEAR_ATTN_MEMORY_BENCH=1 to run the CUDA memory benchmark.')
        if not torch.cuda.is_available():
            self.skipTest('CUDA is required for the linear attention memory benchmark.')

        world_size = int(os.environ.get('QWEN35_LINEAR_ATTN_MEMORY_WORLD_SIZE', '2'))
        if torch.cuda.device_count() < world_size:
            self.skipTest(f'Need at least {world_size} CUDA devices for the linear attention memory benchmark.')

        cases = _parse_memory_bench_cases()
        port = _find_free_port()

        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = os.path.join(temp_dir, 'linear_attn_memory_results.pt')
            mp.spawn(
                _run_linear_attention_memory_worker,
                args=(world_size, port, result_path, cases),
                nprocs=world_size,
                join=True,
            )

            error_logs = []
            for rank in range(world_size):
                error_path = f'{result_path}.rank{rank}.err'
                if os.path.exists(error_path):
                    with open(error_path, 'r', encoding='utf-8') as f:
                        error_logs.append(f'Rank {rank}:\n{f.read()}')
            if error_logs:
                self.fail('\n\n'.join(error_logs))

            results = torch.load(result_path, weights_only=False)

        self.assertEqual(len(results), len(cases))
        for result in results:
            self.assertGreater(result['baseline_peak_allocated_mib_max'], 0.0)
            self.assertGreater(result['sp_peak_allocated_mib_max'], 0.0)

        baseline_peaks = [result['baseline_peak_allocated_mib_max'] for result in results]
        sp_peaks = [result['sp_peak_allocated_mib_max'] for result in results]
        for prev, curr in zip(baseline_peaks, baseline_peaks[1:], strict=False):
            self.assertGreaterEqual(curr + 64.0, prev)
        for prev, curr in zip(sp_peaks, sp_peaks[1:], strict=False):
            self.assertGreaterEqual(curr + 64.0, prev)

        print('Linear attention memory benchmark results:')
        print(json.dumps(results, indent=2))

    def test_text_model_mixed_attention_memory_benchmark_across_seq_and_batch(self):
        if os.environ.get('QWEN35_TEXTMODEL_MEMORY_BENCH') != '1':
            self.skipTest('Set QWEN35_TEXTMODEL_MEMORY_BENCH=1 to run the mixed TextModel CUDA memory benchmark.')
        if not torch.cuda.is_available():
            self.skipTest('CUDA is required for the mixed TextModel memory benchmark.')
        if not is_flash_attn_2_available():
            self.skipTest('flash_attention_2 is required for the mixed TextModel memory benchmark.')

        world_size = int(os.environ.get('QWEN35_TEXTMODEL_MEMORY_WORLD_SIZE', '2'))
        if torch.cuda.device_count() < world_size:
            self.skipTest(f'Need at least {world_size} CUDA devices for the mixed TextModel memory benchmark.')

        cases = _parse_memory_bench_cases(
            env_var='QWEN35_TEXTMODEL_MEMORY_CASES',
            default='1x1024,1x2048,2x2048',
        )
        port = _find_free_port()

        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = os.path.join(temp_dir, 'text_model_memory_results.pt')
            mp.spawn(
                _run_mixed_text_model_memory_worker,
                args=(world_size, port, result_path, cases),
                nprocs=world_size,
                join=True,
            )

            error_logs = []
            for rank in range(world_size):
                error_path = f'{result_path}.rank{rank}.err'
                if os.path.exists(error_path):
                    with open(error_path, 'r', encoding='utf-8') as f:
                        error_logs.append(f'Rank {rank}:\n{f.read()}')
            if error_logs:
                self.fail('\n\n'.join(error_logs))

            results = torch.load(result_path, weights_only=False)

        self.assertEqual(len(results), len(cases))
        for result in results:
            self.assertGreater(result['baseline_peak_allocated_mib_max'], 0.0)
            self.assertGreater(result['sp_peak_allocated_mib_max'], 0.0)

        print('Mixed TextModel memory benchmark results:')
        print(json.dumps(results, indent=2))


if __name__ == '__main__':
    unittest.main()
