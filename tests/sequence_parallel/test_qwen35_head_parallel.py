# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import json
import os
import socket
import sys
import types
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# QWEN35_HEAD_PARALLEL_TEST_WORLD_SIZE=4 \
# PYTHONPATH=src \
# torchrun --standalone --nproc_per_node=4 \
# -m unittest tests.sequence_parallel.test_qwen35_head_parallel.TestQwen35HeadParallel.test_head_parallel_forward_is_deterministic

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# QWEN35_HEAD_PARALLEL_TEST_WORLD_SIZE=4 \
# QWEN35_SP_MEMORY_BENCH=1 \
# QWEN35_SP_MEMORY_BATCH=1 \
# QWEN35_SP_MEMORY_SEQ_LEN=4096 \
# PYTHONPATH=src \
# torchrun --standalone --nproc_per_node=4 \
# -m unittest tests.sequence_parallel.test_qwen35_head_parallel.TestQwen35HeadParallel.test_head_parallel_peak_memory_compare

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# QWEN35_SP_PARITY=1 \
# QWEN35_MODEL_ID=/your/local/qwen3.5/path \
# QWEN35_SP_PARITY_SEQ_LEN=4096 \
# PYTHONPATH=src \
# torchrun --standalone --nproc_per_node=4 \
# -m unittest tests.sequence_parallel.test_qwen35_sp_parity

from twinkle.patch.qwen35_linear_attention_sp import (
    _head_to_seq_shard,
    _interleave_qkv_value_head_params,
    _run_qwen35_head_parallel,
    _select_rule,
    _seq_to_head_shard,
)
from twinkle.patch import Qwen35LinearAttentionSPPatch


def apply_mask_to_padding_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return hidden_states
    return hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)


class _RepeatQKV(torch.nn.Module):

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.cat([hidden_states, hidden_states, hidden_states], dim=-1)


class _ZeroProj(torch.nn.Module):

    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = (*hidden_states.shape[:-1], self.out_dim)
        return hidden_states.new_zeros(shape)


class _IdentityNorm(torch.nn.Module):

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return x + z


class _IdentityOutProj(torch.nn.Module):

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class _DummyHeadParallelModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.num_k_heads = 1
        self.num_v_heads = 1
        self.head_k_dim = 1
        self.head_v_dim = 1
        self.key_dim = 1
        self.value_dim = 1
        self.conv_kernel_size = 3
        self.activation = 'identity'
        self.act = lambda tensor: tensor
        self.in_proj_qkv = _RepeatQKV()
        self.in_proj_z = torch.nn.Identity()
        self.in_proj_b = _ZeroProj(1)
        self.in_proj_a = _ZeroProj(1)
        self.conv1d = torch.nn.Conv1d(3, 3, kernel_size=3, groups=3, bias=False)
        with torch.no_grad():
            self.conv1d.weight.zero_()
            self.conv1d.weight[:, :, 0] = 1.0
        self.dt_bias = torch.nn.Parameter(torch.zeros(1))
        self.A_log = torch.nn.Parameter(torch.zeros(1))
        self.norm = _IdentityNorm()
        self.out_proj = _IdentityOutProj()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _build_actual_module():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as module_impl

    module_impl.FusedRMSNormGated = None
    config = Qwen3_5TextConfig(
        hidden_size=16,
        num_hidden_layers=1,
        layer_types=['linear_attention'],
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_conv_kernel_dim=3,
        hidden_act='silu',
        rms_norm_eps=1e-6,
    )
    module = module_impl.Qwen3_5GatedDeltaNet(config, layer_idx=0)
    module.causal_conv1d_fn = None
    module.causal_conv1d_update = module_impl.torch_causal_conv1d_update
    module.chunk_gated_delta_rule = module_impl.torch_chunk_gated_delta_rule
    module.recurrent_gated_delta_rule = module_impl.torch_recurrent_gated_delta_rule
    return module


def _set_deterministic(seed: int = 42) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _assert_forward_deterministic(fn, repeats: int = 10) -> None:
    with torch.no_grad():
        reference = fn().detach()
        for _ in range(repeats - 1):
            output = fn().detach()
            torch.testing.assert_close(output, reference, rtol=0, atol=0)


def _measure_cuda_peak_bytes(fn, device: torch.device) -> int:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    memory_before = torch.cuda.memory_allocated(device)
    fn()
    torch.cuda.synchronize(device)
    return int(torch.cuda.max_memory_allocated(device) - memory_before)


def _run_full_train_step(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> None:
    module.zero_grad(set_to_none=True)
    output = module(hidden_states, attention_mask=attention_mask)
    output.mean().backward()


def _run_sp_train_step(
    module: torch.nn.Module,
    sequence_parallel,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> None:
    module.zero_grad(set_to_none=True)
    output = _run_qwen35_head_parallel(
        sequence_parallel=sequence_parallel,
        module=module,
        origin_rule=module.chunk_gated_delta_rule,
        origin_causal_conv1d_fn=module.causal_conv1d_fn,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
    )
    output.mean().backward()


def _run_roundtrip_worker(rank: int, world_size: int, port: int):
    dist.init_process_group(
        backend='gloo',
        init_method=f'tcp://127.0.0.1:{port}',
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
    try:
        sequence_parallel = SimpleNamespace(world_size=world_size, sp_world_size=world_size, _sp_group=dist.group.WORLD)
        local_tensor = torch.randn(2, 3, 4, 5, dtype=torch.float32, requires_grad=True)
        roundtrip = _head_to_seq_shard(_seq_to_head_shard(local_tensor, sequence_parallel), sequence_parallel)
        assert torch.allclose(roundtrip.detach(), local_tensor.detach(), atol=1e-6, rtol=1e-6)

        loss = roundtrip.pow(2).mean()
        loss.backward()
        expected_grad = 2.0 * local_tensor.detach() / local_tensor.numel()
        assert torch.allclose(local_tensor.grad, expected_grad, atol=1e-6, rtol=1e-6), (
            f'rank={rank}: roundtrip grad mismatch, max_diff='
            f'{(local_tensor.grad - expected_grad).abs().max().item()}'
        )
    finally:
        dist.destroy_process_group()


def _run_head_parallel_parity_worker(rank: int, world_size: int, port: int):
    dist.init_process_group(
        backend='gloo',
        init_method=f'tcp://127.0.0.1:{port}',
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
    try:
        torch.manual_seed(20260319)
        module = _build_actual_module()
        reference_module = copy.deepcopy(module)
        module.train()
        reference_module.train()

        batch = 2
        seq_len = 8
        hidden_size = module.hidden_size
        local_seq = seq_len // world_size
        seq_start = rank * local_seq
        seq_end = seq_start + local_seq

        full_hidden = torch.randn(batch, seq_len, hidden_size, dtype=torch.float32, requires_grad=True)
        full_attention_mask = torch.ones(batch, seq_len, dtype=torch.int64)
        full_attention_mask[:, -1] = 0

        reference_output = reference_module(full_hidden, attention_mask=full_attention_mask)
        reference_loss = torch.stack([
            reference_output[:, i * local_seq:(i + 1) * local_seq].pow(2).mean() for i in range(world_size)
        ]).sum()
        reference_loss.backward()

        local_hidden = full_hidden.detach()[:, seq_start:seq_end].contiguous().clone().requires_grad_(True)
        local_attention_mask = full_attention_mask[:, seq_start:seq_end].contiguous().clone()
        sequence_parallel = SimpleNamespace(world_size=world_size, sp_world_size=world_size, _sp_group=dist.group.WORLD)
        local_output = _run_qwen35_head_parallel(
            sequence_parallel=sequence_parallel,
            module=module,
            origin_rule=module.chunk_gated_delta_rule,
            origin_causal_conv1d_fn=module.causal_conv1d_fn,
            hidden_states=local_hidden,
            attention_mask=local_attention_mask,
        )
        local_loss = local_output.pow(2).mean()
        local_loss.backward()

        for param in module.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, group=dist.group.WORLD)

        expected_output = reference_output.detach()[:, seq_start:seq_end]
        expected_hidden_grad = full_hidden.grad.detach()[:, seq_start:seq_end]
        assert torch.allclose(local_output.detach(), expected_output, atol=1e-5, rtol=1e-5), (
            f'rank={rank}: output mismatch, max_diff={((local_output.detach() - expected_output).abs().max().item())}'
        )
        assert torch.allclose(local_hidden.grad, expected_hidden_grad, atol=1e-5, rtol=1e-5), (
            f'rank={rank}: hidden grad mismatch, max_diff={((local_hidden.grad - expected_hidden_grad).abs().max().item())}'
        )

        reference_params = dict(reference_module.named_parameters())
        test_params = dict(module.named_parameters())
        for name in (
            'in_proj_qkv.weight',
            'in_proj_z.weight',
            'in_proj_a.weight',
            'dt_bias',
            'A_log',
            'conv1d.weight',
            'out_proj.weight',
        ):
            assert torch.allclose(test_params[name].grad, reference_params[name].grad, atol=1e-5, rtol=1e-5), (
                f'rank={rank}: param grad mismatch for {name}, max_diff='
                f'{((test_params[name].grad - reference_params[name].grad).abs().max().item())}'
            )
    finally:
        dist.destroy_process_group()


def _run_head_parallel_determinism_worker(rank: int, world_size: int, port: int):
    dist.init_process_group(
        backend='gloo',
        init_method=f'tcp://127.0.0.1:{port}',
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
    try:
        _set_deterministic(20260320)
        module = _build_actual_module()
        batch = 2
        seq_len = 8
        hidden_size = module.hidden_size
        local_seq = seq_len // world_size
        seq_start = rank * local_seq
        seq_end = seq_start + local_seq

        full_hidden = torch.randn(batch, seq_len, hidden_size, dtype=torch.float32)
        full_attention_mask = torch.ones(batch, seq_len, dtype=torch.int64)
        local_hidden = full_hidden[:, seq_start:seq_end].contiguous()
        local_attention_mask = full_attention_mask[:, seq_start:seq_end].contiguous()
        sequence_parallel = SimpleNamespace(world_size=world_size, sp_world_size=world_size, _sp_group=dist.group.WORLD)

        _assert_forward_deterministic(
            lambda: _run_qwen35_head_parallel(
                sequence_parallel=sequence_parallel,
                module=module,
                origin_rule=module.chunk_gated_delta_rule,
                origin_causal_conv1d_fn=module.causal_conv1d_fn,
                hidden_states=local_hidden,
                attention_mask=local_attention_mask,
            ),
            repeats=20,
        )
    finally:
        dist.destroy_process_group()


def _run_head_parallel_memory_compare_worker(rank: int, world_size: int, port: int, batch: int, seq_len: int):
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{port}',
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=300),
    )
    try:
        torch.cuda.set_device(rank)
        device = torch.device('cuda', rank)
        _set_deterministic(20260321)

        module = _build_actual_module().to(device)
        module.train()

        hidden_size = module.hidden_size
        if rank == 0:
            full_hidden = torch.randn(batch, seq_len, hidden_size, device=device, dtype=torch.float32)
        else:
            full_hidden = torch.empty(batch, seq_len, hidden_size, device=device, dtype=torch.float32)
        dist.broadcast(full_hidden, src=0)

        full_attention_mask = torch.ones(batch, seq_len, device=device, dtype=torch.int64)
        local_seq = seq_len // world_size
        seq_start = rank * local_seq
        seq_end = seq_start + local_seq
        local_hidden = full_hidden[:, seq_start:seq_end].contiguous()
        local_attention_mask = full_attention_mask[:, seq_start:seq_end].contiguous()
        sequence_parallel = SimpleNamespace(world_size=world_size, sp_world_size=world_size, _sp_group=dist.group.WORLD)

        if rank == 0:
            _run_full_train_step(module, full_hidden, full_attention_mask)
        dist.barrier()

        _run_sp_train_step(module, sequence_parallel, local_hidden, local_attention_mask)
        module.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        dist.barrier()

        baseline_peak = torch.zeros(1, device=device, dtype=torch.int64)
        if rank == 0:
            baseline_peak[0] = _measure_cuda_peak_bytes(
                lambda: _run_full_train_step(module, full_hidden, full_attention_mask),
                device,
            )
            module.zero_grad(set_to_none=True)

        dist.barrier()

        sp_peak = torch.tensor([
            _measure_cuda_peak_bytes(
                lambda: _run_sp_train_step(module, sequence_parallel, local_hidden, local_attention_mask),
                device,
            )
        ], device=device, dtype=torch.int64)
        module.zero_grad(set_to_none=True)

        gathered_sp_peaks = [torch.empty_like(sp_peak) for _ in range(world_size)]
        dist.all_gather(gathered_sp_peaks, sp_peak)

        if rank == 0:
            baseline_peak_bytes = int(baseline_peak.item())
            sp_peak_bytes = [int(item.item()) for item in gathered_sp_peaks]
            max_sp_peak_bytes = max(sp_peak_bytes)
            memory_saving_ratio = 0.0 if baseline_peak_bytes == 0 else 1.0 - (max_sp_peak_bytes / baseline_peak_bytes)
            print(json.dumps({
                'benchmark': 'qwen35_head_parallel_memory',
                'batch': batch,
                'seq_len': seq_len,
                'world_size': world_size,
                'baseline_peak_mib': round(baseline_peak_bytes / (1024**2), 3),
                'sp_peak_mib_per_rank': [round(value / (1024**2), 3) for value in sp_peak_bytes],
                'sp_peak_mib_max': round(max_sp_peak_bytes / (1024**2), 3),
                'memory_saving_ratio': round(memory_saving_ratio, 6),
            }), flush=True)
            assert baseline_peak_bytes > 0
            assert all(value > 0 for value in sp_peak_bytes)
    finally:
        dist.destroy_process_group()


def _build_fake_qwen35_transformers_modules():
    fake_transformers = types.ModuleType('transformers')
    fake_models = types.ModuleType('transformers.models')
    fake_qwen35 = types.ModuleType('transformers.models.qwen3_5')
    fake_modeling = types.ModuleType('transformers.models.qwen3_5.modeling_qwen3_5')

    class BaseGatedDeltaNet(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.layer_idx = 0
            self.chunk_gated_delta_rule = lambda *args, **kwargs: (None, None)
            self.recurrent_gated_delta_rule = None
            self.causal_conv1d_fn = None

        def forward(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None, **kwargs):
            del cache_params, cache_position, attention_mask, kwargs
            return hidden_states + 1

    class FakeModel(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = BaseGatedDeltaNet()
            self.other = torch.nn.Linear(4, 4)

    fake_modeling.Qwen3_5GatedDeltaNet = BaseGatedDeltaNet
    fake_qwen35.Qwen3_5GatedDeltaNet = BaseGatedDeltaNet
    fake_qwen35.modeling_qwen3_5 = fake_modeling
    fake_models.qwen3_5 = fake_qwen35
    fake_transformers.models = fake_models

    patched_modules = {
        'transformers': fake_transformers,
        'transformers.models': fake_models,
        'transformers.models.qwen3_5': fake_qwen35,
        'transformers.models.qwen3_5.modeling_qwen3_5': fake_modeling,
    }
    return patched_modules, BaseGatedDeltaNet, FakeModel


class TestQwen35HeadParallel(unittest.TestCase):

    def test_head_parallel_selects_recurrent_rule_for_misaligned_sequence(self):
        class DummyModule:
            recurrent_gated_delta_rule = staticmethod(lambda *args, **kwargs: None)

        dummy_module = DummyModule()
        rule, chunk_size = _select_rule(dummy_module, object(), full_seq_len=3)
        self.assertIsNone(chunk_size)
        self.assertTrue(callable(rule))

    def test_head_parallel_roundtrip_collectives(self):
        if not dist.is_available():
            self.skipTest('torch.distributed is not available')
        world_size = 2
        port = _find_free_port()
        mp.spawn(
            _run_roundtrip_worker,
            args=(world_size, port),
            nprocs=world_size,
            join=True,
        )

    def test_interleave_qkv_value_head_params_matches_mixed_qkv_layout(self):
        query = torch.tensor([10, 11, 20, 21, 30, 31, 40, 41], dtype=torch.float32)
        key = torch.tensor([50, 51, 60, 61, 70, 71, 80, 81], dtype=torch.float32)
        value = torch.tensor([90, 91, 92, 100, 101, 102, 110, 111, 112, 120, 121, 122], dtype=torch.float32)

        interleaved = _interleave_qkv_value_head_params(query, key, value, local_v_heads=4)
        expected = torch.tensor([
            10, 11, 50, 51, 90, 91, 92,
            20, 21, 60, 61, 100, 101, 102,
            30, 31, 70, 71, 110, 111, 112,
            40, 41, 80, 81, 120, 121, 122,
        ], dtype=torch.float32)

        self.assertTrue(torch.equal(interleaved, expected))

    def test_head_parallel_matches_single_layer_reference(self):
        if not dist.is_available():
            self.skipTest('torch.distributed is not available')
        world_size = 2
        port = _find_free_port()
        mp.spawn(
            _run_head_parallel_parity_worker,
            args=(world_size, port),
            nprocs=world_size,
            join=True,
        )

    def test_head_parallel_forward_is_deterministic(self):
        if not dist.is_available():
            self.skipTest('torch.distributed is not available')
        world_size = int(os.environ.get('QWEN35_HEAD_PARALLEL_TEST_WORLD_SIZE', '2'))
        if world_size < 1:
            self.skipTest(f'Invalid QWEN35_HEAD_PARALLEL_TEST_WORLD_SIZE={world_size}.')
        port = _find_free_port()
        mp.spawn(
            _run_head_parallel_determinism_worker,
            args=(world_size, port),
            nprocs=world_size,
            join=True,
        )

    def test_head_parallel_peak_memory_compare(self):
        if os.environ.get('QWEN35_SP_MEMORY_BENCH', '0') != '1':
            self.skipTest('Set QWEN35_SP_MEMORY_BENCH=1 to enable the peak-memory comparison benchmark.')
        if not torch.cuda.is_available():
            self.skipTest('CUDA is required for peak-memory comparison.')
        world_size = int(os.environ.get('QWEN35_HEAD_PARALLEL_TEST_WORLD_SIZE', '2'))
        if world_size < 1:
            self.skipTest(f'Invalid QWEN35_HEAD_PARALLEL_TEST_WORLD_SIZE={world_size}.')
        if torch.cuda.device_count() < world_size:
            self.skipTest(f'Need at least {world_size} CUDA devices.')

        batch = int(os.environ.get('QWEN35_SP_MEMORY_BATCH', '1'))
        seq_len = int(os.environ.get('QWEN35_SP_MEMORY_SEQ_LEN', '2048'))
        if seq_len % world_size != 0:
            self.skipTest(f'seq_len ({seq_len}) must be divisible by world_size ({world_size}).')

        port = _find_free_port()
        mp.spawn(
            _run_head_parallel_memory_compare_worker,
            args=(world_size, port, batch, seq_len),
            nprocs=world_size,
            join=True,
        )

    def test_local_patch_patches_only_qwen35_linear_layers(self):
        patched_modules, _, FakeModel = _build_fake_qwen35_transformers_modules()
        model = FakeModel()

        with patch.dict(sys.modules, patched_modules, clear=False), patch('twinkle.requires', return_value=None):
            patch_count = Qwen35LinearAttentionSPPatch()(model)

        self.assertEqual(patch_count, 1)
        self.assertTrue(getattr(model, '_twinkle_qwen35_linear_attention_sp_patch_applied', False))
        self.assertEqual(getattr(model, '_twinkle_qwen35_linear_attention_sp_patch_count', None), 1)
        self.assertTrue(getattr(model.linear, '_twinkle_qwen35_linear_attention_sp_patch', False))
        self.assertTrue(hasattr(model.linear, '_twinkle_origin_forward'))
        self.assertTrue(callable(getattr(model.linear, 'twinkle_bind_sequence_parallel', None)))
        self.assertTrue(callable(getattr(model.linear, 'twinkle_unbind_sequence_parallel', None)))
        self.assertFalse(hasattr(model.other, '_twinkle_qwen35_linear_attention_sp_patch'))

    def test_local_patch_is_idempotent(self):
        patched_modules, _, FakeModel = _build_fake_qwen35_transformers_modules()
        model = FakeModel()

        with patch.dict(sys.modules, patched_modules, clear=False), patch('twinkle.requires', return_value=None):
            Qwen35LinearAttentionSPPatch()(model)
            origin_forward = model.linear._twinkle_origin_forward
            second_result = Qwen35LinearAttentionSPPatch()(model)

        self.assertIsNone(second_result)
        self.assertIs(model.linear._twinkle_origin_forward, origin_forward)

    def test_patched_forward_falls_back_to_origin_without_sp_binding(self):
        patched_modules, _, FakeModel = _build_fake_qwen35_transformers_modules()
        model = FakeModel()
        hidden_states = torch.randn(2, 3, 4)

        with patch.dict(sys.modules, patched_modules, clear=False), patch('twinkle.requires', return_value=None):
            Qwen35LinearAttentionSPPatch()(model)
            output = model.linear.forward(hidden_states, cu_seq_lens_q=torch.tensor([0, 3]))

        self.assertTrue(torch.equal(output, hidden_states + 1))

    def test_patched_forward_falls_back_to_origin_for_cache_path(self):
        patched_modules, _, FakeModel = _build_fake_qwen35_transformers_modules()
        model = FakeModel()
        hidden_states = torch.randn(2, 3, 4)
        cache_params = SimpleNamespace(has_previous_state=True)
        sequence_parallel = SimpleNamespace(sp_world_size=2, _sp_group='sp_group')

        with patch.dict(sys.modules, patched_modules, clear=False), patch('twinkle.requires', return_value=None):
            Qwen35LinearAttentionSPPatch()(model)
            model.linear.twinkle_bind_sequence_parallel(sequence_parallel)
            with patch(
                'twinkle.patch.qwen35_linear_attention_sp._run_qwen35_head_parallel',
                side_effect=AssertionError('head-parallel helper should not run on cache path'),
            ):
                output = model.linear.forward(
                    hidden_states,
                    cache_params=cache_params,
                    cache_position=torch.tensor([0]),
                )

        self.assertTrue(torch.equal(output, hidden_states + 1))

    def test_patched_forward_runs_head_parallel_when_sp_is_bound(self):
        patched_modules, _, FakeModel = _build_fake_qwen35_transformers_modules()
        model = FakeModel()
        hidden_states = torch.randn(2, 3, 4)
        attention_mask = torch.ones(2, 3, dtype=torch.int64)
        sequence_parallel = SimpleNamespace(sp_world_size=2, _sp_group='sp_group')
        sentinel = torch.randn(2, 3, 4)

        with patch.dict(sys.modules, patched_modules, clear=False), patch('twinkle.requires', return_value=None):
            Qwen35LinearAttentionSPPatch()(model)
            model.linear.twinkle_bind_sequence_parallel(sequence_parallel)
            with patch('twinkle.patch.qwen35_linear_attention_sp._run_qwen35_head_parallel', return_value=sentinel) as run_mock:
                output = model.linear.forward(hidden_states, attention_mask=attention_mask)

        self.assertTrue(torch.equal(output, sentinel))
        run_mock.assert_called_once()
        self.assertIs(run_mock.call_args.kwargs['sequence_parallel'], sequence_parallel)
        self.assertIs(run_mock.call_args.kwargs['module'], model.linear)

    def test_head_parallel_keeps_z_local_and_segments_packed_sequences(self):
        module = _DummyHeadParallelModule()
        hidden_states = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0], [0.0]]], dtype=torch.float32)
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.int64)
        sequence_parallel = SimpleNamespace(
            world_size=1,
            sp_world_size=1,
            _sp_group=None,
            extra_kwargs={
                'is_packed': True,
                'position_ids': torch.tensor([[0, 1, 2, 0, 1, -1]], dtype=torch.long),
            },
        )

        shard_shapes = []
        conv_calls = []
        rule_calls = []

        def fake_seq_to_head_shard(tensor: torch.Tensor, _sequence_parallel) -> torch.Tensor:
            shard_shapes.append(tuple(tensor.shape))
            return tensor

        def fake_head_to_seq_shard(tensor: torch.Tensor, _sequence_parallel) -> torch.Tensor:
            return tensor

        def fake_causal_conv1d_fn(x: torch.Tensor, weight: torch.Tensor, bias, activation: str, seq_idx=None) -> torch.Tensor:
            del weight, bias, activation, seq_idx
            conv_calls.append(int(x.shape[-1]))
            return x

        def fake_rule(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            g: torch.Tensor,
            beta: torch.Tensor,
            chunk_size=None,
            initial_state=None,
            output_final_state: bool = False,
            use_qk_l2norm_in_kernel: bool = False,
        ):
            del query, key, g, beta, chunk_size, initial_state, output_final_state, use_qk_l2norm_in_kernel
            rule_calls.append(int(value.shape[1]))
            return value, None

        with patch(
            'twinkle.patch.qwen35_linear_attention_sp._seq_to_head_shard',
            side_effect=fake_seq_to_head_shard,
        ), patch(
            'twinkle.patch.qwen35_linear_attention_sp._head_to_seq_shard',
            side_effect=fake_head_to_seq_shard,
        ):
            output = _run_qwen35_head_parallel(
                sequence_parallel=sequence_parallel,
                module=module,
                origin_rule=fake_rule,
                origin_causal_conv1d_fn=fake_causal_conv1d_fn,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )

        self.assertEqual(len(shard_shapes), 3)
        self.assertEqual(conv_calls, [3, 2])
        self.assertEqual(rule_calls, [3, 2])
        expected = torch.tensor([[[2.0], [4.0], [6.0], [8.0], [10.0], [0.0]]], dtype=torch.float32)
        self.assertTrue(torch.equal(output, expected))


if __name__ == '__main__':
    unittest.main()
