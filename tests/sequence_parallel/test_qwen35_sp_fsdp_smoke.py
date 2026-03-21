# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib
import os
import unittest

import torch
import torch.distributed as dist


def _force_qwen35_linear_recurrent(model: torch.nn.Module) -> int:
    patched = 0
    for module in model.modules():
        if module.__class__.__name__ != 'Qwen3_5GatedDeltaNet':
            continue
        module_impl = importlib.import_module(module.__class__.__module__)
        recurrent_rule = getattr(module_impl, 'torch_recurrent_gated_delta_rule', None)
        if recurrent_rule is None:
            recurrent_rule = getattr(module, 'recurrent_gated_delta_rule', None)
        if recurrent_rule is None:
            raise RuntimeError('Qwen3.5 torch_recurrent_gated_delta_rule is unavailable.')

        def recurrent_adapter(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            g: torch.Tensor,
            beta: torch.Tensor,
            chunk_size=None,
            initial_state=None,
            output_final_state: bool = False,
            use_qk_l2norm_in_kernel: bool = False,
            _rule=recurrent_rule,
        ):
            return _rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )

        module.chunk_gated_delta_rule = recurrent_adapter
        module.recurrent_gated_delta_rule = recurrent_adapter
        patched += 1
    return patched


class TestQwen35SPFSDPSmoke(unittest.TestCase):
    """Opt-in smoke test for Qwen3.5 + Native FSDP + SequenceParallel.

    Run manually with torchrun, e.g.:
    QWEN35_MODEL_ID=/path/to/model \
    QWEN35_SP_SMOKE=1 \
    torchrun --nproc_per_node=2 -m pytest -q tests/sequence_parallel/test_qwen35_sp_fsdp_smoke.py -rs
    """

    @classmethod
    def setUpClass(cls):
        if not dist.is_available() or dist.is_initialized():
            return
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        if world_size <= 1:
            return
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend, init_method='env://')

    @classmethod
    def tearDownClass(cls):
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    def test_qwen35_sp_fsdp_smoke(self):
        if os.environ.get('QWEN35_SP_SMOKE', '0') != '1':
            self.skipTest('Set QWEN35_SP_SMOKE=1 to enable this test.')
        if not dist.is_available() or not dist.is_initialized():
            self.skipTest('Run this test with torchrun (distributed initialized).')
        if not torch.cuda.is_available():
            self.skipTest('CUDA is required for this test.')
        if torch.cuda.device_count() < dist.get_world_size():
            self.skipTest('Need at least world_size CUDA devices.')

        model_id = os.environ.get('QWEN35_MODEL_ID')
        if not model_id:
            self.skipTest('Set QWEN35_MODEL_ID to a local Qwen3.5 model path.')

        import numpy as np
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from twinkle.patch import apply_patch
        from twinkle.model.transformers.strategy.sequence_parallel import sequence_parallel
        from twinkle.utils import DeviceMesh

        rank = dist.get_rank()
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)

        local_files_only = os.environ.get('QWEN35_LOCAL_ONLY', '1') == '1'
        attn_impl = os.environ.get('QWEN35_SP_SMOKE_ATTN_IMPL', 'sdpa')
        force_recurrent = os.environ.get('QWEN35_SP_FORCE_RECURRENT', '0') == '1'
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=local_files_only,
            attn_implementation=attn_impl,
        ).to(device)
        if force_recurrent:
            _force_qwen35_linear_recurrent(model)
        apply_patch(model, 'Qwen35LinearAttentionSPPatch')
        model.train()

        world_size = dist.get_world_size()
        device_mesh = DeviceMesh(
            device_type='cuda',
            mesh=np.arange(world_size),
            mesh_dim_names=('fsdp',),
            ulysses_size=world_size,
        )
        sequence_parallel.prepare(
            sp_size=world_size,
            model=model,
            tokenizer=tokenizer,
            device_mesh=device_mesh,
        )

        fsdp_model = FSDP(model, use_orig_params=True, device_id=device)
        optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-6)

        batch_size = int(os.environ.get('QWEN35_SP_SMOKE_BATCH', '1'))
        qwen35_linear_chunk = int(os.environ.get('QWEN35_SP_LINEAR_CHUNK', '64'))
        seq_len = int(os.environ.get('QWEN35_SP_SMOKE_SEQ_LEN', str(qwen35_linear_chunk * world_size)))
        local_seq = seq_len // world_size
        if seq_len % world_size != 0:
            self.skipTest(f'seq_len ({seq_len}) must be divisible by world_size ({world_size}).')
        if local_seq % qwen35_linear_chunk != 0:
            self.skipTest(
                f'local_seq ({local_seq}) must be a multiple of Qwen3.5 linear-attention chunk '
                f'size ({qwen35_linear_chunk}) for this smoke test.')
        vocab_size = min(int(getattr(fsdp_model.module.config, 'vocab_size', 32000)), 32000)

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        labels = input_ids.clone()
        attention_mask = torch.ones_like(input_ids, device=device)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1).contiguous()

        outputs = fsdp_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
        )
        loss = outputs.loss
        self.assertTrue(bool(torch.isfinite(loss).all().item()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


if __name__ == '__main__':
    unittest.main()
