# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import socket
import unittest
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from twinkle.model.transformers.strategy.linear_attention.qwen35_head_parallel import (
    Qwen35HeadParallelHelper,
    _head_to_seq_shard,
    _seq_to_head_shard,
)


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
        helper = Qwen35HeadParallelHelper()
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
        local_output = helper.run(
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


class TestQwen35HeadParallel(unittest.TestCase):

    def test_head_parallel_selects_recurrent_rule_for_misaligned_sequence(self):
        helper = Qwen35HeadParallelHelper()

        class DummyModule:
            recurrent_gated_delta_rule = staticmethod(lambda *args, **kwargs: None)

        dummy_module = DummyModule()
        rule, chunk_size = helper._select_rule(dummy_module, object(), full_seq_len=3)
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


if __name__ == '__main__':
    unittest.main()
