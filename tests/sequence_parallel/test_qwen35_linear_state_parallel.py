# Copyright (c) ModelScope Contributors. All rights reserved.
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import unittest
from datetime import timedelta
from unittest import mock

from twinkle.model.transformers.strategy.sequence_parallel import SequenceParallel, _QwenLinearStateParallelFn


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _mock_chunk_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size=None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    del chunk_size, use_qk_l2norm_in_kernel
    batch, seq_len, num_heads, key_dim = query.shape
    value_dim = value.shape[-1]
    state = (torch.zeros((batch, num_heads, key_dim, value_dim), dtype=query.dtype, device=query.device)
             if initial_state is None else initial_state.to(query.dtype))
    outputs = []
    for i in range(seq_len):
        q_t = query[:, i]
        k_t = key[:, i]
        v_t = value[:, i]
        g_t = torch.sigmoid(g[:, i]).unsqueeze(-1).unsqueeze(-1)
        beta_t = torch.sigmoid(beta[:, i]).unsqueeze(-1).unsqueeze(-1)
        state = state * g_t + beta_t * torch.einsum('bhd,bhe->bhde', k_t, v_t)
        out_t = torch.einsum('bhd,bhde->bhe', q_t, state)
        outputs.append(out_t.unsqueeze(1))
    output = torch.cat(outputs, dim=1).contiguous()
    return output, state if output_final_state else None


def _mock_torch_recurrent_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    del key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel
    return query, None


def _run_state_parallel_worker(rank: int, world_size: int, port: int):
    dist.init_process_group(
        backend='gloo',
        init_method=f'tcp://127.0.0.1:{port}',
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
    try:
        torch.manual_seed(20260305)
        batch = 2
        seq_per_rank = 4
        seq_len = seq_per_rank * world_size
        num_heads = 3
        key_dim = 4
        value_dim = 5
        seq_start = rank * seq_per_rank
        seq_end = seq_start + seq_per_rank

        base_query = torch.randn(batch, seq_len, num_heads, key_dim, dtype=torch.float32)
        base_key = torch.randn(batch, seq_len, num_heads, key_dim, dtype=torch.float32)
        base_value = torch.randn(batch, seq_len, num_heads, value_dim, dtype=torch.float32)
        base_g = torch.randn(batch, seq_len, num_heads, dtype=torch.float32)
        base_beta = torch.randn(batch, seq_len, num_heads, dtype=torch.float32)

        full_query = base_query.clone().requires_grad_(True)
        full_key = base_key.clone().requires_grad_(True)
        full_value = base_value.clone().requires_grad_(True)
        full_g = base_g.clone().requires_grad_(True)
        full_beta = base_beta.clone().requires_grad_(True)
        full_output, _ = _mock_chunk_rule(
            full_query,
            full_key,
            full_value,
            g=full_g,
            beta=full_beta,
            initial_state=None,
            output_final_state=False,
        )
        # Match distributed objective: sum of per-rank local means.
        full_loss = torch.stack([
            full_output[:, i * seq_per_rank:(i + 1) * seq_per_rank].pow(2).mean() for i in range(world_size)
        ]).sum()
        full_loss.backward()

        local_query = base_query[:, seq_start:seq_end].contiguous().clone().requires_grad_(True)
        local_key = base_key[:, seq_start:seq_end].contiguous().clone().requires_grad_(True)
        local_value = base_value[:, seq_start:seq_end].contiguous().clone().requires_grad_(True)
        local_g = base_g[:, seq_start:seq_end].contiguous().clone().requires_grad_(True)
        local_beta = base_beta[:, seq_start:seq_end].contiguous().clone().requires_grad_(True)

        local_output = _QwenLinearStateParallelFn.apply(
            local_query,
            local_key,
            local_value,
            local_g,
            local_beta,
            dist.group.WORLD,
            _mock_chunk_rule,
            None,
            False,
        )
        local_loss = local_output.pow(2).mean()
        local_loss.backward()

        expected_output = full_output[:, seq_start:seq_end].detach()
        expected_query_grad = full_query.grad[:, seq_start:seq_end].detach()
        expected_key_grad = full_key.grad[:, seq_start:seq_end].detach()
        expected_value_grad = full_value.grad[:, seq_start:seq_end].detach()
        expected_g_grad = full_g.grad[:, seq_start:seq_end].detach()
        expected_beta_grad = full_beta.grad[:, seq_start:seq_end].detach()

        assert torch.allclose(local_output.detach(), expected_output, atol=1e-5, rtol=1e-5), (
            f'rank={rank}: output mismatch, max_diff='
            f'{(local_output.detach() - expected_output).abs().max().item()}')
        assert torch.allclose(local_query.grad, expected_query_grad, atol=1e-5, rtol=1e-5), (
            f'rank={rank}: query grad mismatch, max_diff='
            f'{(local_query.grad - expected_query_grad).abs().max().item()}')
        assert torch.allclose(local_key.grad, expected_key_grad, atol=1e-5, rtol=1e-5), (
            f'rank={rank}: key grad mismatch, max_diff='
            f'{(local_key.grad - expected_key_grad).abs().max().item()}')
        assert torch.allclose(local_value.grad, expected_value_grad, atol=1e-5, rtol=1e-5), (
            f'rank={rank}: value grad mismatch, max_diff='
            f'{(local_value.grad - expected_value_grad).abs().max().item()}')
        assert torch.allclose(local_g.grad, expected_g_grad, atol=1e-5, rtol=1e-5), (
            f'rank={rank}: g grad mismatch, max_diff={((local_g.grad - expected_g_grad).abs().max().item())}')
        assert torch.allclose(local_beta.grad, expected_beta_grad, atol=1e-5, rtol=1e-5), (
            f'rank={rank}: beta grad mismatch, max_diff='
            f'{(local_beta.grad - expected_beta_grad).abs().max().item()}')
    finally:
        dist.destroy_process_group()


class TestQwen35LinearStateParallel(unittest.TestCase):

    def test_linear_state_parallel_matches_full(self):
        if not dist.is_available():
            self.skipTest('torch.distributed is not available')
        world_size = 2
        port = _find_free_port()
        mp.spawn(
            _run_state_parallel_worker,
            args=(world_size, port),
            nprocs=world_size,
            join=True,
        )

    def test_wrap_qwen35_chunk_rule_prefers_torch_recurrent_for_misaligned_local_seq(self):

        class DummyModule:
            recurrent_gated_delta_rule = staticmethod(lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError('fused recurrent rule should not be selected')))

        sp = SequenceParallel()
        sp.sp_world_size = 2
        sp._sp_group = object()
        sp.extra_kwargs['is_packed'] = False

        dummy_module = DummyModule()
        wrapped = sp._wrap_qwen35_chunk_rule(dummy_module, _mock_chunk_rule)

        query = torch.randn(1, 16, 2, 4)
        key = torch.randn(1, 16, 2, 4)
        value = torch.randn(1, 16, 2, 5)
        g = torch.randn(1, 16, 2)
        beta = torch.randn(1, 16, 2)

        with mock.patch.object(_QwenLinearStateParallelFn, 'apply', autospec=True) as mock_apply:
            mock_apply.return_value = query
            output, _ = wrapped(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )

        self.assertIs(output, query)
        self.assertIs(mock_apply.call_args.args[7], _mock_torch_recurrent_rule)
        self.assertIsNone(mock_apply.call_args.args[8])


if __name__ == '__main__':
    unittest.main()
