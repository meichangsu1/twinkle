# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import os
import unittest

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _compute_next_token_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def _first_grad_norm(parameters) -> torch.Tensor:
    for param in parameters:
        if param.grad is not None:
            return param.grad.detach().float().norm()
    return torch.tensor(0.0)


def _format_tensor_slice(tensor: torch.Tensor, seq_tokens: int = 2, vocab_tokens: int = 8) -> str:
    if tensor.dim() < 3:
        return str(tuple(tensor.shape))
    slice_ = tensor[0, :min(seq_tokens, tensor.shape[1]), :min(vocab_tokens, tensor.shape[2])].detach().cpu()
    return str(slice_.tolist())


def _resolve_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
        'float16': torch.float16,
        'fp16': torch.float16,
        'float32': torch.float32,
        'fp32': torch.float32,
    }
    key = str(name).strip().lower()
    if key not in mapping:
        raise ValueError(f'Unsupported dtype override: {name}')
    return mapping[key]


def _quantile_via_kthvalue(tensor: torch.Tensor, q: float) -> torch.Tensor:
    flat = tensor.reshape(-1)
    if flat.numel() == 0:
        raise ValueError('quantile input tensor must be non-empty')
    if flat.numel() == 1:
        return flat[0]
    k = min(flat.numel(), max(1, math.ceil(q * flat.numel())))
    return flat.kthvalue(k).values


class TestQwen35SPParity(unittest.TestCase):
    """Opt-in parity test for Qwen3.5: SP=0 vs SP>1.

    Run manually with torchrun, e.g.:
    QWEN35_SP_PARITY=1 \
    QWEN35_MODEL_ID=/path/to/model \
    torchrun --nproc_per_node=2 -m pytest -q tests/sequence_parallel/test_qwen35_sp_parity.py -rs
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

    def test_qwen35_sp_parity(self):
        if os.environ.get('QWEN35_SP_PARITY', '0') != '1':
            self.skipTest('Set QWEN35_SP_PARITY=1 to enable this test.')
        if not dist.is_available() or not dist.is_initialized():
            self.skipTest('Run this test with torchrun (distributed initialized).')
        if not torch.cuda.is_available():
            self.skipTest('CUDA is required for this test.')
        if torch.cuda.device_count() < dist.get_world_size():
            self.skipTest('Need at least world_size CUDA devices.')

        model_id = os.environ.get('QWEN35_MODEL_ID')
        if not model_id:
            self.skipTest('Set QWEN35_MODEL_ID to a local Qwen3.5 model path.')

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        torch.manual_seed(int(os.environ.get('QWEN35_SP_PARITY_SEED', '20260313')))

        qwen35_linear_chunk = int(os.environ.get('QWEN35_SP_LINEAR_CHUNK', '64'))
        batch_size = int(os.environ.get('QWEN35_SP_PARITY_BATCH', '1'))
        seq_len = int(os.environ.get('QWEN35_SP_PARITY_SEQ_LEN', str(qwen35_linear_chunk * world_size)))
        if seq_len % world_size != 0:
            self.skipTest(f'seq_len ({seq_len}) must be divisible by world_size ({world_size}).')
        local_seq = seq_len // world_size
        if local_seq % qwen35_linear_chunk != 0:
            self.skipTest(
                f'local_seq ({local_seq}) must be a multiple of Qwen3.5 linear-attention chunk '
                f'size ({qwen35_linear_chunk}) for exact parity.')

        local_files_only = os.environ.get('QWEN35_LOCAL_ONLY', '1') == '1'
        logits_atol = float(os.environ.get('QWEN35_SP_PARITY_LOGIT_ATOL', '5e-2'))
        logits_relaxed_max_atol = float(os.environ.get('QWEN35_SP_PARITY_LOGIT_RELAXED_MAX_ATOL', '3.0'))
        logits_mean_atol = float(os.environ.get('QWEN35_SP_PARITY_LOGIT_MEAN_ATOL', '1.5e-1'))
        logits_p99_atol = float(os.environ.get('QWEN35_SP_PARITY_LOGIT_P99_ATOL', '1.0'))
        logits_sdpa_fp32_max_atol = float(os.environ.get('QWEN35_SP_PARITY_LOGIT_SDPA_FP32_MAX_ATOL', '3.0'))
        logits_sdpa_fp32_mean_atol = float(os.environ.get('QWEN35_SP_PARITY_LOGIT_SDPA_FP32_MEAN_ATOL', '1.0e-1'))
        logits_sdpa_fp32_p99_atol = float(os.environ.get('QWEN35_SP_PARITY_LOGIT_SDPA_FP32_P99_ATOL', '5.0e-1'))
        loss_atol = float(os.environ.get('QWEN35_SP_PARITY_LOSS_ATOL', '5e-2'))
        grad_atol = float(os.environ.get('QWEN35_SP_PARITY_GRAD_ATOL', '2e-1'))
        # Default to SDPA so the parity test exercises SP semantics without depending on FA2 kernel availability
        # or numerical/runtime quirks in the local CUDA environment. Override explicitly to debug FA2.
        attn_impl = os.environ.get('QWEN35_SP_PARITY_ATTN_IMPL', 'sdpa')
        model_dtype = _resolve_torch_dtype(os.environ.get('QWEN35_SP_PARITY_DTYPE', 'bfloat16'))

        try:
            import numpy as np
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from transformers import AutoModelForCausalLM, AutoTokenizer

            from twinkle.model.transformers.strategy.sequence_parallel import sequence_parallel
            from twinkle.utils import DeviceMesh
        except Exception as exc:
            self.skipTest(f'Required dependencies are unavailable: {exc}')

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )

        with torch.no_grad():
            vocab_size = min(max(int(getattr(tokenizer, 'vocab_size', 32000) or 32000), 1000), 32000)
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            labels = input_ids.clone()
            attention_mask = torch.ones_like(input_ids, device=device)
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1).contiguous()

        seq_start = rank * local_seq
        seq_end = seq_start + local_seq
        labels_local = labels[:, seq_start:seq_end].contiguous()

        # Baseline: no SP
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=model_dtype,
            trust_remote_code=True,
            local_files_only=local_files_only,
            attn_implementation=attn_impl,
        ).to(device)
        base_model.eval()
        base_fsdp = FSDP(base_model, use_orig_params=True, device_id=device)
        base_outputs = base_fsdp(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        logits_base = base_outputs.logits.float()
        logits_base_local = logits_base[:, seq_start:seq_end, :].contiguous()
        loss_base_local = _compute_next_token_ce(logits_base_local, labels_local)
        loss_base_local.backward()
        grad_base = _first_grad_norm(base_fsdp.parameters()).to(device)
        loss_base_metric = _compute_next_token_ce(logits_base, labels).detach().to(device)

        # Free baseline graph before SP run to avoid OOM.
        del base_outputs
        del base_fsdp
        del base_model
        torch.cuda.empty_cache()
        dist.barrier()

        # SP run: prepare sequence parallel and compare local objective + global logits.
        sp_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=model_dtype,
            trust_remote_code=True,
            local_files_only=local_files_only,
            attn_implementation=attn_impl,
        ).to(device)
        sp_model.eval()
        device_mesh = DeviceMesh(
            device_type='cuda',
            mesh=np.arange(world_size),
            mesh_dim_names=('fsdp',),
            ulysses_size=world_size,
        )
        sequence_parallel.prepare(
            sp_size=world_size,
            model=sp_model,
            tokenizer=tokenizer,
            device_mesh=device_mesh,
        )
        sp_fsdp = FSDP(sp_model, use_orig_params=True, device_id=device)

        sp_outputs = sp_fsdp(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        logits_sp_local = sp_outputs.logits[:, :local_seq, :].contiguous().float()
        loss_sp_local = _compute_next_token_ce(logits_sp_local, labels_local)
        loss_sp_local.backward()
        grad_sp = _first_grad_norm(sp_fsdp.parameters()).to(device)

        logits_sp_full = sequence_parallel.gather(logits_sp_local, dim=1, position_ids=position_ids)
        logits_sp_full = logits_sp_full[:, :seq_len, :].contiguous()
        loss_sp_metric = _compute_next_token_ce(logits_sp_full, labels).detach().to(device)

        # Compare scalar metrics globally (averaged across ranks).
        for tensor in (loss_base_metric, loss_sp_metric, grad_base, grad_sp):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor.div_(world_size)

        # Logits parity: compare global logits element-wise.
        max_abs_diff = (logits_base - logits_sp_full).abs().max().detach().to(device)
        dist.all_reduce(max_abs_diff, op=dist.ReduceOp.MAX)
        mean_abs_diff = (logits_base - logits_sp_full).abs().mean().detach().to(device)
        dist.all_reduce(mean_abs_diff, op=dist.ReduceOp.SUM)
        mean_abs_diff.div_(world_size)
        p99_abs_diff = _quantile_via_kthvalue((logits_base - logits_sp_full).abs(), 0.99).detach().to(device)
        dist.all_reduce(p99_abs_diff, op=dist.ReduceOp.MAX)

        local_logits_base = logits_base[:, seq_start:seq_end, :].contiguous()
        local_logits_sp = logits_sp_full[:, seq_start:seq_end, :].contiguous()
        local_max_abs_diff = (local_logits_base - local_logits_sp).abs().max().detach().to(device)
        dist.all_reduce(local_max_abs_diff, op=dist.ReduceOp.MAX)

        diagnostics = (
            f'world_size={world_size}, rank={rank}, '
            f'loss_base={loss_base_metric.item():.6f}, loss_sp={loss_sp_metric.item():.6f}, '
            f'loss_diff={abs((loss_base_metric - loss_sp_metric).item()):.6f}, '
            f'grad_base={grad_base.item():.6f}, grad_sp={grad_sp.item():.6f}, '
            f'grad_diff={abs((grad_base - grad_sp).item()):.6f}, '
            f'max_abs_diff={max_abs_diff.item():.6f}, mean_abs_diff={mean_abs_diff.item():.6f}, '
            f'p99_abs_diff={p99_abs_diff.item():.6f}, '
            f'local_max_abs_diff={local_max_abs_diff.item():.6f}, '
            f'attn_impl={getattr(sp_model.config, "_attn_implementation", "unknown")}, '
            f'dtype={str(model_dtype).replace("torch.", "")}'
        )

        if os.environ.get('QWEN35_SP_PARITY_DEBUG', '0') == '1' and rank == 0:
            print('SP parity diagnostics:', diagnostics, flush=True)
            print('Baseline logits slice:', _format_tensor_slice(logits_base), flush=True)
            print('SP logits slice:', _format_tensor_slice(logits_sp_full), flush=True)

        effective_attn_impl = getattr(sp_model.config, '_attn_implementation', attn_impl or 'unknown')
        is_sdpa = effective_attn_impl == 'sdpa'
        is_low_precision_sdpa = is_sdpa and model_dtype in (torch.bfloat16, torch.float16)
        is_fp32_sdpa = is_sdpa and model_dtype == torch.float32

        if is_low_precision_sdpa:
            self.assertLessEqual(max_abs_diff.item(), logits_relaxed_max_atol, msg=diagnostics)
            self.assertLessEqual(mean_abs_diff.item(), logits_mean_atol, msg=diagnostics)
            self.assertLessEqual(p99_abs_diff.item(), logits_p99_atol, msg=diagnostics)
        elif is_fp32_sdpa:
            self.assertLessEqual(max_abs_diff.item(), logits_sdpa_fp32_max_atol, msg=diagnostics)
            self.assertLessEqual(mean_abs_diff.item(), logits_sdpa_fp32_mean_atol, msg=diagnostics)
            self.assertLessEqual(p99_abs_diff.item(), logits_sdpa_fp32_p99_atol, msg=diagnostics)
        else:
            self.assertLessEqual(max_abs_diff.item(), logits_atol, msg=diagnostics)
        self.assertLessEqual(abs((loss_base_metric - loss_sp_metric).item()), loss_atol, msg=diagnostics)
        self.assertLessEqual(abs((grad_base - grad_sp).item()), grad_atol, msg=diagnostics)


if __name__ == '__main__':
    unittest.main()
