# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib
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


def _resolve_close_defaults(force_recurrent: bool, model_dtype: torch.dtype) -> dict[str, str]:
    low_precision = model_dtype in (torch.bfloat16, torch.float16)
    if force_recurrent:
        # In recurrent-only mode both baseline and SP share the same linear-attention algorithm, so the defaults
        # should reflect a strict parity expectation rather than the looser "training-close" expectation used by the
        # mixed chunk/recurrent path.
        return {
            'logit_atol': '5e-3' if low_precision else '1e-4',
            'loss_atol': '1e-4' if low_precision else '1e-6',
            'loss_rtol': '1e-4' if low_precision else '1e-5',
            'grad_atol': '5e-4' if low_precision else '1e-5',
            'grad_rtol': '1e-3' if low_precision else '1e-4',
        }
    return {
        'logit_atol': '5e-2',
        'loss_atol': '5e-2',
        'loss_rtol': '0.0',
        'grad_atol': '2e-1',
        'grad_rtol': '0.0',
    }


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


def _install_qwen35_linear_rule_capture(model: torch.nn.Module, max_layers: int) -> dict[str, dict[str, torch.Tensor]]:
    captures: dict[str, dict[str, torch.Tensor]] = {}
    if max_layers <= 0:
        return captures
    installed = 0
    for module_name, module in model.named_modules():
        if module.__class__.__name__ != 'Qwen3_5GatedDeltaNet':
            continue
        if installed >= max_layers:
            break
        rule = getattr(module, 'chunk_gated_delta_rule', None)
        if rule is None:
            continue

        def capture_wrapper(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            g: torch.Tensor,
            beta: torch.Tensor,
            chunk_size=None,
            initial_state=None,
            output_final_state: bool = False,
            use_qk_l2norm_in_kernel: bool = False,
            _rule=rule,
            _captures=captures,
            _module_name=module_name,
        ):
            _captures[_module_name] = {
                'query': query.detach().float().contiguous(),
                'key': key.detach().float().contiguous(),
                'value': value.detach().float().contiguous(),
                'g': g.detach().float().contiguous(),
                'beta': beta.detach().float().contiguous(),
            }
            return _rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                chunk_size=chunk_size,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )

        module.chunk_gated_delta_rule = capture_wrapper
        installed += 1
    return captures


def _slice_seq_dim(tensor: torch.Tensor, seq_len: int) -> torch.Tensor:
    if tensor.shape[1] <= seq_len:
        return tensor
    return tensor[:, :seq_len].contiguous()


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
        # Default to SDPA so the parity test exercises SP semantics without depending on FA2 kernel availability
        # or numerical/runtime quirks in the local CUDA environment. Override explicitly to debug FA2.
        attn_impl = os.environ.get('QWEN35_SP_PARITY_ATTN_IMPL', 'sdpa')
        model_dtype = _resolve_torch_dtype(os.environ.get('QWEN35_SP_PARITY_DTYPE', 'bfloat16'))
        force_recurrent = os.environ.get('QWEN35_SP_FORCE_RECURRENT', '0') == '1'
        linear_debug_layers = int(os.environ.get('QWEN35_SP_PARITY_LINEAR_DEBUG_LAYERS', '0'))
        close_defaults = _resolve_close_defaults(force_recurrent, model_dtype)
        logits_atol = float(os.environ.get('QWEN35_SP_PARITY_LOGIT_ATOL', close_defaults['logit_atol']))
        logits_relaxed_max_atol = float(os.environ.get('QWEN35_SP_PARITY_LOGIT_RELAXED_MAX_ATOL', '3.0'))
        logits_mean_atol = float(os.environ.get('QWEN35_SP_PARITY_LOGIT_MEAN_ATOL', '1.5e-1'))
        logits_p99_atol = float(os.environ.get('QWEN35_SP_PARITY_LOGIT_P99_ATOL', '1.0'))
        logits_sdpa_fp32_max_atol = float(os.environ.get('QWEN35_SP_PARITY_LOGIT_SDPA_FP32_MAX_ATOL', '3.0'))
        logits_sdpa_fp32_mean_atol = float(os.environ.get('QWEN35_SP_PARITY_LOGIT_SDPA_FP32_MEAN_ATOL', '1.0e-1'))
        logits_sdpa_fp32_p99_atol = float(os.environ.get('QWEN35_SP_PARITY_LOGIT_SDPA_FP32_P99_ATOL', '5.0e-1'))
        logits_linear_exact_max_atol = float(os.environ.get('QWEN35_SP_PARITY_LOGIT_LINEAR_EXACT_MAX_ATOL', '1.0e-2'))
        logits_linear_exact_mean_atol = float(os.environ.get('QWEN35_SP_PARITY_LOGIT_LINEAR_EXACT_MEAN_ATOL', '1.0e-3'))
        logits_linear_exact_p99_atol = float(os.environ.get('QWEN35_SP_PARITY_LOGIT_LINEAR_EXACT_P99_ATOL', '5.0e-3'))
        loss_atol = float(os.environ.get('QWEN35_SP_PARITY_LOSS_ATOL', close_defaults['loss_atol']))
        loss_rtol = float(os.environ.get('QWEN35_SP_PARITY_LOSS_RTOL', close_defaults['loss_rtol']))
        grad_atol = float(os.environ.get('QWEN35_SP_PARITY_GRAD_ATOL', close_defaults['grad_atol']))
        grad_rtol = float(os.environ.get('QWEN35_SP_PARITY_GRAD_RTOL', close_defaults['grad_rtol']))
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
        forced_recurrent_layers = 0
        if force_recurrent:
            forced_recurrent_layers = _force_qwen35_linear_recurrent(base_model)
        base_linear_captures: dict[str, dict[str, torch.Tensor]] = {}
        if linear_debug_layers > 0:
            base_linear_captures = _install_qwen35_linear_rule_capture(base_model, linear_debug_layers)
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
        if force_recurrent:
            forced_recurrent_layers = _force_qwen35_linear_recurrent(sp_model)
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
        self.assertEqual(sequence_parallel.linear_attention_provider_name, 'qwen35')
        sp_linear_captures: dict[str, dict[str, torch.Tensor]] = {}
        if linear_debug_layers > 0:
            sp_linear_captures = _install_qwen35_linear_rule_capture(sp_model, linear_debug_layers)
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

        if linear_debug_layers > 0 and base_linear_captures and sp_linear_captures:
            linear_reports = []
            for module_name in base_linear_captures.keys():
                if module_name not in sp_linear_captures or not base_linear_captures[module_name]:
                    continue
                sp_capture = sp_linear_captures[module_name]
                base_capture = base_linear_captures[module_name]
                tensor_reports = []
                for tensor_name in ('query', 'key', 'value', 'g', 'beta'):
                    if tensor_name not in sp_capture or tensor_name not in base_capture:
                        continue
                    sp_full = _slice_seq_dim(sequence_parallel.gather(sp_capture[tensor_name], dim=1), seq_len)
                    base_full = _slice_seq_dim(base_capture[tensor_name], seq_len)
                    diff = (base_full - sp_full).abs()
                    tensor_reports.append(
                        f'{tensor_name}:max={diff.max().item():.6f},mean={diff.mean().item():.6f}'
                    )
                if tensor_reports:
                    linear_reports.append(f'{module_name}: ' + '; '.join(tensor_reports))
            if linear_reports and rank == 0:
                print('Qwen3.5 linear pre-rule diffs:', flush=True)
                for report in linear_reports:
                    print(report, flush=True)

        loss_diff = abs((loss_base_metric - loss_sp_metric).item())
        grad_diff = abs((grad_base - grad_sp).item())
        diagnostics = (
            f'world_size={world_size}, rank={rank}, '
            f'loss_base={loss_base_metric.item():.6f}, loss_sp={loss_sp_metric.item():.6f}, '
            f'loss_diff={loss_diff:.6f}, '
            f'grad_base={grad_base.item():.6f}, grad_sp={grad_sp.item():.6f}, '
            f'grad_diff={grad_diff:.6f}, '
            f'max_abs_diff={max_abs_diff.item():.6f}, mean_abs_diff={mean_abs_diff.item():.6f}, '
            f'p99_abs_diff={p99_abs_diff.item():.6f}, '
            f'local_max_abs_diff={local_max_abs_diff.item():.6f}, '
            f'attn_impl={getattr(sp_model.config, "_attn_implementation", "unknown")}, '
            f'dtype={str(model_dtype).replace("torch.", "")}, '
            f'force_recurrent={force_recurrent}, forced_recurrent_layers={forced_recurrent_layers}, '
            f'linear_attention_provider={sequence_parallel.linear_attention_provider_name}, '
            f'loss_atol={loss_atol:.2e}, loss_rtol={loss_rtol:.2e}, '
            f'grad_atol={grad_atol:.2e}, grad_rtol={grad_rtol:.2e}'
        )

        if os.environ.get('QWEN35_SP_PARITY_DEBUG', '0') == '1' and rank == 0:
            print('SP parity diagnostics:', diagnostics, flush=True)
            print('Baseline logits slice:', _format_tensor_slice(logits_base), flush=True)
            print('SP logits slice:', _format_tensor_slice(logits_sp_full), flush=True)

        effective_attn_impl = getattr(sp_model.config, '_attn_implementation', attn_impl or 'unknown')
        is_sdpa = effective_attn_impl == 'sdpa'
        is_low_precision_sdpa = is_sdpa and model_dtype in (torch.bfloat16, torch.float16)
        is_fp32_sdpa = is_sdpa and model_dtype == torch.float32
        is_fp32_linear_exact = (
            force_recurrent
            and model_dtype == torch.float32
            and sequence_parallel.linear_attention_provider_name == 'qwen35'
        )

        if is_low_precision_sdpa:
            self.assertLessEqual(max_abs_diff.item(), logits_relaxed_max_atol, msg=diagnostics)
            self.assertLessEqual(mean_abs_diff.item(), logits_mean_atol, msg=diagnostics)
            self.assertLessEqual(p99_abs_diff.item(), logits_p99_atol, msg=diagnostics)
        elif is_fp32_linear_exact:
            self.assertLessEqual(max_abs_diff.item(), logits_linear_exact_max_atol, msg=diagnostics)
            self.assertLessEqual(mean_abs_diff.item(), logits_linear_exact_mean_atol, msg=diagnostics)
            self.assertLessEqual(p99_abs_diff.item(), logits_linear_exact_p99_atol, msg=diagnostics)
        elif is_fp32_sdpa:
            self.assertLessEqual(max_abs_diff.item(), logits_sdpa_fp32_max_atol, msg=diagnostics)
            self.assertLessEqual(mean_abs_diff.item(), logits_sdpa_fp32_mean_atol, msg=diagnostics)
            self.assertLessEqual(p99_abs_diff.item(), logits_sdpa_fp32_p99_atol, msg=diagnostics)
        else:
            self.assertLessEqual(max_abs_diff.item(), logits_atol, msg=diagnostics)
        self.assertLessEqual(
            loss_diff,
            max(loss_atol, loss_rtol * max(abs(loss_base_metric.item()), abs(loss_sp_metric.item()))),
            msg=diagnostics,
        )
        self.assertLessEqual(
            grad_diff,
            max(grad_atol, grad_rtol * max(abs(grad_base.item()), abs(grad_sp.item()))),
            msg=diagnostics,
        )


if __name__ == '__main__':
    unittest.main()
