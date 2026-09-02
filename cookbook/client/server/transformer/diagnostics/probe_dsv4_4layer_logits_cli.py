#!/usr/bin/env python3
"""Run a four-layer DeepSeek-V4 logits probe directly with torchrun.

This is the local/CLI counterpart of ``probe_dsv4_4layer_logits.py``.  It
constructs ``TransformersModel`` in every torchrun process and therefore does
not use GatewayServer, Ray Serve, sessions, tenants, or LoRA slots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


DEFAULT_INPUT_IDS = [0, 128803, 2788, 6573, 70979, 36005, 320, 128804, 128821]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('no_ep', 'ep_loop', 'ep_gmm'))
    parser.add_argument(
        '--model-id',
        default=os.environ.get('DSV4_MODEL_ID'),
        help='Four-layer BF16 checkpoint. Defaults to DSV4_MODEL_ID.',
    )
    parser.add_argument('--output-dir', type=Path, default=Path('/nas/disk6/ljl/dsv4_ep_diag/results'))
    parser.add_argument('--input-ids', default=','.join(str(item) for item in DEFAULT_INPUT_IDS))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mixed-precision', choices=('no', 'fp16', 'bf16'), default='bf16')
    parser.add_argument(
        '--ep-size',
        type=int,
        default=None,
        help='EP size for ep_loop/ep_gmm. Defaults to the torchrun world size.',
    )
    parser.add_argument(
        '--disable-memory-efficient-init',
        action='store_true',
        help='Load the complete checkpoint in every process before FSDP wrapping.',
    )
    return parser.parse_args()


def _extract_logits(result: Any):
    import torch

    if hasattr(result, 'to_dict'):
        result = result.to_dict()
    elif hasattr(result, 'model_dump'):
        result = result.model_dump()
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], Mapping):
        result = result[0]
    if not isinstance(result, Mapping) or result.get('logits') is None:
        raise RuntimeError(f'forward_only did not return logits; result type={type(result).__name__}')

    logits = result['logits']
    if not isinstance(logits, torch.Tensor):
        logits = torch.as_tensor(logits)
    logits = logits.detach().to(dtype=torch.float32)
    original_shape = tuple(logits.shape)
    while logits.ndim > 3:
        logits = logits[0]
    if logits.ndim == 3:
        logits = logits[0, -1]
    elif logits.ndim == 2:
        logits = logits[-1]
    elif logits.ndim != 1:
        raise RuntimeError(f'Unsupported logits shape: {original_shape}')
    if logits.numel() < 1000:
        raise RuntimeError(f'Last-token logits look too small: shape={tuple(logits.shape)}')
    return logits.contiguous()


def _tensor_sha256(tensor) -> str:
    values = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


def _check_rank_consistency(last_logits):
    """Return the largest rank-to-rank difference without gathering Python objects."""
    import torch
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return 0.0

    rank0_logits = last_logits.clone()
    dist.broadcast(rank0_logits, src=0)
    max_diff = (last_logits - rank0_logits).abs().max()
    dist.all_reduce(max_diff, op=dist.ReduceOp.MAX)
    return float(max_diff.item())


def main() -> None:
    args = parse_args()
    if not args.model_id:
        raise SystemExit('Set DSV4_MODEL_ID or pass --model-id.')

    input_ids = [int(item.strip()) for item in args.input_ids.split(',') if item.strip()]
    if not input_ids:
        raise SystemExit('--input-ids must contain at least one token ID')

    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    enable_ep = args.mode != 'no_ep'
    ep_size = args.ep_size if args.ep_size is not None else (world_size if enable_ep else 1)
    if enable_ep and world_size <= 1:
        raise SystemExit(f'{args.mode} requires at least two torchrun processes; WORLD_SIZE={world_size}.')
    if enable_ep and (ep_size <= 1 or world_size % ep_size != 0):
        raise SystemExit(f'Invalid EP topology: WORLD_SIZE={world_size}, ep_size={ep_size}.')
    if not enable_ep and args.ep_size not in (None, 1):
        raise SystemExit('no_ep only supports --ep-size 1.')

    # Kernel selection is read while EP patches are installed.  Set it before
    # importing Twinkle model modules so CLI and server modes choose the same
    # implementation.
    os.environ['TWINKLE_EP_FORCE_LOOP'] = '1' if args.mode == 'ep_loop' else '0'
    os.environ.setdefault('TWINKLE_EP_DIAGNOSTICS', '1')
    os.environ.setdefault('TWINKLE_TRUST_REMOTE_CODE', '1')

    import torch
    import torch.distributed as dist
    from transformers import AutoConfig

    import twinkle
    from twinkle import DeviceMesh, Platform
    from twinkle.model import TransformersModel

    device_mesh = DeviceMesh.from_sizes(
        fsdp_size=world_size,
        dp_size=1,
        ep_size=ep_size,
        device_type=Platform.get_platform().device_prefix(),
    )
    twinkle.initialize(
        mode='local',
        global_device_mesh=device_mesh,
        seed=args.seed,
        lazy_collect=False,
    )

    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    model = TransformersModel(
        model_id=args.model_id,
        config=config,
        device_mesh=device_mesh,
        strategy='native_fsdp',
        mixed_precision=args.mixed_precision,
        memory_efficient_init=not args.disable_memory_efficient_init,
        fsdp_config={
            'reshard_after_forward': True,
            'expert_parallel': {
                'enabled': enable_ep,
                'ep_size': ep_size,
                'router_dtype': 'fp32',
                'keep_router_logits': False,
            },
        },
    )
    model.set_processor('InputProcessor', padding_side='left', padding_free=False)

    raw_input = {
        'input_ids': input_ids,
        'attention_mask': [1] * len(input_ids),
        'position_ids': list(range(len(input_ids))),
    }
    response = model.forward_only(
        inputs=[raw_input],
        return_logits=True,
    )
    last_logits = _extract_logits(response)
    max_rank_diff = _check_rank_consistency(last_logits)
    last_logits_cpu = last_logits.cpu()

    if rank == 0:
        finite = torch.isfinite(last_logits_cpu)
        top_values, top_indices = torch.topk(last_logits_cpu, k=min(20, last_logits_cpu.numel()))
        report = {
            'execution': 'cli',
            'mode': args.mode,
            'model_id': str(Path(args.model_id).expanduser()),
            'world_size': world_size,
            'ep_size': ep_size,
            'memory_efficient_init': not args.disable_memory_efficient_init,
            'input_ids': input_ids,
            'last_logits_shape': list(last_logits_cpu.shape),
            'dtype_saved': str(last_logits_cpu.dtype),
            'sha256': _tensor_sha256(last_logits_cpu),
            'max_rank_diff': max_rank_diff,
            'finite': bool(finite.all().item()),
            'nan_count': int(torch.isnan(last_logits_cpu).sum().item()),
            'inf_count': int(torch.isinf(last_logits_cpu).sum().item()),
            'sum': last_logits_cpu.sum().item(),
            'abs_sum': last_logits_cpu.abs().sum().item(),
            'min': last_logits_cpu.min().item(),
            'max': last_logits_cpu.max().item(),
            'top_token_ids': top_indices.tolist(),
            'top_logits': top_values.tolist(),
        }

        args.output_dir.mkdir(parents=True, exist_ok=True)
        tensor_path = args.output_dir / f'cli_{args.mode}_last_logits.pt'
        json_path = args.output_dir / f'cli_{args.mode}_last_logits.json'
        torch.save(
            {
                'execution': 'cli',
                'mode': args.mode,
                'input_ids': input_ids,
                'last_logits': last_logits_cpu,
            },
            tensor_path,
        )
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        print(f'Logits saved to: {tensor_path.resolve()}', flush=True)
        print(f'Report saved to: {json_path.resolve()}', flush=True)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == '__main__':
    main()
