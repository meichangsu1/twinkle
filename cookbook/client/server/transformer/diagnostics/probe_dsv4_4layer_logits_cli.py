#!/usr/bin/env python3
"""Run a four-layer DeepSeek-V4 logits probe directly with torchrun.

This is the local/CLI counterpart of ``probe_dsv4_4layer_logits.py``.  It
constructs ``MultiLoraTransformersModel`` in every torchrun process and
therefore does not use GatewayServer, Ray Serve, sessions, tenants, or
server-side LoRA capacity.  A disabled diagnostic adapter is installed to
match the production initialization path.
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
TARGET_PARAMETERS = ['mlp.experts.gate_up_proj', 'mlp.experts.down_proj']


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
    parser.add_argument(
        '--dump-weight-fingerprints',
        action='store_true',
        help='Save local parameter and buffer fingerprints for every rank after FSDP/EP wrapping.',
    )
    parser.add_argument(
        '--weight-fingerprint-samples',
        type=int,
        default=4096,
        help='Maximum number of evenly spaced values sampled from each local tensor.',
    )
    parser.add_argument(
        '--weight-fingerprint-full-hash',
        action='store_true',
        help='Also calculate an exact SHA256 over every local parameter and buffer in bounded CPU chunks.',
    )
    parser.add_argument(
        '--weight-fingerprint-chunk-mib',
        type=int,
        default=64,
        help='Maximum target CPU chunk size used by --weight-fingerprint-full-hash.',
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


def _sample_positions(numel: int, count: int) -> list[int]:
    """Return deterministic, evenly spaced flat indices without float rounding."""
    if numel <= 0 or count <= 0:
        return []
    count = min(numel, count)
    if count == 1:
        return [0]
    return [index * (numel - 1) // (count - 1) for index in range(count)]


def _parameter_local_tensor(parameter):
    """Return the rank-local tensor for both regular Parameters and FSDP2 DTensors."""
    from torch.distributed.tensor import DTensor

    value = parameter.detach()
    if isinstance(value, DTensor):
        value = value.to_local()
    return value


def _tensor_distribution_metadata(tensor) -> dict[str, Any]:
    from torch.distributed.tensor import DTensor

    value = tensor.detach()
    if not isinstance(value, DTensor):
        return {
            'is_dtensor': False,
            'placements': None,
            'device_mesh_shape': None,
            'device_mesh_dim_names': None,
        }
    mesh = value.device_mesh
    mesh_tensor = getattr(mesh, 'mesh', None)
    mesh_dim_names = getattr(mesh, 'mesh_dim_names', None)
    return {
        'is_dtensor': True,
        'placements': [repr(placement) for placement in value.placements],
        'device_mesh_shape': list(mesh_tensor.shape) if mesh_tensor is not None else None,
        'device_mesh_dim_names': list(mesh_dim_names) if mesh_dim_names is not None else None,
    }


def _sample_tensor_at_flat_positions(tensor, positions: list[int]):
    """Sample logical flat positions without flattening/copying the full tensor."""
    import torch

    if not positions:
        return torch.empty(0, dtype=tensor.dtype, device=tensor.device)
    if tensor.ndim == 0:
        return tensor.reshape(1)

    remaining = positions
    coordinate_columns = []
    for size in reversed(tensor.shape):
        coordinate_columns.append([position % size for position in remaining])
        remaining = [position // size for position in remaining]
    coordinates = tuple(
        torch.tensor(column, dtype=torch.long, device=tensor.device)
        for column in reversed(coordinate_columns)
    )
    return tensor[coordinates]


def _full_tensor_sha256(tensor, chunk_mib: int) -> str:
    """Hash a logical tensor in bounded chunks without copying it all to CPU."""
    import torch

    digest = hashlib.sha256()
    if tensor.numel() == 0:
        return digest.hexdigest()
    if tensor.ndim == 0:
        cpu_chunk = tensor.detach().cpu().contiguous().reshape(1)
        digest.update(cpu_chunk.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    row_numel = 1
    for size in tensor.shape[1:]:
        row_numel *= int(size)
    row_bytes = max(1, row_numel * tensor.element_size())
    rows_per_chunk = max(1, (chunk_mib * 1024 * 1024) // row_bytes)
    for start in range(0, int(tensor.shape[0]), rows_per_chunk):
        length = min(rows_per_chunk, int(tensor.shape[0]) - start)
        cpu_chunk = tensor.narrow(0, start, length).detach().cpu().contiguous()
        digest.update(cpu_chunk.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _fingerprint_tensor(tensor, sample_limit: int, *, full_hash: bool, chunk_mib: int) -> dict[str, Any]:
    import torch

    local = _parameter_local_tensor(tensor)
    local_is_meta = bool(getattr(local, 'is_meta', False))
    positions = [] if local_is_meta else _sample_positions(local.numel(), sample_limit)
    if local_is_meta:
        sample_sha256 = None
        sample_values = []
        sample_finite = None
        sample_min = None
        sample_max = None
        sample_sum = None
        sample_abs_sum = None
    elif positions:
        sampled = _sample_tensor_at_flat_positions(local, positions).detach().cpu().contiguous()
        # Hash the original dtype bytes. Converting BF16 directly to NumPy is
        # not supported by every NumPy version, whereas a byte view is.
        sample_sha256 = hashlib.sha256(sampled.view(torch.uint8).numpy().tobytes()).hexdigest()
        sampled_float = sampled.float()
        sample_values = sampled_float.tolist()
        sample_finite = bool(torch.isfinite(sampled_float).all().item())
        sample_min = float(sampled_float.min().item())
        sample_max = float(sampled_float.max().item())
        sample_sum = float(sampled_float.sum().item())
        sample_abs_sum = float(sampled_float.abs().sum().item())
    else:
        sample_sha256 = hashlib.sha256(b'').hexdigest()
        sample_values = []
        sample_finite = True
        sample_min = None
        sample_max = None
        sample_sum = 0.0
        sample_abs_sum = 0.0

    result = {
        'global_shape': list(tensor.shape),
        'global_stride': list(tensor.stride()),
        'local_shape': list(local.shape),
        'local_stride': list(local.stride()),
        'dtype': str(local.dtype),
        'local_numel': local.numel(),
        'local_is_contiguous': local.is_contiguous(),
        'local_is_meta': local_is_meta,
        'sample_count': len(positions),
        'sample_sha256': sample_sha256,
        'sample_finite': sample_finite,
        'sample_sum': sample_sum,
        'sample_abs_sum': sample_abs_sum,
        'sample_min': sample_min,
        'sample_max': sample_max,
        'sample_values': sample_values,
        'full_sha256': _full_tensor_sha256(local, chunk_mib) if full_hash and not local_is_meta else None,
    }
    result.update(_tensor_distribution_metadata(tensor))
    return result


def _write_weight_fingerprints(model, args: argparse.Namespace, *, rank: int, world_size: int) -> Path:
    """Write rank-local post-wrap base-weight fingerprints to a JSON file."""
    import torch

    inner_model = getattr(model, 'model', model)
    tensors = []
    with torch.no_grad():
        for name, parameter in inner_model.named_parameters():
            record = {'kind': 'parameter', 'name': name}
            record.update(
                _fingerprint_tensor(
                    parameter,
                    args.weight_fingerprint_samples,
                    full_hash=args.weight_fingerprint_full_hash,
                    chunk_mib=args.weight_fingerprint_chunk_mib,
                ))
            tensors.append(record)
        for name, buffer in inner_model.named_buffers():
            record = {'kind': 'buffer', 'name': name}
            record.update(
                _fingerprint_tensor(
                    buffer,
                    args.weight_fingerprint_samples,
                    full_hash=args.weight_fingerprint_full_hash,
                    chunk_mib=args.weight_fingerprint_chunk_mib,
                ))
            tensors.append(record)

    payload = {
        'execution': 'cli',
        'mode': args.mode,
        'model_id': str(Path(args.model_id).expanduser()),
        'rank': rank,
        'world_size': world_size,
        'memory_efficient_init': not args.disable_memory_efficient_init,
        'sample_algorithm': 'flat_evenly_spaced_integer_v1',
        'sample_limit_per_parameter': args.weight_fingerprint_samples,
        'full_hash': args.weight_fingerprint_full_hash,
        'full_hash_chunk_mib': args.weight_fingerprint_chunk_mib,
        'parameter_count': sum(record['kind'] == 'parameter' for record in tensors),
        'buffer_count': sum(record['kind'] == 'buffer' for record in tensors),
        'tensor_count': len(tensors),
        'tensors': tensors,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f'cli_{args.mode}_weight_fingerprints_rank{rank}.json'
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return output


def main() -> None:
    args = parse_args()
    if not args.model_id:
        raise SystemExit('Set DSV4_MODEL_ID or pass --model-id.')

    input_ids = [int(item.strip()) for item in args.input_ids.split(',') if item.strip()]
    if not input_ids:
        raise SystemExit('--input-ids must contain at least one token ID')
    if args.weight_fingerprint_samples <= 0:
        raise SystemExit('--weight-fingerprint-samples must be positive')
    if args.weight_fingerprint_chunk_mib <= 0:
        raise SystemExit('--weight-fingerprint-chunk-mib must be positive')

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
    from peft import LoraConfig
    from transformers import AutoConfig

    import twinkle
    from twinkle import DeviceMesh, Platform
    from twinkle.model import MultiLoraTransformersModel

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
    adapter_name = f'dsv4_diag_{args.mode}'
    model = MultiLoraTransformersModel(
        model_id=args.model_id,
        config=config,
        device_mesh=device_mesh,
        strategy='native_fsdp',
        mixed_precision=args.mixed_precision,
        memory_efficient_init=not args.disable_memory_efficient_init,
        max_loras=1,
        max_r=8,
        max_length=512,
        target_modules='all-linear',
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
    model.add_adapter_to_model(
        adapter_name,
        LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=None,
            target_parameters=TARGET_PARAMETERS,
            bias='none',
        ),
        gradient_accumulation_steps=1,
    )
    model.set_processor(
        'InputProcessor',
        adapter_name=adapter_name,
        padding_side='left',
        padding_free=False,
    )

    raw_input = {
        'input_ids': input_ids,
        'attention_mask': [1] * len(input_ids),
        'position_ids': list(range(len(input_ids))),
    }
    response = model.forward_only(
        inputs=[raw_input],
        adapter_name=adapter_name,
        disable_lora=True,
        return_logits=True,
    )
    last_logits = _extract_logits(response)
    fingerprint_path = None
    if args.dump_weight_fingerprints:
        fingerprint_path = _write_weight_fingerprints(
            model,
            args,
            rank=rank,
            world_size=world_size,
        )
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

    if fingerprint_path is not None:
        print(f'Rank {rank} weight fingerprints saved to: {fingerprint_path.resolve()}', flush=True)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == '__main__':
    main()
