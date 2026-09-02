#!/usr/bin/env python3
"""Run the four-layer DeepSeek-V4 logits probe on a multi-node Ray cluster.

The script is a command-line Ray driver.  It creates distributed Twinkle model
actors directly and deliberately does not deploy Ray Serve, GatewayServer,
ModelManagement, sessions, tenants, or LoRA adapters.
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
    parser.add_argument('--ray-address', default=os.environ.get('RAY_ADDRESS', 'auto'))
    parser.add_argument('--world-size', type=int, default=4)
    parser.add_argument('--nproc-per-node', type=int, default=2)
    parser.add_argument('--output-dir', type=Path, default=Path('/nas/disk6/ljl/dsv4_ep_diag/results'))
    parser.add_argument('--input-ids', default=','.join(str(item) for item in DEFAULT_INPUT_IDS))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mixed-precision', choices=('no', 'fp16', 'bf16'), default='bf16')
    parser.add_argument(
        '--ep-size',
        type=int,
        default=None,
        help='EP size for ep_loop/ep_gmm. Defaults to world-size.',
    )
    parser.add_argument(
        '--disable-memory-efficient-init',
        action='store_true',
        help='Load the complete checkpoint in every actor before FSDP wrapping.',
    )
    return parser.parse_args()


def _extract_logits(result: Any):
    import torch

    if callable(result):
        result = result()
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
    logits = logits.detach().cpu().to(dtype=torch.float32)
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


def _validate_topology(args: argparse.Namespace, ray) -> None:
    required_nodes = args.world_size // args.nproc_per_node
    alive_nodes = [node for node in ray.nodes() if node.get('Alive', True)]
    npu_nodes = [
        node for node in alive_nodes
        if int(node.get('Resources', {}).get('NPU', 0)) >= args.nproc_per_node
    ]
    total_npus = int(ray.cluster_resources().get('NPU', 0))
    if total_npus < args.world_size or len(npu_nodes) < required_nodes:
        raise RuntimeError(
            'Ray cluster does not have the requested NPU topology: '
            f'required world_size={args.world_size}, nodes={required_nodes}, '
            f'nproc_per_node={args.nproc_per_node}; found NPU={total_npus}, '
            f'eligible_nodes={len(npu_nodes)}. Start every Ray node with '
            f'--resources=\'{{"NPU": {args.nproc_per_node}}}\'.')


def main() -> None:
    args = parse_args()
    if not args.model_id:
        raise SystemExit('Set DSV4_MODEL_ID or pass --model-id.')
    if args.world_size <= 0 or args.nproc_per_node <= 0:
        raise SystemExit('--world-size and --nproc-per-node must be positive.')
    if args.world_size % args.nproc_per_node != 0:
        raise SystemExit('--world-size must be divisible by --nproc-per-node.')

    input_ids = [int(item.strip()) for item in args.input_ids.split(',') if item.strip()]
    if not input_ids:
        raise SystemExit('--input-ids must contain at least one token ID')

    enable_ep = args.mode != 'no_ep'
    ep_size = args.ep_size if args.ep_size is not None else (args.world_size if enable_ep else 1)
    if enable_ep and (ep_size <= 1 or args.world_size % ep_size != 0):
        raise SystemExit(f'Invalid EP topology: world_size={args.world_size}, ep_size={ep_size}.')
    if not enable_ep and args.ep_size not in (None, 1):
        raise SystemExit('no_ep only supports --ep-size 1.')

    # RayHelper copies the driver's environment into each model actor.  Set EP
    # selection before Twinkle creates the placement groups and runtime_envs.
    os.environ['TWINKLE_EP_FORCE_LOOP'] = '1' if args.mode == 'ep_loop' else '0'
    os.environ.setdefault('TWINKLE_EP_DIAGNOSTICS', '1')
    os.environ.setdefault('TWINKLE_TRUST_REMOTE_CODE', '1')
    os.environ.setdefault('RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES', '1')

    import ray
    import torch

    ray.init(address=args.ray_address, ignore_reinit_error=True)
    twinkle_initialized = False
    try:
        _validate_topology(args, ray)

        import twinkle
        from twinkle import DeviceGroup, DeviceMesh
        from twinkle.model import TransformersModel

        device_mesh = DeviceMesh.from_sizes(
            fsdp_size=args.world_size,
            dp_size=1,
            ep_size=ep_size,
            device_type='npu',
        )
        groups = [
            DeviceGroup(
                name='model',
                ranks=list(range(args.world_size)),
                device_type='NPU',
            )
        ]
        twinkle.initialize(
            mode='ray',
            nproc_per_node=args.nproc_per_node,
            ncpu_proc_per_node=1,
            seed=args.seed,
            groups=groups,
            global_device_mesh=device_mesh,
            lazy_collect=False,
        )
        twinkle_initialized = True

        model = TransformersModel(
            model_id=args.model_id,
            device_mesh=device_mesh,
            remote_group='model',
            instance_id=f'dsv4_4layer_ray_{args.mode}',
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

        finite = torch.isfinite(last_logits)
        top_values, top_indices = torch.topk(last_logits, k=min(20, last_logits.numel()))
        report = {
            'execution': 'ray_cli',
            'mode': args.mode,
            'ray_address': args.ray_address,
            'model_id': args.model_id,
            'world_size': args.world_size,
            'nproc_per_node': args.nproc_per_node,
            'ep_size': ep_size,
            'memory_efficient_init': not args.disable_memory_efficient_init,
            'input_ids': input_ids,
            'last_logits_shape': list(last_logits.shape),
            'dtype_saved': str(last_logits.dtype),
            'sha256': _tensor_sha256(last_logits),
            'finite': bool(finite.all().item()),
            'nan_count': int(torch.isnan(last_logits).sum().item()),
            'inf_count': int(torch.isinf(last_logits).sum().item()),
            'sum': last_logits.sum().item(),
            'abs_sum': last_logits.abs().sum().item(),
            'min': last_logits.min().item(),
            'max': last_logits.max().item(),
            'top_token_ids': top_indices.tolist(),
            'top_logits': top_values.tolist(),
        }

        args.output_dir.mkdir(parents=True, exist_ok=True)
        tensor_path = args.output_dir / f'ray_{args.mode}_last_logits.pt'
        json_path = args.output_dir / f'ray_{args.mode}_last_logits.json'
        torch.save(
            {
                'execution': 'ray_cli',
                'mode': args.mode,
                'input_ids': input_ids,
                'last_logits': last_logits,
            },
            tensor_path,
        )
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        print(f'Logits saved to: {tensor_path.resolve()}', flush=True)
        print(f'Report saved to: {json_path.resolve()}', flush=True)
    finally:
        if twinkle_initialized:
            from twinkle.infra._ray import RayHelper

            RayHelper.teardown()
        ray.shutdown()


if __name__ == '__main__':
    main()
