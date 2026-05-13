import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict

import torch
import torch.distributed as dist
from torch.distributed.checkpoint import save as dcp_save
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from twinkle.hub import HubOperation
from twinkle.model.transformers.strategy.native_fsdp import NativeFSDPStrategy
from twinkle.utils import DeviceMesh, Platform, torch_util

print(
    f'[worker-bootstrap] pid={os.getpid()} RANK={os.environ.get("RANK")} '
    f'LOCAL_RANK={os.environ.get("LOCAL_RANK")} WORLD_SIZE={os.environ.get("WORLD_SIZE")}',
    flush=True,
)


def _rank_prefix() -> str:
    if dist.is_available() and dist.is_initialized():
        return f'[rank{dist.get_rank()}]'
    rank = Platform.get_rank()
    return f'[rank{rank if rank >= 0 else 0}]'


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert a HuggingFace checkpoint to a DCP checkpoint. '
        'Use torchrun when target-world-size is greater than 1.'
    )
    parser.add_argument('--model-id', required=True, help='HF/local model path or Twinkle hub model id.')
    parser.add_argument('--output-dir', required=True, help='Output directory for converted checkpoint artifacts.')
    parser.add_argument(
        '--dcp-subdir',
        default='model_distcp',
        help='Subdirectory under output-dir used to store DCP files.',
    )
    parser.add_argument(
        '--torch-dtype',
        default='auto',
        choices=('auto', 'float32', 'float16', 'bfloat16'),
        help='Optional dtype override passed to from_pretrained.',
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        default=None,
        help='Optional override for config.num_hidden_layers, useful for debug conversion.',
    )
    parser.add_argument(
        '--trust-remote-code',
        action='store_true',
        default=False,
        help='Enable trust_remote_code when loading config/model/tokenizer.',
    )
    parser.add_argument(
        '--target-world-size',
        type=int,
        default=int(os.environ.get('WORLD_SIZE', '1')),
        help='Target world size for the output DCP checkpoint. Values greater than 1 require torchrun.',
    )
    parser.add_argument(
        '--single-process',
        action='store_true',
        default=False,
        help='Explicitly allow single-process full-state DCP conversion.',
    )
    parser.add_argument(
        '--safe-serialization',
        action='store_true',
        default=False,
        help='Also save a HF-format reference checkpoint under output-dir/hf_reference.',
    )
    return parser.parse_args()


def _resolve_model_dir(model_id: str) -> str:
    if os.path.exists(model_id):
        return model_id
    return HubOperation.download_model(model_id)


def _resolve_torch_dtype(dtype_name: str):
    if dtype_name == 'auto':
        return 'auto'
    return getattr(torch, dtype_name)


def _save_conversion_metadata(
    output_dir: str,
    *,
    source_model_dir: str,
    source_model_id: str,
    dcp_subdir: str,
    torch_dtype: str,
    num_layers: int | None,
    target_world_size: int,
    mode: str,
) -> None:
    metadata: Dict[str, Any] = {
        'source_model_id': source_model_id,
        'source_model_dir': source_model_dir,
        'dcp_subdir': dcp_subdir,
        'torch_dtype': torch_dtype,
        'num_layers': num_layers,
        'target_world_size': target_world_size,
        'mode': mode,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(output_dir, 'conversion_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)


def _load_config(source_model_dir: str, args: argparse.Namespace):
    config = AutoConfig.from_pretrained(source_model_dir, trust_remote_code=args.trust_remote_code)
    if args.num_layers is not None and hasattr(config, 'num_hidden_layers'):
        original_num_layers = config.num_hidden_layers
        config.num_hidden_layers = args.num_layers
        for name, value in vars(config).items():
            if isinstance(value, (list, tuple)) and len(value) == original_num_layers:
                setattr(config, name, list(value)[:args.num_layers])
    if hasattr(config, 'use_cache'):
        config.use_cache = False
    return config


def _init_process_group_if_needed(target_world_size: int) -> None:
    if target_world_size <= 1:
        return
    if dist.is_initialized():
        if dist.get_world_size() != target_world_size:
            raise RuntimeError(
                f'target-world-size={target_world_size} but initialized world_size={dist.get_world_size()}.')
        return
    if Platform.get_world_size() != target_world_size:
        raise RuntimeError(
            f'target-world-size={target_world_size} requires torchrun with WORLD_SIZE={target_world_size}. '
            f'Current WORLD_SIZE={Platform.get_world_size()}.')
    torch_util.set_device()
    init_kwargs = {
        'backend': Platform.device_backend(),
        'init_method': 'env://',
        'rank': Platform.get_rank(),
        'world_size': Platform.get_world_size(),
    }
    if Platform.device_backend() in ('nccl', 'hccl'):
        init_kwargs['device_id'] = torch.device(Platform.get_local_device())
    dist.init_process_group(**init_kwargs)


def _init_empty_model(config, trust_remote_code: bool):
    from accelerate import init_empty_weights

    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
    if hasattr(model, 'tie_weights'):
        model.tie_weights()
    return model


def _build_device_mesh(target_world_size: int) -> DeviceMesh:
    return DeviceMesh.from_sizes(
        world_size=target_world_size,
        fsdp_size=target_world_size,
        dp_size=1,
        device_type=Platform.device_prefix(),
    )


def _save_full_dcp(model, dcp_dir: str) -> None:
    state_dict = {'model': model.state_dict()}
    dcp_save(
        state_dict=state_dict,
        checkpoint_id=dcp_dir,
        no_dist=True,
        use_collectives=False,
    )


def _save_sharded_dcp(model, dcp_dir: str) -> None:
    sharded_state_dict = get_model_state_dict(
        model,
        options=StateDictOptions(
            full_state_dict=False,
            cpu_offload=True,
        ),
    )
    dcp_save(
        state_dict={'model': sharded_state_dict},
        checkpoint_id=dcp_dir,
    )


def main() -> None:
    args = _parse_args()
    if args.target_world_size <= 1 and not args.single_process:
        raise RuntimeError(
            'This script is intended for multi-process validation by default. '
            'Launch it with torchrun so WORLD_SIZE > 1, or pass --single-process explicitly.')
    os.makedirs(args.output_dir, exist_ok=True)

    source_model_dir = _resolve_model_dir(args.model_id)
    config = _load_config(source_model_dir, args)
    load_dtype = _resolve_torch_dtype(args.torch_dtype)

    if Platform.is_master():
        tokenizer = AutoTokenizer.from_pretrained(source_model_dir, trust_remote_code=args.trust_remote_code)
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)

    dcp_dir = os.path.join(args.output_dir, args.dcp_subdir)
    os.makedirs(dcp_dir, exist_ok=True)

    if args.target_world_size <= 1:
        print(f'{_rank_prefix()} converter.single_process.before_from_pretrained', flush=True)
        model_load_kwargs: Dict[str, Any] = {
            'config': config,
            'trust_remote_code': args.trust_remote_code,
            'low_cpu_mem_usage': True,
        }
        if load_dtype != 'auto':
            model_load_kwargs['torch_dtype'] = load_dtype
        model = AutoModelForCausalLM.from_pretrained(source_model_dir, **model_load_kwargs)
        model.eval()
        print(f'{_rank_prefix()} converter.single_process.after_from_pretrained', flush=True)
        print(f'{_rank_prefix()} converter.single_process.before_dcp_save', flush=True)
        _save_full_dcp(model, dcp_dir)
        print(f'{_rank_prefix()} converter.single_process.after_dcp_save', flush=True)
        if args.safe_serialization:
            hf_reference_dir = os.path.join(args.output_dir, 'hf_reference')
            model.save_pretrained(hf_reference_dir, safe_serialization=True)
        _save_conversion_metadata(
            args.output_dir,
            source_model_dir=source_model_dir,
            source_model_id=args.model_id,
            dcp_subdir=args.dcp_subdir,
            torch_dtype=args.torch_dtype,
            num_layers=args.num_layers,
            target_world_size=args.target_world_size,
            mode='full_state_single_process',
        )
        return

    _init_process_group_if_needed(args.target_world_size)
    print(f'{_rank_prefix()} converter.multi_process.after_init_process_group', flush=True)

    model_load_kwargs: Dict[str, Any] = {
        'config': config,
        'trust_remote_code': args.trust_remote_code,
        'low_cpu_mem_usage': True,
    }
    if load_dtype != 'auto':
        model_load_kwargs['torch_dtype'] = load_dtype

    if dist.get_rank() == 0:
        print(f'{_rank_prefix()} converter.before_from_pretrained_rank0', flush=True)
        model = AutoModelForCausalLM.from_pretrained(source_model_dir, **model_load_kwargs)
        print(f'{_rank_prefix()} converter.after_from_pretrained_rank0', flush=True)
    else:
        print(f'{_rank_prefix()} converter.before_init_empty_model', flush=True)
        model = _init_empty_model(config, args.trust_remote_code)
        print(f'{_rank_prefix()} converter.after_init_empty_model', flush=True)
    model.eval()

    device_mesh = _build_device_mesh(args.target_world_size)
    strategy = NativeFSDPStrategy(
        device_mesh=device_mesh,
        mixed_precision='bf16',
        memory_efficient_init=True,
        enable_ep=False,
    )
    print(f'{_rank_prefix()} converter.before_wrap_model', flush=True)
    model, _ = strategy.wrap_model(model)
    print(f'{_rank_prefix()} converter.after_wrap_model', flush=True)

    print(f'{_rank_prefix()} converter.before_dcp_save', flush=True)
    _save_sharded_dcp(model, dcp_dir)
    print(f'{_rank_prefix()} converter.after_dcp_save', flush=True)
    dist.barrier()

    if Platform.is_master():
        _save_conversion_metadata(
            args.output_dir,
            source_model_dir=source_model_dir,
            source_model_id=args.model_id,
            dcp_subdir=args.dcp_subdir,
            torch_dtype=args.torch_dtype,
            num_layers=args.num_layers,
            target_world_size=args.target_world_size,
            mode='sharded_target_world_size',
        )


if __name__ == '__main__':
    main()
