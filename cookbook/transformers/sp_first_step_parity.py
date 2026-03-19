import hashlib
import json
import os
import random
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from peft import LoraConfig
from torch.utils.data import DataLoader as TorchDataLoader

import twinkle
from twinkle import DeviceMesh, Platform, get_logger, torch_util
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor
# TWINKLE_PARITY_ULYSSES_SIZE=1 \
# TWINKLE_PARITY_OUTPUT=/tmp/sp0_first_step.json \
# torchrun --standalone --nproc_per_node=4 -m cookbook.transformers.sp_first_step_parity

# TWINKLE_PARITY_ULYSSES_SIZE=2 \
# TWINKLE_PARITY_OUTPUT=/tmp/sp2_first_step.json \
# torchrun --standalone --nproc_per_node=4 -m cookbook.transformers.sp_first_step_parity
# QWEN35_SP_LINEAR_STRICT=1 \
# TWINKLE_PARITY_WRAP_MODE=ddp \
# TWINKLE_PARITY_ULYSSES_SIZE=2 \
# torchrun --standalone --nproc_per_node=4 -m cookbook.transformers.sp_first_step_parity
# QWEN35_SP_LINEAR_STRICT=1 \
# QWEN35_SP_LINEAR_STRICT_TRACE=1 \
# TWINKLE_PARITY_WRAP_MODE=ddp \
# TWINKLE_PARITY_ULYSSES_SIZE=2 \
# TWINKLE_PARITY_BATCH_SYNC_MODE=local \
# torchrun --standalone --nproc_per_node=4 -m cookbook.transformers.sp_first_step_parity


# QWEN35_SP_LINEAR_HEAD_PARALLEL=1 \
# TWINKLE_PARITY_WRAP_MODE=ddp \
# TWINKLE_PARITY_ULYSSES_SIZE=2 \
# TWINKLE_PARITY_DISABLE_GC=0 \
# torchrun --standalone --nproc_per_node=4 -m cookbook.transformers.sp_first_step_parity

# TWINKLE_PARITY_BATCH_SYNC_MODE=local

# QWEN35_SP_LINEAR_HEAD_PARALLEL=1 \
# TWINKLE_PARITY_WRAP_MODE=ddp \
# TWINKLE_PARITY_ULYSSES_SIZE=2 \
# TWINKLE_PARITY_DISABLE_GC=0 \
# TWINKLE_PARITY_OUTPUT=/tmp/qwen35_head_parallel.json \
# torchrun --standalone --nproc_per_node=4 -m cookbook.transformers.sp_first_step_parity



logger = get_logger()

MODEL_ID = os.environ.get('TWINKLE_MODEL_ID', 'ms://Qwen/Qwen3.5-0.8B')
DATASETS = os.environ.get('TWINKLE_DATASETS', 'ms://swift/self-cognition')
ULYSSES_SIZE = int(os.environ.get('TWINKLE_PARITY_ULYSSES_SIZE', '1'))
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', os.environ.get('TWINKLE_PARITY_WORLD_SIZE', '4')))
LOCAL_WORLD_SIZE = int(
    os.environ.get('LOCAL_WORLD_SIZE', os.environ.get('TWINKLE_PARITY_LOCAL_WORLD_SIZE', str(WORLD_SIZE))))
GLOBAL_BATCH_SIZE = int(os.environ.get('TWINKLE_PARITY_GLOBAL_BATCH_SIZE', '8'))
PARITY_SEED = int(os.environ.get('TWINKLE_PARITY_SEED', '42'))
PARITY_STRICT_DETERMINISM = os.environ.get('TWINKLE_PARITY_STRICT_DETERMINISM', '1') == '1'
PARITY_MIXED_PRECISION = os.environ.get('TWINKLE_PARITY_MIXED_PRECISION', 'no')
PARITY_ATTN_IMPL = os.environ.get('TWINKLE_PARITY_ATTN_IMPL', 'eager')
PARITY_DISABLE_GC = os.environ.get('TWINKLE_PARITY_DISABLE_GC', '1') == '1'
PARITY_WRAP_MODE = os.environ.get('TWINKLE_PARITY_WRAP_MODE', 'native_fsdp')
PARITY_BATCH_SYNC_MODE = os.environ.get('TWINKLE_PARITY_BATCH_SYNC_MODE', 'auto').strip().lower()


def _device_backend_prefix() -> str:
    return Platform.device_prefix()


def _cuda_available() -> bool:
    return _device_backend_prefix() == 'cuda' and hasattr(torch, 'cuda') and torch.cuda.is_available()


def _npu_available() -> bool:
    return _device_backend_prefix() == 'npu' and hasattr(torch, 'npu') and torch.npu.is_available()


def _enable_strict_determinism(seed: int) -> None:
    os.environ.setdefault('PYTHONHASHSEED', str(seed))
    if _cuda_available():
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':16:8')
        os.environ.setdefault('NCCL_DETERMINISTIC', '1')
        os.environ.setdefault('FLASH_ATTENTION_DETERMINISTIC', '1')
        os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        if hasattr(torch.backends.cuda.matmul, 'allow_bf16_reduced_precision_reduction'):
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    elif _npu_available():
        os.environ.setdefault('HCCL_DETERMINISTIC', 'true')
        os.environ.setdefault('ASCEND_LAUNCH_BLOCKING', '1')
    torch.use_deterministic_algorithms(True, warn_only=True)


if PARITY_STRICT_DETERMINISM:
    _enable_strict_determinism(PARITY_SEED)

def _build_device_mesh() -> DeviceMesh:
    if PARITY_WRAP_MODE == 'native_fsdp':
        if WORLD_SIZE < 2 or WORLD_SIZE % 2 != 0:
            raise ValueError('TWINKLE_PARITY_WRAP_MODE=native_fsdp requires an even WORLD_SIZE >= 2.')
        return DeviceMesh(
            device_type=_device_backend_prefix(),
            mesh=np.arange(WORLD_SIZE).reshape(WORLD_SIZE // 2, 2),
            mesh_dim_names=('dp', 'fsdp'),
            ulysses_size=ULYSSES_SIZE,
        )
    if PARITY_WRAP_MODE == 'ddp':
        return DeviceMesh(
            device_type=_device_backend_prefix(),
            mesh=np.arange(WORLD_SIZE),
            mesh_dim_names=('dp',),
            ulysses_size=ULYSSES_SIZE,
        )
    raise ValueError(f'Unsupported TWINKLE_PARITY_WRAP_MODE={PARITY_WRAP_MODE}')


device_mesh = _build_device_mesh()

twinkle.initialize(
    mode='local',
    nproc_per_node=LOCAL_WORLD_SIZE,
    global_device_mesh=device_mesh,
    lazy_collect=False,
)


def create_dataset(data_slice=None):
    dataset = Dataset(dataset_meta=DatasetMeta(DATASETS, data_slice=data_slice or range(500)))
    dataset.set_template('Template', model_id=MODEL_ID)
    dataset.map(SelfCognitionProcessor('twinkle模型', 'twinkle团队'))
    dataset.encode(batched=True)
    return dataset


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if _cuda_available():
        torch.cuda.manual_seed_all(seed)
    elif _npu_available():
        torch.npu.manual_seed_all(seed)


def _create_model(num_training_steps: int) -> TransformersModel:
    warmup_steps = int(os.environ.get('TWINKLE_PARITY_WARMUP_STEPS', '0'))
    strategy = 'native_fsdp' if PARITY_WRAP_MODE == 'native_fsdp' else 'accelerate'
    model = TransformersModel(
        model_id=MODEL_ID,
        device_mesh=device_mesh,
        strategy=strategy,
        mixed_precision=PARITY_MIXED_PRECISION,
        attn_implementation=PARITY_ATTN_IMPL,
    )
    if hasattr(model.model, 'config') and model.model.config is not None:
        for attr in ('_attn_implementation', '_attn_implementation_internal'):
            if hasattr(model.model.config, attr):
                setattr(model.model.config, attr, PARITY_ATTN_IMPL)
    if PARITY_DISABLE_GC and hasattr(model.model, 'gradient_checkpointing_disable'):
        model.model.gradient_checkpointing_disable()
        if hasattr(model.model, 'config') and model.model.config is not None and hasattr(model.model.config, 'use_cache'):
            model.model.config.use_cache = False
    lora_config = LoraConfig(target_modules='all-linear', lora_dropout=0.0)
    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=1)
    model.set_optimizer('AdamW', lr=1e-4, adapter_name='default')
    model.set_lr_scheduler(
        scheduler_cls='CosineWarmupScheduler',
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
        adapter_name='default',
    )
    return model


def _ensure_process_group() -> None:
    if dist.is_available() and dist.is_initialized():
        return
    if Platform.get_world_size() <= 1:
        return
    torch_util.set_device()
    backend = Platform.device_backend()
    init_kwargs = {
        'backend': backend,
        'init_method': 'env://',
        'rank': Platform.get_rank(),
        'world_size': Platform.get_world_size(),
    }
    if backend in ('nccl', 'hccl'):
        init_kwargs['device_id'] = torch.device(Platform.get_local_device())
    dist.init_process_group(**init_kwargs)


def _maybe_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _tensor_hash(tensor: torch.Tensor) -> str:
    tensor = torch_util.to_local_tensor(tensor).detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _update_hash(hasher: 'hashlib._Hash', value: Any) -> None:
    if isinstance(value, torch.Tensor):
        local_tensor = torch_util.to_local_tensor(value).detach().cpu().contiguous()
        hasher.update(str(tuple(local_tensor.shape)).encode())
        hasher.update(str(local_tensor.dtype).encode())
        hasher.update(local_tensor.numpy().tobytes())
        return
    if isinstance(value, dict):
        for key in sorted(value.keys()):
            hasher.update(str(key).encode())
            _update_hash(hasher, value[key])
        return
    if isinstance(value, list):
        hasher.update(str(len(value)).encode())
        for item in value:
            _update_hash(hasher, item)
        return
    hasher.update(repr(value).encode())


def _batch_digest(batch: list[dict[str, Any]]) -> dict[str, Any]:
    hasher = hashlib.sha256()
    _update_hash(hasher, batch)

    digest = {
        'num_samples': int(len(batch)),
        'sha256': hasher.hexdigest(),
    }
    if batch and isinstance(batch[0], dict):
        sample = batch[0]
        sample_summary = {}
        for key in sorted(sample.keys()):
            value = sample[key]
            if isinstance(value, torch.Tensor):
                value = torch_util.to_local_tensor(value)
                sample_summary[key] = {
                    'shape': [int(x) for x in value.shape],
                    'dtype': str(value.dtype),
                    'sum': float(value.detach().float().sum().item()),
                    'hash': _tensor_hash(value),
                }
            else:
                sample_summary[key] = repr(value)
        digest['first_sample'] = sample_summary
    return digest


def _clone_state_dict(state_dict):
    cloned = {}
    for key, value in state_dict.items():
        value = torch_util.to_local_tensor(value)
        cloned[key] = value.detach().float().cpu().clone()
    return cloned


def _state_delta_stats(before, after):
    total_sq = 0.0
    total_count = 0
    max_abs = 0.0
    max_key = None
    for key, before_tensor in before.items():
        after_tensor = after.get(key)
        if after_tensor is None:
            continue
        delta = after_tensor.detach().float().cpu() - before_tensor
        if delta.numel() == 0:
            continue
        local_max = float(delta.abs().max().item())
        if local_max > max_abs:
            max_abs = local_max
            max_key = key
        total_sq += float(delta.pow(2).sum().item())
        total_count += delta.numel()
    l2 = total_sq**0.5 if total_sq > 0 else 0.0
    rms = (total_sq / total_count)**0.5 if total_count > 0 else 0.0
    return {
        'param_delta_l2': l2,
        'param_delta_rms': rms,
        'param_delta_max_abs': max_abs,
        'param_delta_max_abs_key': max_key,
    }


def _resolve_batch_sync_mode() -> str:
    if PARITY_BATCH_SYNC_MODE in ('broadcast', 'local'):
        return PARITY_BATCH_SYNC_MODE
    if _npu_available():
        return 'local'
    return 'broadcast'


def _load_global_batch_locally(batch_index: int) -> list[dict[str, Any]]:
    dataset = create_dataset(data_slice=None)
    dataloader = TorchDataLoader(
        dataset,
        batch_size=GLOBAL_BATCH_SIZE,
        num_workers=0,
        shuffle=False,
        collate_fn=lambda x: x,
    )
    for idx, _batch in enumerate(dataloader):
        if idx == batch_index:
            if not isinstance(_batch, list):
                raise TypeError(f'Expected list batch, got: {type(_batch)}')
            return _batch
    raise IndexError(f'Batch index {batch_index} out of range')


def _get_global_batch(batch_index: int) -> list[dict[str, Any]]:
    sync_mode = _resolve_batch_sync_mode()
    if not dist.is_available() or not dist.is_initialized():
        return _load_global_batch_locally(batch_index)

    if sync_mode == 'local':
        if Platform.get_rank() == 0:
            _load_global_batch_locally(batch_index)
        _maybe_barrier()
        return _load_global_batch_locally(batch_index)

    batch = None
    if Platform.get_rank() == 0:
        batch = _load_global_batch_locally(batch_index)
    object_list = [batch]
    dist.broadcast_object_list(object_list, src=0)
    return object_list[0]


def _slice_local_batch(global_batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    batch_slice = device_mesh.get_slice(len(global_batch))
    return global_batch[batch_slice]


def _run_first_train_step(model: TransformersModel, local_batch, init_state):
    model.zero_grad(adapter_name='default')
    outputs = model.forward(inputs=local_batch, adapter_name='default')
    loss = model.calculate_loss(adapter_name='default')
    model.backward(adapter_name='default')
    grad_norm = model.clip_grad_norm(adapter_name='default')
    model.step(adapter_name='default')
    metric = model.calculate_metric(is_training=True, adapter_name='default')
    final_state = _clone_state_dict(model.get_state_dict(adapter_name='default'))
    return {
        'loss': float(loss),
        'grad_norm': float(grad_norm) if grad_norm is not None else None,
        'metric': metric,
        'state_delta': _state_delta_stats(init_state, final_state),
        'local_batch_size': len(local_batch),
        'local_batch_digest': _batch_digest(local_batch),
        'returned_output_keys': sorted(outputs.keys()) if isinstance(outputs, dict) else type(outputs).__name__,
    }


def main():
    batch_index = int(os.environ.get('TWINKLE_PARITY_BATCH_INDEX', '0'))
    seed = PARITY_SEED
    output_path = os.environ.get('TWINKLE_PARITY_OUTPUT')

    _ensure_process_group()
    _set_seed(seed)
    global_batch = _get_global_batch(batch_index)
    global_batch_digest = _batch_digest(global_batch)
    local_batch = _slice_local_batch(global_batch)

    _set_seed(seed)
    model = _create_model(num_training_steps=1)
    logger.info(model.get_train_configs(adapter_name='default'))

    init_state = _clone_state_dict(model.get_state_dict(adapter_name='default'))
    result = _run_first_train_step(model, local_batch, init_state)

    payload = {
        'mode': 'first_step_parity',
        'model_id': MODEL_ID,
        'datasets': DATASETS,
        'seed': int(seed),
        'batch_index': int(batch_index),
        'global_batch_size': int(GLOBAL_BATCH_SIZE),
        'ulysses_size': int(ULYSSES_SIZE),
        'wrap_mode': PARITY_WRAP_MODE,
        'data_world_size': int(device_mesh.data_world_size),
        'data_rank': int(device_mesh.data_rank),
        'rank': int(Platform.get_rank()),
        'env': {
            'TWINKLE_PARITY_WARMUP_STEPS': os.environ.get('TWINKLE_PARITY_WARMUP_STEPS', '0'),
            'TWINKLE_PARITY_STRICT_DETERMINISM': os.environ.get('TWINKLE_PARITY_STRICT_DETERMINISM', '1'),
            'TWINKLE_PARITY_MIXED_PRECISION': PARITY_MIXED_PRECISION,
            'TWINKLE_PARITY_ATTN_IMPL': PARITY_ATTN_IMPL,
            'TWINKLE_PARITY_DISABLE_GC': os.environ.get('TWINKLE_PARITY_DISABLE_GC', '1'),
            'TWINKLE_PARITY_WRAP_MODE': PARITY_WRAP_MODE,
        },
        'global_batch_digest': global_batch_digest,
        'result': result,
    }
    payload_text = json.dumps(payload, ensure_ascii=False, indent=2)
    logger.info(f'First-step parity result:\n{payload_text}')

    if output_path and Platform.get_rank() == 0:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(payload_text)
        logger.info(f'First-step parity result saved to {output_path}')


if __name__ == '__main__':
    main()
