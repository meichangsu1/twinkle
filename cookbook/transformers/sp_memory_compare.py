import gc
import hashlib
import json
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from peft import LoraConfig
from torch.utils.data import DataLoader as TorchDataLoader

import twinkle
from twinkle import DeviceMesh, Platform, get_logger, torch_util
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import AlpacaProcessor, SelfCognitionProcessor

# 2-card SP off
# TWINKLE_MEMORY_ULYSSES_SIZE=1 \
# TWINKLE_MEMORY_OUTPUT=/tmp/sp0_memory.json \
# torchrun --standalone --nproc_per_node=2 -m cookbook.transformers.sp_memory_compare

# 2-card SP on
# TWINKLE_MEMORY_ULYSSES_SIZE=2 \
# TWINKLE_MEMORY_OUTPUT=/tmp/sp2_memory.json \
# TWINKLE_MEMORY_COMPARE_TO=/tmp/sp0_memory.json \
# torchrun --standalone --nproc_per_node=2 -m cookbook.transformers.sp_memory_compare

# 4-card SP off
# TWINKLE_MEMORY_ULYSSES_SIZE=1 \
# TWINKLE_MEMORY_OUTPUT=/tmp/sp0_memory_4g.json \
# torchrun --standalone --nproc_per_node=4 -m cookbook.transformers.sp_memory_compare

# 4-card SP on
# TWINKLE_MEMORY_ULYSSES_SIZE=2 \
# TWINKLE_MEMORY_OUTPUT=/tmp/sp2_memory_4g.json \
# TWINKLE_MEMORY_COMPARE_TO=/tmp/sp0_memory_4g.json \
# torchrun --standalone --nproc_per_node=4 -m cookbook.transformers.sp_memory_compare

logger = get_logger()

MODEL_ID = os.environ.get('TWINKLE_MODEL_ID', 'ms://Qwen/Qwen3.5-0.8B')
DATASETS = os.environ.get('TWINKLE_DATASETS', 'ms://swift/self-cognition')
ULYSSES_SIZE = int(os.environ.get('TWINKLE_MEMORY_ULYSSES_SIZE', '1'))
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', os.environ.get('TWINKLE_MEMORY_WORLD_SIZE', '4')))
LOCAL_WORLD_SIZE = int(os.environ.get('LOCAL_WORLD_SIZE', os.environ.get('TWINKLE_MEMORY_LOCAL_WORLD_SIZE', str(WORLD_SIZE))))
GLOBAL_BATCH_SIZE = int(os.environ.get('TWINKLE_MEMORY_GLOBAL_BATCH_SIZE', '8'))
MEMORY_SEED = int(os.environ.get('TWINKLE_MEMORY_SEED', '42'))
MEMORY_MIXED_PRECISION = os.environ.get('TWINKLE_MEMORY_MIXED_PRECISION', 'bf16')
MEMORY_ATTN_IMPL = os.environ.get('TWINKLE_MEMORY_ATTN_IMPL', '')
MEMORY_WRAP_MODE = os.environ.get('TWINKLE_MEMORY_WRAP_MODE', 'native_fsdp')
MEMORY_DISABLE_GC = os.environ.get('TWINKLE_MEMORY_DISABLE_GC', '0') == '1'
MEMORY_STRICT_DETERMINISM = os.environ.get('TWINKLE_MEMORY_STRICT_DETERMINISM', '0') == '1'
MEMORY_EMPTY_CACHE = os.environ.get('TWINKLE_MEMORY_EMPTY_CACHE', '1') == '1'
MEMORY_PROCESSOR = os.environ.get('TWINKLE_MEMORY_PROCESSOR', '').strip().lower()
TEMPLATE_MAX_LENGTH = os.environ.get('TWINKLE_MEMORY_TEMPLATE_MAX_LENGTH')
TRUNCATION_STRATEGY = os.environ.get('TWINKLE_MEMORY_TRUNCATION_STRATEGY')
SELF_COGNITION_NAME = os.environ.get('TWINKLE_MEMORY_SELF_NAME', 'twinkle模型')
SELF_COGNITION_AUTHOR = os.environ.get('TWINKLE_MEMORY_SELF_AUTHOR', 'twinkle团队')


def _enable_strict_determinism(seed: int) -> None:
    os.environ.setdefault('PYTHONHASHSEED', str(seed))
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
    torch.use_deterministic_algorithms(True, warn_only=True)


if MEMORY_STRICT_DETERMINISM:
    _enable_strict_determinism(MEMORY_SEED)


def _build_device_mesh() -> DeviceMesh:
    if MEMORY_WRAP_MODE == 'native_fsdp':
        if WORLD_SIZE < 2 or WORLD_SIZE % 2 != 0:
            raise ValueError('TWINKLE_MEMORY_WRAP_MODE=native_fsdp requires an even WORLD_SIZE >= 2.')
        return DeviceMesh(
            device_type='cuda',
            mesh=np.arange(WORLD_SIZE).reshape(WORLD_SIZE // 2, 2),
            mesh_dim_names=('dp', 'fsdp'),
            ulysses_size=ULYSSES_SIZE,
        )
    if MEMORY_WRAP_MODE == 'ddp':
        return DeviceMesh(
            device_type='cuda',
            mesh=np.arange(WORLD_SIZE),
            mesh_dim_names=('dp',),
            ulysses_size=ULYSSES_SIZE,
        )
    raise ValueError(f'Unsupported TWINKLE_MEMORY_WRAP_MODE={MEMORY_WRAP_MODE}')


device_mesh = _build_device_mesh()

twinkle.initialize(
    mode='local',
    nproc_per_node=LOCAL_WORLD_SIZE,
    global_device_mesh=device_mesh,
    lazy_collect=False,
)


def _infer_processor_kind() -> str:
    if MEMORY_PROCESSOR:
        return MEMORY_PROCESSOR
    if 'alpaca' in DATASETS.lower():
        return 'alpaca'
    return 'self_cognition'


def create_dataset(data_slice=None):
    dataset = Dataset(dataset_meta=DatasetMeta(DATASETS, data_slice=data_slice or range(500)))
    template_kwargs = {}
    if TEMPLATE_MAX_LENGTH is not None:
        template_kwargs['max_length'] = int(TEMPLATE_MAX_LENGTH)
    if TRUNCATION_STRATEGY:
        template_kwargs['truncation_strategy'] = TRUNCATION_STRATEGY
    dataset.set_template('Template', model_id=MODEL_ID, **template_kwargs)

    processor_kind = _infer_processor_kind()
    if processor_kind == 'alpaca':
        dataset.map(AlpacaProcessor())
    elif processor_kind == 'self_cognition':
        dataset.map(SelfCognitionProcessor(SELF_COGNITION_NAME, SELF_COGNITION_AUTHOR))
    else:
        raise ValueError(f'Unsupported TWINKLE_MEMORY_PROCESSOR={processor_kind}')

    dataset.encode(batched=True)
    return dataset


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _sync_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_memory_peaks() -> None:
    if not torch.cuda.is_available():
        return
    _sync_device()
    torch.cuda.reset_peak_memory_stats()


def _cleanup_device_memory() -> None:
    gc.collect()
    if MEMORY_EMPTY_CACHE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'ipc_collect'):
            torch.cuda.ipc_collect()


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


def _batch_digest(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
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


def _get_global_batch(batch_index: int) -> List[Dict[str, Any]]:
    batch = None
    if Platform.get_rank() == 0:
        dataset = create_dataset(data_slice=None)
        dataloader = TorchDataLoader(
            dataset,
            batch_size=GLOBAL_BATCH_SIZE,
            num_workers=0,
            shuffle=False,
            collate_fn=lambda x: x,
        )
        for idx, current_batch in enumerate(dataloader):
            if idx == batch_index:
                if not isinstance(current_batch, list):
                    raise TypeError(f'Expected list batch, got: {type(current_batch)}')
                batch = current_batch
                break
        if batch is None:
            raise IndexError(f'Batch index {batch_index} out of range')

    if dist.is_available() and dist.is_initialized():
        object_list = [batch]
        dist.broadcast_object_list(object_list, src=0)
        batch = object_list[0]
    return batch


def _slice_local_batch(global_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    batch_slice = device_mesh.get_slice(len(global_batch))
    return global_batch[batch_slice]


def _create_model(num_training_steps: int) -> TransformersModel:
    warmup_steps = int(os.environ.get('TWINKLE_MEMORY_WARMUP_STEPS', '0'))
    strategy = 'native_fsdp' if MEMORY_WRAP_MODE == 'native_fsdp' else 'accelerate'
    model_kwargs = {
        'model_id': MODEL_ID,
        'device_mesh': device_mesh,
        'strategy': strategy,
        'mixed_precision': MEMORY_MIXED_PRECISION,
    }
    if MEMORY_ATTN_IMPL:
        model_kwargs['attn_implementation'] = MEMORY_ATTN_IMPL
    model = TransformersModel(**model_kwargs)
    if MEMORY_ATTN_IMPL and hasattr(model.model, 'config') and model.model.config is not None:
        for attr in ('_attn_implementation', '_attn_implementation_internal'):
            if hasattr(model.model.config, attr):
                setattr(model.model.config, attr, MEMORY_ATTN_IMPL)
    if MEMORY_DISABLE_GC and hasattr(model.model, 'gradient_checkpointing_disable'):
        model.model.gradient_checkpointing_disable()
        if hasattr(model.model, 'config') and model.model.config is not None and hasattr(model.model.config, 'use_cache'):
            model.model.config.use_cache = False
    lora_dropout = float(os.environ.get('TWINKLE_MEMORY_LORA_DROPOUT', '0.0'))
    lora_config = LoraConfig(target_modules='all-linear', lora_dropout=lora_dropout)
    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=1)
    model.set_optimizer('AdamW', lr=1e-4, adapter_name='default')
    model.set_lr_scheduler(
        scheduler_cls='CosineWarmupScheduler',
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
        adapter_name='default',
    )
    return model


def _local_memory_snapshot(stage: str) -> Dict[str, Any]:
    snapshot = {
        'stage': stage,
        'rank': int(Platform.get_rank()),
        'device': str(Platform.get_local_device()),
    }
    if not torch.cuda.is_available():
        snapshot.update({
            'allocated_bytes': 0,
            'reserved_bytes': 0,
            'peak_allocated_bytes': 0,
            'peak_reserved_bytes': 0,
            'free_bytes': 0,
            'total_bytes': 0,
        })
        return snapshot

    _sync_device()
    device_index = torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    allocated_bytes = torch.cuda.memory_allocated(device_index)
    reserved_bytes = torch.cuda.memory_reserved(device_index)
    peak_allocated_bytes = torch.cuda.max_memory_allocated(device_index)
    peak_reserved_bytes = torch.cuda.max_memory_reserved(device_index)
    mib = 1024**2
    snapshot.update({
        'allocated_bytes': int(allocated_bytes),
        'reserved_bytes': int(reserved_bytes),
        'peak_allocated_bytes': int(peak_allocated_bytes),
        'peak_reserved_bytes': int(peak_reserved_bytes),
        'free_bytes': int(free_bytes),
        'total_bytes': int(total_bytes),
        'allocated_mib': round(float(allocated_bytes / mib), 3),
        'reserved_mib': round(float(reserved_bytes / mib), 3),
        'peak_allocated_mib': round(float(peak_allocated_bytes / mib), 3),
        'peak_reserved_mib': round(float(peak_reserved_bytes / mib), 3),
        'free_mib': round(float(free_bytes / mib), 3),
        'total_mib': round(float(total_bytes / mib), 3),
    })
    return snapshot


def _gather_objects(obj: Any) -> List[Any]:
    if dist.is_available() and dist.is_initialized():
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, obj)
        return gathered
    return [obj]


def _summarize_stage_snapshots(rank_stage_snapshots: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    stage_names = []
    for snapshots in rank_stage_snapshots:
        for stage in snapshots.keys():
            if stage not in stage_names:
                stage_names.append(stage)

    summary = {}
    for stage in stage_names:
        per_rank = [snapshots[stage] for snapshots in rank_stage_snapshots if stage in snapshots]
        per_rank.sort(key=lambda item: item['rank'])
        summary[stage] = {
            'global_max_peak_allocated_bytes': int(max(item['peak_allocated_bytes'] for item in per_rank)),
            'global_max_peak_reserved_bytes': int(max(item['peak_reserved_bytes'] for item in per_rank)),
            'global_max_current_allocated_bytes': int(max(item['allocated_bytes'] for item in per_rank)),
            'global_max_current_reserved_bytes': int(max(item['reserved_bytes'] for item in per_rank)),
            'global_max_peak_allocated_mib': round(max(item['peak_allocated_mib'] for item in per_rank), 3),
            'global_max_peak_reserved_mib': round(max(item['peak_reserved_mib'] for item in per_rank), 3),
            'global_max_current_allocated_mib': round(max(item['allocated_mib'] for item in per_rank), 3),
            'global_max_current_reserved_mib': round(max(item['reserved_mib'] for item in per_rank), 3),
            'per_rank': per_rank,
        }
    return summary


def _run_memory_stage(stage_name: str, fn, stage_snapshots: Dict[str, Dict[str, Any]]):
    _maybe_barrier()
    _reset_memory_peaks()
    result = fn()
    _sync_device()
    _maybe_barrier()
    stage_snapshots[stage_name] = _local_memory_snapshot(stage_name)
    return result


def _build_compare_summary(current_payload: Dict[str, Any], compare_to_path: str) -> Optional[Dict[str, Any]]:
    if not compare_to_path:
        return None
    if not os.path.exists(compare_to_path):
        raise FileNotFoundError(f'TWINKLE_MEMORY_COMPARE_TO not found: {compare_to_path}')

    with open(compare_to_path, 'r', encoding='utf-8') as f:
        baseline = json.load(f)

    current_stages = current_payload.get('stage_memory', {})
    baseline_stages = baseline.get('stage_memory', {})
    shared_stages = [stage for stage in baseline_stages.keys() if stage in current_stages]
    stage_deltas = {}
    for stage in shared_stages:
        current_stage = current_stages[stage]
        baseline_stage = baseline_stages[stage]
        stage_deltas[stage] = {
            'baseline_global_max_peak_allocated_mib': baseline_stage['global_max_peak_allocated_mib'],
            'current_global_max_peak_allocated_mib': current_stage['global_max_peak_allocated_mib'],
            'delta_global_max_peak_allocated_mib': round(
                current_stage['global_max_peak_allocated_mib'] - baseline_stage['global_max_peak_allocated_mib'], 3),
            'baseline_global_max_peak_reserved_mib': baseline_stage['global_max_peak_reserved_mib'],
            'current_global_max_peak_reserved_mib': current_stage['global_max_peak_reserved_mib'],
            'delta_global_max_peak_reserved_mib': round(
                current_stage['global_max_peak_reserved_mib'] - baseline_stage['global_max_peak_reserved_mib'], 3),
        }

    return {
        'compare_to': compare_to_path,
        'baseline_run_label': baseline.get('run_label'),
        'baseline_ulysses_size': baseline.get('ulysses_size'),
        'baseline_wrap_mode': baseline.get('wrap_mode'),
        'same_global_batch': baseline.get('global_batch_digest', {}).get('sha256')
        == current_payload.get('global_batch_digest', {}).get('sha256'),
        'stage_deltas': stage_deltas,
    }


def _get_model_patch_name() -> Optional[str]:
    try:
        from twinkle.model.transformers.strategy.sequence_parallel import sequence_parallel
    except Exception:
        return None
    return getattr(sequence_parallel, 'linear_attention_model_patch_name', None)


def main():
    batch_index = int(os.environ.get('TWINKLE_MEMORY_BATCH_INDEX', '0'))
    output_path = os.environ.get('TWINKLE_MEMORY_OUTPUT')
    compare_to_path = os.environ.get('TWINKLE_MEMORY_COMPARE_TO')
    run_label = os.environ.get('TWINKLE_MEMORY_RUN_LABEL', f'sp{ULYSSES_SIZE}')

    _ensure_process_group()
    _set_seed(MEMORY_SEED)
    _cleanup_device_memory()
    _maybe_barrier()
    baseline_snapshot = {'baseline': _local_memory_snapshot('baseline')}

    global_batch = _get_global_batch(batch_index)
    global_batch_digest = _batch_digest(global_batch)
    local_batch = _slice_local_batch(global_batch)
    local_batch_info = {
        'rank': int(Platform.get_rank()),
        'local_batch_size': int(len(local_batch)),
        'local_batch_sha256': _batch_digest(local_batch)['sha256'],
    }

    stage_snapshots = dict(baseline_snapshot)

    _set_seed(MEMORY_SEED)
    model = _run_memory_stage('model_init', lambda: _create_model(num_training_steps=1), stage_snapshots)
    logger.info(model.get_train_configs(adapter_name='default'))
    _run_memory_stage('wrap_model', lambda: model._lazy_wrap_model(), stage_snapshots)
    _run_memory_stage('batch_ready', lambda: None, stage_snapshots)

    outputs = _run_memory_stage('forward', lambda: model.forward(inputs=local_batch, adapter_name='default'),
                                stage_snapshots)
    loss_value = _run_memory_stage('loss', lambda: model.calculate_loss(adapter_name='default'), stage_snapshots)
    _run_memory_stage('backward', lambda: model.backward(adapter_name='default'), stage_snapshots)
    grad_norm = _run_memory_stage('clip_grad',
                                  lambda: model.clip_grad_norm(adapter_name='default'),
                                  stage_snapshots)
    _run_memory_stage('step', lambda: model.step(adapter_name='default'), stage_snapshots)
    _run_memory_stage('zero_grad', lambda: model.zero_grad(adapter_name='default'), stage_snapshots)
    _run_memory_stage('lr_step', lambda: model.lr_step(adapter_name='default'), stage_snapshots)

    rank_stage_snapshots = _gather_objects(stage_snapshots)
    local_batch_infos = _gather_objects(local_batch_info)
    stage_memory = _summarize_stage_snapshots(rank_stage_snapshots)

    payload = {
        'mode': 'sp_memory_compare',
        'run_label': run_label,
        'model_id': MODEL_ID,
        'datasets': DATASETS,
        'processor': _infer_processor_kind(),
        'seed': int(MEMORY_SEED),
        'batch_index': int(batch_index),
        'global_batch_size': int(GLOBAL_BATCH_SIZE),
        'ulysses_size': int(ULYSSES_SIZE),
        'wrap_mode': MEMORY_WRAP_MODE,
        'mixed_precision': MEMORY_MIXED_PRECISION,
        'attn_implementation': MEMORY_ATTN_IMPL or None,
        'data_world_size': int(device_mesh.data_world_size),
        'rank': int(Platform.get_rank()),
        'linear_attention_model_patch_name': _get_model_patch_name(),
        'global_batch_digest': global_batch_digest,
        'local_batches': sorted(local_batch_infos, key=lambda item: item['rank']),
        'result': {
            'loss': float(loss_value),
            'grad_norm': float(grad_norm) if grad_norm is not None else None,
            'returned_output_keys': sorted(outputs.keys()) if isinstance(outputs, dict) else type(outputs).__name__,
        },
        'env': {
            'TWINKLE_MEMORY_ULYSSES_SIZE': str(ULYSSES_SIZE),
            'TWINKLE_MEMORY_WRAP_MODE': MEMORY_WRAP_MODE,
            'TWINKLE_MEMORY_MIXED_PRECISION': MEMORY_MIXED_PRECISION,
            'TWINKLE_MEMORY_ATTN_IMPL': MEMORY_ATTN_IMPL or None,
            'TWINKLE_MEMORY_DISABLE_GC': os.environ.get('TWINKLE_MEMORY_DISABLE_GC', '0'),
            'TWINKLE_MEMORY_STRICT_DETERMINISM': os.environ.get('TWINKLE_MEMORY_STRICT_DETERMINISM', '0'),
            'TWINKLE_MEMORY_EMPTY_CACHE': os.environ.get('TWINKLE_MEMORY_EMPTY_CACHE', '1'),
            'TWINKLE_MEMORY_PROCESSOR': MEMORY_PROCESSOR or None,
            'TWINKLE_MEMORY_TEMPLATE_MAX_LENGTH': TEMPLATE_MAX_LENGTH,
            'TWINKLE_MEMORY_TRUNCATION_STRATEGY': TRUNCATION_STRATEGY,
        },
        'stage_memory': stage_memory,
    }

    if Platform.get_rank() == 0 and compare_to_path:
        payload['comparison'] = _build_compare_summary(payload, compare_to_path)

    payload_text = json.dumps(payload, ensure_ascii=False, indent=2)
    logger.info(f'SP memory compare result:\n{payload_text}')
    if output_path and Platform.get_rank() == 0:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(payload_text)
        logger.info(f'SP memory compare result saved to {output_path}')


if __name__ == '__main__':
    main()
