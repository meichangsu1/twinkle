import numpy as np
import torch
from functools import partial
from peft import LoraConfig

import twinkle
from twinkle import DeviceGroup, DeviceMesh, Platform, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.model.transformers.models import TwinkleQwen3_5ForCausalLM
from twinkle.preprocessor import SelfCognitionProcessor

logger = get_logger()
MODEL_ID = 'ms://Qwen/Qwen3.5-4B'
DATASETS = 'ms://swift/self-cognition'

device_group = [DeviceGroup(
    name='default',
    ranks=[0, 1, 2, 3],
    device_type=Platform.get_platform().device_prefix(),
)]

# FSDP + SP validation over 4 GPUs: dp=2, fsdp=2 (SP only affects input slicing)
device_mesh = DeviceMesh(
    device_type='cuda',
    mesh=np.arange(4).reshape(2, 2),
    mesh_dim_names=('dp', 'fsdp'),
    ulysses_size=2,
)

twinkle.initialize(
    mode='local',
    nproc_per_node=4,
    global_device_mesh=device_mesh,
    lazy_collect=False,
)


def _memory_api():
    device_type = Platform.get_platform().device_prefix()
    device_api = getattr(torch, device_type, None)
    if device_api is None or not hasattr(device_api, 'is_available') or not device_api.is_available():
        return None, None
    return device_type, device_api


def _format_mib(num_bytes):
    return f'{num_bytes / (1024 ** 2):.1f} MiB'


def _get_memory_stats():
    device_type, device_api = _memory_api()
    if device_api is None:
        return {}

    if hasattr(device_api, 'synchronize'):
        device_api.synchronize()

    current_device = device_api.current_device() if hasattr(device_api, 'current_device') else 0
    return {
        'rank': Platform.get_rank(),
        'local_rank': Platform.get_local_rank(),
        'device': f'{device_type}:{current_device}',
        'mem_allocated': _format_mib(device_api.memory_allocated()),
        'mem_reserved': _format_mib(device_api.memory_reserved()),
        'mem_peak_allocated': _format_mib(device_api.max_memory_allocated()),
        'mem_peak_reserved': _format_mib(device_api.max_memory_reserved()),
    }


def _reset_peak_memory_stats():
    _, device_api = _memory_api()
    if device_api is not None and hasattr(device_api, 'reset_peak_memory_stats'):
        device_api.reset_peak_memory_stats()


def _get_runtime_backend_info(model: TransformersModel):
    model._ensure_sp_strategy()

    underlying_model = getattr(model, 'model', None)
    llm_model = getattr(underlying_model, 'model', underlying_model)
    config = getattr(underlying_model, 'config', None)

    attn_implementation = None
    attn_implementation_internal = None
    if config is not None:
        attn_implementation = getattr(config, '_attn_implementation', None)
        attn_implementation_internal = getattr(config, '_attn_implementation_internal', None)

    return {
        'model_cls': type(underlying_model).__name__ if underlying_model is not None else None,
        'llm_model_cls': type(llm_model).__name__ if llm_model is not None else None,
        'attn_implementation': attn_implementation,
        'attn_implementation_internal': attn_implementation_internal,
        'requires_cu_seq_lens_q': bool(getattr(llm_model, 'requires_cu_seq_lens_q', False)),
        'sp_enabled': bool(getattr(model, '_enable_sp', False)),
        'ulysses_size': getattr(getattr(model, 'device_mesh', None), 'ulysses_size', None),
        'sp_strategy_enabled': bool(getattr(getattr(model, 'sp_strategy', None), 'enabled', False)),
        'sp_strategy_ulysses_size': getattr(getattr(model, 'sp_strategy', None), 'ulysses_size', None),
    }


def eval(model):
    dataloader = DataLoader(
        dataset=partial(create_dataset, data_slice=range(100)),
        batch_size=4,
        device_mesh=device_mesh,
    )
    for _, batch in enumerate(dataloader):
        model.forward_only(inputs=batch, adapter_name='default')
        model.calculate_loss(adapter_name='default')
    return model.calculate_metric(is_training=False, adapter_name='default')


def create_dataset(data_slice=None):
    dataset = Dataset(dataset_meta=DatasetMeta(DATASETS, data_slice=range(500)))
    dataset.set_template('Template', model_id=MODEL_ID)
    dataset.map(SelfCognitionProcessor('twinkle模型', 'twinkle团队'))
    dataset.encode(batched=True)
    return dataset


def train():
    dataloader = DataLoader(
        dataset=partial(create_dataset, data_slice=None),
        batch_size=8,
        device_mesh=device_mesh,
    )

    model = TransformersModel(
        model_id=MODEL_ID,
        model_cls=TwinkleQwen3_5ForCausalLM,
        device_mesh=device_mesh,
        strategy='native_fsdp',
    )

    lora_config = LoraConfig(target_modules='all-linear')
    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=1)
    model.set_optimizer('AdamW', lr=1e-4, adapter_name='default')
    model.set_lr_scheduler(
        scheduler_cls='CosineWarmupScheduler',
        num_warmup_steps=5,
        num_training_steps=len(dataloader),
        adapter_name='default',
    )

    logger.info(model.get_train_configs(adapter_name='default'))
    logger.info(f'Total steps: {len(dataloader)}')
    logger.info(f'Backend info: {_get_runtime_backend_info(model)}')
    logger.info(f'Initial memory: {_get_memory_stats()}')
    _reset_peak_memory_stats()

    for step, batch in enumerate(dataloader):
        model.forward_backward(inputs=batch, adapter_name='default')
        model.clip_grad_and_step(adapter_name='default')
        if step % 20 == 0:
            metric = model.calculate_metric(is_training=True, adapter_name='default')
            metric.update(_get_memory_stats())
            logger.info(f'Current is step {step} of {len(dataloader)}, metric: {metric}')
    model.save('last-checkpoint', interval=1)


if __name__ == '__main__':
    train()
