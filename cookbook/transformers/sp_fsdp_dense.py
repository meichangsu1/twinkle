import json
import os
import random
import tempfile

import numpy as np
import torch
import torch.distributed as dist
from functools import partial
from peft import LoraConfig

import twinkle
from twinkle import DeviceGroup, DeviceMesh, Platform, get_logger, torch_util
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor

logger = get_logger()
MODEL_ID = os.environ.get('TWINKLE_MODEL_ID', 'ms://Qwen/Qwen3.5-0.8B')
DATASETS = os.environ.get('TWINKLE_DATASETS', 'ms://swift/self-cognition')

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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _maybe_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _create_model(num_training_steps: int) -> TransformersModel:
    warmup_steps = int(os.environ.get('TWINKLE_DIAG_WARMUP_STEPS', '5'))
    model = TransformersModel(
        model_id=MODEL_ID,
        device_mesh=device_mesh,
        strategy='native_fsdp',
    )

    lora_config = LoraConfig(target_modules='all-linear')
    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=1)
    model.set_optimizer('AdamW', lr=1e-4, adapter_name='default')
    model.set_lr_scheduler(
        scheduler_cls='CosineWarmupScheduler',
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
        adapter_name='default',
    )
    return model


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


def _tensor_digest(tensor: torch.Tensor, max_items: int = 8):
    tensor = tensor.detach().float().cpu()
    flat = tensor.reshape(-1)
    if flat.numel() == 0:
        return {'shape': list(tensor.shape), 'mean': 0.0, 'std': 0.0, 'max_abs': 0.0, 'slice': []}
    return {
        'shape': list(tensor.shape),
        'mean': float(tensor.mean().item()),
        'std': float(tensor.std(unbiased=False).item()) if flat.numel() > 1 else 0.0,
        'max_abs': float(tensor.abs().max().item()),
        'slice': flat[:max_items].tolist(),
    }


def _maybe_concat_outputs(outputs, key: str):
    if isinstance(outputs, dict):
        value = outputs.get(key)
        return value if isinstance(value, torch.Tensor) else None
    if isinstance(outputs, list):
        tensors = []
        for item in outputs:
            if isinstance(item, dict) and isinstance(item.get(key), torch.Tensor):
                tensors.append(item[key].detach().float().cpu())
        if not tensors:
            return None
        try:
            return torch.cat(tensors, dim=0)
        except RuntimeError:
            return tensors[0]
    return None


def _summarize_outputs(outputs):
    summary = {}
    logits = _maybe_concat_outputs(outputs, 'logits')
    if logits is not None:
        summary['logits'] = _tensor_digest(logits)
    logps = _maybe_concat_outputs(outputs, 'logps')
    if logps is not None:
        summary['logps'] = _tensor_digest(logps)
    return summary


def _get_batch(dataloader, batch_index: int):
    for idx, batch in enumerate(dataloader):
        if idx == batch_index:
            return batch
    raise IndexError(f'Batch index {batch_index} out of range')


def _run_eval_once(model, batch):
    model.zero_grad(adapter_name='default')
    outputs = model.forward_only(inputs=batch, adapter_name='default')
    loss = model.calculate_loss(adapter_name='default')
    metric = model.calculate_metric(is_training=False, adapter_name='default')
    return {
        'mode': 'eval',
        'loss': float(loss),
        'metric': metric,
        'outputs': _summarize_outputs(outputs),
    }


def _run_train_no_step_once(model, batch):
    model.zero_grad(adapter_name='default')
    outputs = model.forward(inputs=batch, adapter_name='default')
    loss = model.calculate_loss(adapter_name='default')
    model.backward(adapter_name='default')
    grad_norm = model.clip_grad_norm(adapter_name='default')
    metric = model.calculate_metric(is_training=True, adapter_name='default')
    model.zero_grad(adapter_name='default')
    return {
        'mode': 'train_no_step',
        'loss': float(loss),
        'grad_norm': float(grad_norm) if grad_norm is not None else None,
        'metric': metric,
        'outputs': _summarize_outputs(outputs),
    }


def _run_train_step_once(model, batch, init_state):
    model.zero_grad(adapter_name='default')
    outputs = model.forward(inputs=batch, adapter_name='default')
    loss = model.calculate_loss(adapter_name='default')
    model.backward(adapter_name='default')
    grad_norm = model.clip_grad_norm(adapter_name='default')
    model.step(adapter_name='default')
    metric = model.calculate_metric(is_training=True, adapter_name='default')
    final_state = _clone_state_dict(model.get_state_dict(adapter_name='default'))
    model.zero_grad(adapter_name='default')
    return {
        'mode': 'train_step',
        'loss': float(loss),
        'grad_norm': float(grad_norm) if grad_norm is not None else None,
        'metric': metric,
        'outputs': _summarize_outputs(outputs),
        'state_delta': _state_delta_stats(init_state, final_state),
    }


def single_batch_diagnostic():
    batch_size = int(os.environ.get('TWINKLE_DIAG_BATCH_SIZE', '8'))
    batch_index = int(os.environ.get('TWINKLE_DIAG_BATCH_INDEX', '0'))
    seed = int(os.environ.get('TWINKLE_DIAG_SEED', '1234'))
    output_path = os.environ.get('TWINKLE_DIAG_OUTPUT')

    dataloader = DataLoader(
        dataset=partial(create_dataset, data_slice=None),
        batch_size=batch_size,
        device_mesh=device_mesh,
    )
    batch = _get_batch(dataloader, batch_index)

    _set_seed(seed)
    model = _create_model(len(dataloader))
    logger.info(model.get_train_configs(adapter_name='default'))

    init_state = _clone_state_dict(model.get_state_dict(adapter_name='default'))
    run_id = os.environ.get('TWINKLE_DIAG_RUN_ID', 'default')
    checkpoint_root = os.environ.get(
        'TWINKLE_DIAG_CHECKPOINT_ROOT',
        os.path.join(tempfile.gettempdir(), f'twinkle-sp-diag-{run_id}'),
    )
    os.makedirs(checkpoint_root, exist_ok=True)
    checkpoint_dir = model.save(
        'single-batch-init',
        output_dir=checkpoint_root,
        interval=1,
        adapter_name='default',
        save_optimizer=True,
    )
    _maybe_barrier()

    def reset_model_state():
        model.load(checkpoint_dir, adapter_name='default', load_optimizer=True)
        _maybe_barrier()
        model.zero_grad(adapter_name='default')

    experiments = []

    reset_model_state()
    _set_seed(seed)
    experiments.append(_run_eval_once(model, batch))

    reset_model_state()
    _set_seed(seed)
    experiments.append(_run_train_no_step_once(model, batch))

    reset_model_state()
    _set_seed(seed)
    experiments.append(_run_train_step_once(model, batch, init_state))

    payload = {
        'model_id': MODEL_ID,
        'datasets': DATASETS,
        'seed': seed,
        'batch_index': batch_index,
        'batch_size': batch_size,
        'env': {
            'QWEN35_SP_LINEAR_STRICT': os.environ.get('QWEN35_SP_LINEAR_STRICT', '0'),
            'QWEN35_SP_LINEAR_CONV_HALO': os.environ.get('QWEN35_SP_LINEAR_CONV_HALO', '0'),
            'QWEN35_SP_LINEAR_STRICT_BARRIER': os.environ.get('QWEN35_SP_LINEAR_STRICT_BARRIER', '0'),
            'QWEN35_SP_PARITY_ATTN_IMPL': os.environ.get('QWEN35_SP_PARITY_ATTN_IMPL'),
        },
        'experiments': experiments,
    }

    payload_text = json.dumps(payload, ensure_ascii=False, indent=2)
    logger.info(f'Single-batch diagnostic results:\n{payload_text}')

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(payload_text)
        logger.info(f'Single-batch diagnostic saved to {output_path}')


def train():
    dataloader = DataLoader(
        dataset=partial(create_dataset, data_slice=None),
        batch_size=8,
        device_mesh=device_mesh,
    )

    model = _create_model(len(dataloader))

    logger.info(model.get_train_configs(adapter_name='default'))
    logger.info(f'Total steps: {len(dataloader)}')

    for step, batch in enumerate(dataloader):
        model.forward_backward(inputs=batch, adapter_name='default')
        model.clip_grad_and_step(adapter_name='default')
        if step % 20 == 0:
            metric = model.calculate_metric(is_training=True, adapter_name='default')
            logger.info(f'Current is step {step} of {len(dataloader)}, metric: {metric}')
    model.save('last-checkpoint', interval=1)


if __name__ == '__main__':
    if os.environ.get('TWINKLE_SINGLE_BATCH_DIAGNOSTIC', '0') == '1':
        single_batch_diagnostic()
    else:
        train()
