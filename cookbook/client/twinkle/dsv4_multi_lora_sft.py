# Copyright (c) ModelScope Contributors. All rights reserved.
"""Client-side GSM8K LoRA SFT for a DeepSeek-V4-Flash-0731 server.

Start the DeepSeek-V4 multi-node server first. This client trains one LoRA
tenant on the EP/FSDP model.
"""
import math
import os

from peft import LoraConfig

from twinkle import get_logger, init_twinkle_client
from twinkle.dataset import DatasetMeta
from twinkle_client.dataloader import DataLoader
from twinkle_client.dataset import Dataset
from twinkle_client.model import MultiLoraTransformersModel

logger = get_logger()

SERVER_URL = os.environ.get('TWINKLE_SERVER_URL', 'http://localhost:8000')
SERVER_TOKEN = os.environ.get('TWINKLE_SERVER_TOKEN', 'EMPTY_TOKEN')
SERVED_MODEL_NAME = os.environ.get('TWINKLE_MODEL_ID', 'deepseek-v4-0731-local')
MODEL_PATH = os.environ.get('DSV4_MODEL_ID', '/nas/disk1/random-deepseek-v4-4b')
DATASET_ID = os.environ.get('DATASET_ID')
DATASET_SUBSET = os.environ.get('DATASET_SUBSET', 'default')
DATASET_SPLIT = os.environ.get('DATASET_SPLIT', 'train')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '/tmp/twinkle_dsv4_0731_multi_lora')

ADAPTER_NAMES = tuple(
    name.strip() for name in os.environ.get('ADAPTER_NAMES', 'tenant_a').split(',') if name.strip())
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '32'))
GRAD_ACCUM_STEPS = int(os.environ.get('GRAD_ACCUM_STEPS', '4'))
MAX_STEPS = int(os.environ.get('MAX_STEPS', '10'))
MAX_LENGTH = int(os.environ.get('MAX_LENGTH', '8192'))
TRUNCATION_STRATEGY = os.environ.get('TRUNCATION_STRATEGY', 'delete')
LR = float(os.environ.get('LR', '1e-4'))
LORA_R = int(os.environ.get('LORA_R', '8'))
LORA_ALPHA = int(os.environ.get('LORA_ALPHA', '32'))


def _assert_finite_output(value, path='result') -> None:
    """Fail the inference smoke test when a returned numeric value is NaN/Inf."""
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite_output(item, f'{path}.{key}')
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite_output(item, f'{path}[{index}]')
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f'Inference smoke test produced a non-finite value at {path}: {value}')


def _build_dataset() -> Dataset:
    if not DATASET_ID:
        raise ValueError('Set DATASET_ID to a local GSM8K JSON/JSONL file or directory.')
    if DATASET_ID.startswith(('hf://', 'ms://')):
        raise ValueError(f'DATASET_ID must be local for this recipe, got: {DATASET_ID}')
    if not os.path.exists(DATASET_ID):
        raise FileNotFoundError(f'Local GSM8K dataset not found: {DATASET_ID}')
    dataset = Dataset(dataset_meta=DatasetMeta(
        DATASET_ID,
        subset_name=DATASET_SUBSET,
        split=DATASET_SPLIT,
    ))
    dataset.set_template(
        'DeepseekV4Template',
        model_id=MODEL_PATH,
        max_length=MAX_LENGTH,
        truncation_strategy=TRUNCATION_STRATEGY,
    )
    dataset.map(
        'GSM8KProcessor',
        init_args={
            'system': 'Solve the math problem step by step and put the final answer in \\boxed{}.',
            # SFT needs the reference solution as the assistant target.
            'add_assistant': True,
        },
    )
    dataset.encode()
    return dataset


def _build_lora_config() -> LoraConfig:
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        target_modules='all-linear',
        exclude_modules=['o_a_proj'],
        target_parameters=[
            'mlp.experts.gate_up_proj',
            'mlp.experts.down_proj',
        ],
    )


def train() -> None:
    client = init_twinkle_client(base_url=SERVER_URL, api_key=SERVER_TOKEN)
    supported_models = [item.model_name for item in client.get_server_capabilities().supported_models]
    if SERVED_MODEL_NAME not in supported_models:
        raise RuntimeError(f'{SERVED_MODEL_NAME!r} is not served; available models: {supported_models}')
    if len(ADAPTER_NAMES) != 1:
        raise ValueError('This recipe requires exactly one ADAPTER_NAMES entry.')
    if BATCH_SIZE < 32 or BATCH_SIZE % 32 != 0:
        raise ValueError('BATCH_SIZE must be at least 32 and divisible by 32 for the 32-rank FSDP model.')

    dataset = _build_dataset()
    # Drop the final undersized batch because 32 FSDP ranks require a full
    # global batch here. GSM8K is large enough for the configured run, so each
    # adapter traverses at most one shuffled epoch.
    dataloaders = {
        name: DataLoader(dataset=dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)
        for name in ADAPTER_NAMES
    }
    models = {name: MultiLoraTransformersModel(model_id=SERVED_MODEL_NAME) for name in ADAPTER_NAMES}
    lora_config = _build_lora_config()

    # Register every tenant before set_optimizer(). The first optimizer call
    # materializes and EP/FSDP-shards all preallocated Multi-LoRA slots.
    for name, model in models.items():
        save_dir = os.path.join(OUTPUT_DIR, name)
        os.makedirs(save_dir, exist_ok=True)
        model.add_adapter_to_model(
            name,
            lora_config,
            gradient_accumulation_steps=GRAD_ACCUM_STEPS,
            save_dir=save_dir,
        )
        model.set_template(
            'DeepseekV4Template',
            max_length=MAX_LENGTH,
            truncation_strategy=TRUNCATION_STRATEGY,
        )
        model.set_processor('InputProcessor', padding_side='right')
        model.set_loss('CrossEntropyLoss')

    for model in models.values():
        model.set_optimizer('AdamW', lr=LR, foreach=False)

    iterators = {name: iter(loader) for name, loader in dataloaders.items()}
    completed_steps = {name: 0 for name in ADAPTER_NAMES}
    active_names = set(ADAPTER_NAMES)

    while active_names and any(step < MAX_STEPS for step in completed_steps.values()):
        for name in ADAPTER_NAMES:
            if name not in active_names or completed_steps[name] >= MAX_STEPS:
                continue
            try:
                batch = next(iterators[name])
            except StopIteration:
                active_names.remove(name)
                continue

            model = models[name]
            model.forward_backward(inputs=batch, gradient_accumulation_steps=GRAD_ACCUM_STEPS)
            model.clip_grad_and_step(max_grad_norm=1.0, gradient_accumulation_steps=GRAD_ACCUM_STEPS)
            completed_steps[name] += 1

            if completed_steps[name] % GRAD_ACCUM_STEPS == 0:
                metric = model.calculate_metric(is_training=True)
                logger.info('adapter=%s micro_step=%s metric=%s', name, completed_steps[name], metric.result)

    for name, model in models.items():
        checkpoint = model.save(
            name=f'dsv4-0731-{name}-final',
            save_optimizer=True,
            consumed_train_samples=dataloaders[name].get_state()['consumed_train_samples'],
        )
        logger.info('Saved adapter %s: %s', name, checkpoint.twinkle_path)

    # Run a real no-grad forward pass after saving. Compare the trained LoRA
    # path with the base-model path so both can be verified on the same input.
    # Saving first preserves the expensive training result if this check fails.
    eval_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=False)
    eval_batch = next(iter(eval_loader))
    for name, model in models.items():
        lora_result = model.forward_only(inputs=eval_batch, disable_lora=False).result
        lora_loss = model.calculate_loss().result
        base_result = model.forward_only(inputs=eval_batch, disable_lora=True).result
        base_loss = model.calculate_loss().result
        _assert_finite_output(lora_result, f'{name}.lora')
        _assert_finite_output(base_result, f'{name}.base')
        _assert_finite_output(lora_loss, f'{name}.lora_loss')
        _assert_finite_output(base_loss, f'{name}.base_loss')
        # Use WARNING so the inference result remains visible even when the
        # deployment intentionally suppresses INFO logs to save local storage.
        logger.warning(
            'Inference result: adapter=%s lora_loss=%.6f base_loss=%.6f loss_delta=%.6f '
            'lora_keys=%s base_keys=%s',
            name,
            lora_loss,
            base_loss,
            lora_loss - base_loss,
            list(lora_result) if isinstance(lora_result, dict) else type(lora_result).__name__,
            list(base_result) if isinstance(base_result, dict) else type(base_result).__name__,
        )


if __name__ == '__main__':
    train()
