# Copyright (c) ModelScope Contributors. All rights reserved.
"""Client-side self-cognition SFT for a DeepSeek-V4-Flash-0731 server.

Start the DeepSeek-V4 multi-node server first. This client trains one LoRA
tenant and updates only the routed-expert gate/up/down parameters.
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
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '/tmp/twinkle_dsv4_0731_self_cognition')
MODEL_NAME = os.environ.get('SELF_COGNITION_MODEL_NAME', 'twinkle模型')
MODEL_AUTHOR = os.environ.get('SELF_COGNITION_MODEL_AUTHOR', 'ModelScope社区')

ADAPTER_NAME = os.environ.get('ADAPTER_NAME', 'tenant_a').strip()
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '32'))
GRAD_ACCUM_STEPS = int(os.environ.get('GRAD_ACCUM_STEPS', '1'))
NUM_EPOCHS = int(os.environ.get('NUM_EPOCHS', '3'))
MAX_STEPS = int(os.environ.get('MAX_STEPS', '0'))
MAX_LENGTH = int(os.environ.get('MAX_LENGTH', '8192'))
TRUNCATION_STRATEGY = os.environ.get('TRUNCATION_STRATEGY', 'delete')
LR = float(os.environ.get('LR', '1e-4'))
LORA_R = int(os.environ.get('LORA_R', '8'))
LORA_ALPHA = int(os.environ.get('LORA_ALPHA', '32'))
ROUTED_EXPERT_TARGET_PARAMETERS = [
    'mlp.experts.gate_up_proj',
    'mlp.experts.down_proj',
]


def _assert_finite_output(value, path='result') -> None:
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
        raise ValueError('Set DATASET_ID to a local self-cognition JSON or JSONL file.')
    if DATASET_ID.startswith(('hf://', 'ms://')):
        raise ValueError(f'DATASET_ID must be local for this recipe, got: {DATASET_ID}')
    if not os.path.exists(DATASET_ID):
        raise FileNotFoundError(f'Local self-cognition dataset not found: {DATASET_ID}')

    dataset = Dataset(dataset_meta=DatasetMeta(DATASET_ID))
    dataset.set_template(
        'DeepseekV4Template',
        model_id=MODEL_PATH,
        max_length=MAX_LENGTH,
        truncation_strategy=TRUNCATION_STRATEGY,
    )
    dataset.map(
        'SelfCognitionProcessor',
        init_args={
            'model_name': MODEL_NAME,
            'model_author': MODEL_AUTHOR,
        },
    )
    dataset.encode()
    return dataset


def _build_lora_config() -> LoraConfig:
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        target_modules=None,
        target_parameters=ROUTED_EXPERT_TARGET_PARAMETERS,
        bias='none',
    )


def train() -> None:
    client = init_twinkle_client(base_url=SERVER_URL, api_key=SERVER_TOKEN)
    supported_models = [item.model_name for item in client.get_server_capabilities().supported_models]
    if SERVED_MODEL_NAME not in supported_models:
        raise RuntimeError(f'{SERVED_MODEL_NAME!r} is not served; available models: {supported_models}')
    if not ADAPTER_NAME:
        raise ValueError('ADAPTER_NAME must not be empty.')
    if BATCH_SIZE < 32 or BATCH_SIZE % 32 != 0:
        raise ValueError('BATCH_SIZE must be at least 32 and divisible by 32 for the 32-rank FSDP model.')
    if NUM_EPOCHS <= 0:
        raise ValueError('NUM_EPOCHS must be greater than zero.')
    if MAX_STEPS < 0:
        raise ValueError('MAX_STEPS must be zero (unlimited) or a positive integer.')

    dataset = _build_dataset()
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        drop_last=True,
        shuffle=True,
    )
    model = MultiLoraTransformersModel(model_id=SERVED_MODEL_NAME)
    save_dir = os.path.join(OUTPUT_DIR, ADAPTER_NAME)
    os.makedirs(save_dir, exist_ok=True)
    model.add_adapter_to_model(
        ADAPTER_NAME,
        _build_lora_config(),
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
    model.set_optimizer('AdamW', lr=LR, foreach=False)

    completed_steps = 0
    stop_training = False
    for epoch in range(NUM_EPOCHS):
        logger.info('Starting epoch %s/%s', epoch + 1, NUM_EPOCHS)
        for batch in dataloader:
            if MAX_STEPS > 0 and completed_steps >= MAX_STEPS:
                stop_training = True
                break
            model.forward_backward(inputs=batch, gradient_accumulation_steps=GRAD_ACCUM_STEPS)
            model.clip_grad_and_step(max_grad_norm=1.0, gradient_accumulation_steps=GRAD_ACCUM_STEPS)
            completed_steps += 1

            if completed_steps % GRAD_ACCUM_STEPS == 0:
                metric = model.calculate_metric(is_training=True)
                logger.info(
                    'adapter=%s epoch=%s/%s micro_step=%s metric=%s',
                    ADAPTER_NAME,
                    epoch + 1,
                    NUM_EPOCHS,
                    completed_steps,
                    metric.result,
                )
        if stop_training:
            break

    if completed_steps == 0:
        raise RuntimeError(
            f'No full batch was produced: dataset is smaller than BATCH_SIZE={BATCH_SIZE}. '
            'Use a smaller valid global batch or provide more data.'
        )

    checkpoint = model.save(
        name=f'dsv4-0731-{ADAPTER_NAME}-self-cognition-final',
        save_optimizer=True,
        consumed_train_samples=dataloader.get_state()['consumed_train_samples'],
    )
    logger.info('Saved adapter %s: %s', ADAPTER_NAME, checkpoint.twinkle_path)

    eval_loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        drop_last=True,
        shuffle=False,
    )
    eval_batch = next(iter(eval_loader))
    lora_result = model.forward_only(inputs=eval_batch, disable_lora=False).result
    lora_loss = model.calculate_loss().result
    base_result = model.forward_only(inputs=eval_batch, disable_lora=True).result
    base_loss = model.calculate_loss().result
    _assert_finite_output(lora_result, f'{ADAPTER_NAME}.lora')
    _assert_finite_output(base_result, f'{ADAPTER_NAME}.base')
    _assert_finite_output(lora_loss, f'{ADAPTER_NAME}.lora_loss')
    _assert_finite_output(base_loss, f'{ADAPTER_NAME}.base_loss')
    logger.warning(
        'Inference result: adapter=%s lora_loss=%.6f base_loss=%.6f loss_delta=%.6f '
        'lora_keys=%s base_keys=%s',
        ADAPTER_NAME,
        lora_loss,
        base_loss,
        lora_loss - base_loss,
        list(lora_result) if isinstance(lora_result, dict) else type(lora_result).__name__,
        list(base_result) if isinstance(base_result, dict) else type(base_result).__name__,
    )


if __name__ == '__main__':
    train()
