# Copyright (c) ModelScope Contributors. All rights reserved.
"""Client-side Multi-LoRA SFT for a local DeepSeek-V4-Flash-0731 server.

Start ``server_config_dsv4_0731.yaml`` first. This client creates two LoRA
tenants on the same two-GPU EP/FSDP model and alternates their SFT micro-steps.
The base model and dataset are both local paths on the server machine.
"""
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
DATASET_PATH = os.environ.get('DATASET_ID', '/model/ljl/dataset/self-cognition.jsonl')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '/tmp/twinkle_dsv4_0731_multi_lora')

ADAPTER_NAMES = tuple(
    name.strip() for name in os.environ.get('ADAPTER_NAMES', 'tenant_a,tenant_b').split(',') if name.strip())
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '2'))
GRAD_ACCUM_STEPS = int(os.environ.get('GRAD_ACCUM_STEPS', '4'))
MAX_STEPS = int(os.environ.get('MAX_STEPS', '10'))
MAX_LENGTH = int(os.environ.get('MAX_LENGTH', '2048'))
LR = float(os.environ.get('LR', '1e-4'))
LORA_R = int(os.environ.get('LORA_R', '8'))
LORA_ALPHA = int(os.environ.get('LORA_ALPHA', '32'))


def _build_dataset() -> Dataset:
    dataset = Dataset(dataset_meta=DatasetMeta(DATASET_PATH))
    dataset.set_template('DeepseekV4Template', model_id=MODEL_PATH, max_length=MAX_LENGTH)
    dataset.map(
        'SelfCognitionProcessor',
        init_args={
            'model_name': 'twinkle模型',
            'model_author': 'ModelScope社区',
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
    if len(ADAPTER_NAMES) != 2:
        raise ValueError('This two-slot server example requires exactly two ADAPTER_NAMES.')
    if BATCH_SIZE < 2:
        raise ValueError('BATCH_SIZE must be at least 2 because the model uses two FSDP data ranks.')

    dataset = _build_dataset()
    dataloaders = {name: DataLoader(dataset=dataset, batch_size=BATCH_SIZE) for name in ADAPTER_NAMES}
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
        model.set_template('DeepseekV4Template', max_length=MAX_LENGTH)
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


if __name__ == '__main__':
    train()
