import os

import twinkle
from peft import LoraConfig
from transformers import AutoConfig
from twinkle import DeviceMesh, Platform, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor

logger = get_logger()
# `deepseek-ai/DeepSeek-V4-Flash` uses mixed FP4/FP8 weights.
# Convert the checkpoint before training by following:
# https://gitcode.com/cann/cann-recipes-train/blob/master/llm_pretrain/deepseekv4/README.md#%E6%A8%A1%E5%9E%8B%E6%9D%83%E9%87%8D%E5%87%86%E5%A4%87
# Install `transformers==5.8.0` before running this cookbook.
MODEL_ID = os.environ.get('MODEL_ID', 'ms://deepseek-ai/DeepSeek-V4-Flash')
DATASET_ID = os.environ.get('DATASET_ID', 'ms://swift/self-cognition')
TEMPLATE_ID = os.environ.get('TEMPLATE_ID', 'DeepseekV4Template')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', './output')

NUM_LAYERS = int(os.environ.get('NUM_LAYERS', '4'))

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '4'))
GRAD_ACCUM_STEPS = int(os.environ.get('GRAD_ACCUM_STEPS', '2'))
LR = float(os.environ.get('LR', '1e-4'))
MAX_STEPS = int(os.environ.get('MAX_STEPS', '0'))
SAVE_STEPS = int(os.environ.get('SAVE_STEPS', '50'))
RESHARD_AFTER_FORWARD = os.environ.get('RESHARD_AFTER_FORWARD', '1') == '1'
GRADIENT_CHECKPOINTING = True
IGNORE_MISMATCHED_SIZES = False
LORA_TARGET_MODULES = [
    'q_a_proj',
    'q_b_proj',
    'kv_proj',
    'o_b_proj',
    'gate_proj',
    'up_proj',
    'down_proj',
]
ADAPTER_NAME = 'default'


def _rank_prefix():
    rank = Platform.get_rank()
    local_rank = Platform.get_local_rank()
    world_size = Platform.get_world_size()
    return f'[rank{rank if rank >= 0 else 0}/local{local_rank if local_rank >= 0 else 0}/world{world_size}]'


def debug_log(message):
    print(f'{_rank_prefix()} deepseek_v4_flash.{message}', flush=True)


debug_log(
    f'bootstrap MASTER_ADDR={os.environ.get("MASTER_ADDR")} MASTER_PORT={os.environ.get("MASTER_PORT")} '
    f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")} '
    f'ASCEND_RT_VISIBLE_DEVICES={os.environ.get("ASCEND_RT_VISIBLE_DEVICES")}')

device_mesh = DeviceMesh.from_sizes(
    fsdp_size=4,
    dp_size=1,
    device_type=Platform.get_platform().device_prefix(),
)

debug_log(f'before_twinkle_initialize device_mesh_world_size={device_mesh.world_size}')
twinkle.initialize(mode='local', global_device_mesh=device_mesh)
debug_log('after_twinkle_initialize')


def create_dataset(data_slice=None):
    debug_log(f'create_dataset.start data_slice={data_slice}')
    dataset = Dataset(dataset_meta=DatasetMeta(DATASET_ID, data_slice=data_slice or range(1000)))
    debug_log('create_dataset.after_dataset')
    dataset.set_template(TEMPLATE_ID, model_id=MODEL_ID)
    debug_log('create_dataset.after_set_template')
    dataset.map(SelfCognitionProcessor('twinkle大模型', 'ModelScope社区'))
    debug_log('create_dataset.after_map')
    dataset.encode(batched=True)
    debug_log('create_dataset.after_encode')
    return dataset


def eval(model):
    dataset = create_dataset(data_slice=range(100))
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, device_mesh=device_mesh)
    for _, batch in enumerate(dataloader):
        if callable(batch):
            batch = batch()
        model.forward_only(inputs=batch, adapter_name=ADAPTER_NAME)
        model.calculate_loss(adapter_name=ADAPTER_NAME)
    return model.calculate_metric(is_training=False, adapter_name=ADAPTER_NAME)


def train():
    debug_log('train.start')
    dataset = create_dataset()
    debug_log('train.after_create_dataset')
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, device_mesh=device_mesh)
    debug_log(f'train.after_dataloader len={len(dataloader)} batch_size={BATCH_SIZE}')

    debug_log('before_auto_config')
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    debug_log('after_auto_config')
    if NUM_LAYERS is not None and hasattr(config, 'num_hidden_layers'):
        config.num_hidden_layers = NUM_LAYERS
    if hasattr(config, 'use_cache'):
        config.use_cache = False

    debug_log('before_transformers_model_init')
    model = TransformersModel(
        model_id=MODEL_ID,
        config=config,
        device_mesh=device_mesh,
        strategy='native_fsdp',
        memory_efficient_init=True,
        ignore_mismatched_sizes=IGNORE_MISMATCHED_SIZES,
        fsdp_config={
            'reshard_after_forward': RESHARD_AFTER_FORWARD,
        },
    )
    debug_log('after_transformers_model_init')

    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=LORA_TARGET_MODULES)
    debug_log('before_add_adapter_to_model')
    model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=GRAD_ACCUM_STEPS)
    debug_log('after_add_adapter_to_model')

    if not GRADIENT_CHECKPOINTING:
        model.model.gradient_checkpointing_disable()
        debug_log('after_gradient_checkpointing_disable')

    debug_log('before_set_template')
    model.set_template(TEMPLATE_ID, model_id=MODEL_ID, adapter_name=ADAPTER_NAME)
    debug_log('after_set_template')
    debug_log('before_set_optimizer')
    model.set_optimizer('AdamW', lr=LR, foreach=False, adapter_name=ADAPTER_NAME)
    debug_log('after_set_optimizer')
    debug_log('before_set_lr_scheduler')
    model.set_lr_scheduler(
        scheduler_cls='CosineWarmupScheduler',
        num_warmup_steps=5,
        num_training_steps=len(dataloader),
        adapter_name=ADAPTER_NAME,
    )
    debug_log('after_set_lr_scheduler')

    logger.info(get_device_placement())
    logger.info(model.get_train_configs(adapter_name=ADAPTER_NAME))
    logger.info(
        f'Total steps: {len(dataloader)}, batch_size={BATCH_SIZE}, '
        f'grad_accum={GRAD_ACCUM_STEPS}, lr={LR:.2e}, '
        f'num_layers={NUM_LAYERS}, ignore_mismatched_sizes={IGNORE_MISMATCHED_SIZES}, '
        f'gradient_checkpointing={GRADIENT_CHECKPOINTING}, '
        f'reshard_after_forward={RESHARD_AFTER_FORWARD}, '
        f'lora_target_modules={LORA_TARGET_MODULES}')

    best_loss = float('inf')
    for step, batch in enumerate(dataloader):
        if MAX_STEPS and step >= MAX_STEPS:
            break
        if callable(batch):
            debug_log(f'step_{step}.before_call_batch')
            batch = batch()
            debug_log(f'step_{step}.after_call_batch')
        debug_log(f'step_{step}.before_forward_backward')
        model.forward_backward(
            inputs=batch,
            adapter_name=ADAPTER_NAME,
        )
        debug_log(f'step_{step}.after_forward_backward')
        debug_log(f'step_{step}.before_clip_grad_and_step')
        model.clip_grad_and_step(
            adapter_name=ADAPTER_NAME,
            gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        )
        debug_log(f'step_{step}.after_clip_grad_and_step')

        if step % 20 == 0:
            debug_log(f'step_{step}.before_calculate_metric')
            metric = model.calculate_metric(is_training=True, adapter_name=ADAPTER_NAME)
            debug_log(f'step_{step}.after_calculate_metric')
            logger.info(f'Current is step {step} of {len(dataloader)}, metric: {metric}')

        if step > 0 and step % SAVE_STEPS == 0:
            metrics = eval(model)
            logger.info(f'Eval metric: {metrics}')
            loss = float(metrics['loss'])
            if loss < best_loss:
                model.save(name=f'checkpoint-{step}', output_dir=OUTPUT_DIR, adapter_name=ADAPTER_NAME)
                best_loss = loss

    debug_log('before_save_last_checkpoint')
    model.save(name='last-checkpoint', output_dir=OUTPUT_DIR, adapter_name=ADAPTER_NAME)
    debug_log('after_save_last_checkpoint')


if __name__ == '__main__':
    train()
