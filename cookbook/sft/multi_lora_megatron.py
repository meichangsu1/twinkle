import os
from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import MultiLoraMegatronModel
from twinkle.preprocessor import SelfCognitionProcessor

logger = get_logger()

MODEL_ID = os.getenv('MODEL_ID', 'ms://Qwen/Qwen2.5-7B-Instruct')
DATASET_ID = os.getenv('DATASET_ID', 'ms://swift/self-cognition')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'output/multi_lora_megatron')

TRAIN_SAMPLES = int(os.getenv('TRAIN_SAMPLES', '1000'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '16'))
EPOCHS = int(os.getenv('EPOCHS', '1'))
GRAD_ACC_STEPS = int(os.getenv('GRAD_ACC_STEPS', '1'))
MAX_LENGTH = int(os.getenv('MAX_LENGTH', '1024'))
MAX_LORAS = int(os.getenv('MAX_LORAS', '4'))
MAX_R = int(os.getenv('MAX_R', '32'))
LOG_INTERVAL = int(os.getenv('LOG_INTERVAL', '10'))
SWITCH_EVERY = int(os.getenv('SWITCH_EVERY', '1'))
SAVE_EVERY_EPOCH = os.getenv('SAVE_EVERY_EPOCH', '1') == '1'
MIXED_PRECISION = os.getenv('MIXED_PRECISION', 'bf16')

DP_SIZE = int(os.getenv('DP_SIZE', '1'))
TP_SIZE = int(os.getenv('TP_SIZE', '1'))
PP_SIZE = int(os.getenv('PP_SIZE', '1'))
CP_SIZE = int(os.getenv('CP_SIZE', '1'))
EP_SIZE = int(os.getenv('EP_SIZE', '1'))
SEQUENCE_PARALLEL = os.getenv('SEQUENCE_PARALLEL', '0') == '1'
USE_DISTRIBUTED_OPTIMIZER = os.getenv('USE_DISTRIBUTED_OPTIMIZER', '1') == '1'


def build_device_mesh() -> DeviceMesh:
    kwargs = dict(
        dp_size=DP_SIZE,
        tp_size=TP_SIZE,
        pp_size=PP_SIZE,
        cp_size=CP_SIZE,
        sequence_parallel=SEQUENCE_PARALLEL,
    )
    if EP_SIZE > 1:
        kwargs['ep_size'] = EP_SIZE
    return DeviceMesh.from_sizes(**kwargs)


def create_dataloader(device_mesh: DeviceMesh):
    dataset = Dataset(dataset_meta=DatasetMeta(DATASET_ID, data_slice=range(TRAIN_SAMPLES)))
    dataset.set_template('Template', model_id=MODEL_ID, max_length=MAX_LENGTH)
    dataset.map(SelfCognitionProcessor('twinkle模型', 'twinkle团队'))
    dataset.encode(batched=True)
    return DataLoader(dataset=dataset, batch_size=BATCH_SIZE, device_mesh=device_mesh)


def setup_model(device_mesh: DeviceMesh, total_steps: int):
    model = MultiLoraMegatronModel(
        model_id=MODEL_ID,
        device_mesh=device_mesh,
        mixed_precision=MIXED_PRECISION,
        max_loras=MAX_LORAS,
        max_r=MAX_R,
        use_distributed_optimizer=USE_DISTRIBUTED_OPTIMIZER,
    )

    tenant_settings = {
        'tenant_a': {
            'config': LoraConfig(r=8, lora_alpha=32, target_modules='all-linear'),
            'lr': 1e-4,
        },
        'tenant_b': {
            'config': LoraConfig(r=16, lora_alpha=32, target_modules='all-linear'),
            'lr': 8e-5,
        },
    }
    steps_per_adapter = max(1, (total_steps + len(tenant_settings) - 1) // len(tenant_settings))
    warmup_steps = max(1, steps_per_adapter // 10)

    for adapter_name, settings in tenant_settings.items():
        model.add_adapter_to_model(
            adapter_name,
            settings['config'],
            gradient_accumulation_steps=GRAD_ACC_STEPS,
        )
        model.set_template('Template', model_id=MODEL_ID, max_length=MAX_LENGTH, adapter_name=adapter_name)
        model.set_processor('InputProcessor', padding_side='right', adapter_name=adapter_name)
        model.set_loss('CrossEntropyLoss', adapter_name=adapter_name)
        model.set_optimizer('default', lr=settings['lr'], adapter_name=adapter_name)
        model.set_lr_scheduler(
            'default',
            lr_warmup_steps=warmup_steps,
            lr_decay_steps=steps_per_adapter,
            adapter_name=adapter_name,
        )

    return model, list(tenant_settings.keys())


def train():
    device_mesh = build_device_mesh()
    twinkle.initialize(mode='local', global_device_mesh=device_mesh, lazy_collect=False)

    dataloader = create_dataloader(device_mesh)
    total_steps = len(dataloader) * EPOCHS
    model, adapters = setup_model(device_mesh, total_steps)

    logger.info(get_device_placement())
    for adapter_name in adapters:
        logger.info(model.get_train_configs(adapter_name=adapter_name))

    global_step = 0
    for epoch in range(EPOCHS):
        for _, batch in enumerate(dataloader):
            adapter_name = adapters[(global_step // SWITCH_EVERY) % len(adapters)]
            loss = model.forward_backward(inputs=batch, adapter_name=adapter_name)
            model.clip_grad_and_step(adapter_name=adapter_name)

            if global_step % LOG_INTERVAL == 0:
                metric = model.calculate_metric(is_training=True, adapter_name=adapter_name)
                logger.info(
                    f'epoch={epoch}, global_step={global_step}, adapter={adapter_name}, '
                    f'loss={loss}, metric={metric}'
                )
            global_step += 1

        if SAVE_EVERY_EPOCH:
            for adapter_name in adapters:
                checkpoint_dir = model.save(
                    name=f'{adapter_name}-epoch-{epoch}',
                    output_dir=OUTPUT_DIR,
                    adapter_name=adapter_name,
                )
                logger.info(f'saved checkpoint: {checkpoint_dir}')

    for adapter_name in adapters:
        checkpoint_dir = model.save(
            name=f'{adapter_name}-final',
            output_dir=OUTPUT_DIR,
            adapter_name=adapter_name,
        )
        logger.info(f'saved checkpoint: {checkpoint_dir}')


if __name__ == '__main__':
    train()
