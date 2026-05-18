import os
import hashlib

import torch
import torch.distributed as dist
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

MODEL_GPUS = int(os.environ.get('MODEL_GPUS', os.environ.get('WORLD_SIZE', '4')))
NPROC_PER_NODE = int(os.environ.get('NPROC_PER_NODE', os.environ.get('LOCAL_WORLD_SIZE', '1')))
NUM_LAYERS = int(os.environ.get('NUM_LAYERS', '4'))

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '4'))
GRAD_ACCUM_STEPS = int(os.environ.get('GRAD_ACCUM_STEPS', '2'))
LR = float(os.environ.get('LR', '1e-4'))
MAX_STEPS = int(os.environ.get('MAX_STEPS', '0'))
SAVE_STEPS = int(os.environ.get('SAVE_STEPS', '50'))
RESHARD_AFTER_FORWARD = os.environ.get('RESHARD_AFTER_FORWARD', '1') == '1'
GRADIENT_CHECKPOINTING = True
IGNORE_MISMATCHED_SIZES = False
DEBUG_LAYER_MEMORY = os.environ.get('DEBUG_LAYER_MEMORY', '0') == '1'
DEBUG_LAYER_MEMORY_STEPS = int(os.environ.get('DEBUG_LAYER_MEMORY_STEPS', '1'))
DEBUG_LAYER_MEMORY_INDEX = int(os.environ.get('DEBUG_LAYER_MEMORY_INDEX', '0'))
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

device_type = Platform.get_platform().device_prefix()
device_mesh = DeviceMesh.from_sizes(
    fsdp_size=MODEL_GPUS,
    dp_size=1,
    device_type=device_type,
)

debug_log(
    f'before_twinkle_initialize mode=local model_gpus={MODEL_GPUS} '
    f'nproc_per_node={NPROC_PER_NODE} device_mesh_world_size={device_mesh.world_size}')
twinkle.initialize(mode='local', nproc_per_node=NPROC_PER_NODE, global_device_mesh=device_mesh)
debug_log('after_twinkle_initialize')


def debug_collectives():
    if os.environ.get('DEBUG_COLLECTIVES', '0') != '1':
        return
    if not dist.is_available() or not dist.is_initialized():
        debug_log('debug_collectives.skip_dist_not_initialized')
        return

    rank = dist.get_rank()
    debug_log('debug_collectives.before_broadcast_object_list')
    obj = [{'rank0': 'ok'}] if rank == 0 else [None]
    dist.broadcast_object_list(obj, src=0)
    debug_log(f'debug_collectives.after_broadcast_object_list obj={obj}')

    device = torch.device(Platform.get_local_device())
    tensor = torch.ones(1, device=device) if rank == 0 else torch.zeros(1, device=device)
    debug_log('debug_collectives.before_tensor_broadcast')
    dist.broadcast(tensor, src=0)
    debug_log(f'debug_collectives.after_tensor_broadcast value={tensor.item()}')


def debug_trainable_parameters(model):
    if os.environ.get('DEBUG_TRAINABLE_PARAMS', '1') != '1':
        return

    module = _unwrap_torch_module(model)
    if module is None:
        debug_log(f'trainable_params.skip model_type={type(model).__name__}')
        return

    items = []
    total = 0
    try:
        named_parameters = list(module.named_parameters())
    except AttributeError as exc:
        debug_log(f'trainable_params.skip named_parameters_failed={type(exc).__name__}: {exc}')
        return
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        shape = tuple(param.shape)
        numel = param.numel()
        items.append((name, shape, numel, str(param.dtype), str(param.device)))
        total += numel

    digest = hashlib.sha256(
        '\n'.join(f'{name}|{shape}|{numel}|{dtype}' for name, shape, numel, dtype, _ in items).encode()
    ).hexdigest()[:16]
    debug_log(f'trainable_params.count={len(items)} total_numel={total} digest={digest}')

    if os.environ.get('DEBUG_TRAINABLE_PARAM_DETAIL', '1') == '1':
        for name, shape, numel, dtype, device in items:
            debug_log(f'trainable_param name={name} shape={shape} numel={numel} dtype={dtype} device={device}')


def _unwrap_torch_module(model):
    module = getattr(model, 'model', model)
    if hasattr(module, 'get_base_model'):
        module = module.get_base_model()
    if not isinstance(module, torch.nn.Module):
        return None
    if not all(hasattr(module, attr) for attr in ('_parameters', '_modules', '_buffers')):
        return None
    return module


def _format_bytes(num_bytes):
    num_bytes = float(num_bytes)
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if abs(num_bytes) < 1024.0 or unit == 'TiB':
            return f'{num_bytes:.2f}{unit}'
        num_bytes /= 1024.0


def _tensor_nbytes(value):
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, (list, tuple)):
        return sum(_tensor_nbytes(item) for item in value)
    if isinstance(value, dict):
        return sum(_tensor_nbytes(item) for item in value.values())
    return 0


def _memory_allocated():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated()
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return torch.npu.memory_allocated()
    return 0


def _memory_reserved():
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved()
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return torch.npu.memory_reserved()
    return 0


def _max_memory_allocated():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated()
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return torch.npu.max_memory_allocated()
    return 0


def _max_memory_reserved():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_reserved()
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return torch.npu.max_memory_reserved()
    return 0


def _reset_peak_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    elif hasattr(torch, 'npu') and torch.npu.is_available():
        torch.npu.reset_peak_memory_stats()


def _find_decoder_layers(root):
    for name, module in root.named_modules():
        if name.endswith('model.layers') and isinstance(module, torch.nn.ModuleList):
            return name, module
    for name, module in root.named_modules():
        if name.endswith('layers') and isinstance(module, torch.nn.ModuleList):
            return name, module
    return None, None


def register_decoder_layer_memory_debugger(model):
    if not DEBUG_LAYER_MEMORY:
        return None

    root = _unwrap_torch_module(model)
    if root is None:
        debug_log(f'layer_memory.skip model_type={type(model).__name__}')
        return None

    layers_name, layers = _find_decoder_layers(root)
    if layers is None:
        debug_log('layer_memory.skip decoder_layers_not_found')
        return None
    if DEBUG_LAYER_MEMORY_INDEX >= len(layers):
        debug_log(f'layer_memory.skip layer_index={DEBUG_LAYER_MEMORY_INDEX} num_layers={len(layers)}')
        return None

    layer = layers[DEBUG_LAYER_MEMORY_INDEX]
    param_bytes = sum(param.numel() * param.element_size() for param in layer.parameters(recurse=True))
    trainable_param_bytes = sum(
        param.numel() * param.element_size() for param in layer.parameters(recurse=True) if param.requires_grad
    )
    param_numel = sum(param.numel() for param in layer.parameters(recurse=True))
    trainable_param_numel = sum(param.numel() for param in layer.parameters(recurse=True) if param.requires_grad)
    debug_log(
        f'layer_memory.layer={layers_name}.{DEBUG_LAYER_MEMORY_INDEX} '
        f'param_numel={param_numel} param_bytes={_format_bytes(param_bytes)} '
        f'trainable_param_numel={trainable_param_numel} '
        f'trainable_param_bytes={_format_bytes(trainable_param_bytes)}')

    state = {'step': 0}

    def pre_hook(_module, inputs):
        if state['step'] >= DEBUG_LAYER_MEMORY_STEPS:
            return
        state['before_allocated'] = _memory_allocated()
        state['before_reserved'] = _memory_reserved()
        _reset_peak_memory_stats()
        debug_log(
            f'layer_memory.step_{state["step"]}.before_forward '
            f'input_tensor_bytes={_format_bytes(_tensor_nbytes(inputs))} '
            f'allocated={_format_bytes(state["before_allocated"])} '
            f'reserved={_format_bytes(state["before_reserved"])}')

    def post_hook(_module, inputs, output):
        if state['step'] >= DEBUG_LAYER_MEMORY_STEPS:
            return
        after_allocated = _memory_allocated()
        after_reserved = _memory_reserved()
        peak_allocated = _max_memory_allocated()
        peak_reserved = _max_memory_reserved()
        before_allocated = state.get('before_allocated', after_allocated)
        before_reserved = state.get('before_reserved', after_reserved)
        debug_log(
            f'layer_memory.step_{state["step"]}.after_forward '
            f'input_tensor_bytes={_format_bytes(_tensor_nbytes(inputs))} '
            f'output_tensor_bytes={_format_bytes(_tensor_nbytes(output))} '
            f'allocated_delta={_format_bytes(after_allocated - before_allocated)} '
            f'peak_delta={_format_bytes(peak_allocated - before_allocated)} '
            f'reserved_delta={_format_bytes(after_reserved - before_reserved)} '
            f'peak_reserved_delta={_format_bytes(peak_reserved - before_reserved)} '
            f'allocated={_format_bytes(after_allocated)} '
            f'reserved={_format_bytes(after_reserved)}')

    handles = [
        layer.register_forward_pre_hook(pre_hook),
        layer.register_forward_hook(post_hook),
    ]
    return state, handles


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
    debug_collectives()

    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=LORA_TARGET_MODULES)
    debug_log('before_add_adapter_to_model')
    model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=GRAD_ACCUM_STEPS)
    debug_log('after_add_adapter_to_model')
    debug_trainable_parameters(model)

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
    layer_memory_debug = register_decoder_layer_memory_debugger(model)

    best_loss = float('inf')
    for step, batch in enumerate(dataloader):
        if MAX_STEPS and step >= MAX_STEPS:
            break
        if callable(batch):
            debug_log(f'step_{step}.before_call_batch')
            batch = batch()
            debug_log(f'step_{step}.after_call_batch')
        if layer_memory_debug is not None:
            layer_memory_debug[0]['step'] = step
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
