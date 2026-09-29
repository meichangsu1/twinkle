# Copyright (c) ModelScope Contributors. All rights reserved.
"""Exercise the full DSV4 GRPO actor without rollout or weight synchronization.

This is a memory/compute-path probe, not a training-quality test: completion
token IDs and advantages are synthetic. Run it against the same actor base,
FSDP/EP topology, and sequence length as the GRPO job.
"""
import os
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig

import twinkle
from twinkle import DeviceGroup, DeviceMesh, remote_class, remote_function
from twinkle.model import MultiLoraTransformersModel
from twinkle.processor import InputProcessor
from twinkle.template import DeepseekV4Template
from twinkle.utils.platforms import ensure_npu_backend


TENANT = 'tenant_a'
GIB = 1024**3


@remote_class()
class PreflightActor(MultiLoraTransformersModel):

    @remote_function(dispatch='all', lazy_collect=False)
    def npu_memory(self):
        """Return every rank's current and peak allocator usage."""
        import torch.distributed as dist

        torch.npu.synchronize()
        return {
            'rank': dist.get_rank(),
            'allocated_gib': round(torch.npu.memory_allocated() / GIB, 3),
            'reserved_gib': round(torch.npu.memory_reserved() / GIB, 3),
            'peak_allocated_gib': round(torch.npu.max_memory_allocated() / GIB, 3),
        }


def _positive_env(name, default):
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f'{name} must be positive')
    return value


def _first_dapo_rows(path, count):
    import pyarrow.parquet as pq

    parquet = path / 'dapo-math-17k.parquet' if path.is_dir() else path
    if not parquet.is_file():
        raise FileNotFoundError(parquet)
    rows = []
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=count, columns=['prompt', 'reward_model']):
        rows.extend(batch.to_pylist())
        if len(rows) >= count:
            break
    if len(rows) < count:
        raise ValueError(f'DAPO data has only {len(rows)} rows; need {count}')
    return rows[:count]


def _make_features(model_path, data_path, count, generations, completion_tokens, max_length, vocab_size):
    from .dsv4_dapo import DAPOMathProcessor

    template = DeepseekV4Template(model_id=str(model_path))
    processor = DAPOMathProcessor()
    rng = random.Random(20260929)
    token_ceiling = min(vocab_size, 120000)
    if token_ceiling <= 32:
        raise ValueError(f'Unexpected vocab_size={vocab_size}')
    features = []
    for row in _first_dapo_rows(data_path, count // generations):
        prompt = template.encode(processor.preprocess(row), add_generation_prompt=True)
        prompt_length = len(prompt['input_ids'])
        if prompt_length + completion_tokens > max_length:
            raise ValueError(
                f'Prompt {prompt_length} + completion {completion_tokens} exceeds actor max_length {max_length}')
        for _ in range(generations):
            # Valid, non-special IDs give a real forward with varied routing.
            # They are not model-generated answers or a reward-quality test.
            completion = [rng.randrange(32, token_ceiling) for _ in range(completion_tokens)]
            features.append(template.concat_input_feature(prompt, completion))
    return features


def _memory_summary(actor, stage):
    snapshots = actor.npu_memory()
    if isinstance(snapshots, dict):
        snapshots = [snapshots]
    highest = max(snapshots, key=lambda entry: entry['peak_allocated_gib'])
    print(f'{stage}: ranks={len(snapshots)} highest_peak={highest}', flush=True)


def main():
    ensure_npu_backend()
    if not torch.npu.is_available():
        raise RuntimeError('NPU runtime is unavailable on the driver')
    model_path = Path(os.environ['ACTOR_MODEL']).expanduser().resolve()
    data_path = Path(os.environ['DAPO_PATH']).expanduser().resolve()
    if not (model_path / 'config.json').is_file():
        raise FileNotFoundError(model_path / 'config.json')
    actor_devices = _positive_env('ACTOR_NPUS', '32')
    ep = _positive_env('ACTOR_EP', str(actor_devices))
    per_node = _positive_env('NPUS_PER_NODE', '16')
    rank = _positive_env('LORA_R', '8')
    alpha = _positive_env('LORA_ALPHA', '32')
    max_length = _positive_env('ACTOR_MAX_LENGTH', os.environ.get('MAX_MODEL_LEN', '8192'))
    completion_tokens = _positive_env('PREFLIGHT_COMPLETION_TOKENS', os.environ.get('MAX_NEW_TOKENS', '4096'))
    batch = _positive_env('BATCH_SIZE', '64')
    generations = _positive_env('NUM_GENERATIONS', '4')
    if actor_devices % generations or batch * generations % actor_devices:
        raise ValueError('NUM_GENERATIONS must divide ACTOR_NPUS and BATCH_SIZE * NUM_GENERATIONS '
                         'must be divisible by ACTOR_NPUS')
    microbatches = _positive_env('PREFLIGHT_MICROBATCHES', str(batch * generations // actor_devices))
    precision = os.environ.get('ACTOR_PRECISION', 'bf16')
    if precision not in ('bf16', 'fp16'):
        raise ValueError('ACTOR_PRECISION must be bf16 or fp16')
    if actor_devices % ep:
        raise ValueError('ACTOR_EP must divide ACTOR_NPUS')

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True)
    if config.n_routed_experts % ep:
        raise ValueError('ACTOR_EP must divide n_routed_experts')
    config.use_cache = False
    features = _make_features(model_path, data_path, actor_devices, generations, completion_tokens, max_length,
                              config.vocab_size)
    lengths = [len(feature['input_ids']) for feature in features]
    print(f'Actor-only preflight: ranks={actor_devices} EP={ep} microbatches={microbatches} '
          f'prompt+completion lengths={min(lengths)}..{max(lengths)}', flush=True)

    address = os.environ.get('RAY_ADDRESS', '').strip()
    if not address:
        raise ValueError('Set RAY_ADDRESS to the existing multi-node Ray head')
    import ray

    ray.init(address=address, ignore_reinit_error=True)
    if ray.cluster_resources().get('NPU', 0) < actor_devices:
        raise RuntimeError(f'Ray cluster has fewer than {actor_devices} NPU resources')
    mesh = DeviceMesh.from_sizes(fsdp_size=actor_devices, dp_size=1, ep_size=ep, device_type='npu')
    twinkle.initialize(
        mode='ray',
        nproc_per_node=per_node,
        lazy_collect=False,
        groups=[DeviceGroup(name='actor', ranks=list(range(actor_devices)), device_type='NPU')],
    )
    placeholder = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0, target_modules=['q_a_proj'])
    actor = PreflightActor(
        model_id=str(model_path),
        config=config,
        remote_group='actor',
        device_mesh=mesh,
        dtype=torch.bfloat16 if precision == 'bf16' else torch.float16,
        strategy='native_fsdp',
        mixed_precision=precision,
        memory_efficient_init=True,
        max_loras=1,
        max_r=rank,
        max_length=max_length,
        lora_config=placeholder,
        fsdp_config={'expert_parallel': {'enabled': ep > 1, 'router_dtype': 'fp32'}},
    )
    lora = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0,
        target_modules=[],
        target_parameters=['mlp.experts.gate_up_proj', 'mlp.experts.down_proj'],
    )
    actor.add_adapter_to_model(TENANT, lora, gradient_accumulation_steps=1)
    actor.set_optimizer('AdamW', lr=float(os.environ.get('LR', '5e-6')), adapter_name=TENANT)
    actor.set_loss('GRPOLoss', beta=0.0, epsilon=0.2, adapter_name=TENANT)
    actor.set_processor(InputProcessor, adapter_name=TENANT)
    actor.set_template('DeepseekV4Template', model_id=str(model_path), adapter_name=TENANT)
    _memory_summary(actor, 'before_forward')

    start = time.monotonic()
    for index in range(microbatches):
        # Omitting old_logps makes GRPOLoss use detached current logprobs.
        # Nonzero synthetic advantages still exercise the actual backward path.
        actor.forward_backward(inputs=features, advantages=[1.0] * actor_devices, adapter_name=TENANT)
        _memory_summary(actor, f'after_microbatch_{index + 1}')
    actor.clip_grad_and_step(adapter_name=TENANT)
    _memory_summary(actor, 'after_optimizer_step')
    print(f'PASSED actor-only preflight in {time.monotonic() - start:.1f}s; '
          'this does not validate rollout, weight sync, or reward quality', flush=True)


if __name__ == '__main__':
    main()
