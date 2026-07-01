# Copyright (c) ModelScope Contributors. All rights reserved.
"""Async RL GRPO on GSM8K with TransferQueueDataPlane.

Uses BaseRLPipeline + MultiLoraTransformersModel + TransferQueueDataPlane
to run the full rollout -> reward -> advantage -> train -> clear lifecycle.

Requirements:
  pip install TransferQueue
  pip install -e ".[transformers,ray,test]"

Local data:
  Model: /data/model/Qwen3.5-0.8B
  Data:  /data/gsm8k_train.parquet

Usage:
  python cookbook/rl/async_rl_grpo_gsm8k.py

  # Custom paths
  MODEL_ID=/path/to/model GSM8K_DATA=/path/to/data.parquet python cookbook/rl/async_rl_grpo_gsm8k.py
"""
import os
from typing import Any, Dict, List

from peft import LoraConfig

from twinkle.dataset import Dataset, DatasetMeta
from twinkle.reward import GSM8KAccuracyReward, GSM8KFormatReward
from twinkle_agentic.async_rl import (
    BaseRLPipeline,
    BaseRLPipelineConfig,
    PartitionStatus,
    TransferQueueDataPlane,
    TransferQueueRuntimeConfig,
)

MODEL_ID = os.environ.get('MODEL_ID', '/data/model/Qwen3.5-0.8B')
GSM8K_DATA = os.environ.get('GSM8K_DATA', '/data/gsm8k_train.parquet')
NUM_GENERATIONS = int(os.environ.get('NUM_GENERATIONS', 4))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 1024))
LEARNING_RATE = float(os.environ.get('LR', 2e-5))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 10))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 4))
ADAPTER_NAME = 'gsm8k_lora'


def create_gsm8k_prompts(limit: int = 64) -> List[Dict[str, Any]]:
    dataset = Dataset(DatasetMeta(GSM8K_DATA))
    samples = []
    for i, item in enumerate(dataset):
        if i >= limit:
            break
        samples.append({
            'sample_id': f'gsm8k_{i}',
            'messages': item.get('messages', []),
            'group_id': f'group_{i}',
            'generation_idx': 0,
            'user_data': item.get('user_data', []),
        })
    return samples


def compute_gsm8k_rewards(trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
    accuracy_reward_fn = GSM8KAccuracyReward()
    format_reward_fn = GSM8KFormatReward()
    accuracy_rewards = accuracy_reward_fn(trajectories)
    format_rewards = format_reward_fn(trajectories)
    return [a + f for a, f in zip(accuracy_rewards, format_rewards)]


def main():
    import twinkle
    from twinkle import DeviceMesh, DeviceGroup, get_logger
    from twinkle.model.transformers.multi_lora_transformers import MultiLoraTransformersModel
    from twinkle.processor import InputProcessor
    from twinkle.sampler import vLLMSampler
    from twinkle.checkpoint_engine import CheckpointEngineManager
    from twinkle_agentic.rollout import MultiTurnRollout

    logger = get_logger()

    MODEL_GPUS = 4
    SAMPLER_GPUS = 4
    NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

    device_groups = [
        DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='NPU'),
        DeviceGroup(name='sampler', ranks=list(range(MODEL_GPUS, NUM_GPUS)), device_type='NPU'),
    ]
    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)

    lora_config = LoraConfig(
        target_modules=[
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj',
        ],
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    )

    logger.info(f'Loading model from {MODEL_ID}...')
    model = MultiLoraTransformersModel(
        model_id=MODEL_ID,
        device_mesh=model_mesh,
        remote_group='model',
        mixed_precision='bf16',
    )
    model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=1)
    model.set_optimizer('AdamW', lr=LEARNING_RATE)
    model.set_loss('GRPOLoss', epsilon=0.2)
    model.set_processor(InputProcessor)
    model.set_template('Qwen3_5Template', model_id=MODEL_ID)

    logger.info('Initializing vLLM sampler...')
    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.8,
            'max_model_len': 1536,
            'max_lora_rank': 16,
            'enable_lora': True,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template('Qwen3_5Template', model_id=MODEL_ID)

    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)

    class GSM8KGRPOPipeline(BaseRLPipeline):

        def build_model(self):
            return model

        def build_rollout(self):
            return MultiTurnRollout(sampler=sampler, max_turns=1)

        def build_reward_registry(self):
            return {'gsm8k': compute_gsm8k_rewards}

        def build_data_plane(self):
            return TransferQueueDataPlane(
                tq_config=TransferQueueRuntimeConfig(
                    num_data_storage_units=4,
                    total_storage_size=100000,
                ),
            )

    config = BaseRLPipelineConfig(
        tenant_id='test_tenant',
        training_run_id='gsm8k_grpo',
        base_model_id=MODEL_ID,
        adapter_name=ADAPTER_NAME,
        reward_type='gsm8k',
        loss_type='grpo',
        algorithm='grpo',
        max_staleness=0,
        target_groups_per_partition=NUM_GENERATIONS,
        max_train_partitions=MAX_STEPS,
        reward_batch_size=1024,
        advantage_batch_size=1024,
    )

    pipeline = GSM8KGRPOPipeline(config=config)

    prompt_samples = create_gsm8k_prompts(limit=BATCH_SIZE * MAX_STEPS)
    pipeline.submit_rollout_samples(prompt_samples)

    history = pipeline.run(max_steps=MAX_STEPS)

    trained = sum(1 for step in history if step['train'] is not None)
    print(f'Trained {trained} partitions')

    partitions = pipeline.data_plane.list_partitions(pipeline.context)
    cleared = [p for p in partitions if p.status == PartitionStatus.CLEARED]
    print(f'Cleared {len(cleared)} / {len(partitions)} partitions')

    assert trained > 0, 'No partitions were trained'
    assert len(cleared) == trained, f'Expected {trained} cleared partitions, got {len(cleared)}'
    print('E2E verification PASSED')


if __name__ == '__main__':
    main()
