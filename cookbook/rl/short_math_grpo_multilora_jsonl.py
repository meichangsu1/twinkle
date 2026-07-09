"""Synchronous GSM8K GRPO baseline using MultiLoraTransformersModel.

This script is the single-LoRA sync counterpart for
``cookbook/rl/async_multi_lora_grpo.py``:

  rollout with latest saved adapter_path -> reward/advantage -> train -> save adapter

It intentionally does not use TransferQueue. The goal is to keep the model
class and sampler weight-sync path aligned with async RL while preserving a
stage-barrier synchronous training loop.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from peft import LoraConfig

import twinkle
from twinkle import DeviceGroup, DeviceMesh, Platform, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.metric import CompletionRewardMetric
from twinkle.model import MultiLoraTransformersModel
from twinkle.preprocessor.llm import GSM8KProcessor
from twinkle.processor import InputProcessor
from twinkle.reward import GSM8KAccuracyReward
from twinkle.reward.base import Reward
from twinkle.sampler import vLLMSampler

logger = get_logger()

# ========== Configuration ==========
MODEL_ID = os.environ.get('MODEL_ID', 'ms://Qwen/Qwen3.5-4B')
MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 4))
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 4))
SAMPLER_TP = int(os.environ.get('SAMPLER_TP', '1'))
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

MODEL_TP_SIZE = int(os.environ.get('MODEL_TP_SIZE', '1'))
MODEL_EP_SIZE = int(os.environ.get('MODEL_EP_SIZE', '1'))
MODEL_PP_SIZE = int(os.environ.get('MODEL_PP_SIZE', '1'))
MODEL_DP_SIZE = int(os.environ.get('MODEL_DP_SIZE', str(MODEL_GPUS // (MODEL_TP_SIZE * MODEL_EP_SIZE * MODEL_PP_SIZE))))
SEQUENCE_PARALLEL = bool(int(os.environ.get('SEQUENCE_PARALLEL', '0')))
MIXED_PRECISION = os.environ.get('MIXED_PRECISION', 'bf16')

NUM_GENERATIONS = int(os.environ.get('NUM_GENERATIONS', 8))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 4096))
LEARNING_RATE = float(os.environ.get('LR', 1e-5))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 1000))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 8))
MINI_BATCH_SIZE = int(os.environ.get('MINI_BATCH_SIZE', 8))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', 1))
ADAPTER_NAME = os.environ.get('ADAPTER_NAME', 'tenant_a_gsm8k_lora')
SAVE_STEPS = int(os.environ.get('SAVE_STEPS', 1000))
LORA_RANK = int(os.environ.get('LORA_RANK', 16))
LORA_ALPHA = int(os.environ.get('LORA_ALPHA', LORA_RANK * 2))
LORA_DROPOUT = float(os.environ.get('LORA_DROPOUT', 0.05))
MODEL_MAX_R = int(os.environ.get('MODEL_MAX_R', LORA_RANK))
MAX_LORAS = int(os.environ.get('MAX_LORAS', 1))
MODEL_MAX_LENGTH = int(os.environ.get('MODEL_MAX_LENGTH', 4096))
SAMPLER_MAX_MODEL_LEN = int(os.environ.get('SAMPLER_MAX_MODEL_LEN', 8192))
SAMPLER_MAX_LORA_RANK = int(os.environ.get('SAMPLER_MAX_LORA_RANK', LORA_RANK))
DATA_NUM = int(os.environ.get('DATA_NUM', '0') or 0)
MAX_GRAD_NORM = float(os.environ.get('MAX_GRAD_NORM', 1.0))
NORM_TYPE = int(os.environ.get('NORM_TYPE', 2))
SAVE_OPTIMIZER = bool(int(os.environ.get('SAVE_OPTIMIZER', '0')))
IS_SAMPLER_CHECKPOINT = bool(int(os.environ.get('IS_SAMPLER_CHECKPOINT', '0')))
SYNC_INITIAL_ADAPTER = bool(int(os.environ.get('SYNC_INITIAL_ADAPTER', '0')))

RUN_ID = os.environ.get('RUN_ID') or (
    f'sync_multilora_short_math_seed{os.environ["TWINKLE_SEED"]}'
    if os.environ.get('TWINKLE_SEED') else 'sync_multilora_short_math')
MODE = os.environ.get('MODE', 'sync_multilora_single_lora')
METRICS_JSONL = Path(
    os.environ.get('METRICS_JSONL')
    or Path('outputs/async_rl_experiments') / RUN_ID / 'metrics.jsonl')
ADAPTER_CHECKPOINT_DIR = os.environ.get(
    'ADAPTER_CHECKPOINT_DIR',
    str(Path('output') / 'short_math_grpo_multilora' / RUN_ID / 'lora_sync'),
)
TRAINING_RUN_ID = os.environ.get('TRAINING_RUN_ID', 'gsm8k_grpo_sync_multilora')
TENANT_ID = os.environ.get('TENANT_ID', 'tenant_a')
CONTEXT_KEY = f'{TENANT_ID}/{TRAINING_RUN_ID}/{ADAPTER_NAME}'

SYSTEM_PROMPT = ('You are a helpful math assistant. Solve the problem with minimal but correct reasoning '
                 'and put your final answer within \\boxed{}.')


class JSONLMetricsWriter:

    def __init__(self, path: Path, *, run_id: str, mode: str):
        self.path = path
        self.run_id = run_id
        self.mode = mode
        self.start_time = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('w', encoding='utf-8')

    def close(self) -> None:
        if not self.file.closed:
            self.file.close()

    def write_metadata(self) -> None:
        self.write_event(
            event='run_metadata',
            phase='run',
            elapsed_s=0.0,
            metrics={
                'source': 'cookbook/rl/short_math_grpo_multilora_jsonl.py',
                'reference_source': 'cookbook/rl/short_math_grpo_jsonl.py',
                'model_class': 'MultiLoraTransformersModel',
                'weight_sync': 'adapter_path_after_train',
                'model_id': MODEL_ID,
                'model_gpus': MODEL_GPUS,
                'sampler_gpus': SAMPLER_GPUS,
                'sampler_tp': SAMPLER_TP,
                'num_generations': NUM_GENERATIONS,
                'max_new_tokens': MAX_NEW_TOKENS,
                'learning_rate': LEARNING_RATE,
                'max_steps': MAX_STEPS,
                'batch_size': BATCH_SIZE,
                'mini_batch_size': MINI_BATCH_SIZE,
                'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
                'adapter_name': ADAPTER_NAME,
                'lora_rank': LORA_RANK,
                'model_max_r': MODEL_MAX_R,
                'max_loras': MAX_LORAS,
                'data_num': DATA_NUM or None,
                'sync_initial_adapter': SYNC_INITIAL_ADAPTER,
                'adapter_checkpoint_dir': ADAPTER_CHECKPOINT_DIR,
            },
        )

    def write_train_metrics(self, *, optimizer_step: int, metrics: dict[str, Any], policy_version: int) -> None:
        event_metrics = dict(metrics)
        event_metrics['optimizer_step'] = optimizer_step
        event_metrics['step'] = optimizer_step
        event_metrics['max_steps'] = MAX_STEPS
        self.write_event(
            event='train_batch_done',
            phase='train',
            elapsed_s=_elapsed_s(event_metrics, fallback=time.time() - self.start_time),
            policy_version=policy_version,
            metrics=event_metrics,
        )

    def write_completed(self, *, optim_step: int, policy_version: int) -> None:
        self.write_event(
            event='run_completed',
            phase='run',
            elapsed_s=time.time() - self.start_time,
            policy_version=policy_version,
            metrics={'optim_step': optim_step, 'policy_version': policy_version},
        )

    def write_event(
        self,
        *,
        event: str,
        phase: str,
        elapsed_s: float,
        metrics: dict[str, Any],
        policy_version: int | None = None,
        partition_id: str | None = None,
    ) -> None:
        payload = {
            'ts': time.time(),
            'elapsed_s': elapsed_s,
            'run_id': self.run_id,
            'mode': self.mode,
            'seed': _optional_int(os.environ.get('TWINKLE_SEED') or os.environ.get('SEED')),
            'event': event,
            'phase': phase,
            'context_key': CONTEXT_KEY,
            'adapter_name': ADAPTER_NAME,
            'partition_id': partition_id,
            'policy_version': policy_version,
            'metrics': _json_safe(metrics),
        }
        self.file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + '\n')
        self.file.flush()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _coerce_number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _elapsed_s(metrics: dict[str, Any], *, fallback: float) -> float:
    value = metrics.get('total time elapse')
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.endswith(' seconds'):
            number = _coerce_number(stripped[:-8].strip())
            return number if number is not None else fallback
        if stripped.endswith(' minutes'):
            number = _coerce_number(stripped[:-8].strip())
            return number * 60.0 if number is not None else fallback
        number = _coerce_number(stripped)
        return number if number is not None else fallback
    return fallback


def _safe_name(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', value)


def _adapter_path_from_save_result(save_result: Any) -> str | None:
    if isinstance(save_result, str):
        return save_result
    path = getattr(save_result, 'twinkle_path', None)
    if path is not None:
        return path
    if isinstance(save_result, dict):
        return save_result.get('twinkle_path') or save_result.get('path')
    return None


class GSM8KBrevityReward(Reward):
    """Reward shorter completions that contain a valid answer."""

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for traj in trajectories:
            messages = traj.get('messages', [])
            completion = ''
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    completion = msg.get('content', '')
                    break

            has_answer = bool(
                re.search(r'\\boxed\{[^}]+\}', completion)
                or re.search(r'####\s*[\-\d,\.]+', completion)
            )

            if not has_answer:
                rewards.append(0.0)
            else:
                length = len(completion)
                if length <= 300:
                    rewards.append(1.0)
                else:
                    rewards.append(max(0.0, 1.0 - (length - 300) / 3000))
        return rewards


def create_gsm8k_dataset():
    data_slice = range(DATA_NUM) if DATA_NUM else None
    dataset = Dataset()
    dataset.add_dataset(
        DatasetMeta(
            'ms://modelscope/gsm8k',
            subset_name='main',
            split='train',
            data_slice=data_slice,
        ))
    dataset.set_template(
        'Qwen3_5Template',
        model_id=MODEL_ID,
        max_length=MODEL_MAX_LENGTH,
        truncation_strategy='delete',
        enable_thinking=False,
    )
    dataset.map(GSM8KProcessor(system=SYSTEM_PROMPT))
    dataset.encode(add_generation_prompt=True)
    return dataset


def compute_rewards(
    trajectories: List[Dict[str, Any]],
) -> Tuple[List[float], List[float], List[float]]:
    accuracy_reward_fn = GSM8KAccuracyReward()
    brevity_reward_fn = GSM8KBrevityReward()

    accuracy_rewards = accuracy_reward_fn(trajectories)
    brevity_rewards = brevity_reward_fn(trajectories)
    total_rewards = [a + b for a, b in zip(accuracy_rewards, brevity_rewards)]
    return total_rewards, brevity_rewards, accuracy_rewards


def reward_metrics_for_slice(
    *,
    completion_lengths: List[int],
    total_rewards: List[float],
    brevity_rewards: List[float],
    accuracy_rewards: List[float],
) -> dict[str, Any]:
    metric = CompletionRewardMetric()
    metric.accumulate(
        completion_lengths=completion_lengths,
        rewards={
            'total': total_rewards,
            'brevity': brevity_rewards,
            'accuracy': accuracy_rewards,
        },
    )
    return metric.calculate()


def save_adapter_snapshot(model: MultiLoraTransformersModel, *, policy_version: int) -> str | None:
    name = f'sync-multilora-{_safe_name(RUN_ID)}-{_safe_name(ADAPTER_NAME)}-v{policy_version}'
    save_result = model.save(
        name,
        output_dir=ADAPTER_CHECKPOINT_DIR,
        adapter_name=ADAPTER_NAME,
        save_optimizer=SAVE_OPTIMIZER,
        is_sampler=IS_SAMPLER_CHECKPOINT,
    )
    return _adapter_path_from_save_result(save_result)


def build_model(model_mesh: DeviceMesh) -> MultiLoraTransformersModel:
    lora_config = LoraConfig(
        target_modules='all-linear',
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
    )
    model = MultiLoraTransformersModel(
        model_id=MODEL_ID,
        device_mesh=model_mesh,
        remote_group='model',
        mixed_precision=MIXED_PRECISION,
        max_loras=MAX_LORAS,
        max_r=MODEL_MAX_R,
        max_length=MODEL_MAX_LENGTH,
        target_modules='all-linear',
    )
    model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    model.set_optimizer('AdamW', lr=LEARNING_RATE, adapter_name=ADAPTER_NAME)
    model.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=0, adapter_name=ADAPTER_NAME)
    model.set_loss('GRPOLoss', epsilon=0.2, adapter_name=ADAPTER_NAME)
    model.add_metric('GRPOMetric', adapter_name=ADAPTER_NAME, epsilon=0.2)
    model.set_processor(InputProcessor, adapter_name=ADAPTER_NAME, padding_free=True)
    model.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False, adapter_name=ADAPTER_NAME)
    return model


def build_sampler(sampler_mesh: DeviceMesh) -> vLLMSampler:
    engine_args = {
        'tensor_parallel_size': SAMPLER_TP,
        'gpu_memory_utilization': 0.8,
        'max_model_len': SAMPLER_MAX_MODEL_LEN,
        'max_loras': MAX_LORAS,
        'max_lora_rank': SAMPLER_MAX_LORA_RANK,
        'enable_lora': True,
    }
    sampler_seed = _optional_int(os.environ.get('SAMPLER_SEED'))
    if sampler_seed is not None:
        engine_args['seed'] = sampler_seed
    if bool(int(os.environ.get('ENABLE_TOWER_CONNECTOR_LORA', '0'))):
        engine_args['enable_tower_connector_lora'] = True
    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args=engine_args,
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False)
    return sampler


def main():
    metrics_writer = JSONLMetricsWriter(METRICS_JSONL, run_id=RUN_ID, mode=MODE)
    metrics_writer.write_metadata()
    try:
        _main(metrics_writer)
    finally:
        metrics_writer.close()
        print(METRICS_JSONL)


def _main(metrics_writer: JSONLMetricsWriter):
    device_type = Platform.device_prefix()
    device_groups = [
        DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type=device_type),
        DeviceGroup(
            name='sampler',
            ranks=list(range(MODEL_GPUS, NUM_GPUS)),
            device_type=device_type,
            gpus_per_worker=SAMPLER_TP,
        ),
    ]

    model_mesh = DeviceMesh.from_sizes(
        world_size=MODEL_GPUS,
        dp_size=MODEL_DP_SIZE,
        tp_size=MODEL_TP_SIZE,
        ep_size=MODEL_EP_SIZE,
        pp_size=MODEL_PP_SIZE,
        sequence_parallel=SEQUENCE_PARALLEL,
    )
    sampler_mesh = DeviceMesh.from_sizes(
        world_size=SAMPLER_GPUS,
        dp_size=max(1, SAMPLER_GPUS // SAMPLER_TP),
        tp_size=SAMPLER_TP,
    )
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)

    model = build_model(model_mesh)
    sampler = build_sampler(sampler_mesh)

    global_batch_size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    dataloader = DataLoader(
        dataset=create_gsm8k_dataset,
        batch_size=global_batch_size,
        min_batch_size=global_batch_size,
        device_mesh=model_mesh,
        remote_group='model',
    )

    advantage_fn = GRPOAdvantage()
    sampling_params = SamplingParams(max_tokens=MAX_NEW_TOKENS, num_samples=1, logprobs=1, temperature=1.0, top_p=0.95)

    optim_step = 0
    policy_version = 0
    latest_adapter_path = save_adapter_snapshot(model, policy_version=policy_version) if SYNC_INITIAL_ADAPTER else None

    logger.info('Starting GSM8K GRPO training (MultiLoraTransformersModel sync JSONL baseline)')
    logger.info(get_device_placement())

    for partition_index, batch in enumerate(dataloader):
        if optim_step >= MAX_STEPS:
            break

        partition_id = f'{CONTEXT_KEY}/train_{partition_index}'
        expand_prompts = []
        for prompt in batch:
            expand_prompts.extend([prompt] * NUM_GENERATIONS)

        rollout_start = time.time()
        metrics_writer.write_event(
            event='rollout_started',
            phase='rollout',
            elapsed_s=rollout_start - metrics_writer.start_time,
            policy_version=policy_version,
            partition_id=partition_id,
            metrics={
                'prompt_count': len(batch),
                'sample_count': len(expand_prompts),
                'adapter_path': latest_adapter_path,
            },
        )

        sampler.reset_prefix_cache()
        sample_kwargs: dict[str, Any] = {'adapter_name': ADAPTER_NAME}
        if latest_adapter_path is not None:
            sample_kwargs['adapter_path'] = latest_adapter_path
        sample_responses = sampler.sample(
            expand_prompts,
            sampling_params,
            **sample_kwargs,
        )

        all_input_data: List[Dict[str, Any]] = []
        all_old_logps: List[List[float]] = []
        all_completion_lengths: List[int] = []

        for sample_response in sample_responses:
            for sequence in sample_response.sequences:
                all_input_data.append(sequence.new_input_feature)
                all_old_logps.append([logprob[0][1] for logprob in sequence.logprobs])
                all_completion_lengths.append(len(sequence.tokens))

        total_rewards, brevity_rewards, accuracy_rewards = compute_rewards(all_input_data)
        rollout_latency_s = time.time() - rollout_start
        metrics_writer.write_event(
            event='rollout_done',
            phase='rollout',
            elapsed_s=time.time() - metrics_writer.start_time,
            policy_version=policy_version,
            partition_id=partition_id,
            metrics={
                'prompt_count': len(batch),
                'sample_count': len(all_input_data),
                'rollout_policy_version': policy_version,
                'policy_version_gap': 0.0,
                'rollout_latency_s': rollout_latency_s,
            },
        )

        advantages = advantage_fn(total_rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()

        total_completions = len(all_input_data)
        for mb_start in range(0, total_completions, MINI_BATCH_SIZE):
            mb_end = min(mb_start + MINI_BATCH_SIZE, total_completions)
            mb_inputs = all_input_data[mb_start:mb_end]
            mb_old_logps = all_old_logps[mb_start:mb_end]
            mb_advantages = advantages[mb_start:mb_end]
            mb_completion_lengths = all_completion_lengths[mb_start:mb_end]
            mb_total_rewards = total_rewards[mb_start:mb_end]
            mb_brevity_rewards = brevity_rewards[mb_start:mb_end]
            mb_accuracy_rewards = accuracy_rewards[mb_start:mb_end]

            train_start = time.time()
            model.forward_backward(
                inputs=mb_inputs,
                old_logps=mb_old_logps,
                advantages=mb_advantages,
                adapter_name=ADAPTER_NAME,
            )
            model.clip_grad_and_step(
                adapter_name=ADAPTER_NAME,
                max_grad_norm=MAX_GRAD_NORM,
                norm_type=NORM_TYPE,
            )
            optim_step += 1

            log_dict = reward_metrics_for_slice(
                completion_lengths=mb_completion_lengths,
                total_rewards=mb_total_rewards,
                brevity_rewards=mb_brevity_rewards,
                accuracy_rewards=mb_accuracy_rewards,
            )
            log_dict.update(model.calculate_metric(is_training=True, adapter_name=ADAPTER_NAME))
            log_dict.update({
                'optimizer_step': optim_step,
                'sample_count': len(mb_inputs),
                'prompt_count': len(mb_inputs) / NUM_GENERATIONS,
                'outer_prompt_count': len(batch),
                'outer_sample_count': total_completions,
                'num_generations': NUM_GENERATIONS,
                'rollout_policy_version': policy_version,
                'policy_version_gap': 0.0,
                'train_batch_latency_s': time.time() - train_start,
            })
            logger.info(f'[Step {optim_step}/{MAX_STEPS}] {log_dict}')
            metrics_writer.write_train_metrics(
                optimizer_step=optim_step,
                metrics=log_dict,
                policy_version=policy_version,
            )

            if optim_step >= MAX_STEPS:
                break
            if optim_step % SAVE_STEPS == 0:
                model.save(
                    f'math-grpo-multilora-checkpoint-{optim_step}',
                    output_dir=ADAPTER_CHECKPOINT_DIR,
                    adapter_name=ADAPTER_NAME,
                    save_optimizer=SAVE_OPTIMIZER,
                    is_sampler=IS_SAMPLER_CHECKPOINT,
                )

        sync_start = time.time()
        policy_version += 1
        latest_adapter_path = save_adapter_snapshot(model, policy_version=policy_version)
        metrics_writer.write_event(
            event='weight_sync_done',
            phase='train',
            elapsed_s=time.time() - metrics_writer.start_time,
            policy_version=policy_version,
            partition_id=partition_id,
            metrics={
                'adapter_path': latest_adapter_path,
                'optimizer_step': optim_step,
                'sample_count': total_completions,
                'weight_sync_latency_s': time.time() - sync_start,
            },
        )
        metrics_writer.write_event(
            event='partition_train_done',
            phase='train',
            elapsed_s=time.time() - metrics_writer.start_time,
            policy_version=policy_version,
            partition_id=partition_id,
            metrics={
                'optimizer_step': optim_step,
                'prompt_count': len(batch),
                'sample_count': total_completions,
            },
        )

    logger.info(f'Training completed. optim_steps={optim_step}, policy_version={policy_version}')
    metrics_writer.write_completed(optim_step=optim_step, policy_version=policy_version)
    model.save(
        'math-grpo-multilora-final',
        output_dir=ADAPTER_CHECKPOINT_DIR,
        adapter_name=ADAPTER_NAME,
        save_optimizer=SAVE_OPTIMIZER,
        is_sampler=IS_SAMPLER_CHECKPOINT,
    )


if __name__ == '__main__':
    main()

# Example:
# RUN_ID=sync_multilora_single_lora_seed42 \
# MODE=sync_multilora_single_lora \
# TWINKLE_SEED=42 \
# SAMPLER_SEED=42 \
# MODEL_ID=ms://Qwen/Qwen3.5-0.8B \
# MODEL_GPUS=2 \
# SAMPLER_GPUS=1 \
# SAMPLER_TP=1 \
# DATA_NUM=2000 \
# NUM_GENERATIONS=8 \
# MAX_NEW_TOKENS=1024 \
# LR=5e-5 \
# MAX_STEPS=100 \
# BATCH_SIZE=4 \
# MINI_BATCH_SIZE=8 \
# GRADIENT_ACCUMULATION_STEPS=1 \
# LORA_RANK=16 \
# MODEL_MAX_R=16 \
# SAMPLER_MAX_LORA_RANK=16 \
# ADAPTER_NAME=tenant_a_gsm8k_lora \
# python cookbook/rl/short_math_grpo_multilora_jsonl.py
