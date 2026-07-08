"""GRPO training script for GSM8K dataset with JSONL metric output.

This file is intentionally a standalone variant of short_math_grpo.py. The
training logic stays aligned with that script, while step metrics are written
directly to a metrics.jsonl sidecar for plotting against async RL runs.
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
from twinkle import DeviceGroup, DeviceMesh, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.metric import CompletionRewardMetric
from twinkle.model import TransformersModel
from twinkle.preprocessor.llm import GSM8KProcessor
from twinkle.processor import InputProcessor
from twinkle.reward import GSM8KAccuracyReward
from twinkle.reward.base import Reward
from twinkle.sampler import vLLMSampler

logger = get_logger()

# ========== Configuration ==========
MODEL_ID = os.environ.get('MODEL_ID', 'ms://Qwen/Qwen3.5-4B')
USE_MEGATRON = bool(int(os.environ.get('USE_MEGATRON', '1')))

MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 4))
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 4))
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

NUM_GENERATIONS = int(os.environ.get('NUM_GENERATIONS', 8))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 4096))
LEARNING_RATE = float(os.environ.get('LR', 1e-5))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 1000))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 8))
MINI_BATCH_SIZE = int(os.environ.get('MINI_BATCH_SIZE', 8))
MICRO_BATCH_SIZE = int(os.environ.get('MICRO_BATCH_SIZE', 2))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', 1))
ADAPTER_NAME = os.environ.get('ADAPTER_NAME', 'default')
SAVE_STEPS = int(os.environ.get('SAVE_STEPS', 1000))
LORA_RANK = int(os.environ.get('LORA_RANK', 16))
DATA_NUM = int(os.environ.get('DATA_NUM', '0') or 0)

RUN_ID = os.environ.get('RUN_ID') or (
    f'short_math_seed{os.environ["TWINKLE_SEED"]}' if os.environ.get('TWINKLE_SEED') else 'short_math')
MODE = os.environ.get('MODE', 'short_math_grpo')
METRICS_JSONL = Path(
    os.environ.get('METRICS_JSONL')
    or Path('outputs/async_rl_experiments') / RUN_ID / 'metrics.jsonl')

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
                'source': 'cookbook/rl/short_math_grpo_jsonl.py',
                'reference_source': 'cookbook/rl/short_math_grpo.py',
                'model_id': MODEL_ID,
                'use_megatron': USE_MEGATRON,
                'model_gpus': MODEL_GPUS,
                'sampler_gpus': SAMPLER_GPUS,
                'num_generations': NUM_GENERATIONS,
                'max_new_tokens': MAX_NEW_TOKENS,
                'learning_rate': LEARNING_RATE,
                'max_steps': MAX_STEPS,
                'batch_size': BATCH_SIZE,
                'mini_batch_size': MINI_BATCH_SIZE,
                'micro_batch_size': MICRO_BATCH_SIZE,
                'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
                'lora_rank': LORA_RANK,
                'data_num': DATA_NUM or None,
            },
        )

    def write_train_metrics(self, *, optimizer_step: int, metrics: dict[str, Any]) -> None:
        event_metrics = dict(metrics)
        event_metrics['optimizer_step'] = optimizer_step
        event_metrics['step'] = optimizer_step
        event_metrics['max_steps'] = MAX_STEPS
        self.write_event(
            event='train_batch_done',
            phase='train',
            elapsed_s=_elapsed_s(event_metrics, fallback=time.time() - self.start_time),
            policy_version=optimizer_step,
            metrics=event_metrics,
        )

    def write_completed(self, *, optim_step: int) -> None:
        self.write_event(
            event='run_completed',
            phase='run',
            elapsed_s=time.time() - self.start_time,
            policy_version=optim_step,
            metrics={'optim_step': optim_step},
        )

    def write_event(
        self,
        *,
        event: str,
        phase: str,
        elapsed_s: float,
        metrics: dict[str, Any],
        policy_version: int | None = None,
    ) -> None:
        payload = {
            'ts': time.time(),
            'elapsed_s': elapsed_s,
            'run_id': self.run_id,
            'mode': self.mode,
            'seed': _optional_int(os.environ.get('TWINKLE_SEED') or os.environ.get('SEED')),
            'event': event,
            'phase': phase,
            'adapter_name': ADAPTER_NAME,
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


# ========== Reward Functions ==========
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


# ========== Dataset ==========
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
        max_length=4096,
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


# ========== Main ==========
def main():
    metrics_writer = JSONLMetricsWriter(METRICS_JSONL, run_id=RUN_ID, mode=MODE)
    metrics_writer.write_metadata()
    try:
        _main(metrics_writer)
    finally:
        metrics_writer.close()
        print(METRICS_JSONL)


def _main(metrics_writer: JSONLMetricsWriter):
    device_groups = [
        DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
        DeviceGroup(name='sampler', ranks=list(range(MODEL_GPUS, NUM_GPUS)), device_type='GPU'),
    ]

    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)

    lora_config = LoraConfig(
        target_modules='all-linear',
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        lora_dropout=0.05,
    )

    if USE_MEGATRON:
        from twinkle.model.megatron import MegatronModel
        model = MegatronModel(
            model_id=MODEL_ID,
            device_mesh=model_mesh,
            remote_group='model',
            mixed_precision='bf16',
            variable_seq_lengths=True,
        )
    else:
        model = TransformersModel(
            model_id=MODEL_ID,
            device_mesh=model_mesh,
            remote_group='model',
        )

    model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    if USE_MEGATRON:
        model.set_optimizer('default', lr=LEARNING_RATE)
        model.set_lr_scheduler('default', lr_decay_steps=MAX_STEPS, max_lr=LEARNING_RATE)
    else:
        model.set_optimizer('AdamW', lr=LEARNING_RATE)
        model.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=0)

    model.set_loss('GRPOLoss', epsilon=0.2)
    model.set_processor(InputProcessor, padding_free=True)
    model.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False)

    engine_args = {
        'gpu_memory_utilization': 0.8,
        'max_model_len': 8192,
        'max_lora_rank': max(32, LORA_RANK),
        'enable_lora': True,
        'enable_tower_connector_lora': True,
    }
    sampler_seed = _optional_int(os.environ.get('SAMPLER_SEED'))
    if sampler_seed is not None:
        engine_args['seed'] = sampler_seed
    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args=engine_args,
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False)

    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)

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
    logger.info('Starting GSM8K GRPO training (short reasoning, JSONL metrics)')
    logger.info(get_device_placement())

    for batch in dataloader:
        if optim_step >= MAX_STEPS:
            break

        expand_prompts = []
        for prompt in batch:
            expand_prompts.extend([prompt] * NUM_GENERATIONS)

        ckpt_manager.sync_weights(merge_and_sync=False)
        sampler.reset_prefix_cache()

        sample_responses = sampler.sample(
            expand_prompts,
            sampling_params,
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

            model.forward_backward(
                inputs=mb_inputs,
                old_logps=mb_old_logps,
                advantages=mb_advantages,
                micro_batch_size=MICRO_BATCH_SIZE,
            )
            model.clip_grad_and_step()
            optim_step += 1

            log_dict = reward_metrics_for_slice(
                completion_lengths=mb_completion_lengths,
                total_rewards=mb_total_rewards,
                brevity_rewards=mb_brevity_rewards,
                accuracy_rewards=mb_accuracy_rewards,
            )
            log_dict.update(model.calculate_metric(is_training=True))
            log_dict.update({
                'optimizer_step': optim_step,
                'sample_count': len(mb_inputs),
                'prompt_count': len(mb_inputs) / NUM_GENERATIONS,
                'outer_prompt_count': len(batch),
                'outer_sample_count': total_completions,
                'num_generations': NUM_GENERATIONS,
            })
            logger.info(f'[Step {optim_step}/{MAX_STEPS}] {log_dict}')
            metrics_writer.write_train_metrics(optimizer_step=optim_step, metrics=log_dict)

            if optim_step >= MAX_STEPS:
                break
            if optim_step % SAVE_STEPS == 0:
                model.save(f'math-grpo-checkpoint-{optim_step}')

    logger.info(f'Training completed. optim_steps={optim_step}')
    metrics_writer.write_completed(optim_step=optim_step)
    model.save('math-grpo-final')


if __name__ == '__main__':
    main()

# METRICS_JSONL=outputs/async_rl_experiments/short_math_seed42/metrics.jsonl \
# python cookbook/rl/short_math_grpo_jsonl.py
# RUN_ID=short_math_seed42 \
# TWINKLE_SEED=42 \
# SAMPLER_SEED=42 \
# MODEL_ID=ms://Qwen/Qwen3.5-0.8B \
# USE_MEGATRON=0 \
# MODEL_GPUS=2 \
# SAMPLER_GPUS=1 \
# DATA_NUM=2000 \
# NUM_GENERATIONS=8 \
# MAX_NEW_TOKENS=1024 \
# LR=5e-5 \
# MAX_STEPS=100 \
# BATCH_SIZE=4 \
# MINI_BATCH_SIZE=8 \
# GRADIENT_ACCUMULATION_STEPS=1 \
# LORA_RANK=16 \
# python cookbook/rl/short_math_grpo_jsonl.py
