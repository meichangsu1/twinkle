# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import json
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

from twinkle import get_logger
from twinkle.data_format import Trajectory
from .data_plane import TransferQueueDataPlane, TransferQueueRuntimeConfig
from .metrics import AsyncRLMetricsConfig
from .pipeline import BaseRLPipeline, BaseRLPipelineConfig
from .prompt_loader import PromptLoader
from .scheduling import (PreferCurrentTrainPolicy, WeightedFairRolloutPolicy, WeightedFairTrainPolicy,
                         WorkConservingRolloutPolicy)
from .tq_utils import read_train_batch
from .types import GRPOAdvantageBatch, LoraContext, RolloutOutput
from .workers import TrainerStepResult

logger = get_logger()

_LORA_CONTEXT_MODEL_RESOURCE_KEYS = {
    'mesh',
    'mixed_precision',
    'strategy',
    'ddp_config',
    'fsdp_config',
    'grad_scaler_config',
    'memory_efficient_init',
    'max_loras',
    'max_r',
    'adapter_checkpoint_dir',
}


def lora_context_configs(cfg) -> list[Any]:
    if cfg.get('lora_contexts'):
        return list(cfg.lora_contexts)
    return [cfg.lora_context]


def primary_lora_context(cfg) -> Any:
    return lora_context_configs(cfg)[0]


def build_lora_contexts(cfg) -> list[LoraContext]:
    contexts = []
    for context_cfg in lora_context_configs(cfg):
        contexts.append(
            LoraContext(
                tenant_id=context_cfg.tenant_id,
                training_run_id=context_cfg.training_run_id,
                base_model_id=context_cfg.base_model_id,
                adapter_name=context_cfg.adapter_name,
                reward_type=context_cfg.reward_type,
                tool_profile=context_cfg.get('tool_profile', 'default'),
                algorithm=context_cfg.get('algorithm', cfg.pipeline.get('algorithm', 'grpo')),
                rollout_profile=context_cfg.get('rollout_profile', 'default'),
            ))
    base_models = {context.base_model_id for context in contexts}
    if len(base_models) != 1:
        raise ValueError(f'one async multi-LoRA job must use one base model, got {sorted(base_models)}')
    return contexts


def context_dataset_config(cfg, context_cfg):
    dataset_cfg = context_cfg.get('dataset')
    if dataset_cfg is not None:
        return dataset_cfg
    dataset_cfg = cfg.get('dataset')
    if dataset_cfg is not None:
        return dataset_cfg
    raise ValueError('dataset config is required for each lora context when top-level dataset is not set: '
                     f'training_run_id={context_cfg.training_run_id}, adapter_name={context_cfg.adapter_name}')


def lora_model_config(cfg, context_cfg):
    from omegaconf import OmegaConf

    context_model_cfg = context_cfg.get('model')
    if context_model_cfg is None:
        return cfg.model
    unsupported = sorted(set(context_model_cfg.keys()) & _LORA_CONTEXT_MODEL_RESOURCE_KEYS)
    if unsupported:
        raise ValueError('lora_context.model can only override adapter-level model config. '
                         f'Unsupported resource-level keys for {context_cfg.adapter_name}: {unsupported}')
    return OmegaConf.merge(cfg.model, context_model_cfg)


def lora_context_config_for_context(cfg, context: LoraContext):
    for context_cfg in lora_context_configs(cfg):
        if (context_cfg.tenant_id == context.tenant_id and context_cfg.training_run_id == context.training_run_id
                and context_cfg.adapter_name == context.adapter_name):
            return context_cfg
    raise KeyError(f'cannot find lora context config for {context.key}')


def lora_model_config_for_context(cfg, context: LoraContext):
    return lora_model_config(cfg, lora_context_config_for_context(cfg, context))


def lora_rollout_config(cfg, context_cfg):
    from omegaconf import OmegaConf

    rollout_defaults = OmegaConf.create({'batch_size': cfg.pipeline.default_rollout_batch_size})
    return OmegaConf.merge(rollout_defaults, cfg.pipeline.rollout, context_cfg.get('rollout') or {})


def rollout_batch_groups_for_context(cfg, context_cfg) -> int:
    return int(lora_rollout_config(cfg, context_cfg).batch_size)


def max_pending_prompt_groups_for_context(cfg, context_cfg) -> int:
    rollout_batch_groups = rollout_batch_groups_for_context(cfg, context_cfg)
    staleness_window = max(1, int(cfg.pipeline.max_staleness) + 1)
    return rollout_batch_groups * staleness_window


def rollout_batch_groups_by_context(cfg, contexts: list[LoraContext]) -> dict[str, int]:
    result = {}
    for context_cfg, context in zip(lora_context_configs(cfg), contexts):
        result[context.key] = rollout_batch_groups_for_context(cfg, context_cfg)
    return result


def mini_batch_size_for_context(cfg, context_cfg) -> int | None:
    train_cfg = context_cfg.get('train') or {}
    if train_cfg.get('mini_batch_size') is None:
        return None
    return int(train_cfg.mini_batch_size)


def mini_batch_size_by_context(cfg, contexts: list[LoraContext]) -> dict[str, int]:
    result = {}
    for context_cfg, context in zip(lora_context_configs(cfg), contexts):
        mini_batch_size = mini_batch_size_for_context(cfg, context_cfg)
        if mini_batch_size is not None:
            result[context.key] = mini_batch_size
    return result


def async_rl_metrics_config(cfg) -> AsyncRLMetricsConfig | None:
    experiment_cfg = cfg.get('experiment') or {}
    metrics_cfg = experiment_cfg.get('metrics') or {}
    if metrics_cfg.get('enabled') is False:
        return None
    max_steps = _optional_step_limit(cfg.pipeline.get('max_steps'))
    metadata = {
        'model_id': cfg.model.get('model_id', None),
        'max_steps': max_steps,
        'max_staleness': int(cfg.pipeline.max_staleness),
        'default_rollout_batch_size': int(cfg.pipeline.default_rollout_batch_size),
        'default_mini_batch_size': int(cfg.pipeline.default_mini_batch_size),
        'rollout_policy': str(cfg.pipeline.get('rollout_policy', 'work_conserving')),
        'train_policy': str(cfg.pipeline.get('train_policy', 'prefer_current')),
        'num_loras': len(lora_context_configs(cfg)),
    }
    return AsyncRLMetricsConfig(
        run_id=str(experiment_cfg.get('run_id') or f'async_rl_{int(time.time())}'),
        mode=str(experiment_cfg.get('mode') or 'async'),
        seed=experiment_cfg.get('seed'),
        output_dir=str(metrics_cfg.get('output_dir', 'outputs/async_rl_experiments')),
        enable_jsonl=bool(metrics_cfg.get('enable_jsonl', True)),
        enable_swanlab=bool(metrics_cfg.get('enable_swanlab', False)),
        swanlab_project=str(metrics_cfg.get('swanlab_project', 'twinkle')),
        swanlab_experiment_name=metrics_cfg.get('swanlab_experiment_name'),
        swanlab_mode=str(metrics_cfg.get('swanlab_mode', 'local')),
        swanlab_logdir=metrics_cfg.get('swanlab_logdir'),
        metadata=metadata,
    )


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return cfg.get(key, default)


def _cfg_text(cfg: Any, key: str, default: str) -> str:
    value = _cfg_get(cfg, key, default)
    return str(default if value is None else value)


def _config_to_dict(cfg: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf

    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    return dict(OmegaConf.to_container(cfg, resolve=True))


def _config_kwargs(cfg: Any, *, exclude: set[str] | None = None) -> dict[str, Any]:
    excluded = {'cls'}
    if exclude:
        excluded.update(exclude)
    return {key: value for key, value in _config_to_dict(cfg).items() if key not in excluded}


def _dataset_format(dataset_cfg: Any, dataset_id: str) -> str:
    suffix = Path(dataset_id).suffix.lower().lstrip('.')
    if suffix in {'json', 'jsonl'}:
        return suffix
    configured = _cfg_get(dataset_cfg, 'format') or _cfg_get(dataset_cfg, 'file_type')
    if configured:
        return str(configured).lower().lstrip('.')
    if suffix:
        return suffix
    return ''


def _read_local_json_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == '.jsonl':
        rows = []
        with path.open(encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    with path.open(encoding='utf-8') as file:
        payload = json.load(file)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ('data', 'rows', 'train'):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f'JSON dataset must be a row list or contain data/rows/train list: {path}')


def _local_json_dataset_path(dataset_cfg: Any, dataset_id: str) -> Path | None:
    path = Path(dataset_id)
    dataset_format = _dataset_format(dataset_cfg, dataset_id)
    if path.is_file() and dataset_format in {'json', 'jsonl'}:
        return path
    if path.is_dir() and dataset_format not in {'parquet', 'csv', 'arrow'}:
        split = str(_cfg_get(dataset_cfg, 'split', 'train'))
        suffixes = [dataset_format] if dataset_format in {'json', 'jsonl'} else []
        suffixes.extend(['jsonl', 'json'])
        for suffix in suffixes:
            candidate = path / f'{split}.{suffix}'
            if candidate.exists():
                return candidate
    return None


def _build_lora_config(lora_cfg):
    from peft import LoraConfig

    kwargs = _config_to_dict(lora_cfg)
    if 'r' in kwargs:
        kwargs['r'] = int(kwargs['r'])
    if 'lora_alpha' in kwargs:
        kwargs['lora_alpha'] = int(kwargs['lora_alpha'])
    if 'lora_dropout' in kwargs:
        kwargs['lora_dropout'] = float(kwargs['lora_dropout'])
    return LoraConfig(**kwargs)


def _model_dp_size(cfg) -> int:
    mesh_cfg = cfg.model.mesh
    model_gpus = int(cfg.runtime.model_gpus)
    tp_size = int(mesh_cfg.get('tp_size', 1))
    ep_size = int(mesh_cfg.get('ep_size', 1))
    pp_size = int(mesh_cfg.get('pp_size', 1))
    parallel_size = tp_size * ep_size * pp_size
    if model_gpus % parallel_size != 0:
        raise ValueError(f'runtime.model_gpus={model_gpus} must be divisible by '
                         f'model.mesh tp_size*ep_size*pp_size={parallel_size}')
    return int(mesh_cfg.get('dp_size', model_gpus // parallel_size))


def _optional_step_limit(value: Any) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    return None if parsed <= 0 else parsed


def _validate_train_batch_config(cfg) -> None:
    mini_batch_groups = int(cfg.pipeline.default_mini_batch_size)
    num_generations = int(cfg.pipeline.rollout.get('num_generations', 1))
    for context_cfg in lora_context_configs(cfg):
        rollout_batch_groups = rollout_batch_groups_for_context(cfg, context_cfg)
        if rollout_batch_groups % mini_batch_groups != 0:
            raise ValueError('resolved rollout batch size must be divisible by pipeline.default_mini_batch_size. '
                             'Both values are measured in prompt groups. '
                             f'Got context={context_cfg.adapter_name}, rollout_batch_size={rollout_batch_groups}, '
                             f'default_mini_batch_size={mini_batch_groups}.')
    mini_batch_samples = mini_batch_groups * num_generations
    dp_size = _model_dp_size(cfg)
    if mini_batch_samples < dp_size:
        raise ValueError('pipeline.default_mini_batch_size is measured in prompt groups, and '
                         'default_mini_batch_size * pipeline.rollout.num_generations must be >= '
                         'model data-parallel size. '
                         f'Got default_mini_batch_size={mini_batch_groups}, num_generations={num_generations}, '
                         f'mini_batch_samples={mini_batch_samples}, model_dp_size={dp_size}. '
                         'Increase pipeline.default_mini_batch_size or pipeline.rollout.num_generations, '
                         'or reduce model.mesh.dp_size.')


def _metric_payload(metric: Any) -> dict[str, Any]:
    if metric is None:
        return {}
    result = getattr(metric, 'result', None)
    if result is not None:
        return result
    if hasattr(metric, 'model_dump'):
        dumped = metric.model_dump()
        return dumped.get('result', dumped)
    if isinstance(metric, dict):
        return metric.get('result', metric)
    return {'metric': metric}


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, 'item'):
        try:
            value = value.item()
        except Exception:
            return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        if len(value) == 1:
            return _as_float(value[0])
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _numeric_values(values: Any) -> list[float]:
    if values is None:
        return []
    if hasattr(values, 'detach'):
        values = values.detach()
    if hasattr(values, 'cpu'):
        values = values.cpu()
    if hasattr(values, 'flatten'):
        values = values.flatten()
    if hasattr(values, 'tolist'):
        values = values.tolist()
    if isinstance(values, (list, tuple)):
        result = []
        for value in values:
            converted = _as_float(value)
            if converted is not None:
                result.append(converted)
        return result
    converted = _as_float(values)
    return [] if converted is None else [converted]


def _sequence_len(value: Any) -> int:
    if value is None:
        return 0
    try:
        return len(value)
    except TypeError:
        return 0


def _stats(values: Any) -> dict[str, float]:
    values = _numeric_values(values)
    if not values:
        return {}
    count = len(values)
    mean = sum(values) / count
    variance = sum((value - mean)**2 for value in values) / count
    return {
        'mean': mean,
        'std': variance**0.5,
        'min': min(values),
        'max': max(values),
    }


def _prefixed_stats(prefix: str, values: Any) -> dict[str, float]:
    return {f'{prefix}_{key}': value for key, value in _stats(values).items()}


def _async_train_batch_data_diagnostics(
    *,
    inputs: list[dict[str, Any]],
    rewards: list[float],
    advantages: list[float],
    logprobs: list[list[float]],
) -> dict[str, Any]:
    logprobs_lens = []
    input_lens = []

    for model_input, sample_logprobs in zip(inputs, logprobs):
        logprobs_lens.append(float(_sequence_len(sample_logprobs)))
        input_ids = model_input.get('input_ids')
        input_lens.append(float(_sequence_len(input_ids)))

    diagnostics: dict[str, Any] = {
        'sample_count': len(inputs),
    }
    diagnostics.update(_prefixed_stats('tq_reward', rewards))
    diagnostics.update(_prefixed_stats('advantage', advantages))
    diagnostics.update(_prefixed_stats('logprobs_len', logprobs_lens))
    diagnostics.update(_prefixed_stats('input_len', input_lens))
    return diagnostics


def _short_math_reward_metrics(
    records: list[dict[str, Any]],
    *,
    total_rewards: list[float] | None = None,
) -> dict[str, Any]:
    """Return reward metrics using the same keys as cookbook/rl/short_math_grpo.py."""
    from twinkle.metric import CompletionRewardMetric

    metadata_items = [dict(record.get('metadata') or record) for record in records]
    totals = _numeric_values(total_rewards) if total_rewards is not None else _record_values(
        metadata_items,
        'total_reward',
    )
    brevity = _record_values(metadata_items, 'brevity_reward')
    accuracy = _record_values(metadata_items, 'accuracy_reward')
    completion_lengths = _record_values(metadata_items, 'completion_length')
    rewards = {}
    if totals:
        rewards['total'] = totals
    if brevity:
        rewards['brevity'] = brevity
    if accuracy:
        rewards['accuracy'] = accuracy
    metric = CompletionRewardMetric()
    metric.accumulate(rewards=rewards, completion_lengths=completion_lengths)
    return metric.calculate()


def _record_values(records: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        number = _as_float(value)
        if number is not None:
            values.append(number)
    return values


def build_prompt_dataset_from_config(context_cfg: dict[str, Any], template_cfg: dict[str, Any]):
    """Build a prompt dataset inside the DataLoader worker.

    This function must stay top-level and only receive serializable config
    values, because Ray serializes the callable passed to twinkle.dataloader.DataLoader.
    """
    import twinkle.preprocessor
    from twinkle.dataset import Dataset, DatasetMeta
    from twinkle.preprocessor import Preprocessor
    from twinkle.utils import construct_class

    dataset_cfg = _cfg_get(context_cfg, 'dataset')
    if dataset_cfg is None:
        raise ValueError(f'lora context {context_cfg.get("training_run_id")} has no dataset config')
    data_num = _cfg_get(dataset_cfg, 'data_num')
    data_slice = range(int(data_num)) if data_num else None
    dataset = Dataset()
    dataset_id = _cfg_get(dataset_cfg, 'dataset_id')
    if dataset_id is None:
        raise ValueError(f'lora context {context_cfg.get("training_run_id")} dataset config missing dataset_id')
    dataset_id = str(dataset_id)
    subset_name = _cfg_text(dataset_cfg, 'subset_name', 'default')
    split = _cfg_text(dataset_cfg, 'split', 'train')
    local_json_path = _local_json_dataset_path(dataset_cfg, dataset_id)
    if local_json_path is not None:
        dataset.add_dataset(
            DatasetMeta(
                subset_name=subset_name,
                split=split,
                data_slice=data_slice,
                data=_read_local_json_rows(local_json_path),
            ))
    else:
        dataset.add_dataset(
            DatasetMeta(
                dataset_id,
                subset_name=subset_name,
                split=split,
                data_slice=data_slice,
            ))
    dataset.set_template(
        _cfg_get(template_cfg, 'cls'),
        model_id=_cfg_get(context_cfg, 'base_model_id'),
        max_length=_cfg_get(dataset_cfg, 'max_length', _cfg_get(template_cfg, 'max_length', 4096)),
        truncation_strategy=_cfg_get(template_cfg, 'truncation_strategy', 'delete'),
        enable_thinking=_cfg_get(template_cfg, 'enable_thinking', False),
    )
    processor_cfg = _cfg_get(dataset_cfg, 'processor')
    if processor_cfg is not None:
        processor_cls = _cfg_get(processor_cfg, 'cls')
        processor_kwargs = {k: v for k, v in processor_cfg.items() if k != 'cls'}
        dataset.map(construct_class(processor_cls, Preprocessor, twinkle.preprocessor, **processor_kwargs))
    dataset.encode(add_generation_prompt=bool(_cfg_get(dataset_cfg, 'add_generation_prompt', True)))
    return dataset


def build_base_pipeline_config(cfg) -> BaseRLPipelineConfig:
    _validate_train_batch_config(cfg)
    contexts = build_lora_contexts(cfg)
    primary_context = contexts[0]
    rollout_cfg = cfg.pipeline.rollout
    target_groups_by_context = rollout_batch_groups_by_context(cfg, contexts)
    context_mini_batch_sizes = mini_batch_size_by_context(cfg, contexts)
    return BaseRLPipelineConfig(
        lora_contexts=contexts,
        tenant_id=primary_context.tenant_id,
        training_run_id=primary_context.training_run_id,
        base_model_id=primary_context.base_model_id,
        adapter_name=primary_context.adapter_name,
        reward_type=primary_context.reward_type,
        algorithm=primary_context.algorithm,
        tool_profile=primary_context.tool_profile,
        max_staleness=int(cfg.pipeline.max_staleness),
        default_rollout_batch_size=int(cfg.pipeline.default_rollout_batch_size),
        target_groups_by_context=target_groups_by_context,
        max_concurrency=int(rollout_cfg.get('max_concurrency', 16)),
        default_mini_batch_size=int(cfg.pipeline.default_mini_batch_size),
        mini_batch_size_by_context=context_mini_batch_sizes,
        max_train_steps=_optional_step_limit(cfg.pipeline.get('max_steps')),
        save_name_prefix=cfg.pipeline.save_name_prefix,
        adapter_checkpoint_dir=cfg.model.adapter_checkpoint_dir,
        is_sampler_checkpoint=bool(cfg.pipeline.is_sampler_checkpoint),
        save_optimizer=bool(cfg.pipeline.save_optimizer),
        max_grad_norm=float(cfg.pipeline.max_grad_norm),
        norm_type=int(cfg.pipeline.norm_type),
        metrics=async_rl_metrics_config(cfg),
    )


def validate_training_input_length(
    model_input: dict[str, Any],
    *,
    context: LoraContext,
    partition_id: str,
    max_length: int,
) -> None:
    input_ids = model_input.get('input_ids')
    if input_ids is None:
        return
    length = len(input_ids)
    if length <= max_length:
        return
    raise ValueError('training sample exceeds model.template.max_length: '
                     f'length={length}, max_length={max_length}, '
                     f'context={context.key}, partition_id={partition_id}. '
                     'Reduce sampler.sampling_params.max_tokens and dataset.max_length, '
                     'then clear old overlength TransferQueue partitions before rerun.')


class GSM8KBrevityReward:
    """Reward valid, shorter answers."""

    def __call__(self, trajectories: list[dict[str, Any]], **kwargs) -> list[float]:
        rewards = []
        for traj in trajectories:
            messages = traj.get('messages', [])
            completion = ''
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    completion = msg.get('content', '')
                    break
            has_answer = bool(
                re.search(r'\\boxed\{[^}]+\}', completion) or re.search(r'####\s*[\-\d,\.]+', completion)
                or re.search(r'(?im)^\s*Answer\s*:\s*\S+', completion))
            if not has_answer:
                rewards.append(0.0)
                continue
            length = len(completion)
            rewards.append(1.0 if length <= 300 else max(0.0, 1.0 - (length - 300) / 3000))
        return rewards


class GSM8KReward:

    def __init__(self):
        from twinkle.reward import GSM8KAccuracyReward

        self.accuracy = GSM8KAccuracyReward()
        self.brevity = GSM8KBrevityReward()

    def __call__(self, trajectories: list[dict[str, Any]], **kwargs) -> list[float]:
        accuracy = self.accuracy(trajectories)
        brevity = self.brevity(trajectories)
        total = [a + b for a, b in zip(accuracy, brevity)]
        for trajectory, total_reward, accuracy_reward, brevity_reward in zip(trajectories, total, accuracy, brevity):
            metadata = dict(trajectory.get('metadata') or {})
            metadata.update({
                'total_reward': float(total_reward),
                'accuracy_reward': float(accuracy_reward),
                'brevity_reward': float(brevity_reward),
            })
            completion_length = _as_float(trajectory.get('completion_length'))
            if completion_length is not None:
                metadata['completion_length'] = completion_length
            trajectory['metadata'] = metadata
        return total

    def metric_payload(
        self,
        trajectories: list[dict[str, Any]],
        *,
        rewards: list[float] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        return _short_math_reward_metrics(trajectories, total_rewards=rewards)


class DAPOMathReward:

    def __init__(self):
        from twinkle.reward import DAPOMathAccuracyReward

        self.accuracy = DAPOMathAccuracyReward()
        self.brevity = GSM8KBrevityReward()

    def __call__(self, trajectories: list[dict[str, Any]], **kwargs) -> list[float]:
        accuracy = self.accuracy(trajectories)
        brevity = self.brevity(trajectories)
        total = [a + b for a, b in zip(accuracy, brevity)]
        for trajectory, total_reward, accuracy_reward, brevity_reward in zip(trajectories, total, accuracy, brevity):
            metadata = dict(trajectory.get('metadata') or {})
            metadata.update({
                'total_reward': float(total_reward),
                'accuracy_reward': float(accuracy_reward),
                'brevity_reward': float(brevity_reward),
            })
            trajectory['metadata'] = metadata
        return total

    def metric_payload(
        self,
        trajectories: list[dict[str, Any]],
        *,
        rewards: list[float] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        return _short_math_reward_metrics(trajectories, total_rewards=rewards)


class ServerSingleTurnRollout:
    """One prompt-group rollout adapter for local/server vLLMSampler."""

    def __init__(self, sampler: Any, *, sampling_params: Any, num_generations: int):
        self.sampler = sampler
        self.sampling_params = sampling_params
        self.num_generations = num_generations

    def __call__(self, trajectories: list[Trajectory], **kwargs) -> list[RolloutOutput]:
        adapter_path = kwargs.get('adapter_path')
        adapter_name = kwargs.get('adapter_name', '')
        expanded = []
        for prompt_idx, trajectory in enumerate(trajectories):
            group_id = trajectory.get('group_id') or trajectory.get('sample_id') or f'prompt_{prompt_idx}'
            for generation_idx in range(self.num_generations):
                item = dict(trajectory)
                item['group_id'] = group_id
                item['generation_idx'] = generation_idx
                expanded.append(item)

        responses = self.sampler.sample(
            expanded,
            self.sampling_params,
            adapter_name=adapter_name,
            adapter_path=adapter_path,
        )
        rows: list[RolloutOutput] = []
        for source, response in zip(expanded, responses):
            for sequence in response.sequences:
                row = dict(source)
                row.update(sequence.new_input_feature or {})
                row.setdefault('group_id', source['group_id'])
                row.setdefault('generation_idx', source['generation_idx'])
                row['logprobs'] = self._extract_logps(sequence.logprobs)
                row['stop_reason'] = sequence.stop_reason
                row['completion_length'] = len(sequence.tokens)
                row['rollout_policy_version'] = kwargs.get('policy_version')
                rows.append(row)
        return rows

    @staticmethod
    def _extract_logps(logprobs) -> list[float]:
        values = []
        for item in logprobs or []:
            if not item:
                values.append(0.0)
            else:
                values.append(float(item[0][1]))
        return values


class AsyncMultiLoraGRPOPipeline(BaseRLPipeline):
    """Config-driven server-side multi-LoRA GRPO pipeline implementation."""

    def __init__(self, cfg, *, model_mesh: Any, sampler_mesh: Any):
        self.cfg = cfg
        self.model_mesh = model_mesh
        self.sampler_mesh = sampler_mesh
        self._optimizer_steps_by_context: dict[str, int] = {}
        self._adapter_paths_by_context: dict[str, list[Path]] = {}
        self._adapter_prune_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix='async-rl-adapter-prune',
        )
        super().__init__(config=build_base_pipeline_config(cfg))

    def build_model(self):
        from omegaconf import OmegaConf

        from twinkle.model import MultiLoraTransformersModel
        from twinkle.processor import InputProcessor

        primary_context = primary_lora_context(self.cfg)
        adapter_model_cfgs = [(context_cfg, lora_model_config(self.cfg, context_cfg))
                              for context_cfg in lora_context_configs(self.cfg)]
        max_adapter_rank = max(int(adapter_model_cfg.lora.r) for _, adapter_model_cfg in adapter_model_cfgs)
        configured_max_rank = self.cfg.model.get('max_r')
        max_rank = int(configured_max_rank) if configured_max_rank is not None else max_adapter_rank
        if max_adapter_rank > max_rank:
            raise ValueError(f'max adapter LoRA rank {max_adapter_rank} exceeds model.max_r {max_rank}')
        max_template_length = max(
            int(adapter_model_cfg.template.max_length) for _, adapter_model_cfg in adapter_model_cfgs)
        model_kwargs = {
            k: v
            for k, v in OmegaConf.to_container(self.cfg.model, resolve=True).items() if k in {
                'strategy',
                'ddp_config',
                'fsdp_config',
                'grad_scaler_config',
                'memory_efficient_init',
                'max_loras',
            }
        }
        model = MultiLoraTransformersModel(
            model_id=primary_context.base_model_id,
            device_mesh=self.model_mesh,
            mixed_precision=self.cfg.model.mixed_precision,
            max_r=max_rank,
            max_length=max_template_length,
            target_modules=self.cfg.model.lora.target_modules,
            **({
                'remote_group': 'model'
            } if self.cfg.runtime.mode == 'ray' else {}),
            **model_kwargs,
        )
        for context_cfg, adapter_model_cfg in adapter_model_cfgs:
            adapter_name = context_cfg.adapter_name
            lora_config = _build_lora_config(adapter_model_cfg.lora)
            model.add_adapter_to_model(
                adapter_name,
                lora_config,
                gradient_accumulation_steps=int(adapter_model_cfg.gradient_accumulation_steps),
            )
            optimizer_kwargs = _config_kwargs(adapter_model_cfg.optimizer)
            model.set_optimizer(
                adapter_model_cfg.optimizer.cls,
                adapter_name=adapter_name,
                **optimizer_kwargs,
            )
            lr_scheduler_cfg = adapter_model_cfg.get('lr_scheduler')
            if lr_scheduler_cfg is not None:
                model.set_lr_scheduler(
                    lr_scheduler_cfg.cls,
                    adapter_name=adapter_name,
                    **_config_kwargs(lr_scheduler_cfg),
                )
            model.set_loss(
                adapter_model_cfg.loss.cls,
                adapter_name=adapter_name,
                **_config_kwargs(adapter_model_cfg.loss),
            )
            if adapter_model_cfg.loss.cls == 'GRPOLoss':
                model.add_metric(
                    'GRPOMetric',
                    adapter_name=adapter_name,
                    epsilon=float(adapter_model_cfg.loss.get('epsilon', 0.2)),
                    epsilon_high=adapter_model_cfg.loss.get('epsilon_high'),
                )
            processor_cfg = adapter_model_cfg.get('processor')
            processor_cls = processor_cfg.get('cls', InputProcessor) if processor_cfg is not None else InputProcessor
            processor_kwargs = _config_kwargs(processor_cfg) if processor_cfg is not None else {}
            model.set_processor(processor_cls, adapter_name=adapter_name, **processor_kwargs)
            model.set_template(
                adapter_model_cfg.template.cls,
                adapter_name=adapter_name,
                **_config_kwargs(adapter_model_cfg.template, exclude={'max_length', 'truncation_strategy'}),
            )
        return model

    def build_rollout(self):
        from omegaconf import OmegaConf

        from twinkle.data_format import SamplingParams
        from twinkle.sampler import vLLMSampler

        primary_context = primary_lora_context(self.cfg)
        engine_args = OmegaConf.to_container(self.cfg.sampler.engine_args, resolve=True)
        engine_args.setdefault('tensor_parallel_size', int(self.cfg.runtime.sampler_tp))
        sampler = vLLMSampler(
            model_id=primary_context.base_model_id,
            engine_args=engine_args,
            device_mesh=self.sampler_mesh,
            **({
                'remote_group': 'sampler'
            } if self.cfg.runtime.mode == 'ray' else {}),
        )
        sampler_template_kwargs = {k: v for k, v in self.cfg.sampler.template.items() if k != 'cls'}
        sampler.set_template(
            self.cfg.sampler.template.cls,
            model_id=primary_context.base_model_id,
            **sampler_template_kwargs,
        )
        sampling_params = SamplingParams.from_dict(
            OmegaConf.to_container(self.cfg.sampler.sampling_params, resolve=True))
        return ServerSingleTurnRollout(
            sampler,
            sampling_params=sampling_params,
            num_generations=int(self.cfg.pipeline.rollout.num_generations),
        )

    def build_data_plane(self):
        tq_cfg = self.cfg.transfer_queue
        return TransferQueueDataPlane(
            tq_config=TransferQueueRuntimeConfig(
                init=bool(tq_cfg.get('init', True)),
                total_storage_size=tq_cfg.get('total_storage_size'),
                num_data_storage_units=int(tq_cfg.get('num_data_storage_units', 4)),
                storage_backend=tq_cfg.get('storage_backend', 'SimpleStorage'),
            ))

    def build_prompt_loaders(self):
        from omegaconf import OmegaConf

        from twinkle.dataloader import DataLoader

        loaders = []
        for context_cfg, context in zip(lora_context_configs(self.cfg), self.contexts):
            context_model_cfg = lora_model_config(self.cfg, context_cfg)
            prompt_batch_size = rollout_batch_groups_for_context(self.cfg, context_cfg)
            max_pending_groups = max_pending_prompt_groups_for_context(self.cfg, context_cfg)
            safe_context_key = re.sub(r'[^A-Za-z0-9_.-]+', '_', context.key)
            dataset_factory = partial(
                build_prompt_dataset_from_config,
                OmegaConf.to_container(context_cfg, resolve=True),
                OmegaConf.to_container(context_model_cfg.template, resolve=True),
            )
            dataloader_kwargs = {}
            if self.cfg.runtime.mode == 'ray':
                dataloader_kwargs['remote_group'] = 'model'
                dataloader_kwargs['instance_id'] = f'{os.getpid()}-{safe_context_key}-'
            dataloader = DataLoader(
                dataset=dataset_factory,
                batch_size=prompt_batch_size,
                min_batch_size=prompt_batch_size,
                device_mesh=self.model_mesh,
                **dataloader_kwargs,
            )
            loaders.append(
                PromptLoader(
                    context=context,
                    dataloader=dataloader,
                    rollouter=self.rollouter,
                    max_pending_groups=max_pending_groups,
                    metrics_recorder=self.metrics_recorder,
                ))
        return loaders

    def build_dataset(self, context_cfg):
        from omegaconf import OmegaConf

        return build_prompt_dataset_from_config(
            OmegaConf.to_container(context_cfg, resolve=True),
            OmegaConf.to_container(lora_model_config(self.cfg, context_cfg).template, resolve=True),
        )

    def build_reward_registry(self):
        registry = {}
        for context_cfg in lora_context_configs(self.cfg):
            if context_cfg.reward_type == 'gsm8k':
                registry[context_cfg.reward_type] = GSM8KReward()
            elif context_cfg.reward_type == 'dapo_math':
                registry[context_cfg.reward_type] = DAPOMathReward()
            else:
                raise ValueError(f'unsupported reward_type for AsyncMultiLoraGRPOPipeline: {context_cfg.reward_type}')
        return registry

    def build_advantage_fn(self):
        return grpo_advantage_fn

    def build_rollout_policy(self):
        policy = str(self.cfg.pipeline.get('rollout_policy', 'work_conserving'))
        if policy == 'work_conserving':
            return WorkConservingRolloutPolicy()
        if policy == 'weighted_fair':
            return WeightedFairRolloutPolicy()
        raise ValueError(f'unsupported rollout_policy: {policy!r}')

    def build_train_policy(self):
        policy = str(self.cfg.pipeline.get('train_policy', 'prefer_current'))
        if policy == 'prefer_current':
            return PreferCurrentTrainPolicy()
        if policy == 'weighted_fair':
            return WeightedFairTrainPolicy()
        raise ValueError(f'unsupported train_policy: {policy!r}')

    def build_train_batch_fn(self):
        return self.train_batch

    def build_save_adapter_fn(self):
        return self.save_adapter

    def train_batch(self, context, batch) -> TrainerStepResult:
        partition_id = batch.partition_id
        train_batch = read_train_batch(self.data_plane, batch)
        context_cfg = lora_context_config_for_context(self.cfg, context)
        mini_batch_groups = mini_batch_size_for_context(self.cfg, context_cfg) or int(
            self.cfg.pipeline.default_mini_batch_size)
        num_generations = int(self.cfg.pipeline.rollout.num_generations)
        mini_batch_size = mini_batch_groups * num_generations
        context_model_cfg = lora_model_config_for_context(self.cfg, context)
        max_length = int(context_model_cfg.template.max_length)
        last_metrics: dict[str, Any] = {}
        for mb_start in range(0, train_batch.sample_count, mini_batch_size):
            mb_end = mb_start + mini_batch_size
            inputs = train_batch.inputs[mb_start:mb_end]
            for model_input in inputs:
                validate_training_input_length(
                    model_input,
                    context=context,
                    partition_id=partition_id,
                    max_length=max_length,
                )
            logprobs = train_batch.logprobs[mb_start:mb_end]
            advantages = train_batch.advantages[mb_start:mb_end]
            rewards = train_batch.rewards[mb_start:mb_end]
            sample_tags = list(getattr(batch, 'tags', []) or [])[mb_start:mb_end]
            batch_diagnostics = _async_train_batch_data_diagnostics(
                inputs=inputs,
                rewards=rewards,
                advantages=advantages,
                logprobs=logprobs,
            )
            self.model.forward_backward(
                inputs=inputs,
                old_logps=logprobs,
                advantages=advantages,
                adapter_name=context.adapter_name,
            )
            self.model.clip_grad_and_step(
                adapter_name=context.adapter_name,
                max_grad_norm=float(self.cfg.pipeline.max_grad_norm),
                norm_type=int(self.cfg.pipeline.norm_type),
            )
            last_metrics = _metric_payload(
                self.model.calculate_metric(
                    is_training=True,
                    adapter_name=context.adapter_name,
                ))
            last_metrics.update(_short_math_reward_metrics(sample_tags, total_rewards=rewards))
            last_metrics.update(batch_diagnostics)
            optimizer_step = self._optimizer_steps_by_context.get(context.key, 0) + 1
            self._optimizer_steps_by_context[context.key] = optimizer_step
            last_metrics['optimizer_step'] = optimizer_step
            last_metrics['step'] = optimizer_step
            last_metrics.update({
                'sample_count': len(inputs),
                'prompt_count': len(inputs) / num_generations,
                'num_generations': num_generations,
            })
            logger.info(
                'async_multi_lora_grpo train metrics: context=%s partition=%s mini_batch=%s/%s metrics=%s',
                context.key,
                partition_id,
                mb_start // mini_batch_size + 1,
                (train_batch.sample_count + mini_batch_size - 1) // mini_batch_size,
                last_metrics,
            )

        return TrainerStepResult(metrics=last_metrics)

    def save_adapter(self, context, partition_id: str) -> TrainerStepResult:
        runtime_state = self.lora_runtime_registry.get(context)
        save_name = (f'{self.cfg.pipeline.save_name_prefix}-{context.training_run_id}-'
                     f'{context.adapter_name}-v{runtime_state.policy_version + 1}')
        save_result = self.model.save(
            save_name,
            output_dir=self.cfg.model.adapter_checkpoint_dir,
            adapter_name=context.adapter_name,
            save_optimizer=bool(self.cfg.pipeline.save_optimizer),
            is_sampler=bool(self.cfg.pipeline.is_sampler_checkpoint),
        )
        adapter_path = save_result if isinstance(save_result, str) else getattr(save_result, 'twinkle_path', None)
        self._record_and_prune_adapter_paths(context, adapter_path)
        return TrainerStepResult(adapter_path=adapter_path)

    def _record_and_prune_adapter_paths(self, context, adapter_path: str | None) -> None:
        keep_versions = int(self.cfg.pipeline.get('keep_adapter_versions', 0) or 0)
        if keep_versions <= 0 or adapter_path is None:
            return
        path = Path(adapter_path)
        if not path.exists():
            return
        paths = self._adapter_paths_by_context.setdefault(context.key, [])
        paths.append(path)
        stale_paths = paths[:-keep_versions]
        self._adapter_paths_by_context[context.key] = paths[-keep_versions:]
        for stale_path in stale_paths:
            if stale_path in self._adapter_paths_by_context[context.key]:
                continue
            self._adapter_prune_executor.submit(self._prune_adapter_path, context.key, stale_path)

    @staticmethod
    def _prune_adapter_path(context_key: str, path: Path) -> None:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
            logger.info('Pruned stale adapter checkpoint: context=%s path=%s', context_key, path)
        except OSError as exc:
            logger.warning('Failed to prune stale adapter checkpoint: context=%s path=%s error=%s',
                           context_key, path, exc)

    def shutdown(self) -> None:
        super().shutdown()
        self._adapter_prune_executor.shutdown(wait=False, cancel_futures=False)


def grpo_advantage_fn(batch: GRPOAdvantageBatch, context) -> tuple[list[float], list[float]]:
    from twinkle.advantage import GRPOAdvantage

    rewards = list(batch.rewards)
    if not rewards:
        return [], []
    num_generations = max(1, max(batch.generation_indices) + 1)
    advantages = GRPOAdvantage()(rewards, num_generations=num_generations, scale='group').tolist()
    return advantages, rewards
