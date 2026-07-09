# Copyright (c) ModelScope Contributors. All rights reserved.
"""Stage-barrier multi-LoRA GRPO baseline.

This entrypoint is the synchronous baseline for the async multi-LoRA GRPO
experiment. It uses the same config shape as
``cookbook/rl/async_multi_lora_grpo.yaml`` but enforces:

  all LoRA rollout -> barrier -> all LoRA train -> barrier

No rollout is submitted while any LoRA is training.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

import twinkle
from twinkle import DeviceGroup, DeviceMesh, Platform, get_device_placement, get_logger
from twinkle.data_format import InputFeature, Trajectory
from twinkle_agentic.async_rl.metrics import AsyncRLMetricsConfig, build_metrics_recorder, prefixed_summary
from twinkle_agentic.async_rl.grpo_pipeline import (
    GSM8KReward,
    ServerSingleTurnRollout,
    _build_lora_config,
    _config_kwargs,
    _metric_payload,
    _short_math_reward_metrics,
    _async_train_batch_data_diagnostics,
    async_rl_metrics_config,
    build_lora_contexts,
    build_prompt_dataset_from_config,
    lora_context_config_for_context,
    lora_context_configs,
    lora_model_config,
    lora_model_config_for_context,
    mini_batch_size_for_context,
    primary_lora_context,
    rollout_batch_groups_for_context,
    validate_training_input_length,
)
from twinkle_agentic.async_rl.types import LoraContext
from twinkle_agentic.async_rl.tq_utils import TRANSFORMERS_INPUT_FIELDS

logger = get_logger()


@dataclass
class SyncRolloutBatch:
    context: LoraContext
    context_cfg: Any
    partition_id: str
    prompts: list[Trajectory]
    rows: list[dict[str, Any]]
    rewards: list[float]
    advantages: list[float]
    rollout_policy_version: int


def load_config(path: str):
    return OmegaConf.load(path)


def build_device_meshes(cfg):
    runtime = cfg.runtime
    model_gpus = int(runtime.model_gpus)
    sampler_gpus = int(runtime.sampler_gpus)
    sampler_tp = int(runtime.sampler_tp)
    total_gpus = model_gpus + sampler_gpus
    device_type = Platform.device_prefix()
    if runtime.mode == 'local':
        device_groups = [
            DeviceGroup(name='default', ranks=list(range(total_gpus)), device_type=device_type),
        ]
    else:
        device_groups = [
            DeviceGroup(name='model', ranks=list(range(model_gpus)), device_type=device_type),
            DeviceGroup(
                name='sampler',
                ranks=list(range(model_gpus, total_gpus)),
                device_type=device_type,
                gpus_per_worker=sampler_tp,
            ),
        ]
    model_mesh_cfg = cfg.model.mesh
    model_tp = int(model_mesh_cfg.get('tp_size', 1))
    model_ep = int(model_mesh_cfg.get('ep_size', 1))
    model_pp = int(model_mesh_cfg.get('pp_size', 1))
    model_parallel_size = model_tp * model_ep * model_pp
    if model_gpus % model_parallel_size != 0:
        raise ValueError(
            f'model_gpus={model_gpus} must be divisible by '
            f'tp_size*ep_size*pp_size={model_parallel_size}')
    model_dp = int(model_mesh_cfg.get('dp_size', model_gpus // model_parallel_size))
    model_mesh = DeviceMesh.from_sizes(
        world_size=model_gpus,
        dp_size=model_dp,
        tp_size=model_tp,
        ep_size=model_ep,
        pp_size=model_pp,
        sequence_parallel=bool(model_mesh_cfg.get('sequence_parallel', False)),
    )
    sampler_mesh = DeviceMesh.from_sizes(
        world_size=sampler_gpus,
        dp_size=max(1, sampler_gpus // sampler_tp),
        tp_size=sampler_tp,
    )
    return total_gpus, device_groups, model_mesh, sampler_mesh


def sync_barrier_metrics_config(cfg) -> AsyncRLMetricsConfig | None:
    metrics_config = async_rl_metrics_config(cfg)
    if metrics_config is None:
        return None
    experiment_cfg = cfg.get('experiment') or {}
    if not experiment_cfg.get('run_id'):
        metrics_config.run_id = f'sync_barrier_{int(time.time())}'
    else:
        metrics_config.run_id = _sync_run_id(str(metrics_config.run_id))
    metrics_config.mode = 'sync_barrier'
    metadata = dict(metrics_config.metadata)
    metadata['scheduler'] = 'sync_barrier'
    metadata['allows_rollout_train_overlap'] = False
    metrics_config.metadata = metadata
    return metrics_config


def build_model(cfg, model_mesh):
    from omegaconf import OmegaConf

    from twinkle.model import MultiLoraTransformersModel
    from twinkle.processor import InputProcessor

    primary_context = primary_lora_context(cfg)
    adapter_model_cfgs = [(context_cfg, lora_model_config(cfg, context_cfg))
                          for context_cfg in lora_context_configs(cfg)]
    max_adapter_rank = max(int(adapter_model_cfg.lora.r) for _, adapter_model_cfg in adapter_model_cfgs)
    configured_max_rank = cfg.model.get('max_r')
    max_rank = int(configured_max_rank) if configured_max_rank is not None else max_adapter_rank
    if max_adapter_rank > max_rank:
        raise ValueError(f'max adapter LoRA rank {max_adapter_rank} exceeds model.max_r {max_rank}')
    max_template_length = max(int(adapter_model_cfg.template.max_length) for _, adapter_model_cfg in adapter_model_cfgs)
    model_kwargs = {
        k: v
        for k, v in OmegaConf.to_container(cfg.model, resolve=True).items() if k in {
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
        device_mesh=model_mesh,
        mixed_precision=cfg.model.mixed_precision,
        max_r=max_rank,
        max_length=max_template_length,
        target_modules=cfg.model.lora.target_modules,
        **({
            'remote_group': 'model'
        } if cfg.runtime.mode == 'ray' else {}),
        **model_kwargs,
    )
    for context_cfg, adapter_model_cfg in adapter_model_cfgs:
        adapter_name = context_cfg.adapter_name
        model.add_adapter_to_model(
            adapter_name,
            _build_lora_config(adapter_model_cfg.lora),
            gradient_accumulation_steps=int(adapter_model_cfg.gradient_accumulation_steps),
        )
        model.set_optimizer(
            adapter_model_cfg.optimizer.cls,
            adapter_name=adapter_name,
            **_config_kwargs(adapter_model_cfg.optimizer),
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


def build_rollout(cfg, sampler_mesh) -> ServerSingleTurnRollout:
    from omegaconf import OmegaConf

    from twinkle.data_format import SamplingParams
    from twinkle.sampler import vLLMSampler

    primary_context = primary_lora_context(cfg)
    engine_args = OmegaConf.to_container(cfg.sampler.engine_args, resolve=True)
    engine_args.setdefault('tensor_parallel_size', int(cfg.runtime.sampler_tp))
    sampler = vLLMSampler(
        model_id=primary_context.base_model_id,
        engine_args=engine_args,
        device_mesh=sampler_mesh,
        **({
            'remote_group': 'sampler'
        } if cfg.runtime.mode == 'ray' else {}),
    )
    sampler_template_kwargs = {k: v for k, v in cfg.sampler.template.items() if k != 'cls'}
    sampler.set_template(
        cfg.sampler.template.cls,
        model_id=primary_context.base_model_id,
        **sampler_template_kwargs,
    )
    sampling_params = SamplingParams.from_dict(OmegaConf.to_container(cfg.sampler.sampling_params, resolve=True))
    return ServerSingleTurnRollout(
        sampler,
        sampling_params=sampling_params,
        num_generations=int(cfg.pipeline.rollout.num_generations),
    )


def build_reward_registry(cfg) -> dict[str, Any]:
    registry = {}
    for context_cfg in lora_context_configs(cfg):
        if context_cfg.reward_type == 'gsm8k':
            registry[context_cfg.reward_type] = GSM8KReward()
        else:
            raise ValueError(f'unsupported reward_type for sync barrier GRPO: {context_cfg.reward_type}')
    return registry


class SyncBarrierMultiLoraGRPORunner:

    def __init__(self, cfg, *, model_mesh, sampler_mesh):
        self.cfg = cfg
        self.contexts = build_lora_contexts(cfg)
        self.context_cfgs = {
            context.key: context_cfg
            for context_cfg, context in zip(lora_context_configs(cfg), self.contexts)
        }
        self.metrics_config = sync_barrier_metrics_config(cfg)
        self._normalize_experiment_config()
        self.metrics_recorder = build_metrics_recorder(self.metrics_config)
        self.model = build_model(cfg, model_mesh)
        self.rollout = build_rollout(cfg, sampler_mesh)
        self.reward_registry = build_reward_registry(cfg)
        self.dataloader_iters = self._build_dataloader_iters(model_mesh)
        self.adapter_paths: dict[str, str | None] = {}
        self.policy_versions = {context.key: 0 for context in self.contexts}
        self.optim_steps = {context.key: 0 for context in self.contexts}
        self.trained_partitions = 0
        self._write_config_snapshot()

    def run(self, *, max_steps: int) -> int:
        round_idx = 0
        while self.trained_partitions < max_steps:
            round_start = time.perf_counter()
            self.metrics_recorder.log_event(
                event='barrier_round_started',
                phase='pipeline',
                metrics={
                    'round_idx': round_idx,
                    'trained_partitions': self.trained_partitions,
                    'contexts': len(self.contexts),
                },
            )
            rollout_batches = self._rollout_round(round_idx)
            if not rollout_batches:
                break
            self.metrics_recorder.log_event(
                event='rollout_barrier_reached',
                phase='pipeline',
                metrics={
                    'round_idx': round_idx,
                    'contexts': len(rollout_batches),
                    'sample_count': sum(len(item.rows) for item in rollout_batches),
                },
            )
            self._train_round(round_idx, rollout_batches)
            self.metrics_recorder.log_event(
                event='barrier_round_done',
                phase='pipeline',
                metrics={
                    'round_idx': round_idx,
                    'trained_partitions': self.trained_partitions,
                    'round_latency_s': time.perf_counter() - round_start,
                    'overlap_ratio': 0.0,
                    'pipeline_had_work': True,
                },
            )
            round_idx += 1
        return self.trained_partitions

    def close(self) -> None:
        self.metrics_recorder.close()

    def _build_dataloader_iters(self, model_mesh) -> dict[str, Any]:
        from twinkle.dataloader import DataLoader

        dataloader_iters = {}
        for context in self.contexts:
            context_cfg = self.context_cfgs[context.key]
            context_model_cfg = lora_model_config(self.cfg, context_cfg)
            prompt_batch_size = rollout_batch_groups_for_context(self.cfg, context_cfg)
            safe_context_key = re.sub(r'[^A-Za-z0-9_.-]+', '_', context.key)
            dataset_factory = partial(
                build_prompt_dataset_from_config,
                OmegaConf.to_container(context_cfg, resolve=True),
                OmegaConf.to_container(context_model_cfg.template, resolve=True),
            )
            dataloader = DataLoader(
                dataset=dataset_factory,
                batch_size=prompt_batch_size,
                min_batch_size=prompt_batch_size,
                device_mesh=model_mesh,
                remote_group='model',
                instance_id=f'{os.getpid()}-sync-barrier-{safe_context_key}-',
            )
            dataloader_iters[context.key] = iter(dataloader)
        return dataloader_iters

    def _rollout_round(self, round_idx: int) -> list[SyncRolloutBatch]:
        batches = []
        for context in self.contexts:
            context_cfg = self.context_cfgs[context.key]
            try:
                prompts = list(next(self.dataloader_iters[context.key]))
            except StopIteration:
                logger.info('sync_barrier dataset exhausted: context=%s round=%s', context.key, round_idx)
                return []
            partition_id = context.partition_id(round_idx)
            rollout_policy_version = self.policy_versions[context.key]
            adapter_path = self._ensure_adapter_path(context, partition_id=partition_id)
            reset_prefix_cache = getattr(getattr(self.rollout, 'sampler', None), 'reset_prefix_cache', None)
            if reset_prefix_cache is not None:
                reset_prefix_cache()
            start = time.perf_counter()
            self.metrics_recorder.log_event(
                event='rollout_started',
                phase='rollout',
                context=context,
                partition_id=partition_id,
                policy_version=rollout_policy_version,
                metrics={
                    'round_idx': round_idx,
                    'prompt_groups': len(prompts),
                },
            )
            rows = list(
                self.rollout(
                    prompts,
                    adapter_name=context.adapter_name,
                    adapter_path=adapter_path,
                    policy_version=rollout_policy_version,
                ))
            rewards = self._compute_rewards(context, rows)
            advantages = self._compute_advantages(rewards)
            self.metrics_recorder.log_event(
                event='rollout_done',
                phase='rollout',
                context=context,
                partition_id=partition_id,
                policy_version=rollout_policy_version,
                metrics={
                    'round_idx': round_idx,
                    'prompt_groups': len(prompts),
                    'sample_count': len(rows),
                    'rollout_latency_s': time.perf_counter() - start,
                    'rollout_policy_version': rollout_policy_version,
                    'policy_version_gap': 0,
                    **prefixed_summary('reward', rewards),
                },
            )
            batches.append(
                SyncRolloutBatch(
                    context=context,
                    context_cfg=context_cfg,
                    partition_id=partition_id,
                    prompts=prompts,
                    rows=rows,
                    rewards=rewards,
                    advantages=advantages,
                    rollout_policy_version=rollout_policy_version,
                ))
        return batches

    def _train_round(self, round_idx: int, batches: list[SyncRolloutBatch]) -> None:
        for batch in batches:
            start = time.perf_counter()
            self.metrics_recorder.log_event(
                event='train_claimed',
                phase='train',
                context=batch.context,
                partition_id=batch.partition_id,
                policy_version=self.policy_versions[batch.context.key],
                metrics={
                    'round_idx': round_idx,
                    'prompt_groups': len(batch.prompts),
                    'sample_count': len(batch.rows),
                },
            )
            self._train_context_batch(round_idx, batch)
            self.policy_versions[batch.context.key] += 1
            adapter_path = self._save_adapter(
                batch.context,
                partition_id=batch.partition_id,
                reason='train_done',
            )
            self.adapter_paths[batch.context.key] = adapter_path
            self.trained_partitions += 1
            self.metrics_recorder.log_event(
                event='partition_train_done',
                phase='train',
                context=batch.context,
                partition_id=batch.partition_id,
                policy_version=self.policy_versions[batch.context.key],
                metrics={
                    'round_idx': round_idx,
                    'prompt_groups': len(batch.prompts),
                    'sample_count': len(batch.rows),
                    'train_latency_s': time.perf_counter() - start,
                    'train_steps': self.optim_steps[batch.context.key],
                },
            )

    def _train_context_batch(self, round_idx: int, batch: SyncRolloutBatch) -> None:
        context = batch.context
        context_cfg = lora_context_config_for_context(self.cfg, context)
        mini_batch_groups = mini_batch_size_for_context(self.cfg, context_cfg) or int(
            self.cfg.pipeline.default_mini_batch_size)
        num_generations = int(self.cfg.pipeline.rollout.num_generations)
        mini_batch_size = mini_batch_groups * num_generations
        context_model_cfg = lora_model_config_for_context(self.cfg, context)
        max_length = int(context_model_cfg.template.max_length)
        inputs = [_model_input(row) for row in batch.rows]
        logprobs = [_require_float_logprobs(row, context=context, partition_id=batch.partition_id) for row in batch.rows]
        for mb_start in range(0, len(inputs), mini_batch_size):
            mb_end = min(mb_start + mini_batch_size, len(inputs))
            mb_inputs = inputs[mb_start:mb_end]
            mb_logprobs = logprobs[mb_start:mb_end]
            mb_rewards = batch.rewards[mb_start:mb_end]
            mb_advantages = batch.advantages[mb_start:mb_end]
            for model_input in mb_inputs:
                validate_training_input_length(
                    model_input,
                    context=context,
                    partition_id=batch.partition_id,
                    max_length=max_length,
                )
            train_start = time.perf_counter()
            self.model.forward_backward(
                inputs=mb_inputs,
                old_logps=mb_logprobs,
                advantages=mb_advantages,
                adapter_name=context.adapter_name,
            )
            self.model.clip_grad_and_step(
                adapter_name=context.adapter_name,
                max_grad_norm=float(self.cfg.pipeline.max_grad_norm),
                norm_type=int(self.cfg.pipeline.norm_type),
            )
            self.optim_steps[context.key] += 1
            metrics = _metric_payload(
                self.model.calculate_metric(
                    is_training=True,
                    adapter_name=context.adapter_name,
                ))
            metrics.update(
                _async_train_batch_data_diagnostics(
                    inputs=mb_inputs,
                    rewards=mb_rewards,
                    advantages=mb_advantages,
                    logprobs=mb_logprobs,
                ))
            metrics.update(
                _short_math_reward_metrics(
                    batch.rows[mb_start:mb_end],
                    total_rewards=mb_rewards,
                ))
            metrics.update({
                'round_idx': round_idx,
                'mini_batch_index': mb_start // mini_batch_size,
                'train_step': self.optim_steps[context.key],
                'train_batch_latency_s': time.perf_counter() - train_start,
            })
            self.metrics_recorder.log_event(
                event='train_batch_done',
                phase='train',
                context=context,
                partition_id=batch.partition_id,
                policy_version=self.policy_versions[context.key],
                metrics=metrics,
            )
            logger.info(
                'sync_barrier train metrics: context=%s partition=%s mini_batch=%s metrics=%s',
                context.key,
                batch.partition_id,
                mb_start // mini_batch_size + 1,
                metrics,
            )

    def _compute_rewards(self, context: LoraContext, rows: list[dict[str, Any]]) -> list[float]:
        reward_fn = self.reward_registry.get(context.reward_type)
        if reward_fn is None:
            raise ValueError(f'no reward function registered for context={context.key}, reward_type={context.reward_type}')
        rewards = list(reward_fn(rows))
        if len(rewards) != len(rows):
            raise ValueError(f'reward length mismatch for {context.key}: {len(rewards)} != {len(rows)}')
        return [float(reward) for reward in rewards]

    def _compute_advantages(self, rewards: list[float]) -> list[float]:
        from twinkle.advantage import GRPOAdvantage

        if not rewards:
            return []
        num_generations = int(self.cfg.pipeline.rollout.num_generations)
        if len(rewards) % num_generations != 0:
            raise ValueError(f'reward count must be divisible by num_generations={num_generations}, got {len(rewards)}')
        return GRPOAdvantage()(rewards, num_generations=num_generations, scale='group').tolist()

    def _ensure_adapter_path(self, context: LoraContext, *, partition_id: str) -> str | None:
        if context.key in self.adapter_paths:
            return self.adapter_paths[context.key]
        adapter_path = self._save_adapter(context, partition_id=partition_id, reason='initial_rollout')
        self.adapter_paths[context.key] = adapter_path
        return adapter_path

    def _save_adapter(self, context: LoraContext, *, partition_id: str, reason: str) -> str | None:
        version = self.policy_versions[context.key]
        save_name = (f'sync-barrier-{_safe_name(context.training_run_id)}-'
                     f'{_safe_name(context.adapter_name)}-v{version}-{reason}')
        start = time.perf_counter()
        save_result = self.model.save(
            save_name,
            output_dir=self.cfg.model.adapter_checkpoint_dir,
            adapter_name=context.adapter_name,
            save_optimizer=bool(self.cfg.pipeline.save_optimizer),
            is_sampler=bool(self.cfg.pipeline.is_sampler_checkpoint),
        )
        adapter_path = _adapter_path(save_result)
        self.metrics_recorder.log_event(
            event='weight_sync_done',
            phase='sync',
            context=context,
            partition_id=partition_id,
            policy_version=version,
            metrics={
                'reason': reason,
                'weight_sync_latency_s': time.perf_counter() - start,
                'adapter_path': adapter_path,
            },
        )
        return adapter_path

    def _write_config_snapshot(self) -> None:
        if self.metrics_config is None:
            return
        run_dir = Path(self.metrics_config.output_dir) / _safe_name(self.metrics_config.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.cfg, run_dir / 'config.yaml')

    def _normalize_experiment_config(self) -> None:
        if self.metrics_config is None:
            return
        OmegaConf.update(self.cfg, 'experiment.run_id', self.metrics_config.run_id, merge=True)
        OmegaConf.update(self.cfg, 'experiment.mode', self.metrics_config.mode, merge=True)


def _model_input(row: dict[str, Any]) -> InputFeature:
    values = {field_name: row[field_name] for field_name in TRANSFORMERS_INPUT_FIELDS if field_name in row}
    for field_name in ('input_ids', 'labels'):
        if field_name not in values:
            raise ValueError(f'rollout row missing required model input field {field_name!r}')
    return InputFeature(**values)


def _require_float_logprobs(row: dict[str, Any], *, context: LoraContext, partition_id: str) -> list[float]:
    logprobs = row.get('logprobs')
    if not isinstance(logprobs, list):
        raise TypeError(f'rollout row logprobs must be list[float]: context={context.key}, partition={partition_id}')
    values = []
    for index, value in enumerate(logprobs):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f'rollout row logprobs[{index}] must be float: '
                            f'context={context.key}, partition={partition_id}, got={type(value)!r}')
        values.append(float(value))
    labels = row.get('labels')
    if labels is not None:
        trainable_tokens = sum(1 for label in labels if label != -100)
        if len(values) != trainable_tokens:
            raise ValueError('rollout row logprobs length must match trainable labels: '
                             f'context={context.key}, partition={partition_id}, {len(values)} != {trainable_tokens}')
    return values


def _adapter_path(save_result: Any) -> str | None:
    if save_result is None:
        return None
    if isinstance(save_result, str):
        return save_result
    for attr in ('twinkle_path', 'path', 'checkpoint_dir'):
        value = getattr(save_result, attr, None)
        if value:
            return str(value)
    return str(save_result)


def _safe_name(value: str) -> str:
    return ''.join(char if char.isalnum() or char in '._-' else '_' for char in str(value))


def _sync_run_id(run_id: str) -> str:
    for prefix in ('async_relaxed', 'async_strict', 'async'):
        if run_id == prefix:
            return 'sync_barrier'
        if run_id.startswith(f'{prefix}_'):
            return f'sync_barrier_{run_id[len(prefix) + 1:]}'
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default=Path(__file__).with_name('async_multi_lora_grpo.yaml').as_posix(),
        help='Path to async multi-LoRA GRPO YAML config reused by the sync barrier baseline.',
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    total_gpus, device_groups, model_mesh, sampler_mesh = build_device_meshes(cfg)
    twinkle.initialize(
        mode=cfg.runtime.mode,
        nproc_per_node=total_gpus,
        groups=device_groups,
        lazy_collect=bool(cfg.runtime.get('lazy_collect', False)),
    )

    runner = SyncBarrierMultiLoraGRPORunner(cfg, model_mesh=model_mesh, sampler_mesh=sampler_mesh)
    logger.info('Starting sync-barrier multi-LoRA GRPO baseline')
    logger.info(get_device_placement())
    try:
        trained = runner.run(max_steps=int(cfg.pipeline.max_steps))
    finally:
        runner.close()

    for context in runner.contexts:
        final_name = f'sync-barrier-final-{context.training_run_id}-{context.adapter_name}'
        runner.model.save(
            final_name,
            output_dir=cfg.model.adapter_checkpoint_dir,
            adapter_name=context.adapter_name,
            save_optimizer=bool(cfg.pipeline.save_optimizer),
        )
    logger.info('Sync-barrier training completed. trained_partitions=%s', trained)


if __name__ == '__main__':
    main()
