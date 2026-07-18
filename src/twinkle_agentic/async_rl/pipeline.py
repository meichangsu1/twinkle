# Copyright (c) ModelScope Contributors. All rights reserved.
"""Driver for the independent async-RL Ray workers."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from functools import partial
from typing import Any

from .context_manager import LoraContextManager
from .data_plane import TQDataPlane
from .metrics import JSONLMetricsRecorder
from .scheduler import ContextSchedulePolicy, SchedulerConfig
from .types import LoraContext, PartitionAdmission
from .workers import AdvantageWorker, RolloutWorker, TrainerWorker


@dataclass(frozen=True)
class AsyncMultiLoraGRPOConfig:
    metrics_drain_interval_s: float = 1.0


def _sampler_data_parallel_size(sampler_gpus: int, sampler_tp: int) -> int:
    if sampler_gpus <= 0:
        raise ValueError(f'sampler_gpus must be positive, got {sampler_gpus}')
    if sampler_tp <= 0:
        raise ValueError(f'sampler_tp must be positive, got {sampler_tp}')
    if sampler_gpus % sampler_tp != 0:
        raise ValueError(f'sampler_gpus ({sampler_gpus}) must be divisible by sampler_tp ({sampler_tp})')
    return sampler_gpus // sampler_tp


def _sequence_parallel_size(model_gpus: int, configured_size: int) -> int:
    if configured_size <= 0:
        raise ValueError(f'model.sequence_parallel_size must be positive, got {configured_size}')
    if model_gpus % configured_size:
        raise ValueError(f'runtime.model_gpus ({model_gpus}) must be divisible by '
                         f'model.sequence_parallel_size ({configured_size})')
    return configured_size


def _model_attention_implementation(model_config: Any, *, padding_free: bool,
                                    sequence_parallel_size: int) -> str | None:
    implementation = model_config.get('attn_implementation')
    if implementation is not None:
        implementation = str(implementation)
    if padding_free and sequence_parallel_size > 1 and implementation != 'flash_attention_2':
        raise ValueError(
            'model.attn_implementation must be flash_attention_2 when '
            'model.padding_free=true and model.sequence_parallel_size>1')
    return implementation


def configure_lora_lr_scheduler(model: Any, adapter_name: str, lora_config: dict[str, Any]) -> None:
    """Configure one adapter's learning-rate scheduler from the shared LoRA config."""
    scheduler_config = lora_config.get('lr_scheduler')
    if scheduler_config is None:
        return
    scheduler_config = dict(scheduler_config)
    scheduler_cls = scheduler_config.pop('cls')
    model.set_lr_scheduler(
        scheduler_cls,
        adapter_name=adapter_name,
        **scheduler_config,
    )


def _validate_context_batch_config(
    context_key: str,
    *,
    rollout_groups: int,
    num_generations: int,
    advantage_groups: int,
    train_groups: int,
    micro_batch_size: int,
    sampler_dp: int,
    model_dp: int,
) -> None:
    values = {
        'rollout.batch_size': rollout_groups,
        'rollout.num_generations': num_generations,
        'advantage.groups_per_batch': advantage_groups,
        'train.groups_per_batch': train_groups,
        'train.micro_batch_size': micro_batch_size,
    }
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f'{name} for {context_key} must be positive, got {value}')
    if rollout_groups % sampler_dp:
        raise ValueError(f'rollout.batch_size for {context_key} must be divisible by sampler DP size '
                         f'({sampler_dp}), got {rollout_groups}')
    if rollout_groups % advantage_groups:
        raise ValueError(f'rollout.batch_size for {context_key} must be divisible by '
                         f'advantage.groups_per_batch: {rollout_groups} % {advantage_groups} != 0')
    if rollout_groups % train_groups:
        raise ValueError(f'rollout.batch_size for {context_key} must be divisible by train.groups_per_batch: '
                         f'{rollout_groups} % {train_groups} != 0')
    train_samples = train_groups * num_generations
    if train_samples % model_dp:
        raise ValueError(f'train batch for {context_key} has {train_samples} samples and must be divisible by '
                         f'model DP size {model_dp}')
    samples_per_rank = train_samples // model_dp
    if micro_batch_size > samples_per_rank:
        raise ValueError(f'train.micro_batch_size for {context_key} must not exceed the per-rank train batch '
                         f'({samples_per_rank}), got {micro_batch_size}')


class AsyncMultiLoraGRPOPipeline:
    """Owns the production async-RL runtime and drives its worker services.

    ``from_config`` is the real training construction path.  The explicit
    constructor remains available to fake-TQ tests, where injecting a fake
    sampler/model is the point of the test.
    """

    def __init__(self,
                 *,
                 context_manager: LoraContextManager,
                 rollout_worker: RolloutWorker,
                 advantage_worker: AdvantageWorker,
                 trainer_worker: TrainerWorker,
                 metrics: JSONLMetricsRecorder | None = None,
                 config: AsyncMultiLoraGRPOConfig = AsyncMultiLoraGRPOConfig(),
                 sampler: Any | None = None,
                 resources: dict[str, Any] | None = None):
        self.context_manager = context_manager
        self.rollout_worker = rollout_worker
        self.advantage_worker = advantage_worker
        self.trainer_worker = trainer_worker
        self.sampler = sampler
        self.metrics = metrics
        self.config = config
        self._resources = dict(resources or {})

    @classmethod
    def from_config(cls, raw_config: dict[str, Any]) -> AsyncMultiLoraGRPOPipeline:
        """Build the complete Ray/TQ runtime from the async-RL YAML mapping."""
        import ray
        import transfer_queue as tq
        from omegaconf import OmegaConf
        from peft import LoraConfig

        import twinkle
        from twinkle import DeviceGroup, DeviceMesh
        from twinkle.data_format import SamplingParams
        from twinkle.model import MultiLoraTransformersModel
        from twinkle.processor import InputProcessor
        from .group_sampler import ContextGRPOGroupNSampler

        runtime = raw_config['runtime']
        model_config = raw_config['model']
        lora_config_data = raw_config['lora']
        template_config = raw_config.get('template', {})
        template_cls = template_config.get('cls', 'Qwen3_5Template')
        enable_thinking = bool(template_config.get('enable_thinking', False))
        sampler_gpus = int(runtime['sampler_gpus'])
        sampler_tp = int(runtime['sampler_tp'])
        sampler_dp = _sampler_data_parallel_size(sampler_gpus, sampler_tp)
        model_dp = int(runtime['model_gpus'])
        sequence_parallel_size = _sequence_parallel_size(
            model_dp,
            int(model_config['sequence_parallel_size']),
        )
        padding_free = bool(model_config['padding_free'])
        attn_implementation = _model_attention_implementation(
            model_config,
            padding_free=padding_free,
            sequence_parallel_size=sequence_parallel_size,
        )
        model_max_length = int(model_config['max_length'])
        sampler_config = raw_config['sampler']
        total_gpus = model_dp + sampler_gpus
        device_groups = [
            DeviceGroup('model', list(range(int(runtime['model_gpus']))), device_type='GPU'),
            DeviceGroup(
                'sampler',
                list(range(int(runtime['model_gpus']), total_gpus)),
                device_type='GPU',
                gpus_per_worker=sampler_tp,
            ),
        ]
        twinkle.initialize(mode='ray', nproc_per_node=total_gpus, groups=device_groups, lazy_collect=False)
        tq.init(
            OmegaConf.create(
                {
                    'controller': {
                        'sampler': ContextGRPOGroupNSampler,
                        'polling_mode': bool(raw_config['tq'].get('polling_mode', True)),
                    },
                    'backend': {
                        'SimpleStorage': {
                            'num_data_storage_units': raw_config['tq']['storage_units']
                        }
                    },
                },
                flags={'allow_objects': True}))

        model_mesh = DeviceMesh.from_sizes(
            world_size=model_dp,
            dp_size=model_dp,
            ulysses_size=sequence_parallel_size,
        )
        model_data_parallel_size = model_mesh.data_world_size
        sampler_mesh = DeviceMesh.from_sizes(world_size=sampler_gpus, dp_size=sampler_dp, tp_size=sampler_tp)
        model_kwargs = {}
        if attn_implementation is not None:
            model_kwargs['attn_implementation'] = attn_implementation
        model = MultiLoraTransformersModel(
            model_id=runtime['model_id'],
            device_mesh=model_mesh,
            remote_group='model',
            max_length=model_max_length,
            **model_kwargs,
        )
        lora_config = LoraConfig(
            target_modules='all-linear',
            r=lora_config_data['r'],
            lora_alpha=lora_config_data['alpha'],
            lora_dropout=lora_config_data['dropout'],
        )
        contexts: list[LoraContext] = []
        prompt_sources: dict[str, Any] = {}
        rollout_config: dict[str, dict[str, Any]] = {}
        advantage_groups_per_batch: dict[str, int] = {}
        train_groups_per_batch: dict[str, int] = {}
        micro_batch_sizes: dict[str, int] = {}
        rewards: dict[str, Any] = {}
        initial_paths: dict[str, str] = {}
        for item in raw_config['lora_contexts']:
            context = LoraContext(
                item['tenant_id'],
                item['training_run_id'],
                runtime['model_id'],
                item['adapter_name'],
                item.get('reward_type', 'gsm8k_accuracy'),
            )
            contexts.append(context)
            model.add_adapter_to_model(context.adapter_name, lora_config, gradient_accumulation_steps=1)
            model.set_optimizer('AdamW', lr=lora_config_data['learning_rate'], adapter_name=context.adapter_name)
            configure_lora_lr_scheduler(model, context.adapter_name, lora_config_data)
            model.set_loss('GRPOLoss', epsilon=.2, adapter_name=context.adapter_name)
            model.set_processor(
                InputProcessor,
                adapter_name=context.adapter_name,
                padding_free=padding_free,
            )
            model.set_template(
                template_cls,
                model_id=runtime['model_id'],
                adapter_name=context.adapter_name,
                enable_thinking=enable_thinking,
            )

            rollout = item['rollout']
            advantage = item['advantage']
            train = item['train']
            rollout_batch_size = int(rollout['batch_size'])
            num_generations = int(rollout['num_generations'])
            advantage_groups = int(advantage['groups_per_batch'])
            train_groups = int(train['groups_per_batch'])
            micro_batch_size = int(train['micro_batch_size'])
            _validate_context_batch_config(
                context.key,
                rollout_groups=rollout_batch_size,
                num_generations=num_generations,
                advantage_groups=advantage_groups,
                train_groups=train_groups,
                micro_batch_size=micro_batch_size,
                sampler_dp=sampler_dp,
                model_dp=model_data_parallel_size,
            )
            prompt_sources[context.key] = partial(
                _prompt_batches,
                item['dataset'],
                model_id=runtime['model_id'],
                batch_size=rollout_batch_size,
                template_cls=template_cls,
                enable_thinking=enable_thinking,
            )
            rollout_config[context.key] = {
                'context':
                context,
                'batch_size':
                rollout_batch_size,
                'num_generations':
                num_generations,
                'sampling_params':
                SamplingParams(
                    max_tokens=rollout['max_tokens'],
                    temperature=rollout['temperature'],
                    top_p=rollout['top_p'],
                    repetition_penalty=float(rollout.get('repetition_penalty', 1.0)),
                    logprobs=1,
                    num_samples=1,
                ),
            }
            advantage_groups_per_batch[context.key] = advantage_groups
            train_groups_per_batch[context.key] = train_groups
            micro_batch_sizes[context.key] = micro_batch_size
            rewards[context.key] = _reward_for_context(context)
            initial_paths[context.key] = model.save(
                f'async-{context.adapter_name}-initial',
                output_dir=runtime['output_dir'],
                adapter_name=context.adapter_name,
            )

        manager = create_cpu_actor(
            LoraContextManager,
            max_staleness=runtime['max_staleness'],
            max_steps=runtime['max_steps'],
        )
        for context in contexts:
            ray.get(manager.register_context.remote(context, adapter_path=initial_paths[context.key]))

        from .vllm_sampler_tq import VLLMSamplerTQ
        sampler = VLLMSamplerTQ(
            model_id=runtime['model_id'],
            remote_group='sampler',
            device_mesh=sampler_mesh,
            engine_args={
                'tensor_parallel_size': sampler_tp,
                'enable_lora': True,
                'max_loras': int(runtime['sampler_max_loras']),
                'max_lora_rank': lora_config_data['r'],
                'max_model_len': int(sampler_config['max_model_len']),
                'gpu_memory_utilization': float(sampler_config['gpu_memory_utilization']),
                'max_num_seqs': int(sampler_config['max_num_seqs']),
                'max_num_batched_tokens': int(sampler_config['max_num_batched_tokens']),
                'enforce_eager': bool(sampler_config['enforce_eager']),
            },
            reward_registry=rewards,
            context_manager=manager,
            rollout_max_retries=int(runtime.get('rollout_max_retries', 2)),
            rollout_retry_delay_s=float(runtime.get('rollout_retry_delay_s', 0.5)),
        )
        sampler.set_template(
            template_cls,
            model_id=runtime['model_id'],
            enable_thinking=enable_thinking,
        )

        rollout_worker = create_cpu_actor(
            RolloutWorker,
            context_manager=manager,
            data_plane=TQDataPlane(),
            sampler=sampler,
            prompt_batches=prompt_sources,
            rollout_config=rollout_config,
            scheduler=_scheduler(raw_config['scheduler']['rollout']),
            allow_partial_rollout=runtime['allow_partial_rollout'],
        )
        advantage_worker = create_cpu_actor(
            AdvantageWorker,
            context_manager=manager,
            data_plane=TQDataPlane(),
            advantage_fn=_compute_advantages,
            groups_per_batch=advantage_groups_per_batch,
            scheduler=_scheduler(raw_config['scheduler']['advantage']),
        )
        trainer_worker = create_cpu_actor(
            TrainerWorker,
            context_manager=manager,
            data_plane=TQDataPlane(),
            train_fn=partial(_train_batch, model, micro_batch_sizes),
            save_adapter=partial(_save_adapter, model, runtime['output_dir']),
            groups_per_batch=train_groups_per_batch,
            scheduler=_scheduler(raw_config['scheduler']['train']),
            keep_adapter_versions=runtime['keep_adapter_versions'],
            initial_adapter_paths=initial_paths,
        )
        metrics = JSONLMetricsRecorder(runtime['metrics_path'], run_id='async_multi_lora_grpo', mode='ray')
        return cls(
            context_manager=manager,
            rollout_worker=rollout_worker,
            advantage_worker=advantage_worker,
            trainer_worker=trainer_worker,
            sampler=sampler,
            metrics=metrics,
            resources={
                'model': model,
                'contexts': contexts
            },
        )

    async def run_async(self) -> dict[str, Any]:
        started = time.perf_counter()
        workers = [self.rollout_worker, self.advantage_worker, self.trainer_worker]
        await asyncio.gather(*(worker.start.remote() for worker in workers))
        try:
            while True:
                await self._drain_metrics()
                states = await asyncio.gather(*(worker.get_service_state.remote() for worker in workers))
                failures = [state['failure'] for state in states if state['failure']]
                if failures:
                    raise RuntimeError(f'async RL worker failed: {failures[0]}')
                if self.sampler is not None:
                    await asyncio.to_thread(self.sampler.check_health)
                running = any(bool(state['running']) for state in states)
                if not running:
                    if await self.context_manager.is_run_finished.remote():
                        break
                    raise RuntimeError('async RL workers stopped before all contexts were drained')
                await asyncio.sleep(self.config.metrics_drain_interval_s)
        except Exception as exc:
            if self.metrics is not None:
                self.metrics.flush()
                failure_metrics = {
                    'error': f'{type(exc).__name__}: {exc}',
                    'wall_time_s': time.perf_counter() - started,
                    **self.metrics.stats(),
                }
                self.metrics.record(
                    event='run_failed',
                    metrics=failure_metrics,
                )
                self.metrics.flush()
            raise
        finally:
            await asyncio.gather(*(worker.stop.remote() for worker in workers), return_exceptions=True)
            await self._drain_metrics()
        result = {
            'trained_partitions': await self.context_manager.get_completed_partitions.remote(),
            'wall_time_s': time.perf_counter() - started,
        }
        if self.metrics is not None:
            self.metrics.flush()
            result.update(self.metrics.stats())
            self.metrics.record(event='run_completed', metrics=result)
            self.metrics.flush()
        return result

    def run(self) -> dict[str, Any]:
        try:
            return asyncio.run(self.run_async())
        finally:
            if self.metrics is not None:
                self.metrics.close()

    async def _drain_metrics(self) -> None:
        if self.metrics is None:
            return
        workers = [self.rollout_worker, self.advantage_worker, self.trainer_worker]
        for worker in workers:
            events = await worker.drain_metrics.remote()
            for item in events:
                self.metrics.record(
                    event=item.event,
                    context=item.context,
                    partition_id=item.partition_id,
                    metrics=item.metrics,
                    policy_version=item.policy_version,
                )
        if self.sampler is not None:
            events = await asyncio.to_thread(self.sampler.drain_metrics)
            for item in events:
                self.metrics.record(
                    event=item.event,
                    context=item.context,
                    partition_id=item.partition_id,
                    metrics=item.metrics,
                    policy_version=item.policy_version,
                )


def create_cpu_actor(cls: type, *args: Any, **kwargs: Any) -> Any:
    """Deploy a CPU service as one raw Ray actor; local tests use the class directly."""

    import ray
    actor_class = ray.remote(
        num_cpus=1,
        runtime_env={'env_vars': {
            'TWINKLE_MODE': 'ray'
        }},
    )(
        cls)
    return actor_class.remote(*args, **kwargs)


def _scheduler(config: dict[str, Any]) -> SchedulerConfig:
    return SchedulerConfig(ContextSchedulePolicy(config['policy']), config.get('max_consecutive_units'))


def _prompt_batches(
    dataset_config: dict[str, Any],
    *,
    model_id: str,
    batch_size: int,
    template_cls: str,
    enable_thinking: bool,
):
    """Create a lazy, full-batch-only prompt source for one context."""
    from twinkle.dataloader import DataLoader
    from twinkle.dataset import Dataset, DatasetMeta
    from twinkle.preprocessor import llm as llm_processors

    def batches():
        data_num = dataset_config.get('data_num')
        dataset = Dataset(
            DatasetMeta(
                dataset_config['dataset_id'],
                subset_name=dataset_config.get('subset_name'),
                split=dataset_config.get('split', 'train'),
                data_slice=range(int(data_num)) if data_num is not None else None,
            ))
        dataset.set_template(
            template_cls,
            model_id=model_id,
            max_length=dataset_config['max_length'],
            enable_thinking=enable_thinking,
        )
        processor_name = dataset_config.get('processor', 'GSM8KProcessor')
        processor_cls = getattr(llm_processors, processor_name)
        if processor_name == 'GSM8KProcessor':
            processor = processor_cls(system=dataset_config['system_prompt'])
        else:
            processor = processor_cls()
        dataset.map(processor)
        dataset.encode(add_generation_prompt=True)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, min_batch_size=batch_size)
        remaining = data_num
        remaining = None if remaining is None else int(remaining)
        for batch in loader:
            if len(batch) != batch_size or (remaining is not None and remaining < batch_size):
                return
            yield batch
            if remaining is not None:
                remaining -= batch_size

    return batches()


def _reward_for_context(context: LoraContext) -> Any:
    from twinkle.reward import (BoxedMathAccuracyReward, DAPOMathAccuracyReward, GSM8KAccuracyBrevityReward,
                                GSM8KAccuracyReward)
    if context.reward_type in {'gsm8k', 'gsm8k_accuracy'}:
        return GSM8KAccuracyReward()
    if context.reward_type == 'gsm8k_accuracy_brevity':
        return GSM8KAccuracyBrevityReward()
    if context.reward_type == 'dapo_math_accuracy':
        return DAPOMathAccuracyReward()
    if context.reward_type == 'aime2024_accuracy':
        return BoxedMathAccuracyReward()
    raise ValueError(f'unsupported async-RL reward_type {context.reward_type!r} for {context.key}')


def _compute_advantages(data: Any, admission: PartitionAdmission) -> tuple[list[float], list[float]]:
    from twinkle.advantage import GRPOAdvantage
    rewards = [float(value) for value in data['rewards']]
    advantages = GRPOAdvantage()(rewards, num_generations=admission.num_generations, scale='group').tolist()
    return advantages, rewards


def _train_batch(model: Any, micro_batch_sizes: dict[str, int], data: Any,
                 admission: PartitionAdmission) -> dict[str, Any]:
    from .tq_utils import REQUIRED_MODEL_INPUT_FIELDS

    size = int(data.batch_size[0])
    inputs = [{name: data[name][index] for name in REQUIRED_MODEL_INPUT_FIELDS} for index in range(size)]
    model.forward_backward(
        inputs=inputs,
        old_logps=list(data['logprobs']),
        advantages=list(data['advantages']),
        micro_batch_size=micro_batch_sizes[admission.context.key],
        adapter_name=admission.context.adapter_name,
    )
    model.clip_grad_and_step(adapter_name=admission.context.adapter_name)
    metrics = dict(model.calculate_metric(is_training=True, adapter_name=admission.context.adapter_name))
    metrics['reward'] = sum(float(value) for value in data['rewards']) / size
    return metrics


def _save_adapter(model: Any, output_dir: str, admission: PartitionAdmission) -> str:
    return model.save(
        f'async-{admission.context.adapter_name}-v{admission.step + 1}',
        output_dir=output_dir,
        adapter_name=admission.context.adapter_name,
    )
