# Copyright (c) ModelScope Contributors. All rights reserved.
"""Driver for the independent async-RL Ray workers."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, Sequence

from .context_manager import LoraContextManager
from .data_plane import TQDataPlane
from .metrics import JSONLMetricsRecorder
from .scheduler import ContextSchedulePolicy, SchedulerConfig
from .types import LoraContext, PartitionAdmission
from .workers import AdvantageWorker, RolloutWorker, TrainerWorker


@dataclass(frozen=True)
class AsyncMultiLoraGRPOConfig:
    metrics_drain_interval_s: float = 1.0


@dataclass(frozen=True)
class TrainBatchConfig:
    mini_batch_size: int
    micro_batch_size: int
    dynamic_batching: bool = False
    max_tokens_per_micro_batch: int | None = None
    packing_algorithm: Literal['ffd', 'kk'] = 'ffd'


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


def _native_fsdp_model_kwargs(model_config: dict[str, Any]) -> dict[str, Any]:
    """Build model kwargs while enforcing FSDP for all RL training entrypoints."""
    strategy = str(model_config.get('strategy', 'native_fsdp'))
    if strategy != 'native_fsdp':
        raise ValueError(f'model.strategy must be native_fsdp for RL training, got {strategy!r}')
    return {
        'strategy': strategy,
        'fsdp_config': dict(model_config.get('fsdp_config') or {}),
    }


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


def _context_learning_rate(train_config: dict[str, Any], lora_config: dict[str, Any]) -> float:
    """Resolve one adapter's learning rate, falling back to the global LoRA default."""
    configured = train_config.get('learning_rate', lora_config.get('learning_rate'))
    if configured is None:
        raise ValueError('train.learning_rate or lora.learning_rate must be configured')
    learning_rate = float(configured)
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError(f'train.learning_rate must be a positive finite value, got {configured!r}')
    return learning_rate


def _context_loss_normalization(train_config: dict[str, Any]) -> str:
    normalization = str(train_config.get('loss_normalization', 'sequence_mean'))
    if normalization not in ('sequence_mean', 'token_mean'):
        raise ValueError(
            'train.loss_normalization must be sequence_mean or token_mean, '
            f'got {normalization!r}')
    return normalization


def _validate_context_batch_config(
    context_key: str,
    *,
    rollout_groups: int,
    num_generations: int,
    train: TrainBatchConfig,
    sampler_dp: int,
    model_dp: int,
) -> None:
    values = {
        'rollout.batch_size': rollout_groups,
        'rollout.num_generations': num_generations,
        'train.mini_batch_size': train.mini_batch_size,
        'train.micro_batch_size': train.micro_batch_size,
    }
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f'{name} for {context_key} must be positive, got {value}')
    if rollout_groups % sampler_dp:
        raise ValueError(f'rollout.batch_size for {context_key} must be divisible by sampler DP size '
                         f'({sampler_dp}), got {rollout_groups}')
    partition_samples = rollout_groups * num_generations
    if partition_samples % train.mini_batch_size:
        raise ValueError(
            f'partition for {context_key} has {partition_samples} samples and must be divisible by '
            f'train.mini_batch_size={train.mini_batch_size}')
    if train.mini_batch_size % num_generations:
        raise ValueError(
            f'train.mini_batch_size for {context_key} must preserve complete prompt groups: '
            f'{train.mini_batch_size} % {num_generations} != 0')
    if train.mini_batch_size % model_dp:
        raise ValueError(f'train.mini_batch_size for {context_key} must be divisible by '
                         f'model DP size {model_dp}')
    samples_per_rank = train.mini_batch_size // model_dp
    if train.micro_batch_size > samples_per_rank:
        raise ValueError(f'train.micro_batch_size for {context_key} must not exceed the per-rank train batch '
                         f'({samples_per_rank}), got {train.micro_batch_size}')
    if train.dynamic_batching:
        if train.max_tokens_per_micro_batch is None or train.max_tokens_per_micro_batch <= 0:
            raise ValueError(
                f'train.max_tokens_per_micro_batch for {context_key} must be positive when '
                'train.dynamic_batching=true')
    if train.packing_algorithm not in ('ffd', 'kk'):
        raise ValueError(
            f'train.packing_algorithm for {context_key} must be ffd or kk, '
            f'got {train.packing_algorithm!r}')


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
        model_kwargs = _native_fsdp_model_kwargs(model_config)
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
        train_batch_configs: dict[str, TrainBatchConfig] = {}
        rewards: dict[str, Any] = {}
        evaluation_config: dict[str, dict[str, Any]] = {}
        evaluation_rewards: dict[str, Any] = {}
        initial_paths: dict[str, str] = {}
        global_evaluation = dict(raw_config.get('evaluation') or {})
        for item in raw_config['lora_contexts']:
            train = item['train']
            context = LoraContext(
                item['tenant_id'],
                item['training_run_id'],
                runtime['model_id'],
                item['adapter_name'],
                item.get('reward_type', 'gsm8k_accuracy'),
            )
            contexts.append(context)
            model.add_adapter_to_model(context.adapter_name, lora_config, gradient_accumulation_steps=1)
            model.set_optimizer(
                'AdamW',
                lr=_context_learning_rate(train, lora_config_data),
                adapter_name=context.adapter_name,
            )
            configure_lora_lr_scheduler(model, context.adapter_name, lora_config_data)
            model.set_loss(
                'GRPOLoss',
                epsilon=.2,
                normalization=_context_loss_normalization(train),
                adapter_name=context.adapter_name,
            )
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
                max_length=model_max_length,
            )

            rollout = item['rollout']
            rollout_batch_size = int(rollout['batch_size'])
            num_generations = int(rollout['num_generations'])
            train_batch_config = TrainBatchConfig(
                mini_batch_size=int(train['mini_batch_size']),
                micro_batch_size=int(train['micro_batch_size']),
                dynamic_batching=bool(train.get('dynamic_batching', False)),
                max_tokens_per_micro_batch=(
                    int(train['max_tokens_per_micro_batch'])
                    if train.get('max_tokens_per_micro_batch') is not None else None
                ),
                packing_algorithm=str(train.get('packing_algorithm', 'ffd')),
            )
            _validate_context_batch_config(
                context.key,
                rollout_groups=rollout_batch_size,
                num_generations=num_generations,
                train=train_batch_config,
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
            train_batch_configs[context.key] = train_batch_config
            rewards[context.key] = _reward_for_context(
                context,
                reward_config=item.get('reward'),
                rollout_config=rollout,
            )
            if bool(global_evaluation.get('enabled', False)):
                eval_dataset = item.get('eval_dataset')
                if eval_dataset is None:
                    raise ValueError(f'eval_dataset is required for periodic evaluation of {context.key}')
                eval_batch_size = int(global_evaluation.get('batch_size', 16))
                eval_interval = int(global_evaluation.get('interval', 1))
                if eval_batch_size <= 0 or eval_interval <= 0:
                    raise ValueError('evaluation.batch_size and evaluation.interval must be positive')
                eval_sampling = dict(global_evaluation.get('sampling_params') or {})
                evaluation_config[context.key] = {
                    'interval': eval_interval,
                    'dataset_name': eval_dataset.get('name', eval_dataset['dataset_id']),
                    'prompt_batches': partial(
                        _prompt_batches,
                        eval_dataset,
                        model_id=runtime['model_id'],
                        batch_size=eval_batch_size,
                        template_cls=template_cls,
                        enable_thinking=enable_thinking,
                        full_batches_only=False,
                    ),
                    'sampling_params': SamplingParams(
                        max_tokens=int(eval_sampling.get('max_tokens', rollout['max_tokens'])),
                        temperature=float(eval_sampling.get('temperature', 0.0)),
                        top_p=float(eval_sampling.get('top_p', 1.0)),
                        repetition_penalty=float(eval_sampling.get('repetition_penalty', 1.0)),
                        logprobs=0,
                        num_samples=1,
                    ),
                }
                eval_context = LoraContext(
                    context.tenant_id,
                    context.training_run_id,
                    context.base_model_id,
                    context.adapter_name,
                    eval_dataset['reward_type'],
                )
                evaluation_rewards[context.key] = _reward_for_context(eval_context)
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
        sampler_engine_args = {
            'tensor_parallel_size': sampler_tp,
            'enable_lora': True,
            'max_loras': int(runtime['sampler_max_loras']),
            'max_lora_rank': lora_config_data['r'],
            'max_model_len': int(sampler_config['max_model_len']),
            'gpu_memory_utilization': float(sampler_config['gpu_memory_utilization']),
            'max_num_seqs': int(sampler_config['max_num_seqs']),
            'enforce_eager': bool(sampler_config['enforce_eager']),
            'seed': int(runtime.get('seed', 1)),
        }
        if sampler_config.get('max_num_batched_tokens') is not None:
            sampler_engine_args['max_num_batched_tokens'] = int(sampler_config['max_num_batched_tokens'])
        sampler = VLLMSamplerTQ(
            model_id=runtime['model_id'],
            remote_group='sampler',
            device_mesh=sampler_mesh,
            engine_args=sampler_engine_args,
            reward_registry=rewards,
            context_manager=manager,
            rollout_max_retries=int(runtime.get('rollout_max_retries', 2)),
            rollout_retry_delay_s=float(runtime.get('rollout_retry_delay_s', 0.5)),
        )
        sampler.set_template(
            template_cls,
            model_id=runtime['model_id'],
            enable_thinking=enable_thinking,
            max_length=model_max_length,
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
            scheduler=_scheduler(raw_config['scheduler']['advantage']),
        )
        trainer_worker = create_cpu_actor(
            TrainerWorker,
            context_manager=manager,
            data_plane=TQDataPlane(),
            train_fn=partial(
                _train_batch,
                model,
                train_batch_configs,
                model_data_parallel_size=model_data_parallel_size,
            ),
            save_adapter=partial(_save_adapter, model, runtime['output_dir']),
            mini_batch_sizes={
                key: config.mini_batch_size for key, config in train_batch_configs.items()
            },
            scheduler=_scheduler(raw_config['scheduler']['train']),
            keep_adapter_versions=runtime['keep_adapter_versions'],
            initial_adapter_paths=initial_paths,
            evaluation_config=evaluation_config,
            evaluate_batch=partial(_evaluate_batch, sampler, evaluation_rewards) if evaluation_config else None,
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
    full_batches_only: bool = True,
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
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            min_batch_size=batch_size if full_batches_only else 1,
        )
        remaining = data_num
        remaining = None if remaining is None else int(remaining)
        for batch in loader:
            if full_batches_only and (len(batch) != batch_size or (remaining is not None and remaining < batch_size)):
                return
            yield batch
            if remaining is not None:
                remaining -= batch_size

    return batches()


def _evaluate_batch(
    sampler: Any,
    reward_registry: dict[str, Any],
    prompts: Sequence[dict[str, Any]],
    admission: PartitionAdmission,
    adapter_path: str,
    policy_version: int,
    sampling_params: Any,
) -> dict[str, Any]:
    from .vllm_sampler_tq import _sample_responses_to_rollout_rows

    responses = sampler.evaluate(
        list(prompts),
        sampling_params,
        admission.context.adapter_name,
        adapter_path,
    )
    rows = _sample_responses_to_rollout_rows(list(prompts), responses, policy_version=policy_version)
    reward_fn = reward_registry[admission.context.key]
    rewards = list(reward_fn(rows, context=admission.context))
    return {
        'rewards': rewards,
        'completion_lengths': [int(row['completion_length']) for row in rows],
    }


def _reward_for_context(
    context: LoraContext,
    *,
    reward_config: dict[str, Any] | None = None,
    rollout_config: dict[str, Any] | None = None,
) -> Any:
    from twinkle.reward import (BoxedMathAccuracyReward, DAPOMathAccuracyReward, DAPOMathReward,
                                GSM8KAccuracyBrevityReward, GSM8KAccuracyReward, MathVerifyAccuracyReward)
    if context.reward_type in {'gsm8k', 'gsm8k_accuracy'}:
        return GSM8KAccuracyReward()
    if context.reward_type == 'gsm8k_accuracy_brevity':
        return GSM8KAccuracyBrevityReward()
    if context.reward_type == 'math_verify_accuracy':
        return MathVerifyAccuracyReward()
    if context.reward_type == 'dapo_math_accuracy':
        return DAPOMathAccuracyReward()
    if context.reward_type == 'dapo_math':
        if rollout_config is None:
            raise ValueError(f'rollout config is required for DAPO reward in {context.key}')
        reward_config = dict(reward_config or {})
        return DAPOMathReward(
            max_response_length=int(rollout_config['max_tokens']),
            overlong_buffer_length=int(reward_config['overlong_buffer_length']),
            overlong_penalty_factor=float(reward_config.get('overlong_penalty_factor', 1.0)),
            score_tail_chars=int(reward_config.get('score_tail_chars', 300)),
        )
    if context.reward_type == 'aime2024_accuracy':
        return BoxedMathAccuracyReward()
    raise ValueError(f'unsupported async-RL reward_type {context.reward_type!r} for {context.key}')


def _compute_advantages(data: Any, admission: PartitionAdmission) -> tuple[list[float], list[float]]:
    from twinkle.advantage import GRPOAdvantage
    rewards = [float(value) for value in data['rewards']]
    advantages = GRPOAdvantage()(rewards, num_generations=admission.num_generations, scale='group').tolist()
    return advantages, rewards


def _train_batch(
    model: Any,
    train_batch_configs: dict[str, TrainBatchConfig],
    data: Any,
    admission: PartitionAdmission,
    *,
    model_data_parallel_size: int = 1,
) -> dict[str, Any]:
    from .tq_utils import REQUIRED_MODEL_INPUT_FIELDS

    size = int(data.batch_size[0])
    inputs = [{name: data[name][index] for name in REQUIRED_MODEL_INPUT_FIELDS} for index in range(size)]
    old_logps = list(data['logprobs'])
    advantages = list(data['advantages'])
    config = train_batch_configs[admission.context.key]
    if size != config.mini_batch_size:
        raise ValueError(
            f'train batch for {admission.context.key} has {size} samples; '
            f'expected mini_batch_size={config.mini_batch_size}')

    if size % model_data_parallel_size:
        raise ValueError(f'train batch size {size} must be divisible by model DP size '
                         f'{model_data_parallel_size}')
    model.forward_backward(
        inputs=inputs,
        old_logps=old_logps,
        advantages=advantages,
        adapter_name=admission.context.adapter_name,
        micro_batch_size=config.micro_batch_size,
        dynamic_batching=config.dynamic_batching,
        max_tokens_per_micro_batch=config.max_tokens_per_micro_batch,
        packing_algorithm=config.packing_algorithm,
        sync_gradients=True,
        loss_scale=1.0,
    )

    model.clip_grad_and_step(adapter_name=admission.context.adapter_name)
    metrics = dict(model.calculate_metric(is_training=True, adapter_name=admission.context.adapter_name))
    metrics['mini_batch_size'] = config.mini_batch_size
    metrics['micro_batch_size_per_rank'] = config.micro_batch_size
    metrics['dynamic_batching'] = config.dynamic_batching
    return metrics


def _save_adapter(model: Any, output_dir: str, admission: PartitionAdmission) -> str:
    return model.save(
        f'async-{admission.context.adapter_name}-v{admission.step + 1}',
        output_dir=output_dir,
        adapter_name=admission.context.adapter_name,
    )
