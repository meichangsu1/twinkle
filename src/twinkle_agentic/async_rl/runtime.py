# Copyright (c) ModelScope Contributors. All rights reserved.
"""Long-lived dynamic-tenant runtime for async RL."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import partial
from typing import Any

from .context_manager import ContextStatus
from .pipeline import (
    AsyncMultiLoraGRPOPipeline,
    _prompt_batches,
    _reward_for_context,
)
from .types import LoraContext
from .utils import (
    TrainBatchConfig,
    configure_lora_lr_scheduler,
    resolve_context_learning_rate,
    resolve_context_lora_target_modules,
    resolve_context_loss_config,
    resolve_sequence_parallel_size,
    sampler_data_parallel_size,
    validate_context_batch_config,
)


@dataclass(frozen=True)
class RuntimeTenant:
    context: LoraContext
    initial_adapter_path: str


class AsyncRLRuntime:
    """Own the GPU runtime and mutate its tenant registries transactionally."""

    def __init__(self, pipeline: AsyncMultiLoraGRPOPipeline, config: dict[str, Any]):
        self.pipeline = pipeline
        self.config = config
        self.context_manager = pipeline.context_manager
        self.model = pipeline.model
        self.sampler = pipeline.sampler
        self._workers = (
            pipeline.rollout_worker,
            pipeline.advantage_worker,
            pipeline.trainer_worker,
        )
        self._tenants: dict[str, RuntimeTenant] = {}
        self._operation_lock = asyncio.Lock()
        self._monitor_task: asyncio.Task[None] | None = None
        self._failure: str | None = None

    @classmethod
    def from_config(cls, raw_config: dict[str, Any]) -> tuple[AsyncRLRuntime, list[dict[str, Any]]]:
        from omegaconf import OmegaConf

        config = OmegaConf.to_container(OmegaConf.create(raw_config), resolve=True)
        if not isinstance(config, dict):
            raise TypeError('async-RL service config must resolve to a mapping')
        initial_tenants = [dict(item) for item in config.get('lora_contexts', [])]
        base_config = dict(config)
        base_config['lora_contexts'] = []
        base_config.setdefault('runtime', {})['max_steps'] = None
        pipeline = AsyncMultiLoraGRPOPipeline.from_config(base_config, persistent=True)
        return cls(pipeline, base_config), initial_tenants

    async def start(self) -> None:
        await asyncio.gather(*(worker.start.remote() for worker in self._workers))
        if self._monitor_task is None or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self._monitor())

    async def stop(self) -> None:
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            await asyncio.gather(self._monitor_task, return_exceptions=True)
        await asyncio.gather(*(worker.stop.remote() for worker in self._workers), return_exceptions=True)
        await self.pipeline._drain_metrics()
        if self.pipeline.metrics is not None:
            self.pipeline.metrics.close()

    async def check_health(self) -> None:
        if self._failure is not None:
            raise RuntimeError(self._failure)
        states = await asyncio.gather(*(worker.get_service_state.remote() for worker in self._workers))
        failure = next((state['failure'] for state in states if state['failure']), None)
        if failure:
            raise RuntimeError(str(failure))
        if self.sampler is not None:
            await asyncio.to_thread(self.sampler.check_health)

    def capacity(self) -> int:
        return int(self.config['runtime']['sampler_max_loras'])

    def tenant_count(self) -> int:
        return len(self._tenants)

    async def add_tenant(self, item: dict[str, Any]) -> RuntimeTenant:
        async with self._operation_lock:
            context = self._context(item)
            if context.key in self._tenants:
                raise KeyError(f'context already exists: {context.key}')
            await self.context_manager.reserve_context.remote(
                context,
                max_steps=item['train'].get('max_steps'),
            )

            model_added = False
            reward_added = False
            rollout_added = False
            trainer_added = False
            initial_path: str | None = None
            try:
                prepared = self._prepare_tenant(item, context)
                await asyncio.to_thread(self._add_model_adapter, item, context)
                model_added = True
                await asyncio.to_thread(self._configure_model, item, context)
                requested_path = item.get('initial_adapter_path')
                if requested_path:
                    await asyncio.to_thread(
                        self.model.load,
                        requested_path,
                        load_optimizer=False,
                        adapter_name=context.adapter_name,
                    )
                    initial_path = str(requested_path)
                else:
                    initial_path = await asyncio.to_thread(
                        self.model.save,
                        f'async-{context.adapter_name}-initial',
                        output_dir=self.config['runtime']['output_dir'],
                        adapter_name=context.adapter_name,
                    )

                await asyncio.to_thread(self.sampler.register_reward, context.key, prepared['reward'])
                reward_added = True
                await self.pipeline.rollout_worker.register_context.remote(
                    context,
                    prepared['prompt_source'],
                    prepared['rollout_config'],
                )
                rollout_added = True
                await self.pipeline.trainer_worker.register_context.remote(
                    context,
                    mini_batch_size=prepared['train_config'].mini_batch_size,
                    train_batch_config=prepared['train_config'],
                    initial_adapter_path=initial_path,
                    evaluation_config=prepared['evaluation_config'],
                    evaluation_reward=prepared['evaluation_reward'],
                )
                trainer_added = True
                await self.context_manager.activate_context.remote(context, adapter_path=initial_path)
                tenant = RuntimeTenant(context, initial_path)
                self._tenants[context.key] = tenant
                return tenant
            except Exception:
                if trainer_added:
                    await self.pipeline.trainer_worker.unregister_context.remote(context)
                if rollout_added:
                    await self.pipeline.rollout_worker.unregister_context.remote(context)
                if reward_added:
                    await asyncio.to_thread(self.sampler.unregister_reward, context.key)
                if initial_path:
                    try:
                        await asyncio.to_thread(self.sampler.unload_lora_paths, [initial_path])
                    except Exception:
                        pass
                if model_added:
                    try:
                        await asyncio.to_thread(self.model.remove_adapter, context.adapter_name)
                    except Exception:
                        pass
                try:
                    await self.context_manager.unregister_context.remote(context)
                except Exception:
                    pass
                raise

    async def request_drain(self, context_key: str) -> None:
        await self.context_manager.request_context_drain.remote(context_key)

    async def remove_tenant(self, context_key: str) -> None:
        async with self._operation_lock:
            tenant = self._tenants.get(context_key)
            if tenant is None:
                return
            status = await self.context_manager.context_status.remote(context_key)
            if status not in (ContextStatus.DRAINING, ContextStatus.FINISHED):
                await self.context_manager.request_context_drain.remote(context_key)
            while not await self.context_manager.context_is_drained.remote(context_key):
                await asyncio.sleep(0.05)

            paths = await self.context_manager.context_adapter_paths.remote(context_key)
            await self.pipeline.rollout_worker.unregister_context.remote(context_key)
            await self.pipeline.trainer_worker.unregister_context.remote(context_key)
            await asyncio.to_thread(self.sampler.unregister_reward, context_key)
            if paths:
                await asyncio.to_thread(self.sampler.unload_lora_paths, paths)
            await asyncio.to_thread(self.model.remove_adapter, tenant.context.adapter_name)
            await self.context_manager.unregister_context.remote(context_key)
            self._tenants.pop(context_key, None)

    async def snapshots(self) -> list[dict[str, object]]:
        return await self.context_manager.list_context_snapshots.remote()

    async def _monitor(self) -> None:
        interval = self.pipeline.config.metrics_drain_interval_s
        try:
            while True:
                await self.pipeline._drain_metrics()
                await self.check_health()
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._failure = f'{type(exc).__name__}: {exc}'

    def _context(self, item: dict[str, Any]) -> LoraContext:
        return LoraContext(
            str(item['tenant_id']),
            str(item['training_run_id']),
            str(self.config['runtime']['model_id']),
            str(item.get('_runtime_adapter_name', item['adapter_name'])),
        )

    def _add_model_adapter(self, item: dict[str, Any], context: LoraContext) -> None:
        from peft import LoraConfig

        global_lora = self.config['lora']
        adapter_config = LoraConfig(
            target_modules=resolve_context_lora_target_modules(item, global_lora),
            r=global_lora['r'],
            lora_alpha=global_lora['alpha'],
            lora_dropout=global_lora['dropout'],
        )
        self.model.add_adapter_to_model(context.adapter_name, adapter_config, gradient_accumulation_steps=1)

    def _configure_model(self, item: dict[str, Any], context: LoraContext) -> None:
        from twinkle.processor import InputProcessor

        global_lora = self.config['lora']
        model_config = self.config['model']
        template_config = self.config.get('template', {})
        self.model.set_optimizer(
            'AdamW',
            lr=resolve_context_learning_rate(item['train'], global_lora),
            adapter_name=context.adapter_name,
        )
        configure_lora_lr_scheduler(self.model, context.adapter_name, global_lora)
        loss_cls, loss_kwargs = resolve_context_loss_config(item, self.config.get('loss'))
        self.model.set_loss(loss_cls, adapter_name=context.adapter_name, **loss_kwargs)
        self.model.set_processor(
            InputProcessor,
            adapter_name=context.adapter_name,
            padding_free=bool(model_config['padding_free']),
        )
        self.model.set_template(
            template_config.get('cls', 'Qwen3_5Template'),
            model_id=self.config['runtime']['model_id'],
            adapter_name=context.adapter_name,
            enable_thinking=bool(template_config.get('enable_thinking', False)),
            max_length=int(model_config['max_length']),
        )

    def _prepare_tenant(self, item: dict[str, Any], context: LoraContext) -> dict[str, Any]:
        from twinkle.data_format import SamplingParams

        runtime = self.config['runtime']
        rollout = item['rollout']
        train = item['train']
        model_config = self.config['model']
        template_config = self.config.get('template', {})
        rollout_batch_size = int(rollout['batch_size'])
        num_generations = int(rollout['num_generations'])
        train_config = TrainBatchConfig(
            mini_batch_size=int(train['mini_batch_size']),
            micro_batch_size=int(train['micro_batch_size']),
            dynamic_batching=bool(train.get('dynamic_batching', False)),
            max_tokens_per_micro_batch=(
                int(train['max_tokens_per_micro_batch'])
                if train.get('max_tokens_per_micro_batch') is not None else None
            ),
            packing_algorithm=str(train.get('packing_algorithm', 'ffd')),
        )
        sampler_dp = sampler_data_parallel_size(int(runtime['sampler_gpus']), int(runtime['sampler_tp']))
        sequence_parallel_size = resolve_sequence_parallel_size(
            int(runtime['model_gpus']),
            int(model_config['sequence_parallel_size']),
        )
        validate_context_batch_config(
            context.key,
            rollout_groups=rollout_batch_size,
            num_generations=num_generations,
            train=train_config,
            sampler_dp=sampler_dp,
            model_dp=int(runtime['model_gpus']) // sequence_parallel_size,
        )
        template_cls = template_config.get('cls', 'Qwen3_5Template')
        enable_thinking = bool(template_config.get('enable_thinking', False))
        prompt_source = partial(
            _prompt_batches,
            item['dataset'],
            model_id=runtime['model_id'],
            batch_size=rollout_batch_size,
            template_cls=template_cls,
            enable_thinking=enable_thinking,
        )
        sampling_params = SamplingParams(
            max_tokens=rollout['max_tokens'],
            temperature=rollout['temperature'],
            top_p=rollout['top_p'],
            repetition_penalty=float(rollout.get('repetition_penalty', 1.0)),
            logprobs=1,
            num_samples=1,
        )
        evaluation_config = None
        evaluation_reward = None
        global_evaluation = dict(self.config.get('evaluation') or {})
        if bool(global_evaluation.get('enabled', False)):
            eval_dataset = item.get('eval_dataset')
            if eval_dataset is None:
                raise ValueError(f'eval_dataset is required for periodic evaluation of {context.key}')
            eval_sampling = dict(global_evaluation.get('sampling_params') or {})
            eval_batch_size = int(global_evaluation.get('batch_size', 16))
            evaluation_config = {
                'interval': int(global_evaluation.get('interval', 1)),
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
            evaluation_reward = _reward_for_context(
                eval_dataset.get('reward'),
                context_key=f'{context.key} evaluation',
            )
        return {
            'prompt_source': prompt_source,
            'rollout_config': {
                'context': context,
                'batch_size': rollout_batch_size,
                'num_generations': num_generations,
                'sampling_params': sampling_params,
            },
            'train_config': train_config,
            'reward': _reward_for_context(item.get('reward'), context_key=context.key),
            'evaluation_config': evaluation_config,
            'evaluation_reward': evaluation_reward,
        }
