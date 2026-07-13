# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from twinkle.data_format import Trajectory
from .data_plane import TransferQueueDataPlane, TransferQueueRuntimeConfig
from .metrics import AsyncRLMetricsConfig, AsyncRLMetricsRecorder, build_metrics_recorder
from .prompt_loader import PromptLoader
from .registry import LoraRuntimeRegistry
from .staleness import StalenessManager
from .types import (GRPOAdvantageBatch, LoraContext, PartitionMeta, PartitionStatus, PipelineStepResult,
                    PromptGroupStatus, RolloutCallable)
from .workers import (AdvantageWorker, AsyncRollouter, MultiLoraGRPOTrainConfig, MultiLoraGRPOTrainerWorker,
                      ToolManagerFactory, TrainerScheduler, TrainerStepResult, TrainerWorker)


@dataclass
class BaseRLPipelineConfig:
    """Runtime knobs for the MVP async RL pipeline.

    The first version follows the short_math_grpo client pattern:
    train one or more LoRA adapters with MultiLoraTransformersModel, save
    adapter weights after each train partition, and pass that saved path to
    rollout.
    """

    lora_contexts: list[LoraContext] | None = None
    tenant_id: str = 'default_tenant'
    training_run_id: str = 'default_run'
    base_model_id: str = ''
    adapter_name: str = 'default'
    reward_type: str = 'default'
    algorithm: str = 'grpo'
    tool_profile: str = 'default'
    max_staleness: int = 0
    max_concurrency: int = 16
    default_rollout_batch_size: int = 1
    target_groups_by_context: dict[str, int] = field(default_factory=dict)
    default_num_generations: int = 1
    num_generations_by_context: dict[str, int] = field(default_factory=dict)
    default_mini_batch_size: int = 1
    mini_batch_size_by_context: dict[str, int] = field(default_factory=dict)
    max_train_steps: int | None = None
    save_name_prefix: str = 'async-rl-sampler-weights'
    adapter_checkpoint_dir: str | None = None
    save_optimizer: bool = False
    is_sampler_checkpoint: bool = True
    max_grad_norm: float = 1.0
    norm_type: int = 2
    train_kwargs: dict[str, Any] = field(default_factory=dict)
    tq_config: TransferQueueRuntimeConfig | None = None
    metrics: AsyncRLMetricsConfig | None = None


class BaseRLPipeline(ABC):
    """Compose rollout, TQ data plane, reward, advantage, and trainer workers.

    This class is intentionally a thin orchestrator. Vertical tasks still own
    rollout behavior, reward logic, advantage logic, and model configuration.
    The default train step is compatible with `MultiLoraTransformersModel`.
    """

    def __init__(
        self,
        *,
        config: BaseRLPipelineConfig,
    ):
        self.config = config
        self.model = None
        self.rollout: RolloutCallable | None = None
        self.reward_registry: dict[str, Callable[..., list[float]]] = {}
        self.data_plane = None
        self.lora_runtime_registry = None
        self.staleness_manager = None
        self.tool_manager_factory = None
        self.advantage_fn = None
        self.rollout_policy = None
        self.train_policy = None
        self.train_batch_fn = None
        self.receive_weights_fn = None
        self._sync_step_loop = None
        self.metrics_recorder: AsyncRLMetricsRecorder | None = None

        self.build_components()
        self.allocate_resources()
        self.create_roles()

    def build_components(self) -> None:
        """Create model and rollout components."""
        if self.model is None:
            self.model = self.build_model()
        if self.requires_rollout() and self.rollout is None:
            self.rollout = self.build_rollout()
        if self.model is None:
            raise ValueError('BaseRLPipeline requires build_model() or a model override')
        if self.requires_rollout() and self.rollout is None:
            raise ValueError('BaseRLPipeline requires build_rollout() or a rollout override')

    def requires_rollout(self) -> bool:
        """Whether the configured algorithm needs rollout-side components."""
        return self.config.algorithm.lower() in {'grpo', 'ppo', 'dapo'}

    @abstractmethod
    def build_model(self) -> Any:
        """Build the train-side model resource.

        Pipeline implementations must provide the train-side model resource.
        """
        raise NotImplementedError

    def build_rollout(self) -> RolloutCallable | None:
        """Build the rollout implementation resource.

        Rollout algorithms such as GRPO/PPO/DAPO must override this hook.
        Non-rollout algorithms can leave it unimplemented because
        `build_components()` only calls it when `requires_rollout()` is true.
        """
        raise NotImplementedError(f'{self.__class__.__name__}.build_rollout() is required for '
                                  f'algorithm={self.config.algorithm!r}')

    def build_data_plane(self) -> TransferQueueDataPlane:
        """Build the TransferQueue data plane resource."""
        return TransferQueueDataPlane(tq_config=self.config.tq_config)

    def build_metrics_recorder(self) -> AsyncRLMetricsRecorder:
        return build_metrics_recorder(self.config.metrics)

    def build_reward_registry(self) -> dict[str, Callable[..., list[float]]]:
        return {}

    def build_advantage_fn(
        self, ) -> Callable[[GRPOAdvantageBatch, LoraContext], tuple[list[float], list[float]]] | None:
        return None

    def build_tool_manager_factory(self) -> ToolManagerFactory | None:
        return None

    def build_rollout_policy(self) -> Any | None:
        return None

    def build_train_policy(self) -> Any | None:
        return None

    def build_train_batch_fn(self, ) -> Callable[[LoraContext, Any], TrainerStepResult | dict[str, Any] | None] | None:
        return None

    def build_save_adapter_fn(self, ) -> Callable[[LoraContext, str], TrainerStepResult | dict[str, Any] | None] | None:
        return None

    def build_receive_weights_fn(self) -> Callable[[Any], None] | None:
        return None

    def build_prompt_loaders(self) -> list[PromptLoader]:
        """Build rollout-side prompt loaders.

        Subclasses can wrap `twinkle.dataloader.DataLoader` instances here.
        The default is empty so callers may still push prompts explicitly with
        `submit_prompt_groups()`.
        """
        return []

    def build_default_context(self) -> LoraContext:
        config = self.config
        return LoraContext(
            tenant_id=config.tenant_id,
            training_run_id=config.training_run_id,
            base_model_id=config.base_model_id,
            adapter_name=config.adapter_name,
            reward_type=config.reward_type,
            algorithm=config.algorithm,
            tool_profile=config.tool_profile,
        )

    def build_lora_contexts(self) -> list[LoraContext]:
        """Build the LoRA contexts managed by this pipeline.

        A context identifies one training run and one LoRA adapter. Multi-LoRA
        jobs pass multiple contexts; single-LoRA jobs keep using the legacy
        scalar config fields.
        """
        if self.config.lora_contexts:
            return list(self.config.lora_contexts)
        return [self.build_default_context()]

    def allocate_resources(self) -> None:
        """Initialize shared resources: contexts, TransferQueue data plane, and registries."""
        config = self.config
        self.contexts = self.build_lora_contexts()
        if not self.contexts:
            raise ValueError('BaseRLPipeline requires at least one LoraContext')
        context_keys = [context.key for context in self.contexts]
        if len(context_keys) != len(set(context_keys)):
            raise ValueError(f'duplicate LoraContext keys are not allowed: {context_keys}')
        self.context = self.contexts[0]

        if self.metrics_recorder is None:
            self.metrics_recorder = self.build_metrics_recorder()
        if self.data_plane is None:
            self.data_plane = self.build_data_plane()
        self.data_plane.metrics_recorder = self.metrics_recorder
        if self.lora_runtime_registry is None:
            self.lora_runtime_registry = LoraRuntimeRegistry()
        if self.staleness_manager is None:
            self.staleness_manager = StalenessManager(
                max_staleness=config.max_staleness,
                target_groups_per_partition=config.default_rollout_batch_size,
            )
        for context in self.contexts:
            self.lora_runtime_registry.register(context)

    def create_roles(self) -> None:
        """Create runtime roles for the default GRPO pipeline."""
        algorithm = self.config.algorithm.lower()
        if algorithm != 'grpo':
            raise NotImplementedError(f'BaseRLPipeline only defines default roles for algorithm={algorithm!r}. '
                                      'Override create_roles() in an algorithm-specific pipeline.')
        self.create_grpo_roles()

    def create_grpo_roles(self) -> None:
        """Create the default Multi-LoRA GRPO stage roles."""
        if not self.reward_registry:
            self.reward_registry = self.build_reward_registry()
        self.advantage_fn = self.build_advantage_fn()
        self.tool_manager_factory = self.build_tool_manager_factory()
        self.rollout_policy = self.build_rollout_policy()
        self.train_policy = self.build_train_policy()
        self.train_batch_fn = self.build_train_batch_fn()
        self.save_adapter_fn = self.build_save_adapter_fn()
        self.receive_weights_fn = self.build_receive_weights_fn()
        self.rollouter = self.build_rollouter(
            tool_manager_factory=self.tool_manager_factory,
            rollout_policy=self.rollout_policy,
        )
        self.advantage_worker = self.build_advantage_worker(advantage_fn=self.advantage_fn)
        self.trainer_scheduler = self.build_trainer_scheduler(train_policy=self.train_policy)
        self.trainer_worker = self.build_trainer_worker()
        self.prompt_loaders = self.build_prompt_loaders()
        self.rollouter.attach_prompt_loaders(self.prompt_loaders)

    def build_rollouter(
        self,
        *,
        tool_manager_factory: ToolManagerFactory | None,
        rollout_policy: Any | None,
    ) -> AsyncRollouter:
        config = self.config
        if self.rollout is None:
            raise ValueError('build_rollouter requires a rollout implementation')
        return AsyncRollouter(
            data_plane=self.data_plane,
            lora_runtime_registry=self.lora_runtime_registry,
            staleness_manager=self.staleness_manager,
            rollout=self.rollout,
            tool_manager_factory=tool_manager_factory,
            rollout_policy=rollout_policy,
            reward_registry=self.reward_registry,
            max_concurrency=config.max_concurrency,
            target_groups_per_partition=config.default_rollout_batch_size,
            target_groups_by_context=config.target_groups_by_context,
            num_generations=config.default_num_generations,
            num_generations_by_context=config.num_generations_by_context,
            metrics_recorder=self.metrics_recorder,
        )

    def build_advantage_worker(
        self,
        *,
        advantage_fn: Callable[[GRPOAdvantageBatch, LoraContext], tuple[list[float], list[float]]] | None,
    ) -> AdvantageWorker:
        return AdvantageWorker(
            data_plane=self.data_plane,
            contexts=self.contexts,
            lora_runtime_registry=self.lora_runtime_registry,
            batch_size=self.config.default_mini_batch_size,
            batch_size_by_context=self.config.mini_batch_size_by_context,
            num_generations=self.config.default_num_generations,
            num_generations_by_context=self.config.num_generations_by_context,
            advantage_fn=advantage_fn,
            metrics_recorder=self.metrics_recorder,
        )

    def build_trainer_scheduler(self, *, train_policy: Any | None) -> TrainerScheduler:
        return TrainerScheduler(lora_runtime_registry=self.lora_runtime_registry, train_policy=train_policy)

    def build_trainer_worker(self) -> TrainerWorker:
        if self.train_batch_fn is not None:
            return TrainerWorker(
                data_plane=self.data_plane,
                lora_runtime_registry=self.lora_runtime_registry,
                scheduler=self.trainer_scheduler,
                train_batch_fn=self.train_batch_fn,
                save_adapter_fn=self.save_adapter_fn,
                receive_weights_fn=self.receive_weights_fn,
                train_batch_groups=self.config.default_mini_batch_size,
                train_batch_groups_by_context=self.config.mini_batch_size_by_context,
                metrics_recorder=self.metrics_recorder,
            )

        if self.__class__.train_batch is not BaseRLPipeline.train_batch:
            return TrainerWorker(
                data_plane=self.data_plane,
                lora_runtime_registry=self.lora_runtime_registry,
                scheduler=self.trainer_scheduler,
                train_batch_fn=self.train_batch,
                save_adapter_fn=self.save_adapter_fn,
                receive_weights_fn=self.receive_weights_fn,
                train_batch_groups=self.config.default_mini_batch_size,
                train_batch_groups_by_context=self.config.mini_batch_size_by_context,
                metrics_recorder=self.metrics_recorder,
            )

        return MultiLoraGRPOTrainerWorker(
            data_plane=self.data_plane,
            lora_runtime_registry=self.lora_runtime_registry,
            scheduler=self.trainer_scheduler,
            model=self.model,
            train_config=MultiLoraGRPOTrainConfig(
                save_name_prefix=self.config.save_name_prefix,
                adapter_checkpoint_dir=self.config.adapter_checkpoint_dir,
                save_optimizer=self.config.save_optimizer,
                is_sampler_checkpoint=self.config.is_sampler_checkpoint,
                max_grad_norm=self.config.max_grad_norm,
                norm_type=self.config.norm_type,
                train_kwargs=self.config.train_kwargs,
            ),
            receive_weights_fn=self.receive_weights_fn,
            train_batch_groups=self.config.default_mini_batch_size,
            train_batch_groups_by_context=self.config.mini_batch_size_by_context,
            metrics_recorder=self.metrics_recorder,
        )

    @classmethod
    def build_multilora_model(
        cls,
        *,
        model_id: str,
        adapter_name: str,
        lora_config: Any,
        loss_cls: str = 'GRPOLoss',
        optimizer_cls: str = 'Adam',
        learning_rate: float = 2e-5,
        template_cls: str = 'Qwen3_5Template',
        processor_cls: str = 'InputProcessor',
        gradient_accumulation_steps: int = 1,
        loss_kwargs: dict[str, Any] | None = None,
        optimizer_kwargs: dict[str, Any] | None = None,
        template_kwargs: dict[str, Any] | None = None,
        processor_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Create and configure MultiLoraTransformersModel like the GRPO cookbook."""

        from twinkle_client.model import MultiLoraTransformersModel

        model = MultiLoraTransformersModel(model_id=model_id)
        model.add_adapter_to_model(
            adapter_name,
            lora_config,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        model.set_loss(loss_cls, **(loss_kwargs or {'epsilon': 0.2, 'beta': 0.0}))
        optimizer_params = {'lr': learning_rate}
        optimizer_params.update(optimizer_kwargs or {})
        model.set_optimizer(optimizer_cls, **optimizer_params)
        model.set_processor(processor_cls, **(processor_kwargs or {}))
        model.set_template(template_cls, model_id=model_id, **(template_kwargs or {}))
        return model

    def submit_prompt_groups(self, prompt_groups: Iterable[Trajectory], context: LoraContext | None = None) -> None:
        """Submit prompt groups for rollout.

        These are not trainer batches. The trainer only reads samples that have
        already passed rollout/reward/advantage stages from TransferQueue.
        """
        self.rollouter.enqueue_prompt_groups(context or self.current_context(), prompt_groups)

    async def step_async(self, *, max_train_partitions: int | None = None) -> PipelineStepResult:
        """Run one coarse orchestration cycle: rollout, advantage, then train.

        This is intentionally not a component tick loop. Rollout owns prompt
        loading/task completion/submission, advantage consumes TQ metadata, and
        trainer synchronously drains one selected context until it is blocked.
        """
        self._require_orchestrated_roles()

        rollout_result = self.rollouter.step()
        if asyncio.iscoroutine(rollout_result):
            rollout_result = await rollout_result

        advantage_result = self.advantage_worker.process_available()
        train_result = self.trainer_worker.train_until_blocked(max_partitions=max_train_partitions)

        step_result = PipelineStepResult(
            rollout=None if rollout_result is None else rollout_result.metadata,
            advantage=None if advantage_result is None else advantage_result.metadata,
            train=None if train_result is None else train_result.metadata,
            rollout_events=0 if rollout_result is None else rollout_result.count,
            advantage_groups=0 if advantage_result is None else advantage_result.count,
            train_batches=0 if train_result is None else train_result.train_batches,
            trained_partitions=0 if train_result is None else train_result.trained_partitions,
        )
        self._last_step_had_work = step_result.had_work
        if self.metrics_recorder is not None and self.metrics_recorder.should_record_event('pipeline_step'):
            self.metrics_recorder.log_event(
                event='pipeline_step',
                phase='pipeline',
                metrics={
                    **self._backlog_metrics(),
                    'pipeline_had_work': self._last_step_had_work,
                },
            )
        return step_result

    def step(self) -> PipelineStepResult:
        if self._sync_step_loop is None or self._sync_step_loop.is_closed():
            self._sync_step_loop = asyncio.new_event_loop()
        return self._sync_step_loop.run_until_complete(self.step_async())

    async def run_async(
        self,
        prompt_groups: Iterable[Trajectory] | None = None,
        *,
        max_steps: int | None = None,
    ) -> list[PipelineStepResult]:
        if prompt_groups is not None:
            self.submit_prompt_groups(prompt_groups)
        limit = max_steps if max_steps is not None else self.config.max_train_steps
        history: list[PipelineStepResult] = []
        trained = 0
        while limit is None or trained < limit:
            remaining = None if limit is None else max(0, limit - trained)
            result = await self.step_async(max_train_partitions=remaining)
            history.append(result)
            if result.trained_partitions:
                trained += result.trained_partitions
                if self.should_stop(trained):
                    break
                continue
            if result.had_work:
                continue
            if self._is_drained():
                break
            await asyncio.sleep(0.05)
        return history

    def run(
        self,
        prompt_groups: Iterable[Trajectory] | None = None,
        *,
        max_steps: int | None = None,
    ) -> list[PipelineStepResult]:
        """Drive the async RL loop.

        `prompt_groups` is an optional convenience feed for rollout prompts.
        Training batches are always read from TransferQueue by TrainerWorker.
        """
        return asyncio.run(self.run_async(prompt_groups, max_steps=max_steps))

    def run_until_idle(self, *, max_steps: int | None = None) -> list[PipelineStepResult]:
        """Advance workers without adding new rollout prompts."""
        return self.run(max_steps=max_steps)

    def _require_orchestrated_roles(self) -> None:
        missing = [
            name for name in ('rollouter', 'advantage_worker', 'trainer_worker')
            if not hasattr(self, name)
        ]
        if missing:
            raise NotImplementedError(
                f'{self.__class__.__name__} does not provide orchestrated async RL roles: {missing}')

    def sync_and_clear_completed_partitions(self, metadata: PartitionMeta) -> None:
        """Hook for custom pipelines after a train_k is completed.

        The MVP `TrainerWorker` performs adapter save/version update and clear
        inline. Custom pipelines can override this method together with a
        custom trainer worker if they need a different sync boundary.
        """
        self.data_plane.clear_partition(metadata.context, metadata.partition_id)

    def should_stop(self, trained_steps: int) -> bool:
        return self.config.max_train_steps is not None and trained_steps >= self.config.max_train_steps

    def _backlog_metrics(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            'pending_prompt_groups': 0,
            'inflight_rollout_groups': 0,
            'active_partitions': 0,
            'closed_partitions': 0,
            'rollout_done_groups': 0,
            'advantaging_groups': 0,
            'advantage_done_groups': 0,
            'training_groups': 0,
            'untrained_groups': 0,
        }
        rollouter = getattr(self, 'rollouter', None)
        for context in getattr(self, 'contexts', []):
            if rollouter is not None:
                metrics['pending_prompt_groups'] += rollouter.pending_prompt_group_count(context)
            try:
                runtime_state = self.lora_runtime_registry.get(context)
                metrics['inflight_rollout_groups'] += runtime_state.in_flight_groups
            except KeyError:
                pass
        live_partitions = self._runtime_live_partitions()
        for partition in live_partitions:
            if partition.status == PartitionStatus.ACTIVE:
                metrics['active_partitions'] += 1
            elif partition.status == PartitionStatus.CLOSED:
                metrics['closed_partitions'] += 1
        groups = []
        if live_partitions:
            for partition in live_partitions:
                groups.extend(
                    self.data_plane.list_prompt_groups(
                        partition.context,
                        partition_id=partition.partition_id,
                    ))
        else:
            groups = self.data_plane.list_all_prompt_groups()
        status_keys = {
            PromptGroupStatus.ROLLOUT_DONE: 'rollout_done_groups',
            PromptGroupStatus.ADVANTAGING: 'advantaging_groups',
            PromptGroupStatus.ADVANTAGE_DONE: 'advantage_done_groups',
            PromptGroupStatus.TRAINING: 'training_groups',
        }
        untrained_statuses = {
            PromptGroupStatus.ROLLOUT_DONE,
            PromptGroupStatus.ADVANTAGING,
            PromptGroupStatus.ADVANTAGE_DONE,
            PromptGroupStatus.TRAINING,
        }
        for group in groups:
            key = status_keys.get(group.status)
            if key is not None:
                metrics[key] += 1
            if group.status in untrained_statuses:
                metrics['untrained_groups'] += 1
        return metrics

    def shutdown(self) -> None:
        components = [
            getattr(self, 'rollouter', None),
            getattr(self, 'advantage_worker', None),
            getattr(self, 'trainer_worker', None),
        ]
        for component in components:
            if component is None:
                continue
            shutdown = getattr(component, 'shutdown', None)
            if shutdown is not None:
                shutdown()
        if self.metrics_recorder is not None:
            self.metrics_recorder.close()
        if self._sync_step_loop is not None and not self._sync_step_loop.is_closed():
            self._sync_step_loop.run_until_complete(asyncio.sleep(0))
            self._sync_step_loop.close()
        close = getattr(getattr(self, 'data_plane', None), 'close', None)
        if close is not None:
            close()

    def train_batch(self, context: LoraContext, batch: Any) -> TrainerStepResult:
        """Override hook for one train batch.

        The default GRPO train path is implemented by
        `MultiLoraGRPOTrainerWorker`. New algorithms should prefer overriding
        `build_trainer_worker()`.
        """
        raise NotImplementedError('BaseRLPipeline.train_batch is not implemented. '
                                  'Use MultiLoraGRPOTrainerWorker or override build_trainer_worker().')

    def current_context(self, context: LoraContext | None = None) -> LoraContext:
        return context or self.context

    def current_contexts(self) -> list[LoraContext]:
        return [self.current_context(context) for context in self.contexts]

    def _is_drained(self) -> bool:
        rollouter = getattr(self, 'rollouter', None)
        if rollouter is not None and not rollouter.is_idle():
            return False
        if rollouter is None:
            return True
        for context in self.current_contexts():
            if self.rollouter.pending_prompt_group_count(context) > 0:
                return False
            terminal_group_statuses = {
                PromptGroupStatus.TRAIN_DONE,
                PromptGroupStatus.FAILED,
                PromptGroupStatus.DROPPED,
            }
            terminal_partition_statuses = {
                PartitionStatus.CLEARED,
                PartitionStatus.FAILED,
                PartitionStatus.TRAIN_DONE,
            }
            partitions = self._runtime_live_partitions(context)
            for partition in partitions:
                if partition.status in terminal_partition_statuses:
                    continue
                groups = self.data_plane.list_prompt_groups(context, partition_id=partition.partition_id)
                if partition.status == PartitionStatus.CLOSED and groups and all(group.status in terminal_group_statuses
                                                                                 for group in groups):
                    continue
                return False
        return True

    def _runtime_live_partitions(self, context: LoraContext | None = None) -> list[PartitionMeta]:
        partition_ids = []
        contexts = [context] if context is not None else self.current_contexts()
        for item in contexts:
            try:
                runtime_state = self.lora_runtime_registry.get(item)
            except KeyError:
                continue
            partition_ids.extend(sorted(runtime_state.live_partitions))
        partitions = []
        stale_partition_ids: list[tuple[LoraContext, str]] = []
        for partition_id in partition_ids:
            try:
                partitions.append(self.data_plane.get_rollout_partition(partition_id))
            except KeyError:
                owner = next((item for item in contexts if partition_id.startswith(f'{item.key}/')), None)
                if owner is not None:
                    stale_partition_ids.append((owner, partition_id))
        for owner, partition_id in stale_partition_ids:
            self.lora_runtime_registry.on_partition_cleared(owner, partition_id)
        if partitions or partition_ids:
            return sorted(partitions, key=lambda partition: (partition.created_at, partition.partition_id))
        if context is not None:
            return self.data_plane.list_partitions(context)
        return self.data_plane.list_partitions()
