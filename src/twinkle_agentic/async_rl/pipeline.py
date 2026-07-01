# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional

from .data_plane import TransferQueueDataPlane, TransferQueueRuntimeConfig
from .prompt_loader import PromptLoader
from .registry import LoraAdapterRegistry
from .staleness import StalenessManager
from .types import LoraContext, PartitionMetadata, RolloutCallable, SampleRecord
from .workers import (AdvantageWorker, AsyncRollouter, MultiLoraGRPOTrainConfig, MultiLoraGRPOTrainerWorker,
                      RewardWorker, ToolManagerFactory, TrainerScheduler, TrainerStepResult, TrainerWorker)


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
    max_concurrent_groups: int = 16
    target_groups_per_partition: int = 1
    reward_batch_size: int = 1024
    advantage_batch_size: int = 1024
    train_batch_groups: int = 1
    max_train_partitions: int | None = None
    save_name_prefix: str = 'async-rl-sampler-weights'
    adapter_checkpoint_dir: str | None = None
    save_optimizer: bool = False
    is_sampler_checkpoint: bool = True
    max_grad_norm: float = 1.0
    norm_type: int = 2
    train_kwargs: dict[str, Any] = field(default_factory=dict)
    tq_config: TransferQueueRuntimeConfig | None = None


class BaseRLPipeline:
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
        self.adapter_registry = None
        self.staleness_manager = None
        self.tool_manager_factory = None
        self.advantage_fn = None
        self.rollout_policy = None
        self.train_policy = None
        self.train_batch_fn = None
        self.receive_weights_fn = None
        self._sync_step_loop = None

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

    def build_model(self) -> Any:
        """Build the train-side model resource.

        Subclasses should override this for YAML-driven jobs.
        """
        return None

    def build_rollout(self) -> RolloutCallable | None:
        """Build the rollout implementation resource.

        Subclasses should override this for YAML-driven jobs.
        """
        return None

    def build_data_plane(self) -> TransferQueueDataPlane:
        """Build the TransferQueue data plane resource."""
        return TransferQueueDataPlane(tq_config=self.config.tq_config)

    def build_reward_registry(self) -> dict[str, Callable[..., list[float]]]:
        return {}

    def build_advantage_fn(
        self, ) -> Callable[[list[SampleRecord], LoraContext], tuple[list[float], list[float]]] | None:
        return None

    def build_tool_manager_factory(self) -> ToolManagerFactory | None:
        return None

    def build_rollout_policy(self) -> Any | None:
        return None

    def build_train_policy(self) -> Any | None:
        return None

    def build_train_batch_fn(
        self, ) -> Callable[[LoraContext, str, Any], TrainerStepResult | dict[str, Any] | None] | None:
        return None

    def build_save_adapter_fn(self, ) -> Callable[[LoraContext, str], TrainerStepResult | dict[str, Any] | None] | None:
        return None

    def build_receive_weights_fn(self) -> Callable[[LoraContext], None] | None:
        return None

    def build_prompt_loaders(self) -> list[PromptLoader]:
        """Build rollout-side prompt loaders.

        Subclasses can wrap `twinkle.dataloader.DataLoader` instances here.
        The default is empty so callers may still push prompts explicitly with
        `submit_rollout_samples()`.
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

        if self.data_plane is None:
            self.data_plane = self.build_data_plane()
        if self.adapter_registry is None:
            self.adapter_registry = LoraAdapterRegistry()
        if self.staleness_manager is None:
            self.staleness_manager = StalenessManager(
                max_staleness=config.max_staleness,
                target_groups_per_partition=config.target_groups_per_partition,
            )
        for context in self.contexts:
            self.adapter_registry.register(context)

    def create_roles(self) -> None:
        """Create runtime roles for the default GRPO pipeline."""
        algorithm = self.config.algorithm.lower()
        if algorithm != 'grpo':
            raise NotImplementedError(f'BaseRLPipeline only defines default roles for algorithm={algorithm!r}. '
                                      'Override create_roles() in an algorithm-specific pipeline.')
        self.create_grpo_roles()

    def create_grpo_roles(self) -> None:
        """Create the default Multi-LoRA GRPO component graph."""
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
        self.reward_worker = self.build_reward_worker()
        self.advantage_worker = self.build_advantage_worker(advantage_fn=self.advantage_fn)
        self.trainer_scheduler = self.build_trainer_scheduler(train_policy=self.train_policy)
        self.trainer_worker = self.build_trainer_worker()
        self.prompt_loaders = self.build_prompt_loaders()
        self.components = self.build_pipeline_components()

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
            adapter_registry=self.adapter_registry,
            staleness_manager=self.staleness_manager,
            rollout=self.rollout,
            tool_manager_factory=tool_manager_factory,
            rollout_policy=rollout_policy,
            max_concurrent_groups=config.max_concurrent_groups,
            target_groups_per_partition=config.target_groups_per_partition,
        )

    def build_reward_worker(self) -> RewardWorker:
        return RewardWorker(
            data_plane=self.data_plane,
            reward_registry=self.reward_registry,
            contexts=self.contexts,
            batch_size=self.config.reward_batch_size,
        )

    def build_advantage_worker(
        self,
        *,
        advantage_fn: Callable[[list[SampleRecord], LoraContext], tuple[list[float], list[float]]] | None,
    ) -> AdvantageWorker:
        return AdvantageWorker(
            data_plane=self.data_plane,
            contexts=self.contexts,
            batch_size=self.config.advantage_batch_size,
            advantage_fn=advantage_fn,
        )

    def build_trainer_scheduler(self, *, train_policy: Any | None) -> TrainerScheduler:
        return TrainerScheduler(adapter_registry=self.adapter_registry, train_policy=train_policy)

    def build_trainer_worker(self) -> TrainerWorker:
        if self.train_batch_fn is not None:
            return TrainerWorker(
                data_plane=self.data_plane,
                adapter_registry=self.adapter_registry,
                scheduler=self.trainer_scheduler,
                train_batch_fn=self.train_batch_fn,
                save_adapter_fn=self.save_adapter_fn,
                receive_weights_fn=self.receive_weights_fn,
                train_batch_groups=self.config.train_batch_groups,
            )

        if self.__class__.train_batch is not BaseRLPipeline.train_batch:
            return TrainerWorker(
                data_plane=self.data_plane,
                adapter_registry=self.adapter_registry,
                scheduler=self.trainer_scheduler,
                train_batch_fn=self.train_batch,
                save_adapter_fn=self.save_adapter_fn,
                receive_weights_fn=self.receive_weights_fn,
                train_batch_groups=self.config.train_batch_groups,
            )

        return MultiLoraGRPOTrainerWorker(
            data_plane=self.data_plane,
            adapter_registry=self.adapter_registry,
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
            train_batch_groups=self.config.train_batch_groups,
        )

    def build_pipeline_components(self) -> list[Any]:
        """Return the component graph run by this pipeline.

        Algorithm-specific pipelines should override this if their roles are
        not the default GRPO chain. The default graph is:

        PromptLoader -> AsyncRollouter -> RewardWorker -> AdvantageWorker -> TrainerWorker
        """
        return [
            *self.prompt_loaders,
            self.rollouter,
            self.reward_worker,
            self.advantage_worker,
            self.trainer_worker,
        ]

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

    def add_pending(self, samples: Iterable[SampleRecord], context: LoraContext | None = None) -> None:
        self.submit_rollout_samples(samples, context=context)

    def submit_rollout_samples(self, samples: Iterable[SampleRecord], context: LoraContext | None = None) -> None:
        """Submit prompt groups for rollout.

        These are not trainer batches. The trainer only reads samples that have
        already passed rollout/reward/advantage stages from TransferQueue.
        """
        self.rollouter.enqueue_prompt_groups(context or self.current_context(), samples)

    async def step_async(self) -> dict[str, PartitionMetadata | None]:
        step_result = {'rollout': None, 'reward': None, 'advantage': None, 'train': None}
        self._last_step_had_work = False
        for component in self.components:
            result = component.step()
            if asyncio.iscoroutine(result):
                result = await result
            if result is None:
                continue
            self._last_step_had_work = True
            if result.kind in step_result:
                step_result[result.kind] = result.metadata
        return step_result

    def step(self) -> dict[str, PartitionMetadata | None]:
        if self._sync_step_loop is None or self._sync_step_loop.is_closed():
            self._sync_step_loop = asyncio.new_event_loop()
        return self._sync_step_loop.run_until_complete(self.step_async())

    async def run_async(
        self,
        rollout_samples: Iterable[SampleRecord] | None = None,
        *,
        max_steps: int | None = None,
    ) -> list[dict[str, PartitionMetadata | None]]:
        if rollout_samples is not None:
            self.submit_rollout_samples(rollout_samples)
        limit = max_steps if max_steps is not None else self.config.max_train_partitions
        history: list[dict[str, PartitionMetadata | None]] = []
        trained = 0
        idle_steps = 0
        while limit is None or trained < limit:
            result = await self.step_async()
            history.append(result)
            if result['train'] is not None:
                trained += 1
                idle_steps = 0
                if self.should_stop(trained):
                    break
            elif self._last_step_had_work or any(value is not None for value in result.values()):
                idle_steps = 0
            else:
                idle_steps += 1
                if self._is_drained():
                    break
                await asyncio.sleep(0)
        return history

    def run(
        self,
        rollout_samples: Iterable[SampleRecord] | None = None,
        *,
        max_steps: int | None = None,
    ) -> list[dict[str, PartitionMetadata | None]]:
        """Drive the async RL loop.

        `rollout_samples` is an optional convenience feed for rollout prompts.
        Training batches are always read from TransferQueue by TrainerWorker.
        """
        return asyncio.run(self.run_async(rollout_samples, max_steps=max_steps))

    def run_until_idle(self, *, max_steps: int | None = None) -> list[dict[str, PartitionMetadata | None]]:
        """Advance workers without adding new rollout prompts."""
        return self.run(max_steps=max_steps)

    def sync_and_clear_completed_partitions(self, metadata: PartitionMetadata) -> None:
        """Hook for custom pipelines after a train_k is completed.

        The MVP `TrainerWorker` performs adapter save/version update and clear
        inline. Custom pipelines can override this method together with a
        custom trainer worker if they need a different sync boundary.
        """
        self.data_plane.clear_partition(metadata.context, metadata.partition_id)

    def should_stop(self, trained_partitions: int) -> bool:
        return self.config.max_train_partitions is not None and trained_partitions >= self.config.max_train_partitions

    def shutdown(self) -> None:
        for component in getattr(self, 'components', []):
            shutdown = getattr(component, 'shutdown', None)
            if shutdown is not None:
                shutdown()
        if self._sync_step_loop is not None and not self._sync_step_loop.is_closed():
            self._sync_step_loop.run_until_complete(asyncio.sleep(0))
            self._sync_step_loop.close()
        close = getattr(getattr(self, 'data_plane', None), 'close', None)
        if close is not None:
            close()

    def train_batch(self, context: LoraContext, partition_id: str, dataloader: Any) -> TrainerStepResult:
        """Override hook for one train batch.

        The default GRPO train path is implemented by
        `MultiLoraGRPOTrainerWorker`. New algorithms should prefer overriding
        `build_trainer_worker()` or `build_pipeline_components()`.
        """
        raise NotImplementedError('BaseRLPipeline.train_batch is not implemented. '
                                  'Use MultiLoraGRPOTrainerWorker or override build_trainer_worker().')

    def current_context(self, context: LoraContext | None = None) -> LoraContext:
        base_context = context or self.context
        record = self.adapter_registry.get(base_context)
        return base_context.with_policy_version(record.policy_version, record.adapter_path)

    def current_contexts(self) -> list[LoraContext]:
        return [self.current_context(context) for context in self.contexts]

    def _is_drained(self) -> bool:
        if any(not component.is_idle() for component in self.components):
            return False
        if not hasattr(self, 'rollouter'):
            return True
        for context in self.current_contexts():
            if self.rollouter.pending_prompt_group_count(context) > 0:
                return False
            active_partitions = [
                partition for partition in self.data_plane.get_metadata(context)
                if partition.status.value not in {'CLEARED', 'FAILED', 'CANCELLED'}
            ]
            if active_partitions:
                return False
        return True
