# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import asyncio
import inspect
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional

from twinkle_agentic.tools.tool_manager import ToolManager
from .data_plane import TransferQueueDataPlane
from .registry import AdapterRegistry
from .scheduler import TrainerScheduler
from .rollout_scheduling import WorkConservingRolloutPolicy
from .staleness import StalenessManager
from .types import (ComponentResult, PartitionMetadata, PartitionStatus, RolloutCallable, RolloutContextState,
                    SampleRecord, TrainingContext)


class ToolManagerFactory:
    """Create context-scoped ToolManager instances.

    Profiles are callables so deployments can attach native or remote tools
    without importing untrusted user code in the server process.
    """

    def __init__(self, profiles: dict[str, Callable[[TrainingContext, SampleRecord], ToolManager]] | None = None):
        self._profiles = dict(profiles or {})

    def register(self, profile: str, factory: Callable[[TrainingContext, SampleRecord], ToolManager]) -> None:
        self._profiles[profile] = factory

    def create(self, sample: SampleRecord, context: TrainingContext) -> ToolManager:
        factory = self._profiles.get(context.tool_profile)
        if factory is None:
            return ToolManager()
        return factory(context, sample)


@dataclass(frozen=True)
class RolloutTaskState:
    context: TrainingContext
    prompt_groups: list[SampleRecord]
    group_count: int
    submitted_at: float


class AsyncRollouter:
    """Schedule prompt groups, run rollout, and append results to train partitions."""

    def __init__(
        self,
        *,
        data_plane: TransferQueueDataPlane,
        adapter_registry: AdapterRegistry,
        staleness_manager: StalenessManager,
        rollout: RolloutCallable,
        tool_manager_factory: ToolManagerFactory | None = None,
        rollout_policy: Any | None = None,
        max_concurrent_groups: int = 16,
        target_groups_per_partition: int = 1,
        max_submit_groups: int | None = None,
    ):
        self.data_plane = data_plane
        self.adapter_registry = adapter_registry
        self.staleness_manager = staleness_manager
        self.rollout: RolloutCallable = rollout
        self.tool_manager_factory = tool_manager_factory or ToolManagerFactory()
        self.rollout_policy = rollout_policy or WorkConservingRolloutPolicy()
        self.max_concurrent_groups = max_concurrent_groups
        self.target_groups_per_partition = target_groups_per_partition
        self.max_submit_groups = max_submit_groups or target_groups_per_partition
        self.pending_prompt_groups_by_context: dict[str, Deque[tuple[TrainingContext,
                                                                     SampleRecord]]] = defaultdict(deque)
        self.active_rollout_tasks: dict[asyncio.Task, RolloutTaskState] = {}
        self.finished_rollout_tasks: Deque[asyncio.Task] = deque()
        self.completed_rollout_results: Deque[ComponentResult] = deque()
        self.active_prompt_groups_by_context: dict[str, int] = defaultdict(int)
        self.active_prompt_group_count = 0
        self._last_rollout_submit_time: dict[str, float] = defaultdict(float)
        self._submitted_prompt_groups: dict[str, int] = defaultdict(int)

    def enqueue_prompt_groups(self, context: TrainingContext, prompt_groups: Iterable[SampleRecord]) -> None:
        """Append rollout inputs for a context.

        A prompt group is the scheduling unit for rollout. It may produce one
        or more trajectories depending on the rollout implementation.
        """
        self.adapter_registry.register(context)
        self.data_plane.init_namespace(context)
        queue = self.pending_prompt_groups_by_context[context.key]
        for prompt_group in prompt_groups:
            queue.append((context, prompt_group))

    def pending_prompt_group_count(self, context: TrainingContext) -> int:
        return len(self.pending_prompt_groups_by_context.get(context.key, ()))

    def add_pending(self, context: TrainingContext, samples: Iterable[SampleRecord]) -> None:
        """Backward-compatible alias for `enqueue_prompt_groups`."""
        self.enqueue_prompt_groups(context, samples)

    def build_rollout_state(self, context: TrainingContext) -> RolloutContextState | None:
        """Collect current queue, staleness, partition, and adapter state for scheduling."""
        pending_groups = len(self.pending_prompt_groups_by_context.get(context.key, ()))
        if pending_groups <= 0:
            return None
        partitions = self.data_plane.get_metadata(context)
        record = self.adapter_registry.get(context)
        context = context.with_policy_version(record.policy_version, record.adapter_revision)
        capacity = self.staleness_manager.get_rollout_capacity(context, partitions)
        open_partitions = [p for p in partitions if p.status == PartitionStatus.OPEN]
        train_ready = [p for p in partitions if p.status == PartitionStatus.TRAIN_READY]
        return RolloutContextState(
            context=context,
            pending_groups=pending_groups,
            in_flight_rollouts=record.in_flight_rollouts,
            live_partitions=len([p for p in partitions if p.status != PartitionStatus.CLEARED]),
            open_partitions=len(open_partitions),
            train_ready_partitions=len(train_ready),
            rollout_capacity=capacity.available_groups,
            last_submit_time=self._last_rollout_submit_time[context.key],
            submitted_groups=self._submitted_prompt_groups[context.key],
            weight=record.weight,
        )

    def pick_next_rollout_context(self) -> TrainingContext | None:
        """Choose the next context that is allowed to submit one prompt group."""
        states: list[RolloutContextState] = []
        if self.remaining_group_capacity() <= 0:
            return None
        seen: dict[str, TrainingContext] = {}
        for queue in self.pending_prompt_groups_by_context.values():
            if not queue:
                continue
            context = queue[0][0]
            seen[context.key] = context
        for context in seen.values():
            if not self.adapter_registry.can_accept_rollout(context):
                continue
            if not self.data_plane.check_capacity(context):
                continue
            state = self.build_rollout_state(context)
            active_groups = self.active_prompt_groups_by_context[context.key]
            if state is None or state.rollout_capacity <= active_groups:
                continue
            state.rollout_capacity = max(0, state.rollout_capacity - active_groups)
            states.append(state)
        return self.rollout_policy.pick_next_context(states)

    def pick_next_training_context(self) -> TrainingContext | None:
        """Backward-compatible alias for `pick_next_rollout_context`."""
        return self.pick_next_rollout_context()

    def get_submit_group_count(self, context: TrainingContext) -> int:
        """Return how many prompt groups can be submitted for this context now."""
        queue = self.pending_prompt_groups_by_context.get(context.key)
        if not queue:
            return 0
        state = self.build_rollout_state(context)
        if state is None or state.rollout_capacity <= 0:
            return 0
        active_groups = self.active_prompt_groups_by_context[context.key]
        context_capacity = max(0, state.rollout_capacity - active_groups)
        return max(0, min(len(queue), context_capacity, self.max_submit_groups, self.remaining_group_capacity()))

    def remaining_group_capacity(self) -> int:
        return max(0, self.max_concurrent_groups - self.active_prompt_group_count)

    def pop_prompt_groups(self, context: TrainingContext, count: int) -> list[SampleRecord]:
        queue = self.pending_prompt_groups_by_context[context.key]
        prompt_groups = []
        for _ in range(min(count, len(queue))):
            _, prompt_group = queue.popleft()
            prompt_groups.append(prompt_group)
        return prompt_groups

    async def rollout_prompt_groups(
        self,
        context: TrainingContext,
        prompt_groups: list[SampleRecord],
    ) -> PartitionMetadata:
        """Run rollout for a homogeneous prompt-group batch and append it to train_k."""
        if not prompt_groups:
            raise ValueError('rollout_prompt_groups requires at least one prompt group')

        tool_manager = self.tool_manager_factory.create(prompt_groups[0], context)
        trajectories = [prompt_group.get('trajectory') or prompt_group for prompt_group in prompt_groups]
        self.adapter_registry.on_rollout_started(context)
        try:
            rollout_kwargs = {'tool_manager': tool_manager, 'adapter_name': context.adapter_name}
            if context.adapter_revision is not None:
                rollout_kwargs['adapter_path'] = context.adapter_revision
            result = await self.invoke_rollout(trajectories, rollout_kwargs)
            if inspect.isawaitable(result):
                result = await result
            rollout_rows = list(result)
            partition_id = self.get_or_create_open_partition(context)
            meta = self.data_plane.put_rollout_batch(
                context,
                partition_id,
                rollout_rows,
                ready_groups=len(prompt_groups),
                seal=False,
            )
            self.adapter_registry.on_partition_created(context, partition_id)
            self._last_rollout_submit_time[context.key] = time.time()
            self._submitted_prompt_groups[context.key] += len(prompt_groups)
            return meta
        finally:
            self.adapter_registry.on_rollout_finished(context)

    async def rollout_prompt_group(self, context: TrainingContext, prompt_group: SampleRecord) -> PartitionMetadata:
        """Backward-compatible single-group wrapper around `rollout_prompt_groups`."""
        return await self.rollout_prompt_groups(context, [prompt_group])

    async def run_one_group(self, context: TrainingContext, sample: SampleRecord) -> PartitionMetadata:
        """Backward-compatible alias for `rollout_prompt_group`."""
        return await self.rollout_prompt_group(context, sample)

    async def invoke_rollout(self, trajectories: list[Any], rollout_kwargs: dict[str, Any]):
        """Call sync rollout implementations without blocking the event loop."""
        call = getattr(self.rollout, '__call__', None)
        if inspect.iscoroutinefunction(self.rollout) or inspect.iscoroutinefunction(call):
            return await self.rollout(trajectories, **rollout_kwargs)
        return await asyncio.to_thread(self.rollout, trajectories, **rollout_kwargs)

    async def run_rollout_task(
        self,
        context: TrainingContext,
        prompt_groups: list[SampleRecord],
    ) -> ComponentResult:
        meta = await self.rollout_prompt_groups(context, prompt_groups)
        return ComponentResult(component='rollouter', kind='rollout', metadata=meta, count=len(prompt_groups))

    def on_rollout_task_done(self, task: asyncio.Task) -> None:
        self.finished_rollout_tasks.append(task)

    def submit_rollout_tasks(self) -> int:
        submitted_groups = 0
        while self.remaining_group_capacity() > 0:
            context = self.pick_next_rollout_context()
            if context is None:
                break
            submit_count = self.get_submit_group_count(context)
            if submit_count <= 0:
                break
            prompt_groups = self.pop_prompt_groups(context, submit_count)
            task = asyncio.create_task(self.run_rollout_task(context, prompt_groups))
            self.active_rollout_tasks[task] = RolloutTaskState(
                context=context,
                prompt_groups=prompt_groups,
                group_count=len(prompt_groups),
                submitted_at=time.time(),
            )
            self.active_prompt_group_count += len(prompt_groups)
            self.active_prompt_groups_by_context[context.key] += len(prompt_groups)
            task.add_done_callback(self.on_rollout_task_done)
            submitted_groups += len(prompt_groups)
        return submitted_groups

    def collect_finished_rollout_tasks(self) -> None:
        while self.finished_rollout_tasks:
            task = self.finished_rollout_tasks.popleft()
            state = self.active_rollout_tasks.pop(task, None)
            if state is not None:
                self.active_prompt_group_count = max(0, self.active_prompt_group_count - state.group_count)
                context_key = state.context.key
                self.active_prompt_groups_by_context[context_key] = max(
                    0,
                    self.active_prompt_groups_by_context[context_key] - state.group_count,
                )
                if self.active_prompt_groups_by_context[context_key] == 0:
                    self.active_prompt_groups_by_context.pop(context_key, None)
            if task.cancelled():
                continue
            try:
                result = task.result()
            except Exception as exc:
                if state is not None:
                    self.adapter_registry.mark_failed(state.context, str(exc))
                raise
            self.completed_rollout_results.append(result)

    def get_or_create_open_partition(self, context: TrainingContext) -> str:
        """Return the context's open train partition, or create the next train_k."""
        open_partitions = self.data_plane.list_partitions(context, statuses=[PartitionStatus.OPEN])
        if open_partitions:
            return open_partitions[0].partition_id
        meta = self.data_plane.create_partition(context, target_groups=self.target_groups_per_partition)
        return meta.partition_id

    async def step(self) -> ComponentResult | None:
        self.collect_finished_rollout_tasks()
        submitted_groups = self.submit_rollout_tasks()
        if self.completed_rollout_results:
            return self.completed_rollout_results.popleft()
        if submitted_groups:
            return ComponentResult(component='rollouter', kind='rollout_submit', count=submitted_groups)
        return None

    def is_idle(self) -> bool:
        return (not any(self.pending_prompt_groups_by_context.values()) and not self.active_rollout_tasks
                and not self.finished_rollout_tasks and not self.completed_rollout_results)

    def shutdown(self) -> None:
        for task in list(self.active_rollout_tasks):
            task.cancel()


class RewardWorker:

    def __init__(
        self,
        *,
        data_plane: TransferQueueDataPlane,
        reward_registry: dict[str, Callable[..., list[float]]],
        contexts: list[TrainingContext] | None = None,
        batch_size: int = 1024,
    ):
        self.data_plane = data_plane
        self.reward_registry = reward_registry
        self.contexts = list(contexts or [])
        self.batch_size = batch_size

    def run_once(self, context: TrainingContext, *, batch_size: int = 1024) -> PartitionMetadata:
        meta, samples = self.data_plane.claim_reward_batch(context, batch_size)
        reward_fn = self.reward_registry.get(context.reward_type)
        if reward_fn is None:
            raise KeyError(f'unknown reward_type: {context.reward_type}')
        trajectories = [s.get('trajectory', s) for s in samples]
        rewards = list(reward_fn(trajectories, context=context))
        return self.data_plane.append_rewards(context, meta.partition_id, rewards)

    def step(self) -> ComponentResult | None:
        for context in self.contexts:
            try:
                meta = self.run_once(context, batch_size=self.batch_size)
                return ComponentResult(component='reward_worker', kind='reward', metadata=meta)
            except LookupError:
                continue
        return None

    def is_idle(self) -> bool:
        return not any(
            self.data_plane.list_partitions(context, statuses=[PartitionStatus.ROLLOUT_DONE])
            for context in self.contexts)

    def shutdown(self) -> None:
        return None


class AdvantageWorker:

    def __init__(
        self,
        *,
        data_plane: TransferQueueDataPlane,
        contexts: list[TrainingContext] | None = None,
        batch_size: int = 1024,
        advantage_fn: Callable[[list[SampleRecord], TrainingContext], tuple[list[float], list[float]]] | None = None,
    ):
        self.data_plane = data_plane
        self.contexts = list(contexts or [])
        self.batch_size = batch_size
        self.advantage_fn = advantage_fn or self._default_advantage_fn

    @staticmethod
    def _default_advantage_fn(samples: list[SampleRecord], context: TrainingContext) -> tuple[list[float], list[float]]:
        rewards = [float(sample.get('rewards', sample.get('reward', 0.0))) for sample in samples]
        if not rewards:
            return [], []
        mean_reward = sum(rewards) / len(rewards)
        advantages = [reward - mean_reward for reward in rewards]
        return advantages, rewards

    def run_once(self, context: TrainingContext, *, batch_size: int = 1024) -> PartitionMetadata:
        meta, samples = self.data_plane.claim_advantage_batch(context, batch_size)
        advantages, returns = self.advantage_fn(samples, context)
        return self.data_plane.append_advantages(context, meta.partition_id, advantages, returns)

    def step(self) -> ComponentResult | None:
        for context in self.contexts:
            try:
                meta = self.run_once(context, batch_size=self.batch_size)
                return ComponentResult(component='advantage_worker', kind='advantage', metadata=meta)
            except LookupError:
                continue
        return None

    def is_idle(self) -> bool:
        return not any(
            self.data_plane.list_partitions(context, statuses=[PartitionStatus.REWARD_DONE])
            for context in self.contexts)

    def shutdown(self) -> None:
        return None


@dataclass
class TrainerStepResult:
    adapter_revision: str | None = None
    metrics: dict[str, Any] | None = None


@dataclass(frozen=True)
class MultiLoraGRPOTrainConfig:
    save_name_prefix: str = 'async-rl-sampler-weights'
    adapter_checkpoint_dir: str | None = None
    save_optimizer: bool = False
    is_sampler_checkpoint: bool = True
    max_grad_norm: float = 1.0
    norm_type: int = 2
    train_kwargs: dict[str, Any] | None = None


class TrainerWorker:

    def __init__(
        self,
        *,
        data_plane: TransferQueueDataPlane,
        adapter_registry: AdapterRegistry,
        scheduler: TrainerScheduler,
        train_partition_fn: Callable[[TrainingContext, str, Any], TrainerStepResult | dict[str, Any] | None],
        receive_weights_fn: Callable[[TrainingContext], None] | None = None,
    ):
        self.data_plane = data_plane
        self.adapter_registry = adapter_registry
        self.scheduler = scheduler
        self.train_partition_fn = train_partition_fn
        self.receive_weights_fn = receive_weights_fn
        self.current_context: TrainingContext | None = None

    def step(self) -> Optional[ComponentResult]:
        partition = self.scheduler.next_partition(current_context=self.current_context)
        if partition is None:
            return None
        context = partition.context
        self.current_context = context
        self.adapter_registry.on_train_started(context, partition.partition_id)
        self.data_plane.mark_training(context, partition.partition_id)
        dataloader = self.data_plane.build_streaming_dataloader(context, partition.partition_id)
        try:
            result = self.train_partition_fn(context, partition.partition_id, dataloader)
            adapter_revision = None
            if isinstance(result, TrainerStepResult):
                adapter_revision = result.adapter_revision
            elif isinstance(result, dict):
                adapter_revision = result.get('adapter_revision')
            self.data_plane.mark_trained(context, partition.partition_id)
            self.adapter_registry.on_train_finished(context, partition.partition_id)
            self.adapter_registry.on_weight_sync_started(context)
            new_context = self.adapter_registry.on_weight_sync_finished(context, adapter_revision=adapter_revision)
            if self.receive_weights_fn is not None:
                self.receive_weights_fn(new_context)
            self.data_plane.clear_partition(context, partition.partition_id)
            self.adapter_registry.on_partition_cleared(context, partition.partition_id)
            return ComponentResult(component='trainer_worker', kind='train', metadata=partition)
        except Exception as exc:
            self.adapter_registry.mark_failed(context, str(exc))
            raise

    def run_once(self) -> PartitionMetadata | None:
        result = self.step()
        return None if result is None else result.metadata

    def is_idle(self) -> bool:
        return not self.data_plane.list_train_ready_partitions()

    def shutdown(self) -> None:
        return None


class MultiLoraGRPOTrainerWorker(TrainerWorker):
    """Trainer component for the default Multi-LoRA GRPO path.

    The algorithm-specific train step lives here instead of in BaseRLPipeline,
    so the pipeline can remain a runtime component orchestrator. Other
    algorithms should provide their own trainer component rather than branching
    inside BaseRLPipeline.
    """

    def __init__(
        self,
        *,
        data_plane: TransferQueueDataPlane,
        adapter_registry: AdapterRegistry,
        scheduler: TrainerScheduler,
        model: Any,
        train_config: MultiLoraGRPOTrainConfig | None = None,
        receive_weights_fn: Callable[[TrainingContext], None] | None = None,
    ):
        self.model = model
        self.train_config = train_config or MultiLoraGRPOTrainConfig()
        super().__init__(
            data_plane=data_plane,
            adapter_registry=adapter_registry,
            scheduler=scheduler,
            train_partition_fn=self.train_partition,
            receive_weights_fn=receive_weights_fn,
        )

    def train_partition(self, context: TrainingContext, partition_id: str, dataloader: Any) -> TrainerStepResult:
        batch = list(dataloader)
        inputs = [sample.get('trajectory', sample) for sample in batch]
        advantages = [sample.get('advantages') for sample in batch if 'advantages' in sample]
        old_logps = [sample.get('old_logps') for sample in batch if 'old_logps' in sample]

        config = self.train_config
        kwargs = dict(config.train_kwargs or {})
        if advantages and len(advantages) == len(inputs):
            kwargs.setdefault('advantages', advantages)
        if old_logps and len(old_logps) == len(inputs):
            kwargs.setdefault('old_logps', old_logps)

        self.model.forward_backward(inputs=inputs, adapter_name=context.adapter_name, **kwargs)
        self.model.clip_grad_and_step(
            adapter_name=context.adapter_name,
            max_grad_norm=config.max_grad_norm,
            norm_type=config.norm_type,
        )
        save_kwargs = {
            'adapter_name': context.adapter_name,
            'save_optimizer': config.save_optimizer,
            'is_sampler': config.is_sampler_checkpoint,
        }
        if config.adapter_checkpoint_dir is not None:
            save_kwargs['output_dir'] = config.adapter_checkpoint_dir
        save_result = self.model.save(
            name=f'{config.save_name_prefix}-{context.training_run_id}-{context.adapter_name}',
            **save_kwargs,
        )
        adapter_revision = getattr(save_result, 'twinkle_path', None)
        if adapter_revision is None and isinstance(save_result, str):
            adapter_revision = save_result
        if adapter_revision is None and isinstance(save_result, dict):
            adapter_revision = save_result.get('twinkle_path') or save_result.get('path')
        return TrainerStepResult(adapter_revision=adapter_revision)
