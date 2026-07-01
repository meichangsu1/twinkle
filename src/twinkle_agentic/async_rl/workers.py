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
from .registry import LoraAdapterRegistry
from .scheduling import PreferCurrentTrainPolicy, WorkConservingRolloutPolicy
from .staleness import StalenessManager
from .types import (GroupBatch, GroupStatus, LoraContext, PartitionMetadata, PartitionStatus, RolloutCallable,
                    RolloutContextState, SampleRecord, WorkerResult)


class ToolManagerFactory:
    """Create context-scoped ToolManager instances.

    Profiles are callables so deployments can attach native or remote tools
    without importing untrusted user code in the server process.
    """

    def __init__(self, profiles: dict[str, Callable[[LoraContext, SampleRecord], ToolManager]] | None = None):
        self._profiles = dict(profiles or {})

    def register(self, profile: str, factory: Callable[[LoraContext, SampleRecord], ToolManager]) -> None:
        self._profiles[profile] = factory

    def create(self, sample: SampleRecord, context: LoraContext) -> ToolManager:
        factory = self._profiles.get(context.tool_profile)
        if factory is None:
            return ToolManager()
        return factory(context, sample)


@dataclass(frozen=True)
class RolloutTaskState:
    context: LoraContext
    prompt_group: SampleRecord
    submitted_at: float


class AsyncRollouter:
    """Schedule prompt groups, run rollout, and append results to train partitions."""

    def __init__(
        self,
        *,
        data_plane: TransferQueueDataPlane,
        adapter_registry: LoraAdapterRegistry,
        staleness_manager: StalenessManager,
        rollout: RolloutCallable,
        tool_manager_factory: ToolManagerFactory | None = None,
        rollout_policy: Any | None = None,
        max_concurrent_groups: int = 16,
        target_groups_per_partition: int = 1,
    ):
        self.data_plane = data_plane
        self.adapter_registry = adapter_registry
        self.staleness_manager = staleness_manager
        self.rollout: RolloutCallable = rollout
        self.tool_manager_factory = tool_manager_factory or ToolManagerFactory()
        self.rollout_policy = rollout_policy or WorkConservingRolloutPolicy()
        self.max_concurrent_groups = max_concurrent_groups
        self.target_groups_per_partition = target_groups_per_partition
        self.pending_prompt_groups_by_context: dict[str, Deque[tuple[LoraContext, SampleRecord]]] = defaultdict(deque)
        self.active_rollout_tasks: dict[asyncio.Task, RolloutTaskState] = {}
        self.finished_rollout_tasks: Deque[asyncio.Task] = deque()
        self.completed_rollout_results: Deque[WorkerResult] = deque()
        self.active_prompt_groups_by_context: dict[str, int] = defaultdict(int)
        self.active_prompt_group_count = 0
        self._last_rollout_submit_time: dict[str, float] = defaultdict(float)
        self._submitted_prompt_groups: dict[str, int] = defaultdict(int)

    def enqueue_prompt_groups(self, context: LoraContext, prompt_groups: Iterable[SampleRecord]) -> None:
        """Append rollout inputs for a context.

        A prompt group is the scheduling unit for rollout. It may produce one
        or more trajectories depending on the rollout implementation.
        """
        self.adapter_registry.register(context)
        queue = self.pending_prompt_groups_by_context[context.key]
        for prompt_group in prompt_groups:
            queue.append((context, prompt_group))

    def pending_prompt_group_count(self, context: LoraContext) -> int:
        return len(self.pending_prompt_groups_by_context.get(context.key, ()))

    def add_pending(self, context: LoraContext, samples: Iterable[SampleRecord]) -> None:
        """Backward-compatible alias for `enqueue_prompt_groups`."""
        self.enqueue_prompt_groups(context, samples)

    def build_rollout_state(self, context: LoraContext) -> RolloutContextState | None:
        """Collect current queue, staleness, partition, and adapter state for scheduling."""
        pending_groups = len(self.pending_prompt_groups_by_context.get(context.key, ()))
        if pending_groups <= 0:
            return None
        partitions = self.data_plane.get_metadata(context)
        record = self.adapter_registry.get(context)
        context = context.with_policy_version(record.policy_version, record.adapter_path)
        groups = self.data_plane.list_groups(context)
        capacity = self.staleness_manager.get_rollout_capacity(context, partitions, groups)
        open_partitions = [p for p in partitions if p.status == PartitionStatus.OPEN]
        train_ready = self.data_plane.list_train_ready_partitions()
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

    def pick_next_rollout_context(self) -> LoraContext | None:
        """Choose the next context that is allowed to submit one prompt group."""
        states: list[RolloutContextState] = []
        if self.remaining_group_capacity() <= 0:
            return None
        seen: dict[str, LoraContext] = {}
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

    def remaining_group_capacity(self) -> int:
        return max(0, self.max_concurrent_groups - self.active_prompt_group_count)

    def pop_prompt_group(self, context: LoraContext) -> SampleRecord:
        queue = self.pending_prompt_groups_by_context[context.key]
        _, prompt_group = queue.popleft()
        return prompt_group

    async def rollout_prompt_group(self, context: LoraContext, prompt_group: SampleRecord) -> PartitionMetadata:
        """Run rollout for one prompt group and append its generated samples to train_k."""
        tool_manager = self.tool_manager_factory.create(prompt_group, context)
        trajectory = prompt_group.get('trajectory') or prompt_group
        self.adapter_registry.on_rollout_started(context)
        try:
            rollout_kwargs = {'tool_manager': tool_manager, 'adapter_name': context.adapter_name}
            if context.adapter_path is not None:
                rollout_kwargs['adapter_path'] = context.adapter_path
            result = await self.invoke_rollout([trajectory], rollout_kwargs)
            if inspect.isawaitable(result):
                result = await result
            rollout_rows = list(result)
            partition_id = self.get_or_create_open_partition(context)
            meta = self.data_plane.put_rollout_groups(
                context,
                partition_id,
                rollout_rows,
            )
            self.adapter_registry.on_partition_created(context, partition_id)
            self._last_rollout_submit_time[context.key] = time.time()
            self._submitted_prompt_groups[context.key] += 1
            return meta
        finally:
            self.adapter_registry.on_rollout_finished(context)

    async def invoke_rollout(self, trajectories: list[Any], rollout_kwargs: dict[str, Any]):
        """Call sync rollout implementations without blocking the event loop."""
        call = getattr(self.rollout, '__call__', None)
        if inspect.iscoroutinefunction(self.rollout) or inspect.iscoroutinefunction(call):
            return await self.rollout(trajectories, **rollout_kwargs)
        return await asyncio.to_thread(self.rollout, trajectories, **rollout_kwargs)

    def on_rollout_task_done(self, task: asyncio.Task) -> None:
        self.finished_rollout_tasks.append(task)

    def submit_rollout_tasks(self) -> int:
        submitted_groups = 0
        while self.remaining_group_capacity() > 0:
            context = self.pick_next_rollout_context()
            if context is None:
                break
            prompt_group = self.pop_prompt_group(context)
            task = asyncio.create_task(self.rollout_prompt_group(context, prompt_group))
            self.active_rollout_tasks[task] = RolloutTaskState(
                context=context,
                prompt_group=prompt_group,
                submitted_at=time.time(),
            )
            self.active_prompt_group_count += 1
            self.active_prompt_groups_by_context[context.key] += 1
            task.add_done_callback(self.on_rollout_task_done)
            submitted_groups += 1
        return submitted_groups

    def collect_finished_rollout_tasks(self) -> None:
        while self.finished_rollout_tasks:
            task = self.finished_rollout_tasks.popleft()
            state = self.active_rollout_tasks.pop(task, None)
            if state is not None:
                self.active_prompt_group_count = max(0, self.active_prompt_group_count - 1)
                context_key = state.context.key
                self.active_prompt_groups_by_context[context_key] = max(
                    0,
                    self.active_prompt_groups_by_context[context_key] - 1,
                )
                if self.active_prompt_groups_by_context[context_key] == 0:
                    self.active_prompt_groups_by_context.pop(context_key, None)
            if task.cancelled():
                continue
            try:
                meta = task.result()
            except Exception as exc:
                if state is not None:
                    self.adapter_registry.mark_failed(state.context, str(exc))
                raise
            self.completed_rollout_results.append(
                WorkerResult(component='rollouter', kind='rollout', metadata=meta, count=1))

    def get_or_create_open_partition(self, context: LoraContext) -> str:
        """Return the context's open train partition, or create the next train_k."""
        open_partitions = self.data_plane.list_partitions(context, statuses=[PartitionStatus.OPEN])
        if open_partitions:
            return open_partitions[0].partition_id
        meta = self.data_plane.create_partition(context, target_groups=self.target_groups_per_partition)
        return meta.partition_id

    async def step(self) -> WorkerResult | None:
        self.collect_finished_rollout_tasks()
        submitted_groups = self.submit_rollout_tasks()
        if self.completed_rollout_results:
            return self.completed_rollout_results.popleft()
        if submitted_groups:
            return WorkerResult(component='rollouter', kind='rollout_submit', count=submitted_groups)
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
        contexts: list[LoraContext] | None = None,
        batch_size: int = 1024,
    ):
        self.data_plane = data_plane
        self.reward_registry = reward_registry
        self.contexts = list(contexts or [])
        self.batch_size = batch_size

    def process_reward_batch(self, context: LoraContext, *, batch_size: int = 1024) -> PartitionMetadata:
        group_batch = self.data_plane.claim_reward_groups(context, batch_size)
        reward_fn = self.reward_registry.get(context.reward_type)
        if reward_fn is None:
            raise KeyError(f'unknown reward_type: {context.reward_type}')
        trajectories = [s.get('trajectory', s) for s in group_batch.samples]
        rewards = list(reward_fn(trajectories, context=context))
        return self.data_plane.append_group_rewards(context, group_batch, rewards)

    def step(self) -> WorkerResult | None:
        for context in self.contexts:
            try:
                meta = self.process_reward_batch(context, batch_size=self.batch_size)
                return WorkerResult(component='reward_worker', kind='reward', metadata=meta)
            except LookupError:
                continue
        return None

    def is_idle(self) -> bool:
        return not any(
            self.data_plane.list_groups(context, statuses=[GroupStatus.ROLLOUT_DONE]) for context in self.contexts)

    def shutdown(self) -> None:
        return None


class AdvantageWorker:

    def __init__(
        self,
        *,
        data_plane: TransferQueueDataPlane,
        contexts: list[LoraContext] | None = None,
        batch_size: int = 1024,
        advantage_fn: Callable[[list[SampleRecord], LoraContext], tuple[list[float], list[float]]] | None = None,
    ):
        self.data_plane = data_plane
        self.contexts = list(contexts or [])
        self.batch_size = batch_size
        self.advantage_fn = advantage_fn or self._default_advantage_fn

    @staticmethod
    def _default_advantage_fn(samples: list[SampleRecord], context: LoraContext) -> tuple[list[float], list[float]]:
        rewards = [float(sample.get('rewards', sample.get('reward', 0.0))) for sample in samples]
        if not rewards:
            return [], []
        mean_reward = sum(rewards) / len(rewards)
        advantages = [reward - mean_reward for reward in rewards]
        return advantages, rewards

    def process_advantage_batch(self, context: LoraContext, *, batch_size: int = 1024) -> PartitionMetadata:
        group_batch = self.data_plane.claim_advantage_groups(context, batch_size)
        advantages, returns = self.advantage_fn(group_batch.samples, context)
        return self.data_plane.append_group_advantages(context, group_batch, advantages, returns)

    def step(self) -> WorkerResult | None:
        for context in self.contexts:
            try:
                meta = self.process_advantage_batch(context, batch_size=self.batch_size)
                return WorkerResult(component='advantage_worker', kind='advantage', metadata=meta)
            except LookupError:
                continue
        return None

    def is_idle(self) -> bool:
        return not any(
            self.data_plane.list_groups(context, statuses=[GroupStatus.REWARD_DONE]) for context in self.contexts)

    def shutdown(self) -> None:
        return None


class TrainerScheduler:

    def __init__(self, *, adapter_registry: LoraAdapterRegistry, train_policy: Any | None = None):
        self.adapter_registry = adapter_registry
        self.train_policy = train_policy or PreferCurrentTrainPolicy()

    def next_partition(
        self,
        candidates: list[PartitionMetadata],
        current_context: LoraContext | None = None,
    ) -> PartitionMetadata | None:
        filtered = []
        for partition in candidates:
            if not self.adapter_registry.can_train(partition.context):
                continue
            filtered.append(partition)
        return self.train_policy.pick_next_partition(filtered, current_context)


@dataclass
class TrainerStepResult:
    adapter_path: str | None = None
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
        adapter_registry: LoraAdapterRegistry,
        scheduler: TrainerScheduler,
        train_batch_fn: Callable[[LoraContext, str, Any], TrainerStepResult | dict[str, Any] | None],
        save_adapter_fn: Callable[[LoraContext, str], TrainerStepResult | dict[str, Any] | None] | None = None,
        receive_weights_fn: Callable[[LoraContext], None] | None = None,
        train_batch_groups: int = 1,
    ):
        self.data_plane = data_plane
        self.adapter_registry = adapter_registry
        self.scheduler = scheduler
        self.train_batch_fn = train_batch_fn
        self.save_adapter_fn = save_adapter_fn
        self.receive_weights_fn = receive_weights_fn
        self.train_batch_groups = train_batch_groups
        self.current_context: LoraContext | None = None

    def step(self) -> WorkerResult | None:
        partition = self.scheduler.next_partition(
            self.data_plane.list_train_ready_partitions(min_groups=self.train_batch_groups),
            self.current_context,
        )
        if partition is None:
            return None
        context = partition.context
        self.current_context = context
        self.adapter_registry.on_train_started(context, partition.partition_id)
        try:
            group_batch = self.data_plane.claim_train_groups(context, partition.partition_id, self.train_batch_groups)
            dataloader = group_batch.samples
            result = self.train_batch_fn(context, partition.partition_id, dataloader)
            self.data_plane.mark_groups_train_done(context, group_batch)
            if not self.data_plane.is_partition_train_done(context, partition.partition_id):
                self.adapter_registry.on_train_finished(context, partition.partition_id)
                return WorkerResult(
                    component='trainer_worker',
                    kind='train_batch',
                    metadata=partition,
                    count=group_batch.group_count,
                )

            sync_result = self.save_adapter_fn(context, partition.partition_id) if self.save_adapter_fn else result
            adapter_path = self._adapter_path_from_result(sync_result)
            meta = self.data_plane.mark_trained(context, partition.partition_id)
            self.adapter_registry.on_train_finished(context, partition.partition_id)
            self.adapter_registry.on_weight_sync_started(context)
            new_context = self.adapter_registry.on_weight_sync_finished(context, adapter_path=adapter_path)
            if self.receive_weights_fn is not None:
                self.receive_weights_fn(new_context)
            self.data_plane.clear_partition(context, partition.partition_id)
            self.adapter_registry.on_partition_cleared(context, partition.partition_id)
            return WorkerResult(
                component='trainer_worker',
                kind='train',
                metadata=meta,
                count=group_batch.group_count,
            )
        except Exception as exc:
            self.adapter_registry.on_train_finished(context, partition.partition_id)
            self.adapter_registry.mark_failed(context, str(exc))
            raise

    def train_next_batch(self) -> PartitionMetadata | None:
        result = self.step()
        return None if result is None else result.metadata

    def is_idle(self) -> bool:
        return not self.data_plane.list_train_ready_partitions(min_groups=self.train_batch_groups)

    def shutdown(self) -> None:
        return None

    @staticmethod
    def _adapter_path_from_result(result: TrainerStepResult | dict[str, Any] | None) -> str | None:
        if isinstance(result, TrainerStepResult):
            return result.adapter_path
        if isinstance(result, dict):
            return result.get('adapter_path')
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
        adapter_registry: LoraAdapterRegistry,
        scheduler: TrainerScheduler,
        model: Any,
        train_config: MultiLoraGRPOTrainConfig | None = None,
        receive_weights_fn: Callable[[LoraContext], None] | None = None,
        train_batch_groups: int = 1,
    ):
        self.model = model
        self.train_config = train_config or MultiLoraGRPOTrainConfig()
        super().__init__(
            data_plane=data_plane,
            adapter_registry=adapter_registry,
            scheduler=scheduler,
            train_batch_fn=self.train_batch,
            save_adapter_fn=self.save_adapter,
            receive_weights_fn=receive_weights_fn,
            train_batch_groups=train_batch_groups,
        )

    def train_batch(self, context: LoraContext, partition_id: str, dataloader: Any) -> TrainerStepResult:
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
        return TrainerStepResult()

    def save_adapter(self, context: LoraContext, partition_id: str) -> TrainerStepResult:
        config = self.train_config
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
        adapter_path = getattr(save_result, 'twinkle_path', None)
        if adapter_path is None and isinstance(save_result, str):
            adapter_path = save_result
        if adapter_path is None and isinstance(save_result, dict):
            adapter_path = save_result.get('twinkle_path') or save_result.get('path')
        return TrainerStepResult(adapter_path=adapter_path)
