# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import asyncio
import inspect
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Iterable

from twinkle.data_format import Trajectory
from twinkle_agentic.tools.tool_manager import ToolManager
from .data_plane import TransferQueueDataPlane
from .metrics import AsyncRLMetricsRecorder, NoopMetricsRecorder, prefixed_summary
from .registry import LoraRuntimeRegistry
from .scheduling import PreferCurrentTrainPolicy, WorkConservingRolloutPolicy
from .staleness import StalenessManager
from .tq_utils import columns_to_tq_fields, read_train_batch
from .types import (ComponentResult, GRPOAdvantageBatch, LoraContext, PartitionMeta, PartitionStatus, PromptGroupRef,
                    PromptGroupStatus, RolloutCallable, RolloutScheduleCandidate, TrainBatchCandidate, TrainStageResult,
                    TransformersTrainBatch)


class ToolManagerFactory:
    """Create context-scoped ToolManager instances.

    Profiles are callables so deployments can attach native or remote tools
    without importing untrusted user code in the server process.
    """

    def __init__(self, profiles: dict[str, Callable[[LoraContext, Trajectory], ToolManager]] | None = None):
        self._profiles = dict(profiles or {})

    def register(self, profile: str, factory: Callable[[LoraContext, Trajectory], ToolManager]) -> None:
        self._profiles[profile] = factory

    def create(self, prompt_group: Trajectory, context: LoraContext) -> ToolManager:
        factory = self._profiles.get(context.tool_profile)
        if factory is None:
            return ToolManager()
        return factory(context, prompt_group)


TERMINAL_PARTITION_STATUSES = {
    PartitionStatus.TRAIN_DONE,
    PartitionStatus.CLEARED,
    PartitionStatus.FAILED,
}


def _metrics_from_result(result: Any) -> dict[str, Any]:
    if isinstance(result, TrainerStepResult):
        return dict(result.metrics or {})
    if isinstance(result, dict):
        return dict(result.get('metrics') or {})
    return {}


def _batch_policy_gap_metrics(batch: Any, *, current_policy_version: int) -> dict[str, float]:
    gaps = []
    for tag in getattr(batch, 'tags', None) or []:
        rollout_policy_version = tag.get('rollout_policy_version')
        if rollout_policy_version is not None:
            gaps.append(float(current_policy_version - int(rollout_policy_version)))
    return prefixed_summary('policy_version_gap', gaps)


def _assert_generation_tag_schema(
    tags: Iterable[dict[str, Any]],
    *,
    expected_num_generations: int,
    context_key: str,
    partition_id: str,
) -> None:
    tags = list(tags)
    assert expected_num_generations > 0, (
        f'num_generations must be positive for context {context_key}, got {expected_num_generations}')
    assert tags, f'empty sample tags for context {context_key}, partition {partition_id}'
    assert len(tags) % expected_num_generations == 0, (
        f'sample tag count must be divisible by num_generations for context {context_key}, '
        f'partition {partition_id}: {len(tags)} % {expected_num_generations} != 0')
    for offset in range(0, len(tags), expected_num_generations):
        chunk = tags[offset:offset + expected_num_generations]
        group_id = chunk[0].get('group_id')
        assert group_id is not None, f'missing group_id at sample offset {offset} in partition {partition_id}'
        for tag_index, tag in enumerate(chunk):
            assert 'generation_idx' in tag, (
                f'missing generation_idx at sample offset {offset + tag_index} in partition {partition_id}')
        group_ids = [tag.get('group_id') for tag in chunk]
        generation_indices = [int(tag['generation_idx']) for tag in chunk]
        assert all(item == group_id for item in group_ids), (
            f'samples for one GRPO group must be contiguous in partition {partition_id}, '
            f'offset={offset}, group_ids={group_ids}')
        assert generation_indices == list(range(expected_num_generations)), (
            f'group {group_id} generation_idx must be 0..{expected_num_generations - 1} in order, '
            f'got {generation_indices}')


def _partition_train_id(partition_id: str) -> int:
    suffix = partition_id.rsplit('/', 1)[-1]
    if not suffix.startswith('train_'):
        raise ValueError(f'partition id must end with train_<id>, got {partition_id!r}')
    try:
        return int(suffix[len('train_'):])
    except ValueError as exc:
        raise ValueError(f'partition id must end with numeric train id, got {partition_id!r}') from exc


class AsyncRollouter:
    """Schedule prompt groups, run rollout, and append results to train partitions."""

    def __init__(
        self,
        *,
        data_plane: TransferQueueDataPlane,
        lora_runtime_registry: LoraRuntimeRegistry,
        staleness_manager: StalenessManager,
        rollout: RolloutCallable,
        tool_manager_factory: ToolManagerFactory | None = None,
        rollout_policy: Any | None = None,
        max_concurrency: int = 16,
        target_groups_per_partition: int = 1,
        target_groups_by_context: dict[str, int] | None = None,
        num_generations: int = 1,
        num_generations_by_context: dict[str, int] | None = None,
        metrics_recorder: AsyncRLMetricsRecorder | None = None,
    ):
        self.data_plane = data_plane
        self.lora_runtime_registry = lora_runtime_registry
        self.staleness_manager = staleness_manager
        self.rollout: RolloutCallable = rollout
        self.tool_manager_factory = tool_manager_factory or ToolManagerFactory()
        self.rollout_policy = rollout_policy or WorkConservingRolloutPolicy()
        self.max_concurrency = max_concurrency
        self.target_groups_per_partition = target_groups_per_partition
        self.target_groups_by_context = dict(target_groups_by_context or {})
        self.num_generations = int(num_generations)
        self.num_generations_by_context = dict(num_generations_by_context or {})
        self.metrics_recorder = metrics_recorder or NoopMetricsRecorder()
        self.pending_prompt_groups_by_context: dict[str, Deque[tuple[LoraContext, Trajectory]]] = defaultdict(deque)
        self.active_tasks: set[asyncio.Task] = set()
        self.prompt_loaders: list[Any] = []
        self._last_rollout_submit_time: dict[str, float] = defaultdict(float)
        self._submitted_prompt_groups: dict[str, int] = defaultdict(int)
        self._next_train_id_by_context: dict[str, int] = {}

    def attach_prompt_loaders(self, prompt_loaders: Iterable[Any]) -> None:
        self.prompt_loaders = list(prompt_loaders)

    def enqueue_prompt_groups(self, context: LoraContext, prompt_groups: Iterable[Trajectory]) -> None:
        """Append rollout inputs for a context.

        A prompt group is the scheduling unit for rollout. It may produce one
        or more trajectories depending on the rollout implementation.
        """
        self.lora_runtime_registry.register(context)
        queue = self.pending_prompt_groups_by_context[context.key]
        for prompt_group in prompt_groups:
            queue.append((context, prompt_group))

    def pending_prompt_group_count(self, context: LoraContext) -> int:
        return len(self.pending_prompt_groups_by_context.get(context.key, ()))

    def _list_live_partitions(self, context: LoraContext) -> list[PartitionMeta]:
        runtime_state = self.lora_runtime_registry.get(context)
        partitions = []
        stale_partition_ids = []
        for partition_id in sorted(runtime_state.live_partitions):
            try:
                partitions.append(self.data_plane.get_rollout_partition(partition_id))
            except KeyError:
                stale_partition_ids.append(partition_id)
        for partition_id in stale_partition_ids:
            self.lora_runtime_registry.on_partition_cleared(context, partition_id)
        if partitions:
            return sorted(partitions, key=lambda partition: (partition.created_at, partition.partition_id))
        return self.data_plane.list_partitions(context)

    def _build_rollout_candidate(self, context: LoraContext) -> RolloutScheduleCandidate | None:
        """Collect current queue, staleness, partition, and adapter state for scheduling."""
        pending_groups = len(self.pending_prompt_groups_by_context.get(context.key, ()))
        if pending_groups <= 0:
            return None
        runtime_state = self.lora_runtime_registry.get(context)
        partitions = self._list_live_partitions(context)
        active_partitions = [p for p in partitions if p.status == PartitionStatus.ACTIVE]
        active_partition = active_partitions[0] if active_partitions else None
        free_slots = 0
        if active_partition is not None:
            free_slots = self._free_group_slots(active_partition)
        elif self._can_create_next_rollout_partition(context):
            free_slots = self._target_groups_for_context(context)
        return RolloutScheduleCandidate(
            context=context,
            pending_groups=pending_groups,
            in_flight_groups=runtime_state.in_flight_groups,
            live_partitions=len([p for p in partitions if p.status != PartitionStatus.CLEARED]),
            active_partitions=len(active_partitions),
            rollout_capacity=free_slots,
            rollout_cost=1.0,
            last_submit_time=self._last_rollout_submit_time[context.key],
            submitted_groups=self._submitted_prompt_groups[context.key],
            weight=runtime_state.weight,
        )

    def _pick_next_rollout_candidate(self) -> LoraContext | None:
        """Choose the next context that is allowed to submit one rollout batch."""
        candidates: list[RolloutScheduleCandidate] = []
        if self._remaining_task_capacity() <= 0:
            return None
        seen: dict[str, LoraContext] = {}
        for queue in self.pending_prompt_groups_by_context.values():
            if not queue:
                continue
            context = queue[0][0]
            seen[context.key] = context
        for context in seen.values():
            if not self.lora_runtime_registry.can_accept_rollout(context):
                continue
            candidate = self._build_rollout_candidate(context)
            if candidate is None or candidate.rollout_capacity <= 0:
                continue
            candidates.append(candidate)
        return self.rollout_policy.pick_next_context(candidates)

    def _remaining_task_capacity(self) -> int:
        return max(0, self.max_concurrency - len(self.active_tasks))

    def _target_groups_for_context(self, context: LoraContext) -> int:
        return int(self.target_groups_by_context.get(context.key, self.target_groups_per_partition))

    def _num_generations_for_context(self, context: LoraContext) -> int:
        value = int(self.num_generations_by_context.get(context.key, self.num_generations))
        assert value > 0, f'num_generations must be positive for context {context.key}, got {value}'
        return value

    def _peek_next_train_id(self, context: LoraContext) -> int:
        if context.key not in self._next_train_id_by_context:
            self._next_train_id_by_context[context.key] = self.data_plane.peek_next_train_id(context)
        return self._next_train_id_by_context[context.key]

    def _allocate_next_partition_id(self, context: LoraContext) -> str:
        train_id = self._peek_next_train_id(context)
        self._next_train_id_by_context[context.key] = train_id + 1
        return context.partition_id(train_id)

    def _free_group_slots(self, partition: PartitionMeta) -> int:
        if partition.status != PartitionStatus.ACTIVE:
            return 0
        group_count = len(self.data_plane.list_prompt_groups(partition.context, partition_id=partition.partition_id))
        return max(0, partition.target_groups - group_count)

    def _can_create_next_rollout_partition(self, context: LoraContext) -> bool:
        partitions = self._list_live_partitions(context)
        live_partitions = [partition for partition in partitions if partition.status not in TERMINAL_PARTITION_STATUSES]
        if not live_partitions:
            return True
        oldest_train_id = min(_partition_train_id(partition.partition_id) for partition in live_partitions)
        next_train_id = self._peek_next_train_id(context)
        return next_train_id - oldest_train_id <= self.staleness_manager.max_staleness

    def _pop_prompt_group(self, context: LoraContext) -> Trajectory:
        queue = self.pending_prompt_groups_by_context[context.key]
        _, prompt_group = queue.popleft()
        return prompt_group

    async def _run_prompt_group_rollout(
        self,
        group_ref: PromptGroupRef,
        prompt_group: Trajectory,
    ) -> PartitionMeta:
        """Run rollout for one prompt group and append its generated samples to train_k."""
        return await self._run_prompt_group_rollout_batch([(group_ref, prompt_group)])

    async def _run_prompt_group_rollout_batch(
        self,
        items: list[tuple[PromptGroupRef, Trajectory]],
    ) -> PartitionMeta:
        """Run one sampler-facing rollout batch and write each prompt group independently."""
        assert items, 'rollout batch must not be empty'
        group_refs = [group_ref for group_ref, _ in items]
        prompt_groups = [prompt_group for _, prompt_group in items]
        groups = [self.data_plane.get_prompt_group(group_ref) for group_ref in group_refs]
        context = groups[0].context
        partition_id = group_refs[0].partition_id
        assert all(group.context.key == context.key for group in groups), 'rollout batch must use one context'
        assert all(group_ref.partition_id == partition_id for group_ref in group_refs), (
            'rollout batch must use one partition')
        policy_version = groups[0].rollout_policy_version
        adapter_path = groups[0].rollout_adapter_path
        assert all(group.rollout_policy_version == policy_version for group in groups), (
            'rollout batch must use one policy version')
        assert all(group.rollout_adapter_path == adapter_path for group in groups), (
            'rollout batch must use one adapter path')

        for group_ref in group_refs:
            self.data_plane.update_prompt_group_status(group_ref, PromptGroupStatus.RUNNING)
            self.lora_runtime_registry.on_rollout_started(context)
        start = time.perf_counter()
        num_generations = self._num_generations_for_context(context)
        for group_ref in group_refs:
            self.metrics_recorder.log_event(
                event='rollout_started',
                phase='rollout',
                context=context,
                partition_id=partition_id,
                policy_version=policy_version,
                metrics={
                    'group_id': group_ref.group_id,
                    'submit_batch_group_count': len(group_refs),
                    'inflight_rollout_groups': self.lora_runtime_registry.get(context).in_flight_groups,
                    'num_generations': num_generations,
                },
            )
        try:
            tool_managers = [
                self.tool_manager_factory.create(prompt_group, context) for prompt_group in prompt_groups
            ]
            rollout_inputs = []
            for group_ref, prompt_group in items:
                trajectory = dict(prompt_group)
                trajectory['group_id'] = group_ref.group_id
                rollout_inputs.append(trajectory)
            rollout_kwargs = {
                'tool_manager': tool_managers[0] if len(tool_managers) == 1 else tool_managers,
                'context': context,
                'partition_id': partition_id,
                'group_refs': group_refs,
                'groups': groups,
                'adapter_name': context.adapter_name,
                'policy_version': policy_version,
                'num_generations': num_generations,
            }
            if adapter_path is not None:
                rollout_kwargs['adapter_path'] = adapter_path
            submission = await self._invoke_rollout(rollout_inputs, rollout_kwargs)
            if not isinstance(submission, dict):
                raise TypeError(f'TQ rollout must return submission metadata, got {type(submission)!r}')
            submitted_groups = int(submission.get('submitted_prompt_groups', 0))
            assert submitted_groups == len(group_refs), (
                f'TQ rollout submitted {submitted_groups} groups, expected {len(group_refs)}')
            last_meta = self.data_plane.get_rollout_partition(partition_id)
            self._last_rollout_submit_time[context.key] = time.time()
            self._submitted_prompt_groups[context.key] += len(group_refs)
            return last_meta
        except Exception as exc:
            for group_ref, group in zip(group_refs, groups):
                current = self.data_plane.get_prompt_group(group_ref)
                if current.status in {PromptGroupStatus.ROLLOUT_DONE, PromptGroupStatus.DROPPED}:
                    continue
                self.data_plane.update_prompt_group_status(
                    group_ref,
                    PromptGroupStatus.FAILED,
                    extra_tag={'error': str(exc)},
                )
                self.metrics_recorder.log_event(
                    event='rollout_failed',
                    phase='rollout',
                    context=context,
                    partition_id=partition_id,
                    policy_version=group.rollout_policy_version,
                    metrics={
                        'group_id': group_ref.group_id,
                        'submit_batch_group_count': len(group_refs),
                        'rollout_latency_s': time.perf_counter() - start,
                        'error': str(exc),
                    },
                )
            raise
        finally:
            for _ in group_refs:
                self.lora_runtime_registry.on_rollout_finished(context)

    async def _invoke_rollout(
        self,
        trajectories: list[Trajectory],
        rollout_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Call sync rollout implementations without blocking the event loop."""
        call = getattr(self.rollout, '__call__', None)
        if inspect.iscoroutinefunction(self.rollout) or inspect.iscoroutinefunction(call):
            return await self.rollout(trajectories, **rollout_kwargs)
        return await asyncio.to_thread(self.rollout, trajectories, **rollout_kwargs)

    def _submit_prompt_group_tasks(self) -> int:
        submitted_groups = 0
        while self._remaining_task_capacity() > 0:
            context = self._pick_next_rollout_candidate()
            if context is None:
                break
            partition = self._get_active_or_create_rollout_partition(context)
            if partition is None:
                continue
            runtime_state = self.lora_runtime_registry.get(context)
            free_slots = self._free_group_slots(partition)
            batch_group_count = min(
                self.pending_prompt_group_count(context),
                free_slots,
            )
            if batch_group_count <= 0:
                continue
            batch_items: list[tuple[PromptGroupRef, Trajectory]] = []
            for _ in range(batch_group_count):
                try:
                    group_ref = self.data_plane.create_prompt_group(
                        context,
                        partition,
                        runtime_state=runtime_state,
                    )
                except LookupError:
                    break
                prompt_group = self._pop_prompt_group(context)
                batch_items.append((group_ref, prompt_group))
            if not batch_items:
                continue
            task = asyncio.create_task(self._run_prompt_group_rollout_batch(batch_items))
            self.active_tasks.add(task)
            self._close_partition_if_full(partition)
            submitted_groups += len(batch_items)
            self.metrics_recorder.log_event(
                event='rollout_submitted',
                phase='rollout',
                context=context,
                partition_id=partition.partition_id,
                policy_version=runtime_state.policy_version,
                metrics={
                    'submitted_groups': submitted_groups,
                    'submit_batch_group_count': len(batch_items),
                    'active_tasks': len(self.active_tasks),
                    'pending_prompt_groups': self.pending_prompt_group_count(context),
                    'rollout_capacity': self._free_group_slots(partition),
                    'max_concurrency': self.max_concurrency,
                },
            )
        return submitted_groups

    async def _collect_finished_tasks(self) -> int:
        done = [task for task in self.active_tasks if task.done()]
        for task in done:
            self.active_tasks.remove(task)
            if task.cancelled():
                continue
            try:
                await task
            except Exception as exc:
                # Batch/group-level failure is already persisted by the rollout task.
                # The exception is consumed here so asyncio does not leak it.
                _ = exc
        return len(done)

    def _get_active_or_create_rollout_partition(self, context: LoraContext) -> PartitionMeta | None:
        """Return the context's active rollout partition, or create the next train_k."""
        active_partitions = [
            partition for partition in self._list_live_partitions(context) if partition.status == PartitionStatus.ACTIVE
        ]
        active_partition = active_partitions[0] if active_partitions else None
        if active_partition is not None:
            if self._free_group_slots(active_partition) > 0:
                return active_partition
            self.data_plane.update_partition_status(active_partition.partition_id, PartitionStatus.CLOSED)
        runtime_state = self.lora_runtime_registry.get(context)
        if not self._can_create_next_rollout_partition(context):
            return None
        meta = self.data_plane.create_rollout_partition(
            context,
            target_groups=self._target_groups_for_context(context),
            partition_id=self._allocate_next_partition_id(context),
        )
        self.lora_runtime_registry.on_partition_created(context, meta.partition_id)
        return meta

    def _close_partition_if_full(self, partition: PartitionMeta) -> None:
        if partition.status != PartitionStatus.ACTIVE:
            return
        if self._free_group_slots(partition) == 0:
            self.data_plane.update_partition_status(partition.partition_id, PartitionStatus.CLOSED)

    def _context_prompt_source_exhausted(self, context: LoraContext) -> bool:
        loaders = [
            loader for loader in self.prompt_loaders
            if getattr(getattr(loader, 'context', None), 'key', None) == context.key
        ]
        if not loaders:
            return True
        return all(getattr(loader, 'exhausted', False) for loader in loaders)

    def _close_exhausted_partial_partitions(self) -> int:
        closed = 0
        contexts = {
            context
            for queue in self.pending_prompt_groups_by_context.values()
            for context, _ in queue
        }
        contexts.update(getattr(loader, 'context') for loader in self.prompt_loaders if getattr(loader, 'context', None))
        for runtime_state in self.lora_runtime_registry.list_states():
            for partition_id in runtime_state.live_partitions:
                try:
                    contexts.add(self.data_plane.get_rollout_partition(partition_id).context)
                except KeyError:
                    continue
        for context in contexts:
            if self.pending_prompt_group_count(context) > 0:
                continue
            if not self._context_prompt_source_exhausted(context):
                continue
            for partition in self._list_live_partitions(context):
                if partition.status != PartitionStatus.ACTIVE:
                    continue
                if self._free_group_slots(partition) <= 0:
                    continue
                group_count = len(self.data_plane.list_prompt_groups(context, partition_id=partition.partition_id))
                if group_count <= 0:
                    continue
                self.data_plane.update_partition_status(partition.partition_id, PartitionStatus.CLOSED)
                closed += 1
        return closed

    def _load_prompt_stage(self) -> int:
        loaded = 0
        for loader in self.prompt_loaders:
            result = loader.step()
            if result is not None:
                loaded += result.count
        return loaded

    async def step(self) -> ComponentResult | None:
        loaded_prompt_groups = self._load_prompt_stage()
        completed = await self._collect_finished_tasks()
        submitted_groups = self._submit_prompt_group_tasks()
        closed_partitions = self._close_exhausted_partial_partitions()
        if loaded_prompt_groups or completed or submitted_groups or closed_partitions:
            return ComponentResult(
                component='rollouter',
                kind='rollout',
                count=loaded_prompt_groups + completed + submitted_groups + closed_partitions,
            )
        return None

    def is_idle(self) -> bool:
        return (
            all(loader.is_idle() for loader in self.prompt_loaders)
            and not any(self.pending_prompt_groups_by_context.values())
            and not self.active_tasks
        )

    def shutdown(self) -> None:
        for task in list(self.active_tasks):
            task.cancel()
        for loader in self.prompt_loaders:
            shutdown = getattr(loader, 'shutdown', None)
            if shutdown is not None:
                shutdown()


class AdvantageWorker:

    def __init__(
        self,
        *,
        data_plane: TransferQueueDataPlane,
        contexts: list[LoraContext] | None = None,
        lora_runtime_registry: LoraRuntimeRegistry | None = None,
        batch_size: int = 1024,
        batch_size_by_context: dict[str, int] | None = None,
        num_generations: int = 1,
        num_generations_by_context: dict[str, int] | None = None,
        advantage_fn: Callable[[GRPOAdvantageBatch, LoraContext], tuple[list[float], list[float]]] | None = None,
        metrics_recorder: AsyncRLMetricsRecorder | None = None,
    ):
        self.data_plane = data_plane
        self.contexts = list(contexts or [])
        self.lora_runtime_registry = lora_runtime_registry
        self.batch_size = batch_size
        self.batch_size_by_context = dict(batch_size_by_context or {})
        self.num_generations = int(num_generations)
        self.num_generations_by_context = dict(num_generations_by_context or {})
        self.advantage_fn = advantage_fn or self._default_advantage_fn
        self.metrics_recorder = metrics_recorder or NoopMetricsRecorder()

    @staticmethod
    def _default_advantage_fn(batch: GRPOAdvantageBatch, context: LoraContext) -> tuple[list[float], list[float]]:
        rewards = list(batch.rewards)
        if not rewards:
            return [], []
        mean_reward = sum(rewards) / len(rewards)
        advantages = [reward - mean_reward for reward in rewards]
        return advantages, rewards

    def process_advantage_batch(
        self,
        context: LoraContext,
        *,
        partition_id: str,
        groups: list[Any],
    ) -> PartitionMeta:
        if not groups:
            raise ValueError('groups must not be empty')
        start = time.perf_counter()
        batch = self.data_plane.claim_prompt_groups(
            groups,
            ready_status=PromptGroupStatus.ROLLOUT_DONE,
            claim_status=PromptGroupStatus.ADVANTAGING,
            fields=['rewards'],
        )
        sample_count = len(batch.keys)
        self.metrics_recorder.log_event(
            event='advantage_started',
            phase='advantage',
            context=context,
            partition_id=partition_id,
            metrics={
                'group_count': len(groups),
                'sample_count': sample_count,
            },
        )
        try:
            self._compute_advantages(context, batch)
            self.data_plane.mark_prompt_groups(groups, PromptGroupStatus.ADVANTAGE_DONE)
            self.metrics_recorder.log_event(
                event='advantage_done',
                phase='advantage',
                context=context,
                partition_id=partition_id,
                metrics={
                    'group_count': len(groups),
                    'sample_count': sample_count,
                    'advantage_latency_s': time.perf_counter() - start,
                },
            )
        except Exception as exc:
            self.data_plane.mark_prompt_groups(groups, PromptGroupStatus.FAILED, extra_tag={'error': str(exc)})
            self.metrics_recorder.log_event(
                event='advantage_failed',
                phase='advantage',
                context=context,
                partition_id=partition_id,
                metrics={
                    'group_count': len(groups),
                    'sample_count': len(batch.keys),
                    'advantage_latency_s': time.perf_counter() - start,
                    'error': str(exc),
                },
            )
            raise
        return self.data_plane.get_rollout_partition(partition_id)

    def _compute_advantages(self, context: LoraContext, batch: Any) -> Any:
        columns = self.data_plane.read_batch_fields(batch, fields=['rewards'])
        num_generations = self.num_generations_for_context(context)
        _assert_generation_tag_schema(
            batch.tags,
            expected_num_generations=num_generations,
            context_key=context.key,
            partition_id=batch.partition_id,
        )
        advantage_batch = GRPOAdvantageBatch(
            rewards=columns['rewards'],
            sample_keys=list(batch.keys),
            group_ids=[str(tag['group_id']) for tag in batch.tags],
            generation_indices=[int(tag['generation_idx']) for tag in batch.tags],
            num_generations=num_generations,
        )
        advantages, returns = self.advantage_fn(advantage_batch, context)
        sample_count = len(batch.keys)
        if len(advantages) != sample_count or len(returns) != sample_count:
            raise ValueError(f'advantage_fn returned {len(advantages)} advantages and {len(returns)} returns '
                             f'for {sample_count} samples')
        self.data_plane.write_batch_fields(
            batch,
            fields=columns_to_tq_fields(
                {
                    'advantages': advantages,
                    'returns': returns,
                },
                sample_count,
            ),
        )
        return batch

    def process_available(self) -> ComponentResult | None:
        if self.lora_runtime_registry is None:
            return None
        count = 0
        last_meta = None
        for context in self.contexts:
            max_groups = self.batch_size_for_context(context)
            try:
                partition_ids = sorted(self.lora_runtime_registry.get(context).live_partitions)
            except KeyError:
                continue
            for partition_id in partition_ids:
                while True:
                    groups = self._select_advantage_groups(context, partition_id=partition_id, max_groups=max_groups)
                    if not groups:
                        break
                    last_meta = self.process_advantage_batch(context, partition_id=partition_id, groups=groups)
                    count += len(groups)
        if last_meta is None:
            return None
        return ComponentResult(component='advantage_worker', kind='advantage', metadata=last_meta, count=count)

    def step(self) -> ComponentResult | None:
        return self.process_available()

    def batch_size_for_context(self, context: LoraContext) -> int:
        return int(self.batch_size_by_context.get(context.key, self.batch_size))

    def num_generations_for_context(self, context: LoraContext) -> int:
        value = int(self.num_generations_by_context.get(context.key, self.num_generations))
        assert value > 0, f'num_generations must be positive for context {context.key}, got {value}'
        return value

    def is_idle(self) -> bool:
        if self.lora_runtime_registry is None:
            return True
        for context in self.contexts:
            partition_ids = sorted(self.lora_runtime_registry.get(context).live_partitions)
            max_groups = self.batch_size_for_context(context)
            for partition_id in partition_ids:
                if self._select_advantage_groups(context, partition_id=partition_id, max_groups=max_groups):
                    return False
        return True

    def shutdown(self) -> None:
        return None

    def _select_advantage_groups(
        self,
        context: LoraContext,
        *,
        partition_id: str,
        max_groups: int,
    ) -> list[Any]:
        groups = self.data_plane.list_prompt_groups(
            context,
            partition_id=partition_id,
            statuses=[PromptGroupStatus.ROLLOUT_DONE],
        )
        groups = sorted(groups, key=lambda group: (group.created_at, group.partition_id, group.group_id))
        return groups[:max_groups]


class TrainerScheduler:

    def __init__(self, *, lora_runtime_registry: LoraRuntimeRegistry, train_policy: Any | None = None):
        self.lora_runtime_registry = lora_runtime_registry
        self.train_policy = train_policy or PreferCurrentTrainPolicy()

    def next_batch(
        self,
        candidates: list[TrainBatchCandidate],
        current_context: LoraContext | None = None,
    ) -> TrainBatchCandidate | None:
        filtered = []
        for candidate in candidates:
            if not self.lora_runtime_registry.can_train(candidate.context):
                continue
            filtered.append(candidate)
        return self.train_policy.pick_next_batch(filtered, current_context)


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
        lora_runtime_registry: LoraRuntimeRegistry,
        scheduler: TrainerScheduler,
        train_batch_fn: Callable[[LoraContext, Any], TrainerStepResult | dict[str, Any] | None],
        save_adapter_fn: Callable[[LoraContext, str], TrainerStepResult | dict[str, Any] | None] | None = None,
        receive_weights_fn: Callable[[Any], None] | None = None,
        train_batch_groups: int = 1,
        train_batch_groups_by_context: dict[str, int] | None = None,
        metrics_recorder: AsyncRLMetricsRecorder | None = None,
    ):
        self.data_plane = data_plane
        self.lora_runtime_registry = lora_runtime_registry
        self.scheduler = scheduler
        self.train_batch_fn = train_batch_fn
        self.save_adapter_fn = save_adapter_fn
        self.receive_weights_fn = receive_weights_fn
        self.train_batch_groups = train_batch_groups
        self.train_batch_groups_by_context = dict(train_batch_groups_by_context or {})
        self.metrics_recorder = metrics_recorder or NoopMetricsRecorder()
        self.current_context: LoraContext | None = None

    def step(self) -> ComponentResult | None:
        candidate = self.scheduler.next_batch(
            self.list_train_batch_candidates(),
            self.current_context,
        )
        if candidate is None:
            self._drop_untrainable_closed_tail_partitions()
            return None
        return self._train_candidate(candidate)

    def train_until_blocked(
        self,
        *,
        max_partitions: int | None = None,
    ) -> TrainStageResult | None:
        return self.train_one_partition(max_partitions=max_partitions)

    def train_one_partition(
        self,
        *,
        max_partitions: int | None = None,
    ) -> TrainStageResult | None:
        if max_partitions is not None and max_partitions <= 0:
            return None
        candidates = self.list_train_batch_candidates()
        candidate = self.scheduler.next_batch(
            candidates,
            self.current_context,
        )
        if candidate is None:
            self._drop_untrainable_closed_tail_partitions()
            return None

        selected_context = candidate.context
        selected_partition_id = candidate.partition.partition_id
        train_batches = 0
        trained_partitions = 0
        last_meta: PartitionMeta | None = None
        while candidate is not None:
            result = self._train_candidate(candidate)
            train_batches += 1
            last_meta = result.metadata
            if result.kind == 'train':
                trained_partitions += 1
                if max_partitions is not None and trained_partitions >= max_partitions:
                    break
                break
            candidate = self._next_candidate_for_partition(selected_context, selected_partition_id)

        return TrainStageResult(
            train_batches=train_batches,
            trained_partitions=trained_partitions,
            metadata=last_meta,
        )

    def _next_candidate_for_partition(self, context: LoraContext, partition_id: str) -> TrainBatchCandidate | None:
        candidates = [
            candidate for candidate in self.list_train_batch_candidates()
            if candidate.context.key == context.key and candidate.partition.partition_id == partition_id
        ]
        return self.scheduler.next_batch(candidates, context)

    def _drop_untrainable_closed_tail_partitions(self) -> int:
        dropped = 0
        terminal_statuses = {PromptGroupStatus.TRAIN_DONE, PromptGroupStatus.FAILED, PromptGroupStatus.DROPPED}
        blocking_statuses = {
            PromptGroupStatus.PENDING,
            PromptGroupStatus.RUNNING,
            PromptGroupStatus.ROLLOUT_DONE,
            PromptGroupStatus.ADVANTAGING,
            PromptGroupStatus.TRAINING,
        }
        for partition in self.list_live_partitions():
            if partition.status != PartitionStatus.CLOSED:
                continue
            groups = self.data_plane.list_prompt_groups(partition.context, partition_id=partition.partition_id)
            if not groups:
                continue
            if any(group.status in blocking_statuses for group in groups):
                continue
            ready_groups = [group for group in groups if group.status == PromptGroupStatus.ADVANTAGE_DONE]
            required_groups = self.train_batch_groups_for_context(partition.context)
            if len(ready_groups) >= required_groups:
                continue
            if any(group.status not in terminal_statuses and group.status != PromptGroupStatus.ADVANTAGE_DONE
                   for group in groups):
                continue
            trained_group_count = len([group for group in groups if group.status == PromptGroupStatus.TRAIN_DONE])
            failed_group_count = len([group for group in groups if group.status == PromptGroupStatus.FAILED])
            already_dropped_group_count = len([group for group in groups if group.status == PromptGroupStatus.DROPPED])
            drop_reason = 'partial_tail_below_mini_batch' if ready_groups else 'closed_terminal_partition'
            if trained_group_count > 0:
                start = time.perf_counter()
                sync_result = self.save_adapter_fn(partition.context,
                                                   partition.partition_id) if self.save_adapter_fn else None
                adapter_path = self._adapter_path_from_result(sync_result)
                if ready_groups:
                    self.data_plane.mark_prompt_groups(
                        ready_groups,
                        PromptGroupStatus.DROPPED,
                        extra_tag={'drop_reason': drop_reason},
                    )
                self.data_plane.update_partition_status(partition.partition_id, PartitionStatus.TRAIN_DONE)
                self.lora_runtime_registry.on_weight_sync_started(partition.context)
                runtime_state = self.lora_runtime_registry.on_weight_sync_finished(
                    partition.context,
                    adapter_path=adapter_path,
                )
                if self.receive_weights_fn is not None:
                    self.receive_weights_fn(runtime_state)
                self.metrics_recorder.log_event(
                    event='weight_sync_done',
                    phase='train',
                    context=partition.context,
                    partition_id=partition.partition_id,
                    policy_version=runtime_state.policy_version,
                    metrics={
                        'group_count': trained_group_count,
                        'dropped_tail_groups': len(ready_groups),
                        'failed_groups': failed_group_count,
                        'already_dropped_groups': already_dropped_group_count,
                        'adapter_path': adapter_path,
                    },
                )
                self.data_plane.clear_partition(partition.context, partition.partition_id)
                self.lora_runtime_registry.on_partition_cleared(partition.context, partition.partition_id)
                self.metrics_recorder.log_event(
                    event='partition_train_done',
                    phase='train',
                    context=partition.context,
                    partition_id=partition.partition_id,
                    policy_version=runtime_state.policy_version,
                    metrics={
                        'group_count': trained_group_count,
                        'dropped_tail_groups': len(ready_groups),
                        'failed_groups': failed_group_count,
                        'already_dropped_groups': already_dropped_group_count,
                        'partition_train_latency_s': time.perf_counter() - start,
                    },
                )
                dropped += 1
                continue
            if ready_groups:
                self.data_plane.mark_prompt_groups(
                    ready_groups,
                    PromptGroupStatus.DROPPED,
                    extra_tag={'drop_reason': drop_reason},
                )
            self.data_plane.clear_partition(partition.context, partition.partition_id)
            self.lora_runtime_registry.on_partition_cleared(partition.context, partition.partition_id)
            self.metrics_recorder.log_event(
                event='partition_dropped',
                phase='train',
                context=partition.context,
                partition_id=partition.partition_id,
                metrics={
                    'group_count': len(ready_groups),
                    'required_groups': required_groups,
                    'failed_groups': failed_group_count,
                    'already_dropped_groups': already_dropped_group_count,
                    'drop_reason': drop_reason,
                },
            )
            dropped += 1
        return dropped

    def _next_candidate_for_context(self, context: LoraContext) -> TrainBatchCandidate | None:
        candidates = [
            candidate for candidate in self.list_train_batch_candidates()
            if candidate.context.key == context.key
        ]
        return self.scheduler.next_batch(candidates, context)

    def _train_candidate(self, candidate: TrainBatchCandidate) -> ComponentResult:
        partition = candidate.partition
        context = partition.context
        self.current_context = context
        self.lora_runtime_registry.on_train_started(context, partition.partition_id)
        batch = None
        try:
            start = time.perf_counter()
            groups = self.select_train_groups(
                context,
                partition.partition_id,
                max_groups=self.train_batch_groups_for_context(context),
            )
            batch = self.data_plane.claim_prompt_groups(
                groups,
                ready_status=PromptGroupStatus.ADVANTAGE_DONE,
                claim_status=PromptGroupStatus.TRAINING,
                fields=['input_ids', 'labels', 'logprobs', 'advantages'],
            )
            current_policy_version = self.lora_runtime_registry.get(context).policy_version
            gap_metrics = _batch_policy_gap_metrics(batch, current_policy_version=current_policy_version)
            self.metrics_recorder.log_event(
                event='train_claimed',
                phase='train',
                context=context,
                partition_id=partition.partition_id,
                policy_version=current_policy_version,
                metrics={
                    'group_count': len(groups),
                    'sample_count': len(batch.keys),
                    **gap_metrics,
                },
            )
            result = self.train_batch_fn(context, batch)
            self.data_plane.mark_prompt_groups(groups, PromptGroupStatus.TRAIN_DONE)
            group_count = len(groups)
            train_metrics = _metrics_from_result(result)
            self.metrics_recorder.log_event(
                event='train_batch_done',
                phase='train',
                context=context,
                partition_id=partition.partition_id,
                policy_version=current_policy_version,
                metrics={
                    'group_count': group_count,
                    'sample_count': len(batch.keys),
                    'train_batch_latency_s': time.perf_counter() - start,
                    **gap_metrics,
                    **train_metrics,
                },
            )
            batch = None
            if not self.partition_training_complete(context, partition.partition_id):
                self.lora_runtime_registry.on_train_finished(context, partition.partition_id)
                return ComponentResult(
                    component='trainer_worker',
                    kind='train_batch',
                    metadata=partition,
                    count=group_count,
                )

            sync_result = self.save_adapter_fn(context, partition.partition_id) if self.save_adapter_fn else result
            adapter_path = self._adapter_path_from_result(sync_result)
            meta = self.data_plane.update_partition_status(partition.partition_id, PartitionStatus.TRAIN_DONE)
            self.lora_runtime_registry.on_train_finished(context, partition.partition_id)
            self.lora_runtime_registry.on_weight_sync_started(context)
            runtime_state = self.lora_runtime_registry.on_weight_sync_finished(context, adapter_path=adapter_path)
            if self.receive_weights_fn is not None:
                self.receive_weights_fn(runtime_state)
            self.metrics_recorder.log_event(
                event='weight_sync_done',
                phase='train',
                context=context,
                partition_id=partition.partition_id,
                policy_version=runtime_state.policy_version,
                metrics={
                    'group_count': group_count,
                    'adapter_path': adapter_path,
                },
            )
            self.data_plane.clear_partition(context, partition.partition_id)
            self.lora_runtime_registry.on_partition_cleared(context, partition.partition_id)
            self.metrics_recorder.log_event(
                event='partition_train_done',
                phase='train',
                context=context,
                partition_id=partition.partition_id,
                policy_version=runtime_state.policy_version,
                metrics={
                    'group_count': group_count,
                    'partition_train_latency_s': time.perf_counter() - start,
                },
            )
            return ComponentResult(
                component='trainer_worker',
                kind='train',
                metadata=meta,
                count=group_count,
            )
        except Exception as exc:
            if batch is not None:
                self.data_plane.mark_prompt_groups(groups, PromptGroupStatus.FAILED, extra_tag={'error': str(exc)})
                self.metrics_recorder.log_event(
                    event='train_failed',
                    phase='train',
                    context=context,
                    partition_id=partition.partition_id,
                    metrics={
                        'sample_count': len(batch.keys),
                        'error': str(exc),
                    },
                )
            self.lora_runtime_registry.on_train_finished(context, partition.partition_id)
            self.lora_runtime_registry.mark_failed(context, str(exc))
            raise

    def train_next_batch(self) -> PartitionMeta | None:
        result = self.step()
        return None if result is None else result.metadata

    def is_idle(self) -> bool:
        return not self.list_train_batch_candidates()

    def shutdown(self) -> None:
        return None

    def read_train_batch(
        self,
        batch: Any,
        data_fields: list[str] | None = None,
    ) -> TransformersTrainBatch:
        return read_train_batch(self.data_plane, batch, data_fields=data_fields)

    def list_train_batch_candidates(self, *, min_groups: int | None = None) -> list[TrainBatchCandidate]:
        candidates = []
        for partition in self.list_live_partitions():
            if partition.status in {PartitionStatus.CLEARED, PartitionStatus.TRAIN_DONE, PartitionStatus.FAILED}:
                continue
            required_groups = min_groups if min_groups is not None else self.train_batch_groups_for_context(
                partition.context)
            ready_groups = self.data_plane.list_prompt_groups(
                partition.context,
                partition_id=partition.partition_id,
                statuses=[PromptGroupStatus.ADVANTAGE_DONE],
            )
            if len(ready_groups) >= required_groups:
                candidates.append(
                    TrainBatchCandidate(
                        context=partition.context,
                        partition=partition,
                        available_groups=len(ready_groups),
                    ))
        return sorted(candidates, key=lambda candidate: (candidate.created_at, candidate.partition_id))

    def list_live_partitions(self) -> list[PartitionMeta]:
        partition_ids = sorted({
            partition_id
            for runtime_state in self.lora_runtime_registry.list_states()
            for partition_id in runtime_state.live_partitions
        })
        if not partition_ids:
            return self.data_plane.list_partitions()
        partitions = []
        for partition_id in partition_ids:
            try:
                partitions.append(self.data_plane.get_rollout_partition(partition_id))
            except KeyError:
                continue
        return sorted(partitions, key=lambda partition: (partition.created_at, partition.partition_id))

    def train_batch_groups_for_context(self, context: LoraContext) -> int:
        return int(self.train_batch_groups_by_context.get(context.key, self.train_batch_groups))

    def select_train_groups(self, context: LoraContext, partition_id: str, *, max_groups: int) -> list[Any]:
        groups = self.data_plane.list_prompt_groups(
            context,
            partition_id=partition_id,
            statuses=[PromptGroupStatus.ADVANTAGE_DONE],
        )
        if len(groups) < max_groups:
            raise LookupError(f'partition {partition_id} has only {len(groups)} train-ready groups')
        return groups[:max_groups]

    def partition_training_complete(self, context: LoraContext, partition_id: str) -> bool:
        try:
            partition = self.data_plane.get_rollout_partition(partition_id)
        except KeyError:
            return False
        if partition.context.key != context.key or partition.status != PartitionStatus.CLOSED:
            return False
        groups = self.data_plane.list_prompt_groups(context, partition_id=partition_id)
        if not groups:
            return False
        terminal_statuses = {PromptGroupStatus.TRAIN_DONE, PromptGroupStatus.FAILED, PromptGroupStatus.DROPPED}
        return all(group.status in terminal_statuses for group in groups)

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
        lora_runtime_registry: LoraRuntimeRegistry,
        scheduler: TrainerScheduler,
        model: Any,
        train_config: MultiLoraGRPOTrainConfig | None = None,
        receive_weights_fn: Callable[[Any], None] | None = None,
        train_batch_groups: int = 1,
        train_batch_groups_by_context: dict[str, int] | None = None,
        metrics_recorder: AsyncRLMetricsRecorder | None = None,
    ):
        self.model = model
        self.train_config = train_config or MultiLoraGRPOTrainConfig()
        super().__init__(
            data_plane=data_plane,
            lora_runtime_registry=lora_runtime_registry,
            scheduler=scheduler,
            train_batch_fn=self.train_batch,
            save_adapter_fn=self.save_adapter,
            receive_weights_fn=receive_weights_fn,
            train_batch_groups=train_batch_groups,
            train_batch_groups_by_context=train_batch_groups_by_context,
            metrics_recorder=metrics_recorder,
        )

    def train_batch(self, context: LoraContext, tq_batch: Any) -> TrainerStepResult:
        train_batch = self.read_train_batch(tq_batch)
        inputs = train_batch.inputs

        config = self.train_config
        kwargs = dict(config.train_kwargs or {})
        if train_batch.advantages and len(train_batch.advantages) == len(inputs):
            kwargs.setdefault('advantages', train_batch.advantages)
        if train_batch.logprobs and len(train_batch.logprobs) == len(inputs):
            kwargs.setdefault('old_logps', train_batch.logprobs)

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
        if adapter_path is not None and os.path.exists(adapter_path):
            adapter_path = os.path.abspath(adapter_path)
        return TrainerStepResult(adapter_path=adapter_path)
