# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import asyncio
import inspect
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
from .tq_utils import ROLLOUT_TRAIN_FIELDS, columns_to_tq_fields, read_train_batch, rows_to_tq_fields
from .types import (ComponentResult, GRPOAdvantageBatch, LoraContext, PartitionMeta, PartitionStatus, PromptGroupRef,
                    PromptGroupStatus, RolloutCallable, RolloutOutput, RolloutScheduleCandidate, TrainBatchCandidate,
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


def _require_rollout_logprobs(sample: dict[str, Any], *, sample_key: str) -> list[float]:
    logprobs = sample.get('logprobs')
    if not isinstance(logprobs, list):
        raise TypeError(f'rollout sample {sample_key!r} logprobs must be list[float], got {type(logprobs)!r}')

    values: list[float] = []
    for index, value in enumerate(logprobs):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f'rollout sample {sample_key!r} logprobs[{index}] must be a float, got {type(value)!r}')
        values.append(float(value))

    labels = sample.get('labels')
    if labels is not None:
        trainable_tokens = sum(1 for label in labels if label != -100)
        if len(values) != trainable_tokens:
            raise ValueError(f'rollout sample {sample_key!r} logprobs length must match trainable labels: '
                             f'{len(values)} != {trainable_tokens}')
    return values


def _rollout_sample_fields(sample: dict[str, Any]) -> dict[str, Any]:
    return {field_name: sample[field_name] for field_name in ROLLOUT_TRAIN_FIELDS if field_name in sample}


def _sample_tag(
    *,
    context: LoraContext,
    group: Any,
    sample: dict[str, Any],
    sample_key: str,
    generation_idx: int,
    logprobs: list[float],
) -> dict[str, Any]:
    tag = context.metadata()
    tag.update(dict(sample.get('metadata') or {}))
    context.validate_metadata(tag)
    for state_field in ('partition_id', 'partition_status', 'group_status', 'num_samples', 'sample_keys'):
        tag.pop(state_field, None)
    tag.update({
        'record_type': 'sample',
        'sample_status': 'success',
        'sample_id': sample.get('sample_id', sample_key),
        'group_id': group.group_id,
        'generation_idx': generation_idx,
        'rollout_policy_version': group.rollout_policy_version,
        'rollout_adapter_path': group.rollout_adapter_path,
        'logprobs_length': len(logprobs),
    })
    trainable_tokens = _trainable_token_count(sample.get('labels'))
    if trainable_tokens is not None:
        tag['trainable_tokens'] = trainable_tokens
    for sample_field, tag_field in (
        ('input_ids', 'input_length'),
        ('labels', 'label_length'),
        ('attention_mask', 'attention_length'),
    ):
        length = _safe_len(sample.get(sample_field))
        if length is not None:
            tag[tag_field] = length
    for field_name in ('stop_reason', 'truncated', 'turns'):
        if field_name in sample:
            tag[field_name] = sample[field_name]
    return tag


def _trainable_token_count(labels: Any) -> int | None:
    if labels is None:
        return None
    return sum(1 for label in labels if label != -100)


def _safe_len(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return len(value)
    except TypeError:
        return None


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
        reward_registry: dict[str, Callable[..., list[float]]] | None = None,
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
        self.reward_registry = dict(reward_registry or {})
        self.max_concurrency = max_concurrency
        self.target_groups_per_partition = target_groups_per_partition
        self.target_groups_by_context = dict(target_groups_by_context or {})
        self.num_generations = int(num_generations)
        self.num_generations_by_context = dict(num_generations_by_context or {})
        self.metrics_recorder = metrics_recorder or NoopMetricsRecorder()
        self.pending_prompt_groups_by_context: dict[str, Deque[tuple[LoraContext, Trajectory]]] = defaultdict(deque)
        self.active_tasks: set[asyncio.Task] = set()
        self._last_rollout_submit_time: dict[str, float] = defaultdict(float)
        self._submitted_prompt_groups: dict[str, int] = defaultdict(int)
        self._next_train_id_by_context: dict[str, int] = {}

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
        """Choose the next context that is allowed to submit one prompt group."""
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
        group = self.data_plane.get_prompt_group(group_ref)
        context = group.context
        tool_manager = self.tool_manager_factory.create(prompt_group, context)
        trajectory = prompt_group
        self.data_plane.update_prompt_group_status(group_ref, PromptGroupStatus.RUNNING)
        self.lora_runtime_registry.on_rollout_started(context)
        start = time.perf_counter()
        self.metrics_recorder.log_event(
            event='rollout_started',
            phase='rollout',
            context=context,
            partition_id=group_ref.partition_id,
            policy_version=group.rollout_policy_version,
            metrics={
                'group_id': group_ref.group_id,
                'inflight_rollout_groups': self.lora_runtime_registry.get(context).in_flight_groups,
                'num_generations': self._num_generations_for_context(context),
            },
        )
        try:
            num_generations = self._num_generations_for_context(context)
            rollout_kwargs = {
                'tool_manager': tool_manager,
                'adapter_name': context.adapter_name,
                'policy_version': group.rollout_policy_version,
                'num_generations': num_generations,
            }
            if group.rollout_adapter_path is not None:
                rollout_kwargs['adapter_path'] = group.rollout_adapter_path
            rollout_rows = list(await self._invoke_rollout([trajectory], rollout_kwargs))
            rewards = self._compute_rewards(context, rollout_rows)
            reward_metrics = self._compute_reward_metrics(context, rollout_rows, rewards)
            current_policy_version = self.lora_runtime_registry.get(context).policy_version
            if self.staleness_manager.is_group_too_stale(
                    rollout_policy_version=group.rollout_policy_version,
                    current_policy_version=current_policy_version,
            ):
                self.data_plane.update_prompt_group_status(
                    group_ref,
                    PromptGroupStatus.DROPPED,
                    extra_tag={'drop_reason': 'stale_after_rollout'},
                )
                self.metrics_recorder.log_event(
                    event='stale_dropped',
                    phase='rollout',
                    context=context,
                    partition_id=group_ref.partition_id,
                    policy_version=current_policy_version,
                    metrics={
                        'group_id': group_ref.group_id,
                        'rollout_policy_version': group.rollout_policy_version,
                        'policy_version_gap': current_policy_version - group.rollout_policy_version,
                        'rollout_latency_s': time.perf_counter() - start,
                    },
                )
                return self.data_plane.get_rollout_partition(group_ref.partition_id)
            meta, sample_keys = self._write_rollout_samples(
                group_ref,
                rollout_rows,
                rewards=rewards,
                expected_num_generations=num_generations,
            )
            self.data_plane.update_prompt_group_status(
                group_ref,
                PromptGroupStatus.ROLLOUT_DONE,
                sample_keys=sample_keys,
            )
            self._last_rollout_submit_time[context.key] = time.time()
            self._submitted_prompt_groups[context.key] += 1
            self.metrics_recorder.log_event(
                event='rollout_done',
                phase='rollout',
                context=context,
                partition_id=group_ref.partition_id,
                policy_version=current_policy_version,
                metrics={
                    'group_id': group_ref.group_id,
                    'sample_count': len(sample_keys),
                    'rollout_latency_s': time.perf_counter() - start,
                    'rollout_policy_version': group.rollout_policy_version,
                    'policy_version_gap': current_policy_version - group.rollout_policy_version,
                    'num_generations': num_generations,
                    **reward_metrics,
                },
            )
            return meta
        except Exception as exc:
            self.data_plane.update_prompt_group_status(
                group_ref,
                PromptGroupStatus.FAILED,
                extra_tag={'error': str(exc)},
            )
            self.metrics_recorder.log_event(
                event='rollout_failed',
                phase='rollout',
                context=context,
                partition_id=group_ref.partition_id,
                policy_version=group.rollout_policy_version,
                metrics={
                    'group_id': group_ref.group_id,
                    'rollout_latency_s': time.perf_counter() - start,
                    'error': str(exc),
                },
            )
            raise
        finally:
            self.lora_runtime_registry.on_rollout_finished(context)

    async def _invoke_rollout(
        self,
        trajectories: list[Trajectory],
        rollout_kwargs: dict[str, Any],
    ) -> Iterable[RolloutOutput]:
        """Call sync rollout implementations without blocking the event loop."""
        call = getattr(self.rollout, '__call__', None)
        if inspect.iscoroutinefunction(self.rollout) or inspect.iscoroutinefunction(call):
            return await self.rollout(trajectories, **rollout_kwargs)
        return await asyncio.to_thread(self.rollout, trajectories, **rollout_kwargs)

    def _compute_rewards(self, context: LoraContext, rollout_rows: list[RolloutOutput]) -> list[float] | None:
        reward_fn = self.reward_registry.get(context.key) or self.reward_registry.get(context.reward_type)
        if reward_fn is None:
            return None
        return list(reward_fn(rollout_rows, context=context))

    def _compute_reward_metrics(
        self,
        context: LoraContext,
        rollout_rows: list[RolloutOutput],
        rewards: list[float] | None,
    ) -> dict[str, Any]:
        reward_fn = self.reward_registry.get(context.key) or self.reward_registry.get(context.reward_type)
        metric_payload = getattr(reward_fn, 'metric_payload', None)
        if metric_payload is None:
            return {}
        return dict(metric_payload(rollout_rows, rewards=rewards, context=context))

    def _write_rollout_samples(
        self,
        group_ref: PromptGroupRef,
        samples: Iterable[RolloutOutput],
        *,
        rewards: list[float] | None = None,
        expected_num_generations: int,
    ) -> tuple[PartitionMeta, list[str]]:
        group_samples = [dict(sample) for sample in samples]
        assert expected_num_generations > 0, f'expected_num_generations must be positive, got {expected_num_generations}'
        assert len(group_samples) == expected_num_generations, (
            f'group {group_ref.group_id} expected {expected_num_generations} rollout samples, '
            f'got {len(group_samples)}')
        if rewards is not None:
            assert len(rewards) == len(group_samples), (
                f'reward count {len(rewards)} does not match sample count {len(group_samples)}')
        group = self.data_plane.get_prompt_group(group_ref)
        if group.status not in {PromptGroupStatus.PENDING, PromptGroupStatus.RUNNING}:
            raise ValueError(f'group {group.group_id} is not pending/running: {group.status}')
        partition = self.data_plane.get_rollout_partition(group_ref.partition_id)

        sample_keys: list[str] = []
        sample_fields: list[dict[str, Any]] = []
        sample_tags: list[dict[str, Any]] = []
        generation_indices: list[int] = []
        reward_iter = iter(rewards or [])
        for sample_index, trajectory in enumerate(group_samples):
            sample = dict(trajectory)
            if rewards is not None:
                sample['rewards'] = next(reward_iter)
            generation_idx = int(sample.get('generation_idx', sample_index))
            key = f'samples/{group_ref.group_id}/{generation_idx}'
            if key in sample_keys:
                raise ValueError(f'duplicate rollout sample key {key!r}')
            generation_indices.append(generation_idx)
            logprobs = _require_rollout_logprobs(sample, sample_key=key)
            sample['logprobs'] = logprobs
            fields = _rollout_sample_fields(sample)
            tag = _sample_tag(
                context=group.context,
                group=group,
                sample=sample,
                sample_key=key,
                generation_idx=generation_idx,
                logprobs=logprobs,
            )
            sample_keys.append(key)
            sample_fields.append(fields)
            sample_tags.append(tag)

        assert generation_indices == list(range(expected_num_generations)), (
            f'group {group.group_id} generation_idx must be 0..{expected_num_generations - 1} in order, '
            f'got {generation_indices}')

        self.data_plane.write_sample_batch(
            partition_id=group_ref.partition_id,
            keys=sample_keys,
            fields=rows_to_tq_fields(sample_fields),
            tags=sample_tags,
        )
        return partition, sample_keys

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
            try:
                group_ref = self.data_plane.create_prompt_group(
                    context,
                    partition,
                    runtime_state=runtime_state,
                )
            except LookupError:
                continue
            prompt_group = self._pop_prompt_group(context)
            task = asyncio.create_task(self._run_prompt_group_rollout(group_ref, prompt_group))
            self.active_tasks.add(task)
            self._close_partition_if_full(partition)
            submitted_groups += 1
            self.metrics_recorder.log_event(
                event='rollout_submitted',
                phase='rollout',
                context=context,
                partition_id=partition.partition_id,
                policy_version=runtime_state.policy_version,
                metrics={
                    'submitted_groups': submitted_groups,
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
                # Group-level failure is already persisted by _run_prompt_group_rollout.
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

    async def step(self) -> ComponentResult | None:
        completed = await self._collect_finished_tasks()
        submitted_groups = self._submit_prompt_group_tasks()
        if completed or submitted_groups:
            return ComponentResult(component='rollouter', kind='rollout', count=completed + submitted_groups)
        return None

    def is_idle(self) -> bool:
        return not any(self.pending_prompt_groups_by_context.values()) and not self.active_tasks

    def shutdown(self) -> None:
        for task in list(self.active_tasks):
            task.cancel()


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

    def step(self) -> ComponentResult | None:
        if self.lora_runtime_registry is None:
            return None
        count = 0
        last_meta = None
        for context in self.contexts:
            max_groups = self.batch_size_for_context(context)
            partition_ids = sorted(self.lora_runtime_registry.get(context).live_partitions)
            for partition_id in partition_ids:
                groups = self._select_advantage_groups(context, partition_id=partition_id, max_groups=max_groups)
                if not groups:
                    continue
                last_meta = self.process_advantage_batch(context, partition_id=partition_id, groups=groups)
                count += len(groups)
        if last_meta is None:
            return None
        return ComponentResult(component='advantage_worker', kind='advantage', metadata=last_meta, count=count)

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
            return None
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
        return TrainerStepResult(adapter_path=adapter_path)
