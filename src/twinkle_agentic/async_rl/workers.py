# Copyright (c) ModelScope Contributors. All rights reserved.
"""Long-lived, context-scheduling async-RL workers."""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from .context_manager import ContextStatus, LoraContextManager
from .data_plane import TQDataPlane
from .metrics import MetricsBuffer, training_policy_metrics
from .scheduler import ContextScheduler, ScheduleCandidate, SchedulerConfig
from .types import LoraContext, MetricEvent, PartitionAdmission, PreparedPartition


class _Worker:
    """A long-lived Ray service with one privately owned background loop."""

    def __init__(self):
        self._service_task: asyncio.Task[None] | None = None
        self._stop_requested = False
        self._failure: str | None = None

    async def start(self) -> None:
        if self._service_task is not None and not self._service_task.done():
            return
        self._stop_requested = False
        self._failure = None
        self._service_task = asyncio.create_task(self._run_service())

    async def stop(self) -> None:
        self._stop_requested = True
        if self._service_task is None or self._service_task.done():
            return
        self._service_task.cancel()
        try:
            await self._service_task
        except asyncio.CancelledError:
            pass

    async def get_service_state(self) -> dict[str, str | bool | None]:
        return {
            'running': self._service_task is not None and not self._service_task.done(),
            'failure': self._failure,
        }

    async def _run_service(self) -> None:
        try:
            await self._serve()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self._failure = f'{type(exc).__name__}: {exc}'

    async def _serve(self) -> None:
        raise NotImplementedError


class RolloutWorker(_Worker):
    """Admits full prompt batches and submits them to the sampler without waiting."""

    def __init__(self,
                 *,
                 context_manager: LoraContextManager,
                 data_plane: TQDataPlane,
                 sampler: Any,
                 prompt_batches: dict[str, Iterable[Sequence[dict[str, Any]]]
                                      | Callable[[], Iterable[Sequence[dict[str, Any]]]]],
                 rollout_config: dict[str, dict[str, Any]],
                 scheduler: SchedulerConfig,
                 allow_partial_rollout: bool = False,
                 idle_delay_s: float = 0.05):
        super().__init__()
        self.data_plane = data_plane
        self.sampler = sampler
        self.context_manager = context_manager
        self.idle_delay_s = idle_delay_s
        self.metrics = MetricsBuffer()
        self.rollout_config = rollout_config
        self.scheduler = ContextScheduler(scheduler)
        self.allow_partial_rollout = allow_partial_rollout
        self._prompt_batch_iterators = {
            key: iter(value() if callable(value) else value)
            for key, value in prompt_batches.items()
        }
        self._next_batch_tasks: dict[str, asyncio.Task[Sequence[dict[str, Any]] | None]] = {}

    def drain_metrics(self) -> list[MetricEvent]:
        return self.metrics.drain()

    def _record(self,
                event: str,
                context: LoraContext | None,
                partition_id: str | None,
                metrics: dict[str, Any],
                policy_version: int | None = None) -> None:
        self.metrics.record(event, context, partition_id, metrics, policy_version)

    def _start_next_batch(self, key: str) -> None:
        if key not in self._next_batch_tasks:
            self._next_batch_tasks[key] = asyncio.create_task(
                asyncio.to_thread(next, self._prompt_batch_iterators[key], None))

    async def stop(self) -> None:
        await super().stop()
        pending = list(self._next_batch_tasks.values())
        self._next_batch_tasks.clear()
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def _serve(self) -> None:
        exhausted: set[str] = set()
        for key in self._prompt_batch_iterators:
            self._start_next_batch(key)
        while not self._stop_requested:
            if await self.context_manager.is_rollout_admission_closed.remote():
                return
            candidates = []
            for key in self._prompt_batch_iterators:
                if key in exhausted:
                    continue
                if await self.context_manager.context_status.remote(key) is not ContextStatus.ACTIVE:
                    exhausted.add(key)
                    continue
                config = self.rollout_config[key]
                if self._next_batch_tasks[key].done():
                    candidates.append(ScheduleCandidate(config['context']))
            candidate = self.scheduler.choose(candidates)
            if candidate is None:
                if len(exhausted) == len(self._prompt_batch_iterators):
                    return
                await asyncio.sleep(self.idle_delay_s)
                continue
            key = candidate.context.key
            config = self.rollout_config[key]
            batch_task = self._next_batch_tasks[key]
            try:
                batch = batch_task.result()
            except Exception as exc:
                self._next_batch_tasks.pop(key)
                self._record('rollout_failed', candidate.context, None, {'error': f'prompt loading failed: {exc}'})
                raise RuntimeError(f'prompt loading failed for {key}: {exc}') from exc
            if batch is None or len(batch) != int(config['batch_size']):
                self._next_batch_tasks.pop(key)
                exhausted.add(key)
                await self.context_manager.on_dataset_exhausted.remote(candidate.context)
                self.scheduler.on_blocked(candidate)
                continue
            admission = await self.context_manager.request_rollout_partition.remote(
                candidate.context,
                target_groups=len(batch),
                num_generations=int(config['num_generations']),
            )
            if admission is None:
                self.scheduler.on_blocked(candidate)
                await asyncio.sleep(self.idle_delay_s)
                continue
            self._next_batch_tasks.pop(key)
            try:
                prepared = await self.data_plane.prepare_rollout_partition(
                    admission,
                    list(batch),
                    config['sampling_params'],
                )
                await asyncio.to_thread(
                    self.sampler.sample,
                    list(prepared.groups),
                    prepared.sampling_params,
                    self.allow_partial_rollout,
                )
            except Exception as exc:
                self._record(
                    'rollout_failed',
                    admission.context,
                    admission.partition_id,
                    {'error': str(exc)},
                )
                raise RuntimeError(f'rollout submission failed for {admission.partition_id}: {exc}') from exc
            self._start_next_batch(key)
            self.scheduler.on_success(candidate)
            self._record('rollout_submitted', admission.context, admission.partition_id, {
                'prompt_count': admission.target_groups,
                'sample_count': admission.sample_count
            })


class AdvantageWorker(_Worker):

    def __init__(self,
                 *,
                 context_manager: LoraContextManager,
                 data_plane: TQDataPlane,
                 advantage_fn: Callable[[Any, PartitionAdmission], tuple[Sequence[float], Sequence[float]]],
                 groups_per_batch: dict[str, int],
                 scheduler: SchedulerConfig,
                 idle_delay_s: float = 0.05):
        super().__init__()
        self.data_plane = data_plane
        self.context_manager = context_manager
        self.idle_delay_s = idle_delay_s
        self.metrics = MetricsBuffer()
        self.advantage_fn = advantage_fn
        self.groups_per_batch = groups_per_batch
        self.scheduler = ContextScheduler(scheduler)

    def drain_metrics(self) -> list[MetricEvent]:
        return self.metrics.drain()

    def _record(self,
                event: str,
                context: LoraContext | None,
                partition_id: str | None,
                metrics: dict[str, Any],
                policy_version: int | None = None) -> None:
        self.metrics.record(event, context, partition_id, metrics, policy_version)

    async def _serve(self) -> None:
        while not self._stop_requested and not await self.context_manager.is_run_finished.remote():
            leases = await self.context_manager.list_live_partitions.remote()
            blocked: set[str] = set()
            progressed = False
            for _ in range(len(leases)):
                candidates = [
                    ScheduleCandidate(admission.context, admission) for admission in leases
                    if admission.partition_id not in blocked
                ]
                candidate = self.scheduler.choose(candidates)
                if candidate is None:
                    break
                admission = candidate.partition
                batch = await self.data_plane.claim_advantage_batch(admission,
                                                                    self.groups_per_batch[admission.context.key])
                if batch is None:
                    blocked.add(admission.partition_id)
                    self.scheduler.on_blocked(candidate)
                    continue
                started = time.perf_counter()
                try:
                    advantages, returns = self.advantage_fn(batch.data, admission)
                    await self.data_plane.write_advantages(batch, advantages=advantages, returns=returns)
                except Exception as exc:
                    self._record('advantage_failed', admission.context, admission.partition_id, {'error': str(exc)})
                    raise RuntimeError(f'advantage failed for {admission.partition_id}: {exc}') from exc
                self.scheduler.on_success(candidate)
                policy = await self.context_manager.get_rollout_policy.remote(admission.context)
                self._record('advantage_done', admission.context, admission.partition_id, {
                    'sample_count': len(advantages),
                    'advantage_latency_s': time.perf_counter() - started
                }, policy.version)
                progressed = True
                break
            if not progressed:
                await asyncio.sleep(self.idle_delay_s)


class TrainerWorker(_Worker):

    def __init__(self,
                 *,
                 context_manager: LoraContextManager,
                 data_plane: TQDataPlane,
                 train_fn: Callable[[Any, PartitionAdmission], dict[str, Any] | None],
                 save_adapter: Callable[[PartitionAdmission], str],
                 groups_per_batch: dict[str, int],
                 scheduler: SchedulerConfig,
                 keep_adapter_versions: int = 0,
                 initial_adapter_paths: dict[str, str] | None = None,
                 remove_adapter: Callable[[str], None] | None = None,
                 idle_delay_s: float = 0.05):
        super().__init__()
        self.data_plane = data_plane
        self.context_manager = context_manager
        self.idle_delay_s = idle_delay_s
        self.metrics = MetricsBuffer()
        self.train_fn = train_fn
        self.save_adapter = save_adapter
        self.groups_per_batch = groups_per_batch
        self.scheduler = ContextScheduler(scheduler)
        self.keep_adapter_versions = max(0, int(keep_adapter_versions))
        self._adapter_history: dict[str, list[str]] = defaultdict(list)
        for context_key, path in (initial_adapter_paths or {}).items():
            if path:
                self._adapter_history[context_key].append(path)
        self.remove_adapter = remove_adapter or _remove_local_adapter

    def drain_metrics(self) -> list[MetricEvent]:
        return self.metrics.drain()

    def _record(self,
                event: str,
                context: LoraContext | None,
                partition_id: str | None,
                metrics: dict[str, Any],
                policy_version: int | None = None) -> None:
        self.metrics.record(event, context, partition_id, metrics, policy_version)

    async def _serve(self) -> None:
        while not self._stop_requested and not await self.context_manager.is_run_finished.remote():
            leases = await self.context_manager.list_trainable_partitions.remote()
            blocked: set[str] = set()
            progressed = False
            for _ in range(len(leases)):
                candidates = [
                    ScheduleCandidate(admission.context, admission) for admission in leases
                    if admission.partition_id not in blocked
                ]
                candidate = self.scheduler.choose(candidates)
                if candidate is None:
                    break
                admission = candidate.partition
                batch = await self.data_plane.claim_training_batch(admission,
                                                                   self.groups_per_batch[admission.context.key])
                if batch is None:
                    if await self.data_plane.is_training_consumed(admission):
                        try:
                            await self.context_manager.on_partition_training_started.remote(admission)
                            await self._finish_partition(admission)
                            self.scheduler.on_success(candidate)
                            progressed = True
                            break
                        except Exception as exc:
                            self._record('train_failed', admission.context, admission.partition_id, {'error': str(exc)})
                            raise RuntimeError(
                                f'training completion failed for {admission.partition_id}: {exc}') from exc
                    blocked.add(admission.partition_id)
                    self.scheduler.on_blocked(candidate)
                    continue
                try:
                    await self.context_manager.on_partition_training_started.remote(admission)
                    policy = await self.context_manager.get_rollout_policy.remote(admission.context)
                    started = time.perf_counter()
                    metrics = dict(self.train_fn(batch.data, admission) or {})
                except Exception as exc:
                    self._record('train_failed', admission.context, admission.partition_id, {'error': str(exc)})
                    raise RuntimeError(f'training failed for {admission.partition_id}: {exc}') from exc
                metrics.setdefault('sample_count', len(batch.data['input_ids']))
                metrics['train_latency_s'] = time.perf_counter() - started
                metrics.update(training_policy_metrics(batch.sample_tags, policy.version))
                self.scheduler.on_success(candidate)
                self._record('train_step_done', admission.context, admission.partition_id, metrics, policy.version)
                progressed = True
                break
            if not progressed:
                await asyncio.sleep(self.idle_delay_s)

    async def _finish_partition(self, admission: PartitionAdmission) -> None:
        adapter_path = self.save_adapter(admission)
        policy = await self.context_manager.on_partition_trained.remote(admission, adapter_path=adapter_path)
        self._record('policy_published', admission.context, admission.partition_id, {'adapter_path': adapter_path},
                     policy.version)
        await self.data_plane.clear_partition(admission)
        await self.context_manager.on_partition_cleared.remote(admission)
        self._adapter_history[admission.context.key].append(adapter_path)
        await self._prune_adapter_history(admission.context.key)
        self._record('partition_done', admission.context, admission.partition_id, {}, policy.version)

    async def _prune_adapter_history(self, context_key: str) -> None:
        protected = set(await self.context_manager.adapter_paths_to_keep.remote())
        history = self._adapter_history[context_key]
        retained_history = set(history[-self.keep_adapter_versions:]) if self.keep_adapter_versions else set()
        retained = protected | retained_history
        stale = [path for path in history if path not in retained]
        for path in stale:
            self.remove_adapter(path)
        self._adapter_history[context_key] = [path for path in history if path in retained]


def _remove_local_adapter(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
