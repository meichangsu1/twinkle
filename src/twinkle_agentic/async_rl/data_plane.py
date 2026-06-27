# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set

from .types import PartitionMetadata, PartitionStatus, QueueMetadata, SampleRecord, TaskName, TrainingContext


@dataclass
class TransferQueueRuntimeConfig:
    """TransferQueue initialization and capacity guard config.
    Capacity auto-calculation:
      samples_per_partition = target_groups * num_generations
      max_live_partitions = max_staleness + 1
      max_rows = samples_per_partition * max_live_partitions
      max_tq_bytes = estimate_bytes_per_sample * max_rows * safety_factor
    """

    # TQ backend
    total_storage_size: Optional[int] = None
    num_data_storage_units: int = 4
    storage_backend: str = 'SimpleStorage'
    controller: Dict[str, Any] = field(default_factory=dict)
    backend: Dict[str, Any] = field(default_factory=dict)
    init: bool = True

    # Capacity planning inputs
    target_groups: int = 128
    num_generations: int = 8
    max_staleness: int = 1
    estimate_bytes_per_sample: Optional[int] = None
    safety_factor: float = 1.2

    # Capacity guard thresholds (None = auto-calculate in __post_init__)
    max_rows: Optional[int] = None
    max_rows_per_context: Optional[int] = None
    max_tq_bytes: Optional[int] = None
    max_live_partitions_per_context: Optional[int] = None

    # Runtime
    lease_timeout: float = 300.0

    def __post_init__(self):
        samples_per_partition = self.target_groups * self.num_generations
        max_live_partitions = self.max_staleness + 1

        if self.max_rows is None:
            self.max_rows = samples_per_partition * max_live_partitions

        if self.max_rows_per_context is None:
            self.max_rows_per_context = self.max_rows

        if self.max_tq_bytes is None and self.estimate_bytes_per_sample is not None:
            self.max_tq_bytes = int(self.estimate_bytes_per_sample * self.max_rows * self.safety_factor)

        if self.max_live_partitions_per_context is None:
            self.max_live_partitions_per_context = max_live_partitions


class TransferQueueDataPlane:
    """The only data-plane boundary for async RL TransferQueue access."""

    def __init__(self, tq_client: Optional[Any] = None, tq_config: Optional[TransferQueueRuntimeConfig] = None):
        self.tq_config = tq_config or TransferQueueRuntimeConfig()
        self.tq = tq_client or self._init_transfer_queue(self.tq_config)
        self._meta: Dict[str, PartitionMetadata] = {}
        self._next_train_id: Dict[str, int] = defaultdict(int)
        self._consumed: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self._leases: Dict[str, Dict[str, Any]] = {}
        self._trainer_steps: Dict[str, int] = defaultdict(int)
        self._lock = threading.RLock()

    def _init_transfer_queue(self, config: TransferQueueRuntimeConfig):
        try:
            import transfer_queue as tq
        except ImportError as exc:
            raise RuntimeError(
                'transfer_queue is required for TransferQueueDataPlane. '
                'Pass an explicit tq_client only in unit tests/local mocks.'
            ) from exc
        if config.init:
            tq.init(self._build_tq_config(config))
        return tq

    @staticmethod
    def _build_tq_config(config: TransferQueueRuntimeConfig):
        try:
            from omegaconf import OmegaConf
        except ImportError as exc:
            raise RuntimeError('omegaconf is required to initialize transfer_queue config') from exc
        backend_config = dict(config.backend)
        simple_storage = dict(backend_config.get('SimpleStorage') or {})
        simple_storage.setdefault('num_data_storage_units', config.num_data_storage_units)
        if config.total_storage_size is not None:
            simple_storage.setdefault('total_storage_size', config.total_storage_size)
        elif config.max_tq_bytes is not None:
            simple_storage.setdefault('total_storage_size', config.max_tq_bytes)
        backend_config.setdefault('storage_backend', config.storage_backend)
        backend_config['SimpleStorage'] = simple_storage
        return OmegaConf.create(
            {'controller': config.controller, 'backend': backend_config},
            flags={'allow_objects': True},
        )

    def close(self) -> None:
        if hasattr(self.tq, 'close'):
            self.tq.close()

    def init_namespace(self, context: TrainingContext) -> None:
        context.metadata()

    def next_partition_id(self, context: TrainingContext) -> str:
        with self._lock:
            self._load_partition_meta()
            train_id = self._next_train_id[context.key]
            while context.partition_id(train_id) in self._meta:
                train_id += 1
            self._next_train_id[context.key] = train_id + 1
            return context.partition_id(train_id)

    def create_partition(
        self,
        context: TrainingContext,
        *,
        target_groups: int,
        partition_id: Optional[str] = None,
    ) -> PartitionMetadata:
        partition_id = partition_id or self.next_partition_id(context)
        with self._lock:
            self._load_partition_meta()
            existing = self._meta.get(partition_id)
            if existing is not None:
                raise ValueError(f'partition already exists: {partition_id}')
            meta = PartitionMetadata(
                context=context,
                partition_id=partition_id,
                policy_version=context.policy_version,
                target_groups=target_groups,
            )
            self._meta[partition_id] = meta
        return meta

    def put_rollout_batch(
        self,
        context: TrainingContext,
        partition_id: str,
        trajectories: List[SampleRecord],
        *,
        ready_groups: int = 1,
        seal: bool = False,
    ) -> PartitionMetadata:
        with self._lock:
            meta = self._meta.get(partition_id)
            if meta is None:
                meta = self.create_partition(context, target_groups=ready_groups, partition_id=partition_id)
            if meta.context.key != context.key:
                raise ValueError(f'partition {partition_id} belongs to {meta.context.key}, not {context.key}')
            if meta.status != PartitionStatus.OPEN:
                raise ValueError(f'partition {partition_id} is not open for rollout append: {meta.status}')
            keys = []
            all_fields = []
            all_tags = []
            for idx, trajectory in enumerate(trajectories):
                sample = dict(trajectory)
                sample_meta = dict(sample.get('metadata') or {})
                sample_meta.update(context.metadata())
                sample_meta.setdefault('partition_id', partition_id)
                context.validate_metadata(sample_meta, strict_policy_version=False)
                key = sample.get('sample_id') or f'{partition_id}/sample_{meta.num_rows + idx}'
                fields = {k: v for k, v in sample.items() if k not in {'metadata', 'sample_id'}}
                tag = dict(sample_meta)
                tag.update(meta.tag())
                keys.append(key)
                all_fields.append(fields)
                all_tags.append(tag)
            if keys:
                self._kv_batch_put(keys, partition_id, all_fields, all_tags)
            meta.num_rows += len(keys)
            meta.ready_groups += ready_groups
            if seal or (meta.target_groups and meta.ready_groups >= meta.target_groups):
                meta.status = PartitionStatus.ROLLOUT_DONE
            meta.touch()
            self._meta[partition_id] = meta
            self._sync_partition_status(meta)
            return meta

    def list_partitions(
        self,
        context: Optional[TrainingContext] = None,
        *,
        statuses: Optional[Iterable[PartitionStatus]] = None,
    ) -> list[PartitionMetadata]:
        self._load_partition_meta()
        status_set = set(statuses) if statuses is not None else None
        with self._lock:
            partitions = list(self._meta.values())
        if context is not None:
            partitions = [p for p in partitions if p.context.key == context.key]
        if status_set is not None:
            partitions = [p for p in partitions if p.status in status_set]
        return sorted(partitions, key=lambda p: (p.created_at, p.partition_id))

    def get_metadata(self, context: Optional[TrainingContext] = None) -> QueueMetadata | list[PartitionMetadata]:
        """Return QueueMetadata if context is specified, else list of all partitions.

        When called with a context, returns a QueueMetadata object containing
        aggregate information for that scope.
        When called without context, returns the raw partition list for backward compat.
        """
        partitions = self.list_partitions(context)
        if context is not None:
            active = [p for p in partitions if p.status not in (PartitionStatus.CLEARED, PartitionStatus.CANCELLED)]
            total_rows = sum(p.num_rows for p in active)
            return QueueMetadata(
                context=context,
                active_partitions=active,
                total_rows=total_rows,
                trainer_step=self._trainer_steps.get(context.key, 0),
                current_policy_version=context.policy_version,
            )
        return partitions

    def check_capacity(self, context: TrainingContext) -> bool:
        live_partitions = [p for p in self.list_partitions() if p.status != PartitionStatus.CLEARED]
        total_rows = sum(p.num_rows for p in live_partitions)
        context_rows = sum(p.num_rows for p in live_partitions if p.context.key == context.key)
        if total_rows >= self.tq_config.max_rows:
            return False
        if context_rows >= self.tq_config.max_rows_per_context:
            return False
        context_live = len([p for p in live_partitions if p.context.key == context.key])
        if context_live >= self.tq_config.max_live_partitions_per_context:
            return False
        return True

    def claim_reward_batch(
        self,
        context: TrainingContext,
        batch_size: int,
        *,
        worker_id: Optional[str] = None,
    ) -> tuple[PartitionMetadata, list[SampleRecord]]:
        return self._claim_samples(context, batch_size, [PartitionStatus.ROLLOUT_DONE], TaskName.REWARD, worker_id=worker_id)

    def claim_reward_ready_groups(
        self,
        context: TrainingContext,
        num_generations: int,
        max_groups: int,
        *,
        worker_id: Optional[str] = None,
    ) -> tuple[PartitionMetadata, list[list[SampleRecord]]]:
        """Claim reward-ready data at the group level.

        Finds the first ROLLOUT_DONE partition for the given context, groups
        samples by group_id, and returns up to max_groups complete groups.
        Each group contains exactly num_generations samples.

        Returns:
            (PartitionMetadata, list of groups) where each group is a
            list[SampleRecord] of num_generations trajectories.
        """
        partitions = self.list_partitions(context, statuses=[PartitionStatus.ROLLOUT_DONE])
        if not partitions:
            raise LookupError(f'no reward-ready partition for {context.key}')
        meta = partitions[0]
        if worker_id is not None:
            self.claim_partition_with_lease(context, meta.partition_id, worker_id=worker_id)
        samples = self._get_samples(meta.partition_id)
        groups: Dict[str, list[SampleRecord]] = defaultdict(list)
        for sample in samples:
            gid = sample.get('group_id', sample['sample_id'])
            groups[gid].append(sample)
        result_groups = []
        for gid, group_samples in groups.items():
            if len(group_samples) >= num_generations:
                result_groups.append(group_samples[:num_generations])
            if len(result_groups) >= max_groups:
                break
        return meta, result_groups

    def append_rewards(
        self,
        context: TrainingContext,
        partition_id: str,
        rewards: list[float],
        *,
        field_name: str = 'rewards',
    ) -> PartitionMetadata:
        meta = self._meta.get(partition_id)
        if meta is not None and meta.context.key != context.key:
            raise ValueError(f'partition {partition_id} belongs to {meta.context.key}, not {context.key}')
        samples = self._get_samples(partition_id)
        if len(rewards) != len(samples):
            raise ValueError(f'reward count {len(rewards)} does not match sample count {len(samples)}')
        updates = {}
        for sample, reward in zip(samples, rewards):
            context.validate_metadata(sample.get('metadata') or {}, strict_policy_version=False)
            updates[sample['sample_id']] = {field_name: reward}
        self._batch_update_samples(partition_id, updates)
        meta = self._meta[partition_id]
        meta.status = PartitionStatus.REWARD_DONE
        meta.touch()
        self._sync_partition_status(meta)
        return meta

    def claim_advantage_batch(
        self,
        context: TrainingContext,
        batch_size: int,
        *,
        worker_id: Optional[str] = None,
    ) -> tuple[PartitionMetadata, list[SampleRecord]]:
        return self._claim_samples(
            context, batch_size, [PartitionStatus.REWARD_DONE], TaskName.ADVANTAGE, worker_id=worker_id,
        )

    def append_advantages(
        self,
        context: TrainingContext,
        partition_id: str,
        advantages: list[float],
        returns: Optional[list[float]] = None,
    ) -> PartitionMetadata:
        meta = self._meta.get(partition_id)
        if meta is not None and meta.context.key != context.key:
            raise ValueError(f'partition {partition_id} belongs to {meta.context.key}, not {context.key}')
        samples = self._get_samples(partition_id)
        if len(advantages) != len(samples):
            raise ValueError(f'advantage count {len(advantages)} does not match sample count {len(samples)}')
        if returns is None:
            returns = advantages
        updates = {}
        for sample, advantage, ret in zip(samples, advantages, returns):
            context.validate_metadata(sample.get('metadata') or {}, strict_policy_version=False)
            updates[sample['sample_id']] = {'advantages': advantage, 'returns': ret}
        self._batch_update_samples(partition_id, updates)
        meta = self._meta[partition_id]
        meta.status = PartitionStatus.TRAIN_READY
        meta.touch()
        self._sync_partition_status(meta)
        return meta

    def list_train_ready_partitions(self) -> list[PartitionMetadata]:
        return self.list_partitions(statuses=[PartitionStatus.TRAIN_READY])

    def mark_training(self, context: TrainingContext, partition_id: str) -> PartitionMetadata:
        return self._mark_status(context, partition_id, PartitionStatus.TRAINING)

    def mark_trained(self, context: TrainingContext, partition_id: str) -> PartitionMetadata:
        with self._lock:
            self._trainer_steps[context.key] = self._trainer_steps.get(context.key, 0) + 1
        return self._mark_status(context, partition_id, PartitionStatus.TRAIN_DONE)

    def build_streaming_dataset(
        self,
        context: TrainingContext,
        partition_id: str,
        *,
        batch_size: int = 32,
        data_fields: Optional[List[str]] = None,
        task_name: Optional[str] = None,
        dp_rank: int = 0,
    ) -> '_StreamingDatasetWrapper':
        """Build a streaming dataset that wraps TQ Client API for batch iteration.

        Uses TQ Client API (get_meta + get_data) to stream batches from the partition.
        Each iteration yields a list[SampleRecord] batch and auto-acks consumed samples.

        Args:
            context: Training context for partition ownership validation
            partition_id: Partition to stream from
            batch_size: Number of samples per batch
            data_fields: Fields to retrieve (None = all fields)
            task_name: Task name for consumption tracking (defaults to TASK_TRAIN)
            dp_rank: Data parallel rank for distributed training

        Returns:
            _StreamingDatasetWrapper that yields list[SampleRecord] batches
        """
        if task_name is None:
            task_name = TaskName.TRAIN
        meta = self._meta[partition_id]
        if meta.context.key != context.key:
            raise ValueError(f'partition {partition_id} belongs to {meta.context.key}, not {context.key}')
        return _StreamingDatasetWrapper(
            data_plane=self,
            context=context,
            partition_id=partition_id,
            batch_size=batch_size,
            data_fields=data_fields,
            task_name=task_name,
            dp_rank=dp_rank,
        )

    def build_streaming_dataloader(
        self,
        context: TrainingContext,
        partition_id: str,
        *,
        task_name: Optional[str] = None,
        required_fields: Optional[frozenset] = None,
    ) -> list[SampleRecord]:
        meta = self._meta[partition_id]
        if meta.context.key != context.key:
            raise ValueError(f'partition {partition_id} belongs to {meta.context.key}, not {context.key}')
        samples = self._get_samples(partition_id)
        if task_name is not None:
            consumed = self._consumed[partition_id].get(task_name, set())
            samples = [s for s in samples if s['sample_id'] not in consumed]
        if required_fields is not None:
            incomplete = []
            for s in samples:
                missing = required_fields - set(s.keys())
                if missing:
                    incomplete.append((s['sample_id'], missing))
            if incomplete:
                raise ValueError(
                    f'{len(incomplete)} samples missing required fields: '
                    f'{incomplete[0][0]} missing {incomplete[0][1]}'
                )
        return samples

    def ack_rows(
        self,
        context: TrainingContext,
        partition_id: str,
        sample_ids: List[str],
        *,
        task_name: Optional[str] = None,
    ) -> int:
        if task_name is None:
            task_name = TaskName.TRAIN
        meta = self._meta.get(partition_id)
        if meta is not None and meta.context.key != context.key:
            raise ValueError(f'partition {partition_id} belongs to {meta.context.key}, not {context.key}')
        with self._lock:
            consumed = self._consumed[partition_id][task_name]
            before = len(consumed)
            consumed.update(sample_ids)
            return len(consumed) - before

    def get_consumed_count(self, partition_id: str, *, task_name: Optional[str] = None) -> int:
        if task_name is None:
            task_name = TaskName.TRAIN
        return len(self._consumed[partition_id].get(task_name, set()))

    def claim_partition_with_lease(
        self,
        context: TrainingContext,
        partition_id: str,
        *,
        worker_id: str,
        timeout: Optional[float] = None,
    ) -> PartitionMetadata:
        lease_timeout = timeout if timeout is not None else self.tq_config.lease_timeout
        with self._lock:
            self._recover_expired_leases()
            existing = self._leases.get(partition_id)
            if existing is not None:
                if existing['worker_id'] != worker_id:
                    raise RuntimeError(
                        f'partition {partition_id} is leased by {existing["worker_id"]} '
                        f'until {existing["deadline"]}'
                    )
            meta = self._meta.get(partition_id)
            if meta is not None and meta.context.key != context.key:
                raise ValueError(f'partition {partition_id} belongs to {meta.context.key}, not {context.key}')
            self._leases[partition_id] = {
                'worker_id': worker_id,
                'deadline': time.time() + lease_timeout,
                'context_key': context.key,
            }
            if meta is not None:
                meta.owner_worker_id = worker_id
                meta.lease_deadline = self._leases[partition_id]['deadline']
                meta.touch()
            return meta

    def release_lease(self, partition_id: str, *, worker_id: str) -> None:
        with self._lock:
            existing = self._leases.get(partition_id)
            if existing is None:
                return
            if existing['worker_id'] != worker_id:
                raise RuntimeError(
                    f'partition {partition_id} is leased by {existing["worker_id"]}, not {worker_id}'
                )
            del self._leases[partition_id]
            meta = self._meta.get(partition_id)
            if meta is not None:
                meta.owner_worker_id = None
                meta.lease_deadline = None

    def renew_lease(self, partition_id: str, *, worker_id: str, timeout: Optional[float] = None) -> None:
        lease_timeout = timeout if timeout is not None else self.tq_config.lease_timeout
        with self._lock:
            existing = self._leases.get(partition_id)
            if existing is None:
                raise RuntimeError(f'no active lease on partition {partition_id}')
            if existing['worker_id'] != worker_id:
                raise RuntimeError(
                    f'partition {partition_id} is leased by {existing["worker_id"]}, not {worker_id}'
                )
            existing['deadline'] = time.time() + lease_timeout
            meta = self._meta.get(partition_id)
            if meta is not None:
                meta.lease_deadline = existing['deadline']

    def clear_partition(self, context: TrainingContext, partition_id: str) -> None:
        meta = self._meta.get(partition_id)
        if meta is not None and meta.context.key != context.key:
            raise ValueError(f'partition {partition_id} belongs to {meta.context.key}, not {context.key}')
        keys = list(self.tq.kv_list(partition_id=partition_id).get(partition_id, {}))
        if keys:
            self.tq.kv_clear(keys=keys, partition_id=partition_id)
        if meta is not None:
            meta.status = PartitionStatus.CLEARED
            meta.touch()
        with self._lock:
            self._consumed.pop(partition_id, None)
            self._leases.pop(partition_id, None)

    def clear_namespace(self, context: TrainingContext) -> int:
        partitions = self.list_partitions(context)
        cleared = 0
        for partition in partitions:
            if partition.status != PartitionStatus.CLEARED:
                self.clear_partition(context, partition.partition_id)
                cleared += 1
        return cleared

    def _claim_samples(
        self,
        context: TrainingContext,
        batch_size: int,
        statuses: Iterable[PartitionStatus],
        task_name: str,
        *,
        worker_id: Optional[str] = None,
    ) -> tuple[PartitionMetadata, list[SampleRecord]]:
        partitions = self.list_partitions(context, statuses=statuses)
        if not partitions:
            raise LookupError(f'no {task_name}-ready partition for {context.key}')
        meta = partitions[0]
        if worker_id is not None:
            self.claim_partition_with_lease(context, meta.partition_id, worker_id=worker_id)
        return meta, self._get_samples(meta.partition_id)[:batch_size]

    def _mark_status(self, context: TrainingContext, partition_id: str, status: PartitionStatus) -> PartitionMetadata:
        meta = self._meta[partition_id]
        if meta.context.key != context.key:
            raise ValueError(f'partition {partition_id} belongs to {meta.context.key}, not {context.key}')
        meta.status = status
        meta.touch()
        self._sync_partition_status(meta)
        return meta

    def _get_samples(self, partition_id: str) -> list[SampleRecord]:
        tags_by_key = self.tq.kv_list(partition_id=partition_id).get(partition_id, {})
        keys = list(tags_by_key)
        if not keys:
            return []
        data = self.tq.kv_batch_get(keys=keys, partition_id=partition_id)
        rows = self._rows_from_tq_data(data, len(keys))
        samples = []
        for key, row in zip(keys, rows):
            copied = dict(row)
            copied['sample_id'] = key
            copied['metadata'] = dict(tags_by_key.get(key) or {})
            samples.append(copied)
        return samples

    def _batch_update_samples(self, partition_id: str, updates: Dict[str, Dict[str, Any]]) -> None:
        tags_by_key = self.tq.kv_list(partition_id=partition_id).get(partition_id, {})
        keys = list(updates.keys())
        all_fields = [updates[key] for key in keys]
        all_tags = [dict(tags_by_key.get(key) or {}) for key in keys]
        if keys:
            self._kv_batch_put(keys, partition_id, all_fields, all_tags)

    def _sync_partition_status(self, meta: PartitionMetadata) -> None:
        tags_by_key = self.tq.kv_list(partition_id=meta.partition_id).get(meta.partition_id, {})
        keys = list(tags_by_key.keys())
        if not keys:
            return
        all_tags = []
        for key in keys:
            updated = dict(tags_by_key[key])
            updated.update(meta.tag())
            all_tags.append(updated)
        self._kv_batch_put(keys, meta.partition_id, [None] * len(keys), all_tags)

    def _kv_batch_put(
        self,
        keys: List[str],
        partition_id: str,
        fields_list: List[Optional[Dict[str, Any]]],
        tags_list: List[Dict[str, Any]],
    ) -> None:
        if hasattr(self.tq, 'kv_batch_put'):
            self.tq.kv_batch_put(keys=keys, partition_id=partition_id, fields=None, tags=tags_list)
            for key, fields in zip(keys, fields_list):
                if fields is not None:
                    self.tq.kv_put(key=key, partition_id=partition_id, fields=fields)
        else:
            for key, fields, tag in zip(keys, fields_list, tags_list):
                self.tq.kv_put(key=key, partition_id=partition_id, fields=fields, tag=tag)

    def _recover_expired_leases(self) -> None:
        now = time.time()
        expired = [pid for pid, lease in self._leases.items() if lease['deadline'] <= now]
        for pid in expired:
            del self._leases[pid]
            meta = self._meta.get(pid)
            if meta is not None:
                meta.owner_worker_id = None
                meta.lease_deadline = None

    def _load_partition_meta(self) -> None:
        for partition_id, tags_by_key in self.tq.kv_list().items():
            tag = next(iter(tags_by_key.values()), None)
            if not tag:
                continue
            meta = self._meta_from_tag(partition_id, tag, num_rows=len(tags_by_key))
            if meta is not None:
                self._meta[partition_id] = meta

    @staticmethod
    def _rows_from_tq_data(data: Any, size: int) -> list[SampleRecord]:
        if hasattr(data, 'to_dict'):
            data = data.to_dict()
        if isinstance(data, dict):
            rows = [dict() for _ in range(size)]
            for field_name, value in data.items():
                values = TransferQueueDataPlane._split_field(value, size)
                for row, item in zip(rows, values):
                    row[field_name] = item
            return rows
        if isinstance(data, list):
            return [dict(item) for item in data]
        return [{'data': data}]

    @staticmethod
    def _split_field(value: Any, size: int) -> list[Any]:
        if size == 1:
            if hasattr(value, '__len__') and not isinstance(value, (str, bytes, dict)):
                return [value[0]]
            return [value]
        if hasattr(value, 'unbind'):
            return list(value.unbind(0))
        if hasattr(value, 'tolist'):
            value = value.tolist()
        if isinstance(value, list) and len(value) == size:
            return value
        return [value for _ in range(size)]

    @staticmethod
    def _meta_from_tag(partition_id: str, tag: Dict[str, Any], *, num_rows: int) -> Optional[PartitionMetadata]:
        try:
            context = TrainingContext(
                tenant_id=tag['tenant_id'],
                training_run_id=tag['training_run_id'],
                base_model_id=tag['base_model_id'],
                adapter_name=tag['adapter_name'],
                adapter_revision=tag.get('adapter_revision'),
                policy_version=int(tag.get('policy_version', 0)),
                env_type=tag.get('env_type', 'tool_calling'),
                tool_profile=tag.get('tool_profile', 'default'),
                reward_type=tag.get('reward_type', 'default'),
                loss_type=tag.get('loss_type', 'default'),
                algorithm=tag.get('algorithm', 'grpo'),
            )
            return PartitionMetadata(
                context=context,
                partition_id=partition_id,
                policy_version=int(tag.get('partition_policy_version', tag.get('policy_version', 0))),
                target_groups=int(tag.get('target_groups', 0)),
                ready_groups=int(tag.get('ready_groups', 0)),
                status=PartitionStatus(tag.get('status', PartitionStatus.OPEN.value)),
                num_rows=num_rows,
            )
        except (KeyError, ValueError):
            return None


class _StreamingDatasetWrapper:
    """Wrapper around TQ Client API for streaming batch iteration.

    Uses TQ Client API (get_meta + get_data) to stream batches from a partition.
    Each iteration yields a list[SampleRecord] batch and auto-acks consumed samples
    via the DataPlane's ack_rows method.

    This wraps TQ's native streaming capabilities and provides:
    - Automatic conversion from TensorDict to SampleRecord (dict)
    - Automatic ack tracking via DataPlane.ack_rows()
    - Simple iterator interface yielding list[SampleRecord] batches
    """

    def __init__(
        self,
        data_plane: TransferQueueDataPlane,
        context: TrainingContext,
        partition_id: str,
        batch_size: int,
        data_fields: Optional[List[str]],
        task_name: str,
        dp_rank: int,
    ):
        self._data_plane = data_plane
        self._context = context
        self._partition_id = partition_id
        self._batch_size = batch_size
        self._data_fields = data_fields
        self._task_name = task_name
        self._dp_rank = dp_rank
        self._client = data_plane.tq.get_client() if hasattr(data_plane.tq, 'get_client') else None
        self._total_acked = 0

    def __iter__(self) -> Iterator[list[SampleRecord]]:
        """Iterate over batches, yielding list[SampleRecord] and auto-acking."""
        if self._client is None:
            yield from self._iter_via_kv_api()
        else:
            yield from self._iter_via_client_api()

    def _iter_via_client_api(self) -> Iterator[list[SampleRecord]]:
        """Stream using TQ Client API (get_meta + get_data)."""
        fields = self._data_fields or ['messages', 'group_id', 'generation_idx', 'old_logps',
                                        'rewards', 'advantages', 'returns']
        while True:
            try:
                batch_meta = self._client.get_meta(
                    data_fields=fields,
                    batch_size=self._batch_size,
                    partition_id=self._partition_id,
                    mode='fetch',
                    task_name=self._task_name,
                    sampling_config={'dp_rank': self._dp_rank},
                )
            except Exception:
                break
            if batch_meta is None or batch_meta.size == 0:
                break
            data = self._client.get_data(batch_meta)
            samples = self._data_plane._rows_from_tq_data(data, batch_meta.size)
            keys = self._client.kv_retrieve_keys(
                global_indexes=batch_meta.global_indexes,
                partition_id=self._partition_id,
            )
            tags_by_key = self._data_plane.tq.kv_list(partition_id=self._partition_id).get(self._partition_id, {})
            for i, (key, sample) in enumerate(zip(keys, samples)):
                sample['sample_id'] = key if isinstance(key, str) else str(key)
                sample['metadata'] = dict(tags_by_key.get(key, {}))
            sample_ids = [s['sample_id'] for s in samples]
            self._data_plane.ack_rows(self._context, self._partition_id, sample_ids, task_name=self._task_name)
            self._total_acked += len(samples)
            yield samples

    def _iter_via_kv_api(self) -> Iterator[list[SampleRecord]]:
        """Fallback: stream using KV API when Client API is unavailable."""
        all_samples = self._data_plane._get_samples(self._partition_id)
        consumed = self._data_plane._consumed[self._partition_id].get(self._task_name, set())
        remaining = [s for s in all_samples if s['sample_id'] not in consumed]
        for i in range(0, len(remaining), self._batch_size):
            batch = remaining[i:i + self._batch_size]
            sample_ids = [s['sample_id'] for s in batch]
            self._data_plane.ack_rows(self._context, self._partition_id, sample_ids, task_name=self._task_name)
            self._total_acked += len(batch)
            yield batch

    @property
    def total_acked(self) -> int:
        return self._total_acked
