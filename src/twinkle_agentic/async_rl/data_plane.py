# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .types import PartitionMetadata, PartitionStatus, SampleRecord, TrainingContext


@dataclass
class TransferQueueRuntimeConfig:
    """TransferQueue initialization and lightweight capacity guard config."""

    total_storage_size: int | None = None
    max_rows: int | None = None
    max_rows_per_context: int | None = None
    num_data_storage_units: int = 4
    storage_backend: str = 'SimpleStorage'
    controller: dict[str, Any] = field(default_factory=dict)
    backend: dict[str, Any] = field(default_factory=dict)
    init: bool = True


class TransferQueueDataPlane:
    """The only data-plane boundary for async RL TransferQueue access."""

    def __init__(self, tq_client: Any | None = None, tq_config: TransferQueueRuntimeConfig | None = None):
        self.tq_config = tq_config or TransferQueueRuntimeConfig()
        self.tq = tq_client or self._init_transfer_queue(self.tq_config)
        self._meta: dict[str, PartitionMetadata] = {}
        self._next_train_id: dict[str, int] = defaultdict(int)
        self._lock = threading.RLock()

    def _init_transfer_queue(self, config: TransferQueueRuntimeConfig):
        try:
            import transfer_queue as tq
        except ImportError as exc:
            raise RuntimeError('transfer_queue is required for TransferQueueDataPlane. '
                               'Pass an explicit tq_client only in unit tests/local mocks.') from exc
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
        backend_config.setdefault('storage_backend', config.storage_backend)
        backend_config['SimpleStorage'] = simple_storage
        return OmegaConf.create(
            {
                'controller': config.controller,
                'backend': backend_config
            },
            flags={'allow_objects': True},
        )

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
        partition_id: str | None = None,
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
        trajectories: list[SampleRecord],
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
            for idx, trajectory in enumerate(trajectories):
                sample = dict(trajectory)
                sample_meta = dict(sample.get('metadata') or {})
                sample_meta.update(context.metadata())
                sample_meta.setdefault('partition_id', partition_id)
                context.validate_metadata(sample_meta)
                key = sample.get('sample_id') or f'{partition_id}/sample_{meta.num_rows + idx}'
                fields = {k: v for k, v in sample.items() if k not in {'metadata', 'sample_id'}}
                tag = dict(sample_meta)
                tag.update(meta.tag())
                self.tq.kv_put(key=key, partition_id=partition_id, fields=fields, tag=tag)
                keys.append(key)
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
        context: TrainingContext | None = None,
        *,
        statuses: Iterable[PartitionStatus] | None = None,
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

    def get_metadata(self, context: TrainingContext | None = None) -> list[PartitionMetadata]:
        return self.list_partitions(context)

    def check_capacity(self, context: TrainingContext) -> bool:
        live_partitions = [p for p in self.list_partitions() if p.status != PartitionStatus.CLEARED]
        total_rows = sum(p.num_rows for p in live_partitions)
        context_rows = sum(p.num_rows for p in live_partitions if p.context.key == context.key)
        if self.tq_config.max_rows is not None and total_rows >= self.tq_config.max_rows:
            return False
        if self.tq_config.max_rows_per_context is not None and context_rows >= self.tq_config.max_rows_per_context:
            return False
        return True

    def claim_reward_batch(self, context: TrainingContext,
                           batch_size: int) -> tuple[PartitionMetadata, list[SampleRecord]]:
        return self._claim_samples(context, batch_size, [PartitionStatus.ROLLOUT_DONE], 'reward')

    def append_rewards(
        self,
        context: TrainingContext,
        partition_id: str,
        rewards: list[float],
        *,
        field_name: str = 'rewards',
    ) -> PartitionMetadata:
        samples = self._get_samples(partition_id)
        if len(rewards) != len(samples):
            raise ValueError(f'reward count {len(rewards)} does not match sample count {len(samples)}')
        updates = {}
        for sample, reward in zip(samples, rewards):
            context.validate_metadata(sample.get('metadata') or {}, strict_policy_version=False)
            updates[sample['sample_id']] = {field_name: reward}
        self._update_samples(partition_id, updates)
        meta = self._meta[partition_id]
        meta.status = PartitionStatus.REWARD_DONE
        meta.touch()
        self._sync_partition_status(meta)
        return meta

    def claim_advantage_batch(self, context: TrainingContext,
                              batch_size: int) -> tuple[PartitionMetadata, list[SampleRecord]]:
        return self._claim_samples(context, batch_size, [PartitionStatus.REWARD_DONE], 'advantage')

    def append_advantages(
        self,
        context: TrainingContext,
        partition_id: str,
        advantages: list[float],
        returns: list[float] | None = None,
    ) -> PartitionMetadata:
        samples = self._get_samples(partition_id)
        if len(advantages) != len(samples):
            raise ValueError(f'advantage count {len(advantages)} does not match sample count {len(samples)}')
        if returns is None:
            returns = advantages
        updates = {}
        for sample, advantage, ret in zip(samples, advantages, returns):
            context.validate_metadata(sample.get('metadata') or {}, strict_policy_version=False)
            updates[sample['sample_id']] = {'advantages': advantage, 'returns': ret}
        self._update_samples(partition_id, updates)
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
        return self._mark_status(context, partition_id, PartitionStatus.TRAIN_DONE)

    def build_streaming_dataloader(self, context: TrainingContext, partition_id: str):
        meta = self._meta[partition_id]
        if meta.context.key != context.key:
            raise ValueError(f'partition {partition_id} belongs to {meta.context.key}, not {context.key}')
        return self._get_samples(partition_id)

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

    def _claim_samples(
        self,
        context: TrainingContext,
        batch_size: int,
        statuses: Iterable[PartitionStatus],
        stage: str,
    ) -> tuple[PartitionMetadata, list[SampleRecord]]:
        partitions = self.list_partitions(context, statuses=statuses)
        if not partitions:
            raise LookupError(f'no {stage}-ready partition for {context.key}')
        meta = partitions[0]
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

    def _update_samples(self, partition_id: str, updates: dict[str, dict[str, Any]]) -> None:
        tags_by_key = self.tq.kv_list(partition_id=partition_id).get(partition_id, {})
        for key, fields in updates.items():
            self.tq.kv_put(
                key=key,
                partition_id=partition_id,
                fields=fields,
                tag=dict(tags_by_key.get(key) or {}),
            )

    def _sync_partition_status(self, meta: PartitionMetadata) -> None:
        tags_by_key = self.tq.kv_list(partition_id=meta.partition_id).get(meta.partition_id, {})
        for key, tag in tags_by_key.items():
            updated = dict(tag)
            updated.update(meta.tag())
            self.tq.kv_put(key=key, partition_id=meta.partition_id, tag=updated)

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
    def _meta_from_tag(partition_id: str, tag: dict[str, Any], *, num_rows: int) -> PartitionMetadata | None:
        try:
            context = TrainingContext(
                tenant_id=tag['tenant_id'],
                training_run_id=tag['training_run_id'],
                base_model_id=tag['base_model_id'],
                adapter_name=tag['adapter_name'],
                adapter_path=tag.get('adapter_path'),
                policy_version=int(tag.get('policy_version', 0)),
                tool_profile=tag.get('tool_profile', 'default'),
                reward_type=tag.get('reward_type', 'default'),
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
