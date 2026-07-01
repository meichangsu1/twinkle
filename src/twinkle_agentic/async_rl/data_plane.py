# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import threading
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .types import GroupBatch, GroupMetadata, GroupStatus, LoraContext, PartitionMetadata, PartitionStatus, SampleRecord


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
        self._group_meta: dict[str, dict[str, GroupMetadata]] = defaultdict(dict)
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

    def next_partition_id(self, context: LoraContext) -> str:
        with self._lock:
            self._load_partition_meta()
            train_id = self._next_train_id[context.key]
            while context.partition_id(train_id) in self._meta:
                train_id += 1
            self._next_train_id[context.key] = train_id + 1
            return context.partition_id(train_id)

    def create_partition(
        self,
        context: LoraContext,
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
        context: LoraContext,
        partition_id: str,
        trajectories: list[SampleRecord],
        *,
        ready_groups: int = 1,
        seal: bool = False,
    ) -> PartitionMetadata:
        return self.put_rollout_groups(context, partition_id, self._normalize_rollout_groups(trajectories), seal=seal)

    def put_rollout_groups(
        self,
        context: LoraContext,
        partition_id: str,
        groups: Iterable[Iterable[SampleRecord]],
        *,
        seal: bool = False,
    ) -> PartitionMetadata:
        normalized_groups = self._normalize_rollout_groups(groups)
        with self._lock:
            meta = self._meta.get(partition_id)
            if meta is None:
                meta = self.create_partition(context, target_groups=len(normalized_groups), partition_id=partition_id)
            if meta.context.key != context.key:
                raise ValueError(f'partition {partition_id} belongs to {meta.context.key}, not {context.key}')
            if meta.status != PartitionStatus.OPEN:
                raise ValueError(f'partition {partition_id} is not open for rollout append: {meta.status}')
            row_count = 0
            for group in normalized_groups:
                if not group:
                    continue
                group_id = str(group[0].get('group_id') or group[0].get('sample_id')
                               or f'group_{meta.ready_groups + len(self._group_meta[partition_id])}')
                group_meta = GroupMetadata(
                    context=context,
                    partition_id=partition_id,
                    group_id=group_id,
                    policy_version=context.policy_version,
                    adapter_path=context.adapter_path,
                    num_samples=len(group),
                    status=GroupStatus.ROLLOUT_DONE,
                )
                self._group_meta[partition_id][group_id] = group_meta
                group_tag = meta.tag()
                group_tag.update(group_meta.tag())
                self.tq.kv_put(key=group_meta.key, partition_id=partition_id, fields={}, tag=group_tag)

                for sample_index, trajectory in enumerate(group):
                    sample = dict(trajectory)
                    sample['group_id'] = group_id
                    sample.setdefault('generation_idx', sample_index)
                    sample_meta = dict(sample.get('metadata') or {})
                    sample_meta.update(context.metadata())
                    sample_meta.setdefault('partition_id', partition_id)
                    context.validate_metadata(sample_meta)
                    generation_idx = sample.get('generation_idx', sample_index)
                    key = f'{partition_id}/samples/{group_id}/{generation_idx}'
                    fields = {k: v for k, v in sample.items() if k not in {'metadata', 'sample_id'}}
                    tag = dict(sample_meta)
                    tag.update(meta.tag())
                    tag.update({
                        'record_type': 'sample',
                        'group_id': group_id,
                        'generation_idx': generation_idx,
                        'sample_id': sample.get('sample_id', key),
                    })
                    self.tq.kv_put(key=key, partition_id=partition_id, fields=fields, tag=tag)
                    row_count += 1
            meta.num_rows += row_count
            meta.ready_groups += len(normalized_groups)
            if seal or (meta.target_groups and meta.ready_groups >= meta.target_groups):
                meta.status = PartitionStatus.SEALED
            meta.touch()
            self._meta[partition_id] = meta
            self._sync_partition_status(meta)
            return meta

    def list_partitions(
        self,
        context: LoraContext | None = None,
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

    def get_metadata(self, context: LoraContext | None = None) -> list[PartitionMetadata]:
        return self.list_partitions(context)

    def check_capacity(self, context: LoraContext) -> bool:
        live_partitions = [p for p in self.list_partitions() if p.status != PartitionStatus.CLEARED]
        total_rows = sum(p.num_rows for p in live_partitions)
        context_rows = sum(p.num_rows for p in live_partitions if p.context.key == context.key)
        if self.tq_config.max_rows is not None and total_rows >= self.tq_config.max_rows:
            return False
        if self.tq_config.max_rows_per_context is not None and context_rows >= self.tq_config.max_rows_per_context:
            return False
        return True

    def claim_reward_groups(self, context: LoraContext, max_groups: int) -> GroupBatch:
        return self._claim_groups(
            context=context,
            max_groups=max_groups,
            from_status=GroupStatus.ROLLOUT_DONE,
            to_status=GroupStatus.REWARDING,
            stage='reward',
        )

    def claim_reward_batch(self, context: LoraContext, batch_size: int) -> tuple[PartitionMetadata, list[SampleRecord]]:
        batch = self.claim_reward_groups(context, batch_size)
        return self._meta[batch.partition_id], batch.samples

    def append_rewards(
        self,
        context: LoraContext,
        partition_id: str,
        rewards: list[float],
        *,
        field_name: str = 'rewards',
    ) -> PartitionMetadata:
        groups = [
            group for group in self.list_groups(context, partition_id=partition_id, statuses=[GroupStatus.REWARDING])
        ]
        return self.append_group_rewards(
            context,
            self._build_group_batch(context, groups),
            rewards,
            field_name=field_name,
        )

    def append_group_rewards(
        self,
        context: LoraContext,
        group_batch: GroupBatch,
        rewards: list[float],
        *,
        field_name: str = 'rewards',
    ) -> PartitionMetadata:
        samples = group_batch.samples
        if len(rewards) != len(samples):
            raise ValueError(f'reward count {len(rewards)} does not match sample count {len(samples)}')
        updates = {}
        for sample, reward in zip(samples, rewards):
            context.validate_metadata(sample.get('metadata') or {}, strict_policy_version=False)
            updates[sample['sample_id']] = {field_name: reward}
        self._update_samples(group_batch.partition_id, updates)
        for group in group_batch.groups:
            group.status = GroupStatus.REWARD_DONE
            group.touch()
            self._sync_group_status(group)
        meta = self._meta[group_batch.partition_id]
        meta.touch()
        self._sync_partition_status(meta)
        return meta

    def claim_advantage_groups(self, context: LoraContext, max_groups: int) -> GroupBatch:
        return self._claim_groups(
            context=context,
            max_groups=max_groups,
            from_status=GroupStatus.REWARD_DONE,
            to_status=GroupStatus.ADVANTAGING,
            stage='advantage',
        )

    def claim_advantage_batch(self, context: LoraContext,
                              batch_size: int) -> tuple[PartitionMetadata, list[SampleRecord]]:
        batch = self.claim_advantage_groups(context, batch_size)
        return self._meta[batch.partition_id], batch.samples

    def append_advantages(
        self,
        context: LoraContext,
        partition_id: str,
        advantages: list[float],
        returns: list[float] | None = None,
    ) -> PartitionMetadata:
        groups = [
            group for group in self.list_groups(context, partition_id=partition_id, statuses=[GroupStatus.ADVANTAGING])
        ]
        return self.append_group_advantages(context, self._build_group_batch(context, groups), advantages, returns)

    def append_group_advantages(
        self,
        context: LoraContext,
        group_batch: GroupBatch,
        advantages: list[float],
        returns: list[float] | None = None,
    ) -> PartitionMetadata:
        samples = group_batch.samples
        if len(advantages) != len(samples):
            raise ValueError(f'advantage count {len(advantages)} does not match sample count {len(samples)}')
        if returns is None:
            returns = advantages
        updates = {}
        for sample, advantage, ret in zip(samples, advantages, returns):
            context.validate_metadata(sample.get('metadata') or {}, strict_policy_version=False)
            updates[sample['sample_id']] = {'advantages': advantage, 'returns': ret}
        self._update_samples(group_batch.partition_id, updates)
        for group in group_batch.groups:
            group.status = GroupStatus.ADVANTAGE_DONE
            group.touch()
            self._sync_group_status(group)
        meta = self._meta[group_batch.partition_id]
        meta.touch()
        self._sync_partition_status(meta)
        return meta

    def list_train_ready_partitions(self, *, min_groups: int = 1) -> list[PartitionMetadata]:
        self._load_partition_meta()
        candidates = []
        for meta in self.list_partitions():
            if meta.status in {PartitionStatus.CLEARED, PartitionStatus.TRAIN_DONE, PartitionStatus.FAILED}:
                continue
            available = self.count_groups(meta.context, meta.partition_id, statuses=[GroupStatus.ADVANTAGE_DONE])
            if available >= min_groups or (meta.status == PartitionStatus.SEALED and available > 0):
                candidates.append(meta)
        return candidates

    def mark_training(self, context: LoraContext, partition_id: str) -> PartitionMetadata:
        return self._mark_status(context, partition_id, PartitionStatus.TRAINING)

    def mark_trained(self, context: LoraContext, partition_id: str) -> PartitionMetadata:
        return self._mark_status(context, partition_id, PartitionStatus.TRAIN_DONE)

    def claim_train_groups(self, context: LoraContext, partition_id: str, max_groups: int) -> GroupBatch:
        with self._lock:
            groups = self.list_groups(context, partition_id=partition_id, statuses=[GroupStatus.ADVANTAGE_DONE])
            if not groups:
                raise LookupError(f'no train-ready group for {context.key} partition={partition_id}')
            meta = self._meta[partition_id]
            if len(groups) < max_groups and meta.status != PartitionStatus.SEALED:
                raise LookupError(f'partition {partition_id} has only {len(groups)} train-ready groups')
            selected = groups[:max_groups]
            for group in selected:
                group.status = GroupStatus.TRAINING
                group.touch()
                self._sync_group_status(group)
            return self._build_group_batch(context, selected)

    def mark_groups_train_done(self, context: LoraContext, group_batch: GroupBatch) -> PartitionMetadata:
        for group in group_batch.groups:
            if group.context.key != context.key:
                raise ValueError(f'group {group.group_id} belongs to {group.context.key}, not {context.key}')
            group.status = GroupStatus.TRAIN_DONE
            group.touch()
            self._sync_group_status(group)
        meta = self._meta[group_batch.partition_id]
        meta.touch()
        self._sync_partition_status(meta)
        return meta

    def seal_partition(self, context: LoraContext, partition_id: str) -> PartitionMetadata:
        return self._mark_status(context, partition_id, PartitionStatus.SEALED)

    def is_partition_train_done(self, context: LoraContext, partition_id: str) -> bool:
        meta = self._meta[partition_id]
        if meta.context.key != context.key:
            raise ValueError(f'partition {partition_id} belongs to {meta.context.key}, not {context.key}')
        if meta.status != PartitionStatus.SEALED:
            return False
        groups = self.list_groups(context, partition_id=partition_id)
        return bool(groups) and all(group.status == GroupStatus.TRAIN_DONE for group in groups)

    def build_streaming_dataloader(self, context: LoraContext, partition_id: str):
        meta = self._meta[partition_id]
        if meta.context.key != context.key:
            raise ValueError(f'partition {partition_id} belongs to {meta.context.key}, not {context.key}')
        return self._get_samples(partition_id)

    def clear_partition(self, context: LoraContext, partition_id: str) -> None:
        meta = self._meta.get(partition_id)
        if meta is not None and meta.context.key != context.key:
            raise ValueError(f'partition {partition_id} belongs to {meta.context.key}, not {context.key}')
        keys = list(self.tq.kv_list(partition_id=partition_id).get(partition_id, {}))
        if keys:
            self.tq.kv_clear(keys=keys, partition_id=partition_id)
        if meta is not None:
            meta.status = PartitionStatus.CLEARED
            meta.touch()
        self._group_meta.pop(partition_id, None)

    def list_groups(
        self,
        context: LoraContext | None = None,
        *,
        partition_id: str | None = None,
        statuses: Iterable[GroupStatus] | None = None,
    ) -> list[GroupMetadata]:
        self._load_partition_meta()
        status_set = set(statuses) if statuses is not None else None
        groups = []
        partition_ids = [partition_id] if partition_id is not None else list(self._group_meta)
        for pid in partition_ids:
            groups.extend(self._group_meta.get(pid, {}).values())
        if context is not None:
            groups = [group for group in groups if group.context.key == context.key]
        if status_set is not None:
            groups = [group for group in groups if group.status in status_set]
        return sorted(groups, key=lambda group: (group.created_at, group.partition_id, group.group_id))

    def count_groups(
        self,
        context: LoraContext,
        partition_id: str,
        *,
        statuses: Iterable[GroupStatus] | None = None,
    ) -> int:
        return len(self.list_groups(context, partition_id=partition_id, statuses=statuses))

    def _claim_samples(
        self,
        context: LoraContext,
        batch_size: int,
        statuses: Iterable[PartitionStatus],
        stage: str,
    ) -> tuple[PartitionMetadata, list[SampleRecord]]:
        partitions = self.list_partitions(context, statuses=statuses)
        if not partitions:
            raise LookupError(f'no {stage}-ready partition for {context.key}')
        meta = partitions[0]
        return meta, self._get_samples(meta.partition_id)[:batch_size]

    def _claim_groups(
        self,
        *,
        context: LoraContext,
        max_groups: int,
        from_status: GroupStatus,
        to_status: GroupStatus,
        stage: str,
    ) -> GroupBatch:
        if max_groups <= 0:
            raise ValueError(f'{stage} max_groups must be positive')
        with self._lock:
            groups = self.list_groups(context, statuses=[from_status])
            if not groups:
                raise LookupError(f'no {stage}-ready group for {context.key}')
            partition_id = groups[0].partition_id
            partition_groups = [group for group in groups if group.partition_id == partition_id]
            selected = partition_groups[:max_groups]
            for group in selected:
                group.status = to_status
                group.touch()
                self._sync_group_status(group)
            return self._build_group_batch(context, selected)

    def _build_group_batch(self, context: LoraContext, groups: list[GroupMetadata]) -> GroupBatch:
        if not groups:
            raise LookupError(f'empty group batch for {context.key}')
        partition_ids = {group.partition_id for group in groups}
        if len(partition_ids) != 1:
            raise ValueError(f'group batch crosses partitions: {sorted(partition_ids)}')
        for group in groups:
            if group.context.key != context.key:
                raise ValueError(f'group {group.group_id} belongs to {group.context.key}, not {context.key}')
        samples = self._get_samples_for_groups(groups)
        return GroupBatch(
            context=context,
            partition_id=groups[0].partition_id,
            groups=list(groups),
            samples=samples,
        )

    def _mark_status(self, context: LoraContext, partition_id: str, status: PartitionStatus) -> PartitionMetadata:
        meta = self._meta[partition_id]
        if meta.context.key != context.key:
            raise ValueError(f'partition {partition_id} belongs to {meta.context.key}, not {context.key}')
        meta.status = status
        meta.touch()
        self._sync_partition_status(meta)
        return meta

    def _get_samples(self, partition_id: str) -> list[SampleRecord]:
        tags_by_key = self.tq.kv_list(partition_id=partition_id).get(partition_id, {})
        keys = sorted(
            [key for key, tag in tags_by_key.items() if tag.get('record_type', 'sample') == 'sample'],
            key=lambda key: self._sample_sort_key(key,
                                                  tags_by_key.get(key) or {}),
        )
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

    def _get_samples_for_groups(self, groups: list[GroupMetadata]) -> list[SampleRecord]:
        group_ids = {group.group_id for group in groups}
        partition_id = groups[0].partition_id
        return [
            sample for sample in self._get_samples(partition_id)
            if str(sample.get('metadata', {}).get('group_id', sample.get('group_id'))) in group_ids
        ]

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

    def _sync_group_status(self, group: GroupMetadata) -> None:
        tag = dict(self.tq.kv_list(partition_id=group.partition_id).get(group.partition_id, {}).get(group.key) or {})
        tag.update(group.tag())
        if group.partition_id in self._meta:
            partition_tag = self._meta[group.partition_id].tag()
            partition_tag.update(tag)
            tag = partition_tag
        self._group_meta[group.partition_id][group.group_id] = group
        self.tq.kv_put(key=group.key, partition_id=group.partition_id, tag=tag)

    def _load_partition_meta(self) -> None:
        for partition_id, tags_by_key in self.tq.kv_list().items():
            tag = next((tag for tag in tags_by_key.values() if 'tenant_id' in tag), None)
            if not tag:
                continue
            num_rows = sum(1 for tag in tags_by_key.values() if tag.get('record_type', 'sample') == 'sample')
            meta = self._meta_from_tag(partition_id, tag, num_rows=num_rows)
            if meta is not None:
                self._meta[partition_id] = meta
            for group_tag in tags_by_key.values():
                if group_tag.get('record_type') != 'group':
                    continue
                group = self._group_meta_from_tag(group_tag)
                if group is not None:
                    self._group_meta[partition_id][group.group_id] = group

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
    def _sample_sort_key(key: str, tag: dict[str, Any]) -> tuple[str, int, str]:
        group_id = str(tag.get('group_id', ''))
        try:
            generation_idx = int(tag.get('generation_idx', 0))
        except (TypeError, ValueError):
            generation_idx = 0
        return group_id, generation_idx, key

    @staticmethod
    def _normalize_rollout_groups(groups: Iterable[Any]) -> list[list[SampleRecord]]:
        items = list(groups)
        if not items:
            return []
        first = items[0]
        if isinstance(first, dict):
            grouped: OrderedDict[str, list[SampleRecord]] = OrderedDict()
            for idx, item in enumerate(items):
                sample = dict(item)
                group_id = str(sample.get('group_id') or sample.get('sample_id') or f'group_{idx}')
                sample['group_id'] = group_id
                grouped.setdefault(group_id, []).append(sample)
            return list(grouped.values())
        normalized = []
        for group_idx, item in enumerate(items):
            group_samples = [dict(sample) for sample in item]
            for sample in group_samples:
                sample.setdefault('group_id', f'group_{group_idx}')
            normalized.append(group_samples)
        return normalized

    @staticmethod
    def _meta_from_tag(partition_id: str, tag: dict[str, Any], *, num_rows: int) -> PartitionMetadata | None:
        try:
            context = LoraContext(
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
                status=PartitionStatus(tag.get('partition_status', tag.get('status', PartitionStatus.OPEN.value))),
                num_rows=num_rows,
            )
        except (KeyError, ValueError):
            return None

    @staticmethod
    def _group_meta_from_tag(tag: dict[str, Any]) -> GroupMetadata | None:
        try:
            context = LoraContext(
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
            return GroupMetadata(
                context=context,
                partition_id=tag['partition_id'],
                group_id=str(tag['group_id']),
                policy_version=int(tag.get('group_policy_version', tag.get('policy_version', 0))),
                adapter_path=tag.get('group_adapter_path', tag.get('adapter_path')),
                num_samples=int(tag.get('num_samples', 0)),
                status=GroupStatus(tag.get('group_status', GroupStatus.ROLLOUT_DONE.value)),
                created_at=float(tag.get('created_at', 0.0)) or 0.0,
                updated_at=float(tag.get('updated_at', 0.0)) or 0.0,
            )
        except (KeyError, ValueError):
            return None
