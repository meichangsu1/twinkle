# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from transfer_queue import KVBatchMeta
from typing import Any, Iterable

from .metrics import AsyncRLMetricsRecorder, NoopMetricsRecorder
from .types import (LoraContext, LoraRuntimeState, PartitionMeta, PartitionStatus, PromptGroupMeta, PromptGroupRef,
                    PromptGroupStatus)

PARTITION_KEY = '__partition__'


@dataclass
class TransferQueueRuntimeConfig:
    """TransferQueue initialization config."""

    total_storage_size: int | None = None
    num_data_storage_units: int = 4
    storage_backend: str = 'SimpleStorage'
    controller: dict[str, Any] = field(default_factory=dict)
    backend: dict[str, Any] = field(default_factory=dict)
    init: bool = True


class TransferQueueDataPlane:
    """The only data-plane boundary for async RL TransferQueue access.

    This object is driven by the pipeline's single driver thread. If future
    workers claim/write TQ state concurrently, use TQ-side atomic operations
    instead of process-local locks.
    """

    def __init__(
        self,
        tq_client: Any | None = None,
        tq_config: TransferQueueRuntimeConfig | None = None,
        metrics_recorder: AsyncRLMetricsRecorder | None = None,
    ):
        self.tq_config = tq_config or TransferQueueRuntimeConfig()
        self.tq = tq_client or self._init_transfer_queue(self.tq_config)
        self.metrics_recorder = metrics_recorder or NoopMetricsRecorder()

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

    def _next_partition_id(self, context: LoraContext) -> str:
        return context.partition_id(self.peek_next_train_id(context))

    def peek_next_train_id(self, context: LoraContext) -> int:
        next_train_id = 0
        for partition in self.list_partitions(context):
            suffix = partition.partition_id.rsplit('/', 1)[-1]
            if not suffix.startswith('train_'):
                continue
            try:
                next_train_id = max(next_train_id, int(suffix[len('train_'):]) + 1)
            except ValueError:
                continue
        return next_train_id

    def create_rollout_partition(
        self,
        context: LoraContext,
        *,
        target_groups: int,
        partition_id: str | None = None,
    ) -> PartitionMeta:
        partition_id = partition_id or self._next_partition_id(context)
        partition_tag = self._list_partition_tags(partition_id).get(PARTITION_KEY, {})
        existing = self._partition_from_tag(partition_id, partition_tag)
        if existing is not None:
            raise ValueError(f'partition already exists: {partition_id}')
        partition = PartitionMeta(
            context=context,
            partition_id=partition_id,
            target_groups=target_groups,
        )
        self._put_partition_tag(partition)
        return partition

    def update_partition_status(self, partition_id: str, status: PartitionStatus) -> PartitionMeta:
        partition = self.get_rollout_partition(partition_id)
        partition.status = status
        partition.touch()
        self._put_partition_tag(partition)
        return partition

    def create_prompt_group(
        self,
        context: LoraContext,
        partition: PartitionMeta,
        *,
        runtime_state: LoraRuntimeState,
    ) -> PromptGroupRef:
        stored_partition = self.get_rollout_partition(partition.partition_id)
        if stored_partition.context.key != context.key:
            raise ValueError(
                f'partition {partition.partition_id} belongs to {stored_partition.context.key}, not {context.key}')
        group_id = str(uuid.uuid4())
        group = PromptGroupMeta(
            context=context,
            partition_id=stored_partition.partition_id,
            group_id=group_id,
            rollout_policy_version=runtime_state.policy_version,
            rollout_adapter_path=runtime_state.adapter_path,
            num_samples=0,
            status=PromptGroupStatus.PENDING,
        )
        self._put_group_tag(group)
        return PromptGroupRef(partition_id=stored_partition.partition_id, group_id=group_id)

    def get_prompt_group(self, group_ref: PromptGroupRef) -> PromptGroupMeta:
        tags_by_key = self._list_partition_tags(group_ref.partition_id)
        group = self._prompt_group_from_tag(tags_by_key.get(group_ref.key, {}))
        if group is None:
            raise KeyError(f'unknown prompt group: {group_ref.partition_id}/{group_ref.group_id}')
        return group

    def get_rollout_partition(self, partition_id: str) -> PartitionMeta:
        partition_tag = self._list_partition_tags(partition_id).get(PARTITION_KEY, {})
        partition = self._partition_from_tag(partition_id, partition_tag)
        if partition is None:
            raise KeyError(f'unknown partition: {partition_id}')
        return partition

    def update_prompt_group_status(
        self,
        group_ref: PromptGroupRef,
        status: PromptGroupStatus,
        *,
        sample_keys: list[str] | None = None,
        extra_tag: dict[str, Any] | None = None,
    ) -> None:
        group = self.get_prompt_group(group_ref)
        if sample_keys is not None:
            group.sample_keys = list(sample_keys)
            group.num_samples = len(sample_keys)
        group.status = status
        group.touch()
        self._put_group_tags([group], extra_tag=extra_tag)

    def write_sample_batch(
        self,
        *,
        partition_id: str,
        keys: list[str],
        fields: Any,
        tags: list[dict[str, Any]],
    ) -> None:
        if not keys:
            return
        start = time.perf_counter()
        self.tq.kv_batch_put(
            keys=keys,
            partition_id=partition_id,
            fields=fields,
            tags=tags,
        )
        self._record_tq_event(
            'kv_batch_put',
            start,
            partition_id=partition_id,
            metrics={'write_samples': len(keys)},
        )

    def list_partitions(
        self,
        context: LoraContext | None = None,
        *,
        statuses: Iterable[PartitionStatus] | None = None,
    ) -> list[PartitionMeta]:
        status_set = set(statuses) if statuses is not None else None
        partitions = []
        for partition_id, tags_by_key in self._list_all_tags().items():
            partition = self._partition_from_tag(partition_id, tags_by_key.get(PARTITION_KEY, {}))
            if partition is None:
                continue
            if context is not None and partition.context.key != context.key:
                continue
            if status_set is not None and partition.status not in status_set:
                continue
            partitions.append(partition)
        return sorted(partitions, key=lambda p: (p.created_at, p.partition_id))

    def read_batch_fields(self, batch: Any, fields: list[str] | str | None = None) -> Any:
        start = time.perf_counter()
        result = self.tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
        self._record_tq_event(
            'kv_batch_get',
            start,
            partition_id=batch.partition_id,
            metrics={'read_samples': len(batch.keys)},
        )
        return result

    def write_batch_fields(self, batch: Any, fields: Any) -> None:
        start = time.perf_counter()
        self.tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=fields)
        self._record_tq_event(
            'kv_batch_put',
            start,
            partition_id=batch.partition_id,
            metrics={'write_samples': len(batch.keys)},
        )

    def claim_prompt_groups(
        self,
        groups: Iterable[PromptGroupMeta],
        *,
        ready_status: PromptGroupStatus,
        claim_status: PromptGroupStatus,
        fields: list[str] | None = None,
    ) -> KVBatchMeta:
        groups = list(groups)
        if not groups:
            raise ValueError('groups must not be empty')
        partition_ids = {group.partition_id for group in groups}
        if len(partition_ids) != 1:
            raise ValueError(f'claim batch must belong to one partition, got {sorted(partition_ids)}')
        context_keys = {group.context.key for group in groups}
        if len(context_keys) != 1:
            raise ValueError(f'claim batch must belong to one context, got {sorted(context_keys)}')
        partition_id = groups[0].partition_id
        context = groups[0].context
        tags_by_key = self._list_partition_tags(partition_id, context=context)
        stored_groups = []
        for group in groups:
            stored_group = self._prompt_group_from_tag(tags_by_key.get(group.key, {}))
            if stored_group is None:
                raise KeyError(f'unknown prompt group: {partition_id}/{group.group_id}')
            if stored_group.context.key != context.key:
                raise ValueError(f'group {group.group_id} belongs to {stored_group.context.key}, not {context.key}')
            stored_groups.append(stored_group)
        batch = self._build_sample_batch_for_groups(
            stored_groups,
            tags_by_key=tags_by_key,
            ready_status=ready_status,
            fields=fields,
        )
        for group in stored_groups:
            group.status = claim_status
            group.touch()
        self._put_group_tags(stored_groups)
        return batch

    def mark_groups(
        self,
        *,
        partition_id: str,
        group_ids: Iterable[str],
        status: PromptGroupStatus,
        extra_tag: dict[str, Any] | None = None,
    ) -> None:
        group_ids = [str(group_id) for group_id in group_ids]
        if not group_ids:
            raise ValueError('group_ids must not be empty')
        tags_by_key = self._list_partition_tags(partition_id)
        groups = []
        for group_id in group_ids:
            group = self._prompt_group_from_tag(tags_by_key.get(f'groups/{group_id}', {}))
            if group is None:
                raise KeyError(f'unknown prompt group: {partition_id}/{group_id}')
            group.status = status
            group.touch()
            groups.append(group)
        self._put_group_tags(groups, extra_tag=extra_tag)

    def mark_prompt_groups(
        self,
        groups: Iterable[PromptGroupMeta],
        status: PromptGroupStatus,
        *,
        extra_tag: dict[str, Any] | None = None,
    ) -> None:
        groups = list(groups)
        if not groups:
            raise ValueError('groups must not be empty')
        for group in groups:
            group.status = status
            group.touch()
        self._put_group_tags(groups, extra_tag=extra_tag)

    def clear_partition(self, context: LoraContext, partition_id: str) -> None:
        tags_by_key = self._list_partition_tags(partition_id, context=context)
        partition = self._partition_from_tag(partition_id, tags_by_key.get(PARTITION_KEY, {}))
        if partition is not None and partition.context.key != context.key:
            raise ValueError(f'partition {partition_id} belongs to {partition.context.key}, not {context.key}')
        keys = list(tags_by_key)
        if keys:
            start = time.perf_counter()
            self.tq.kv_clear(keys=keys, partition_id=partition_id)
            self._record_tq_event(
                'kv_clear',
                start,
                context=context,
                partition_id=partition_id,
                metrics={'cleared_keys': len(keys)},
            )

    def list_prompt_groups(
        self,
        context: LoraContext | None = None,
        *,
        partition_id: str,
        statuses: Iterable[PromptGroupStatus] | None = None,
    ) -> list[PromptGroupMeta]:
        if partition_id is None:
            raise ValueError('partition_id is required; use list_all_prompt_groups() for global scans')
        tags_by_key = self._list_partition_tags(partition_id, context=context)
        return self._filter_prompt_groups({partition_id: tags_by_key}, context=context, statuses=statuses)

    def list_all_prompt_groups(
        self,
        context: LoraContext | None = None,
        *,
        statuses: Iterable[PromptGroupStatus] | None = None,
    ) -> list[PromptGroupMeta]:
        return self._filter_prompt_groups(self._list_all_tags(), context=context, statuses=statuses)

    def _filter_prompt_groups(
        self,
        tags_by_partition: dict[str, dict[str, dict[str, Any]]],
        *,
        context: LoraContext | None = None,
        statuses: Iterable[PromptGroupStatus] | None = None,
    ) -> list[PromptGroupMeta]:
        status_set = set(statuses) if statuses is not None else None
        groups = []
        for tags_by_key in tags_by_partition.values():
            for group in self._groups_from_tags(tags_by_key.values()):
                if context is not None and group.context.key != context.key:
                    continue
                if status_set is not None and group.status not in status_set:
                    continue
                groups.append(group)
        return sorted(groups, key=lambda group: (group.created_at, group.partition_id, group.group_id))

    @staticmethod
    def _build_sample_batch_for_groups(
        groups: list[PromptGroupMeta],
        *,
        tags_by_key: dict[str, dict[str, Any]],
        ready_status: PromptGroupStatus,
        fields: list[str] | None,
    ) -> KVBatchMeta:
        keys: list[str] = []
        tags: list[dict[str, Any]] = []
        for group in groups:
            if group.status != ready_status:
                raise LookupError(f'group {group.group_id} is not {ready_status.value}: {group.status}')
            if len(group.sample_keys) != group.num_samples:
                raise ValueError(f'group {group.group_id} has num_samples={group.num_samples} '
                                 f'but {len(group.sample_keys)} sample keys')
            for sample_key in group.sample_keys:
                tag = dict(tags_by_key.get(sample_key) or {})
                if not tag:
                    raise ValueError(f'group {group.group_id} sample key {sample_key!r} has no TQ tag')
                if tag.get('record_type') != 'sample':
                    raise ValueError(f'group {group.group_id} sample key {sample_key!r} is not a sample tag')
                if tag.get('group_id') != group.group_id:
                    raise ValueError(f'group {group.group_id} sample key {sample_key!r} tag belongs to '
                                     f'{tag.get("group_id")!r}')
                if 'generation_idx' not in tag:
                    raise ValueError(f'group {group.group_id} sample key {sample_key!r} missing generation_idx')
                keys.append(sample_key)
                tags.append(tag)
        return KVBatchMeta(
            partition_id=groups[0].partition_id,
            keys=list(keys),
            tags=list(tags),
            fields=None if fields is None else list(fields),
        )

    def _put_partition_tag(self, partition: PartitionMeta) -> None:
        tag = dict(partition.tag())
        tag['record_type'] = 'partition'
        start = time.perf_counter()
        self.tq.kv_put(key=PARTITION_KEY, partition_id=partition.partition_id, tag=tag)
        self._record_tq_event('kv_put', start, context=partition.context, partition_id=partition.partition_id)

    def _put_group_tag(self, group: PromptGroupMeta, *, extra_tag: dict[str, Any] | None = None) -> None:
        self._put_group_tags([group], extra_tag=extra_tag)

    def _put_group_tags(
        self,
        groups: Iterable[PromptGroupMeta],
        *,
        extra_tag: dict[str, Any] | None = None,
    ) -> None:
        groups = list(groups)
        if not groups:
            return
        partition_ids = {group.partition_id for group in groups}
        if len(partition_ids) != 1:
            raise ValueError(f'group batch must belong to one partition, got {sorted(partition_ids)}')
        partition_id = groups[0].partition_id
        keys = []
        tags = []
        for group in groups:
            tag = dict(group.tag())
            if extra_tag:
                tag.update(extra_tag)
            keys.append(group.key)
            tags.append(tag)
        start = time.perf_counter()
        self.tq.kv_batch_put(keys=keys, partition_id=partition_id, tags=tags)
        self._record_tq_event(
            'kv_batch_put',
            start,
            context=groups[0].context,
            partition_id=partition_id,
            metrics={'write_group_tags': len(keys)},
        )

    def _list_all_tags(self) -> dict[str, dict[str, dict[str, Any]]]:
        start = time.perf_counter()
        all_tags = self.tq.kv_list()
        self._record_tq_event('kv_list', start, metrics={'partitions': len(all_tags)})
        return all_tags

    def _list_partition_tags(
        self,
        partition_id: str,
        *,
        context: LoraContext | None = None,
    ) -> dict[str, dict[str, Any]]:
        start = time.perf_counter()
        tags_by_key = self.tq.kv_list(partition_id=partition_id).get(partition_id, {})
        self._record_tq_event(
            'kv_list',
            start,
            context=context,
            partition_id=partition_id,
            metrics={'keys': len(tags_by_key)},
        )
        return tags_by_key

    @staticmethod
    def _groups_from_tags(tags: Iterable[dict[str, Any]]) -> list[PromptGroupMeta]:
        groups = []
        for tag in tags:
            if tag.get('record_type') != 'group':
                continue
            group = TransferQueueDataPlane._prompt_group_from_tag(tag)
            if group is not None:
                groups.append(group)
        return groups

    @staticmethod
    def _partition_from_tag(partition_id: str, tag: dict[str, Any]) -> PartitionMeta | None:
        try:
            context = LoraContext(
                tenant_id=tag['tenant_id'],
                training_run_id=tag['training_run_id'],
                base_model_id=tag['base_model_id'],
                adapter_name=tag['adapter_name'],
                tool_profile=tag.get('tool_profile', 'default'),
                reward_type=tag.get('reward_type', 'default'),
                algorithm=tag.get('algorithm', 'grpo'),
                rollout_profile=tag.get('rollout_profile', 'default'),
            )
            return PartitionMeta(
                context=context,
                partition_id=partition_id,
                target_groups=int(tag.get('target_groups', 0)),
                status=PartitionStatus(tag.get('partition_status', tag.get('status', PartitionStatus.ACTIVE.value))),
                created_at=float(tag.get('created_at', 0.0)) or 0.0,
                updated_at=float(tag.get('updated_at', 0.0)) or 0.0,
            )
        except (KeyError, ValueError):
            return None

    @staticmethod
    def _prompt_group_from_tag(tag: dict[str, Any]) -> PromptGroupMeta | None:
        try:
            context = LoraContext(
                tenant_id=tag['tenant_id'],
                training_run_id=tag['training_run_id'],
                base_model_id=tag['base_model_id'],
                adapter_name=tag['adapter_name'],
                tool_profile=tag.get('tool_profile', 'default'),
                reward_type=tag.get('reward_type', 'default'),
                algorithm=tag.get('algorithm', 'grpo'),
                rollout_profile=tag.get('rollout_profile', 'default'),
            )
            return PromptGroupMeta(
                context=context,
                partition_id=tag['partition_id'],
                group_id=str(tag['group_id']),
                rollout_policy_version=int(tag.get('rollout_policy_version', 0)),
                rollout_adapter_path=tag.get('rollout_adapter_path'),
                num_samples=int(tag.get('num_samples', 0)),
                sample_keys=list(tag.get('sample_keys') or []),
                status=PromptGroupStatus(tag.get('group_status', PromptGroupStatus.PENDING.value)),
                created_at=float(tag.get('created_at', 0.0)) or 0.0,
                updated_at=float(tag.get('updated_at', 0.0)) or 0.0,
            )
        except (KeyError, ValueError):
            return None

    def _record_tq_event(
        self,
        op: str,
        start: float,
        *,
        context: LoraContext | None = None,
        partition_id: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        payload = {f'{op}_latency_ms': (time.perf_counter() - start) * 1000.0}
        if metrics:
            payload.update(metrics)
        self.metrics_recorder.log_event(
            event=op,
            phase='tq',
            context=context,
            partition_id=partition_id,
            metrics=payload,
        )
