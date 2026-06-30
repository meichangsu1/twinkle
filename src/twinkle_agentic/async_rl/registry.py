# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import threading
from typing import Dict, Iterable, Optional

from .types import AdapterRecord, AdapterState, TrainingContext


class AdapterRegistry:
    """Runtime state table for one-LoRA-per-training-run async RL."""

    def __init__(self):
        self._records: dict[str, AdapterRecord] = {}
        self._lock = threading.RLock()

    def register(self,
                 context: TrainingContext,
                 *,
                 weight: float = 1.0,
                 state: AdapterState = AdapterState.ACTIVE) -> AdapterRecord:
        with self._lock:
            key = context.key
            existing = self._records.get(key)
            if existing is not None:
                return existing
            record = AdapterRecord(
                tenant_id=context.tenant_id,
                training_run_id=context.training_run_id,
                adapter_name=context.adapter_name,
                base_model_id=context.base_model_id,
                state=state,
                policy_version=context.policy_version,
                adapter_path=context.adapter_path,
                weight=weight,
            )
            self._records[key] = record
            return record

    def get(self, context: TrainingContext | str) -> AdapterRecord:
        key = context if isinstance(context, str) else context.key
        with self._lock:
            if key not in self._records:
                raise KeyError(f'unknown adapter context: {key}')
            return self._records[key]

    def list_records(self) -> list[AdapterRecord]:
        with self._lock:
            return list(self._records.values())

    def contexts(self) -> Iterable[str]:
        with self._lock:
            return tuple(self._records)

    def can_accept_rollout(self, context: TrainingContext) -> bool:
        record = self.get(context)
        return record.state == AdapterState.ACTIVE and not record.sync_in_progress

    def can_train(self, context: TrainingContext) -> bool:
        record = self.get(context)
        return (record.state == AdapterState.ACTIVE and not record.sync_in_progress
                and record.training_partition is None)

    def on_rollout_started(self, context: TrainingContext) -> None:
        with self._lock:
            record = self.get(context)
            record.in_flight_rollouts += 1
            record.touch()

    def on_rollout_finished(self, context: TrainingContext) -> None:
        with self._lock:
            record = self.get(context)
            record.in_flight_rollouts = max(0, record.in_flight_rollouts - 1)
            record.touch()

    def on_partition_created(self, context: TrainingContext, partition_id: str) -> None:
        with self._lock:
            record = self.get(context)
            record.live_partitions.add(partition_id)
            record.touch()

    def on_partition_cleared(self, context: TrainingContext, partition_id: str) -> None:
        with self._lock:
            record = self.get(context)
            record.live_partitions.discard(partition_id)
            if record.training_partition == partition_id:
                record.training_partition = None
            record.touch()

    def on_train_started(self, context: TrainingContext, partition_id: str) -> None:
        with self._lock:
            record = self.get(context)
            if record.training_partition is not None and record.training_partition != partition_id:
                raise RuntimeError(f'adapter {record.key} is already training {record.training_partition}')
            record.training_partition = partition_id
            record.touch()

    def on_train_finished(self, context: TrainingContext, partition_id: str) -> None:
        with self._lock:
            record = self.get(context)
            if record.training_partition == partition_id:
                record.training_partition = None
            record.touch()

    def on_weight_sync_started(self, context: TrainingContext) -> None:
        with self._lock:
            record = self.get(context)
            record.sync_in_progress = True
            record.touch()

    def on_weight_sync_finished(
        self,
        context: TrainingContext,
        *,
        adapter_path: str | None = None,
        policy_version: int | None = None,
    ) -> TrainingContext:
        with self._lock:
            record = self.get(context)
            record.sync_in_progress = False
            record.policy_version = record.policy_version + 1 if policy_version is None else policy_version
            if adapter_path is not None:
                record.adapter_path = adapter_path
            record.touch()
            return context.with_policy_version(record.policy_version, record.adapter_path)

    def mark_failed(self, context: TrainingContext, error: str) -> None:
        with self._lock:
            record = self.get(context)
            record.state = AdapterState.FAILED
            record.last_error = error
            record.touch()
