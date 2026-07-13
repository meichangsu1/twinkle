# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import threading
from typing import Iterable

from .types import LoraAdapterState, LoraContext, LoraRuntimeState


class LoraRuntimeRegistry:
    """Runtime state table for one-LoRA-per-training-run async RL."""

    def __init__(self):
        self._states: dict[str, LoraRuntimeState] = {}
        self._lock = threading.RLock()

    def register(self,
                 context: LoraContext,
                 *,
                 weight: float = 1.0,
                 state: LoraAdapterState = LoraAdapterState.ACTIVE) -> LoraRuntimeState:
        with self._lock:
            key = context.key
            existing = self._states.get(key)
            if existing is not None:
                return existing
            runtime_state = LoraRuntimeState(
                tenant_id=context.tenant_id,
                training_run_id=context.training_run_id,
                adapter_name=context.adapter_name,
                base_model_id=context.base_model_id,
                state=state,
                policy_version=0,
                adapter_path=None,
                weight=weight,
            )
            self._states[key] = runtime_state
            return runtime_state

    def get(self, context: LoraContext | str) -> LoraRuntimeState:
        key = context if isinstance(context, str) else context.key
        with self._lock:
            if key not in self._states:
                raise KeyError(f'unknown adapter context: {key}')
            return self._states[key]

    def list_states(self) -> list[LoraRuntimeState]:
        with self._lock:
            return list(self._states.values())

    def contexts(self) -> Iterable[str]:
        with self._lock:
            return tuple(self._states)

    def can_accept_rollout(self, context: LoraContext) -> bool:
        runtime_state = self.get(context)
        return runtime_state.state == LoraAdapterState.ACTIVE and not runtime_state.sync_in_progress

    def can_train(self, context: LoraContext) -> bool:
        runtime_state = self.get(context)
        return (runtime_state.state == LoraAdapterState.ACTIVE and not runtime_state.sync_in_progress
                and runtime_state.training_partition is None)

    def on_rollout_started(self, context: LoraContext) -> None:
        with self._lock:
            runtime_state = self.get(context)
            runtime_state.in_flight_groups += 1
            runtime_state.touch()

    def on_rollout_finished(self, context: LoraContext) -> None:
        with self._lock:
            runtime_state = self.get(context)
            runtime_state.in_flight_groups = max(0, runtime_state.in_flight_groups - 1)
            runtime_state.touch()

    def on_partition_created(self, context: LoraContext, partition_id: str) -> None:
        with self._lock:
            runtime_state = self.get(context)
            runtime_state.live_partitions.add(partition_id)
            runtime_state.touch()

    def on_partition_cleared(self, context: LoraContext, partition_id: str) -> None:
        with self._lock:
            runtime_state = self.get(context)
            runtime_state.live_partitions.discard(partition_id)
            if runtime_state.training_partition == partition_id:
                runtime_state.training_partition = None
            runtime_state.touch()

    def on_train_started(self, context: LoraContext, partition_id: str) -> None:
        with self._lock:
            runtime_state = self.get(context)
            if runtime_state.training_partition is not None and runtime_state.training_partition != partition_id:
                raise RuntimeError(
                    f'adapter {runtime_state.key} is already training {runtime_state.training_partition}')
            runtime_state.training_partition = partition_id
            runtime_state.touch()

    def on_train_finished(self, context: LoraContext, partition_id: str) -> None:
        with self._lock:
            runtime_state = self.get(context)
            if runtime_state.training_partition == partition_id:
                runtime_state.training_partition = None
            runtime_state.touch()

    def on_weight_sync_started(self, context: LoraContext) -> None:
        with self._lock:
            runtime_state = self.get(context)
            runtime_state.sync_in_progress = True
            runtime_state.touch()

    def on_weight_sync_finished(
        self,
        context: LoraContext,
        *,
        adapter_path: str | None = None,
        policy_version: int | None = None,
    ) -> LoraRuntimeState:
        with self._lock:
            runtime_state = self.get(context)
            runtime_state.sync_in_progress = False
            runtime_state.policy_version = (
                runtime_state.policy_version + 1 if policy_version is None else policy_version)
            if adapter_path is not None:
                runtime_state.adapter_path = adapter_path
            runtime_state.touch()
            return runtime_state

    def mark_failed(self, context: LoraContext, error: str) -> None:
        with self._lock:
            runtime_state = self.get(context)
            runtime_state.state = LoraAdapterState.FAILED
            runtime_state.last_error = error
            runtime_state.touch()
