# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

from .types import PartitionMetadata, TrainingContext


def _oldest_partition(partitions: Iterable[PartitionMetadata]) -> PartitionMetadata:
    return min(partitions, key=lambda p: (p.created_at, p.partition_id))


class PreferCurrentTrainPolicy:
    """Keep current adapter if it has work; otherwise switch immediately."""

    def __init__(self, switch_penalty: float = 0.0):
        self.switch_penalty = switch_penalty

    def pick_next_partition(
        self,
        candidates: list[PartitionMetadata],
        current_context: TrainingContext | None = None,
    ) -> PartitionMetadata | None:
        if not candidates:
            return None
        if current_context is not None:
            same = [p for p in candidates if p.context.key == current_context.key]
            if same:
                return _oldest_partition(same)

        grouped: dict[str, list[PartitionMetadata]] = defaultdict(list)
        for partition in candidates:
            grouped[partition.context.key].append(partition)

        if self.switch_penalty > 0 and current_context is not None:
            best_group = max(grouped.values(), key=lambda g: len(g))
            if len(best_group) < self.switch_penalty:
                return None

        selected_group = max(
            grouped.values(),
            key=lambda group: (len(group), -_oldest_partition(group).created_at),
        )
        return _oldest_partition(selected_group)


class CostAwareTrainPolicy:
    """Explicitly model LoRA switch cost and batch same-adapter partitions."""

    def __init__(self, switch_cost: float = 1.0, adapter_registry: Optional[Any] = None):
        self.switch_cost = switch_cost
        self.adapter_registry = adapter_registry
        self._last_context_key: Optional[str] = None

    def pick_next_partition(
        self,
        candidates: List[PartitionMetadata],
        current_context: Optional[TrainingContext] = None,
    ) -> Optional[PartitionMetadata]:
        if not candidates:
            return None

        grouped: Dict[str, list[PartitionMetadata]] = defaultdict(list)
        for p in candidates:
            grouped[p.context.key].append(p)

        current_key = current_context.key if current_context else self._last_context_key

        if current_key and current_key in grouped:
            self._last_context_key = current_key
            return _oldest_partition(grouped[current_key])

        best_key = max(
            grouped.keys(),
            key=lambda k: len(grouped[k]) - (self.switch_cost if k != current_key else 0.0),
        )
        self._last_context_key = best_key
        return _oldest_partition(grouped[best_key])


class SJFTrainPolicy:
    """Shortest job first: pick the partition with fewest rows."""

    def pick_next_partition(
        self,
        candidates: List[PartitionMetadata],
        current_context: Optional[TrainingContext] = None,
    ) -> Optional[PartitionMetadata]:
        if not candidates:
            return None
        return min(candidates, key=lambda p: (p.num_rows, p.created_at, p.partition_id))


class DeficitFairTrainPolicy:
    """Weighted deficit round-robin over train_k partitions."""

    def __init__(
        self,
        quantum: float = 1.0,
        adapter_registry: Optional[Any] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.quantum = quantum
        self.adapter_registry = adapter_registry
        self.weights = weights or {}
        self.deficit: Dict[str, float] = defaultdict(float)
        self._cursor = 0

    def _get_weight(self, context_key: str) -> float:
        if context_key in self.weights:
            return self.weights[context_key]
        if self.adapter_registry is not None:
            try:
                return self.adapter_registry.get(context_key).weight
            except KeyError:
                pass
        return 1.0

    def pick_next_partition(
        self,
        candidates: list[PartitionMetadata],
        current_context: TrainingContext | None = None,
    ) -> PartitionMetadata | None:
        if not candidates:
            return None
        grouped: Dict[str, list[PartitionMetadata]] = defaultdict(list)
        for partition in candidates:
            grouped[partition.context.key].append(partition)
        keys = sorted(grouped)
        n = len(keys)
        for i in range(n):
            idx = (self._cursor + i) % n
            key = keys[idx]
            self.deficit[key] += self._get_weight(key) * self.quantum
            cost = 1.0
            if self.deficit[key] >= cost:
                self.deficit[key] -= cost
                self._cursor = (idx + 1) % n
                return _oldest_partition(grouped[key])
        self._cursor = (self._cursor + 1) % n
        return None


class StrideTrainPolicy:
    """Deterministic proportional-share scheduling (Xen-style stride)."""

    STRIDE_UNIT = 1000000

    def __init__(
        self,
        adapter_registry: Optional[Any] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.adapter_registry = adapter_registry
        self.weights = weights or {}
        self._pass_values: Dict[str, int] = defaultdict(int)

    def _get_stride(self, context_key: str) -> int:
        weight = self.weights.get(context_key)
        if weight is None and self.adapter_registry is not None:
            try:
                weight = self.adapter_registry.get(context_key).weight
            except KeyError:
                weight = 1.0
        if weight is None:
            weight = 1.0
        return int(self.STRIDE_UNIT / max(weight, 0.01))

    def pick_next_partition(
        self,
        candidates: List[PartitionMetadata],
        current_context: Optional[TrainingContext] = None,
    ) -> Optional[PartitionMetadata]:
        if not candidates:
            return None
        grouped: Dict[str, list[PartitionMetadata]] = defaultdict(list)
        for p in candidates:
            grouped[p.context.key].append(p)
        best_key = min(grouped.keys(), key=lambda k: self._pass_values[k])
        self._pass_values[best_key] += self._get_stride(best_key)
        return _oldest_partition(grouped[best_key])


class WeightedFairQueueTrainPolicy:
    """Weighted fair queuing with virtual finish time (Demers et al. 1989)."""

    def __init__(
        self,
        adapter_registry: Optional[Any] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.adapter_registry = adapter_registry
        self.weights = weights or {}
        self._virtual_time: float = 0.0
        self._vft: Dict[str, float] = {}

    def _get_weight(self, context_key: str) -> float:
        if context_key in self.weights:
            return self.weights[context_key]
        if self.adapter_registry is not None:
            try:
                return self.adapter_registry.get(context_key).weight
            except KeyError:
                pass
        return 1.0

    def pick_next_partition(
        self,
        candidates: List[PartitionMetadata],
        current_context: Optional[TrainingContext] = None,
    ) -> Optional[PartitionMetadata]:
        if not candidates:
            return None
        grouped: Dict[str, list[PartitionMetadata]] = defaultdict(list)
        for p in candidates:
            grouped[p.context.key].append(p)
        active_keys = list(grouped.keys())
        total_w = sum(self._get_weight(k) for k in active_keys)
        for key in active_keys:
            if key not in self._vft:
                self._vft[key] = self._virtual_time
        best_key = min(active_keys, key=lambda k: self._vft[k])
        w = self._get_weight(best_key)
        self._vft[best_key] += 1.0 / w
        self._virtual_time += 1.0 / total_w
        return _oldest_partition(grouped[best_key])


class LRUTrainPolicy:
    """Least recently used: pick the adapter trained longest ago."""

    def __init__(self):
        self._last_trained_at: Dict[str, float] = {}

    def pick_next_partition(
        self,
        candidates: List[PartitionMetadata],
        current_context: Optional[TrainingContext] = None,
    ) -> Optional[PartitionMetadata]:
        if not candidates:
            return None
        grouped: Dict[str, list[PartitionMetadata]] = defaultdict(list)
        for p in candidates:
            grouped[p.context.key].append(p)
        best_key = min(grouped.keys(), key=lambda k: self._last_trained_at.get(k, 0.0))
        self._last_trained_at[best_key] = time.time()
        return _oldest_partition(grouped[best_key])


class EDFTrainPolicy:
    """Earliest deadline first: pick the most urgent partition.

    Urgency combines age, live partition pressure and in-flight rollouts.
    """

    def __init__(self, max_staleness: int = 1, adapter_registry: Optional[Any] = None):
        self.max_staleness = max_staleness
        self.adapter_registry = adapter_registry

    def _urgency(self, partition: PartitionMetadata) -> float:
        score = time.time() - partition.created_at
        if self.adapter_registry is not None:
            try:
                record = self.adapter_registry.get(partition.context)
                max_live = self.max_staleness + 1
                score += (len(record.live_partitions) / max_live) * 100.0
                score += record.in_flight_rollouts * 10.0
            except KeyError:
                pass
        return -score

    def pick_next_partition(
        self,
        candidates: List[PartitionMetadata],
        current_context: Optional[TrainingContext] = None,
    ) -> Optional[PartitionMetadata]:
        if not candidates:
            return None
        return min(candidates, key=lambda p: self._urgency(p))


class PriorityTrainPolicy:
    """Static priority with aging to prevent starvation."""

    def __init__(
        self,
        aging_rate: float = 0.1,
        adapter_registry: Optional[Any] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.aging_rate = aging_rate
        self.adapter_registry = adapter_registry
        self.weights = weights or {}
        self._wait_start: Dict[str, float] = {}

    def _base_priority(self, context_key: str) -> float:
        if context_key in self.weights:
            return self.weights[context_key]
        if self.adapter_registry is not None:
            try:
                return self.adapter_registry.get(context_key).weight
            except KeyError:
                pass
        return 1.0

    def pick_next_partition(
        self,
        candidates: List[PartitionMetadata],
        current_context: Optional[TrainingContext] = None,
    ) -> Optional[PartitionMetadata]:
        if not candidates:
            return None
        now = time.time()
        grouped: Dict[str, list[PartitionMetadata]] = defaultdict(list)
        for p in candidates:
            grouped[p.context.key].append(p)
            if p.context.key not in self._wait_start:
                self._wait_start[p.context.key] = p.created_at
        best_key = max(
            grouped.keys(),
            key=lambda k: self._base_priority(k) + (now - self._wait_start.get(k, now)) * self.aging_rate,
        )
        self._wait_start[best_key] = now
        return _oldest_partition(grouped[best_key])


class AdaptiveTrainPolicy:
    """Dynamically switch strategy based on system load."""

    def __init__(
        self,
        adapter_registry: Optional[Any] = None,
        max_staleness: int = 1,
        switch_cost: float = 1.0,
        high_load_threshold: float = 3.0,
        switch_rate_threshold: float = 0.7,
        window_size: int = 20,
    ):
        self.adapter_registry = adapter_registry
        self._policies: Dict[str, Any] = {
            'prefer_current': PreferCurrentTrainPolicy(),
            'cost_aware': CostAwareTrainPolicy(switch_cost=switch_cost, adapter_registry=adapter_registry),
            'lru': LRUTrainPolicy(),
            'edf': EDFTrainPolicy(max_staleness=max_staleness, adapter_registry=adapter_registry),
        }
        self._history: list[str] = []
        self._window_size = window_size
        self._high_load_threshold = high_load_threshold
        self._switch_rate_threshold = switch_rate_threshold

    @property
    def current_policy_name(self) -> Optional[str]:
        return self._history[-1] if self._history else None

    def pick_next_partition(
        self,
        candidates: List[PartitionMetadata],
        current_context: Optional[TrainingContext] = None,
    ) -> Optional[PartitionMetadata]:
        if not candidates:
            return None
        n_adapters = len(set(p.context.key for p in candidates))
        load_ratio = len(candidates) / max(n_adapters, 1)

        name = 'prefer_current'
        if len(self._history) >= self._window_size:
            recent = self._history[-self._window_size:]
            switch_count = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i - 1])
            switch_rate = switch_count / (self._window_size - 1)
            if switch_rate > self._switch_rate_threshold:
                name = 'prefer_current'
            elif load_ratio >= self._high_load_threshold:
                name = 'edf'
            elif load_ratio >= 1.5:
                name = 'cost_aware'
            else:
                name = 'prefer_current'
        else:
            if load_ratio >= self._high_load_threshold:
                name = 'edf'
            elif load_ratio >= 1.5:
                name = 'cost_aware'

        self._history.append(name)
        return self._policies[name].pick_next_partition(candidates, current_context)


class FIFOTrainPolicy:
    """Strict first-in-first-out by partition creation time."""

    def pick_next_partition(
        self,
        candidates: List[PartitionMetadata],
        current_context: Optional[TrainingContext] = None,
    ) -> Optional[PartitionMetadata]:
        if not candidates:
            return None
        return _oldest_partition(candidates)
