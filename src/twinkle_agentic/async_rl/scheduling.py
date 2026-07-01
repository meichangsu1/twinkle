# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional

from .types import LoraContext, PartitionMetadata, RolloutContextState


def _oldest_partition(partitions: Iterable[PartitionMetadata]) -> PartitionMetadata:
    return min(partitions, key=lambda p: (p.created_at, p.partition_id))


class WorkConservingRolloutPolicy:
    """Prefer contexts that are most likely to keep trainer fed."""

    def pick_next_context(self, candidates: list[RolloutContextState]) -> LoraContext | None:
        candidates = [c for c in candidates if c.pending_groups > 0 and c.rollout_capacity > 0]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda c: (
                c.open_partitions > 0,
                c.live_partitions,
                c.in_flight_rollouts,
                c.last_submit_time,
                c.context_key,
            ),
        ).context


class WeightedFairRolloutPolicy:
    """Weighted fair scheduling over rollout prompt groups.

    Implemented with deficit round-robin accounting: each context accrues
    `weight * quantum` credit and one prompt-group rollout costs 1 credit.
    """

    def __init__(self, quantum: float = 1.0):
        self.quantum = quantum
        self.deficit: dict[str, float] = defaultdict(float)
        self._cursor = 0

    def pick_next_context(self, candidates: list[RolloutContextState]) -> LoraContext | None:
        candidates = [c for c in candidates if c.pending_groups > 0 and c.rollout_capacity > 0]
        if not candidates:
            return None
        candidates = sorted(candidates, key=lambda c: c.context_key)
        n = len(candidates)
        for i in range(n):
            idx = (self._cursor + i) % n
            state = candidates[idx]
            key = state.context_key
            self.deficit[key] += state.weight * self.quantum
            cost = 1.0
            if self.deficit[key] >= cost:
                self.deficit[key] -= cost
                self._cursor = (idx + 1) % n
                return state.context
        self._cursor = (self._cursor + 1) % n
        return None


class PreferCurrentTrainPolicy:
    """Keep current adapter if it has work; otherwise switch immediately."""

    def pick_next_partition(
        self,
        candidates: list[PartitionMetadata],
        current_context: LoraContext | None = None,
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
        selected_group = max(
            grouped.values(),
            key=lambda group: (len(group), -_oldest_partition(group).created_at),
        )
        return _oldest_partition(selected_group)


class WeightedFairTrainPolicy:
    """Weighted fair scheduling over train candidates.

    Implemented with deficit round-robin accounting. The current MVP uses
    equal context weights and returns the oldest partition for the selected
    context.
    """

    def __init__(self, quantum: float = 1.0):
        self.quantum = quantum
        self.deficit: dict[str, float] = defaultdict(float)
        self._cursor = 0

    def pick_next_partition(
        self,
        candidates: list[PartitionMetadata],
        current_context: LoraContext | None = None,
    ) -> PartitionMetadata | None:
        if not candidates:
            return None
        grouped: dict[str, list[PartitionMetadata]] = defaultdict(list)
        weights: dict[str, float] = {}
        for partition in candidates:
            grouped[partition.context.key].append(partition)
            weights[partition.context.key] = 1.0
        keys = sorted(grouped)
        n = len(keys)
        for i in range(n):
            idx = (self._cursor + i) % n
            key = keys[idx]
            self.deficit[key] += weights[key] * self.quantum
            cost = 1.0
            if self.deficit[key] >= cost:
                self.deficit[key] -= cost
                self._cursor = (idx + 1) % n
                return _oldest_partition(grouped[key])
        self._cursor = (self._cursor + 1) % n
        return None
