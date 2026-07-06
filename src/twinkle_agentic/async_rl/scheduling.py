# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .types import LoraContext, RolloutScheduleCandidate, TrainBatchCandidate


def _oldest_train_candidate(candidates: Iterable[TrainBatchCandidate]) -> TrainBatchCandidate:
    return min(candidates, key=lambda c: (c.created_at, c.partition_id))


class WorkConservingRolloutPolicy:
    """Prefer contexts that are most likely to keep trainer fed."""

    def pick_next_context(self, candidates: list[RolloutScheduleCandidate]) -> LoraContext | None:
        candidates = [c for c in candidates if c.pending_groups > 0 and c.rollout_capacity > 0]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda c: (
                c.active_partitions > 0,
                c.live_partitions,
                c.in_flight_groups,
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

    def pick_next_context(self, candidates: list[RolloutScheduleCandidate]) -> LoraContext | None:
        candidates = [c for c in candidates if c.pending_groups > 0 and c.rollout_capacity > 0]
        if not candidates:
            return None
        candidates = sorted(candidates, key=lambda c: c.context_key)
        n = len(candidates)
        for i in range(n):
            idx = (self._cursor + i) % n
            candidate = candidates[idx]
            key = candidate.context_key
            self.deficit[key] += candidate.weight * self.quantum
            cost = 1.0
            if self.deficit[key] >= cost:
                self.deficit[key] -= cost
                self._cursor = (idx + 1) % n
                return candidate.context
        self._cursor = (self._cursor + 1) % n
        return None


class PreferCurrentTrainPolicy:
    """Keep current adapter if it has work; otherwise switch immediately."""

    def pick_next_batch(
        self,
        candidates: list[TrainBatchCandidate],
        current_context: LoraContext | None = None,
    ) -> TrainBatchCandidate | None:
        if not candidates:
            return None
        if current_context is not None:
            same = [candidate for candidate in candidates if candidate.context.key == current_context.key]
            if same:
                return _oldest_train_candidate(same)

        grouped: dict[str, list[TrainBatchCandidate]] = defaultdict(list)
        for candidate in candidates:
            grouped[candidate.context.key].append(candidate)
        selected_group = max(
            grouped.values(),
            key=lambda group:
            (sum(candidate.available_groups for candidate in group), -_oldest_train_candidate(group).created_at),
        )
        return _oldest_train_candidate(selected_group)


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

    def pick_next_batch(
        self,
        candidates: list[TrainBatchCandidate],
        current_context: LoraContext | None = None,
    ) -> TrainBatchCandidate | None:
        if not candidates:
            return None
        grouped: dict[str, list[TrainBatchCandidate]] = defaultdict(list)
        weights: dict[str, float] = {}
        for candidate in candidates:
            grouped[candidate.context.key].append(candidate)
            weights[candidate.context.key] = 1.0
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
                return _oldest_train_candidate(grouped[key])
        self._cursor = (self._cursor + 1) % n
        return None
