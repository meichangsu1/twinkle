# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from .types import RolloutContextState, TrainingContext


class WorkConservingRolloutPolicy:
    """Prefer contexts that are most likely to keep trainer fed."""

    def pick_next_context(self, candidates: list[RolloutContextState]) -> TrainingContext | None:
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


class DeficitFairRolloutPolicy:
    """Weighted deficit round-robin over rollout prompt groups."""

    def __init__(self, quantum: float = 1.0):
        self.quantum = quantum
        self.deficit: dict[str, float] = defaultdict(float)
        self._cursor = 0

    def pick_next_context(self, candidates: list[RolloutContextState]) -> TrainingContext | None:
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
