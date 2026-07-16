# Copyright (c) ModelScope Contributors. All rights reserved.
"""TransferQueue sampler for context-specific GRPO group sizes."""

from __future__ import annotations

from transfer_queue import GRPOGroupNSampler
from typing import Any


class ContextGRPOGroupNSampler(GRPOGroupNSampler):
    """Select complete prompt groups using the request's generation count."""

    def sample(
        self,
        ready_indexes: list[int],
        batch_size: int,
        task_name: str = '',
        partition_id: str = '',
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[int], list[int]]:
        group_size = int(kwargs['n_samples_per_prompt'])
        if group_size <= 0:
            raise ValueError(f'n_samples_per_prompt must be positive, got {group_size}')
        if batch_size % group_size:
            raise ValueError(f'batch_size ({batch_size}) must be a multiple of n_samples_per_prompt ({group_size})')

        states = self._states.get(partition_id, {}).get(task_name, {})
        dp_rank = kwargs.get('dp_rank')
        batch_index = kwargs.get('batch_index')
        if dp_rank in states and batch_index in states[dp_rank]:
            return states[dp_rank][batch_index]

        ready = sorted(ready_indexes)
        selected: list[int] = []
        offset = 0
        while offset <= len(ready) - group_size and len(selected) < batch_size:
            group = ready[offset:offset + group_size]
            if all(right - left == 1 for left, right in zip(group, group[1:])):
                selected.extend(group)
                offset += group_size
            else:
                offset += 1

        if len(selected) != batch_size:
            return [], []

        result = (selected, selected.copy())
        if dp_rank is not None:
            states.setdefault(dp_rank, {})[batch_index] = result
            self._states.setdefault(partition_id, {})[task_name] = states
        return result
