# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass

from .types import LoraContext, PromptGroupMeta, PromptGroupStatus


@dataclass
class StalenessManager:
    max_staleness: int = 0
    target_groups_per_partition: int = 1

    def can_create_next_rollout_partition(
        self,
        context: LoraContext,
        *,
        current_policy_version: int,
        groups: list[PromptGroupMeta] | None = None,
    ) -> bool:
        live_groups = [
            group for group in groups or [] if group.context.key == context.key and group.status not in {
                PromptGroupStatus.TRAIN_DONE,
                PromptGroupStatus.FAILED,
                PromptGroupStatus.DROPPED,
            }
        ]
        if not live_groups:
            return True
        oldest_policy_version = min(group.rollout_policy_version for group in live_groups)
        return current_policy_version - oldest_policy_version <= self.max_staleness

    def is_group_too_stale(
        self,
        *,
        rollout_policy_version: int,
        current_policy_version: int,
    ) -> bool:
        return current_policy_version - rollout_policy_version > self.max_staleness
