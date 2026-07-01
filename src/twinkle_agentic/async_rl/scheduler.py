# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .data_plane import TransferQueueDataPlane
from .registry import AdapterRegistry
from .train_scheduling import (
    AdaptiveTrainPolicy,
    CostAwareTrainPolicy,
    DeficitFairTrainPolicy,
    EDFTrainPolicy,
    FIFOTrainPolicy,
    LRUTrainPolicy,
    PreferCurrentTrainPolicy,
    PriorityTrainPolicy,
    SJFTrainPolicy,
    StrideTrainPolicy,
    WeightedFairQueueTrainPolicy,
)
from .types import AdapterState, PartitionMetadata, PartitionStatus, TrainingContext


@dataclass
class TrainerSchedulerConfig:
    policy: str = 'prefer_current'
    switch_penalty: float = 0.0
    switch_cost: float = 1.0
    fairness_quantum: float = 1.0
    aging_rate: float = 0.1
    max_staleness: int = 1
    adaptive_high_load_threshold: float = 3.0
    adaptive_switch_rate_threshold: float = 0.7
    weights: Dict[str, float] = field(default_factory=dict)
    fairness_unit: str = 'partition'

    def build_policy(self, adapter_registry: AdapterRegistry) -> Any:
        if self.policy == 'prefer_current':
            return PreferCurrentTrainPolicy(switch_penalty=self.switch_penalty)
        if self.policy == 'cost_aware':
            return CostAwareTrainPolicy(switch_cost=self.switch_cost, adapter_registry=adapter_registry)
        if self.policy == 'sjf':
            return SJFTrainPolicy()
        if self.policy == 'fair':
            return DeficitFairTrainPolicy(
                quantum=self.fairness_quantum,
                adapter_registry=adapter_registry,
                weights=self.weights,
            )
        if self.policy == 'stride':
            return StrideTrainPolicy(adapter_registry=adapter_registry, weights=self.weights)
        if self.policy == 'wfq':
            return WeightedFairQueueTrainPolicy(adapter_registry=adapter_registry, weights=self.weights)
        if self.policy == 'lru':
            return LRUTrainPolicy()
        if self.policy == 'edf':
            return EDFTrainPolicy(max_staleness=self.max_staleness, adapter_registry=adapter_registry)
        if self.policy == 'priority':
            return PriorityTrainPolicy(
                aging_rate=self.aging_rate,
                adapter_registry=adapter_registry,
                weights=self.weights,
            )
        if self.policy == 'adaptive':
            return AdaptiveTrainPolicy(
                adapter_registry=adapter_registry,
                max_staleness=self.max_staleness,
                switch_cost=self.switch_cost,
                high_load_threshold=self.adaptive_high_load_threshold,
                switch_rate_threshold=self.adaptive_switch_rate_threshold,
            )
        if self.policy == 'fifo':
            return FIFOTrainPolicy()
        raise ValueError(f'unknown train schedule policy: {self.policy!r}')


@dataclass(frozen=True)
class RejectedPartition:
    partition_id: str
    context_key: str
    reason: str


@dataclass(frozen=True)
class ScheduleDecision:
    selected: Optional[PartitionMetadata]
    reason: str
    total_candidates: int
    eligible_count: int
    rejected: list[RejectedPartition]
    elapsed_ms: float


class TrainerScheduler:

    def __init__(
        self,
        *,
        adapter_registry: AdapterRegistry,
        data_plane: Optional[TransferQueueDataPlane] = None,
        train_policy: Optional[Any] = None,
        config: Optional[TrainerSchedulerConfig] = None,
        supported_reward_type: Optional[str] = None,
        supported_loss_type: Optional[str] = None,
        supported_algorithm: Optional[str] = None,
    ):
        self.adapter_registry = adapter_registry
        self.data_plane = data_plane
        self.config = config or TrainerSchedulerConfig()
        self.train_policy = train_policy or self.config.build_policy(adapter_registry)
        self.supported_reward_type = supported_reward_type
        self.supported_loss_type = supported_loss_type
        self.supported_algorithm = supported_algorithm

    def is_compatible(self, partition: PartitionMetadata) -> bool:
        """Check if partition's reward_type/loss_type/algorithm matches trainer config."""
        ctx = partition.context
        if self.supported_reward_type is not None and ctx.reward_type != self.supported_reward_type:
            return False
        if self.supported_loss_type is not None and ctx.loss_type != self.supported_loss_type:
            return False
        if self.supported_algorithm is not None and ctx.algorithm != self.supported_algorithm:
            return False
        return True

    def list_train_ready_partitions(self) -> list[PartitionMetadata]:
        """Query TransferQueueDataPlane for TRAIN_READY partitions."""
        if self.data_plane is None:
            return []
        return self.data_plane.list_train_ready_partitions()

    def _apply_gating(
        self,
        candidates: List[PartitionMetadata],
    ) -> tuple[list[PartitionMetadata], list[RejectedPartition]]:
        eligible: list[PartitionMetadata] = []
        rejected: list[RejectedPartition] = []

        for partition in candidates:
            if partition.status != PartitionStatus.TRAIN_READY:
                rejected.append(RejectedPartition(
                    partition_id=partition.partition_id,
                    context_key=partition.context.key,
                    reason='not_train_ready',
                ))
                continue

            try:
                record = self.adapter_registry.get(partition.context)
            except KeyError:
                rejected.append(RejectedPartition(
                    partition_id=partition.partition_id,
                    context_key=partition.context.key,
                    reason='unknown_adapter',
                ))
                continue

            if record.state in (AdapterState.FAILED, AdapterState.CANCELLED, AdapterState.DRAINING):
                rejected.append(RejectedPartition(
                    partition_id=partition.partition_id,
                    context_key=partition.context.key,
                    reason='adapter_terminal_state',
                ))
                continue

            if not self.adapter_registry.can_train(partition.context):
                rejected.append(RejectedPartition(
                    partition_id=partition.partition_id,
                    context_key=partition.context.key,
                    reason='cannot_train',
                ))
                continue

            if not self.is_compatible(partition):
                rejected.append(RejectedPartition(
                    partition_id=partition.partition_id,
                    context_key=partition.context.key,
                    reason='incompatible',
                ))
                continue

            eligible.append(partition)

        return eligible, rejected

    def next_partition(
        self,
        candidates: Optional[List[PartitionMetadata]] = None,
        current_context: Optional[TrainingContext] = None,
    ) -> Optional[PartitionMetadata]:
        """Select next partition to train.

        If candidates is None, queries TransferQueueDataPlane via list_train_ready_partitions().
        Otherwise uses the provided candidates list.
        """
        if candidates is None:
            candidates = self.list_train_ready_partitions()
        if not candidates:
            return None
        eligible, _ = self._apply_gating(candidates)
        if not eligible:
            return None
        return self.train_policy.pick_next_partition(eligible, current_context)

    def list_eligible_partitions(
        self,
        candidates: List[PartitionMetadata],
        current_context: Optional[TrainingContext] = None,
    ) -> list[PartitionMetadata]:
        eligible, _ = self._apply_gating(candidates)
        return eligible

    def get_schedule_decision(
        self,
        candidates: List[PartitionMetadata],
        current_context: Optional[TrainingContext] = None,
    ) -> ScheduleDecision:
        t0 = time.monotonic()
        total = len(candidates)

        if not candidates:
            return ScheduleDecision(
                selected=None,
                reason='no_candidates',
                total_candidates=0,
                eligible_count=0,
                rejected=[],
                elapsed_ms=_elapsed_ms(t0),
            )

        eligible, rejected = self._apply_gating(candidates)

        if not eligible:
            return ScheduleDecision(
                selected=None,
                reason='all_rejected',
                total_candidates=total,
                eligible_count=0,
                rejected=rejected,
                elapsed_ms=_elapsed_ms(t0),
            )

        selected = self.train_policy.pick_next_partition(eligible, current_context)
        reason = 'selected' if selected is not None else 'policy_returned_none'

        return ScheduleDecision(
            selected=selected,
            reason=reason,
            total_candidates=total,
            eligible_count=len(eligible),
            rejected=rejected,
            elapsed_ms=_elapsed_ms(t0),
        )


def _elapsed_ms(t0: float) -> float:
    return (time.monotonic() - t0) * 1000.0
