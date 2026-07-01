# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Awaitable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple


class PartitionStatus(StrEnum):
    OPEN = 'OPEN'
    SEALED = 'SEALED'
    TRAIN_DONE = 'TRAIN_DONE'
    CLEARED = 'CLEARED'
    FAILED = 'FAILED'
    CANCELLED = 'CANCELLED'
    # Backward-compatible names. New async RL flow tracks these states on
    # GroupMetadata instead of PartitionMetadata.
    ROLLOUT_DONE = 'ROLLOUT_DONE'
    REWARD_DONE = 'REWARD_DONE'
    TRAIN_READY = 'TRAIN_READY'
    TRAINING = 'TRAINING'


class GroupStatus(StrEnum):
    ROLLOUT_DONE = 'ROLLOUT_DONE'
    REWARDING = 'REWARDING'
    REWARD_DONE = 'REWARD_DONE'
    ADVANTAGING = 'ADVANTAGING'
    ADVANTAGE_DONE = 'ADVANTAGE_DONE'
    TRAINING = 'TRAINING'
    TRAIN_DONE = 'TRAIN_DONE'
    FAILED = 'FAILED'
    DROPPED = 'DROPPED'


class LoraAdapterState(StrEnum):
    LOADING = 'LOADING'
    ACTIVE = 'ACTIVE'
    DRAINING = 'DRAINING'
    CANCELLED = 'CANCELLED'
    FAILED = 'FAILED'


@dataclass(frozen=True)
class LoraContext:
    tenant_id: str
    training_run_id: str
    base_model_id: str
    adapter_name: str
    adapter_path: str | None = None
    policy_version: int = 0
    tool_profile: str = 'default'
    reward_type: str = 'default'
    algorithm: str = 'grpo'

    @property
    def key(self) -> str:
        return f'{self.tenant_id}/{self.training_run_id}/{self.adapter_name}'

    def partition_id(self, train_id: int | str) -> str:
        suffix = train_id if isinstance(train_id, str) and train_id.startswith('train_') else f'train_{train_id}'
        return f'{self.key}/{suffix}'

    def with_policy_version(self, policy_version: int, adapter_path: str | None = None) -> LoraContext:
        return LoraContext(
            tenant_id=self.tenant_id,
            training_run_id=self.training_run_id,
            base_model_id=self.base_model_id,
            adapter_name=self.adapter_name,
            adapter_path=self.adapter_path if adapter_path is None else adapter_path,
            policy_version=policy_version,
            tool_profile=self.tool_profile,
            reward_type=self.reward_type,
            algorithm=self.algorithm,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            'tenant_id': self.tenant_id,
            'training_run_id': self.training_run_id,
            'base_model_id': self.base_model_id,
            'adapter_name': self.adapter_name,
            'adapter_path': self.adapter_path,
            'policy_version': self.policy_version,
            'tool_profile': self.tool_profile,
            'reward_type': self.reward_type,
            'algorithm': self.algorithm,
        }

    def validate_metadata(self, metadata: dict[str, Any], *, strict_policy_version: bool = True) -> None:
        expected = self.metadata()
        for key, expected_value in expected.items():
            if key == 'adapter_path':
                continue
            if key == 'policy_version' and not strict_policy_version:
                continue
            actual_value = metadata.get(key)
            if actual_value != expected_value:
                raise ValueError(
                    f'context metadata mismatch for {key}: expected {expected_value!r}, got {actual_value!r}')


@dataclass
class PartitionMetadata:
    context: LoraContext
    partition_id: str
    policy_version: int
    target_groups: int = 0
    ready_groups: int = 0
    status: PartitionStatus = PartitionStatus.OPEN
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    num_rows: int = 0

    @property
    def logical_train_id(self) -> str:
        return self.partition_id.rsplit('/', 1)[-1]

    def touch(self) -> None:
        self.updated_at = time.time()

    def tag(self) -> dict[str, Any]:
        tag = self.context.metadata()
        # Sample-level policy_version / adapter_path must remain attached
        # to each row. Partition tags carry lifecycle state and the version that
        # opened the partition, but must not overwrite row generation metadata.
        tag.pop('policy_version', None)
        tag.pop('adapter_path', None)
        tag.update({
            'partition_id': self.partition_id,
            'partition_policy_version': self.policy_version,
            'target_groups': self.target_groups,
            'ready_groups': self.ready_groups,
            'partition_status': self.status.value,
            'num_rows': self.num_rows,
        })
        return tag


@dataclass
class GroupMetadata:
    context: LoraContext
    partition_id: str
    group_id: str
    policy_version: int
    adapter_path: str | None = None
    num_samples: int = 0
    status: GroupStatus = GroupStatus.ROLLOUT_DONE
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def key(self) -> str:
        return f'{self.partition_id}/groups/{self.group_id}'

    def touch(self) -> None:
        self.updated_at = time.time()

    def tag(self) -> dict[str, Any]:
        tag = self.context.metadata()
        tag.update({
            'record_type': 'group',
            'partition_id': self.partition_id,
            'group_id': self.group_id,
            'group_policy_version': self.policy_version,
            'group_adapter_path': self.adapter_path,
            'num_samples': self.num_samples,
            'group_status': self.status.value,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
        })
        return tag


@dataclass
class GroupBatch:
    context: LoraContext
    partition_id: str
    groups: list[GroupMetadata]
    samples: list[SampleRecord]

    @property
    def group_count(self) -> int:
        return len(self.groups)

    @property
    def sample_count(self) -> int:
        return len(self.samples)


@dataclass
class AdapterRecord:
    tenant_id: str
    training_run_id: str
    adapter_name: str
    base_model_id: str
    state: LoraAdapterState = LoraAdapterState.LOADING
    policy_version: int = 0
    adapter_path: str | None = None
    train_slot_name: str | None = None
    rollout_slot_name: str | None = None
    live_partitions: set[str] = field(default_factory=set)
    in_flight_rollouts: int = 0
    training_partition: str | None = None
    sync_in_progress: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_error: str | None = None
    weight: float = 1.0

    @property
    def key(self) -> str:
        return f'{self.tenant_id}/{self.training_run_id}/{self.adapter_name}'

    def touch(self) -> None:
        self.updated_at = time.time()


@dataclass(frozen=True)
class RolloutCapacity:
    available_groups: int
    action: str = 'submit'
    reason: str = ''
    sleep_seconds: float = 0.0

    @property
    def can_submit(self) -> bool:
        return self.available_groups > 0 and self.action == 'submit'


@dataclass
class RolloutContextState:
    context: LoraContext
    pending_groups: int
    in_flight_rollouts: int
    live_partitions: int
    open_partitions: int
    train_ready_partitions: int
    rollout_capacity: int
    last_submit_time: float = 0.0
    submitted_groups: int = 0
    weight: float = 1.0

    @property
    def context_key(self) -> str:
        return self.context.key


@dataclass(frozen=True)
class WorkerResult:
    component: str
    kind: str
    metadata: PartitionMetadata | None = None
    count: int = 0


SampleRecord = Dict[str, Any]
RewardFn = Any
AdvantageFn = Any
TrainResult = Dict[str, Any]
ContextKey = Tuple[str, str, str]


class RolloutCallable(Protocol):
    """Callable contract used by AsyncRollouter.

    Implementations may be a concrete Rollout subclass, a server-side adapter,
    or an async wrapper. The async RL layer only requires batched trajectory
    input and iterable sample-row output.
    """

    def __call__(
        self,
        trajectories: Sequence[Any],
        **kwargs: Any,
    ) -> Iterable[SampleRecord] | Awaitable[Iterable[SampleRecord]]:
        ...
