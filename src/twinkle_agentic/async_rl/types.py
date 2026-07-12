# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Awaitable, Iterable, Protocol, Sequence

from twinkle.data_format import InputFeature, Trajectory


class PartitionStatus(StrEnum):
    ACTIVE = 'ACTIVE'
    CLOSED = 'CLOSED'
    TRAIN_DONE = 'TRAIN_DONE'
    CLEARED = 'CLEARED'
    FAILED = 'FAILED'


class PromptGroupStatus(StrEnum):
    PENDING = 'PENDING'
    RUNNING = 'RUNNING'
    ROLLOUT_DONE = 'ROLLOUT_DONE'
    FAILED = 'FAILED'
    DROPPED = 'DROPPED'
    ADVANTAGING = 'ADVANTAGING'
    ADVANTAGE_DONE = 'ADVANTAGE_DONE'
    TRAINING = 'TRAINING'
    TRAIN_DONE = 'TRAIN_DONE'


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
    tool_profile: str = 'default'
    reward_type: str = 'default'
    algorithm: str = 'grpo'
    rollout_profile: str = 'default'

    @property
    def key(self) -> str:
        return f'{self.tenant_id}/{self.training_run_id}/{self.adapter_name}'

    def partition_id(self, train_id: int | str) -> str:
        suffix = train_id if isinstance(train_id, str) and train_id.startswith('train_') else f'train_{train_id}'
        return f'{self.key}/{suffix}'

    def metadata(self) -> dict[str, Any]:
        return {
            'tenant_id': self.tenant_id,
            'training_run_id': self.training_run_id,
            'base_model_id': self.base_model_id,
            'adapter_name': self.adapter_name,
            'tool_profile': self.tool_profile,
            'reward_type': self.reward_type,
            'algorithm': self.algorithm,
            'rollout_profile': self.rollout_profile,
        }

    def validate_metadata(self, metadata: dict[str, Any], *, strict_policy_version: bool = True) -> None:
        expected = self.metadata()
        for key, expected_value in expected.items():
            actual_value = metadata.get(key)
            if actual_value != expected_value:
                raise ValueError(
                    f'context metadata mismatch for {key}: expected {expected_value!r}, got {actual_value!r}')


@dataclass
class PartitionMeta:
    context: LoraContext
    partition_id: str
    target_groups: int = 0
    status: PartitionStatus = PartitionStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.updated_at = time.time()

    def tag(self) -> dict[str, Any]:
        tag = self.context.metadata()
        tag.update({
            'record_type': 'partition',
            'partition_id': self.partition_id,
            'target_groups': self.target_groups,
            'partition_status': self.status.value,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
        })
        return tag


@dataclass
class PromptGroupMeta:
    context: LoraContext
    partition_id: str
    group_id: str
    rollout_policy_version: int
    rollout_adapter_path: str | None = None
    num_samples: int = 0
    sample_keys: list[str] = field(default_factory=list)
    status: PromptGroupStatus = PromptGroupStatus.PENDING
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def key(self) -> str:
        return f'groups/{self.group_id}'

    def touch(self) -> None:
        self.updated_at = time.time()

    def tag(self) -> dict[str, Any]:
        tag = self.context.metadata()
        tag.update({
            'record_type': 'group',
            'partition_id': self.partition_id,
            'group_id': self.group_id,
            'rollout_policy_version': self.rollout_policy_version,
            'rollout_adapter_path': self.rollout_adapter_path,
            'num_samples': self.num_samples,
            'sample_keys': list(self.sample_keys),
            'group_status': self.status.value,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
        })
        return tag


class RolloutOutput(Trajectory, InputFeature, total=False):
    sample_id: str
    group_id: str
    generation_idx: int
    logprobs: list[float]
    rewards: float
    advantages: float
    returns: float
    turns: int
    stop_reason: str | None
    truncated: bool
    rollout_policy_version: int | None
    rollout_adapter_path: str | None
    metadata: dict[str, Any]


@dataclass
class TransformersTrainBatch:
    inputs: list[InputFeature]
    logprobs: list[list[float]]
    advantages: list[float]
    rewards: list[float]
    sample_keys: list[str]

    @property
    def sample_count(self) -> int:
        return len(self.sample_keys)


@dataclass
class GRPOAdvantageBatch:
    rewards: list[float]
    sample_keys: list[str]
    group_ids: list[str]
    generation_indices: list[int]
    num_generations: int

    @property
    def sample_count(self) -> int:
        return len(self.sample_keys)


@dataclass(frozen=True)
class PromptGroupRef:
    partition_id: str
    group_id: str

    @property
    def key(self) -> str:
        return f'groups/{self.group_id}'


@dataclass
class LoraRuntimeState:
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
    in_flight_groups: int = 0
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


@dataclass
class RolloutScheduleCandidate:
    context: LoraContext
    pending_groups: int
    in_flight_groups: int
    live_partitions: int
    active_partitions: int
    rollout_capacity: int
    rollout_cost: float = 1.0
    last_submit_time: float = 0.0
    submitted_groups: int = 0
    weight: float = 1.0

    @property
    def context_key(self) -> str:
        return self.context.key


@dataclass(frozen=True)
class TrainBatchCandidate:
    context: LoraContext
    partition: PartitionMeta
    available_groups: int

    @property
    def partition_id(self) -> str:
        return self.partition.partition_id

    @property
    def created_at(self) -> float:
        return self.partition.created_at


@dataclass(frozen=True)
class ComponentResult:
    component: str
    kind: str
    metadata: PartitionMeta | None = None
    count: int = 0


@dataclass(frozen=True)
class TrainStageResult:
    train_batches: int = 0
    trained_partitions: int = 0
    metadata: PartitionMeta | None = None

    @property
    def had_work(self) -> bool:
        return self.train_batches > 0 or self.trained_partitions > 0


@dataclass(frozen=True)
class PipelineStepResult:
    rollout: PartitionMeta | None = None
    advantage: PartitionMeta | None = None
    train: PartitionMeta | None = None
    prompt_groups: int = 0
    rollout_events: int = 0
    advantage_groups: int = 0
    train_batches: int = 0
    trained_partitions: int = 0

    @property
    def had_work(self) -> bool:
        return any((
            self.prompt_groups,
            self.rollout_events,
            self.advantage_groups,
            self.train_batches,
            self.trained_partitions,
        ))

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        if key not in {'rollout', 'advantage', 'train'}:
            raise KeyError(key)
        return getattr(self, key)

    def values(self) -> tuple[PartitionMeta | None, PartitionMeta | None, PartitionMeta | None]:
        return self.rollout, self.advantage, self.train


class RolloutCallable(Protocol):
    """Callable contract used by AsyncRollouter.

    Implementations may be a concrete Rollout subclass, a server-side adapter,
    or an async wrapper. The async RL layer only requires batched trajectory
    input and iterable sample output.
    """

    def __call__(
        self,
        trajectories: Sequence[Trajectory],
        **kwargs: Any,
    ) -> Iterable[RolloutOutput] | Awaitable[Iterable[RolloutOutput]]:
        ...
