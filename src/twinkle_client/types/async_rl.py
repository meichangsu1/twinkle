# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared request and response models for the async-RL tenant control plane."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AsyncRLTenantStatus(StrEnum):
    ADDING = 'ADDING'
    ACTIVE = 'ACTIVE'
    DRAINING = 'DRAINING'
    REMOVED = 'REMOVED'
    FAILED = 'FAILED'


class AsyncRLDatasetConfig(BaseModel):
    model_config = ConfigDict(extra='allow')

    dataset_id: str
    max_length: int = Field(gt=0)


class AsyncRLRolloutConfig(BaseModel):
    model_config = ConfigDict(extra='allow')

    batch_size: int = Field(gt=0)
    num_generations: int = Field(gt=0)
    max_tokens: int = Field(ge=0)
    temperature: float = Field(ge=0)
    top_p: float = Field(gt=0, le=1)
    repetition_penalty: float = Field(default=1.0, gt=0)


class AsyncRLTrainConfig(BaseModel):
    model_config = ConfigDict(extra='allow')

    mini_batch_size: int = Field(gt=0)
    micro_batch_size: int = Field(gt=0)
    dynamic_batching: bool = False
    max_tokens_per_micro_batch: int | None = Field(default=None, gt=0)
    packing_algorithm: str = 'ffd'
    max_steps: int | None = Field(default=None, ge=0)


class AsyncRLRewardConfig(BaseModel):
    model_config = ConfigDict(extra='allow')

    class_path: str = Field(min_length=1)
    kwargs: dict[str, Any] = Field(default_factory=dict)


class AsyncRLTenantConfig(BaseModel):
    """One dynamic tenant; its shape matches one ``lora_contexts`` YAML item."""

    model_config = ConfigDict(extra='forbid')

    tenant_id: str
    training_run_id: str
    adapter_name: str
    dataset: AsyncRLDatasetConfig
    rollout: AsyncRLRolloutConfig
    train: AsyncRLTrainConfig
    reward: AsyncRLRewardConfig | None = None
    eval_dataset: AsyncRLDatasetConfig | None = None
    lora: dict[str, Any] | None = None
    loss: dict[str, Any] | None = None
    initial_adapter_path: str | None = None


class AsyncRLTenantInfo(BaseModel):
    context_id: str
    context_key: str
    tenant_id: str
    training_run_id: str
    adapter_name: str
    status: AsyncRLTenantStatus
    policy_version: int = 0
    live_partitions: int = 0
    completed_partitions: int = 0
    error: str | None = None


class AsyncRLTenantListResponse(BaseModel):
    tenants: list[AsyncRLTenantInfo] = Field(default_factory=list)
