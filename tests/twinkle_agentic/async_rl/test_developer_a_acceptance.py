# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for Developer A acceptance criteria (详细设计 7.1):

1. TransferQueueRuntimeConfig basic fields
2. get_metadata returns list
3. Row field schema constants
4. Basic partition lifecycle
"""
from __future__ import annotations

from twinkle_agentic.async_rl import (
    PartitionStatus,
    TrainingContext,
    TransferQueueDataPlane,
    TransferQueueRuntimeConfig,
)

from .fakes import FakeTransferQueueClient

# ── Row field schema (defined locally for testing, documented in API doc) ──

ROLLOUT_FIELDS = frozenset({
    'messages',
    'group_id',
    'generation_idx',
    'old_logps',
})

REWARD_FIELDS = frozenset({
    'rewards',
})

ADVANTAGE_FIELDS = frozenset({
    'advantages',
    'returns',
})

TRAIN_REQUIRED_FIELDS = frozenset(
    ROLLOUT_FIELDS | REWARD_FIELDS | ADVANTAGE_FIELDS
)

SAMPLE_ISOLATION_TAG_FIELDS = frozenset({
    'tenant_id',
    'training_run_id',
    'base_model_id',
    'adapter_name',
    'adapter_revision',
    'policy_version',
    'partition_id',
    'group_id',
    'generation_idx',
})

TASK_NAMES = frozenset({'rollout', 'reward', 'advantage', 'train'})


def make_context(name='lora', *, tenant='tenant_a', run='run_001', version=0):
    return TrainingContext(
        tenant_id=tenant,
        training_run_id=run,
        base_model_id='Qwen/Qwen3.5-0.8B',
        adapter_name=name,
        policy_version=version,
        reward_type='constant',
        loss_type='grpo',
        algorithm='grpo',
    )


def make_sample(i=0, group_id=None):
    return {
        'sample_id': f'sample_{i}',
        'messages': [{'role': 'user', 'content': f'q{i}'}],
        'group_id': group_id or f'g{i}',
        'generation_idx': i % 4,
        'old_logps': [0.1, 0.2, 0.3],
    }


# ── TransferQueueRuntimeConfig ─────────────────────────────────────────────


class TestTransferQueueRuntimeConfig:

    def test_default_fields(self):
        config = TransferQueueRuntimeConfig()
        assert config.num_data_storage_units == 4
        assert config.storage_backend == 'SimpleStorage'
        assert config.init is True

    def test_max_rows_explicit(self):
        config = TransferQueueRuntimeConfig(max_rows=500)
        assert config.max_rows == 500

    def test_max_rows_per_context_explicit(self):
        config = TransferQueueRuntimeConfig(max_rows_per_context=100)
        assert config.max_rows_per_context == 100

    def test_max_rows_defaults_none(self):
        config = TransferQueueRuntimeConfig()
        assert config.max_rows is None

    def test_max_rows_per_context_defaults_none(self):
        config = TransferQueueRuntimeConfig()
        assert config.max_rows_per_context is None

    def test_check_capacity_with_max_rows(self):
        config = TransferQueueRuntimeConfig(max_rows=2)
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient(), tq_config=config)
        ctx = make_context()
        assert dp.check_capacity(ctx)

        p = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, p.partition_id, [make_sample(0), make_sample(1)], seal=True)
        assert not dp.check_capacity(ctx)

    def test_check_capacity_without_limits(self):
        config = TransferQueueRuntimeConfig()
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient(), tq_config=config)
        ctx = make_context()
        assert dp.check_capacity(ctx)


# ── get_metadata ─────────────────────────────────────────────────────────


class TestGetMetadata:

    def test_get_metadata_with_context_returns_list(self):
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
        ctx = make_context()
        dp.init_namespace(ctx)

        result = dp.get_metadata(ctx)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_get_metadata_without_context_returns_list(self):
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
        ctx = make_context()
        dp.create_partition(ctx, target_groups=1)

        result = dp.get_metadata()
        assert isinstance(result, list)

    def test_get_metadata_active_partitions(self):
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
        ctx = make_context()

        p0 = dp.create_partition(ctx, target_groups=1)
        p1 = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, p0.partition_id, [make_sample(0)], seal=True)
        dp.put_rollout_batch(ctx, p1.partition_id, [make_sample(1)], seal=True)

        partitions = dp.get_metadata(ctx)
        assert len(partitions) == 2

    def test_get_metadata_oldest_partition(self):
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
        ctx = make_context()

        p0 = dp.create_partition(ctx, target_groups=1)
        dp.create_partition(ctx, target_groups=1)

        partitions = dp.get_metadata(ctx)
        oldest = min(partitions, key=lambda p: (p.created_at, p.partition_id))
        assert oldest.partition_id == p0.partition_id

    def test_get_metadata_iterable(self):
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
        ctx = make_context()

        dp.create_partition(ctx, target_groups=1)
        dp.create_partition(ctx, target_groups=1)

        partitions = dp.get_metadata(ctx)
        assert len(list(partitions)) == 2

    def test_get_metadata_len(self):
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
        ctx = make_context()

        dp.create_partition(ctx, target_groups=1)
        partitions = dp.get_metadata(ctx)
        assert len(partitions) == 1


# ── Row field schema constants ────────────────────────────────────────────


class TestRowFieldSchema:

    def test_rollout_fields(self):
        assert 'messages' in ROLLOUT_FIELDS
        assert 'group_id' in ROLLOUT_FIELDS
        assert 'generation_idx' in ROLLOUT_FIELDS
        assert 'old_logps' in ROLLOUT_FIELDS

    def test_reward_fields(self):
        assert 'rewards' in REWARD_FIELDS

    def test_advantage_fields(self):
        assert 'advantages' in ADVANTAGE_FIELDS
        assert 'returns' in ADVANTAGE_FIELDS

    def test_train_required_fields_is_union(self):
        assert TRAIN_REQUIRED_FIELDS == ROLLOUT_FIELDS | REWARD_FIELDS | ADVANTAGE_FIELDS

    def test_sample_isolation_tag_fields(self):
        assert 'tenant_id' in SAMPLE_ISOLATION_TAG_FIELDS
        assert 'training_run_id' in SAMPLE_ISOLATION_TAG_FIELDS
        assert 'adapter_name' in SAMPLE_ISOLATION_TAG_FIELDS
        assert 'policy_version' in SAMPLE_ISOLATION_TAG_FIELDS
        assert 'partition_id' in SAMPLE_ISOLATION_TAG_FIELDS
        assert 'group_id' in SAMPLE_ISOLATION_TAG_FIELDS
        assert 'generation_idx' in SAMPLE_ISOLATION_TAG_FIELDS

    def test_task_names(self):
        assert TASK_NAMES == frozenset({'rollout', 'reward', 'advantage', 'train'})

    def test_build_streaming_dataloader_returns_samples(self):
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
        ctx = make_context()
        p = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, p.partition_id, [make_sample(0)], seal=True)
        dp.append_rewards(ctx, p.partition_id, [1.0])
        dp.append_advantages(ctx, p.partition_id, [0.5])

        samples = dp.build_streaming_dataloader(ctx, p.partition_id)
        assert len(samples) == 1
        assert 'rewards' in samples[0]
        assert 'advantages' in samples[0]
        assert 'returns' in samples[0]


# ── Basic partition lifecycle ─────────────────────────────────────────────


class TestPartitionLifecycle:

    def test_full_lifecycle(self):
        """End-to-end: rollout -> reward -> advantage -> train -> clear."""
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
        ctx = make_context()
        p = dp.create_partition(ctx, target_groups=1)

        dp.put_rollout_batch(ctx, p.partition_id, [make_sample(0)], seal=True)
        assert dp.list_partitions(ctx)[0].status == PartitionStatus.ROLLOUT_DONE

        meta, samples = dp.claim_reward_batch(ctx, batch_size=10)
        assert len(samples) == 1
        dp.append_rewards(ctx, p.partition_id, [1.0])
        assert dp.list_partitions(ctx)[0].status == PartitionStatus.REWARD_DONE

        meta, samples = dp.claim_advantage_batch(ctx, batch_size=10)
        dp.append_advantages(ctx, p.partition_id, [0.5])
        assert dp.list_partitions(ctx)[0].status == PartitionStatus.TRAIN_READY

        dp.mark_training(ctx, p.partition_id)
        assert dp.list_partitions(ctx)[0].status == PartitionStatus.TRAINING

        dataloader = dp.build_streaming_dataloader(ctx, p.partition_id)
        assert len(dataloader) == 1

        dp.mark_trained(ctx, p.partition_id)
        assert dp.list_partitions(ctx)[0].status == PartitionStatus.TRAIN_DONE

        dp.clear_partition(ctx, p.partition_id)
        assert dp.list_partitions(ctx)[0].status == PartitionStatus.CLEARED
