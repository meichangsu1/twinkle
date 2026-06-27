# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for Developer A acceptance criteria (详细设计 7.1):

1. Capacity auto-calculation: target_groups * num_generations * (max_staleness + 1)
2. QueueMetadata aggregate query
3. claim_reward_ready_groups (group-level claim)
4. Row field schema constants
5. Multi-worker claim exclusion (same sample not claimed by multiple workers)
"""
from __future__ import annotations

import pytest

from twinkle_agentic.async_rl import (
    QueueMetadata,
    PartitionStatus,
    TaskName,
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


# ── Capacity auto-calculation ─────────────────────────────────────────────


class TestCapacityAutoCalculation:

    def test_max_rows_auto_calculated_default(self):
        config = TransferQueueRuntimeConfig()
        assert config.max_rows == 128 * 8 * (1 + 1)

    def test_max_rows_auto_calculated_custom(self):
        config = TransferQueueRuntimeConfig(target_groups=64, num_generations=4, max_staleness=0)
        assert config.max_rows == 64 * 4 * 1

    def test_max_live_partitions_per_context_auto(self):
        config = TransferQueueRuntimeConfig(max_staleness=2)
        assert config.max_live_partitions_per_context == 3

    def test_max_tq_bytes_auto_calculated(self):
        config = TransferQueueRuntimeConfig(
            target_groups=10, num_generations=2, max_staleness=0,
            estimate_bytes_per_sample=1000, safety_factor=1.5,
        )
        expected = int(1000 * (10 * 2 * 1) * 1.5)
        assert config.max_tq_bytes == expected

    def test_max_tq_bytes_none_when_no_estimate(self):
        config = TransferQueueRuntimeConfig()
        assert config.max_tq_bytes is None

    def test_max_rows_explicit_overrides_auto(self):
        config = TransferQueueRuntimeConfig(max_rows=500, target_groups=10, num_generations=2, max_staleness=0)
        assert config.max_rows == 500

    def test_max_rows_auto_when_none(self):
        config = TransferQueueRuntimeConfig(target_groups=10, num_generations=2, max_staleness=0)
        assert config.max_rows == 20

    def test_max_rows_per_context_explicit(self):
        config = TransferQueueRuntimeConfig(max_rows_per_context=100)
        assert config.max_rows_per_context == 100

    def test_max_rows_per_context_defaults_to_global(self):
        config = TransferQueueRuntimeConfig(max_rows=500)
        assert config.max_rows_per_context == 500

    def test_max_live_partitions_per_context_explicit(self):
        config = TransferQueueRuntimeConfig(max_live_partitions_per_context=5)
        assert config.max_live_partitions_per_context == 5

    def test_max_live_partitions_per_context_defaults(self):
        config = TransferQueueRuntimeConfig(max_staleness=3)
        assert config.max_live_partitions_per_context == 4

    def test_check_capacity_uses_auto_calculated_max_rows(self):
        config = TransferQueueRuntimeConfig(target_groups=1, num_generations=2, max_staleness=0)
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient(), tq_config=config)
        ctx = make_context()
        assert dp.check_capacity(ctx)

        p = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, p.partition_id, [make_sample(0), make_sample(1)], seal=True)
        assert not dp.check_capacity(ctx)

    def test_check_capacity_enforces_max_live_partitions(self):
        config = TransferQueueRuntimeConfig(max_staleness=0)
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient(), tq_config=config)
        ctx = make_context()

        p0 = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, p0.partition_id, [make_sample(0)], seal=True)
        assert not dp.check_capacity(ctx)


# ── QueueMetadata ─────────────────────────────────────────────────────────


class TestQueueMetadata:

    def test_get_metadata_with_context_returns_queue_metadata(self):
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
        ctx = make_context()
        dp.init_namespace(ctx)

        result = dp.get_metadata(ctx)
        assert isinstance(result, QueueMetadata)
        assert result.context.key == ctx.key
        assert result.live_partition_count == 0
        assert result.total_rows == 0

    def test_get_metadata_without_context_returns_list(self):
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
        ctx = make_context()
        dp.create_partition(ctx, target_groups=1)

        result = dp.get_metadata()
        assert isinstance(result, list)

    def test_queue_metadata_active_partitions(self):
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
        ctx = make_context()

        p0 = dp.create_partition(ctx, target_groups=1)
        p1 = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, p0.partition_id, [make_sample(0)], seal=True)
        dp.put_rollout_batch(ctx, p1.partition_id, [make_sample(1)], seal=True)

        qm = dp.get_metadata(ctx)
        assert qm.live_partition_count == 2
        assert qm.total_rows == 2

    def test_queue_metadata_excludes_cleared(self):
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
        ctx = make_context()

        p0 = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, p0.partition_id, [make_sample(0)], seal=True)
        dp.clear_partition(ctx, p0.partition_id)

        qm = dp.get_metadata(ctx)
        assert qm.live_partition_count == 0

    def test_queue_metadata_oldest_partition(self):
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
        ctx = make_context()

        p0 = dp.create_partition(ctx, target_groups=1)
        p1 = dp.create_partition(ctx, target_groups=1)

        qm = dp.get_metadata(ctx)
        assert qm.oldest_partition.partition_id == p0.partition_id

    def test_queue_metadata_iterable(self):
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
        ctx = make_context()

        dp.create_partition(ctx, target_groups=1)
        dp.create_partition(ctx, target_groups=1)

        qm = dp.get_metadata(ctx)
        partitions = list(qm)
        assert len(partitions) == 2

    def test_queue_metadata_len(self):
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
        ctx = make_context()

        dp.create_partition(ctx, target_groups=1)
        qm = dp.get_metadata(ctx)
        assert len(qm) == 1

    def test_queue_metadata_trainer_step_increments(self):
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
        ctx = make_context()

        p = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, p.partition_id, [make_sample(0)], seal=True)
        dp.append_rewards(ctx, p.partition_id, [1.0])
        dp.append_advantages(ctx, p.partition_id, [0.5])
        dp.mark_training(ctx, p.partition_id)
        dp.mark_trained(ctx, p.partition_id)

        qm = dp.get_metadata(ctx)
        assert qm.trainer_step == 1

    def test_queue_metadata_current_policy_version(self):
        ctx = make_context(version=5)
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())

        qm = dp.get_metadata(ctx)
        assert qm.current_policy_version == 5


# ── claim_reward_ready_groups (group-level claim) ─────────────────────────


class TestClaimRewardReadyGroups:

    def test_claim_reward_ready_groups_returns_groups(self):
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
        ctx = make_context()
        p = dp.create_partition(ctx, target_groups=2)

        samples = [
            make_sample(0, group_id='g0'),
            make_sample(1, group_id='g0'),
            make_sample(2, group_id='g0'),
            make_sample(3, group_id='g0'),
            make_sample(4, group_id='g1'),
            make_sample(5, group_id='g1'),
            make_sample(6, group_id='g1'),
            make_sample(7, group_id='g1'),
        ]
        dp.put_rollout_batch(ctx, p.partition_id, samples, ready_groups=2, seal=True)

        meta, groups = dp.claim_reward_ready_groups(ctx, num_generations=4, max_groups=2)
        assert meta.partition_id == p.partition_id
        assert len(groups) == 2
        assert len(groups[0]) == 4
        assert len(groups[1]) == 4

    def test_claim_reward_ready_groups_respects_max_groups(self):
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
        ctx = make_context()
        p = dp.create_partition(ctx, target_groups=2)

        samples = [make_sample(i, group_id=f'g{i // 4}') for i in range(8)]
        dp.put_rollout_batch(ctx, p.partition_id, samples, ready_groups=2, seal=True)

        meta, groups = dp.claim_reward_ready_groups(ctx, num_generations=4, max_groups=1)
        assert len(groups) == 1

    def test_claim_reward_ready_groups_skips_incomplete_groups(self):
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
        ctx = make_context()
        p = dp.create_partition(ctx, target_groups=1)

        samples = [
            make_sample(0, group_id='g0'),
            make_sample(1, group_id='g0'),
            make_sample(2, group_id='g1'),
        ]
        dp.put_rollout_batch(ctx, p.partition_id, samples, seal=True)

        meta, groups = dp.claim_reward_ready_groups(ctx, num_generations=4, max_groups=10)
        assert len(groups) == 0

    def test_claim_reward_ready_groups_raises_when_no_ready(self):
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
        ctx = make_context()

        with pytest.raises(LookupError, match='no reward-ready partition'):
            dp.claim_reward_ready_groups(ctx, num_generations=4, max_groups=1)

    def test_claim_reward_ready_groups_with_lease(self):
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
        ctx = make_context()
        p = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, p.partition_id, [make_sample(i, group_id='g0') for i in range(4)], seal=True)

        meta, groups = dp.claim_reward_ready_groups(ctx, num_generations=4, max_groups=1, worker_id='w1')
        assert len(groups) == 1

        with pytest.raises(RuntimeError, match='leased by w1'):
            dp.claim_reward_ready_groups(ctx, num_generations=4, max_groups=1, worker_id='w2')


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

    def test_build_streaming_dataloader_validates_required_fields(self):
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
        ctx = make_context()
        p = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, p.partition_id, [make_sample(0)], seal=True)
        dp.append_rewards(ctx, p.partition_id, [1.0])
        dp.append_advantages(ctx, p.partition_id, [0.5])

        samples = dp.build_streaming_dataloader(ctx, p.partition_id, required_fields=REWARD_FIELDS)
        assert len(samples) == 1
        assert 'rewards' in samples[0]

    def test_build_streaming_dataloader_raises_on_missing_fields(self):
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
        ctx = make_context()
        p = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, p.partition_id, [make_sample(0)], seal=True)

        with pytest.raises(ValueError, match='missing required fields'):
            dp.build_streaming_dataloader(ctx, p.partition_id, required_fields=TRAIN_REQUIRED_FIELDS)

    def test_build_streaming_dataloader_train_required_fields_after_full_pipeline(self):
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
        ctx = make_context()
        p = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, p.partition_id, [make_sample(0)], seal=True)
        dp.append_rewards(ctx, p.partition_id, [1.0])
        dp.append_advantages(ctx, p.partition_id, [0.5])

        samples = dp.build_streaming_dataloader(ctx, p.partition_id, required_fields=TRAIN_REQUIRED_FIELDS)
        assert len(samples) == 1
        assert 'rewards' in samples[0]
        assert 'advantages' in samples[0]
        assert 'returns' in samples[0]


# ── Multi-worker claim exclusion (验收标准) ───────────────────────────────


class TestMultiWorkerClaimExclusion:
    """Acceptance criteria: 同一个 sample 不会被多个 reward / advantage worker 重复 claim。"""

    def test_two_reward_workers_cannot_claim_same_partition(self):
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
        ctx = make_context()
        p = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, p.partition_id, [make_sample(0)], seal=True)

        dp.claim_reward_batch(ctx, batch_size=10, worker_id='reward_worker_1')

        with pytest.raises(RuntimeError, match='leased by reward_worker_1'):
            dp.claim_reward_batch(ctx, batch_size=10, worker_id='reward_worker_2')

    def test_two_advantage_workers_cannot_claim_same_partition(self):
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
        ctx = make_context()
        p = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, p.partition_id, [make_sample(0)], seal=True)
        dp.append_rewards(ctx, p.partition_id, [1.0])

        dp.claim_advantage_batch(ctx, batch_size=10, worker_id='adv_worker_1')

        with pytest.raises(RuntimeError, match='leased by adv_worker_1'):
            dp.claim_advantage_batch(ctx, batch_size=10, worker_id='adv_worker_2')

    def test_worker_can_claim_after_lease_released(self):
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
        ctx = make_context()
        p = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, p.partition_id, [make_sample(0)], seal=True)

        dp.claim_reward_batch(ctx, batch_size=10, worker_id='w1')
        dp.release_lease(p.partition_id, worker_id='w1')

        meta, samples = dp.claim_reward_batch(ctx, batch_size=10, worker_id='w2')
        assert len(samples) == 1

    def test_worker_can_claim_after_lease_expires(self):
        dp = TransferQueueDataPlane(
            tq_client=FakeTransferQueueClient(),
            tq_config=TransferQueueRuntimeConfig(lease_timeout=0.1),
        )
        ctx = make_context()
        p = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, p.partition_id, [make_sample(0)], seal=True)

        dp.claim_reward_batch(ctx, batch_size=10, worker_id='w1')

        import time
        time.sleep(0.2)

        meta, samples = dp.claim_reward_batch(ctx, batch_size=10, worker_id='w2')
        assert len(samples) == 1

    def test_different_partitions_can_be_claimed_by_different_workers(self):
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
        ctx = make_context()

        p0 = dp.create_partition(ctx, target_groups=1)
        p1 = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, p0.partition_id, [make_sample(0)], seal=True)
        dp.put_rollout_batch(ctx, p1.partition_id, [make_sample(1)], seal=True)

        dp.claim_reward_batch(ctx, batch_size=10, worker_id='w1')
        dp.append_rewards(ctx, p0.partition_id, [1.0])
        dp.release_lease(p0.partition_id, worker_id='w1')

        meta, samples = dp.claim_reward_batch(ctx, batch_size=10, worker_id='w2')
        assert meta.partition_id == p1.partition_id

    def test_full_lifecycle_no_duplicate_claim(self):
        """End-to-end: rollout -> reward -> advantage -> train -> clear with worker exclusion."""
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
        ctx = make_context()
        p = dp.create_partition(ctx, target_groups=1)

        dp.put_rollout_batch(ctx, p.partition_id, [make_sample(0)], seal=True)

        meta, samples = dp.claim_reward_batch(ctx, batch_size=10, worker_id='rw1')
        with pytest.raises(RuntimeError):
            dp.claim_reward_batch(ctx, batch_size=10, worker_id='rw2')
        dp.append_rewards(ctx, p.partition_id, [1.0])
        dp.release_lease(p.partition_id, worker_id='rw1')

        meta, samples = dp.claim_advantage_batch(ctx, batch_size=10, worker_id='aw1')
        with pytest.raises(RuntimeError):
            dp.claim_advantage_batch(ctx, batch_size=10, worker_id='aw2')
        dp.append_advantages(ctx, p.partition_id, [0.5])
        dp.release_lease(p.partition_id, worker_id='aw1')

        dp.mark_training(ctx, p.partition_id)
        dataloader = dp.build_streaming_dataloader(ctx, p.partition_id, task_name=TaskName.TRAIN)
        assert len(dataloader) == 1
        dp.ack_rows(ctx, p.partition_id, [dataloader[0]['sample_id']], task_name=TaskName.TRAIN)
        dp.mark_trained(ctx, p.partition_id)

        dp.clear_partition(ctx, p.partition_id)
        assert dp.list_partitions(ctx)[0].status == PartitionStatus.CLEARED
