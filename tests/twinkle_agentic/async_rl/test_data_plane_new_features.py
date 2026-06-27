# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for new TransferQueueDataPlane features:
  - kv_batch_put optimization
  - ack mechanism (ack_rows, consumed tracking)
  - lease/claim mutual exclusion
  - clear_namespace
  - close()
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from twinkle_agentic.async_rl import (
    PartitionStatus,
    TaskName,
    TrainingContext,
    TransferQueueDataPlane,
    TransferQueueRuntimeConfig,
)

from .fakes import FakeTransferQueueClient


def make_context(name='lora', *, tenant='tenant_a', run='run_001'):
    return TrainingContext(
        tenant_id=tenant,
        training_run_id=run,
        base_model_id='Qwen/Qwen3.5-0.8B',
        adapter_name=name,
        reward_type='constant',
        loss_type='grpo',
        algorithm='grpo',
    )


def make_sample(i=0):
    return {
        'sample_id': f'sample_{i}',
        'messages': [{'role': 'user', 'content': f'question {i}'}],
        'group_id': f'g{i}',
        'generation_idx': 0,
    }


class TestKVBatchPutOptimization:

    def test_put_rollout_batch_uses_kv_batch_put(self):
        fake = FakeTransferQueueClient()
        dp = TransferQueueDataPlane(tq_client=fake)
        ctx = make_context()
        partition = dp.create_partition(ctx, target_groups=1)

        with patch.object(fake, 'kv_batch_put', wraps=fake.kv_batch_put) as mock_batch:
            dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0), make_sample(1)], seal=True)
            assert mock_batch.called, 'put_rollout_batch should use kv_batch_put'

    def test_sync_partition_status_uses_kv_batch_put(self):
        fake = FakeTransferQueueClient()
        dp = TransferQueueDataPlane(tq_client=fake)
        ctx = make_context()
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)

        with patch.object(fake, 'kv_batch_put', wraps=fake.kv_batch_put) as mock_batch:
            dp.mark_training(ctx, partition.partition_id)
            assert mock_batch.called, '_sync_partition_status should use kv_batch_put'

    def test_append_rewards_uses_batch_update(self):
        fake = FakeTransferQueueClient()
        dp = TransferQueueDataPlane(tq_client=fake)
        ctx = make_context()
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0), make_sample(1)], seal=True)

        with patch.object(fake, 'kv_batch_put', wraps=fake.kv_batch_put) as mock_batch:
            dp.append_rewards(ctx, partition.partition_id, [1.0, 0.5])
            assert mock_batch.called, 'append_rewards should use batch update'


class TestAckMechanism:

    def test_ack_rows_tracks_consumed_samples(self):
        fake = FakeTransferQueueClient()
        dp = TransferQueueDataPlane(tq_client=fake)
        ctx = make_context()
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0), make_sample(1)], seal=True)

        new_acked = dp.ack_rows(ctx, partition.partition_id, ['sample_0'], task_name=TaskName.TRAIN)
        assert new_acked == 1
        assert dp.get_consumed_count(partition.partition_id, task_name=TaskName.TRAIN) == 1

    def test_ack_rows_idempotent(self):
        fake = FakeTransferQueueClient()
        dp = TransferQueueDataPlane(tq_client=fake)
        ctx = make_context()
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)

        dp.ack_rows(ctx, partition.partition_id, ['sample_0'], task_name=TaskName.TRAIN)
        new_acked = dp.ack_rows(ctx, partition.partition_id, ['sample_0'], task_name=TaskName.TRAIN)
        assert new_acked == 0
        assert dp.get_consumed_count(partition.partition_id, task_name=TaskName.TRAIN) == 1

    def test_build_streaming_dataloader_filters_acked_rows(self):
        fake = FakeTransferQueueClient()
        dp = TransferQueueDataPlane(tq_client=fake)
        ctx = make_context()
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0), make_sample(1), make_sample(2)], seal=True)

        dp.ack_rows(ctx, partition.partition_id, ['sample_0', 'sample_1'], task_name=TaskName.TRAIN)
        samples = dp.build_streaming_dataloader(ctx, partition.partition_id, task_name=TaskName.TRAIN)
        assert len(samples) == 1
        assert samples[0]['sample_id'] == 'sample_2'

    def test_build_streaming_dataloader_without_task_name_returns_all(self):
        fake = FakeTransferQueueClient()
        dp = TransferQueueDataPlane(tq_client=fake)
        ctx = make_context()
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0), make_sample(1)], seal=True)

        dp.ack_rows(ctx, partition.partition_id, ['sample_0'], task_name=TaskName.TRAIN)
        samples = dp.build_streaming_dataloader(ctx, partition.partition_id)
        assert len(samples) == 2

    def test_ack_rows_per_task_isolation(self):
        fake = FakeTransferQueueClient()
        dp = TransferQueueDataPlane(tq_client=fake)
        ctx = make_context()
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)

        dp.ack_rows(ctx, partition.partition_id, ['sample_0'], task_name=TaskName.TRAIN)
        assert dp.get_consumed_count(partition.partition_id, task_name=TaskName.TRAIN) == 1
        assert dp.get_consumed_count(partition.partition_id, task_name='eval') == 0

    def test_ack_rows_rejects_cross_context(self):
        fake = FakeTransferQueueClient()
        dp = TransferQueueDataPlane(tq_client=fake)
        ctx = make_context()
        other = make_context('other')
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)

        with pytest.raises(ValueError, match='belongs to'):
            dp.ack_rows(other, partition.partition_id, ['sample_0'])

    def test_clear_partition_resets_consumed(self):
        fake = FakeTransferQueueClient()
        dp = TransferQueueDataPlane(tq_client=fake)
        ctx = make_context()
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)

        dp.ack_rows(ctx, partition.partition_id, ['sample_0'], task_name=TaskName.TRAIN)
        assert dp.get_consumed_count(partition.partition_id, task_name=TaskName.TRAIN) == 1

        dp.clear_partition(ctx, partition.partition_id)
        assert dp.get_consumed_count(partition.partition_id, task_name=TaskName.TRAIN) == 0


class TestLeaseClaimMechanism:

    def test_claim_partition_with_lease(self):
        fake = FakeTransferQueueClient()
        dp = TransferQueueDataPlane(tq_client=fake)
        ctx = make_context()
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)

        meta = dp.claim_partition_with_lease(ctx, partition.partition_id, worker_id='worker_1')
        assert meta.owner_worker_id == 'worker_1'
        assert meta.lease_deadline is not None
        assert meta.lease_deadline > time.time()

    def test_lease_blocks_other_workers(self):
        fake = FakeTransferQueueClient()
        dp = TransferQueueDataPlane(tq_client=fake)
        ctx = make_context()
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)

        dp.claim_partition_with_lease(ctx, partition.partition_id, worker_id='worker_1', timeout=60)

        with pytest.raises(RuntimeError, match='leased by worker_1'):
            dp.claim_partition_with_lease(ctx, partition.partition_id, worker_id='worker_2')

    def test_same_worker_can_reclaim(self):
        fake = FakeTransferQueueClient()
        dp = TransferQueueDataPlane(tq_client=fake)
        ctx = make_context()
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)

        dp.claim_partition_with_lease(ctx, partition.partition_id, worker_id='worker_1', timeout=60)
        meta = dp.claim_partition_with_lease(ctx, partition.partition_id, worker_id='worker_1', timeout=60)
        assert meta.owner_worker_id == 'worker_1'

    def test_release_lease(self):
        fake = FakeTransferQueueClient()
        dp = TransferQueueDataPlane(tq_client=fake)
        ctx = make_context()
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)

        dp.claim_partition_with_lease(ctx, partition.partition_id, worker_id='worker_1')
        dp.release_lease(partition.partition_id, worker_id='worker_1')

        meta = dp.claim_partition_with_lease(ctx, partition.partition_id, worker_id='worker_2')
        assert meta.owner_worker_id == 'worker_2'

    def test_release_lease_rejects_wrong_worker(self):
        fake = FakeTransferQueueClient()
        dp = TransferQueueDataPlane(tq_client=fake)
        ctx = make_context()
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)

        dp.claim_partition_with_lease(ctx, partition.partition_id, worker_id='worker_1')

        with pytest.raises(RuntimeError, match='leased by worker_1'):
            dp.release_lease(partition.partition_id, worker_id='worker_2')

    def test_renew_lease(self):
        fake = FakeTransferQueueClient()
        dp = TransferQueueDataPlane(tq_client=fake)
        ctx = make_context()
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)

        dp.claim_partition_with_lease(ctx, partition.partition_id, worker_id='worker_1', timeout=10)
        meta_before = dp._meta[partition.partition_id]
        deadline_before = meta_before.lease_deadline

        time.sleep(0.1)
        dp.renew_lease(partition.partition_id, worker_id='worker_1', timeout=60)
        meta_after = dp._meta[partition.partition_id]
        assert meta_after.lease_deadline > deadline_before

    def test_expired_lease_auto_recovered(self):
        fake = FakeTransferQueueClient()
        dp = TransferQueueDataPlane(tq_client=fake)
        ctx = make_context()
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)

        dp.claim_partition_with_lease(ctx, partition.partition_id, worker_id='worker_1', timeout=0.1)
        time.sleep(0.2)

        meta = dp.claim_partition_with_lease(ctx, partition.partition_id, worker_id='worker_2')
        assert meta.owner_worker_id == 'worker_2'

    def test_claim_with_lease_rejects_cross_context(self):
        fake = FakeTransferQueueClient()
        dp = TransferQueueDataPlane(tq_client=fake)
        ctx = make_context()
        other = make_context('other')
        partition = dp.create_partition(ctx, target_groups=1)

        with pytest.raises(ValueError, match='belongs to'):
            dp.claim_partition_with_lease(other, partition.partition_id, worker_id='worker_1')

    def test_clear_partition_releases_lease(self):
        fake = FakeTransferQueueClient()
        dp = TransferQueueDataPlane(tq_client=fake)
        ctx = make_context()
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)

        dp.claim_partition_with_lease(ctx, partition.partition_id, worker_id='worker_1')
        dp.clear_partition(ctx, partition.partition_id)

        assert partition.partition_id not in dp._leases


class TestClearNamespace:

    def test_clear_namespace_clears_all_partitions_for_context(self):
        fake = FakeTransferQueueClient()
        dp = TransferQueueDataPlane(tq_client=fake)
        ctx = make_context()

        p0 = dp.create_partition(ctx, target_groups=1)
        p1 = dp.create_partition(ctx, target_groups=1)
        p2 = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, p0.partition_id, [make_sample(0)], seal=True)
        dp.put_rollout_batch(ctx, p1.partition_id, [make_sample(1)], seal=True)
        dp.put_rollout_batch(ctx, p2.partition_id, [make_sample(2)], seal=True)

        cleared = dp.clear_namespace(ctx)
        assert cleared == 3

        partitions = dp.list_partitions(ctx)
        assert all(p.status == PartitionStatus.CLEARED for p in partitions)

    def test_clear_namespace_does_not_affect_other_contexts(self):
        fake = FakeTransferQueueClient()
        dp = TransferQueueDataPlane(tq_client=fake)
        ctx_a = make_context('lora_a', tenant='tenant_a')
        ctx_b = make_context('lora_b', tenant='tenant_b')

        pa = dp.create_partition(ctx_a, target_groups=1)
        pb = dp.create_partition(ctx_b, target_groups=1)
        dp.put_rollout_batch(ctx_a, pa.partition_id, [make_sample(0)], seal=True)
        dp.put_rollout_batch(ctx_b, pb.partition_id, [make_sample(1)], seal=True)

        dp.clear_namespace(ctx_a)

        parts_a = dp.list_partitions(ctx_a)
        parts_b = dp.list_partitions(ctx_b)
        assert all(p.status == PartitionStatus.CLEARED for p in parts_a)
        assert all(p.status != PartitionStatus.CLEARED for p in parts_b)

    def test_clear_namespace_returns_zero_when_already_cleared(self):
        fake = FakeTransferQueueClient()
        dp = TransferQueueDataPlane(tq_client=fake)
        ctx = make_context()

        p = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, p.partition_id, [make_sample(0)], seal=True)
        dp.clear_partition(ctx, p.partition_id)

        cleared = dp.clear_namespace(ctx)
        assert cleared == 0

    def test_clear_namespace_empty_context(self):
        fake = FakeTransferQueueClient()
        dp = TransferQueueDataPlane(tq_client=fake)
        ctx = make_context()

        cleared = dp.clear_namespace(ctx)
        assert cleared == 0


class TestStreamingDataset:
    """Tests for build_streaming_dataset() method."""

    def test_streaming_dataset_yields_batches(self):
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
        ctx = make_context()
        p = dp.create_partition(ctx, target_groups=1)
        samples = [make_sample(i) for i in range(10)]
        dp.put_rollout_batch(ctx, p.partition_id, samples, seal=True)
        dp.append_rewards(ctx, p.partition_id, [1.0] * 10)
        dp.append_advantages(ctx, p.partition_id, [0.5] * 10)

        dataset = dp.build_streaming_dataset(ctx, p.partition_id, batch_size=3, task_name=TaskName.TRAIN)
        batches = list(dataset)
        assert len(batches) == 4
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert len(batches[2]) == 3
        assert len(batches[3]) == 1
        assert dataset.total_acked == 10

    def test_streaming_dataset_auto_acks(self):
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
        ctx = make_context()
        p = dp.create_partition(ctx, target_groups=1)
        samples = [make_sample(i) for i in range(5)]
        dp.put_rollout_batch(ctx, p.partition_id, samples, seal=True)
        dp.append_rewards(ctx, p.partition_id, [1.0] * 5)
        dp.append_advantages(ctx, p.partition_id, [0.5] * 5)

        dataset = dp.build_streaming_dataset(ctx, p.partition_id, batch_size=2, task_name=TaskName.TRAIN)
        batch1 = next(iter(dataset))
        assert len(batch1) == 2
        assert dp.get_consumed_count(p.partition_id, task_name=TaskName.TRAIN) == 2

    def test_streaming_dataset_rejects_cross_context(self):
        dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
        ctx = make_context()
        other = make_context('other')
        p = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, p.partition_id, [make_sample(0)], seal=True)

        with pytest.raises(ValueError, match='belongs to'):
            dp.build_streaming_dataset(other, p.partition_id, batch_size=1)


class TestClose:

    def test_close_calls_tq_close(self):
        fake = FakeTransferQueueClient()
        fake.close = MagicMock()
        dp = TransferQueueDataPlane(tq_client=fake)

        dp.close()
        fake.close.assert_called_once()

    def test_close_without_close_method(self):
        fake = FakeTransferQueueClient()
        dp = TransferQueueDataPlane(tq_client=fake)
        dp.close()
