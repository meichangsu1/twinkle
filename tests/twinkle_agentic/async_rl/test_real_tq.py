# Copyright (c) ModelScope Contributors. All rights reserved.
"""Integration tests using real TransferQueue on NPU (Ascend 910B3).

Requires:
  pip install TransferQueue
  NPU environment with CANN 9.0+

Ray is initialized once per session via conftest.py (num_gpus=0 for NPU).
"""
from __future__ import annotations

import pytest

from twinkle_agentic.async_rl import (
    AdvantageWorker,
    PartitionStatus,
    RewardWorker,
    StalenessManager,
    TrainingContext,
    TransferQueueDataPlane,
    TransferQueueRuntimeConfig,
)

tq = pytest.importorskip('transfer_queue', reason='TransferQueue not installed')


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


def make_sample(i=0):
    return {
        'sample_id': f'sample_{i}',
        'messages': [{'role': 'user', 'content': f'question {i}'}],
        'group_id': f'g{i}',
        'generation_idx': 0,
    }


@pytest.fixture
def real_dp():
    """Create a TransferQueueDataPlane with real TQ backend."""
    config = TransferQueueRuntimeConfig(
        num_data_storage_units=4,
        total_storage_size=100000,
        init=True,
    )
    dp = TransferQueueDataPlane(tq_config=config)
    yield dp


class TestRealTransferQueueDataPlane:

    def test_init_namespace(self, real_dp):
        ctx = make_context()
        real_dp.init_namespace(ctx)

    def test_create_partition(self, real_dp):
        ctx = make_context()
        real_dp.init_namespace(ctx)
        partition = real_dp.create_partition(ctx, target_groups=1)
        assert partition.status == PartitionStatus.OPEN
        assert partition.context.key == ctx.key

    def test_put_rollout_batch_and_read_back(self, real_dp):
        ctx = make_context()
        real_dp.init_namespace(ctx)
        partition = real_dp.create_partition(ctx, target_groups=1)

        meta = real_dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)
        assert meta.status == PartitionStatus.ROLLOUT_DONE
        assert meta.num_rows == 1

        samples = real_dp.build_streaming_dataloader(ctx, partition.partition_id)
        assert len(samples) == 1
        assert samples[0]['sample_id'] == 'sample_0'
        assert samples[0]['metadata']['adapter_name'] == 'lora'
        assert samples[0]['metadata']['tenant_id'] == 'tenant_a'

    def test_full_lifecycle_with_real_tq(self, real_dp):
        """OPEN -> ROLLOUT_DONE -> REWARD_DONE -> TRAIN_READY -> TRAINING -> TRAIN_DONE -> CLEARED."""
        ctx = make_context()
        real_dp.init_namespace(ctx)
        partition = real_dp.create_partition(ctx, target_groups=1)

        meta = real_dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)
        assert meta.status == PartitionStatus.ROLLOUT_DONE

        meta = real_dp.append_rewards(ctx, partition.partition_id, [1.0])
        assert meta.status == PartitionStatus.REWARD_DONE

        meta = real_dp.append_advantages(ctx, partition.partition_id, [0.5])
        assert meta.status == PartitionStatus.TRAIN_READY

        meta = real_dp.mark_training(ctx, partition.partition_id)
        assert meta.status == PartitionStatus.TRAINING

        meta = real_dp.mark_trained(ctx, partition.partition_id)
        assert meta.status == PartitionStatus.TRAIN_DONE

        real_dp.clear_partition(ctx, partition.partition_id)
        partitions = real_dp.list_partitions(ctx)
        assert partitions[0].status == PartitionStatus.CLEARED

    def test_reward_and_advantage_workers_with_real_tq(self, real_dp):
        ctx = make_context()
        real_dp.init_namespace(ctx)
        partition = real_dp.create_partition(ctx, target_groups=1)
        real_dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)

        reward_worker = RewardWorker(
            data_plane=real_dp,
            reward_registry={'constant': lambda trajectories, **_: [1.0]},
        )
        meta = reward_worker.run_once(ctx)
        assert meta.status == PartitionStatus.REWARD_DONE

        adv_worker = AdvantageWorker(data_plane=real_dp)
        meta = adv_worker.run_once(ctx)
        assert meta.status == PartitionStatus.TRAIN_READY

    def test_multi_partition_with_real_tq(self, real_dp):
        ctx = make_context('multi_part_lora', tenant='multi_part_tenant', run='multi_part_run')
        real_dp.init_namespace(ctx)

        p0 = real_dp.create_partition(ctx, target_groups=1)
        p1 = real_dp.create_partition(ctx, target_groups=1)

        real_dp.put_rollout_batch(ctx, p0.partition_id, [make_sample(0)], seal=True)
        real_dp.put_rollout_batch(ctx, p1.partition_id, [make_sample(1)], seal=True)

        partitions = real_dp.list_partitions(ctx, statuses=[PartitionStatus.ROLLOUT_DONE])
        assert len(partitions) == 2

    def test_multi_context_isolation_with_real_tq(self, real_dp):
        ctx_a = make_context('lora_a', tenant='tenant_a', run='run_1')
        ctx_b = make_context('lora_b', tenant='tenant_b', run='run_2')
        real_dp.init_namespace(ctx_a)
        real_dp.init_namespace(ctx_b)

        pa = real_dp.create_partition(ctx_a, target_groups=1)
        pb = real_dp.create_partition(ctx_b, target_groups=1)

        real_dp.put_rollout_batch(ctx_a, pa.partition_id, [make_sample(0)], seal=True)
        real_dp.put_rollout_batch(ctx_b, pb.partition_id, [make_sample(1)], seal=True)

        parts_a = real_dp.list_partitions(ctx_a)
        parts_b = real_dp.list_partitions(ctx_b)
        assert len(parts_a) == 1
        assert len(parts_b) == 1
        assert parts_a[0].context.adapter_name == 'lora_a'
        assert parts_b[0].context.adapter_name == 'lora_b'

    def test_check_capacity_with_real_tq(self, real_dp):
        ctx = make_context('cap_lora', tenant='cap_tenant', run='cap_run')
        real_dp.init_namespace(ctx)
        real_dp.tq_config = TransferQueueRuntimeConfig(max_rows_per_context=2)

        assert real_dp.check_capacity(ctx)

        p = real_dp.create_partition(ctx, target_groups=1)
        real_dp.put_rollout_batch(ctx, p.partition_id, [make_sample(0), make_sample(1), make_sample(2)], seal=True)
        assert not real_dp.check_capacity(ctx)

    def test_staleness_manager_with_real_tq(self, real_dp):
        ctx = make_context('stale_lora', tenant='stale_tenant', run='stale_run')
        real_dp.init_namespace(ctx)
        manager = StalenessManager(max_staleness=1, target_groups_per_partition=1)

        capacity = manager.get_rollout_capacity(ctx, real_dp.get_metadata(ctx))
        assert capacity.available_groups == 2

        p0 = real_dp.create_partition(ctx, target_groups=1)
        real_dp.put_rollout_batch(ctx, p0.partition_id, [make_sample(0)], seal=True)
        capacity = manager.get_rollout_capacity(ctx, real_dp.get_metadata(ctx))
        assert capacity.available_groups == 1

    def test_clear_partition_removes_tq_data(self, real_dp):
        ctx = make_context()
        real_dp.init_namespace(ctx)
        partition = real_dp.create_partition(ctx, target_groups=1)
        real_dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)

        tq_data = real_dp.tq.kv_list(partition_id=partition.partition_id)
        assert len(tq_data.get(partition.partition_id, {})) > 0

        real_dp.clear_partition(ctx, partition.partition_id)

        tq_data = real_dp.tq.kv_list(partition_id=partition.partition_id)
        assert len(tq_data.get(partition.partition_id, {})) == 0

    def test_cross_context_rejected_with_real_tq(self, real_dp):
        ctx = make_context()
        other = make_context('other')
        real_dp.init_namespace(ctx)
        partition = real_dp.create_partition(ctx, target_groups=1)

        with pytest.raises(ValueError, match='belongs to'):
            real_dp.put_rollout_batch(other, partition.partition_id, [make_sample(0)])

    def test_mixed_policy_version_with_real_tq(self, real_dp):
        """Test mixed policy_version in same partition with real TQ backend."""
        ctx_v0 = make_context(version=0)
        ctx_v1 = make_context(version=1)
        
        real_dp.init_namespace(ctx_v0)
        partition = real_dp.create_partition(ctx_v0, target_groups=2)
        
        # Write samples with policy_version=0
        samples_v0 = [
            {
                'sample_id': 'v0_sample_0',
                'messages': [{'role': 'user', 'content': 'q0'}],
                'group_id': 'g0',
                'generation_idx': 0,
                'old_logps': [0.1, 0.2],
            }
        ]
        real_dp.put_rollout_batch(ctx_v0, partition.partition_id, samples_v0, ready_groups=1, seal=False)
        
        # Write samples with policy_version=1 to same partition
        samples_v1 = [
            {
                'sample_id': 'v1_sample_0',
                'messages': [{'role': 'user', 'content': 'q1'}],
                'group_id': 'g1',
                'generation_idx': 0,
                'old_logps': [0.3, 0.4],
            }
        ]
        real_dp.put_rollout_batch(ctx_v1, partition.partition_id, samples_v1, ready_groups=1, seal=True)
        
        # Read back and verify both versions present
        dataloader = real_dp.build_streaming_dataloader(ctx_v1, partition.partition_id)
        assert len(dataloader) == 2
        
        sample_ids = {s['sample_id'] for s in dataloader}
        assert sample_ids == {'v0_sample_0', 'v1_sample_0'}
        
        # Verify each sample retains its own policy_version
        for sample in dataloader:
            if sample['sample_id'] == 'v0_sample_0':
                assert sample['metadata']['policy_version'] == 0
                assert sample['old_logps'] == [0.1, 0.2]
            else:
                assert sample['metadata']['policy_version'] == 1
                assert sample['old_logps'] == [0.3, 0.4]
