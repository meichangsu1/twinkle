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
    TaskName,
    TrainingContext,
    TransferQueueDataPlane,
    TransferQueueRuntimeConfig,
)

tq = pytest.importorskip('transfer_queue', reason='TransferQueue not installed')

# ── Row field schema (defined locally for testing, documented in API doc) ──

TRAIN_REQUIRED_FIELDS = frozenset({
    'messages',
    'group_id',
    'generation_idx',
    'old_logps',
    'rewards',
    'advantages',
    'returns',
})



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

    def test_field_state_flow_with_real_tq(self, real_dp):
        """Verify field state at each stage: rollout -> reward -> advantage -> train -> clear."""
        ctx = make_context()
        real_dp.init_namespace(ctx)
        partition = real_dp.create_partition(ctx, target_groups=1)

        # Stage 1: Rollout - write sample with messages, group_id, generation_idx, old_logps
        sample = {
            'sample_id': 'sample_0',
            'messages': [{'role': 'user', 'content': 'What is 2+2?'}],
            'group_id': 'group_0',
            'generation_idx': 0,
            'old_logps': [0.1, 0.2, 0.3],
        }
        meta = real_dp.put_rollout_batch(ctx, partition.partition_id, [sample], seal=True)
        assert meta.status == PartitionStatus.ROLLOUT_DONE

        # Verify fields after rollout
        samples = real_dp.build_streaming_dataloader(ctx, partition.partition_id)
        assert len(samples) == 1
        s = samples[0]
        assert 'messages' in s
        assert 'group_id' in s
        assert 'generation_idx' in s
        assert 'old_logps' in s
        assert s['group_id'] == 'group_0'
        assert s['generation_idx'] == 0
        assert 'rewards' not in s
        assert 'advantages' not in s
        assert 'returns' not in s

        # Stage 2: Reward - append rewards field
        meta = real_dp.append_rewards(ctx, partition.partition_id, [1.5])
        assert meta.status == PartitionStatus.REWARD_DONE

        # Verify fields after reward
        samples = real_dp.build_streaming_dataloader(ctx, partition.partition_id)
        s = samples[0]
        assert 'messages' in s
        assert 'group_id' in s
        assert 'old_logps' in s
        assert 'rewards' in s
        assert s['rewards'] == 1.5
        assert 'advantages' not in s
        assert 'returns' not in s

        # Stage 3: Advantage - append advantages and returns fields
        meta = real_dp.append_advantages(ctx, partition.partition_id, [0.8], returns=[1.2])
        assert meta.status == PartitionStatus.TRAIN_READY

        # Verify fields after advantage
        samples = real_dp.build_streaming_dataloader(ctx, partition.partition_id)
        s = samples[0]
        assert 'messages' in s
        assert 'group_id' in s
        assert 'old_logps' in s
        assert 'rewards' in s
        assert s['rewards'] == 1.5
        assert 'advantages' in s
        assert s['advantages'] == 0.8
        assert 'returns' in s
        assert s['returns'] == 1.2

        # Stage 4: Train - mark training, read data, ack, mark trained
        meta = real_dp.mark_training(ctx, partition.partition_id)
        assert meta.status == PartitionStatus.TRAINING

        samples = real_dp.build_streaming_dataloader(ctx, partition.partition_id)
        s = samples[0]
        assert 'messages' in s
        assert 'group_id' in s
        assert 'generation_idx' in s
        assert 'old_logps' in s
        assert 'rewards' in s
        assert 'advantages' in s
        assert 'returns' in s

        acked = real_dp.ack_rows(ctx, partition.partition_id, ['sample_0'], task_name=TaskName.TRAIN)
        assert acked == 1
        assert real_dp.get_consumed_count(partition.partition_id, task_name=TaskName.TRAIN) == 1

        meta = real_dp.mark_trained(ctx, partition.partition_id)
        assert meta.status == PartitionStatus.TRAIN_DONE

        # Stage 5: Clear - verify data is removed from TQ
        real_dp.clear_partition(ctx, partition.partition_id)
        partitions = real_dp.list_partitions(ctx)
        assert partitions[0].status == PartitionStatus.CLEARED

        tq_data = real_dp.tq.kv_list(partition_id=partition.partition_id)
        assert len(tq_data.get(partition.partition_id, {})) == 0

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

    def test_lease_claim_mutual_exclusion_with_real_tq(self, real_dp):
        """Test lease/claim mutual exclusion with real TQ backend."""
        ctx = make_context('lease_lora', tenant='lease_tenant', run='lease_run')
        real_dp.init_namespace(ctx)
        partition = real_dp.create_partition(ctx, target_groups=1)
        real_dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)
        
        # Worker 1 claims with lease
        real_dp.claim_partition_with_lease(ctx, partition.partition_id, worker_id='worker_1', timeout=60)
        
        # Worker 2 should be rejected
        with pytest.raises(RuntimeError, match='is leased by'):
            real_dp.claim_partition_with_lease(ctx, partition.partition_id, worker_id='worker_2', timeout=60)
        
        # Worker 1 releases
        real_dp.release_lease(partition.partition_id, worker_id='worker_1')
        
        # Worker 2 can now claim
        real_dp.claim_partition_with_lease(ctx, partition.partition_id, worker_id='worker_2', timeout=60)
        real_dp.release_lease(partition.partition_id, worker_id='worker_2')

    def test_ack_rows_with_real_tq(self, real_dp):
        """Test ack_rows consumption tracking with real TQ backend."""
        ctx = make_context()
        real_dp.init_namespace(ctx)
        partition = real_dp.create_partition(ctx, target_groups=1)
        
        samples = [make_sample(i) for i in range(5)]
        real_dp.put_rollout_batch(ctx, partition.partition_id, samples, seal=True)
        real_dp.append_rewards(ctx, partition.partition_id, [1.0] * 5)
        real_dp.append_advantages(ctx, partition.partition_id, [0.5] * 5)
        
        # Read with task_name=TaskName.TRAIN
        dataloader = real_dp.build_streaming_dataloader(ctx, partition.partition_id, task_name=TaskName.TRAIN)
        assert len(dataloader) == 5
        
        # Ack first 3 samples
        acked = real_dp.ack_rows(ctx, partition.partition_id, ['sample_0', 'sample_1', 'sample_2'], task_name=TaskName.TRAIN)
        assert acked == 3
        assert real_dp.get_consumed_count(partition.partition_id, task_name=TaskName.TRAIN) == 3
        
        # Read again, should only get 2 unacked samples
        dataloader = real_dp.build_streaming_dataloader(ctx, partition.partition_id, task_name=TaskName.TRAIN)
        assert len(dataloader) == 2
        sample_ids = {s['sample_id'] for s in dataloader}
        assert sample_ids == {'sample_3', 'sample_4'}
        
        # Ack remaining
        acked = real_dp.ack_rows(ctx, partition.partition_id, ['sample_3', 'sample_4'], task_name=TaskName.TRAIN)
        assert acked == 2
        assert real_dp.get_consumed_count(partition.partition_id, task_name=TaskName.TRAIN) == 5
        
        # Read again, should be empty
        dataloader = real_dp.build_streaming_dataloader(ctx, partition.partition_id, task_name=TaskName.TRAIN)
        assert len(dataloader) == 0

    def test_claim_reward_ready_groups_with_real_tq(self, real_dp):
        """Test group-level claim with real TQ backend."""
        ctx = make_context('group_claim_lora', tenant='group_claim_tenant', run='group_claim_run')
        real_dp.init_namespace(ctx)
        partition = real_dp.create_partition(ctx, target_groups=2)
        
        # Write 2 groups, each with 4 generations
        samples = []
        for g in range(2):
            for gen in range(4):
                samples.append({
                    'sample_id': f'g{g}_gen{gen}',
                    'messages': [{'role': 'user', 'content': f'question {g}'}],
                    'group_id': f'group_{g}',
                    'generation_idx': gen,
                })
        real_dp.put_rollout_batch(ctx, partition.partition_id, samples, ready_groups=2, seal=True)
        
        # Claim 1 group
        meta, groups = real_dp.claim_reward_ready_groups(ctx, num_generations=4, max_groups=1)
        assert len(groups) == 1
        assert len(groups[0]) == 4
        assert all(s['group_id'] == 'group_0' for s in groups[0])

    def test_clear_namespace_with_real_tq(self, real_dp):
        """Test batch clear all partitions in a namespace with real TQ backend."""
        ctx = make_context('clear_ns_lora', tenant='clear_ns_tenant', run='clear_ns_run')
        real_dp.init_namespace(ctx)
        
        # Create 3 partitions
        p1 = real_dp.create_partition(ctx, target_groups=1)
        p2 = real_dp.create_partition(ctx, target_groups=1)
        p3 = real_dp.create_partition(ctx, target_groups=1)
        
        real_dp.put_rollout_batch(ctx, p1.partition_id, [make_sample(0)], seal=True)
        real_dp.put_rollout_batch(ctx, p2.partition_id, [make_sample(1)], seal=True)
        real_dp.put_rollout_batch(ctx, p3.partition_id, [make_sample(2)], seal=True)
        
        # Clear all
        cleared = real_dp.clear_namespace(ctx)
        assert cleared == 3
        
        # Verify all cleared
        partitions = real_dp.list_partitions(ctx)
        assert all(p.status == PartitionStatus.CLEARED for p in partitions)
        
        # Verify TQ data is gone
        for p in [p1, p2, p3]:
            tq_data = real_dp.tq.kv_list(partition_id=p.partition_id)
            assert len(tq_data.get(p.partition_id, {})) == 0

    def test_close_with_real_tq(self):
        """Test close() resource cleanup with real TQ backend."""
        config = TransferQueueRuntimeConfig(
            num_data_storage_units=4,
            total_storage_size=100000,
            init=True,
        )
        dp = TransferQueueDataPlane(tq_config=config)
        
        ctx = make_context()
        dp.init_namespace(ctx)
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)
        
        # Close should not raise
        dp.close()

    def test_required_fields_validation_with_real_tq(self, real_dp):
        """Test required_fields validation with real TQ backend."""
        ctx = make_context()
        real_dp.init_namespace(ctx)
        partition = real_dp.create_partition(ctx, target_groups=1)
        
        # Write sample without old_logps
        sample = {
            'sample_id': 'sample_0',
            'messages': [{'role': 'user', 'content': 'question'}],
            'group_id': 'g0',
            'generation_idx': 0,
        }
        real_dp.put_rollout_batch(ctx, partition.partition_id, [sample], seal=True)
        real_dp.append_rewards(ctx, partition.partition_id, [1.0])
        real_dp.append_advantages(ctx, partition.partition_id, [0.5])
        
        # Should raise when requiring old_logps
        with pytest.raises(ValueError, match='missing required fields'):
            real_dp.build_streaming_dataloader(
                ctx, 
                partition.partition_id, 
                required_fields=TRAIN_REQUIRED_FIELDS
            )

    def test_queue_metadata_with_real_tq(self, real_dp):
        """Test QueueMetadata aggregation with real TQ backend."""
        ctx = make_context('qmeta_lora', tenant='qmeta_tenant', run='qmeta_run')
        real_dp.init_namespace(ctx)
        
        # Create 2 active partitions
        p1 = real_dp.create_partition(ctx, target_groups=1)
        p2 = real_dp.create_partition(ctx, target_groups=1)
        real_dp.put_rollout_batch(ctx, p1.partition_id, [make_sample(0), make_sample(1)], seal=True)
        real_dp.put_rollout_batch(ctx, p2.partition_id, [make_sample(2)], seal=True)
        
        # Create 1 cleared partition
        p3 = real_dp.create_partition(ctx, target_groups=1)
        real_dp.put_rollout_batch(ctx, p3.partition_id, [make_sample(3)], seal=True)
        real_dp.clear_partition(ctx, p3.partition_id)
        
        # Get metadata
        qm = real_dp.get_metadata(ctx)
        assert qm.live_partition_count == 2
        assert qm.total_rows == 3
        assert qm.oldest_partition.partition_id == p1.partition_id
        assert qm.current_policy_version == 0
        
        # Should be iterable
        partition_ids = [p.partition_id for p in qm]
        assert len(partition_ids) == 2

    def test_metadata_stage_progress_with_real_tq(self, real_dp):
        """Verify metadata reflects completion progress at each stage."""
        ctx = make_context('progress_lora', tenant='progress_tenant', run='progress_run')
        real_dp.init_namespace(ctx)
        
        # Create 4 partitions at different stages
        p1 = real_dp.create_partition(ctx, target_groups=1)
        p2 = real_dp.create_partition(ctx, target_groups=1)
        p3 = real_dp.create_partition(ctx, target_groups=1)
        p4 = real_dp.create_partition(ctx, target_groups=1)
        
        # p1: ROLLOUT_DONE
        real_dp.put_rollout_batch(ctx, p1.partition_id, [make_sample(0)], seal=True)
        
        # p2: REWARD_DONE
        real_dp.put_rollout_batch(ctx, p2.partition_id, [make_sample(1)], seal=True)
        real_dp.append_rewards(ctx, p2.partition_id, [1.0])
        
        # p3: TRAIN_READY
        real_dp.put_rollout_batch(ctx, p3.partition_id, [make_sample(2)], seal=True)
        real_dp.append_rewards(ctx, p3.partition_id, [1.0])
        real_dp.append_advantages(ctx, p3.partition_id, [0.5])
        
        # p4: TRAINING
        real_dp.put_rollout_batch(ctx, p4.partition_id, [make_sample(3)], seal=True)
        real_dp.append_rewards(ctx, p4.partition_id, [1.0])
        real_dp.append_advantages(ctx, p4.partition_id, [0.5])
        real_dp.mark_training(ctx, p4.partition_id)
        
        # Verify metadata at each stage
        qm = real_dp.get_metadata(ctx)
        assert qm.live_partition_count == 4
        assert qm.total_rows == 4
        assert qm.oldest_partition.partition_id == p1.partition_id
        
        # Count partitions at each stage
        partitions = list(qm)
        status_counts = {}
        for p in partitions:
            status_counts[p.status] = status_counts.get(p.status, 0) + 1
        
        assert status_counts.get(PartitionStatus.ROLLOUT_DONE, 0) == 1
        assert status_counts.get(PartitionStatus.REWARD_DONE, 0) == 1
        assert status_counts.get(PartitionStatus.TRAIN_READY, 0) == 1
        assert status_counts.get(PartitionStatus.TRAINING, 0) == 1
        
        # Move p4 to TRAIN_DONE
        real_dp.mark_trained(ctx, p4.partition_id)
        
        # Verify progress changed
        qm = real_dp.get_metadata(ctx)
        partitions = list(qm)
        status_counts = {}
        for p in partitions:
            status_counts[p.status] = status_counts.get(p.status, 0) + 1
        
        assert status_counts.get(PartitionStatus.TRAINING, 0) == 0
        assert status_counts.get(PartitionStatus.TRAIN_DONE, 0) == 1
        
        # Clear p4 and verify live count decreases
        real_dp.clear_partition(ctx, p4.partition_id)
        qm = real_dp.get_metadata(ctx)
        assert qm.live_partition_count == 3
        assert qm.total_rows == 3  # p4 data removed
        
        # Verify oldest is still p1
        assert qm.oldest_partition.partition_id == p1.partition_id
        
        # Clear p1 and verify oldest changes to p2
        real_dp.clear_partition(ctx, p1.partition_id)
        qm = real_dp.get_metadata(ctx)
        assert qm.live_partition_count == 2
        assert qm.oldest_partition.partition_id == p2.partition_id

    def test_capacity_auto_calculation_with_real_tq(self):
        """Test capacity auto-calculation with real TQ backend."""
        config = TransferQueueRuntimeConfig(
            target_groups=2,
            num_generations=4,
            max_staleness=1,
            num_data_storage_units=4,
            total_storage_size=100000,
            init=True,
        )
        dp = TransferQueueDataPlane(tq_config=config)
        
        # Auto-calculated: max_rows = 2 * 4 * (1+1) = 16
        assert config.max_rows == 16
        assert config.max_live_partitions_per_context == 2
        
        ctx = make_context('cap_auto_lora', tenant='cap_auto_tenant', run='cap_auto_run')
        dp.init_namespace(ctx)
        
        # Should have capacity initially
        assert dp.check_capacity(ctx)
        
        # Fill to max_live_partitions (2 partitions)
        p1 = dp.create_partition(ctx, target_groups=2)
        p2 = dp.create_partition(ctx, target_groups=2)
        dp.put_rollout_batch(ctx, p1.partition_id, [make_sample(i) for i in range(8)], ready_groups=2, seal=True)
        dp.put_rollout_batch(ctx, p2.partition_id, [make_sample(i) for i in range(8, 16)], ready_groups=2, seal=True)
        
        # Should be at capacity (2 live partitions = max_live_partitions)
        assert not dp.check_capacity(ctx)

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
