# Copyright (c) ModelScope Contributors. All rights reserved.
"""Comprehensive verification tests for TransferQueueDataPlane.

Covers spec section 15:
  15.1 TQ interface call correctness
  15.2 Code correctness
  15.3 Design doc consistency
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

import pytest

from twinkle_agentic.async_rl import (
    AdapterRegistry,
    AdvantageWorker,
    PartitionStatus,
    RewardWorker,
    StalenessManager,
    TrainingContext,
    TransferQueueDataPlane,
    TransferQueueRuntimeConfig,
)


class RecordingFakeTQ:
    """FakeTransferQueueClient that records all calls for interface verification."""

    def __init__(self):
        self.fields: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        self.tags: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        self.calls: List[tuple] = []

    def kv_put(self, key: str, partition_id: str, fields=None, tag=None):
        self.calls.append(('kv_put', {'key': key, 'partition_id': partition_id, 'fields': fields, 'tag': tag}))
        if fields:
            current = dict(self.fields[partition_id].get(key) or {})
            current.update(dict(fields))
            self.fields[partition_id][key] = current
        elif key not in self.fields[partition_id]:
            self.fields[partition_id][key] = {}
        if tag:
            current_tag = dict(self.tags[partition_id].get(key) or {})
            current_tag.update(dict(tag))
            self.tags[partition_id][key] = current_tag

    def kv_batch_get(self, keys, partition_id: str, select_fields=None):
        self.calls.append(('kv_batch_get', {'keys': keys, 'partition_id': partition_id, 'select_fields': select_fields}))
        if isinstance(keys, str):
            keys = [keys]
        selected_fields = select_fields
        if isinstance(selected_fields, str):
            selected_fields = [selected_fields]
        rows = [dict(self.fields[partition_id].get(key) or {}) for key in keys]
        field_names = set()
        for row in rows:
            field_names.update(row)
        if selected_fields is not None:
            field_names.intersection_update(selected_fields)
        return {field_name: [row.get(field_name) for row in rows] for field_name in field_names}

    def kv_list(self, partition_id=None):
        self.calls.append(('kv_list', {'partition_id': partition_id}))
        if partition_id is not None:
            return {partition_id: dict(self.tags.get(partition_id) or {})}
        return {pid: dict(tags) for pid, tags in self.tags.items()}

    def kv_clear(self, keys, partition_id: str):
        self.calls.append(('kv_clear', {'keys': keys, 'partition_id': partition_id}))
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            self.fields.get(partition_id, {}).pop(key, None)
            self.tags.get(partition_id, {}).pop(key, None)

    def get_calls(self, method_name: str) -> list:
        return [args for name, args in self.calls if name == method_name]


def make_context(name='lora', *, tenant='tenant_a', run='run_001', version=0):
    return TrainingContext(
        tenant_id=tenant,
        training_run_id=run,
        base_model_id='Qwen/Qwen3.5-0.8B',
        adapter_name=name,
        policy_version=version,
        reward_type='f1_reward',
        loss_type='grpo',
        algorithm='grpo',
    )


def make_sample(i=0, group_id=None):
    return {
        'sample_id': f'sample_{i}',
        'messages': [{'role': 'user', 'content': f'question {i}'}],
        'group_id': group_id or f'g{i}',
        'generation_idx': 0,
    }


# =============================================================================
# 15.1 TQ Interface Call Correctness
# =============================================================================


class TestTQInterfaceCorrectness:
    """Verify data_plane calls TQ with correct parameter types and structures."""

    def test_kv_put_parameter_types_on_rollout_write(self):
        fake = RecordingFakeTQ()
        dp = TransferQueueDataPlane(tq_client=fake)
        ctx = make_context()
        partition = dp.create_partition(ctx, target_groups=1)

        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)

        put_calls = fake.get_calls('kv_put')
        assert len(put_calls) >= 1
        for call in put_calls:
            assert isinstance(call['key'], str), f'key must be str, got {type(call["key"])}'
            assert isinstance(call['partition_id'], str), f'partition_id must be str, got {type(call["partition_id"])}'
            if call['fields'] is not None:
                assert isinstance(call['fields'], dict), f'fields must be dict, got {type(call["fields"])}'
                assert 'metadata' not in call['fields'], 'fields must not contain metadata key'
                assert 'sample_id' not in call['fields'], 'fields must not contain sample_id key'
            if call['tag'] is not None:
                assert isinstance(call['tag'], dict), f'tag must be dict, got {type(call["tag"])}'

    def test_kv_put_tag_contains_context_metadata(self):
        fake = RecordingFakeTQ()
        dp = TransferQueueDataPlane(tq_client=fake)
        ctx = make_context()
        partition = dp.create_partition(ctx, target_groups=1)

        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)

        put_calls = fake.get_calls('kv_put')
        sample_put = [c for c in put_calls if c['fields'] is not None]
        assert len(sample_put) >= 1
        tag = sample_put[0]['tag']
        assert tag['tenant_id'] == 'tenant_a'
        assert tag['training_run_id'] == 'run_001'
        assert tag['adapter_name'] == 'lora'
        assert tag['base_model_id'] == 'Qwen/Qwen3.5-0.8B'
        assert 'status' in tag
        assert 'partition_id' in tag

    def test_kv_batch_get_parameter_types(self):
        fake = RecordingFakeTQ()
        dp = TransferQueueDataPlane(tq_client=fake)
        ctx = make_context()
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)

        samples = dp.build_streaming_dataloader(ctx, partition.partition_id)

        get_calls = fake.get_calls('kv_batch_get')
        assert len(get_calls) >= 1
        for call in get_calls:
            assert isinstance(call['keys'], list), f'keys must be list, got {type(call["keys"])}'
            assert all(isinstance(k, str) for k in call['keys']), 'all keys must be str'
            assert isinstance(call['partition_id'], str), f'partition_id must be str, got {type(call["partition_id"])}'

    def test_kv_list_parameter_types(self):
        fake = RecordingFakeTQ()
        dp = TransferQueueDataPlane(tq_client=fake)
        ctx = make_context()
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)

        dp.list_partitions(ctx)

        list_calls = fake.get_calls('kv_list')
        assert len(list_calls) >= 1
        for call in list_calls:
            pid = call['partition_id']
            assert pid is None or isinstance(pid, str), f'partition_id must be str or None, got {type(pid)}'

    def test_kv_clear_parameter_types(self):
        fake = RecordingFakeTQ()
        dp = TransferQueueDataPlane(tq_client=fake)
        ctx = make_context()
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)

        dp.clear_partition(ctx, partition.partition_id)

        clear_calls = fake.get_calls('kv_clear')
        assert len(clear_calls) == 1
        call = clear_calls[0]
        assert isinstance(call['keys'], list), f'keys must be list, got {type(call["keys"])}'
        assert all(isinstance(k, str) for k in call['keys']), 'all keys must be str'
        assert isinstance(call['partition_id'], str), f'partition_id must be str, got {type(call["partition_id"])}'

    def test_kv_put_tag_only_update_does_not_pass_fields(self):
        """_sync_partition_status should call kv_put with tag only (no fields)."""
        fake = RecordingFakeTQ()
        dp = TransferQueueDataPlane(tq_client=fake)
        ctx = make_context()
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)

        fake.calls.clear()
        dp.mark_training(ctx, partition.partition_id)

        put_calls = fake.get_calls('kv_put')
        for call in put_calls:
            assert call['fields'] is None, f'_sync_partition_status must not pass fields, got {call["fields"]}'
            assert call['tag'] is not None, '_sync_partition_status must pass tag'

    def test_kv_put_on_append_rewards_writes_correct_fields(self):
        fake = RecordingFakeTQ()
        dp = TransferQueueDataPlane(tq_client=fake)
        ctx = make_context()
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)

        dp.append_rewards(ctx, partition.partition_id, [0.95])

        put_calls = fake.get_calls('kv_put')
        reward_puts = [c for c in put_calls if c['fields'] is not None and 'rewards' in (c['fields'] or {})]
        assert len(reward_puts) >= 1
        assert reward_puts[0]['fields']['rewards'] == 0.95

    def test_kv_put_on_append_advantages_writes_correct_fields(self):
        fake = RecordingFakeTQ()
        dp = TransferQueueDataPlane(tq_client=fake)
        ctx = make_context()
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)
        dp.append_rewards(ctx, partition.partition_id, [0.95])

        dp.append_advantages(ctx, partition.partition_id, [0.5], [0.95])

        put_calls = fake.get_calls('kv_put')
        adv_puts = [c for c in put_calls if c['fields'] is not None and 'advantages' in (c['fields'] or {})]
        assert len(adv_puts) >= 1
        assert adv_puts[0]['fields']['advantages'] == 0.5
        assert adv_puts[0]['fields']['returns'] == 0.95


# =============================================================================
# 15.3 Design Doc Consistency
# =============================================================================


class TestDesignDocConsistency:
    """Verify code logic matches multilora-async-rl design doc requirements."""

    def test_namespace_format_matches_design_doc(self):
        """Design doc section 2.2: namespace = {tenant_id}/{training_run_id}/{adapter_name}/train_{k}"""
        ctx = make_context('code_lora', tenant='tenant_a', run='code_grpo_001')
        assert ctx.partition_id(7) == 'tenant_a/code_grpo_001/code_lora/train_7'

    def test_namespace_isolation_different_tenants(self):
        """Design doc section 2.3: different tenants must have different namespaces."""
        ctx_a = make_context('lora', tenant='tenant_a', run='run_1')
        ctx_b = make_context('lora', tenant='tenant_b', run='run_1')
        assert ctx_a.key != ctx_b.key
        assert ctx_a.partition_id(0) != ctx_b.partition_id(0)

    def test_same_train_k_belongs_to_one_context(self):
        """Design doc section 2.3: same train_k can only belong to one context."""
        ctx = make_context('lora')
        other = make_context('other_lora')
        dp = TransferQueueDataPlane(tq_client=RecordingFakeTQ())
        partition = dp.create_partition(ctx, target_groups=1)

        with pytest.raises(ValueError, match='belongs to'):
            dp.put_rollout_batch(other, partition.partition_id, [make_sample(0)])

    def test_state_machine_open_to_rollout_done_on_seal(self):
        """Design doc section 3 step 11: rollout writes seal partition to ROLLOUT_DONE."""
        ctx = make_context()
        dp = TransferQueueDataPlane(tq_client=RecordingFakeTQ())
        partition = dp.create_partition(ctx, target_groups=1)
        assert partition.status == PartitionStatus.OPEN

        meta = dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)
        assert meta.status == PartitionStatus.ROLLOUT_DONE

    def test_state_machine_rollout_done_to_reward_done(self):
        """Design doc section 3 step 13: reward worker appends rewards -> REWARD_DONE."""
        ctx = make_context()
        dp = TransferQueueDataPlane(tq_client=RecordingFakeTQ())
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)

        meta = dp.append_rewards(ctx, partition.partition_id, [1.0])
        assert meta.status == PartitionStatus.REWARD_DONE

    def test_state_machine_reward_done_to_train_ready(self):
        """Design doc section 3 step 14: advantage worker appends -> TRAIN_READY."""
        ctx = make_context()
        dp = TransferQueueDataPlane(tq_client=RecordingFakeTQ())
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)
        dp.append_rewards(ctx, partition.partition_id, [1.0])

        meta = dp.append_advantages(ctx, partition.partition_id, [0.5])
        assert meta.status == PartitionStatus.TRAIN_READY

    def test_state_machine_train_ready_to_training_to_train_done(self):
        """Design doc section 3 steps 18-19: trainer marks TRAINING then TRAIN_DONE."""
        ctx = make_context()
        dp = TransferQueueDataPlane(tq_client=RecordingFakeTQ())
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)
        dp.append_rewards(ctx, partition.partition_id, [1.0])
        dp.append_advantages(ctx, partition.partition_id, [0.5])

        meta = dp.mark_training(ctx, partition.partition_id)
        assert meta.status == PartitionStatus.TRAINING

        meta = dp.mark_trained(ctx, partition.partition_id)
        assert meta.status == PartitionStatus.TRAIN_DONE

    def test_state_machine_train_done_to_cleared(self):
        """Design doc section 3 step 22: clear partition after weight sync -> CLEARED."""
        ctx = make_context()
        dp = TransferQueueDataPlane(tq_client=RecordingFakeTQ())
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)
        dp.append_rewards(ctx, partition.partition_id, [1.0])
        dp.append_advantages(ctx, partition.partition_id, [0.5])
        dp.mark_training(ctx, partition.partition_id)
        dp.mark_trained(ctx, partition.partition_id)

        dp.clear_partition(ctx, partition.partition_id)
        meta = dp.list_partitions(ctx)[0]
        assert meta.status == PartitionStatus.CLEARED

    def test_full_lifecycle_state_transitions(self):
        """Verify complete OPEN -> ROLLOUT_DONE -> REWARD_DONE -> TRAIN_READY -> TRAINING -> TRAIN_DONE -> CLEARED."""
        ctx = make_context()
        dp = TransferQueueDataPlane(tq_client=RecordingFakeTQ())
        partition = dp.create_partition(ctx, target_groups=1)
        assert partition.status == PartitionStatus.OPEN

        meta = dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)
        assert meta.status == PartitionStatus.ROLLOUT_DONE

        meta = dp.append_rewards(ctx, partition.partition_id, [1.0])
        assert meta.status == PartitionStatus.REWARD_DONE

        meta = dp.append_advantages(ctx, partition.partition_id, [0.5])
        assert meta.status == PartitionStatus.TRAIN_READY

        meta = dp.mark_training(ctx, partition.partition_id)
        assert meta.status == PartitionStatus.TRAINING

        meta = dp.mark_trained(ctx, partition.partition_id)
        assert meta.status == PartitionStatus.TRAIN_DONE

        dp.clear_partition(ctx, partition.partition_id)
        meta = dp.list_partitions(ctx)[0]
        assert meta.status == PartitionStatus.CLEARED

    def test_capacity_guard_max_rows(self):
        """Design doc section 3 step 7: check_capacity enforces row limits."""
        ctx = make_context()
        dp = TransferQueueDataPlane(
            tq_client=RecordingFakeTQ(),
            tq_config=TransferQueueRuntimeConfig(max_rows=2),
        )
        assert dp.check_capacity(ctx)

        p = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, p.partition_id, [make_sample(0), make_sample(1), make_sample(2)], seal=True)
        assert not dp.check_capacity(ctx)

    def test_capacity_guard_max_rows_per_context(self):
        """Design doc section 2.3: per-context capacity isolation."""
        ctx_a = make_context('lora_a', tenant='tenant_a')
        ctx_b = make_context('lora_b', tenant='tenant_b')
        dp = TransferQueueDataPlane(
            tq_client=RecordingFakeTQ(),
            tq_config=TransferQueueRuntimeConfig(max_rows_per_context=1),
        )

        pa = dp.create_partition(ctx_a, target_groups=1)
        dp.put_rollout_batch(ctx_a, pa.partition_id, [make_sample(0)], seal=True)
        assert not dp.check_capacity(ctx_a)
        assert dp.check_capacity(ctx_b)

    def test_metadata_validation_on_write(self):
        """Design doc section 5.1: context metadata takes precedence; sample metadata is overwritten."""
        ctx = make_context()
        dp = TransferQueueDataPlane(tq_client=RecordingFakeTQ())
        partition = dp.create_partition(ctx, target_groups=1)

        bad_sample = dict(make_sample(0))
        bad_sample['metadata'] = {'adapter_name': 'wrong_adapter'}
        meta = dp.put_rollout_batch(ctx, partition.partition_id, [bad_sample], seal=True)
        assert meta.status == PartitionStatus.ROLLOUT_DONE

    def test_list_train_ready_partitions(self):
        """Design doc section 3 step 15: TrainerScheduler queries TRAIN_READY partitions."""
        ctx = make_context()
        dp = TransferQueueDataPlane(tq_client=RecordingFakeTQ())
        p = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, p.partition_id, [make_sample(0)], seal=True)
        dp.append_rewards(ctx, p.partition_id, [1.0])
        dp.append_advantages(ctx, p.partition_id, [0.5])

        ready = dp.list_train_ready_partitions()
        assert len(ready) == 1
        assert ready[0].partition_id == p.partition_id
        assert ready[0].status == PartitionStatus.TRAIN_READY

    def test_grpo_group_same_adapter_allows_same_key_different_version(self):
        """Design doc section 2.3: partition is scoped by adapter key, not policy_version.

        context.key = tenant_id/training_run_id/adapter_name (no policy_version).
        Same adapter with different policy_version shares the same partition namespace.
        Policy version is tracked per-sample in metadata tags.
        """
        ctx_v0 = make_context(version=0)
        ctx_v1 = make_context(version=1)
        assert ctx_v0.key == ctx_v1.key
        dp = TransferQueueDataPlane(tq_client=RecordingFakeTQ())

        p = dp.create_partition(ctx_v0, target_groups=1)
        meta = dp.put_rollout_batch(ctx_v1, p.partition_id, [make_sample(0)], seal=True)
        assert meta.status == PartitionStatus.ROLLOUT_DONE

    def test_grpo_group_rejects_different_adapter(self):
        """Design doc section 2.3: different adapter must not share partition."""
        ctx_a = make_context('lora_a')
        ctx_b = make_context('lora_b')
        assert ctx_a.key != ctx_b.key
        dp = TransferQueueDataPlane(tq_client=RecordingFakeTQ())

        p = dp.create_partition(ctx_a, target_groups=1)
        with pytest.raises(ValueError, match='belongs to'):
            dp.put_rollout_batch(ctx_b, p.partition_id, [make_sample(0)])

    def test_auto_seal_when_ready_groups_reach_target(self):
        """Partition auto-seals when ready_groups >= target_groups without explicit seal."""
        ctx = make_context()
        dp = TransferQueueDataPlane(tq_client=RecordingFakeTQ())
        partition = dp.create_partition(ctx, target_groups=2)

        meta = dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], ready_groups=1)
        assert meta.status == PartitionStatus.OPEN

        meta = dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(1)], ready_groups=1)
        assert meta.status == PartitionStatus.ROLLOUT_DONE

    def test_clear_partition_removes_tq_data(self):
        """Design doc section 3 step 22: clear_partition must remove data from TQ."""
        fake = RecordingFakeTQ()
        ctx = make_context()
        dp = TransferQueueDataPlane(tq_client=fake)
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)

        assert len(fake.fields[partition.partition_id]) > 0
        dp.clear_partition(ctx, partition.partition_id)
        assert len(fake.fields.get(partition.partition_id, {})) == 0

    def test_multi_context_isolation_in_same_data_plane(self):
        """Design doc section 2: multiple tenants/runs/adapters coexist in same data plane."""
        fake = RecordingFakeTQ()
        dp = TransferQueueDataPlane(tq_client=fake)

        ctx_a = make_context('lora_a', tenant='tenant_a', run='run_1')
        ctx_b = make_context('lora_b', tenant='tenant_b', run='run_2')

        pa = dp.create_partition(ctx_a, target_groups=1)
        pb = dp.create_partition(ctx_b, target_groups=1)

        dp.put_rollout_batch(ctx_a, pa.partition_id, [make_sample(0)], seal=True)
        dp.put_rollout_batch(ctx_b, pb.partition_id, [make_sample(1)], seal=True)

        partitions_a = dp.list_partitions(ctx_a)
        partitions_b = dp.list_partitions(ctx_b)
        assert len(partitions_a) == 1
        assert len(partitions_b) == 1
        assert partitions_a[0].context.adapter_name == 'lora_a'
        assert partitions_b[0].context.adapter_name == 'lora_b'

        dp.append_rewards(ctx_a, pa.partition_id, [1.0])
        dp.append_advantages(ctx_a, pa.partition_id, [0.5])

        ready = dp.list_train_ready_partitions()
        assert len(ready) == 1
        assert ready[0].context.key == ctx_a.key

    def test_reward_count_mismatch_raises(self):
        """append_rewards must reject mismatched reward count."""
        ctx = make_context()
        dp = TransferQueueDataPlane(tq_client=RecordingFakeTQ())
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0), make_sample(1)], seal=True)

        with pytest.raises(ValueError, match='reward count'):
            dp.append_rewards(ctx, partition.partition_id, [1.0])

    def test_advantage_count_mismatch_raises(self):
        """append_advantages must reject mismatched advantage count."""
        ctx = make_context()
        dp = TransferQueueDataPlane(tq_client=RecordingFakeTQ())
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)
        dp.append_rewards(ctx, partition.partition_id, [1.0])

        with pytest.raises(ValueError, match='advantage count'):
            dp.append_advantages(ctx, partition.partition_id, [0.5, 0.3])

    def test_build_streaming_dataloader_returns_samples_with_metadata(self):
        """build_streaming_dataloader must return samples with sample_id and metadata."""
        ctx = make_context()
        dp = TransferQueueDataPlane(tq_client=RecordingFakeTQ())
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)

        samples = dp.build_streaming_dataloader(ctx, partition.partition_id)
        assert len(samples) == 1
        assert 'sample_id' in samples[0]
        assert 'metadata' in samples[0]
        assert samples[0]['metadata']['adapter_name'] == 'lora'
        assert samples[0]['metadata']['tenant_id'] == 'tenant_a'

    def test_build_streaming_dataloader_rejects_cross_context(self):
        """build_streaming_dataloader must reject cross-context access."""
        ctx = make_context()
        other = make_context('other')
        dp = TransferQueueDataPlane(tq_client=RecordingFakeTQ())
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)

        with pytest.raises(ValueError, match='belongs to'):
            dp.build_streaming_dataloader(other, partition.partition_id)

    def test_append_rewards_rejects_cross_context(self):
        """append_rewards must reject cross-context access to prevent data corruption."""
        ctx = make_context()
        other = make_context('other')
        dp = TransferQueueDataPlane(tq_client=RecordingFakeTQ())
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)

        with pytest.raises(ValueError, match='metadata mismatch'):
            dp.append_rewards(other, partition.partition_id, [1.0])

    def test_append_advantages_rejects_cross_context(self):
        """append_advantages must reject cross-context access to prevent data corruption."""
        ctx = make_context()
        other = make_context('other')
        dp = TransferQueueDataPlane(tq_client=RecordingFakeTQ())
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)
        dp.append_rewards(ctx, partition.partition_id, [1.0])

        with pytest.raises(ValueError, match='metadata mismatch'):
            dp.append_advantages(other, partition.partition_id, [0.5])

    def test_cross_context_data_isolation(self):
        """Verify that operations on one context cannot affect another context's data."""
        ctx_a = make_context('tenant_a', run='run_a')
        ctx_b = make_context('tenant_b', run='run_b')
        dp = TransferQueueDataPlane(tq_client=RecordingFakeTQ())

        # Create partitions for both contexts
        partition_a = dp.create_partition(ctx_a, target_groups=1)
        partition_b = dp.create_partition(ctx_b, target_groups=1)

        # Add data to both partitions
        dp.put_rollout_batch(ctx_a, partition_a.partition_id, [make_sample(0)], seal=True)
        dp.put_rollout_batch(ctx_b, partition_b.partition_id, [make_sample(1)], seal=True)

        # Try to append rewards using wrong context
        with pytest.raises(ValueError, match='metadata mismatch'):
            dp.append_rewards(ctx_b, partition_a.partition_id, [2.0])

        # Verify correct context can still append
        dp.append_rewards(ctx_a, partition_a.partition_id, [1.0])

        # Verify data integrity
        samples_a = dp.build_streaming_dataloader(ctx_a, partition_a.partition_id)
        assert len(samples_a) == 1
        assert samples_a[0]['rewards'] == 1.0

        # Try to append advantages using wrong context
        with pytest.raises(ValueError, match='metadata mismatch'):
            dp.append_advantages(ctx_b, partition_a.partition_id, [0.5])

        # Verify correct context can still append
        dp.append_advantages(ctx_a, partition_a.partition_id, [0.5])

        # Verify data integrity
        samples_a = dp.build_streaming_dataloader(ctx_a, partition_a.partition_id)
        assert samples_a[0]['advantages'] == 0.5

    def test_claim_reward_batch_returns_partition_and_samples(self):
        """claim_reward_batch must return (PartitionMetadata, list[SampleRecord])."""
        ctx = make_context()
        dp = TransferQueueDataPlane(tq_client=RecordingFakeTQ())
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)

        meta, samples = dp.claim_reward_batch(ctx, batch_size=10)
        assert meta.partition_id == partition.partition_id
        assert len(samples) == 1
        assert 'sample_id' in samples[0]

    def test_claim_advantage_batch_returns_partition_and_samples(self):
        """claim_advantage_batch must return (PartitionMetadata, list[SampleRecord])."""
        ctx = make_context()
        dp = TransferQueueDataPlane(tq_client=RecordingFakeTQ())
        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)
        dp.append_rewards(ctx, partition.partition_id, [1.0])

        meta, samples = dp.claim_advantage_batch(ctx, batch_size=10)
        assert meta.partition_id == partition.partition_id
        assert len(samples) == 1

    def test_claim_raises_when_no_ready_partition(self):
        """claim must raise LookupError when no partition in target status."""
        ctx = make_context()
        dp = TransferQueueDataPlane(tq_client=RecordingFakeTQ())

        with pytest.raises(LookupError, match='no reward-ready partition'):
            dp.claim_reward_batch(ctx, batch_size=10)

    def test_get_metadata_returns_all_partitions_for_context(self):
        """get_metadata is used by StalenessManager to compute capacity."""
        ctx = make_context()
        dp = TransferQueueDataPlane(tq_client=RecordingFakeTQ())
        p0 = dp.create_partition(ctx, target_groups=1)
        p1 = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, p0.partition_id, [make_sample(0)], seal=True)
        dp.put_rollout_batch(ctx, p1.partition_id, [make_sample(1)], seal=True)

        metadata = dp.get_metadata(ctx)
        assert len(metadata) == 2


# =============================================================================
# Integration: data_plane with workers (design doc section 3 sequence)
# =============================================================================


class TestDataPlaneWorkerIntegration:
    """Verify data_plane works correctly with all worker types per design doc section 3."""

    def test_rollouter_to_reward_to_advantage_to_trainer_flow(self):
        """Design doc section 3.1: complete sequence from rollout to clear."""
        ctx = make_context()
        dp = TransferQueueDataPlane(tq_client=RecordingFakeTQ())
        registry = AdapterRegistry()
        registry.register(ctx)

        partition = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, partition.partition_id, [make_sample(0)], seal=True)
        assert partition.status == PartitionStatus.ROLLOUT_DONE

        reward_worker = RewardWorker(data_plane=dp, reward_registry={'f1_reward': lambda trajectories, **_: [1.0]})
        meta = reward_worker.run_once(ctx)
        assert meta.status == PartitionStatus.REWARD_DONE

        adv_worker = AdvantageWorker(data_plane=dp)
        meta = adv_worker.run_once(ctx)
        assert meta.status == PartitionStatus.TRAIN_READY

        ready = dp.list_train_ready_partitions()
        assert len(ready) == 1

        dp.mark_training(ctx, partition.partition_id)
        dataloader = dp.build_streaming_dataloader(ctx, partition.partition_id)
        assert len(dataloader) == 1

        dp.mark_trained(ctx, partition.partition_id)
        registry.on_train_started(ctx, partition.partition_id)
        registry.on_train_finished(ctx, partition.partition_id)
        registry.on_weight_sync_started(ctx)
        new_ctx = registry.on_weight_sync_finished(ctx, adapter_revision='/tmp/lora-v1')
        assert new_ctx.policy_version == 1

        dp.clear_partition(ctx, partition.partition_id)
        registry.on_partition_cleared(ctx, partition.partition_id)

        assert dp.list_partitions(ctx)[0].status == PartitionStatus.CLEARED
        assert registry.get(ctx).live_partitions == set()

    def test_staleness_manager_uses_data_plane_metadata(self):
        """Design doc section 3 step 4-5: StalenessManager reads from data_plane."""
        ctx = make_context()
        dp = TransferQueueDataPlane(tq_client=RecordingFakeTQ())
        manager = StalenessManager(max_staleness=1, target_groups_per_partition=1)

        capacity = manager.get_rollout_capacity(ctx, dp.get_metadata(ctx))
        assert capacity.available_groups == 2

        p0 = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, p0.partition_id, [make_sample(0)], seal=True)
        capacity = manager.get_rollout_capacity(ctx, dp.get_metadata(ctx))
        assert capacity.available_groups == 1

        p1 = dp.create_partition(ctx, target_groups=1)
        dp.put_rollout_batch(ctx, p1.partition_id, [make_sample(1)], seal=True)
        capacity = manager.get_rollout_capacity(ctx, dp.get_metadata(ctx))
        assert capacity.available_groups == 0
        assert capacity.action == 'sleep'
