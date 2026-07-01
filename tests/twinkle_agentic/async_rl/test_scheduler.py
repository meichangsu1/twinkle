# Copyright (c) ModelScope Contributors. All rights reserved.
import time

import pytest

from twinkle_agentic.async_rl import (
    AdapterRegistry,
    AdapterState,
    AdaptiveTrainPolicy,
    CostAwareTrainPolicy,
    DeficitFairTrainPolicy,
    EDFTrainPolicy,
    FIFOTrainPolicy,
    LRUTrainPolicy,
    PartitionMetadata,
    PartitionStatus,
    PreferCurrentTrainPolicy,
    PriorityTrainPolicy,
    RejectedPartition,
    SJFTrainPolicy,
    ScheduleDecision,
    StrideTrainPolicy,
    TrainerScheduler,
    TrainerSchedulerConfig,
    TrainingContext,
    WeightedFairQueueTrainPolicy,
)


def _ctx(name='a', tenant='t', run='r'):
    return TrainingContext(
        tenant_id=tenant,
        training_run_id=run,
        base_model_id='base',
        adapter_name=f'lora_{name}',
    )


def _partition(context, *, status=PartitionStatus.TRAIN_READY, num_rows=8, created_at=None):
    return PartitionMetadata(
        context=context,
        partition_id=f'{context.key}/train_0',
        policy_version=context.policy_version,
        target_groups=1,
        ready_groups=1,
        status=status,
        created_at=created_at or time.time(),
        num_rows=num_rows,
    )


def _register_active(registry, context, weight=1.0):
    registry.register(context, weight=weight)


# ---------------------------------------------------------------------------
# Gating tests
# ---------------------------------------------------------------------------


class TestGating:

    def test_rejects_non_train_ready(self):
        registry = AdapterRegistry()
        ctx = _ctx()
        _register_active(registry, ctx)
        scheduler = TrainerScheduler(adapter_registry=registry)
        p = _partition(ctx, status=PartitionStatus.ROLLOUT_DONE)
        assert scheduler.next_partition([p]) is None

    def test_rejects_failed_adapter(self):
        registry = AdapterRegistry()
        ctx = _ctx()
        _register_active(registry, ctx)
        registry.mark_failed(ctx, 'boom')
        scheduler = TrainerScheduler(adapter_registry=registry)
        p = _partition(ctx)
        assert scheduler.next_partition([p]) is None

    def test_rejects_cancelled_adapter(self):
        registry = AdapterRegistry()
        ctx = _ctx()
        registry.register(ctx, state=AdapterState.CANCELLED)
        scheduler = TrainerScheduler(adapter_registry=registry)
        p = _partition(ctx)
        assert scheduler.next_partition([p]) is None

    def test_rejects_draining_adapter(self):
        registry = AdapterRegistry()
        ctx = _ctx()
        registry.register(ctx, state=AdapterState.DRAINING)
        scheduler = TrainerScheduler(adapter_registry=registry)
        p = _partition(ctx)
        assert scheduler.next_partition([p]) is None

    def test_rejects_syncing_adapter(self):
        registry = AdapterRegistry()
        ctx = _ctx()
        _register_active(registry, ctx)
        registry.on_weight_sync_started(ctx)
        scheduler = TrainerScheduler(adapter_registry=registry)
        p = _partition(ctx)
        assert scheduler.next_partition([p]) is None

    def test_rejects_training_adapter(self):
        registry = AdapterRegistry()
        ctx = _ctx()
        _register_active(registry, ctx)
        registry.on_train_started(ctx, 'some_partition')
        scheduler = TrainerScheduler(adapter_registry=registry)
        p = _partition(ctx)
        assert scheduler.next_partition([p]) is None

    def test_rejects_unknown_adapter(self):
        registry = AdapterRegistry()
        ctx = _ctx()
        scheduler = TrainerScheduler(adapter_registry=registry)
        p = _partition(ctx)
        assert scheduler.next_partition([p]) is None

    def test_passes_active_adapter(self):
        registry = AdapterRegistry()
        ctx = _ctx()
        _register_active(registry, ctx)
        scheduler = TrainerScheduler(adapter_registry=registry)
        p = _partition(ctx)
        result = scheduler.next_partition([p])
        assert result is not None
        assert result.partition_id == p.partition_id

    def test_is_compatible_hook(self):
        registry = AdapterRegistry()
        ctx = _ctx()
        _register_active(registry, ctx)

        class StrictScheduler(TrainerScheduler):
            def is_compatible(self, partition):
                return partition.context.algorithm == 'ppo'

        scheduler = StrictScheduler(adapter_registry=registry)
        p = _partition(ctx)
        assert scheduler.next_partition([p]) is None


# ---------------------------------------------------------------------------
# PreferCurrentTrainPolicy
# ---------------------------------------------------------------------------


class TestPreferCurrentTrainPolicy:

    def test_stays_on_same_adapter(self):
        policy = PreferCurrentTrainPolicy()
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        pa = _partition(ctx_a, created_at=100)
        pb = _partition(ctx_b, created_at=50)
        result = policy.pick_next_partition([pa, pb], current_context=ctx_a)
        assert result.context.key == ctx_a.key

    def test_switches_when_no_ready(self):
        policy = PreferCurrentTrainPolicy()
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        pb = _partition(ctx_b, created_at=50)
        result = policy.pick_next_partition([pb], current_context=ctx_a)
        assert result.context.key == ctx_b.key

    def test_picks_most_ready(self):
        policy = PreferCurrentTrainPolicy()
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        pa1 = _partition(ctx_a, created_at=100)
        pb1 = _partition(ctx_b, created_at=50)
        pb2 = PartitionMetadata(
            context=ctx_b, partition_id=f'{ctx_b.key}/train_1',
            policy_version=0, target_groups=1, ready_groups=1,
            status=PartitionStatus.TRAIN_READY, created_at=60, num_rows=8,
        )
        result = policy.pick_next_partition([pa1, pb1, pb2])
        assert result.context.key == ctx_b.key

    def test_tiebreak_oldest(self):
        policy = PreferCurrentTrainPolicy()
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        pa = _partition(ctx_a, created_at=100)
        pb = _partition(ctx_b, created_at=50)
        result = policy.pick_next_partition([pa, pb])
        assert result.context.key == ctx_b.key

    def test_switch_penalty_blocks(self):
        policy = PreferCurrentTrainPolicy(switch_penalty=3.0)
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        pb = _partition(ctx_b, created_at=50)
        result = policy.pick_next_partition([pb], current_context=ctx_a)
        assert result is None

    def test_switch_penalty_allows(self):
        policy = PreferCurrentTrainPolicy(switch_penalty=1.0)
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        pb = _partition(ctx_b, created_at=50)
        result = policy.pick_next_partition([pb], current_context=ctx_a)
        assert result is not None

    def test_empty_candidates(self):
        policy = PreferCurrentTrainPolicy()
        assert policy.pick_next_partition([]) is None


# ---------------------------------------------------------------------------
# CostAwareTrainPolicy
# ---------------------------------------------------------------------------


class TestCostAwareTrainPolicy:

    def test_stays_on_same_adapter(self):
        policy = CostAwareTrainPolicy(switch_cost=10.0)
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        pa = _partition(ctx_a)
        pb = _partition(ctx_b)
        result = policy.pick_next_partition([pa, pb], current_context=ctx_a)
        assert result.context.key == ctx_a.key

    def test_picks_highest_benefit(self):
        policy = CostAwareTrainPolicy(switch_cost=0.0)
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        pa = _partition(ctx_a, created_at=100)
        pb1 = _partition(ctx_b, created_at=50)
        pb2 = PartitionMetadata(
            context=ctx_b, partition_id=f'{ctx_b.key}/train_1',
            policy_version=0, target_groups=1, ready_groups=1,
            status=PartitionStatus.TRAIN_READY, created_at=60, num_rows=8,
        )
        result = policy.pick_next_partition([pa, pb1, pb2])
        assert result.context.key == ctx_b.key

    def test_switch_cost_deters(self):
        policy = CostAwareTrainPolicy(switch_cost=100.0)
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        pa1 = _partition(ctx_a, created_at=100)
        pa2 = PartitionMetadata(
            context=ctx_a, partition_id=f'{ctx_a.key}/train_1',
            policy_version=0, target_groups=1, ready_groups=1,
            status=PartitionStatus.TRAIN_READY, created_at=110, num_rows=8,
        )
        pb = _partition(ctx_b, created_at=50)
        policy._last_context_key = ctx_a.key
        result = policy.pick_next_partition([pa1, pa2, pb])
        assert result.context.key == ctx_a.key

    def test_empty_candidates(self):
        policy = CostAwareTrainPolicy()
        assert policy.pick_next_partition([]) is None


# ---------------------------------------------------------------------------
# SJFTrainPolicy
# ---------------------------------------------------------------------------


class TestSJFTrainPolicy:

    def test_picks_smallest_partition(self):
        policy = SJFTrainPolicy()
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        pa = _partition(ctx_a, num_rows=100, created_at=100)
        pb = _partition(ctx_b, num_rows=10, created_at=50)
        result = policy.pick_next_partition([pa, pb])
        assert result.context.key == ctx_b.key

    def test_tiebreak_oldest(self):
        policy = SJFTrainPolicy()
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        pa = _partition(ctx_a, num_rows=10, created_at=100)
        pb = _partition(ctx_b, num_rows=10, created_at=50)
        result = policy.pick_next_partition([pa, pb])
        assert result.context.key == ctx_b.key

    def test_ignores_current_context(self):
        policy = SJFTrainPolicy()
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        pa = _partition(ctx_a, num_rows=100, created_at=100)
        pb = _partition(ctx_b, num_rows=10, created_at=50)
        result = policy.pick_next_partition([pa, pb], current_context=ctx_a)
        assert result.context.key == ctx_b.key

    def test_empty_candidates(self):
        policy = SJFTrainPolicy()
        assert policy.pick_next_partition([]) is None


# ---------------------------------------------------------------------------
# DeficitFairTrainPolicy
# ---------------------------------------------------------------------------


class TestDeficitFairTrainPolicy:

    def test_round_robin_equal_weight(self):
        policy = DeficitFairTrainPolicy(quantum=1.0)
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        pa = _partition(ctx_a, created_at=100)
        pb = _partition(ctx_b, created_at=50)
        first = policy.pick_next_partition([pa, pb])
        second = policy.pick_next_partition([pa, pb])
        assert first.context.key != second.context.key

    def test_weighted(self):
        policy = DeficitFairTrainPolicy(quantum=1.0, weights={'t/r/lora_a': 3.0, 't/r/lora_b': 1.0})
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        pa = _partition(ctx_a, created_at=100)
        pb = _partition(ctx_b, created_at=50)
        picks = [policy.pick_next_partition([pa, pb]) for _ in range(4)]
        a_count = sum(1 for p in picks if p.context.key == ctx_a.key)
        assert a_count >= 2

    def test_uses_adapter_registry_weight(self):
        registry = AdapterRegistry()
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        registry.register(ctx_a, weight=3.0)
        registry.register(ctx_b, weight=1.0)
        policy = DeficitFairTrainPolicy(quantum=1.0, adapter_registry=registry)
        pa = _partition(ctx_a, created_at=100)
        pb = _partition(ctx_b, created_at=50)
        picks = [policy.pick_next_partition([pa, pb]) for _ in range(4)]
        a_count = sum(1 for p in picks if p.context.key == ctx_a.key)
        assert a_count >= 2

    def test_explicit_weights_override(self):
        registry = AdapterRegistry()
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        registry.register(ctx_a, weight=1.0)
        registry.register(ctx_b, weight=1.0)
        policy = DeficitFairTrainPolicy(
            quantum=1.0, adapter_registry=registry,
            weights={'t/r/lora_a': 10.0, 't/r/lora_b': 1.0},
        )
        pa = _partition(ctx_a, created_at=100)
        pb = _partition(ctx_b, created_at=50)
        picks = [policy.pick_next_partition([pa, pb]) for _ in range(11)]
        a_count = sum(1 for p in picks if p.context.key == ctx_a.key)
        assert a_count > 5

    def test_empty_candidates(self):
        policy = DeficitFairTrainPolicy()
        assert policy.pick_next_partition([]) is None


# ---------------------------------------------------------------------------
# StrideTrainPolicy
# ---------------------------------------------------------------------------


class TestStrideTrainPolicy:

    def test_equal_weight_alternates(self):
        policy = StrideTrainPolicy(weights={'t/r/lora_a': 1.0, 't/r/lora_b': 1.0})
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        pa = _partition(ctx_a, created_at=100)
        pb = _partition(ctx_b, created_at=50)
        first = policy.pick_next_partition([pa, pb])
        second = policy.pick_next_partition([pa, pb])
        assert first.context.key != second.context.key

    def test_higher_weight_more_often(self):
        policy = StrideTrainPolicy(weights={'t/r/lora_a': 3.0, 't/r/lora_b': 1.0})
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        pa = _partition(ctx_a, created_at=100)
        pb = _partition(ctx_b, created_at=50)
        picks = [policy.pick_next_partition([pa, pb]) for _ in range(8)]
        a_count = sum(1 for p in picks if p.context.key == ctx_a.key)
        assert a_count >= 4

    def test_deterministic(self):
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        pa = _partition(ctx_a, created_at=100)
        pb = _partition(ctx_b, created_at=50)

        policy1 = StrideTrainPolicy(weights={'t/r/lora_a': 2.0, 't/r/lora_b': 1.0})
        picks1 = [policy1.pick_next_partition([pa, pb]).context.key for _ in range(6)]

        policy2 = StrideTrainPolicy(weights={'t/r/lora_a': 2.0, 't/r/lora_b': 1.0})
        picks2 = [policy2.pick_next_partition([pa, pb]).context.key for _ in range(6)]

        assert picks1 == picks2

    def test_empty_candidates(self):
        policy = StrideTrainPolicy()
        assert policy.pick_next_partition([]) is None


# ---------------------------------------------------------------------------
# WeightedFairQueueTrainPolicy
# ---------------------------------------------------------------------------


class TestWeightedFairQueueTrainPolicy:

    def test_higher_weight_lower_delay(self):
        policy = WeightedFairQueueTrainPolicy(weights={'t/r/lora_a': 3.0, 't/r/lora_b': 1.0})
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        pa = _partition(ctx_a, created_at=100)
        pb = _partition(ctx_b, created_at=50)
        picks = [policy.pick_next_partition([pa, pb]) for _ in range(8)]
        a_count = sum(1 for p in picks if p.context.key == ctx_a.key)
        assert a_count >= 4

    def test_virtual_time_advances(self):
        policy = WeightedFairQueueTrainPolicy()
        ctx_a = _ctx('a')
        pa = _partition(ctx_a)
        v0 = policy._virtual_time
        policy.pick_next_partition([pa])
        assert policy._virtual_time > v0

    def test_single_adapter_always_picks(self):
        policy = WeightedFairQueueTrainPolicy()
        ctx_a = _ctx('a')
        pa = _partition(ctx_a)
        for _ in range(5):
            result = policy.pick_next_partition([pa])
            assert result.context.key == ctx_a.key

    def test_empty_candidates(self):
        policy = WeightedFairQueueTrainPolicy()
        assert policy.pick_next_partition([]) is None


# ---------------------------------------------------------------------------
# LRUTrainPolicy
# ---------------------------------------------------------------------------


class TestLRUTrainPolicy:

    def test_picks_least_recently_trained(self):
        policy = LRUTrainPolicy()
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        pa = _partition(ctx_a, created_at=100)
        pb = _partition(ctx_b, created_at=50)
        first = policy.pick_next_partition([pa, pb])
        second = policy.pick_next_partition([pa, pb])
        assert first.context.key != second.context.key

    def test_updates_timestamp(self):
        policy = LRUTrainPolicy()
        ctx_a = _ctx('a')
        pa = _partition(ctx_a)
        policy.pick_next_partition([pa])
        assert ctx_a.key in policy._last_trained_at

    def test_new_adapter_first(self):
        policy = LRUTrainPolicy()
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        pa = _partition(ctx_a, created_at=100)
        pb = _partition(ctx_b, created_at=50)
        policy._last_trained_at[ctx_a.key] = time.time()
        result = policy.pick_next_partition([pa, pb])
        assert result.context.key == ctx_b.key

    def test_prevents_starvation(self):
        policy = LRUTrainPolicy()
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        ctx_c = _ctx('c')
        pa = _partition(ctx_a, created_at=100)
        pb = _partition(ctx_b, created_at=50)
        pc = _partition(ctx_c, created_at=75)
        seen = set()
        for _ in range(3):
            result = policy.pick_next_partition([pa, pb, pc])
            seen.add(result.context.key)
        assert len(seen) == 3

    def test_empty_candidates(self):
        policy = LRUTrainPolicy()
        assert policy.pick_next_partition([]) is None


# ---------------------------------------------------------------------------
# EDFTrainPolicy
# ---------------------------------------------------------------------------


class TestEDFTrainPolicy:

    def test_picks_oldest_without_registry(self):
        policy = EDFTrainPolicy()
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        pa = _partition(ctx_a, created_at=100)
        pb = _partition(ctx_b, created_at=50)
        result = policy.pick_next_partition([pa, pb])
        assert result.context.key == ctx_b.key

    def test_prefers_high_live_count(self):
        registry = AdapterRegistry()
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        registry.register(ctx_a)
        registry.register(ctx_b)
        registry.get(ctx_a).live_partitions = {'p1', 'p2', 'p3'}
        registry.get(ctx_b).live_partitions = {'p1'}
        policy = EDFTrainPolicy(max_staleness=1, adapter_registry=registry)
        pa = _partition(ctx_a, created_at=time.time())
        pb = _partition(ctx_b, created_at=time.time())
        result = policy.pick_next_partition([pa, pb])
        assert result.context.key == ctx_a.key

    def test_prefers_high_in_flight(self):
        registry = AdapterRegistry()
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        registry.register(ctx_a)
        registry.register(ctx_b)
        registry.get(ctx_a).in_flight_rollouts = 10
        policy = EDFTrainPolicy(max_staleness=1, adapter_registry=registry)
        pa = _partition(ctx_a, created_at=time.time())
        pb = _partition(ctx_b, created_at=time.time())
        result = policy.pick_next_partition([pa, pb])
        assert result.context.key == ctx_a.key

    def test_empty_candidates(self):
        policy = EDFTrainPolicy()
        assert policy.pick_next_partition([]) is None


# ---------------------------------------------------------------------------
# PriorityTrainPolicy
# ---------------------------------------------------------------------------


class TestPriorityTrainPolicy:

    def test_high_weight_first(self):
        policy = PriorityTrainPolicy(weights={'t/r/lora_a': 10.0, 't/r/lora_b': 1.0})
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        pa = _partition(ctx_a, created_at=100)
        pb = _partition(ctx_b, created_at=50)
        result = policy.pick_next_partition([pa, pb])
        assert result.context.key == ctx_a.key

    def test_aging_promotes(self):
        policy = PriorityTrainPolicy(aging_rate=100.0, weights={'t/r/lora_a': 1.0, 't/r/lora_b': 10.0})
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        pa = _partition(ctx_a, created_at=time.time() - 1000)
        pb = _partition(ctx_b, created_at=time.time())
        policy._wait_start[ctx_a.key] = time.time() - 1000
        result = policy.pick_next_partition([pa, pb])
        assert result.context.key == ctx_a.key

    def test_no_starvation(self):
        policy = PriorityTrainPolicy(aging_rate=1000.0, weights={'t/r/lora_a': 10.0, 't/r/lora_b': 1.0})
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        pa = _partition(ctx_a, created_at=time.time() - 100)
        pb = _partition(ctx_b, created_at=time.time() - 100)
        seen = set()
        for _ in range(10):
            result = policy.pick_next_partition([pa, pb])
            seen.add(result.context.key)
            time.sleep(0.02)
        assert len(seen) == 2

    def test_empty_candidates(self):
        policy = PriorityTrainPolicy()
        assert policy.pick_next_partition([]) is None


# ---------------------------------------------------------------------------
# AdaptiveTrainPolicy
# ---------------------------------------------------------------------------


class TestAdaptiveTrainPolicy:

    def test_low_load_prefers_current(self):
        policy = AdaptiveTrainPolicy(high_load_threshold=100.0)
        ctx_a = _ctx('a')
        pa = _partition(ctx_a)
        result = policy.pick_next_partition([pa])
        assert result is not None
        assert policy.current_policy_name == 'prefer_current'

    def test_high_load_edf(self):
        policy = AdaptiveTrainPolicy(high_load_threshold=0.5)
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        partitions = [_partition(ctx_a, created_at=100), _partition(ctx_b, created_at=50)]
        for _ in range(25):
            policy.pick_next_partition(partitions)
        assert policy.current_policy_name == 'edf'

    def test_medium_load_cost_aware(self):
        policy = AdaptiveTrainPolicy(high_load_threshold=100.0)
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        pa2 = PartitionMetadata(
            context=ctx_a, partition_id=f'{ctx_a.key}/train_1',
            policy_version=0, target_groups=1, ready_groups=1,
            status=PartitionStatus.TRAIN_READY, created_at=110, num_rows=8,
        )
        partitions = [_partition(ctx_a, created_at=100), _partition(ctx_b, created_at=50), pa2]
        policy.pick_next_partition(partitions)
        assert policy.current_policy_name == 'cost_aware'

    def test_empty_candidates(self):
        policy = AdaptiveTrainPolicy()
        assert policy.pick_next_partition([]) is None


# ---------------------------------------------------------------------------
# FIFOTrainPolicy
# ---------------------------------------------------------------------------


class TestFIFOTrainPolicy:

    def test_picks_oldest(self):
        policy = FIFOTrainPolicy()
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        pa = _partition(ctx_a, created_at=100)
        pb = _partition(ctx_b, created_at=50)
        result = policy.pick_next_partition([pa, pb])
        assert result.context.key == ctx_b.key

    def test_ignores_current_context(self):
        policy = FIFOTrainPolicy()
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        pa = _partition(ctx_a, created_at=100)
        pb = _partition(ctx_b, created_at=50)
        result = policy.pick_next_partition([pa, pb], current_context=ctx_a)
        assert result.context.key == ctx_b.key

    def test_empty_candidates(self):
        policy = FIFOTrainPolicy()
        assert policy.pick_next_partition([]) is None


# ---------------------------------------------------------------------------
# TrainerSchedulerConfig
# ---------------------------------------------------------------------------


class TestTrainerSchedulerConfig:

    def test_builds_prefer_current(self):
        registry = AdapterRegistry()
        config = TrainerSchedulerConfig(policy='prefer_current')
        policy = config.build_policy(registry)
        assert isinstance(policy, PreferCurrentTrainPolicy)

    def test_builds_cost_aware(self):
        registry = AdapterRegistry()
        config = TrainerSchedulerConfig(policy='cost_aware')
        policy = config.build_policy(registry)
        assert isinstance(policy, CostAwareTrainPolicy)

    def test_builds_sjf(self):
        registry = AdapterRegistry()
        config = TrainerSchedulerConfig(policy='sjf')
        policy = config.build_policy(registry)
        assert isinstance(policy, SJFTrainPolicy)

    def test_builds_fair(self):
        registry = AdapterRegistry()
        config = TrainerSchedulerConfig(policy='fair')
        policy = config.build_policy(registry)
        assert isinstance(policy, DeficitFairTrainPolicy)

    def test_builds_stride(self):
        registry = AdapterRegistry()
        config = TrainerSchedulerConfig(policy='stride')
        policy = config.build_policy(registry)
        assert isinstance(policy, StrideTrainPolicy)

    def test_builds_wfq(self):
        registry = AdapterRegistry()
        config = TrainerSchedulerConfig(policy='wfq')
        policy = config.build_policy(registry)
        assert isinstance(policy, WeightedFairQueueTrainPolicy)

    def test_builds_lru(self):
        registry = AdapterRegistry()
        config = TrainerSchedulerConfig(policy='lru')
        policy = config.build_policy(registry)
        assert isinstance(policy, LRUTrainPolicy)

    def test_builds_edf(self):
        registry = AdapterRegistry()
        config = TrainerSchedulerConfig(policy='edf')
        policy = config.build_policy(registry)
        assert isinstance(policy, EDFTrainPolicy)

    def test_builds_priority(self):
        registry = AdapterRegistry()
        config = TrainerSchedulerConfig(policy='priority')
        policy = config.build_policy(registry)
        assert isinstance(policy, PriorityTrainPolicy)

    def test_builds_adaptive(self):
        registry = AdapterRegistry()
        config = TrainerSchedulerConfig(policy='adaptive')
        policy = config.build_policy(registry)
        assert isinstance(policy, AdaptiveTrainPolicy)

    def test_builds_fifo(self):
        registry = AdapterRegistry()
        config = TrainerSchedulerConfig(policy='fifo')
        policy = config.build_policy(registry)
        assert isinstance(policy, FIFOTrainPolicy)

    def test_unknown_policy_raises(self):
        registry = AdapterRegistry()
        config = TrainerSchedulerConfig(policy='nonexistent')
        with pytest.raises(ValueError, match='unknown train schedule policy'):
            config.build_policy(registry)

    def test_params_passed(self):
        registry = AdapterRegistry()
        config = TrainerSchedulerConfig(
            policy='prefer_current',
            switch_penalty=5.0,
        )
        policy = config.build_policy(registry)
        assert policy.switch_penalty == 5.0


# ---------------------------------------------------------------------------
# ScheduleDecision
# ---------------------------------------------------------------------------


class TestScheduleDecision:

    def test_no_candidates(self):
        registry = AdapterRegistry()
        scheduler = TrainerScheduler(adapter_registry=registry)
        decision = scheduler.get_schedule_decision([])
        assert decision.reason == 'no_candidates'
        assert decision.selected is None
        assert decision.total_candidates == 0

    def test_all_rejected(self):
        registry = AdapterRegistry()
        ctx = _ctx()
        scheduler = TrainerScheduler(adapter_registry=registry)
        p = _partition(ctx, status=PartitionStatus.ROLLOUT_DONE)
        decision = scheduler.get_schedule_decision([p])
        assert decision.reason == 'all_rejected'
        assert decision.eligible_count == 0
        assert len(decision.rejected) == 1
        assert decision.rejected[0].reason == 'not_train_ready'

    def test_selected(self):
        registry = AdapterRegistry()
        ctx = _ctx()
        _register_active(registry, ctx)
        scheduler = TrainerScheduler(adapter_registry=registry)
        p = _partition(ctx)
        decision = scheduler.get_schedule_decision([p])
        assert decision.reason == 'selected'
        assert decision.selected is not None
        assert decision.eligible_count == 1

    def test_policy_returned_none(self):
        registry = AdapterRegistry()
        ctx = _ctx()
        _register_active(registry, ctx)

        class NonePolicy:
            def pick_next_partition(self, candidates, current_context=None):
                return None

        scheduler = TrainerScheduler(adapter_registry=registry, train_policy=NonePolicy())
        p = _partition(ctx)
        decision = scheduler.get_schedule_decision([p])
        assert decision.reason == 'policy_returned_none'
        assert decision.selected is None

    def test_elapsed_ms(self):
        registry = AdapterRegistry()
        ctx = _ctx()
        _register_active(registry, ctx)
        scheduler = TrainerScheduler(adapter_registry=registry)
        p = _partition(ctx)
        decision = scheduler.get_schedule_decision([p])
        assert decision.elapsed_ms >= 0.0


# ---------------------------------------------------------------------------
# list_eligible_partitions
# ---------------------------------------------------------------------------


class TestListEligiblePartitions:

    def test_filters_correctly(self):
        registry = AdapterRegistry()
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        _register_active(registry, ctx_a)
        scheduler = TrainerScheduler(adapter_registry=registry)
        pa = _partition(ctx_a)
        pb = _partition(ctx_b, status=PartitionStatus.ROLLOUT_DONE)
        eligible = scheduler.list_eligible_partitions([pa, pb])
        assert len(eligible) == 1
        assert eligible[0].context.key == ctx_a.key


# ---------------------------------------------------------------------------
# Integration: Scheduler + TrainerWorker
# ---------------------------------------------------------------------------


class TestSchedulerIntegration:

    def test_multi_adapter_prefer_current(self):
        registry = AdapterRegistry()
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        _register_active(registry, ctx_a)
        _register_active(registry, ctx_b)
        scheduler = TrainerScheduler(
            adapter_registry=registry,
            train_policy=PreferCurrentTrainPolicy(),
        )
        pa = _partition(ctx_a, created_at=100)
        pb = _partition(ctx_b, created_at=50)
        result = scheduler.next_partition([pa, pb], current_context=ctx_a)
        assert result.context.key == ctx_a.key

    def test_multi_adapter_fair(self):
        registry = AdapterRegistry()
        ctx_a = _ctx('a')
        ctx_b = _ctx('b')
        _register_active(registry, ctx_a)
        _register_active(registry, ctx_b)
        scheduler = TrainerScheduler(
            adapter_registry=registry,
            train_policy=DeficitFairTrainPolicy(quantum=1.0),
        )
        pa = _partition(ctx_a, created_at=100)
        pb = _partition(ctx_b, created_at=50)
        first = scheduler.next_partition([pa, pb])
        second = scheduler.next_partition([pa, pb])
        assert first.context.key != second.context.key

    def test_adapter_state_transitions(self):
        registry = AdapterRegistry()
        ctx = _ctx()
        _register_active(registry, ctx)
        scheduler = TrainerScheduler(adapter_registry=registry)
        p = _partition(ctx)

        assert scheduler.next_partition([p]) is not None

        registry.on_weight_sync_started(ctx)
        assert scheduler.next_partition([p]) is None

        registry.on_weight_sync_finished(ctx)
        assert scheduler.next_partition([p]) is not None

    def test_config_driven_scheduler(self):
        registry = AdapterRegistry()
        ctx = _ctx()
        _register_active(registry, ctx)
        config = TrainerSchedulerConfig(policy='fifo')
        scheduler = TrainerScheduler(adapter_registry=registry, config=config)
        p = _partition(ctx)
        result = scheduler.next_partition([p])
        assert result is not None

    def test_list_train_ready_partitions_with_data_plane(self):
        from .fakes import FakeTransferQueueClient
        from twinkle_agentic.async_rl import TransferQueueDataPlane, TransferQueueRuntimeConfig

        registry = AdapterRegistry()
        ctx = _ctx()
        _register_active(registry, ctx)

        tq_config = TransferQueueRuntimeConfig(init=False)
        data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient(), tq_config=tq_config)
        data_plane.init_namespace(ctx)

        p = data_plane.create_partition(ctx, target_groups=1)
        trajectories = [
            {'sample_id': f's{i}', 'messages': [], 'group_id': 'g0', 'generation_idx': i, 'old_logps': []}
            for i in range(8)
        ]
        data_plane.put_rollout_batch(ctx, p.partition_id, trajectories, ready_groups=1, seal=True)
        meta, samples = data_plane.claim_reward_batch(ctx, batch_size=1024)
        data_plane.append_rewards(ctx, p.partition_id, [1.0] * len(samples))
        meta, samples = data_plane.claim_advantage_batch(ctx, batch_size=1024)
        data_plane.append_advantages(ctx, p.partition_id, [0.0] * len(samples))

        scheduler = TrainerScheduler(adapter_registry=registry, data_plane=data_plane)
        ready = scheduler.list_train_ready_partitions()
        assert len(ready) == 1
        assert ready[0].partition_id == p.partition_id

    def test_next_partition_no_arg_queries_data_plane(self):
        from .fakes import FakeTransferQueueClient
        from twinkle_agentic.async_rl import TransferQueueDataPlane, TransferQueueRuntimeConfig

        registry = AdapterRegistry()
        ctx = _ctx()
        _register_active(registry, ctx)

        tq_config = TransferQueueRuntimeConfig(init=False)
        data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient(), tq_config=tq_config)
        data_plane.init_namespace(ctx)

        p = data_plane.create_partition(ctx, target_groups=1)
        trajectories = [
            {'sample_id': f's{i}', 'messages': [], 'group_id': 'g0', 'generation_idx': i, 'old_logps': []}
            for i in range(8)
        ]
        data_plane.put_rollout_batch(ctx, p.partition_id, trajectories, ready_groups=1, seal=True)
        meta, samples = data_plane.claim_reward_batch(ctx, batch_size=1024)
        data_plane.append_rewards(ctx, p.partition_id, [1.0] * len(samples))
        meta, samples = data_plane.claim_advantage_batch(ctx, batch_size=1024)
        data_plane.append_advantages(ctx, p.partition_id, [0.0] * len(samples))

        scheduler = TrainerScheduler(adapter_registry=registry, data_plane=data_plane)
        result = scheduler.next_partition()
        assert result is not None
        assert result.partition_id == p.partition_id

    def test_list_train_ready_partitions_without_data_plane(self):
        registry = AdapterRegistry()
        scheduler = TrainerScheduler(adapter_registry=registry)
        ready = scheduler.list_train_ready_partitions()
        assert ready == []
