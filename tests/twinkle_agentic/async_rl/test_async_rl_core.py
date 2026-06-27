import asyncio

import pytest

from twinkle_agentic.async_rl import (
    AdapterRegistry,
    AdvantageWorker,
    AsyncRollouter,
    DeficitFairRolloutPolicy,
    PartitionStatus,
    PromptFeeder,
    PreferCurrentTrainPolicy,
    RewardWorker,
    StalenessManager,
    TrainerScheduler,
    TrainerWorker,
    TrainingContext,
    TransferQueueDataPlane,
    TransferQueueRuntimeConfig,
    WorkConservingRolloutPolicy,
)

from .fakes import FakeTransferQueueClient


def make_context(name='a', *, tenant='tenant', run='run', version=0):
    return TrainingContext(
        tenant_id=tenant,
        training_run_id=run,
        base_model_id='base',
        adapter_name=name,
        policy_version=version,
        reward_type='constant',
        loss_type='grpo',
        algorithm='grpo',
    )


def make_sample(i=0):
    return {
        'sample_id': f'sample_{i}',
        'messages': [{'role': 'user', 'content': f'q{i}'}],
        'group_id': f'g{i}',
        'generation_idx': 0,
    }


def test_training_context_namespace_and_metadata_validation():
    context = make_context('lora')
    assert context.partition_id(3) == 'tenant/run/lora/train_3'
    metadata = context.metadata()
    context.validate_metadata(metadata)
    metadata['adapter_name'] = 'other'
    with pytest.raises(ValueError, match='adapter_name'):
        context.validate_metadata(metadata)


@pytest.mark.skipif(
    __import__('importlib').util.find_spec('transfer_queue') is not None,
    reason='transfer_queue is installed',
)
def test_default_data_plane_requires_real_transfer_queue_when_not_installed():
    with pytest.raises(RuntimeError, match='transfer_queue is required'):
        TransferQueueDataPlane()


def test_data_plane_rollout_reward_advantage_and_clear():
    context = make_context('lora')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    data_plane.init_namespace(context)
    partition = data_plane.create_partition(context, target_groups=1)

    meta = data_plane.put_rollout_batch(context, partition.partition_id, [make_sample(0)], seal=True)
    assert meta.status == PartitionStatus.ROLLOUT_DONE

    reward_worker = RewardWorker(data_plane=data_plane, reward_registry={'constant': lambda trajectories, **_: [1.0]})
    meta = reward_worker.run_once(context)
    assert meta.status == PartitionStatus.REWARD_DONE

    adv_worker = AdvantageWorker(data_plane=data_plane)
    meta = adv_worker.run_once(context)
    assert meta.status == PartitionStatus.TRAIN_READY
    assert data_plane.list_train_ready_partitions()[0].partition_id == partition.partition_id

    data_plane.clear_partition(context, partition.partition_id)
    assert data_plane.list_partitions(context)[0].status == PartitionStatus.CLEARED


def test_data_plane_partition_id_is_rollout_step_not_policy_version():
    context = make_context('lora', version=3)
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())

    partition = data_plane.create_partition(context, target_groups=1)

    assert partition.partition_id == 'tenant/run/lora/train_0'
    assert partition.policy_version == 3
    next_partition = data_plane.create_partition(context, target_groups=1)
    assert next_partition.partition_id == 'tenant/run/lora/train_1'


def test_data_plane_rejects_cross_context_append():
    context = make_context('lora')
    other = make_context('other')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    partition = data_plane.create_partition(context, target_groups=1)
    with pytest.raises(ValueError, match='belongs to'):
        data_plane.put_rollout_batch(other, partition.partition_id, [make_sample(0)])


def test_data_plane_check_capacity_by_row_limits():
    context = make_context('lora')
    other = make_context('other')
    data_plane = TransferQueueDataPlane(
        tq_client=FakeTransferQueueClient(),
        tq_config=TransferQueueRuntimeConfig(max_rows=2, max_rows_per_context=1),
    )
    assert data_plane.check_capacity(context)

    p0 = data_plane.create_partition(context, target_groups=1)
    data_plane.put_rollout_batch(context, p0.partition_id, [make_sample(0)], seal=True)
    assert not data_plane.check_capacity(context)
    assert data_plane.check_capacity(other)

    p1 = data_plane.create_partition(other, target_groups=1)
    data_plane.put_rollout_batch(other, p1.partition_id, [make_sample(1)], seal=True)
    assert not data_plane.check_capacity(other)


def test_adapter_registry_blocks_current_adapter_during_sync_only():
    registry = AdapterRegistry()
    a = make_context('a')
    b = make_context('b', run='run_b')
    registry.register(a)
    registry.register(b)

    assert registry.can_accept_rollout(a)
    registry.on_weight_sync_started(a)
    assert not registry.can_accept_rollout(a)
    assert registry.can_accept_rollout(b)

    updated = registry.on_weight_sync_finished(a, adapter_revision='/tmp/a')
    assert updated.policy_version == 1
    assert updated.adapter_revision == '/tmp/a'
    assert registry.can_accept_rollout(a)


def test_staleness_capacity_by_live_partitions():
    context = make_context('a')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    manager = StalenessManager(max_staleness=1, target_groups_per_partition=1)

    assert manager.get_rollout_capacity(context, data_plane.get_metadata(context)).available_groups == 2

    p0 = data_plane.create_partition(context, target_groups=1)
    data_plane.put_rollout_batch(context, p0.partition_id, [make_sample(0)], seal=True)
    assert manager.get_rollout_capacity(context, data_plane.get_metadata(context)).available_groups == 1

    p1 = data_plane.create_partition(context, target_groups=1)
    data_plane.put_rollout_batch(context, p1.partition_id, [make_sample(1)], seal=True)
    capacity = manager.get_rollout_capacity(context, data_plane.get_metadata(context))
    assert capacity.available_groups == 0
    assert capacity.action == 'sleep'


def test_staleness_allows_filling_current_open_partition_at_limit():
    context = make_context('a')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    manager = StalenessManager(max_staleness=0, target_groups_per_partition=2)

    partition = data_plane.create_partition(context, target_groups=2)
    data_plane.put_rollout_batch(context, partition.partition_id, [make_sample(0)], seal=False)

    capacity = manager.get_rollout_capacity(context, data_plane.get_metadata(context))
    assert capacity.available_groups == 1
    assert capacity.action == 'submit'


def test_data_plane_allows_mixed_policy_versions_inside_open_partition():
    context_v0 = make_context('lora', version=0)
    context_v1 = context_v0.with_policy_version(1, adapter_revision='/tmp/lora-v1')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())

    partition = data_plane.create_partition(context_v0, target_groups=2)
    data_plane.put_rollout_batch(context_v0, partition.partition_id, [make_sample(0)], seal=False)
    meta = data_plane.put_rollout_batch(context_v1, partition.partition_id, [make_sample(1)], seal=False)

    assert meta.status == PartitionStatus.ROLLOUT_DONE
    rows = data_plane.build_streaming_dataloader(context_v1, partition.partition_id)
    assert [row['metadata']['policy_version'] for row in rows] == [0, 1]
    assert rows[0]['metadata']['partition_policy_version'] == 0

    reward_worker = RewardWorker(data_plane=data_plane, reward_registry={'constant': lambda trajectories, **_: [1.0, 1.0]})
    reward_worker.run_once(context_v1)
    AdvantageWorker(data_plane=data_plane).run_once(context_v1)
    assert data_plane.list_train_ready_partitions()[0].partition_id == partition.partition_id


def test_work_conserving_rollout_policy_prefers_less_live_work():
    a = make_context('a')
    b = make_context('b', run='run_b')
    policy = WorkConservingRolloutPolicy()
    from twinkle_agentic.async_rl import RolloutContextState

    selected = policy.pick_next_context([
        RolloutContextState(a, pending_groups=1, in_flight_rollouts=2, live_partitions=2, open_partitions=1,
                            train_ready_partitions=0, rollout_capacity=1),
        RolloutContextState(b, pending_groups=1, in_flight_rollouts=0, live_partitions=0, open_partitions=0,
                            train_ready_partitions=0, rollout_capacity=1),
    ])
    assert selected == b


def test_deficit_fair_rollout_policy_alternates_candidates():
    a = make_context('a')
    b = make_context('b', run='run_b')
    policy = DeficitFairRolloutPolicy()
    from twinkle_agentic.async_rl import RolloutContextState

    states = [
        RolloutContextState(a, 10, 0, 0, 0, 0, 1),
        RolloutContextState(b, 10, 0, 0, 0, 0, 1),
    ]
    assert policy.pick_next_context(states) == a
    assert policy.pick_next_context(states) == b


def test_prefer_current_train_policy_keeps_current_then_switches():
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    a = make_context('a')
    b = make_context('b', run='run_b')
    pa = data_plane.create_partition(a, target_groups=1)
    pb = data_plane.create_partition(b, target_groups=1)
    pa.status = PartitionStatus.TRAIN_READY
    pb.status = PartitionStatus.TRAIN_READY

    policy = PreferCurrentTrainPolicy()
    assert policy.pick_next_partition([pa, pb], current_context=a) == pa
    assert policy.pick_next_partition([pb], current_context=a) == pb


def test_async_rollouter_and_trainer_worker_mvp_flow():
    context = make_context('lora')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    registry = AdapterRegistry()
    registry.register(context)

    class EchoRollout:
        def __call__(self, trajectories, **kwargs):
            out = []
            for traj in trajectories:
                copied = dict(traj)
                copied['messages'] = list(copied.get('messages', [])) + [{'role': 'assistant', 'content': 'ok'}]
                out.append(copied)
            return out

    rollouter = AsyncRollouter(
        data_plane=data_plane,
        adapter_registry=registry,
        staleness_manager=StalenessManager(max_staleness=0, target_groups_per_partition=1),
        rollout=EchoRollout(),
        max_concurrent_groups=1,
    )
    rollouter.add_pending(context, [make_sample(0)])
    result = asyncio.run(rollouter.step())
    assert result is not None
    assert result.component == 'rollouter'
    assert result.kind == 'rollout'
    meta = result.metadata
    assert meta.status == PartitionStatus.ROLLOUT_DONE
    assert meta.partition_id == context.partition_id(0)

    reward_result = RewardWorker(
        data_plane=data_plane,
        reward_registry={'constant': lambda trajectories, **_: [1.0]},
        contexts=[context],
    ).step()
    assert reward_result.kind == 'reward'
    advantage_result = AdvantageWorker(data_plane=data_plane, contexts=[context]).step()
    assert advantage_result.kind == 'advantage'

    received = []

    def train_fn(ctx, partition_id, dataloader):
        assert ctx == context
        assert len(dataloader) == 1
        return {'adapter_revision': '/tmp/adapter-lora-v1'}

    trainer = TrainerWorker(
        data_plane=data_plane,
        adapter_registry=registry,
        scheduler=TrainerScheduler(adapter_registry=registry),
        train_partition_fn=train_fn,
        receive_weights_fn=lambda ctx: received.append(ctx),
    )
    train_result = trainer.step()
    assert train_result is not None
    assert train_result.kind == 'train'
    assert trainer.run_once() is None
    assert received[0].policy_version == 1
    assert received[0].adapter_revision == '/tmp/adapter-lora-v1'
    assert data_plane.list_partitions(context)[0].status == PartitionStatus.CLEARED
    assert registry.get(context).live_partitions == set()


def test_async_rollouter_accumulates_prompt_groups_into_one_train_partition():
    context = make_context('lora')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    registry = AdapterRegistry()
    registry.register(context)

    class EchoRollout:
        def __call__(self, trajectories, **kwargs):
            return [dict(trajectory) for trajectory in trajectories]

    rollouter = AsyncRollouter(
        data_plane=data_plane,
        adapter_registry=registry,
        staleness_manager=StalenessManager(max_staleness=0, target_groups_per_partition=2),
        rollout=EchoRollout(),
        max_concurrent_groups=1,
        target_groups_per_partition=2,
    )
    rollouter.enqueue_prompt_groups(context, [make_sample(0), make_sample(1)])

    first = asyncio.run(rollouter.step()).metadata
    first_status = first.status
    second = asyncio.run(rollouter.step()).metadata

    assert first.partition_id == second.partition_id == context.partition_id(0)
    assert first_status == PartitionStatus.OPEN
    assert second.status == PartitionStatus.ROLLOUT_DONE
    assert data_plane.list_partitions(context)[0].num_rows == 2


def test_prompt_feeder_is_pipeline_source_component():
    context = make_context('lora')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    registry = AdapterRegistry()
    registry.register(context)

    class NoopRollout:
        def __call__(self, trajectories, **kwargs):
            return trajectories

    rollouter = AsyncRollouter(
        data_plane=data_plane,
        adapter_registry=registry,
        staleness_manager=StalenessManager(max_staleness=0, target_groups_per_partition=1),
        rollout=NoopRollout(),
    )
    feeder = PromptFeeder(context=context, dataloader=[[make_sample(0)]], rollouter=rollouter)

    result = feeder.step()

    assert result.component == 'prompt_feeder'
    assert result.kind == 'prompt'
    assert result.count == 1
    assert rollouter.pending_prompt_group_count(context) == 1
    assert feeder.step() is None
    assert feeder.exhausted
