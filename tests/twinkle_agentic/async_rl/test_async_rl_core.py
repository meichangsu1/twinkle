import asyncio
import json

import pytest
from omegaconf import OmegaConf

from twinkle_agentic.async_rl import (
    LoraRuntimeRegistry,
    LoraRuntimeState,
    AdvantageWorker,
    AsyncRollouter,
    WeightedFairRolloutPolicy,
    PromptGroupStatus,
    PromptGroupRef,
    PartitionStatus,
    PromptLoader,
    PreferCurrentTrainPolicy,
    StalenessManager,
    TrainerScheduler,
    TrainerWorker,
    TrainBatchCandidate,
    LoraContext,
    TransferQueueDataPlane,
    TransferQueueRuntimeConfig,
    TransformersTrainBatch,
    WorkConservingRolloutPolicy,
)
from twinkle_agentic.async_rl.grpo_pipeline import _async_train_batch_data_diagnostics, lora_model_config
from twinkle_agentic.async_rl.metrics import AsyncRLMetricsConfig, JSONLMetricsRecorder, flatten_for_swanlab
from twinkle_agentic.async_rl.workers import columns_to_tq_fields, rows_to_tq_fields

from .fakes import FakeTransferQueueClient


def make_context(name='a', *, tenant='tenant', run='run', version=0):
    return LoraContext(
        tenant_id=tenant,
        training_run_id=run,
        base_model_id='base',
        adapter_name=name,
        reward_type='constant',
        algorithm='grpo',
    )


def make_sample(i=0):
    return {
        'sample_id': f'sample_{i}',
        'messages': [{'role': 'user', 'content': f'q{i}'}],
        'input_ids': [10 + i, 20 + i],
        'labels': [-100, 20 + i],
        'attention_mask': [1, 1],
        'logprobs': [-0.1],
        'rewards': float(i),
        'group_id': f'g{i}',
        'generation_idx': 0,
    }


def make_runtime_state(context, *, policy_version=0, adapter_path=None):
    return LoraRuntimeState(
        tenant_id=context.tenant_id,
        training_run_id=context.training_run_id,
        adapter_name=context.adapter_name,
        base_model_id=context.base_model_id,
        policy_version=policy_version,
        adapter_path=adapter_path,
    )


def make_rollout_sample_writer(data_plane):
    return AsyncRollouter(
        data_plane=data_plane,
        lora_runtime_registry=LoraRuntimeRegistry(),
        staleness_manager=StalenessManager(),
        rollout=lambda trajectories, **_: trajectories,
    )


def write_rollout_samples(data_plane, group_ref, samples, *, rewards=None):
    return make_rollout_sample_writer(data_plane).write_rollout_samples(group_ref, samples, rewards=rewards)


def batch_group_ids(batch):
    seen = set()
    group_ids = []
    for tag in batch.tags:
        group_id = tag['group_id']
        if group_id not in seen:
            seen.add(group_id)
            group_ids.append(group_id)
    return group_ids


def complete_prompt_group(data_plane, context, partition, sample, *, policy_version=0, adapter_path=None):
    group_ref = data_plane.create_prompt_group(
        context,
        partition,
        runtime_state=make_runtime_state(context, policy_version=policy_version, adapter_path=adapter_path),
    )
    meta, sample_keys = write_rollout_samples(data_plane, group_ref, [sample])
    data_plane.update_prompt_group_status(group_ref, PromptGroupStatus.ROLLOUT_DONE, sample_keys=sample_keys)
    if len(data_plane.list_prompt_groups(context, partition_id=partition.partition_id)) >= partition.target_groups:
        meta = data_plane.update_partition_status(partition.partition_id, PartitionStatus.CLOSED)
    return meta


def make_trainer_for_batch_view(data_plane):
    registry = LoraRuntimeRegistry()
    return TrainerWorker(
        data_plane=data_plane,
        lora_runtime_registry=registry,
        scheduler=TrainerScheduler(lora_runtime_registry=registry),
        train_batch_fn=lambda *_: None,
    )


def test_lora_context_namespace_and_metadata_validation():
    context = make_context('lora')
    assert context.partition_id(3) == 'tenant/run/lora/train_3'
    metadata = context.metadata()
    context.validate_metadata(metadata)


def test_jsonl_metrics_recorder_writes_event_sidecar(tmp_path):
    config = AsyncRLMetricsConfig(
        run_id='run/seed:42',
        mode='async_strict',
        seed=42,
        output_dir=str(tmp_path),
        metadata={'num_loras': 2},
    )
    recorder = JSONLMetricsRecorder(config)
    context = make_context('lora')

    recorder.log_event(
        event='train_done',
        phase='train',
        context=context,
        partition_id=context.partition_id(0),
        policy_version=1,
        metrics={'reward_mean': 0.5},
    )
    recorder.close()

    path = tmp_path / 'run_seed_42' / 'metrics.jsonl'
    events = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    assert events[0]['event'] == 'run_metadata'
    assert events[0]['metrics']['num_loras'] == 2
    assert events[1]['event'] == 'train_done'
    assert events[1]['context_key'] == context.key
    assert events[1]['adapter_name'] == 'lora'
    assert events[1]['partition_id'] == context.partition_id(0)
    assert events[1]['policy_version'] == 1
    assert events[1]['metrics']['reward_mean'] == 0.5


def test_swanlab_flatten_uses_stable_metric_namespaces():
    payload = {
        'elapsed_s': 3.0,
        'phase': 'train',
        'event': 'train_batch_done',
        'adapter_name': 'lora',
        'policy_version': 2,
        'metrics': {
            'reward_mean': 0.7,
            'policy_version_gap_mean': 0.0,
            'untrained_groups': 4,
            'error': 'ignored',
        },
    }

    flat = flatten_for_swanlab(payload)
    assert flat['global/wall_time_s'] == 3.0
    assert flat['context/lora/policy_version'] == 2
    assert flat['context/lora/reward_mean'] == 0.7
    assert flat['staleness/policy_version_gap_mean'] == 0.0
    assert flat['backlog/untrained_groups'] == 4
    assert 'context/lora/error' not in flat
    metadata['adapter_name'] = 'other'
    with pytest.raises(ValueError, match='adapter_name'):
        context.validate_metadata(metadata)


def test_lora_context_model_config_overrides_adapter_fields_only():
    cfg = OmegaConf.create({
        'model': {
            'mixed_precision': 'bf16',
            'lora': {
                'target_modules': 'all-linear',
                'r': 16,
                'lora_alpha': 32,
                'lora_dropout': 0.05,
            },
            'gradient_accumulation_steps': 4,
            'loss': {
                'cls': 'GRPOLoss',
                'epsilon': 0.2,
            },
            'optimizer': {
                'cls': 'AdamW',
                'lr': 5.0e-5,
            },
            'lr_scheduler': {
                'cls': 'LinearWarmupScheduler',
                'num_warmup_steps': 0,
            },
            'processor': {
                'cls': 'InputProcessor',
            },
            'template': {
                'cls': 'Qwen3_5Template',
                'max_length': 4096,
                'enable_thinking': False,
            },
        },
        'lora_contexts': [{
            'adapter_name': 'lora_a',
            'model': {
                'lora': {
                    'r': 8,
                    'lora_alpha': 16,
                },
                'optimizer': {
                    'lr': 3.0e-5,
                },
            },
        }],
    })

    model_cfg = lora_model_config(cfg, cfg.lora_contexts[0])

    assert model_cfg.lora.r == 8
    assert model_cfg.lora.lora_alpha == 16
    assert model_cfg.lora.lora_dropout == 0.05
    assert model_cfg.optimizer.cls == 'AdamW'
    assert model_cfg.optimizer.lr == 3.0e-5
    assert model_cfg.template.max_length == 4096

    cfg.lora_contexts[0].model.mixed_precision = 'fp16'
    with pytest.raises(ValueError, match='resource-level'):
        lora_model_config(cfg, cfg.lora_contexts[0])


def test_data_plane_rollout_reward_advantage_and_clear():
    context = make_context('lora')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    partition = data_plane.create_rollout_partition(context, target_groups=1)

    meta = complete_prompt_group(data_plane, context, partition, make_sample(0))
    assert meta.status == PartitionStatus.CLOSED
    group = data_plane.list_prompt_groups(context)[0]
    assert group.status == PromptGroupStatus.ROLLOUT_DONE
    assert group.sample_keys == ['samples/group_0/0']
    tags = data_plane.tq.kv_list(partition_id=partition.partition_id)[partition.partition_id]
    assert set(tags) == {'__partition__', 'groups/group_0', 'samples/group_0/0'}
    assert tags['__partition__']['partition_status'] == PartitionStatus.CLOSED.value
    group_tag = tags[group.key]
    assert group_tag['sample_keys'] == group.sample_keys
    sample_tag = tags['samples/group_0/0']
    assert sample_tag['sample_status'] == 'success'
    assert sample_tag['group_id'] == 'group_0'
    assert sample_tag['generation_idx'] == 0
    assert 'group_status' not in sample_tag
    assert 'partition_status' not in sample_tag

    adv_worker = AdvantageWorker(data_plane=data_plane)
    meta = adv_worker.process_advantage_batch(context)
    assert meta.status == PartitionStatus.CLOSED
    assert data_plane.list_prompt_groups(context)[0].status == PromptGroupStatus.ADVANTAGE_DONE

    data_plane.clear_partition(context, partition.partition_id)
    assert data_plane.list_partitions(context)[0].status == PartitionStatus.CLEARED


def test_data_plane_group_claim_consumption_is_exclusive():
    context = make_context('lora')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    partition = data_plane.create_rollout_partition(context, target_groups=2)
    complete_prompt_group(data_plane, context, partition, make_sample(0))
    complete_prompt_group(data_plane, context, partition, make_sample(1))

    first = data_plane.claim_prompt_group_samples(
        context=context,
        partition_id=partition.partition_id,
        ready_status=PromptGroupStatus.ROLLOUT_DONE,
        claim_status=PromptGroupStatus.ADVANTAGING,
        max_groups=1,
    )
    second = data_plane.claim_prompt_group_samples(
        context=context,
        partition_id=partition.partition_id,
        ready_status=PromptGroupStatus.ROLLOUT_DONE,
        claim_status=PromptGroupStatus.ADVANTAGING,
        max_groups=1,
    )

    assert len(first.keys) == 1
    assert len(second.keys) == 1
    assert first.keys != second.keys
    assert [group.status for group in data_plane.list_prompt_groups(context)] == [
        PromptGroupStatus.ADVANTAGING,
        PromptGroupStatus.ADVANTAGING,
    ]


def test_claim_prompt_group_samples_does_not_cross_partitions():
    context = make_context('lora')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    first_partition = data_plane.create_rollout_partition(context, target_groups=1)
    second_partition = data_plane.create_rollout_partition(context, target_groups=1)
    complete_prompt_group(data_plane, context, first_partition, make_sample(0))
    complete_prompt_group(data_plane, context, second_partition, make_sample(1))

    batch = data_plane.claim_prompt_group_samples(
        context=context,
        partition_id=first_partition.partition_id,
        ready_status=PromptGroupStatus.ROLLOUT_DONE,
        claim_status=PromptGroupStatus.ADVANTAGING,
        max_groups=1,
    )

    assert len(batch.keys) == 1
    assert batch.partition_id == first_partition.partition_id
    assert data_plane.list_prompt_groups(context, partition_id=second_partition.partition_id)[0].status == (
        PromptGroupStatus.ROLLOUT_DONE)


def test_data_plane_partition_id_is_rollout_step_not_policy_version():
    context = make_context('lora', version=3)
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())

    partition = data_plane.create_rollout_partition(context, target_groups=1)

    assert partition.partition_id == 'tenant/run/lora/train_0'
    next_partition = data_plane.create_rollout_partition(context, target_groups=1)
    assert next_partition.partition_id == 'tenant/run/lora/train_1'


def test_data_plane_rejects_cross_context_append():
    context = make_context('lora')
    other = make_context('other')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    partition = data_plane.create_rollout_partition(context, target_groups=1)
    with pytest.raises(ValueError, match='belongs to'):
        data_plane.create_prompt_group(other, partition, runtime_state=make_runtime_state(other))


def test_lora_adapter_registry_blocks_current_adapter_during_sync_only():
    registry = LoraRuntimeRegistry()
    a = make_context('a')
    b = make_context('b', run='run_b')
    registry.register(a)
    registry.register(b)

    assert registry.can_accept_rollout(a)
    registry.on_weight_sync_started(a)
    assert not registry.can_accept_rollout(a)
    assert registry.can_accept_rollout(b)

    updated = registry.on_weight_sync_finished(a, adapter_path='/tmp/a')
    assert updated.policy_version == 1
    assert updated.adapter_path == '/tmp/a'
    assert registry.can_accept_rollout(a)


def test_staleness_blocks_next_partition_by_live_group_version():
    context = make_context('a')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    manager = StalenessManager(max_staleness=1, target_groups_per_partition=1)

    assert manager.can_create_next_rollout_partition(context, current_policy_version=0, groups=[])
    p0 = data_plane.create_rollout_partition(context, target_groups=1)
    complete_prompt_group(data_plane, context, p0, make_sample(0), policy_version=0)
    assert manager.can_create_next_rollout_partition(
        context,
        current_policy_version=1,
        groups=data_plane.list_prompt_groups(context),
    )
    p1 = data_plane.create_rollout_partition(context, target_groups=1)
    complete_prompt_group(data_plane, context, p1, make_sample(1), policy_version=1)
    assert not manager.can_create_next_rollout_partition(
        context,
        current_policy_version=2,
        groups=data_plane.list_prompt_groups(context),
    )


def test_data_plane_prompt_group_records_are_partition_scoped():
    context = make_context('a')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())

    partition = data_plane.create_rollout_partition(context, target_groups=2)
    data_plane.create_prompt_group(context, partition, runtime_state=make_runtime_state(context))
    data_plane.create_prompt_group(context, partition, runtime_state=make_runtime_state(context))

    groups = data_plane.list_prompt_groups(context, partition_id=partition.partition_id)
    assert [group.group_id for group in groups] == ['group_0', 'group_1']


def test_data_plane_allows_mixed_policy_versions_inside_active_partition():
    context = make_context('lora')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())

    partition = data_plane.create_rollout_partition(context, target_groups=2)
    g0 = data_plane.create_prompt_group(context, partition, runtime_state=make_runtime_state(context))
    _, sample_keys = write_rollout_samples(data_plane, g0, [make_sample(0)])
    data_plane.update_prompt_group_status(g0, PromptGroupStatus.ROLLOUT_DONE, sample_keys=sample_keys)
    g1 = data_plane.create_prompt_group(
        context,
        partition,
        runtime_state=make_runtime_state(context, policy_version=1, adapter_path='/tmp/lora-v1'),
    )
    meta, sample_keys = write_rollout_samples(data_plane, g1, [make_sample(1)])
    data_plane.update_prompt_group_status(g1, PromptGroupStatus.ROLLOUT_DONE, sample_keys=sample_keys)
    meta = data_plane.update_partition_status(partition.partition_id, PartitionStatus.CLOSED)

    assert meta.status == PartitionStatus.CLOSED
    batch = data_plane.claim_prompt_group_samples(
        context=context,
        partition_id=partition.partition_id,
        ready_status=PromptGroupStatus.ROLLOUT_DONE,
        claim_status=PromptGroupStatus.ADVANTAGING,
        max_groups=2,
    )
    assert batch_group_ids(batch) == ['group_0', 'group_1']
    assert [tag['rollout_policy_version'] for tag in batch.tags] == [0, 1]
    assert len(batch.keys) == 2
    data_plane.write_batch_fields(
        batch,
        {'advantages': [0.0, 0.0], 'returns': [0.0, 0.0]},
    )


def test_data_plane_builds_transformers_train_batch_without_sample_rows():
    context = make_context('lora')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    partition = data_plane.create_rollout_partition(context, target_groups=1)
    complete_prompt_group(data_plane, context, partition, make_sample(0))

    adv_batch = data_plane.claim_prompt_group_samples(
        context=context,
        partition_id=partition.partition_id,
        ready_status=PromptGroupStatus.ROLLOUT_DONE,
        claim_status=PromptGroupStatus.ADVANTAGING,
        max_groups=1,
    )
    data_plane.write_batch_fields(
        adv_batch,
        {'advantages': [0.25], 'returns': [1.0]},
    )
    data_plane.mark_batch_groups(adv_batch, PromptGroupStatus.ADVANTAGE_DONE)
    train_batch_meta = data_plane.claim_prompt_group_samples(
        context=context,
        partition_id=partition.partition_id,
        ready_status=PromptGroupStatus.ADVANTAGE_DONE,
        claim_status=PromptGroupStatus.TRAINING,
        max_groups=1,
    )

    train_batch = make_trainer_for_batch_view(data_plane).read_train_batch(train_batch_meta)

    assert isinstance(train_batch, TransformersTrainBatch)
    assert train_batch.inputs == [{
        'input_ids': [10, 20],
        'labels': [-100, 20],
        'attention_mask': [1, 1],
    }]
    assert train_batch.logprobs == [[-0.1]]
    assert train_batch.advantages == [0.25]
    assert 'logprobs' not in train_batch.inputs[0]
    assert 'advantages' not in train_batch.inputs[0]


def test_data_plane_train_batch_requires_encoded_input_feature_fields():
    context = make_context('lora')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    partition = data_plane.create_rollout_partition(context, target_groups=1)
    sample = make_sample(0)
    sample.pop('input_ids')
    complete_prompt_group(data_plane, context, partition, sample)

    adv_batch = data_plane.claim_prompt_group_samples(
        context=context,
        partition_id=partition.partition_id,
        ready_status=PromptGroupStatus.ROLLOUT_DONE,
        claim_status=PromptGroupStatus.ADVANTAGING,
        max_groups=1,
    )
    data_plane.write_batch_fields(
        adv_batch,
        {'advantages': [0.0], 'returns': [0.0]},
    )
    data_plane.mark_batch_groups(adv_batch, PromptGroupStatus.ADVANTAGE_DONE)
    train_batch_meta = data_plane.claim_prompt_group_samples(
        context=context,
        partition_id=partition.partition_id,
        ready_status=PromptGroupStatus.ADVANTAGE_DONE,
        claim_status=PromptGroupStatus.TRAINING,
        max_groups=1,
    )

    with pytest.raises(ValueError, match='input_ids'):
        make_trainer_for_batch_view(data_plane).read_train_batch(train_batch_meta)


def test_async_train_batch_data_diagnostics_accepts_tensor_values():
    torch = pytest.importorskip('torch')

    diagnostics = _async_train_batch_data_diagnostics(
        inputs=[{
            'input_ids': torch.tensor([1, 2, 3]),
        }],
        rewards=torch.tensor([1.0]),
        advantages=torch.tensor([0.5]),
        logprobs=[torch.tensor([-0.1, -0.2])],
    )

    assert diagnostics['tq_reward_mean'] == 1.0
    assert diagnostics['advantage_mean'] == 0.5
    assert diagnostics['logprobs_len_mean'] == 2.0
    assert diagnostics['input_len_mean'] == 3.0


def test_rollout_sample_logprobs_must_match_trainable_labels():
    context = make_context('lora')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    partition = data_plane.create_rollout_partition(context, target_groups=1)
    group_ref = data_plane.create_prompt_group(context, partition, runtime_state=make_runtime_state(context))
    sample = make_sample(0)
    sample['logprobs'] = [-0.1, -0.2]

    with pytest.raises(ValueError, match='logprobs length'):
        write_rollout_samples(data_plane, group_ref, [sample])


def test_rollout_sample_logprobs_must_be_float_list():
    context = make_context('lora')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    partition = data_plane.create_rollout_partition(context, target_groups=1)
    group_ref = data_plane.create_prompt_group(context, partition, runtime_state=make_runtime_state(context))
    sample = make_sample(0)
    sample['logprobs'] = [{'token_id': 20, 'logprob': -0.1}]

    with pytest.raises(TypeError, match='must be a float'):
        write_rollout_samples(data_plane, group_ref, [sample])


def test_tq_field_converters_are_strict():
    with pytest.raises(ValueError, match='fields mismatch'):
        rows_to_tq_fields([{'input_ids': [1], 'labels': [1]}, {'input_ids': [2]}])

    with pytest.raises(ValueError, match='must contain 2 values'):
        columns_to_tq_fields({'advantages': [0.0]}, 2)

    with pytest.raises(TypeError, match='must be a list'):
        columns_to_tq_fields({'advantages': (0.0,)}, 1)


def test_work_conserving_rollout_policy_prefers_less_live_work():
    a = make_context('a')
    b = make_context('b', run='run_b')
    policy = WorkConservingRolloutPolicy()
    from twinkle_agentic.async_rl import RolloutScheduleCandidate

    selected = policy.pick_next_context([
        RolloutScheduleCandidate(
            a, pending_groups=1, in_flight_groups=2, live_partitions=2, active_partitions=1, rollout_capacity=1),
        RolloutScheduleCandidate(
            b, pending_groups=1, in_flight_groups=0, live_partitions=0, active_partitions=0, rollout_capacity=1),
    ])
    assert selected == b


def test_weighted_fair_rollout_policy_alternates_candidates():
    a = make_context('a')
    b = make_context('b', run='run_b')
    policy = WeightedFairRolloutPolicy()
    from twinkle_agentic.async_rl import RolloutScheduleCandidate

    candidates = [
        RolloutScheduleCandidate(a, 10, 0, 0, 0, 1),
        RolloutScheduleCandidate(b, 10, 0, 0, 0, 1),
    ]
    assert policy.pick_next_context(candidates) == a
    assert policy.pick_next_context(candidates) == b


def test_prefer_current_train_policy_keeps_current_then_switches():
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    a = make_context('a')
    b = make_context('b', run='run_b')
    pa = data_plane.create_rollout_partition(a, target_groups=1)
    pb = data_plane.create_rollout_partition(b, target_groups=1)
    pa.status = PartitionStatus.CLOSED
    pb.status = PartitionStatus.CLOSED

    policy = PreferCurrentTrainPolicy()
    ca = TrainBatchCandidate(context=a, partition=pa, available_groups=1)
    cb = TrainBatchCandidate(context=b, partition=pb, available_groups=1)
    assert policy.pick_next_batch([ca, cb], current_context=a).partition == pa
    assert policy.pick_next_batch([cb], current_context=a).partition == pb


def test_async_rollouter_and_trainer_worker_mvp_flow():
    context = make_context('lora')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    registry = LoraRuntimeRegistry()
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
        lora_runtime_registry=registry,
        staleness_manager=StalenessManager(max_staleness=0, target_groups_per_partition=1),
        rollout=EchoRollout(),
        max_concurrency=1,
    )
    rollouter.enqueue_prompt_groups(context, [make_sample(0)])

    async def drive_rollout():
        submit_result = await rollouter.step()
        assert submit_result is not None
        assert submit_result.component == 'rollouter'
        assert submit_result.kind == 'rollout'
        for _ in range(10):
            result = await rollouter.step()
            if data_plane.list_prompt_groups(context, statuses=[PromptGroupStatus.ROLLOUT_DONE]):
                return result
            await asyncio.sleep(0)
        raise AssertionError('rollout task did not complete')

    result = asyncio.run(drive_rollout())
    assert result is not None
    meta = data_plane.list_partitions(context)[0]
    assert meta.status == PartitionStatus.CLOSED
    assert meta.partition_id == context.partition_id(0)

    advantage_result = AdvantageWorker(data_plane=data_plane, contexts=[context]).step()
    assert advantage_result.kind == 'advantage'

    received = []

    def train_fn(ctx, batch):
        assert ctx == context
        assert len(batch.keys) == 1
        assert batch_group_ids(batch) == ['group_0']
        assert trainer.read_train_batch(batch).sample_count == 1
        return {'adapter_path': '/tmp/adapter-lora-v1'}

    trainer = TrainerWorker(
        data_plane=data_plane,
        lora_runtime_registry=registry,
        scheduler=TrainerScheduler(lora_runtime_registry=registry),
        train_batch_fn=train_fn,
        receive_weights_fn=lambda ctx: received.append(ctx),
    )
    train_result = trainer.step()
    assert train_result is not None
    assert train_result.kind == 'train'
    assert trainer.train_next_batch() is None
    assert received[0].policy_version == 1
    assert received[0].adapter_path == '/tmp/adapter-lora-v1'
    assert data_plane.list_partitions(context)[0].status == PartitionStatus.CLEARED
    assert registry.get(context).live_partitions == set()


def test_trainer_trains_group_batches_and_syncs_only_when_partition_done():
    context = make_context('lora')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    registry = LoraRuntimeRegistry()
    registry.register(context)
    partition = data_plane.create_rollout_partition(context, target_groups=4)
    complete_prompt_group(data_plane, context, partition, make_sample(0))
    complete_prompt_group(data_plane, context, partition, make_sample(1))
    complete_prompt_group(data_plane, context, partition, make_sample(2))
    complete_prompt_group(data_plane, context, partition, make_sample(3))
    AdvantageWorker(data_plane=data_plane, contexts=[context], batch_size=4).step()

    train_batches = []
    received = []

    def train_fn(ctx, batch):
        train_batches.append(batch_group_ids(batch))
        assert trainer.read_train_batch(batch).sample_count == 2
        return {'adapter_path': f'/tmp/{ctx.adapter_name}-v1'}

    trainer = TrainerWorker(
        data_plane=data_plane,
        lora_runtime_registry=registry,
        scheduler=TrainerScheduler(lora_runtime_registry=registry),
        train_batch_fn=train_fn,
        receive_weights_fn=lambda ctx: received.append(ctx),
        train_batch_groups=2,
    )

    first = trainer.step()
    assert first.kind == 'train_batch'
    assert received == []
    assert data_plane.list_partitions(context)[0].status == PartitionStatus.CLOSED

    second = trainer.step()
    assert second.kind == 'train'
    assert received[0].policy_version == 1
    assert received[0].adapter_path == '/tmp/lora-v1'
    assert train_batches == [['group_0', 'group_1'], ['group_2', 'group_3']]
    assert data_plane.list_partitions(context)[0].status == PartitionStatus.CLEARED


def test_trainer_mini_batch_size_is_context_level():
    context_a = make_context('a')
    context_b = make_context('b', run='run_b')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    registry = LoraRuntimeRegistry()
    registry.register(context_a)
    registry.register(context_b)

    partition_a = data_plane.create_rollout_partition(context_a, target_groups=1)
    complete_prompt_group(data_plane, context_a, partition_a, make_sample(0))
    AdvantageWorker(data_plane=data_plane, contexts=[context_a], batch_size=1).step()

    partition_b = data_plane.create_rollout_partition(context_b, target_groups=2)
    complete_prompt_group(data_plane, context_b, partition_b, make_sample(1))
    adv_batch_b = data_plane.claim_prompt_group_samples(
        context=context_b,
        partition_id=partition_b.partition_id,
        ready_status=PromptGroupStatus.ROLLOUT_DONE,
        claim_status=PromptGroupStatus.ADVANTAGING,
        max_groups=1,
    )
    data_plane.write_batch_fields(adv_batch_b, {'advantages': [0.0], 'returns': [0.0]})
    data_plane.mark_batch_groups(adv_batch_b, PromptGroupStatus.ADVANTAGE_DONE)

    trained_contexts = []

    def train_fn(ctx, batch):
        trained_contexts.append(ctx.key)
        assert batch_group_ids(batch) == ['group_0']
        return {'adapter_path': f'/tmp/{ctx.adapter_name}-v1'}

    trainer = TrainerWorker(
        data_plane=data_plane,
        lora_runtime_registry=registry,
        scheduler=TrainerScheduler(lora_runtime_registry=registry),
        train_batch_fn=train_fn,
        train_batch_groups=2,
        train_batch_groups_by_context={context_a.key: 1},
    )

    candidates = trainer.list_train_batch_candidates()
    assert [candidate.context.key for candidate in candidates] == [context_a.key]
    result = trainer.step()

    assert result.kind == 'train'
    assert trained_contexts == [context_a.key]


def test_async_rollouter_accumulates_prompt_groups_into_one_rollout_partition():
    context = make_context('lora')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    registry = LoraRuntimeRegistry()
    registry.register(context)

    class EchoRollout:
        def __init__(self):
            self.batch_sizes = []

        def __call__(self, trajectories, **kwargs):
            self.batch_sizes.append(len(trajectories))
            return [dict(trajectory) for trajectory in trajectories]

    rollout = EchoRollout()
    rollouter = AsyncRollouter(
        data_plane=data_plane,
        lora_runtime_registry=registry,
        staleness_manager=StalenessManager(max_staleness=0, target_groups_per_partition=2),
        rollout=rollout,
        max_concurrency=2,
        target_groups_per_partition=2,
    )
    rollouter.enqueue_prompt_groups(context, [make_sample(0), make_sample(1)])

    async def drive_rollout():
        submit_result = await rollouter.step()
        assert submit_result is not None
        assert submit_result.kind == 'rollout'
        assert submit_result.count == 2
        results = []
        for _ in range(10):
            result = await rollouter.step()
            if len(data_plane.list_prompt_groups(context, statuses=[PromptGroupStatus.ROLLOUT_DONE])) == 2:
                return [result]
            await asyncio.sleep(0)
        raise AssertionError('rollout task did not complete')

    results = asyncio.run(drive_rollout())
    meta = data_plane.list_partitions(context)[0]

    assert results
    assert rollout.batch_sizes == [1, 1]
    assert meta.partition_id == context.partition_id(0)
    assert meta.status == PartitionStatus.CLOSED
    tags = data_plane.tq.kv_list(partition_id=meta.partition_id)[meta.partition_id]
    assert len([tag for tag in tags.values() if tag.get('record_type') == 'sample']) == 2


def test_async_rollouter_writes_fast_task_before_slow_tail():
    context = make_context('lora')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    registry = LoraRuntimeRegistry()
    registry.register(context)

    class VariableSpeedRollout:

        async def __call__(self, trajectories, **kwargs):
            delay = float(trajectories[0].get('delay', 0.0))
            await asyncio.sleep(delay)
            return [dict(trajectory) for trajectory in trajectories]

    fast = make_sample(0)
    fast['delay'] = 0.0
    slow = make_sample(1)
    slow['delay'] = 0.05
    rollouter = AsyncRollouter(
        data_plane=data_plane,
        lora_runtime_registry=registry,
        staleness_manager=StalenessManager(max_staleness=1, target_groups_per_partition=2),
        rollout=VariableSpeedRollout(),
        max_concurrency=2,
        target_groups_per_partition=2,
    )
    rollouter.enqueue_prompt_groups(context, [slow, fast])

    async def drive_until_first_rollout():
        submit_result = await rollouter.step()
        assert submit_result.kind == 'rollout'
        assert submit_result.count == 2
        for _ in range(20):
            result = await rollouter.step()
            partition_id = data_plane.list_partitions(context)[0].partition_id
            tags = data_plane.tq.kv_list(partition_id=partition_id)[partition_id]
            if len([tag for tag in tags.values() if tag.get('record_type') == 'sample']) == 1:
                return result
            await asyncio.sleep(0.005)
        raise AssertionError('fast rollout task did not complete first')

    result = asyncio.run(drive_until_first_rollout())

    assert result.kind == 'rollout'
    assert result.count == 1
    partition_id = data_plane.list_partitions(context)[0].partition_id
    tags = data_plane.tq.kv_list(partition_id=partition_id)[partition_id]
    assert len([tag for tag in tags.values() if tag.get('record_type') == 'sample']) == 1
    assert not rollouter.is_idle()


def test_prompt_loader_is_pipeline_source_component():
    context = make_context('lora')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    registry = LoraRuntimeRegistry()
    registry.register(context)

    class NoopRollout:
        def __call__(self, trajectories, **kwargs):
            return trajectories

    rollouter = AsyncRollouter(
        data_plane=data_plane,
        lora_runtime_registry=registry,
        staleness_manager=StalenessManager(max_staleness=0, target_groups_per_partition=1),
        rollout=NoopRollout(),
    )
    loader = PromptLoader(context=context, dataloader=[[make_sample(0)]], rollouter=rollouter)

    result = loader.step()

    assert result.component == 'prompt_loader'
    assert result.kind == 'prompt'
    assert result.count == 1
    assert rollouter.pending_prompt_group_count(context) == 1
    assert loader.step() is None
    assert loader.exhausted
