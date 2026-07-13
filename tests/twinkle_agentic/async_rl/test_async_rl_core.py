import asyncio
import json
from uuid import UUID

import pytest
from omegaconf import OmegaConf

from twinkle.data_format import SamplingParams
from twinkle_agentic.async_rl import (
    LoraRuntimeRegistry,
    LoraRuntimeState,
    AdvantageWorker,
    AsyncRollouter,
    WeightedFairRolloutPolicy,
    PromptGroupStatus,
    PromptGroupMeta,
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
from twinkle_agentic.async_rl.grpo_pipeline import (
    _async_train_batch_data_diagnostics,
    _short_math_reward_metrics,
    lora_model_config,
)
from twinkle_agentic.async_rl.metrics import AsyncRLMetricsConfig, JSONLMetricsRecorder, flatten_for_swanlab
from twinkle_agentic.async_rl.tq_utils import columns_to_tq_fields, rows_to_tq_fields
from twinkle_agentic.async_rl.vllm_sampler_tq import TQSamplerRollout

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


def write_rollout_samples(data_plane, group_ref, samples, *, rewards=None):
    group = data_plane.get_prompt_group(group_ref)
    partition = data_plane.get_rollout_partition(group_ref.partition_id)
    sample_keys = []
    sample_fields = []
    sample_tags = []
    for sample_index, raw_sample in enumerate(samples):
        sample = dict(raw_sample)
        if rewards is not None:
            sample['rewards'] = float(rewards[sample_index])
        generation_idx = int(sample.get('generation_idx', sample_index))
        sample_key = f'samples/{group_ref.group_id}/{generation_idx}'
        sample_keys.append(sample_key)
        sample_fields.append({
            field_name: sample[field_name]
            for field_name in (
                'input_ids',
                'labels',
                'attention_mask',
                'logprobs',
                'rewards',
                'advantages',
                'returns',
            )
            if field_name in sample
        })
        tag = group.context.metadata()
        tag.update(dict(sample.get('metadata') or {}))
        for state_field in ('partition_id', 'partition_status', 'group_status', 'num_samples', 'sample_keys'):
            tag.pop(state_field, None)
        tag.update({
            'record_type': 'sample',
            'sample_status': 'success',
            'sample_id': sample.get('sample_id', sample_key),
            'group_id': group.group_id,
            'generation_idx': generation_idx,
            'rollout_policy_version': group.rollout_policy_version,
            'rollout_adapter_path': group.rollout_adapter_path,
            'logprobs_length': len(sample.get('logprobs') or []),
        })
        sample_tags.append(tag)
    data_plane.write_sample_batch(
        partition_id=group_ref.partition_id,
        keys=sample_keys,
        fields=rows_to_tq_fields(sample_fields),
        tags=sample_tags,
    )
    return partition, sample_keys


def batch_group_ids(batch):
    seen = set()
    group_ids = []
    for tag in batch.tags:
        group_id = tag['group_id']
        if group_id not in seen:
            seen.add(group_id)
            group_ids.append(group_id)
    return group_ids


def mark_batch_group_ids(data_plane, batch, status):
    data_plane.mark_groups(
        partition_id=batch.partition_id,
        group_ids=batch_group_ids(batch),
        status=status,
    )


def claim_groups_by_status(data_plane, context, partition_id, ready_status, claim_status, *, max_groups):
    groups = data_plane.list_prompt_groups(context, partition_id=partition_id, statuses=[ready_status])
    return data_plane.claim_prompt_groups(
        groups[:max_groups],
        ready_status=ready_status,
        claim_status=claim_status,
    )


def assert_uuid(value):
    UUID(str(value))


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


def test_jsonl_metrics_recorder_filters_high_frequency_control_events_by_default(tmp_path):
    config = AsyncRLMetricsConfig(
        run_id='run',
        output_dir=str(tmp_path),
    )
    recorder = JSONLMetricsRecorder(config)
    recorder.log_event(event='kv_list', phase='tq', metrics={'keys': 9})
    recorder.log_event(event='pipeline_step', phase='pipeline', metrics={'pipeline_had_work': False})
    recorder.log_event(event='rollout_done', phase='rollout', metrics={'sample_count': 8})
    recorder.close()

    path = tmp_path / 'run' / 'metrics.jsonl'
    events = [json.loads(line)['event'] for line in path.read_text(encoding='utf-8').splitlines()]
    assert events == ['rollout_done']


def test_jsonl_metrics_recorder_can_record_control_events_when_enabled(tmp_path):
    config = AsyncRLMetricsConfig(
        run_id='run',
        output_dir=str(tmp_path),
        record_tq_events=True,
        record_pipeline_steps=True,
    )
    recorder = JSONLMetricsRecorder(config)
    recorder.log_event(event='kv_list', phase='tq', metrics={'keys': 9})
    recorder.log_event(event='pipeline_step', phase='pipeline', metrics={'pipeline_had_work': False})
    recorder.close()

    path = tmp_path / 'run' / 'metrics.jsonl'
    events = [json.loads(line)['event'] for line in path.read_text(encoding='utf-8').splitlines()]
    assert events == ['kv_list', 'pipeline_step']


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
    group = data_plane.list_prompt_groups(context, partition_id=partition.partition_id)[0]
    assert group.status == PromptGroupStatus.ROLLOUT_DONE
    assert_uuid(group.group_id)
    assert group.sample_keys == [f'samples/{group.group_id}/0']
    tags = data_plane.tq.kv_list(partition_id=partition.partition_id)[partition.partition_id]
    assert set(tags) == {'__partition__', f'groups/{group.group_id}', f'samples/{group.group_id}/0'}
    assert tags['__partition__']['partition_status'] == PartitionStatus.CLOSED.value
    group_tag = tags[group.key]
    assert group_tag['sample_keys'] == group.sample_keys
    sample_tag = tags[f'samples/{group.group_id}/0']
    assert sample_tag['sample_status'] == 'success'
    assert sample_tag['group_id'] == group.group_id
    assert sample_tag['generation_idx'] == 0
    assert 'group_status' not in sample_tag
    assert 'partition_status' not in sample_tag

    registry = LoraRuntimeRegistry()
    registry.register(context)
    registry.on_partition_created(context, partition.partition_id)
    adv_worker = AdvantageWorker(data_plane=data_plane, lora_runtime_registry=registry)
    groups = data_plane.list_prompt_groups(
        context,
        partition_id=partition.partition_id,
        statuses=[PromptGroupStatus.ROLLOUT_DONE],
    )
    meta = adv_worker.process_advantage_batch(context, partition_id=partition.partition_id, groups=groups)
    assert meta.status == PartitionStatus.CLOSED
    assert data_plane.list_prompt_groups(context, partition_id=partition.partition_id)[0].status == (
        PromptGroupStatus.ADVANTAGE_DONE)

    data_plane.clear_partition(context, partition.partition_id)
    assert data_plane.list_partitions(context) == []


def test_data_plane_group_claim_consumption_is_exclusive():
    context = make_context('lora')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    partition = data_plane.create_rollout_partition(context, target_groups=2)
    complete_prompt_group(data_plane, context, partition, make_sample(0))
    complete_prompt_group(data_plane, context, partition, make_sample(1))

    first = claim_groups_by_status(
        data_plane,
        context,
        partition.partition_id,
        PromptGroupStatus.ROLLOUT_DONE,
        PromptGroupStatus.ADVANTAGING,
        max_groups=1,
    )
    second = claim_groups_by_status(
        data_plane,
        context,
        partition.partition_id,
        PromptGroupStatus.ROLLOUT_DONE,
        PromptGroupStatus.ADVANTAGING,
        max_groups=1,
    )

    assert len(first.keys) == 1
    assert len(second.keys) == 1
    assert first.keys != second.keys
    assert [group.status for group in data_plane.list_prompt_groups(context, partition_id=partition.partition_id)] == [
        PromptGroupStatus.ADVANTAGING,
        PromptGroupStatus.ADVANTAGING,
    ]


def test_claim_prompt_groups_does_not_cross_partitions():
    context = make_context('lora')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    first_partition = data_plane.create_rollout_partition(context, target_groups=1)
    second_partition = data_plane.create_rollout_partition(context, target_groups=1)
    complete_prompt_group(data_plane, context, first_partition, make_sample(0))
    complete_prompt_group(data_plane, context, second_partition, make_sample(1))

    batch = claim_groups_by_status(
        data_plane,
        context,
        first_partition.partition_id,
        PromptGroupStatus.ROLLOUT_DONE,
        PromptGroupStatus.ADVANTAGING,
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
        groups=data_plane.list_all_prompt_groups(context),
    )
    p1 = data_plane.create_rollout_partition(context, target_groups=1)
    complete_prompt_group(data_plane, context, p1, make_sample(1), policy_version=1)
    assert not manager.can_create_next_rollout_partition(
        context,
        current_policy_version=2,
        groups=data_plane.list_all_prompt_groups(context),
    )


def test_strict_rollouter_blocks_new_partition_until_live_partition_finishes():
    context = make_context('a')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    registry = LoraRuntimeRegistry()
    registry.register(context)
    rollouter = AsyncRollouter(
        data_plane=data_plane,
        lora_runtime_registry=registry,
        staleness_manager=StalenessManager(max_staleness=0),
        rollout=lambda trajectories, **_: trajectories,
    )
    partition = data_plane.create_rollout_partition(context, target_groups=1)
    group_ref = data_plane.create_prompt_group(context, partition, runtime_state=registry.get(context))
    data_plane.update_prompt_group_status(group_ref, PromptGroupStatus.ADVANTAGE_DONE)

    assert not rollouter._can_create_next_rollout_partition(context)

    data_plane.update_partition_status(partition.partition_id, PartitionStatus.TRAIN_DONE)

    assert rollouter._can_create_next_rollout_partition(context)


def test_relaxed_rollouter_limits_created_partitions_by_staleness_window():
    context = make_context('a')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    registry = LoraRuntimeRegistry()
    registry.register(context)
    rollouter = AsyncRollouter(
        data_plane=data_plane,
        lora_runtime_registry=registry,
        staleness_manager=StalenessManager(max_staleness=1),
        rollout=lambda trajectories, **_: trajectories,
    )
    p0 = data_plane.create_rollout_partition(context, target_groups=1)
    g0 = data_plane.create_prompt_group(context, p0, runtime_state=registry.get(context))
    data_plane.update_prompt_group_status(g0, PromptGroupStatus.ADVANTAGE_DONE)
    p1 = data_plane.create_rollout_partition(context, target_groups=1)
    g1 = data_plane.create_prompt_group(context, p1, runtime_state=registry.get(context))
    data_plane.update_prompt_group_status(g1, PromptGroupStatus.ADVANTAGE_DONE)

    assert not rollouter._can_create_next_rollout_partition(context)

    data_plane.update_prompt_group_status(g0, PromptGroupStatus.TRAIN_DONE)
    data_plane.update_partition_status(p0.partition_id, PartitionStatus.TRAIN_DONE)

    assert rollouter._can_create_next_rollout_partition(context)


def test_relaxed_rollouter_uses_partition_step_span_not_live_count():
    context = make_context('a')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    registry = LoraRuntimeRegistry()
    registry.register(context)
    rollouter = AsyncRollouter(
        data_plane=data_plane,
        lora_runtime_registry=registry,
        staleness_manager=StalenessManager(max_staleness=2),
        rollout=lambda trajectories, **_: trajectories,
    )
    p0 = data_plane.create_rollout_partition(context, target_groups=1)
    g0 = data_plane.create_prompt_group(context, p0, runtime_state=registry.get(context))
    data_plane.update_prompt_group_status(g0, PromptGroupStatus.ADVANTAGE_DONE)
    p1 = data_plane.create_rollout_partition(context, target_groups=1)
    g1 = data_plane.create_prompt_group(context, p1, runtime_state=registry.get(context))
    data_plane.update_prompt_group_status(g1, PromptGroupStatus.TRAIN_DONE)
    data_plane.update_partition_status(p1.partition_id, PartitionStatus.TRAIN_DONE)
    p2 = data_plane.create_rollout_partition(context, target_groups=1)
    g2 = data_plane.create_prompt_group(context, p2, runtime_state=registry.get(context))
    data_plane.update_prompt_group_status(g2, PromptGroupStatus.ADVANTAGE_DONE)

    assert not rollouter._can_create_next_rollout_partition(context)


def test_data_plane_prompt_group_records_are_partition_scoped():
    context = make_context('a')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())

    partition = data_plane.create_rollout_partition(context, target_groups=2)
    data_plane.create_prompt_group(context, partition, runtime_state=make_runtime_state(context))
    data_plane.create_prompt_group(context, partition, runtime_state=make_runtime_state(context))

    groups = data_plane.list_prompt_groups(context, partition_id=partition.partition_id)
    assert len(groups) == 2
    assert len({group.group_id for group in groups}) == 2
    for group in groups:
        assert_uuid(group.group_id)


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
    batch = claim_groups_by_status(
        data_plane,
        context,
        partition.partition_id,
        PromptGroupStatus.ROLLOUT_DONE,
        PromptGroupStatus.ADVANTAGING,
        max_groups=2,
    )
    assert batch_group_ids(batch) == [g0.group_id, g1.group_id]
    assert [tag['rollout_policy_version'] for tag in batch.tags] == [0, 1]
    assert len(batch.keys) == 2
    data_plane.write_batch_fields(
        batch,
        {'advantages': [0.0, 0.0], 'returns': [0.0, 0.0]},
    )


def test_mark_groups_writes_group_tags_in_one_batch():
    context = make_context('lora')
    tq_client = FakeTransferQueueClient()
    data_plane = TransferQueueDataPlane(tq_client=tq_client)
    partition = data_plane.create_rollout_partition(context, target_groups=2)
    complete_prompt_group(data_plane, context, partition, make_sample(0))
    complete_prompt_group(data_plane, context, partition, make_sample(1))

    batch = claim_groups_by_status(
        data_plane,
        context,
        partition.partition_id,
        PromptGroupStatus.ROLLOUT_DONE,
        PromptGroupStatus.ADVANTAGING,
        max_groups=2,
    )
    tq_client.kv_batch_put_calls.clear()

    mark_batch_group_ids(data_plane, batch, PromptGroupStatus.ADVANTAGE_DONE)

    assert len(tq_client.kv_batch_put_calls) == 1
    call = tq_client.kv_batch_put_calls[0]
    assert call['partition_id'] == partition.partition_id
    assert call['keys'] == [f'groups/{group_id}' for group_id in batch_group_ids(batch)]
    assert call['has_tags'] is True
    assert call['has_fields'] is False
    assert [group.status for group in data_plane.list_prompt_groups(context, partition_id=partition.partition_id)] == [
        PromptGroupStatus.ADVANTAGE_DONE,
        PromptGroupStatus.ADVANTAGE_DONE,
    ]


def test_data_plane_builds_transformers_train_batch_without_sample_rows():
    context = make_context('lora')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    partition = data_plane.create_rollout_partition(context, target_groups=1)
    complete_prompt_group(data_plane, context, partition, make_sample(0))

    adv_batch = claim_groups_by_status(
        data_plane,
        context,
        partition.partition_id,
        PromptGroupStatus.ROLLOUT_DONE,
        PromptGroupStatus.ADVANTAGING,
        max_groups=1,
    )
    data_plane.write_batch_fields(
        adv_batch,
        {'advantages': [0.25], 'returns': [1.0]},
    )
    mark_batch_group_ids(data_plane, adv_batch, PromptGroupStatus.ADVANTAGE_DONE)
    train_batch_meta = claim_groups_by_status(
        data_plane,
        context,
        partition.partition_id,
        PromptGroupStatus.ADVANTAGE_DONE,
        PromptGroupStatus.TRAINING,
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


def test_data_plane_does_not_feed_scalar_length_to_trainer_inputs():
    context = make_context('lora')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    partition = data_plane.create_rollout_partition(context, target_groups=1)
    sample = make_sample(0)
    sample['length'] = len(sample['input_ids'])
    complete_prompt_group(data_plane, context, partition, sample)

    adv_batch = claim_groups_by_status(
        data_plane,
        context,
        partition.partition_id,
        PromptGroupStatus.ROLLOUT_DONE,
        PromptGroupStatus.ADVANTAGING,
        max_groups=1,
    )
    data_plane.write_batch_fields(
        adv_batch,
        {'advantages': [0.25], 'returns': [1.0]},
    )
    mark_batch_group_ids(data_plane, adv_batch, PromptGroupStatus.ADVANTAGE_DONE)
    train_batch_meta = claim_groups_by_status(
        data_plane,
        context,
        partition.partition_id,
        PromptGroupStatus.ADVANTAGE_DONE,
        PromptGroupStatus.TRAINING,
        max_groups=1,
    )

    train_batch = make_trainer_for_batch_view(data_plane).read_train_batch(train_batch_meta)

    assert 'length' not in train_batch.inputs[0]


def test_data_plane_train_batch_requires_encoded_input_feature_fields():
    context = make_context('lora')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    partition = data_plane.create_rollout_partition(context, target_groups=1)
    sample = make_sample(0)
    sample.pop('input_ids')
    complete_prompt_group(data_plane, context, partition, sample)

    adv_batch = claim_groups_by_status(
        data_plane,
        context,
        partition.partition_id,
        PromptGroupStatus.ROLLOUT_DONE,
        PromptGroupStatus.ADVANTAGING,
        max_groups=1,
    )
    data_plane.write_batch_fields(
        adv_batch,
        {'advantages': [0.0], 'returns': [0.0]},
    )
    mark_batch_group_ids(data_plane, adv_batch, PromptGroupStatus.ADVANTAGE_DONE)
    train_batch_meta = claim_groups_by_status(
        data_plane,
        context,
        partition.partition_id,
        PromptGroupStatus.ADVANTAGE_DONE,
        PromptGroupStatus.TRAINING,
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


def test_async_train_batch_data_diagnostics_accepts_tensor_vectors():
    torch = pytest.importorskip('torch')

    diagnostics = _async_train_batch_data_diagnostics(
        inputs=[{
            'input_ids': torch.tensor([1, 2, 3]),
        }, {
            'input_ids': torch.tensor([1, 2, 3, 4]),
        }],
        rewards=torch.tensor([1.0, 2.0]),
        advantages=torch.tensor([0.5, -0.5]),
        logprobs=[torch.tensor([-0.1, -0.2]), torch.tensor([-0.3])],
    )

    assert diagnostics['tq_reward_mean'] == 1.5
    assert diagnostics['advantage_mean'] == 0.0
    assert diagnostics['logprobs_len_mean'] == 1.5
    assert diagnostics['input_len_mean'] == 3.5


def test_short_math_reward_metrics_accepts_tensor_rewards():
    torch = pytest.importorskip('torch')

    metrics = _short_math_reward_metrics(
        [{'completion_length': 3}, {'completion_length': 5}],
        total_rewards=[torch.tensor(1.0), torch.tensor(2.0)],
    )

    assert metrics['train/total_reward'] == 1.5
    assert metrics['train/total_reward_std'] > 0.0
    assert metrics['train/completion_length'] == 4.0


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
            group_refs = list(kwargs['group_refs'])
            for group_ref, trajectory in zip(group_refs, trajectories):
                _, sample_keys = write_rollout_samples(data_plane, group_ref, [dict(trajectory)], rewards=[1.0])
                data_plane.update_prompt_group_status(
                    group_ref,
                    PromptGroupStatus.ROLLOUT_DONE,
                    sample_keys=sample_keys,
                )
            return {
                'submission_id': 'sub-1',
                'submitted_prompt_groups': len(trajectories),
                'submitted_samples': len(trajectories),
            }

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
            if data_plane.list_all_prompt_groups(context, statuses=[PromptGroupStatus.ROLLOUT_DONE]):
                return result
            await asyncio.sleep(0)
        raise AssertionError('rollout task did not complete')

    result = asyncio.run(drive_rollout())
    assert result is not None
    meta = data_plane.list_partitions(context)[0]
    assert meta.status == PartitionStatus.CLOSED
    assert meta.partition_id == context.partition_id(0)

    advantage_result = AdvantageWorker(data_plane=data_plane, contexts=[context], lora_runtime_registry=registry).step()
    assert advantage_result.kind == 'advantage'

    received = []

    def train_fn(ctx, batch):
        assert ctx == context
        assert len(batch.keys) == 1
        group_ids = batch_group_ids(batch)
        assert len(group_ids) == 1
        assert_uuid(group_ids[0])
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
    assert data_plane.list_partitions(context) == []
    assert registry.get(context).live_partitions == set()


def test_trainer_trains_group_batches_and_syncs_only_when_partition_done():
    context = make_context('lora')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    registry = LoraRuntimeRegistry()
    registry.register(context)
    partition = data_plane.create_rollout_partition(context, target_groups=4)
    registry.on_partition_created(context, partition.partition_id)
    complete_prompt_group(data_plane, context, partition, make_sample(0))
    complete_prompt_group(data_plane, context, partition, make_sample(1))
    complete_prompt_group(data_plane, context, partition, make_sample(2))
    complete_prompt_group(data_plane, context, partition, make_sample(3))
    AdvantageWorker(data_plane=data_plane, contexts=[context], lora_runtime_registry=registry, batch_size=4).step()
    expected_group_ids = [
        group.group_id
        for group in data_plane.list_prompt_groups(
            context,
            partition_id=partition.partition_id,
            statuses=[PromptGroupStatus.ADVANTAGE_DONE],
        )
    ]

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
    assert train_batches == [expected_group_ids[:2], expected_group_ids[2:]]
    assert data_plane.list_partitions(context) == []


def test_trainer_stage_trains_selected_partition_until_done():
    context_a = make_context('a')
    context_b = make_context('b', run='run_b')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    registry = LoraRuntimeRegistry()
    registry.register(context_a)
    registry.register(context_b)

    partition_a = data_plane.create_rollout_partition(context_a, target_groups=2)
    registry.on_partition_created(context_a, partition_a.partition_id)
    complete_prompt_group(data_plane, context_a, partition_a, make_sample(0))
    complete_prompt_group(data_plane, context_a, partition_a, make_sample(1))

    partition_b = data_plane.create_rollout_partition(context_b, target_groups=1)
    registry.on_partition_created(context_b, partition_b.partition_id)
    complete_prompt_group(data_plane, context_b, partition_b, make_sample(2))

    AdvantageWorker(
        data_plane=data_plane,
        contexts=[context_a, context_b],
        lora_runtime_registry=registry,
        batch_size=2,
    ).process_available()

    trained_contexts = []

    def train_fn(ctx, batch):
        trained_contexts.append(ctx.key)
        assert trainer.read_train_batch(batch).sample_count == 1
        return {'adapter_path': f'/tmp/{ctx.adapter_name}-v1'}

    trainer = TrainerWorker(
        data_plane=data_plane,
        lora_runtime_registry=registry,
        scheduler=TrainerScheduler(lora_runtime_registry=registry),
        train_batch_fn=train_fn,
        train_batch_groups=1,
    )

    result = trainer.train_one_partition()

    assert result.train_batches == 2
    assert result.trained_partitions == 1
    assert trained_contexts == [context_a.key, context_a.key]
    assert data_plane.list_partitions(context_a) == []
    assert data_plane.list_prompt_groups(
        context_b,
        partition_id=partition_b.partition_id,
        statuses=[PromptGroupStatus.ADVANTAGE_DONE],
    )


def test_trainer_stage_returns_after_one_partition_even_when_context_has_more_ready():
    context = make_context('lora')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    registry = LoraRuntimeRegistry()
    registry.register(context)

    partition_0 = data_plane.create_rollout_partition(context, target_groups=1)
    registry.on_partition_created(context, partition_0.partition_id)
    complete_prompt_group(data_plane, context, partition_0, make_sample(0))

    partition_1 = data_plane.create_rollout_partition(context, target_groups=1)
    registry.on_partition_created(context, partition_1.partition_id)
    complete_prompt_group(data_plane, context, partition_1, make_sample(1))

    AdvantageWorker(
        data_plane=data_plane,
        contexts=[context],
        lora_runtime_registry=registry,
        batch_size=2,
    ).process_available()

    trained_partitions = []

    def train_fn(ctx, batch):
        trained_partitions.append(batch.partition_id)
        assert trainer.read_train_batch(batch).sample_count == 1
        return {'adapter_path': f'/tmp/{ctx.adapter_name}-v1'}

    trainer = TrainerWorker(
        data_plane=data_plane,
        lora_runtime_registry=registry,
        scheduler=TrainerScheduler(lora_runtime_registry=registry),
        train_batch_fn=train_fn,
        train_batch_groups=1,
    )

    result = trainer.train_one_partition()

    assert result.train_batches == 1
    assert result.trained_partitions == 1
    assert trained_partitions == [partition_0.partition_id]
    assert [partition.partition_id for partition in data_plane.list_partitions(context)] == [partition_1.partition_id]
    assert data_plane.list_prompt_groups(
        context,
        partition_id=partition_1.partition_id,
        statuses=[PromptGroupStatus.ADVANTAGE_DONE],
    )


def test_trainer_mini_batch_size_is_context_level():
    context_a = make_context('a')
    context_b = make_context('b', run='run_b')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    registry = LoraRuntimeRegistry()
    registry.register(context_a)
    registry.register(context_b)

    partition_a = data_plane.create_rollout_partition(context_a, target_groups=1)
    registry.on_partition_created(context_a, partition_a.partition_id)
    complete_prompt_group(data_plane, context_a, partition_a, make_sample(0))
    AdvantageWorker(data_plane=data_plane, contexts=[context_a], lora_runtime_registry=registry, batch_size=1).step()

    partition_b = data_plane.create_rollout_partition(context_b, target_groups=2)
    complete_prompt_group(data_plane, context_b, partition_b, make_sample(1))
    adv_batch_b = claim_groups_by_status(
        data_plane,
        context_b,
        partition_b.partition_id,
        PromptGroupStatus.ROLLOUT_DONE,
        PromptGroupStatus.ADVANTAGING,
        max_groups=1,
    )
    data_plane.write_batch_fields(adv_batch_b, {'advantages': [0.0], 'returns': [0.0]})
    mark_batch_group_ids(data_plane, adv_batch_b, PromptGroupStatus.ADVANTAGE_DONE)

    trained_contexts = []

    def train_fn(ctx, batch):
        trained_contexts.append(ctx.key)
        group_ids = batch_group_ids(batch)
        assert len(group_ids) == 1
        assert_uuid(group_ids[0])
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

    class SubmitOnlyTQRollout:
        def __init__(self):
            self.batch_sizes = []

        def __call__(self, trajectories, **kwargs):
            self.batch_sizes.append(len(trajectories))
            return {
                'submission_id': f'sub-{len(self.batch_sizes)}',
                'submitted_prompt_groups': len(trajectories),
                'submitted_samples': len(trajectories) * int(kwargs['num_generations']),
            }

    rollout = SubmitOnlyTQRollout()
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
        for _ in range(10):
            await rollouter.step()
            if not rollouter.active_tasks:
                return
            await asyncio.sleep(0)
        raise AssertionError('rollout submission task did not complete')

    asyncio.run(drive_rollout())
    meta = data_plane.list_partitions(context)[0]

    assert rollout.batch_sizes == [1, 1]
    assert meta.partition_id == context.partition_id(0)
    assert meta.status == PartitionStatus.CLOSED
    groups = data_plane.list_prompt_groups(context, partition_id=meta.partition_id)
    assert [group.status for group in groups] == [PromptGroupStatus.RUNNING, PromptGroupStatus.RUNNING]
    tags = data_plane.tq.kv_list(partition_id=meta.partition_id)[meta.partition_id]
    assert len([tag for tag in tags.values() if tag.get('record_type') == 'sample']) == 0


def test_async_rollouter_batches_prompt_groups_for_one_sampler_call():
    context = make_context('lora')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    registry = LoraRuntimeRegistry()
    registry.register(context)

    class SubmitOnlyTQRollout:
        def __init__(self):
            self.batch_sizes = []

        def __call__(self, trajectories, **kwargs):
            self.batch_sizes.append(len(trajectories))
            return {
                'submission_id': f'sub-{len(self.batch_sizes)}',
                'submitted_prompt_groups': len(trajectories),
                'submitted_samples': len(trajectories) * int(kwargs['num_generations']),
            }

    rollout = SubmitOnlyTQRollout()
    rollouter = AsyncRollouter(
        data_plane=data_plane,
        lora_runtime_registry=registry,
        staleness_manager=StalenessManager(max_staleness=0, target_groups_per_partition=2),
        rollout=rollout,
        max_concurrency=1,
        target_groups_per_partition=2,
    )
    rollouter.enqueue_prompt_groups(context, [make_sample(0), make_sample(1)])

    async def drive_rollout():
        submit_result = await rollouter.step()
        assert submit_result is not None
        assert submit_result.kind == 'rollout'
        assert submit_result.count == 2
        for _ in range(10):
            result = await rollouter.step()
            if not rollouter.active_tasks:
                return result
            await asyncio.sleep(0)
        raise AssertionError('rollout submission task did not complete')

    result = asyncio.run(drive_rollout())
    meta = data_plane.list_partitions(context)[0]
    tags = data_plane.tq.kv_list(partition_id=meta.partition_id)[meta.partition_id]

    assert result.kind == 'rollout'
    assert result.count == 1
    assert rollout.batch_sizes == [2]
    assert meta.status == PartitionStatus.CLOSED
    groups = data_plane.list_prompt_groups(context, partition_id=meta.partition_id)
    assert [group.status for group in groups] == [PromptGroupStatus.RUNNING, PromptGroupStatus.RUNNING]
    assert len([tag for tag in tags.values() if tag.get('record_type') == 'sample']) == 0


def test_tq_sampler_rollout_builds_async_rl_request():
    context = make_context('lora')
    group_ref = PromptGroupRef(partition_id=context.partition_id(0), group_id='group-1')
    group = PromptGroupMeta(
        context=context,
        partition_id=group_ref.partition_id,
        group_id=group_ref.group_id,
        rollout_policy_version=3,
        rollout_adapter_path='adapter-path',
        status=PromptGroupStatus.RUNNING,
    )

    class FakeSampler:

        def __init__(self):
            self.request = None

        def sample(self, request):
            self.request = request
            return {
                'submission_id': 'sub-1',
                'submitted_prompt_groups': len(request['group_refs']),
                'submitted_samples': len(request['group_refs']) * request['num_generations'],
            }

    sampler = FakeSampler()
    rollout = TQSamplerRollout(
        sampler,
        sampling_params=SamplingParams(max_tokens=16, num_samples=1, logprobs=1),
        default_num_generations=8,
    )

    result = rollout(
        [{'messages': [{'role': 'user', 'content': 'q'}], 'group_id': group_ref.group_id}],
        context=context,
        partition_id=group_ref.partition_id,
        group_refs=[group_ref],
        groups=[group],
        policy_version=3,
        adapter_name=context.adapter_name,
        adapter_path='adapter-path',
    )

    assert result['submitted_prompt_groups'] == 1
    assert result['submitted_samples'] == 8
    assert sampler.request['context'] == context
    assert sampler.request['partition_id'] == group_ref.partition_id
    assert sampler.request['group_refs'] == [group_ref]
    assert sampler.request['groups'] == [group]
    assert sampler.request['num_generations'] == 8
    assert sampler.request['rollout_policy_version'] == 3
    assert sampler.request['adapter_path'] == 'adapter-path'
    assert sampler.request['sampling_params'].num_samples == 1


def test_async_rollouter_tq_mode_submits_without_writing_samples():
    context = make_context('lora')
    data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient())
    registry = LoraRuntimeRegistry()
    registry.register(context)

    class SubmitOnlyTQRollout:
        def __init__(self):
            self.calls = []

        def __call__(self, trajectories, **kwargs):
            self.calls.append((list(trajectories), dict(kwargs)))
            return {
                'submission_id': 'sub-1',
                'submitted_prompt_groups': len(trajectories),
                'submitted_samples': len(trajectories) * int(kwargs['num_generations']),
            }

    rollout = SubmitOnlyTQRollout()
    rollouter = AsyncRollouter(
        data_plane=data_plane,
        lora_runtime_registry=registry,
        staleness_manager=StalenessManager(max_staleness=0, target_groups_per_partition=2),
        rollout=rollout,
        max_concurrency=1,
        target_groups_per_partition=2,
        num_generations=8,
    )
    rollouter.enqueue_prompt_groups(context, [make_sample(0), make_sample(1)])

    async def drive_submission():
        submit_result = await rollouter.step()
        assert submit_result is not None
        assert submit_result.count == 2
        for _ in range(10):
            await rollouter.step()
            if not rollouter.active_tasks:
                return
            await asyncio.sleep(0)
        raise AssertionError('TQ rollout submission task did not finish')

    asyncio.run(drive_submission())

    assert len(rollout.calls) == 1
    trajectories, kwargs = rollout.calls[0]
    assert len(trajectories) == 2
    assert kwargs['context'] == context
    assert kwargs['partition_id'] == context.partition_id(0)
    assert len(kwargs['group_refs']) == 2
    assert len(kwargs['groups']) == 2
    assert kwargs['num_generations'] == 8

    partition = data_plane.list_partitions(context)[0]
    groups = data_plane.list_prompt_groups(context, partition_id=partition.partition_id)
    assert [group.status for group in groups] == [PromptGroupStatus.RUNNING, PromptGroupStatus.RUNNING]
    tags = data_plane.tq.kv_list(partition_id=partition.partition_id)[partition.partition_id]
    assert len([tag for tag in tags.values() if tag.get('record_type') == 'sample']) == 0

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

    assert loader.step() is None
    result = None
    for _ in range(20):
        result = loader.step()
        if result is not None:
            break

    assert result is not None
    assert result.component == 'prompt_loader'
    assert result.kind == 'prompt'
    assert result.count == 1
    assert rollouter.pending_prompt_group_count(context) == 1
    assert loader.step() is None
    assert loader.exhausted
