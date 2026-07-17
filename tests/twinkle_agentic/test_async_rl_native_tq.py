from __future__ import annotations

import asyncio
import inspect
import time

import pytest

from twinkle import DeviceMesh
from twinkle.data_format import SampledSequence, SampleResponse, SamplingParams
from twinkle.infra import _dispatch_args
from twinkle_agentic.async_rl import (AsyncMultiLoraGRPOPipeline, ContextSchedulePolicy, ContextScheduler,
                                      ContextStatus, LoraContext, LoraContextManager, PartitionStatus, ScheduleCandidate,
                                      SchedulerConfig, TQDataPlane, TrainerWorker)
from twinkle_agentic.async_rl.data_plane import build_rollout_group_sample_write
from twinkle_agentic.async_rl.metrics import training_policy_metrics
from twinkle_agentic.async_rl.group_sampler import ContextGRPOGroupNSampler
from twinkle_agentic.async_rl.pipeline import (create_cpu_actor, _sampler_data_parallel_size,
                                                _sequence_parallel_size,
                                                _validate_context_batch_config)
from twinkle_agentic.async_rl.types import (PartitionAdmission, PreparedPartition, PromptGroup, RolloutPolicy)
from twinkle_agentic.async_rl.vllm_sampler_tq import VLLMSamplerTQ, _PromptGroupRolloutStats
from twinkle_agentic.async_rl.workers import RolloutWorker


class LocalActorHandle:
    def __init__(self, target):
        self.target = target

    def __getattr__(self, name):
        method = getattr(self.target, name)

        class RemoteMethod:
            async def remote(_, *args, **kwargs):
                result = method(*args, **kwargs)
                return await result if inspect.isawaitable(result) else result

        return RemoteMethod()


def test_cpu_service_actor_uses_twinkle_ray_mode(monkeypatch):
    import ray

    captured = {}

    class ActorClass:

        @staticmethod
        def remote(*args, **kwargs):
            captured['actor_args'] = args
            captured['actor_kwargs'] = kwargs
            return 'actor'

    def fake_remote(**options):
        captured['options'] = options
        return lambda cls: ActorClass

    monkeypatch.setattr(ray, 'remote', fake_remote)

    assert create_cpu_actor(object, 'value', enabled=True) == 'actor'
    assert captured['options'] == {
        'num_cpus': 1,
        'runtime_env': {
            'env_vars': {
                'TWINKLE_MODE': 'ray'
            }
        },
    }
    assert captured['actor_args'] == ('value', )
    assert captured['actor_kwargs'] == {'enabled': True}


def test_sequence_parallel_size_must_divide_model_gpus():
    assert _sequence_parallel_size(2, 1) == 1
    assert _sequence_parallel_size(2, 2) == 2

    with pytest.raises(ValueError, match='must be divisible'):
        _sequence_parallel_size(2, 3)


class PolicyProvider:

    def __init__(self, policies):
        self.policies = iter(policies)

    def get_rollout_policy(self, _context):
        return next(self.policies)


class GenerationHarness:
    _merge_partial_responses = VLLMSamplerTQ._merge_partial_responses

    def __init__(self, policies, responses):
        self.context_manager = LocalActorHandle(PolicyProvider(policies))
        self.responses = iter(responses)
        self.rollout_max_retries = 1
        self.rollout_retry_delay_s = 0
        self.calls = []
        self.template = type('Template', (), {'decode': staticmethod(lambda tokens: str(tokens))})()

    async def _load_lora_for_policy(self, policy):
        return policy.version

    async def _sample_single(self, feat, sampling_params, *, lora_request, multi_modal_data, logprobs_only):
        self.calls.append((list(feat['input_ids']), sampling_params.max_tokens, lora_request))
        return next(self.responses)


def _context(name: str = 'adapter') -> LoraContext:
    return LoraContext('tenant', f'run_{name}', 'model', name)


def _sample_response(tokens, stop_reason, input_ids):
    return SampleResponse(
        prompt_token_ids=[1, 2],
        sequences=[
            SampledSequence(
                stop_reason=stop_reason,
                tokens=tokens,
                logprobs=[[(token, -.1)] for token in tokens],
                new_input_feature={
                    'input_ids': input_ids,
                    'labels': [-100, -100, *tokens],
                },
            )
        ],
    )


def test_sampler_data_parallel_size_is_derived_from_gpu_and_tp_sizes():
    assert _sampler_data_parallel_size(8, 2) == 4
    assert _sampler_data_parallel_size(1, 1) == 1


def test_sampler_parallelism_rejects_incomplete_tp_group():
    try:
        _sampler_data_parallel_size(3, 2)
    except ValueError as exc:
        assert 'must be divisible' in str(exc)
    else:
        raise AssertionError('expected invalid sampler GPU/TP layout to fail')


def test_context_batch_config_accepts_group_aligned_dp_batches():
    _validate_context_batch_config(
        'tenant/run/adapter',
        rollout_groups=8,
        num_generations=4,
        advantage_groups=2,
        train_groups=2,
        micro_batch_size=2,
        sampler_dp=2,
        model_dp=2,
    )


def test_context_batch_config_rejects_partition_tail_and_undersized_rank_batch():
    try:
        _validate_context_batch_config(
            'tenant/run/adapter',
            rollout_groups=8,
            num_generations=4,
            advantage_groups=3,
            train_groups=2,
            micro_batch_size=1,
            sampler_dp=2,
            model_dp=2,
        )
    except ValueError as exc:
        assert 'advantage.groups_per_batch' in str(exc)
    else:
        raise AssertionError('expected a non-divisible advantage batch to fail')

    try:
        _validate_context_batch_config(
            'tenant/run/adapter',
            rollout_groups=8,
            num_generations=2,
            advantage_groups=1,
            train_groups=1,
            micro_batch_size=2,
            sampler_dp=2,
            model_dp=2,
        )
    except ValueError as exc:
        assert 'per-rank train batch' in str(exc)
    else:
        raise AssertionError('expected an oversized micro batch to fail')


def test_sampler_dp_dispatch_slices_complete_groups_without_duplication():
    mesh = DeviceMesh.from_sizes(world_size=4, dp_size=2, tp_size=2)
    admission = object()
    groups = ['group_0', 'group_1', 'group_2', 'group_3']
    dispatched = _dispatch_args(
        workers=['dp_0', 'dp_1'],
        dispatch='slice_dp',
        execute='all',
        device_mesh=mesh,
        args=(admission, groups, 'sampling_params', False),
        kwargs={},
    )

    assert [worker for worker, _, _ in dispatched] == ['dp_0', 'dp_1']
    assert [args[1] for _, args, _ in dispatched] == [groups[:2], groups[2:]]
    assert all(args[0] is admission for _, args, _ in dispatched)
    assert [group for _, args, _ in dispatched for group in args[1]] == groups


@pytest.mark.parametrize(('dp_size', 'expected_event'), [(1, 'rollout_partition_done'), (2, 'rollout_shard_done')])
def test_sampler_reports_submission_throughput_at_partition_or_shard_scope(dp_size, expected_event):
    context = _context()
    admission = PartitionAdmission(context, context.partition_id(0), 0, 2, 2, 0)
    groups = [
        PromptGroup(context, admission, f'{admission.partition_id}/group_{index}', {}, object())
        for index in range(2)
    ]

    class RolloutMetricsHarness:
        def __init__(self):
            self.device_mesh = DeviceMesh.from_sizes(world_size=dp_size, dp_size=dp_size)
            self.events = []

        async def _run_prompt_group(self, *, group, **_kwargs):
            index = int(group.group_id.rsplit('_', 1)[1])
            lengths = ((10, 20), (30, 40))[index]
            reasons = (('stop', 'length'), ('stop', 'stop'))[index]
            return _PromptGroupRolloutStats(lengths, reasons, (index + 1, index + 1))

        async def _emit(self, event, event_context, partition_id, metrics):
            self.events.append((event, event_context, partition_id, metrics))

    sampler = RolloutMetricsHarness()
    asyncio.run(
        VLLMSamplerTQ._sample_prompt_groups(
            sampler,
            'submission',
            groups,
            SamplingParams(max_tokens=64),
            False,
            time.perf_counter() - 1,
        ))

    event, event_context, partition_id, metrics = sampler.events[-1]
    assert event == expected_event
    assert event_context == context
    assert partition_id == admission.partition_id
    assert metrics['prompt_group_count'] == 2
    assert metrics['sample_count'] == 4
    assert metrics['output_tokens'] == 100
    assert metrics['completion_length_mean'] == 25
    assert metrics['completion_truncated_samples'] == 1
    assert metrics['policy_version_min'] == 1
    assert metrics['policy_version_max'] == 2
    assert metrics['sampler_dp_size'] == dp_size
    assert metrics['output_tokens_per_s'] == pytest.approx(100 / metrics['rollout_latency_s'])


def test_aborted_generation_restarts_from_original_prompt_when_partial_is_disabled():
    context = _context()
    policies = [
        RolloutPolicy(context.key, context.adapter_name, 3, 'adapter-v3'),
        RolloutPolicy(context.key, context.adapter_name, 4, 'adapter-v4'),
    ]
    sampler = GenerationHarness(
        policies,
        [
            _sample_response([7], 'abort', [1, 2, 7]),
            _sample_response([8], 'stop', [1, 2, 8]),
        ],
    )
    generated = asyncio.run(
        VLLMSamplerTQ._generate_sample(
            sampler,
            context,
            {
                'input_ids': [1, 2],
                'labels': [-100, -100]
            },
            SamplingParams(max_tokens=4, logprobs=1),
            multi_modal_data=None,
            logprobs_only=False,
            allow_partial_rollout=False,
        ))

    assert sampler.calls == [([1, 2], 4, 3), ([1, 2], 4, 4)]
    assert generated.response.sequences[0].tokens == [8]
    assert [policy.version for policy in generated.policies] == [4]
    assert generated.retry_count == 1
    assert generated.was_aborted
    assert not generated.resumed_partial_output


def test_aborted_generation_continues_from_partial_tokens_when_enabled():
    context = _context()
    policies = [
        RolloutPolicy(context.key, context.adapter_name, 3, 'adapter-v3'),
        RolloutPolicy(context.key, context.adapter_name, 4, 'adapter-v4'),
    ]
    sampler = GenerationHarness(
        policies,
        [
            _sample_response([7], 'abort', [1, 2, 7]),
            _sample_response([8], 'stop', [1, 2, 7, 8]),
        ],
    )
    generated = asyncio.run(
        VLLMSamplerTQ._generate_sample(
            sampler,
            context,
            {
                'input_ids': [1, 2],
                'labels': [-100, -100]
            },
            SamplingParams(max_tokens=4, logprobs=1),
            multi_modal_data=None,
            logprobs_only=False,
            allow_partial_rollout=True,
        ))

    assert sampler.calls == [([1, 2], 4, 3), ([1, 2, 7], 3, 4)]
    assert generated.response.sequences[0].tokens == [7, 8]
    assert [policy.version for policy in generated.policies] == [3, 4]
    assert generated.retry_count == 1
    assert generated.was_aborted
    assert generated.resumed_partial_output


def test_training_policy_metrics_use_final_version_and_partial_span():
    metrics = training_policy_metrics((
        {
            'final_policy_version': 3,
            'policy_version_span': 1
        },
        {
            'final_policy_version': 4,
            'policy_version_span': 0
        },
    ), train_policy_version=5)

    assert metrics == {
        'policy_version_gap_mean': 1.5,
        'policy_version_gap_p95': 2,
        'policy_version_gap_max': 2,
        'rollout_policy_span_mean': 0.5,
        'rollout_policy_span_max': 1,
    }


def test_training_policy_metrics_reject_future_rollout_version():
    try:
        training_policy_metrics(({
            'final_policy_version': 6,
            'policy_version_span': 0
        }, ), train_policy_version=5)
    except ValueError as exc:
        assert 'older than rollout versions' in str(exc)
    else:
        raise AssertionError('expected a future rollout policy version to fail')


def test_unified_staleness_admission():
    context = _context()
    zero = LoraContextManager(max_staleness=0)
    zero.register_context(context)
    first = zero.request_rollout_partition(context, target_groups=1, num_generations=2)
    assert first is not None
    assert zero.request_rollout_partition(context, target_groups=1, num_generations=2) is None

    one = LoraContextManager(max_staleness=1)
    one.register_context(context)
    assert one.request_rollout_partition(context, target_groups=1, num_generations=2) is not None
    assert one.request_rollout_partition(context, target_groups=1, num_generations=2) is not None
    assert one.request_rollout_partition(context, target_groups=1, num_generations=2) is None


def test_rollout_worker_retains_prefetched_batch_until_admission_succeeds():
    context = _context()

    class AdmissionGate:
        def __init__(self):
            self.blocked = True
            self.attempts = 0
            self.accepted = False

        def is_rollout_admission_closed(self):
            return False

        def context_status(self, _context):
            return ContextStatus.ACTIVE

        def request_rollout_partition(self, _context, *, target_groups, num_generations):
            self.attempts += 1
            if self.blocked or self.accepted:
                return None
            self.accepted = True
            return PartitionAdmission(context, context.partition_id(0), 0, target_groups, num_generations, 0)

    class DataPlane:
        async def prepare_rollout_partition(self, admission, _prompts, sampling_params):
            return PreparedPartition(admission, (), sampling_params)

    class Sampler:
        def __init__(self, loop):
            self.submitted = asyncio.Event()
            self.loop = loop

        def sample(self, _groups, _sampling_params, _allow_partial_rollout):
            self.loop.call_soon_threadsafe(self.submitted.set)

    loaded_batches = []

    def batches():
        for value in (1, 2):
            loaded_batches.append(value)
            yield [{'input_ids': [value]}]

    async def run():
        manager = AdmissionGate()
        sampler = Sampler(asyncio.get_running_loop())
        worker = RolloutWorker(
            context_manager=LocalActorHandle(manager),
            data_plane=DataPlane(),
            sampler=sampler,
            prompt_batches={context.key: batches()},
            rollout_config={
                context.key: {
                    'context': context,
                    'batch_size': 1,
                    'num_generations': 2,
                    'sampling_params': {},
                }
            },
            scheduler=SchedulerConfig(ContextSchedulePolicy.ROUND_ROBIN, 1),
            idle_delay_s=.001,
        )
        await worker.start()
        while manager.attempts == 0:
            await asyncio.sleep(.001)
        prefetched_task = worker._next_batch_tasks[context.key]
        await asyncio.sleep(.01)
        assert worker._next_batch_tasks[context.key] is prefetched_task
        assert loaded_batches == [1]

        manager.blocked = False
        await asyncio.wait_for(sampler.submitted.wait(), timeout=1)
        while loaded_batches == [1]:
            await asyncio.sleep(.001)
        assert loaded_batches == [1, 2]
        await worker.stop()

    asyncio.run(run())


def test_partition_clear_releases_capacity_only_after_publish():
    context = _context()
    manager = LoraContextManager(max_staleness=0)
    manager.register_context(context, adapter_path='initial')
    admission = manager.request_rollout_partition(context, target_groups=1, num_generations=2)
    manager.on_partition_training_started(admission)
    policy = manager.on_partition_trained(admission, adapter_path='v1')
    assert policy.version == 1
    assert manager.request_rollout_partition(context, target_groups=1, num_generations=2) is None
    manager.on_partition_cleared(admission)
    assert manager.request_rollout_partition(context, target_groups=1, num_generations=2) is not None


def test_context_trains_partitions_in_step_order():
    context = _context()
    manager = LoraContextManager(max_staleness=1)
    manager.register_context(context)
    first = manager.request_rollout_partition(context, target_groups=1, num_generations=2)
    second = manager.request_rollout_partition(context, target_groups=1, num_generations=2)

    assert manager.list_trainable_partitions() == [first]
    manager.on_partition_training_started(first)
    assert manager.list_trainable_partitions() == [first]

    try:
        manager.on_partition_training_started(second)
    except RuntimeError as exc:
        assert f'already trains {first.partition_id}' in str(exc)
    else:
        raise AssertionError('expected the next partition to remain blocked')

    manager.on_partition_trained(first, adapter_path='v1')
    manager.on_partition_cleared(first)
    assert manager.list_trainable_partitions() == [second]
    manager.on_partition_training_started(second)


def test_scheduler_supports_round_robin_sticky_and_oldest():
    a, b = _context('a'), _context('b')
    candidates = [ScheduleCandidate(a), ScheduleCandidate(b)]
    round_robin = ContextScheduler(SchedulerConfig(ContextSchedulePolicy.ROUND_ROBIN, 1))
    assert round_robin.choose(candidates).context == a
    round_robin.on_success(candidates[0])
    assert round_robin.choose(candidates).context == b

    sticky = ContextScheduler(SchedulerConfig(ContextSchedulePolicy.STICKY, None))
    sticky.on_success(candidates[1])
    assert sticky.choose(candidates).context == b
    sticky.on_blocked(candidates[1])
    assert sticky.choose(candidates).context == a

    manager = LoraContextManager(max_staleness=2)
    manager.register_context(a)
    manager.register_context(b)
    old = manager.request_rollout_partition(a, target_groups=1, num_generations=2)
    new = manager.request_rollout_partition(b, target_groups=1, num_generations=2)
    oldest = ContextScheduler(SchedulerConfig(ContextSchedulePolicy.OLDEST_PARTITION, 1))
    assert oldest.choose([ScheduleCandidate(b, new), ScheduleCandidate(a, old)]).partition == old


def test_sticky_scheduler_switches_context_at_consecutive_cap():
    a, b = _context('a'), _context('b')
    candidates = [ScheduleCandidate(a), ScheduleCandidate(b)]
    scheduler = ContextScheduler(SchedulerConfig(ContextSchedulePolicy.STICKY, 1))

    first = scheduler.choose(candidates)
    scheduler.on_success(first)

    assert scheduler.choose(candidates).context == b


def test_context_group_sampler_uses_request_generation_count():
    sampler = ContextGRPOGroupNSampler()

    selected, consumed = sampler.sample(
        [0, 1, 4, 5, 6, 7],
        batch_size=4,
        partition_id='train_0',
        task_name='advantage/context',
        n_samples_per_prompt=4,
    )

    assert selected == [4, 5, 6, 7]
    assert consumed == selected

    selected, consumed = sampler.sample(
        [8, 9, 12],
        batch_size=2,
        partition_id='train_1',
        task_name='advantage/context',
        n_samples_per_prompt=2,
    )

    assert selected == [8, 9]
    assert consumed == selected


def test_partition_status_only_represents_normal_lifecycle():
    assert set(PartitionStatus) == {
        PartitionStatus.ROLLOUT,
        PartitionStatus.TRAINING,
        PartitionStatus.PUBLISHED,
    }


def test_context_finishes_after_exhaustion_and_clear():
    context = _context()
    manager = LoraContextManager()
    manager.register_context(context)
    admission = manager.request_rollout_partition(context, target_groups=1, num_generations=2)
    manager.on_dataset_exhausted(context)
    assert not manager.is_run_finished()
    manager.on_partition_training_started(admission)
    manager.on_partition_trained(admission, adapter_path='v1')
    manager.on_partition_cleared(admission)
    assert manager.is_run_finished()


def test_pipeline_fails_fast_when_a_worker_service_fails():
    context = _context()
    manager = LoraContextManager()
    manager.register_context(context)

    class FailedWorker:
        async def start(self):
            return None

        async def stop(self):
            return None

        async def get_service_state(self):
            return {'running': False, 'failure': 'CUDA out of memory'}

        def drain_metrics(self):
            return []

    worker = LocalActorHandle(FailedWorker())
    pipeline = AsyncMultiLoraGRPOPipeline(
        context_manager=LocalActorHandle(manager),
        rollout_worker=worker,
        advantage_worker=worker,
        trainer_worker=worker,
    )

    try:
        asyncio.run(pipeline.run_async())
    except RuntimeError as exc:
        assert 'CUDA out of memory' in str(exc)
    else:
        raise AssertionError('expected worker failure to fail the pipeline')


def test_global_max_steps_limits_admission_and_closes_after_completion():
    first, second = _context('a'), _context('b')
    manager = LoraContextManager(max_staleness=1, max_steps=1)
    manager.register_context(first)
    manager.register_context(second)
    first_admission = manager.request_rollout_partition(first, target_groups=1, num_generations=2)
    assert manager.request_rollout_partition(second, target_groups=1, num_generations=2) is None
    manager.on_partition_training_started(first_admission)
    manager.on_partition_trained(first_admission, adapter_path='v1')
    manager.on_partition_cleared(first_admission)
    assert manager.is_rollout_admission_closed()
    assert manager.is_run_finished()


def test_zero_max_steps_finishes_without_admission():
    context = _context()
    manager = LoraContextManager(max_steps=0)
    manager.register_context(context)
    assert manager.request_rollout_partition(context, target_groups=1, num_generations=2) is None
    assert manager.is_rollout_admission_closed()
    assert manager.is_run_finished()


def test_rollout_sample_tags_use_new_context_descriptor_only():
    context = _context()
    admission = PartitionAdmission(context, context.partition_id(0), 0, 1, 2, 0)
    group = PromptGroup(context, admission, f'{admission.partition_id}/group_0', {}, storage=None)
    fields, tags = build_rollout_group_sample_write(
        group,
        [
            {
                'generation_idx': 0,
                'labels': [-100, 1],
                'logprobs': [-.1],
                'rollout_policy_version': 3,
                'rollout_adapter_path': 'adapter-v3',
            },
            {
                'generation_idx': 1,
                'labels': [-100, 2],
                'logprobs': [-.2],
                'rollout_policy_version': 4,
                'rollout_adapter_path': 'adapter-v4',
            },
        ],
        rewards=[1., 0.],
        expected_num_generations=2,
    )
    assert [row['rewards'] for row in fields] == [1., 0.]
    assert [tag['generation_idx'] for tag in tags] == [0, 1]
    assert all(tag['context_key'] == context.key for tag in tags)
    assert [tag['rollout_policy_version'] for tag in tags] == [3, 4]


def test_data_plane_completes_rollout_with_full_training_trajectory():
    class Metadata:
        def __init__(self):
            self.size = 2
            self.custom_meta = [{}, {}]

        def update_custom_meta(self, updates):
            for tag, update in zip(self.custom_meta, updates):
                tag.update(update)

    class Client:
        def __init__(self):
            self.written = None
            self.calls = []

        async def async_put(self, data, metadata=None, partition_id=None):
            self.calls.append('fields')
            self.written = data
            return metadata

        async def async_set_custom_meta(self, _metadata):
            self.calls.append('tags')
            return None

    context = _context()
    admission = PartitionAdmission(context, context.partition_id(0), 0, 1, 2, 0)
    metadata = Metadata()
    group = PromptGroup(context, admission, f'{admission.partition_id}/group_0', {}, metadata)
    client = Client()
    rows = [{
        'input_ids': [1, 2, token],
        'labels': [-100, -100, token],
        'attention_mask': [1, 1, 1],
        'logprobs': [-.1],
        'generation_idx': generation_idx,
        'rollout_policy_version': 3,
        'rollout_policy_versions': [3],
        'initial_policy_version': 3,
        'final_policy_version': 3,
        'policy_version_span': 0,
        'rollout_adapter_path': 'adapter-v3',
        'completion_length': 1,
    } for generation_idx, token in enumerate((7, 8))]

    asyncio.run(
        TQDataPlane(client).complete_rollout_group(
            group,
            rollout_rows=rows,
            rewards=[1., 0.],
            submission_id='submission',
        ))

    assert set(client.written.keys()) == {'input_ids', 'labels', 'attention_mask', 'logprobs', 'rewards'}
    assert client.calls == ['tags', 'fields']
    assert [tag['rollout_status'] for tag in metadata.custom_meta] == ['ROLLOUT_DONE', 'ROLLOUT_DONE']
    assert [tag['submission_id'] for tag in metadata.custom_meta] == ['submission', 'submission']


def test_data_plane_rejects_rollout_without_complete_model_inputs():
    context = _context()
    admission = PartitionAdmission(context, context.partition_id(0), 0, 1, 1, 0)
    metadata = type('Metadata', (), {'size': 1})()
    group = PromptGroup(context, admission, f'{admission.partition_id}/group_0', {}, metadata)
    row = {
        'input_ids': [1, 2],
        'labels': [-100, 2],
        'logprobs': [-.1],
        'generation_idx': 0,
        'rollout_policy_version': 0,
    }

    try:
        asyncio.run(
            TQDataPlane(object()).complete_rollout_group(
                group,
                rollout_rows=[row],
                rewards=[1.],
                submission_id='submission',
            ))
    except ValueError as exc:
        assert 'attention_mask' in str(exc)
    else:
        raise AssertionError('expected incomplete rollout model fields to fail')


def test_checkpoint_retention_preserves_current_policy_and_history_window():
    context = _context()
    manager = LoraContextManager()
    manager.register_context(context, adapter_path='initial')
    removed = []
    worker = TrainerWorker(
        context_manager=LocalActorHandle(manager),
        data_plane=TQDataPlane(),
        train_fn=lambda _data, _admission: {},
        save_adapter=lambda _admission: 'unused',
        groups_per_batch={context.key: 1},
        scheduler=SchedulerConfig(ContextSchedulePolicy.STICKY, None),
        keep_adapter_versions=1,
        initial_adapter_paths={context.key: 'initial'},
        remove_adapter=removed.append,
    )
    admission = manager.request_rollout_partition(context, target_groups=1, num_generations=2)
    manager.on_partition_training_started(admission)
    manager.on_partition_trained(admission, adapter_path='current')
    manager.on_partition_cleared(admission)
    worker._adapter_history[context.key].append('current')
    async def prune():
        await worker._prune_adapter_history(context)
        await worker.stop()

    asyncio.run(prune())
    assert removed == ['initial']
    assert worker._adapter_history[context.key] == ['current']
    prune_events = [event for event in worker.drain_metrics() if event.event == 'adapter_pruned']
    assert len(prune_events) == 1
    assert prune_events[0].context == context
    assert prune_events[0].metrics['adapter_path'] == 'initial'
    assert prune_events[0].metrics['adapter_prune_latency_s'] >= 0
