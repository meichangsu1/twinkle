from __future__ import annotations

import asyncio
import inspect
import json
import time

import pytest

from cookbook.rl.sync_barrier_multi_lora_grpo import SyncBarrierMultiLoraGRPO
from twinkle import DeviceMesh
from twinkle.data_format import SampledSequence, SampleResponse, SamplingParams
from twinkle.infra import _dispatch_args
from twinkle.metric import MetricRecord
from twinkle_agentic.async_rl import (AsyncMultiLoraGRPOPipeline, ContextSchedulePolicy, ContextScheduler,
                                      ContextStatus, LoraContext, LoraContextManager, ScheduleCandidate,
                                      SchedulerConfig, TQDataPlane, TrainerWorker)
from twinkle_agentic.async_rl.data_plane import build_rollout_group_sample_write
from twinkle_agentic.async_rl.metrics import training_policy_metrics
from twinkle_agentic.async_rl.native_tq import ContextGRPOGroupNSampler
from twinkle.model.micro_batch import MicroBatchConfig, plan_micro_batches
from twinkle_agentic.async_rl.pipeline import create_cpu_actor, _reward_for_context, _train_batch
from twinkle_agentic.async_rl.types import (PartitionAdmission, PreparedPartition, PromptGroup, RolloutPolicy)
from twinkle_agentic.async_rl.utils import (
    TrainBatchConfig,
    build_native_fsdp_model_kwargs,
    configure_lora_lr_scheduler,
    resolve_context_learning_rate,
    resolve_context_lora_target_modules,
    resolve_context_loss_config,
    resolve_model_attention_implementation,
    resolve_sequence_parallel_size,
    sampler_data_parallel_size,
    validate_context_batch_config,
)
from twinkle_agentic.async_rl.vllm_sampler_tq import (
    VLLMSamplerTQ,
    _GeneratedSample,
    _PromptGroupRolloutStats,
)
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


def test_unload_lora_paths_does_not_require_pruned_checkpoint(tmp_path):
    removed_paths = []

    class Engine:

        async def remove_loras(self, paths):
            removed_paths.extend(paths)

    class Completed:

        @staticmethod
        def result():
            return None

    sampler = object.__new__(VLLMSamplerTQ)
    sampler.engine = Engine()

    def submit(coro):
        asyncio.run(coro)
        return Completed()

    sampler._submit_in_loop = submit
    pruned_path = tmp_path / 'already-pruned'
    sampler.unload_lora_paths([str(pruned_path)])

    assert removed_paths == [str(pruned_path.resolve())]


def test_sequence_parallel_size_must_divide_model_gpus():
    assert resolve_sequence_parallel_size(2, 1) == 1
    assert resolve_sequence_parallel_size(2, 2) == 2

    with pytest.raises(ValueError, match='must be divisible'):
        resolve_sequence_parallel_size(2, 3)


def test_padding_free_sequence_parallel_requires_flash_attention():
    assert resolve_model_attention_implementation(
        {'attn_implementation': 'flash_attention_2'},
        padding_free=True,
        sequence_parallel_size=2,
    ) == 'flash_attention_2'

    with pytest.raises(ValueError, match='model.attn_implementation'):
        resolve_model_attention_implementation({}, padding_free=True, sequence_parallel_size=2)


def test_train_batch_preserves_position_ids_from_tq():
    class Batch(dict):
        batch_size = (1, )

    class Model:
        inputs = None

        def forward_backward(self, *, inputs, **_kwargs):
            self.inputs = inputs

        def clip_grad_and_step(self, **_kwargs):
            return None

        def calculate_metric(self, **_kwargs):
            return {}

    context = _context()
    admission = PartitionAdmission(context, context.partition_id(0), 0, 1, 1, 0)
    data = Batch({
        'input_ids': [[1, 2]],
        'labels': [[-100, 2]],
        'attention_mask': [[1, 1]],
        'position_ids': [[0, 1]],
        'logprobs': [[-.1]],
        'advantages': [1.],
        'rewards': [1.],
    })
    model = Model()

    _train_batch(model, {context.key: TrainBatchConfig(1, 1)}, data, admission)

    assert model.inputs == [{
        'input_ids': [1, 2],
        'labels': [-100, 2],
        'attention_mask': [1, 1],
        'position_ids': [0, 1],
    }]


def test_train_batch_accumulates_real_micro_batches_before_one_optimizer_step():
    class Batch(dict):
        batch_size = (4, )

    class Model:
        def __init__(self):
            self.calls = []
            self.optimizer_steps = 0

        def forward_backward(self, **kwargs):
            self.calls.append(kwargs)
            return lambda: {
                'micro_batch_count': 4,
                'micro_batch_samples_mean': 1.0,
                'micro_batch_tokens_mean': 1.0,
                'micro_batch_tokens_max': 1,
            }

        def clip_grad_and_step(self, **_kwargs):
            self.optimizer_steps += 1

        def calculate_metric(self, **_kwargs):
            return {}

    context = _context()
    admission = PartitionAdmission(context, context.partition_id(0), 0, 1, 4, 0)
    data = Batch({
        'input_ids': [[index] for index in range(4)],
        'labels': [[index] for index in range(4)],
        'attention_mask': [[1] for _ in range(4)],
        'position_ids': [[0] for _ in range(4)],
        'logprobs': [[-.1] for _ in range(4)],
        'advantages': [1., 2., 3., 4.],
        'rewards': [1., 1., 1., 1.],
    })
    model = Model()

    metrics = _train_batch(
        model,
        {context.key: TrainBatchConfig(4, 1)},
        data,
        admission,
        model_data_parallel_size=1,
    )

    assert [len(call['inputs']) for call in model.calls] == [4]
    assert model.calls[0]['advantages'] == [1., 2., 3., 4.]
    assert model.calls[0]['micro_batch_size'] == 1
    assert model.calls[0]['loss_scale'] == 1.0
    assert model.optimizer_steps == 1
    assert metrics['micro_batch_size_per_rank'] == 1
    assert 'micro_batch_count' not in metrics


def test_dynamic_micro_batch_planner_honors_per_rank_sample_and_token_limits():
    lengths = [10, 9, 8, 7, 4, 3, 2, 1]
    inputs = [{'input_ids': list(range(length))} for length in lengths]
    config = MicroBatchConfig(
        micro_batch_size=3,
        dynamic_batching=True,
        max_tokens_per_micro_batch=18,
    )

    batches = plan_micro_batches(inputs, config, padding_free=False)

    assert sorted(index for batch in batches for index in batch) == list(range(8))
    for batch in batches:
        assert len(batch) <= 3
        padded_tokens = max(lengths[index] for index in batch) * len(batch)
        assert padded_tokens <= 18


def test_sync_training_batch_preserves_position_ids():
    rows = [{
        'input_ids': [1, 2],
        'labels': [-100, 2],
        'attention_mask': [1, 1],
        'position_ids': [0, 1],
        'logprobs': [-.1],
    }]

    batch = SyncBarrierMultiLoraGRPO._training_batch(rows, rewards=[1.], advantages=[0.])

    assert 'position_ids' in batch.keys()
    assert batch['position_ids'][0] == [0, 1]


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
    assert sampler_data_parallel_size(8, 2) == 4
    assert sampler_data_parallel_size(1, 1) == 1


def test_sampler_parallelism_rejects_incomplete_tp_group():
    try:
        sampler_data_parallel_size(3, 2)
    except ValueError as exc:
        assert 'must be divisible' in str(exc)
    else:
        raise AssertionError('expected invalid sampler GPU/TP layout to fail')


def test_lora_lr_scheduler_uses_shared_adapter_config():
    calls = []

    class Model:
        def set_lr_scheduler(self, scheduler_cls, **kwargs):
            calls.append((scheduler_cls, kwargs))

    configure_lora_lr_scheduler(
        Model(),
        'tenant_lora',
        {
            'lr_scheduler': {
                'cls': 'CosineAnnealingLR',
                'T_max': 2000,
                'eta_min': 0.0,
            },
        },
    )

    assert calls == [('CosineAnnealingLR', {
        'adapter_name': 'tenant_lora',
        'T_max': 2000,
        'eta_min': 0.0,
    })]


def test_context_learning_rate_overrides_global_default():
    assert resolve_context_learning_rate({'learning_rate': 5e-6}, {'learning_rate': 1e-6}) == pytest.approx(5e-6)
    assert resolve_context_learning_rate({}, {'learning_rate': 1e-6}) == pytest.approx(1e-6)


def test_context_lora_target_modules_override_global_default():
    defaults = {'target_modules': 'all-linear'}

    assert resolve_context_lora_target_modules({}, defaults) == 'all-linear'
    assert resolve_context_lora_target_modules(
        {'lora': {'target_modules': ['q_proj', 'v_proj']}},
        defaults,
    ) == ['q_proj', 'v_proj']


@pytest.mark.parametrize('value', ['', [], [None], {'q_proj': True}])
def test_context_lora_target_modules_reject_invalid_values(value):
    with pytest.raises(ValueError, match='target_modules'):
        resolve_context_lora_target_modules(
            {'lora': {'target_modules': value}},
            {'target_modules': 'all-linear'},
        )


def test_context_loss_config_overrides_global_defaults():
    loss_cls, loss_kwargs = resolve_context_loss_config(
        {
            'loss': {
                'cls': 'GSPOLoss',
                'normalization': 'token_mean',
            }
        },
        {
            'cls': 'GRPOLoss',
            'epsilon': 0.2,
            'normalization': 'sequence_mean',
        },
    )

    assert loss_cls == 'GSPOLoss'
    assert loss_kwargs == {
        'epsilon': 0.2,
        'normalization': 'token_mean',
    }


def test_context_loss_config_uses_grpo_defaults():
    assert resolve_context_loss_config({}) == (
        'GRPOLoss',
        {
            'epsilon': 0.2,
            'normalization': 'sequence_mean',
        },
    )


def test_context_loss_config_rejects_empty_class_name():
    with pytest.raises(ValueError, match='loss.cls'):
        resolve_context_loss_config({'loss': {'cls': ''}})


@pytest.mark.parametrize('value', [0, -1e-6, float('inf')])
def test_context_learning_rate_rejects_invalid_values(value):
    with pytest.raises(ValueError, match='positive finite'):
        resolve_context_learning_rate({'learning_rate': value}, {'learning_rate': 1e-6})


def test_rl_model_kwargs_enforce_native_fsdp():
    assert build_native_fsdp_model_kwargs({}) == {
        'strategy': 'native_fsdp',
        'fsdp_config': {},
    }
    assert build_native_fsdp_model_kwargs({
        'strategy': 'native_fsdp',
        'fsdp_config': {'reshard_after_forward': False},
    }) == {
        'strategy': 'native_fsdp',
        'fsdp_config': {'reshard_after_forward': False},
    }
    with pytest.raises(ValueError, match='must be native_fsdp'):
        build_native_fsdp_model_kwargs({'strategy': 'accelerate'})


def test_reward_factory_loads_class_and_resolved_kwargs():
    reward = _reward_for_context(
        {
            'class_path': 'twinkle.reward.DAPOMathReward',
            'kwargs': {
                'max_response_length': 8192,
                'overlong_buffer_length': 4096,
                'overlong_penalty_factor': 1.0,
                'score_tail_chars': 300,
            },
        },
        context_key='tenant/run/adapter',
    )

    assert reward.max_response_length == 8192
    assert reward.overlong_buffer_length == 4096


def test_reward_factory_rejects_non_reward_class():
    with pytest.raises(TypeError, match='Reward subclass'):
        _reward_for_context(
            {'class_path': 'collections.Counter'},
            context_key='tenant/run/adapter',
        )


def test_context_batch_config_accepts_group_aligned_dp_batches():
    validate_context_batch_config(
        'tenant/run/adapter',
        rollout_groups=8,
        num_generations=4,
        train=TrainBatchConfig(mini_batch_size=8, micro_batch_size=2),
        sampler_dp=2,
        model_dp=2,
    )


def test_context_batch_config_allows_training_group_to_span_model_dp_ranks():
    validate_context_batch_config(
        'tenant/run/adapter',
        rollout_groups=8,
        num_generations=4,
        train=TrainBatchConfig(mini_batch_size=16, micro_batch_size=2),
        sampler_dp=1,
        model_dp=8,
    )


def test_context_batch_config_rejects_partition_tail_and_undersized_rank_batch():
    try:
        validate_context_batch_config(
            'tenant/run/adapter',
            rollout_groups=6,
            num_generations=4,
            train=TrainBatchConfig(mini_batch_size=6, micro_batch_size=1),
            sampler_dp=2,
            model_dp=2,
        )
    except ValueError as exc:
        assert 'complete prompt groups' in str(exc)
    else:
        raise AssertionError('expected a split prompt group to fail')

    try:
        validate_context_batch_config(
            'tenant/run/adapter',
            rollout_groups=8,
            num_generations=2,
            train=TrainBatchConfig(mini_batch_size=2, micro_batch_size=2),
            sampler_dp=2,
            model_dp=2,
        )
    except ValueError as exc:
        assert 'per-rank train batch' in str(exc)
    else:
        raise AssertionError('expected an oversized micro batch to fail')


def test_context_batch_config_requires_token_limit_for_dynamic_batching():
    with pytest.raises(ValueError, match='max_tokens_per_micro_batch'):
        validate_context_batch_config(
            'tenant/run/adapter',
            rollout_groups=8,
            num_generations=4,
            train=TrainBatchConfig(
                mini_batch_size=8,
                micro_batch_size=2,
                dynamic_batching=True,
            ),
            sampler_dp=1,
            model_dp=1,
        )


def test_sampler_dp_dispatch_slices_complete_groups_without_duplication():
    mesh = DeviceMesh.from_sizes(world_size=4, dp_size=2, tp_size=2)
    groups = ['group_0', 'group_1', 'group_2', 'group_3']
    dispatched = _dispatch_args(
        workers=['dp_0', 'dp_1'],
        dispatch='slice_dp',
        execute='all',
        device_mesh=mesh,
        args=(groups, 'sampling_params', False),
        kwargs={},
    )

    assert [worker for worker, _, _ in dispatched] == ['dp_0', 'dp_1']
    assert [args[0] for _, args, _ in dispatched] == [groups[:2], groups[2:]]
    assert [group for _, args, _ in dispatched for group in args[0]] == groups


@pytest.mark.parametrize(('dp_size', 'expected_scope'), [(1, 'partition'), (2, 'shard')])
def test_sampler_reports_submission_throughput_at_partition_or_shard_scope(dp_size, expected_scope):
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

        def _record_metrics(self, group, values, **kwargs):
            self.events.append((group, values, kwargs))

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

    recorded_group, metrics, record_options = sampler.events[-1]
    assert recorded_group.context == context
    assert recorded_group.partition_id == admission.partition_id
    assert record_options['attributes']['scope'] == expected_scope
    assert metrics['prompt_group_count'] == 2
    assert metrics['sample_count'] == 4
    assert metrics['output_tokens'] == 100
    assert metrics['completion_length_mean'] == 25
    assert metrics['completion_truncated_count'] == 1
    assert metrics['policy_version_min'] == 1
    assert metrics['policy_version_max'] == 2
    assert metrics['sampler_dp_size'] == dp_size
    assert metrics['output_tokens_per_s'] == pytest.approx(100 / metrics['rollout_latency_s'])


def test_sampler_writes_one_atomic_rollout_file_per_prompt_group(tmp_path):
    context = _context()
    admission = PartitionAdmission(context, context.partition_id(3), 3, 1, 2, 0)
    group = PromptGroup(
        context,
        admission,
        f'{admission.partition_id}/group_0',
        {'user_data': [('ground_truth', '"42"')]},
        object(),
    )
    policy = RolloutPolicy(context.key, context.adapter_name, 7, '/tmp/adapter-v7')
    generated = [
        _GeneratedSample(
            SampleResponse(
                sequences=[SampledSequence('stop', [20 + index], decoded=f'completion-{index}')],
                prompt_token_ids=[10, 11],
            ),
            (policy,),
            attempts=1,
            was_aborted=False,
            resumed_partial_output=False,
        )
        for index in range(2)
    ]
    rows = [
        {
            'generation_idx': index,
            'rollout_policy_version': 7,
            'initial_policy_version': 7,
            'final_policy_version': 7,
            'rollout_policy_versions': [7],
            'rollout_adapter_path': '/tmp/adapter-v7',
            'stop_reason': 'stop',
            'logprobs': [-0.1],
        }
        for index in range(2)
    ]

    class Template:
        @staticmethod
        def decode(token_ids, **_kwargs):
            return ' '.join(map(str, token_ids))

    sampler = object.__new__(VLLMSamplerTQ)
    sampler.rollout_output_dir = tmp_path
    sampler.rollout_output_include_token_ids = False
    sampler.template = Template()

    sampler._write_rollout_group('submission-1', group, generated, rows, [1.0, 0.0])
    sampler._write_rollout_group('submission-2', group, generated, rows, [1.0, 0.0])

    output_path = (
        tmp_path
        / context.tenant_id
        / context.training_run_id
        / context.adapter_name
        / 'policy_7'
        / 'train_3-group_0.jsonl'
    )
    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]['submission_id'] == 'submission-2'
    assert records[0]['prompt'] == '10 11'
    assert records[0]['completion'] == '20'
    assert records[0]['ground_truth'] == '42'
    assert records[0]['reward'] == 1.0
    assert records[0]['head_version'] == 7
    assert records[0]['tail_version'] == 7
    assert 'prompt_token_ids' not in records[0]


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
    assert generated.initial_policy.version == 3
    assert generated.final_policy.version == 4
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

        def drain_metric_records(self):
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


def test_pipeline_drains_actor_metric_buffers_when_reporting_is_disabled():
    class BufferedWorker:
        def __init__(self):
            self.drain_count = 0

        def drain_metric_records(self):
            self.drain_count += 1
            return [MetricRecord(stage='train', values={'loss': 1.0})]

    class BufferedSampler:
        def __init__(self):
            self.drain_count = 0

        def drain_metric_records(self):
            self.drain_count += 1
            return [MetricRecord(stage='rollout', values={'sample_count': 1})]

    workers = [BufferedWorker() for _ in range(3)]
    sampler = BufferedSampler()
    pipeline = AsyncMultiLoraGRPOPipeline(
        context_manager=object(),
        rollout_worker=LocalActorHandle(workers[0]),
        advantage_worker=LocalActorHandle(workers[1]),
        trainer_worker=LocalActorHandle(workers[2]),
        sampler=sampler,
        metrics=None,
    )

    asyncio.run(pipeline._drain_metrics())

    assert [worker.drain_count for worker in workers] == [1, 1, 1]
    assert sampler.drain_count == 1


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
    group = PromptGroup(context, admission, f'{admission.partition_id}/group_0', {}, batch_meta=None)
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
        'position_ids': [0, 1, 2],
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

    assert set(client.written.keys()) == {
        'input_ids', 'labels', 'attention_mask', 'position_ids', 'logprobs', 'rewards'
    }
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
        assert 'position_ids' in str(exc)
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
        mini_batch_sizes={context.key: 2},
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
    prune_events = [
        record for record in worker.drain_metric_records()
        if record.stage == 'policy' and record.attributes.get('operation') == 'adapter_prune'
    ]
    assert len(prune_events) == 1
    assert prune_events[0].context_key == context.key
    assert prune_events[0].attributes['adapter_path'] == 'initial'
    assert prune_events[0].values['adapter_prune_latency_s'] >= 0


def test_trainer_periodically_evaluates_published_policy():
    context = _context()
    manager = LoraContextManager()
    manager.register_context(context, adapter_path='initial')
    admission = manager.request_rollout_partition(context, target_groups=1, num_generations=2)
    calls = []

    def evaluate_batch(batch, evaluated_admission, adapter_path, policy_version, sampling_params):
        calls.append((list(batch), evaluated_admission, adapter_path, policy_version, sampling_params))
        return {
            'rewards': [1.0] * len(batch),
            'completion_lengths': [10] * len(batch),
        }

    worker = TrainerWorker(
        context_manager=LocalActorHandle(manager),
        data_plane=TQDataPlane(),
        train_fn=lambda _data, _admission: {},
        save_adapter=lambda _admission: 'unused',
        mini_batch_sizes={context.key: 2},
        scheduler=SchedulerConfig(ContextSchedulePolicy.STICKY, None),
        evaluation_config={
            context.key: {
                'interval': 5,
                'dataset_name': 'validation',
                'prompt_batches': lambda: [[{'input_ids': [1]}], [{'input_ids': [2]}]],
                'sampling_params': 'params',
            }
        },
        evaluate_batch=evaluate_batch,
    )
    worker._optimizer_steps[context.key] = 50

    async def evaluate():
        await worker._evaluate_policy(admission, 'adapter-v4', 4)
        await worker._evaluate_policy(admission, 'adapter-v5', 5)

    asyncio.run(evaluate())
    assert len(calls) == 2
    records = [record for record in worker.drain_metric_records() if record.stage == 'evaluation']
    assert len(records) == 1
    assert records[0].policy_version == 5
    assert records[0].optimizer_step == 50
    assert records[0].values['accuracy'] == 1.0
    assert records[0].values['prompt_count'] == 2
    assert records[0].values['sample_count'] == 2
    assert records[0].values['completion_length'] == 10
