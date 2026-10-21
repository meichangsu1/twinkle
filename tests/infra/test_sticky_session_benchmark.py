# Copyright (c) ModelScope Contributors. All rights reserved.
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import json
import pytest


from scripts.benchmark_sticky_session import (
    BenchmarkWorker,
    build_workload,
    check_trial,
    comparisons,
    local_ray_options,
    parse_args,
    prompt_key,
    request_metrics,
    summarize,
    worker_options,
)


def test_workload_is_fixed_append_only_and_unique():
    def encode(text):
        return list(text.encode())
    first = build_workload(encode, 3, 4, 64, 32, 42)
    assert first == build_workload(encode, 3, 4, 64, 32, 42)
    assert len({tuple(episode['prompts'][0]) for episode in first}) == 3
    for episode in first:
        assert [len(tokens) for tokens in episode['prompts']] == [64, 96, 128, 160]
        for previous, current in zip(episode['prompts'], episode['prompts'][1:], strict=False):
            assert current[:len(previous)] == previous


def test_variable_workload_freezes_lengths_budgets_and_waits_with_one_seed():
    options = {'max_tokens': 16, 'prompt_tokens_range': [64, 160], 'append_tokens_range': [32, 64],
               'output_tokens_range': [2, 16], 'tool_wait_ms': [1, 7]}
    workload = build_workload(lambda text: list(text.encode()), 20, 4, 160, 64, 42, **options)
    assert workload == build_workload(lambda text: list(text.encode()), 20, 4, 160, 64, 42, **options)
    assert workload != build_workload(lambda text: list(text.encode()), 20, 4, 160, 64, 43, **options)
    assert len({len(episode['prompts'][0]) for episode in workload}) > 1
    assert len({limit for episode in workload for limit in episode['max_tokens']}) > 1
    for episode in workload:
        assert 64 <= len(episode['prompts'][0]) <= 160
        assert len(episode['max_tokens']) == len(episode['tool_wait_s']) == 4
        assert all(2 <= limit <= 16 for limit in episode['max_tokens'])
        assert episode['tool_wait_s'][0] == 0
        assert all(0.001 <= value <= 0.007 for value in episode['tool_wait_s'][1:])
        for previous, current in zip(episode['prompts'], episode['prompts'][1:], strict=False):
            assert 32 <= len(current) - len(previous) <= 64
            assert current[:len(previous)] == previous


def test_request_metrics_does_not_mix_vllm_clock_domains():
    metrics = SimpleNamespace(arrival_time=1700000000.0, queued_ts=10.0, scheduled_ts=11.0,
                              first_token_ts=11.25, first_token_latency=1.5)
    values = request_metrics(SimpleNamespace(metrics=metrics, num_cached_tokens=128))
    assert values == {'cached_tokens': 128, 'engine_ttft_s': 1.5,
                      'engine_prefill_s': 0.25, 'engine_queue_s': 1.0}


def test_request_metrics_supports_legacy_fields_and_missing_observations():
    metrics = SimpleNamespace(arrival_time=10.0, first_scheduled_time=11.0, first_token_time=11.5)
    assert request_metrics(SimpleNamespace(metrics=metrics)) == {
        'cached_tokens': None, 'engine_ttft_s': 1.5, 'engine_prefill_s': 0.5, 'engine_queue_s': 1.0,
    }
    assert all(value is None for value in request_metrics(SimpleNamespace()).values())


def test_summary_uses_token_weighted_cache_hits_and_only_completed_episode_latencies():
    rows = [
        {'episode_id': 'A', 'turn': 0, 'instance_id': 'one', 'input_tokens': 100, 'output_tokens': 2,
         'cached_tokens': 0, 'engine_call_s': 0.2, 'started_s': 0.0, 'finished_s': 0.5, 'latency_s': 0.5},
        {'episode_id': 'A', 'turn': 1, 'instance_id': 'one', 'input_tokens': 200, 'output_tokens': 2,
         'cached_tokens': 100, 'engine_call_s': 0.2, 'started_s': 0.5, 'finished_s': 1.0,
         'latency_s': 0.5, 'episode_finished': True},
        {'episode_id': 'B', 'turn': 0, 'error': 'failed', 'started_s': 0.0, 'finished_s': 0.1},
    ]
    result = summarize(rows, 2.0, {})
    assert result['cache_hit_fraction'] == pytest.approx(1 / 3)
    assert result['continuation_cache_hit_fraction'] == 0.5
    assert result['episodes_per_s'] == 0.5
    assert result['episode_latency_p95_s'] == 1.0
    assert result['errors'] == 1
    assert result['error_examples'][0]['error'] == 'failed'
    assert result['engine_ttft_s'] == {'count': 0, 'p50': None, 'p95': None}
    rows[0]['cached_tokens'] = None
    assert summarize(rows, 2.0, {})['cache_hit_fraction'] is None


def test_comparisons_exclude_failed_trials():
    assert comparisons([{'group': 'B', 'repeat': 0, 'concurrency': 2, 'errors': 1},
                        {'group': 'D', 'repeat': 0, 'concurrency': 2, 'errors': 0}]) == []


@pytest.mark.parametrize('prompt_length,output_length', [(3, 1), (8192, 63)])
def test_vllm_observer_preserves_generation_and_extracts_request_metrics(monkeypatch, prompt_length, output_length):
    import asyncio

    import ray

    import twinkle.infra as infra
    import twinkle.sampler.vllm_sampler as sampler_module
    from twinkle.template import Template

    tokens = list(range(prompt_length))
    completion = [4] * output_length
    model_limit = 12288 if prompt_length > 128 else 128
    output_limit = output_length + 2

    async def generate(*args, **kwargs):
        yield SimpleNamespace(
            request_id='engine-generated-id', prompt_token_ids=tokens, num_cached_tokens=2,
            outputs=[SimpleNamespace(token_ids=completion)],
            metrics=SimpleNamespace(first_token_latency=0.1, scheduled_ts=10.0, first_token_ts=10.05),
        )

    class FakeSampler:

        def __init__(self, model, engine_args):
            assert engine_args['disable_log_stats'] is False
            self.engine = SimpleNamespace(engine=SimpleNamespace(
                generate=generate,
                vllm_config=SimpleNamespace(cache_config=SimpleNamespace(
                    enable_prefix_caching=engine_args['enable_prefix_caching'])),
            ))

        def set_template(self, template_cls, **kwargs):
            # Exercise the real post-generation concat/check pipeline without downloading a tokenizer.
            self.template = Template.__new__(Template)
            self.template.max_length = kwargs.get('max_length', 8192)
            self.template.truncation_strategy = 'raise'

        def sample(self, inputs, sampling_params):
            assert sampling_params.max_tokens == output_limit
            async def drain():
                outputs = [output async for output in self.engine.engine.generate()]
                assert outputs[-1].outputs[0].token_ids == completion
                feature = self.template.concat_input_feature(inputs[0], completion)
                assert len(feature['input_ids']) == len(feature['labels']) == prompt_length + output_length
                assert feature['input_ids'] == tokens + completion
                return [SimpleNamespace(prompt_token_ids=inputs[0]['input_ids'],
                                        sequences=[SimpleNamespace(tokens=completion)])]

            return asyncio.run(drain())

    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2')
    monkeypatch.setattr(ray, 'get_runtime_context', lambda: SimpleNamespace(
        get_actor_id=lambda: 'replica', get_node_id=lambda: 'node', get_accelerator_ids=lambda: {'GPU': ['2']}))
    monkeypatch.setattr(infra, 'initialize', lambda **kwargs: None)
    monkeypatch.setattr(sampler_module, 'vLLMSampler', FakeSampler)
    worker = BenchmarkWorker({'backend': 'vllm', 'device': 'cuda', 'model': 'unused', 'tensor_parallel_size': 1,
                              'gpu_memory_utilization': 0.5, 'max_model_len': model_limit, 'seed': 0,
                              'max_tokens': 128}, True)
    result = asyncio.run(worker._twinkle_async_sample([{'input_ids': tokens}], max_tokens=output_limit))[0]
    assert result['engine_request_id'] == 'engine-generated-id'
    assert result['cached_tokens'] == 2
    assert result['engine_prefill_s'] == pytest.approx(0.05)
    assert result['output_tokens'] == output_length
    assert result['metrics_source'] == 'vllm_request_output'
    assert worker.observations == {} and worker.active == 0
    assert worker.info()['device_ids'] == ['2']


def test_local_npu_resources_use_visible_device_count(monkeypatch):
    import torch

    import twinkle.utils.platforms.npu as npu_module

    monkeypatch.setattr(npu_module, 'ensure_npu_backend', lambda: None)
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(is_available=lambda: True, device_count=lambda: 4), raising=False)
    config = {'backend': 'vllm', 'device': 'npu', 'replicas': 2}
    assert local_ray_options(config) == {'num_cpus': 2, 'resources': {'NPU': 4}}
    monkeypatch.setattr(torch.npu, 'device_count', lambda: 0)
    with pytest.raises(RuntimeError, match='No visible NPU'):
        local_ray_options(config)
    config['backend'] = 'mock'
    assert local_ray_options(config) == {'num_cpus': 2}


def test_mock_never_requests_accelerators():
    config = {'backend': 'mock', 'device': 'npu', 'concurrency': [4], 'tensor_parallel_size': 2}
    assert worker_options(config) == {'num_cpus': 1, 'num_gpus': 0, 'max_concurrency': 6}


@pytest.mark.parametrize('enabled', [False, True])
def test_prefix_cache_flag_is_explicit_in_actual_engine_arguments(monkeypatch, enabled):
    import inspect

    from twinkle.hub import HubOperation
    from twinkle.sampler.vllm_sampler.vllm_engine import VLLMEngine

    class FakeArgs:
        __signature__ = inspect.Signature([
            inspect.Parameter('enable_prefix_caching', inspect.Parameter.KEYWORD_ONLY, default=True),
        ])

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def create_engine_config(self, **kwargs):
            return self.kwargs

    modules = {
        'vllm.engine.arg_utils': {'AsyncEngineArgs': FakeArgs},
        'vllm.usage.usage_lib': {'UsageContext': SimpleNamespace(OPENAI_API_SERVER='unused')},
        'vllm.v1.engine.async_llm': {
            'AsyncLLM': SimpleNamespace(from_vllm_config=lambda **kwargs: kwargs['vllm_config']),
        },
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(HubOperation, 'download_model', lambda model: model)
    engine = VLLMEngine('unused', enable_prefix_caching=enabled)
    assert engine.engine == {'enable_prefix_caching': enabled}


@pytest.mark.parametrize('options', [
    ['--replicas', '1'], ['--concurrency', '0'], ['--groups', 'B', 'B'],
    ['--max-model-len', '1'], ['--backend', 'vllm'],
    ['--prompt-tokens-range', '128', '64'], ['--output-tokens-range', '0', '8'],
    ['--tool-wait-ms', '-1', '10'], ['--tool-wait-ms', '1', 'nan'],
    ['--prompt-tokens-range', '8000', '8192'],
])
def test_invalid_benchmark_configuration_fails_before_ray_startup(tmp_path, options):
    with pytest.raises(SystemExit):
        parse_args(['--backend', 'mock', '--output-dir', str(tmp_path / 'result'), *options])


@pytest.mark.parametrize('pending', [False, True])
def test_completed_trial_keeps_bindings_but_rejects_inflight_work(tmp_path, pending):
    trial = {'group': 'D', 'repeat': 0, 'concurrency': 2, 'errors': 0, 'error_examples': [],
             'router_after': {'session_count': 4, 'total_inflight': int(pending),
                              'pending_calls': int(pending),
                              'pending_methods': {'sample': 1} if pending else {}}}
    if pending:
        with pytest.raises(RuntimeError, match='total_inflight=1, pending_calls=1'):
            check_trial(trial, tmp_path)
    else:
        check_trial(trial, tmp_path)


@pytest.mark.parametrize('variable', [False, True])
def test_mock_benchmark_exercises_current_dispatcher_without_http(tmp_path, variable):
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / 'benchmark'
    options = (['--prompt-tokens-range', '64', '128', '--append-tokens-range', '32', '48',
                '--output-tokens-range', '1', '8', '--tool-wait-ms', '1', '5'] if variable else [])
    result = subprocess.run([
        sys.executable, str(root / 'scripts/benchmark_sticky_session.py'),
        '--backend', 'mock', '--episodes', '4', '--turns', '3', '--prompt-tokens', '64',
        '--append-tokens', '32', '--max-tokens', '4', '--concurrency', '2', '3',
        '--repeats', '2', '--warmup-requests', '1', '--output-dir', str(output),
        *options,
    ], cwd=root, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / 'summary.json').read_text())
    records = [json.loads(line) for line in (output / 'requests.jsonl').read_text().splitlines()]
    workload = {row['episode_id']: row for row in
                (json.loads(line) for line in (output / 'workload.jsonl').read_text().splitlines())}
    assert summary['synthetic'] is True
    assert summary['routing_kind'] == 'driver_local_continuous_work'
    assert {trial['group'] for trial in summary['trials']} == {'A', 'B', 'D'}
    assert len(summary['trials']) == 12
    assert len(records) == 12 * 4 * 3
    assert all(record['metrics_source'] == 'synthetic_mock' for record in records)
    for record in records:
        episode = workload[record['episode_id']]
        turn = record['turn']
        assert record['max_tokens'] == record['output_tokens'] == episode['max_tokens'][turn]
        assert record['planned_tool_wait_s'] == episode['tool_wait_s'][turn]
        assert record['input_hash'] == prompt_key(episode['prompts'][turn])
        assert record['session_id'] == (
            f"{record['group']}:{record['repeat']}:{record['concurrency']}:{record['episode_id']}"
            if record['session_affinity'] else None)
    for trial in summary['trials']:
        assert trial['errors'] == 0
        assert trial['completed_episodes'] == 4
        assert trial['router_after']['total_inflight'] == trial['router_after']['pending_calls'] == 0
        assert trial['router_after']['pending_methods'] == {}
        assert trial['router_after']['session_count'] == (4 if trial['session_affinity'] else 0)
        if trial['session_affinity']:
            assert trial['worker_switches'] == 0
        if not trial['prefix_cache']:
            assert trial['cache_hit_fraction'] == 0
        if trial['group'] == 'D':
            assert trial['continuation_cache_hit_fraction'] > 0
    assert len(summary['comparisons_D_over_B']) == 2

