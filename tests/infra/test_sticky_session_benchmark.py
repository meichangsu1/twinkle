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
    run_trial,
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


@pytest.mark.parametrize('fail_turn', [None, 1])
def test_replay_preserves_plan_and_session_but_releases_execution_during_tool_wait(
        monkeypatch, fail_turn, tmp_path, caplog):
    from io import StringIO

    import scripts.benchmark_sticky_session as benchmark
    from twinkle.infra.routing import RequestRouter

    workload = [{'episode_id': 'episode', 'prompts': [[1], [1, 2], [1, 2, 3]],
                 'max_tokens': [2, 4, 8], 'tool_wait_s': [0, 0.002, 0.004]}]
    clock = [100.0]
    monkeypatch.setattr(benchmark.time, 'perf_counter', lambda: clock[0])
    def replay(group):
        router = RequestRouter({'one': None, 'two': None})
        calls, waits, session_ids = [], [], []

        class Sampler:

            def status(self):
                return router.get_status()

            def sample(self, inputs, session_ids=None, max_tokens=None):
                sid = session_ids[0] if session_ids else None
                lease = router.acquire(sid)
                try:
                    assert set(inputs[0]) == {'input_ids'}
                    calls.append((list(inputs[0]['input_ids']), max_tokens, sid))
                    clock[0] += 0.01
                    if len(calls) - 1 == fail_turn:
                        raise RuntimeError('planned failure')
                    return [{'instance_id': lease.instance_id, 'input_tokens': len(inputs[0]['input_ids']),
                             'output_tokens': max_tokens, 'cached_tokens': 0, 'engine_call_s': 0.01}]
                finally:
                    router.release(lease)

            def release_sessions(self, ids):
                session_ids.extend(ids)
                router.release_sessions(ids)

        def wait(seconds):
            status = router.get_status()
            assert status['active_leases'] == status['total_inflight'] == 0
            assert status['session_count'] == (1 if group == 'D' else 0)
            waits.append(seconds)
            clock[0] += seconds

        monkeypatch.setattr(benchmark.time, 'sleep', wait)
        output = StringIO()
        result = run_trial(Sampler(), workload, group, 0, 1, output)
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        assert result['errors'] == (0 if fail_turn is None else 1)
        assert result['router_after']['active_leases'] == result['router_after']['session_count'] == 0
        if group == 'B':
            assert all(call[2] is None for call in calls)
        else:
            assert all(call[2] == session_ids[0] for call in calls)
        assert [row['max_tokens'] for row in rows] == workload[0]['max_tokens'][:len(calls)]
        assert result['planned_tool_wait_s'] == pytest.approx(sum(waits))
        if fail_turn is None:
            check_trial(result, tmp_path)
            assert result['episode_latency_p95_s'] == pytest.approx(0.036)
            assert result['latency_s']['p95'] == pytest.approx(0.01)
            assert result['error_examples'] == []
        else:
            assert result['error_examples'] == [{
                'episode_id': 'episode', 'turn': 1, 'input_tokens': 2, 'max_tokens': 4,
                'error_type': 'RuntimeError', 'error': 'RuntimeError: planned failure',
            }]
            with pytest.raises(RuntimeError, match='sampling_errors=1, active_leases=0, session_count=0') as error:
                check_trial(result, tmp_path)
            assert 'episode=episode, turn=1, input_tokens=2, max_tokens=4' in str(error.value)
            assert 'planned failure' in str(error.value)
            assert 'Sampling failed:' in caplog.text and 'planned failure' in caplog.text
        return [(tokens, budget) for tokens, budget, _ in calls], waits

    assert replay('B') == replay('D')


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


@pytest.mark.parametrize('leases,sessions', [(1, 0), (0, 1)])
def test_trial_error_distinguishes_routing_leak_from_sampling_failure(tmp_path, leases, sessions):
    result = {'group': 'D', 'repeat': 0, 'concurrency': 8, 'errors': 0, 'error_examples': [],
              'router_after': {'active_leases': leases, 'session_count': sessions}}
    with pytest.raises(RuntimeError) as error:
        check_trial(result, tmp_path)
    assert f'sampling_errors=0, active_leases={leases}, session_count={sessions}' in str(error.value)
    assert 'First sampling error' not in str(error.value)


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


@pytest.mark.parametrize('tp', [1, 2])
def test_ray_npu_replicas_receive_disjoint_devices_without_cuda(tmp_path, tp):
    # Real Ray allocation with simulated NPU resources: no torch_npu/model/hardware.
    script = tmp_path / 'npu_allocation.py'
    script.write_text('''
import os
import sys
import ray
from scripts.benchmark_sticky_session import worker_options, worker_device_info

tp = int(sys.argv[1])
visible_ids = [str(2 * index + 1) for index in range(2 * tp)]
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = ','.join(visible_ids)
os.environ['RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES'] = '1'
config = {'backend': 'vllm', 'device': 'npu', 'tensor_parallel_size': tp, 'concurrency': [2]}
options = worker_options(config)
assert options['num_gpus'] == 0 and options['resources'] == {'NPU': tp}
ray.init(num_cpus=2, resources={'NPU': 2 * tp}, include_dashboard=False, log_to_driver=False)
try:
    @ray.remote
    class Probe:
        def info(self):
            assigned = ray.get_runtime_context().get_accelerator_ids()['NPU']
            assert os.environ['ASCEND_RT_VISIBLE_DEVICES'] == ','.join(assigned)
            info = worker_device_info(config)
            assert os.environ['ASCEND_RT_VISIBLE_DEVICES'] == ','.join(info['device_ids'])
            return info

    actors = [Probe.options(**options).remote() for _ in range(2)]
    infos = ray.get([actor.info.remote() for actor in actors], timeout=30)
    assert all(info['device'] == 'npu' and len(info['device_ids']) == tp for info in infos)
    assert set(infos[0]['device_ids'] + infos[1]['device_ids']) == set(visible_ids)
    assert set(infos[0]['device_ids']).isdisjoint(infos[1]['device_ids'])
finally:
    ray.shutdown()
''')
    root = Path(__file__).resolve().parents[2]
    # Pass the repository path explicitly because this script lives outside it.
    result = subprocess.run([sys.executable, '-c',
                             f'import sys; sys.path.insert(0, {str(root)!r}); '
                             f'exec(compile(open({str(script)!r}).read(), {str(script)!r}, "exec"))', str(tp)],
                            cwd=root, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


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


@pytest.mark.parametrize('variable', [False, True])
def test_mock_benchmark_runs_full_matrix_and_releases_routes(tmp_path, variable):
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
    assert len(summary['trials']) == 16
    assert len(records) == 16 * 4 * 3
    assert all(record['metrics_source'] == 'synthetic_mock' for record in records)
    for record in records:
        episode = workload[record['episode_id']]
        turn = record['turn']
        assert record['max_tokens'] == record['output_tokens'] == episode['max_tokens'][turn]
        assert record['planned_tool_wait_s'] == episode['tool_wait_s'][turn]
        assert record['input_hash'] == prompt_key(episode['prompts'][turn])
    for trial in summary['trials']:
        assert trial['errors'] == 0
        assert trial['completed_episodes'] == 4
        assert trial['router_after']['total_inflight'] == trial['router_after']['session_count'] == 0
        assert trial['router_after']['active_leases'] == 0
        if trial['session_affinity']:
            assert trial['worker_switches'] == 0
        if not trial['prefix_cache']:
            assert trial['cache_hit_fraction'] == 0
        if trial['group'] == 'D':
            assert trial['continuation_cache_hit_fraction'] > 0
    assert len(summary['comparisons_D_over_B']) == 2
