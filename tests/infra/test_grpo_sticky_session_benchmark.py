# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import copy
import subprocess
import sys
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import json
import pytest

from scripts.benchmark_grpo_sticky_session import (
    expand_groups,
    observe_engine,
    paired_comparisons,
    parse_args,
    prompt_key,
    read_inputs,
    run_experiment,
    summarize_observations,
    summarize_trial,
    train_batches,
)


def config(**overrides):
    return {'group': 'D', 'seed': 42, 'steps': 2, 'warmup_steps': 1, 'batch_size': 2,
            'num_generations': 2, 'mini_batch_size': 2, 'micro_batch_size': 1,
            'plan_hash': 'shared-plan', 'max_prompt_tokens': 32, **overrides}


def inputs(count=4):
    return [{'input_ids': [index, 100], 'user_data': [['ground_truth', '1']]} for index in range(count)]


@pytest.mark.parametrize('affinity', [False, True])
def test_group_ids_are_shared_only_within_each_prompt_and_each_call(affinity):
    prompts = inputs(2)
    before = copy.deepcopy(prompts)
    expanded, sessions = expand_groups(prompts, 4, affinity)
    assert expanded == [prompts[0]] * 4 + [prompts[1]] * 4
    assert prompts == before
    _, next_sessions = expand_groups(prompts, 4, affinity)
    if affinity:
        assert sessions[:4] == [sessions[0]] * 4
        assert sessions[4:] == [sessions[4]] * 4
        assert len(set(sessions)) == 2
        assert set(sessions).isdisjoint(next_sessions)
    else:
        assert sessions is next_sessions is None


def test_native_observer_preserves_duplicate_prompts_and_missing_metrics():
    class Engine:
        async def generate(self, request_id):
            yield SimpleNamespace(request_id=request_id, prompt_token_ids=[1, 2],
                                  outputs=[SimpleNamespace(token_ids=[3])], num_cached_tokens=None)
            yield SimpleNamespace(request_id=request_id, prompt_token_ids=[1, 2],
                                  outputs=[SimpleNamespace(token_ids=[3, 4])], num_cached_tokens=None)

    engine, observations = Engine(), []
    observe_engine(engine, observations, 'replica-1')

    async def run():
        await asyncio.gather(*[drain(engine.generate(str(index))) for index in range(8)])

    async def drain(generator):
        return [item async for item in generator]

    asyncio.run(run())
    assert len(observations) == 8
    assert {row['engine_request_id'] for row in observations} == {str(index) for index in range(8)}
    assert all(row['cached_tokens'] is None and row['output_tokens'] == 2 for row in observations)


def test_cache_weighting_replica_counts_and_missing_observations():
    prompts = [{'input_ids': [1] * 10}, {'input_ids': [2] * 30}]
    rows = [{'input_hash': prompt_key(prompt['input_ids']), 'input_tokens': len(prompt['input_ids']),
             'output_tokens': 2, 'instance_id': 'one' if index == 0 else 'two',
             'cached_tokens': len(prompt['input_ids']) // 2}
            for prompt in prompts for index in range(2)]
    result = summarize_observations(rows, prompts, 2)
    assert result['cache_hit_fraction'] == 0.5
    assert result['group_replica_counts'] == [2, 2]
    assert result['cache_metric_coverage'] == 1
    rows[0]['cached_tokens'] = None
    result = summarize_observations(rows, prompts, 2)
    assert result['cache_hit_fraction'] is result['cached_tokens'] is None
    assert result['cache_metric_coverage'] == 0.75
    with pytest.raises(RuntimeError, match='multiset'):
        summarize_observations(rows[:-1], prompts, 2)


@pytest.mark.parametrize('failure', [None, 'sample', 'score', 'train'])
@pytest.mark.parametrize('group', ['A', 'B', 'D'])
def test_training_updates_use_production_group_advantages_and_release_sessions(group, failure):
    import torch

    from twinkle.advantage import GRPOAdvantage
    from twinkle.infra.routing import RequestRouter

    events, observed, session_batches = [], [], []
    router = RequestRouter({'one': None, 'two': None}, max_sessions=2)
    before = inputs()
    plan = copy.deepcopy(before)

    class Sampler:
        def reset_prefix_cache(self):
            events.append('reset')

        def sample(self, expanded, params, session_ids=None):
            assert events[-2:] == ['sync', 'reset']
            events.append('sample')
            session_batches.append(session_ids)
            leases = router.acquire_batch([(sid, 1) for sid in (session_ids or [None] * len(expanded))])
            try:
                if failure == 'sample':
                    raise RuntimeError('sample failed')
                responses = []
                for index, (prompt, lease) in enumerate(zip(expanded, leases, strict=True)):
                    observed.append({'input_hash': prompt_key(prompt['input_ids']),
                                     'instance_id': lease.instance_id, 'input_tokens': len(prompt['input_ids']),
                                     'output_tokens': 1, 'cached_tokens': 0 if group == 'A' else 1})
                    feature = {**prompt, 'candidate': index % 2}
                    sequence = SimpleNamespace(tokens=[index % 2], logprobs=[[(index % 2, -0.5)]],
                                               new_input_feature=feature, stop_reason='stop')
                    responses.append(SimpleNamespace(sequences=[sequence]))
                return responses
            finally:
                router.release_batch([lease.lease_id for lease in leases])

        def release_sessions(self, sessions):
            events.append('release')
            router.release_sessions(sessions)

        def take_observations(self):
            result = observed[:]
            observed.clear()
            return result

    # Actual PyTorch backward + optimizer updates, with a tiny CPU policy.
    class Model:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.tensor(0.0))
            self.optimizer = torch.optim.SGD([self.weight], lr=0.1)
            self.update_count = 0

        def forward_backward(self, *, inputs, old_logps, advantages, micro_batch_size):
            assert old_logps == [[-0.5]] * 2
            assert advantages[0] < 0 < advantages[1]
            assert [feature['candidate'] for feature in inputs] == [0, 1]
            assert micro_batch_size == 1
            if failure == 'train':
                raise RuntimeError('train failed')
            signs = torch.tensor([-1.0, 1.0])
            logps = torch.log_softmax(self.weight * signs, dim=0)
            loss = -(logps * torch.tensor(advantages)).mean()
            loss.backward()

        def clip_grad_and_step(self):
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.update_count += 1

        def synchronize(self):
            events.append('synchronize')

        def calculate_metric(self, is_training):
            assert is_training
            return {'weight': self.weight.item()}

    def reward(features):
        if failure == 'score':
            raise RuntimeError('score failed')
        rewards = [float(feature['candidate']) for feature in features]
        return rewards, rewards

    model = Model()
    checkpoint = SimpleNamespace(sync_weights=lambda **kwargs: events.append('sync'))
    steps, requests = StringIO(), StringIO()
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            train_batches(config(group=group), plan, model, Sampler(), checkpoint, None, reward,
                          GRPOAdvantage(), steps, requests, router.get_status)
    else:
        rows = train_batches(config(group=group), plan, model, Sampler(), checkpoint, None, reward,
                             GRPOAdvantage(), steps, requests, router.get_status)
        assert model.weight.item() > 0
        assert model.update_count == 4  # Two full batches, each containing two optimizer mini-batches.
        assert events.count('synchronize') == 2
        assert [row['measured'] for row in rows] == [False, True]
        assert all(row['optimizer_updates'] == 2 for row in rows)
        assert all(row['model_metrics']['weight'] > 0 for row in rows)
        summary = summarize_trial(config(group=group), rows)
        assert summary['trajectories'] == 4 and summary['optimizer_updates'] == 2
        assert summary['cache_hit_fraction'] == (0 if group == 'A' else 0.5)
        assert summary['nonzero_advantage_fraction'] == 1
        assert len(requests.getvalue().splitlines()) == 8
        if group == 'D':
            assert set(session_batches[0]).isdisjoint(session_batches[1])
            assert summary['mean_replicas_per_prompt'] == 1
    assert router.get_status()['session_count'] == router.get_status()['active_leases'] == 0
    assert plan == before


def test_input_plan_rejects_missing_labels_and_bad_lengths(tmp_path):
    path = tmp_path / 'inputs.jsonl'
    path.write_text('\n'.join(json.dumps(feature) for feature in inputs()))
    assert read_inputs(path, config(plan_hash=None)) == inputs()
    with pytest.raises(ValueError, match='exactly'):
        read_inputs(path, config(steps=3, plan_hash=None))
    with pytest.raises(ValueError, match='length'):
        read_inputs(path, config(max_prompt_tokens=1, plan_hash=None))
    with pytest.raises(ValueError, match='hash changed'):
        read_inputs(path, config())
    path.write_text('\n'.join(json.dumps({'input_ids': [1]}) for _ in range(4)))
    with pytest.raises(ValueError, match='ground truth'):
        read_inputs(path, config(plan_hash=None))


@pytest.mark.parametrize('flags', [
    ['--temperature', '0'], ['--warmup-steps', '50'], ['--replicas', '1'],
    ['--mini-batch-size', '7'], ['--max-model-len', '1024'], ['--seeds', '42', '42'],
])
def test_invalid_experiments_fail_before_loading_models(flags):
    with pytest.raises(SystemExit):
        parse_args(['--model', 'local-model', '--output-dir', 'unused', *flags])


def test_seed_pairs_and_failed_trials_do_not_pollute_ratios():
    baseline = {'status': 'complete', 'group': 'B', 'seed': 42, 'plan_hash': 'fixed',
                'trajectories': 100, 'optimizer_updates': 10, 'sample_trajectories_per_s': 2,
                'train_trajectories_per_s': 1, 'stage_totals_s': {'total_s': 100},
                'output_tokens': 1000, 'accuracy': 0.5}
    candidate = {**baseline, 'group': 'D', 'sample_trajectories_per_s': 3,
                 'train_trajectories_per_s': 1.25, 'stage_totals_s': {'total_s': 80}}
    results = paired_comparisons([baseline, candidate, {'status': 'failed', 'group': 'A', 'seed': 42}])
    assert len(results) == 1
    assert results[0]['sample_throughput_ratio'] == 1.5
    assert results[0]['total_time_reduction'] == pytest.approx(0.2)
    with pytest.raises(ValueError, match='different inputs'):
        paired_comparisons([baseline, {**candidate, 'plan_hash': 'different'}])


def test_parent_uses_fresh_processes_rotates_order_and_records_failures(tmp_path, monkeypatch):
    import scripts.benchmark_grpo_sticky_session as benchmark

    output = tmp_path / 'run'
    calls = []

    def launch(flag, config_file, log_file):
        child = json.loads(config_file.read_text())
        if flag == '--prepare-config':
            (output / 'inputs.jsonl').write_text('{}\n')
            return
        calls.append((child['group'], child['seed']))
        if len(calls) == 4:
            raise RuntimeError('worker startup failed')
        summary = {'status': 'complete', 'group': child['group'], 'seed': child['seed'],
                   'plan_hash': child['plan_hash'], 'trajectories': 4, 'optimizer_updates': 2,
                   'sample_trajectories_per_s': 1, 'train_trajectories_per_s': 0.5,
                   'stage_totals_s': {'total_s': 8}, 'output_tokens': 4, 'accuracy': 0.5}
        (config_file.parent / 'summary.json').write_text(json.dumps(summary))

    monkeypatch.setattr(benchmark, 'launch_process', launch)
    with pytest.raises(RuntimeError, match='startup failed'):
        run_experiment({'output_dir': str(output), 'groups': ['A', 'B', 'D'], 'seeds': [42, 43]})
    assert calls == [('A', 42), ('B', 42), ('D', 42), ('B', 43)]
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['status'] == 'failed'
    assert len(summary['trials']) == 4
    assert summary['trials'][-1]['status'] == 'failed'
    assert (output / 'B-seed-43' / 'error.json').exists()


def test_observed_subclasses_run_through_real_ray_continuous_work():
    pytest.importorskip('ray')
    code = '''
import asyncio
import threading
from types import SimpleNamespace
import ray
import twinkle
from twinkle import DeviceGroup, DeviceMesh, remote_class, remote_function
import twinkle.model
import twinkle.sampler
from scripts.benchmark_grpo_sticky_session import observed_classes, prompt_key

class Engine:
    def __init__(self, cache):
        self.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(enable_prefix_caching=cache),
                                          model_config=SimpleNamespace(logprobs_mode='raw_logprobs'))
        self.counter = 0
    async def generate(self, tokens):
        self.counter += 1
        yield SimpleNamespace(request_id=str(self.counter), prompt_token_ids=tokens,
                              outputs=[SimpleNamespace(token_ids=[9])], num_cached_tokens=0)

@remote_class()
class FakeModel:
    def __init__(self, device_mesh=None, **kwargs):
        self.device_mesh = device_mesh

@remote_class(max_concurrency=24)
class FakeSampler:
    def __init__(self, model_id, engine_args=None, device_mesh=None, **kwargs):
        self.engine = SimpleNamespace(engine=Engine(engine_args['enable_prefix_caching']))
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
    def _run_in_loop(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result()
    @remote_function(dispatch='slice_dp', collect='flatten', lazy_collect=False, enable_continous_work=True)
    def sample(self, inputs, sampling_params=None, *, session_ids=None):
        async def generate():
            results = []
            for feature in inputs:
                async for output in self.engine.engine.generate(feature['input_ids']):
                    pass
                results.append(prompt_key(output.prompt_token_ids))
            return results
        return self._run_in_loop(generate())

twinkle.model.TransformersModel = FakeModel
twinkle.sampler.vLLMSampler = FakeSampler
ray.init(num_cpus=12, namespace='grpo-benchmark-test', include_dashboard=False)
try:
    twinkle.initialize(mode='ray', lazy_collect=False, ncpu_proc_per_node=3,
                       groups=[DeviceGroup(name='model', ranks=1, device_type='CPU'),
                               DeviceGroup(name='sampler', ranks=2, device_type='CPU')])
    Model, Sampler = observed_classes()
    model = Model(device_mesh=DeviceMesh.from_sizes(device_type='cpu'), remote_group='model')
    model.synchronize()
    sampler = Sampler('fake', engine_args={'enable_prefix_caching': True}, remote_group='sampler',
                      device_mesh=DeviceMesh.from_sizes(world_size=2, dp_size=2, device_type='cpu'),
                      routing_config={'max_sessions': 2})
    assert len(sampler.benchmark_info()) == 2
    for affinity in (True, False):
        features = [{'input_ids': [1, 2]}] * 3 + [{'input_ids': [3, 4]}] * 3
        sessions = ['a'] * 3 + ['b'] * 3 if affinity else None
        result = sampler.sample(features, session_ids=sessions)
        assert result == [prompt_key(feature['input_ids']) for feature in features]
        if sessions:
            sampler.release_sessions(['a', 'b'])
        rows = sampler.take_observations()
        assert len(rows) == 6
        assert len({row['instance_id'] for row in rows}) == 2
        assert sampler.take_observations() == []
        counts = {key: len({row['instance_id'] for row in rows if row['input_hash'] == key})
                  for key in set(result)}
        assert set(counts.values()) == ({1} if affinity else {2})
        state = ray.get(sampler._request_router.get_status.remote())
        assert state['session_count'] == state['active_leases'] == 0
finally:
    ray.shutdown()
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[2],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
