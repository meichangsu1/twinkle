# Copyright (c) ModelScope Contributors. All rights reserved.
"""Real CPU Ray actors exercise the routing/dispatch boundary, without vLLM."""
import asyncio

import pytest

import twinkle.infra as infra
from twinkle.infra.ray_routing import get_handle_router, get_shared_router


@infra.remote_class(max_concurrency=4)
class SamplingClient:

    def __init__(self, **kwargs):
        self.engine_kwargs = kwargs

    @infra.remote_function(enable_continous_work=True, lazy_collect=False)
    def sample(self, inputs, sampling_params=None, *, session_ids=None, adapter_paths=None):
        assert session_ids is None  # Dispatch metadata must not reach the sampler body.
        return inputs


@pytest.fixture(scope='module')
def ray_pool():
    ray = pytest.importorskip('ray')

    @ray.remote(num_cpus=0, max_concurrency=4)
    class Worker:

        def __init__(self, name):
            self.name = name
            self.inputs_seen = []

        async def _twinkle_async_sample(self, inputs, sampling_params=None, adapter_paths=None):
            if isinstance(inputs, dict):
                inputs = [inputs]
            self.inputs_seen.extend(inputs)
            await asyncio.sleep(max((value.get('delay', 0) for value in inputs), default=0))
            if any(value.get('fail') for value in inputs):
                raise RuntimeError('worker generation failed')
            if inputs and 'messages' in inputs[0]:
                from twinkle.data_format import SampledSequence, SampleResponse
                return [SampleResponse(sequences=[SampledSequence('stop', [], decoded=self.name)])
                        for _ in inputs]
            if adapter_paths is not None:
                return [(self.name, value['id'], adapter) for value, adapter in zip(inputs, adapter_paths, strict=True)]
            return [(self.name, value['id']) for value in inputs]

        def seen(self):
            return self.inputs_seen

    started_here = not ray.is_initialized()
    if started_here:
        ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False)
    workers = [Worker.remote('A'), Worker.remote('B')]
    yield ray, workers
    for worker in workers:
        ray.kill(worker)
    if started_here:
        ray.shutdown()


@pytest.fixture
def client(ray_pool, monkeypatch):
    ray, workers = ray_pool
    monkeypatch.setattr(infra, '_mode', 'ray')
    client = object.__new__(SamplingClient)
    client._actors = workers
    client._max_concurrency = 4
    client._routing_config = {'max_sessions': 10000}
    get_handle_router(client, workers)
    return client


def item(name, **kwargs):
    return {'id': name, **kwargs}


def test_global_router_sticky_order_and_input_isolation(client, ray_pool):
    ray, workers = ray_pool
    original = [item('one'), item('two'), item('three')]
    sessions = ['shared', 'other', 'shared']
    first = client.sample(original, session_ids=sessions)
    assert [value for _, value in first] == ['one', 'two', 'three']
    assert first[0][0] == first[2][0]
    other = object.__new__(SamplingClient)
    other._actors = workers
    other._routing_config = {'max_sessions': 10000}
    second = other.sample(inputs=[item('again')], sampling_params={'temperature': 0.0}, session_ids=['shared'])
    assert second[0][0] == first[0][0]
    assert client._request_router._actor_id == other._request_router._actor_id
    assert original == [{'id': 'one'}, {'id': 'two'}, {'id': 'three'}]
    assert sessions == ['shared', 'other', 'shared']
    for worker in workers:
        assert all('session_id' not in value and '_twinkle_routing' not in value
                   for value in ray.get(worker.seen.remote()))
    client.release_sessions(['shared', 'other'])
    assert ray.get(client._request_router.get_status.remote())['total_inflight'] == 0
    assert client.sample([]) == []
    assert client.sample(item('single'))[0][1] == 'single'


def test_two_rollout_actors_share_binding_even_after_handle_was_used(client, ray_pool):
    ray, _ = ray_pool
    expected = client.sample([item('initial')], session_ids=['cross-actor'])[0][0]

    @ray.remote(num_cpus=0)
    class RolloutClient:

        def __init__(self, sampler):
            infra._mode = 'ray'
            self.sampler = sampler

        def generate(self, name):
            return self.sampler.sample([item(name)], session_ids=['cross-actor'])

    actors = [RolloutClient.remote(client) for _ in range(2)]
    try:
        results = ray.get([actor.generate.remote(str(i)) for i, actor in enumerate(actors)])
        assert all(result[0][0] == expected for result in results)
        assert ray.get(client._request_router.get_status.remote())['total_inflight'] == 0
    finally:
        for actor in actors:
            ray.kill(actor)
        client.release_sessions(['cross-actor'])


def test_timeout_releases_call_occupancy_and_allows_session_cleanup(client, ray_pool):
    ray, _ = ray_pool
    client._ray_get_timeout = 0.01
    with pytest.raises(ray.exceptions.GetTimeoutError):
        client.sample([item('slow', delay=0.4)], session_ids=['timeout'])
    assert ray.get(client._request_router.get_status.remote())['total_inflight'] == 0
    client.release_sessions(['timeout'])
    status = ray.get(client._request_router.get_status.remote())
    assert status['total_inflight'] == 0
    assert 'timeout' not in status['sessions']


def test_failed_generation_releases_lease(client, ray_pool):
    ray, _ = ray_pool
    with pytest.raises(Exception, match='worker generation failed'):
        client.sample([item('broken', fail=True)], session_ids=['failure'])
    infra._cw_barrier(client, 'after_error')
    assert ray.get(client._request_router.get_status.remote())['total_inflight'] == 0
    client.release_sessions(['failure'])


def test_partial_submission_failure_releases_all_occupancy(client, ray_pool):
    ray, workers = ray_pool
    calls = 0

    def execute(name, jobs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('submission failed')
        worker, args, kwargs = jobs[0]
        return [getattr(worker, name).remote(*args, **kwargs)]

    with pytest.raises(RuntimeError, match='submission failed'):
        infra._run_continous_work(client, 'sample', execute, workers,
                                 ([item('one'), item('two')],), {}, 2, None,
                                 session_ids=['part-1', 'part-2'])
    infra._cw_barrier(client, 'after_submit_error')
    assert ray.get(client._request_router.get_status.remote())['total_inflight'] == 0
    client.release_sessions(['part-1', 'part-2'])


def test_pool_config_mismatch_is_rejected(ray_pool):
    _, workers = ray_pool
    with pytest.raises(ValueError, match='mismatch'):
        get_shared_router(workers, {'max_sessions': 1})


def test_session_ids_and_adapters_stay_aligned_after_grouping(client, ray_pool):
    ray, _ = ray_pool
    results = client.sample(
        [item('one'), item('two'), item('three')],
        session_ids=['alignment-A', 'alignment-B', 'alignment-A'],
        adapter_paths=['adapter-1', None, 'adapter-3'],
    )
    assert [(name, adapter) for _, name, adapter in results] == [
        ('one', 'adapter-1'), ('two', None), ('three', 'adapter-3'),
    ]
    assert results[0][0] == results[2][0]
    client.release_sessions(['alignment-A', 'alignment-B'])
    assert ray.get(client._request_router.get_status.remote())['total_inflight'] == 0


@pytest.mark.parametrize('session_ids', ['id', [], ['one', 'two'], [''], [1]])
def test_invalid_session_ids_do_not_acquire_occupancy(client, ray_pool, session_ids):
    ray, _ = ray_pool
    before = ray.get(client._request_router.get_status.remote())
    with pytest.raises(ValueError, match='session_id'):
        client.sample([item('invalid')], session_ids=session_ids)
    assert ray.get(client._request_router.get_status.remote()) == before


def test_omitted_and_none_ids_use_ordinary_routing(client, ray_pool):
    ray, _ = ray_pool
    before = ray.get(client._request_router.get_status.remote())['session_count']
    client.sample([item('plain')])
    client.sample([item('none')], session_ids=[None])
    assert client.sample(item('single'), session_ids=['single-session'])[0][1] == 'single'
    client.release_sessions(['single-session'])
    assert client.sample([], session_ids=[]) == []
    with pytest.raises(ValueError, match='one entry per input'):
        client.sample([], session_ids=['extra'])
    assert ray.get(client._request_router.get_status.remote())['session_count'] == before


def test_session_ids_require_concrete_inputs_for_routing(client, ray_pool):
    ray, _ = ray_pool
    with pytest.raises(ValueError, match='concrete input batch'):
        client.sample(ray.put([item('deferred')]), session_ids=['deferred'])


def test_routing_config_is_not_forwarded_to_local_engine(monkeypatch):
    monkeypatch.setattr(infra, '_mode', 'local')
    sampler = SamplingClient(routing_config={'max_sessions': 2})
    assert sampler._routing_config == {'max_sessions': 2}
    assert sampler.engine_kwargs == {}
    original = [item('local')]
    assert sampler.sample(original, session_ids=['s']) == [{'id': 'local'}]
    assert original == [{'id': 'local'}]
    sampler.release_sessions(['s'])


def test_worker_wrapper_consumes_session_ids(monkeypatch):
    monkeypatch.setattr(infra, '_mode', 'ray')
    sampler = object.__new__(SamplingClient)
    original = [item('worker')]
    assert sampler.sample(original, session_ids=['s']) is original


def test_remote_constructor_creates_shared_router_and_consumes_config(ray_pool, monkeypatch):
    from twinkle.infra._ray import RayHelper
    ray, workers = ray_pool
    monkeypatch.setattr(infra, '_mode', 'ray')
    monkeypatch.delenv('CLUSTER_NAME', raising=False)
    forwarded = []

    def create_workers(*args, **kwargs):
        forwarded.append(kwargs)
        return workers

    monkeypatch.setattr(RayHelper, 'create_workers', create_workers)
    sampler = SamplingClient(remote_group='sampling-pool', routing_config={'max_sessions': 10000})
    assert '_request_router' in sampler.__dict__
    assert 'routing_config' not in forwarded[0]
    assert ray.get(sampler._request_router.get_config.remote()) == {'max_sessions': 10000}
    assert sampler.sample([item('constructed')])[0][1] == 'constructed'


def test_vllm_public_sample_has_keyword_only_session_ids():
    import inspect

    from twinkle.sampler.vllm_sampler import vLLMSampler
    parameters = inspect.signature(vLLMSampler.sample).parameters
    assert parameters['session_ids'].kind == inspect.Parameter.KEYWORD_ONLY
    assert parameters['session_ids'].default is None
    assert 'request_ids' not in parameters
    assert vLLMSampler.sample._supports_session_affinity


@pytest.mark.parametrize('custom_callback', [False, True], ids=['default', 'custom'])
def test_multi_turn_rollouts_on_two_actors_share_pool_and_close_sessions(client, ray_pool, custom_callback):
    ray, _ = ray_pool

    @ray.remote(num_cpus=0)
    class RolloutActor:

        def __init__(self, sampler, custom_callback):
            from twinkle_agentic.rollout.multi_turn import MultiTurnRollout
            infra._mode = 'ray'

            def callback(sampler, api, params, *, input_feature, adapter_kwargs, trajectory,
                         trajectory_index, turn, followups):
                return sampler.sample([input_feature], params, **adapter_kwargs)[0].sequences[0]

            self.rollout = MultiTurnRollout(
                sampler, followup_fn=lambda trajectory, count: 'continue' if count == 0 else None,
                response_callback=callback if custom_callback else None)

        def run(self):
            return self.rollout([{'messages': [{'role': 'user', 'content': 'question'}]}])[0]

    actors = [RolloutActor.remote(client, custom_callback) for _ in range(2)]
    try:
        outputs = ray.get([actor.run.remote() for actor in actors])
        for output in outputs:
            assistants = [message['content'] for message in output['messages'] if message['role'] == 'assistant']
            assert len(assistants) == 2 and assistants[0] == assistants[1]
            assert '_twinkle_routing' not in output and 'session_id' not in output
        state = ray.get(client._request_router.get_status.remote())
        assert state['session_count'] == state['total_inflight'] == 0
    finally:
        for actor in actors:
            ray.kill(actor)
