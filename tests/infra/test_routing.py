# Copyright (c) ModelScope Contributors. All rights reserved.
import threading
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor

import pytest

from twinkle.infra.routing import LeastInflightPolicy, RequestRouter, normalize_routing_config


class FirstMinimum:

    def select(self, loads):
        return min(loads, key=loads.get)


def router(limit=10000):
    return RequestRouter({'A': 'handle-A', 'B': 'handle-B'}, max_sessions=limit, policy=FirstMinimum())


def test_least_load_affinity_units_and_duplicate_release():
    r = router()
    a = r.acquire('s1', units=3)
    b = r.acquire('s2')
    again = r.acquire('s1')
    assert (a.instance_id, b.instance_id, again.instance_id) == ('A', 'B', 'A')
    assert a.handle == 'handle-A'
    assert len({a.lease_id, b.lease_id, again.lease_id}) == 3
    assert r.get_status()['loads'] == {'A': 4, 'B': 1}
    r.release(a)
    r.release(a)
    r.release_batch([b.lease_id, again.lease_id])
    assert r.get_status()['total_inflight'] == 0
    assert r.get_status()['session_count'] == 2
    r.release_sessions(['s1', 's2', 'missing'])
    assert r.get_status()['session_count'] == 0


def test_lru_refresh_and_waiting_tool_session_eviction():
    r = router(2)
    for sid in ['old', 'new', 'old']:
        r.release(r.acquire(sid))
    r.release(r.acquire('third'))
    assert list(r.get_status()['sessions']) == ['old', 'third']
    # A session waiting on a tool has no executing lease and may lose affinity.
    r.release(r.acquire('new'))
    assert list(r.get_status()['sessions']) == ['third', 'new']


def test_protect_inflight_and_fallback_without_blocking():
    r = router(1)
    protected = r.acquire('protected')
    fallback = r.acquire('other')
    state = r.get_status()
    assert state['sessions'] == {'protected': 'A'}
    assert state['capacity_fallbacks'] == 1
    assert fallback.instance_id == 'B'
    with pytest.raises(RuntimeError, match='in flight'):
        r.release_sessions(['protected'])
    r.release(protected)
    r.release(fallback)
    r.release(r.acquire('other'))
    assert r.get_status()['sessions'] == {'other': 'A'}


def test_remove_instance_with_inflight_lease_and_rebind():
    r = router()
    old = r.acquire('s')
    r.remove_instances(['A'])
    new = r.acquire('s')
    assert new.instance_id == 'B'
    r.release(old)
    r.release(new)
    assert r.get_status()['loads'] == {'B': 0}
    r.add_instances({'C': 'handle-C'})
    r.remove_instances(['B', 'C'])
    with pytest.raises(RuntimeError, match='No available'):
        r.acquire()


def test_failed_batch_acquire_rolls_back_bindings_and_load():
    class FailingPolicy:

        def __init__(self):
            self.calls = 0

        def select(self, loads):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError('policy failed')
            return next(iter(loads))

    r = RequestRouter({'A': object()}, policy=FailingPolicy())
    with pytest.raises(RuntimeError, match='policy failed'):
        r.acquire_batch([('one', 1), ('two', 1)])
    assert r.get_status()['active_leases'] == r.get_status()['session_count'] == 0


def test_concurrent_acquire_is_atomic():
    r = router()
    barrier = threading.Barrier(8)

    def work(_):
        lease = r.acquire('shared')
        barrier.wait(timeout=5)
        assert r.get_status()['session_count'] == 1
        r.release(lease)
        return lease.instance_id

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert len(set(pool.map(work, range(8)))) == 1
    assert r.get_status()['total_inflight'] == 0


def test_tied_loads_use_random_candidates(monkeypatch):
    seen = []

    def choose(candidates):
        seen.append(candidates)
        return candidates[-1]

    monkeypatch.setattr('twinkle.infra.routing.random.choice', choose)
    assert LeastInflightPolicy().select({'A': 1, 'B': 0, 'C': 0}) == 'C'
    assert seen == [['B', 'C']]


@pytest.mark.parametrize('config', [{'max_sessions': 0}, {'max_sessions': True}, {'unknown': 1}, 4])
def test_config_validation(config):
    with pytest.raises(ValueError):
        normalize_routing_config(config)


def test_validation_happens_before_batch_admission():
    r = router()
    with pytest.raises(ValueError):
        r.acquire_batch([('valid', 1), ('', 1)])
    assert r.get_status()['active_leases'] == 0


def test_async_group_lease_is_released_at_confirmed_completion_not_collect():
    r = router()
    lease = r.acquire('group', units=8)
    task = Future()
    task.add_done_callback(lambda future: r.release(lease))
    # A cancellation request alone is not proof that the backend has stopped.
    cancellation_requested = True
    assert cancellation_requested and r.get_status()['total_inflight'] == 8
    task.set_exception(CancelledError())  # Backend confirms termination.
    assert r.get_status()['total_inflight'] == 0
    # No consumer has collected a result, and the session remains independent.
    assert r.get_status()['session_count'] == 1
    r.release_sessions(['group'])
