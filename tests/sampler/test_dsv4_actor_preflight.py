"""CPU tests of diagnostic tracing/collection, not HCCL hardware validation."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import ray
import torch

from cookbook.rl.grpo import dsv4_actor_only_preflight as probe
from twinkle.model.transformers import transformers as implementation


def _fake_call(refs, collected):
    def result():
        return collected

    result._futures = refs
    return result


def test_later_ready_failure_is_not_hidden_by_earlier_pending_rank(monkeypatch):
    refs = [object(), object()]
    result = _fake_call(refs, None)
    actor = SimpleNamespace(_lazy_collect=False)

    def invoke(**kwargs):
        assert actor._lazy_collect is True
        return result

    actor.clip_grad_and_step = invoke
    monkeypatch.setattr(ray, 'wait', lambda pending, **kw: ([refs[1]], [refs[0]]))
    seen = []

    def get(ref):
        seen.append(ref)
        raise RuntimeError('rank 1 failed before the collective')

    monkeypatch.setattr(ray, 'get', get)
    with pytest.raises(RuntimeError, match='rank 1 failed'):
        probe._probe_call(actor, 'clip_grad_and_step')
    assert seen == [refs[1]]
    assert actor._lazy_collect is False


def test_out_of_order_completions_preserve_original_collector(monkeypatch):
    refs = [object(), object()]
    expected = ['rank 0', 'rank 1']
    result = _fake_call(refs, expected)
    actor = SimpleNamespace(npu_memory=lambda: result)
    responses = iter([([refs[1]], [refs[0]]), ([refs[0]], [])])
    monkeypatch.setattr(ray, 'wait', lambda *a, **kw: next(responses))
    monkeypatch.setattr(ray, 'get', lambda ref: None)
    assert probe._probe_call(actor, 'npu_memory') is expected
    assert not hasattr(actor, '_lazy_collect')


@pytest.mark.parametrize('existing_flag', [None, False, True])
def test_uninitialized_module_driver_restores_flag_without_module_hooks(monkeypatch, existing_flag):
    # remote_class does not initialize nn.Module on the driver: only the
    # remote workers have _parameters/_modules. Match that actual proxy type.
    actor = object.__new__(probe.PreflightActor)
    assert '_parameters' not in vars(actor)
    if existing_flag is not None:
        vars(actor)['_lazy_collect'] = existing_flag
    expected = ['rank 0']

    def invoke():
        assert actor._lazy_collect is True
        return _fake_call([], expected)

    vars(actor)['npu_memory'] = invoke
    monkeypatch.setattr(ray, 'wait', lambda *a, **kw: pytest.fail('No pending references'))
    assert probe._probe_call(actor, 'npu_memory') is expected
    if existing_flag is None:
        assert '_lazy_collect' not in vars(actor)
    else:
        assert actor._lazy_collect is existing_flag


def test_uninitialized_module_driver_restores_flag_when_dispatch_fails():
    actor = object.__new__(probe.PreflightActor)

    def invoke():
        assert actor._lazy_collect is True
        raise RuntimeError('dispatch failed')

    vars(actor)['npu_memory'] = invoke
    with pytest.raises(RuntimeError, match='dispatch failed'):
        probe._probe_call(actor, 'npu_memory')
    assert '_lazy_collect' not in vars(actor)


def test_timeout_identifies_pending_worker_indices(monkeypatch):
    refs = [object(), object()]
    actor = SimpleNamespace(npu_memory=lambda: _fake_call(refs, None))
    ticks = iter([0, 0, 2, 2])
    monkeypatch.setenv('PREFLIGHT_RPC_TIMEOUT', '1')
    monkeypatch.setattr(probe.time, 'monotonic', lambda: next(ticks))
    monkeypatch.setattr(ray, 'wait', lambda *a, **kw: ([], refs))
    with pytest.raises(TimeoutError, match=r'pending worker indices: \[0, 1\]'):
        probe._probe_call(actor, 'npu_memory')


@pytest.fixture
def fake_clip(monkeypatch):
    events = []
    actor = object.__new__(probe.PreflightActor)
    torch.nn.Module.__init__(actor)
    group = SimpleNamespace(group_name='token_pg')

    @contextmanager
    def adapter(*args, **kwargs):
        events.append('original_adapter.enter')
        yield 'slot_0'
        events.append('original_adapter.exit')

    actor.multi_adapter = SimpleNamespace(adapter=adapter)
    monkeypatch.setattr(probe, '_trace', lambda stage, **kw: events.append(stage))
    monkeypatch.setattr(probe, '_group_details', lambda group: dict(group_name=group.group_name))
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(synchronize=lambda: events.append('device_sync')), raising=False)

    def gather(value, mesh, received_group):
        assert value == [3] and received_group is group
        events.append('original_gather')
        return [3, 3]

    def norm(parameters, **kwargs):
        assert kwargs['num_tokens'] == 6 and kwargs['group'] is group
        events.append('original_norm')
        return 2.0

    def clip(self, max_grad_norm, norm_type, **kwargs):
        with self.multi_adapter.adapter(kwargs['adapter_name']):
            tokens = implementation.torch_util.gather_object([3], None, group)
            return implementation.normalize_and_clip_grad_norm([], num_tokens=sum(tokens), group=group)

    monkeypatch.setattr(implementation.torch_util, 'gather_object', gather)
    monkeypatch.setattr(implementation, 'normalize_and_clip_grad_norm', norm)
    monkeypatch.setattr(probe.MultiLoraTransformersModel, 'clip_grad_norm', clip)
    return actor, events, adapter, gather, norm


@pytest.mark.parametrize('sync', ['0', '1'])
def test_clip_traces_existing_operations_and_restores_patches(fake_clip, monkeypatch, sync):
    actor, events, adapter, gather, norm = fake_clip
    monkeypatch.setenv('PREFLIGHT_SYNC_BEFORE_GATHER', sync)
    assert actor.clip_grad_norm(adapter_name=probe.TENANT) == 2.0
    assert events.index('adapter.ready') < events.index('token_gather.enter')
    assert events.index('token_gather.exit') < events.index('gradient_norm.enter')
    assert ('device_sync' in events) == (sync == '1')
    assert actor.multi_adapter.adapter is adapter
    assert implementation.torch_util.gather_object is gather
    assert implementation.normalize_and_clip_grad_norm is norm


def test_gather_failure_is_logged_and_patches_are_restored(fake_clip, monkeypatch):
    actor, events, adapter, gather, norm = fake_clip
    monkeypatch.setenv('PREFLIGHT_SYNC_BEFORE_GATHER', '0')

    def failing_gather(*args):
        raise RuntimeError('collective failed')

    monkeypatch.setattr(implementation.torch_util, 'gather_object', failing_gather)
    with pytest.raises(RuntimeError, match='collective failed'):
        actor.clip_grad_norm(adapter_name=probe.TENANT)
    assert 'token_gather.error' in events
    assert 'gradient_norm.enter' not in events
    assert actor.multi_adapter.adapter is adapter
    assert implementation.torch_util.gather_object is failing_gather
    assert implementation.normalize_and_clip_grad_norm is norm


def test_adapter_activation_failure_is_distinguished_from_gather(fake_clip, monkeypatch):
    actor, events, adapter, gather, norm = fake_clip

    @contextmanager
    def failing_adapter(*args, **kwargs):
        raise RuntimeError('activation failed')
        yield  # pragma: no cover -- make this a contextmanager

    actor.multi_adapter.adapter = failing_adapter
    with pytest.raises(RuntimeError, match='activation failed'):
        actor.clip_grad_norm(adapter_name=probe.TENANT)
    assert 'adapter_context.error' in events
    assert 'token_gather.enter' not in events


def test_device_sync_failure_is_reported_before_the_collective(fake_clip, monkeypatch):
    actor, events, adapter, gather, norm = fake_clip
    monkeypatch.setenv('PREFLIGHT_SYNC_BEFORE_GATHER', '1')

    def synchronize():
        raise RuntimeError('earlier asynchronous device operation failed')

    monkeypatch.setattr(torch.npu, 'synchronize', synchronize)
    with pytest.raises(RuntimeError, match='earlier asynchronous'):
        actor.clip_grad_norm(adapter_name=probe.TENANT)
    assert 'pre_token_gather_sync.error' in events
    assert 'token_gather.enter' not in events
    assert implementation.torch_util.gather_object is gather


def test_optimizer_probe_keeps_the_existing_step_order(monkeypatch):
    actor = object.__new__(probe.PreflightActor)
    torch.nn.Module.__init__(actor)
    actor.optimizer_group = {probe.TENANT: SimpleNamespace(cur_step=4, gradient_accumulation_steps=1)}
    calls = []
    events = []
    monkeypatch.setattr(probe, '_trace', lambda stage, **kw: events.append(stage))
    actor.clip_grad_norm = lambda *a, **kw: calls.append('clip')
    actor.step = lambda **kw: calls.append('step')
    actor.zero_grad = lambda **kw: calls.append('zero')
    actor.lr_step = lambda **kw: calls.append('lr')
    actor.clip_grad_and_step(adapter_name=probe.TENANT)
    assert calls == ['clip', 'step', 'zero', 'lr']
    assert events[0] == 'optimizer_update.enter'
    assert events[-1] == 'optimizer_update.exit'
