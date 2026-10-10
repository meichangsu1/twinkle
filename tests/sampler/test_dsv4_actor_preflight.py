"""CPU tests of diagnostic tracing/collection, not HCCL hardware validation."""
from contextlib import contextmanager
from types import SimpleNamespace

import json
import numpy as np
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


@pytest.fixture
def real_batch(tmp_path, monkeypatch):
    model_path = tmp_path / 'base'
    model_path.mkdir()
    (model_path / 'config.json').write_text('{}')
    monkeypatch.setenv('ACTOR_MODEL', str(model_path))
    monkeypatch.setenv('LR', '5e-6')
    monkeypatch.setenv('NPUS_PER_NODE', '2')
    for key in ('LORA_TARGET_MODULES', 'LORA_TARGET_PARAMETERS', 'TWINKLE_FAIL_FAST'):
        monkeypatch.delenv(key, raising=False)
    actor = SimpleNamespace(device_mesh=SimpleNamespace(world_size=2, ep_size=2))
    features = [dict(input_ids=[10, 20 + i, 30 + i], labels=[20 + i, 30 + i, -100],
                     attention_mask=[1, 1, 1], completion_mask=[0, 1, 1]) for i in range(4)]
    logps = [[-0.25 - i, -0.5 - i] for i in range(4)]
    advantages = [-0.5, 1.5, 0.0, 0.0]
    path = probe.save_actor_batch(tmp_path, actor, 0, 2, features, logps, advantages)
    return path, probe._read_actor_batch(path)


def test_batch_capture_preserves_inputs_logprobs_and_resolved_topology(real_batch):
    path, batch = real_batch
    assert batch['format'] == 'dsv4_actor_batch_v1'
    assert batch['training_step'] == 0 and batch['micro_size'] == 2
    assert batch['features'][3]['input_ids'] == [10, 23, 33]
    assert batch['old_logps'][3] == [-3.25, -3.5]
    assert batch['advantages'] == [-0.5, 1.5, 0.0, 0.0]
    assert batch['actor_env']['ACTOR_NPUS'] == batch['actor_env']['ACTOR_EP'] == '2'
    assert batch['actor_env']['LR'] == '5e-6'
    assert batch['actor_env']['LORA_TARGET_PARAMETERS'] == 'mlp.experts.gate_up_proj,mlp.experts.down_proj'
    assert 'RAY_ADDRESS' not in batch['actor_env']


def test_batch_capture_supports_cpu_arrays_and_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setenv('ACTOR_MODEL', str(tmp_path))
    actor = SimpleNamespace(device_mesh=SimpleNamespace(world_size=1, ep_size=1))
    features = [dict(input_ids=torch.tensor([1, 2]), labels=np.array([2, -100]), length=np.int64(2))]
    path = probe.save_actor_batch(tmp_path, actor, 0, 1, features, [[-0.125]], [0.0])
    original = path.read_bytes()
    batch = probe._read_actor_batch(path)
    assert batch['features'] == [dict(input_ids=[1, 2], labels=[2, -100], length=2)]
    with pytest.raises(FileExistsError):
        probe.save_actor_batch(tmp_path, actor, 0, 1, [], [], [])
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match='CPU inputs'):
        probe._json_array(torch.empty(1, device='meta'))


@pytest.mark.parametrize('field,value', [('format', 'other'), ('features', []), ('old_logps', []),
                                        ('advantages', [0.0]), ('micro_size', 0), ('micro_size', 3),
                                        ('micro_size', 1)])
def test_bad_replay_packet_fails_before_model_setup(real_batch, field, value):
    path, batch = real_batch
    batch[field] = value
    path.write_text(json.dumps(batch))
    with pytest.raises(ValueError):
        probe._read_actor_batch(path)


def test_replay_uses_distinct_saved_microbatches_and_only_then_updates(real_batch, monkeypatch):
    path, batch = real_batch
    actor = object()
    calls = []

    def invoke(received_actor, method, **kwargs):
        assert received_actor is actor
        calls.append((method, kwargs))
        return {'loss': torch.tensor(0.25)}

    monkeypatch.setattr(probe, '_probe_call', invoke)
    monkeypatch.setattr(probe, '_memory_summary', lambda *a: pytest.fail('No extra NPU memory sync'))
    probe._replay_actor_batch(actor, batch)
    assert [method for method, _ in calls] == ['forward_backward', 'forward_backward', 'clip_grad_and_step']
    for index, (_, kwargs) in enumerate(calls[:2]):
        start = index * 2
        assert kwargs == dict(inputs=batch['features'][start:start + 2], old_logps=batch['old_logps'][start:start + 2],
                              advantages=batch['advantages'][start:start + 2], adapter_name=probe.TENANT)


def test_replay_does_not_update_after_a_failed_microbatch(real_batch, monkeypatch):
    path, batch = real_batch
    calls = []

    def invoke(actor, method, **kwargs):
        calls.append(method)
        raise RuntimeError('rank 1 loss computation failed')

    monkeypatch.setattr(probe, '_probe_call', invoke)
    with pytest.raises(RuntimeError, match='rank 1 loss'):
        probe._replay_actor_batch(object(), batch)
    assert calls == ['forward_backward']


def test_replay_keeps_ragged_features_masks_and_rollout_logprobs(real_batch, monkeypatch):
    path, batch = real_batch
    batch['features'][1] = dict(input_ids=[10, 21, 31, 41], labels=[-100, 31, 41, -100],
                                completion_mask=[0, 1, 1, 0])
    batch['old_logps'][1] = [-0.75, -1.25]
    batch['features'][3] = dict(input_ids=[10, 23], labels=[23, -100], completion_mask=[1, 0])
    batch['old_logps'][3] = [-2.5]
    path.write_text(json.dumps(batch))
    restored = probe._read_actor_batch(path)
    calls = []
    monkeypatch.setattr(probe, '_probe_call',
                        lambda actor, method, **kw: calls.append((method, kw)) or {'loss': 0.0})
    probe._replay_actor_batch(object(), restored)
    assert calls[0][1]['inputs'][1] == batch['features'][1]
    assert calls[1][1]['inputs'][1] == batch['features'][3]
    assert calls[1][1]['old_logps'] == batch['old_logps'][2:]


def test_replay_rejects_explicit_extra_sync_before_runtime_setup(real_batch, monkeypatch):
    path, batch = real_batch
    monkeypatch.setenv('ACTOR_REPLAY_BATCH', str(path))
    monkeypatch.setenv('PREFLIGHT_SYNC_BEFORE_GATHER', '1')
    monkeypatch.setattr(probe, 'ensure_npu_backend', lambda: pytest.fail('Reject timing changes before model setup'))
    with pytest.raises(ValueError, match='PREFLIGHT_SYNC_BEFORE_GATHER=0'):
        probe.main()


def test_replay_main_needs_no_dataset_and_syncs_memory_only_after_update(real_batch, monkeypatch, capsys):
    path, batch = real_batch
    monkeypatch.setenv('ACTOR_REPLAY_BATCH', str(path))
    monkeypatch.setenv('RAY_ADDRESS', 'head:6379')
    monkeypatch.setenv('ACTOR_NPUS', '99')  # Captured actor configuration wins.
    monkeypatch.setenv('BATCH_SIZE', '0')  # Synthetic-only settings are not read.
    monkeypatch.setenv('PREFLIGHT_SYNC_BEFORE_GATHER', '0')
    monkeypatch.delenv('DAPO_PATH', raising=False)
    monkeypatch.setattr(probe, 'ensure_npu_backend', lambda: None)
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(is_available=lambda: True), raising=False)
    monkeypatch.setattr(probe, '_make_features', lambda *a: pytest.fail('Do not re-encode saved input tokens'))
    monkeypatch.setattr(ray, 'init', lambda **kw: None)
    monkeypatch.setattr(ray, 'cluster_resources', lambda: {'NPU': 2})
    import transformers
    monkeypatch.setattr(transformers.AutoConfig, 'from_pretrained',
                        lambda *a, **kw: SimpleNamespace(n_routed_experts=2, use_cache=True))
    monkeypatch.setattr(probe.twinkle, 'initialize', lambda **kw: None)
    settings, events = {}, []

    class Actor:
        def __init__(self, **kwargs):
            settings.update(kwargs)

        def add_adapter_to_model(self, *args, **kwargs):
            pass

        def __getattr__(self, method):
            assert method.startswith('set_')
            return lambda *a, **kw: settings.update({method: kw})

    monkeypatch.setattr(probe, 'PreflightActor', Actor)
    monkeypatch.setattr(probe, '_probe_call',
                        lambda actor, method, **kw: events.append(method) or {'loss': torch.tensor(0.0)})
    monkeypatch.setattr(probe, '_memory_summary', lambda actor, stage: events.append(stage))
    probe.main()
    assert events == ['forward_backward', 'forward_backward', 'clip_grad_and_step', 'after_optimizer_step']
    assert settings['device_mesh'].world_size == 2
    assert settings['set_template']['enable_thinking'] is False
    assert 'PASSED actor batch replay' in capsys.readouterr().out
