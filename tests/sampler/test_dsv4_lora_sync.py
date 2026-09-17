"""CPU protocol tests. Hardware numerical/transport acceptance is separate."""
import asyncio
import importlib.util
import json
import pytest
import subprocess
import sys
import torch
import types
from pathlib import Path

from twinkle.checkpoint_engine.manager import CheckpointEngineManager
from twinkle.sampler.vllm_sampler.dsv4_lora import PREFIX, TARGETS, DeepSeekV4NvidiaLoraAdapter, target_peft_config

ADAPTER_CONFIG = dict(class_path='twinkle.sampler.vllm_sampler.dsv4_lora.DeepSeekV4NvidiaLoraAdapter', options={})


def fixture_config(dtype=torch.bfloat16):
    return dict(
        model_config=dict(model_type='deepseek_v4', num_hidden_layers=2,
                          n_routed_experts=3, hidden_size=8, moe_intermediate_size=4),
        peft_config=dict(r=2, lora_alpha=8, target_modules=[], target_parameters=sorted(TARGETS)),
        dtype=str(dtype))


def new_adapter(metadata, cls=DeepSeekV4NvidiaLoraAdapter):
    adapter = cls()
    adapter.initialize(
        dict(
            device='cuda',
            enable_lora=True,
            pp_size=1,
            ep_enabled=False,
            model_config=metadata['model_config'],
            max_lora_rank=128,
            lora_dtype=metadata['dtype']), {})
    adapter.begin_update(metadata["peft_config"])
    return adapter


def convert_groups(source, metadata):
    adapter = new_adapter(metadata)
    for group in source:
        adapter.consume_weights(group.items())
    return adapter.finish_update()[0].items()


def groups(dtype=torch.bfloat16):
    torch.manual_seed(123)
    result = []
    for layer in range(2):
        for projection, ins, outs in [('gate_up_proj', 8, 8), ('down_proj', 4, 8)]:
            prefix = f'model.layers.{layer}.mlp.experts.{projection}'
            result.append({
                f'{prefix}.lora_A.weight': torch.randn(3, 2, ins).to(dtype),
                f'{prefix}.lora_B.weight': torch.randn(3, outs, 2).to(dtype)
            })
    return result


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_conversion_matches_independent_offline_recipe(dtype):
    source = groups(dtype)
    actual = dict(convert_groups(source, fixture_config(dtype)))
    expected = {}
    for index, group in enumerate(source):
        a, b = group.values()
        layer = index // 2
        for expert in range(3):
            for projection, bp in ([('w1', b[:, :4]), ('w3', b[:, 4:])] if index % 2 == 0 else [('w2', b)]):
                prefix = f'{PREFIX}.{layer}.ffn.experts.{expert}.{projection}'
                expected[f'{prefix}.lora_A.weight'] = a[expert]
                expected[f'{prefix}.lora_B.weight'] = bp[expert]
    assert actual.keys() == expected.keys()
    assert len(actual) == 36
    assert all(actual[k].dtype == dtype and torch.equal(actual[k], expected[k]) for k in actual)
    for group in source:
        for tensor in group.values():
            tensor.zero_()
    assert any(t.count_nonzero() for t in actual.values())  # owned storage


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_parity_with_existing_nvidia_offline_converter(tmp_path, dtype):
    from safetensors.torch import load_file, save_file
    converter = Path(__file__).resolve().parents[2] / 'output' / 'convert_twinkle_dsv4_lora_for_vllm.py'
    if not converter.is_file():
        pytest.skip('Optional local offline oracle is not version-controlled; independent recipe test still runs')
    source = tmp_path / 'source'
    base = tmp_path / 'base'
    source.mkdir()
    base.mkdir()
    m = fixture_config(dtype)
    (source / 'adapter_config.json').write_text(
        json.dumps(dict(r=2, lora_alpha=8, target_modules=[], target_parameters=sorted(TARGETS))), encoding='utf-8')
    (base / 'config.json').write_text(json.dumps(m['model_config']), encoding='utf-8')
    state = groups(dtype)
    save_file({('base_model.model.base_model.model.' + k.replace('.gate_up_proj', '.base_layer').replace('.down_proj', '')): v for group in state for k, v in group.items()}, str(source / 'adapter_model.safetensors'))
    subprocess.run([
        sys.executable,
        str(converter),
        str(source),
        str(tmp_path / 'converted'), '--backend', 'nvidia', '--base-model',
        str(base)
    ],
                   check=True,
                   capture_output=True,
                   timeout=60)
    oracle = load_file(str(tmp_path / 'converted' / 'adapter_model.safetensors'))
    actual = dict(convert_groups(state, m))
    assert actual.keys() == oracle.keys()
    assert all(actual[k].dtype == oracle[k].dtype and torch.equal(actual[k], oracle[k]) for k in actual)


def test_scaling_and_rejections():
    m = fixture_config()
    adapter = new_adapter(m)
    assert adapter.scale == 4 and adapter.peft_config['target_modules'] == ['experts']
    m['peft_config']['use_rslora'] = True
    assert new_adapter(m).scale == 8 / (2**0.5)
    config = dict(m['peft_config'])
    for bad in [dict(target_modules='all-linear'), dict(use_dora=True), dict(rank_pattern={'x': 1})]:
        with pytest.raises(ValueError):
            target_peft_config(config | bad)
    with pytest.raises(ValueError, match='Incomplete'):
        list(convert_groups(groups()[:-1], fixture_config()))
    with pytest.raises(ValueError, match='duplicate'):
        list(convert_groups(groups() + groups()[:1], fixture_config()))
    with pytest.raises(ValueError, match='dtype'):
        list(convert_groups(groups(torch.float32), fixture_config()))


def test_explicit_tensor_request_payload_roundtrip(monkeypatch):
    import msgspec

    import twinkle.patch.vllm_lora_weights as patch

    class Request(msgspec.Struct, omit_defaults=True, array_like=True):
        lora_name: str
        lora_int_id: int
        lora_path: str
        load_inplace: bool = False

    monkeypatch.setitem(sys.modules, 'vllm.lora.request', types.SimpleNamespace(LoRARequest=Request))
    spec = importlib.util.spec_from_file_location('twinkle.patch._request_probe', patch.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    request = module.TensorLoRARequest(
        'tenant', 111, 'unused', peft_config={'r': 2}, lora_tensors={})
    assert isinstance(request, Request) and request.load_inplace is False
    encoded = msgspec.msgpack.encode(request)
    restored = msgspec.msgpack.decode(encoded, type=module.TensorLoRARequest)
    assert restored.peft_config == {'r': 2} and restored.lora_tensors == {}


def test_worker_processes_then_uses_existing_inplace_loader(monkeypatch):
    import twinkle.patch.vllm_lora_weights as patch
    from twinkle.sampler.vllm_sampler.vllm_worker_extension import TwinkleWorkerExtension
    monkeypatch.setattr(patch, 'TensorLoRARequest', types.SimpleNamespace)
    worker = object.__new__(TwinkleWorkerExtension)
    worker.rank = 0
    worker.device = torch.device('cuda')
    worker.vllm_config = types.SimpleNamespace(
        lora_config=types.SimpleNamespace(max_lora_rank=2, lora_dtype=torch.bfloat16),
        model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(**fixture_config()['model_config'])),
        parallel_config=types.SimpleNamespace(
            pipeline_parallel_size=1, enable_expert_parallel=False, tensor_parallel_size=1))
    calls = []
    worker.list_loras = lambda: {111}
    worker.remove_lora = lambda i: calls.append(('remove', i))
    worker.add_lora = lambda req: calls.append(('add', req)) or True
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    source = [item for group in groups() for item in group.items()]
    config = fixture_config()['peft_config']
    with pytest.raises(ValueError, match='Incomplete'):
        worker._load_weights(source[:-1], config, False, lora_only=True, weight_adapter=ADAPTER_CONFIG)
    assert calls == []
    worker._load_weights(source, config, False, lora_only=True, weight_adapter=ADAPTER_CONFIG)
    assert [c[0] for c in calls] == ['add']
    assert calls[0][1].load_inplace is True
    assert calls[0][1].lora_int_id == 111
    assert calls[0][1].lora_tensors.keys() == dict(convert_groups(groups(), fixture_config())).keys()
    assert not worker._get_weight_adapter(ADAPTER_CONFIG).pending


def test_default_load_hook_is_identity_and_keeps_inplace_install(monkeypatch):
    import twinkle.patch.vllm_lora_weights as patch
    from twinkle.sampler.vllm_sampler.vllm_worker_extension import TwinkleWorkerExtension
    worker = object.__new__(TwinkleWorkerExtension)
    worker._get_weight_adapter = lambda _: pytest.fail('Default path must not initialize a processor')
    weights = [('model.q_proj.lora_A.weight', torch.ones(2, 4))]
    config = {'r': 2, 'lora_alpha': 4, 'target_modules': ['q_proj']}
    actual_weights, actual_config = worker._process_lora_weights(weights, config)
    assert actual_weights is weights and actual_config is config
    monkeypatch.setattr(patch, 'TensorLoRARequest', types.SimpleNamespace)
    requests = []
    worker.add_lora = lambda request: requests.append(request)
    worker.remove_lora = lambda _: pytest.fail('Must keep the existing inplace policy')
    for _ in range(2):
        worker._load_weights(weights, config, True)
    assert len(requests) == 2
    for request in requests:
        assert request.load_inplace is True and request.lora_int_id == 111
        assert request.peft_config is config
        assert next(iter(request.lora_tensors.values())) is weights[0][1]


def test_base_load_does_not_call_lora_hook():
    from twinkle.sampler.vllm_sampler.vllm_worker_extension import TwinkleWorkerExtension
    worker = object.__new__(TwinkleWorkerExtension)
    worker._process_lora_weights = lambda *args: pytest.fail('Base loading must not use a LoRA processor')
    weights = [('model.q_proj.weight', torch.ones(4, 4))]
    loaded = []
    worker.model_runner = types.SimpleNamespace(model=types.SimpleNamespace(load_weights=loaded.extend))
    worker._load_weights(weights, None, False, weight_adapter=ADAPTER_CONFIG)
    assert loaded == weights


def test_ipc_multibucket_reuse_and_tp_barrier_before_ack(monkeypatch):
    """Exercise the actual receiver, including a tensor split across buckets."""
    import zmq

    import twinkle.sampler.vllm_sampler.vllm_worker_extension as extension
    source = [item for group in groups() for item in group.items()]
    buffer = torch.empty(128, dtype=torch.uint8)
    chunks = []
    for name, tensor in source:
        raw = tensor.view(-1).view(torch.uint8)
        split = max(2, raw.numel() // 2)
        for start in range(0, raw.numel(), split):
            part = raw[start:start + split].clone()
            chunks.append((part,
                           dict(
                               name=name,
                               dtype=tensor.dtype,
                               shape=tensor.shape,
                               offset=0,
                               nbytes=part.numel(),
                               chunk_offset=start,
                               total_nbytes=raw.numel())))
    events = []

    class Socket:
        index = -1

        def connect(self, endpoint):
            pass

        def recv_pyobj(self):
            self.index += 1
            if self.index == 0:
                return ('fake', [])
            payload, meta = chunks[self.index - 1]
            buffer.fill_(255)  # invalidate the preceding IPC bucket
            buffer[:payload.numel()].copy_(payload)
            events.append('receive')
            return dict(bucket_meta=[meta], is_last=self.index == len(chunks))

        def send(self, payload):
            if self.index:
                assert events[-1] == 'barrier'
            events.append('ack')

        def close(self):
            pass

    socket = Socket()
    worker = object.__new__(extension.TwinkleWorkerExtension)
    worker.rank, worker.device = 0, torch.device('cpu')
    worker.model_runner = types.SimpleNamespace(parallel_config=types.SimpleNamespace(tensor_parallel_size=2))
    worker._zmq_ctx = types.SimpleNamespace(socket=lambda *a: socket)
    monkeypatch.setattr(extension, 'configure_zmq_socket', lambda *a, **kw: None)
    monkeypatch.setattr(extension, '_rebuild_ipc', lambda *a: buffer)
    for name in ('synchronize', 'ipc_collect', 'empty_cache'):
        monkeypatch.setattr(extension.Torch, name, lambda: None)
    monkeypatch.setattr(torch.distributed, 'broadcast_object_list', lambda *a, **kw: None)
    monkeypatch.setattr(torch.distributed, 'barrier', lambda **kw: events.append('barrier'))
    monkeypatch.setitem(
        sys.modules, 'vllm.distributed',
        types.SimpleNamespace(get_tp_group=lambda: types.SimpleNamespace(cpu_group=object(), ranks=[0, 1])))
    installed = []

    adapter = new_adapter(fixture_config())
    worker._get_weight_adapter = lambda config: adapter

    import twinkle.patch.vllm_lora_weights as patch
    monkeypatch.setattr(patch, 'TensorLoRARequest', types.SimpleNamespace)
    worker.add_lora = lambda request: installed.append(request.lora_tensors) or True
    report = worker.update_weights_from_ipc(
        zmq_handle='ipc://test', lora_only=True, peft_config=fixture_config()['peft_config'],
        weight_adapter=ADAPTER_CONFIG)
    assert report is None
    assert len(installed) == 1
    assert all(torch.equal(installed[0][k], v) for k, v in convert_groups(groups(), fixture_config()))
    assert len(chunks) > len(source)


def test_original_save_preserves_tenant_and_actual_rank():
    from peft import LoraConfig

    from twinkle.model.multi_lora_target_parameters import TargetParameterLoraManager
    module = torch.nn.Module()
    module.experts = torch.nn.Module()
    module.experts.is_transposed = False
    module.experts.gate_up_proj = torch.nn.Parameter(torch.randn(3, 8, 8))
    module.experts.down_proj = torch.nn.Parameter(torch.randn(3, 8, 4))
    manager = TargetParameterLoraManager(max_loras=2, max_r=4)
    targets = ['experts.gate_up_proj', 'experts.down_proj']
    manager.patch(module, targets)
    for name, slot, r in [('tenant_a', 'lora_0', 2), ('tenant_b', 'lora_1', 3)]:
        manager.acquire(name, slot, LoraConfig(r=r, lora_alpha=8, target_modules=[], target_parameters=targets))
    with torch.no_grad():
        for _, p in manager.named_slot_parameters('tenant_b'):
            p.fill_(7)
    for tenant, rank in [('tenant_a', 2), ('tenant_b', 3)]:
        saved = manager.get_state_dict(tenant)
        expected = {}
        for wrapper in manager.wrappers:
            expected.update(wrapper.get_state_dict(manager.tenant_to_slot[tenant]))
        assert saved.keys() == expected.keys()
        for key in saved:
            assert torch.equal(saved[key], expected[key])
            assert saved[key].shape[1 if '.lora_A.' in key else 2] == rank
            if tenant == 'tenant_b':
                assert (saved[key] == 7).all()
    assert manager.get_state_dict('tenant_without_target_parameters') == {}


class FakePeer:

    def __init__(self, calls, kind):
        self.calls, self.kind = calls, kind
        self._actors = [object()]
        self.device_mesh = types.SimpleNamespace(world_size=1, data_world_size=1)
        self.fail = None

    def __getattr__(self, name):

        def method(*args, **kwargs):
            self.calls.append((self.kind, name, args, kwargs))

            def result():
                if self.fail == name:
                    raise RuntimeError('injected failure')
                if name == 'get_state_keys':
                    return ['model.layers.0.qkv_proj.weight']
                if name == 'get_peft_config_dict':
                    return {'r': 2}
                return {}

            return result if name in {'init_checkpoint_process_group', 'send_weights', 'receive_weights'} else result()

        return method


def peers(monkeypatch):
    backend = types.SimpleNamespace(build_topology=lambda *args: ({}, {}))
    monkeypatch.setattr(CheckpointEngineManager, 'decide_backend_engine', lambda *args: backend)
    calls = []
    actor, sampler = FakePeer(calls, 'actor'), FakePeer(calls, 'sampler')
    return CheckpointEngineManager(actor, sampler), calls


def test_manager_first_and_repeated_lora_only_reuse_existing_transport(monkeypatch):
    manager, calls = peers(monkeypatch)
    for _ in range(20):
        report = manager.sync_weights(merge_and_sync=False, lora_only=True, adapter_name='tenant_a')
        assert report is None and manager.base_sync_done is False
    assert not hasattr(manager, '_lora_version') and not hasattr(manager, '_lora_sync_lock')
    assert not any(n in {'get_state_keys', 'load_full_weights_from_path', 'check_lora_sync_bucket'}
                   for _, n, _, _ in calls)
    for kind, name, args, kwargs in calls:
        if name in {'send_weights', 'receive_weights'}:
            assert kwargs['lora_only'] is True
        if name == 'prepare_checkpoint_engine':
            assert kwargs == {}
        if name == 'receive_weights':
            assert set(kwargs) == {'base_sync_done', 'peft_config', 'lora_only'}
    manager.sampler.fail = 'receive_weights'
    with pytest.raises(RuntimeError, match='injected'):
        manager.sync_weights(merge_and_sync=False, lora_only=True, adapter_name='tenant_a')


def test_manager_legacy_modes_unchanged(monkeypatch):
    for merge in (False, True):
        manager, calls = peers(monkeypatch)
        for _ in range(2):
            assert manager.sync_weights(merge_and_sync=merge) is None
        assert manager.base_sync_done is True
        sends = [kw for _, n, _, kw in calls if n == 'send_weights']
        receives = [kw for _, n, _, kw in calls if n == 'receive_weights']
        assert [kw['base_sync_done'] for kw in sends] == [False, True]
        assert all(kw['merge_and_sync'] == merge and 'lora_only' not in kw for kw in sends)
        assert receives[0]['peft_config'] is None
        assert receives[1]['peft_config'] == (None if merge else {'r': 2})
        assert not any(n == 'check_lora_sync_bucket' for _, n, _, _ in calls)


def test_invalid_lora_only_options_do_not_start_transport(monkeypatch):
    manager, calls = peers(monkeypatch)
    for kwargs in [
            dict(lora_only=True, adapter_name='tenant_a'),
            dict(lora_only=True, merge_and_sync=False),
    ]:
        with pytest.raises(ValueError):
            manager.sync_weights(**kwargs)
    assert calls == []


def test_manager_keeps_original_constructor_signature(monkeypatch):
    import inspect
    manager, calls = peers(monkeypatch)
    assert list(inspect.signature(CheckpointEngineManager.__init__).parameters) == [
        'self', 'model', 'sampler', 'platform']
    assert '_bucket_size_mb' not in vars(manager) and '_ipc_bucket_size_mb' not in vars(manager)
    assert calls == []


def test_ray_slice_master_flag_is_unwrapped():
    from twinkle.checkpoint_engine.mixin import CheckpointEngineMixin
    from twinkle.infra import _dispatch_args
    workers = [object() for _ in range(4)]
    dispatched = _dispatch_args(workers, 'slice', 'all', None, ([True, False, False, False], ), {})
    flags = []
    for _, args, kwargs in dispatched:
        engine = types.SimpleNamespace(is_master=None)
        engine.prepare = lambda: engine.is_master
        mixin = CheckpointEngineMixin()
        mixin._get_or_create_checkpoint_engine = lambda: engine
        flags.append(CheckpointEngineMixin.prepare_checkpoint_engine.__wrapped__(mixin, *args, **kwargs))
    assert flags == [True, False, False, False]


@pytest.mark.parametrize('lora_only', [False, True])
@pytest.mark.parametrize('failure', [False, True])
def test_receiver_reuses_original_refresh(monkeypatch, lora_only, failure):
    from unittest.mock import AsyncMock
    from twinkle.sampler.vllm_sampler.vllm_sampler import vLLMSampler
    sampler = object.__new__(vLLMSampler)
    engine = types.SimpleNamespace(
        refresh_synced_lora=AsyncMock(), invalidate_synced_lora=lambda: pytest.fail('unexpected invalidation'))

    async def update(stream, **kwargs):
        assert kwargs['base_sync_done'] == (not lora_only)
        assert 'bucket_size_mb' not in kwargs
        if lora_only:
            assert kwargs['lora_only']
        else:
            assert 'lora_only' not in kwargs
        if failure:
            raise RuntimeError('injected receiver failure')

    engine.update_weights = update
    sampler.engine = engine
    sampler._get_or_create_checkpoint_engine = lambda: types.SimpleNamespace(receive_weights=lambda: object())
    sampler._run_in_loop = asyncio.run
    def call():
        return vLLMSampler.receive_weights.__wrapped__(
            sampler, base_sync_done=not lora_only, lora_only=lora_only,
            peft_config=fixture_config()['peft_config'])
    if failure:
        with pytest.raises(RuntimeError):
            call()
        engine.refresh_synced_lora.assert_not_awaited()
    else:
        assert call() is None
        engine.refresh_synced_lora.assert_awaited_once()


@pytest.mark.parametrize('bucket_size', [None, 32 << 20])
def test_prepare_preserves_original_bucket_configuration(bucket_size):
    from twinkle.checkpoint_engine.mixin import CheckpointEngineMixin
    mixin = CheckpointEngineMixin()
    if bucket_size is not None:
        mixin._bucket_size = bucket_size
    expected_size = mixin._bucket_size
    engine = types.SimpleNamespace(is_master=False, prepare=lambda: {})
    calls = []

    def get_engine():
        calls.append('get')
        mixin._checkpoint_engine = engine
        return engine

    mixin._get_or_create_checkpoint_engine = get_engine
    method = CheckpointEngineMixin.prepare_checkpoint_engine.__wrapped__
    method(mixin, True)
    method(mixin, True)
    assert mixin._checkpoint_engine is engine and mixin._bucket_size == expected_size
    assert len(calls) == 2


def test_source_identity_is_explicit_not_key_or_shape_inference():
    from copy import deepcopy
    m = deepcopy(fixture_config())
    raw = {k: v for g in groups() for k, v in g.items()}
    renamed = raw
    adapter = new_adapter(m)
    # B arrives first, and A/B of different parameters interleave across buckets.
    items = list(renamed.items())
    for item in items[1::2] + items[::2]:
        adapter.consume_weights([item])
    actual, _, report = adapter.finish_update()
    expected = dict(convert_groups(groups(), fixture_config()))
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[k], v) for k, v in expected.items())
    assert not adapter.pending
    assert report['source_tensor_count'] == 8 and report['tensor_count'] == 36
    assert report['target_nbytes'] > report['source_nbytes']  # duplicated shared A, only on rollout
    for value in raw.values():
        value.zero_()
    assert all(torch.equal(actual[k], v) for k, v in expected.items())


@pytest.mark.parametrize('bad', ['name', 'shape', 'dtype', 'old_wrapper_name'])
def test_invalid_stream_rejected_before_installation(bad):
    adapter = new_adapter(fixture_config())
    name, value = next(iter(groups()[0].items()))
    if bad == 'name':
        name = 'opaque_source_0'
    elif bad == 'old_wrapper_name':
        name = name.replace('gate_up_proj', 'base_layer')
    elif bad == 'shape':
        value = value[:, :, :-1]
    else:
        value = value.float()
    with pytest.raises(ValueError):
        adapter.consume_weights([(name, value)])


def test_no_weight_adapter_fails_before_consuming_stream(monkeypatch):
    monkeypatch.setitem(sys.modules, 'vllm.platforms', types.SimpleNamespace(current_platform=None))
    from twinkle.sampler.vllm_sampler.vllm_engine import VLLMEngine
    engine = object.__new__(VLLMEngine)
    engine.weight_adapter = None

    def forbidden():
        raise AssertionError('Transport must not be consumed')
        yield

    with pytest.raises(ValueError, match='explicit weight_adapter'):
        asyncio.run(engine.update_weights(forbidden(), lora_only=True, peft_config=fixture_config()['peft_config']))


def test_numeric_hook_precedes_split_and_constants_survive_updates(monkeypatch):
    import twinkle.sampler.vllm_sampler.weight_sync as sync
    from twinkle.sampler.vllm_sampler.vllm_worker_extension import TwinkleWorkerExtension
    events = []

    class NumericProbe(DeepSeekV4NvidiaLoraAdapter):

        def initialize(self, context, options):
            super().initialize(context, options)
            self.constant = torch.tensor(2.0)
            events.append('initialize')

        def adapt_numeric_pair(self, layer, projection, a, b):
            events.append('numeric')
            return a * self.constant, b

        def convert_layout(self, projection, a, b):
            events.append('layout')
            yield from super().convert_layout(projection, a, b)

    worker = object.__new__(TwinkleWorkerExtension)
    worker.device, worker.rank = torch.device('cuda'), 0
    worker.vllm_config = types.SimpleNamespace(
        lora_config=types.SimpleNamespace(max_lora_rank=2, lora_dtype=torch.bfloat16),
        model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(**fixture_config()['model_config'])),
        parallel_config=types.SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1, enable_expert_parallel=False))

    def create(config, context):
        adapter = NumericProbe()
        adapter.initialize(context, config['options'])
        return adapter

    monkeypatch.setattr(sync, 'create_weight_adapter', create)
    for _ in range(3):
        result, _ = worker._process_lora_weights(
            [item for g in groups() for item in g.items()], fixture_config()['peft_config'], ADAPTER_CONFIG)
        result = dict(result)
        adapter = worker._get_weight_adapter(ADAPTER_CONFIG)
        expected = dict(convert_groups(groups(), fixture_config()))
        assert all(
            torch.equal(value, expected[name] * (2 if '.lora_A.' in name else 1)) for name, value in result.items())
        assert adapter.constant.item() == 2 and not adapter.pending and not adapter.converted
    assert events.count('initialize') == 1
    assert events[1:] == ['numeric', 'layout'] * 12


def test_target_record_names_match_save_without_touching_base():
    from peft import LoraConfig

    from twinkle.model.multi_lora_target_parameters import TargetParameterLoraManager
    from twinkle.model.transformers.weight_sync import iter_lora_source_groups
    root = torch.nn.Module()
    root.model = torch.nn.Module()
    root.model.layers = torch.nn.ModuleList()
    for _ in range(2):
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        layer.mlp.experts = torch.nn.Module()
        layer.mlp.experts.is_transposed = False
        # Opposite registration order; export names must use the actual record.
        layer.mlp.experts.down_proj = torch.nn.Parameter(torch.randn(3, 8, 4, dtype=torch.bfloat16))
        layer.mlp.experts.gate_up_proj = torch.nn.Parameter(torch.randn(3, 8, 8, dtype=torch.bfloat16))
        root.model.layers.append(layer)
    manager = TargetParameterLoraManager(2, 4)
    manager.patch(root, sorted(TARGETS))
    for tenant, slot, rank in [('tenant_a', 'lora_0', 2), ('tenant_b', 'lora_1', 3)]:
        config = LoraConfig(r=rank, lora_alpha=8, target_modules=[], target_parameters=sorted(TARGETS))
        manager.acquire(tenant, slot, config)
    model = types.SimpleNamespace(
        model=root,
        _check_adapter_valid=lambda _: None,
        _lazy_wrap_model=lambda: None,
        multi_adapter=types.SimpleNamespace(
            target_parameter_manager=manager,
            find_lora_by_tenant=lambda t: types.SimpleNamespace(
                adapter_name=manager.tenant_to_slot[t], tenant_config=manager.tenant_configs[t])),
        hf_config=types.SimpleNamespace(to_dict=lambda: fixture_config()['model_config']),
        strategy=types.SimpleNamespace(gather_adapter_state_dict=lambda model, group, slot: group))
    for tenant in ('tenant_a', 'tenant_b'):
        state = {k: v for group in iter_lora_source_groups(model, tenant) for k, v in group.items()}
        saved = manager.get_state_dict(tenant)
        for wrapper in manager.wrappers:
            for side in ('A', 'B'):
                key = f'{wrapper.record.key}.lora_{side}.weight'
                assert torch.equal(saved[f'{wrapper.peft_key_prefix}.lora_{side}.weight'], state[key])
        metadata = fixture_config()
        metadata['peft_config'] = manager.tenant_configs[tenant].to_dict()
        assert len(dict(convert_groups([state], metadata))) == 36


@pytest.mark.parametrize('rank', [0, -1])
def test_actor_sends_raw_groups_and_non_senders_still_gather(monkeypatch, rank):
    import threading
    import twinkle.model.transformers.weight_sync as source_sync
    from twinkle.model.transformers.multi_lora_transformers import MultiLoraTransformersModel
    from twinkle.utils import Platform
    from twinkle.utils.framework import Torch
    metadata = 'tenant_a'
    collected, sent = [], []
    caller_thread = threading.get_ident()
    device_threads = []
    wrapped = []

    def collect(model, meta):
        assert meta == metadata
        assert wrapped == [True]
        assert not torch.is_grad_enabled()
        assert threading.get_ident() != caller_thread
        assert device_threads == [threading.get_ident()]
        for group in groups():
            collected.append(list(group))
            yield group

    async def send(weights):
        sent.extend(weights)

    monkeypatch.setattr(source_sync, 'iter_lora_source_groups', collect)
    monkeypatch.setattr(Platform, 'get_local_device', lambda: 'cpu')
    monkeypatch.setattr(Torch, 'set_device', lambda _: device_threads.append(threading.get_ident()))
    monkeypatch.setattr(Torch, 'is_gpu_available', lambda: False)
    engine = types.SimpleNamespace(rank=rank, send_weights=send)
    from twinkle.model.transformers.transformers import TransformersModel
    assert 'send_weights' not in MultiLoraTransformersModel.__dict__
    model = types.SimpleNamespace(_get_or_create_checkpoint_engine=lambda: engine,
                                  _check_adapter_valid=lambda name: None,
                                  _lazy_wrap_model=lambda: wrapped.append(True))
    model._export_lora_weights = types.MethodType(MultiLoraTransformersModel._export_lora_weights, model)
    report = TransformersModel.send_weights.__wrapped__(model, adapter_name=metadata, lora_only=True)
    assert len(collected) == 4
    assert report is None
    expected = {k: v for group in groups() for k, v in group.items()}
    assert len(sent) == (8 if rank == 0 else 0)
    for name, value in sent:
        assert '.ffn.' not in name and value.ndim == 3
        assert torch.equal(value, expected[name])


@pytest.mark.parametrize('mode', ['base', 'merged', 'peft', 'source_lora'])
@pytest.mark.parametrize('failure', [None, 'export', 'send'])
def test_parent_send_reuses_transport_and_propagates_errors(monkeypatch, mode, failure):
    import peft.utils
    import twinkle.model.transformers.transformers as model_module
    import twinkle.model.transformers.weight_sync as source_sync
    from twinkle.model.transformers.transformers import TransformersModel
    from twinkle.utils import Platform
    from twinkle.utils.framework import Torch

    monkeypatch.setattr(Platform, 'get_local_device', lambda: 'cpu')
    monkeypatch.setattr(Torch, 'set_device', lambda _: None)
    monkeypatch.setattr(Torch, 'is_gpu_available', lambda: False)
    sent, exports, merges = [], [], []
    value = torch.ones(2, dtype=torch.bfloat16)
    base_name = 'base_model.model.model.q_proj.base_layer.weight'
    lora_name = 'base_model.model.model.q_proj.lora_A.weight'
    source_name = 'model.layers.0.mlp.experts.gate_up_proj.lora_A.weight'
    lora_only = mode == 'source_lora'

    class FakePeft:
        def state_dict(self):
            assert not lora_only
            exports.append('base')
            if failure == 'export':
                raise RuntimeError('export failure')
            return {base_name: value, lora_name: value}

        def merge_adapter(self):
            merges.append('merge')

        def unmerge_adapter(self):
            merges.append('unmerge')

    base_model = FakePeft()

    def export_peft(model, adapter_name):
        assert mode == 'peft' and model is base_model and adapter_name == 'default'
        exports.append('peft')
        if failure == 'export':
            raise RuntimeError('export failure')
        return {lora_name: value}

    def export_source(model, adapter_name):
        assert lora_only and adapter_name == 'default'
        exports.append('source_lora')
        if failure == 'export':
            raise RuntimeError('export failure')
        yield {source_name: value}

    def unwrap(model):
        assert not lora_only  # Raw LoRA must never inspect the full base state.
        return base_model

    monkeypatch.setattr(model_module, 'PeftModel', FakePeft)
    monkeypatch.setattr(peft.utils, 'get_peft_model_state_dict', export_peft)
    monkeypatch.setattr(source_sync, 'iter_lora_source_groups', export_source)
    monkeypatch.setattr(Torch, 'to_local_tensor', lambda tensor: tensor)

    async def send(weights):
        if failure == 'send':
            raise RuntimeError('send failure')
        sent.extend(weights)

    model = types.SimpleNamespace(
        model=base_model, strategy=types.SimpleNamespace(unwrap_model=unwrap),
        _get_default_group=lambda: 'default', _check_adapter_valid=lambda name: None,
        _lazy_wrap_model=lambda: None,
        _get_or_create_checkpoint_engine=lambda: types.SimpleNamespace(rank=0, send_weights=send))
    from twinkle.model.transformers.multi_lora_transformers import MultiLoraTransformersModel
    method = MultiLoraTransformersModel._export_lora_weights if lora_only else TransformersModel._export_lora_weights
    model._export_lora_weights = types.MethodType(method, model)
    kwargs = dict(lora_only=lora_only, base_sync_done=mode in ('merged', 'peft'),
                  merge_and_sync=mode == 'merged')
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            TransformersModel.send_weights.__wrapped__(model, **kwargs)
    else:
        report = TransformersModel.send_weights.__wrapped__(model, **kwargs)
        assert len(sent) == 1 and torch.equal(sent[0][1], value)
        assert sent[0][0] == {
            'base': 'model.q_proj.base_layer.weight',
            'merged': 'model.q_proj.weight',
            'peft': 'model.q_proj.lora_A.weight',
            'source_lora': source_name,
        }[mode]
        assert report is None
        assert merges == (['merge', 'unmerge'] if mode == 'merged' else [])
    expected_export = 'base' if mode in ('base', 'merged') else mode
    # Legacy base/PEFT extraction occurs before starting the sending thread.
    assert exports == ([] if failure == 'send' and mode in ('merged', 'source_lora') else [expected_export])


def test_conversion_failure_releases_staging_and_does_not_install(monkeypatch):
    from twinkle.sampler.vllm_sampler.vllm_worker_extension import TwinkleWorkerExtension
    worker = object.__new__(TwinkleWorkerExtension)
    adapter = new_adapter(fixture_config())
    worker._get_weight_adapter = lambda _: adapter
    worker.add_lora = lambda _: pytest.fail('Must not install an invalid adapter')
    def fail(*args):
        raise RuntimeError('injected converter failure')
    monkeypatch.setattr(adapter, 'adapt_numeric_pair', fail)
    with pytest.raises(RuntimeError, match='converter failure'):
        worker._load_weights(list(groups()[0].items()), fixture_config()['peft_config'],
                             False, lora_only=True, weight_adapter=ADAPTER_CONFIG)
    assert not adapter.pending and not adapter.converted


@pytest.mark.parametrize('npu', [False, True])
def test_grpo_example_uses_none_returning_sync(monkeypatch, tmp_path, npu):
    from cookbook.rl.grpo import dsv4_lora_h800 as example
    monkeypatch.setenv('REPORT_DIR', str(tmp_path / 'reports'))
    monkeypatch.setenv('STEPS', '1')
    monkeypatch.setenv('BATCH_SIZE', '1')
    monkeypatch.setenv('NUM_GENERATIONS', '2')
    monkeypatch.delenv('FINAL_CHECKPOINT_DIR', raising=False)
    monkeypatch.setattr(example, 'local_gsm8k', lambda: [{'user_data': 'answer'}])
    monkeypatch.setattr(example, 'GSM8KProcessor', lambda: types.SimpleNamespace(preprocess=lambda row: row))
    monkeypatch.setattr(example, 'GSM8KAccuracyReward', lambda: lambda inputs: [0, 1])
    monkeypatch.setattr(example, 'GRPOAdvantage', lambda: lambda *args, **kwargs: torch.tensor([-1, 1]))
    events = []
    feature = {'input_ids': [1, 2]}
    sequence = types.SimpleNamespace(
        tokens=[2], logprobs=[[(2, -0.5)]], decoded='answer', new_input_feature=feature)

    def sync(**kwargs):
        events.append('sync')
        assert kwargs == dict(merge_and_sync=False, lora_only=True, adapter_name=example.TENANT)
        return None

    def sample(*args):
        events.append('sample')
        return [types.SimpleNamespace(sequences=[sequence, sequence], prompt_token_ids=[1])]

    def train(**kwargs):
        events.append('train')
        assert kwargs['inputs'] == [feature, feature] and kwargs['old_logps'] == [[-0.5], [-0.5]]

    model = types.SimpleNamespace(
        device_mesh=types.SimpleNamespace(data_world_size=1), forward_backward=train,
        clip_grad_and_step=lambda **kwargs: events.append('step'))
    monkeypatch.setattr(example, 'build_workers', lambda: (
        model, types.SimpleNamespace(sample=sample), types.SimpleNamespace(sync_weights=sync)))
    if npu:
        from cookbook.rl.grpo import dsv4_lora_npu
        monkeypatch.setattr(dsv4_lora_npu, 'build_workers', example.build_workers)
        dsv4_lora_npu.main()
    else:
        example.main()
    assert events == ['sync', 'sample', 'train', 'step', 'sync']
    report = json.loads((tmp_path / 'reports' / 'round_0.json').read_text())
    final = json.loads((tmp_path / 'reports' / 'final_sync.json').read_text())
    assert report['sync_seconds'] >= 0 and report['training_step'] == 0
    assert final['sync_seconds'] >= 0 and final['training_step'] == 1
