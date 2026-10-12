"""CPU lifecycle tests; no real HCCL transfer or NPU memory validation."""
import importlib
import json
import types
import weakref

import pytest
import torch

from twinkle.checkpoint_engine.hccl_checkpoint_engine import HCCLCheckpointEngine, MasterMetadata
from twinkle.checkpoint_engine.mixin import CheckpointEngineMixin

hccl_module = importlib.import_module('twinkle.checkpoint_engine.hccl_checkpoint_engine')
mixin_module = importlib.import_module('twinkle.checkpoint_engine.mixin')


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.delenv('TWINKLE_CKPT_MEMORY_AUDIT', raising=False)
    events, buffers, communicators = [], [], []
    zeros, tensor = torch.zeros, torch.tensor

    def allocate(*args, **kwargs):
        kwargs['device'] = 'cpu'
        result = zeros(*args, **kwargs)
        buffers.append(weakref.ref(result))
        return result

    def empty_cache():
        assert all(reference() is None for reference in buffers)
        events.append(('empty_cache',))

    monkeypatch.setattr(torch, 'npu', types.SimpleNamespace(
        current_device=lambda: 0,
        synchronize=lambda device: events.append(('synchronize', device)),
        empty_cache=empty_cache), raising=False)
    monkeypatch.setattr(torch, 'zeros', allocate)
    monkeypatch.setattr(torch, 'tensor', lambda *args, **kwargs: tensor(
        *args, **dict(kwargs, device='cpu')))

    def open_channel(engine):
        engine._zmq_ctx = types.SimpleNamespace(term=lambda: events.append(('context_term',)))
        engine.socket = types.SimpleNamespace(close=lambda: events.append(('socket_close',)))

    def start_server(engine):
        engine.ip = '127.0.0.1'
        engine.zmq_port = 10000 + len(buffers)
        engine.dist_port = engine.zmq_port + 1
        open_channel(engine)

    def init_group(**kwargs):
        handle = object()
        communicator = types.SimpleNamespace(
            available=True,
            comm=handle,
            hccl=types.SimpleNamespace(hcclCommDestroy=lambda comm: events.append(('destroy', comm))),
            all_reduce=lambda signal: events.append(('all_reduce', kwargs['rank'])))
        communicators.append(communicator)
        return communicator

    monkeypatch.setattr(HCCLCheckpointEngine, '_start_zmq_server', start_server)
    monkeypatch.setattr(HCCLCheckpointEngine, '_connect_zmq_client', lambda engine, metadata: open_channel(engine))
    monkeypatch.setattr(hccl_module, 'stateless_init_process_group', init_group)
    return types.SimpleNamespace(events=events, buffers=buffers, communicators=communicators)


def _install_memory_audit(monkeypatch, runtime, *, keep_view=False):
    """Model active vs cached allocations; these are not real NPU counters."""
    records = []
    cache = {'after_empty_cache': None}

    def allocated():
        return sum(buffer.numel() * buffer.element_size() for ref in runtime.buffers
                   if (buffer := ref()) is not None)

    def reserved():
        return cache['after_empty_cache'] if cache['after_empty_cache'] is not None else len(runtime.buffers) * 32

    def empty_cache():
        cache['after_empty_cache'] = allocated()
        runtime.events.append(('empty_cache',))
        if not keep_view:
            assert allocated() == 0

    def log(format, *args):
        if format.startswith('[CKPT-MEM]'):
            records.append(json.loads(args[0]))
            runtime.events.append(('audit', records[-1]['stage']))

    monkeypatch.setenv('TWINKLE_CKPT_MEMORY_AUDIT', '1')
    monkeypatch.setenv('RANK', '9')
    monkeypatch.setattr(torch.npu, 'memory_allocated', lambda device: allocated(), raising=False)
    monkeypatch.setattr(torch.npu, 'memory_reserved', lambda device: reserved(), raising=False)
    monkeypatch.setattr(torch.npu, 'mem_get_info', lambda device: (1024 - reserved(), 1024), raising=False)
    monkeypatch.setattr(torch.npu, 'empty_cache', empty_cache)
    monkeypatch.setattr(hccl_module.logger, 'info', log)
    return records


def test_memory_audit_distinguishes_dropped_tensors_and_released_cache(monkeypatch, runtime):
    records = _install_memory_audit(monkeypatch, runtime)
    engine = HCCLCheckpointEngine(bucket_size=32, rebuild_group=True)
    engine.prepare()
    engine.init_process_group(-1, 3, None)
    engine.finalize()

    assert [record['stage'] for record in records] == [
        'prepare.before_allocate', 'prepare.after_allocate', 'finalize.before_release',
        'finalize.after_drop_refs', 'finalize.after_empty_cache']
    assert [record['allocated_bytes'] for record in records] == [0, 64, 64, 0, 0]
    assert [record['reserved_bytes'] for record in records] == [0, 64, 64, 64, 0]
    assert [record['free_bytes'] for record in records] == [1024, 960, 960, 960, 1024]
    assert records[2]['send_buf_bytes'] == records[2]['recv_buf_bytes'] == 32
    assert records[3]['send_buf_bytes'] == records[3]['recv_buf_bytes'] == 0
    assert all(record['rank'] == 9 and record['rebuild_group'] for record in records)
    assert records[2]['checkpoint_rank'] == -1 and records[3]['checkpoint_rank'] is None
    assert runtime.events.count(('synchronize', 0)) == 1  # Only the pre-existing finalize barrier.
    assert runtime.events.index(('synchronize', 0)) < runtime.events.index(('audit', 'finalize.before_release'))


def test_memory_audit_exposes_skipped_finalize(monkeypatch, runtime):
    records = _install_memory_audit(monkeypatch, runtime)
    engine = HCCLCheckpointEngine(bucket_size=32, rebuild_group=False)
    engine.prepare()
    engine.finalize()
    assert records[-1]['stage'] == 'finalize.skipped'
    assert not records[-1]['rebuild_group']
    assert records[-1]['allocated_bytes'] == 64
    assert records[-1]['send_buf_bytes'] == records[-1]['recv_buf_bytes'] == 32
    assert not any(event[0] in ('synchronize', 'empty_cache') for event in runtime.events)


def test_memory_audit_does_not_claim_zero_allocation_when_view_survives(monkeypatch, runtime):
    records = _install_memory_audit(monkeypatch, runtime, keep_view=True)
    engine = HCCLCheckpointEngine(bucket_size=32, rebuild_group=True)
    engine.prepare()
    held_view = engine.send_buf[:1]
    engine.finalize()
    assert engine.send_buf is None and engine.recv_buf is None
    assert held_view.numel() == 1
    assert records[-1]['send_buf_bytes'] == records[-1]['recv_buf_bytes'] == 0
    assert records[-1]['allocated_bytes'] == records[-1]['reserved_bytes'] == 32


@pytest.mark.parametrize('rank', [0, 1, -1])
def test_rebuild_releases_resources_for_twenty_updates(runtime, rank):
    engine = HCCLCheckpointEngine(bucket_size=32, rebuild_group=True)
    engine.is_master = rank == 0
    ports = []
    for _ in range(20):
        metadata = engine.prepare() or MasterMetadata('127.0.0.1', 10000, '127.0.0.1', 10001)
        ports.append(metadata.dist_port)
        engine.init_process_group(rank, 3, metadata)
        communicator = engine.pyhccl
        start = len(runtime.events)

        engine.finalize()

        cleanup = runtime.events[start:]
        assert cleanup[0] == ('synchronize', 0)
        if rank >= 0:
            assert cleanup[1] == ('destroy', communicator.comm)
            assert ('socket_close',) in cleanup and ('context_term',) in cleanup
        else:
            assert cleanup == [('synchronize', 0), ('empty_cache',)]
        assert cleanup[-1] == ('empty_cache',)
        assert engine.pyhccl is None and engine.socket is None and engine._zmq_ctx is None
        assert engine.send_buf is None and engine.recv_buf is None
        assert engine.rank is None and engine.world_size is None
        assert not engine._prepared and not engine._group_initialized
        assert all(reference() is None for reference in runtime.buffers)
        # A second finalize must not destroy an already destroyed communicator.
        completed = list(runtime.events)
        engine.finalize()
        assert runtime.events == completed

    assert len(runtime.buffers) == 40
    assert len(runtime.communicators) == (20 if rank >= 0 else 0)
    assert sum(event[0] == 'empty_cache' for event in runtime.events) == 20
    if rank == 0:
        assert len(set(ports)) == 20


@pytest.mark.parametrize('rank', [0, 1, -1])
def test_reuse_keeps_original_buffers_and_communicator(runtime, rank):
    engine = HCCLCheckpointEngine(bucket_size=32, rebuild_group=False)
    engine.is_master = rank == 0
    for _ in range(20):
        metadata = engine.prepare() or MasterMetadata('127.0.0.1', 10000, '127.0.0.1', 10001)
        engine.init_process_group(rank, 3, metadata)
        events = list(runtime.events)
        engine.finalize()
        assert runtime.events == events
        assert engine._prepared and engine._group_initialized
        assert engine.rank == rank

    assert len(runtime.buffers) == 2
    assert runtime.buffers[0]() is engine.send_buf
    assert runtime.buffers[1]() is engine.recv_buf
    assert len(runtime.communicators) == (1 if rank >= 0 else 0)
    assert not any(event[0] in ('destroy', 'empty_cache', 'synchronize') for event in runtime.events)


def test_destroy_failure_is_not_silently_ignored(runtime):
    engine = HCCLCheckpointEngine(bucket_size=32, rebuild_group=True)
    engine.is_master = True
    metadata = engine.prepare()
    engine.init_process_group(0, 2, metadata)
    communicator = engine.pyhccl

    def fail(comm):
        raise RuntimeError('HCCL destroy failed')

    communicator.hccl.hcclCommDestroy = fail
    with pytest.raises(RuntimeError, match='HCCL destroy failed'):
        engine.finalize()
    assert engine.pyhccl is communicator
    assert engine._prepared and engine._group_initialized
    assert engine.send_buf is not None and engine.recv_buf is not None
    assert not any(event[0] in ('socket_close', 'context_term', 'empty_cache') for event in runtime.events)


def test_disabled_communicator_has_no_native_handle_to_destroy(runtime):
    engine = HCCLCheckpointEngine(bucket_size=32, rebuild_group=True)
    engine.prepare()
    # PyHcclCommunicator does not allocate a native handle for world_size=1.
    engine.pyhccl = types.SimpleNamespace(available=False)
    engine.finalize()
    assert engine.pyhccl is None and engine.send_buf is None and engine.recv_buf is None
    assert runtime.events == [('synchronize', 0), ('empty_cache',)]


@pytest.mark.parametrize('platform', ['GPU', 'NPU'])
@pytest.mark.parametrize('rebuild_group', [False, True])
def test_prepare_configures_both_backends_without_environment(monkeypatch, platform, rebuild_group):
    monkeypatch.setattr(mixin_module.Platform, 'get_platform', lambda: types.SimpleNamespace(__name__=platform))
    calls = []

    def create(bucket_size, **kwargs):
        calls.append((bucket_size, kwargs))
        return types.SimpleNamespace(prepare=lambda: 'metadata', **kwargs)

    import twinkle.checkpoint_engine as engines
    monkeypatch.setattr(engines, 'HCCLCheckpointEngine', create)
    monkeypatch.setattr(engines, 'NCCLCheckpointEngine', create)
    worker = CheckpointEngineMixin()
    worker._bucket_size = 123
    prepare = CheckpointEngineMixin.prepare_checkpoint_engine.__wrapped__
    for _ in range(20):
        assert prepare(worker, [True], rebuild_group=rebuild_group) == 'metadata'
        engine = worker._get_or_create_checkpoint_engine()
        assert engine.rebuild_group is rebuild_group and engine.is_master
    assert calls == [(123, {'rebuild_group': rebuild_group})]


@pytest.mark.parametrize('platform', ['GPU', 'NPU'])
def test_prepare_defaults_to_reuse(monkeypatch, platform):
    monkeypatch.setattr(mixin_module.Platform, 'get_platform', lambda: types.SimpleNamespace(__name__=platform))

    def create(bucket_size, **kwargs):
        return types.SimpleNamespace(prepare=lambda: None, **kwargs)

    import twinkle.checkpoint_engine as engines
    monkeypatch.setattr(engines, 'HCCLCheckpointEngine', create)
    monkeypatch.setattr(engines, 'NCCLCheckpointEngine', create)
    worker = CheckpointEngineMixin()
    CheckpointEngineMixin.prepare_checkpoint_engine.__wrapped__(worker, False)
    assert not worker._checkpoint_engine.rebuild_group
