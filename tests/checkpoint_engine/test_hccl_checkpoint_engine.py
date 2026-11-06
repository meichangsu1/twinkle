# Copyright (c) ModelScope Contributors. All rights reserved.
"""Native HCCL group setup and full-weight transfer (CPU substitutes for NPU only)."""
from __future__ import annotations

import asyncio
import builtins
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist

from twinkle.checkpoint_engine import hccl_checkpoint_engine as mod

META = mod.MasterMetadata(zmq_ip='127.0.0.1', zmq_port=111, dist_ip='127.0.0.1', dist_port=222)


@pytest.fixture
def backend(monkeypatch):
    groups = []

    class ProcessGroupHCCL:
        Options = SimpleNamespace

        def __init__(self, store, rank, world_size, options):
            self.store = store
            self.rank = rank
            self.world_size = world_size
            self.options = options
            self.work = SimpleNamespace(wait=Mock())
            self.broadcast = Mock(return_value=self.work)
            groups.append(self)

    for name in ('torch_npu', 'torch_npu._C', 'torch_npu._C._distributed_c10d'):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    native = sys.modules['torch_npu._C._distributed_c10d']
    native.ProcessGroupHCCL = ProcessGroupHCCL

    npu = SimpleNamespace(current_device=lambda: 3, set_device=Mock(), synchronize=Mock())
    monkeypatch.setattr(torch, 'npu', npu, raising=False)
    original_zeros = torch.zeros

    def zeros(*args, **kwargs):
        if str(kwargs.get('device', '')).startswith('npu'):
            kwargs['device'] = 'cpu'
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, 'zeros', zeros)
    store = Mock()
    monkeypatch.setattr(dist, 'TCPStore', store)
    ports = Mock()
    monkeypatch.setattr(mod, 'ensure_hccl_socket_env', ports)
    connect_client = mod.HCCLCheckpointEngine._connect_zmq_client
    monkeypatch.setattr(mod.HCCLCheckpointEngine, '_connect_zmq_client', Mock())
    monkeypatch.setenv('TWINKLE_DIST_TIMEOUT_SECONDS', '17')
    monkeypatch.setenv('TWINKLE_CKPT_HCCL_META_TIMEOUT_S', '5')

    original_import = builtins.__import__

    def independent_import(name, *args, **kwargs):
        if name.split('.')[0] in ('vllm', 'vllm_ascend', 'sglang'):
            raise AssertionError(f'Weight transport must not import {name}')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', independent_import)
    return SimpleNamespace(
        groups=groups, native=native, npu=npu, store=store, ports=ports, connect_client=connect_client)


@pytest.mark.parametrize('rank', [0, 1])
def test_init_uses_native_group_without_a_sampler_dependency(backend, rank):
    engine = mod.HCCLCheckpointEngine(bucket_size=16)
    engine.init_process_group(rank, 2, META)

    backend.store.assert_called_once_with(
        host_name=META.dist_ip, port=META.dist_port, world_size=2, is_master=(rank == 0),
        timeout=timedelta(seconds=300), wait_for_workers=True, use_libuv=False)
    group, = backend.groups
    assert group.store is backend.store.return_value
    assert (group.rank, group.world_size) == (rank, 2)
    assert group.options.global_ranks_in_group == [0, 1]
    assert group.options.group_id == 'twinkle_ckpt:127.0.0.1:222'
    assert group.options._timeout == timedelta(seconds=17)
    backend.ports.assert_called_once_with(META.dist_port)
    backend.npu.set_device.assert_called_once_with(3)
    tensors, options = group.broadcast.call_args.args
    assert options.rootRank == 0 and tensors[0].dtype == torch.int32
    group.work.wait.assert_called_once_with()
    backend.npu.synchronize.assert_called_once_with(3)
    assert engine._group_initialized


def test_nonparticipating_trainer_does_not_create_a_group(backend):
    engine = mod.HCCLCheckpointEngine(bucket_size=16)
    engine.init_process_group(-1, 3, META)

    assert engine.rank == -1 and engine._group_initialized
    assert not backend.groups
    backend.store.assert_not_called()
    backend.npu.set_device.assert_not_called()


def test_finalize_preserves_persistent_group_and_close_releases_it(backend, monkeypatch):
    engine = mod.HCCLCheckpointEngine(bucket_size=16, rebuild_group=False)
    engine.init_process_group(0, 2, META)
    group = engine._pg
    engine.finalize()
    engine.init_process_group(0, 2, META)

    assert engine._pg is group and len(backend.groups) == 1
    group.broadcast.assert_called_once()
    destroy = Mock(side_effect=AssertionError('Cannot destroy an unregistered group via dist'))
    monkeypatch.setattr(dist, 'destroy_process_group', destroy)
    socket = engine.socket = Mock()
    context = engine._zmq_ctx = Mock()
    engine.close()

    socket.close.assert_called_once()
    context.term.assert_called_once()
    assert engine._pg is None and engine._store is None
    assert not engine._group_initialized and engine.rank is None
    assert engine.rebuild_group is False
    destroy.assert_not_called()
    engine.close()


def test_finalize_allows_group_recreation(backend):
    engine = mod.HCCLCheckpointEngine(bucket_size=16, rebuild_group=True)
    engine.init_process_group(0, 2, META)
    engine.finalize()
    engine.init_process_group(0, 2, META)

    assert len(backend.groups) == 2
    assert engine._pg is backend.groups[1]


def test_broadcast_waits_on_the_raw_group_and_propagates_failures(backend):
    engine = mod.HCCLCheckpointEngine(bucket_size=16)
    engine.init_process_group(0, 2, META)
    group = engine._pg
    group.broadcast.reset_mock()
    group.work.wait.reset_mock()
    tensor = torch.ones(4)
    engine._broadcast(tensor, src=1)

    tensors, options = group.broadcast.call_args.args
    assert tensors[0] is tensor and options.rootRank == 1
    group.work.wait.assert_called_once_with()
    group.work.wait.side_effect = RuntimeError('collective failed')
    with pytest.raises(RuntimeError, match='collective failed'):
        engine._broadcast(tensor)


@pytest.mark.skipif(not dist.is_gloo_available(), reason='CPU loopback requires Gloo')
def test_chunked_weight_roundtrip_reuses_group_without_changing_training_group(backend, monkeypatch):
    """Real TCPStore/Gloo/ZMQ transfer, including split weights, empty tensors and two versions."""
    if dist.is_initialized():
        pytest.skip('This test owns its single-rank training group')
    # The fixture replaced TCPStore; recover the real binding for this loopback.
    monkeypatch.setattr(dist, 'TCPStore', torch._C._distributed_c10d.TCPStore)
    monkeypatch.setenv('TWINKLE_DIST_TIMEOUT_SECONDS', '5')

    def gloo_group(store, rank, world_size, options):
        return dist.ProcessGroupGloo(store, rank, world_size, options._timeout)

    gloo_group.Options = SimpleNamespace
    backend.native.ProcessGroupHCCL = gloo_group
    monkeypatch.setattr(mod, 'find_node_ip', lambda: '127.0.0.1')
    monkeypatch.setattr(mod.HCCLCheckpointEngine, '_connect_zmq_client', backend.connect_client)
    original_to = torch.Tensor.to

    def to(tensor, *args, **kwargs):
        device = args[0] if args else kwargs.get('device')
        if isinstance(device, str) and device.startswith('npu'):
            return tensor
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'to', to)
    sender = mod.HCCLCheckpointEngine(bucket_size=16, rebuild_group=False)
    sender.is_master = True
    receiver = mod.HCCLCheckpointEngine(bucket_size=16, rebuild_group=False)
    metadata = sender.prepare()
    receiver.prepare()

    expected = [{
        'matrix': torch.arange(7, dtype=torch.float32).reshape(1, 7) + version,
        'bias': torch.arange(3, dtype=torch.float32) + version,
        'empty': torch.empty(0, 2),
    } for version in (0, 10)]

    def send():
        try:
            sender.init_process_group(0, 2, metadata)
            group = sender._pg
            for weights in expected:
                asyncio.run(sender.send_weights(iter(weights.items())))
                sender.finalize()
                sender.init_process_group(0, 2, metadata)
                assert sender._pg is group
        finally:
            sender.close()

    def receive():
        async def collect():
            return {name: tensor.clone() async for name, tensor in receiver.receive_weights()}

        try:
            receiver.init_process_group(1, 2, metadata)
            group = receiver._pg
            for weights in expected:
                received = asyncio.run(collect())
                assert received.keys() == weights.keys()
                for name, tensor in weights.items():
                    torch.testing.assert_close(received[name], tensor, rtol=0, atol=0)
                receiver.finalize()
                receiver.init_process_group(1, 2, metadata)
                assert receiver._pg is group
        finally:
            receiver.close()

    try:
        dist.init_process_group('gloo', store=dist.HashStore(), rank=0, world_size=1)
        training_group = dist.group.WORLD
        registered_groups = set(dist.distributed_c10d._world.pg_map)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(send), pool.submit(receive)]
            for future in futures:
                future.result(timeout=20)
        assert dist.group.WORLD is training_group
        assert set(dist.distributed_c10d._world.pg_map) == registered_groups
        tensor = torch.tensor([7.0])
        dist.all_reduce(tensor)
        assert tensor.item() == 7
    finally:
        sender.close()
        receiver.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def _npu_roundtrip_worker(rank, training_port, metadata_queue):
    import torch_npu  # noqa: F401

    torch.npu.set_device(rank)
    engine = mod.HCCLCheckpointEngine(bucket_size=16, rebuild_group=False)
    engine.is_master = rank == 0
    try:
        dist.init_process_group(
            'hccl', init_method=f'tcp://127.0.0.1:{training_port}', rank=rank, world_size=2,
            timeout=timedelta(seconds=30))
        training_group = dist.group.WORLD
        metadata = engine.prepare()
        if rank == 0:
            metadata_queue.put(metadata)
        else:
            metadata = metadata_queue.get(timeout=30)
        engine.init_process_group(rank, 2, metadata)
        group = engine._pg

        for version in (0, 10):
            expected = {
                'matrix': torch.arange(7, dtype=torch.float32).reshape(1, 7) + version,
                'bias': torch.arange(3, dtype=torch.float32) + version,
                'empty': torch.empty(0, 2),
            }
            if rank == 0:
                weights = ((name, tensor.to(f'npu:{rank}')) for name, tensor in expected.items())
                asyncio.run(engine.send_weights(weights))
            else:
                async def collect():
                    return {name: tensor.cpu() async for name, tensor in engine.receive_weights()}

                received = asyncio.run(collect())
                assert received.keys() == expected.keys()
                for name, tensor in expected.items():
                    torch.testing.assert_close(received[name], tensor, rtol=0, atol=0)
            engine.finalize()
            engine.init_process_group(rank, 2, metadata)
            assert engine._pg is group
            assert dist.group.WORLD is training_group
            signal = torch.ones(1, device=f'npu:{rank}')
            dist.all_reduce(signal)
            assert signal.item() == 2
        del group
    finally:
        engine.close()
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.accel(2)
def test_native_hccl_roundtrip_alongside_training_group(monkeypatch):
    """Two-NPU smoke without model weights, Ray or a rollout backend."""
    npu = getattr(torch, 'npu', None)
    if npu is None or not npu.is_available() or npu.device_count() < 2:
        pytest.skip('Requires two Ascend NPUs')
    monkeypatch.setenv('TWINKLE_DIST_TIMEOUT_SECONDS', '30')
    monkeypatch.setenv('TWINKLE_CKPT_HCCL_META_TIMEOUT_S', '30')
    context = torch.multiprocessing.get_context('spawn')
    queue = context.Queue()
    training_port = mod.find_free_port()
    processes = [context.Process(target=_npu_roundtrip_worker, args=(rank, training_port, queue))
                 for rank in range(2)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=60)
        assert [process.exitcode for process in processes] == [0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        queue.close()
