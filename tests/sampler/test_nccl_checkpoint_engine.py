"""CPU lifecycle mocks; not a CUDA/NCCL transport acceptance test."""
import importlib
import types
import weakref

import pytest
import torch

from twinkle.checkpoint_engine.nccl_checkpoint_engine import NCCLCheckpointEngine, MasterMetadata

nccl_module = importlib.import_module('twinkle.checkpoint_engine.nccl_checkpoint_engine')


@pytest.mark.parametrize('rank', [0, 1, -1])
@pytest.mark.parametrize('rebuild_group', [False, True])
def test_nccl_prepare_init_finalize_for_twenty_updates(monkeypatch, rank, rebuild_group):
    buffers, groups, stores, closed = [], [], [], []
    zeros = torch.zeros

    def allocate(*args, **kwargs):
        result = zeros(*args, **dict(kwargs, device='cpu'))
        if result.numel() == 32:
            buffers.append(weakref.ref(result))
        return result

    class Resource:
        pass

    def create_resource(collection):
        resource = Resource()
        collection.append(weakref.ref(resource))
        return resource

    def start_server(engine):
        engine.ip = '127.0.0.1'
        engine.listen_port = 10000 + len(buffers)
        engine.socket = types.SimpleNamespace(close=lambda: closed.append(rank))

    def connect_client(engine, metadata):
        engine.socket = types.SimpleNamespace(close=lambda: closed.append(rank))

    monkeypatch.setattr(torch, 'zeros', allocate)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(nccl_module, 'find_free_port', lambda: 11000 + len(buffers))
    monkeypatch.setattr(nccl_module, '_pg_broadcast', lambda *args, **kwargs: None)
    monkeypatch.setattr(nccl_module.dist, 'TCPStore', lambda **kwargs: create_resource(stores))
    monkeypatch.setattr(nccl_module.dist, 'ProcessGroupNCCL',
                        lambda *args: create_resource(groups), raising=False)
    monkeypatch.setattr(NCCLCheckpointEngine, '_start_zmq_server', start_server)
    monkeypatch.setattr(NCCLCheckpointEngine, '_connect_zmq_client', connect_client)
    engine = NCCLCheckpointEngine(bucket_size=32, rebuild_group=rebuild_group)
    engine.is_master = rank == 0
    ports = []

    for _ in range(20):
        metadata = engine.prepare() or MasterMetadata('127.0.0.1', 10000, '127.0.0.1', 11000)
        ports.append(metadata.nccl_store_port)
        engine.init_process_group(rank, 3, metadata)
        engine.finalize()

        if rebuild_group:
            assert not engine._prepared and not engine._group_initialized
            assert engine.send_buf is None and engine.recv_buf is None
            assert engine._pg is None and engine._store is None and engine.socket is None
            assert engine.rank is None and engine.world_size is None
            assert all(reference() is None for reference in buffers + groups + stores)
            # Finalization is idempotent, including socket cleanup.
            previous_closed = list(closed)
            engine.finalize()
            assert closed == previous_closed
        else:
            assert engine._prepared and engine._group_initialized
            assert buffers[0]() is engine.send_buf and buffers[1]() is engine.recv_buf
            assert engine.rank == rank
            if rank >= 0:
                assert groups[0]() is engine._pg and stores[0]() is engine._store
            assert not closed

    cycles = 20 if rebuild_group else 1
    assert len(buffers) == 2 * cycles
    assert len(groups) == len(stores) == (cycles if rank >= 0 else 0)
    assert len(closed) == (20 if rebuild_group and rank >= 0 else 0)
    if rank == 0:
        assert len(set(ports)) == cycles
