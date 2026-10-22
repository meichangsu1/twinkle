# Copyright (c) ModelScope Contributors. All rights reserved.
"""vLLM constructor compatibility and TCPStore listening-fd ownership."""
import gc
import json
import os
import resource
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch.distributed as dist

from twinkle.utils.torch_utils import stateless_init_process_group


@dataclass
class LegacyGroup:
    rank: int
    world_size: int
    store: object
    socket: object
    data_expiration_seconds: int = 3600


@dataclass
class ModernGroup:
    rank: int
    world_size: int
    store: object
    data_expiration_seconds: int = 3600


@pytest.fixture
def vllm_modules(monkeypatch):
    """Replace only GPU/NPU dependencies; the store can still be real PyTorch."""
    names = [
        'vllm', 'vllm.distributed', 'vllm.distributed.utils',
        'vllm.distributed.device_communicators', 'vllm.distributed.device_communicators.pynccl',
        'vllm_ascend', 'vllm_ascend.distributed', 'vllm_ascend.distributed.device_communicators',
        'vllm_ascend.distributed.device_communicators.pyhccl',
    ]
    modules = {}
    for name in names:
        module = ModuleType(name)
        module.__path__ = []
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)

    class Communicator:
        def __init__(self, group, device):
            self.group = group
            self.device = device

    modules['vllm.distributed.device_communicators.pynccl'].PyNcclCommunicator = Communicator
    modules['vllm_ascend.distributed.device_communicators.pyhccl'].PyHcclCommunicator = Communicator
    from twinkle.utils import platforms
    ports = []
    monkeypatch.setattr(platforms, 'ensure_hccl_socket_env', ports.append)
    return SimpleNamespace(utils=modules['vllm.distributed.utils'], ports=ports)


@pytest.mark.parametrize('group_cls', [LegacyGroup, ModernGroup], ids=['legacy', 'modern'])
@pytest.mark.parametrize('backend', ['nccl', 'hccl'])
@pytest.mark.parametrize('rank,mode', [(0, 'automatic'), (0, 'socket'), (0, 'fd'), (1, 'automatic')],
                         ids=['master-created', 'master-provided', 'master-fd', 'receiver'])
def test_constructor_compatibility(monkeypatch, vllm_modules, group_cls, backend, rank, mode):
    vllm_modules.utils.StatelessProcessGroup = group_cls
    stores = []

    def create_store(**kwargs):
        store = SimpleNamespace(options=kwargs)
        stores.append(store)
        return store

    monkeypatch.setattr(dist, 'TCPStore', create_store)
    listener = socket.socket() if mode != 'automatic' else None
    listen_options = {}
    if listener is not None:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
    port = listener.getsockname()[1] if listener is not None else 0
    if mode == 'socket':
        listen_options['listen_socket'] = listener
    elif mode == 'fd':
        listen_options['listen_fd'] = listener.detach()
    communicator = None
    try:
        communicator = stateless_init_process_group(
            '127.0.0.1', port, rank, 2, device=0, backend=backend, **listen_options)
        group = communicator.group
        assert (group.rank, group.world_size, group.data_expiration_seconds) == (rank, 2, 3600)
        assert group.store is stores[0] and communicator.device == 0
        options = stores[0].options
        assert options['host_name'] == '127.0.0.1' and options['port'] == port
        assert options['is_master'] == (rank == 0)
        assert options['world_size'] == 2 and options['use_libuv'] is False
        assert options['timeout'] == timedelta(seconds=300)
        assert vllm_modules.ports == ([port] if backend == 'hccl' else [])
        if rank == 0:
            fd = options['master_listen_fd']
            os.fstat(fd)  # Still open after the helper's local socket goes out of scope.
            if group_cls is LegacyGroup:
                assert group.socket is None  # Never keep a second fd owner on legacy vLLM either.
            else:
                assert not hasattr(group, 'socket')
            if listener is not None:
                assert listener.fileno() == -1  # TCPStore owns this descriptor now.
            if mode == 'fd':
                assert fd == listen_options['listen_fd']
        else:
            assert options['master_listen_fd'] is None
            if group_cls is LegacyGroup:
                assert group.socket is None
    finally:
        # The stand-in store does not own descriptors, unlike a real TCPStore.
        if communicator is not None and rank == 0:
            socket.close(stores[0].options['master_listen_fd'])
        if listener is not None:
            listener.close()


def test_store_failure_closes_transferred_fd(monkeypatch, vllm_modules):
    vllm_modules.utils.StatelessProcessGroup = ModernGroup
    transferred = []

    def failed_store(**kwargs):
        transferred.append(kwargs['master_listen_fd'])
        raise RuntimeError('store initialization failed')

    monkeypatch.setattr(dist, 'TCPStore', failed_store)
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        with pytest.raises(RuntimeError, match='store initialization failed'):
            stateless_init_process_group('127.0.0.1', listener.getsockname()[1], 0, 2,
                                         device=0, backend='hccl', listen_socket=listener)
        assert listener.fileno() == -1
        with pytest.raises(OSError):
            os.fstat(transferred[0])


@pytest.mark.parametrize('already_closed', [False, True])
def test_store_failure_closes_provided_fd_without_masking_error(monkeypatch, vllm_modules, already_closed):
    vllm_modules.utils.StatelessProcessGroup = LegacyGroup

    def failed_store(**kwargs):
        if already_closed:
            socket.close(kwargs['master_listen_fd'])
        raise RuntimeError('original store error')

    monkeypatch.setattr(dist, 'TCPStore', failed_store)
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        port, fd = listener.getsockname()[1], listener.detach()
        with pytest.raises(RuntimeError, match='original store error'):
            stateless_init_process_group('127.0.0.1', port, 0, 2, device=0, listen_fd=fd)
        with pytest.raises(OSError):
            os.fstat(fd)


class Communicator:
    def __init__(self, group, device):
        assert device == 'cpu'
        self.group = group


def metadata_and_communicator_stubs(api='current'):
    utils = ModuleType('vllm.distributed.utils')
    utils.StatelessProcessGroup = {'current': ModernGroup, 'legacy': LegacyGroup}[api]
    modules = {utils.__name__: utils}
    for module_name, class_name in (
            ('vllm.distributed.device_communicators.pynccl', 'PyNcclCommunicator'),
            ('vllm_ascend.distributed.device_communicators.pyhccl', 'PyHcclCommunicator')):
        module = ModuleType(module_name)
        setattr(module, class_name, Communicator)
        modules[module_name] = module
    return modules


def run_worker(rank, port, mode, backend, api):
    # A regression may abort in the native TCPStore thread; keep that isolated
    # from pytest and avoid writing a large core dump.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    from torch.distributed import TCPStore

    from twinkle.utils.torch_utils import stateless_init_process_group

    sys.modules.update(metadata_and_communicator_stubs(api))
    listener = None
    kwargs = {}
    if rank == 0 and mode != 'automatic':
        listener = socket.socket()
        listener.bind(('127.0.0.1', port))
        listener.listen()
        if mode == 'socket':
            kwargs['listen_socket'] = listener
        else:
            kwargs['listen_fd'] = listener.detach()
    communicator = stateless_init_process_group('127.0.0.1', port, rank, 3, device='cpu', backend=backend, **kwargs)
    if listener is not None:
        assert listener.fileno() == -1
        listener.close()
    gc.collect()
    store = communicator.group.store
    store.set(f'value/{rank}', str(rank))
    store.wait([f'value/{peer}' for peer in range(3)])
    values = [store.get(f'value/{peer}').decode() for peer in range(3)]
    assert values == ['0', '1', '2']
    if rank == 0:
        # A new connection must still work after Python socket cleanup.
        probe = TCPStore('127.0.0.1', port, is_master=False, timeout=timedelta(seconds=10), use_libuv=False)
        probe.set('probe', 'alive')
        assert store.get('probe') == b'alive'
    store.set(f'finished/{rank}', 'yes')
    store.wait([f'finished/{peer}' for peer in range(3)])
    sys.stdout.write(json.dumps({'rank': rank, 'values': values}) + '\n')


@pytest.mark.parametrize(('mode', 'backend', 'api'), [
    (mode, backend, 'current') for mode in ('automatic', 'socket', 'fd') for backend in ('nccl', 'hccl')
] + [('automatic', 'hccl', 'legacy')])
def test_three_process_store_survives_python_socket_cleanup(mode, backend, api):
    root = Path(__file__).resolve().parents[2]
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        port = reservation.getsockname()[1]
    environment = {**os.environ, 'PYTHONPATH': str(root / 'src') + os.pathsep + os.environ.get('PYTHONPATH', '')}
    processes = []
    try:
        for rank in range(3):
            processes.append(subprocess.Popen(
                [sys.executable, '-u', str(Path(__file__).resolve()), str(rank), str(port), mode, backend, api],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment))
        deadline = time.monotonic() + 45
        for process in processes:
            output, errors = process.communicate(timeout=max(0.1, deadline - time.monotonic()))
            assert process.returncode == 0, output + errors
            assert json.loads(output)['values'] == ['0', '1', '2']
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)


if __name__ == '__main__':
    run_worker(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5])
