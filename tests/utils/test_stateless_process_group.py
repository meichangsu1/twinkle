# Copyright (c) ModelScope Contributors. All rights reserved.
"""Exercise the real CPU TCPStore transport without importing accelerator runtimes."""
import gc
import os
import resource
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import ModuleType

import json
import pytest


@dataclass
class MetadataGroup:
    """Current vLLM constructor: TCPStore owns the listener, with no socket field."""
    rank: int
    world_size: int
    store: object
    data_expiration_seconds: int = 3600


@dataclass
class LegacyMetadataGroup:
    rank: int
    world_size: int
    store: object
    socket: object
    data_expiration_seconds: int = 3600


class Communicator:
    def __init__(self, group, device):
        assert device == 'cpu'
        self.group = group


def metadata_and_communicator_stubs(api='current'):
    utils = ModuleType('vllm.distributed.utils')
    utils.StatelessProcessGroup = {'current': MetadataGroup, 'legacy': LegacyMetadataGroup}[api]
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


def test_failed_store_creation_closes_transferred_descriptor(monkeypatch):
    import torch.distributed

    from twinkle.utils.torch_utils import stateless_init_process_group

    for name, module in metadata_and_communicator_stubs().items():
        monkeypatch.setitem(sys.modules, name, module)
    descriptor = None

    def fail_store(**kwargs):
        nonlocal descriptor
        descriptor = kwargs['master_listen_fd']
        raise RuntimeError('store construction failed')

    monkeypatch.setattr(torch.distributed, 'TCPStore', fail_store)
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
        with pytest.raises(RuntimeError, match='store construction failed'):
            stateless_init_process_group('127.0.0.1', port, 0, 1, device='cpu', listen_socket=listener)
        assert listener.fileno() == -1
        with pytest.raises(OSError):
            os.fstat(descriptor)


if __name__ == '__main__':
    run_worker(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5])
