# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

from twinkle import DeviceGroup, Platform
from twinkle.infra._ray.resource_manager import ResourceManager, _gpu_placement_group_cpus


def test_gpu_placement_group_cpus_uses_per_process_default(monkeypatch):
    monkeypatch.delenv('TWINKLE_GPU_PG_CPUS_PER_PROC', raising=False)

    assert _gpu_placement_group_cpus(node_cpu=192, nproc_per_node=2) == 8


def test_gpu_placement_group_cpus_respects_node_cap(monkeypatch):
    monkeypatch.setenv('TWINKLE_GPU_PG_CPUS_PER_PROC', '16')

    assert _gpu_placement_group_cpus(node_cpu=192, nproc_per_node=8) == 48


def test_gpu_placement_group_cpus_can_be_configured(monkeypatch):
    monkeypatch.setenv('TWINKLE_GPU_PG_CPUS_PER_PROC', '8')

    assert _gpu_placement_group_cpus(node_cpu=192, nproc_per_node=2) == 16


@pytest.fixture
def ray_allocation(monkeypatch):
    """Return allocation metadata without starting Ray or probing accelerator hardware."""
    ray = ModuleType('ray')
    util = ModuleType('ray.util')
    placement = ModuleType('ray.util.placement_group')

    class PlacementGroup:

        def ready(self):
            return None

    class Remote:

        def __init__(self, function):
            self.function = function

        def options(self, **kwargs):
            return self

        def remote(self):
            return self.function()

    ray.remote = Remote
    ray.get = lambda value: value
    ray.get_runtime_context = lambda: SimpleNamespace(gcs_address='test-ray')
    ray.nodes = lambda: [{'NodeID': 'test-node', 'Resources': {'CPU': 64, 'NPU': 4, 'GPU': 4}}]
    ray.util = util
    util.placement_group = lambda bundles: PlacementGroup()
    placement.PlacementGroup = PlacementGroup
    for name, module in [('ray', ray), ('ray.util', util), ('ray.util.placement_group', placement)]:
        monkeypatch.setitem(sys.modules, name, module)
    platforms = {
        name: type(name, (), {'visible_device_env': staticmethod(lambda key=key: key)})
        for name, key in [('NPU', 'ASCEND_RT_VISIBLE_DEVICES'), ('GPU', 'CUDA_VISIBLE_DEVICES')]
    }
    monkeypatch.setattr(Platform, 'get_platform', lambda name: platforms[name])
    monkeypatch.setattr(Platform, 'get_node_rank', lambda: 0)
    return ray


@pytest.mark.parametrize('device_type, sampler_devices', [('NPU', [5, 6]), ('GPU', [6, 5])])
def test_worker_device_order_from_ray_allocation(monkeypatch, ray_allocation, device_type, sampler_devices):
    # Ray may hand the bundle back in an order that is invalid for a multi-NPU worker.
    env = 'ASCEND_RT_VISIBLE_DEVICES' if device_type == 'NPU' else 'CUDA_VISIBLE_DEVICES'
    monkeypatch.setenv(env, '1,4,6,5')
    groups = [
        DeviceGroup(name='model', ranks=[0, 1], device_type=device_type),
        DeviceGroup(name='sampler', ranks=[2, 3], device_type=device_type, gpus_per_worker=2),
    ]
    manager = ResourceManager(nproc_per_node=4, ncpu_proc_per_node=1, groups=groups)

    assert [worker['gpu_rank'] for worker in manager.get_group('model')] == [[1], [4]]
    assert [worker['gpu_rank'] for worker in manager.get_group('sampler')] == [sampler_devices]
    assert os.environ[env] == '1,4,6,5', 'Only the worker allocation should be normalized.'


@pytest.mark.parametrize('device_type, expected', [
    ('NPU', [['2', '10'], ['5', '6']]),
    ('GPU', [['10', '2'], ['6', '5']]),
])
def test_worker_device_order_from_external_allocation(ray_allocation, device_type, expected):
    devices = ['10', '2', '6', '5']
    group = DeviceGroup(name='sampler', ranks=4, device_type=device_type, gpus_per_worker=2)
    manager = ResourceManager.from_allocation(group, object(), devices)

    assert [worker['gpu_rank'] for worker in manager.get_group('sampler')] == expected
    assert devices == ['10', '2', '6', '5']
    assert group.ranks == [0, 1]
