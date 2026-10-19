# Copyright (c) ModelScope Contributors. All rights reserved.
from importlib.machinery import ModuleSpec
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch.distributed as dist

from twinkle.utils.framework import Framework, Platform


@pytest.fixture
def npu_collective(monkeypatch):
    monkeypatch.setattr(Platform, 'device_prefix', lambda: 'npu')
    monkeypatch.setattr(dist, 'is_available', lambda: True)
    monkeypatch.setattr(dist, 'is_initialized', lambda: True)
    world_size = Mock(return_value=2)
    monkeypatch.setattr(dist, 'get_world_size', world_size)

    def gather(output, value, group):
        output[:] = [value, [11]]

    all_gather = Mock(side_effect=gather)
    monkeypatch.setattr(dist, 'all_gather_object', all_gather)
    mesh = SimpleNamespace(has_dim=lambda dim: False, cp_world_size=1)
    return mesh, world_size, all_gather


@pytest.mark.parametrize('missing', ['megatron', 'megatron.core'])
@pytest.mark.parametrize('world_size', [1, 2])
def test_npu_gather_without_megatron_preserves_group(monkeypatch, npu_collective, missing, world_size):
    mesh, get_world_size, all_gather = npu_collective
    get_world_size.return_value = world_size
    if missing == 'megatron':
        # Exercise real find_spec(): looking up megatron.core raises when its
        # parent cannot be imported, rather than simply returning None.
        monkeypatch.setitem(sys.modules, 'megatron', None)
        monkeypatch.delitem(sys.modules, 'megatron.core', raising=False)
    else:
        package = ModuleType('megatron')
        package.__spec__ = ModuleSpec('megatron', loader=None, is_package=True)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, 'megatron', package)
        monkeypatch.setitem(sys.modules, 'megatron.core', None)

    caller_group = object()
    expected = [7] if world_size == 1 else [7, 11]
    assert Framework.gather_object([7], mesh, caller_group) == expected
    get_world_size.assert_called_once_with(group=caller_group)
    if world_size == 1:
        all_gather.assert_not_called()
    else:
        assert all_gather.call_args.kwargs['group'] is caller_group


@pytest.mark.parametrize(('initialized', 'cp_world_size'), [(False, 1), (True, 1), (True, 2)])
def test_npu_gather_uses_initialized_megatron_gloo_group(monkeypatch, npu_collective, initialized, cp_world_size):
    mesh, get_world_size, all_gather = npu_collective
    mesh.cp_world_size = cp_world_size
    gloo_group = object()
    mpu = SimpleNamespace(
        model_parallel_is_initialized=lambda: initialized,
        get_data_parallel_group_gloo=Mock(return_value=gloo_group),
    )
    package = ModuleType('megatron')
    package.__spec__ = ModuleSpec('megatron', loader=None, is_package=True)
    package.__path__ = []
    core = ModuleType('megatron.core')
    core.__spec__ = ModuleSpec('megatron.core', loader=None)
    core.parallel_state = mpu
    monkeypatch.setitem(sys.modules, 'megatron', package)
    monkeypatch.setitem(sys.modules, 'megatron.core', core)

    caller_group = object()
    assert Framework.gather_object([7], mesh, caller_group) == [7, 11]
    expected_group = gloo_group if initialized else caller_group
    get_world_size.assert_called_once_with(group=expected_group)
    assert all_gather.call_args.kwargs['group'] is expected_group
    if initialized:
        mpu.get_data_parallel_group_gloo.assert_called_once_with(with_context_parallel=cp_world_size > 1)
    else:
        mpu.get_data_parallel_group_gloo.assert_not_called()
