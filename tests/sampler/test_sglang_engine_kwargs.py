# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
import sys

import pytest

from twinkle.sampler.sglang_sampler import sglang_engine


@dataclass
class ModernServerArgs:
    model_path: str
    tp_size: int = 1
    dp_size: int = 1
    enable_memory_saver: bool = False
    ep_size: int = 1


@dataclass
class LegacyServerArgs(ModernServerArgs):
    enable_ep_moe: bool = False


@pytest.fixture
def backend(monkeypatch):
    def install(server_args_cls):
        calls = []

        def engine(**kwargs):
            args = server_args_cls(**kwargs)
            calls.append(kwargs)
            return SimpleNamespace(server_args=args)

        package = ModuleType('sglang')
        package.__path__ = []
        package.Engine = engine
        srt = ModuleType('sglang.srt')
        srt.__path__ = []
        server_args = ModuleType('sglang.srt.server_args')
        server_args.ServerArgs = server_args_cls
        for name, module in [('sglang', package), ('sglang.srt', srt),
                             ('sglang.srt.server_args', server_args)]:
            monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setattr(sglang_engine, 'requires', lambda package: None)
        return calls

    return install


def test_removed_ep_switch_does_not_break_disabled_default(backend):
    calls = backend(ModernServerArgs)
    sglang_engine.SGLangEngine('local-model', tensor_parallel_size=2, enable_ep_moe=False, ep_size=1)

    assert len(calls) == 1
    assert 'enable_ep_moe' not in calls[0]
    assert calls[0]['tp_size'] == 2
    assert calls[0]['ep_size'] == 1


def test_removed_ep_switch_rejects_explicit_enable(backend):
    calls = backend(ModernServerArgs)
    with pytest.raises(ValueError, match='configure ep_size instead'):
        sglang_engine.SGLangEngine('local-model', enable_ep_moe=True)
    assert calls == []


@pytest.mark.parametrize('enabled', [False, True])
def test_legacy_server_keeps_supported_ep_switch(backend, enabled):
    calls = backend(LegacyServerArgs)
    sglang_engine.SGLangEngine('local-model', enable_ep_moe=enabled)

    assert calls[0]['enable_ep_moe'] is enabled


def test_modern_ep_size_is_preserved(backend):
    calls = backend(ModernServerArgs)
    sglang_engine.SGLangEngine('local-model', tensor_parallel_size=2, ep_size=2)

    assert calls[0]['ep_size'] == 2
    assert 'enable_ep_moe' not in calls[0]


def test_other_unknown_arguments_are_still_rejected(backend):
    backend(ModernServerArgs)
    with pytest.raises(TypeError, match='unknown_option'):
        sglang_engine.SGLangEngine('local-model', unknown_option=True)
