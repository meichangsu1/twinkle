# Copyright (c) ModelScope Contributors. All rights reserved.
from contextlib import nullcontext
from copy import deepcopy
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import sys

import pytest

from twinkle.model.transformers.strategy.deepspeed import DeepSpeedStrategy


@pytest.mark.parametrize('auto_batch_size', [True, False])
def test_zero3_loading_uses_accelerates_loading_config(monkeypatch, auto_batch_size):
    config = {
        'zero_optimization': {'stage': 3},
        'train_batch_size': 'auto' if auto_batch_size else 32,
        'train_micro_batch_size_per_gpu': 'auto' if auto_batch_size else 2,
        'gradient_accumulation_steps': 'auto' if auto_batch_size else 4,
    }
    original = deepcopy(config)
    loading_config = deepcopy(config)
    if auto_batch_size:
        loading_config.pop('train_batch_size')
        loading_config['train_micro_batch_size_per_gpu'] = 1
        loading_config['gradient_accumulation_steps'] = 1
    strategy = DeepSpeedStrategy.__new__(DeepSpeedStrategy)
    strategy._deepspeed_plugin = SimpleNamespace(
        deepspeed_config=config, dschf=SimpleNamespace(config=loading_config))
    deepspeed = ModuleType('deepspeed')

    def zero_init(*, config_dict_or_path):
        # DeepSpeed validates these fields before allocating any parameters.
        for field in ('train_batch_size', 'train_micro_batch_size_per_gpu', 'gradient_accumulation_steps'):
            if field in config_dict_or_path:
                assert config_dict_or_path[field] > 0
        assert config_dict_or_path['zero_optimization']['stage'] == 3
        return nullcontext()

    deepspeed.zero = SimpleNamespace(Init=Mock(side_effect=zero_init))
    monkeypatch.setitem(sys.modules, 'deepspeed', deepspeed)

    with strategy.pretrained_load_context():
        pass

    assert deepspeed.zero.Init.call_args.kwargs['config_dict_or_path'] == loading_config
    assert config == original


@pytest.mark.parametrize('stage', [1, 2])
def test_non_zero3_loading_does_not_require_a_loading_config(stage):
    strategy = DeepSpeedStrategy.__new__(DeepSpeedStrategy)
    strategy._deepspeed_plugin = SimpleNamespace(deepspeed_config={'zero_optimization': {'stage': stage}})

    with strategy.pretrained_load_context():
        pass
