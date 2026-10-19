# Copyright (c) ModelScope Contributors. All rights reserved.
"""Check trainer export against the names accepted by vLLM's LoRA loaders."""
from itertools import groupby
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from twinkle.checkpoint_engine.manager import CheckpointEngineManager
from twinkle.model.transformers.transformers import TransformersModel
from twinkle.sampler.vllm_sampler.vllm_worker_extension import TwinkleWorkerExtension


class LegacyLoRA(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.base_layer = layer


def _auto_load(module, weights):
    """The recursive loader delegates to a module's load_weights when present."""
    for first, entries in groupby(weights, key=lambda item: item[0].split('.', 1)[0]):
        child = getattr(module, first)
        entries = list(entries)
        if isinstance(child, nn.Parameter):
            assert len(entries) == 1
            with torch.no_grad():
                child.copy_(entries[0][1])
        else:
            relative_weights = [(name.split('.', 1)[1], tensor) for name, tensor in entries]
            loader = getattr(child, 'load_weights', None)
            if callable(loader):
                loader(relative_weights)
            else:
                _auto_load(child, relative_weights)


class DelegatingLoRA(LegacyLoRA):
    def load_weights(self, weights):
        # vLLM's BaseLayerWithLoRA now delegates directly to base_layer.
        _auto_load(self.base_layer, weights)


class InferenceModel(nn.Module):
    def __init__(self, wrapper):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = wrapper(nn.Embedding(8, 4))
        self.model.q_proj = wrapper(nn.Linear(4, 4))
        self.model.norm = nn.LayerNorm(4)

    def load_weights(self, weights):
        _auto_load(self, weights)


def _worker(model, drafter=None):
    # Skip __new__, which installs patches in an actual vLLM worker process.
    worker = object.__new__(TwinkleWorkerExtension)
    worker.model_runner = SimpleNamespace(model=model, drafter=drafter)
    return worker


@pytest.mark.parametrize('wrapper', [DelegatingLoRA, LegacyLoRA])
def test_initial_peft_base_sync_loads_embeddings_linears_and_norms(wrapper):
    train_model = nn.Module()
    train_model.model = nn.Module()
    train_model.model.embed_tokens = nn.Embedding(8, 4)
    train_model.model.q_proj = nn.Linear(4, 4)
    train_model.model.norm = nn.LayerNorm(4)
    peft_model = get_peft_model(train_model, LoraConfig(target_modules=['q_proj'], r=2, lora_alpha=2))

    trainer = TransformersModel.__new__(TransformersModel)
    nn.Module.__init__(trainer)
    trainer.model = peft_model
    trainer.strategy = SimpleNamespace(unwrap_model=lambda model: model)
    worker = _worker(InferenceModel(wrapper))
    manager = CheckpointEngineManager.__new__(CheckpointEngineManager)
    manager._model_keys = None
    manager.sampler = worker
    manager._ensure_model_keys()

    weights = list(trainer._get_weight_generator(
        adapter_name='default', merge_and_sync=False, model_keys=manager._model_keys))
    suffix = '' if wrapper is DelegatingLoRA else '.base_layer'
    assert [name for name, _ in weights] == [
        f'model.embed_tokens{suffix}.weight',
        f'model.q_proj{suffix}.weight',
        f'model.q_proj{suffix}.bias',
        'model.norm.weight',
        'model.norm.bias',
    ]
    # Execute the production receiver's base-weight loading path as well.
    worker._load_weights(weights, peft_config=None, base_sync_done=False)
    assert torch.equal(worker.model_runner.model.model.embed_tokens.base_layer.weight,
                       train_model.model.embed_tokens.weight)
    assert torch.equal(worker.model_runner.model.model.q_proj.base_layer.weight,
                       train_model.model.q_proj.base_layer.weight)
    assert torch.equal(worker.model_runner.model.model.q_proj.base_layer.bias,
                       train_model.model.q_proj.base_layer.bias)
    assert torch.equal(worker.model_runner.model.model.norm.weight, train_model.model.norm.weight)


def test_drafter_reports_loadable_keys_as_well():
    worker = _worker(InferenceModel(DelegatingLoRA),
                     SimpleNamespace(model=InferenceModel(DelegatingLoRA)))
    keys = worker.get_state_keys()
    assert keys.count('model.embed_tokens.weight') == 2
    assert all('.base_layer.' not in key for key in keys)


def test_qwen_multimodal_prefix_and_fused_projection_expansion_preserve_loader_names():
    model = nn.Module()
    model.language_model = nn.Module()
    model.language_model.model = nn.Module()
    model.language_model.model.embed_tokens = DelegatingLoRA(nn.Embedding(8, 4))
    model.language_model.model.in_proj_qkvz = DelegatingLoRA(nn.Linear(4, 4))
    # A pool can contain modules with both loading contracts.
    model.language_model.model.legacy = LegacyLoRA(nn.Linear(4, 4))
    worker = _worker(model)
    manager = CheckpointEngineManager.__new__(CheckpointEngineManager)
    manager._model_keys = None
    manager.sampler = worker
    manager._ensure_model_keys()
    assert 'model.language_model.embed_tokens.weight' in manager._model_keys
    assert 'model.language_model.in_proj_qkv.weight' in manager._model_keys
    assert 'model.language_model.in_proj_z.weight' in manager._model_keys
    assert 'model.language_model.legacy.base_layer.weight' in manager._model_keys
    assert 'model.language_model.embed_tokens.base_layer.weight' not in manager._model_keys
