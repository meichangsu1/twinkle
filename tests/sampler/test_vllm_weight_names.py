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
from twinkle.patch.vllm_weight_loader import patch_lora_weight_loader
from twinkle.sampler.vllm_sampler.vllm_worker_extension import TwinkleWorkerExtension

from .test_vllm_tied_weight_sync import loaders  # noqa: F401


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


def _load_by_internal_name(self, weights):
    params = dict(self.named_parameters(remove_duplicate=False))
    loaded = set()
    with torch.no_grad():
        for name, tensor in weights:
            params[name].copy_(tensor)
            loaded.add(name)
    return loaded


def _wrapper(loaders, kind):
    if kind == 'plain':
        return lambda layer: layer
    if kind == 'legacy':
        return LegacyLoRA
    if kind == 'standard':
        return loaders.LoRA
    if kind == 'inherited':
        class InheritedLoRA(loaders.LoRA):
            pass
        return InheritedLoRA
    base = loaders.LoRA if kind == 'custom_lora' else LegacyLoRA

    class CustomLoader(base):
        load_weights = _load_by_internal_name

    return CustomLoader


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


@pytest.mark.parametrize('kind', ['plain', 'legacy', 'standard', 'inherited', 'custom', 'custom_lora'])
def test_initial_peft_base_sync_loads_embeddings_linears_and_norms(loaders, kind):
    wrapper = _wrapper(loaders, kind)
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
    suffix = '' if kind == 'plain' else '.base_layer'
    assert [name for name, _ in weights] == [
        f'model.embed_tokens{suffix}.weight',
        f'model.q_proj{suffix}.weight',
        f'model.q_proj{suffix}.bias',
        'model.norm.weight',
        'model.norm.bias',
    ]
    # Discovery above precedes patch installation, as it does before the
    # first sync. Only the standard LoRA loader adapts internal input names.
    patch_lora_weight_loader()
    worker._load_weights(weights, peft_config=None, base_sync_done=False)
    actual = dict(worker.model_runner.model.named_parameters())
    assert torch.equal(actual[f'model.embed_tokens{suffix}.weight'], train_model.model.embed_tokens.weight)
    assert torch.equal(actual[f'model.q_proj{suffix}.weight'], train_model.model.q_proj.base_layer.weight)
    assert torch.equal(actual[f'model.q_proj{suffix}.bias'], train_model.model.q_proj.base_layer.bias)
    assert torch.equal(worker.model_runner.model.model.norm.weight, train_model.model.norm.weight)


def test_drafter_reports_internal_keys_as_well(loaders):
    worker = _worker(InferenceModel(loaders.LoRA),
                     SimpleNamespace(model=InferenceModel(loaders.LoRA)))
    keys = worker.get_state_keys()
    assert keys.count('model.embed_tokens.base_layer.weight') == 2
    assert keys.count('model.norm.weight') == 2


def test_qwen_multimodal_prefix_and_fused_projection_expansion_preserve_internal_names(loaders):
    model = nn.Module()
    model.language_model = nn.Module()
    model.language_model.model = nn.Module()
    model.language_model.model.embed_tokens = loaders.LoRA(nn.Embedding(8, 4))
    model.language_model.model.in_proj_qkvz = loaders.LoRA(nn.Linear(4, 4))
    # A pool can contain modules with both loading contracts.
    model.language_model.model.legacy = LegacyLoRA(nn.Linear(4, 4))
    worker = _worker(model)
    manager = CheckpointEngineManager.__new__(CheckpointEngineManager)
    manager._model_keys = None
    manager.sampler = worker
    manager._ensure_model_keys()
    assert 'model.language_model.embed_tokens.base_layer.weight' in manager._model_keys
    assert 'model.language_model.in_proj_qkv.base_layer.weight' in manager._model_keys
    assert 'model.language_model.in_proj_z.base_layer.weight' in manager._model_keys
    assert 'model.language_model.legacy.base_layer.weight' in manager._model_keys
    assert 'model.language_model.embed_tokens.weight' not in manager._model_keys
