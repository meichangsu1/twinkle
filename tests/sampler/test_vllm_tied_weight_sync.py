# Copyright (c) ModelScope Contributors. All rights reserved.
"""CPU coverage of LoRA loaded names and tied embeddings across sync buckets.

The loader doubles preserve the alias and loaded-name contracts from vLLM
84030bbe3, without importing its accelerator-dependent model modules.
"""
import asyncio
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
from multiprocessing import shared_memory
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import zmq
from torch import nn

from twinkle.patch.vllm_weight_loader import patch_lora_weight_loader, streaming_weight_sync
from twinkle.sampler.vllm_sampler.vllm_worker_extension import TwinkleWorkerExtension


@pytest.fixture
def loaders(monkeypatch):
    class Embedding(nn.Embedding):
        pass

    class BaseLayerWithLoRA(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.base_layer = layer

        def load_weights(self, weights):
            base_loader = getattr(self.base_layer, 'load_weights', None)
            if callable(base_loader):
                return base_loader(weights)
            return AutoWeightsLoader(self.base_layer).load_weights(weights)

    class AutoWeightsLoader:
        def __init__(self, module):
            self.module = module
            self._skipped_aliases = {}
            self._loaded_params_are_complete = True
            self.aliased_params = {}
            canonical = {}
            for prefix, child in module.named_modules(remove_duplicate=False):
                if isinstance(child, Embedding):
                    for name, param in child.named_parameters(remove_duplicate=False):
                        qualname = f'{prefix}.{name}' if prefix else name
                        first = canonical.setdefault(id(param), qualname)
                        if first != qualname:
                            self.aliased_params[qualname] = first

        def _load_module(self, module, prefix, weights):
            if module is not self.module and callable(getattr(module, 'load_weights', None)):
                loaded = module.load_weights(weights)
                if loaded is None:
                    self._loaded_params_are_complete = False
                else:
                    yield from (f'{prefix}{name}' for name in loaded)
                return
            for name, group in groupby(weights, key=lambda item: item[0].split('.', 1)[0]):
                if name in module._parameters:
                    for suffix, value in group:
                        assert suffix == name
                        with torch.no_grad():
                            module._parameters[name].copy_(value)
                        yield f'{prefix}{name}'
                else:
                    relative = ((key.split('.', 1)[1], value) for key, value in group)
                    yield from self._load_module(getattr(module, name), f'{prefix}{name}.', relative)

        def _check_skipped_aliases(self, autoloaded_weights):
            if not self._loaded_params_are_complete:
                return
            for alias, canonical in self._skipped_aliases.items():
                if canonical not in autoloaded_weights:
                    raise ValueError(f'{alias!r} was skipped because it is tied to {canonical!r}, '
                                     f'but {canonical!r} was not found in the checkpoint')

        def load_weights(self, weights):
            def filtered():
                for name, value in weights:
                    if name in self.aliased_params:
                        self._skipped_aliases[name] = self.aliased_params[name]
                    else:
                        yield name, value

            loaded = set(self._load_module(self.module, '', filtered()))
            self._check_skipped_aliases(loaded)
            return loaded

    lora_module = ModuleType('vllm.lora.layers')
    lora_module.BaseLayerWithLoRA = BaseLayerWithLoRA
    utils_module = ModuleType('vllm.model_executor.models.utils')
    utils_module.AutoWeightsLoader = AutoWeightsLoader
    monkeypatch.setitem(sys.modules, lora_module.__name__, lora_module)
    monkeypatch.setitem(sys.modules, utils_module.__name__, utils_module)
    return SimpleNamespace(Embedding=Embedding, LoRA=BaseLayerWithLoRA, Loader=AutoWeightsLoader)


def _model(loaders, prefix='', lora=True):
    root = nn.Module()
    language = nn.Module()
    language.model = nn.Module()
    embedding = loaders.Embedding(8, 4)
    language.model.embed_tokens = loaders.LoRA(embedding) if lora else embedding
    language.lm_head = loaders.Embedding(8, 4)
    language.lm_head.weight = embedding.weight
    if prefix:
        root.language_model = language
    else:
        root = language
    return root, embedding.weight, language.lm_head.weight


def test_unpatched_loader_reproduces_false_uninitialized_tied_weight_error(loaders):
    model, embedding, _ = _model(loaders, prefix='language_model.')
    expected = torch.ones(8, 4)
    weights = [('language_model.model.embed_tokens.weight', expected),
               ('language_model.lm_head.weight', expected)]
    with pytest.raises(ValueError, match=r'embed_tokens.base_layer.weight.*not found'):
        loaders.Loader(model).load_weights(iter(weights))
    # The error is about the returned name: the actual tensor was written.
    assert torch.equal(embedding, expected)


@pytest.mark.parametrize('prefix', ['', 'language_model.'])
@pytest.mark.parametrize('lora', [True, False])
@pytest.mark.parametrize('bucket_order', ['together', 'embedding_first', 'alias_first'])
@pytest.mark.parametrize('name_style', ['checkpoint', 'internal'])
def test_tied_weights_load_once_across_buckets(loaders, prefix, lora, bucket_order, name_style):
    patch_lora_weight_loader()
    model, embedding, head = _model(loaders, prefix, lora)
    expected = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    suffix = '.base_layer' if lora and name_style == 'internal' else ''
    canonical = (f'{prefix}model.embed_tokens{suffix}.weight', expected)
    alias = (f'{prefix}lm_head.weight', torch.full((8, 4), -999.))
    buckets = {'together': [[canonical, alias]], 'embedding_first': [[canonical], [alias]],
               'alias_first': [[alias], [canonical]]}[bucket_order]
    with streaming_weight_sync():
        for bucket in buckets:
            loaders.Loader(model).load_weights(iter(bucket))
    assert head is embedding
    assert torch.equal(embedding, expected)


@pytest.mark.parametrize('lora', [True, False])
def test_a_previous_sync_does_not_hide_missing_canonical_weights(loaders, lora):
    patch_lora_weight_loader()
    model, _, _ = _model(loaders, lora=lora)
    with streaming_weight_sync():
        loaders.Loader(model).load_weights(iter([('model.embed_tokens.weight', torch.ones(8, 4))]))
    with pytest.raises(ValueError, match='not found in the checkpoint'):
        with streaming_weight_sync():
            loaders.Loader(model).load_weights(iter([('lm_head.weight', torch.ones(8, 4))]))


def test_loaded_names_are_not_shared_between_models(loaders):
    patch_lora_weight_loader()
    first, _, _ = _model(loaders)
    second, _, _ = _model(loaders)
    with pytest.raises(ValueError, match='not found in the checkpoint'):
        with streaming_weight_sync():
            loaders.Loader(first).load_weights(iter([('model.embed_tokens.weight', torch.ones(8, 4))]))
            loaders.Loader(second).load_weights(iter([('lm_head.weight', torch.ones(8, 4))]))


def test_alias_checks_outside_sync_still_run_immediately(loaders):
    patch_lora_weight_loader()
    model, _, _ = _model(loaders)
    with pytest.raises(ValueError, match='not found in the checkpoint'):
        loaders.Loader(model).load_weights(iter([('lm_head.weight', torch.ones(8, 4))]))


def test_failed_sync_preserves_original_error_and_resets_context(loaders):
    patch_lora_weight_loader()
    model, _, _ = _model(loaders)
    error = RuntimeError('transfer failed')
    with pytest.raises(RuntimeError) as exc_info:
        with streaming_weight_sync():
            loaders.Loader(model).load_weights(iter([('lm_head.weight', torch.ones(8, 4))]))
            raise error
    assert exc_info.value is error
    with pytest.raises(ValueError, match='not found in the checkpoint'):
        loaders.Loader(model).load_weights(iter([('lm_head.weight', torch.ones(8, 4))]))


def test_concurrent_syncs_have_independent_alias_checks(loaders):
    patch_lora_weight_loader()
    model, _, _ = _model(loaders)

    async def run():
        canonical_loaded = asyncio.Event()
        alias_checked = asyncio.Event()

        async def complete_sync():
            with streaming_weight_sync():
                loaders.Loader(model).load_weights(iter([('model.embed_tokens.weight', torch.ones(8, 4))]))
                canonical_loaded.set()
                await alias_checked.wait()

        async def incomplete_sync():
            await canonical_loaded.wait()
            try:
                with pytest.raises(ValueError, match='not found in the checkpoint'):
                    with streaming_weight_sync():
                        loaders.Loader(model).load_weights(iter([('lm_head.weight', torch.ones(8, 4))]))
            finally:
                alias_checked.set()

        await asyncio.wait_for(asyncio.gather(complete_sync(), incomplete_sync()), timeout=3)

    asyncio.run(run())


@pytest.mark.parametrize('missing_embedding', [False, True])
@pytest.mark.parametrize('name_style', ['checkpoint', 'internal'])
def test_production_worker_checks_aliases_at_end_of_real_shm_zmq_stream(
        loaders, monkeypatch, missing_embedding, name_style):
    monkeypatch.setenv('TWINKLE_VLLM_IPC_TIMEOUT_S', '3')
    for method in ('synchronize', 'ipc_collect', 'empty_cache'):
        monkeypatch.setattr(f'twinkle.sampler.vllm_sampler.vllm_worker_extension.Torch.{method}', lambda: None)
    model, embedding, head = _model(loaders, prefix='language_model.')
    model.load_weights = lambda weights: loaders.Loader(model).load_weights(iter(weights))
    worker = object.__new__(TwinkleWorkerExtension)
    worker.device = torch.device('cpu')
    worker.rank = 0
    worker.model_runner = SimpleNamespace(model=model, parallel_config=SimpleNamespace(tensor_parallel_size=1))
    expected = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    suffix = '.base_layer' if name_style == 'internal' else ''
    buckets = [] if missing_embedding else [(f'language_model.model.embed_tokens{suffix}.weight', expected)]
    buckets.append(('language_model.lm_head.weight', torch.full((8, 4), -999.)))
    endpoint = f'ipc:///tmp/twinkle-tied-{uuid.uuid4().hex[:12]}.sock'
    shm = shared_memory.SharedMemory(create=True, size=expected.nbytes)
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REQ)
    socket.linger = 0
    socket.rcvtimeo = socket.sndtimeo = 3000
    socket.bind(endpoint)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(worker.update_weights_from_ipc, use_shm=True, zmq_handle=endpoint)
            socket.send_pyobj({'name': shm.name, 'size': shm.size})
            socket.recv()
            for index, (name, tensor) in enumerate(buckets):
                shm.buf[:tensor.nbytes] = tensor.numpy().tobytes()
                socket.send_pyobj({'bucket_meta': [{
                    'name': name, 'shape': tensor.shape, 'dtype': tensor.dtype, 'offset': 0,
                    'nbytes': tensor.nbytes, 'chunk_offset': 0, 'total_nbytes': tensor.nbytes,
                }], 'is_last': index == len(buckets) - 1})
                socket.recv()
            if missing_embedding:
                with pytest.raises(ValueError, match='not found in the checkpoint'):
                    result.result(timeout=5)
            else:
                result.result(timeout=5)
                assert embedding is head
                assert torch.equal(embedding, expected)
    finally:
        socket.close()
        ctx.term()
        shm.close()
        shm.unlink()
        if getattr(worker, '_zmq_ctx', None) is not None:
            worker._zmq_ctx.destroy(linger=0)
        Path(endpoint.removeprefix('ipc://')).unlink(missing_ok=True)


def test_patch_is_idempotent_and_reports_internal_parameter_names(loaders):
    patch_lora_weight_loader()
    first_load = loaders.LoRA.load_weights
    first_check = loaders.Loader._check_skipped_aliases
    patch_lora_weight_loader()
    assert loaders.LoRA.load_weights is first_load
    assert loaders.Loader._check_skipped_aliases is first_check
    wrapper = loaders.LoRA(loaders.Embedding(8, 4))
    assert wrapper.load_weights(iter([('weight', torch.ones(8, 4))])) == {'base_layer.weight'}


@pytest.mark.parametrize('name', ['weight', 'base_layer.base_layer.weight'])
def test_nested_standard_lora_wrappers_strip_and_report_each_own_prefix(loaders, name):
    patch_lora_weight_loader()
    embedding = loaders.Embedding(8, 4)
    wrapper = loaders.LoRA(loaders.LoRA(embedding))
    expected = torch.ones(8, 4)
    assert wrapper.load_weights(iter([(name, expected)])) == {'base_layer.base_layer.weight'}
    assert torch.equal(embedding.weight, expected)


def test_only_the_leading_wrapper_prefix_is_removed(loaders):
    patch_lora_weight_loader()
    base = nn.Module()
    base.custom = nn.Module()
    base.custom.base_layer = loaders.Embedding(8, 4)
    wrapper = loaders.LoRA(base)
    expected = torch.ones(8, 4)
    loaded = wrapper.load_weights(iter([('base_layer.custom.base_layer.weight', expected)]))
    assert loaded == {'base_layer.custom.base_layer.weight'}
    assert torch.equal(base.custom.base_layer.weight, expected)


@pytest.mark.parametrize('returned', [None, {'base_layer.weight'}])
def test_preserve_unknown_or_already_internal_loaded_names(loaders, returned):
    loaders.LoRA.load_weights = lambda self, weights: returned
    patch_lora_weight_loader()
    wrapper = loaders.LoRA(loaders.Embedding(8, 4))
    assert wrapper.load_weights(iter([])) == returned


def test_older_loader_without_these_hooks_is_unchanged(loaders, monkeypatch):
    monkeypatch.delattr(loaders.LoRA, 'load_weights')
    monkeypatch.delattr(loaders.Loader, '_check_skipped_aliases')
    patch_lora_weight_loader()
    assert not hasattr(loaders.LoRA, 'load_weights')
    assert not hasattr(loaders.Loader, '_check_skipped_aliases')
