# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import json
import msgspec
import pytest
import torch
from safetensors.torch import save_file

from twinkle.sampler.vllm_sampler.vllm_engine import VLLMEngine


@pytest.fixture
def lora_runtime(monkeypatch, request):
    mixed = getattr(request, 'param', True)
    fields = [('lora_name', str), ('lora_int_id', int), ('lora_path', str), ('load_inplace', bool, False)]
    if mixed:
        fields.append(('is_3d_lora_weight', bool, False))
    request_cls = msgspec.defstruct('LoRARequest', fields, array_like=True)

    class Manager:
        def _load_adapter(self, request):
            return SimpleNamespace(native_request=request)

    legacy_patch = Mock()
    helper = SimpleNamespace(validate_legal=Mock())
    modules = {
        'vllm.lora.request': {'LoRARequest': request_cls},
        'vllm.lora.worker_manager': {'LRUCacheWorkerLoRAManager': Manager},
        'vllm.lora.models': {'LoRAModel': SimpleNamespace},
        'vllm.lora.peft_helper': {'PEFTHelper': SimpleNamespace(from_dict=lambda config: helper)},
        'twinkle.patch.vllm_moe_loader': {'patch_qwen35_moe_is_3d_moe_weight_false': legacy_patch},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, 'vllm.transformers_utils.tokenizer_group', None)
    name = 'twinkle.patch.vllm_lora_weights'
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[2] / 'src/twinkle/patch/vllm_lora_weights.py')
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    monkeypatch.setitem(sys.modules, name, patch)
    patch.VLLMLoraWeights()(None)
    assert legacy_patch.call_count == (0 if mixed else 1)
    return SimpleNamespace(patch=patch, manager_cls=Manager, request_cls=request_cls)


def _weights(layout):
    prefix = 'base_model.model.language_model.model.layers.0'
    if layout == 'packed':
        return {
            f'{prefix}.mlp.experts.base_layer.lora_A.weight': torch.randn(4, 6),
            f'{prefix}.mlp.experts.base_layer.lora_B.weight': torch.randn(8, 4),
            f'{prefix}.mlp.experts.lora_A.weight': torch.randn(4, 4),
            f'{prefix}.mlp.experts.lora_B.weight': torch.randn(6, 4),
        }
    if layout == 'per_expert':
        result = {}
        for expert in range(2):
            for projection, in_features, out_features in [('gate_proj', 6, 4), ('up_proj', 6, 4), ('down_proj', 4, 6)]:
                name = f'{prefix}.mlp.experts.{expert}.{projection}'
                result[f'{name}.lora_A.weight'] = torch.randn(2, in_features)
                result[f'{name}.lora_B.weight'] = torch.randn(out_features, 2)
        return result
    return {f'{prefix}.self_attn.q_proj.lora_A.weight': torch.randn(2, 6),
            f'{prefix}.self_attn.q_proj.lora_B.weight': torch.randn(6, 2)}


@pytest.mark.parametrize('layout', ['packed', 'per_expert', 'dense'])
@pytest.mark.parametrize('filename', ['adapter_model.safetensors', 'adapter_model.bin', 'adapter_model.pt'])
def test_file_adapter_preserves_native_loading_and_carries_layout(tmp_path, lora_runtime, layout, filename):
    weights = _weights(layout)
    if filename.endswith('.safetensors'):
        save_file(weights, tmp_path / filename)
    else:
        torch.save(weights, tmp_path / filename)
    (tmp_path / 'adapter_config.json').write_text(json.dumps({'r': 2, 'lora_alpha': 4}))
    engine = VLLMEngine.__new__(VLLMEngine)
    engine._next_lora_id = 1
    engine.engine = SimpleNamespace(add_lora=AsyncMock())

    request = asyncio.run(engine._load_lora(str(tmp_path)))

    assert request.is_3d_lora_weight == (layout == 'packed')
    engine.engine.add_lora.assert_awaited_once_with(request)
    # The file request must bypass the tensor patch's copied module validation.
    loaded = lora_runtime.manager_cls()._load_adapter(request)
    assert loaded.native_request is request
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(request), type=lora_runtime.request_cls)
    assert decoded.is_3d_lora_weight == request.is_3d_lora_weight


@pytest.mark.parametrize('layout', ['packed', 'per_expert', 'dense'])
def test_memory_adapter_preserves_tensors_and_carries_layout_to_manager(lora_runtime, layout):
    from twinkle.sampler.vllm_sampler.vllm_worker_extension import TwinkleWorkerExtension

    weights = _weights(layout)
    worker = object.__new__(TwinkleWorkerExtension)
    worker.add_lora = Mock()
    named_weights = [(name.replace('.lora_A.', '.lora_A.tenant.').replace('.lora_B.', '.lora_B.tenant.'), tensor)
                     for name, tensor in weights.items()]

    worker._load_weights(named_weights, peft_config={'r': 2, 'lora_alpha': 4}, base_sync_done=True)

    request = worker.add_lora.call_args.args[0]
    assert isinstance(request, lora_runtime.patch.TensorLoRARequest)
    assert request.load_inplace and request.is_3d_lora_weight == (layout == 'packed')
    assert request.lora_tensors.keys() == weights.keys()
    for name, tensor in weights.items():
        assert request.lora_tensors[name] is tensor

    calls = []

    def from_tensors(*, tensors, skip_prefixes=None, **kwargs):
        calls.append(dict(tensors=tensors, skip_prefixes=skip_prefixes, **kwargs))
        return SimpleNamespace(tensors=tensors)

    manager = lora_runtime.manager_cls()
    manager._lora_model_cls = SimpleNamespace(from_lora_tensors=from_tensors)
    manager._adapter_manager = SimpleNamespace(
        model=SimpleNamespace(hf_to_vllm_mapper=SimpleNamespace(get_rename_mapper=lambda: 'renames'),
                              lora_skip_prefixes=['mtp']),
        _validate_modules_to_save=Mock(), _validate_token_classification_lora=Mock())
    manager.lora_config = SimpleNamespace(lora_dtype=torch.float32)
    manager.vocab_size = 100

    loaded = manager._load_adapter(request)

    assert loaded.is_3d_lora_weight == (layout == 'packed')
    assert loaded.tensors is request.lora_tensors
    assert calls[0]['weights_mapper'] == 'renames' and calls[0]['skip_prefixes'] == ['mtp']
    manager._adapter_manager._validate_modules_to_save.assert_called_once_with(loaded)
    manager._adapter_manager._validate_token_classification_lora.assert_called_once_with(loaded)


@pytest.mark.parametrize('lora_runtime', [False], indirect=True)
def test_legacy_file_request_keeps_native_loading(tmp_path, lora_runtime):
    (tmp_path / 'adapter_config.json').write_text('{}')
    engine = VLLMEngine.__new__(VLLMEngine)
    engine._next_lora_id = 1
    engine.engine = SimpleNamespace(add_lora=AsyncMock())

    request = asyncio.run(engine._load_lora(str(tmp_path)))

    assert not hasattr(request, 'is_3d_lora_weight')
    assert lora_runtime.manager_cls()._load_adapter(request).native_request is request


@pytest.mark.parametrize('lora_runtime', [False], indirect=True)
def test_legacy_memory_request_preserves_embedding_and_mapper_arguments(lora_runtime):
    from twinkle.sampler.vllm_sampler.vllm_worker_extension import TwinkleWorkerExtension

    worker = object.__new__(TwinkleWorkerExtension)
    worker.add_lora = Mock()
    worker._load_weights(list(_weights('dense').items()), peft_config={'r': 2}, base_sync_done=True)
    request = worker.add_lora.call_args.args[0]
    assert not hasattr(request, 'is_3d_lora_weight')

    def from_tensors(*, tensors, lora_model_id, peft_helper, device, dtype, weights_mapper,
                     embedding_modules, embedding_padding_modules, target_embedding_padding):
        assert weights_mapper == 'renames' and device == 'cpu'
        assert embedding_modules == {'embed_tokens': 'input_embeddings'}
        assert embedding_padding_modules == ['embed_tokens'] and target_embedding_padding == 104
        return SimpleNamespace(tensors=tensors, extra_vocab_size=0)

    manager = lora_runtime.manager_cls()
    manager._lora_model_cls = SimpleNamespace(from_lora_tensors=from_tensors)
    manager._adapter_manager = SimpleNamespace(model=SimpleNamespace(
        hf_to_vllm_mapper=SimpleNamespace(get_unstacked_mapper=lambda: 'renames')))
    manager.lora_config = SimpleNamespace(lora_dtype=torch.float32, lora_extra_vocab_size=4)
    manager.vocab_size = 100
    manager.embedding_modules = {'embed_tokens': 'input_embeddings'}
    manager.embedding_padding_modules = ['embed_tokens']

    assert manager._load_adapter(request).tensors is request.lora_tensors


@pytest.mark.parametrize('mixed_supported,enable_lora,override,expected', [
    (True, True, None, True),
    (True, True, False, False),
    (True, False, None, False),
    (False, True, None, None),
])
def test_engine_defaults_to_mixed_only_when_supported(monkeypatch, mixed_supported, enable_lora, override, expected):
    from twinkle.sampler.vllm_sampler import vllm_engine

    class Args:
        def __init__(self, model, enable_lora=False, enable_mixed_moe_lora_format=False):
            self.config = SimpleNamespace(model=model, mixed=enable_mixed_moe_lora_format)

        def create_engine_config(self, **kwargs):
            return self.config

    class LegacyArgs(Args):
        def __init__(self, model, enable_lora=False):
            self.config = SimpleNamespace(model=model, mixed=None)

    async_llm = SimpleNamespace(from_vllm_config=Mock(return_value=object()))
    modules = {
        'vllm.engine.arg_utils': {'AsyncEngineArgs': Args if mixed_supported else LegacyArgs},
        'vllm.usage.usage_lib': {'UsageContext': SimpleNamespace(OPENAI_API_SERVER='test')},
        'vllm.v1.engine.async_llm': {'AsyncLLM': async_llm},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(vllm_engine.Platform, 'apply_vllm_patch', lambda: None)
    monkeypatch.setattr(vllm_engine, 'get_vllm_max_lora_rank', lambda rank: rank)
    engine = VLLMEngine.__new__(VLLMEngine)
    engine.__dict__.update(
        model_id='test', enable_sampling_replay=False, tensor_parallel_size=1,
        gpu_memory_utilization=.5, max_num_seqs=4, trust_remote_code=False,
        enforce_eager=True, dtype='float32', load_format='auto', enable_prefix_caching=False,
        max_model_len=None, quantization=None, enable_sleep_mode=False, runner=None, logprobs_mode=None,
        enable_lora=enable_lora, max_loras=2, max_lora_rank=8,
        engine_kwargs={} if override is None else {'enable_mixed_moe_lora_format': override})

    engine._create_engine()

    assert async_llm.from_vllm_config.call_args.kwargs['vllm_config'].mixed == expected
