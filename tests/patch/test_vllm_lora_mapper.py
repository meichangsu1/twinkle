# Copyright (c) ModelScope Contributors. All rights reserved.
"""CPU coverage of tensor-adapter naming before vLLM packs QKV and gate/up.

The doubles follow the mapper/name-parser contracts in vLLM 84030bbe3.
"""
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from twinkle.patch.vllm_lora_weights import TensorLoRARequest, VLLMLoraWeights


class RenameMapper:
    def _map_name(self, name):
        return name.replace('model.', 'language_model.model.', 1)


class StackingMapper(RenameMapper):
    """Base-weight mapping also merges constituent projection names."""
    stacked = {'q_proj': 'qkv_proj', 'k_proj': 'qkv_proj', 'v_proj': 'qkv_proj',
               'gate_proj': 'gate_up_proj', 'up_proj': 'gate_up_proj'}

    def _map_name(self, name):
        name = super()._map_name(name)
        for source, target in self.stacked.items():
            name = name.replace(f'.{source}.', f'.{target}.')
        return name

    def get_rename_mapper(self):
        return RenameMapper()


@pytest.fixture
def load_adapter(monkeypatch):
    class PEFTHelper:
        @classmethod
        def from_dict(cls, config):
            return cls()

        @classmethod
        def from_local_dir(cls, *args):
            return cls()

        def validate_legal(self, config):
            pass

    class LoRAModel:
        @classmethod
        def from_lora_tensors(cls, tensors, weights_mapper=None, **kwargs):
            loras = {}
            for name, tensor in tensors.items():
                name = name.removeprefix('base_model.model.')
                if weights_mapper is not None:
                    name = weights_mapper._map_name(name)
                module_name, lora_part, _ = name.rsplit('.', 2)
                layer = loras.setdefault(module_name, SimpleNamespace())
                setattr(layer, lora_part.lower(), tensor)
            return SimpleNamespace(loras=loras)

        @classmethod
        def from_local_checkpoint(cls, *args, **kwargs):
            return cls.from_lora_tensors(tensors=disk_tensors, **kwargs)

    class WorkerManager:
        def _load_adapter(self, request):
            raise AssertionError('Patch was not installed')

    for name, attrs in {
        'vllm.lora.worker_manager': {'LRUCacheWorkerLoRAManager': WorkerManager},
        'vllm.lora.models': {'LoRAModel': LoRAModel},
        'vllm.lora.utils': {'get_adapter_absolute_path': lambda path: path},
        'vllm.lora.peft_helper': {'PEFTHelper': PEFTHelper},
        'vllm.transformers_utils.tokenizer_group': {'TokenizerGroup': None},
        'twinkle.patch.vllm_moe_loader': {'patch_qwen35_moe_is_3d_moe_weight_false': lambda: None},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    VLLMLoraWeights()(None)
    disk_tensors = {}

    def load(tensors, mapper, from_tensors=True):
        nonlocal disk_tensors
        disk_tensors = tensors
        manager = WorkerManager()
        manager._adapter_manager = SimpleNamespace(
            supported_lora_modules=['qkv_proj', 'gate_up_proj', 'down_proj'],
            packed_modules_mapping={'qkv_proj': ['q_proj', 'k_proj', 'v_proj'],
                                    'gate_up_proj': ['gate_proj', 'up_proj']},
            model=SimpleNamespace(hf_to_vllm_mapper=mapper),
        )
        manager._lora_model_cls = LoRAModel
        manager.lora_config = SimpleNamespace(lora_dtype=torch.float32)
        manager.vocab_size = 32
        manager.max_position_embeddings = 128
        # Skip the real request constructor; no engine or accelerator is needed.
        request = object.__new__(TensorLoRARequest) if from_tensors else SimpleNamespace()
        request.lora_int_id = 111
        request.lora_path = '/unused-adapter'
        request.peft_config = {'r': 2, 'lora_alpha': 4}
        request.lora_tensors = tensors
        return manager._load_adapter(request)

    return load


@pytest.mark.parametrize('from_tensors', [True, False])
@pytest.mark.parametrize('partial', [True, False])
@pytest.mark.parametrize('peft_prefix', ['', 'base_model.model.'])
def test_projected_loras_remain_distinct_before_packing(load_adapter, from_tensors, partial, peft_prefix):
    projections = ['q_proj', 'v_proj', 'up_proj'] if partial else [
        'q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj']
    weights = {}
    for index, name in enumerate(projections, 1):
        part = 'self_attn' if name in ('q_proj', 'k_proj', 'v_proj') else 'mlp'
        prefix = f'{peft_prefix}model.layers.0.{part}.{name}'
        weights[f'{prefix}.lora_A.weight'] = torch.full((2, 4), float(index))
        weights[f'{prefix}.lora_B.weight'] = torch.full((6, 2), float(index + 1))

    model = load_adapter(weights, StackingMapper(), from_tensors=from_tensors)
    expected_names = {name.removeprefix(peft_prefix).replace('model.', 'language_model.model.', 1).rsplit('.', 2)[0]
                      for name in weights}
    assert set(model.loras) == expected_names
    for name, tensor in weights.items():
        mapped = name.removeprefix(peft_prefix).replace('model.', 'language_model.model.', 1)
        module_name, lora_part, _ = mapped.rsplit('.', 2)
        assert torch.equal(getattr(model.loras[module_name], lora_part.lower()), tensor)


@pytest.mark.parametrize('mapper', [None, RenameMapper()])
def test_older_mapper_without_stacking_helper_keeps_its_contract(load_adapter, mapper):
    weights = {'model.layers.0.self_attn.q_proj.lora_A.weight': torch.ones(2, 4),
               'model.layers.0.self_attn.q_proj.lora_B.weight': torch.ones(6, 2)}
    model = load_adapter(weights, mapper)
    prefix = 'language_model.' if mapper is not None else ''
    assert set(model.loras) == {f'{prefix}model.layers.0.self_attn.q_proj'}
