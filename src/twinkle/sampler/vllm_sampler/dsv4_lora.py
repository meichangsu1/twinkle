# Copyright (c) ModelScope Contributors. All rights reserved.
"""DeepSeek-V4 LoRA adaptation after rollout has received the complete adapter."""
import math
import re
from copy import deepcopy
from typing import NamedTuple

import torch

from .weight_sync import RolloutWeightAdapter

TARGETS = {'mlp.experts.gate_up_proj', 'mlp.experts.down_proj'}
PREFIX = 'base_model.model.model.layers'
SOURCE_NAME = re.compile(r'(?:.*\.)?layers\.(\d+)\.(.+)\.lora_([AB])\.weight')


class NormalModuleRule(NamedTuple):
    nvidia: str
    ascend: str
    quarot: str


# Source module -> vLLM names and, when needed, the Ascend QuaRot coordinate rule.
NORMAL_MODULE_RULES = {
    'self_attn.compressor.indexer.weights_proj': NormalModuleRule(
        'attn.indexer.weights_proj', 'self_attn.indexer.weights_proj', 'attention_input'),
    'self_attn.compressor.indexer.scorer.weights_proj': NormalModuleRule(
        'attn.indexer.weights_proj', 'self_attn.indexer.weights_proj', 'attention_input'),
    'self_attn.compressor.indexer.q_b_proj': NormalModuleRule(
        'attn.indexer.wq_b', 'self_attn.indexer.wq_b', 'identity'),
    'self_attn.compressor.indexer.kv_proj': NormalModuleRule(
        'attn.indexer.compressor.wkv', 'self_attn.indexer.compressor.wkv', 'attention_input'),
    'self_attn.compressor.indexer.gate_proj': NormalModuleRule(
        'attn.indexer.compressor.wgate', 'self_attn.indexer.compressor.wgate', 'attention_input'),
    'self_attn.compressor.kv_proj': NormalModuleRule(
        'attn.compressor.wkv', 'self_attn.compressor.wkv', 'attention_input'),
    'self_attn.compressor.gate_proj': NormalModuleRule(
        'attn.compressor.wgate', 'self_attn.compressor.wgate', 'attention_input'),
    'self_attn.q_a_proj': NormalModuleRule('attn.wq_a', 'self_attn.wq_a', 'attention_input'),
    'self_attn.q_b_proj': NormalModuleRule('attn.wq_b', 'self_attn.wq_b', 'identity'),
    'self_attn.kv_proj': NormalModuleRule('attn.wkv', 'self_attn.wkv', 'attention_input'),
    'self_attn.o_a_proj': NormalModuleRule('attn.wo_a', 'self_attn.wo_a', 'identity'),
    'self_attn.o_b_proj': NormalModuleRule('attn.wo_b', 'self_attn.wo_b', 'hidden_output'),
    'mlp.shared_experts.gate_proj': NormalModuleRule(
        'ffn.shared_experts.w1', 'mlp.shared_experts.gate_proj', 'ffn_input'),
    'mlp.shared_experts.down_proj': NormalModuleRule(
        'ffn.shared_experts.w2', 'mlp.shared_experts.down_proj', 'hidden_output'),
    'mlp.shared_experts.up_proj': NormalModuleRule(
        'ffn.shared_experts.w3', 'mlp.shared_experts.up_proj', 'ffn_input'),
}


class DeepSeekV4LoraAdapter(RolloutWeightAdapter):
    backend = 'nvidia'

    def _initialize(self, target_context):
        if not target_context['enable_lora']:
            raise ValueError('LoRA synchronization requires enable_lora')
        if target_context['pp_size'] != 1 or target_context['ep_enabled']:
            raise ValueError('Rollout synchronization supports node-local TP, not PP/EP')
        dims = target_context['model_config']
        if dims.get('model_type') != 'deepseek_v4':
            raise ValueError('Expected DeepSeek-V4 rollout')
        for key in ('num_hidden_layers', 'n_routed_experts', 'hidden_size', 'moe_intermediate_size'):
            if type(dims.get(key)) is not int or dims[key] <= 0:
                raise ValueError(f'Invalid target dimension: {key}')
        self.target = target_context

    def _update_config(self, peft_config):
        targets = set(peft_config.get('target_parameters') or [])
        modules = peft_config.get('target_modules')
        if targets - TARGETS or not (targets or modules):
            raise ValueError('Unsupported DSV4 LoRA targets')
        if modules is not None and not isinstance(modules, (str, list, tuple, set)):
            raise ValueError('Unsupported DSV4 LoRA targets')
        unsupported = ('use_dora', 'use_qalora', 'rank_pattern', 'alpha_pattern', 'modules_to_save',
                       'layer_replication', 'trainable_token_indices', 'lora_bias')
        if any(peft_config.get(key) for key in unsupported) or peft_config.get('bias', 'none') != 'none':
            raise ValueError('Unsupported LoRA configuration')
        rank = peft_config['r']
        if type(rank) is not int or rank <= 0 or not math.isfinite(float(peft_config['lora_alpha'])):
            raise ValueError('Invalid LoRA rank/alpha')
        if rank > self.target['max_lora_rank']:
            raise ValueError('Insufficient rollout LoRA rank capacity')
        dtype = str(self.target['lora_dtype'])
        if dtype not in ('torch.bfloat16', 'torch.float16'):
            raise ValueError('LoRA dtype must be BF16/FP16')
        return targets, rank, dtype

    def transform_pair(self, layer, suffix, a, b):
        """Change coordinates before splitting a routed projection; NVIDIA is identity."""
        return a, b

    def map_name(self, layer, expert, projection, side):
        return f'{PREFIX}.{layer}.ffn.experts.{expert}.{projection}.lora_{side}.weight'

    def normal_runtime_target(self, mapped_suffix):
        if mapped_suffix == 'ffn.shared_experts.w2':
            return 'down_proj'
        return mapped_suffix.rsplit('.', 1)[-1]

    def _outputs(self, layer, suffix, a, b):
        if suffix not in TARGETS:
            mapped = getattr(NORMAL_MODULE_RULES[suffix], self.backend)
            runtime = self.normal_runtime_target(mapped)
            yield f'{PREFIX}.{layer}.{mapped}.lora_A.weight', a, runtime
            yield f'{PREFIX}.{layer}.{mapped}.lora_B.weight', b, runtime
            return
        intermediate = self.target['model_config']['moe_intermediate_size']
        projection = suffix.rsplit('.', 1)[-1]
        branches = (('w1', b[:, :intermediate]), ('w3', b[:, intermediate:])) \
            if projection == 'gate_up_proj' else (('w2', b),)
        for expert in range(a.shape[0]):
            for target, b_part in branches:
                yield self.map_name(layer, expert, target, 'A'), a[expert], 'experts'
                yield self.map_name(layer, expert, target, 'B'), b_part[expert], 'experts'

    @torch.no_grad()
    def process(self, weights, peft_config):
        """Convert one complete adapter without keeping any per-update state."""
        targets, rank, dtype = self._update_config(peft_config)
        dims = self.target['model_config']
        pairs = {}
        for name, tensor in weights:
            match = SOURCE_NAME.fullmatch(name)
            if match is None:
                raise ValueError(f'Unsupported source name: {name}')
            layer, suffix, side = int(match[1]), match[2], match[3]
            if not 0 <= layer < dims['num_hidden_layers']:
                raise ValueError(f'Unsupported DSV4 LoRA layer: {name}')
            if suffix in TARGETS:
                if suffix not in targets:
                    raise ValueError(f'Unexpected routed LoRA target: {name}')
                experts, hidden, intermediate = (dims[key] for key in
                                                 ('n_routed_experts', 'hidden_size', 'moe_intermediate_size'))
                ins, outs = ((hidden, 2 * intermediate) if suffix.endswith('gate_up_proj')
                             else (intermediate, hidden))
                expected = (experts, rank, ins) if side == 'A' else (experts, outs, rank)
                if tuple(tensor.shape) != expected or str(tensor.dtype) != dtype:
                    raise ValueError(f'Invalid source tensor shape/dtype for {name}: expected {expected}, {dtype}')
            elif suffix in NORMAL_MODULE_RULES:
                if (tensor.ndim != 2 or str(tensor.dtype) != dtype
                        or (tensor.shape[0] if side == 'A' else tensor.shape[1]) != rank):
                    raise ValueError(f'Invalid ordinary LoRA shape/dtype: {name}')
            else:
                raise ValueError(f'Unsupported DSV4 LoRA module: {name}')
            pair = pairs.setdefault((layer, suffix), {})
            if side in pair:
                raise ValueError(f'Duplicate source LoRA tensor: {name}')
            pair[side] = tensor

        if not pairs:
            raise ValueError('Empty source LoRA')
        if any(set(pair) != {'A', 'B'} for pair in pairs.values()):
            raise ValueError('Incomplete source LoRA pair')
        routed_count = sum(suffix in TARGETS for _, suffix in pairs)
        if routed_count != dims['num_hidden_layers'] * len(targets):
            raise ValueError('Incomplete source routed LoRA')

        converted = {}
        runtime_targets = set()
        for (layer, suffix), pair in pairs.items():
            a, b = pair['A'], pair['B']
            new_a, new_b = self.transform_pair(layer, suffix, a, b)
            if (new_a.shape != a.shape or new_b.shape != b.shape
                    or str(new_a.dtype) != dtype or str(new_b.dtype) != dtype):
                raise ValueError(f'Numeric adaptation changed A/B shape or dtype: {layer}.{suffix}')
            for name, value, runtime in self._outputs(layer, suffix, new_a, new_b):
                if name in converted:
                    raise ValueError(f'Conflicting LoRA target: {name}')
                converted[name] = value.detach().clone().contiguous()
                runtime_targets.add(runtime)

        target_config = deepcopy(peft_config)
        target_config.update(
            target_modules=sorted(runtime_targets), target_parameters=None,
            exclude_modules=None, inference_mode=True, bias='none', modules_to_save=None)
        return list(converted.items()), target_config


class DeepSeekV4NvidiaLoraAdapter(DeepSeekV4LoraAdapter):

    def initialize(self, target_context, options):
        if options:
            raise ValueError('The NVIDIA adapter has no rotation or quantization options')
        if torch.device(target_context['device']).type != 'cuda':
            raise ValueError('NVIDIA LoRA synchronization requires CUDA and enable_lora')
        self._initialize(target_context)
