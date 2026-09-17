# Copyright (c) ModelScope Contributors. All rights reserved.
"""DeepSeek-V4 routed LoRA layout handling, executed only on rollout workers."""
import math
import re
import time
import torch
from copy import deepcopy

from .weight_sync import RolloutWeightAdapter

TARGETS = {'mlp.experts.gate_up_proj', 'mlp.experts.down_proj'}
PREFIX = 'base_model.model.model.layers'
PARAMETER = re.compile(r'(?:.*\.)?layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$')


def target_peft_config(source):
    config = deepcopy(source)
    if config.get('target_modules') or set(config.get('target_parameters') or []) != TARGETS:
        raise ValueError('DSV4 synchronization requires routed-only gate_up_proj/down_proj LoRA')
    for key in ('use_dora', 'use_qalora', 'rank_pattern', 'alpha_pattern', 'modules_to_save', 'layer_replication',
                'trainable_token_indices', 'lora_bias'):
        if config.get(key):
            raise ValueError(f'Unsupported LoRA option: {key}')
    if config.get('bias', 'none') != 'none':
        raise ValueError('Only LoRA bias=none is supported')
    if type(config['r']) is not int or config['r'] <= 0 or not math.isfinite(float(config['lora_alpha'])):
        raise ValueError('Invalid LoRA rank/alpha')
    config.update(
        target_modules=['experts'],
        target_parameters=None,
        exclude_modules=None,
        inference_mode=True,
        bias='none',
        modules_to_save=None)
    return config


class DeepSeekV4LoraAdapter(RolloutWeightAdapter):

    def _initialize(self, target_context):
        if not target_context['enable_lora']:
            raise ValueError('LoRA synchronization requires enable_lora')
        if target_context['pp_size'] != 1 or target_context['ep_enabled']:
            raise ValueError('Rollout synchronization supports node-local TP, not PP/EP')
        self.target = target_context
        self.abort_update()

    def process(self, weights, peft_config):
        """Adapt one complete LoRA without changing the worker's loading policy."""
        try:
            self.begin_update(peft_config)
            self.consume_weights(weights)
            tensors, config, _ = self.finish_update()
            return list(tensors.items()), config
        finally:
            self.abort_update()

    def begin_update(self, peft_config):
        self.abort_update()
        dims = self.target['model_config']
        if dims.get('model_type') != 'deepseek_v4':
            raise ValueError('Expected DeepSeek-V4 rollout')
        for key in ('num_hidden_layers', 'n_routed_experts', 'hidden_size', 'moe_intermediate_size'):
            if type(dims.get(key)) is not int or dims[key] <= 0:
                raise ValueError(f'Invalid target dimension: {key}')
        self.peft_config = target_peft_config(peft_config)
        r = self.peft_config['r']
        if r > self.target['max_lora_rank']:
            raise ValueError('Insufficient rollout LoRA rank capacity')
        self.scale = float(self.peft_config['lora_alpha']) / (math.sqrt(r) if self.peft_config.get('use_rslora') else r)
        self.dtype = str(self.target['lora_dtype'])
        if self.dtype not in ('torch.bfloat16', 'torch.float16'):
            raise ValueError('LoRA dtype must be BF16/FP16')
        self.expected_target_count = dims['num_hidden_layers'] * dims['n_routed_experts'] * 6

    def adapt_numeric_pair(self, layer, projection, a, b):
        """Default identity; a backend may change coordinates before splitting."""
        return a, b

    def convert_layout(self, projection, a, b):
        """Yield independent 2-D tensors, retaining gate/up's shared source A."""
        i = self.target['model_config']['moe_intermediate_size']
        projections = [('w1', b[:, :i]), ('w3', b[:, i:])] if projection == 'gate_up_proj' else [('w2', b)]
        for expert in range(a.shape[0]):
            for target, bp in projections:
                yield expert, target, 'A', a[expert].detach().clone().contiguous()
                yield expert, target, 'B', bp[expert].detach().clone().contiguous()

    def map_name(self, layer, expert, projection, side):
        return f'{PREFIX}.{layer}.ffn.experts.{expert}.{projection}.lora_{side}.weight'

    @torch.no_grad()
    def consume_weights(self, weights):
        for name, tensor in weights:
            match = re.fullmatch(r'(.+)\.lora_([AB])\.weight', name)
            parameter = PARAMETER.fullmatch(match[1]) if match else None
            if parameter is None:
                raise ValueError(f'Unsupported source name: {name}')
            layer, projection, side = int(parameter[1]), parameter[2], match[2]
            slot = (layer, projection, side)
            dims = self.target['model_config']
            if slot in self.seen or not 0 <= layer < dims['num_hidden_layers']:
                raise ValueError(f'Unexpected or duplicate source tensor: {name}')
            e, h, i = (dims[k] for k in ('n_routed_experts', 'hidden_size', 'moe_intermediate_size'))
            ins, outs = (h, 2 * i) if projection == 'gate_up_proj' else (i, h)
            r = self.peft_config['r']
            expected = (e, r, ins) if side == 'A' else (e, outs, r)
            if tuple(tensor.shape) != expected or str(tensor.dtype) != self.dtype:
                raise ValueError(f'Invalid source tensor shape/dtype for {name}: expected {expected}, {self.dtype}')
            self.seen.add(slot)
            self.source_nbytes += tensor.numel() * tensor.element_size()
            group = self.pending.setdefault((layer, projection), {})
            group[side] = tensor
            if set(group) != {'A', 'B'}:
                continue
            if tensor.is_cuda:
                torch.cuda.synchronize(tensor.device)
            start = time.monotonic()
            a, b = self.adapt_numeric_pair(layer, projection, group['A'], group['B'])
            if a.shape != group['A'].shape or b.shape != group['B'].shape:
                raise ValueError('Numeric adaptation changed A/B dimensions')
            for expert, target, ab, value in self.convert_layout(projection, a, b):
                target_name = self.map_name(layer, expert, target, ab)
                if target_name in self.converted or str(value.dtype) != self.dtype:
                    raise ValueError(f'Duplicate target or changed dtype: {target_name}')
                self.converted[target_name] = value
                self.target_nbytes += value.numel() * value.element_size()
            if tensor.is_cuda:
                torch.cuda.synchronize(tensor.device)
            self.convert_seconds += time.monotonic() - start
            del self.pending[(layer, projection)]

    def finish_update(self):
        if self.pending or len(self.seen) != self.target['model_config']['num_hidden_layers'] * 4:
            raise ValueError('Incomplete source LoRA stream')
        if len(self.converted) != self.expected_target_count:
            raise ValueError('Incomplete converted LoRA')
        # Explicit output shape/key validation before calling the existing loader.
        dims = self.target['model_config']
        r, h, i = self.peft_config['r'], dims['hidden_size'], dims['moe_intermediate_size']
        for layer in range(dims['num_hidden_layers']):
            for expert in range(dims['n_routed_experts']):
                for proj in ('w1', 'w2', 'w3'):
                    for side in ('A', 'B'):
                        name = self.map_name(layer, expert, proj, side)
                        expected = ((r, i if proj == 'w2' else h) if side == 'A' else (h if proj == 'w2' else i, r))
                        value = self.converted.get(name)
                        if value is None or tuple(value.shape) != expected or str(value.dtype) != self.dtype:
                            raise ValueError(f'Invalid converted tensor: {name}')
        report = dict(
            source_tensor_count=len(self.seen),
            tensor_count=len(self.converted),
            source_nbytes=self.source_nbytes,
            target_nbytes=self.target_nbytes,
            convert_seconds=self.convert_seconds,
            dtype=self.dtype,
            scale=self.scale)
        return self.converted, self.peft_config, report

    def abort_update(self):
        self.pending = {}
        self.converted = {}
        self.seen = set()
        self.source_nbytes = self.target_nbytes = 0
        self.convert_seconds = 0.0


class DeepSeekV4NvidiaLoraAdapter(DeepSeekV4LoraAdapter):

    def initialize(self, target_context, options):
        if options:
            raise ValueError('The NVIDIA adapter has no rotation or quantization options')
        if torch.device(target_context['device']).type != 'cuda':
            raise ValueError('NVIDIA LoRA synchronization requires CUDA and enable_lora')
        self._initialize(target_context)
