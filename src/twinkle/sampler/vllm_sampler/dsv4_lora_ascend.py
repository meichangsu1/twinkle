# Copyright (c) ModelScope Contributors. All rights reserved.
"""Opt-in Ascend W8A8_DYNAMIC QuaRot adaptation for source-native routed LoRA.

Only checkpoint headers, FFN norms and Q are read at initialization. Online
adapter tensors still arrive through the existing Checkpoint Engine. CPU FP32
math deliberately matches the validated offline converter; no base is reloaded
or quantized here. This is not a general ModelSlim recipe detector.
"""
import json
import re
import torch
from pathlib import Path
from safetensors import safe_open

from .dsv4_lora import PREFIX, DeepSeekV4LoraAdapter

QUAROT_RECIPE = 'w8a8_dynamic_quarot_v1'
_DIMENSIONS = ('num_hidden_layers', 'n_routed_experts', 'hidden_size', 'moe_intermediate_size')


def _read_json(path):
    with path.open(encoding='utf-8') as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return value


class _Checkpoint:
    """Index/header-only lookup; read only explicitly requested constants."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve(strict=True)
        if not self.directory.is_dir():
            raise ValueError(f'Expected local checkpoint directory: {self.directory}')
        self.locations = {}
        self._paths = {}
        indices = sorted(self.directory.glob('*.safetensors.index.json'))
        if len(indices) > 1:
            raise ValueError(f'Multiple safetensors indices: {self.directory}')
        if indices:
            for key, filename in _read_json(indices[0])['weight_map'].items():
                self._add(key, self.checked_path(filename))
        else:
            for file in sorted(self.directory.glob('*.safetensors')):
                with safe_open(str(file), framework='pt', device='cpu') as handle:
                    for key in handle.keys():
                        self._add(key, file)
        if not self.locations:
            raise ValueError(f'No checkpoint tensors: {self.directory}')

    def checked_path(self, name):
        if name in self._paths:
            return self._paths[name]
        path = (self.directory / name).resolve(strict=True)
        path.relative_to(self.directory)
        self._paths[name] = path
        return path

    def _add(self, key, path):
        match = re.search(r'(?:^|\.)(layers\.\d+\..*)$', key)
        normalized = match[1] if match else key
        if normalized in self.locations:
            raise ValueError(f'Ambiguous normalized checkpoint key: {normalized}')
        self.locations[normalized] = (key, path)

    def select(self, candidates):
        found = [key for key in candidates if key in self.locations]
        if len(found) != 1:
            raise ValueError(f'Expected one checkpoint key from {candidates}, found {found}')
        return found[0]

    def norm(self, layer, hidden):
        key = self.select([f'layers.{layer}.ffn_norm.weight', f'layers.{layer}.post_attention_layernorm.weight'])
        actual, path = self.locations[key]
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            if handle.get_slice(actual).get_shape() != [hidden]:
                raise ValueError(f'Layer {layer}: invalid FFN norm dimensions')
            value = handle.get_tensor(actual)
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f'Layer {layer}: expected finite floating-point FFN norm')
        return value.float().clone()


@torch.no_grad()
@torch.autocast(device_type='cpu', enabled=False)
def _load_constants(training_base, target_context):
    """Use Q from the actual rollout model, never an independent quant-base option."""
    model_path = target_context.get('model_path')
    if not model_path:
        raise ValueError('Ascend QuaRot requires the actual rollout model_path in target_context')
    original, quant = _Checkpoint(training_base), _Checkpoint(model_path)
    source_config = _read_json(original.directory / 'config.json')
    quant_config = _read_json(quant.directory / 'config.json')
    target = target_context['model_config']
    for config in (source_config, quant_config, target):
        if config.get('model_type') != 'deepseek_v4':
            raise ValueError('QuaRot requires DeepSeek-V4 source and target bases')
        for key in _DIMENSIONS:
            if type(config.get(key)) is not int or config[key] <= 0 or config[key] != target.get(key):
                raise ValueError(f'QuaRot base configuration mismatch: {key}')
    hidden = target['hidden_size']
    description = _read_json(quant.directory / 'quant_model_description.json')
    try:
        rotation_file = description['optional']['quarot']['rotation_map']['global_rotation']
    except (KeyError, TypeError) as exc:
        raise ValueError('Quantized base has no global QuaRot rotation recipe') from exc
    path = quant.checked_path(rotation_file)
    with safe_open(str(path), framework='pt', device='cpu') as handle:
        if handle.get_slice('global_rotation').get_shape() != [hidden, hidden]:
            raise ValueError('QuaRot rotation dimensions do not match the training base')
        rotation = handle.get_tensor('global_rotation')
    if not rotation.is_floating_point() or not torch.isfinite(rotation).all():
        raise ValueError('Expected finite floating-point QuaRot rotation')
    rotation = rotation.float().clone()
    # Same deterministic, inexpensive orthogonality probe as the offline path.
    probe = torch.randn(hidden, 8, dtype=torch.float32, device='cpu', generator=torch.Generator().manual_seed(42))
    error = float((rotation.T @ (rotation @ probe) - probe).norm() / probe.norm())
    if not error <= 0.005:
        raise ValueError(f'Rotation fails approximate orthogonality check: {error}')
    gammas = {}
    for layer in range(target['num_hidden_layers']):
        gammas[layer] = original.norm(layer, hidden)
        norm = quant.norm(layer, hidden)
        if not torch.equal(norm, torch.ones_like(norm)):
            raise ValueError(f'Layer {layer}: expected all-one quantized FFN norm (norm-fusion recipe)')
        for expert in range(target['n_routed_experts']):
            for proj in ('w1', 'w2', 'w3'):
                key = quant.select([
                    f'layers.{layer}.{parent}.experts.{expert}.{proj}.weight' for parent in ('ffn', 'mlp')
                ])
                actual = quant.locations[key][0]
                if description.get(actual, description.get(key)) != 'W8A8_DYNAMIC':
                    raise ValueError(f'{actual}: expected W8A8_DYNAMIC')
    return rotation, gammas


@torch.no_grad()
@torch.autocast(device_type='cpu', enabled=False)
def transform_quarot_pair(a, b, projection, gamma, rotation):
    """FP32 CPU coordinate transform, preserving source dtype/device and scaling."""
    if a.dtype not in (torch.bfloat16, torch.float16) or b.dtype != a.dtype or a.device != b.device:
        raise ValueError('QuaRot expects BF16/FP16 A/B on the same device')
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Expected finite LoRA weights')
    if projection == 'gate_up_proj':
        source = a.detach().cpu()
        result = torch.empty_like(source)
        for start in range(0, source.shape[0], 16):
            result[start:start + 16] = (
                (source[start:start + 16].float() * gamma[None, None, :]) @ rotation).to(a.dtype)
        output = result.to(a.device), b
    elif projection == 'down_proj':
        source = b.detach().cpu()
        result = torch.empty_like(source)
        for start in range(0, source.shape[0], 16):
            result[start:start + 16] = (rotation.T @ source[start:start + 16].float()).to(b.dtype)
        output = a, result.to(b.device)
    else:
        raise ValueError(f'Unsupported QuaRot target: {projection}')
    if not all(torch.isfinite(value).all() for value in output):
        raise ValueError('QuaRot adaptation overflowed the adapter storage dtype')
    return output


class DeepSeekV4AscendQuaRotLoraAdapter(DeepSeekV4LoraAdapter):
    """Explicitly selected audited norm-fusion/global-rotation recipe, routed-only."""

    def initialize(self, target_context, options):
        if str(target_context['device']).split(':', 1)[0] not in ('npu', 'privateuseone'):
            raise ValueError('Ascend QuaRot synchronization requires NPU')
        self._initialize(target_context)
        if (set(options) != {'training_base_model', 'recipe'} or options.get('recipe') != QUAROT_RECIPE
                or not isinstance(options.get('training_base_model'), str) or not options['training_base_model']):
            raise ValueError(f'Ascend QuaRot requires training_base_model and recipe={QUAROT_RECIPE!r}')
        self.rotation, self.gammas = _load_constants(options['training_base_model'], target_context)

    def adapt_numeric_pair(self, layer, projection, a, b):
        return transform_quarot_pair(a, b, projection, self.gammas[layer], self.rotation)

    def map_name(self, layer, expert, projection, side):
        return f'{PREFIX}.{layer}.mlp.experts.{expert}.{projection}.lora_{side}.weight'
