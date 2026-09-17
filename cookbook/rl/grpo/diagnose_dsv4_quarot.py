#!/usr/bin/env python3
"""CPU-only sampled weight-transform audit. Never loads or rewrites a model.

Compare one BF16 expert with a symmetric per-channel INT8 checkpoint using
random matrix-vector probes. This identifies candidate representation changes,
not end-to-end LoRA correctness. Requires only torch and safetensors.
"""
import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open


def canonical(key):
    match = re.search(r'(?:^|\.)(layers\.\d+\..*)$', key)
    return match.group(1) if match else key


class Checkpoint:
    def __init__(self, directory):
        self.directory = Path(directory).resolve(strict=True)
        self.locations = {}
        self.reads = []
        indices = sorted(self.directory.glob('*.safetensors.index.json'))
        if len(indices) > 1:
            raise ValueError(f'Multiple safetensors indices: {indices}')
        if indices:
            mapping = json.loads(indices[0].read_text())['weight_map']
            for key, filename in mapping.items():
                self.add(key, self.checked_path(filename))
        else:
            # Header inspection only; tensor data is not loaded here.
            for file in sorted(self.directory.glob('*.safetensors')):
                with safe_open(str(file), framework='pt', device='cpu') as handle:
                    for key in handle.keys():
                        self.add(key, file)
        if not self.locations:
            raise ValueError(f'No checkpoint tensors found in {self.directory}')

    def checked_path(self, name):
        path = (self.directory / name).resolve(strict=True)
        path.relative_to(self.directory)
        return path

    def add(self, key, path):
        normalized = canonical(key)
        if normalized in self.locations:
            raise ValueError(f'Ambiguous normalized key: {normalized}')
        self.locations[normalized] = (key, path)

    def select(self, candidates, required=True):
        found = [key for key in candidates if key in self.locations]
        if len(found) > 1:
            raise ValueError(f'Ambiguous candidates: {found}')
        if not found and required:
            raise KeyError(f'None of these keys found: {candidates}')
        return found[0] if found else None

    def read(self, key, index=None):
        actual, file = self.locations[key]
        with safe_open(str(file), framework='pt', device='cpu') as handle:
            view = handle.get_slice(actual)
            shape = view.get_shape()
            # Do not accidentally materialize all 256 fused experts.
            if index is None:
                if len(shape) > 2:
                    raise ValueError(f'Refusing full tensor read: {actual} {shape}')
                tensor = handle.get_tensor(actual)
            else:
                if len(shape) != 3 or not 0 <= index < shape[0]:
                    raise ValueError(f'Invalid expert slice: {actual} {shape}, expert={index}')
                tensor = view[index, :, :]
        self.reads.append(dict(key=actual, file=str(file), stored_shape=shape,
                               selected_shape=list(tensor.shape), dtype=str(tensor.dtype), expert=index))
        return tensor


def weight_key(checkpoint, layer, expert, proj):
    return checkpoint.select([
        f'layers.{layer}.{parent}.experts.{expert}.{proj}.weight'
        for parent in ('ffn', 'mlp')
    ], required=False)


def original_weight(checkpoint, layer, expert, proj, hidden, intermediate):
    key = weight_key(checkpoint, layer, expert, proj)
    if key:
        value = checkpoint.read(key)
    else:
        name = 'down_proj' if proj == 'w2' else 'gate_up_proj'
        fused = checkpoint.select([
            f'layers.{layer}.mlp.experts.{name}',
            f'layers.{layer}.mlp.experts.{name}.weight',
        ])
        value = checkpoint.read(fused, index=expert)
        if proj != 'w2':
            if tuple(value.shape) != (2 * intermediate, hidden):
                raise ValueError(f'Unsupported fused gate/up layout: {tuple(value.shape)}')
            value = value[:intermediate] if proj == 'w1' else value[intermediate:]
    if value.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f'Original weights must be floating BF16/FP16/FP32, got {value.dtype}')
    expected = (hidden, intermediate) if proj == 'w2' else (intermediate, hidden)
    if tuple(value.shape) != expected:
        raise ValueError(f'{proj}: source shape {tuple(value.shape)} != {expected}')
    return value.float()


def quantized_weight(checkpoint, description, layer, expert, proj, expected):
    key = weight_key(checkpoint, layer, expert, proj)
    if key is None:
        raise KeyError(f'Expected per-expert w1/w2/w3 checkpoint: layer={layer}, expert={expert}, {proj}')
    actual, _ = checkpoint.locations[key]
    kind = description.get(actual, description.get(key))
    if kind != 'W8A8_DYNAMIC':
        raise ValueError(f'{actual}: expected W8A8_DYNAMIC descriptor, got {kind!r}')
    value = checkpoint.read(key)
    if value.dtype != torch.int8 or tuple(value.shape) != expected:
        raise ValueError(f'Unsupported quantized storage: {value.dtype} {tuple(value.shape)}; expected INT8 {expected}')
    prefix = key.removesuffix('.weight')
    scale = checkpoint.read(prefix + '.weight_scale')
    if tuple(scale.shape) not in ((expected[0],), (expected[0], 1)):
        raise ValueError(f'Expected per-output-channel scale, got {tuple(scale.shape)}')
    scale = scale.float().reshape(-1, 1)
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError('Quantization scales must be positive and finite')
    offset_key = prefix + '.weight_offset'
    if offset_key in checkpoint.locations:
        offset = checkpoint.read(offset_key)
        if not torch.isfinite(offset.float()).all() or torch.count_nonzero(offset):
            raise ValueError('Nonzero/invalid weight_offset: refusing to guess the asymmetric dequantization convention')
    else:
        raise ValueError('Missing weight_offset: symmetric quantization has not been verified')
    return value.float() * scale


def metrics(reference, candidate):
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise ValueError('Nonfinite comparison input')
    ref = reference.double().reshape(-1)
    cand = candidate.double().reshape(-1)
    error = cand - ref
    eps = 1e-30
    return dict(relative_l2=float(error.norm() / ref.norm().clamp_min(eps)),
                cosine=float(torch.dot(ref, cand) / (ref.norm() * cand.norm()).clamp_min(eps)),
                max_abs_error=float(error.abs().max()),
                rmse=float(error.square().mean().sqrt()))


def compare_projection(weight, dequant, rotation, gamma, proj, probes):
    observed = dequant @ probes
    results = []

    def add(name, candidate):
        results.append(dict(name=name, **metrics(observed, candidate)))

    add('identity: W', weight @ probes)
    if proj in ('w1', 'w3'):
        diagonals = [('none', None)]
        if gamma is not None:
            diagonals.append(('gamma', gamma))
            if torch.all(gamma.abs() > 1e-12):
                diagonals.append(('inverse_gamma', gamma.reciprocal()))
        for name, diagonal in diagonals:
            for orient, q in [('Q', rotation), ('Q.T', rotation.T)]:
                # Candidate W D Q. Associate products to avoid full W@Q.
                transformed = q @ probes
                if diagonal is not None:
                    transformed = diagonal[:, None] * transformed
                add(f'W D({name}) {orient}', weight @ transformed)
            if diagonal is not None:
                add(f'W D({name})', weight @ (diagonal[:, None] * probes))
    else:
        intermediate = weight @ probes
        add('Q W', rotation @ intermediate)
        add('Q.T W', rotation.T @ intermediate)
    return sorted(results, key=lambda item: item['relative_l2'])


def run(args, report):
    torch.set_num_threads(args.threads)
    original = Checkpoint(args.bf16_model)
    quant = Checkpoint(args.quant_model)
    report['source_reads'] = original.reads
    report['quantized_reads'] = quant.reads
    config = json.loads((original.directory / 'config.json').read_text())
    hidden = int(config['hidden_size'])
    intermediate = int(config['moe_intermediate_size'])
    if not 0 <= args.layer < int(config['num_hidden_layers']):
        raise ValueError('Layer is out of range')
    if not 0 <= args.expert < int(config['n_routed_experts']):
        raise ValueError('Expert is out of range')
    qconfig = json.loads((quant.directory / 'config.json').read_text())
    for key in ('hidden_size', 'moe_intermediate_size', 'n_routed_experts', 'num_hidden_layers'):
        if int(qconfig[key]) != int(config[key]):
            raise ValueError(f'Base configuration mismatch: {key}')
    description = json.loads((quant.directory / 'quant_model_description.json').read_text())
    relative = description['optional']['quarot']['rotation_map']['global_rotation']
    path = quant.checked_path(relative)
    with safe_open(str(path), framework='pt', device='cpu') as handle:
        if handle.get_slice('global_rotation').get_shape() != [hidden, hidden]:
            raise ValueError('Rotation shape does not match hidden_size')
        rotation = handle.get_tensor('global_rotation').float()
    if not torch.isfinite(rotation).all():
        raise ValueError('Nonfinite rotation matrix')
    rng = torch.Generator(device='cpu').manual_seed(args.seed)
    probe = torch.randn(hidden, args.probes, generator=rng) / hidden**0.5
    report['rotation'] = dict(path=str(path), shape=list(rotation.shape),
                              orthogonality_probe=metrics(probe, rotation.T @ (rotation @ probe)),
                              transpose_difference=metrics(rotation, rotation.T))
    norm_names = [f'layers.{args.layer}.ffn_norm.weight',
                  f'layers.{args.layer}.post_attention_layernorm.weight']
    norm_key = original.select(norm_names, required=False)
    gamma = original.read(norm_key).float() if norm_key else None
    report['source_norm_key'] = norm_key
    if gamma is not None:
        if tuple(gamma.shape) != (hidden,) or not torch.isfinite(gamma).all():
            raise ValueError('Invalid source norm')
        report['source_norm'] = dict(min=float(gamma.min()), max=float(gamma.max()),
                                     max_distance_from_one=float((gamma - 1).abs().max()))
    else:
        report['notes'].append('Source FFN norm not found: norm-fusion candidates were not tested.')
    quant_norm_key = quant.select(norm_names, required=False)
    if quant_norm_key:
        quant_norm = quant.read(quant_norm_key).float()
        report['quantized_norm'] = dict(key=quant_norm_key, shape=list(quant_norm.shape),
                                        max_distance_from_one=float((quant_norm - 1).abs().max()))
    report['projections'] = {}
    for proj in ('w1', 'w3', 'w2'):
        print(f'Comparing layer {args.layer}, expert {args.expert}, {proj} ...', flush=True)
        weight = original_weight(original, args.layer, args.expert, proj, hidden, intermediate)
        dequant = quantized_weight(quant, description, args.layer, args.expert, proj, tuple(weight.shape))
        probes = torch.randn(weight.shape[1], args.probes, generator=rng) / weight.shape[1]**0.5
        scores = compare_projection(weight, dequant, rotation, gamma, proj, probes)
        report['projections'][proj] = scores
        for row in scores[:3]:
            print(f"  {row['name']}: relative_l2={row['relative_l2']:.6g}, cosine={row['cosine']:.6g}", flush=True)
        del weight, dequant
    report['completed'] = True
    report['notes'].extend([
        'Smallest error is only the best among tested hypotheses, not proof that the transform is complete.',
        'Orthogonal symmetric Q can make Q and Q.T indistinguishable; do not infer an orientation from ties.',
        'No real adapter, activation quantization, routing, generation, or NPU runtime was tested.',
        'Large errors for all candidates may indicate smoothing, different base provenance, or unsupported transforms.',
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bf16-model', required=True, type=Path)
    parser.add_argument('--quant-model', required=True, type=Path)
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--expert', type=int, default=0)
    parser.add_argument('--probes', type=int, default=32)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    if min(args.probes, args.threads) < 1 or args.probes > 256:
        parser.error('threads must be positive; probes must be in 1..256')
    report = dict(scope='Sampled CPU weight-transform hypotheses, NOT end-to-end LoRA validation',
                  completed=False, arguments={k: str(v) if isinstance(v, Path) else v
                                              for k, v in vars(args).items()},
                  torch_version=torch.__version__, notes=[])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: do not overwrite previous evidence or model files.
    with args.report.open('x', encoding='utf-8') as stream:
        try:
            with torch.inference_mode():
                run(args, report)
        except Exception as exc:
            report['fatal_error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            print(f'Report saved: {args.report}', flush=True)


if __name__ == '__main__':
    main()
