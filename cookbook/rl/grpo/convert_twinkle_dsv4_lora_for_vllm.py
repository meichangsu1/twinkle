#!/usr/bin/env python3
"""Convert a Twinkle Transformers DeepSeek-V4 LoRA adapter for vLLM.

The converter handles the modules produced by the recommended Twinkle config::

    LoraConfig(
        target_modules="all-linear",
        exclude_modules=["o_a_proj"],
        target_parameters=[
            "mlp.experts.gate_up_proj",
            "mlp.experts.down_proj",
        ],
    )

That includes regular attention/compressor/indexer linears, shared experts,
and Twinkle's 3D routed-expert parameters. vLLM-Ascend is the default target;
NVIDIA vLLM (H800/H100) can be selected explicitly. Routed experts can be
emitted as per-expert 2D w1/w2/w3 tensors (the default) or as the original 3D
fused PEFT parameters.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


LORA_WEIGHT_RE = re.compile(
    r"^(?P<module>.+)\.lora_(?P<side>[AB])(?:\.[^.]+)?\.weight$"
)
LAYER_MODULE_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.(?P<suffix>.+)$"
)
ROUTED_EXPERT_SUFFIX = "mlp.experts"
CANONICAL_LAYER_PREFIX = "base_model.model.model.layers.{layer}"
ROUTED_TARGET_PARAMETERS = [
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
]

# Longest/specialized paths must be checked before their parent paths.
# Destination names below use the original/NVIDIA DeepSeek-V4 adapter-visible
# layout. vLLM-Ascend uses a self_attn/mlp hierarchy and is handled separately.
NORMAL_MODULE_MAPPING = {
    "self_attn.compressor.indexer.scorer.weights_proj": "attn.indexer.weights_proj",
    # Transformers 5.14 checkpoints may place weights_proj directly on the
    # indexer instead of under an intermediate scorer module.
    "self_attn.compressor.indexer.weights_proj": "attn.indexer.weights_proj",
    "self_attn.compressor.indexer.q_b_proj": "attn.indexer.wq_b",
    "self_attn.compressor.indexer.kv_proj": "attn.indexer.compressor.wkv",
    "self_attn.compressor.indexer.gate_proj": "attn.indexer.compressor.wgate",
    "self_attn.compressor.kv_proj": "attn.compressor.wkv",
    "self_attn.compressor.gate_proj": "attn.compressor.wgate",
    "self_attn.q_a_proj": "attn.wq_a",
    "self_attn.q_b_proj": "attn.wq_b",
    "self_attn.kv_proj": "attn.wkv",
    "self_attn.o_a_proj": "attn.wo_a",
    "self_attn.o_b_proj": "attn.wo_b",
    "mlp.shared_experts.gate_proj": "ffn.shared_experts.w1",
    "mlp.shared_experts.down_proj": "ffn.shared_experts.w2",
    "mlp.shared_experts.up_proj": "ffn.shared_experts.w3",
}


def backend_module_suffix(mapped_suffix: str, backend: str) -> str:
    if backend == "nvidia":
        return mapped_suffix
    if mapped_suffix.startswith("attn."):
        return "self_attn." + mapped_suffix.removeprefix("attn.")
    if mapped_suffix.startswith("ffn."):
        return "mlp." + mapped_suffix.removeprefix("ffn.")
    raise ValueError(f"cannot map module to {backend}: {mapped_suffix}")


def runtime_target_name(mapped_suffix: str, backend: str) -> str:
    """Return the vLLM runtime suffix used to select LoRA wrappers."""
    output_name = mapped_suffix.rsplit(".", 1)[-1]
    if backend == "ascend":
        return {
            "w1": "gate_proj",
            "w2": "down_proj",
            "w3": "up_proj",
        }.get(output_name, output_name)
    # NVIDIA keeps routed/shared w1 and w3 adapter-visible. Shared w2 is
    # renamed to the runtime down_proj by the model's hf_to_vllm_mapper.
    if mapped_suffix.endswith(".shared_experts.w2"):
        return "down_proj"
    return output_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Twinkle adapter directory")
    parser.add_argument("output", type=Path, help="New vLLM adapter directory (must not exist)")
    parser.add_argument(
        "--backend",
        choices=("nvidia", "ascend"),
        default="ascend",
        help="vLLM DeepSeek-V4 implementation to target (default: ascend for NPU)",
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        help="Optional base-model directory used to validate dimensions and expert count",
    )
    parser.add_argument(
        "--quarot-model",
        type=Path,
        help="Opt-in: adapt routed-only LoRA to this audited Ascend W8A8_DYNAMIC QuaRot base. Requires --base-model (training BF16 base) and --format 2d.",
    )
    parser.add_argument(
        "--format",
        choices=("2d", "3d"),
        default="2d",
        help=(
            "Routed-expert layout. DeepSeek-V4 loaders that list "
            "experts.<id>.w1/w2/w3 as targets require 2d (default). "
            "Regular linears and shared experts are always emitted as 2D LoRA tensors."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def locate_weights(source: Path) -> Path:
    candidates = [
        source / "adapter_model.safetensors",
        source / "adapter_model.bin",
    ]
    for candidate in candidates:
        if candidate.is_file():
            if candidate.suffix != ".safetensors":
                raise ValueError("Only adapter_model.safetensors is supported")
            return candidate
    raise FileNotFoundError(f"adapter_model.safetensors not found under {source}")


def canonicalize_module(raw_module: str) -> tuple[str, int]:
    """Strip PEFT wrappers and return ``model.layers.N...`` plus base-layer depth."""
    marker = "model.layers."
    marker_index = raw_module.rfind(marker)
    if marker_index < 0:
        raise ValueError(f"unsupported non-layer LoRA module: {raw_module}")

    module = raw_module[marker_index:]
    base_layer_depth = 0
    while module.endswith(".base_layer"):
        module = module.removesuffix(".base_layer")
        base_layer_depth += 1
    if LAYER_MODULE_RE.fullmatch(module) is None:
        raise ValueError(f"invalid DeepSeek-V4 layer module: {raw_module}")
    return module, base_layer_depth


def parse_lora_key(key: str) -> tuple[str, int, str] | None:
    match = LORA_WEIGHT_RE.fullmatch(key)
    if match is None:
        return None
    module, base_layer_depth = canonicalize_module(match.group("module"))
    return module, base_layer_depth, match.group("side")


def split_layer_module(module: str) -> tuple[int, str]:
    match = LAYER_MODULE_RE.fullmatch(module)
    if match is None:
        raise ValueError(f"invalid DeepSeek-V4 layer module: {module}")
    return int(match.group("layer")), match.group("suffix")


def map_normal_module(module: str, backend: str) -> tuple[int, str]:
    layer, suffix = split_layer_module(module)
    try:
        mapped_suffix = NORMAL_MODULE_MAPPING[suffix]
    except KeyError as exc:
        supported = "\n  ".join(sorted(NORMAL_MODULE_MAPPING))
        raise ValueError(
            f"unsupported DeepSeek-V4 LoRA module: {module}\n"
            f"Supported Transformers module suffixes:\n  {supported}"
        ) from exc
    return layer, backend_module_suffix(mapped_suffix, backend)


def validate_2d_pair(module: str, a_tensor: Any, b_tensor: Any) -> int:
    if a_tensor.ndim != 2 or b_tensor.ndim != 2:
        raise ValueError(
            f"{module}: expected 2D LoRA tensors, got "
            f"A={tuple(a_tensor.shape)}, B={tuple(b_tensor.shape)}"
        )
    if a_tensor.shape[0] != b_tensor.shape[1]:
        raise ValueError(
            f"{module}: LoRA ranks differ: "
            f"A={tuple(a_tensor.shape)}, B={tuple(b_tensor.shape)}"
        )
    return int(a_tensor.shape[0])


def classify_routed_pair(
    layer: int,
    a_tensor: Any,
    b_tensor: Any,
    hidden_size: int | None,
    intermediate_size: int | None,
    num_experts: int | None,
) -> str:
    if a_tensor.ndim != 3 or b_tensor.ndim != 3:
        raise ValueError(
            f"layer {layer}: expected 3D routed-expert LoRA tensors, got "
            f"A={tuple(a_tensor.shape)}, B={tuple(b_tensor.shape)}"
        )
    if a_tensor.shape[0] != b_tensor.shape[0]:
        raise ValueError(f"layer {layer}: A/B expert counts differ")
    if a_tensor.shape[1] != b_tensor.shape[2]:
        raise ValueError(f"layer {layer}: A/B LoRA ranks differ")
    if num_experts is not None and a_tensor.shape[0] != num_experts:
        raise ValueError(
            f"layer {layer}: expected {num_experts} experts, got {a_tensor.shape[0]}"
        )

    input_size = int(a_tensor.shape[2])
    output_size = int(b_tensor.shape[1])
    if hidden_size is not None and intermediate_size is not None:
        if input_size == hidden_size and output_size == 2 * intermediate_size:
            return "gate_up_proj"
        if input_size == intermediate_size and output_size == hidden_size:
            return "down_proj"
        raise ValueError(
            f"layer {layer}: unsupported routed-expert LoRA shapes "
            f"A={tuple(a_tensor.shape)}, B={tuple(b_tensor.shape)}; expected "
            f"gate_up A=[E,r,{hidden_size}], B=[E,{2 * intermediate_size},r] or "
            f"down A=[E,r,{intermediate_size}], B=[E,{hidden_size},r]"
        )

    # DeepSeek-V4 has hidden_size > moe_intermediate_size. This fallback is
    # intentionally limited to choosing between the two routed targets.
    return "gate_up_proj" if input_size > output_size // 2 else "down_proj"


def add_tensor(converted: dict[str, Any], key: str, tensor: Any) -> None:
    if key in converted:
        raise ValueError(f"multiple source tensors map to {key}")
    converted[key] = tensor.contiguous()


def transform_quarot_pair(a, b, target, gamma, rotation):
    """Apply the audited offline weight transform, before per-expert splitting.

    gate/up: delta' = delta D(gamma) Q; down: delta' = Q.T delta.
    All compute is FP32 on CPU; preserve adapter storage dtype and scaling.
    """
    import torch
    if a.device.type != 'cpu' or b.device.type != 'cpu':
        raise ValueError('QuaRot conversion expects CPU tensors')
    with torch.inference_mode():
        if target == 'gate_up_proj':
            # Batch across experts to avoid materializing dense delta matrices.
            result = torch.empty_like(a)
            for start in range(0, a.shape[0], 16):
                result[start:start + 16] = (
                    (a[start:start + 16].float() * gamma[None, None, :]) @ rotation
                ).to(a.dtype)
            return result, b
        if target == 'down_proj':
            result = torch.empty_like(b)
            for start in range(0, b.shape[0], 16):
                result[start:start + 16] = (
                    rotation.T @ b[start:start + 16].float()
                ).to(b.dtype)
            return a, result
    raise ValueError(f'Unsupported QuaRot target: {target}')


class QuaRotAdapterTransform:
    def __init__(self, source_base, quant_base, layers, hidden_size, num_experts):
        import hashlib
        import torch
        from safetensors import safe_open
        # Reuse the CPU/header-only reader from the diagnostic script.
        try:
            from diagnose_dsv4_quarot import Checkpoint
        except ImportError as exc:
            raise RuntimeError('Place diagnose_dsv4_quarot.py beside this converter for --quarot-model') from exc
        original = Checkpoint(source_base)
        quant = Checkpoint(quant_base)
        source_config = load_json(original.directory / 'config.json')
        quant_config = load_json(quant.directory / 'config.json')
        for name in ('hidden_size', 'moe_intermediate_size', 'n_routed_experts', 'num_hidden_layers'):
            if int(source_config[name]) != int(quant_config[name]):
                raise ValueError(f'QuaRot base configuration mismatch: {name}')
        desc = load_json(quant.directory / 'quant_model_description.json')
        path = quant.checked_path(desc['optional']['quarot']['rotation_map']['global_rotation'])
        with safe_open(str(path), framework='pt', device='cpu') as f:
            if f.get_slice('global_rotation').get_shape() != [hidden_size, hidden_size]:
                raise ValueError('QuaRot rotation dimensions do not match the training base')
            self.rotation = f.get_tensor('global_rotation').float()
        if not torch.isfinite(self.rotation).all():
            raise ValueError('Nonfinite QuaRot rotation')
        rng = torch.Generator().manual_seed(42)
        probe = torch.randn(hidden_size, 8, generator=rng)
        residual = self.rotation.T @ (self.rotation @ probe) - probe
        orthogonal_error = float(residual.norm() / probe.norm())
        if orthogonal_error > 0.005:
            raise ValueError(f'Rotation fails approximate orthogonality check: {orthogonal_error}')
        self.gammas = {}
        norms = {}
        for layer in sorted(layers):
            if not 0 <= layer < int(source_config['num_hidden_layers']):
                raise ValueError(f'Out-of-range adapter layer: {layer}')
            candidates = [f'layers.{layer}.ffn_norm.weight',
                          f'layers.{layer}.post_attention_layernorm.weight']
            gamma = original.read(original.select(candidates)).float()
            quant_norm = quant.read(quant.select(candidates)).float()
            if (tuple(gamma.shape) != (hidden_size,) or not torch.isfinite(gamma).all()
                    or tuple(quant_norm.shape) != (hidden_size,)
                    or not torch.equal(quant_norm, torch.ones_like(quant_norm))):
                raise ValueError(f'Layer {layer}: expected finite source gamma and all-one quantized FFN norm')
            # Verify descriptions for all routed experts touched by this adapter.
            for expert in range(num_experts):
                for proj in ('w1', 'w2', 'w3'):
                    key = quant.select([f'layers.{layer}.{parent}.experts.{expert}.{proj}.weight'
                                        for parent in ('ffn', 'mlp')])
                    actual = quant.locations[key][0]
                    if desc.get(actual, desc.get(key)) != 'W8A8_DYNAMIC':
                        raise ValueError(f'{actual}: expected W8A8_DYNAMIC')
            self.gammas[layer] = gamma
            norms[str(layer)] = dict(min=float(gamma.min()), max=float(gamma.max()))
        self.report = dict(
            scope='Offline routed-expert LoRA basis adaptation; not end-to-end runtime validation',
            training_base=str(original.directory), quantized_base=str(quant.directory),
            rotation_path=str(path), rotation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            orthogonality_probe_relative_l2=orthogonal_error, source_norms=norms,
            gate_up_formula='A_new = (A * gamma) @ Q; B_new = B',
            down_formula='A_new = A; B_new = Q.T @ B',
            warning='Assumes audited norm-fusion/global-rotation recipe across selected layers; other smoothing transforms are not inferred.',
        )

    def apply(self, layer, a, b, target):
        import torch
        for tensor in (a, b):
            if tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16) or not torch.isfinite(tensor).all():
                raise ValueError('Expected finite floating-point LoRA weights')
        a, b = transform_quarot_pair(a, b, target, self.gammas[layer], self.rotation)
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('QuaRot adaptation overflowed the adapter storage dtype')
        return a, b


def main() -> None:
    args = parse_args()
    try:
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required: python -m pip install safetensors") from exc

    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    config_path = source / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    adapter_config = load_json(config_path)
    weights = load_file(str(locate_weights(source)), device="cpu")

    hidden_size = intermediate_size = num_experts = None
    if args.base_model is not None:
        base_config = load_json(args.base_model.resolve() / "config.json")
        hidden_size = int(base_config["hidden_size"])
        intermediate_size = int(base_config["moe_intermediate_size"])
        num_experts = int(base_config["n_routed_experts"])

    routed_groups: dict[tuple[int, int], dict[str, tuple[str, Any]]] = defaultdict(dict)
    normal_groups: dict[str, dict[str, tuple[str, Any]]] = defaultdict(dict)
    non_lora_keys: list[str] = []

    for key, tensor in weights.items():
        parsed = parse_lora_key(key)
        if parsed is None:
            non_lora_keys.append(key)
            continue
        module, base_layer_depth, side = parsed
        layer, suffix = split_layer_module(module)
        if suffix == ROUTED_EXPERT_SUFFIX:
            group = routed_groups[(layer, base_layer_depth)]
        else:
            if base_layer_depth:
                raise ValueError(
                    f"unexpected .base_layer wrapper on regular LoRA module: {key}"
                )
            group = normal_groups[module]
        if side in group:
            raise ValueError(f"duplicate LoRA {side} tensor for {module}")
        group[side] = (key, tensor)

    if non_lora_keys:
        details = "\n  ".join(non_lora_keys[:20])
        raise ValueError(f"adapter contains unexpected non-LoRA tensors:\n  {details}")
    if not routed_groups and not normal_groups:
        raise ValueError("no DeepSeek-V4 LoRA tensors were found")

    quarot = None
    if args.quarot_model is not None:
        if args.backend != 'ascend' or args.format != '2d' or args.base_model is None:
            raise ValueError('--quarot-model requires --backend ascend --format 2d --base-model TRAINING_BF16_BASE')
        if normal_groups or not routed_groups:
            raise ValueError('QuaRot adaptation currently supports routed-expert-only adapters; refusing partially converted attention/shared LoRA')
        if adapter_config.get('use_dora') or adapter_config.get('rank_pattern') or adapter_config.get('alpha_pattern'):
            raise ValueError('QuaRot mode does not support DoRA or per-module rank/alpha patterns')
        quarot = QuaRotAdapterTransform(args.base_model, args.quarot_model,
                                       {layer for layer, _ in routed_groups}, hidden_size, num_experts)

    converted: dict[str, Any] = {}
    source_keys: set[str] = set()
    ranks: set[int] = set()
    output_targets: set[str] = set()
    runtime_targets: set[str] = set()
    normal_counts: dict[str, int] = defaultdict(int)

    # Convert ordinary attention/compressor/indexer/shared-expert linears.
    for module, pair in sorted(normal_groups.items()):
        if set(pair) != {"A", "B"}:
            raise ValueError(f"{module}: incomplete A/B pair")
        key_a, tensor_a = pair["A"]
        key_b, tensor_b = pair["B"]
        rank = validate_2d_pair(module, tensor_a, tensor_b)
        layer, mapped_suffix = map_normal_module(module, args.backend)
        prefix = CANONICAL_LAYER_PREFIX.format(layer=layer)
        target_module = f"{prefix}.{mapped_suffix}"
        add_tensor(converted, f"{target_module}.lora_A.weight", tensor_a)
        add_tensor(converted, f"{target_module}.lora_B.weight", tensor_b)
        source_keys.update((key_a, key_b))
        ranks.add(rank)
        output_targets.add(mapped_suffix.rsplit(".", 1)[-1])
        runtime_targets.add(runtime_target_name(mapped_suffix, args.backend))
        category = "shared_expert" if ".shared_experts." in mapped_suffix else "attention"
        normal_counts[category] += 1

    # Classify and convert Twinkle's two 3D routed-expert parameter pairs.
    targets_by_layer: dict[int, set[str]] = defaultdict(set)
    expert_counts: set[int] = set()
    for (layer, base_layer_depth), pair in sorted(routed_groups.items()):
        if set(pair) != {"A", "B"}:
            raise ValueError(
                f"layer {layer}, base_layer_depth={base_layer_depth}: incomplete A/B pair"
            )
        key_a, tensor_a = pair["A"]
        key_b, tensor_b = pair["B"]
        target = classify_routed_pair(
            layer,
            tensor_a,
            tensor_b,
            hidden_size,
            intermediate_size,
            num_experts,
        )
        if target in targets_by_layer[layer]:
            raise ValueError(f"layer {layer}: duplicate routed target {target}")
        targets_by_layer[layer].add(target)
        source_keys.update((key_a, key_b))
        ranks.add(int(tensor_a.shape[1]))
        expert_counts.add(int(tensor_a.shape[0]))

        if quarot is not None:
            print(f'QuaRot adaptation: layer={layer}, target={target}', flush=True)
            tensor_a, tensor_b = quarot.apply(layer, tensor_a, tensor_b, target)

        moe_name = "ffn" if args.backend == "nvidia" else "mlp"
        prefix = f"{CANONICAL_LAYER_PREFIX.format(layer=layer)}.{moe_name}.experts"
        if args.format == "3d":
            add_tensor(converted, f"{prefix}.{target}.lora_A.weight", tensor_a)
            add_tensor(converted, f"{prefix}.{target}.lora_B.weight", tensor_b)
            output_targets.add(target)
        elif target == "gate_up_proj":
            if tensor_b.shape[1] % 2:
                raise ValueError(
                    f"layer {layer}: gate_up LoRA B output dimension must be even, "
                    f"got {tensor_b.shape[1]}"
                )
            split_size = tensor_b.shape[1] // 2
            # Transformers gate_up_proj stores [gate (w1), up (w3)]. Both
            # halves share A and have separate B projections.
            b_w1, b_w3 = tensor_b.split(split_size, dim=1)
            for expert_id in range(tensor_a.shape[0]):
                expert_prefix = f"{prefix}.{expert_id}"
                add_tensor(converted, f"{expert_prefix}.w1.lora_A.weight", tensor_a[expert_id].clone())
                add_tensor(converted, f"{expert_prefix}.w1.lora_B.weight", b_w1[expert_id].clone())
                add_tensor(converted, f"{expert_prefix}.w3.lora_A.weight", tensor_a[expert_id].clone())
                add_tensor(converted, f"{expert_prefix}.w3.lora_B.weight", b_w3[expert_id].clone())
            output_targets.update(("w1", "w3"))
        else:
            for expert_id in range(tensor_a.shape[0]):
                expert_prefix = f"{prefix}.{expert_id}"
                add_tensor(converted, f"{expert_prefix}.w2.lora_A.weight", tensor_a[expert_id].clone())
                add_tensor(converted, f"{expert_prefix}.w2.lora_B.weight", tensor_b[expert_id].clone())
            output_targets.add("w2")

    if routed_groups:
        # The checkpoint contains experts.<id>.w1/w2/w3 (2D) or the two
        # fused expert parameters (3D), but the module wrapped by vLLM is the
        # MoERunner named ``experts``.
        runtime_targets.add("experts")

    required_routed_targets = {"gate_up_proj", "down_proj"}
    for layer, targets in sorted(targets_by_layer.items()):
        if targets != required_routed_targets:
            raise ValueError(
                f"layer {layer}: expected routed targets "
                f"{sorted(required_routed_targets)}, got {sorted(targets)}"
            )
    if len(ranks) != 1:
        raise ValueError(f"inconsistent LoRA ranks: {sorted(ranks)}")
    if len(expert_counts) > 1:
        raise ValueError(f"inconsistent routed expert counts: {sorted(expert_counts)}")
    if source_keys != set(weights):
        extra = sorted(set(weights) - source_keys)
        raise ValueError(f"adapter contains unconverted tensors: {extra[:20]}")

    rank = next(iter(ranks))
    config_rank = int(adapter_config.get("r", rank))
    if config_rank != rank:
        raise ValueError(f"adapter config rank {config_rank} does not match tensor rank {rank}")

    adapter_config["r"] = rank
    adapter_config["target_modules"] = sorted(runtime_targets) or None
    adapter_config["target_parameters"] = (
        ROUTED_TARGET_PARAMETERS if args.format == "3d" and routed_groups else None
    )
    adapter_config["exclude_modules"] = None
    adapter_config["inference_mode"] = True
    adapter_config["bias"] = "none"
    adapter_config["modules_to_save"] = None

    output.mkdir(parents=True)
    save_file(converted, str(output / "adapter_model.safetensors"))
    with (output / "adapter_config.json").open("w", encoding="utf-8") as stream:
        json.dump(adapter_config, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    if quarot is not None:
        with (output / 'quarot_conversion_report.json').open('w', encoding='utf-8') as stream:
            json.dump(quarot.report, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    for filename in ("README.md", "tokenizer_config.json"):
        source_file = source / filename
        if source_file.is_file():
            shutil.copy2(source_file, output / filename)

    print(f"Converted adapter:       {output}")
    print(f"vLLM backend:            {args.backend}")
    print(f"LoRA rank:               {rank}")
    print(f"Attention/aux modules:   {normal_counts['attention']}")
    print(f"Shared-expert modules:   {normal_counts['shared_expert']}")
    print(f"Routed-expert layers:    {len(targets_by_layer)}")
    if expert_counts:
        print(f"Routed experts/layer:    {next(iter(expert_counts))}")
    print(f"Output tensors:          {len(converted)}")
    print(f"Checkpoint targets:      {sorted(output_targets)}")
    print(f"vLLM runtime targets:    {sorted(runtime_targets)}")
    if routed_groups:
        if args.format == "3d":
            print('Routed layout:           3D fused PEFT (use "is_3d_lora_weight": true)')
        else:
            print('Routed layout:           2D per-expert (use "is_3d_lora_weight": false)')


if __name__ == "__main__":
    main()
