#!/usr/bin/env python3
"""Compare the DeepSeek-V4 square expert layout on NPU GMM against F.linear."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from twinkle.kernel.ops.moe.npu import GmmFunction, _normalize_packed_expert_weights


class PackedExperts(nn.Module):

    def __init__(self, gate_up_proj: torch.Tensor, down_proj: torch.Tensor):
        super().__init__()
        self.gate_up_proj = nn.Parameter(gate_up_proj, requires_grad=False)
        self.down_proj = nn.Parameter(down_proj, requires_grad=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--dtype', choices=('float16', 'bfloat16'), default='bfloat16')
    parser.add_argument('--atol', type=float, default=2e-2)
    parser.add_argument('--rtol', type=float, default=1e-2)
    parser.add_argument('--output', type=Path, default=Path('output/dsv4_ep_diag/npu_gmm_layout.json'))
    return parser.parse_args()


def reference_forward(
    inputs: torch.Tensor,
    counts: list[int],
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
) -> torch.Tensor:
    outputs = []
    start = 0
    for expert, count in enumerate(counts):
        expert_input = inputs[start:start + count]
        gate_up = F.linear(expert_input, gate_up_proj[expert])
        gate, up = gate_up.chunk(2, dim=-1)
        outputs.append(F.linear(F.silu(gate) * up, down_proj[expert]))
        start += count
    return torch.cat(outputs, dim=0)


def gmm_forward(
    inputs: torch.Tensor,
    counts: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    import torch_npu

    gate_up = GmmFunction.apply(inputs, counts, gate_up_weight)
    activated = torch_npu.npu_swiglu(gate_up, dim=-1)
    return GmmFunction.apply(activated, counts, down_weight)


def main() -> None:
    args = parse_args()
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise SystemExit('torch_npu is required; run this script in the Ascend container.') from exc

    if not torch.npu.is_available():
        raise SystemExit('torch.npu.is_available() is False')

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.npu.set_device(device.index or 0)
    torch.manual_seed(20260901)
    torch.npu.manual_seed_all(20260901)

    # Preserve the DeepSeek-V4 relation hidden == 2 * intermediate. The gate/up
    # matrix is deliberately non-symmetric so an omitted transpose is visible.
    experts = 2
    hidden = 64
    intermediate = 32
    token_counts = [8, 8]
    inputs = torch.randn(sum(token_counts), hidden, device=device, dtype=dtype) * 0.1
    gate_up_proj = torch.randn(experts, 2 * intermediate, hidden, device=device, dtype=dtype) * 0.02
    down_proj = torch.randn(experts, hidden, intermediate, device=device, dtype=dtype) * 0.02
    module = PackedExperts(gate_up_proj, down_proj).to(device)

    normalized_gate_up, normalized_down = _normalize_packed_expert_weights(module, dtype, hidden)
    counts = torch.tensor(token_counts, device=device, dtype=torch.int64)

    with torch.no_grad():
        expected = reference_forward(inputs, token_counts, gate_up_proj, down_proj)
        actual = gmm_forward(inputs, counts, normalized_gate_up, normalized_down)
        # Reproduce the old DeepSeek-V4 bug: the square gate/up tensor was not transposed.
        old_bug = gmm_forward(inputs, counts, gate_up_proj, down_proj.transpose(1, 2))
        torch.npu.synchronize()

    difference = (actual.float() - expected.float()).abs()
    old_difference = (old_bug.float() - expected.float()).abs()
    passed = torch.allclose(actual.float(), expected.float(), rtol=args.rtol, atol=args.atol)
    report = {
        'device': str(device),
        'dtype': str(dtype),
        'input_shape': list(inputs.shape),
        'gate_up_shape_transformers': list(gate_up_proj.shape),
        'down_shape_transformers': list(down_proj.shape),
        'gate_up_shape_gmm': list(normalized_gate_up.shape),
        'down_shape_gmm': list(normalized_down.shape),
        'rtol': args.rtol,
        'atol': args.atol,
        'max_abs_diff': difference.max().item(),
        'mean_abs_diff': difference.mean().item(),
        'old_bug_max_abs_diff': old_difference.max().item(),
        'old_bug_mean_abs_diff': old_difference.mean().item(),
        'passed': bool(passed),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print(f'Report saved to: {args.output.resolve()}')
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
