#!/usr/bin/env python3
"""Numerically compare two last-token-logit files produced by the DSV4 probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--rtol', type=float, default=1e-2)
    parser.add_argument('--atol', type=float, default=2e-2)
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def load_logits(path: Path) -> tuple[str, list[int], torch.Tensor]:
    payload = torch.load(path, map_location='cpu', weights_only=True)
    return (
        str(payload.get('mode', path.stem)),
        list(payload['input_ids']),
        payload['last_logits'].float(),
    )


def main() -> None:
    args = parse_args()
    reference_mode, reference_ids, reference = load_logits(args.reference)
    candidate_mode, candidate_ids, candidate = load_logits(args.candidate)

    if reference_ids != candidate_ids:
        raise RuntimeError(f'Input IDs differ: {reference_ids} != {candidate_ids}')
    if reference.shape != candidate.shape:
        raise RuntimeError(f'Logit shapes differ: {tuple(reference.shape)} != {tuple(candidate.shape)}')

    difference = (reference - candidate).abs()
    close = torch.isclose(reference, candidate, rtol=args.rtol, atol=args.atol)
    top_k = min(20, reference.numel())
    reference_top = torch.topk(reference, k=top_k).indices.tolist()
    candidate_top = torch.topk(candidate, k=top_k).indices.tolist()

    report = {
        'reference': str(args.reference),
        'reference_mode': reference_mode,
        'candidate': str(args.candidate),
        'candidate_mode': candidate_mode,
        'input_ids': reference_ids,
        'shape': list(reference.shape),
        'rtol': args.rtol,
        'atol': args.atol,
        'allclose': bool(close.all().item()),
        'close_fraction': float(close.float().mean().item()),
        'max_abs_diff': float(difference.max().item()),
        'mean_abs_diff': float(difference.mean().item()),
        'reference_top20': reference_top,
        'candidate_top20': candidate_top,
        'top20_overlap': len(set(reference_top) & set(candidate_top)) / top_k,
    }

    output = args.output or args.reference.with_name(f'compare_{reference_mode}_vs_{candidate_mode}.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'Report saved to: {output.resolve()}')
    if not report['allclose']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
