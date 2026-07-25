# Copyright (c) ModelScope Contributors. All rights reserved.
"""Compare Twinkle and AReaL math-verify semantics on saved predictions."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class Score:
    correct: bool
    error: str | None = None


def twinkle_explicit_score(completion: str, ground_truth: str) -> Score:
    """Match MathVerifyAccuracyReward used by Twinkle."""
    from math_verify.grader import verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse

    extraction_config = (
        ExprExtractionConfig(try_extract_without_anchor=True),
        LatexExtractionConfig(),
    )
    try:
        gold = parse(
            ground_truth,
            extraction_config=extraction_config,
            parsing_timeout=None,
        )
        answer = parse(
            completion,
            extraction_config=extraction_config,
            parsing_timeout=None,
        )
        if not gold or not answer:
            return Score(False)
        return Score(bool(verify(gold, answer, float_rounding=6, timeout_seconds=None)))
    except Exception as exc:
        return Score(False, f'{type(exc).__name__}: {exc}')


def areal_workflow_score(completion: str, ground_truth: str) -> Score:
    """Match examples/math/gsm8k_rl.py through MathAgent.math_reward_fn."""
    from math_verify import parse, verify

    try:
        answer = parse(completion)
        gold = parse(ground_truth)
        return Score(bool(verify(answer, gold)))
    except Exception as exc:
        return Score(False, f'{type(exc).__name__}: {exc}')


def load_predictions(path: Path) -> list[dict[str, Any]]:
    predictions = []
    with path.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {'completion', 'completion_length'} - row.keys()
            if missing:
                raise ValueError(
                    f'{path}:{line_number} is missing required fields: {sorted(missing)}')
            if 'ground_truth' not in row and 'expected' not in row:
                raise ValueError(
                    f'{path}:{line_number} requires ground_truth or expected')
            predictions.append(row)
    if not predictions:
        raise ValueError(f'prediction file is empty: {path}')
    return predictions


def length_bucket(length: int) -> str:
    if length < 512:
        return '<512'
    if length < 1024:
        return '512-1023'
    return '>=1024'


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        'count': count,
        'completion_length_mean': (
            sum(int(row['completion_length']) for row in rows) / count if count else None),
        'twinkle_explicit_accuracy': (
            sum(row['twinkle_explicit_correct'] for row in rows) / count if count else None),
        'areal_workflow_accuracy': (
            sum(row['areal_workflow_correct'] for row in rows) / count if count else None),
        'reward_disagreement_count': sum(row['reward_disagreement'] for row in rows),
        'reward_disagreement_ratio': (
            sum(row['reward_disagreement'] for row in rows) / count if count else None),
    }


def score_predictions(
    predictions: list[dict[str, Any]],
    *,
    generation_token_limit: int | None,
) -> list[dict[str, Any]]:
    scored = []
    scorers: tuple[tuple[str, Callable[[str, str], Score]], ...] = (
        ('twinkle_explicit', twinkle_explicit_score),
        ('areal_workflow', areal_workflow_score),
    )
    for index, row in enumerate(predictions):
        completion = str(row['completion'])
        ground_truth = str(row.get('ground_truth', row.get('expected', '')))
        result = dict(row)
        result['prediction_index'] = index
        for name, scorer in scorers:
            score = scorer(completion, ground_truth)
            result[f'{name}_correct'] = score.correct
            result[f'{name}_error'] = score.error
        result['reward_disagreement'] = (
            result['twinkle_explicit_correct'] != result['areal_workflow_correct'])
        result['length_bucket'] = length_bucket(int(row['completion_length']))
        result['at_generation_limit'] = (
            generation_token_limit is not None
            and int(row['completion_length']) >= generation_token_limit
        )
        scored.append(result)
    return scored


def build_summary(
    predictions_path: Path,
    scored: list[dict[str, Any]],
    *,
    generation_token_limit: int | None,
) -> dict[str, Any]:
    disagreements = [row for row in scored if row['reward_disagreement']]
    twinkle_only = [
        row for row in disagreements
        if row['twinkle_explicit_correct'] and not row['areal_workflow_correct']
    ]
    areal_only = [
        row for row in disagreements
        if row['areal_workflow_correct'] and not row['twinkle_explicit_correct']
    ]
    at_limit = [row for row in scored if row['at_generation_limit']]
    buckets = {
        name: summarize_rows([row for row in scored if row['length_bucket'] == name])
        for name in ('<512', '512-1023', '>=1024')
    }
    summary = {
        'predictions_path': str(predictions_path),
        'generation_token_limit': generation_token_limit,
        **summarize_rows(scored),
        'twinkle_only_correct_count': len(twinkle_only),
        'areal_only_correct_count': len(areal_only),
        'twinkle_error_count': sum(row['twinkle_explicit_error'] is not None for row in scored),
        'areal_error_count': sum(row['areal_workflow_error'] is not None for row in scored),
        'length_buckets': buckets,
        'at_generation_limit': summarize_rows(at_limit),
    }
    stored = [row for row in scored if 'correct' in row]
    if stored:
        summary['stored_accuracy'] = (
            sum(float(row['correct']) for row in stored) / len(stored))
        summary['stored_vs_twinkle_disagreement_count'] = sum(
            bool(row['correct']) != row['twinkle_explicit_correct']
            for row in stored
        )
    if at_limit:
        summary['at_generation_limit']['ratio'] = len(at_limit) / len(scored)
        summary['at_generation_limit']['twinkle_only_correct_ratio'] = (
            sum(
                row['twinkle_explicit_correct'] and not row['areal_workflow_correct']
                for row in at_limit
            ) / len(at_limit)
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions_path', type=Path)
    parser.add_argument(
        '--generation-token-limit',
        type=int,
        default=None,
        help='Marks samples at or above this completion length; omitted means unknown.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Defaults to <predictions directory>/reward_comparison.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = load_predictions(args.predictions_path)
    scored = score_predictions(
        predictions,
        generation_token_limit=args.generation_token_limit,
    )
    output_dir = args.output_dir or args.predictions_path.parent / 'reward_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)

    disagreements_path = output_dir / 'reward_disagreements.jsonl'
    with disagreements_path.open('w', encoding='utf-8') as stream:
        for row in scored:
            if row['reward_disagreement']:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')

    summary = build_summary(
        args.predictions_path,
        scored,
        generation_token_limit=args.generation_token_limit,
    )
    summary['reward_disagreements_path'] = str(disagreements_path)
    summary_path = output_dir / 'summary.json'
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
