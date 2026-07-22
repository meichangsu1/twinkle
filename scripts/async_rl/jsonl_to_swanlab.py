#!/usr/bin/env python3
"""Replay native async-RL JSONL business metrics into SwanLab."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value if math.isfinite(float(value)) else None
    if isinstance(value, str):
        match = re.fullmatch(r'\s*tensor\(([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\)\s*', value)
        if match:
            return float(match.group(1))
        try:
            number = float(value)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def _safe(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', value).strip('_') or 'default'


def replay(
    input_path: Path,
    logdir: Path,
    *,
    project: str,
    experiment: str,
    smooth_span: int = 0,
) -> int:
    try:
        import swanlab
    except ImportError as exc:
        raise SystemExit("SwanLab is not installed. Install it with: pip install 'swanlab[dashboard]'") from exc

    run = swanlab.init(
        project=project,
        experiment_name=experiment,
        logdir=str(logdir),
        mode='local',
    )
    if smooth_span < 0:
        raise ValueError(f'smooth_span must be non-negative, got {smooth_span}')

    logged = 0
    reward_ema: dict[str, float] = {}
    ema_alpha = 2.0 / (smooth_span + 1) if smooth_span else 0.0
    event_steps: dict[tuple[str, str], int] = defaultdict(int)
    rows_by_step: dict[int, dict[str, float | int]] = defaultdict(dict)
    with input_path.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            event = json.loads(line)
            event_name = _safe(str(event.get('event', 'event')))
            context = event.get('context_key')
            prefix = f'context/{_safe(context)}' if context else 'global'
            values: dict[str, float | int] = {}
            for name, value in (event.get('metrics') or {}).items():
                number = _number(value)
                if number is not None:
                    values[f'{prefix}/{event_name}/{_safe(str(name))}'] = number
                    if smooth_span and event_name == 'train_step_done' and name == 'reward':
                        previous = reward_ema.get(prefix)
                        smoothed = float(number) if previous is None else ema_alpha * float(number) + (
                            1.0 - ema_alpha) * previous
                        reward_ema[prefix] = smoothed
                        values[f'{prefix}/{event_name}/reward_ema{smooth_span}'] = smoothed
            if not values:
                continue
            raw_step = (event.get('metrics') or {}).get('step')
            raw_step = raw_step if raw_step is not None else (event.get('metrics') or {}).get('optimizer_step')
            explicit_step = _number(raw_step)
            if explicit_step is None:
                counter_key = (str(context or 'global'), event_name)
                event_steps[counter_key] += 1
                step = event_steps[counter_key]
            else:
                step = int(explicit_step)
            rows_by_step[step].update(values)

    for step in sorted(rows_by_step):
        run.log(rows_by_step[step], step=step)
        logged += 1
    swanlab.finish()
    print(f'logged {logged} logical steps to {logdir}')
    return logged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('metrics_jsonl', type=Path)
    parser.add_argument('--logdir', type=Path, default=Path('outputs/swanlab_async_rl'))
    parser.add_argument('--project', default='twinkle-async-rl')
    parser.add_argument('--experiment', default=None)
    parser.add_argument(
        '--smooth-span',
        type=int,
        default=0,
        help='EMA span for train_step_done/reward; 0 disables smoothing',
    )
    args = parser.parse_args()
    experiment = args.experiment or args.metrics_jsonl.stem
    replay(
        args.metrics_jsonl,
        args.logdir,
        project=args.project,
        experiment=experiment,
        smooth_span=args.smooth_span,
    )


if __name__ == '__main__':
    main()
