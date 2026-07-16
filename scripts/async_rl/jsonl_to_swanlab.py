#!/usr/bin/env python3
"""Replay native async-RL JSONL business metrics into SwanLab."""

from __future__ import annotations

import argparse
import json
import math
import re
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


def replay(input_path: Path, logdir: Path, *, project: str, experiment: str) -> int:
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
    logged = 0
    with input_path.open(encoding='utf-8') as stream:
        for event_index, line in enumerate(stream, start=1):
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
            if not values:
                continue
            raw_step = (event.get('metrics') or {}).get('step')
            raw_step = raw_step if raw_step is not None else (event.get('metrics') or {}).get('optimizer_step')
            step = int(_number(raw_step) or event_index)
            run.log(values, step=step)
            logged += 1
    swanlab.finish()
    print(f'logged {logged} events to {logdir}')
    return logged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('metrics_jsonl', type=Path)
    parser.add_argument('--logdir', type=Path, default=Path('outputs/swanlab_async_rl'))
    parser.add_argument('--project', default='twinkle-async-rl')
    parser.add_argument('--experiment', default=None)
    args = parser.parse_args()
    experiment = args.experiment or args.metrics_jsonl.stem
    replay(args.metrics_jsonl, args.logdir, project=args.project, experiment=experiment)


if __name__ == '__main__':
    main()
