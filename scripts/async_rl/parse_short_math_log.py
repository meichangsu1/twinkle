#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
import time
from pathlib import Path
from typing import Any

STEP_PATTERN = re.compile(r'\[Step\s+(\d+)/(\d+)\]\s+(\{.*\})')


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert short_math_grpo.py logs to metrics.jsonl.')
    parser.add_argument('log_file', type=Path)
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--mode', default='short_math_grpo')
    parser.add_argument('--adapter-name', default='default')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    run_id = args.run_id or args.log_file.stem
    output = args.output or Path('outputs/async_rl_experiments') / run_id / 'metrics.jsonl'
    output.parent.mkdir(parents=True, exist_ok=True)
    events = parse_short_math_log(
        args.log_file,
        run_id=run_id,
        mode=args.mode,
        adapter_name=args.adapter_name,
    )
    with output.open('w', encoding='utf-8') as file:
        for event in events:
            file.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + '\n')
    print(output)


def parse_short_math_log(
    path: Path,
    *,
    run_id: str,
    mode: str,
    adapter_name: str,
) -> list[dict[str, Any]]:
    events = [
        {
            'ts': time.time(),
            'elapsed_s': 0.0,
            'run_id': run_id,
            'mode': mode,
            'seed': None,
            'event': 'run_metadata',
            'phase': 'run',
            'adapter_name': adapter_name,
            'metrics': {
                'source': str(path),
            },
        }
    ]
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        match = STEP_PATTERN.search(line)
        if match is None:
            continue
        step = int(match.group(1))
        max_steps = int(match.group(2))
        raw_metrics = ast.literal_eval(match.group(3))
        metrics = _coerce_metrics(raw_metrics)
        metrics['step'] = step
        metrics['max_steps'] = max_steps
        events.append({
            'ts': time.time(),
            'elapsed_s': _elapsed_s(raw_metrics, fallback=float(step)),
            'run_id': run_id,
            'mode': mode,
            'seed': None,
            'event': 'train_batch_done',
            'phase': 'train',
            'adapter_name': adapter_name,
            'policy_version': step,
            'metrics': metrics,
        })
    return events


def _coerce_metrics(raw_metrics: dict[str, Any]) -> dict[str, Any]:
    metrics = {}
    for key, value in raw_metrics.items():
        metrics[str(key)] = _coerce_value(value)
    return metrics


def _coerce_value(value: Any) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        number = _coerce_number(stripped)
        if number is not None:
            return number
        if stripped.endswith(' seconds'):
            number = _coerce_number(stripped[:-8].strip())
            return number if number is not None else value
        if stripped.endswith(' minutes'):
            number = _coerce_number(stripped[:-8].strip())
            return number * 60.0 if number is not None else value
        if stripped.endswith(' iters/s'):
            number = _coerce_number(stripped[:-8].strip())
            return number if number is not None else value
        if stripped.endswith('%'):
            number = _coerce_number(stripped[:-1].strip())
            return number if number is not None else value
    return value


def _coerce_number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _elapsed_s(raw_metrics: dict[str, Any], *, fallback: float) -> float:
    value = raw_metrics.get('total time elapse')
    if value is None:
        return fallback
    converted = _coerce_value(value)
    return float(converted) if isinstance(converted, (int, float)) else fallback


if __name__ == '__main__':
    main()
