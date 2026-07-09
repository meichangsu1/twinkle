#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description='Replay async RL metrics.jsonl files into an offline SwanLab logdir.')
    parser.add_argument(
        'runs',
        nargs='+',
        type=Path,
        help='Run directories or metrics.jsonl files.',
    )
    parser.add_argument('--project', default='twinkle-async-rl', help='SwanLab project name.')
    parser.add_argument(
        '--logdir',
        type=Path,
        default=Path('outputs/async_rl_experiments/swanlab'),
        help='Local SwanLab log directory.',
    )
    parser.add_argument('--mode', default='local', help='SwanLab mode. Use local for offline dashboards.')
    parser.add_argument('--name-prefix', default='', help='Optional prefix for SwanLab experiment names.')
    parser.add_argument('--dry-run', action='store_true', help='Parse and flatten events without importing SwanLab.')
    args = parser.parse_args()

    runs = [_load_run(path) for path in args.runs]
    if args.dry_run:
        for run in runs:
            count = sum(1 for _ in _iter_flattened_events(run['events']))
            print(f'{run["run_id"]}: {count} scalar event payloads')
        return

    try:
        import swanlab
    except ImportError as exc:
        raise SystemExit('swanlab is required. Install it in the local visualization environment.') from exc

    args.logdir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        experiment_name = f'{args.name_prefix}{run["run_id"]}'
        swanlab.init(
            project=args.project,
            experiment_name=experiment_name,
            mode=args.mode,
            logdir=str(args.logdir),
            config=_run_config(run),
        )
        logged = 0
        for step, flat in enumerate(_iter_flattened_events(run['events']), start=1):
            swanlab.log(flat, step=step)
            logged += 1
        if hasattr(swanlab, 'finish'):
            swanlab.finish()
        print(f'{experiment_name}: replayed {logged} events')
    if args.mode == 'local':
        print(f'Open the offline dashboard with: swanlab watch -l {args.logdir}')


def _load_run(path: Path) -> dict[str, Any]:
    metrics_path = path / 'metrics.jsonl' if path.is_dir() else path
    events = _read_events(metrics_path)
    run_id = _first_value(events, 'run_id') or metrics_path.parent.name
    mode = _first_value(events, 'mode') or run_id
    return {
        'run_id': _safe_name(str(run_id)),
        'mode': str(mode),
        'path': metrics_path,
        'events': events,
    }


def _read_events(path: Path) -> list[dict[str, Any]]:
    events = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def _first_value(events: list[dict[str, Any]], key: str) -> Any:
    for event in events:
        value = event.get(key)
        if value is not None:
            return value
    return None


def _run_config(run: dict[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {
        'run_id': run['run_id'],
        'mode': run['mode'],
        'source_metrics_jsonl': str(run['path']),
    }
    for event in run['events']:
        if event.get('event') == 'run_metadata':
            config.update(event.get('metrics') or {})
            break
    seed = _first_value(run['events'], 'seed')
    if seed is not None:
        config['seed'] = seed
    return _jsonable(config)


def _iter_flattened_events(events: list[dict[str, Any]]):
    event_counts: dict[str, int] = defaultdict(int)
    train_steps: dict[str, int] = defaultdict(int)
    rollout_done: dict[str, int] = defaultdict(int)
    partition_done: dict[str, int] = defaultdict(int)
    for event in events:
        name = str(event.get('event') or 'unknown')
        context = str(event.get('adapter_name') or event.get('context_key') or 'global')
        event_counts[name] += 1
        if name == 'train_batch_done':
            train_steps[context] += 1
        elif name == 'rollout_done':
            rollout_done[context] += 1
        elif name == 'partition_train_done':
            partition_done[context] += 1

        flat = _flatten_event(event)
        flat[f'events/{name}'] = event_counts[name]
        flat['global/event_count'] = sum(event_counts.values())
        if name == 'train_batch_done':
            flat['global/train_steps'] = sum(train_steps.values())
            flat[f'context/{context}/train_steps'] = train_steps[context]
            optimizer_step = _scalar((event.get('metrics') or {}).get('optimizer_step'))
            if optimizer_step is not None:
                flat['global/optimizer_step'] = optimizer_step
        elif name == 'rollout_done':
            flat['global/rollout_groups'] = sum(rollout_done.values())
            flat[f'context/{context}/rollout_groups'] = rollout_done[context]
        elif name == 'partition_train_done':
            flat['global/train_partitions'] = sum(partition_done.values())
            flat[f'context/{context}/train_partitions'] = partition_done[context]
        if flat:
            yield flat


def _flatten_event(event: dict[str, Any]) -> dict[str, int | float]:
    adapter_name = event.get('adapter_name')
    phase = str(event.get('phase') or 'global')
    flat: dict[str, int | float] = {}
    elapsed_s = _scalar(event.get('elapsed_s'))
    if elapsed_s is not None:
        flat['global/wall_time_s'] = elapsed_s
    policy_version = _scalar(event.get('policy_version'))
    if policy_version is not None:
        key = f'context/{adapter_name}/policy_version' if adapter_name else 'global/policy_version'
        flat[key] = policy_version
    for key, value in (event.get('metrics') or {}).items():
        scalar = _scalar(value)
        if scalar is None:
            continue
        flat[_metric_key(phase, adapter_name, str(key))] = scalar
    return flat


def _metric_key(phase: str, adapter_name: str | None, key: str) -> str:
    backlog_keys = {
        'pending_prompt_groups',
        'inflight_rollout_groups',
        'active_partitions',
        'closed_partitions',
        'rollout_done_groups',
        'advantaging_groups',
        'advantage_done_groups',
        'training_groups',
        'untrained_groups',
    }
    if key in backlog_keys:
        return f'backlog/{key}'
    if key.startswith('policy_version_gap') or key.startswith('stale_'):
        return f'staleness/{key}'
    if phase == 'tq':
        return f'tq/{key}'
    if adapter_name:
        return f'context/{adapter_name}/{key}'
    return f'global/{key}'


def _scalar(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        number = float(value)
        return value if math.isfinite(number) else None
    if isinstance(value, str):
        text = value.strip()
        if not _NUMERIC_RE.fullmatch(text):
            return None
        number = float(text)
        return number if math.isfinite(number) else None
    return None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _safe_name(value: str) -> str:
    return ''.join(char if char.isalnum() or char in '._-' else '_' for char in value)


_NUMERIC_RE = re.compile(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?')

if __name__ == '__main__':
    main()
