#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize async RL metrics.jsonl into summary.json.')
    parser.add_argument('metrics_jsonl', type=Path)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    events = _read_events(args.metrics_jsonl)
    summary = summarize_events(events)
    output = args.output or args.metrics_jsonl.with_name('summary.json')
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(output)


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    wall_time_s = max((float(event.get('elapsed_s') or 0.0) for event in events), default=0.0)
    train_events = [event for event in events if event.get('event') == 'train_batch_done']
    partition_done_events = [event for event in events if event.get('event') == 'partition_train_done']
    rollout_done_events = [event for event in events if event.get('event') == 'rollout_done']
    stale_events = [event for event in events if event.get('event') == 'stale_dropped']

    trained_samples = sum(_metric_number(event, 'sample_count') for event in train_events)
    rollout_groups = len(rollout_done_events)
    rollout_samples = sum(_metric_number(event, 'sample_count') for event in rollout_done_events)
    train_partitions = len(partition_done_events)
    train_steps = len(train_events)
    hours = wall_time_s / 3600.0 if wall_time_s > 0 else 0.0

    gap_values = _metric_values(events, 'policy_version_gap_mean')
    reward_values = _metric_values(train_events, 'reward_mean')
    accuracy_values = _metric_values(train_events, 'accuracy_reward_mean')

    by_context: dict[str, dict[str, Any]] = defaultdict(lambda: {
        'train_steps': 0,
        'rollout_groups': 0,
        'reward_mean_last': None,
        'accuracy_reward_mean_last': None,
    })
    for event in train_events:
        key = str(event.get('context_key') or event.get('adapter_name') or 'unknown')
        by_context[key]['train_steps'] += 1
        reward = _metric_optional_number(event, 'reward_mean')
        accuracy = _metric_optional_number(event, 'accuracy_reward_mean')
        if reward is not None:
            by_context[key]['reward_mean_last'] = reward
        if accuracy is not None:
            by_context[key]['accuracy_reward_mean_last'] = accuracy
    for event in rollout_done_events:
        key = str(event.get('context_key') or event.get('adapter_name') or 'unknown')
        by_context[key]['rollout_groups'] += int(_metric_number(event, 'groups') or 1)

    return {
        'wall_time_s': wall_time_s,
        'train_steps': train_steps,
        'train_partitions': train_partitions,
        'train_steps_per_hour': _rate(train_steps, hours),
        'train_partitions_per_hour': _rate(train_partitions, hours),
        'trained_samples_per_sec': _rate(trained_samples, wall_time_s),
        'rollout_groups_per_sec': _rate(rollout_groups, wall_time_s),
        'rollout_samples_per_sec': _rate(rollout_samples, wall_time_s),
        'policy_version_gap': _summary(gap_values),
        'stale_dropped_groups': len(stale_events),
        'stale_drop_ratio': len(stale_events) / max(len(stale_events) + len(rollout_done_events), 1),
        'reward_mean': _summary(reward_values),
        'accuracy_reward_mean': _summary(accuracy_values),
        'per_context': dict(sorted(by_context.items())),
        'event_counts': _event_counts(events),
    }


def _read_events(path: Path) -> list[dict[str, Any]]:
    events = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def _event_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for event in events:
        counts[str(event.get('event') or 'unknown')] += 1
    return dict(sorted(counts.items()))


def _metric_optional_number(event: dict[str, Any], key: str) -> float | None:
    value = (event.get('metrics') or {}).get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_number(event: dict[str, Any], key: str) -> float:
    value = _metric_optional_number(event, key)
    return value if value is not None else 0.0


def _metric_values(events: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for event in events:
        value = _metric_optional_number(event, key)
        if value is not None:
            values.append(value)
    return values


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {'mean': None, 'std': None, 'min': None, 'max': None, 'p95': None, 'last': None}
    ordered = sorted(values)
    count = len(ordered)
    mean = sum(ordered) / count
    variance = sum((value - mean)**2 for value in ordered) / count
    p95_index = min(count - 1, int(math.ceil(count * 0.95)) - 1)
    return {
        'mean': mean,
        'std': variance**0.5,
        'min': ordered[0],
        'max': ordered[-1],
        'p95': ordered[p95_index],
        'last': values[-1],
    }


def _rate(value: float, denominator: float) -> float:
    return value / denominator if denominator > 0 else 0.0


if __name__ == '__main__':
    main()
