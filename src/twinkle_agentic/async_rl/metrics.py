# Copyright (c) ModelScope Contributors. All rights reserved.
"""Small JSONL metrics sink for native async RL business events."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import Counter, defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import LoraContext, MetricEvent


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)]


def rollout_performance_metrics(
    rows: Sequence[dict[str, Any]],
    *,
    rollout_latency_s: float | None = None,
) -> dict[str, float | int]:
    if not rows:
        raise ValueError('rollout metrics require at least one sample')
    completion_lengths = [int(row['completion_length']) for row in rows]
    output_tokens = sum(completion_lengths)
    truncated_samples = sum(row.get('stop_reason') == 'length' for row in rows)
    metrics: dict[str, float | int] = {
        'sample_count': len(rows),
        'completion_length_mean': output_tokens / len(rows),
        'completion_length_p95': _p95(completion_lengths),
        'completion_length_max': max(completion_lengths),
        'output_tokens': output_tokens,
        'completion_truncated_samples': truncated_samples,
        'completion_truncated_ratio': truncated_samples / len(rows),
    }
    if rollout_latency_s is not None:
        metrics['rollout_latency_s'] = rollout_latency_s
        metrics['output_tokens_per_s'] = output_tokens / rollout_latency_s
    return metrics


def training_policy_metrics(
    sample_tags: tuple[dict[str, Any], ...],
    train_policy_version: int,
) -> dict[str, float | int]:
    if not sample_tags:
        raise ValueError('training batch must contain sample policy tags')
    final_versions = [int(tag['final_policy_version']) for tag in sample_tags]
    spans = [int(tag['policy_version_span']) for tag in sample_tags]
    gaps = [int(train_policy_version) - version for version in final_versions]
    if any(gap < 0 for gap in gaps):
        raise ValueError(
            f'training policy version {train_policy_version} is older than rollout versions {final_versions}')
    return {
        'policy_version_gap_mean': sum(gaps) / len(gaps),
        'policy_version_gap_p95': _p95(gaps),
        'policy_version_gap_max': max(gaps),
        'rollout_policy_span_mean': sum(spans) / len(spans),
        'rollout_policy_span_max': max(spans),
    }


def advantage_signal_metrics(
    rewards: Sequence[float],
    advantages: Sequence[float],
    *,
    num_generations: int,
    zero_tolerance: float = 1e-8,
) -> dict[str, float | int]:
    """Summarize whether GRPO groups provide a useful learning signal."""
    if num_generations <= 0:
        raise ValueError(f'num_generations must be positive, got {num_generations}')
    if len(rewards) != len(advantages):
        raise ValueError(f'rewards and advantages must have equal length: {len(rewards)} != {len(advantages)}')
    if len(rewards) == 0 or len(rewards) % num_generations:
        raise ValueError(
            f'advantage metrics require complete groups: sample_count={len(rewards)}, '
            f'num_generations={num_generations}')

    reward_values = [float(value) for value in rewards]
    advantage_values = [float(value) for value in advantages]
    group_reward_stds: list[float] = []
    zero_advantage_groups = 0
    for start in range(0, len(reward_values), num_generations):
        group_rewards = reward_values[start:start + num_generations]
        group_advantages = advantage_values[start:start + num_generations]
        reward_mean = sum(group_rewards) / num_generations
        group_reward_stds.append(
            math.sqrt(sum((value - reward_mean)**2 for value in group_rewards) / num_generations))
        if max(abs(value) for value in group_advantages) <= zero_tolerance:
            zero_advantage_groups += 1

    advantage_mean = sum(advantage_values) / len(advantage_values)
    group_count = len(group_reward_stds)
    return {
        'group_count': group_count,
        'group_reward_std_mean': sum(group_reward_stds) / group_count,
        'zero_advantage_group_ratio': zero_advantage_groups / group_count,
        'positive_advantage_ratio': sum(value > zero_tolerance for value in advantage_values) / len(advantage_values),
        'advantage_mean': advantage_mean,
        'advantage_std': math.sqrt(
            sum((value - advantage_mean)**2 for value in advantage_values) / len(advantage_values)),
    }


class MetricsBuffer:
    """In-process business-event buffer owned by one worker actor."""

    def __init__(self):
        self._events: deque[MetricEvent] = deque()

    def record(self,
               event: str,
               context: LoraContext | None,
               partition_id: str | None,
               metrics: dict[str, Any],
               policy_version: int | None = None) -> None:
        self._events.append(MetricEvent(event, context, partition_id, dict(metrics), policy_version))

    def drain(self) -> list[MetricEvent]:
        events = list(self._events)
        self._events.clear()
        return events


@dataclass
class _ScalarSummary:
    count: int = 0
    total: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    last: float = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        self.last = value

    def as_dict(self) -> dict[str, float | int]:
        return {
            'count': self.count,
            'last': self.last,
            'mean': self.total / self.count,
            'min': self.minimum,
            'max': self.maximum,
        }


def _numeric_metric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


class JSONLMetricsRecorder:
    """Buffer business events and write a compact summary when the run closes."""

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str,
        mode: str = 'async',
        flush_every: int = 64,
        flush_interval_s: float = 2.0,
        summary_path: str | Path | None = None,
    ):
        if flush_every <= 0:
            raise ValueError(f'flush_every must be positive, got {flush_every}')
        if flush_interval_s <= 0:
            raise ValueError(f'flush_interval_s must be positive, got {flush_interval_s}')
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path = Path(summary_path) if summary_path is not None else self.path.with_name('summary.json')
        self.run_id = run_id
        self.mode = mode
        self.started_at = time.time()
        self.flush_every = flush_every
        self.flush_interval_s = flush_interval_s
        self._lock = threading.Lock()
        self._stream = self.path.open('a', encoding='utf-8')
        self._pending_lines: list[str] = []
        self._last_flush = time.monotonic()
        self._closed = False
        self._event_count = 0
        self._flush_count = 0
        self._write_latency_s = 0.0
        self._event_counts: Counter[str] = Counter()
        self._context_event_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self._context_rollout_samples: Counter[str] = Counter()
        self._context_trained_samples: Counter[str] = Counter()
        self._context_policy_versions: dict[str, int] = {}
        self._metric_summaries: dict[str, _ScalarSummary] = defaultdict(_ScalarSummary)
        self._terminal_event: str | None = None
        self._terminal_metrics: dict[str, Any] = {}

    def record(
        self,
        *,
        event: str,
        context: Any = None,
        partition_id: str | None = None,
        metrics: dict[str, Any] | None = None,
        phase: str | None = None,
        policy_version: int | None = None,
    ) -> None:
        context_key = getattr(context, 'key', None) if context is not None else None
        payload = {
            'ts': time.time(),
            'elapsed_s': time.time() - self.started_at,
            'run_id': self.run_id,
            'mode': self.mode,
            'event': event,
            'phase': phase or event.split('_', 1)[0],
            'context_key': context_key,
            'partition_id': partition_id,
            'policy_version': policy_version,
            'metrics': dict(metrics or {}),
        }
        line = json.dumps(payload, ensure_ascii=True, default=str) + '\n'
        with self._lock:
            if self._closed:
                raise RuntimeError('cannot record metrics after recorder is closed')
            self._pending_lines.append(line)
            self._event_count += 1
            self._update_summary(event, context_key, policy_version, payload['metrics'])
            if (len(self._pending_lines) >= self.flush_every
                    or time.monotonic() - self._last_flush >= self.flush_interval_s):
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            if not self._closed:
                self._flush_locked()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._flush_locked()
            self._stream.close()
            self._write_summary_locked()
            self._closed = True

    def stats(self) -> dict[str, float | int]:
        with self._lock:
            return {
                'metrics_event_count': self._event_count,
                'metrics_flush_count': self._flush_count,
                'metrics_write_latency_s': self._write_latency_s,
            }

    def _flush_locked(self) -> None:
        if not self._pending_lines:
            return
        started = time.perf_counter()
        self._stream.writelines(self._pending_lines)
        self._pending_lines.clear()
        self._stream.flush()
        self._write_latency_s += time.perf_counter() - started
        self._flush_count += 1
        self._last_flush = time.monotonic()

    def _update_summary(
        self,
        event: str,
        context_key: str | None,
        policy_version: int | None,
        metrics: dict[str, Any],
    ) -> None:
        self._event_counts[event] += 1
        if context_key is not None:
            self._context_event_counts[context_key][event] += 1
            if policy_version is not None:
                self._context_policy_versions[context_key] = policy_version
            sample_count = _numeric_metric(metrics.get('sample_count'))
            if sample_count is not None:
                if event == 'rollout_done':
                    self._context_rollout_samples[context_key] += int(sample_count)
                elif event == 'train_step_done':
                    self._context_trained_samples[context_key] += int(sample_count)
        if event in {
                'rollout_done', 'rollout_partition_done', 'advantage_done', 'train_step_done', 'partition_done',
                'policy_published'
        }:
            for name, value in metrics.items():
                number = _numeric_metric(value)
                if number is not None:
                    self._metric_summaries[f'{event}/{name}'].add(number)
        if event in {'run_completed', 'run_failed'}:
            self._terminal_event = event
            self._terminal_metrics = dict(metrics)

    def _summary(self) -> dict[str, Any]:
        wall_time_s = _numeric_metric(self._terminal_metrics.get('wall_time_s'))
        if wall_time_s is None:
            wall_time_s = time.time() - self.started_at
        rollout_groups = self._event_counts['rollout_done']
        train_steps = self._event_counts['train_step_done']
        trained_partitions = self._event_counts['partition_done']
        terminal_partitions = _numeric_metric(self._terminal_metrics.get('trained_partitions'))
        if terminal_partitions is not None:
            trained_partitions = int(terminal_partitions)
        rollout_samples = sum(self._context_rollout_samples.values())
        trained_samples = sum(self._context_trained_samples.values())
        per_context = {}
        for context_key in self._context_event_counts:
            counts = self._context_event_counts[context_key]
            per_context[context_key] = {
                'rollout_groups': counts['rollout_done'],
                'rollout_samples': self._context_rollout_samples[context_key],
                'train_steps': counts['train_step_done'],
                'trained_samples': self._context_trained_samples[context_key],
                'trained_partitions': counts['partition_done'],
                'policy_version': self._context_policy_versions.get(context_key),
            }
        return {
            'run_id': self.run_id,
            'mode': self.mode,
            'status': self._terminal_event or 'stopped',
            'wall_time_s': wall_time_s,
            'event_counts': dict(sorted(self._event_counts.items())),
            'rollout_groups': rollout_groups,
            'rollout_samples': rollout_samples,
            'train_steps': train_steps,
            'trained_samples': trained_samples,
            'trained_partitions': trained_partitions,
            'rollout_groups_per_sec': rollout_groups / wall_time_s if wall_time_s > 0 else 0.0,
            'rollout_samples_per_sec': rollout_samples / wall_time_s if wall_time_s > 0 else 0.0,
            'train_steps_per_hour': train_steps * 3600 / wall_time_s if wall_time_s > 0 else 0.0,
            'trained_samples_per_sec': trained_samples / wall_time_s if wall_time_s > 0 else 0.0,
            'train_partitions_per_hour': trained_partitions * 3600 / wall_time_s if wall_time_s > 0 else 0.0,
            'per_context': per_context,
            'metrics': {
                name: summary.as_dict()
                for name, summary in sorted(self._metric_summaries.items())
            },
            'result': self._terminal_metrics,
        }

    def _write_summary_locked(self) -> None:
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.summary_path.with_suffix(f'{self.summary_path.suffix}.tmp')
        with temporary_path.open('w', encoding='utf-8') as stream:
            json.dump(self._summary(), stream, ensure_ascii=True, indent=2, default=str)
            stream.write('\n')
        os.replace(temporary_path, self.summary_path)


class CompositeMetricsRecorder:
    """Fan out the same event to a small set of recorders."""

    def __init__(self, *recorders: Any):
        self.recorders = tuple(recorders)

    def record(self, **event: Any) -> None:
        for recorder in self.recorders:
            recorder.record(**event)

    def flush(self) -> None:
        for recorder in self.recorders:
            flush = getattr(recorder, 'flush', None)
            if flush is not None:
                flush()

    def close(self) -> None:
        for recorder in self.recorders:
            close = getattr(recorder, 'close', None)
            if close is not None:
                close()
