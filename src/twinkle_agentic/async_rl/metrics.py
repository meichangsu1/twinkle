# Copyright (c) ModelScope Contributors. All rights reserved.
"""Small JSONL metrics sink for native async RL business events."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from .types import LoraContext, MetricEvent


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)]


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


class JSONLMetricsRecorder:
    """Write one business event per JSONL record.

    The recorder is intentionally synchronous and tiny.  It is suitable for a
    CPU driver or a dedicated metrics actor; TQ operation-level calls are never
    sent here.
    """

    def __init__(self, path: str | Path, *, run_id: str, mode: str = 'async'):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.mode = mode
        self.started_at = time.time()
        self._lock = threading.Lock()

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
        with self._lock:
            with self.path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(payload, ensure_ascii=True, default=str) + '\n')


class CompositeMetricsRecorder:
    """Fan out the same event to a small set of recorders."""

    def __init__(self, *recorders: Any):
        self.recorders = tuple(recorders)

    def record(self, **event: Any) -> None:
        for recorder in self.recorders:
            recorder.record(**event)
