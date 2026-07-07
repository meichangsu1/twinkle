# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _default_run_id() -> str:
    return time.strftime('async_rl_%Y%m%d_%H%M%S')


@dataclass
class AsyncRLMetricsConfig:
    run_id: str = field(default_factory=_default_run_id)
    mode: str = 'async'
    seed: int | None = None
    output_dir: str = 'outputs/async_rl_experiments'
    enable_jsonl: bool = True
    enable_swanlab: bool = False
    swanlab_project: str = 'twinkle'
    metadata: dict[str, Any] = field(default_factory=dict)


class AsyncRLMetricsRecorder:
    """Event recorder interface for async RL experiments."""

    def log_event(
        self,
        *,
        event: str,
        phase: str,
        context: Any | None = None,
        partition_id: str | None = None,
        policy_version: int | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        return None

    def close(self) -> None:
        return None


class NoopMetricsRecorder(AsyncRLMetricsRecorder):
    pass


class JSONLMetricsRecorder(AsyncRLMetricsRecorder):

    def __init__(self, config: AsyncRLMetricsConfig):
        self.config = config
        self.start_time = time.time()
        self.run_dir = Path(config.output_dir) / _safe_path_segment(config.run_id)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / 'metrics.jsonl'
        self._file = self.path.open('a', encoding='utf-8')
        if config.metadata:
            self.log_event(event='run_metadata', phase='run', metrics=config.metadata)

    def log_event(
        self,
        *,
        event: str,
        phase: str,
        context: Any | None = None,
        partition_id: str | None = None,
        policy_version: int | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        payload = event_payload(
            config=self.config,
            start_time=self.start_time,
            event=event,
            phase=phase,
            context=context,
            partition_id=partition_id,
            policy_version=policy_version,
            metrics=metrics,
        )
        self._file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + '\n')
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class SwanLabMetricsRecorder(AsyncRLMetricsRecorder):

    def __init__(self, config: AsyncRLMetricsConfig):
        self.config = config
        self.start_time = time.time()
        self.step = 0
        try:
            import swanlab
        except ImportError:
            self._swanlab = None
            return
        self._swanlab = swanlab
        self._swanlab.init(project=config.swanlab_project, config=_jsonable(config.metadata))

    def log_event(
        self,
        *,
        event: str,
        phase: str,
        context: Any | None = None,
        partition_id: str | None = None,
        policy_version: int | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if self._swanlab is None:
            return
        payload = event_payload(
            config=self.config,
            start_time=self.start_time,
            event=event,
            phase=phase,
            context=context,
            partition_id=partition_id,
            policy_version=policy_version,
            metrics=metrics,
        )
        self.step += 1
        self._swanlab.log(flatten_for_swanlab(payload), step=self.step)


class CompositeMetricsRecorder(AsyncRLMetricsRecorder):

    def __init__(self, recorders: list[AsyncRLMetricsRecorder]):
        self.recorders = recorders

    def log_event(
        self,
        *,
        event: str,
        phase: str,
        context: Any | None = None,
        partition_id: str | None = None,
        policy_version: int | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        for recorder in self.recorders:
            recorder.log_event(
                event=event,
                phase=phase,
                context=context,
                partition_id=partition_id,
                policy_version=policy_version,
                metrics=metrics,
            )

    def close(self) -> None:
        for recorder in self.recorders:
            recorder.close()


def build_metrics_recorder(config: AsyncRLMetricsConfig | None) -> AsyncRLMetricsRecorder:
    if config is None:
        return NoopMetricsRecorder()
    recorders: list[AsyncRLMetricsRecorder] = []
    if config.enable_jsonl:
        recorders.append(JSONLMetricsRecorder(config))
    if config.enable_swanlab:
        recorders.append(SwanLabMetricsRecorder(config))
    if not recorders:
        return NoopMetricsRecorder()
    if len(recorders) == 1:
        return recorders[0]
    return CompositeMetricsRecorder(recorders)


def event_payload(
    *,
    config: AsyncRLMetricsConfig,
    start_time: float,
    event: str,
    phase: str,
    context: Any | None,
    partition_id: str | None,
    policy_version: int | None,
    metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    now = time.time()
    payload = {
        'ts': now,
        'elapsed_s': now - start_time,
        'run_id': config.run_id,
        'mode': config.mode,
        'seed': config.seed,
        'event': event,
        'phase': phase,
        'partition_id': partition_id,
        'policy_version': policy_version,
        'metrics': _jsonable(metrics or {}),
    }
    if context is not None:
        payload.update({
            'context_key': getattr(context, 'key', None),
            'adapter_name': getattr(context, 'adapter_name', None),
            'tenant_id': getattr(context, 'tenant_id', None),
            'training_run_id': getattr(context, 'training_run_id', None),
        })
    return _jsonable(payload)


def flatten_for_swanlab(payload: dict[str, Any]) -> dict[str, Any]:
    adapter_name = payload.get('adapter_name')
    phase = str(payload.get('phase') or 'global')
    flat: dict[str, Any] = {
        'global/wall_time_s': payload.get('elapsed_s'),
    }
    policy_version = payload.get('policy_version')
    if policy_version is not None:
        key = f'context/{adapter_name}/policy_version' if adapter_name else 'global/policy_version'
        flat[key] = policy_version
    for key, value in (payload.get('metrics') or {}).items():
        scalar = _swanlab_scalar(value)
        if scalar is not None:
            flat[_swanlab_metric_key(phase, adapter_name, key)] = scalar
    return flat


def summarize_numbers(values: list[float] | tuple[float, ...]) -> dict[str, float]:
    if not values:
        return {}
    sorted_values = sorted(float(value) for value in values)
    count = len(sorted_values)
    mean = sum(sorted_values) / count
    variance = sum((value - mean)**2 for value in sorted_values) / count
    p95_index = min(count - 1, int(math.ceil(count * 0.95)) - 1)
    return {
        'mean': mean,
        'std': variance**0.5,
        'min': sorted_values[0],
        'max': sorted_values[-1],
        'p95': sorted_values[p95_index],
    }


def prefixed_summary(prefix: str, values: list[float] | tuple[float, ...]) -> dict[str, float]:
    return {f'{prefix}_{key}': value for key, value in summarize_numbers(values).items()}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, 'model_dump'):
        return _jsonable(value.model_dump())
    if hasattr(value, 'tolist'):
        return _jsonable(value.tolist())
    return str(value)


def _swanlab_metric_key(phase: str, adapter_name: str | None, key: str) -> str:
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


def _swanlab_scalar(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return value
    return None


def _safe_path_segment(value: str) -> str:
    return ''.join(char if char.isalnum() or char in '._-' else '_' for char in value)
