#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate async RL experiment plots from metrics.jsonl files.')
    parser.add_argument(
        'runs',
        nargs='+',
        type=Path,
        help='Run directories or metrics.jsonl files.',
    )
    parser.add_argument('--output', type=Path, default=None, help='Output plot directory.')
    parser.add_argument(
        '--gpu-csv',
        action='append',
        type=Path,
        default=[],
        help='Optional gpu_util.csv file. May be passed multiple times.',
    )
    args = parser.parse_args()

    plt = _import_matplotlib()
    run_metrics = [_load_run(path) for path in args.runs]
    output_dir = args.output or _default_output_dir(args.runs)
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric_key, file_prefix in (
        ('train/total_reward', 'total_reward'),
        ('train/accuracy_reward', 'accuracy_reward'),
        ('train/brevity_reward', 'brevity_reward'),
        ('train/completion_length', 'completion_length'),
        ('loss', 'loss'),
        ('grad_norm', 'grad_norm'),
        ('tq_reward_mean', 'tq_reward_mean'),
        ('reward_mean', 'reward_mean'),
        ('accuracy_reward_mean', 'accuracy_reward_mean'),
    ):
        _plot_metric_over_wall_time(plt, run_metrics, metric_key, output_dir / f'{file_prefix}_vs_wall_time.png')
        _plot_metric_over_step(plt, run_metrics, metric_key, output_dir / f'{file_prefix}_vs_train_step.png')
    _plot_train_partitions_per_hour(plt, run_metrics, output_dir / 'train_partitions_per_hour.png')
    _plot_overlap_timeline(plt, run_metrics, output_dir / 'rollout_train_overlap_timeline.png')
    _plot_policy_gap_histogram(plt, run_metrics, output_dir / 'policy_version_gap_histogram.png')
    _plot_backlog(plt, run_metrics, output_dir / 'backlog_over_time.png')
    _plot_context_fairness(plt, run_metrics, output_dir / 'per_context_train_steps.png')
    if args.gpu_csv:
        _plot_gpu_utilization(plt, args.gpu_csv, output_dir / 'gpu_utilization_timeline.png')

    print(output_dir)


def _import_matplotlib():
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit('matplotlib is required to generate plots. Install matplotlib in the experiment env.') from exc
    return plt


def _load_run(path: Path) -> dict[str, Any]:
    metrics_path = path / 'metrics.jsonl' if path.is_dir() else path
    events = _read_events(metrics_path)
    run_id = _first_value(events, 'run_id') or metrics_path.parent.name
    mode = _first_value(events, 'mode') or run_id
    return {
        'run_id': str(run_id),
        'mode': str(mode),
        'label': str(run_id),
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


def _default_output_dir(paths: list[Path]) -> Path:
    if len(paths) == 1:
        base = paths[0] if paths[0].is_dir() else paths[0].parent
        return base / 'plots'
    common = paths[0] if paths[0].is_dir() else paths[0].parent
    return common.parent / 'plots'


def _metric(event: dict[str, Any], key: str) -> float | None:
    value = (event.get('metrics') or {}).get(key)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _train_points(run: dict[str, Any], metric_key: str) -> tuple[list[float], list[int], list[float]]:
    wall_time = []
    steps = []
    values = []
    step = 0
    for event in run['events']:
        if event.get('event') != 'train_batch_done':
            continue
        value = _metric(event, metric_key)
        if value is None:
            continue
        step += 1
        logged_step = _metric(event, 'optimizer_step')
        if logged_step is None:
            logged_step = _metric(event, 'step')
        if logged_step is None:
            logged_step = _metric(event, 'train_step')
        wall_time.append(float(event.get('elapsed_s') or 0.0))
        steps.append(int(logged_step) if logged_step is not None else step)
        values.append(value)
    return wall_time, steps, values


def _plot_metric_over_wall_time(plt, runs: list[dict[str, Any]], metric_key: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False
    for run in runs:
        wall_time, _, values = _train_points(run, metric_key)
        if not values:
            continue
        ax.plot(wall_time, values, marker='o', linewidth=1.5, label=run['label'])
        plotted = True
    _finish_line_plot(
        fig,
        ax,
        output,
        title=f'{metric_key} vs wall-clock',
        xlabel='wall-clock time (s)',
        ylabel=metric_key,
        plotted=plotted,
    )


def _plot_metric_over_step(plt, runs: list[dict[str, Any]], metric_key: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False
    for run in runs:
        _, steps, values = _train_points(run, metric_key)
        if not values:
            continue
        ax.plot(steps, values, marker='o', linewidth=1.5, label=run['label'])
        plotted = True
    _finish_line_plot(
        fig,
        ax,
        output,
        title=f'{metric_key} vs train step',
        xlabel='train step',
        ylabel=metric_key,
        plotted=plotted,
    )


def _plot_train_partitions_per_hour(plt, runs: list[dict[str, Any]], output: Path) -> None:
    labels = []
    values = []
    for run in runs:
        wall_time = max((float(event.get('elapsed_s') or 0.0) for event in run['events']), default=0.0)
        partitions = sum(1 for event in run['events'] if event.get('event') == 'partition_train_done')
        labels.append(run['label'])
        values.append(partitions / (wall_time / 3600.0) if wall_time > 0 else 0.0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color='#4C78A8')
    ax.set_title('train partitions per hour')
    ax.set_ylabel('partitions/hour')
    ax.tick_params(axis='x', rotation=20)
    _save(fig, output)


def _plot_overlap_timeline(plt, runs: list[dict[str, Any]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, max(3, 1.2 * len(runs))))
    plotted = False
    ytick_positions = []
    ytick_labels = []
    row = 0
    for run in runs:
        intervals = _phase_intervals(run['events'])
        rollout = intervals.get('rollout', [])
        train = intervals.get('train', [])
        if rollout:
            ax.broken_barh(rollout, (row, 0.35), facecolors='#59A14F', label='rollout' if not plotted else None)
            plotted = True
        if train:
            ax.broken_barh(train, (row + 0.4, 0.35), facecolors='#E15759', label='train' if row == 0 else None)
            plotted = True
        ytick_positions.append(row + 0.35)
        ytick_labels.append(run['label'])
        row += 1
    ax.set_title('rollout/train overlap timeline')
    ax.set_xlabel('wall-clock time (s)')
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels)
    if plotted:
        ax.legend(loc='best')
    else:
        _write_no_data(ax)
    _save(fig, output)


def _phase_intervals(events: list[dict[str, Any]]) -> dict[str, list[tuple[float, float]]]:
    intervals: dict[str, list[tuple[float, float]]] = {'rollout': [], 'train': []}
    rollout_starts: dict[tuple[str, str], float] = {}
    train_starts: dict[tuple[str, str], float] = {}
    for event in events:
        phase = event.get('phase')
        name = event.get('event')
        key = (str(event.get('context_key') or event.get('adapter_name') or ''), str(event.get('partition_id') or ''))
        ts = float(event.get('elapsed_s') or 0.0)
        if name == 'rollout_started':
            group_id = str((event.get('metrics') or {}).get('group_id') or len(rollout_starts))
            rollout_starts[(key[0], key[1], group_id)] = ts
        elif name == 'rollout_done':
            starts = [item for item in rollout_starts if item[0] == key[0] and item[1] == key[1]]
            start_key = starts[0] if starts else None
            start = rollout_starts.pop(start_key, ts) if start_key is not None else ts
            intervals['rollout'].append((start, max(ts - start, 0.001)))
        elif name == 'train_claimed' or (phase == 'train' and name == 'train_started'):
            train_starts[key] = ts
        elif name == 'partition_train_done':
            start = train_starts.pop(key, ts)
            intervals['train'].append((start, max(ts - start, 0.001)))
    return intervals


def _plot_policy_gap_histogram(plt, runs: list[dict[str, Any]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False
    for run in runs:
        values = []
        for event in run['events']:
            for key in ('policy_version_gap_mean', 'policy_version_gap'):
                value = _metric(event, key)
                if value is not None:
                    values.append(value)
                    break
        if values:
            ax.hist(values, bins=min(20, max(1, len(set(values)))), alpha=0.45, label=run['label'])
            plotted = True
    ax.set_title('policy version gap histogram')
    ax.set_xlabel('policy version gap')
    ax.set_ylabel('count')
    if plotted:
        ax.legend(loc='best')
    else:
        _write_no_data(ax)
    _save(fig, output)


def _plot_backlog(plt, runs: list[dict[str, Any]], output: Path) -> None:
    keys = [
        'pending_prompt_groups',
        'inflight_rollout_groups',
        'untrained_groups',
        'rollout_done_groups',
        'advantage_done_groups',
        'training_groups',
    ]
    fig, axes = plt.subplots(len(runs), 1, figsize=(9, max(3, 2.6 * len(runs))), squeeze=False)
    for ax, run in zip(axes[:, 0], runs):
        plotted = False
        step_events = [event for event in run['events'] if event.get('event') == 'pipeline_step']
        for key in keys:
            xs = []
            ys = []
            for event in step_events:
                value = _metric(event, key)
                if value is not None:
                    xs.append(float(event.get('elapsed_s') or 0.0))
                    ys.append(value)
            if ys:
                ax.plot(xs, ys, label=key)
                plotted = True
        ax.set_title(f'backlog over time: {run["label"]}')
        ax.set_xlabel('wall-clock time (s)')
        ax.set_ylabel('groups / partitions')
        if plotted:
            ax.legend(loc='best', fontsize='small')
        else:
            _write_no_data(ax)
    _save(fig, output)


def _plot_context_fairness(plt, runs: list[dict[str, Any]], output: Path) -> None:
    labels = []
    values = []
    for run in runs:
        counts: dict[str, int] = defaultdict(int)
        for event in run['events']:
            if event.get('event') == 'train_batch_done':
                context = str(event.get('adapter_name') or event.get('context_key') or 'unknown')
                counts[f'{run["label"]}\n{context}'] += 1
        for label, count in sorted(counts.items()):
            labels.append(label)
            values.append(count)
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(labels)), 4.5))
    if values:
        ax.bar(labels, values, color='#F28E2B')
        ax.tick_params(axis='x', rotation=45)
    else:
        _write_no_data(ax)
    ax.set_title('per-context train steps fairness')
    ax.set_ylabel('train_batch_done count')
    _save(fig, output)


def _plot_gpu_utilization(plt, paths: list[Path], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    plotted = False
    for path in paths:
        samples = _read_gpu_csv(path)
        grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for sample in samples:
            role = sample.get('role') or f'gpu{sample.get("gpu_index", "")}'
            try:
                elapsed = float(sample['elapsed_s'])
                util = float(sample['gpu_util'])
            except (KeyError, TypeError, ValueError):
                continue
            grouped[str(role)].append((elapsed, util))
        for role, points in sorted(grouped.items()):
            xs, ys = zip(*points)
            ax.plot(xs, ys, label=f'{path.parent.name}:{role}')
            plotted = True
    ax.set_title('GPU utilization timeline')
    ax.set_xlabel('wall-clock time (s)')
    ax.set_ylabel('GPU util (%)')
    if plotted:
        ax.legend(loc='best', fontsize='small')
    else:
        _write_no_data(ax)
    _save(fig, output)


def _read_gpu_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as file:
        return list(csv.DictReader(file))


def _finish_line_plot(fig, ax, output: Path, *, title: str, xlabel: str, ylabel: str, plotted: bool) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if plotted:
        ax.legend(loc='best')
    else:
        _write_no_data(ax)
    _save(fig, output)


def _write_no_data(ax) -> None:
    ax.text(0.5, 0.5, 'no data', transform=ax.transAxes, ha='center', va='center')


def _save(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    fig.clf()


if __name__ == '__main__':
    main()
