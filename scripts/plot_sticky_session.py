#!/usr/bin/env python
# Copyright (c) ModelScope Contributors. All rights reserved.
"""Plot a sticky-session benchmark summary without Ray, models or accelerators.

python scripts/plot_sticky_session.py /tmp/sticky-session-npu/summary.json
"""
from __future__ import annotations
import argparse
import math
import statistics
from pathlib import Path

import json

GROUP_LABELS = {
    'A': 'A: affinity off / cache off', 'B': 'B: affinity off / cache on',
    'C': 'C: affinity on / cache off', 'D': 'D: affinity on / cache on',
}
COLORS = {'A': '#7C8799', 'B': '#3479B5', 'C': '#D58A20', 'D': '#16866B'}
METRICS = [
    ('Episode throughput (higher is better)', ('episodes_per_s',), 1, 'episodes / s'),
    ('Episode p95 latency (lower is better)', ('episode_latency_p95_s',), 1, 'seconds'),
    ('Continuation prefix-cache hit rate', ('continuation_cache_hit_fraction',), 100, '% of input tokens'),
    ('Engine TTFT p50 (lower is better)', ('engine_ttft_s', 'p50'), 1000, 'milliseconds'),
    ('Engine prefill p50 (lower is better)', ('engine_prefill_s', 'p50'), 1000, 'milliseconds'),
    ('Replica switch rate (lower is better)', ('worker_switch_fraction',), 100, '% of turn transitions'),
]


def number(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
        return float(value)
    return None


def metric_value(trial, path):
    value = trial
    for name in path:
        value = value.get(name) if isinstance(value, dict) else None
    return number(value)


def select_trials(summary):
    """Exclude failed/incomplete measurements, and reject ambiguous repeat keys."""
    selected = []
    seen = set()
    expected = summary.get('config', {}).get('episodes')
    for trial in summary['trials']:
        key = (trial['group'], trial['concurrency'], trial['repeat'])
        if key in seen:
            raise ValueError(f'Duplicate group/concurrency/repeat: {key}')
        seen.add(key)
        completed = trial.get('completed_episodes', 0)
        if trial.get('errors', 0) or completed <= 0 or (expected is not None and completed != expected):
            continue
        if trial['group'] not in GROUP_LABELS:
            raise ValueError(f'Unknown benchmark group: {trial["group"]}')
        selected.append(trial)
    return selected


def metric_series(trials, group, concurrencies, path, scale=1):
    return [[value * scale for trial in trials
             if trial['group'] == group and trial['concurrency'] == concurrency
             and (value := metric_value(trial, path)) is not None]
            for concurrency in concurrencies]


def paired_ratios(trials, concurrencies, metric):
    """Pair the same repeat before taking the median; never divide two medians."""
    indexed = {(trial['group'], trial['concurrency'], trial['repeat']): trial for trial in trials}
    result = []
    for concurrency in concurrencies:
        values = []
        for trial in trials:
            if trial['group'] != 'D' or trial['concurrency'] != concurrency:
                continue
            baseline = indexed.get(('B', concurrency, trial['repeat']))
            if baseline is not None:
                numerator, denominator = metric_value(trial, (metric,)), metric_value(baseline, (metric,))
                if numerator is not None and denominator is not None and denominator > 0:
                    values.append(numerator / denominator)
        result.append(values)
    return result


def instance_shares(summary, trials):
    """Mean per-trial input-token shares, including registered replicas with zero work."""
    rows, labels = [], []
    for group in GROUP_LABELS:
        group_trials = [trial for trial in trials if trial['group'] == group]
        ids = set()
        for trial in group_trials:
            ids.update(trial.get('router_after', {}).get('active_instances', []))
            ids.update(trial.get('per_instance', {}))
        for pool in summary.get('replica_pools', []):
            ids.update(info['instance_id'] for info in pool if group_trials
                       and info['prefix_cache'] == group_trials[0]['prefix_cache'])
        ids = sorted(ids)
        for concurrency in sorted({trial['concurrency'] for trial in group_trials}):
            shares = []
            for trial in group_trials:
                if trial['concurrency'] != concurrency:
                    continue
                work = trial.get('per_instance', {})
                total = sum(instance['input_tokens'] for instance in work.values())
                if total > 0:
                    shares.append([100 * work.get(key, {}).get('input_tokens', 0) / total for key in ids])
            if shares:
                rows.append([statistics.mean(values) for values in zip(*shares, strict=True)])
                labels.append(f'{group} / concurrency {concurrency}')
    return labels, rows


def draw_series(ax, values, color, label, offset=0):
    medians = [statistics.median(items) if items else math.nan for items in values]
    lows = [min(items) if items else math.nan for items in values]
    highs = [max(items) if items else math.nan for items in values]
    positions = [index + offset for index in range(len(values))]
    ax.plot(positions, medians, color=color, marker='o', markersize=5, linewidth=2, label=label)
    ax.fill_between(positions, lows, highs, color=color, alpha=0.12)
    for position, items in zip(positions, values, strict=True):
        # Deterministic small offsets keep repeated measurements visible.
        xs = [position + 0.12 * (index - (len(items) - 1) / 2) / max(1, len(items) - 1)
              for index in range(len(items))]
        ax.scatter(xs, items, color=color, alpha=0.5, s=13, zorder=3)


def style_axis(ax, concurrencies, title, ylabel, available, percent=False):
    ax.set(title=title, xlabel='Concurrent episodes', ylabel=ylabel)
    ax.set_xticks(range(len(concurrencies)), [str(value) for value in concurrencies])
    ax.set_xlim(-0.35, len(concurrencies) - 0.65)
    ax.set_ylim(bottom=0, top=100 if percent else None)
    ax.grid(axis='y', color='#E4E7EB', linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    if not available:
        ax.text(0.5, 0.5, 'Metric unavailable', ha='center', va='center', color='#7C8799', transform=ax.transAxes)


def caption(summary, excluded):
    config = summary.get('config', {})
    model = (config.get('model') or config.get('backend', 'unknown')).rstrip('/').rsplit('/', 1)[-1]
    device = 'CPU mock' if summary.get('synthetic') else config.get('device', 'unspecified device')
    prefix = 'SYNTHETIC / MOCK | ' if summary.get('synthetic') else ''
    title = f'{prefix}Sticky session | {model} | {device}'
    if summary.get('error') or excluded:
        title += ' | INCOMPLETE RUN'
    footer = f'Missing metrics stay unavailable. Excluded failed/incomplete trials: {excluded}.'
    if summary.get('synthetic'):
        footer += '\nSynthetic data: these plots do not measure NPU/GPU performance.'
    if summary.get('error'):
        footer += f'\nRun error: {str(summary["error"])[:160]}'
    return title, footer


def make_figures(summary, trials):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11})
    concurrencies = sorted({trial['concurrency'] for trial in trials})
    groups = [group for group in GROUP_LABELS if any(trial['group'] == group for trial in trials)]
    title, footer = caption(summary, len(summary['trials']) - len(trials))
    spread_note = 'Dots: repeats; line: median; band: min-max (not a confidence interval).'
    figures = {}

    overview, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (name, path, scale, unit) in zip(axes.flat, METRICS, strict=True):
        available = False
        for index, group in enumerate(groups):
            values = metric_series(trials, group, concurrencies, path, scale)
            available |= any(values)
            draw_series(ax, values, COLORS[group], group, offset=(index - (len(groups) - 1) / 2) * 0.055)
        style_axis(ax, concurrencies, name, unit, available, percent=scale == 100)
    overview.suptitle(title, y=0.98, fontsize=15)
    overview.legend(handles=[Line2D([], [], color=COLORS[group], marker='o', label=GROUP_LABELS[group])
                             for group in groups], loc='upper center', bbox_to_anchor=(0.5, 0.94),
                    ncol=2, frameon=False)
    overview.text(0.03, 0.025, f'{spread_note}\n{footer}', va='bottom', fontsize=9, color='#59636F')
    overview.tight_layout(rect=(0.01, 0.09, 0.99, 0.87), h_pad=2.4)
    figures['overview'] = overview

    ratios, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, metric, name in zip(axes, ['episodes_per_s', 'episode_latency_p95_s'],
                                ['Throughput D/B (>1 favors sticky)', 'Episode p95 D/B (<1 favors sticky)'],
                                strict=True):
        values = paired_ratios(trials, concurrencies, metric)
        draw_series(ax, values, COLORS['D'], 'D/B')
        ax.axhline(1, color='#7C8799', linestyle='--', linewidth=1)
        style_axis(ax, concurrencies, name, 'Ratio', any(values))
        if not any(values):
            ax.text(0.5, 0.4, 'Needs valid B + D from the same repeat', ha='center',
                    color='#7C8799', fontsize=9, transform=ax.transAxes)
    ratios.suptitle(title, fontsize=13)
    ratios.text(0.03, 0.025, f'{spread_note}\n{footer}', va='bottom', fontsize=8, color='#59636F')
    ratios.tight_layout(rect=(0.01, 0.17, 0.99, 0.92))
    figures['sticky_ratio'] = ratios

    labels, rows = instance_shares(summary, trials)
    load, ax = plt.subplots(figsize=(10, max(4.5, len(rows) * 0.4 + 2)))
    if rows:
        width = max(map(len, rows))
        matrix = [row + [math.nan] * (width - len(row)) for row in rows]
        heatmap = ax.imshow(matrix, vmin=0, vmax=100, cmap='Blues', aspect='auto')
        ax.set_xticks(range(width), [f'Replica {index + 1}' for index in range(width)])
        ax.set_yticks(range(len(labels)), labels)
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                ax.text(j, i, f'{value:.1f}%', ha='center', va='center',
                        color='white' if value > 55 else '#243447')
        load.colorbar(heatmap, ax=ax, label='Mean input-token share across repeats (%)')
    else:
        ax.text(0.5, 0.5, 'Per-instance workload unavailable', ha='center', transform=ax.transAxes)
    ax.set_title('Input-token distribution (includes idle replicas)')
    load.suptitle(title, fontsize=13)
    note = 'Replica indices are local to each cache-config pool; token share is not device utilization.'
    load.text(0.03, 0.025, f'{note}\n{footer}', fontsize=8, va='bottom', color='#59636F')
    load.tight_layout(rect=(0.01, 0.18, 0.99, 0.92))
    figures['instance_load'] = load
    return figures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path, help='summary.json or its containing directory')
    parser.add_argument('--output-dir', type=Path, help='Default: <summary directory>/plots')
    parser.add_argument('--formats', nargs='+', choices=['png', 'svg', 'pdf'], default=['png'])
    parser.add_argument('--dpi', type=int, default=160)
    args = parser.parse_args(argv)
    if args.dpi < 1:
        parser.error('--dpi must be positive')
    source = args.summary / 'summary.json' if args.summary.is_dir() else args.summary
    try:
        summary = json.loads(source.read_text())
        trials = select_trials(summary)
        if not trials:
            raise ValueError('No successful, complete trials to plot')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(f'Cannot plot {source}: {exc}')
    try:
        figures = make_figures(summary, trials)
    except ImportError:
        parser.error('Plotting requires matplotlib: python -m pip install matplotlib')
    import matplotlib.pyplot as plt

    destination = args.output_dir or source.parent / 'plots'
    destination.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        for extension in dict.fromkeys(args.formats):
            target = destination / f'{name}.{extension}'
            figure.savefig(target, dpi=args.dpi, facecolor='white')
            print(target)  # noqa: T201
        plt.close(figure)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
