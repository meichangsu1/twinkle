#!/usr/bin/env python
# Copyright (c) ModelScope Contributors. All rights reserved.
"""Plot the parent summary.json from benchmark_grpo_sticky_session.py offline."""
from __future__ import annotations
import argparse
import statistics
import sys
from pathlib import Path

import json

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COLORS = {'A': '#8b8b8b', 'B': '#327bb6', 'D': '#188b74'}
LABELS = {'A': 'A: cache off / affinity off', 'B': 'B: cache on / affinity off',
          'D': 'D: cache on / group affinity on'}


def load_summary(path):
    path = Path(path)
    if path.is_dir():
        path /= 'summary.json'
    summary = json.loads(path.read_text())
    if summary.get('benchmark') not in ('real_grpo_group_affinity', 'synthetic_multiturn_grpo_affinity'):
        raise ValueError('Expected the parent summary from benchmark_grpo_sticky_session.py')
    trials = [trial for trial in summary['trials'] if trial['status'] == 'complete']
    if not trials:
        raise ValueError('No completed real training trials to plot')
    keys = [(trial['group'], trial['seed']) for trial in trials]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate group/seed trials')
    if len({trial['plan_hash'] for trial in trials}) != 1:
        raise ValueError('Trials used different input plans')
    return path, summary, trials


def render(path, output_dir=None, formats=('png',), dpi=160):
    path, summary, trials = load_summary(path)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Recompute pairs from the trial records; never trust stale derived ratios.
    from scripts.benchmark_grpo_sticky_session import paired_comparisons
    pairs = paired_comparisons(trials)
    output = Path(output_dir) if output_dir else path.parent / 'plots'
    output.mkdir(parents=True, exist_ok=True)
    groups = [group for group in COLORS if any(trial['group'] == group for trial in trials)]
    synthetic = summary['benchmark'] == 'synthetic_multiturn_grpo_affinity'
    task = 'Synthetic multi-turn GRPO' if synthetic else 'GRPO'
    title = f'{task} | {summary["config"]["model"]} | {summary["config"]["device"]}'
    labels = {**LABELS, 'D': 'D: cache on / episode affinity on'} if synthetic else LABELS
    if summary['status'] != 'complete':
        title += ' | INCOMPLETE RUN'
    footer = ('Dots: independent seed trials; bars: median; whiskers: min-max (not a confidence interval).\n'
              'Warmup batches excluded. Missing metrics stay unavailable; generated trajectories can differ.')
    artifacts = []

    def save(figure, name):
        for extension in formats:
            destination = output / f'{name}.{extension}'
            figure.savefig(destination, dpi=dpi, bbox_inches='tight')
            artifacts.append(destination)
        plt.close(figure)

    def bars(axis, metric, ylabel, scale=1):
        for index, group in enumerate(groups):
            values = [trial.get(metric) for trial in trials if trial['group'] == group]
            values = [value * scale for value in values if value is not None]
            if not values:
                axis.text(index, 0, 'N/A', ha='center', va='bottom')
                continue
            median = statistics.median(values)
            axis.bar(index, median, width=0.55, color=COLORS[group], alpha=0.75)
            axis.errorbar(index, median, yerr=[[median - min(values)], [max(values) - median]],
                          color=COLORS[group], capsize=5)
            offsets = [0] if len(values) == 1 else [-0.12 + 0.24 * i / (len(values) - 1) for i in range(len(values))]
            axis.scatter([index + delta for delta in offsets], values, color=COLORS[group],
                         edgecolors='white', zorder=3)
        axis.set_xticks(range(len(groups)), groups)
        axis.set_xlim(-0.6, len(groups) - 0.4)
        axis.set_ylabel(ylabel)
        axis.set_ylim(bottom=0)
        axis.grid(axis='y', alpha=0.2)
        axis.set_axisbelow(True)

    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = [
        ('sample_trajectories_per_s', 'Sampling throughput (higher is better)', 'trajectories / s', 1),
        ('train_trajectories_per_s', 'End-to-end training throughput (higher is better)', 'trajectories / s', 1),
        ('cache_hit_fraction', 'Token-weighted prefix-cache hits', '% of input tokens', 100),
        ('completion_tokens_mean', 'Actual completion length', 'tokens / completion', 1),
        ('accuracy', 'Training rollout answer accuracy', '%', 100),
        ('mean_replicas_per_prompt', 'Instances serving one prompt group', 'instances / prompt', 1),
    ]
    if synthetic:
        panels[3] = ('completion_tokens_mean', 'Actual generated tokens across all turns', 'tokens / episode', 1)
        panels[4] = ('accuracy', 'Synthetic lookup accuracy across turns', '%', 100)
        panels[5] = ('mean_replicas_per_episode', 'Instances serving one episode', 'instances / episode', 1)
    for axis, (metric, panel_title, ylabel, scale) in zip(axes.flat, panels, strict=True):
        bars(axis, metric, ylabel, scale)
        axis.set_title(panel_title, fontsize=11)
    figure.suptitle(title)
    figure.legend([plt.Rectangle((0, 0), 1, 1, color=COLORS[group]) for group in groups],
                  [labels[group] for group in groups], loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=3)
    figure.text(0.02, 0.01, footer, fontsize=9, color='#555555')
    figure.tight_layout(rect=(0, 0.08, 1, 0.89))
    save(figure, 'overview')

    figure, axis = plt.subplots(figsize=(10, 6))
    bottoms = [0.0] * len(groups)
    stages = [('sync_s', 'Weight sync'), ('cache_reset_s', 'Cache reset'), ('sample_s', 'Sampling'),
              ('reward_advantage_s', 'Reward + advantage'), ('train_s', 'Training update'), ('overhead_s', 'Other')]
    for stage, label in stages:
        values = [statistics.mean(trial['stage_totals_s'][stage] / trial['measured_batches']
                                  for trial in trials if trial['group'] == group) for group in groups]
        axis.bar(groups, values, bottom=bottoms, label=label)
        bottoms = [before + value for before, value in zip(bottoms, values, strict=True)]
    axis.set_ylabel('seconds / complete rollout batch')
    axis.set_title(title + '\nMean stage times across seed trials; device synchronization included')
    axis.legend(loc='upper left', bbox_to_anchor=(1, 1))
    figure.tight_layout()
    save(figure, 'stage_times')

    if synthetic:
        figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        for axis, (metric, panel_title, ylabel, scale) in zip(axes, [
                ('continuation_cache_hit_fraction', 'Prefix-cache hits after the first turn', '% of input tokens', 100),
                ('replica_switch_rate', 'Replica switches between consecutive turns', '% of transitions', 100),
                ('initial_prompt_tokens_mean', 'Actual initial prompt length', 'tokens', 1)], strict=True):
            bars(axis, metric, ylabel, scale)
            axis.set_title(panel_title, fontsize=10)
        figure.suptitle(title)
        figure.tight_layout()
        save(figure, 'multiturn')

    if pairs:
        figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        comparisons = [key for key in ('B/A', 'D/B', 'D/A') if any(pair['comparison'] == key for pair in pairs)]
        for axis, (metric, panel_title) in zip(axes, [
                ('sample_throughput_ratio', 'Sampling throughput ratio'),
                ('train_throughput_ratio', 'Training throughput ratio'),
                ('output_tokens_ratio', 'Actual output-token ratio')], strict=True):
            for index, comparison in enumerate(comparisons):
                values = [pair[metric] for pair in pairs if pair['comparison'] == comparison]
                median = statistics.median(values)
                axis.bar(index, median, color='#327bb6', alpha=0.65)
                axis.errorbar(index, median, yerr=[[median - min(values)], [max(values) - median]],
                              color='#333333', capsize=4)
                axis.scatter([index] * len(values), values, color='#188b74', zorder=3)
                axis.text(index, median, f'{median:.3f}x', ha='center', va='bottom')
            axis.axhline(1, color='#777777', linestyle='--', linewidth=1)
            axis.set_xticks(range(len(comparisons)), comparisons)
            axis.set_title(panel_title)
            axis.set_ylim(bottom=0)
        figure.suptitle('Paired by seed; each ratio is computed before aggregation')
        figure.tight_layout()
        save(figure, 'paired_ratios')
    return artifacts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary')
    parser.add_argument('--output-dir')
    parser.add_argument('--formats', nargs='+', choices=['png', 'svg', 'pdf'], default=['png'])
    parser.add_argument('--dpi', type=int, default=160)
    args = parser.parse_args(argv)
    for path in render(args.summary, args.output_dir, args.formats, args.dpi):
        sys.stdout.write(str(path) + '\n')


if __name__ == '__main__':
    main()
