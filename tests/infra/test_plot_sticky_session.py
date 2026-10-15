# Copyright (c) ModelScope Contributors. All rights reserved.
import subprocess
import sys
from pathlib import Path

import json
import pytest

from scripts.plot_sticky_session import instance_shares, metric_series, paired_ratios, select_trials


def trial(group='B', repeat=0, concurrency=4, **kwargs):
    return {
        'group': group, 'repeat': repeat, 'concurrency': concurrency, 'errors': 0,
        'completed_episodes': 4, 'prefix_cache': group in ('B', 'D'),
        'episodes_per_s': 10, 'episode_latency_p95_s': 1,
        'continuation_cache_hit_fraction': 0.5, 'worker_switch_fraction': 0,
        'engine_ttft_s': {'p50': None, 'count': 0},
        'per_instance': {'one': {'input_tokens': 100}},
        'router_after': {'active_instances': ['one', 'two']},
        **kwargs,
    }


def test_excludes_failed_and_incomplete_trials_and_rejects_duplicate_repeat_keys():
    data = {'config': {'episodes': 4}, 'trials': [
        trial(), trial(repeat=1, errors=1), trial(repeat=2, completed_episodes=3),
    ]}
    assert select_trials(data) == [data['trials'][0]]
    data['trials'].append(trial())
    with pytest.raises(ValueError, match='Duplicate'):
        select_trials(data)


def test_missing_metrics_stay_missing_and_actual_zero_is_retained():
    values = [trial(continuation_cache_hit_fraction=0), trial(repeat=1, continuation_cache_hit_fraction=None)]
    assert metric_series(values, 'B', [4, 8], ('continuation_cache_hit_fraction',), 100) == [[0], []]
    assert metric_series(values, 'B', [4], ('engine_ttft_s', 'p50'), 1000) == [[]]


def test_ratios_pair_repeats_instead_of_dividing_group_medians():
    values = []
    for repeat, (baseline, sticky) in enumerate(zip([1, 10, 100], [2, 30, 100], strict=True)):
        values += [trial('B', repeat, episodes_per_s=baseline), trial('D', repeat, episodes_per_s=sticky)]
    # Median paired ratio = 2; ratio of group medians would incorrectly be 3.
    assert paired_ratios(values, [4, 8], 'episodes_per_s') == [[2, 3, 1], []]
    values += [trial('D', 3), trial('B', 4, episodes_per_s=0), trial('D', 4)]
    assert paired_ratios(values, [4], 'episodes_per_s') == [[2, 3, 1]]


def test_load_shares_include_idle_registered_replicas_and_normalize_each_repeat():
    values = [trial(), trial(repeat=1, per_instance={'two': {'input_tokens': 300}})]
    labels, rows = instance_shares({}, values)
    assert labels == ['B / concurrency 4']
    assert rows == [[50, 50]]  # Equal repeat weights; do not pool 100 vs 300 tokens.
    assert instance_shares({}, [values[0]])[1] == [[100, 0]]
    values[0]['router_after'] = {}
    summary = {'replica_pools': [[{'instance_id': key, 'prefix_cache': True} for key in ['one', 'two']]]}
    assert instance_shares(summary, [values[0]])[1] == [[100, 0]]


def test_cli_renders_headless_and_marks_mock_incomplete_and_missing_metrics(tmp_path):
    pytest.importorskip('matplotlib')
    root = Path(__file__).resolve().parents[2]
    data = {
        'config': {'episodes': 4, 'backend': 'mock', 'replicas': 2}, 'synthetic': True,
        'trials': [trial(group, repeat=repeat, concurrency=concurrency,
                         episodes_per_s=10 + repeat + concurrency * 0.1,
                         episode_latency_p95_s=1 + repeat * 0.1)
                   for group in ['B', 'D'] for concurrency in [4, 8] for repeat in [0, 1]]
                  + [trial('A', errors=1), trial('C', completed_episodes=1)],
    }
    (tmp_path / 'summary.json').write_text(json.dumps(data))
    result = subprocess.run([sys.executable, str(root / 'scripts/plot_sticky_session.py'),
                             str(tmp_path), '--formats', 'png', 'svg'],
                            cwd=root, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    for name in ['overview', 'sticky_ratio', 'instance_load']:
        png = tmp_path / 'plots' / f'{name}.png'
        assert png.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
        assert png.stat().st_size > 10000
        svg = (tmp_path / 'plots' / f'{name}.svg').read_text()
        assert 'SYNTHETIC / MOCK' in svg and 'INCOMPLETE RUN' in svg
        assert 'Excluded failed/incomplete trials: 2' in svg
    assert 'Metric unavailable' in (tmp_path / 'plots' / 'overview.svg').read_text()


def test_cli_reports_no_complete_trials_before_importing_plotting_libraries(tmp_path):
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / 'summary.json'
    source.write_text(json.dumps({'trials': [trial(errors=1)]}))
    result = subprocess.run([sys.executable, str(root / 'scripts/plot_sticky_session.py'), str(source)],
                            cwd=root, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert 'No successful, complete trials' in result.stderr
    assert not (tmp_path / 'plots').exists()
