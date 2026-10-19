# Copyright (c) ModelScope Contributors. All rights reserved.
import copy

import json
import pytest

from scripts.benchmark_grpo_sticky_session import paired_comparisons
from scripts.plot_grpo_sticky_session import load_summary, render


def example():
    trials = []
    for seed in (42, 43, 44):
        for group, factor in (('A', 1), ('B', 1.1), ('D', 1.2)):
            stages = {'sync_s': 1, 'cache_reset_s': 0.1, 'sample_s': 4 / factor,
                      'train_s': 3, 'reward_advantage_s': 0.2, 'overhead_s': 0.2}
            stages['total_s'] = sum(stages.values())
            trials.append({'status': 'complete', 'group': group, 'seed': seed, 'plan_hash': 'fixed',
                           'trajectories': 16, 'optimizer_updates': 4, 'measured_batches': 2,
                           'stage_totals_s': stages, 'sample_trajectories_per_s': 16 / stages['sample_s'],
                           'train_trajectories_per_s': 16 / stages['total_s'], 'output_tokens': 128,
                           'cache_hit_fraction': None if group == 'A' else 0.5,
                           'completion_tokens_mean': 8, 'accuracy': 0.5, 'mean_replicas_per_prompt': 1})
    return {'benchmark': 'real_grpo_group_affinity', 'status': 'complete',
            'config': {'model': 'TEST FIXTURE - NOT PERFORMANCE DATA', 'device': 'cpu'},
            'trials': trials, 'comparisons': paired_comparisons(trials)}


def test_real_training_summary_renders_all_formats_and_missing_metrics(tmp_path):
    pytest.importorskip('matplotlib')
    path = tmp_path / 'summary.json'
    path.write_text(json.dumps(example()))
    artifacts = render(path, formats=('png', 'svg', 'pdf'), dpi=70)
    assert len(artifacts) == 9
    assert all(artifact.stat().st_size > 1000 for artifact in artifacts)


@pytest.mark.parametrize('change,match', [
    ('kind', 'parent summary'), ('empty', 'No completed'), ('duplicate', 'Duplicate'), ('plan', 'different input'),
])
def test_invalid_or_unmatched_runs_fail_before_plotting(tmp_path, change, match):
    data = copy.deepcopy(example())
    if change == 'kind':
        data['benchmark'] = 'other'
    elif change == 'empty':
        data['trials'] = []
    elif change == 'duplicate':
        data['trials'].append(copy.deepcopy(data['trials'][0]))
    else:
        data['trials'][0]['plan_hash'] = 'other'
    (tmp_path / 'summary.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match=match):
        load_summary(tmp_path)


def test_incomplete_run_ignores_failed_trials(tmp_path):
    data = example()
    data['status'] = 'failed'
    data['trials'].append({'group': 'D', 'seed': 99, 'status': 'failed', 'error': 'worker failed'})
    (tmp_path / 'summary.json').write_text(json.dumps(data))
    _, summary, trials = load_summary(tmp_path)
    assert summary['status'] == 'failed'
    assert len(trials) == 9
