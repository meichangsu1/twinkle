from __future__ import annotations

import json

import pytest

from twinkle_agentic.async_rl.metrics import (JSONLMetricsRecorder, advantage_signal_metrics,
                                              rollout_performance_metrics)


def test_advantage_signal_metrics_report_zero_and_nonzero_groups():
    metrics = advantage_signal_metrics(
        rewards=[1.0, 1.0, 0.0, 1.0],
        advantages=[0.0, 0.0, -1.0, 1.0],
        num_generations=2,
    )

    assert metrics['group_count'] == 2
    assert metrics['group_reward_std_mean'] == pytest.approx(0.25)
    assert metrics['zero_advantage_group_ratio'] == pytest.approx(0.5)
    assert metrics['positive_advantage_ratio'] == pytest.approx(0.25)
    assert metrics['advantage_mean'] == pytest.approx(0.0)
    assert metrics['advantage_std'] == pytest.approx(2**-0.5)


def test_advantage_signal_metrics_reject_incomplete_groups():
    with pytest.raises(ValueError, match='complete groups'):
        advantage_signal_metrics([1.0, 0.0, 1.0], [1.0, -1.0, 0.0], num_generations=2)


def test_rollout_performance_metrics_include_tokens_and_truncation():
    metrics = rollout_performance_metrics(
        [
            {
                'completion_length': 3,
                'stop_reason': 'stop'
            },
            {
                'completion_length': 7,
                'stop_reason': 'length'
            },
        ],
        rollout_latency_s=2.0,
    )

    assert metrics == {
        'sample_count': 2,
        'completion_length_mean': 5.0,
        'completion_length_p95': 7,
        'completion_length_max': 7,
        'output_tokens': 10,
        'completion_truncated_samples': 1,
        'completion_truncated_ratio': 0.5,
        'rollout_latency_s': 2.0,
        'output_tokens_per_s': 5.0,
    }


def test_jsonl_metrics_recorder_flushes_in_batches(tmp_path):
    path = tmp_path / 'metrics.jsonl'
    recorder = JSONLMetricsRecorder(path, run_id='test', flush_every=2, flush_interval_s=60)

    recorder.record(event='first')
    assert path.read_text() == ''

    recorder.record(event='second')
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record['event'] for record in records] == ['first', 'second']
    assert recorder.stats()['metrics_event_count'] == 2
    assert recorder.stats()['metrics_flush_count'] == 1
    recorder.close()


def test_jsonl_metrics_recorder_close_flushes_pending_events(tmp_path):
    path = tmp_path / 'metrics.jsonl'
    recorder = JSONLMetricsRecorder(path, run_id='test', flush_every=64, flush_interval_s=60)

    recorder.record(event='run_completed')
    recorder.close()
    recorder.close()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record['event'] for record in records] == ['run_completed']
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary['status'] == 'run_completed'
    assert summary['event_counts'] == {'run_completed': 1}


def test_jsonl_metrics_recorder_writes_experiment_summary(tmp_path):
    path = tmp_path / 'metrics.jsonl'
    recorder = JSONLMetricsRecorder(path, run_id='test', mode='async_strict')

    class Context:
        key = 'tenant/run/adapter'

    context = Context()
    recorder.record(
        event='rollout_done',
        context=context,
        policy_version=0,
        metrics={'sample_count': 4, 'reward': 0.5, 'completion_length_mean': 12.0},
    )
    recorder.record(
        event='train_step_done',
        context=context,
        policy_version=0,
        metrics={'sample_count': 4, 'loss': '0.25'},
    )
    recorder.record(event='partition_done', context=context, policy_version=1)
    recorder.record(event='run_completed', metrics={'trained_partitions': 1, 'wall_time_s': 10.0})
    recorder.close()

    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary['run_id'] == 'test'
    assert summary['mode'] == 'async_strict'
    assert summary['rollout_groups'] == 1
    assert summary['rollout_samples'] == 4
    assert summary['train_steps'] == 1
    assert summary['trained_samples'] == 4
    assert summary['trained_partitions'] == 1
    assert summary['train_steps_per_hour'] == 360.0
    assert summary['per_context']['tenant/run/adapter'] == {
        'rollout_groups': 1,
        'rollout_samples': 4,
        'train_steps': 1,
        'trained_samples': 4,
        'trained_partitions': 1,
        'policy_version': 1,
    }
    assert summary['metrics']['rollout_done/reward']['mean'] == 0.5
    assert summary['metrics']['train_step_done/loss']['mean'] == 0.25
