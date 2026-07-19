import pytest

from twinkle.reward import DAPOMathAccuracyReward, DAPOMathReward


def _trajectory(completion: str, ground_truth: str) -> dict:
    return {
        'messages': [
            {'role': 'user', 'content': 'Solve it.'},
            {'role': 'assistant', 'content': completion},
        ],
        'user_data': [('ground_truth', ground_truth)],
    }


def _rollout_trajectory(completion: str, ground_truth: str, completion_length: int) -> dict:
    trajectory = _trajectory(completion, ground_truth)
    trajectory['completion_length'] = completion_length
    return trajectory


def test_dapo_math_reward_accepts_answer_line_and_boxed_formats():
    reward = DAPOMathAccuracyReward()

    assert reward([
        _trajectory('Reasoning.\nAnswer: 34', '34'),
        _trajectory('Reasoning.\nAnswer: $34$', '34'),
        _trajectory('Reasoning. \\boxed{34}', '34'),
        _trajectory('Reasoning.\nAnswer: 35', '34'),
    ]) == [1.0, 1.0, 1.0, 0.0]


def test_dapo_math_reward_uses_last_answer_line():
    reward = DAPOMathAccuracyReward()

    assert reward([_trajectory('Answer: 12\nCorrection.\nAnswer: 13', '13')]) == [1.0]


def test_dapo_math_reward_reports_accuracy_metric():
    reward = DAPOMathAccuracyReward()

    assert reward.metric_payload([], rewards=[1.0, 0.0]) == {'accuracy_reward': 0.5}


def test_dapo_training_reward_uses_signed_accuracy_and_overlong_shaping():
    reward = DAPOMathReward(
        max_response_length=8192,
        overlong_buffer_length=4096,
        overlong_penalty_factor=1.0,
    )
    trajectories = [
        _rollout_trajectory('Answer: 34', '34', 4096),
        _rollout_trajectory('Answer: 34', '34', 6144),
        _rollout_trajectory('Answer: 35', '34', 4096),
        _rollout_trajectory('Answer: 35', '34', 8192),
    ]

    assert reward(trajectories) == [1.0, 0.5, -1.0, -2.0]
    assert reward.metric_payload(trajectories, rewards=reward(trajectories)) == {
        'total_reward': -0.375,
        'accuracy_reward': 0.5,
        'overlong_reward': -0.375,
        'overlong_ratio': 0.5,
    }


def test_dapo_training_reward_scores_only_completion_tail():
    reward = DAPOMathReward(max_response_length=8192, overlong_buffer_length=4096, score_tail_chars=300)
    trailing_garbage = 'Answer: 34' + (' ' * 301)

    assert reward([_rollout_trajectory(trailing_garbage, '34', 500)]) == [-1.0]


def test_dapo_training_reward_requires_token_completion_length():
    reward = DAPOMathReward(max_response_length=8192, overlong_buffer_length=4096)

    with pytest.raises(ValueError, match='completion_length'):
        reward([_trajectory('Answer: 34', '34')])
