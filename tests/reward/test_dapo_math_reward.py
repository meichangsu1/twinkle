from twinkle.reward import DAPOMathAccuracyReward


def _trajectory(completion: str, ground_truth: str) -> dict:
    return {
        'messages': [
            {'role': 'user', 'content': 'Solve it.'},
            {'role': 'assistant', 'content': completion},
        ],
        'user_data': [('ground_truth', ground_truth)],
    }


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
