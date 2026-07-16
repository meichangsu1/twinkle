from twinkle.reward import BoxedMathAccuracyReward


def _trajectory(completion: str, ground_truth: str) -> dict:
    return {
        'messages': [
            {
                'role': 'user',
                'content': 'Solve it.',
            },
            {
                'role': 'assistant',
                'content': completion,
            },
        ],
        'user_data': [('ground_truth', ground_truth)],
    }


def test_boxed_math_accuracy_reward_compares_numeric_answers():
    reward = BoxedMathAccuracyReward()

    assert reward([
        _trajectory('Reasoning. \\boxed{033}', '33'),
        _trajectory('Reasoning. \\boxed{32}', '33'),
        _trajectory('The answer is 33.', '33'),
    ]) == [1.0, 0.0, 0.0]


def test_boxed_math_accuracy_reward_handles_nested_latex():
    reward = BoxedMathAccuracyReward()

    assert reward([_trajectory('Reasoning. \\boxed{\\frac{1}{2}}', '\\frac{1}{2}')]) == [1.0]


def test_boxed_math_accuracy_reward_reports_accuracy_metric():
    reward = BoxedMathAccuracyReward()

    assert reward.metric_payload([], rewards=[1.0, 0.0, 1.0]) == {
        'accuracy_reward': 2 / 3,
    }
