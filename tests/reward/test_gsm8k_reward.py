from twinkle.reward import GSM8KAccuracyBrevityReward, GSM8KBrevityReward


def _trajectory(completion: str, ground_truth: str = '42') -> dict:
    return {
        'messages': [{'role': 'assistant', 'content': completion}],
        'user_data': [('ground_truth', ground_truth)],
    }


def test_gsm8k_brevity_reward_requires_answer_and_decays_with_length():
    reward = GSM8KBrevityReward(full_reward_length=20, decay_length=100)

    values = reward([
        _trajectory('short \\boxed{42}'),
        _trajectory('x' * 30 + ' \\boxed{42}'),
        _trajectory('no final answer'),
    ])

    assert values == [1.0, 0.79, 0.0]


def test_gsm8k_accuracy_brevity_reward_sums_components():
    reward = GSM8KAccuracyBrevityReward()
    trajectories = [
        _trajectory('brief \\boxed{42}'),
        _trajectory('brief \\boxed{7}'),
    ]

    values = reward(trajectories)
    metrics = reward.metric_payload(trajectories, rewards=values)

    assert values == [2.0, 1.0]
    assert metrics == {
        'total_reward': 1.5,
        'accuracy_reward': 0.5,
        'brevity_reward': 1.0,
    }
