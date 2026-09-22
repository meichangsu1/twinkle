"""DAPO input and reward regressions; no model or accelerator required."""

from cookbook.rl.grpo.dsv4_dapo import DAPOMathAccuracyReward, DAPOMathProcessor


def test_dapo_prompt_and_ground_truth():
    row = {
        'prompt': [{'role': 'user', 'content': 'Solve it.\nAnswer:'}],
        'reward_model': {'ground_truth': '34', 'style': 'rule-lighteval/MATH_v2'},
    }
    trajectory = DAPOMathProcessor().preprocess(row)
    assert trajectory['messages'] == row['prompt']
    assert trajectory['user_data'] == [('ground_truth', '34')]


def test_dapo_reward_only_uses_final_answer_line():
    reward = DAPOMathAccuracyReward()

    def trajectory(text):
        return {'messages': [{'role': 'assistant', 'content': text}], 'user_data': [('ground_truth', '34')]}

    assert reward([trajectory('Answer: 34'), trajectory('34 appears in reasoning only'),
                   trajectory('Answer: 12\nAnswer: 34'), trajectory('Answer: 34\nAnswer: 12')]) == [1.0, 0.0, 1.0, 0.0]
