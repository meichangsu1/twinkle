# Copyright (c) ModelScope Contributors. All rights reserved.
"""DAPO-Math prompt and answer handling for the DSV4 GRPO example."""
import re

from twinkle.data_format import Message, Trajectory, user_data_get
from twinkle.reward.math_reward import MathReward


class DAPOMathProcessor:

    def preprocess(self, row):
        prompt = row['prompt']
        reward_model = row['reward_model']
        if not prompt or not isinstance(prompt, list):
            raise ValueError('DAPO row must have a nonempty prompt message list')
        if not isinstance(reward_model, dict) or not reward_model.get('ground_truth'):
            raise ValueError('DAPO row is missing reward_model.ground_truth')
        messages = [Message(role=item['role'], content=item['content']) for item in prompt]
        return Trajectory(messages=messages, user_data=[('ground_truth', reward_model['ground_truth'])])


class DAPOMathAccuracyReward:
    """Score only the final Answer: line, as requested by the DAPO prompt."""

    _answer_line = re.compile(r'^\s*Answer:\s*(.*?)\s*$', re.IGNORECASE | re.MULTILINE)

    def __call__(self, trajectories):
        rewards = []
        for trajectory in trajectories:
            completion = trajectory['messages'][-1]['content']
            truth = user_data_get(trajectory.get('user_data'), 'ground_truth')
            matches = self._answer_line.findall(completion)
            if not matches or not truth or not matches[-1].strip():
                rewards.append(0.0)
                continue
            predicted = MathReward.extract_boxed_result(matches[-1].strip())
            expected = MathReward.extract_boxed_result(str(truth).strip())
            correct = False
            try:
                from math_verify import parse, verify
                correct = bool(verify(parse(expected), parse(predicted)))
            except (ImportError, ValueError, TypeError):
                pass
            if not correct:
                try:
                    correct = bool(MathReward.compare_consecutive(predicted, expected))
                except (ValueError, TypeError):
                    pass
            rewards.append(float(correct))
        return rewards
