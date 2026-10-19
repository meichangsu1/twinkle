# Copyright (c) ModelScope Contributors. All rights reserved.
"""DAPO-Math prompt and answer handling for the DSV4 GRPO example."""
import re
from decimal import Decimal, InvalidOperation

from twinkle.data_format import Message, Trajectory, user_data_get
from twinkle.reward.math_reward import MathReward


class DAPOMathProcessor:

    def preprocess(self, row):
        prompt = row['prompt']
        reward_model = row['reward_model']
        if not prompt or not isinstance(prompt, list):
            raise ValueError('DAPO row must have a nonempty prompt message list')
        if (not isinstance(reward_model, dict) or reward_model.get('ground_truth') is None
                or not str(reward_model['ground_truth']).strip()):
            raise ValueError('DAPO row is missing reward_model.ground_truth')
        messages = [Message(role=item['role'], content=item['content']) for item in prompt]
        return Trajectory(messages=messages, user_data=[('ground_truth', reward_model['ground_truth'])])


class DAPOMathAccuracyReward:
    """Compare the final Answer: line, or a boxed answer, with the ground truth."""

    _answer_line = re.compile(
        r'^[ \t]*(?:\*\*)?Answer[ \t]*(?:\*\*)?[:：][ \t]*(.*?)[ \t]*$', re.IGNORECASE | re.MULTILINE)

    @staticmethod
    def _normalize_answer(answer):
        answer = answer.strip().strip('*').strip().removesuffix('.').strip()
        if '\\boxed{' in answer:
            boxed = MathReward.extract_boxed_result(answer)
            if boxed == answer:  # Incomplete boxed expression, not a final answer.
                return ''
            answer = boxed.strip()
        for left, right in [('$$', '$$'), ('$', '$'), (r'\(', r'\)'), (r'\[', r'\]')]:
            if answer.startswith(left) and answer.endswith(right) and len(answer) >= len(left) + len(right):
                answer = answer[len(left):-len(right)].strip()
        answer = answer.replace('−', '-')
        if re.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?', answer):
            answer = answer.replace(',', '')
        return answer

    @classmethod
    def extract_answer(cls, completion):
        # Also support regrading old reports that retained the actual EOS marker.
        completion = completion.split('<｜end▁of▁sentence｜>', 1)[0]
        matches = cls._answer_line.findall(completion)
        if matches:
            return cls._normalize_answer(matches[-1])
        if '\\boxed{' in completion:
            return cls._normalize_answer(MathReward.extract_boxed_result(completion))
        return ''

    def __call__(self, trajectories):
        rewards = []
        for trajectory in trajectories:
            completion = trajectory['messages'][-1]['content']
            truth = user_data_get(trajectory.get('user_data'), 'ground_truth')
            predicted = self.extract_answer(completion)
            expected = self._normalize_answer(str(truth)) if truth is not None else ''
            if not predicted or not expected:
                rewards.append(0.0)
                continue
            # Decimal compares complete numeric answers exactly, without float
            # rounding or optional symbolic-parser dependencies. A mismatch is
            # final: do not let a parser reinterpret a wrong numeric answer.
            try:
                first, second = Decimal(predicted), Decimal(expected)
            except InvalidOperation:
                pass
            else:
                rewards.append(float(first.is_finite() and second.is_finite() and first == second))
                continue
            if predicted == expected:
                rewards.append(1.0)
                continue
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
