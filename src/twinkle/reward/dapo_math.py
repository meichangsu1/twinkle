# Copyright (c) ModelScope Contributors. All rights reserved.
"""Accuracy reward for the Answer-style format used by DAPO-Math."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from twinkle.data_format import Trajectory, user_data_get
from twinkle.reward.base import Reward
from twinkle.reward.math_reward import MathReward


_ANSWER_LINE = re.compile(r'^\s*Answer\s*:\s*(.+?)\s*$', re.IGNORECASE | re.MULTILINE)


class DAPOMathAccuracyReward(Reward):
    """Compare the final ``Answer:`` or boxed result with the ground truth."""

    def __call__(self, trajectories: list[Trajectory], **kwargs: Any) -> list[float]:
        return [self._score(trajectory) for trajectory in trajectories]

    def metric_payload(
        self,
        trajectories: list[Trajectory],
        *,
        rewards: list[float],
        **kwargs: Any,
    ) -> dict[str, float]:
        return {'accuracy_reward': sum(rewards) / len(rewards)}

    @classmethod
    def _score(cls, trajectory: Trajectory) -> float:
        completion = cls._last_assistant_content(trajectory)
        prediction = cls.extract_answer(completion)
        ground_truth = str(user_data_get(trajectory.get('user_data'), 'ground_truth', '')).strip()
        if not prediction or not ground_truth:
            return 0.0
        if cls._decimal_equal(prediction, ground_truth):
            return 1.0
        return float(MathReward.compare_consecutive(prediction, ground_truth))

    @staticmethod
    def extract_answer(completion: str) -> str:
        if '\\boxed{' in completion:
            return MathReward.extract_boxed_result(completion).strip()
        matches = _ANSWER_LINE.findall(completion)
        if not matches:
            return ''
        answer = matches[-1].strip().rstrip('.')
        if len(answer) >= 2 and answer.startswith('$') and answer.endswith('$'):
            answer = answer[1:-1].strip()
        return answer

    @staticmethod
    def _last_assistant_content(trajectory: Trajectory) -> str:
        for message in reversed(trajectory.get('messages', [])):
            if message.get('role') == 'assistant':
                return str(message.get('content', ''))
        return ''

    @staticmethod
    def _decimal_equal(first: str, second: str) -> bool:
        try:
            return Decimal(first.replace(',', '')) == Decimal(second.replace(',', ''))
        except InvalidOperation:
            return False
