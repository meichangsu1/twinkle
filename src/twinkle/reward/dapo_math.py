import re
from typing import Any, Dict, List

from twinkle.data_format import user_data_get
from twinkle.reward.base import Reward
from .gsm8k import _extract_last_boxed


class DAPOMathAccuracyReward(Reward):
    """Accuracy reward for DAPO/verl math rows.

    DAPO math prompts usually require the final answer to be written as
    ``Answer: ...``. For compatibility with existing math prompts, boxed and
    GSM8K-style ``####`` answers are accepted as fallbacks.
    """

    @staticmethod
    def extract_answer(completion: str) -> str:
        text = completion[-1000:] if len(completion) > 1000 else completion
        matches = re.findall(r'(?im)^\s*Answer\s*:\s*(.+?)\s*$', text)
        if matches:
            return _normalize_answer(matches[-1])
        inline = re.findall(r'(?i)Answer\s*:\s*([^\n]+)', text)
        if inline:
            return _normalize_answer(inline[-1])
        boxed = _extract_last_boxed(text)
        if boxed:
            return _normalize_answer(boxed)
        gsm8k = re.findall(r'####\s*([^\n]+)', text)
        if gsm8k:
            return _normalize_answer(gsm8k[-1])
        return ''

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for trajectory in trajectories:
            completion = _last_assistant_content(trajectory)
            ground_truth = user_data_get(trajectory.get('user_data'), 'ground_truth', '') or ''
            predicted = self.extract_answer(completion)
            expected = _normalize_answer(str(ground_truth))
            rewards.append(1.0 if _answers_equal(predicted, expected) else 0.0)
        return rewards


def _last_assistant_content(trajectory: Dict[str, Any]) -> str:
    for message in reversed(trajectory.get('messages') or []):
        if message.get('role') == 'assistant':
            return str(message.get('content') or '')
    return ''


def _answers_equal(predicted: str, expected: str) -> bool:
    if not predicted or not expected:
        return False
    try:
        return abs(float(predicted) - float(expected)) < 1e-5
    except (TypeError, ValueError, OverflowError):
        return predicted == expected


def _normalize_answer(value: str) -> str:
    value = str(value).strip()
    value = value.strip('`').strip()
    value = value.strip('$').strip()
    value = re.sub(r'\\boxed\s*\{(.+)\}', r'\1', value)
    value = value.replace(',', '')
    value = value.replace(' ', '')
    trailing = re.match(r'^(.+?)(?:[.;。]|</?answer>)?$', value, flags=re.IGNORECASE)
    return trailing.group(1).strip() if trailing else value
