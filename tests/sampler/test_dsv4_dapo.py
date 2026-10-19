"""DAPO input and reward regressions; no model or accelerator required."""
import sys
from types import SimpleNamespace

import pytest

from cookbook.rl.grpo.dsv4_dapo import DAPOMathAccuracyReward, DAPOMathProcessor
from twinkle.reward.math_reward import MathReward


def trajectory(text, truth='34'):
    return {'messages': [{'role': 'assistant', 'content': text}], 'user_data': [('ground_truth', truth)]}


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

    assert reward([trajectory('Answer: 34'), trajectory('34 appears in reasoning only'),
                   trajectory('Answer: 12\nAnswer: 34'), trajectory('Answer: 34\nAnswer: 12')]) == [1.0, 0.0, 1.0, 0.0]


@pytest.mark.parametrize('text,truth', [
    ('Answer: $34$', '34'),
    ('Answer: 34.', '34'),
    (r'Answer: \(34\)', '34'),
    (r'Answer: \[34\]', '34'),
    ('Answer: $$34$$', '34'),
    ('Answer: 34.0', '34'),
    ('Answer: 3.4e1', '34'),
    ('Answer: 1,234', '1234'),
    ('Answer: −34', '-34'),
    ('Answer: **34**', '34'),
    ('**Answer:** 34', '34'),
    ('**Answer: 34**', '34'),
    ('Answer：34', '34'),
    (r'\boxed{34}', '34'),
    (r'Answer: $\boxed{34}$', '34'),
    ('Answer: 7<｜end▁of▁sentence｜>', '7'),
    ('Answer: 7<｜end▁of▁sentence｜>\nAnswer: 12', '7'),
    ('Answer: 0', '0'),
    ('Answer: 0', 0),
    ('Answer: -0.0', '0'),
    ('Answer: 9007199254740993', '9007199254740993'),
])
def test_numeric_answers_do_not_require_symbolic_parsers(monkeypatch, text, truth):
    monkeypatch.setitem(sys.modules, 'math_verify', None)
    monkeypatch.setattr(MathReward, 'compare_consecutive', lambda *args: pytest.fail('Numeric grading needs no parser'))
    assert DAPOMathAccuracyReward()([trajectory(text, truth)]) == [1.0]


@pytest.mark.parametrize('text,truth', [
    ('Answer: 35', '34'),
    ('Answer: -34', '34'),
    ('Answer: 0', '7'),
    ('Answer: 9007199254740992', '9007199254740993'),
    ('Answer: NaN', 'NaN'),
    ('Answer: Infinity', 'Infinity'),
])
def test_numeric_mismatches_do_not_fall_back_to_permissive_parsing(monkeypatch, text, truth):
    def unexpected_parser(*args):
        pytest.fail('A numeric mismatch must not be reinterpreted by a symbolic parser')

    monkeypatch.setitem(sys.modules, 'math_verify', SimpleNamespace(parse=unexpected_parser))
    monkeypatch.setattr(MathReward, 'compare_consecutive', unexpected_parser)
    assert DAPOMathAccuracyReward()([trajectory(text, truth)]) == [0.0]


@pytest.mark.parametrize('text', [
    '34 appears in reasoning only',
    'Answer:',
    'Answer: 34\nAnswer:',
    r'Answer: \boxed{34',
])
def test_no_complete_final_answer_scores_zero(text):
    assert DAPOMathAccuracyReward()([trajectory(text)]) == [0.0]


def test_explicit_final_answer_takes_priority_over_boxed_reasoning():
    assert DAPOMathAccuracyReward()([trajectory(r'Intermediate: \boxed{34}' + '\nAnswer: 12')]) == [0.0]


def test_symbolic_answers_still_use_math_verify(monkeypatch):
    def parse(text):
        return [{'2': 2, r'\sqrt{4}': 2}[text]]

    monkeypatch.setitem(sys.modules, 'math_verify', SimpleNamespace(parse=parse, verify=lambda gold, pred: gold == pred))
    assert DAPOMathAccuracyReward()([trajectory(r'Answer: \sqrt{4}', '2')]) == [1.0]


def test_numeric_zero_ground_truth_is_preserved_by_processor():
    row = {'prompt': [{'role': 'user', 'content': 'Zero?'}], 'reward_model': {'ground_truth': 0}}
    actual = DAPOMathProcessor().preprocess(row)
    assert actual['user_data'] == [('ground_truth', 0)]


@pytest.mark.parametrize('truth', [None, ''])
def test_processor_rejects_missing_ground_truth(truth):
    row = {'prompt': [{'role': 'user', 'content': 'Question'}], 'reward_model': {'ground_truth': truth}}
    with pytest.raises(ValueError, match='ground_truth'):
        DAPOMathProcessor().preprocess(row)
