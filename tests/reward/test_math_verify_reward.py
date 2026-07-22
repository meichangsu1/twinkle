import sys
from types import SimpleNamespace

from twinkle.reward import MathVerifyAccuracyReward


def test_math_verify_reward_uses_complete_completion_and_ground_truth(monkeypatch):
    calls = []

    def parse(value):
        calls.append(('parse', value))
        return f'parsed:{value}'

    def verify(answer, gold):
        calls.append(('verify', answer, gold))
        return True

    monkeypatch.setitem(sys.modules, 'math_verify', SimpleNamespace(parse=parse, verify=verify))
    trajectory = {
        'messages': [
            {'role': 'user', 'content': 'question'},
            {'role': 'assistant', 'content': 'reasoning \\boxed{33}'},
        ],
        'user_data': [('ground_truth', 'reference reasoning\n#### 33')],
    }

    reward = MathVerifyAccuracyReward()
    assert reward([trajectory]) == [1.0]
    assert calls == [
        ('parse', 'reasoning \\boxed{33}'),
        ('parse', 'reference reasoning\n#### 33'),
        ('verify', 'parsed:reasoning \\boxed{33}', 'parsed:reference reasoning\n#### 33'),
    ]
