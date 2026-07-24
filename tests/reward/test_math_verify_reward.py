import sys
from types import ModuleType

from twinkle.reward import MathVerifyAccuracyReward


def test_math_verify_reward_uses_complete_completion_and_ground_truth(monkeypatch):
    calls = []

    class ExprExtractionConfig:
        def __init__(self, *, try_extract_without_anchor):
            self.try_extract_without_anchor = try_extract_without_anchor

    class LatexExtractionConfig:
        pass

    def parse(value, *, extraction_config, parsing_timeout):
        calls.append((
            'parse',
            value,
            tuple(type(config).__name__ for config in extraction_config),
            extraction_config[0].try_extract_without_anchor,
            parsing_timeout,
        ))
        return f'parsed:{value}'

    def verify(gold, answer, *, float_rounding, timeout_seconds):
        calls.append(('verify', gold, answer, float_rounding, timeout_seconds))
        return True

    parser_module = ModuleType('math_verify.parser')
    parser_module.ExprExtractionConfig = ExprExtractionConfig
    parser_module.LatexExtractionConfig = LatexExtractionConfig
    parser_module.parse = parse
    grader_module = ModuleType('math_verify.grader')
    grader_module.verify = verify
    monkeypatch.setitem(sys.modules, 'math_verify.parser', parser_module)
    monkeypatch.setitem(sys.modules, 'math_verify.grader', grader_module)
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
        (
            'parse',
            'reference reasoning\n#### 33',
            ('ExprExtractionConfig', 'LatexExtractionConfig'),
            True,
            None,
        ),
        (
            'parse',
            'reasoning \\boxed{33}',
            ('ExprExtractionConfig', 'LatexExtractionConfig'),
            True,
            None,
        ),
        (
            'verify',
            'parsed:reference reasoning\n#### 33',
            'parsed:reasoning \\boxed{33}',
            6,
            None,
        ),
    ]
