from twinkle.preprocessor import AIME2024Processor, AReaLGSM8KProcessor, DAPOMathProcessor


def test_dapo_math_processor_preserves_prompt_and_ground_truth():
    row = {
        'prompt': [{
            'role': 'user',
            'content': 'Solve the problem and put the answer in a box.',
        }],
        'reward_model': {
            'ground_truth': '34',
            'style': 'rule-lighteval/MATH_v2',
        },
    }

    trajectory = DAPOMathProcessor().preprocess(row)

    assert trajectory['messages'] == row['prompt']
    assert trajectory['user_data'] == [('ground_truth', '34')]


def test_aime2024_processor_uses_maxwell_schema():
    trajectory = AIME2024Processor().preprocess({
        'Problem': 'What is 16 + 17?',
        'Answer': 33,
    })

    assert trajectory['messages'] == [{
        'role': 'user',
        'content': ('What is 16 + 17?\n'
                    'The answer format must be: \\boxed{The final answer goes here.}'),
    }]
    assert trajectory['user_data'] == [('ground_truth', '33')]


def test_areal_gsm8k_processor_matches_user_only_prompt():
    trajectory = AReaLGSM8KProcessor().preprocess({
        'question': 'What is 16 + 17?',
        'answer': 'Work.\n#### 33',
    })

    assert trajectory['messages'] == [{
        'role': 'user',
        'content': 'What is 16 + 17?\nPlease put your final answer within \\boxed{}.',
    }]
    assert trajectory['user_data'] == [('ground_truth', 'Work.\n#### 33')]
