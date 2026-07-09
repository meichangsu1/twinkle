# Copyright (c) ModelScope Contributors. All rights reserved.
import re
from typing import Any, Dict, List

from twinkle.data_format import Message, Trajectory
from .base import Preprocessor


class CompetitionMathProcessor(Preprocessor):

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        rows = self.map_row_to_col(rows)
        return rows

    def preprocess(self, row) -> Dict[str, Any]:
        problem = row['problem']
        solution = row['solution']
        messages = [
            Message(role='user', content=problem),
            Message(role='assistant', content=solution),
        ]
        return Trajectory(messages=messages)


class CompetitionMathGRPOProcessor(Preprocessor):

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        rows = self.map_row_to_col(rows)
        return rows

    def preprocess(self, row) -> Trajectory:
        problem = row['problem']
        solution = row['solution']
        messages = [
            Message(
                role='system',
                content='You are a helpful math assistant. Respond with only the final answer in the form '
                '\\boxed{...} and nothing else.'),
            Message(role='user', content=problem),
            Message(role='assistant', content=''),
        ]
        return Trajectory(messages=messages, user_data=[('solution', solution)])


class SelfCognitionProcessor(Preprocessor):

    def __init__(self, model_name, model_author):
        self.model_name = model_name
        self.model_author = model_author

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        rows = self.map_row_to_col(rows)
        return rows

    def preprocess(self, row) -> Trajectory:
        problem = row['query'].replace('{{NAME}}', self.model_name).replace('{{AUTHOR}}', self.model_author)
        solution = row['response'].replace('{{NAME}}', self.model_name).replace('{{AUTHOR}}', self.model_author)
        messages = [
            Message(role='system', content='You are a helpful assistant.'),
            Message(role='user', content=problem),
            Message(role='assistant', content=solution),
        ]
        return Trajectory(messages=messages)


class AlpacaProcessor(Preprocessor):

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        rows = self.map_row_to_col(rows)
        return rows

    def preprocess(self, row) -> Trajectory:
        instruction = row.get('instruction') or ''
        input_text = row.get('input') or ''
        output_text = row.get('output') or ''
        prompt = instruction if not input_text else f'{instruction}\n{input_text}'
        messages = [
            Message(role='user', content=prompt),
            Message(role='assistant', content=output_text),
        ]
        return Trajectory(messages=messages)


class CountdownProcessor(Preprocessor):
    system_prompt = ('You are a helpful assistant. You first thinks about the reasoning process '
                     'in the mind and then provides the user with the answer.')

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        rows = self.map_row_to_col(rows)
        return rows

    def preprocess(self, row) -> Trajectory:
        nums = row.get('nums', [])
        target = row.get('response', row.get('target', 0))

        query = f"""Using the numbers {nums}, create an equation that equals {target}.
You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.
Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags,
for example <answer> (1 + 2) / 3 * 4 = 4 </answer>."""

        messages = [
            Message(role='system', content=self.system_prompt),
            Message(role='user', content=query),
        ]
        return Trajectory(messages=messages, user_data=[{'target': target, 'nums': nums}])


class GSM8KProcessor(Preprocessor):
    """Preprocessor for GSM8K dataset (prompt-only, for on-policy generation).

    GSM8K fields: question (str), answer (str ending with '#### <number>')
    Extracts the ground truth number and stores it in user_data for reward.
    Only includes system + user messages; assistant response is generated on-policy.
    """
    system_prompt = ('You are a helpful math assistant. Solve the problem step by step '
                     'and put your final answer within \\boxed{}.')

    def __init__(self, system=None, add_assistant=False):
        self.system = system
        if self.system is None:
            self.system = self.system_prompt
        self.add_assistant = add_assistant

    def extract_ground_truth(self, answer_str: str) -> str:
        """Extract the number after '####' from GSM8K answer."""
        match = re.search(r'####\s*([\-\d,.]+)', answer_str)
        if match:
            return match.group(1).replace(',', '').strip()
        return ''

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        rows = self.map_row_to_col(rows)
        return rows

    def preprocess(self, row) -> Trajectory:
        question = row['question']
        answer = row.get('answer', '')
        ground_truth = self.extract_ground_truth(answer)

        messages = [
            Message(role='system', content=self.system),
            Message(role='user', content=question),
        ]
        if self.add_assistant:
            messages.append(Message(role='assistant', content=answer))
        return Trajectory(
            messages=messages,
            user_data=[('ground_truth', ground_truth)],
        )


class DAPOMathProcessor(Preprocessor):
    """Preprocessor for DAPO/verl math parquet rows.

    Expected fields:
      prompt: list[{role, content}]
      reward_model: {ground_truth, style}
      data_source / ability / extra_info: optional metadata
    """

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        return self.map_row_to_col(rows)

    def preprocess(self, row) -> Trajectory:
        prompt = row.get('prompt') or []
        if not isinstance(prompt, list):
            raise TypeError(f'DAPOMathProcessor expected prompt list, got {type(prompt)!r}')
        messages = []
        for message in prompt:
            if not isinstance(message, dict):
                raise TypeError(f'DAPOMathProcessor expected prompt message dict, got {type(message)!r}')
            role = message.get('role')
            content = message.get('content')
            if role is None or content is None:
                raise ValueError(f'DAPOMathProcessor prompt message missing role/content: {message!r}')
            messages.append(Message(role=role, content=content))

        reward_model = row.get('reward_model') or {}
        if not isinstance(reward_model, dict):
            raise TypeError(f'DAPOMathProcessor expected reward_model dict, got {type(reward_model)!r}')
        ground_truth = str(reward_model.get('ground_truth', '')).strip()
        user_data = [
            ('ground_truth', ground_truth),
            ('reward_style', str(reward_model.get('style', ''))),
            ('data_source', str(row.get('data_source', ''))),
            ('ability', str(row.get('ability', ''))),
        ]
        extra_info = row.get('extra_info') or {}
        if isinstance(extra_info, dict) and extra_info.get('index') is not None:
            user_data.append(('index', str(extra_info['index'])))
        return Trajectory(messages=messages, user_data=user_data)


class AIMEProcessor(Preprocessor):
    """Preprocessor for original AIME rows.

    Expected fields:
      ID: problem id
      Problem: math problem text
      Solution: optional reference solution
      Answer: final answer
    """
    prompt_template = (
        'Solve the following math problem step by step. The last line of your response should be of the form '
        'Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n'
        '{problem}\n\n'
        'Remember to put your answer on its own line after "Answer:".')

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        return self.map_row_to_col(rows)

    def preprocess(self, row) -> Trajectory:
        missing = [field for field in ('ID', 'Problem', 'Answer') if row.get(field) is None]
        if missing:
            keys = sorted(str(key) for key in row.keys())
            raise KeyError(f'AIMEProcessor expected ID/Problem/Answer fields, missing {missing}; got fields: {keys}')

        problem = str(row['Problem']).strip()
        ground_truth = str(row['Answer']).strip()
        user_data = [
            ('ground_truth', ground_truth),
            ('data_source', 'aime'),
            ('ability', 'MATH'),
        ]
        user_data.append(('id', str(row['ID'])))
        if row.get('Solution') is not None:
            user_data.append(('solution', str(row['Solution'])))

        return Trajectory(
            messages=[Message(role='user', content=self.prompt_template.format(problem=problem))],
            user_data=user_data,
        )
