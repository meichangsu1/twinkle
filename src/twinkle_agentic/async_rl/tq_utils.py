# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from numbers import Number
from typing import Any

from twinkle.data_format import InputFeature

from .data_plane import TransferQueueDataPlane
from .types import TransformersTrainBatch

TRANSFORMERS_INPUT_FIELDS = (
    'input_ids',
    'labels',
    'attention_mask',
    'position_ids',
    'cu_seqlens',
    'completion_mask',
    'pixel_values',
    'image_grid_thw',
    'video_pixel_values',
    'video_grid_thw',
    'input_features',
    'feature_attention_mask',
)
REQUIRED_MODEL_INPUT_FIELDS = ('input_ids', 'labels')
TRAIN_LOSS_FIELDS = ('logprobs', 'advantages', 'rewards')
REQUIRED_TRAIN_LOSS_FIELDS = ('logprobs', 'advantages')
REWARD_FIELD = 'rewards'
ROLLOUT_TRAIN_FIELDS = (*TRANSFORMERS_INPUT_FIELDS, 'logprobs', 'rewards', 'advantages', 'returns')


def rows_to_tq_fields(rows: list[dict[str, Any]]):
    from tensordict import TensorDict

    if not rows:
        return TensorDict({}, batch_size=[0])
    field_names = tuple(rows[0].keys())
    expected = set(field_names)
    for row_index, row in enumerate(rows):
        actual = set(row)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f'TQ row {row_index} fields mismatch: missing={missing}, extra={extra}')
    columns = {field_name: [row[field_name] for row in rows] for field_name in field_names}
    return columns_to_tq_fields(columns, len(rows))


def columns_to_tq_fields(columns: dict[str, list[Any]], size: int):
    import torch
    from tensordict import TensorDict
    from tensordict.tensorclass import NonTensorStack

    if size < 0:
        raise ValueError(f'TQ field size must be non-negative, got {size}')
    packed = {}
    for field_name, values in columns.items():
        if not isinstance(values, list):
            raise TypeError(f'TQ field {field_name!r} must be a list, got {type(values)!r}')
        if len(values) != size:
            raise ValueError(f'TQ field {field_name!r} must contain {size} values, got {len(values)}')
        if all(isinstance(item, Number) and not isinstance(item, bool) for item in values):
            packed[field_name] = torch.tensor(values)
        else:
            packed[field_name] = NonTensorStack(*values)
    return TensorDict(packed, batch_size=[size])


def read_train_batch(
    data_plane: TransferQueueDataPlane,
    batch: Any,
    data_fields: list[str] | None = None,
) -> TransformersTrainBatch:
    selected = _selected_train_fields(data_fields)
    columns = data_plane.read_batch_fields(batch, fields=selected)
    size = len(batch.keys)
    input_columns = {
        field_name: columns[field_name]
        for field_name in TRANSFORMERS_INPUT_FIELDS if field_name in columns
    }
    for field_name in REQUIRED_MODEL_INPUT_FIELDS:
        values = input_columns.get(field_name)
        if values is None or any(value is None for value in values):
            raise ValueError(f'TQ batch missing required model input field {field_name!r}')

    inputs: list[InputFeature] = []
    for index in range(size):
        inputs.append(
            InputFeature(**{
                field_name: values[index]
                for field_name, values in input_columns.items() if values[index] is not None
            }))

    return TransformersTrainBatch(
        inputs=inputs,
        logprobs=columns['logprobs'],
        advantages=columns['advantages'],
        rewards=columns[REWARD_FIELD],
        sample_keys=list(batch.keys),
    )


def _selected_train_fields(data_fields: list[str] | None) -> list[str]:
    if data_fields is None:
        return sorted(set(TRANSFORMERS_INPUT_FIELDS) | set(TRAIN_LOSS_FIELDS))
    return sorted(
        set(data_fields)
        | set(REQUIRED_MODEL_INPUT_FIELDS)
        | set(REQUIRED_TRAIN_LOSS_FIELDS)
        | {REWARD_FIELD})
