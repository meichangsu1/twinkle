from __future__ import annotations

import pytest

import twinkle_client.types as types
from twinkle.server.sampler.twinkle_handlers import _sample_models_to_rows


def _response(tokens: list[int]) -> types.SampleResponseModel:
    return types.SampleResponseModel(
        sequences=[
            types.SampledSequenceModel(
                stop_reason='stop',
                tokens=[token],
                logprobs=[[(token, -0.1)]],
                new_input_feature={'input_ids': [token], 'labels': [token]},
            )
            for token in tokens
        ],
        prompt_logprobs=[-0.2],
    )


def test_async_sampler_flattens_generations_to_tagged_tq_rows() -> None:
    rows, tags = _sample_models_to_rows(
        [_response([10, 11]), _response([20, 21])],
        group_ids=['group-a', 'group-b'],
        policy_version=7,
        adapter_uri='twinkle://policy-7',
    )

    assert [row['tokens'] for row in rows] == [[10], [11], [20], [21]]
    assert [(tag['group_id'], tag['generation_idx']) for tag in tags] == [
        ('group-a', 0),
        ('group-a', 1),
        ('group-b', 0),
        ('group-b', 1),
    ]
    assert {tag['rollout_policy_version'] for tag in tags} == {7}
    assert {tag['rollout_adapter_uri'] for tag in tags} == {'twinkle://policy-7'}


def test_async_sampler_rejects_group_id_count_mismatch() -> None:
    with pytest.raises(ValueError, match='group_ids contains 1 values for 2'):
        _sample_models_to_rows(
            [_response([10]), _response([20])],
            group_ids=['only-one'],
            policy_version=0,
            adapter_uri=None,
        )
