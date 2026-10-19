# Copyright (c) ModelScope Contributors. All rights reserved.
"""CPU regressions for reward text decoding; no vLLM engine or model needed."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from twinkle.data_format import SampledSequence, SampleResponse, SamplingMask, SamplingParams
from twinkle.sampler.vllm_sampler.vllm_sampler import vLLMSampler
from twinkle.template.base import Template


@pytest.fixture
def template():
    eos = '<｜end▁of▁sentence｜>'
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({'Answer:': 0, '7': 1, eos: 2, '<unk>': 3}, unk_token='<unk>')),
        eos_token=eos,
        unk_token='<unk>',
    )
    # Exercise the real Template decode/concat methods without downloading a model.
    result = object.__new__(Template)
    result.processor = tokenizer
    result.max_length = 32
    result.truncation_strategy = 'raise'
    return result


@pytest.mark.parametrize('tokens,reason,expected', [
    ([0, 1, 2], 'stop', 'Answer: 7'),
    ([0, 1], 'length', 'Answer: 7'),
    ([], 'stop', ''),
])
def test_reward_text_skips_eos_without_changing_training_tokens(template, tokens, reason, expected):
    logprobs = [[(token, -float(index + 1))] for index, token in enumerate(tokens)]
    mask = SamplingMask(token_ids=list(tokens), offsets=list(range(len(tokens) + 1)))
    sequence = SampledSequence(stop_reason=reason, tokens=tokens, logprobs=logprobs, sampling_mask=mask)
    response = SampleResponse(sequences=[sequence], prompt_token_ids=[1], prompt_logprobs=[None])
    engine = SimpleNamespace(sample=AsyncMock(return_value=response))
    sampler = SimpleNamespace(engine=engine, template=template)
    feature = dict(input_ids=[1], labels=[-100])
    original_feature, original_tokens, original_logprobs = deepcopy((feature, tokens, logprobs))

    actual = asyncio.run(vLLMSampler._sample_single(sampler, feature, SamplingParams(logprobs=1)))
    sampled = actual.sequences[0]

    assert sampled.decoded == expected
    assert sampled.tokens is tokens and tokens == original_tokens
    assert sampled.logprobs is logprobs and logprobs == original_logprobs
    assert len(sampled.tokens) == len(sampled.logprobs)
    assert sampled.sampling_mask is mask
    assert sampled.stop_reason == reason
    assert sampled.new_input_feature['input_ids'] == feature['input_ids'] + original_tokens
    assert sampled.new_input_feature['labels'] == original_tokens + [-100]
    assert feature == original_feature
    assert actual.prompt_token_ids == [1]
    assert actual.prompt_logprobs == [None]


def test_logprobs_only_does_not_decode_or_append_tokens(template):
    template.decode = Mock(side_effect=AssertionError('logprobs-only must not decode'))
    template.concat_input_feature = Mock(side_effect=AssertionError('logprobs-only must not append tokens'))
    response = SampleResponse(sequences=[SampledSequence(stop_reason='stop', tokens=[0, 1, 2])], prompt_token_ids=[1])
    sampler = SimpleNamespace(engine=SimpleNamespace(sample=AsyncMock(return_value=response)), template=template)
    feature = dict(input_ids=[1], labels=[-100])

    actual = asyncio.run(vLLMSampler._sample_single(sampler, feature, SamplingParams(), logprobs_only=True))

    assert actual.sequences[0].decoded is None
    assert actual.sequences[0].tokens == []
    assert actual.sequences[0].new_input_feature == feature
    template.decode.assert_not_called()
    template.concat_input_feature.assert_not_called()
