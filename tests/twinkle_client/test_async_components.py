# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import asyncio

from twinkle_client.types import DataRef


class _Response:

    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self):
        return self._payload


def test_model_forward_backward_sends_multiple_data_refs(monkeypatch) -> None:
    import twinkle_client.http as http_module
    from twinkle_client.model import multi_lora_transformers as module

    calls = []

    def post(*, url, json_data=None, **_kwargs):
        calls.append((url, json_data))
        if url.endswith('/create'):
            return _Response({})
        return _Response({'result': {'loss': 1.0}})

    monkeypatch.setattr(http_module, 'get_base_url', lambda: 'http://server/api/v1')
    monkeypatch.setattr(module, 'http_post', post)

    model = module.MultiLoraTransformersModel('ms://base')
    model.adapter_name = 'adapter'
    refs = [
        DataRef(ref_id='data-1', size=2, fields=['train_input']),
        DataRef(ref_id='data-2', size=2, fields=['train_input']),
    ]
    model.forward_backward(
        refs,
        input_field='train_input',
        kwarg_fields={'advantages': 'advantage'},
    )

    url, body = calls[-1]
    assert url.endswith('/model/base/twinkle/forward_backward')
    assert body['input_refs'] == [ref.model_dump() for ref in refs]
    assert body['input_field'] == 'train_input'
    assert body['kwarg_fields'] == {'advantages': 'advantage'}
    assert body['adapter_name'] == 'adapter'


def test_model_forward_accepts_data_ref_without_a_separate_submit_api(monkeypatch) -> None:
    import twinkle_client.http as http_module
    from twinkle_client.model import multi_lora_transformers as module

    calls = []

    def post(url, json_data=None, **_kwargs):
        calls.append((url, json_data))
        return _Response({} if url.endswith('/create') else {'result': {'value': 1}})

    monkeypatch.setattr(http_module, 'get_base_url', lambda: 'http://server/api/v1')
    monkeypatch.setattr(module, 'http_post', post)

    model = module.MultiLoraTransformersModel('ms://base')
    ref = DataRef(ref_id='data-1', size=2, fields=['train_input'])
    model.forward(ref, input_field='train_input')

    url, body = calls[-1]
    assert url.endswith('/model/base/twinkle/forward')
    assert body['input_refs'] == [ref.model_dump()]
    assert body['input_field'] == 'train_input'


def test_sampler_async_data_plane_path_returns_reference_without_materializing(monkeypatch) -> None:
    import twinkle_client.http as http_module
    from twinkle_client.sampler import vllm_sampler as module

    output_ref = DataRef(
        ref_id='rollout-1',
        size=4,
        fields=['train_input', 'sampled_logprobs', 'decoded'],
        kind='rollout',
    )
    calls = []

    def post(*, url, json_data=None, **_kwargs):
        calls.append((url, json_data))
        if url.endswith('/create'):
            return _Response({})
        return _Response(output_ref.model_dump())

    monkeypatch.setattr(http_module, 'get_base_url', lambda: 'http://server/api/v1')
    monkeypatch.setattr(module, 'http_post', post)
    sampler = module.vLLMSampler('ms://base')

    result = asyncio.run(sampler.asample_to_data_plane(
        [{'input_ids': [1]}],
        num_samples=4,
        group_ids=['group-1'],
    ))

    assert result == output_ref
    url, body = calls[-1]
    assert url.endswith('/sampler/base/twinkle/sample_to_data_plane')
    assert body['num_samples'] == 4
    assert body['group_ids'] == ['group-1']
