# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import asyncio

import pytest

from twinkle_client.types import ComponentTaskRef, DataRef, DataRowsResponse


class _Response:

    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self):
        return self._payload


def test_remote_task_uses_existing_future_endpoint(monkeypatch) -> None:
    from twinkle_client import remote_task as module

    responses = iter([_Response({'type': 'try_again'}), _Response({'value': 7})])
    calls = []

    def post(url, json_data, **kwargs):
        calls.append((url, json_data, kwargs))
        return next(responses)

    monkeypatch.setattr(module, 'http_post', post)
    monkeypatch.setattr(module, 'get_base_url', lambda: 'http://server/api/v1')

    task = module.RemoteTask(ComponentTaskRef(request_id='req-1', model_id='adapter'))
    assert task.result(timeout=1) == {'value': 7}
    assert [call[:2] for call in calls] == [
        ('http://server/api/v1/retrieve_future', {'request_id': 'req-1'}),
        ('http://server/api/v1/retrieve_future', {'request_id': 'req-1'}),
    ]
    assert all(0 < call[2]['timeout'] <= 1 for call in calls)


def test_remote_task_async_result_uses_non_blocking_http_polling(monkeypatch) -> None:
    import httpx
    from twinkle_client import remote_task as module

    responses = iter([_Response({'type': 'try_again'}), _Response({'value': 9})])
    calls = []

    class AsyncClient:

        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url, *, headers, json, timeout):
            calls.append((url, headers, json, timeout))
            return next(responses)

    monkeypatch.setattr(httpx, 'AsyncClient', AsyncClient)
    monkeypatch.setattr(module, 'get_base_url', lambda: 'http://server/api/v1')
    monkeypatch.setattr(module, '_build_headers', lambda: {'Authorization': 'Bearer token'})

    task = module.RemoteTask(ComponentTaskRef(request_id='req-async', model_id='adapter'))
    assert asyncio.run(task.aresult(timeout=1)) == {'value': 9}
    assert [call[2] for call in calls] == [
        {'request_id': 'req-async'},
        {'request_id': 'req-async'},
    ]
    assert all(0 < call[3] <= 1 for call in calls)


def test_remote_task_sync_timeout_bounds_long_poll(monkeypatch) -> None:
    import requests
    from twinkle_client import remote_task as module

    observed = []

    def post(_url, *, json_data, timeout):
        observed.append((json_data, timeout))
        raise requests.Timeout('long poll exceeded client deadline')

    monkeypatch.setattr(module, 'http_post', post)
    monkeypatch.setattr(module, 'get_base_url', lambda: 'http://server/api/v1')

    task = module.RemoteTask('req-timeout')
    with pytest.raises(TimeoutError, match='within 0.1s'):
        task.result(timeout=0.1)
    assert observed[0][0] == {'request_id': 'req-timeout'}
    assert 0 < observed[0][1] <= 0.1


def test_model_component_submits_data_ref_without_control_plane(monkeypatch) -> None:
    import twinkle_client.http as http_module
    from twinkle_client.model import multi_lora_transformers as module

    calls = []

    def post(*, url, json_data=None, **_kwargs):
        calls.append((url, json_data))
        if url.endswith('/create'):
            return _Response({})
        return _Response({'request_id': 'req-model', 'model_id': 'session-adapter'})

    monkeypatch.setattr(http_module, 'get_base_url', lambda: 'http://server/api/v1')
    monkeypatch.setattr(module, 'http_post', post)
    monkeypatch.setattr('twinkle_client.remote_task.get_base_url', lambda: 'http://server/api/v1')

    model = module.MultiLoraTransformersModel('ms://base')
    model.adapter_name = 'adapter'
    ref = DataRef(ref_id='data-1', size=4, fields=['input_ids'])
    task = model.submit_forward_backward(ref, advantages=[1, -1, 1, -1])

    assert task.request_id == 'req-model'
    url, body = calls[-1]
    assert url.endswith('/model/base/twinkle/submit_forward_backward')
    assert body['input_ref'] == ref.model_dump()
    assert body['adapter_name'] == 'adapter'
    assert body['kwargs']['advantages'] == [1, -1, 1, -1]


def test_sampler_component_fetches_and_releases_data_plane_output(monkeypatch) -> None:
    import twinkle_client.http as http_module
    from twinkle_client.sampler import vllm_sampler as module

    monkeypatch.setattr(http_module, 'get_base_url', lambda: 'http://server/api/v1')
    monkeypatch.setattr(module, 'http_post', lambda **_kwargs: _Response({}))
    sampler = module.vLLMSampler('ms://base')

    output_ref = DataRef(ref_id='rollout-1', size=1, fields=['tokens'], kind='rollout')

    class _DoneTask:

        async def aresult(self):
            return {'output_ref': output_ref.model_dump()}

    released = []
    monkeypatch.setattr(sampler, 'submit_sample', lambda *_args, **_kwargs: _DoneTask())
    monkeypatch.setattr(
        sampler.data_plane,
        'get_batch',
        lambda ref: DataRowsResponse(
                rows=[{
                    'tokens': [],
                    'stop_reason': 'stop',
                }],
            tags=[{'prompt_index': 0, 'generation_idx': 0}],
        ),
    )
    monkeypatch.setattr(sampler.data_plane, 'release', lambda ref: released.append(ref))

    responses = asyncio.run(sampler.asample([{'input_ids': [1]}]))
    assert len(responses) == 1
    assert len(responses[0].sequences) == 1
    assert responses[0].sequences[0].tokens == []
    assert released == [output_ref]
