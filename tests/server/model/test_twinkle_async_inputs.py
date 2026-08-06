from __future__ import annotations

import pytest
from fastapi import FastAPI
from starlette.requests import Request

import twinkle_client.types as types
from twinkle.server.model.twinkle_handlers import (
    _model_result_rows,
    _register_twinkle_routes,
)


def test_model_result_rows_keeps_one_output_row_per_sample() -> None:
    assert _model_result_rows(
        {'logps': [[-1.0], [-2.0]], 'loss': 0.25},
        batch_size=2,
    ) == [
        {'logps': [-1.0], 'loss': 0.25},
        {'logps': [-2.0], 'loss': 0.25},
    ]


class _SchedulingManagement:

    def __init__(self):
        self.data_world_size = 2
        self.scheduled = []
        self.model_calls = []
        self.model = self

    async def _on_request_start(self, _request):
        return 'token'

    def assert_resource_exists(self, _adapter_name):
        return None

    def forward_backward(self, *, inputs, adapter_name, **kwargs):
        self.model_calls.append((inputs, adapter_name, kwargs))
        return {'loss': 1.0}

    async def schedule_task(self, task, **kwargs):
        self.scheduled.append(kwargs)
        await task()
        return {'request_id': 'request-1', 'model_id': kwargs.get('model_id')}


@pytest.mark.asyncio
async def test_async_forward_backward_schedules_without_algorithm_metadata() -> None:
    management = _SchedulingManagement()
    app = FastAPI()
    _register_twinkle_routes(app, lambda: management)
    route = next(route for route in app.routes if getattr(route, 'path', None) == '/twinkle/submit_forward_backward')
    request = Request({'type': 'http', 'headers': []})
    request.state.session_id = 'session'
    body = types.AsyncForwardBackwardRequest(
        adapter_name='adapter',
        inputs=[{'input_ids': [index]} for index in range(8)],
        kwargs={
            'old_logps': [[-0.1]] * 8,
            'advantages': [1.0] * 8,
        },
    )

    await route.endpoint(request, body, management)

    assert management.scheduled[-1]['batch_size'] == 8
    assert management.scheduled[-1]['data_world_size'] == 2
    assert 'batch_size_multiple' not in management.scheduled[-1]
    _, adapter_name, forwarded_kwargs = management.model_calls[-1]
    assert adapter_name == 'session-adapter'
    assert forwarded_kwargs == body.kwargs
