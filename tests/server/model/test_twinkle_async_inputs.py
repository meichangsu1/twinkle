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
        self.data_plane = self
        self.rows = {
            'data-a': [{
                'train_input': {'input_ids': [index]},
                'sampled_logprobs': [-0.1],
                'advantage': 1.0,
            } for index in range(4)],
            'data-b': [{
                'train_input': {'input_ids': [index]},
                'sampled_logprobs': [-0.2],
                'advantage': -1.0,
            } for index in range(4, 8)],
        }

    async def _on_request_start(self, _request):
        return 'token'

    def assert_resource_exists(self, _adapter_name):
        return None

    def forward_backward(self, *, inputs, adapter_name, **kwargs):
        self.model_calls.append((inputs, adapter_name, kwargs))
        return {'loss': 1.0}

    async def get(self, ref, *, fields=None):
        rows = self.rows[ref.ref_id]
        if fields is None:
            return rows
        return [{field: row[field] for field in fields} for row in rows]

    async def schedule_task_and_wait(self, task, **kwargs):
        self.scheduled.append(kwargs)
        return await task()


@pytest.mark.asyncio
async def test_forward_backward_resolves_multiple_data_refs_and_field_kwargs() -> None:
    management = _SchedulingManagement()
    app = FastAPI()
    _register_twinkle_routes(app, lambda: management)
    route = next(route for route in app.routes if getattr(route, 'path', None) == '/twinkle/forward_backward')
    request = Request({'type': 'http', 'headers': []})
    request.state.session_id = 'session'
    body = types.ForwardRequest(
        adapter_name='adapter',
        input_refs=[
            types.DataRef(ref_id='data-a', size=4, num_tokens=4),
            types.DataRef(ref_id='data-b', size=4, num_tokens=4),
        ],
        input_field='train_input',
        kwarg_fields={
            'old_logps': 'sampled_logprobs',
            'advantages': 'advantage',
        },
    )

    await route.endpoint(request, body, management)

    assert management.scheduled[-1]['batch_size'] == 8
    assert management.scheduled[-1]['data_world_size'] == 2
    inputs, adapter_name, forwarded_kwargs = management.model_calls[-1]
    assert adapter_name == 'session-adapter'
    assert [row['input_ids'].tolist() for row in inputs] == [[index] for index in range(8)]
    assert forwarded_kwargs == {
        'old_logps': [[-0.1]] * 4 + [[-0.2]] * 4,
        'advantages': [1.0] * 4 + [-1.0] * 4,
    }
