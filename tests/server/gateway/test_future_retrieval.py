# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

tinker = pytest.importorskip('tinker')

from twinkle.server.gateway.tinker_handlers import _register_tinker_routes


def _make_client(get_future: AsyncMock) -> TestClient:
    management = MagicMock()
    management.state.get_future = get_future
    management.supported_models = []

    app = FastAPI()

    @app.middleware('http')
    async def _set_request_state(request: Request, call_next):
        authorization = request.headers.get('Authorization', '')
        request.state.token = authorization.removeprefix('Bearer ')
        request.state.session_id = request.headers.get('X-Twinkle-Session-Id', '')
        return await call_next(request)

    _register_tinker_routes(app, lambda: management)
    return TestClient(app)


def test_retrieve_future_uses_request_id_as_capability() -> None:
    get_future = AsyncMock(return_value={'status': 'completed', 'result': {'value': 7}})
    client = _make_client(get_future)

    response = client.post(
        '/retrieve_future',
        json={'request_id': 'req-1'},
        headers={
            'Authorization': 'Bearer tenant-token',
            'X-Twinkle-Session-Id': 'session-1',
        },
    )

    assert response.status_code == 200
    assert response.json() == {'value': 7}
    get_future.assert_awaited_once_with('req-1')


def test_not_yet_visible_future_returns_try_again(monkeypatch) -> None:
    monkeypatch.setenv('TWINKLE_LONG_POLL_TIMEOUT', '0')
    get_future = AsyncMock(return_value=None)
    client = _make_client(get_future)

    response = client.post(
        '/retrieve_future',
        json={'request_id': 'unknown'},
        headers={
            'Authorization': 'Bearer tenant-token',
            'X-Twinkle-Session-Id': 'session-1',
        },
    )

    assert response.status_code == 200
    assert response.json() == {'type': 'try_again'}


def test_initial_cross_replica_miss_is_long_polled(monkeypatch) -> None:
    monkeypatch.setenv('TWINKLE_LONG_POLL_TIMEOUT', '1')
    monkeypatch.setenv('TWINKLE_POLL_INTERVAL', '0')
    get_future = AsyncMock(side_effect=[
        None,
        {'status': 'completed', 'result': {'value': 9}},
    ])
    client = _make_client(get_future)

    response = client.post(
        '/retrieve_future',
        json={'request_id': 'req-replicated'},
        headers={'Authorization': 'Bearer tenant-token'},
    )

    assert response.status_code == 200
    assert response.json() == {'value': 9}
    assert get_future.await_count == 2
