# Copyright (c) ModelScope Contributors. All rights reserved.
"""Future handle returned by an individual Model or Sampler component."""
from __future__ import annotations

import asyncio
import time
from typing import Any

from twinkle_client.http import get_base_url, http_post
from twinkle_client.http.http_utils import _build_headers
from twinkle_client.types.component import ComponentTaskRef


class RemoteTaskError(RuntimeError):
    pass


class RemoteTask:
    """A wrapper over the server's existing component future registry."""

    def __init__(self, task: ComponentTaskRef | str):
        self.request_id = task if isinstance(task, str) else task.request_id
        self.model_id = None if isinstance(task, str) else task.model_id
        self._url = f'{get_base_url()}/retrieve_future'

    def poll(self, timeout: float | None = None) -> Any | None:
        request_kwargs = {} if timeout is None else {'timeout': timeout}
        response = http_post(
            self._url,
            json_data={'request_id': self.request_id},
            **request_kwargs,
        )
        response.raise_for_status()
        return self._resolve_payload(response.json())

    @staticmethod
    def _resolve_payload(payload: Any) -> Any | None:
        if isinstance(payload, dict) and payload.get('type') == 'try_again':
            return None
        if isinstance(payload, dict) and 'error' in payload:
            raise RemoteTaskError(payload['error'])
        return payload

    def result(self, timeout: float | None = None) -> Any:
        import requests

        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError(f'component task {self.request_id} did not finish within {timeout}s')
            try:
                result = self.poll(timeout=remaining)
            except requests.Timeout as exc:
                raise TimeoutError(
                    f'component task {self.request_id} did not finish within {timeout}s') from exc
            if result is not None:
                return result
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f'component task {self.request_id} did not finish within {timeout}s')

    async def aresult(self, timeout: float | None = None) -> Any:
        import httpx
        deadline = None if timeout is None else time.monotonic() + timeout
        poll_interval = 0.05
        async with httpx.AsyncClient(timeout=600) as client:
            while True:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(
                        f'component task {self.request_id} did not finish within {timeout}s')
                try:
                    response = await client.post(
                        self._url,
                        headers=_build_headers(),
                        json={'request_id': self.request_id},
                        timeout=remaining if remaining is not None else 600,
                    )
                except httpx.TimeoutException as exc:
                    raise TimeoutError(
                        f'component task {self.request_id} did not finish within {timeout}s') from exc
                response.raise_for_status()
                result = self._resolve_payload(response.json())
                if result is not None:
                    return result
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(f'component task {self.request_id} did not finish within {timeout}s')
                remaining = None if deadline is None else deadline - time.monotonic()
                await asyncio.sleep(
                    poll_interval if remaining is None else min(poll_interval, max(remaining, 0.0)))
                poll_interval = min(poll_interval * 1.5, 1.0)

    def __await__(self):
        return self.aresult().__await__()
