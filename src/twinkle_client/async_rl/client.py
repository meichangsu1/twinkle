# Copyright (c) ModelScope Contributors. All rights reserved.
"""Client for the async-RL dynamic tenant control plane."""

from __future__ import annotations

import time

from twinkle_client.http import get_base_url, http_delete, http_get, http_post
from twinkle_client.types.async_rl import (
    AsyncRLTenantConfig,
    AsyncRLTenantInfo,
    AsyncRLTenantListResponse,
    AsyncRLTenantStatus,
)


class AsyncRLClient:

    def __init__(self):
        self.server_url = f'{get_base_url()}/async-rl'

    def add_tenant(self, config: AsyncRLTenantConfig | dict) -> AsyncRLTenantInfo:
        request = AsyncRLTenantConfig.model_validate(config)
        response = http_post(
            url=f'{self.server_url}/tenants',
            json_data=request.model_dump(exclude_none=True),
        )
        return AsyncRLTenantInfo(**response.json())

    def get_tenant(self, context_id: str) -> AsyncRLTenantInfo:
        response = http_get(url=f'{self.server_url}/tenants/{context_id}')
        return AsyncRLTenantInfo(**response.json())

    def list_tenants(self) -> list[AsyncRLTenantInfo]:
        response = http_get(url=f'{self.server_url}/tenants')
        return AsyncRLTenantListResponse(**response.json()).tenants

    def remove_tenant(self, context_id: str) -> AsyncRLTenantInfo:
        response = http_delete(url=f'{self.server_url}/tenants/{context_id}')
        return AsyncRLTenantInfo(**response.json())

    def wait_for_status(
        self,
        context_id: str,
        statuses: set[AsyncRLTenantStatus | str],
        *,
        poll_interval: float = 1.0,
        timeout: float | None = None,
    ) -> AsyncRLTenantInfo:
        expected = {AsyncRLTenantStatus(status) for status in statuses}
        started = time.monotonic()
        while True:
            tenant = self.get_tenant(context_id)
            if tenant.status in expected:
                return tenant
            if tenant.status is AsyncRLTenantStatus.FAILED:
                raise RuntimeError(f'async-RL tenant {context_id} failed: {tenant.error}')
            if timeout is not None and time.monotonic() - started >= timeout:
                raise TimeoutError(f'timed out waiting for async-RL tenant {context_id}: {sorted(expected)}')
            time.sleep(poll_interval)
