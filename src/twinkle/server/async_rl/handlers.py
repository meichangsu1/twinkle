# Copyright (c) ModelScope Contributors. All rights reserved.
"""HTTP routes for the async-RL tenant control plane."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import Depends, FastAPI, HTTPException, Request, Response

from twinkle.server.utils.validation import get_token_from_request
from twinkle_client.types.async_rl import (
    AsyncRLTenantConfig,
    AsyncRLTenantInfo,
    AsyncRLTenantListResponse,
)

from .app import AsyncRLManagement, TenantCapacityError, TenantConflictError


def register_async_rl_routes(app: FastAPI, self_fn: Callable[[], AsyncRLManagement]) -> None:

    def owner(request: Request) -> str:
        return AsyncRLManagement.owner_fingerprint(get_token_from_request(request))

    @app.post('/tenants', response_model=AsyncRLTenantInfo, status_code=202)
    async def add_tenant(
        request: Request,
        body: AsyncRLTenantConfig,
        response: Response,
        self: AsyncRLManagement = Depends(self_fn),
    ) -> AsyncRLTenantInfo:
        try:
            info, created = await self.add_tenant(owner(request), body)
        except TenantConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except TenantCapacityError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not created:
            response.status_code = 200
        return info

    @app.get('/tenants', response_model=AsyncRLTenantListResponse)
    async def list_tenants(
        request: Request,
        self: AsyncRLManagement = Depends(self_fn),
    ) -> AsyncRLTenantListResponse:
        return AsyncRLTenantListResponse(tenants=await self.list_tenants(owner(request)))

    @app.get('/tenants/{context_id}', response_model=AsyncRLTenantInfo)
    async def get_tenant(
        context_id: str,
        request: Request,
        self: AsyncRLManagement = Depends(self_fn),
    ) -> AsyncRLTenantInfo:
        try:
            return await self.get_tenant(owner(request), context_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail='Tenant not found') from exc

    @app.delete('/tenants/{context_id}', response_model=AsyncRLTenantInfo, status_code=202)
    async def remove_tenant(
        context_id: str,
        request: Request,
        self: AsyncRLManagement = Depends(self_fn),
    ) -> AsyncRLTenantInfo:
        try:
            return await self.remove_tenant(owner(request), context_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail='Tenant not found') from exc
