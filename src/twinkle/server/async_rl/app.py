# Copyright (c) ModelScope Contributors. All rights reserved.
"""Ray Serve management application for dynamic async-RL tenants."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI

from twinkle.server.deployment import bind_deployment, build_deployment_app
from twinkle_client.types.async_rl import AsyncRLTenantConfig, AsyncRLTenantInfo, AsyncRLTenantStatus


class TenantConflictError(RuntimeError):
    pass


class TenantCapacityError(RuntimeError):
    pass


@dataclass
class _TenantRecord:
    context_id: str
    owner: str
    config: AsyncRLTenantConfig
    config_digest: str
    status: AsyncRLTenantStatus = AsyncRLTenantStatus.ADDING
    policy_version: int = 0
    live_partitions: int = 0
    completed_partitions: int = 0
    error: str | None = None
    cancel_requested: bool = False
    runtime_context_key: str | None = None

    @property
    def context_key(self) -> str:
        return f'{self.config.tenant_id}/{self.config.training_run_id}/{self.config.adapter_name}'

    def response(self) -> AsyncRLTenantInfo:
        return AsyncRLTenantInfo(
            context_id=self.context_id,
            context_key=self.context_key,
            tenant_id=self.config.tenant_id,
            training_run_id=self.config.training_run_id,
            adapter_name=self.config.adapter_name,
            status=self.status,
            policy_version=self.policy_version,
            live_partitions=self.live_partitions,
            completed_partitions=self.completed_partitions,
            error=self.error,
        )


class AsyncRLManagement:
    """Single-owner HTTP control plane around :class:`AsyncRLRuntime`."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        initial_owner_token: str | None = None,
        runtime_factory: Any = None,
    ):
        self.config = dict(config)
        self.initial_owner_token = initial_owner_token or os.environ.get('TWINKLE_SERVER_TOKEN', 'EMPTY_TOKEN')
        if runtime_factory is None:
            from twinkle_agentic.async_rl import AsyncRLRuntime
            runtime_factory = AsyncRLRuntime
        self.runtime_factory = runtime_factory
        self.runtime, self._initial_specs = self.runtime_factory.from_config(self.config)
        self._records: dict[str, _TenantRecord] = {}
        self._start_lock = asyncio.Lock()
        self._records_lock = asyncio.Lock()
        self._started = False
        self._operation_tasks: set[asyncio.Task[None]] = set()
        self._reconciler_task: asyncio.Task[None] | None = None
        self._startup_task: asyncio.Task[None] | None = None
        try:
            self._startup_task = asyncio.get_running_loop().create_task(self._start())
        except RuntimeError:
            pass

    async def ensure_started(self) -> None:
        if self._started:
            return
        if self._startup_task is not None and self._startup_task is not asyncio.current_task():
            await self._startup_task
            return
        await self._start()

    async def _start(self) -> None:
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return
            await self.runtime.start()
            self._started = True
            self._reconciler_task = asyncio.create_task(self._reconcile())
            owner = self.owner_fingerprint(self.initial_owner_token)
            for raw_spec in self._initial_specs:
                initial_token = raw_spec.pop('owner_token', self.initial_owner_token)
                await self.add_tenant(
                    self.owner_fingerprint(str(initial_token)) if initial_token else owner,
                    AsyncRLTenantConfig.model_validate(raw_spec),
                )

    async def shutdown(self) -> None:
        if self._startup_task is not None and not self._startup_task.done():
            await asyncio.gather(self._startup_task, return_exceptions=True)
        tasks = tuple(self._operation_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._reconciler_task is not None:
            self._reconciler_task.cancel()
            await asyncio.gather(self._reconciler_task, return_exceptions=True)
        await self.runtime.stop()

    async def add_tenant(
        self,
        owner: str,
        config: AsyncRLTenantConfig,
    ) -> tuple[AsyncRLTenantInfo, bool]:
        await self.ensure_started()
        digest = self._config_digest(config)
        context_key = f'{config.tenant_id}/{config.training_run_id}/{config.adapter_name}'
        async with self._records_lock:
            for record in self._records.values():
                if record.owner != owner or record.context_key != context_key:
                    continue
                if record.status in (AsyncRLTenantStatus.REMOVED, AsyncRLTenantStatus.FAILED):
                    continue
                if record.config_digest != digest:
                    raise TenantConflictError(f'context {context_key} already exists with different configuration')
                return record.response(), False
            in_use = sum(
                record.status in (
                    AsyncRLTenantStatus.ADDING,
                    AsyncRLTenantStatus.ACTIVE,
                    AsyncRLTenantStatus.DRAINING,
                )
                for record in self._records.values()
            )
            if in_use >= self.runtime.capacity():
                raise TenantCapacityError(f'async-RL LoRA capacity reached: {in_use}/{self.runtime.capacity()}')
            record = _TenantRecord(
                context_id=str(uuid.uuid4()),
                owner=owner,
                config=config,
                config_digest=digest,
            )
            self._records[record.context_id] = record
            self._track(asyncio.create_task(self._add(record)))
            return record.response(), True

    async def remove_tenant(self, owner: str, context_id: str) -> AsyncRLTenantInfo:
        await self.ensure_started()
        async with self._records_lock:
            record = self._owned_record(owner, context_id)
            if record.status in (AsyncRLTenantStatus.REMOVED, AsyncRLTenantStatus.FAILED):
                return record.response()
            if record.status is AsyncRLTenantStatus.ADDING:
                record.cancel_requested = True
                return record.response()
            if record.status is not AsyncRLTenantStatus.DRAINING:
                record.status = AsyncRLTenantStatus.DRAINING
                await self.runtime.request_drain(record.runtime_context_key or record.context_key)
                self._track(asyncio.create_task(self._remove(record)))
            return record.response()

    async def get_tenant(self, owner: str, context_id: str) -> AsyncRLTenantInfo:
        await self.ensure_started()
        return self._owned_record(owner, context_id).response()

    async def list_tenants(self, owner: str) -> list[AsyncRLTenantInfo]:
        await self.ensure_started()
        return [record.response() for record in self._records.values() if record.owner == owner]

    async def _add(self, record: _TenantRecord) -> None:
        try:
            runtime_config = record.config.model_dump(exclude_none=True)
            runtime_config['_runtime_adapter_name'] = f'ar-{record.context_id[:8]}-{record.config.adapter_name}'
            tenant = await self.runtime.add_tenant(runtime_config)
            record.runtime_context_key = tenant.context.key
            if record.cancel_requested:
                record.status = AsyncRLTenantStatus.DRAINING
                await self.runtime.request_drain(record.runtime_context_key)
                await self._remove(record)
            else:
                record.status = AsyncRLTenantStatus.ACTIVE
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            record.status = AsyncRLTenantStatus.FAILED
            record.error = f'{type(exc).__name__}: {exc}'

    async def _remove(self, record: _TenantRecord) -> None:
        try:
            await self.runtime.remove_tenant(record.runtime_context_key or record.context_key)
            record.status = AsyncRLTenantStatus.REMOVED
            record.live_partitions = 0
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            record.status = AsyncRLTenantStatus.FAILED
            record.error = f'{type(exc).__name__}: {exc}'

    async def _reconcile(self) -> None:
        try:
            while True:
                await self.runtime.check_health()
                snapshots = {
                    snapshot['context'].key: snapshot
                    for snapshot in await self.runtime.snapshots()
                }
                for record in list(self._records.values()):
                    snapshot = snapshots.get(record.runtime_context_key or record.context_key)
                    if snapshot is None:
                        continue
                    record.policy_version = int(snapshot['policy_version'])
                    record.live_partitions = int(snapshot['live_partitions'])
                    record.completed_partitions = int(snapshot['completed_partitions'])
                    if str(snapshot['status']) == 'FINISHED' and record.status is AsyncRLTenantStatus.ACTIVE:
                        record.status = AsyncRLTenantStatus.DRAINING
                        self._track(asyncio.create_task(self._remove(record)))
                await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            for record in self._records.values():
                if record.status not in (AsyncRLTenantStatus.REMOVED, AsyncRLTenantStatus.FAILED):
                    record.status = AsyncRLTenantStatus.FAILED
                    record.error = error

    def _owned_record(self, owner: str, context_id: str) -> _TenantRecord:
        record = self._records.get(context_id)
        if record is None or record.owner != owner:
            raise KeyError(context_id)
        return record

    def _track(self, task: asyncio.Task[None]) -> None:
        self._operation_tasks.add(task)
        task.add_done_callback(self._operation_tasks.discard)

    @staticmethod
    def owner_fingerprint(token: str) -> str:
        return hashlib.sha256(token.encode('utf-8')).hexdigest()

    @staticmethod
    def _config_digest(config: AsyncRLTenantConfig) -> str:
        payload = json.dumps(config.model_dump(mode='json', exclude_none=True), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def build_async_rl_app(
    config: dict[str, Any],
    deploy_options: dict[str, Any],
    initial_owner_token: str | None = None,
):
    from .handlers import register_async_rl_routes

    autoscaling = deploy_options.get('autoscaling_config')
    if autoscaling and (
        int(autoscaling.get('min_replicas', 1)) != 1
        or int(autoscaling.get('max_replicas', 1)) != 1
    ):
        raise ValueError('async_rl deployment must use exactly one replica')
    if 'num_replicas' in deploy_options and int(deploy_options['num_replicas']) != 1:
        raise ValueError('async_rl deployment must use exactly one replica')
    deploy_options = dict(deploy_options)
    if not autoscaling:
        deploy_options.setdefault('num_replicas', 1)

    def register_routes(app: FastAPI, get_self: Any) -> None:
        register_async_rl_routes(app, get_self)

    async def shutdown(servable: AsyncRLManagement) -> None:
        await servable.shutdown()

    app = build_deployment_app('AsyncRL', register_routes, on_shutdown=shutdown)
    return bind_deployment(
        app,
        AsyncRLManagement,
        deploy_options,
        deployment_name='AsyncRLManagement',
        bind_kwargs={
            'config': config,
            'initial_owner_token': initial_owner_token,
        },
    )
