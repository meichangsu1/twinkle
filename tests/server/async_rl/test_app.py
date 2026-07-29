from __future__ import annotations

import asyncio

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from twinkle.server.async_rl.app import AsyncRLManagement, TenantConflictError
from twinkle.server.async_rl.handlers import register_async_rl_routes
from twinkle_client.types.async_rl import AsyncRLTenantConfig, AsyncRLTenantStatus


def _config(adapter_name: str = 'adapter') -> AsyncRLTenantConfig:
    return AsyncRLTenantConfig(
        tenant_id='tenant',
        training_run_id='run',
        adapter_name=adapter_name,
        dataset={'dataset_id': 'dataset', 'max_length': 128},
        rollout={
            'batch_size': 1,
            'num_generations': 2,
            'max_tokens': 16,
            'temperature': 1.0,
            'top_p': 1.0,
        },
        train={
            'mini_batch_size': 2,
            'micro_batch_size': 1,
        },
        reward={'class_path': 'example.Reward'},
    )


class _FakeRuntime:

    def __init__(self):
        self.contexts = {}
        self.started = False
        self.removed = []

    @classmethod
    def from_config(cls, config):
        return cls(), []

    async def start(self):
        self.started = True

    async def stop(self):
        self.started = False

    async def check_health(self):
        pass

    def capacity(self):
        return 4

    async def add_tenant(self, item):
        adapter_name = item.get('_runtime_adapter_name', item['adapter_name'])
        key = f'{item["tenant_id"]}/{item["training_run_id"]}/{adapter_name}'
        self.contexts[key] = {
            'context': type('Context', (), {'key': key})(),
            'status': 'ACTIVE',
            'policy_version': 0,
            'live_partitions': 0,
            'completed_partitions': 0,
        }
        return type('Tenant', (), {
            'context': type('Context', (), {'key': key})(),
        })()

    async def request_drain(self, key):
        self.contexts[key]['status'] = 'DRAINING'

    async def remove_tenant(self, key):
        self.removed.append(key)
        self.contexts.pop(key, None)

    async def snapshots(self):
        return list(self.contexts.values())


async def _wait_status(management, owner, context_id, status):
    for _ in range(100):
        info = await management.get_tenant(owner, context_id)
        if info.status is status:
            return info
        await asyncio.sleep(0.001)
    raise AssertionError(f'tenant did not reach {status}')


def test_management_add_is_idempotent_and_owner_isolated():
    async def run():
        management = AsyncRLManagement({}, runtime_factory=_FakeRuntime)
        owner = management.owner_fingerprint('owner')
        other = management.owner_fingerprint('other')

        first, created = await management.add_tenant(owner, _config())
        assert created
        active = await _wait_status(management, owner, first.context_id, AsyncRLTenantStatus.ACTIVE)

        duplicate, created = await management.add_tenant(owner, _config())
        assert not created
        assert duplicate.context_id == active.context_id

        try:
            await management.get_tenant(other, first.context_id)
        except KeyError:
            pass
        else:
            raise AssertionError('another owner must not see the tenant')

        changed = _config()
        changed.rollout.top_p = 0.5
        try:
            await management.add_tenant(owner, changed)
        except TenantConflictError:
            pass
        else:
            raise AssertionError('different duplicate configuration must conflict')
        await management.shutdown()

    asyncio.run(run())


def test_management_gracefully_removes_and_keeps_tombstone():
    async def run():
        management = AsyncRLManagement({}, runtime_factory=_FakeRuntime)
        owner = management.owner_fingerprint('owner')
        tenant, _ = await management.add_tenant(owner, _config())
        await _wait_status(management, owner, tenant.context_id, AsyncRLTenantStatus.ACTIVE)

        draining = await management.remove_tenant(owner, tenant.context_id)
        assert draining.status is AsyncRLTenantStatus.DRAINING
        removed = await _wait_status(management, owner, tenant.context_id, AsyncRLTenantStatus.REMOVED)
        assert removed.live_partitions == 0
        assert len(await management.list_tenants(owner)) == 1
        await management.shutdown()

    asyncio.run(run())


def test_http_routes_return_202_and_hide_other_owners():
    management = AsyncRLManagement({}, runtime_factory=_FakeRuntime)
    app = FastAPI()

    @app.middleware('http')
    async def inject_token(request: Request, call_next):
        request.state.token = request.headers.get('x-test-token', '')
        return await call_next(request)

    @app.on_event('shutdown')
    async def shutdown():
        await management.shutdown()

    register_async_rl_routes(app, lambda: management)
    payload = _config().model_dump()
    with TestClient(app) as client:
        created = client.post('/tenants', json=payload, headers={'x-test-token': 'owner'})
        assert created.status_code == 202
        context_id = created.json()['context_id']

        visible = client.get(f'/tenants/{context_id}', headers={'x-test-token': 'owner'})
        assert visible.status_code == 200
        hidden = client.get(f'/tenants/{context_id}', headers={'x-test-token': 'other'})
        assert hidden.status_code == 404

        removed = client.delete(f'/tenants/{context_id}', headers={'x-test-token': 'owner'})
        assert removed.status_code == 202
