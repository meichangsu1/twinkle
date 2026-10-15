# Copyright (c) ModelScope Contributors. All rights reserved.
"""Ray deployment of the shared instance router."""
from __future__ import annotations
import hashlib
import threading
from collections.abc import Mapping, Sequence
from typing import Any

import json

from .routing import RequestRouter, normalize_routing_config

_POOL_ROUTERS: dict[tuple[str, str, str], Any] = {}
_POOL_ROUTER_LOCK = threading.Lock()


def instance_id(handle) -> str:
    return handle._actor_id.hex()


def get_shared_router(workers: Sequence[Any], config: Mapping[str, Any] | None = None):
    import ray

    config = normalize_routing_config(config)
    instances = {instance_id(worker): worker for worker in workers}
    if not instances:
        raise ValueError('Cannot create a router for an empty instance pool')
    pool_key = hashlib.sha256(json.dumps(sorted(instances)).encode()).hexdigest()
    # Named actors are scoped by Ray namespace. Non-detached ownership keeps the
    # router's lifetime tied to the job that creates the sampling pool.
    runtime = ray.get_runtime_context()
    job_id = str(runtime.get_job_id())
    key = (job_id, runtime.namespace, pool_key)
    with _POOL_ROUTER_LOCK:
        # A non-detached named actor still uses reference counting. Retain a
        # job-local reference across temporary sampler handles and caller exits.
        for previous in list(_POOL_ROUTERS):
            if previous[0] != job_id:
                del _POOL_ROUTERS[previous]
        actor = _POOL_ROUTERS.get(key)
        if actor is None:
            actor = ray.remote(RequestRouter).options(
                name=f'twinkle-request-router-{pool_key}', get_if_exists=True,
                num_cpus=0).remote(instances, **config)
            _POOL_ROUTERS[key] = actor
    actual = ray.get(actor.get_config.remote())
    if actual != config:
        raise ValueError(f'Routing configuration mismatch for shared pool: {actual} != {config}')
    return actor


def get_handle_router(owner, workers):
    ids = tuple(sorted(instance_id(worker) for worker in workers))
    # Only actor handles and immutable ids are kept on the sampler handle; they
    # can be sent to other rollout actors without duplicating routing state.
    if getattr(owner, '_routing_pool_ids', None) != ids:
        router = get_shared_router(workers, getattr(owner, '_routing_config', None))
        owner._request_router = router
        owner._routing_pool_ids = ids
    return owner._request_router


def release_handle_sessions(owner, ids: Sequence[str]) -> None:
    router = getattr(owner, '_request_router', None)
    if router is None:
        return  # Local engine, or a handle which has never acquired routing.
    import ray

    ray.get(router.release_sessions.remote(list(ids)))
