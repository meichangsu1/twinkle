# Copyright (c) ModelScope Contributors. All rights reserved.
"""Backend-independent instance routing. No generation results or KV data live here."""
from __future__ import annotations
import random
import threading
import uuid
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


class RoutingPolicy(Protocol):

    def select(self, loads: Mapping[str, int]) -> str:
        ...


class LeastInflightPolicy:

    def select(self, loads: Mapping[str, int]) -> str:
        minimum = min(loads.values())
        return random.choice([key for key, value in loads.items() if value == minimum])


@dataclass(frozen=True)
class RouteLease:
    lease_id: str
    instance_id: str
    handle: Any


def normalize_routing_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if config is not None and not isinstance(config, Mapping):
        raise ValueError('routing_config must be a mapping')
    result = {'max_sessions': 10000, **dict(config or {})}
    if set(result) != {'max_sessions'}:
        raise ValueError('Unknown routing_config field; supported: max_sessions')
    limit = result['max_sessions']
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError('max_sessions must be a positive integer')
    return result


class RequestRouter:
    """Strict affinity while a binding exists; bounded LRU and stateless fallback.

    Locks support local callers. Ray deployments use this class directly as an actor.
    Leases are independent of sessions, submissions, training groups and policies.
    """

    def __init__(self, instances: Mapping[str, Any], *, max_sessions: int = 10000,
                 policy: RoutingPolicy | None = None):
        self._config = normalize_routing_config({'max_sessions': max_sessions})
        self._policy = policy if policy is not None else LeastInflightPolicy()
        self._lock = threading.Lock()
        self._instances = dict(instances)
        self._loads = dict.fromkeys(self._instances, 0)
        self._sessions: OrderedDict[str, str] = OrderedDict()
        self._session_loads: dict[str, int] = {}
        self._leases: dict[str, tuple[RouteLease, str | None, int]] = {}
        self._fallbacks = 0

    @staticmethod
    def _validate(session_id: str | None, units: int) -> None:
        if session_id is not None and (not isinstance(session_id, str) or not session_id):
            raise ValueError('session_id must be a non-empty string or None')
        if isinstance(units, bool) or not isinstance(units, int) or units < 1:
            raise ValueError('units must be a positive integer')

    def acquire(self, session_id: str | None = None, units: int = 1) -> RouteLease:
        return self.acquire_batch([(session_id, units)])[0]

    def acquire_batch(self, requests: Sequence[tuple[str | None, int]]) -> list[RouteLease]:
        requests = list(requests)
        for sid, units in requests:
            self._validate(sid, units)
        with self._lock:
            if requests and not self._instances:
                raise RuntimeError('No available rollout instances')
            leases = []
            previous_sessions = self._sessions.copy()
            previous_fallbacks = self._fallbacks
            try:
                for sid, units in requests:
                    target = self._sessions.get(sid) if sid is not None else None
                    if target not in self._instances:
                        candidates = {key: self._loads[key] for key in self._instances}
                        target = self._policy.select(candidates)
                        if target not in candidates:
                            raise ValueError('Routing policy selected an unavailable instance')
                        if sid is not None:
                            self._bind(sid, target)
                    if sid in self._sessions:
                        self._sessions.move_to_end(sid)
                    lease = RouteLease(uuid.uuid4().hex, target, self._instances[target])
                    self._leases[lease.lease_id] = (lease, sid, units)
                    self._loads[target] += units
                    if sid is not None:
                        self._session_loads[sid] = self._session_loads.get(sid, 0) + units
                    leases.append(lease)
            except BaseException:
                for lease in leases:
                    self._release(lease.lease_id)
                self._sessions = previous_sessions
                self._fallbacks = previous_fallbacks
                raise
            return leases

    def _bind(self, sid: str, target: str) -> None:
        if len(self._sessions) >= self._config['max_sessions']:
            victim = next((key for key in self._sessions if not self._session_loads.get(key)), None)
            if victim is None:
                self._fallbacks += 1
                return
            del self._sessions[victim]
        self._sessions[sid] = target

    def release(self, lease: RouteLease) -> None:
        self.release_batch([lease.lease_id])

    def release_batch(self, lease_ids: Sequence[str]) -> None:
        with self._lock:
            for lease_id in lease_ids:
                self._release(lease_id)

    def _release(self, lease_id: str) -> None:
        entry = self._leases.pop(lease_id, None)
        if entry is None:
            return
        lease, sid, units = entry
        self._loads[lease.instance_id] -= units
        if lease.instance_id not in self._instances and not self._loads[lease.instance_id]:
            del self._loads[lease.instance_id]
        if sid is not None:
            self._session_loads[sid] -= units
            if not self._session_loads[sid]:
                del self._session_loads[sid]

    def release_sessions(self, ids: Sequence[str]) -> None:
        with self._lock:
            if any(self._session_loads.get(sid) for sid in ids):
                raise RuntimeError('Cannot release a session with requests in flight')
            for sid in ids:
                self._sessions.pop(sid, None)

    def add_instances(self, instances: Mapping[str, Any]) -> None:
        with self._lock:
            if any(self._loads.get(key, 0) for key in instances):
                raise RuntimeError('Cannot replace an instance with requests in flight')
            self._instances.update(instances)
            for key in instances:
                self._loads[key] = 0

    def remove_instances(self, ids: Sequence[str]) -> None:
        with self._lock:
            for key in ids:
                self._instances.pop(key, None)
                if not self._loads.get(key):
                    self._loads.pop(key, None)
            for sid, target in list(self._sessions.items()):
                if target not in self._instances:
                    del self._sessions[sid]

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                'config': dict(self._config),
                'loads': dict(self._loads),
                'total_inflight': sum(self._loads.values()),
                'active_leases': len(self._leases),
                'active_instances': list(self._instances),
                'sessions': dict(self._sessions),
                'session_count': len(self._sessions),
                'capacity_fallbacks': self._fallbacks,
            }

    def get_config(self) -> dict[str, Any]:
        return dict(self._config)
