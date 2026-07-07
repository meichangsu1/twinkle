# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from typing import Any, Iterable

from twinkle.data_format import Trajectory
from .metrics import AsyncRLMetricsRecorder, NoopMetricsRecorder
from .types import ComponentResult, LoraContext
from .workers import AsyncRollouter


class PromptLoader:
    """Load raw prompt groups from a Twinkle DataLoader into AsyncRollouter.

    This is the rollout-side data ingress. It wraps an iterable such as
    `twinkle.dataloader.DataLoader` and never reads training samples from
    TransferQueue. Trainer-side TQ reading remains owned by TrainerWorker.
    """

    def __init__(
        self,
        *,
        context: LoraContext,
        dataloader: Iterable[Any],
        rollouter: AsyncRollouter,
        max_pending_groups: int | None = None,
        metrics_recorder: AsyncRLMetricsRecorder | None = None,
    ):
        self.context = context
        self.dataloader = dataloader
        self.rollouter = rollouter
        self.max_pending_groups = max_pending_groups
        self.metrics_recorder = metrics_recorder or NoopMetricsRecorder()
        self._iterator = None
        self.exhausted = False
        self.submitted_groups = 0

    def can_load(self) -> bool:
        if self.exhausted:
            return False
        if self.max_pending_groups is None:
            return True
        pending = self.rollouter.pending_prompt_group_count(self.context)
        return pending < self.max_pending_groups

    def step(self) -> ComponentResult | None:
        """Read one dataloader batch and enqueue it as rollout prompt groups."""
        if not self.can_load():
            return None
        if self._iterator is None:
            self._iterator = iter(self.dataloader)
        try:
            batch = next(self._iterator)
        except StopIteration:
            self.exhausted = True
            return None

        prompt_groups = self._normalize_batch(batch)
        if not prompt_groups:
            return None
        self.rollouter.enqueue_prompt_groups(self.context, prompt_groups)
        self.submitted_groups += len(prompt_groups)
        self.metrics_recorder.log_event(
            event='prompt_loaded',
            phase='prompt',
            context=self.context,
            metrics={
                'prompt_groups': len(prompt_groups),
                'pending_prompt_groups': self.rollouter.pending_prompt_group_count(self.context),
                'max_pending_groups': self.max_pending_groups,
                'submitted_groups': self.submitted_groups,
            },
        )
        return ComponentResult(component='prompt_loader', kind='prompt', count=len(prompt_groups))

    def is_idle(self) -> bool:
        return not self.can_load()

    def shutdown(self) -> None:
        for method_name in ('shutdown', 'close'):
            method = getattr(self.dataloader, method_name, None)
            if method is not None:
                method()
                return

    @staticmethod
    def _normalize_batch(batch: Any) -> list[Trajectory]:
        if batch is None:
            return []
        if isinstance(batch, list):
            return batch
        if isinstance(batch, tuple):
            return list(batch)
        return [batch]
