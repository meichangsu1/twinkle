import torch

from twinkle.kernel.ops import ep as ep_ops
from twinkle.kernel.ops.ep import loop as loop_ops


class _Backend(ep_ops.EpExpertsGmm):

    def __init__(self, name: str, value: float, *, fallback: bool = False):
        self.name = name
        self.value = value
        self.fallback = fallback

    def ineligible_reason(self, experts_mod):
        return None

    def forward(self, experts_mod, permuted_tokens, num_global_sum_tokens_per_local_expert, experts_per_rank):
        return torch.full_like(permuted_tokens, self.value)


def test_ep_forward_uses_accelerated_backend_by_default(monkeypatch):
    monkeypatch.delenv('TWINKLE_EP_FORCE_LOOP', raising=False)
    monkeypatch.setattr(ep_ops, '_IMPLS', [_Backend('accelerated', 1.0), _Backend('loop', 2.0, fallback=True)])
    result = ep_ops.ep_forward(None, torch.zeros(2, 3), torch.tensor([2]), 1)
    assert torch.equal(result, torch.ones(2, 3))


def test_ep_forward_can_force_reference_loop(monkeypatch):
    monkeypatch.setenv('TWINKLE_EP_FORCE_LOOP', '1')
    monkeypatch.setattr(ep_ops, '_IMPLS', [_Backend('accelerated', 1.0), _Backend('loop', 2.0, fallback=True)])
    monkeypatch.setattr(loop_ops, 'LoopEpExpertsGmm', lambda: _Backend('loop', 2.0, fallback=True))
    result = ep_ops.ep_forward(None, torch.zeros(2, 3), torch.tensor([2]), 1)
    assert torch.equal(result, torch.full((2, 3), 2.0))
