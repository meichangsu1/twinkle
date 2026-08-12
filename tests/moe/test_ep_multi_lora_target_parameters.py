import pytest
import sys
import torch
import types
from torch import nn


def _ensure_dummy_zmq():
    if "zmq" in sys.modules:
        return
    sys.modules["zmq"] = types.SimpleNamespace(
        Context=object,
        Socket=object,
        RCVTIMEO=1,
        SNDTIMEO=2,
        LINGER=3,
    )


def test_ep_target_parameter_lora_gather_dim_matches_peft_flattening():
    _ensure_dummy_zmq()
    from twinkle.model.transformers.strategy.native_fsdp import _ep_expert_state_dict_gather_dim

    assert _ep_expert_state_dict_gather_dim("model.layers.0.mlp.experts.lora_A.weight") == 0
    assert _ep_expert_state_dict_gather_dim("model.layers.0.mlp.experts.base_layer.lora_A.weight") == 0
    assert _ep_expert_state_dict_gather_dim("model.layers.0.mlp.experts.lora_B.weight") == 1
    assert _ep_expert_state_dict_gather_dim("model.layers.0.mlp.experts.base_layer.lora_B.weight") == 1
    assert _ep_expert_state_dict_gather_dim(
        "model.layers.0.mlp.experts._twinkle_lora_gate_up_proj.lora_B.lora_0.weight") == 0


class _FakeTensorExperts(nn.Module):

    def __init__(self, *, device="cpu", dtype=torch.float32):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(4, 3, 8, device=device, dtype=dtype))
        self.down_proj = nn.Parameter(torch.empty(4, 4, 3, device=device, dtype=dtype))
        self.num_experts = 4


def test_target_parameter_lora_slots_stay_meta_until_fsdp_materialization():
    from twinkle.model.multi_lora_target_parameters import TargetParameterLoraManager

    model = nn.Module()
    model.experts = _FakeTensorExperts(device="meta")
    manager = TargetParameterLoraManager(max_loras=2, max_r=4)
    manager.patch(model, ["experts.gate_up_proj", "experts.down_proj"])

    for wrapper in manager.wrappers:
        assert all(param.is_meta for param in wrapper.lora_A.values())
        assert all(param.is_meta for param in wrapper.lora_B.values())


def test_target_parameter_lora_defers_initial_snapshot_on_source_rank():
    from twinkle.model.multi_lora_target_parameters import TargetParameterLoraManager

    model = nn.Module()
    model.experts = _FakeTensorExperts()
    manager = TargetParameterLoraManager(max_loras=2, max_r=4, defer_initial_weights=True)
    manager.patch(model, ["experts.gate_up_proj", "experts.down_proj"])

    for wrapper in manager.wrappers:
        assert all(not param.is_meta for param in wrapper.lora_A.values())
        assert all(torch.count_nonzero(param) == 0 for param in wrapper.lora_B.values())
        assert wrapper._initial_lora_A == {}

    manager.save_initial_weights()

    for wrapper in manager.wrappers:
        assert set(wrapper._initial_lora_A) == {"lora_0", "lora_1"}


def test_ep_shards_target_parameter_lora_slots_on_meta():
    _ensure_dummy_zmq()
    from twinkle.model.multi_lora_target_parameters import TargetParameterLoraManager
    from twinkle.model.transformers.moe.expert_parallel import _shard_tensor_experts

    model = nn.Module()
    model.experts = _FakeTensorExperts(device="meta")
    manager = TargetParameterLoraManager(max_loras=2, max_r=4)
    manager.patch(model, ["experts.gate_up_proj", "experts.down_proj"])

    _shard_tensor_experts(model.experts, 2, 4)

    assert model.experts.gate_up_proj.shape[0] == 2
    assert model.experts.down_proj.shape[0] == 2
    for wrapper in manager.wrappers:
        assert wrapper.num_experts == 2
        assert all(param.shape[0] == 2 and param.is_meta for param in wrapper.lora_A.values())
        assert all(param.shape[0] == 2 and param.is_meta for param in wrapper.lora_B.values())


def test_target_parameter_slot_reset_uses_materialized_ep_local_snapshot():
    _ensure_dummy_zmq()
    from peft import LoraConfig
    from twinkle.model.multi_lora_target_parameters import TargetParameterLoraManager
    from twinkle.model.transformers.moe.expert_parallel import _shard_tensor_experts

    torch.manual_seed(0)
    model = nn.Module()
    model.experts = _FakeTensorExperts()
    manager = TargetParameterLoraManager(max_loras=1, max_r=4)
    targets = ["experts.gate_up_proj", "experts.down_proj"]
    manager.patch(model, targets)
    manager.acquire(
        "tenant_a",
        "lora_0",
        LoraConfig(r=2, lora_alpha=4, target_modules=[], target_parameters=targets),
    )

    _shard_tensor_experts(model.experts, 2, 4)
    manager.save_initial_weights()
    initial_a = [wrapper.lora_A["lora_0"].detach().clone() for wrapper in manager.wrappers]

    with torch.no_grad():
        for wrapper in manager.wrappers:
            wrapper.lora_A["lora_0"].add_(1)
            wrapper.lora_B["lora_0"].add_(1)
    manager.release("tenant_a")

    for wrapper, expected_a in zip(manager.wrappers, initial_a):
        assert torch.equal(wrapper.lora_A["lora_0"], expected_a)
        assert torch.count_nonzero(wrapper.lora_B["lora_0"]) == 0


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason="Need 4 GPUs")
def test_ep_fsdp_multi_lora_target_parameter_checkpoint_smoke():
    pytest.skip("Run this smoke in the DSV4 EP/FSDP integration environment with a local model fixture.")
