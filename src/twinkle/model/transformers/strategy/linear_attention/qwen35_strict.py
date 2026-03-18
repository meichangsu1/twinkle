import os
from typing import Optional

import torch


class _GatherForwardSplitBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, sequence_parallel, dim: int):
        ctx.sequence_parallel = sequence_parallel
        ctx.dim = dim
        return sequence_parallel.gather(
            tensor.contiguous(),
            dim=dim,
            position_ids=sequence_parallel.real_position_ids,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = ctx.sequence_parallel.split(
            grad_output.contiguous(),
            dim=ctx.dim,
            position_ids=ctx.sequence_parallel.real_position_ids,
        )
        return grad_input.contiguous(), None, None


class _SplitForwardGatherBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, sequence_parallel, dim: int):
        ctx.sequence_parallel = sequence_parallel
        ctx.dim = dim
        return sequence_parallel.split(
            tensor.contiguous(),
            dim=dim,
            position_ids=sequence_parallel.real_position_ids,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = ctx.sequence_parallel.gather(
            grad_output.contiguous(),
            dim=ctx.dim,
            position_ids=ctx.sequence_parallel.real_position_ids,
        )
        return grad_input.contiguous(), None, None


def _gather_optional_tensor(
    tensor: Optional[torch.Tensor],
    sequence_parallel,
    local_seq_len: int,
) -> Optional[torch.Tensor]:
    if tensor is None or not torch.is_tensor(tensor):
        return tensor

    gather_dims = []
    if tensor.dim() == 1:
        if tensor.shape[0] == local_seq_len:
            gather_dims.append(0)
    else:
        if tensor.shape[-1] == local_seq_len:
            gather_dims.append(tensor.dim() - 1)
        if tensor.dim() >= 3 and tensor.shape[-2] == local_seq_len:
            gather_dims.append(tensor.dim() - 2)
        elif tensor.dim() == 2 and tensor.shape[0] == local_seq_len:
            gather_dims.append(0)

    dedup_dims = []
    for dim in gather_dims:
        if dim not in dedup_dims:
            dedup_dims.append(dim)

    output = tensor.contiguous()
    for dim in dedup_dims:
        output = sequence_parallel.gather(output, dim=dim, position_ids=sequence_parallel.real_position_ids)
    return output.contiguous()


class Qwen35StrictFullSeqHelper:

    def __init__(self):
        self.enabled = os.environ.get('QWEN35_SP_LINEAR_STRICT', '0') == '1'

    def validate_runtime(self, sequence_parallel) -> None:
        if not self.enabled:
            return
        device_mesh = getattr(sequence_parallel, 'device_mesh', None)
        fsdp_world_size = getattr(device_mesh, 'fsdp_world_size', None) if device_mesh is not None else None
        if fsdp_world_size is not None and fsdp_world_size > 1:
            raise RuntimeError(
                'SequenceParallel: Qwen3.5 strict full-seq is only supported without FSDP sharding. '
                'Use DDP/accelerate-style wrapping when QWEN35_SP_LINEAR_STRICT=1.'
            )

    def run(
        self,
        sequence_parallel,
        module: torch.nn.Module,
        origin_forward,
        origin_rule,
        origin_causal_conv1d_fn,
        hidden_states: torch.Tensor,
        cache_params=None,
        cache_position: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        self.validate_runtime(sequence_parallel)

        full_hidden_states = _GatherForwardSplitBackward.apply(hidden_states.contiguous(), sequence_parallel, 1)
        local_seq_len = hidden_states.shape[1]
        full_attention_mask = _gather_optional_tensor(attention_mask, sequence_parallel, local_seq_len)
        full_cache_position = _gather_optional_tensor(cache_position, sequence_parallel, local_seq_len)

        saved_rule = module.chunk_gated_delta_rule
        saved_causal_conv1d_fn = module.causal_conv1d_fn
        module.chunk_gated_delta_rule = origin_rule
        module.causal_conv1d_fn = origin_causal_conv1d_fn
        try:
            full_output = origin_forward(
                full_hidden_states,
                cache_params=cache_params,
                cache_position=full_cache_position,
                attention_mask=full_attention_mask,
            )
        finally:
            module.chunk_gated_delta_rule = saved_rule
            module.causal_conv1d_fn = saved_causal_conv1d_fn

        return _SplitForwardGatherBackward.apply(full_output.contiguous(), sequence_parallel, 1)
