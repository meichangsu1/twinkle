from typing import Literal, Optional

import os
import torch
import torch.distributed as dist

from ..sequence_parallel import _gather_attention_mask_for_sp, _gather_position_ids_for_sp


def _trace_collective(
    event: str,
    sequence_parallel,
    tensor: Optional[torch.Tensor] = None,
    *,
    label: Optional[str] = None,
    dim: Optional[int] = None,
    grad_mode: Optional[str] = None,
) -> None:
    if os.environ.get('QWEN35_SP_LINEAR_STRICT_TRACE', '0') != '1':
        return

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else int(os.environ.get('RANK', '-1'))
    sp_group = getattr(sequence_parallel, '_sp_group', None)
    sp_rank = dist.get_rank(sp_group) if sp_group is not None and dist.is_available() and dist.is_initialized() else 0
    payload = {
        'rank': rank,
        'sp_rank': sp_rank,
        'event': event,
    }
    if label is not None:
        payload['label'] = label
    if dim is not None:
        payload['dim'] = dim
    if grad_mode is not None:
        payload['grad_mode'] = grad_mode
    if tensor is not None:
        payload['shape'] = tuple(int(x) for x in tensor.shape)
        payload['dtype'] = str(tensor.dtype)
        payload['device'] = str(tensor.device)
    text = ' '.join(f'{key}={value}' for key, value in payload.items())
    print(f'[qwen35_strict_collective] {text}', flush=True)


def _reduce_scatter_along_dim(
    tensor: torch.Tensor,
    scatter_dim: int,
    sequence_parallel,
) -> torch.Tensor:
    if sequence_parallel.world_size == 1:
        return tensor

    sp_group = sequence_parallel._sp_group
    sp_world_size = sequence_parallel.sp_world_size
    scatter_dim = scatter_dim if scatter_dim >= 0 else tensor.dim() + scatter_dim
    if scatter_dim < 0 or scatter_dim >= tensor.dim():
        raise ValueError(f'Invalid scatter_dim={scatter_dim} for tensor dim={tensor.dim()}')

    tensor = tensor.contiguous().movedim(scatter_dim, 0).contiguous()
    if tensor.shape[0] % sp_world_size != 0:
        raise ValueError(
            f'The dimension to reduce_scatter ({tensor.shape[0]}) is not a multiple of sp world size ({sp_world_size}).')

    local_size = tensor.shape[0] // sp_world_size
    output = torch.empty((local_size, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
    if hasattr(dist, 'reduce_scatter_tensor'):
        dist.reduce_scatter_tensor(output, tensor, group=sp_group)
    else:
        chunks = list(torch.split(tensor, local_size, dim=0))
        dist.reduce_scatter(output, chunks, group=sp_group)
    return output.movedim(0, scatter_dim).contiguous()


class _GatherFromSequenceParallelRegion(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        input_: torch.Tensor,
        sequence_parallel,
        dim: int,
        output_grad_mode: Literal['split', 'reduce_scatter'],
        trace_label: Optional[str],
    ):
        ctx.sequence_parallel = sequence_parallel
        ctx.dim = dim
        ctx.output_grad_mode = output_grad_mode
        ctx.trace_label = trace_label
        _trace_collective(
            'gather_forward',
            sequence_parallel,
            input_,
            label=trace_label,
            dim=dim,
            grad_mode=output_grad_mode,
        )
        return sequence_parallel.gather(
            input_.contiguous(),
            dim=dim,
            position_ids=sequence_parallel.real_position_ids,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        _trace_collective(
            'gather_backward:before',
            ctx.sequence_parallel,
            grad_output,
            label=ctx.trace_label,
            dim=ctx.dim,
            grad_mode=ctx.output_grad_mode,
        )
        if ctx.output_grad_mode == 'reduce_scatter':
            grad_input = _reduce_scatter_along_dim(grad_output.contiguous(), ctx.dim, ctx.sequence_parallel)
        else:
            grad_input = ctx.sequence_parallel.split(
                grad_output.contiguous(),
                dim=ctx.dim,
                position_ids=ctx.sequence_parallel.real_position_ids,
            )
        _trace_collective(
            'gather_backward:after',
            ctx.sequence_parallel,
            grad_input,
            label=ctx.trace_label,
            dim=ctx.dim,
            grad_mode=ctx.output_grad_mode,
        )
        return grad_input.contiguous(), None, None, None, None


class _ScatterToSequenceParallelRegion(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_: torch.Tensor, sequence_parallel, dim: int, trace_label: Optional[str]):
        ctx.sequence_parallel = sequence_parallel
        ctx.dim = dim
        ctx.trace_label = trace_label
        _trace_collective('scatter_forward', sequence_parallel, input_, label=trace_label, dim=dim)
        return sequence_parallel.split(
            input_.contiguous(),
            dim=dim,
            position_ids=sequence_parallel.real_position_ids,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        _trace_collective('scatter_backward:before', ctx.sequence_parallel, grad_output, label=ctx.trace_label, dim=ctx.dim)
        grad_input = ctx.sequence_parallel.gather(
            grad_output.contiguous(),
            dim=ctx.dim,
            position_ids=ctx.sequence_parallel.real_position_ids,
        )
        _trace_collective('scatter_backward:after', ctx.sequence_parallel, grad_input, label=ctx.trace_label, dim=ctx.dim)
        return grad_input.contiguous(), None, None, None


def gather_from_sequence_parallel_region(
    input_: torch.Tensor,
    sequence_parallel,
    dim: int,
    output_grad_mode: Literal['split', 'reduce_scatter'] = 'split',
    trace_label: Optional[str] = None,
) -> torch.Tensor:
    return _GatherFromSequenceParallelRegion.apply(input_, sequence_parallel, dim, output_grad_mode, trace_label)


def scatter_to_sequence_parallel_region(
    input_: torch.Tensor,
    sequence_parallel,
    dim: int,
    trace_label: Optional[str] = None,
) -> torch.Tensor:
    return _ScatterToSequenceParallelRegion.apply(input_, sequence_parallel, dim, trace_label)


def gather_attention_mask_from_sequence_parallel_region(
    attention_mask: Optional[torch.Tensor],
    sequence_parallel,
    local_seq_len: int,
) -> Optional[torch.Tensor]:
    return _gather_attention_mask_for_sp(
        attention_mask,
        local_seq_len,
        sequence_parallel.sp_world_size,
        sequence_parallel._sp_group,
    )


def gather_cache_position_from_sequence_parallel_region(
    cache_position: Optional[torch.Tensor],
    sequence_parallel,
) -> Optional[torch.Tensor]:
    return _gather_position_ids_for_sp(
        cache_position,
        sequence_parallel.sp_world_size,
        sequence_parallel._sp_group,
    )
