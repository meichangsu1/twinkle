import importlib
import os
import torch
import torch.distributed as dist
from dataclasses import asdict, dataclass, is_dataclass
from functools import partial
from transformers import PreTrainedTokenizer
from typing import Any, Dict, Optional, Tuple, Union

from twinkle.utils import DeviceMesh
from twinkle.utils.transformers_utils import get_llm_model


def _sp_linear_collective_debug_enabled() -> bool:
    return os.environ.get('QWEN35_SP_LINEAR_DEBUG_COLLECTIVES', '1') == '1'


def _sp_linear_collective_debug(message: str, sp_group: Optional[dist.ProcessGroup] = None) -> None:
    if not _sp_linear_collective_debug_enabled():
        return
    global_rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    group_rank = None
    if sp_group is not None and dist.is_available() and dist.is_initialized():
        group_rank = dist.get_rank(sp_group)
    prefix = f'[QWEN35_SP_LINEAR_DEBUG][global_rank={global_rank}'
    if group_rank is not None:
        prefix += f', sp_rank={group_rank}'
    prefix += ']'
    print(f'{prefix} {message}', flush=True)


def _sp_qwen35_decoder_debug_enabled() -> bool:
    return os.environ.get('QWEN35_SP_DEBUG_DECODER_LAYERS', '0') == '1'


def get_config_attr(config, key, default=None):
    return getattr(config, key, default)


def _extract_text_position_ids(position_ids: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Extract text-aligned position ids as [B, S] from 1D/2D/3D/4D inputs."""
    if position_ids is None or not torch.is_tensor(position_ids):
        return None
    if position_ids.dim() == 1:
        return position_ids.unsqueeze(0)
    if position_ids.dim() == 2:
        return position_ids
    # Channel-first multimodal layouts, e.g. [3,B,S] / [4,B,S].
    return position_ids[0]


def _split_position_ids_for_sp(
    position_ids: Optional[torch.Tensor],
    sp_world_size: Optional[int],
    sp_group: Optional[dist.ProcessGroup],
) -> Optional[torch.Tensor]:
    if position_ids is None or not torch.is_tensor(position_ids) or sp_world_size is None or sp_world_size <= 1:
        return position_ids
    rank = dist.get_rank(sp_group) if sp_group is not None else 0
    seq_len = position_ids.size(-1)
    if seq_len % sp_world_size != 0:
        raise ValueError(f'position_ids seq_len ({seq_len}) must be divisible by sp_world_size ({sp_world_size}).')
    local_seq = seq_len // sp_world_size
    return torch.split(position_ids, local_seq, dim=-1)[rank].contiguous()


def _gather_position_ids_for_sp(
    position_ids: Optional[torch.Tensor],
    sp_world_size: Optional[int],
    sp_group: Optional[dist.ProcessGroup],
) -> Optional[torch.Tensor]:
    if position_ids is None or not torch.is_tensor(position_ids) or sp_world_size is None or sp_world_size <= 1:
        return position_ids
    position_ids = position_ids.contiguous()
    local_shape = tuple(position_ids.shape)
    gather_shape = (local_shape[0] * sp_world_size, *local_shape[1:])
    gathered = torch.empty(gather_shape, dtype=position_ids.dtype, device=position_ids.device)
    dist.all_gather_into_tensor(gathered, position_ids, group=sp_group)
    return torch.cat(gathered.split(local_shape[0], dim=0), dim=-1).contiguous()


def _gather_tensor_along_dim_for_sp(
    tensor: Optional[torch.Tensor],
    gather_dim: int,
    sp_world_size: Optional[int],
    sp_group: Optional[dist.ProcessGroup],
) -> Optional[torch.Tensor]:
    if tensor is None or not torch.is_tensor(tensor) or sp_world_size is None or sp_world_size <= 1:
        return tensor
    gather_dim = gather_dim if gather_dim >= 0 else tensor.dim() + gather_dim
    if gather_dim < 0 or gather_dim >= tensor.dim():
        raise ValueError(f'Invalid gather_dim={gather_dim} for tensor dim={tensor.dim()}')
    tensor = tensor.contiguous()
    tensor = tensor.movedim(gather_dim, 0).contiguous()
    local_size = tensor.shape[0]
    gather_shape = (local_size * sp_world_size, *tensor.shape[1:])
    gathered = torch.empty(gather_shape, dtype=tensor.dtype, device=tensor.device)
    dist.all_gather_into_tensor(gathered, tensor, group=sp_group)
    gathered = torch.cat(gathered.split(local_size, dim=0), dim=0).contiguous()
    return gathered.movedim(0, gather_dim).contiguous()


def _get_attention_mask_sequence_dims(
    attention_mask: Optional[torch.Tensor],
    seq_len: int,
) -> Tuple[int, ...]:
    if attention_mask is None or not torch.is_tensor(attention_mask):
        return tuple()
    if attention_mask.dim() == 0:
        return tuple()
    if attention_mask.dim() == 1:
        return (0,) if attention_mask.shape[0] == seq_len else tuple()
    if attention_mask.dim() == 2:
        if attention_mask.shape[1] == seq_len:
            return (1,)
        if attention_mask.shape[0] == seq_len:
            return (0,)
        return tuple()

    dims = []
    if attention_mask.shape[-1] == seq_len:
        dims.append(attention_mask.dim() - 1)
    if attention_mask.dim() >= 3 and attention_mask.shape[-2] == seq_len:
        dims.append(attention_mask.dim() - 2)
    if not dims and attention_mask.shape[1] == seq_len:
        dims.append(1)
    # Keep order stable and remove duplicates.
    dedup = []
    for dim in dims:
        if dim not in dedup:
            dedup.append(dim)
    return tuple(dedup)


def _gather_attention_mask_for_sp(
    attention_mask: Optional[torch.Tensor],
    local_seq_len: int,
    sp_world_size: Optional[int],
    sp_group: Optional[dist.ProcessGroup],
) -> Optional[torch.Tensor]:
    if attention_mask is None or not torch.is_tensor(attention_mask):
        return attention_mask
    if sp_world_size is None or sp_world_size <= 1:
        return attention_mask

    gathered = attention_mask.contiguous()
    seq_dims = _get_attention_mask_sequence_dims(gathered, local_seq_len)
    for dim in seq_dims:
        gathered = _gather_tensor_along_dim_for_sp(gathered, dim, sp_world_size, sp_group)
    return gathered.contiguous()


def _assert_attention_mask_matches_sequence(
    attention_mask: Optional[torch.Tensor],
    expected_seq_len: int,
) -> None:
    if attention_mask is None or not torch.is_tensor(attention_mask):
        return
    seq_dims = _get_attention_mask_sequence_dims(attention_mask, expected_seq_len)
    if not seq_dims:
        raise ValueError(
            'SequenceParallel: gathered attention_mask shape is incompatible with attention sequence length. '
            f'attention_mask.shape={tuple(attention_mask.shape)}, expected_seq_len={expected_seq_len}.')


def get_cu_seqlens_from_position_ids(position_ids: torch.LongTensor):
    text_position_ids = _extract_text_position_ids(position_ids)
    if text_position_ids is None:
        raise ValueError('position_ids must be a tensor when computing cu_seqlens.')
    # Packed mode is expected to operate on batch-size 1 streams.
    position_ids = text_position_ids[0]
    seq_start_indices = torch.where(position_ids == 0)[0]
    if seq_start_indices.numel() == 0:
        return torch.tensor([0, len(position_ids)], device=position_ids.device)
    seq_end_indices = torch.cat([seq_start_indices[1:], torch.tensor([len(position_ids)], device=position_ids.device)])
    seq_lengths = seq_end_indices - seq_start_indices
    cu_seqlens = torch.cumsum(torch.cat([torch.tensor([0], device=position_ids.device), seq_lengths]), dim=0)
    return cu_seqlens


def _normalize_flash_position_ids(position_ids: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """FlashAttention packed detection expects text-aligned [B, S] position ids."""
    if position_ids is None or not torch.is_tensor(position_ids):
        return position_ids
    return _extract_text_position_ids(position_ids)


def _sp_prev_next_rank(
    sp_group: Optional[dist.ProcessGroup],
) -> Tuple[int, int, Optional[int], Optional[int]]:
    """Return (group_rank, group_world_size, prev_global_rank, next_global_rank)."""
    if sp_group is None:
        return 0, 1, None, None
    sp_rank = dist.get_rank(sp_group)
    sp_world_size = dist.get_world_size(sp_group)
    prev_rank = dist.get_global_rank(sp_group, sp_rank - 1) if sp_rank > 0 else None
    next_rank = dist.get_global_rank(sp_group, sp_rank + 1) if sp_rank + 1 < sp_world_size else None
    return sp_rank, sp_world_size, prev_rank, next_rank


class _QwenLinearStateParallelFn(torch.autograd.Function):
    """State-parallel autograd for Qwen3.5 linear attention chunk rule.

    Forward: recv previous recurrent state -> run local chunk rule -> send final recurrent state.
    Backward: recv grad(final_state) -> recompute local chunk rule -> send grad(initial_state).
    """

    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        sp_group: Optional[dist.ProcessGroup],
        chunk_rule,
        chunk_size: Optional[int],
        use_qk_l2norm_in_kernel: bool,
    ) -> torch.Tensor:
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        g = g.contiguous()
        beta = beta.contiguous()

        ctx.sp_group = sp_group
        ctx.chunk_rule = chunk_rule
        ctx.chunk_size = int(chunk_size) if chunk_size is not None else None
        ctx.use_qk_l2norm_in_kernel = bool(use_qk_l2norm_in_kernel)

        sp_rank, sp_world_size, prev_rank, next_rank = _sp_prev_next_rank(sp_group)
        ctx.sp_rank = sp_rank
        ctx.sp_world_size = sp_world_size
        ctx.prev_rank = prev_rank
        ctx.next_rank = next_rank

        state_shape = (query.shape[0], query.shape[2], query.shape[3], value.shape[3])
        initial_state = torch.zeros(state_shape, dtype=torch.float32, device=query.device)
        if sp_group is not None and sp_world_size > 1 and prev_rank is not None:
            dist.recv(initial_state, src=prev_rank, group=sp_group)
        initial_state = initial_state.contiguous()

        chunk_kwargs = {
            'g': g,
            'beta': beta,
            'initial_state': (initial_state if prev_rank is not None else None),
            'output_final_state': True,
            'use_qk_l2norm_in_kernel': ctx.use_qk_l2norm_in_kernel,
        }
        if ctx.chunk_size is not None:
            chunk_kwargs['chunk_size'] = ctx.chunk_size
        local_output, final_state = chunk_rule(query, key, value, **chunk_kwargs)
        local_output = local_output.contiguous()
        if sp_group is not None and sp_world_size > 1 and next_rank is not None:
            dist.send(final_state.contiguous().to(torch.float32), dst=next_rank, group=sp_group)

        # Always save a tensor initial_state so backward can re-run deterministically.
        ctx.save_for_backward(query, key, value, g, beta, initial_state)
        return local_output

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ):
        query, key, value, g, beta, initial_state = ctx.saved_tensors
        grad_output = grad_output.contiguous()

        grad_final_state = torch.zeros_like(initial_state, dtype=torch.float32, device=grad_output.device)
        if ctx.sp_group is not None and ctx.sp_world_size > 1 and ctx.next_rank is not None:
            dist.recv(grad_final_state, src=ctx.next_rank, group=ctx.sp_group)
        grad_final_state = grad_final_state.contiguous()

        query_ = query.detach().requires_grad_(True)
        key_ = key.detach().requires_grad_(True)
        value_ = value.detach().requires_grad_(True)
        g_ = g.detach().requires_grad_(True)
        beta_ = beta.detach().requires_grad_(True)
        initial_state_ = initial_state.detach().requires_grad_(True)

        with torch.enable_grad():
            chunk_kwargs = {
                'g': g_,
                'beta': beta_,
                'initial_state': (initial_state_ if ctx.prev_rank is not None else None),
                'output_final_state': True,
                'use_qk_l2norm_in_kernel': ctx.use_qk_l2norm_in_kernel,
            }
            if ctx.chunk_size is not None:
                chunk_kwargs['chunk_size'] = ctx.chunk_size
            local_output, final_state = ctx.chunk_rule(query_, key_, value_, **chunk_kwargs)
        local_output = local_output.contiguous()
        final_state = final_state.contiguous()

        grad_args = [query_, key_, value_, g_, beta_]
        if ctx.prev_rank is not None:
            grad_args.append(initial_state_)

        grad_outputs = [grad_output, grad_final_state.to(final_state.dtype)]
        grads = torch.autograd.grad(
            outputs=(local_output, final_state),
            inputs=tuple(grad_args),
            grad_outputs=tuple(grad_outputs),
            allow_unused=True,
        )

        grad_query, grad_key, grad_value, grad_g, grad_beta = grads[:5]
        grad_initial_state = grads[5] if len(grads) > 5 else None

        if ctx.sp_group is not None and ctx.sp_world_size > 1 and ctx.prev_rank is not None:
            if grad_initial_state is None:
                grad_initial_state = torch.zeros_like(initial_state, dtype=torch.float32, device=grad_output.device)
            dist.send(grad_initial_state.contiguous().to(torch.float32), dst=ctx.prev_rank, group=ctx.sp_group)

        return grad_query, grad_key, grad_value, grad_g, grad_beta, None, None, None, None


class _LeftHaloExchangeFn(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        halo: int,
        seq_dim: int,
        sp_group: Optional[dist.ProcessGroup],
        tag_base: int,
    ) -> torch.Tensor:
        ctx.halo = int(halo)
        ctx.seq_dim = seq_dim if seq_dim >= 0 else tensor.dim() + seq_dim
        ctx.sp_group = sp_group
        ctx.forward_tag = int(tag_base) * 2
        ctx.backward_tag = ctx.forward_tag + 1
        if ctx.halo <= 0 or sp_group is None or dist.get_world_size(sp_group) <= 1:
            ctx.local_seq_len = tensor.size(ctx.seq_dim)
            ctx.prev_rank = None
            ctx.next_rank = None
            return tensor

        ctx.local_seq_len = tensor.size(ctx.seq_dim)
        if ctx.local_seq_len < ctx.halo:
            raise RuntimeError(
                'SequenceParallel: local sequence length must be at least halo size for Qwen3.5 conv halo exchange. '
                f'local_seq_len={ctx.local_seq_len}, halo={ctx.halo}. '
                'Use QWEN35_SP_LINEAR_STRICT=1 for very short local sequences.'
            )

        _, _, prev_rank, next_rank = _sp_prev_next_rank(sp_group)
        ctx.prev_rank = prev_rank
        ctx.next_rank = next_rank

        halo_shape = list(tensor.shape)
        halo_shape[ctx.seq_dim] = ctx.halo
        left_halo = tensor.new_zeros(halo_shape)
        ops = []
        if prev_rank is not None:
            ops.append(dist.P2POp(dist.irecv, left_halo, prev_rank, sp_group, ctx.forward_tag))
        if next_rank is not None:
            tail = tensor.narrow(ctx.seq_dim, ctx.local_seq_len - ctx.halo, ctx.halo).contiguous()
            ops.append(dist.P2POp(dist.isend, tail, next_rank, sp_group, ctx.forward_tag))
        if ops:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        return torch.cat([left_halo, tensor], dim=ctx.seq_dim).contiguous()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.halo <= 0 or ctx.sp_group is None or dist.get_world_size(ctx.sp_group) <= 1:
            return grad_output, None, None, None

        grad_left = grad_output.narrow(ctx.seq_dim, 0, ctx.halo).contiguous()
        grad_local = grad_output.narrow(ctx.seq_dim, ctx.halo, ctx.local_seq_len).contiguous()

        grad_from_next = None
        ops = []
        if ctx.prev_rank is not None:
            ops.append(dist.P2POp(dist.isend, grad_left, ctx.prev_rank, ctx.sp_group, ctx.backward_tag))
        if ctx.next_rank is not None:
            halo_shape = list(grad_local.shape)
            halo_shape[ctx.seq_dim] = ctx.halo
            grad_from_next = grad_local.new_zeros(halo_shape)
            ops.append(dist.P2POp(dist.irecv, grad_from_next, ctx.next_rank, ctx.sp_group, ctx.backward_tag))
        if ops:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        if grad_from_next is not None:
            grad_local.narrow(ctx.seq_dim, ctx.local_seq_len - ctx.halo, ctx.halo).add_(grad_from_next)

        return grad_local.contiguous(), None, None, None, None


def _get_raw_data_world_size(device_mesh: DeviceMesh) -> int:
    dp_world_size = device_mesh.dp_world_size or 1
    fsdp_world_size = device_mesh.fsdp_world_size or 1
    if dp_world_size <= 0:
        dp_world_size = 1
    if fsdp_world_size <= 0:
        fsdp_world_size = 1
    return dp_world_size * fsdp_world_size


def _get_raw_data_rank(device_mesh: DeviceMesh, rank: int) -> Optional[int]:
    coord = device_mesh._get_coord_for_rank(rank)
    if coord is None:
        return None

    dp_rank = None
    fsdp_rank = None
    if device_mesh.has_dim('dp'):
        dp_rank = coord[device_mesh._get_dim_index('dp')]
    if device_mesh.has_dim('fsdp'):
        fsdp_rank = coord[device_mesh._get_dim_index('fsdp')]

    fsdp_world_size = device_mesh.fsdp_world_size
    data_rank = dp_rank if dp_rank is not None else None
    if fsdp_world_size is not None and fsdp_world_size > 1:
        if dp_rank is not None and fsdp_rank is not None:
            data_rank = dp_rank * fsdp_world_size + fsdp_rank
        elif fsdp_rank is not None:
            data_rank = fsdp_rank

    if data_rank is None:
        data_rank = 0
    return int(data_rank)


def _get_sp_group_from_device_mesh(
    device_mesh: Optional[DeviceMesh],
    sp_size: int,
) -> Optional[dist.ProcessGroup]:
    """Return the SP (sequence-parallel) process group for the current rank.

    If the mesh defines an explicit "sp" dimension, use it directly. Otherwise,
    derive SP groups by chunking data-parallel ranks (dp/fsdp) while keeping
    all other mesh dimensions (tp/pp/ep/etc.) fixed.

    Example (no explicit "sp" dim, sp_size=2):
        mesh_dim_names = ("dp", "fsdp", "tp")
        mesh = np.arange(8).reshape(2, 2, 2)
        # coords are (dp, fsdp, tp). dp/fsdp are "data" dims; tp is "non-data".
        # raw_data_rank = dp * fsdp_world_size + fsdp, so ranges [0..3].
        # group_id = raw_data_rank // sp_size partitions data ranks into 2 groups.
        #
        # For tp=0:
        #   data ranks 0,1 -> group_id=0  => ranks at coords:
        #     (dp=0,fsdp=0,tp=0) -> rank 0
        #     (dp=0,fsdp=1,tp=0) -> rank 2
        #   data ranks 2,3 -> group_id=1  => ranks at coords:
        #     (dp=1,fsdp=0,tp=0) -> rank 4
        #     (dp=1,fsdp=1,tp=0) -> rank 6
        #
        # For tp=1:
        #   data ranks 0,1 -> group_id=0  => ranks at coords:
        #     (dp=0,fsdp=0,tp=1) -> rank 1
        #     (dp=0,fsdp=1,tp=1) -> rank 3
        #   data ranks 2,3 -> group_id=1  => ranks at coords:
        #     (dp=1,fsdp=0,tp=1) -> rank 5
        #     (dp=1,fsdp=1,tp=1) -> rank 7
        #
        # Final SP groups (keyed by (group_id, non_data_key)):
        #   (0, (tp=0)) -> [0, 2]
        #   (1, (tp=0)) -> [4, 6]
        #   (0, (tp=1)) -> [1, 3]
        #   (1, (tp=1)) -> [5, 7]
        #
        # Each SP group has size=2 and never crosses tp.
    """
    if device_mesh is None or sp_size <= 1:
        return None
    if device_mesh.has_dim('sp'):
        return device_mesh.create_process_group(['sp'])
    if not dist.is_available() or not dist.is_initialized():
        return None

    raw_data_world_size = _get_raw_data_world_size(device_mesh)
    if raw_data_world_size % sp_size != 0:
        raise ValueError(f'data_world_size ({raw_data_world_size}) must be divisible by sp_size ({sp_size}).')

    rank = dist.get_rank()
    ref_coord = device_mesh._get_coord_for_rank(rank)
    if ref_coord is None:
        return None

    non_data_indices = []
    if device_mesh.mesh_dim_names is not None:
        for i, name in enumerate(device_mesh.mesh_dim_names):
            if name in ('dp', 'fsdp'):
                continue
            non_data_indices.append(i)

    # Group ranks by (data-parallel chunk, non-data mesh coordinates).
    groups: Dict[Tuple[int, Tuple[int, ...]], list[int]] = {}
    for r in device_mesh.mesh.flatten().tolist():
        r = int(r)
        coord = device_mesh._get_coord_for_rank(r)
        if coord is None:
            continue
        raw_rank = _get_raw_data_rank(device_mesh, r)
        if raw_rank is None:
            continue
        group_id = raw_rank // sp_size
        non_data_key = tuple(coord[i] for i in non_data_indices)
        key = (group_id, non_data_key)
        groups.setdefault(key, []).append(r)

    group_list = []
    for key, ranks in groups.items():
        ranks = sorted(ranks)
        if len(ranks) != sp_size:
            raise ValueError(f'SP group size mismatch for key={key}: expected {sp_size}, got {len(ranks)}')
        group_list.append((key, ranks))

    group_list.sort(key=lambda item: item[0])

    sp_group = None
    for _, ranks in group_list:
        pg = dist.new_group(ranks=ranks)
        if rank in ranks:
            sp_group = pg
    return sp_group


class GatherLoss(torch.autograd.Function):
    """Gather loss from sequence group."""

    @staticmethod
    def forward(ctx, loss, labels, gather_idx=None, position_ids=None):
        """
        Args:
            loss: loss tensor after splitting
            labels: labels tensor after splitting
            gather_idx: gather the tensors on this dim
        """
        ctx.scatter_shape = loss.shape[gather_idx or 0]
        ctx.gather_idx = gather_idx or 0
        if position_ids is not None:
            position_ids = sequence_parallel.pad(position_ids, padding_value=-1, position_ids=position_ids)
        ctx.position_ids = position_ids
        # Gather split losses/labels to compute aux losses on full sequence length.
        output = sequence_parallel.gather(loss, dim=ctx.gather_idx, position_ids=position_ids)
        if labels is not None:
            labels_output = sequence_parallel.gather(labels, dim=ctx.gather_idx, position_ids=position_ids)
        else:
            labels_output = None
        return output, labels_output

    @staticmethod
    def backward(ctx, *grad_output):
        # Split grads back to local sequence chunk.
        _grad = grad_output[0]
        if sequence_parallel.world_size > 1 and sequence_parallel._sp_group is not None:
            # Gather replicates the sequence dimension across SP ranks. Scale once here
            # so downstream FSDP avg does not shrink this path by an extra SP factor.
            _grad = _grad * sequence_parallel.world_size
            _grad = sequence_parallel.split(_grad, dim=ctx.gather_idx, position_ids=ctx.position_ids).contiguous()
        return _grad, None, None, None


class _GatherForwardSplitBackward(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        gather_dim: int,
        sp_world_size: Optional[int],
        sp_group: Optional[dist.ProcessGroup],
        debug_label: Optional[str],
    ) -> torch.Tensor:
        ctx.gather_dim = gather_dim
        ctx.sp_world_size = sp_world_size
        ctx.sp_group = sp_group
        ctx.debug_label = debug_label or 'gather_forward_split_backward'
        if sp_world_size is None or sp_world_size <= 1 or sp_group is None:
            return tensor
        _sp_linear_collective_debug(
            f'{ctx.debug_label}: before forward gather dim={gather_dim}, shape={tuple(tensor.shape)}',
            sp_group,
        )
        output = _gather_tensor_along_dim_for_sp(tensor, gather_dim, sp_world_size, sp_group)
        _sp_linear_collective_debug(
            f'{ctx.debug_label}: after forward gather shape={tuple(output.shape)}',
            sp_group,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.sp_world_size is None or ctx.sp_world_size <= 1 or ctx.sp_group is None:
            return grad_output, None, None, None, None
        _sp_linear_collective_debug(
            f'{ctx.debug_label}: before backward split dim={ctx.gather_dim}, shape={tuple(grad_output.shape)}',
            ctx.sp_group,
        )
        rank = dist.get_rank(ctx.sp_group)
        gather_dim = ctx.gather_dim if ctx.gather_dim >= 0 else grad_output.dim() + ctx.gather_dim
        dim_size = grad_output.size(gather_dim)
        if dim_size % ctx.sp_world_size != 0:
            raise ValueError(
                f'Cannot split gathered grad_output size {dim_size} on dim {gather_dim} by '
                f'sp_world_size={ctx.sp_world_size}.'
            )
        local_size = dim_size // ctx.sp_world_size
        grad_input = torch.split(grad_output.contiguous(), local_size, dim=gather_dim)[rank].contiguous()
        _sp_linear_collective_debug(
            f'{ctx.debug_label}: after backward split shape={tuple(grad_input.shape)}',
            ctx.sp_group,
        )
        return grad_input, None, None, None, None


class _SplitForwardGatherBackward(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        split_dim: int,
        sp_world_size: Optional[int],
        sp_group: Optional[dist.ProcessGroup],
        debug_label: Optional[str],
    ) -> torch.Tensor:
        ctx.split_dim = split_dim
        ctx.sp_world_size = sp_world_size
        ctx.sp_group = sp_group
        ctx.debug_label = debug_label or 'split_forward_gather_backward'
        if sp_world_size is None or sp_world_size <= 1 or sp_group is None:
            return tensor
        _sp_linear_collective_debug(
            f'{ctx.debug_label}: before forward split dim={split_dim}, shape={tuple(tensor.shape)}',
            sp_group,
        )
        rank = dist.get_rank(sp_group)
        split_dim = split_dim if split_dim >= 0 else tensor.dim() + split_dim
        dim_size = tensor.size(split_dim)
        if dim_size % sp_world_size != 0:
            raise ValueError(
                f'Cannot split tensor size {dim_size} on dim {split_dim} by sp_world_size={sp_world_size}.'
            )
        local_size = dim_size // sp_world_size
        output = torch.split(tensor.contiguous(), local_size, dim=split_dim)[rank].contiguous()
        _sp_linear_collective_debug(
            f'{ctx.debug_label}: after forward split shape={tuple(output.shape)}',
            sp_group,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.sp_world_size is None or ctx.sp_world_size <= 1 or ctx.sp_group is None:
            return grad_output, None, None, None, None
        _sp_linear_collective_debug(
            f'{ctx.debug_label}: before backward gather dim={ctx.split_dim}, shape={tuple(grad_output.shape)}',
            ctx.sp_group,
        )
        grad_input = _gather_tensor_along_dim_for_sp(
            grad_output.contiguous(),
            ctx.split_dim,
            ctx.sp_world_size,
            ctx.sp_group,
        )
        _sp_linear_collective_debug(
            f'{ctx.debug_label}: after backward gather shape={tuple(grad_input.shape)}',
            ctx.sp_group,
        )
        return grad_input, None, None, None, None


# Code borrowed from deepspeed, here is why:
# 1. Reduce the dependency
# 2. The original code is complex
def _generate_layout_params(scatter_idx, seq_world_size, input):
    if scatter_idx < 2:
        bs, global_seq_len, num_local_head, head_dim = input.shape
        pre_all2all_inp_shape = [bs, seq_world_size, global_seq_len // seq_world_size, num_local_head, head_dim]
        pre_all2all_permute_idx = (1, 0, 2, 3, 4)

        post_all2all_permute_idx = (1, 2, 0, 3, 4)
        post_all2all_res_shape = [bs, global_seq_len // seq_world_size, seq_world_size * num_local_head, head_dim]
    else:
        bs, local_seq_len, num_total_head, head_dim = input.shape
        assert num_total_head % seq_world_size == 0, (f'Number of heads ({num_total_head}) must be divisible '
                                                      f'by the sequence parallel size ({seq_world_size})!')
        pre_all2all_inp_shape = [bs, local_seq_len, seq_world_size, num_total_head // seq_world_size, head_dim]
        pre_all2all_permute_idx = (2, 0, 1, 3, 4)

        post_all2all_permute_idx = (1, 0, 2, 3, 4)
        post_all2all_res_shape = [bs, seq_world_size * local_seq_len, num_total_head // seq_world_size, head_dim]

    return pre_all2all_permute_idx, pre_all2all_inp_shape, post_all2all_permute_idx, post_all2all_res_shape


def post_all2all(permute_idx, res_shape):
    """
    Post-processing function for `all2all` communication.
    """

    def post_func(input):
        if permute_idx is not None:
            input = input.permute(permute_idx).contiguous()
        output = input.reshape(res_shape).contiguous()

        return output

    return post_func


def pre_all2all_fun(permute_idx, inp_shape, input):
    """
    Pre-processing function for `all2all` communication.
    """
    input_t = input.reshape(inp_shape).contiguous()
    if permute_idx is not None:
        input_t = input_t.permute(permute_idx).contiguous()
    return input_t


def single_all_to_all(input, scatter_idx, gather_idx, group, **kwargs):
    seq_world_size = dist.get_world_size(group)
    num_heads = input.shape[2]
    if num_heads % seq_world_size != 0 and not scatter_idx < 2:
        raise NotImplementedError(f'num_heads {num_heads} cannot be split by sp world size {seq_world_size}')
    pre_all2all_permute_idx, pre_all2all_inp_shape, post_all2all_permute_idx, post_all2all_res_shape = (
        _generate_layout_params(scatter_idx, seq_world_size, input))

    input_t = pre_all2all_fun(pre_all2all_permute_idx, pre_all2all_inp_shape, input)

    post_all2all_fun = post_all2all(post_all2all_permute_idx, post_all2all_res_shape)
    output = torch.empty_like(input_t)
    dist.all_to_all_single(output, input_t, group=group)

    res = post_all2all_fun(output)
    return res


class _SeqAllToAll(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: torch.Tensor,
        scatter_idx: int,
        gather_idx: int,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        _sp_linear_collective_debug(
            f'seq_all_to_all: before forward scatter_idx={scatter_idx}, gather_idx={gather_idx}, '
            f'shape={tuple(input.shape)}',
            group,
        )
        res = single_all_to_all(input, scatter_idx, gather_idx, group)
        _sp_linear_collective_debug(
            f'seq_all_to_all: after forward shape={tuple(res.shape)}',
            group,
        )
        return res

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor) -> Tuple[None, torch.Tensor, None, None]:
        # Reverse scatter/gather in backward to match forward layout transform.
        _sp_linear_collective_debug(
            f'seq_all_to_all: before backward scatter_idx={ctx.gather_idx}, gather_idx={ctx.scatter_idx}, '
            f'shape={tuple(grad_output[0].shape)}',
            ctx.group,
        )
        return None, _SeqAllToAll.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx), None, None


class DistributedAttention(torch.nn.Module):

    def __init__(
        self,
        local_attention,
        sequence_parallel,
        scatter_idx: int = 2,
        gather_idx: int = 1,
    ) -> None:
        super().__init__()
        self.local_attn = local_attention
        self.sequence_parallel = sequence_parallel
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor, *args:
                Any, **kwargs) -> torch.Tensor:
        if self.sequence_parallel.world_size == 1:
            return self.local_attn(query, key, value, attention_mask, *args, **kwargs)

        _sp_linear_collective_debug(
            f'distributed_attention: enter query={tuple(query.shape)}, key={tuple(key.shape)}, '
            f'value={tuple(value.shape)}, attention_mask={None if attention_mask is None else tuple(attention_mask.shape)}',
            self.sequence_parallel._sp_group,
        )
        # All-to-all to assemble full sequence for attention, then split back after.
        if self.sequence_parallel.sp_world_size > 1:
            query_layer = _SeqAllToAll.apply(self.sequence_parallel._sp_group, query, self.scatter_idx, self.gather_idx)
            key_layer = _SeqAllToAll.apply(self.sequence_parallel._sp_group, key, self.scatter_idx, self.gather_idx)
            value_layer = _SeqAllToAll.apply(self.sequence_parallel._sp_group, value, self.scatter_idx, self.gather_idx)
        else:
            query_layer, key_layer, value_layer = query, key, value

        attention_mask = _gather_attention_mask_for_sp(
            attention_mask,
            local_seq_len=key.shape[1],
            sp_world_size=self.sequence_parallel.sp_world_size,
            sp_group=self.sequence_parallel._sp_group,
        )
        _assert_attention_mask_matches_sequence(attention_mask, expected_seq_len=key_layer.shape[1])

        position_ids = kwargs.pop('position_ids', None)
        if position_ids is not None:
            position_ids = _gather_position_ids_for_sp(
                position_ids,
                self.sequence_parallel.sp_world_size,
                self.sequence_parallel._sp_group,
            )

        context_layer = self.local_attn(
            query_layer, key_layer, value_layer, attention_mask, *args, position_ids=position_ids, **kwargs)

        if self.sequence_parallel.sp_world_size > 1:
            output = _SeqAllToAll.apply(self.sequence_parallel._sp_group, context_layer, self.gather_idx,
                                        self.scatter_idx)
        else:
            output = context_layer

        _sp_linear_collective_debug(
            f'distributed_attention: exit output={tuple(output.shape)}',
            self.sequence_parallel._sp_group,
        )
        return output


# main content copied from ms-swift
class SequenceParallel:

    _global_inited: bool = False

    def __init__(self):
        self.sp_world_size = None
        self.dp_world_size = None
        self.world_size = None
        self.model_dtype = None
        self.tokenizer = None
        self.device_mesh = None
        self._sp_group = None
        self.num_heads = None
        self.causal_mask_func = None
        self.has_qwen35_linear_attn = False
        self.qwen35_linear_strict_full_seq = os.environ.get('QWEN35_SP_LINEAR_STRICT', '0') == '1'
        self.qwen35_linear_conv_halo = os.environ.get('QWEN35_SP_LINEAR_CONV_HALO', '0') == '1'
        self.qwen35_linear_disable_gc = os.environ.get(
            'QWEN35_SP_LINEAR_DISABLE_GC',
            os.environ.get('QWEN35_SP_LINEAR_CONV_HALO_DISABLE_GC', '1'),
        ) == '1'
        self.qwen35_linear_conv_halo_disable_gc = os.environ.get('QWEN35_SP_LINEAR_CONV_HALO_DISABLE_GC', '1') == '1'
        self.extra_kwargs = {}

    @property
    def real_position_ids(self) -> torch.Tensor:
        """The real position ids, this is different from the position_ids in mrope"""
        return self.extra_kwargs.get('position_ids')

    @property
    def text_position_ids(self) -> Optional[torch.Tensor]:
        """Text-aligned position ids with shape [B, S]."""
        return self.extra_kwargs.get('text_position_ids')

    def _prepare_flash_attn(self, base_model: torch.nn.Module):
        try:
            from transformers import masking_utils

            _origin_flash_attention_mask = masking_utils.flash_attention_mask

            # Patch attention masks for SP: avoid masking when full sequence is reconstructed.
            def flash_attention_mask(batch_size,
                                     cache_position,
                                     kv_length,
                                     kv_offset=0,
                                     mask_function=masking_utils.causal_mask_function,
                                     attention_mask=None,
                                     **kwargs):
                if self.world_size == 1:
                    return _origin_flash_attention_mask(batch_size, cache_position, kv_length, kv_offset, mask_function,
                                                        attention_mask, **kwargs)
                if attention_mask is not None:
                    if attention_mask.all():
                        attention_mask = None

                return attention_mask

            masking_utils.flash_attention_mask = flash_attention_mask
            masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping['flash_attention_2'] = flash_attention_mask

            def sdpa_mask(batch_size, cache_position, kv_length, *args, **kwargs):
                if self.world_size == 1:
                    return masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping['sdpa_origin'](batch_size,
                                                                                                     cache_position,
                                                                                                     kv_length, *args,
                                                                                                     **kwargs)
                # Rebuild cache positions from full-sequence text-aligned position ids.
                device = cache_position.device
                text_position_ids = _extract_text_position_ids(self.real_position_ids)
                if text_position_ids is None:
                    return masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping['sdpa_origin'](
                        batch_size, cache_position, kv_length, *args, **kwargs)
                cache_position = text_position_ids[0]
                cache_position = self.pad(cache_position, padding_value=-1, position_ids=text_position_ids, dim=0)
                cache_position = torch.arange(0, cache_position.shape[0], device=device)
                kv_length = cache_position.shape[0]
                return masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping['sdpa_origin'](batch_size,
                                                                                                 cache_position,
                                                                                                 kv_length, *args,
                                                                                                 **kwargs)

            masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping[
                'sdpa_origin'] = masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping['sdpa']
            masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping['sdpa'] = sdpa_mask

            def _prepare_full_sequence_mask_inputs(config, inputs_embeds, attention_mask, cache_position=None):
                if (attention_mask is not None and torch.is_tensor(attention_mask) and attention_mask.dim() == 2
                        and getattr(config, '_attn_implementation', None) in {'sdpa', 'eager'}):
                    # SDPA/eager both consume a 4D causal mask built from the 2D padding mask before the attention
                    # kernel runs. If we feed the per-rank 2D padding mask here, each rank builds a local causal
                    # block; later gathering only the mask tensor cannot recover the true global mask semantics.
                    # Gather the 2D padding mask first so HF materializes the full causal mask once.
                    attention_mask = _gather_attention_mask_for_sp(
                        attention_mask,
                        local_seq_len=attention_mask.shape[-1],
                        sp_world_size=self.sp_world_size,
                        sp_group=self._sp_group,
                    )
                inputs_embeds = torch.ones(
                    (inputs_embeds.shape[0], inputs_embeds.shape[1] * self.sp_world_size, inputs_embeds.shape[2]),
                    dtype=inputs_embeds.dtype,
                    device=inputs_embeds.device)
                if cache_position is None:
                    cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
                else:
                    cache_position = torch.arange(0, inputs_embeds.shape[1], device=cache_position.device)
                return inputs_embeds, attention_mask, cache_position

            def create_causal_mask(config,
                                   inputs_embeds=None,
                                   attention_mask=None,
                                   cache_position=None,
                                   past_key_values=None,
                                   position_ids=None,
                                   *args,
                                   **kwargs):
                if inputs_embeds is None and 'input_embeds' in kwargs:
                    inputs_embeds = kwargs.pop('input_embeds')
                if self.world_size == 1:
                    return masking_utils.origin_create_causal_mask(
                        config,
                        inputs_embeds,
                        attention_mask,
                        cache_position,
                        past_key_values,
                        position_ids,
                        *args,
                        **kwargs,
                    )
                inputs_embeds, attention_mask, cache_position = _prepare_full_sequence_mask_inputs(
                    config, inputs_embeds, attention_mask, cache_position)
                return masking_utils.origin_create_causal_mask(
                    config,
                    inputs_embeds,
                    attention_mask,
                    cache_position,
                    past_key_values,
                    position_ids,
                    *args,
                    **kwargs,
                )

            def create_sliding_window_causal_mask(config,
                                                  inputs_embeds=None,
                                                  attention_mask=None,
                                                  cache_position=None,
                                                  past_key_values=None,
                                                  position_ids=None,
                                                  *args,
                                                  **kwargs):
                if inputs_embeds is None and 'input_embeds' in kwargs:
                    inputs_embeds = kwargs.pop('input_embeds')
                if self.world_size == 1:
                    return masking_utils.origin_create_sliding_window_causal_mask(
                        config,
                        inputs_embeds,
                        attention_mask,
                        cache_position,
                        past_key_values,
                        position_ids,
                        *args,
                        **kwargs)
                inputs_embeds, attention_mask, cache_position = _prepare_full_sequence_mask_inputs(
                    config, inputs_embeds, attention_mask, cache_position)
                return masking_utils.origin_create_sliding_window_causal_mask(
                    config,
                    inputs_embeds,
                    attention_mask,
                    cache_position,
                    past_key_values,
                    position_ids,
                    *args,
                    **kwargs)

            masking_utils.origin_create_causal_mask = masking_utils.create_causal_mask
            masking_utils.create_causal_mask = create_causal_mask
            if hasattr(masking_utils, 'create_sliding_window_causal_mask'):
                masking_utils.origin_create_sliding_window_causal_mask = masking_utils.create_sliding_window_causal_mask
                masking_utils.create_sliding_window_causal_mask = create_sliding_window_causal_mask

            # Some models bind these helpers via `from transformers.masking_utils import create_causal_mask`
            # at import time, so patching masking_utils alone is not enough. Patch the model module globals too.
            model_module = importlib.import_module(base_model.__class__.__module__)
            if hasattr(model_module, 'create_causal_mask'):
                if not hasattr(model_module, 'origin_create_causal_mask'):
                    model_module.origin_create_causal_mask = model_module.create_causal_mask
                model_module.create_causal_mask = create_causal_mask
            if hasattr(model_module, 'create_sliding_window_causal_mask'):
                if not hasattr(model_module, 'origin_create_sliding_window_causal_mask'):
                    model_module.origin_create_sliding_window_causal_mask = model_module.create_sliding_window_causal_mask
                model_module.create_sliding_window_causal_mask = create_sliding_window_causal_mask
        except ImportError:
            pass

        if hasattr(base_model, 'language_model'):
            text_model = base_model.language_model
        else:
            text_model = base_model

        from transformers.modeling_flash_attention_utils import is_flash_attn_available
        if is_flash_attn_available():
            # TODO this works for multi-modal models like qwen2.5-vl
            # SDPA is not supported here, because we need to copy the code to our project, which will bring
            # more work for maintaining.
            from transformers import modeling_flash_attention_utils
            from transformers.modeling_flash_attention_utils import _flash_attention_forward
            _distributed_flash_attention = DistributedAttention(_flash_attention_forward, self)

            modeling_flash_attention_utils._flash_attention_forward_origin = _flash_attention_forward

            def flash_attention_forward(query_states: torch.Tensor, key_states: torch.Tensor,
                                        value_states: torch.Tensor, attention_mask: Optional[torch.Tensor], q_len,
                                        *args, **kwargs):
                if self.world_size == 1:
                    return _flash_attention_forward(query_states, key_states, value_states, attention_mask, q_len,
                                                    *args, **kwargs)
                return _distributed_flash_attention(query_states, key_states, value_states, attention_mask,
                                                    q_len * self.sp_world_size, *args, **kwargs)

            modeling_flash_attention_utils._flash_attention_forward = flash_attention_forward

        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        def _normalize_flash_attention_mask(
            attention_mask: Optional[torch.Tensor],
            kwargs: Dict[str, Any],
        ) -> Optional[torch.Tensor]:
            if attention_mask is None or not torch.is_tensor(attention_mask):
                return attention_mask
            if self.extra_kwargs.get('is_packed', False):
                return attention_mask
            if any(kwargs.get(name) is not None for name in ('cu_seq_lens_q', 'cu_seq_lens_k', 'cu_seqlens_q', 'cu_seqlens_k')):
                return attention_mask
            if attention_mask.dim() == 2 and bool(attention_mask.detach().all().item()):
                return None
            return attention_mask

        def local_flash_attn(module: torch.nn.Module, query_states, key_states, value_states, attention_mask, *args,
                             dist_attn, **kwargs):
            if self.world_size == 1 or module.__class__ not in [m.__class__ for m in text_model.modules()]:
                return ALL_ATTENTION_FUNCTIONS['flash_attention_2_origin'](module, query_states, key_states,
                                                                           value_states, attention_mask, *args,
                                                                           **kwargs)
            attention_mask = _normalize_flash_attention_mask(attention_mask, kwargs)
            if dist_attn.local_attn is None:

                def _attention(query, key, value, *args, **kwargs):
                    query = query.transpose(1, 2)
                    key = key.transpose(1, 2)
                    value = value.transpose(1, 2)
                    position_ids = kwargs.get('position_ids')
                    if position_ids is None:
                        position_ids = self.real_position_ids
                    position_ids = _normalize_flash_position_ids(position_ids)
                    # Packed batches (produced by PackingDataset + padding_free collate) require FA2 varlen
                    # semantics to avoid cross-subsequence attention. We derive cu_seqlens from position_ids
                    # resets (0,1,...) and pass cu_seq_lens_* to FA2.
                    if self.extra_kwargs.get('is_packed', False):
                        if position_ids is not None:
                            kwargs['position_ids'] = position_ids
                        # Treat SP-alignment padding (-1) as separate 1-token sequences by mapping -1 -> 0.
                        pos = position_ids
                        if pos is None:
                            raise RuntimeError('SequenceParallel: packed mode requires position_ids.')
                        pos = pos.clone()
                        pos[pos < 0] = 0

                        cu_seqlens = get_cu_seqlens_from_position_ids(pos).to(torch.int32)
                        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
                        assert query.shape[2] == cu_seqlens[-1]
                        kwargs['cu_seq_lens_q'] = cu_seqlens
                        kwargs['cu_seq_lens_k'] = cu_seqlens
                        kwargs['max_length_q'] = max_seqlen
                        kwargs['max_length_k'] = max_seqlen
                        # Do not use attention_mask-based unpadding when using explicit cu_seqlens.
                        if len(args) > 0:
                            args = (None, *args[1:])
                    elif 'cu_seq_lens_q' in kwargs:
                        if position_ids is not None:
                            kwargs['position_ids'] = position_ids
                        text_position_ids = position_ids
                        if text_position_ids is None:
                            raise RuntimeError('SequenceParallel: cu_seqlens mode requires position_ids.')
                        text_position_ids = self.pad(
                            text_position_ids,
                            padding_value=-1,
                            position_ids=text_position_ids,
                        )
                        cu_seqlens = get_cu_seqlens_from_position_ids(text_position_ids).to(torch.int32)
                        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
                        assert query.shape[2] == cu_seqlens[-1]
                        kwargs['cu_seq_lens_q'] = cu_seqlens
                        kwargs['cu_seq_lens_k'] = cu_seqlens
                        kwargs['max_length_q'] = max_seqlen
                        kwargs['max_length_k'] = max_seqlen
                    else:
                        # Qwen3.5 applies RoPE before the FA2 kernel call, so plain (non-packed) text batches do not
                        # need position_ids here. Keeping 3D mRoPE position_ids can make HF treat batch_size=1 inputs
                        # as packed and incorrectly enter the varlen kernel path.
                        kwargs.pop('position_ids', None)
                    return ALL_ATTENTION_FUNCTIONS['flash_attention_2_origin'](module, query, key, value, *args,
                                                                               **kwargs)[0]

                dist_attn.local_attn = _attention

            return dist_attn(
                query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), attention_mask,
                *args, **kwargs), None

        def local_sdpa_attn(module: torch.nn.Module, query_states, key_states, value_states, attention_mask, *args,
                            dist_attn, **kwargs):
            # Bypass SP logic when world_size == 1 (SP disabled) or module not in text_model
            if self.world_size == 1 or module.__class__ not in [m.__class__ for m in text_model.modules()]:
                return ALL_ATTENTION_FUNCTIONS['sdpa_origin'](module, query_states, key_states, value_states,
                                                              attention_mask, *args, **kwargs)
            # Policy: packed (PackingDataset/padding-free) batches require FlashAttention2 varlen/packed semantics.
            # SDPA does not have a native packed/varlen interface; supporting packed batches would require building a
            # large block-diagonal causal mask (slow / memory heavy).
            if self.extra_kwargs.get('is_packed', False):
                raise RuntimeError(
                    'SequenceParallel: detected packed batch (position_ids contains multiple sequences). '
                    'SDPA backend is not supported for packed batches; please use flash_attention_2.')
            if dist_attn.local_attn is None:

                def _attention(query, key, value, *args, **kwargs):
                    query = query.transpose(1, 2)
                    key = key.transpose(1, 2)
                    value = value.transpose(1, 2)
                    return ALL_ATTENTION_FUNCTIONS['sdpa_origin'](module, query, key, value, *args, **kwargs)[0]

                dist_attn.local_attn = _attention
            return dist_attn(
                query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), attention_mask,
                *args, **kwargs), None

        def local_eager_attn(module: torch.nn.Module, query_states, key_states, value_states, attention_mask, *args,
                             dist_attn, **kwargs):
            if self.world_size == 1 or module.__class__ not in [m.__class__ for m in text_model.modules()]:
                return ALL_ATTENTION_FUNCTIONS['eager_origin'](
                    module, query_states, key_states, value_states, attention_mask, *args, **kwargs)

            if self.extra_kwargs.get('is_packed', False):
                raise RuntimeError(
                    'SequenceParallel: detected packed batch (position_ids contains multiple sequences). '
                    'eager backend is not supported for packed batches.')

            if dist_attn.local_attn is None:

                def _attention(query, key, value, *inner_args, **inner_kwargs):
                    query = query.transpose(1, 2)
                    key = key.transpose(1, 2)
                    value = value.transpose(1, 2)
                    return ALL_ATTENTION_FUNCTIONS['eager_origin'](
                        module, query, key, value, *inner_args, **inner_kwargs)[0]

                dist_attn.local_attn = _attention

            return dist_attn(
                query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), attention_mask,
                *args, **kwargs), None

        ALL_ATTENTION_FUNCTIONS['flash_attention_2_origin'] = ALL_ATTENTION_FUNCTIONS['flash_attention_2']
        ALL_ATTENTION_FUNCTIONS['sdpa_origin'] = ALL_ATTENTION_FUNCTIONS['sdpa']
        eager_origin = None
        if 'eager' in ALL_ATTENTION_FUNCTIONS:
            eager_origin = ALL_ATTENTION_FUNCTIONS['eager']
            ALL_ATTENTION_FUNCTIONS['eager_origin'] = eager_origin
        else:
            eager_origin = getattr(model_module, 'eager_attention_forward', None)
            if eager_origin is not None:
                ALL_ATTENTION_FUNCTIONS['eager_origin'] = eager_origin
        ALL_ATTENTION_FUNCTIONS['flash_attention_2'] = partial(
            local_flash_attn, dist_attn=DistributedAttention(None, self))
        ALL_ATTENTION_FUNCTIONS['sdpa'] = partial(local_sdpa_attn, dist_attn=DistributedAttention(None, self))
        if 'eager_origin' in ALL_ATTENTION_FUNCTIONS:
            ALL_ATTENTION_FUNCTIONS['eager'] = partial(local_eager_attn, dist_attn=DistributedAttention(None, self))
        if eager_origin is not None and hasattr(model_module, 'eager_attention_forward'):
            if not hasattr(model_module, 'origin_eager_attention_forward'):
                model_module.origin_eager_attention_forward = model_module.eager_attention_forward
            model_module.eager_attention_forward = partial(local_eager_attn, dist_attn=DistributedAttention(None, self))

    def _wrap_qwen35_chunk_rule(self, module: torch.nn.Module, origin_rule):

        recurrent_rule = getattr(module, 'recurrent_gated_delta_rule', None)
        module_impl = importlib.import_module(module.__class__.__module__)
        torch_recurrent_rule = getattr(module_impl, 'torch_recurrent_gated_delta_rule', None)

        def recurrent_rule_wrapper(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            g: torch.Tensor,
            beta: torch.Tensor,
            chunk_size: Optional[int] = None,
            initial_state: Optional[torch.Tensor] = None,
            output_final_state: bool = False,
            use_qk_l2norm_in_kernel: bool = False,
        ):
            rule = torch_recurrent_rule or recurrent_rule
            if rule is None:
                raise RuntimeError('SequenceParallel: Qwen3.5 recurrent_gated_delta_rule is unavailable.')
            return rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )

        def wrapped_chunk_rule(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            g: torch.Tensor,
            beta: torch.Tensor,
            chunk_size: Optional[int] = None,
            initial_state: Optional[torch.Tensor] = None,
            output_final_state: bool = False,
            use_qk_l2norm_in_kernel: bool = False,
        ):
            call_kwargs = {
                'g': g,
                'beta': beta,
                'initial_state': initial_state,
                'output_final_state': output_final_state,
                'use_qk_l2norm_in_kernel': use_qk_l2norm_in_kernel,
            }
            if chunk_size is not None:
                call_kwargs['chunk_size'] = chunk_size

            if self.sp_world_size is None or self.sp_world_size <= 1 or self._sp_group is None:
                return origin_rule(query, key, value, **call_kwargs)
            # Respect cache/incremental path semantics.
            if initial_state is not None or output_final_state:
                return origin_rule(query, key, value, **call_kwargs)

            if self.extra_kwargs.get('is_packed', False):
                raise RuntimeError(
                    'SequenceParallel: packed batches are not supported for Qwen3.5 linear attention under SP. '
                    'Packed reset semantics for recurrent states are not implemented.')

            sp_rule = origin_rule
            sp_chunk_size = int(chunk_size) if chunk_size is not None else None
            effective_chunk_size = sp_chunk_size or 64
            if recurrent_rule is not None and query.shape[1] % effective_chunk_size != 0:
                # Qwen3.5's chunk rule is only exact when split points align with chunk boundaries. If SP splits a
                # sequence inside a chunk (e.g. seq_len=32, sp=2, local_seq=16, default chunk_size=64), carrying only
                # the final recurrent state is not enough to reproduce the monolithic chunked prefill path. Fall back
                # to the recurrent rule for exact state hand-off across SP ranks.
                sp_rule = recurrent_rule_wrapper
                sp_chunk_size = None

            output = _QwenLinearStateParallelFn.apply(
                query,
                key,
                value,
                g,
                beta,
                self._sp_group,
                sp_rule,
                sp_chunk_size,
                bool(use_qk_l2norm_in_kernel),
            )
            return output, None

        wrapped_chunk_rule._twinkle_origin_rule = origin_rule
        wrapped_chunk_rule._twinkle_wrapped_module = module.__class__.__name__
        return wrapped_chunk_rule

    def _wrap_qwen35_causal_conv1d_fn(
        self,
        module: torch.nn.Module,
        origin_causal_conv1d_fn,
    ):

        def run_origin(
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: Optional[torch.Tensor],
            activation: str,
            seq_idx=None,
        ) -> torch.Tensor:
            if origin_causal_conv1d_fn is not None:
                return origin_causal_conv1d_fn(
                    x=x,
                    weight=weight,
                    bias=bias,
                    activation=activation,
                    seq_idx=seq_idx,
                )
            conv_weight = weight.unsqueeze(1) if weight.dim() == 2 else weight
            conv_bias = bias if bias is not None else module.conv1d.bias
            conv_output = torch.nn.functional.conv1d(
                x,
                conv_weight,
                bias=conv_bias,
                padding=module.conv_kernel_size - 1,
                groups=module.conv_dim,
            )
            conv_output = conv_output[:, :, :x.shape[-1]]
            return module.act(conv_output)

        def wrapped_causal_conv1d(
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: Optional[torch.Tensor],
            activation: str,
            seq_idx=None,
        ) -> torch.Tensor:
            if (
                not self.qwen35_linear_conv_halo
                or self.qwen35_linear_strict_full_seq
                or self.sp_world_size is None
                or self.sp_world_size <= 1
                or self._sp_group is None
            ):
                return run_origin(x, weight, bias, activation, seq_idx)

            halo = int(module.conv_kernel_size) - 1
            if halo <= 0:
                return run_origin(x, weight, bias, activation, seq_idx)

            extended_x = _LeftHaloExchangeFn.apply(
                x.contiguous(),
                halo,
                -1,
                self._sp_group,
                int(getattr(module, 'layer_idx', 0)),
            )
            extended_output = run_origin(extended_x, weight, bias, activation, seq_idx)
            local_seq_len = x.shape[-1]
            return extended_output.narrow(-1, halo, local_seq_len).contiguous()

        wrapped_causal_conv1d._twinkle_origin_causal_conv1d_fn = origin_causal_conv1d_fn
        wrapped_causal_conv1d._twinkle_wrapped_module = module.__class__.__name__
        return wrapped_causal_conv1d

    def _wrap_qwen35_linear_forward(
        self,
        module: torch.nn.Module,
        origin_forward,
        origin_chunk_rule,
        origin_recurrent_rule,
        origin_causal_conv1d_fn,
    ):

        halo_causal_conv1d_fn = self._wrap_qwen35_causal_conv1d_fn(module, origin_causal_conv1d_fn)

        def wrapped_forward(
            hidden_states: torch.Tensor,
            cache_params=None,
            cache_position: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
        ):
            linear_sp_enabled = self.qwen35_linear_strict_full_seq or self.qwen35_linear_conv_halo
            if (
                not linear_sp_enabled
                or self.sp_world_size is None
                or self.sp_world_size <= 1
                or self._sp_group is None
            ):
                return origin_forward(
                    hidden_states,
                    cache_params=cache_params,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                )

            if self.extra_kwargs.get('is_packed', False):
                raise RuntimeError(
                    'SequenceParallel: packed batches are not supported for Qwen3.5 linear attention under SP '
                    'strict/halo modes.'
                )

            use_precomputed_states = (
                cache_params is not None
                and getattr(cache_params, 'has_previous_state', False)
                and hidden_states.shape[1] == 1
                and cache_position is not None
            )
            if use_precomputed_states:
                # Decode already carries conv/recurrent state through cache_params; strict full-sequence recomputation
                # is only needed for prefill, where local shards otherwise miss causal-conv left context.
                return origin_forward(
                    hidden_states,
                    cache_params=cache_params,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                )

            if (
                self.qwen35_linear_conv_halo
                and not self.qwen35_linear_strict_full_seq
                and not use_precomputed_states
            ):
                saved_causal_conv1d_fn = module.causal_conv1d_fn
                module.causal_conv1d_fn = halo_causal_conv1d_fn
                try:
                    return origin_forward(
                        hidden_states,
                        cache_params=cache_params,
                        cache_position=cache_position,
                        attention_mask=attention_mask,
                    )
                finally:
                    module.causal_conv1d_fn = saved_causal_conv1d_fn

            layer_idx = int(getattr(module, 'layer_idx', 0))
            debug_label = f'qwen35_linear_layer_{layer_idx}'
            full_hidden_states = _GatherForwardSplitBackward.apply(
                hidden_states,
                1,
                self.sp_world_size,
                self._sp_group,
                f'{debug_label}:hidden_states',
            )
            _sp_linear_collective_debug(
                f'{debug_label}: before gather attention_mask '
                f'shape={None if attention_mask is None else tuple(attention_mask.shape)}',
                self._sp_group,
            )
            full_attention_mask = _gather_attention_mask_for_sp(
                attention_mask,
                local_seq_len=hidden_states.shape[1],
                sp_world_size=self.sp_world_size,
                sp_group=self._sp_group,
            )
            _sp_linear_collective_debug(
                f'{debug_label}: after gather attention_mask '
                f'shape={None if full_attention_mask is None else tuple(full_attention_mask.shape)}',
                self._sp_group,
            )
            full_cache_position = cache_position
            if full_cache_position is not None and torch.is_tensor(full_cache_position):
                full_cache_position = torch.arange(
                    0,
                    full_hidden_states.shape[1],
                    device=full_cache_position.device,
                    dtype=full_cache_position.dtype,
                )

            saved_chunk_rule = module.chunk_gated_delta_rule
            saved_recurrent_rule = module.recurrent_gated_delta_rule
            module.chunk_gated_delta_rule = origin_chunk_rule
            if origin_recurrent_rule is not None:
                module.recurrent_gated_delta_rule = origin_recurrent_rule
            try:
                _sp_linear_collective_debug(
                    f'{debug_label}: before origin_forward full_hidden_states shape={tuple(full_hidden_states.shape)}',
                    self._sp_group,
                )
                full_output = origin_forward(
                    full_hidden_states,
                    cache_params=cache_params,
                    cache_position=full_cache_position,
                    attention_mask=full_attention_mask,
                )
                _sp_linear_collective_debug(
                    f'{debug_label}: after origin_forward full_output shape={tuple(full_output.shape)}',
                    self._sp_group,
                )
            finally:
                module.chunk_gated_delta_rule = saved_chunk_rule
                module.recurrent_gated_delta_rule = saved_recurrent_rule

            return _SplitForwardGatherBackward.apply(
                full_output,
                1,
                self.sp_world_size,
                self._sp_group,
                f'{debug_label}:full_output',
            )

        wrapped_forward._twinkle_origin_forward = origin_forward
        wrapped_forward._twinkle_wrapped_module = module.__class__.__name__
        return wrapped_forward

    def _wrap_qwen35_attention_debug_forward(
        self,
        module: torch.nn.Module,
        origin_forward,
    ):
        modeling_module = importlib.import_module(module.__class__.__module__)
        apply_rotary_pos_emb = getattr(modeling_module, 'apply_rotary_pos_emb')
        all_attention_functions = getattr(modeling_module, 'ALL_ATTENTION_FUNCTIONS')
        eager_attention_forward = getattr(modeling_module, 'eager_attention_forward')
        layer_idx = int(getattr(module, 'layer_idx', 0))
        debug_label = f'qwen35_attention_layer_{layer_idx}'

        def wrapped_forward(
            hidden_states: torch.Tensor,
            position_embeddings,
            attention_mask: Optional[torch.Tensor],
            past_key_values=None,
            cache_position: Optional[torch.Tensor] = None,
            **kwargs,
        ):
            _sp_linear_collective_debug(
                f'{debug_label}: before qkv_proj hidden_states={tuple(hidden_states.shape)}',
                self._sp_group,
            )
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, module.head_dim)

            query_states, gate = torch.chunk(
                module.q_proj(hidden_states).view(*input_shape, -1, module.head_dim * 2), 2, dim=-1
            )
            gate = gate.reshape(*input_shape, -1)
            key_states = module.k_proj(hidden_states)
            value_states = module.v_proj(hidden_states)
            _sp_linear_collective_debug(
                f'{debug_label}: after qkv_proj q={tuple(query_states.shape)} '
                f'k={tuple(key_states.shape)} v={tuple(value_states.shape)}',
                self._sp_group,
            )

            query_states = module.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
            key_states = module.k_norm(key_states.view(hidden_shape)).transpose(1, 2)
            value_states = value_states.view(hidden_shape).transpose(1, 2)
            _sp_linear_collective_debug(
                f'{debug_label}: after qk_norm q={tuple(query_states.shape)} '
                f'k={tuple(key_states.shape)} v={tuple(value_states.shape)}',
                self._sp_group,
            )

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            _sp_linear_collective_debug(
                f'{debug_label}: after rotary q={tuple(query_states.shape)} k={tuple(key_states.shape)}',
                self._sp_group,
            )

            if past_key_values is not None:
                cache_kwargs = {'sin': sin, 'cos': cos, 'cache_position': cache_position}
                key_states, value_states = past_key_values.update(key_states, value_states, module.layer_idx, cache_kwargs)
                _sp_linear_collective_debug(
                    f'{debug_label}: after cache_update k={tuple(key_states.shape)} v={tuple(value_states.shape)}',
                    self._sp_group,
                )

            attention_interface = all_attention_functions.get_interface(
                module.config._attn_implementation,
                eager_attention_forward,
            )
            _sp_linear_collective_debug(
                f'{debug_label}: before attention_interface impl={module.config._attn_implementation} '
                f'attention_mask={None if attention_mask is None else tuple(attention_mask.shape)}',
                self._sp_group,
            )
            attn_output, attn_weights = attention_interface(
                module,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not module.training else module.attention_dropout,
                scaling=module.scaling,
                **kwargs,
            )
            _sp_linear_collective_debug(
                f'{debug_label}: after attention_interface attn_output={tuple(attn_output.shape)}',
                self._sp_group,
            )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn_output * torch.sigmoid(gate)
            attn_output = module.o_proj(attn_output)
            _sp_linear_collective_debug(
                f'{debug_label}: after o_proj attn_output={tuple(attn_output.shape)}',
                self._sp_group,
            )
            return attn_output, attn_weights

        wrapped_forward._twinkle_origin_forward = origin_forward
        wrapped_forward._twinkle_wrapped_module = module.__class__.__name__
        return wrapped_forward

    def _prepare_qwen35_linear_attn(self, base_model: torch.nn.Module):
        if hasattr(base_model, 'language_model'):
            text_model = base_model.language_model
        else:
            text_model = base_model

        has_linear_attn = False
        for module in text_model.modules():
            if module.__class__.__name__ != 'Qwen3_5GatedDeltaNet':
                continue
            origin_rule = getattr(module, 'chunk_gated_delta_rule', None)
            if origin_rule is None:
                continue
            origin_recurrent_rule = getattr(module, 'recurrent_gated_delta_rule', None)
            origin_forward = getattr(module, 'forward', None)
            has_linear_attn = True
            if getattr(module, '_twinkle_sp_linear_chunk_patched', False):
                continue
            module.chunk_gated_delta_rule = self._wrap_qwen35_chunk_rule(module, origin_rule)
            if origin_forward is not None:
                module.forward = self._wrap_qwen35_linear_forward(
                    module,
                    origin_forward,
                    origin_rule,
                    origin_recurrent_rule,
                    getattr(module, 'causal_conv1d_fn', None),
                )
            module._twinkle_sp_linear_chunk_patched = True

        if _sp_qwen35_decoder_debug_enabled() and hasattr(text_model, 'layers'):
            for layer_idx, decoder_layer in enumerate(text_model.layers):
                if decoder_layer.__class__.__name__ != 'Qwen3_5DecoderLayer':
                    continue
                if getattr(decoder_layer, '_twinkle_sp_decoder_debug_hooked', False):
                    continue

                def _make_pre_hook(idx):

                    def _pre_hook(_module, _args, _kwargs):
                        hidden_states = None
                        if 'hidden_states' in _kwargs:
                            hidden_states = _kwargs['hidden_states']
                        elif _args:
                            hidden_states = _args[0]
                        _sp_linear_collective_debug(
                            f'qwen35_decoder_layer_{idx}: before forward '
                            f'layer_type={getattr(_module, "layer_type", "unknown")} '
                            f'hidden_states={None if hidden_states is None else tuple(hidden_states.shape)}',
                            self._sp_group,
                        )

                    return _pre_hook

                def _make_post_hook(idx):

                    def _post_hook(_module, _args, _kwargs, output):
                        hidden_states = output[0] if isinstance(output, (tuple, list)) else output
                        _sp_linear_collective_debug(
                            f'qwen35_decoder_layer_{idx}: after forward '
                            f'layer_type={getattr(_module, "layer_type", "unknown")} '
                            f'hidden_states={None if hidden_states is None else tuple(hidden_states.shape)}',
                            self._sp_group,
                        )

                    return _post_hook

                decoder_layer.register_forward_pre_hook(_make_pre_hook(layer_idx), with_kwargs=True)
                decoder_layer.register_forward_hook(_make_post_hook(layer_idx), with_kwargs=True)
                decoder_layer._twinkle_sp_decoder_debug_hooked = True

                full_attn = getattr(decoder_layer, 'self_attn', None)
                if (
                    _sp_qwen35_decoder_debug_enabled()
                    and full_attn is not None
                    and full_attn.__class__.__name__ == 'Qwen3_5Attention'
                    and not getattr(full_attn, '_twinkle_sp_attn_debug_hooked', False)
                ):
                    origin_attn_forward = getattr(full_attn, 'forward', None)
                    if origin_attn_forward is not None:
                        full_attn.forward = self._wrap_qwen35_attention_debug_forward(full_attn, origin_attn_forward)

                    def _make_attn_pre_hook(idx):

                        def _pre_hook(_module, _args, _kwargs):
                            hidden_states = None
                            if 'hidden_states' in _kwargs:
                                hidden_states = _kwargs['hidden_states']
                            elif _args:
                                hidden_states = _args[0]
                            attention_mask = _kwargs.get('attention_mask', None)
                            _sp_linear_collective_debug(
                                f'qwen35_attention_layer_{idx}: before forward '
                                f'hidden_states={None if hidden_states is None else tuple(hidden_states.shape)} '
                                f'attention_mask={None if attention_mask is None else tuple(attention_mask.shape)}',
                                self._sp_group,
                            )

                        return _pre_hook

                    def _make_attn_post_hook(idx):

                        def _post_hook(_module, _args, _kwargs, output):
                            attn_output = output[0] if isinstance(output, (tuple, list)) else output
                            _sp_linear_collective_debug(
                                f'qwen35_attention_layer_{idx}: after forward '
                                f'attn_output={None if attn_output is None else tuple(attn_output.shape)}',
                                self._sp_group,
                            )

                        return _post_hook

                    full_attn.register_forward_pre_hook(_make_attn_pre_hook(layer_idx), with_kwargs=True)
                    full_attn.register_forward_hook(_make_attn_post_hook(layer_idx), with_kwargs=True)
                    full_attn._twinkle_sp_attn_debug_hooked = True

        self.has_qwen35_linear_attn = has_linear_attn
        self.extra_kwargs['has_qwen35_linear_attn'] = has_linear_attn

    def _prepare_forward_hook(self, base_model: torch.nn.Module):

        def pre_forward_split_hook(_self, args, kwargs):
            if self.world_size == 1:
                return args, kwargs
            # Pad to multiple of SP size and split inputs per SP rank before forward.
            input_ids = kwargs.get('input_ids', None)
            inputs_embeds = kwargs.get('inputs_embeds', None)
            position_ids = kwargs['position_ids']
            attention_mask = kwargs.get('attention_mask', None)
            extra_names = []
            extra_split_values = []
            for name in ('mm_token_type_ids', 'token_type_ids'):
                value = kwargs.get(name, None)
                if value is not None:
                    extra_names.append(name)
                    extra_split_values.append((value, 0, -1))
            if hasattr(_self, 'language_model'):
                embed_tokens = getattr(_self.language_model, 'embed_tokens', None)
            else:
                embed_tokens = getattr(_self, 'embed_tokens', None)
            input_ids, inputs_embeds, _, position_ids, attention_mask, _, extra_values = self.pad_and_split_inputs(
                input_ids,
                inputs_embeds,
                None,
                position_ids,
                attention_mask,
                None,
                embed_tokens=embed_tokens,
                real_position_ids=self.real_position_ids,
                extra_split_values=extra_split_values if extra_split_values else None,
            )
            kwargs['input_ids'] = input_ids
            kwargs['inputs_embeds'] = inputs_embeds
            kwargs['position_ids'] = position_ids
            kwargs['attention_mask'] = attention_mask
            for i, name in enumerate(extra_names):
                kwargs[name] = extra_values[i]
            return args, kwargs

        base_model.register_forward_pre_hook(pre_forward_split_hook, with_kwargs=True)

    def _prepare_moe_aux_loss(self, base_model: torch.nn.Module):

        def moe_aux_loss_hook(module, args, kwargs, output):
            router_logits = getattr(output, 'router_logits', None)
            if router_logits is None:
                return output

            attention_mask = kwargs['attention_mask']
            if attention_mask is None:
                batch_size = 1
            else:
                batch_size = attention_mask.shape[0]

            assert router_logits[0].shape[0] % batch_size == 0
            seq_len = router_logits[0].shape[0] // batch_size

            _gathered_logits = []
            for i in range(batch_size):
                _slice = slice(i * seq_len, (i + 1) * seq_len)
                _bs_logits = [logit[_slice] for logit in router_logits]
                compute_device = _bs_logits[0].device
                _bs_logits = torch.stack([layer_gate.to(compute_device) for layer_gate in _bs_logits], dim=0)
                _bs_logits, _ = GatherLoss.apply(_bs_logits, None, 1, self.real_position_ids)
                _gathered_logits.append(_bs_logits)
            router_logits = torch.stack(_gathered_logits, dim=0)
            text_position_ids = _extract_text_position_ids(self.real_position_ids)
            if text_position_ids is not None:
                router_logits = router_logits[:, :, :text_position_ids.shape[1], :]
            output['router_logits'] = tuple(
                [logit.reshape(-1, logit.shape[-1]) for logit in router_logits.split(1, dim=1)])
            return output

        base_model.register_forward_hook(moe_aux_loss_hook, with_kwargs=True)

    @staticmethod
    def _is_moe_model(config) -> bool:
        if 'Moe' in config.__class__.__name__:
            return True
        for key in ['num_experts', 'num_experts_per_tok', 'moe_intermediate_size']:
            if get_config_attr(config, key):
                return True
        return False

    @staticmethod
    def _disable_gradient_checkpointing_for_model(model: torch.nn.Module) -> bool:
        disabled = False
        visited = set()
        candidates = [model]
        llm_outer = get_llm_model(model, inner_backbone=False)
        llm_model = get_llm_model(model, inner_backbone=True)
        for candidate in (llm_outer, llm_model):
            if candidate is not None and candidate is not model:
                candidates.append(candidate)
        for candidate in candidates:
            if candidate is None or id(candidate) in visited:
                continue
            visited.add(id(candidate))
            if hasattr(candidate, 'gradient_checkpointing'):
                try:
                    candidate.gradient_checkpointing = False
                    disabled = True
                except Exception:
                    pass
            if hasattr(candidate, 'gradient_checkpointing_disable'):
                try:
                    candidate.gradient_checkpointing_disable()
                    disabled = True
                except Exception:
                    pass
            candidate_config = getattr(candidate, 'config', None)
            if candidate_config is not None and hasattr(candidate_config, 'use_cache'):
                try:
                    candidate_config.use_cache = False
                except Exception:
                    pass
            if candidate_config is not None and hasattr(candidate_config, 'gradient_checkpointing'):
                try:
                    candidate_config.gradient_checkpointing = False
                except Exception:
                    pass
            for module in candidate.modules():
                if hasattr(module, 'gradient_checkpointing'):
                    try:
                        module.gradient_checkpointing = False
                        disabled = True
                    except Exception:
                        pass
                if hasattr(module, '_gradient_checkpointing_func'):
                    try:
                        module._gradient_checkpointing_func = None
                    except Exception:
                        pass
        return disabled

    def prepare(
        self,
        sp_size: int,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizer,
        device_mesh: Optional[DeviceMesh] = None,
    ):
        self.num_heads = get_config_attr(model.config, 'num_key_value_heads')
        if self.num_heads is None:
            self.num_heads = get_config_attr(model.config, 'num_attention_heads')
        assert self.num_heads is not None, 'Cannot find num_heads config in config.json'
        if sp_size > 1 and self.num_heads % sp_size != 0:
            raise ValueError(
                f'sp_size ({sp_size}) must divide num_heads ({self.num_heads}) for ulysses sequence parallel.')
        self.world_size = sp_size

        llm_model = get_llm_model(model)

        if hasattr(llm_model, 'language_model'):
            if hasattr(llm_model.language_model, '_update_causal_mask'):
                self.causal_mask_func = llm_model.language_model._update_causal_mask
        else:
            if hasattr(llm_model, '_update_causal_mask'):
                self.causal_mask_func = llm_model._update_causal_mask

        if not SequenceParallel._global_inited:
            # these operations are global initializations and patches
            self._init_device_mesh(device_mesh)
            self._prepare_flash_attn(llm_model)
            SequenceParallel._global_inited = True

        # Model-specific patch: Qwen3.5 linear attention state passing for SP.
        self._prepare_qwen35_linear_attn(llm_model)
        if (
            (self.qwen35_linear_conv_halo or self.qwen35_linear_strict_full_seq)
            and self.has_qwen35_linear_attn
            and self.qwen35_linear_disable_gc
        ):
            was_gc_enabled = bool(getattr(model, 'is_gradient_checkpointing', False))
            if not was_gc_enabled:
                was_gc_enabled = bool(getattr(llm_model, 'is_gradient_checkpointing', False))
            if was_gc_enabled:
                disabled = self._disable_gradient_checkpointing_for_model(model)
                self.extra_kwargs['qwen35_linear_disabled_gradient_checkpointing'] = disabled
        self._prepare_forward_hook(llm_model)

        if SequenceParallel._is_moe_model(getattr(model, 'config', None)):
            self._prepare_moe_aux_loss(llm_model)

        self.model_dtype = next(model.parameters()).dtype
        self.tokenizer = tokenizer

    def pad(self, tensor, padding_value, position_ids=None, dim=1):
        """Pad tensor for sequence parallel"""
        world_size = self.world_size

        def _do_pad(tensor):
            # Ensure seq length is divisible by SP size to allow even split.
            length = tensor.shape[dim]
            pad_num = world_size - (length % world_size)
            if pad_num == 0 or pad_num == world_size:
                return tensor
            if not isinstance(padding_value, torch.Tensor):
                # ids
                pad_shape = ((*tensor.shape[:dim], pad_num, *tensor.shape[dim + 1:]) if dim != -1 else
                             (*tensor.shape[:dim], pad_num))
                pad = torch.full(pad_shape, padding_value, dtype=tensor.dtype, device=tensor.device)
                tensor = torch.cat([tensor, pad], dim=dim)
            else:
                # For embeddings
                tensor = torch.cat([tensor, padding_value.unsqueeze(0).repeat(tensor.shape[0], pad_num, 1)], dim=dim)
            return tensor

        return _do_pad(tensor)

    def gather(self, local_output, dim: int, position_ids=None):
        """Gather tensor for sequence parallel - reverse of split"""
        if self.world_size == 1:
            return local_output

        # Gather local chunks from each SP rank and concatenate along sequence dim.
        gathered_sp = torch.empty(
            [local_output.shape[0] * self.sp_world_size] + list(local_output.shape[1:]),
            dtype=local_output.dtype,
            device=local_output.device)
        dist.all_gather_into_tensor(gathered_sp, local_output, group=self._sp_group)
        gathered_sp = torch.cat(gathered_sp.split(local_output.shape[0], dim=0), dim=dim)
        return gathered_sp.contiguous()

    def split(self, input, dim: int, position_ids=None):
        """Split tensor for sequence parallel"""
        if self.world_size == 1:
            return input

        # Split along sequence dimension; each rank keeps its local slice.
        rank = dist.get_rank(self._sp_group) if self._sp_group is not None else 0
        dim_size = input.size(dim)
        assert dim_size % self.sp_world_size == 0, (f'The dimension to split ({dim_size}) is not a multiple of '
                                                    f'world size ({self.sp_world_size}), cannot split tensor evenly')

        tensor_list = torch.split(input, dim_size // self.sp_world_size, dim=dim)
        output = tensor_list[rank].contiguous()
        return output

    def _split_attention_mask(self, attention_mask: Optional[torch.Tensor], seq_len: Optional[int],
                              real_position_ids: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Split attention_mask to align with sequence-parallel local chunks."""
        if attention_mask is None or self.world_size == 1:
            return attention_mask

        if attention_mask.dim() == 2:
            return self.split(attention_mask, dim=1, position_ids=real_position_ids)

        if seq_len is None:
            return attention_mask

        # Prefer splitting sequence-related trailing dims.
        split_dims = []
        if attention_mask.shape[-1] == seq_len:
            split_dims.append(-1)
        if attention_mask.dim() >= 3 and attention_mask.shape[-2] == seq_len:
            split_dims.append(-2)
        # Fallback for uncommon layouts where sequence is on dim=1.
        if not split_dims and attention_mask.dim() > 1 and attention_mask.shape[1] == seq_len:
            split_dims.append(1)

        output = attention_mask
        for dim in split_dims:
            output = self.split(output, dim=dim, position_ids=real_position_ids)
        return output

    def pad_and_split_inputs(self,
                             input_ids,
                             input_embeds,
                             labels,
                             position_ids,
                             attention_mask,
                             loss_scale,
                             embed_tokens=None,
                             real_position_ids=None,
                             extra_split_values=None):
        """Common implementation for padding and splitting inputs

        Pad to a length divisible by the sequence-parallel size, then split across SP ranks.

        Args:
            input_ids: input_ids
            input_embeds: input_embeds
            labels: labels
            position_ids: position_ids or, position_ids for mrope
            attention_mask: attention_mask
            loss_scale: loss_scale
            embed_tokens: embed_tokens
            real_position_ids: the real position_ids to represent the seq length information
            extra_split_values: List of Tuples for extra split values, e.g.: (tensor, pad_value, split_dim)
        """
        tokenizer = self.tokenizer
        real_position_ids = real_position_ids if real_position_ids is not None else position_ids
        text_position_ids = _extract_text_position_ids(real_position_ids)
        # Track packed batches to drive attention backend behavior (packed => require flash_attention_2 varlen).
        self.extra_kwargs['is_packed'] = self._is_packed_position_ids(text_position_ids)
        self.extra_kwargs['has_qwen35_linear_attn'] = self.has_qwen35_linear_attn
        if self.world_size is not None and self.world_size > 1 and self.has_qwen35_linear_attn and self.extra_kwargs[
                'is_packed']:
            raise RuntimeError(
                'SequenceParallel: packed batches are not supported for Qwen3.5 linear attention under SP. '
                'Packed reset semantics for recurrent states are not implemented.')
        extra_values = []
        batch_size = input_ids.shape[
            0] if input_ids is not None else input_embeds.shape[0] if input_embeds is not None else None
        if input_ids is not None:
            input_ids = self.pad(input_ids, padding_value=tokenizer.pad_token_id, position_ids=real_position_ids)
            self.extra_kwargs['input_ids'] = input_ids.clone()
        if input_embeds is not None:
            pad_emb = torch.zeros(
                (1, embed_tokens.weight.shape[-1])).to(embed_tokens.weight.device).to(embed_tokens.weight.dtype)
            input_embeds = self.pad(input_embeds, padding_value=pad_emb, position_ids=real_position_ids)
        batch_size = input_ids.shape[
            0] if input_ids is not None else input_embeds.shape[0] if input_embeds is not None else 1
        if position_ids is not None:
            position_ids = self.pad(position_ids, padding_value=-1, position_ids=real_position_ids, dim=-1)
        if labels is not None:
            labels = self.pad(labels, padding_value=-100, position_ids=real_position_ids)
        if loss_scale is not None:
            loss_scale = self.pad(loss_scale, padding_value=0., position_ids=real_position_ids)
        if real_position_ids is not None:
            real_position_ids = self.pad(real_position_ids, padding_value=-1, position_ids=real_position_ids)
        text_position_ids = _extract_text_position_ids(real_position_ids)
        if real_position_ids is not None:
            self.extra_kwargs['position_ids'] = real_position_ids.clone()
        if text_position_ids is not None and batch_size is not None and text_position_ids.shape[0] == batch_size:
            self.extra_kwargs['text_position_ids'] = text_position_ids.clone()
        # Build a 2D attention_mask whenever we padded for SP alignment so FlashAttention2 can unpad correctly.
        # For packed batches (batch_size==1 with multiple position_id resets), relying on position_ids alone is
        # unsafe if we also appended SP-alignment padding (position_ids=-1), because HF's FA2 varlen path will
        # include the padded tail in the last segment when attention_mask is None.
        if (input_ids is not None or input_embeds is not None) and batch_size > 1:
            # not padding_free, so not ring-attention
            inputs = input_ids if input_ids is not None else input_embeds
            attn_shape = inputs.shape[1]  # The sequence length
            if attention_mask is None:
                # Mask out padded positions introduced by sequence-parallel padding.
                # `real_position_ids` is padded with `-1` (see above), so use it to build a valid-token mask.
                if text_position_ids is not None:
                    attention_mask = (text_position_ids != -1).to(dtype=torch.int64)
                else:
                    attention_mask = torch.ones((batch_size, attn_shape), dtype=torch.int64, device=inputs.device)
            # no need position_ids here, because padding_free does not need attention_mask,
            # so this is not ring-attention
            attention_mask = self.pad(attention_mask, padding_value=0)
            cache_position = torch.arange(0, attn_shape, device=inputs.device)
            # For SP>1 keep 2D masks and let full-attention gather to global mask later.
            # Prebuilding 4D causal masks before split can break full-sequence semantics.
            if (self.world_size == 1 and hasattr(self, 'causal_mask_func') and self.causal_mask_func is not None
                    and not self.has_qwen35_linear_attn):
                attention_mask = self.causal_mask_func(attention_mask, inputs.to(self.model_dtype), cache_position,
                                                       None, None)
        if extra_split_values is not None:
            for (tensor, pad_value, split_dim) in extra_split_values:
                extra_values.append(
                    self.pad(tensor, padding_value=pad_value, position_ids=real_position_ids, dim=split_dim))
        if input_ids is not None:
            input_ids = self.split(input_ids, dim=1, position_ids=real_position_ids)
        if input_embeds is not None:
            input_embeds = self.split(input_embeds, dim=1, position_ids=real_position_ids)
        seq_len = input_ids.shape[1] if input_ids is not None else input_embeds.shape[1] if input_embeds is not None else None
        attention_mask = self._split_attention_mask(attention_mask, seq_len=seq_len, real_position_ids=real_position_ids)
        if labels is not None:
            if self.extra_kwargs.get('is_packed', False) and text_position_ids is not None:
                # PackingDataset + padding_free collate concatenates multiple sequences into a single token stream.
                # `position_ids` resets to 0 at each boundary, but our labels are already next-token aligned by
                # Template._roll_labels(). Therefore the cross-subsequence supervision term lives at the *previous*
                # token index (the token right before a boundary start).
                #
                # Example (boundary at index b where position_ids[b] == 0):
                # - Bad term is: token[b-1] predicting token[b]
                # - In next-token-aligned labels, this appears at labels[b-1]
                boundary_starts = (text_position_ids == 0)
                prev = torch.zeros_like(boundary_starts, dtype=torch.bool)
                # Mask token b-1 when boundary starts at b.
                prev[..., :-1] = boundary_starts[..., 1:]
                labels = labels.clone()
                labels[prev] = -100
                # Also avoid any potential wrap-around supervision at the end of the concatenated stream.
                labels[..., -1] = -100
            labels = self.split(labels, dim=-1, position_ids=real_position_ids)
        if loss_scale is not None:
            loss_scale = torch.roll(loss_scale, shifts=-1, dims=-1)
            loss_scale = self.split(loss_scale, dim=-1, position_ids=real_position_ids)

        if position_ids is not None:
            position_ids = _split_position_ids_for_sp(position_ids, self.sp_world_size, self._sp_group)
        if extra_split_values is not None:
            for i in range(len(extra_values)):
                extra_values[i] = self.split(
                    extra_values[i], dim=extra_split_values[i][2], position_ids=real_position_ids)
        return input_ids, input_embeds, labels, position_ids, attention_mask, loss_scale, extra_values

    def _init_device_mesh(self, device_mesh: Optional[DeviceMesh] = None):
        """Initialize process groups for sequence parallel."""
        if not isinstance(device_mesh, DeviceMesh):
            raise RuntimeError('SequenceParallel requires a twinkle DeviceMesh for initialization.')

        self.device_mesh = device_mesh
        self.sp_world_size = self.world_size
        self.dp_world_size = device_mesh.data_world_size or 1
        self._sp_group = _get_sp_group_from_device_mesh(device_mesh, self.sp_world_size)
        if self._sp_group is None and self.sp_world_size > 1:
            raise RuntimeError('Failed to create sequence-parallel group from DeviceMesh.')

    @staticmethod
    def _is_packed_position_ids(position_ids: Optional[torch.Tensor]) -> bool:
        """Heuristic: detect packed samples by multiple (0,1,...) resets in position_ids.

        PackingDataset packs multiple sequences into one row by resetting position_ids to 0/1/... at each boundary.
        """
        text_position_ids = _extract_text_position_ids(position_ids)
        if text_position_ids is None:
            return False
        if text_position_ids.dim() == 1:
            text_position_ids = text_position_ids.unsqueeze(0)
        if text_position_ids.dim() != 2:
            return False
        # A batch may contain multiple packed samples; consider it "packed" if any row is packed.
        for i in range(text_position_ids.size(0)):
            row = text_position_ids[i]
            zero_count = int((row == 0).sum().item())
            one_count = int((row == 1).sum().item())
            if zero_count > 1 and one_count > 1:
                return True
        return False

    def prepare_inputs(self, inputs):
        """Prepare inputs

        1. set extra_kwargs['position_ids']
        2. split labels
        """
        position_ids = None
        input_ids = inputs.get('input_ids')
        position_ids = inputs.get('position_ids')
        text_position_ids = _extract_text_position_ids(position_ids)
        if position_ids is not None:
            self.extra_kwargs['position_ids'] = position_ids.clone()
        if text_position_ids is not None:
            self.extra_kwargs['text_position_ids'] = text_position_ids.clone()
        self.extra_kwargs['is_packed'] = self._is_packed_position_ids(text_position_ids)
        self.extra_kwargs['has_qwen35_linear_attn'] = self.has_qwen35_linear_attn
        if self.world_size is not None and self.world_size > 1 and self.has_qwen35_linear_attn and self.extra_kwargs[
                'is_packed']:
            raise RuntimeError(
                'SequenceParallel: packed batches are not supported for Qwen3.5 linear attention under SP. '
                'Packed reset semantics for recurrent states are not implemented.')
        if input_ids is not None:
            self.extra_kwargs['input_ids'] = input_ids.clone()
        if 'labels' in inputs:
            labels = inputs['labels']
            _, _, labels, _, _, _, _ = self.pad_and_split_inputs(
                None, None, labels, None, None, None, real_position_ids=position_ids)
            inputs['labels'] = labels
        return inputs


sequence_parallel = SequenceParallel()


@dataclass(frozen=True)
class SequenceParallelConfig:
    enabled: bool = True
    ulysses_size: Optional[int] = None
    gather_logits: bool = True
    loss_reduction: str = 'mean'
    compensate_fsdp_avg: bool = False


def _get_ulysses_size(device_mesh, sp_config: Optional[Dict[str, Any]] = None) -> int:
    if sp_config:
        cfg_size = sp_config.get('ulysses_size')
        if cfg_size is not None:
            return int(cfg_size)
    if device_mesh is None:
        return 1
    if getattr(device_mesh, 'ulysses_size', None) is not None:
        return int(device_mesh.ulysses_size)
    return 1


class SequenceParallelStrategy:
    """Ulysses sequence-parallel strategy implementation."""

    def __init__(
        self,
        device_mesh=None,
        sp_config: Optional[Union[Dict[str, Any], SequenceParallelConfig]] = None,
        model: Optional[torch.nn.Module] = None,
        tokenizer_id: Optional[str] = None,
    ):
        self.device_mesh = device_mesh
        if isinstance(sp_config, SequenceParallelConfig):
            self.sp_config = asdict(sp_config)
        elif sp_config is not None and is_dataclass(sp_config):
            self.sp_config = asdict(sp_config)
        else:
            self.sp_config = sp_config or {}
        self.enabled = bool(self.sp_config.get('enabled', True))
        self.ulysses_size = _get_ulysses_size(device_mesh, self.sp_config)
        self._model_ref = model
        self._tokenizer_id = tokenizer_id
        self._tokenizer = None
        self._initialized = False

    def _get_tokenizer(self) -> Optional[PreTrainedTokenizer]:
        if self._tokenizer is not None:
            return self._tokenizer
        if not self._tokenizer_id:
            return None
        try:
            from twinkle.template import Template

            self._tokenizer = Template(self._tokenizer_id).tokenizer
            return self._tokenizer
        except Exception:
            return None

    def initialize(self) -> bool:
        if not self.enabled or self.ulysses_size <= 1:
            return False
        if not dist.is_initialized():
            raise RuntimeError('torch.distributed must be initialized before enabling sequence parallel.')
        if not isinstance(self.device_mesh, DeviceMesh):
            raise RuntimeError('SequenceParallelStrategy requires a twinkle DeviceMesh when ulysses_size > 1.')
        if self._model_ref is None:
            raise RuntimeError('SequenceParallelStrategy requires a model reference to initialize.')
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            raise RuntimeError('SequenceParallelStrategy requires a tokenizer to initialize.')
        sequence_parallel.prepare(
            self.ulysses_size,
            self._model_ref,
            tokenizer,
            device_mesh=self.device_mesh,
        )
        self._initialized = True
        return True

    def preprocess_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled or self.ulysses_size <= 1:
            return inputs
        return sequence_parallel.prepare_inputs(inputs)

    def postprocess_outputs(self, outputs: Any) -> Any:
        if (not self.enabled or self.ulysses_size <= 1 or not self.sp_config.get('gather_logits', True)):
            return outputs
        # Twinkle expects dict-like ModelOutput containers in the main training path
        # (uses `.get(...)` and `outputs[...] = ...`). Keep SP postprocess consistent.
        if outputs is None or not hasattr(outputs, 'get') or not hasattr(outputs, '__setitem__'):
            raise TypeError('SequenceParallelStrategy.postprocess_outputs expects a dict-like ModelOutput. '
                            f'Got type={type(outputs)}')
        logits = outputs.get('logits', None)
        if logits is None or not torch.is_tensor(logits) or logits.dim() < 2:
            return outputs
        gathered = sequence_parallel.gather(logits, dim=1, position_ids=sequence_parallel.real_position_ids)
        # Scheme A: SP pads to make seq_len divisible by sp_size. Trim back to the original
        # (unpadded) length using the cached real_position_ids.
        text_pos = sequence_parallel.text_position_ids
        if text_pos is not None and torch.is_tensor(text_pos) and text_pos.dim() >= 2:
            gathered = gathered[:, :text_pos.shape[1]].contiguous()
        outputs['logits'] = gathered
        return outputs

    def reduce_loss(self, loss: torch.Tensor, labels: Optional[torch.Tensor], ignore_index: int = -100) -> torch.Tensor:
        if not self.enabled or self.ulysses_size <= 1:
            return loss
        if labels is None or sequence_parallel._sp_group is None:
            return loss
        # Compute global loss via autograd-aware all-reduce.
        reduction = str(self.sp_config.get('loss_reduction', 'mean')).lower()
        if reduction == 'none':
            raise ValueError("SequenceParallelStrategy.reduce_loss only supports reduction='sum' or 'mean'. "
                             'Please aggregate per-token losses before calling reduce_loss.')
        compensate_fsdp_avg = bool(self.sp_config.get('compensate_fsdp_avg', False))
        compensate_factor = float(self.ulysses_size if compensate_fsdp_avg else 1.0)
        sum_metric_scale = float(self.ulysses_size)

        class _ReduceSequenceParallelLoss(torch.autograd.Function):

            @staticmethod
            def forward(ctx, local_mean: torch.Tensor, num_valid_tokens: torch.Tensor) -> torch.Tensor:
                local_tokens = num_valid_tokens.detach().clone()
                local_sum = local_mean * local_tokens
                if local_tokens.item() == 0:
                    local_sum = torch.nan_to_num(local_sum)
                global_sum = local_sum.detach().clone()
                dist.all_reduce(global_sum, group=sequence_parallel._sp_group)
                global_tokens = num_valid_tokens.detach().clone()
                dist.all_reduce(global_tokens, group=sequence_parallel._sp_group)
                ctx.save_for_backward(local_tokens, global_tokens)
                if global_tokens.item() == 0:
                    return local_sum
                return global_sum / global_tokens

            @staticmethod
            def backward(ctx, grad_output: torch.Tensor):
                local_tokens, global_tokens = ctx.saved_tensors
                if global_tokens.item() == 0:
                    return torch.zeros_like(grad_output), None
                # d(global_mean)/d(local_mean) = local_tokens / global_tokens.
                grad_local_mean = grad_output * (local_tokens / global_tokens) * compensate_factor
                return grad_local_mean, None

        class _ReduceSequenceParallelSum(torch.autograd.Function):

            @staticmethod
            def forward(ctx, local_sum: torch.Tensor) -> torch.Tensor:
                ctx.sum_metric_scale = sum_metric_scale
                global_sum = local_sum.detach().clone()
                dist.all_reduce(global_sum, group=sequence_parallel._sp_group)
                # Keep logging/metric value aligned with non-SP sum semantics under
                # outer collect='mean' by removing one SP replication factor.
                return global_sum / ctx.sum_metric_scale

            @staticmethod
            def backward(ctx, grad_output: torch.Tensor):
                # Keep training gradient scale unchanged; forward-side scaling is for
                # logging/metric alignment under outer collect='mean'.
                return grad_output

        if reduction == 'sum':
            return _ReduceSequenceParallelSum.apply(loss)

        # Default to mean reduction: `loss` is local mean.
        num_valid_tokens = (labels != ignore_index).sum().to(loss.device)
        return _ReduceSequenceParallelLoss.apply(loss, num_valid_tokens)

    def wrap_model(self, model, optimizer=None):
        self.initialize()
        return model, optimizer

    def unwrap_model(self, model):
        return model
