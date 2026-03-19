import importlib
import torch
import torch.distributed as dist
from dataclasses import asdict, dataclass, is_dataclass
from functools import partial
from transformers import PreTrainedTokenizer
from typing import Any, Dict, Optional, Tuple, Union

from twinkle.utils import DeviceMesh
from twinkle.utils.transformers_utils import get_llm_model


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
        res = single_all_to_all(input, scatter_idx, gather_idx, group)
        return res

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor) -> Tuple[None, torch.Tensor, None, None]:
        # Reverse scatter/gather in backward to match forward layout transform.
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
        self.linear_attention_model_patch = None
        self.linear_attention_model_patch_name = None
        self.extra_kwargs = {}

    @property
    def real_position_ids(self) -> torch.Tensor:
        """The real position ids, this is different from the position_ids in mrope"""
        return self.extra_kwargs.get('position_ids')

    @property
    def text_position_ids(self) -> Optional[torch.Tensor]:
        """Text-aligned position ids with shape [B, S]."""
        return self.extra_kwargs.get('text_position_ids')

    def _resolve_linear_attention_model_patch(self, base_model: torch.nn.Module):
        from .linear_attention import LINEAR_ATTENTION_MODEL_PATCHES

        for model_patch in LINEAR_ATTENTION_MODEL_PATCHES:
            if model_patch.match(base_model):
                return model_patch
        return None

    def _activate_linear_attention_model_patch(self, llm_model: torch.nn.Module, model: torch.nn.Module) -> None:
        self.linear_attention_model_patch = None
        self.linear_attention_model_patch_name = None
        self.extra_kwargs.pop('linear_attention_model_patch', None)
        self.extra_kwargs.pop('linear_attention_model_patch_impl', None)
        self.extra_kwargs.pop('linear_attention_model_patch_disabled_gradient_checkpointing', None)
        self.extra_kwargs.pop('linear_attention_model_patch_disabled_gradient_checkpointing_layers', None)

        if self.world_size is None or self.world_size <= 1:
            return

        model_patch = self._resolve_linear_attention_model_patch(llm_model)
        if model_patch is None:
            return
        if not model_patch.patch(llm_model, self):
            return

        self.linear_attention_model_patch = model_patch
        self.linear_attention_model_patch_name = model_patch.name
        self.extra_kwargs['linear_attention_model_patch'] = model_patch.name
        self.extra_kwargs['linear_attention_model_patch_impl'] = getattr(model_patch, 'impl_name', 'state_parallel')
        disabled = model_patch.maybe_disable_gc(model, self)
        self.extra_kwargs['linear_attention_model_patch_impl'] = getattr(model_patch, 'impl_name', 'state_parallel')
        self.extra_kwargs['linear_attention_model_patch_disabled_gradient_checkpointing_layers'] = list(disabled)

    def _validate_linear_attention_inputs(self) -> None:
        if self.linear_attention_model_patch is None:
            return
        self.linear_attention_model_patch.validate_inputs(self)

    def _should_build_causal_mask(self) -> bool:
        if self.linear_attention_model_patch is None:
            return True
        return bool(self.linear_attention_model_patch.should_build_causal_mask())

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
                cached_full_attention_mask = self.extra_kwargs.get('strict_full_attention_mask')
                if (cached_full_attention_mask is not None and torch.is_tensor(cached_full_attention_mask)
                        and cached_full_attention_mask.dim() == 2
                        and getattr(config, '_attn_implementation', None) in {'sdpa', 'eager'}):
                    # pad_and_split_inputs already cached the full 2D padding mask before SP split. Reuse it here
                    # instead of issuing another runtime gather from local shards, which can desync full-attention
                    # models that materialize causal masks through create_causal_mask.
                    attention_mask = cached_full_attention_mask.to(device=inputs_embeds.device)
                elif (attention_mask is not None and torch.is_tensor(attention_mask) and attention_mask.dim() == 2
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
                cached_full_cache_position = self.extra_kwargs.get('strict_full_cache_position')
                if cached_full_cache_position is not None and torch.is_tensor(cached_full_cache_position):
                    cache_position = cached_full_cache_position.to(device=inputs_embeds.device)
                elif cache_position is None:
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
                        packed_position_ids = self.real_position_ids if self.real_position_ids is not None else position_ids
                        if packed_position_ids is not None:
                            kwargs['position_ids'] = packed_position_ids
                        # Treat SP-alignment padding (-1) as separate 1-token sequences by mapping -1 -> 0.
                        pos = packed_position_ids
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
                        text_position_ids = self.real_position_ids if self.real_position_ids is not None else position_ids
                        if text_position_ids is not None:
                            kwargs['position_ids'] = text_position_ids
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

        self._activate_linear_attention_model_patch(llm_model, model)
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
        self._validate_linear_attention_inputs()
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
            # Keep model-facing position_ids non-negative. Some optimized CUDA kernels assume valid gather indices and
            # can assert on the SP-alignment sentinel (-1). We keep the true padded layout in real_position_ids for
            # mask / packed-sequence bookkeeping.
            position_ids = self.pad(position_ids, padding_value=0, position_ids=real_position_ids, dim=-1)
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
        self.extra_kwargs.pop('strict_full_attention_mask', None)
        self.extra_kwargs.pop('strict_full_cache_position', None)
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
            self.extra_kwargs['strict_full_attention_mask'] = attention_mask.clone()
            self.extra_kwargs['strict_full_cache_position'] = cache_position.clone()
            # For SP>1 keep 2D masks and let full-attention gather to global mask later.
            # Prebuilding 4D causal masks before split can break full-sequence semantics.
            if (self.world_size == 1 and hasattr(self, 'causal_mask_func') and self.causal_mask_func is not None
                    and self._should_build_causal_mask()):
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
        self._validate_linear_attention_inputs()
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
