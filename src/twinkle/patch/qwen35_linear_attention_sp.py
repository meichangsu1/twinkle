import importlib
from types import MethodType
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .base import Patch
from ..model.transformers.strategy.sequence_parallel import _SeqAllToAll, _extract_text_position_ids, _gather_position_ids_for_sp


def _sp_rank(sequence_parallel) -> int:
    sp_group = getattr(sequence_parallel, '_sp_group', None)
    if sp_group is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank(sp_group)
    return 0


def _seq_to_head_shard(tensor: torch.Tensor, sequence_parallel) -> torch.Tensor:
    return _SeqAllToAll.apply(sequence_parallel._sp_group, tensor.contiguous(), 2, 1)


def _head_to_seq_shard(tensor: torch.Tensor, sequence_parallel) -> torch.Tensor:
    return _SeqAllToAll.apply(sequence_parallel._sp_group, tensor.contiguous(), 1, 2)


def _slice_conv_channel_params(
    flat_params: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    head_indices: torch.Tensor,
) -> torch.Tensor:
    reshaped = flat_params.reshape(num_heads, head_dim, *flat_params.shape[1:])
    return reshaped.index_select(0, head_indices).reshape(-1, *flat_params.shape[1:]).contiguous()


def _has_nontrivial_sequence_spans(
    sequence_spans: Optional[List[List[Tuple[int, int]]]],
    seq_len: int,
) -> bool:
    if sequence_spans is None:
        return False
    for row_spans in sequence_spans:
        if len(row_spans) != 1 or row_spans[0] != (0, seq_len):
            return True
    return False


def _run_depthwise_causal_conv(
    tensor: torch.Tensor,
    *,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    module: torch.nn.Module,
    origin_causal_conv1d_fn,
    sequence_spans: Optional[List[List[Tuple[int, int]]]] = None,
) -> torch.Tensor:
    batch_size, seq_len, num_heads, inner_dim = tensor.shape
    if _has_nontrivial_sequence_spans(sequence_spans, seq_len):
        conv_output = tensor.new_zeros((batch_size, seq_len, num_heads, inner_dim))
        for batch_idx, spans in enumerate(sequence_spans or []):
            for start, end in spans:
                if end <= start:
                    continue
                conv_output[batch_idx:batch_idx + 1, start:end] = _run_depthwise_causal_conv(
                    tensor[batch_idx:batch_idx + 1, start:end],
                    weight=weight,
                    bias=bias,
                    module=module,
                    origin_causal_conv1d_fn=origin_causal_conv1d_fn,
                    sequence_spans=None,
                )
        return conv_output.contiguous()

    conv_input = tensor.reshape(batch_size, seq_len, num_heads * inner_dim).transpose(1, 2).contiguous()

    if origin_causal_conv1d_fn is not None:
        conv_output = origin_causal_conv1d_fn(
            x=conv_input,
            weight=weight,
            bias=bias,
            activation=module.activation,
            seq_idx=None,
        )
    else:
        conv_weight = weight.unsqueeze(1) if weight.dim() == 2 else weight
        conv_output = F.conv1d(
            conv_input,
            conv_weight,
            bias=bias,
            padding=module.conv_kernel_size - 1,
            groups=conv_input.shape[1],
        )
        conv_output = conv_output[:, :, :seq_len]
        conv_output = module.act(conv_output)

    return conv_output.transpose(1, 2).reshape(batch_size, seq_len, num_heads, inner_dim).contiguous()


def _interleave_qkv_value_head_params(
    query_params: torch.Tensor,
    key_params: torch.Tensor,
    value_params: torch.Tensor,
    *,
    local_v_heads: int,
) -> torch.Tensor:
    query_params = query_params.reshape(local_v_heads, -1, *query_params.shape[1:])
    key_params = key_params.reshape(local_v_heads, -1, *key_params.shape[1:])
    value_params = value_params.reshape(local_v_heads, -1, *value_params.shape[1:])
    return torch.cat([query_params, key_params, value_params], dim=1).reshape(-1, *query_params.shape[2:]).contiguous()


def _resolve_full_text_position_ids(
    sequence_parallel,
    *,
    expected_seq_len: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    extra_kwargs = getattr(sequence_parallel, 'extra_kwargs', {}) or {}
    candidates = [
        getattr(sequence_parallel, 'text_position_ids', None),
        getattr(sequence_parallel, 'real_position_ids', None),
        extra_kwargs.get('text_position_ids'),
        extra_kwargs.get('position_ids'),
    ]
    sp_world_size = int(getattr(sequence_parallel, 'sp_world_size', 1) or 1)
    sp_group = getattr(sequence_parallel, '_sp_group', None)

    for position_ids in candidates:
        text_position_ids = _extract_text_position_ids(position_ids)
        if text_position_ids is None:
            continue
        if text_position_ids.shape[-1] == expected_seq_len:
            return text_position_ids.to(device=device).contiguous()
        if text_position_ids.shape[-1] * sp_world_size == expected_seq_len and sp_group is not None:
            gathered = _gather_position_ids_for_sp(text_position_ids, sp_world_size, sp_group)
            if gathered is not None and gathered.shape[-1] == expected_seq_len:
                return gathered.to(device=device).contiguous()
    return None


def _build_sequence_spans(text_position_ids: Optional[torch.Tensor]) -> Optional[List[List[Tuple[int, int]]]]:
    if text_position_ids is None or not torch.is_tensor(text_position_ids):
        return None
    if text_position_ids.dim() == 1:
        text_position_ids = text_position_ids.unsqueeze(0)
    if text_position_ids.dim() != 2:
        raise ValueError(
            'SequenceParallel: expected text-aligned position_ids with shape [B, S] for Qwen3.5 head-parallel varlen.'
        )

    sequence_spans: List[List[Tuple[int, int]]] = []
    for row in text_position_ids:
        valid = row.ge(0)
        if not bool(valid.any()):
            sequence_spans.append([])
            continue

        prev_valid = torch.cat([valid.new_zeros(1), valid[:-1]], dim=0)
        start_mask = valid & ((~prev_valid) | row.eq(0))
        start_indices = torch.nonzero(start_mask, as_tuple=False).flatten()
        if start_indices.numel() == 0:
            start_indices = torch.nonzero(valid, as_tuple=False).flatten()[:1]

        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
        last_valid_end = int(valid_indices[-1].item()) + 1

        row_spans: List[Tuple[int, int]] = []
        start_values = [int(idx) for idx in start_indices.tolist()]
        end_values = start_values[1:] + [last_valid_end]
        for start, end in zip(start_values, end_values):
            if end <= start:
                continue
            invalid = torch.nonzero(~valid[start:end], as_tuple=False)
            if invalid.numel() > 0:
                end = start + int(invalid[0].item())
            if end > start:
                row_spans.append((start, end))
        sequence_spans.append(row_spans)
    return sequence_spans
def _validate_runtime(sequence_parallel, module: torch.nn.Module) -> None:
    sp_world_size = int(sequence_parallel.sp_world_size or 1)
    if sp_world_size <= 1:
        return
    if module.num_v_heads % sp_world_size != 0:
        raise RuntimeError(
            'SequenceParallel: Qwen3.5 head-parallel linear attention requires '
            f'sp_world_size ({sp_world_size}) to divide linear_num_value_heads ({module.num_v_heads}).'
        )


def _select_rule(module: torch.nn.Module, origin_rule, full_seq_len: int):
    recurrent_rule = getattr(module, 'recurrent_gated_delta_rule', None)
    module_impl = importlib.import_module(module.__class__.__module__)
    torch_recurrent_rule = getattr(module_impl, 'torch_recurrent_gated_delta_rule', None)
    effective_chunk_size = 64
    if recurrent_rule is not None and full_seq_len % effective_chunk_size != 0:
        rule = torch_recurrent_rule or recurrent_rule

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
            del chunk_size
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

        return recurrent_rule_wrapper, None

    return origin_rule, None


def _run_rule(
    module: torch.nn.Module,
    origin_rule,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    sequence_spans: Optional[List[List[Tuple[int, int]]]],
) -> torch.Tensor:
    seq_len = query.shape[1]
    if not _has_nontrivial_sequence_spans(sequence_spans, seq_len):
        rule, chunk_size = _select_rule(module, origin_rule, seq_len)
        chunk_kwargs = {
            'g': g,
            'beta': beta,
            'initial_state': None,
            'output_final_state': False,
            'use_qk_l2norm_in_kernel': True,
        }
        if chunk_size is not None:
            chunk_kwargs['chunk_size'] = chunk_size
        core_attn_out, _ = rule(query, key, value, **chunk_kwargs)
        return core_attn_out

    core_attn_out = value.new_zeros(value.shape)
    for batch_idx, spans in enumerate(sequence_spans or []):
        for start, end in spans:
            if end <= start:
                continue
            rule, chunk_size = _select_rule(module, origin_rule, end - start)
            chunk_kwargs = {
                'g': g[batch_idx:batch_idx + 1, start:end],
                'beta': beta[batch_idx:batch_idx + 1, start:end],
                'initial_state': None,
                'output_final_state': False,
                'use_qk_l2norm_in_kernel': True,
            }
            if chunk_size is not None:
                chunk_kwargs['chunk_size'] = chunk_size
            segment_out, _ = rule(
                query[batch_idx:batch_idx + 1, start:end],
                key[batch_idx:batch_idx + 1, start:end],
                value[batch_idx:batch_idx + 1, start:end],
                **chunk_kwargs,
            )
            core_attn_out[batch_idx:batch_idx + 1, start:end] = segment_out
    return core_attn_out


def _run_qwen35_head_parallel(
    sequence_parallel,
    module: torch.nn.Module,
    origin_rule,
    origin_causal_conv1d_fn,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _validate_runtime(sequence_parallel, module)

    module_impl = importlib.import_module(module.__class__.__module__)
    apply_mask_to_padding_states = getattr(module_impl, 'apply_mask_to_padding_states')

    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, local_seq_len, _ = hidden_states.shape
    if local_seq_len <= 0:
        raise RuntimeError('SequenceParallel: Qwen3.5 head-parallel linear attention requires non-empty local sequences.')

    mixed_qkv = module.in_proj_qkv(hidden_states)
    z = module.in_proj_z(hidden_states).reshape(batch_size, local_seq_len, module.num_v_heads, module.head_v_dim)
    b = module.in_proj_b(hidden_states).reshape(batch_size, local_seq_len, module.num_v_heads, 1)
    a = module.in_proj_a(hidden_states).reshape(batch_size, local_seq_len, module.num_v_heads, 1)

    query_proj, key_proj, value_proj = torch.split(
        mixed_qkv,
        [module.key_dim, module.key_dim, module.value_dim],
        dim=-1,
    )
    query_proj = query_proj.reshape(batch_size, local_seq_len, module.num_k_heads, module.head_k_dim)
    key_proj = key_proj.reshape(batch_size, local_seq_len, module.num_k_heads, module.head_k_dim)
    value_proj = value_proj.reshape(batch_size, local_seq_len, module.num_v_heads, module.head_v_dim)

    qk_expand = module.num_v_heads // module.num_k_heads
    if qk_expand * module.num_k_heads != module.num_v_heads:
        raise RuntimeError(
            'SequenceParallel: Qwen3.5 head-parallel linear attention requires '
            'linear_num_value_heads to be an integer multiple of linear_num_key_heads.'
        )

    query_proj = query_proj.repeat_interleave(qk_expand, dim=2)
    key_proj = key_proj.repeat_interleave(qk_expand, dim=2)
    mixed_qkv_expanded = torch.cat([query_proj, key_proj, value_proj], dim=-1)

    mixed_qkv_hp = _seq_to_head_shard(mixed_qkv_expanded, sequence_parallel)
    b_hp = _seq_to_head_shard(b, sequence_parallel)
    a_hp = _seq_to_head_shard(a, sequence_parallel)
    full_seq_len = mixed_qkv_hp.shape[1]

    full_text_position_ids = _resolve_full_text_position_ids(
        sequence_parallel,
        expected_seq_len=full_seq_len,
        device=hidden_states.device,
    )
    is_packed = bool(getattr(sequence_parallel, 'extra_kwargs', {}).get('is_packed', False))
    if is_packed and full_text_position_ids is None:
        raise RuntimeError(
            'SequenceParallel: packed Qwen3.5 head-parallel linear attention requires full position_ids metadata.'
        )
    sequence_spans = _build_sequence_spans(full_text_position_ids)

    sp_world_size = int(sequence_parallel.sp_world_size or 1)
    local_v_heads = module.num_v_heads // sp_world_size
    sp_rank = _sp_rank(sequence_parallel)
    start_v = sp_rank * local_v_heads
    end_v = start_v + local_v_heads
    local_value_head_indices = torch.arange(start_v, end_v, device=hidden_states.device, dtype=torch.long)
    local_qk_head_indices = torch.div(local_value_head_indices, qk_expand, rounding_mode='floor')

    conv_weight = module.conv1d.weight.squeeze(1)
    q_weight = conv_weight[:module.key_dim]
    k_weight = conv_weight[module.key_dim:2 * module.key_dim]
    v_weight = conv_weight[2 * module.key_dim:]
    local_q_weight = _slice_conv_channel_params(
        q_weight,
        num_heads=module.num_k_heads,
        head_dim=module.head_k_dim,
        head_indices=local_qk_head_indices,
    )
    local_k_weight = _slice_conv_channel_params(
        k_weight,
        num_heads=module.num_k_heads,
        head_dim=module.head_k_dim,
        head_indices=local_qk_head_indices,
    )
    local_v_weight = _slice_conv_channel_params(
        v_weight,
        num_heads=module.num_v_heads,
        head_dim=module.head_v_dim,
        head_indices=local_value_head_indices,
    )
    local_conv_weight = _interleave_qkv_value_head_params(
        local_q_weight,
        local_k_weight,
        local_v_weight,
        local_v_heads=local_v_heads,
    )

    conv_bias = module.conv1d.bias
    local_conv_bias = None
    if conv_bias is not None:
        q_bias = conv_bias[:module.key_dim]
        k_bias = conv_bias[module.key_dim:2 * module.key_dim]
        v_bias = conv_bias[2 * module.key_dim:]
        local_q_bias = _slice_conv_channel_params(
            q_bias,
            num_heads=module.num_k_heads,
            head_dim=module.head_k_dim,
            head_indices=local_qk_head_indices,
        )
        local_k_bias = _slice_conv_channel_params(
            k_bias,
            num_heads=module.num_k_heads,
            head_dim=module.head_k_dim,
            head_indices=local_qk_head_indices,
        )
        local_v_bias = _slice_conv_channel_params(
            v_bias,
            num_heads=module.num_v_heads,
            head_dim=module.head_v_dim,
            head_indices=local_value_head_indices,
        )
        local_conv_bias = _interleave_qkv_value_head_params(
            local_q_bias,
            local_k_bias,
            local_v_bias,
            local_v_heads=local_v_heads,
        )

    mixed_qkv_hp = _run_depthwise_causal_conv(
        mixed_qkv_hp,
        weight=local_conv_weight,
        bias=local_conv_bias,
        module=module,
        origin_causal_conv1d_fn=origin_causal_conv1d_fn,
        sequence_spans=sequence_spans,
    )
    query_hp, key_hp, value_hp = torch.split(
        mixed_qkv_hp,
        [module.head_k_dim, module.head_k_dim, module.head_v_dim],
        dim=-1,
    )

    local_dt_bias = module.dt_bias[start_v:end_v]
    local_a_log = module.A_log[start_v:end_v]
    beta = b_hp.squeeze(-1).sigmoid()
    g = -local_a_log.float().exp() * F.softplus(a_hp.squeeze(-1).float() + local_dt_bias)

    core_attn_out = _run_rule(
        module,
        origin_rule,
        query_hp,
        key_hp,
        value_hp,
        g,
        beta,
        sequence_spans,
    )
    core_attn_out = _head_to_seq_shard(core_attn_out, sequence_parallel)
    core_attn_out = core_attn_out.reshape(batch_size, local_seq_len, module.num_v_heads, module.head_v_dim)
    core_attn_out = module.norm(
        core_attn_out.reshape(-1, module.head_v_dim),
        z.reshape(-1, module.head_v_dim),
    )
    core_attn_out = core_attn_out.reshape(batch_size, local_seq_len, module.value_dim)
    return module.out_proj(core_attn_out)


def _bind_sequence_parallel(self, sequence_parallel, *, impl_name: Optional[str] = None) -> None:
    del impl_name
    self._twinkle_sequence_parallel = sequence_parallel


def _unbind_sequence_parallel(self) -> None:
    self._twinkle_sequence_parallel = None


def _should_run_qwen35_sp_forward(
    self,
    sequence_parallel,
    *,
    cache_params=None,
    cache_position: Optional[torch.Tensor] = None,
    hidden_states: Optional[torch.Tensor] = None,
) -> bool:
    del cache_position
    if sequence_parallel is None:
        return False
    sp_world_size = int(getattr(sequence_parallel, 'sp_world_size', 1) or 1)
    if sp_world_size <= 1 or getattr(sequence_parallel, '_sp_group', None) is None:
        return False
    if cache_params is not None:
        return False
    if hidden_states is not None and hidden_states.shape[1] == 1:
        return False
    return True


def _qwen35_linear_attention_sp_forward(
    self,
    hidden_states: torch.Tensor,
    cache_params=None,
    cache_position: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
):
    origin_forward = getattr(self, '_twinkle_origin_forward')
    sequence_parallel = getattr(self, '_twinkle_sequence_parallel', None)

    if not _should_run_qwen35_sp_forward(
        self,
        sequence_parallel,
        cache_params=cache_params,
        cache_position=cache_position,
        hidden_states=hidden_states,
    ):
        kwargs.pop('cu_seq_lens_q', None)
        return origin_forward(
            hidden_states,
            cache_params=cache_params,
            cache_position=cache_position,
            attention_mask=attention_mask,
            **kwargs,
        )

    return _run_qwen35_head_parallel(
        sequence_parallel=sequence_parallel,
        module=self,
        origin_rule=getattr(self, 'chunk_gated_delta_rule', None),
        origin_causal_conv1d_fn=getattr(self, 'causal_conv1d_fn', None),
        hidden_states=hidden_states,
        attention_mask=attention_mask,
    )


class Qwen35LinearAttentionSPPatch(Patch):

    def __call__(self, module, *args, **kwargs):
        del args, kwargs
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

        from twinkle import requires

        requires('transformers<5.0')
        if getattr(module, '_twinkle_qwen35_linear_attention_sp_patch_applied', False):
            return

        patched_modules = 0
        for submodule in module.modules():
            if not isinstance(submodule, Qwen3_5GatedDeltaNet):
                continue
            if getattr(submodule, 'chunk_gated_delta_rule', None) is None:
                continue
            if getattr(submodule, '_twinkle_qwen35_linear_attention_sp_patch', False):
                continue

            submodule._twinkle_origin_forward = submodule.forward
            submodule._twinkle_sequence_parallel = None
            submodule.twinkle_bind_sequence_parallel = MethodType(_bind_sequence_parallel, submodule)
            submodule.twinkle_unbind_sequence_parallel = MethodType(_unbind_sequence_parallel, submodule)
            submodule.forward = MethodType(_qwen35_linear_attention_sp_forward, submodule)
            submodule._twinkle_qwen35_linear_attention_sp_patch = True
            patched_modules += 1

        module._twinkle_qwen35_linear_attention_sp_patch_applied = True
        module._twinkle_qwen35_linear_attention_sp_patch_count = patched_modules
        return patched_modules


__all__ = ['Qwen35LinearAttentionSPPatch']
