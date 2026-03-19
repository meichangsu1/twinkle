import importlib
import os
from typing import Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist

from ..sequence_parallel import _SeqAllToAll


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


def _run_depthwise_causal_conv(
    tensor: torch.Tensor,
    *,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    module: torch.nn.Module,
    origin_causal_conv1d_fn,
) -> torch.Tensor:
    batch_size, seq_len, num_heads, inner_dim = tensor.shape
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


class Qwen35HeadParallelHelper:

    impl_name = 'head_parallel'

    @property
    def enabled(self) -> bool:
        return os.environ.get('QWEN35_SP_LINEAR_HEAD_PARALLEL', '0') == '1'

    def validate_runtime(self, sequence_parallel, module: torch.nn.Module) -> None:
        sp_world_size = int(sequence_parallel.sp_world_size or 1)
        if sp_world_size <= 1:
            return
        if module.num_v_heads % sp_world_size != 0:
            raise RuntimeError(
                'SequenceParallel: Qwen3.5 head-parallel linear attention requires '
                f'sp_world_size ({sp_world_size}) to divide linear_num_value_heads ({module.num_v_heads}).'
            )

    def _select_rule(self, module: torch.nn.Module, origin_rule, full_seq_len: int):
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

    def run(
        self,
        sequence_parallel,
        module: torch.nn.Module,
        origin_rule,
        origin_causal_conv1d_fn,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.validate_runtime(sequence_parallel, module)

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
        z_hp = _seq_to_head_shard(z, sequence_parallel)
        b_hp = _seq_to_head_shard(b, sequence_parallel)
        a_hp = _seq_to_head_shard(a, sequence_parallel)

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
        local_conv_weight = torch.cat([local_q_weight, local_k_weight, local_v_weight], dim=0)

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
            local_conv_bias = torch.cat([local_q_bias, local_k_bias, local_v_bias], dim=0)

        mixed_qkv_hp = _run_depthwise_causal_conv(
            mixed_qkv_hp,
            weight=local_conv_weight,
            bias=local_conv_bias,
            module=module,
            origin_causal_conv1d_fn=origin_causal_conv1d_fn,
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

        rule, chunk_size = self._select_rule(module, origin_rule, query_hp.shape[1])
        chunk_kwargs = {
            'g': g,
            'beta': beta,
            'initial_state': None,
            'output_final_state': False,
            'use_qk_l2norm_in_kernel': True,
        }
        if chunk_size is not None:
            chunk_kwargs['chunk_size'] = chunk_size
        core_attn_out, _ = rule(query_hp, key_hp, value_hp, **chunk_kwargs)

        core_attn_out = core_attn_out.reshape(-1, module.head_v_dim)
        z_hp = z_hp.reshape(-1, module.head_v_dim)
        core_attn_out = module.norm(core_attn_out, z_hp)
        core_attn_out = core_attn_out.reshape(batch_size, -1, local_v_heads, module.head_v_dim)

        core_attn_out = _head_to_seq_shard(core_attn_out, sequence_parallel)
        core_attn_out = core_attn_out.reshape(batch_size, local_seq_len, module.value_dim)
        return module.out_proj(core_attn_out)
