import importlib
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from twinkle.utils.transformers_utils import get_llm_model

from .qwen35_strict import Qwen35StrictFullSeqHelper


def _get_text_model(base_model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(base_model, 'language_model'):
        return base_model.language_model
    return base_model


def _sp_prev_next_rank(
    sp_group: Optional[dist.ProcessGroup],
) -> Tuple[int, int, Optional[int], Optional[int]]:
    if sp_group is None:
        return 0, 1, None, None
    sp_rank = dist.get_rank(sp_group)
    sp_world_size = dist.get_world_size(sp_group)
    prev_rank = dist.get_global_rank(sp_group, sp_rank - 1) if sp_rank > 0 else None
    next_rank = dist.get_global_rank(sp_group, sp_rank + 1) if sp_rank + 1 < sp_world_size else None
    return sp_rank, sp_world_size, prev_rank, next_rank


class _QwenLinearStateParallelFn(torch.autograd.Function):
    """State-parallel autograd for Qwen3.5 linear-attention chunk rule."""

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

        ctx.save_for_backward(query, key, value, g, beta, initial_state)
        return local_output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
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
                f'local_seq_len={ctx.local_seq_len}, halo={ctx.halo}.'
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


class Qwen35LinearAttentionSPModelPatch:

    name = 'qwen35'

    def __init__(self):
        self.patched_linear_layer_indices = set()
        self.strict_full_seq = Qwen35StrictFullSeqHelper()

    def match(self, base_model: torch.nn.Module) -> bool:
        text_model = _get_text_model(base_model)
        for module in text_model.modules():
            if module.__class__.__name__ != 'Qwen3_5GatedDeltaNet':
                continue
            if getattr(module, 'chunk_gated_delta_rule', None) is not None:
                return True
        return False

    def should_build_causal_mask(self) -> bool:
        return False

    def validate_inputs(self, sequence_parallel) -> None:
        if sequence_parallel.world_size is None or sequence_parallel.world_size <= 1:
            return
        if sequence_parallel.extra_kwargs.get('is_packed', False):
            raise RuntimeError(
                'SequenceParallel: packed batches are not supported for Qwen3.5 linear attention under SP. '
                'Packed reset semantics for recurrent states are not implemented.'
            )

    def maybe_disable_gc(self, model: torch.nn.Module, sequence_parallel) -> list[int]:
        llm_model = get_llm_model(model)
        was_gc_enabled = bool(getattr(model, 'is_gradient_checkpointing', False))
        if not was_gc_enabled:
            was_gc_enabled = bool(getattr(llm_model, 'is_gradient_checkpointing', False))
        if not was_gc_enabled:
            return []
        return self._disable_gc_for_patched_layers(model)

    def patch(self, base_model: torch.nn.Module, sequence_parallel) -> bool:
        text_model = _get_text_model(base_model)
        has_linear_attn = False
        self.patched_linear_layer_indices = set()
        for module in text_model.modules():
            if module.__class__.__name__ != 'Qwen3_5GatedDeltaNet':
                continue
            origin_rule = getattr(module, 'chunk_gated_delta_rule', None)
            if origin_rule is None:
                continue
            has_linear_attn = True
            layer_idx = getattr(module, 'layer_idx', None)
            if isinstance(layer_idx, int) and layer_idx >= 0:
                self.patched_linear_layer_indices.add(layer_idx)
            if getattr(module, '_twinkle_qwen35_linear_sp_patched', False):
                continue
            origin_forward = getattr(module, 'forward', None)
            module.chunk_gated_delta_rule = self._wrap_chunk_rule(sequence_parallel, module, origin_rule)
            if origin_forward is not None:
                module.forward = self._wrap_linear_forward(
                    sequence_parallel,
                    module,
                    origin_forward,
                    origin_rule,
                    getattr(module, 'causal_conv1d_fn', None),
                )
            module._twinkle_qwen35_linear_sp_patched = True
        return has_linear_attn

    def _disable_gc_for_patched_layers(self, model: torch.nn.Module) -> list[int]:
        text_model = _get_text_model(get_llm_model(model))
        layers = getattr(text_model, 'layers', None)
        if layers is None:
            return []

        disabled_layers = []
        for layer_idx, layer in enumerate(layers):
            if layer_idx not in self.patched_linear_layer_indices:
                continue
            if getattr(layer, 'layer_type', None) != 'linear_attention':
                continue

            was_enabled = bool(getattr(layer, 'gradient_checkpointing', False))
            if hasattr(layer, 'gradient_checkpointing'):
                try:
                    layer.gradient_checkpointing = False
                except Exception:
                    pass
            if hasattr(layer, '_gradient_checkpointing_func'):
                try:
                    layer._gradient_checkpointing_func = None
                except Exception:
                    pass
            if was_enabled:
                disabled_layers.append(layer_idx)

        return disabled_layers

    def _wrap_chunk_rule(self, sequence_parallel, module: torch.nn.Module, origin_rule):
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

            if sequence_parallel.sp_world_size is None or sequence_parallel.sp_world_size <= 1 or sequence_parallel._sp_group is None:
                return origin_rule(query, key, value, **call_kwargs)
            if initial_state is not None or output_final_state:
                return origin_rule(query, key, value, **call_kwargs)

            self.validate_inputs(sequence_parallel)

            sp_rule = origin_rule
            sp_chunk_size = int(chunk_size) if chunk_size is not None else None
            effective_chunk_size = sp_chunk_size or 64
            if recurrent_rule is not None and query.shape[1] % effective_chunk_size != 0:
                sp_rule = recurrent_rule_wrapper
                sp_chunk_size = None

            output = _QwenLinearStateParallelFn.apply(
                query,
                key,
                value,
                g,
                beta,
                sequence_parallel._sp_group,
                sp_rule,
                sp_chunk_size,
                bool(use_qk_l2norm_in_kernel),
            )
            return output, None

        wrapped_chunk_rule._twinkle_origin_rule = origin_rule
        wrapped_chunk_rule._twinkle_wrapped_module = module.__class__.__name__
        return wrapped_chunk_rule

    def _wrap_causal_conv1d_fn(self, sequence_parallel, module: torch.nn.Module, origin_causal_conv1d_fn):

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
            if sequence_parallel.sp_world_size is None or sequence_parallel.sp_world_size <= 1 or sequence_parallel._sp_group is None:
                return run_origin(x, weight, bias, activation, seq_idx)

            halo = int(module.conv_kernel_size) - 1
            if halo <= 0:
                return run_origin(x, weight, bias, activation, seq_idx)

            extended_x = _LeftHaloExchangeFn.apply(
                x.contiguous(),
                halo,
                -1,
                sequence_parallel._sp_group,
                int(getattr(module, 'layer_idx', 0)),
            )
            extended_output = run_origin(extended_x, weight, bias, activation, seq_idx)
            local_seq_len = x.shape[-1]
            return extended_output.narrow(-1, halo, local_seq_len).contiguous()

        wrapped_causal_conv1d._twinkle_origin_causal_conv1d_fn = origin_causal_conv1d_fn
        wrapped_causal_conv1d._twinkle_wrapped_module = module.__class__.__name__
        return wrapped_causal_conv1d

    def _wrap_linear_forward(
        self,
        sequence_parallel,
        module: torch.nn.Module,
        origin_forward,
        origin_rule,
        origin_causal_conv1d_fn,
    ):
        halo_causal_conv1d_fn = self._wrap_causal_conv1d_fn(sequence_parallel, module, origin_causal_conv1d_fn)

        def wrapped_forward(
            hidden_states: torch.Tensor,
            cache_params=None,
            cache_position: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
        ):
            if sequence_parallel.sp_world_size is None or sequence_parallel.sp_world_size <= 1 or sequence_parallel._sp_group is None:
                return origin_forward(
                    hidden_states,
                    cache_params=cache_params,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                )

            self.validate_inputs(sequence_parallel)

            use_precomputed_states = (
                cache_params is not None
                and getattr(cache_params, 'has_previous_state', False)
                and hidden_states.shape[1] == 1
                and cache_position is not None
            )
            if use_precomputed_states:
                return origin_forward(
                    hidden_states,
                    cache_params=cache_params,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                )

            if self.strict_full_seq.enabled:
                return self.strict_full_seq.run(
                    sequence_parallel=sequence_parallel,
                    module=module,
                    origin_forward=origin_forward,
                    origin_rule=origin_rule,
                    origin_causal_conv1d_fn=origin_causal_conv1d_fn,
                    hidden_states=hidden_states,
                    cache_params=cache_params,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                )

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

        wrapped_forward._twinkle_origin_forward = origin_forward
        wrapped_forward._twinkle_wrapped_module = module.__class__.__name__
        return wrapped_forward
