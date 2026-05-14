import functools
import torch
import torch_npu


class GmmFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, group_list: torch.Tensor, weight_ekn: torch.Tensor):
        assert x.dim() == 2, f'x must be [M, K], got {tuple(x.shape)}'
        assert group_list.dim() == 1, f'group_list must be [E], got {tuple(group_list.shape)}'
        assert weight_ekn.dim() == 3, f'weight_ekn must be [E, K, N], got {tuple(weight_ekn.shape)}'
        assert group_list.numel() == weight_ekn.size(0), (
            f'group_list len {group_list.numel()} != num_experts {weight_ekn.size(0)}')
        assert x.size(1) == weight_ekn.size(1), (
            f'input dim mismatch: x.shape={tuple(x.shape)}, weight_ekn.shape={tuple(weight_ekn.shape)}')

        group_list = group_list.to(torch.int64)

        ctx.save_for_backward(x, group_list, weight_ekn)

        outputs = torch_npu.npu_grouped_matmul(
            [x],
            [weight_ekn],
            group_list=group_list,
            group_type=0,
            split_item=2,
            group_list_type=1,
        )
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, group_list, weight_ekn = ctx.saved_tensors

        grad_input = torch_npu.npu_grouped_matmul(
            [grad_output],
            [weight_ekn.transpose(-2, -1).contiguous()],
            bias=None,
            group_list=group_list,
            group_type=0,
            split_item=2,
            group_list_type=1,
        )[0]

        grad_weight = torch_npu.npu_grouped_matmul(
            [x.transpose(0, 1)],
            [grad_output],
            bias=None,
            group_list=group_list,
            group_type=2,
            split_item=3,
            group_list_type=1,
        )[0]

        return grad_input, None, grad_weight.contiguous()


def _grouped_mm_npu(input: torch.Tensor, weight_ekn: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    assert input.dim() == 2, f'input must be [M, K], got {tuple(input.shape)}'
    assert weight_ekn.dim() == 3, f'weight_ekn must be [E, K, N], got {tuple(weight_ekn.shape)}'
    assert offs.dim() == 1, f'offs must be [E], got {tuple(offs.shape)}'
    assert weight_ekn.size(0) == offs.numel(), (
        f'weight_ekn.size(0)={weight_ekn.size(0)} != offs.numel()={offs.numel()}')

    counts = torch.empty_like(offs)
    counts[0] = offs[0]
    if offs.numel() > 1:
        counts[1:] = offs[1:] - offs[:-1]
    counts = counts.to(torch.int64)

    return GmmFunction.apply(input, counts, weight_ekn)


def apply_hf_moe_grouped_mm_patch():
    import transformers.integrations.moe as hf_moe

    hf_moe._grouped_mm = _grouped_mm_npu
    print('[PATCH] transformers.integrations.moe._grouped_mm -> _grouped_mm_npu')


def _deepseek_v4_rms_norm_forward_npu(self, hidden_states: torch.Tensor) -> torch.Tensor:
    if hidden_states.dtype != self.weight.dtype:
        hidden_states = hidden_states.to(self.weight.dtype)
    return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]


def _deepseek_v4_unweighted_rms_norm_forward_npu(self, hidden_states: torch.Tensor) -> torch.Tensor:
    weight = getattr(self, '_twinkle_npu_rms_weight', None)
    if (
        weight is None
        or weight.device != hidden_states.device
        or weight.dtype != hidden_states.dtype
        or weight.shape[-1] != hidden_states.shape[-1]
    ):
        weight = torch.ones(hidden_states.shape[-1], device=hidden_states.device, dtype=hidden_states.dtype)
        self._twinkle_npu_rms_weight = weight
    return torch_npu.npu_rms_norm(hidden_states, weight, epsilon=self.eps)[0]


def apply_deepseek_v4_rms_norm_patch():
    try:
        from transformers.models.deepseek_v4 import modeling_deepseek_v4
    except Exception as exc:
        print(f'[PATCH] skip DeepSeek V4 RMSNorm patch: {exc}')
        return

    if not getattr(modeling_deepseek_v4.DeepseekV4RMSNorm.forward, '_twinkle_npu_patched', False):
        _deepseek_v4_rms_norm_forward_npu._twinkle_npu_patched = True
        _deepseek_v4_rms_norm_forward_npu._twinkle_old_forward = modeling_deepseek_v4.DeepseekV4RMSNorm.forward
        modeling_deepseek_v4.DeepseekV4RMSNorm.forward = _deepseek_v4_rms_norm_forward_npu

    if not getattr(modeling_deepseek_v4.DeepseekV4UnweightedRMSNorm.forward, '_twinkle_npu_patched', False):
        _deepseek_v4_unweighted_rms_norm_forward_npu._twinkle_npu_patched = True
        _deepseek_v4_unweighted_rms_norm_forward_npu._twinkle_old_forward = (
            modeling_deepseek_v4.DeepseekV4UnweightedRMSNorm.forward
        )
        modeling_deepseek_v4.DeepseekV4UnweightedRMSNorm.forward = _deepseek_v4_unweighted_rms_norm_forward_npu

    print('[PATCH] DeepseekV4RMSNorm/DeepseekV4UnweightedRMSNorm.forward -> torch_npu.npu_rms_norm')



def _npu_sparse_attn_shared_kv(query: torch.Tensor, ori_kv: torch.Tensor, cmp_kv: torch.Tensor,
                              cmp_indices: torch.Tensor, sinks: torch.Tensor, scale: float,
                              cmp_ratio: int, sliding_window: int) -> torch.Tensor:
    """Adapter from HF DeepSeek V4 attention tensors to MindSpeed SFA.

    HF uses [B, H, S, D] for q/kv. MindSpeed SparseAttnSharedKV expects BSND query
    and split original/compressed shared-KV tensors.
    """
    from mindspeed.ops.npu_sparse_attn_shared_kv import SparseAttnSharedKV

    query = query.transpose(1, 2).contiguous()  # [B, S, H, D]
    ori_kv = ori_kv.squeeze(1).contiguous()  # [B, S_ori, D]

    if cmp_kv is not None:
        cmp_kv = cmp_kv.squeeze(1).contiguous()  # [B, S_cmp, D]

    if cmp_indices is not None:
        cmp_indices = cmp_indices.to(torch.int32).contiguous().unsqueeze(2)  # [B, S, 1, K]

    batch_size, q_len, num_heads, head_dim = query.shape
    kv_len = ori_kv.shape[1]
    topk = 0 if cmp_indices is None or cmp_ratio != 4 else cmp_indices.shape[-1]

    output = SparseAttnSharedKV.apply(
        query,
        ori_kv.unsqueeze(2).contiguous(),
        None if cmp_kv is None else cmp_kv.unsqueeze(2).contiguous(),
        None,  # cu_seq_lens_q; TND is not supported by this adapter.
        None,  # cu_seq_lens_ori_kv
        None,  # cu_seq_lens_cmp_kv
        None,  # ori_sparse_indices; original KV uses band mask mode.
        cmp_indices if cmp_ratio == 4 else None,
        sinks.float(),
        scale,
        cmp_ratio,
        4,  # ori_mask_mode: band/sliding original KV
        3,  # cmp_mask_mode: sparse compressed KV
        sliding_window - 1,
        0,
        num_heads,
        1,  # DeepSeek V4 has shared KV / single KV head.
        head_dim,
        batch_size,
        q_len,
        kv_len,
        topk,
        'BSND',
        'BSND',
    )
    return output.transpose(1, 2).contiguous()  # [B, H, S, D]


def _align_npu_tensor_format(tensor: torch.Tensor, ref_tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != 'npu' or ref_tensor.device.type != 'npu':
        return tensor
    try:
        ref_format = torch_npu.get_npu_format(ref_tensor)
        tensor = torch_npu.npu_format_cast(tensor, ref_format)
    except Exception:
        pass
    return tensor.contiguous()


def apply_deepseek_v4_indexer_capture_patch():
    try:
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
            DeepseekV4Indexer,
            apply_rotary_pos_emb,
        )
    except Exception as exc:
        print(f'[PATCH] skip DeepseekV4Indexer capture patch: {exc}')
        return

    if getattr(DeepseekV4Indexer.forward, '_twinkle_npu_patched', False):
        return

    old_forward = DeepseekV4Indexer.forward

    @functools.wraps(old_forward)
    def new_forward(self, hidden_states, q_residual, position_ids, past_key_values, layer_idx):
        if (
            hidden_states.device.type != 'npu'
            or past_key_values is not None
            or getattr(self, '_twinkle_npu_li_disabled', False)
        ):
            indices = old_forward(self, hidden_states, q_residual, position_ids, past_key_values, layer_idx)
            self._twinkle_last_top_k_indices = indices
            return indices

        try:
            import mindspeed.ops.npu_lightning_indexer as mindspeed_li

            batch, seq_len, _ = hidden_states.shape
            cache_layer = past_key_values.layers[layer_idx] if past_key_values is not None else None
            kv = self.kv_proj(hidden_states)
            gate = self.gate_proj(hidden_states)

            if cache_layer is None:
                usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
                chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
            else:
                chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights(
                    'indexer', kv, gate
                )

            if chunk_kv.shape[1] > 0:
                n_windows = chunk_kv.shape[1] // self.compress_rate
                ratio = self.compress_rate
                chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
                chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + self.position_bias.to(chunk_gate.dtype)

                new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))
                new_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, self.head_dim), float('-inf'))
                new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim:]
                new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim:]
                if n_windows > 1:
                    new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, :self.head_dim]
                    new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, :self.head_dim]
                if cache_layer is not None:
                    prior_kv, prior_gate = cache_layer.update_overlap_state(
                        'indexer', chunk_kv, chunk_gate, self.head_dim
                    )
                    if prior_kv is not None:
                        new_kv[:, 0, :ratio] = prior_kv.to(new_kv.dtype)
                        new_gate[:, 0, :ratio] = prior_gate.to(new_gate.dtype)

                compressed = self.kv_norm(
                    (new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2)
                )
                positions = torch.arange(n_windows, device=compressed.device)
                positions = positions * self.compress_rate + first_window_position
                positions = positions.unsqueeze(0).expand(batch, -1)
                cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
                compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
            else:
                compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

            compressed_kv = (
                compressed if cache_layer is None else cache_layer.update_compressor_states('indexer', compressed)
            )
            compressed_len = compressed_kv.shape[1]
            if compressed_len == 0:
                indices = old_forward(self, hidden_states, q_residual, position_ids, past_key_values, layer_idx)
                self._twinkle_last_top_k_indices = indices
                return indices

            cos_q, sin_q = self.rotary_emb(hidden_states, position_ids=position_ids, layer_type=self.rope_layer_type)
            q = self.q_b_proj(q_residual).view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
            q = apply_rotary_pos_emb(q, cos_q, sin_q).transpose(1, 2)
            weights = self.weights_proj(hidden_states).float() * self.weights_scaling

            top_k_indices, index_score = mindspeed_li.npu_lightning_indexer(
                q.to(torch.bfloat16),
                compressed_kv.to(torch.bfloat16).unsqueeze(2),
                weights.to(torch.bfloat16),
                sparse_count=min(self.index_topk, compressed_len),
                sparse_mode=3,
                cmp_ratio=self.compress_rate,
            )
            top_k_indices = top_k_indices.squeeze(2)
            index_score = index_score.squeeze(2)

            causal_threshold = (position_ids + 1) // self.compress_rate
            invalid = top_k_indices >= causal_threshold.unsqueeze(-1)
            top_k_indices = torch.where(invalid, torch.full_like(top_k_indices, -1), top_k_indices)

            if not getattr(self, '_twinkle_npu_li_logged', False):
                self._twinkle_npu_li_logged = True
                print(
                    '[PATCH] DeepSeek V4 NPU Lightning Indexer used: '
                    f'layer_idx={layer_idx}, '
                    f'q={tuple(q.shape)}, compressed_kv={tuple(compressed_kv.shape)}, '
                    f'top_k_indices={tuple(top_k_indices.shape)}',
                    flush=True,
                )

            self._twinkle_last_top_k_indices = top_k_indices
            self._twinkle_last_index_score = index_score
            return top_k_indices
        except Exception as exc:
            self._twinkle_npu_li_disabled = True
            indices = old_forward(self, hidden_states, q_residual, position_ids, past_key_values, layer_idx)
            self._twinkle_last_top_k_indices = indices
            print(
                '[PATCH] DeepSeek V4 NPU Lightning Indexer fallback to HF indexer: '
                f'layer_idx={layer_idx}, error={type(exc).__name__}: {exc}',
                flush=True,
            )
            return indices

    new_forward._twinkle_npu_patched = True
    new_forward._twinkle_old_forward = old_forward
    DeepseekV4Indexer.forward = new_forward
    print('[PATCH] DeepseekV4Indexer.forward -> NPU Lightning Indexer with HF fallback')


def apply_deepseek_v4_sfa_attention_patch():
    try:
        import torch.nn.functional as F
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
            DeepseekV4Attention,
            apply_rotary_pos_emb,
            eager_attention_forward,
        )
    except Exception as exc:
        print(f'[PATCH] skip DeepseekV4 SFA attention patch: {exc}')
        return

    if getattr(DeepseekV4Attention.forward, '_twinkle_npu_patched', False):
        return

    def _shape_dtype(x):
        if x is None:
            return None
        return {'shape': tuple(x.shape), 'dtype': str(x.dtype), 'device': str(x.device)}

    def _get_position_cos_sin(self, position_embeddings):
        if isinstance(position_embeddings, dict):
            rope_layer_type = getattr(self, 'rope_layer_type', 'main')
            return position_embeddings[rope_layer_type]
        return position_embeddings

    def _get_compress_ratio(self):
        if self.layer_type == 'compressed_sparse_attention':
            return self.config.compress_rates['compressed_sparse_attention']
        if self.layer_type == 'heavily_compressed_attention':
            return self.config.compress_rates['heavily_compressed_attention']
        return 0

    def new_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = _get_position_cos_sin(self, position_embeddings)

        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
        q = self.q_b_norm(q)
        q = apply_rotary_pos_emb(q, cos, sin)

        kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)
        kv = apply_rotary_pos_emb(kv, cos, sin)

        if past_key_values is not None:
            kv = past_key_values.update(kv, kv, self.layer_idx)[0]

        ori_kv = kv
        compressed_kv = None
        block_bias = None
        top_k_indices = None

        if self.compressor is not None:
            compressed_kv, block_bias = self.compressor(
                hidden_states, q_residual, position_ids, past_key_values, self.layer_idx
            )
            indexer = getattr(self.compressor, 'indexer', None)
            if indexer is not None:
                top_k_indices = getattr(indexer, '_twinkle_last_top_k_indices', None)

        use_npu_sfa = (
            q.device.type == 'npu'
            and self.compressor is not None
            and self.layer_type == 'compressed_sparse_attention'
            and compressed_kv is not None
            and top_k_indices is not None
            and not getattr(self, '_twinkle_npu_sfa_disabled', False)
        )

        if use_npu_sfa:
            try:
                attn_output = _npu_sparse_attn_shared_kv(
                    query=q,
                    ori_kv=ori_kv,
                    cmp_kv=compressed_kv,
                    cmp_indices=top_k_indices,
                    sinks=self.sinks,
                    scale=self.scaling,
                    cmp_ratio=_get_compress_ratio(self),
                    sliding_window=self.sliding_window,
                )
                if not getattr(self, '_twinkle_npu_sfa_logged', False):
                    self._twinkle_npu_sfa_logged = True
                    print(
                        '[PATCH] DeepSeek V4 NPU SFA used: '
                        f'layer_idx={getattr(self, "layer_idx", None)}, '
                        f'q={tuple(q.shape)}, ori_kv={tuple(ori_kv.shape)}, '
                        f'compressed_kv={tuple(compressed_kv.shape)}, '
                        f'top_k_indices={tuple(top_k_indices.shape)}',
                        flush=True,
                    )
                attn_weights = None
            except Exception as exc:
                self._twinkle_npu_sfa_disabled = True
                print(
                    '[PATCH] DeepSeek V4 NPU SFA fallback to HF attention: '
                    f'layer_idx={getattr(self, "layer_idx", None)}, '
                    f'layer_type={getattr(self, "layer_type", None)}, '
                    f'cmp_ratio={_get_compress_ratio(self)}, '
                    f'sliding_window={getattr(self, "sliding_window", None)}, '
                    f'q={_shape_dtype(q)}, '
                    f'ori_kv={_shape_dtype(ori_kv)}, '
                    f'compressed_kv={_shape_dtype(compressed_kv)}, '
                    f'top_k_indices={_shape_dtype(top_k_indices)}, '
                    f'sinks={_shape_dtype(self.sinks)}, '
                    f'error={type(exc).__name__}: {exc}',
                    flush=True,
                )
                use_npu_sfa = False

        if not use_npu_sfa:
            if compressed_kv is not None:
                ori_kv = _align_npu_tensor_format(ori_kv, ori_kv)
                compressed_kv = _align_npu_tensor_format(compressed_kv, ori_kv)
                kv = torch.cat([ori_kv, compressed_kv], dim=2)
            else:
                kv = ori_kv

            if isinstance(attention_mask, torch.Tensor) and kv.shape[2] > attention_mask.shape[-1]:
                if block_bias is not None:
                    attention_mask = torch.cat([attention_mask, block_bias.to(attention_mask.dtype)], dim=-1)
                else:
                    attention_mask = F.pad(attention_mask, (0, kv.shape[2] - attention_mask.shape[-1]), value=0.0)

            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, eager_attention_forward
            )
            attn_output, attn_weights = attention_interface(
                self,
                q,
                kv,
                kv,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                s_aux=self.sinks,
                **kwargs,
            )

        attn_output = apply_rotary_pos_emb(attn_output.transpose(1, 2), cos, -sin).transpose(1, 2)
        grouped = attn_output.reshape(*input_shape, self.config.o_groups, -1)
        grouped = self.o_a_proj(grouped).flatten(2)
        output = self.o_b_proj(grouped)
        return output, attn_weights

    new_forward._twinkle_npu_patched = True
    new_forward._twinkle_old_forward = DeepseekV4Attention.forward
    DeepseekV4Attention.forward = new_forward
    print('[PATCH] DeepseekV4Attention.forward -> NPU SFA adapter with HF fallback')


def apply_deepseek_v4_sfa_patch():
    apply_deepseek_v4_indexer_capture_patch()
    apply_deepseek_v4_sfa_attention_patch()


def apply_npu_patch():
    import torch
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    apply_hf_moe_grouped_mm_patch()
    apply_deepseek_v4_rms_norm_patch()
    apply_deepseek_v4_sfa_patch()
