import os
import sys
from typing import Optional

import torch
import torch.distributed as dist

from .sp_collectives import (gather_attention_mask_from_sequence_parallel_region,
                             gather_cache_position_from_sequence_parallel_region,
                             gather_from_sequence_parallel_region, scatter_to_sequence_parallel_region)


class Qwen35StrictFullSeqHelper:

    def __init__(self):
        self.enabled = os.environ.get('QWEN35_SP_LINEAR_STRICT', '0') == '1'
        self.trace_enabled = os.environ.get('QWEN35_SP_LINEAR_STRICT_TRACE', '0') == '1'
        self._trace_seq = 0

    @staticmethod
    def _rank() -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return int(os.environ.get('RANK', '-1'))

    @staticmethod
    def _sp_rank(sequence_parallel) -> int:
        sp_group = getattr(sequence_parallel, '_sp_group', None)
        if sp_group is not None and dist.is_available() and dist.is_initialized():
            return dist.get_rank(sp_group)
        return 0

    def _trace(self, sequence_parallel, module: torch.nn.Module, event: str, tensor: Optional[torch.Tensor] = None, **kwargs):
        if not self.trace_enabled:
            return

        layer_idx = getattr(module, 'layer_idx', None)
        rank = self._rank()
        sp_rank = self._sp_rank(sequence_parallel)
        payload = {
            'seq': self._trace_seq,
            'rank': rank,
            'sp_rank': sp_rank,
            'layer_idx': layer_idx,
            'event': event,
        }
        if tensor is not None:
            payload['shape'] = tuple(int(x) for x in tensor.shape)
            payload['dtype'] = str(tensor.dtype)
            payload['device'] = str(tensor.device)
        for key, value in kwargs.items():
            payload[key] = value
        self._trace_seq += 1
        text = ' '.join(f'{key}={value}' for key, value in payload.items())
        print(f'[qwen35_strict] {text}', file=sys.stderr, flush=True)

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
        self._trace(sequence_parallel, module, 'run:enter', tensor=hidden_states)

        self._trace(sequence_parallel, module, 'hidden_states:gather:before', tensor=hidden_states, dim=1)
        full_hidden_states = gather_from_sequence_parallel_region(
            hidden_states.contiguous(),
            sequence_parallel,
            1,
            output_grad_mode='split',
            trace_label=f'layer{getattr(module, "layer_idx", "na")}:hidden_states',
        )
        self._trace(sequence_parallel, module, 'hidden_states:gather:after', tensor=full_hidden_states, dim=1)
        local_seq_len = hidden_states.shape[1]
        cached_attention_mask = sequence_parallel.extra_kwargs.get('strict_full_attention_mask')
        if cached_attention_mask is not None:
            full_attention_mask = cached_attention_mask
            self._trace(sequence_parallel, module, 'attention_mask:cached', tensor=full_attention_mask)
        else:
            self._trace(sequence_parallel, module, 'attention_mask:gather:before', tensor=attention_mask)
            full_attention_mask = gather_attention_mask_from_sequence_parallel_region(
                attention_mask,
                sequence_parallel,
                local_seq_len,
            )
            self._trace(sequence_parallel, module, 'attention_mask:gather:after', tensor=full_attention_mask)

        cached_cache_position = sequence_parallel.extra_kwargs.get('strict_full_cache_position')
        if cached_cache_position is not None:
            full_cache_position = cached_cache_position
            self._trace(sequence_parallel, module, 'cache_position:cached', tensor=full_cache_position)
        else:
            self._trace(sequence_parallel, module, 'cache_position:gather:before', tensor=cache_position)
            full_cache_position = gather_cache_position_from_sequence_parallel_region(
                cache_position,
                sequence_parallel,
            )
            self._trace(sequence_parallel, module, 'cache_position:gather:after', tensor=full_cache_position)

        saved_rule = module.chunk_gated_delta_rule
        saved_causal_conv1d_fn = module.causal_conv1d_fn
        module.chunk_gated_delta_rule = origin_rule
        module.causal_conv1d_fn = origin_causal_conv1d_fn
        try:
            self._trace(sequence_parallel, module, 'origin_forward:before', tensor=full_hidden_states)
            full_output = origin_forward(
                full_hidden_states,
                cache_params=cache_params,
                cache_position=full_cache_position,
                attention_mask=full_attention_mask,
            )
            self._trace(sequence_parallel, module, 'origin_forward:after', tensor=full_output)
        finally:
            module.chunk_gated_delta_rule = saved_rule
            module.causal_conv1d_fn = saved_causal_conv1d_fn

        self._trace(sequence_parallel, module, 'output:split:before', tensor=full_output, dim=1)
        local_output = scatter_to_sequence_parallel_region(
            full_output.contiguous(),
            sequence_parallel,
            1,
            trace_label=f'layer{getattr(module, "layer_idx", "na")}:output',
        )
        self._trace(sequence_parallel, module, 'output:split:after', tensor=local_output, dim=1)
        return local_output
