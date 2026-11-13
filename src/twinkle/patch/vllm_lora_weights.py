# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
import os
import re
import torch
from dataclasses import field
from typing import Dict, Optional

from twinkle import requires
from .base import Patch

try:
    from vllm.lora.request import LoRARequest
except (ModuleNotFoundError, ImportError):
    LoRARequest = object


def is_packed_moe_lora(names) -> bool:
    """Recognize PEFT expert parameters rather than per-expert linear adapters."""
    return any(re.search(r'(?:^|\.)experts(?:\.base_layer)*\.lora_[AB]\.weight$', name) for name in names)


def is_packed_moe_lora_checkpoint(path: str) -> bool:
    from safetensors import safe_open

    tensor_path = os.path.join(path, 'adapter_model.safetensors')
    if os.path.isfile(tensor_path):
        with safe_open(tensor_path, framework='pt', device='cpu') as tensors:
            return is_packed_moe_lora(tensors.keys())
    for filename in ('adapter_model.bin', 'adapter_model.pt'):
        tensor_path = os.path.join(path, filename)
        if os.path.isfile(tensor_path):
            return is_packed_moe_lora(torch.load(tensor_path, map_location='cpu', weights_only=True))
    raise FileNotFoundError(f'No adapter weights found in {path}')


class TensorLoRARequest(LoRARequest):
    peft_config: dict = field(default=None)
    lora_tensors: dict = field(default=None)
    lora_embeddings: Optional[Dict[str, torch.Tensor]] = None

    @property
    def config(self):
        return self.peft_config

    @property
    def embeddings(self):
        return self.lora_embeddings


class VLLMLoraWeights(Patch):

    def __call__(self, sampler, **kwargs):
        # Mixed-format vLLM selects the layout per adapter; retain the old path on older builds.
        if not hasattr(LoRARequest, 'is_3d_lora_weight'):
            from twinkle.patch.vllm_moe_loader import patch_qwen35_moe_is_3d_moe_weight_false

            patch_qwen35_moe_is_3d_moe_weight_false()

        _sampler_ref = sampler

        def _get_tokenizer():
            """Get tokenizer lazily from sampler's template."""
            if _sampler_ref and _sampler_ref.template is not None:
                return _sampler_ref.template.tokenizer
            return None

        from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
        try:
            from vllm.lora.models import LoRAModel
        except ImportError:
            # vllm >= 0.13 https://github.com/vllm-project/vllm/pull/30253
            from vllm.lora.lora_model import LoRAModel

        try:
            from vllm.transformers_utils.tokenizer_group import TokenizerGroup
        except ImportError:
            # removed in https://github.com/vllm-project/vllm/pull/24078
            TokenizerGroup = None

        def patched_load_adapter(self: LRUCacheWorkerLoRAManager, lora_request: TensorLoRARequest) -> LoRAModel:
            # Native file loading preserves mixed-format validation and EP handling.
            if not isinstance(lora_request, TensorLoRARequest):
                return self._old_load_adapter(lora_request)

            try:
                from vllm.lora.peft_helper import PEFTHelper

                peft_helper = PEFTHelper.from_dict(lora_request.peft_config)
                # Validates the LoRA configuration against requirements before
                # loading weights, throwing an exception if validation fails.
                peft_helper.validate_legal(self.lora_config)
                # For some models like Qwen2VL, we need to use hf_to_vllm_mapper
                # to ensure correct loading of lora weights. Keep only renames so
                # constituent names (e.g. `q_proj`) survive for the LoRA manager
                # to pack; base-weight stacking maps would overwrite adapters.
                # (align with vllm.lora.worker_manager.WorkerLoRAManager._load_adapter)
                model = self._adapter_manager.model
                hf_to_vllm_mapper = getattr(model, 'hf_to_vllm_mapper', None)
                if hf_to_vllm_mapper is not None:
                    if hasattr(hf_to_vllm_mapper, 'get_rename_mapper'):
                        hf_to_vllm_mapper = hf_to_vllm_mapper.get_rename_mapper()
                    elif hasattr(hf_to_vllm_mapper, 'get_unstacked_mapper'):
                        # Compatibility with vLLM versions using the previous API.
                        hf_to_vllm_mapper = hf_to_vllm_mapper.get_unstacked_mapper()

                lora_request_kwargs = {
                    'peft_helper': peft_helper,
                    'lora_model_id': lora_request.lora_int_id,
                    'device': 'cpu',
                    'dtype': self.lora_config.lora_dtype,
                    'weights_mapper': hf_to_vllm_mapper,
                }
                if hasattr(self, 'embedding_padding_modules'):
                    lora_request_kwargs['embedding_modules'] = self.embedding_modules
                    lora_request_kwargs['embedding_padding_modules'] = self.embedding_padding_modules
                else:
                    lora_request_kwargs['model_vocab_size'] = self.vocab_size
                if hasattr(self.lora_config, 'lora_extra_vocab_size'):
                    # lora_extra_vocab_size is removed in vllm >= 0.12
                    # https://github.com/vllm-project/vllm/issues/23474
                    lora_request_kwargs['target_embedding_padding'] = (
                        self.vocab_size + self.lora_config.lora_extra_vocab_size)

                if 'skip_prefixes' in inspect.signature(self._lora_model_cls.from_lora_tensors).parameters:
                    lora_request_kwargs['skip_prefixes'] = getattr(model, 'lora_skip_prefixes', None)
                lora = self._lora_model_cls.from_lora_tensors(
                    tensors=lora_request.lora_tensors,
                    **lora_request_kwargs,
                )
                # The adapter manager uses this flag to convert packed MoE tensors before activation.
                if hasattr(lora_request, 'is_3d_lora_weight'):
                    lora.is_3d_lora_weight = lora_request.is_3d_lora_weight
                for name in ('_validate_modules_to_save', '_validate_token_classification_lora'):
                    validate = getattr(self._adapter_manager, name, None)
                    if validate is not None:
                        validate(lora)
            except Exception as e:
                raise e

            if hasattr(self.lora_config, 'lora_extra_vocab_size'):
                if lora.extra_vocab_size > self.lora_config.lora_extra_vocab_size:
                    raise ValueError(f'LoRA added vocab size {lora.extra_vocab_size} is greater than '
                                     f'lora_extra_vocab_size {self.lora_config.lora_extra_vocab_size}.')
            return lora

        # Cache the cache-wrapped template tokenizer (keyed by id) so we wrap once, not per request.
        _wrapped_tok_cache: Dict[int, object] = {}

        def _ensure_max_token_id(tokenizer):
            """
            vllm's Processor._validate_model_input reads ``tokenizer.max_token_id``, an attribute
            that only exists on vllm's ``CachedTokenizer`` wrapper. The sampler template tokenizer is
            a RAW HF tokenizer (never passed through vllm's ``get_cached_tokenizer``), so validation
            raises ``AttributeError: ... has no attribute max_token_id``. Wrap it once to add the attr.
            """
            if tokenizer is None or hasattr(tokenizer, 'max_token_id'):
                return tokenizer
            key = id(tokenizer)
            wrapped = _wrapped_tok_cache.get(key)
            if wrapped is None:
                try:
                    from vllm.transformers_utils.tokenizer import get_cached_tokenizer
                    wrapped = get_cached_tokenizer(tokenizer)
                except Exception:
                    wrapped = tokenizer
                _wrapped_tok_cache[key] = wrapped
            return wrapped

        def patched_get_lora_tokenizer(self: TokenizerGroup, lora_request: LoRARequest):
            # since we pass dummy path, skip get tokenizer from path
            # Use lazy tokenizer access
            tokenizer = _get_tokenizer()
            if tokenizer is None:
                # Fallback to the original method if tokenizer not available
                return self._old_get_lora_tokenizer(lora_request)
            return _ensure_max_token_id(tokenizer)

        if not hasattr(LRUCacheWorkerLoRAManager, '_old_load_adapter'):
            _old_load_adapter = LRUCacheWorkerLoRAManager._load_adapter
            LRUCacheWorkerLoRAManager._load_adapter = patched_load_adapter
            LRUCacheWorkerLoRAManager._old_load_adapter = _old_load_adapter
            if TokenizerGroup is not None:
                TokenizerGroup._old_get_lora_tokenizer = TokenizerGroup.get_lora_tokenizer
                TokenizerGroup.get_lora_tokenizer = patched_get_lora_tokenizer
