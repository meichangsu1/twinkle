from .qwen35 import Qwen35LinearAttentionSPModelPatch


LINEAR_ATTENTION_MODEL_PATCHES = (Qwen35LinearAttentionSPModelPatch(), )

__all__ = ['LINEAR_ATTENTION_MODEL_PATCHES', 'Qwen35LinearAttentionSPModelPatch']
