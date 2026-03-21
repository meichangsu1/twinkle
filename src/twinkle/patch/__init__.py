# Copyright (c) ModelScope Contributors. All rights reserved.
import sys
from typing import Any, Type, Union

from .base import Patch


def apply_patch(module: Any, patch_cls: Union[Patch, Type[Patch], str], *args, **kwargs):
    from ..utils import construct_class
    patch_ins = construct_class(patch_cls, Patch, sys.modules[__name__])
    return patch_ins(module, *args, **kwargs)


def __getattr__(name: str):
    if name == 'Qwen35LinearAttentionSPPatch':
        from .qwen35_linear_attention_sp import Qwen35LinearAttentionSPPatch

        return Qwen35LinearAttentionSPPatch
    raise AttributeError(name)


__all__ = ['apply_patch', 'Patch', 'Qwen35LinearAttentionSPPatch']
