# Copyright (c) ModelScope Contributors. All rights reserved.
"""Replaceable rollout-side weight adaptation; transport and installation stay unchanged."""
import json
from abc import ABC, abstractmethod
from importlib import import_module


class RolloutWeightAdapter(ABC):
    """A load-time hook. Never sends weights or installs/removes adapters."""

    @abstractmethod
    def initialize(self, target_context, options):
        """Cache target constants. No adapter tensors or base reload."""

    @abstractmethod
    def process(self, weights, peft_config):
        """Return (weights, peft_config) for the existing loader.

        Input tensors already own their storage. Context/constants are supplied
        once by initialize(); a processor must release per-update staging on error.
        """


def normalize_weight_adapter(config):
    if config is None:
        return None
    if not isinstance(config, dict) or set(config) - {'class_path', 'options'}:
        raise ValueError('weight_adapter requires class_path and optional options')
    path = config.get('class_path')
    if not isinstance(path, str) or '.' not in path or not isinstance(config.get('options', {}), dict):
        raise ValueError('Invalid weight_adapter class_path/options')
    return json.loads(json.dumps(dict(class_path=path, options=config.get('options', {})), allow_nan=False))


def create_weight_adapter(config, target_context):
    config = normalize_weight_adapter(config)
    if config is None:
        raise ValueError('Raw LoRA synchronization requires an explicit rollout weight_adapter')
    module, name = config['class_path'].rsplit('.', 1)
    cls = getattr(import_module(module), name)
    if not isinstance(cls, type) or not issubclass(cls, RolloutWeightAdapter):
        raise TypeError('weight_adapter must implement RolloutWeightAdapter')
    adapter = cls()
    adapter.initialize(target_context, config['options'])
    return adapter
