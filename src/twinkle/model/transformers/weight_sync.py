# Copyright (c) ModelScope Contributors. All rights reserved.
"""Source-native LoRA export; no inference naming or layout rules."""
import re

LAYER = re.compile(r'(?:^|\.)layers\.(\d+)\.')


def iter_lora_source_groups(model, adapter_name):
    """Gather one tenant's LoRA layer by layer without a full state dict."""
    multi_adapter = model.multi_adapter
    tenant = multi_adapter.find_lora_by_tenant(adapter_name)
    slot = tenant.adapter_name
    slot_marker = f'.{slot}.'
    pattern = re.compile(rf'\.lora_([AB])\.{re.escape(slot)}\.weight$')
    modules = getattr(multi_adapter, 'module', None)
    if modules is None:
        modules = []
    elif not isinstance(modules, list):
        modules = [modules]
    ordinary = {}
    for module in modules:
        for name, parameter in module.named_parameters():
            if multi_adapter._is_target_parameter_lora_name(name):
                continue
            if slot_marker not in name and not name.endswith(f'.{slot}'):
                continue
            if '.lora_' not in name:
                continue
            if not multi_adapter.match_target_modules(name, tenant.tenant_config.target_modules):
                continue
            match = pattern.search(name)
            if match is None:
                raise ValueError(f'Unsupported trainable LoRA tensor during synchronization: {name}')
            source_name = name.replace(slot_marker, '.')
            layer = LAYER.search(source_name)
            if layer is None:
                raise ValueError(f'Unsupported trainable LoRA tensor during synchronization: {name}')
            ordinary.setdefault(int(layer[1]), []).append((source_name, name, parameter))
    for layer in sorted(ordinary):
        group = {}
        for source_name, name, parameter in sorted(ordinary[layer], key=lambda item: item[0]):
            tensor = multi_adapter._slice_rank_tensor(
                name, multi_adapter._read_param_tensor(parameter), tenant.tenant_config.r)
            if tensor is None:
                raise ValueError(f'LoRA tensor is unavailable during synchronization: {name}')
            if source_name in group:
                raise ValueError(f'Duplicate trainable LoRA tensor during synchronization: {source_name}')
            group[source_name] = tensor
        yield model.strategy.gather_adapter_state_dict(model.model, group, slot)

    manager = multi_adapter.target_parameter_manager
    slot = manager.tenant_to_slot.get(adapter_name)
    if slot is None:
        return
    routed = {}
    for wrapper in manager.wrappers:
        layer = LAYER.search(wrapper.record.key)
        if layer is None:
            raise ValueError(f'Unsupported routed LoRA target: {wrapper.record.key}')
        routed.setdefault(int(layer[1]), []).append(wrapper)
    for layer in sorted(routed):
        wrappers = sorted(routed[layer], key=lambda wrapper: wrapper.record.key)
        source = {}
        for wrapper in wrappers:
            source.update(wrapper.get_state_dict(slot))
        # Keep the PEFT keys until EP gather has reconstructed each tensor.
        gathered = model.strategy.gather_adapter_state_dict(model.model, source, tenant.adapter_name)
        yield {
            f'{wrapper.record.key}.lora_{side}.weight': gathered[f'{wrapper.peft_key_prefix}.lora_{side}.weight']
            for wrapper in wrappers for side in ('A', 'B')
        }
