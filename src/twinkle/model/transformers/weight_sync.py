# Copyright (c) ModelScope Contributors. All rights reserved.
"""Source-native target-parameter LoRA export; no inference naming rules."""


def iter_lora_source_groups(model, adapter_name):
    tenant = model.multi_adapter.find_lora_by_tenant(adapter_name)
    manager = model.multi_adapter.target_parameter_manager
    slot = manager.tenant_to_slot[adapter_name]
    for wrapper in manager.wrappers:
        # Gather with the original keys required by the existing strategy.
        group = model.strategy.gather_adapter_state_dict(
            model.model, wrapper.get_state_dict(slot), tenant.adapter_name)
        yield {
            f'{wrapper.record.key}.lora_{side}.weight': group[f'{wrapper.peft_key_prefix}.lora_{side}.weight']
            for side in ('A', 'B')
        }
