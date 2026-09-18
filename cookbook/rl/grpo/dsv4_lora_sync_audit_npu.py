# Copyright (c) ModelScope Contributors. All rights reserved.
"""Ascend HCCL/SHM + QuaRot LoRA audit with an independent offline oracle."""
from .dsv4_lora_npu import connect_existing_ray_if_configured
from .dsv4_lora_sync_audit import main as run_audit


def main():
    connect_existing_ray_if_configured()
    run_audit(backend='ascend')


if __name__ == '__main__':
    main()
