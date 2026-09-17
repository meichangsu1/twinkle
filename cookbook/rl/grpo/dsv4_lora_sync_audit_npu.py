# Copyright (c) ModelScope Contributors. All rights reserved.
"""Ascend HCCL/SHM + QuaRot LoRA audit with an independent offline oracle."""
from .dsv4_lora_sync_audit import main as run_audit


def main():
    run_audit(backend='ascend')


if __name__ == '__main__':
    main()
