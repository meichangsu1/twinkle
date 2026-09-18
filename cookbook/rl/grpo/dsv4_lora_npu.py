# Copyright (c) ModelScope Contributors. All rights reserved.
"""Ascend DSV4 GRPO; use an audited W8A8_DYNAMIC QuaRot rollout base.

Run with python -m cookbook.rl.grpo.dsv4_lora_npu. See dsv4_lora_h800_read_me.md.
"""
import os

from twinkle.model import MultiLoraTransformersModel
from twinkle.sampler import vLLMSampler
from .dsv4_lora_h800 import build_workers as _build_workers
from .dsv4_lora_h800 import main as run_grpo


def build_workers(model_cls=MultiLoraTransformersModel, sampler_cls=vLLMSampler, extra_engine_args=None):
    return _build_workers(model_cls, sampler_cls, extra_engine_args, backend='ascend')


def connect_existing_ray_if_configured():
    """Connect the driver to an explicitly configured multi-node cluster.

    With no RAY_ADDRESS, Twinkle keeps its existing single-node auto-start
    behavior.  With RAY_ADDRESS, connect before ``twinkle.initialize()`` so
    RayHelper does not try to redeclare custom NPU resources on an existing
    cluster.
    """
    address = os.environ.get('RAY_ADDRESS', '').strip()
    if not address:
        return

    import ray
    if not ray.is_initialized():
        ray.init(address=address, ignore_reinit_error=True)


def main():
    connect_existing_ray_if_configured()
    run_grpo(worker_builder=build_workers)


if __name__ == '__main__':
    main()
