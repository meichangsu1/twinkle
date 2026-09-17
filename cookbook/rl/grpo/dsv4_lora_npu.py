# Copyright (c) ModelScope Contributors. All rights reserved.
"""Two-node Ascend DSV4 GRPO; use an audited W8A8_DYNAMIC QuaRot rollout base.

Run with python -m cookbook.rl.grpo.dsv4_lora_npu. See dsv4_lora_h800_read_me.md.
"""
from twinkle.model import MultiLoraTransformersModel
from twinkle.sampler import vLLMSampler
from .dsv4_lora_h800 import build_workers as _build_workers
from .dsv4_lora_h800 import main as run_grpo


def build_workers(model_cls=MultiLoraTransformersModel, sampler_cls=vLLMSampler, extra_engine_args=None):
    return _build_workers(model_cls, sampler_cls, extra_engine_args, backend='ascend')


def main():
    run_grpo(worker_builder=build_workers)


if __name__ == '__main__':
    main()
