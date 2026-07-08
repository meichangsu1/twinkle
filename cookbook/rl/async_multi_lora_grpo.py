# Copyright (c) ModelScope Contributors. All rights reserved.
"""Server-side async GRPO cookbook with TransferQueue + Multi-LoRA.

This entrypoint is the in-process/server MVP:

  dataset prompts -> AsyncRollouter(inline reward) -> TransferQueue
  -> AdvantageWorker -> TrainerWorker -> LoRA save -> vLLM adapter_path

It does not use the Twinkle client/server submission path. The script owns
resource initialization, model construction, sampler construction, and the
pipeline run loop.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

import twinkle
from twinkle import DeviceGroup, DeviceMesh, Platform, get_device_placement, get_logger
from twinkle_agentic.async_rl import AsyncMultiLoraGRPOPipeline

logger = get_logger()


def load_config(path: str):
    return OmegaConf.load(path)


def build_device_meshes(cfg):
    runtime = cfg.runtime
    model_gpus = int(runtime.model_gpus)
    sampler_gpus = int(runtime.sampler_gpus)
    sampler_tp = int(runtime.sampler_tp)
    total_gpus = model_gpus + sampler_gpus
    device_type = Platform.device_prefix()
    if runtime.mode == 'local':
        device_groups = [
            DeviceGroup(name='default', ranks=list(range(total_gpus)), device_type=device_type),
        ]
    else:
        device_groups = [
            DeviceGroup(name='model', ranks=list(range(model_gpus)), device_type=device_type),
            DeviceGroup(
                name='sampler',
                ranks=list(range(model_gpus, total_gpus)),
                device_type=device_type,
                gpus_per_worker=sampler_tp,
            ),
        ]
    model_mesh_cfg = cfg.model.mesh
    model_tp = int(model_mesh_cfg.get('tp_size', 1))
    model_ep = int(model_mesh_cfg.get('ep_size', 1))
    model_pp = int(model_mesh_cfg.get('pp_size', 1))
    model_parallel_size = model_tp * model_ep * model_pp
    if model_gpus % model_parallel_size != 0:
        raise ValueError(
            f'model_gpus={model_gpus} must be divisible by '
            f'tp_size*ep_size*pp_size={model_parallel_size}')
    model_dp = int(model_mesh_cfg.get('dp_size', model_gpus // model_parallel_size))
    model_mesh = DeviceMesh.from_sizes(
        world_size=model_gpus,
        dp_size=model_dp,
        tp_size=model_tp,
        ep_size=model_ep,
        pp_size=model_pp,
        sequence_parallel=bool(model_mesh_cfg.get('sequence_parallel', False)),
    )
    sampler_mesh = DeviceMesh.from_sizes(
        world_size=sampler_gpus,
        dp_size=max(1, sampler_gpus // sampler_tp),
        tp_size=sampler_tp,
    )
    return total_gpus, device_groups, model_mesh, sampler_mesh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default=Path(__file__).with_suffix('.yaml').as_posix(),
        help='Path to server-side async multi-LoRA GRPO YAML config.',
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    total_gpus, device_groups, model_mesh, sampler_mesh = build_device_meshes(cfg)
    twinkle.initialize(
        mode=cfg.runtime.mode,
        nproc_per_node=total_gpus,
        groups=device_groups,
        lazy_collect=bool(cfg.runtime.get('lazy_collect', False)),
    )

    pipeline = AsyncMultiLoraGRPOPipeline(cfg, model_mesh=model_mesh, sampler_mesh=sampler_mesh)

    logger.info('Starting server-side async multi-LoRA GRPO')
    logger.info(get_device_placement())
    concurrent = str(cfg.pipeline.get('scheduler', 'serial')) == 'concurrent'
    history = pipeline.run_until_idle(max_steps=int(cfg.pipeline.max_steps), concurrent=concurrent)
    trained = sum(1 for item in history if item.get('train') is not None)
    logger.info('async_multi_lora_grpo progress: trained_partitions=%s', trained)

    pipeline.shutdown()
    for context in pipeline.current_contexts():
        final_name = f'async-grpo-final-{context.training_run_id}-{context.adapter_name}'
        pipeline.model.save(
            final_name,
            output_dir=cfg.model.adapter_checkpoint_dir,
            adapter_name=context.adapter_name,
            save_optimizer=bool(cfg.pipeline.save_optimizer),
        )
    logger.info('Training completed. trained_partitions=%s', trained)


if __name__ == '__main__':
    main()
