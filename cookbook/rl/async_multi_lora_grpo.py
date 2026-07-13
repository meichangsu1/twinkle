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
import time
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

import twinkle
from twinkle.data_format import SamplingParams
from twinkle.reward import DAPOMathAccuracyReward, GSM8KAccuracyReward
from twinkle import DeviceGroup, DeviceMesh, Platform, get_device_placement, get_logger
from twinkle_agentic.async_rl import AsyncMultiLoraGRPOPipeline
from twinkle_agentic.async_rl.grpo_pipeline import (build_prompt_dataset_from_config, lora_context_config_for_context,
                                                   lora_model_config)

logger = get_logger()


def optional_step_limit(value):
    if value is None:
        return None
    parsed = int(value)
    return None if parsed <= 0 else parsed


def load_config(path: str):
    return OmegaConf.load(path)


def config_to_dict(value) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    return dict(OmegaConf.to_container(value, resolve=True) or {})


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


def adapter_path_from_save_result(save_result) -> str | None:
    if isinstance(save_result, str):
        return save_result
    return getattr(save_result, 'twinkle_path', None)


def recorder_elapsed_s(recorder) -> float | None:
    start_time = getattr(recorder, 'start_time', None)
    if start_time is not None:
        return time.time() - float(start_time)
    for child in getattr(recorder, 'recorders', []) or []:
        elapsed_s = recorder_elapsed_s(child)
        if elapsed_s is not None:
            return elapsed_s
    return None


def eval_config(cfg) -> dict[str, Any]:
    return config_to_dict(cfg.pipeline.get('eval'))


def eval_enabled(cfg) -> bool:
    return bool(eval_config(cfg).get('enabled', True))


def eval_batch_size(cfg) -> int:
    batch_size = int(eval_config(cfg).get('batch_size', 16))
    if batch_size <= 0:
        raise ValueError(f'pipeline.eval.batch_size must be positive, got {batch_size}')
    return batch_size


def eval_sampling_params(cfg) -> SamplingParams:
    eval_cfg = eval_config(cfg)
    sampling_cfg = {
        'max_tokens': int(cfg.sampler.sampling_params.get('max_tokens', 1024)),
        'num_samples': 1,
        'temperature': 0.0,
        'top_p': 1.0,
    }
    sampling_cfg.update(config_to_dict(eval_cfg.get('sampling_params')))
    return SamplingParams.from_dict(sampling_cfg)


def build_eval_dataset(cfg, context_cfg):
    eval_dataset_cfg = context_cfg.get('eval_dataset')
    if eval_dataset_cfg is None:
        return None
    context_payload = config_to_dict(context_cfg)
    context_payload['dataset'] = config_to_dict(eval_dataset_cfg)
    template_payload = config_to_dict(lora_model_config(cfg, context_cfg).template)
    return build_prompt_dataset_from_config(context_payload, template_payload)


def build_eval_batches(cfg, context_cfg) -> tuple[str, list[list[Any]]]:
    eval_dataset_cfg = context_cfg.get('eval_dataset')
    if eval_dataset_cfg is None:
        return '', []
    dataset = build_eval_dataset(cfg, context_cfg)
    if dataset is None:
        return '', []
    dataset_len = len(dataset)
    data_num = int(eval_dataset_cfg.get('data_num', 0) or 0)
    max_prompts = min(data_num, dataset_len) if data_num > 0 else dataset_len
    batch_size = eval_batch_size(cfg)
    batches: list[list[Any]] = []
    batch: list[Any] = []
    for index in range(max_prompts):
        batch.append(dataset[index])
        if len(batch) == batch_size:
            batches.append(batch)
            batch = []
    if batch:
        batches.append(batch)
    eval_name = str(eval_dataset_cfg.get('name') or eval_dataset_cfg.get('dataset_id') or 'eval')
    logger.info(
        'Loaded async validation batches: adapter=%s eval_dataset=%s dataset_len=%s max_prompts=%s batches=%s',
        context_cfg.adapter_name,
        eval_name,
        dataset_len,
        max_prompts,
        len(batches),
    )
    return eval_name, batches


def accuracy_reward_for_context(context):
    if context.reward_type == 'gsm8k':
        return GSM8KAccuracyReward()
    if context.reward_type == 'dapo_math':
        return DAPOMathAccuracyReward()
    raise ValueError(f'unsupported reward_type for async validation: {context.reward_type}')


def run_final_validation(pipeline: AsyncMultiLoraGRPOPipeline, cfg, adapter_paths: dict[str, str]) -> None:
    if not eval_enabled(cfg):
        logger.info('Async validation disabled by pipeline.eval.enabled=false')
        return
    rollout = pipeline.rollout
    sampler = getattr(rollout, 'sampler', None)
    if sampler is None:
        raise RuntimeError('Async validation requires the pipeline rollout to expose a sampler')
    sampling_params = eval_sampling_params(cfg)

    for context in pipeline.current_contexts():
        context_cfg = lora_context_config_for_context(cfg, context)
        eval_dataset_name, eval_batches = build_eval_batches(cfg, context_cfg)
        if not eval_batches:
            logger.info('Validation skipped for adapter=%s: no eval_dataset configured', context.adapter_name)
            continue
        adapter_path = adapter_paths.get(context.adapter_name)
        if adapter_path is None:
            raise RuntimeError(f'No final adapter path available for validation: {context.adapter_name}')

        runtime_state = pipeline.lora_runtime_registry.get(context)
        policy_version = runtime_state.policy_version
        partition_id = f'{context.key}/final_eval'
        eval_start = time.time()
        pipeline.metrics_recorder.log_event(
            event='eval_started',
            phase='eval',
            context=context,
            partition_id=partition_id,
            policy_version=policy_version,
            metrics={
                'dataset': str(context_cfg.dataset.get('name') or context_cfg.dataset.get('dataset_id')),
                'eval_dataset': eval_dataset_name,
                'adapter_path': adapter_path,
                'eval_batch_count': len(eval_batches),
            },
        )

        reset_prefix_cache = getattr(sampler, 'reset_prefix_cache', None)
        if reset_prefix_cache is not None:
            reset_prefix_cache()

        prompt_count = 0
        sample_count = 0
        completion_lengths: list[int] = []
        accuracy_rewards: list[float] = []
        accuracy_reward = accuracy_reward_for_context(context)
        for batch in eval_batches:
            prompt_count += len(batch)
            responses = sampler.sample(
                batch,
                sampling_params,
                adapter_name=context.adapter_name,
                adapter_path=adapter_path,
            )
            input_data = []
            for response in responses:
                for sequence in response.sequences:
                    input_data.append(sequence.new_input_feature)
                    completion_lengths.append(len(sequence.tokens))
            sample_count += len(input_data)
            accuracy_rewards.extend(float(value) for value in accuracy_reward(input_data))

        metrics = {
            'eval/accuracy': sum(accuracy_rewards) / len(accuracy_rewards) if accuracy_rewards else 0.0,
            'eval/sample_count': sample_count,
            'eval/prompt_count': prompt_count,
            'eval/completion_length': (
                sum(completion_lengths) / len(completion_lengths) if completion_lengths else 0.0
            ),
            'dataset': str(context_cfg.dataset.get('name') or context_cfg.dataset.get('dataset_id')),
            'eval_dataset': eval_dataset_name,
            'optimizer_step': pipeline._optimizer_steps_by_context.get(context.key, 0),
            'step': pipeline._optimizer_steps_by_context.get(context.key, 0),
            'policy_version': policy_version,
            'eval_num_samples': sampling_params.num_samples,
            'eval_latency_s': time.time() - eval_start,
        }
        logger.info('[%s Final eval step %s version %s] %s',
                    context.adapter_name, metrics['optimizer_step'], policy_version, metrics)
        pipeline.metrics_recorder.log_event(
            event='eval_done',
            phase='eval',
            context=context,
            partition_id=partition_id,
            policy_version=policy_version,
            metrics=metrics,
        )


def log_training_completed(
    pipeline: AsyncMultiLoraGRPOPipeline,
    *,
    trained_partitions: int,
    final_adapter_paths: dict[str, str],
) -> float | None:
    train_wall_time_s = recorder_elapsed_s(pipeline.metrics_recorder)
    pipeline.metrics_recorder.log_event(
        event='training_completed',
        phase='run',
        metrics={
            'wall_time_s': train_wall_time_s,
            'train_wall_time_s': train_wall_time_s,
            'trained_partitions': trained_partitions,
            'final_adapter_paths': final_adapter_paths,
            'per_context': {
                context.key: {
                    'adapter_name': context.adapter_name,
                    'optimizer_steps': pipeline._optimizer_steps_by_context.get(context.key, 0),
                    'policy_version': pipeline.lora_runtime_registry.get(context).policy_version,
                }
                for context in pipeline.current_contexts()
            },
        },
    )
    return train_wall_time_s


def log_run_completed(
    pipeline: AsyncMultiLoraGRPOPipeline,
    *,
    trained_partitions: int,
    train_wall_time_s: float | None,
) -> None:
    total_wall_time_s = recorder_elapsed_s(pipeline.metrics_recorder)
    eval_wall_time_s = (
        max(0.0, total_wall_time_s - train_wall_time_s)
        if total_wall_time_s is not None and train_wall_time_s is not None
        else None
    )
    pipeline.metrics_recorder.log_event(
        event='run_completed',
        phase='run',
        metrics={
            'wall_time_s': train_wall_time_s,
            'train_wall_time_s': train_wall_time_s,
            'total_wall_time_s': total_wall_time_s,
            'eval_wall_time_s': eval_wall_time_s,
            'trained_partitions': trained_partitions,
            'per_context': {
                context.key: {
                    'adapter_name': context.adapter_name,
                    'optimizer_steps': pipeline._optimizer_steps_by_context.get(context.key, 0),
                    'policy_version': pipeline.lora_runtime_registry.get(context).policy_version,
                }
                for context in pipeline.current_contexts()
            },
        },
    )


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
    trained = 0
    try:
        logger.info('Starting server-side async multi-LoRA GRPO')
        logger.info(get_device_placement())
        history = pipeline.run_until_idle(max_steps=optional_step_limit(cfg.pipeline.get('max_steps')))
        trained = sum(
            getattr(item, 'trained_partitions', 1 if item.get('train') is not None else 0)
            for item in history
        )
        logger.info('async_multi_lora_grpo progress: trained_partitions=%s', trained)

        final_adapter_paths: dict[str, str] = {}
        for context in pipeline.current_contexts():
            final_name = f'async-grpo-final-{context.training_run_id}-{context.adapter_name}'
            save_result = pipeline.model.save(
                final_name,
                output_dir=cfg.model.adapter_checkpoint_dir,
                adapter_name=context.adapter_name,
                save_optimizer=bool(cfg.pipeline.save_optimizer),
            )
            adapter_path = adapter_path_from_save_result(save_result)
            if adapter_path is None:
                raise RuntimeError(f'Final adapter save did not return an adapter path: {context.adapter_name}')
            final_adapter_paths[context.adapter_name] = adapter_path

        train_wall_time_s = log_training_completed(
            pipeline,
            trained_partitions=trained,
            final_adapter_paths=final_adapter_paths,
        )
        run_final_validation(pipeline, cfg, final_adapter_paths)
        log_run_completed(
            pipeline,
            trained_partitions=trained,
            train_wall_time_s=train_wall_time_s,
        )
    finally:
        pipeline.shutdown()
    logger.info('Training completed. trained_partitions=%s', trained)


if __name__ == '__main__':
    main()
