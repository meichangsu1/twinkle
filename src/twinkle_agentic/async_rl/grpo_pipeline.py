# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import re
import os
from functools import partial
from typing import Any, Dict, List

from .data_plane import TransferQueueDataPlane, TransferQueueRuntimeConfig
from .pipeline import BaseRLPipeline, BaseRLPipelineConfig
from .prompt_feeder import PromptFeeder
from .types import TrainingContext
from .workers import TrainerStepResult


def training_context_configs(cfg) -> list[Any]:
    if cfg.get('training_contexts'):
        return list(cfg.training_contexts)
    return [cfg.training_context]


def primary_training_context(cfg) -> Any:
    return training_context_configs(cfg)[0]


def build_training_contexts(cfg) -> list[TrainingContext]:
    contexts = []
    for context_cfg in training_context_configs(cfg):
        contexts.append(
            TrainingContext(
                tenant_id=context_cfg.tenant_id,
                training_run_id=context_cfg.training_run_id,
                base_model_id=context_cfg.base_model_id,
                adapter_name=context_cfg.adapter_name,
                reward_type=context_cfg.reward_type,
                tool_profile=context_cfg.get('tool_profile', 'default'),
                algorithm=context_cfg.get('algorithm', cfg.pipeline.get('algorithm', 'grpo')),
            ))
    base_models = {context.base_model_id for context in contexts}
    if len(base_models) != 1:
        raise ValueError(f'one async multi-LoRA job must use one base model, got {sorted(base_models)}')
    return contexts


def context_dataset_config(cfg, context_cfg):
    dataset_cfg = context_cfg.get('dataset')
    if dataset_cfg is not None:
        return dataset_cfg
    dataset_cfg = cfg.get('dataset')
    if dataset_cfg is not None:
        return dataset_cfg
    raise ValueError('dataset config is required for each training context when top-level dataset is not set: '
                     f'training_run_id={context_cfg.training_run_id}, adapter_name={context_cfg.adapter_name}')


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return cfg.get(key, default)


def build_prompt_dataset_from_config(context_cfg: dict[str, Any], template_cfg: dict[str, Any]):
    """Build a prompt dataset inside the DataLoader worker.

    This function must stay top-level and only receive serializable config
    values, because Ray serializes the callable passed to twinkle.dataloader.DataLoader.
    """
    import twinkle.preprocessor
    from twinkle.dataset import Dataset, DatasetMeta
    from twinkle.preprocessor import Preprocessor
    from twinkle.utils import construct_class

    dataset_cfg = _cfg_get(context_cfg, 'dataset')
    if dataset_cfg is None:
        raise ValueError(f'training context {context_cfg.get("training_run_id")} has no dataset config')
    data_num = _cfg_get(dataset_cfg, 'data_num')
    data_slice = range(int(data_num)) if data_num else None
    dataset = Dataset()
    dataset.add_dataset(
        DatasetMeta(
            _cfg_get(dataset_cfg, 'dataset_id'),
            subset_name=_cfg_get(dataset_cfg, 'subset_name'),
            split=_cfg_get(dataset_cfg, 'split', 'train'),
            data_slice=data_slice,
        ))
    dataset.set_template(
        _cfg_get(template_cfg, 'cls'),
        model_id=_cfg_get(context_cfg, 'base_model_id'),
        max_length=_cfg_get(dataset_cfg, 'max_length', _cfg_get(template_cfg, 'max_length', 4096)),
        truncation_strategy=_cfg_get(template_cfg, 'truncation_strategy', 'delete'),
        enable_thinking=_cfg_get(template_cfg, 'enable_thinking', False),
    )
    processor_cfg = _cfg_get(dataset_cfg, 'processor')
    if processor_cfg is not None:
        processor_cls = _cfg_get(processor_cfg, 'cls')
        processor_kwargs = {k: v for k, v in processor_cfg.items() if k != 'cls'}
        dataset.map(construct_class(processor_cls, Preprocessor, twinkle.preprocessor, **processor_kwargs))
    dataset.encode(add_generation_prompt=bool(_cfg_get(dataset_cfg, 'add_generation_prompt', True)))
    return dataset


def build_base_pipeline_config(cfg) -> BaseRLPipelineConfig:
    contexts = build_training_contexts(cfg)
    primary_context = contexts[0]
    return BaseRLPipelineConfig(
        training_contexts=contexts,
        tenant_id=primary_context.tenant_id,
        training_run_id=primary_context.training_run_id,
        base_model_id=primary_context.base_model_id,
        adapter_name=primary_context.adapter_name,
        reward_type=primary_context.reward_type,
        algorithm=primary_context.algorithm,
        tool_profile=primary_context.tool_profile,
        max_staleness=int(cfg.pipeline.max_staleness),
        target_groups_per_partition=int(cfg.pipeline.target_groups_per_partition),
        max_concurrent_groups=int(cfg.pipeline.max_concurrent_groups),
        max_submit_groups=cfg.pipeline.get('max_submit_groups'),
        reward_batch_size=int(cfg.pipeline.reward_batch_size),
        advantage_batch_size=int(cfg.pipeline.advantage_batch_size),
        max_train_partitions=int(cfg.pipeline.max_steps),
        save_name_prefix=cfg.pipeline.save_name_prefix,
        adapter_checkpoint_dir=cfg.model.adapter_checkpoint_dir,
        is_sampler_checkpoint=bool(cfg.pipeline.is_sampler_checkpoint),
        save_optimizer=bool(cfg.pipeline.save_optimizer),
        max_grad_norm=float(cfg.pipeline.max_grad_norm),
        norm_type=int(cfg.pipeline.norm_type),
    )


_MODEL_INPUT_FIELDS = {
    'messages',
    'input_ids',
    'labels',
    'attention_mask',
    'position_ids',
    'cu_seqlens',
    'pixel_values',
    'image_grid_thw',
    'video_pixel_values',
    'video_grid_thw',
    'input_features',
    'feature_attention_mask',
}


def model_input_from_training_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Return only model-consumable fields from a TQ training row.

    TQ rows contain both model input fields and training/runtime metadata
    such as old_logps, advantages, rewards, policy_version and group_id.
    Those fields are consumed by the loss or scheduler, not by InputProcessor.
    """
    trajectory = sample.get('trajectory')
    source = trajectory if isinstance(trajectory, dict) else sample
    model_input = {key: value for key, value in source.items() if key in _MODEL_INPUT_FIELDS}
    if not model_input:
        raise ValueError(f'training sample has no model input fields: keys={sorted(source.keys())}')
    return model_input


def validate_training_input_length(
    model_input: dict[str, Any],
    *,
    context: TrainingContext,
    partition_id: str,
    max_length: int,
) -> None:
    input_ids = model_input.get('input_ids')
    if input_ids is None:
        return
    length = len(input_ids)
    if length <= max_length:
        return
    raise ValueError(
        'training sample exceeds model.template.max_length: '
        f'length={length}, max_length={max_length}, '
        f'context={context.key}, partition_id={partition_id}. '
        'Reduce sampler.sampling_params.max_tokens and dataset.max_length, '
        'then clear old overlength TransferQueue partitions before rerun.')


class GSM8KBrevityReward:
    """Reward valid, shorter answers."""

    def __call__(self, trajectories: list[dict[str, Any]], **kwargs) -> list[float]:
        rewards = []
        for traj in trajectories:
            messages = traj.get('messages', [])
            completion = ''
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    completion = msg.get('content', '')
                    break
            has_answer = bool(re.search(r'\\boxed\{[^}]+\}', completion) or re.search(r'####\s*[\-\d,\.]+', completion))
            if not has_answer:
                rewards.append(0.0)
                continue
            length = len(completion)
            rewards.append(1.0 if length <= 300 else max(0.0, 1.0 - (length - 300) / 3000))
        return rewards


class GSM8KReward:

    def __init__(self):
        from twinkle.reward import GSM8KAccuracyReward

        self.accuracy = GSM8KAccuracyReward()
        self.brevity = GSM8KBrevityReward()

    def __call__(self, trajectories: list[dict[str, Any]], **kwargs) -> list[float]:
        accuracy = self.accuracy(trajectories)
        brevity = self.brevity(trajectories)
        return [a + b for a, b in zip(accuracy, brevity)]


class ServerSingleTurnRollout:
    """One prompt-group rollout adapter for local/server vLLMSampler."""

    def __init__(self, sampler: Any, *, sampling_params: Any, num_generations: int):
        self.sampler = sampler
        self.sampling_params = sampling_params
        self.num_generations = num_generations

    def __call__(self, trajectories: list[dict[str, Any]], **kwargs) -> list[dict[str, Any]]:
        adapter_path = kwargs.get('adapter_path')
        adapter_name = kwargs.get('adapter_name', '')
        expanded = []
        for prompt_idx, trajectory in enumerate(trajectories):
            group_id = trajectory.get('group_id') or trajectory.get('sample_id') or f'prompt_{prompt_idx}'
            for generation_idx in range(self.num_generations):
                item = dict(trajectory)
                item['group_id'] = group_id
                item['generation_idx'] = generation_idx
                expanded.append(item)

        responses = self.sampler.sample(
            expanded,
            self.sampling_params,
            adapter_name=adapter_name,
            adapter_path=adapter_path,
        )
        rows: list[dict[str, Any]] = []
        for source, response in zip(expanded, responses):
            for sequence in response.sequences:
                row = dict(sequence.new_input_feature or source)
                row.setdefault('group_id', source['group_id'])
                row.setdefault('generation_idx', source['generation_idx'])
                row['old_logps'] = self._extract_logps(sequence.logprobs)
                row['stop_reason'] = sequence.stop_reason
                row['policy_version'] = kwargs.get('policy_version')
                rows.append(row)
        return rows

    @staticmethod
    def _extract_logps(logprobs) -> list[float]:
        values = []
        for item in logprobs or []:
            if not item:
                values.append(0.0)
            else:
                values.append(float(item[0][1]))
        return values


class AsyncMultiLoraGRPOPipeline(BaseRLPipeline):
    """Config-driven server-side multi-LoRA GRPO pipeline implementation."""

    def __init__(self, cfg, *, model_mesh: Any, sampler_mesh: Any):
        self.cfg = cfg
        self.model_mesh = model_mesh
        self.sampler_mesh = sampler_mesh
        super().__init__(config=build_base_pipeline_config(cfg))

    def build_model(self):
        from omegaconf import OmegaConf
        from peft import LoraConfig

        from twinkle.model import MultiLoraTransformersModel
        from twinkle.processor import InputProcessor

        primary_context = primary_training_context(self.cfg)
        lora_cfg = self.cfg.model.lora
        lora_config = LoraConfig(
            target_modules=lora_cfg.target_modules,
            r=int(lora_cfg.r),
            lora_alpha=int(lora_cfg.lora_alpha),
            lora_dropout=float(lora_cfg.lora_dropout),
        )
        model_kwargs = {
            k: v
            for k, v in OmegaConf.to_container(self.cfg.model, resolve=True).items() if k in {
                'strategy',
                'ddp_config',
                'fsdp_config',
                'grad_scaler_config',
                'memory_efficient_init',
                'max_loras',
            }
        }
        model = MultiLoraTransformersModel(
            model_id=primary_context.base_model_id,
            device_mesh=self.model_mesh,
            mixed_precision=self.cfg.model.mixed_precision,
            max_r=int(self.cfg.model.get('max_r', lora_cfg.r)),
            max_length=int(self.cfg.model.template.max_length),
            target_modules=lora_cfg.target_modules,
            **({
                'remote_group': 'model'
            } if self.cfg.runtime.mode == 'ray' else {}),
            **model_kwargs,
        )
        loss_kwargs = {k: v for k, v in self.cfg.model.loss.items() if k != 'cls'}
        template_kwargs = {
            k: v
            for k, v in self.cfg.model.template.items() if k not in {'cls', 'max_length', 'truncation_strategy'}
        }
        for context_cfg in training_context_configs(self.cfg):
            adapter_name = context_cfg.adapter_name
            model.add_adapter_to_model(
                adapter_name,
                lora_config,
                gradient_accumulation_steps=int(self.cfg.model.gradient_accumulation_steps),
            )
            model.set_optimizer(
                self.cfg.model.optimizer.cls,
                lr=float(self.cfg.model.optimizer.lr),
                adapter_name=adapter_name,
            )
            scheduler_kwargs = {k: v for k, v in self.cfg.model.lr_scheduler.items() if k != 'cls'}
            model.set_lr_scheduler(
                self.cfg.model.lr_scheduler.cls,
                adapter_name=adapter_name,
                **scheduler_kwargs,
            )
            model.set_loss(self.cfg.model.loss.cls, adapter_name=adapter_name, **loss_kwargs)
            model.set_processor(InputProcessor, adapter_name=adapter_name)
            model.set_template(
                self.cfg.model.template.cls,
                model_id=primary_context.base_model_id,
                adapter_name=adapter_name,
                **template_kwargs,
            )
        return model

    def build_rollout(self):
        from omegaconf import OmegaConf

        from twinkle.data_format import SamplingParams
        from twinkle.sampler import vLLMSampler

        primary_context = primary_training_context(self.cfg)
        engine_args = OmegaConf.to_container(self.cfg.sampler.engine_args, resolve=True)
        engine_args.setdefault('tensor_parallel_size', int(self.cfg.runtime.sampler_tp))
        sampler = vLLMSampler(
            model_id=primary_context.base_model_id,
            engine_args=engine_args,
            device_mesh=self.sampler_mesh,
            **({
                'remote_group': 'sampler'
            } if self.cfg.runtime.mode == 'ray' else {}),
        )
        sampler_template_kwargs = {k: v for k, v in self.cfg.sampler.template.items() if k != 'cls'}
        sampler.set_template(
            self.cfg.sampler.template.cls,
            model_id=primary_context.base_model_id,
            **sampler_template_kwargs,
        )
        sampling_params = SamplingParams.from_dict(
            OmegaConf.to_container(self.cfg.sampler.sampling_params, resolve=True))
        return ServerSingleTurnRollout(
            sampler,
            sampling_params=sampling_params,
            num_generations=int(self.cfg.sampler.num_generations),
        )

    def build_data_plane(self):
        tq_cfg = self.cfg.transfer_queue
        return TransferQueueDataPlane(
            tq_config=TransferQueueRuntimeConfig(
                init=bool(tq_cfg.get('init', True)),
                total_storage_size=tq_cfg.get('total_storage_size'),
                max_rows=tq_cfg.get('max_rows'),
                max_rows_per_context=tq_cfg.get('max_rows_per_context'),
                num_data_storage_units=int(tq_cfg.get('num_data_storage_units', 4)),
                storage_backend=tq_cfg.get('storage_backend', 'SimpleStorage'),
            ))

    def build_prompt_feeders(self):
        from omegaconf import OmegaConf

        from twinkle.dataloader import DataLoader

        feeders = []
        max_pending_groups = self.cfg.pipeline.get('prompt_max_pending_groups')
        for context_cfg, context in zip(training_context_configs(self.cfg), self.contexts):
            dataset_cfg = context_dataset_config(self.cfg, context_cfg)
            safe_context_key = re.sub(r'[^A-Za-z0-9_.-]+', '_', context.key)
            dataset_factory = partial(
                build_prompt_dataset_from_config,
                OmegaConf.to_container(context_cfg, resolve=True),
                OmegaConf.to_container(self.cfg.model.template, resolve=True),
            )
            dataloader = DataLoader(
                dataset=dataset_factory,
                batch_size=int(dataset_cfg.batch_size),
                min_batch_size=int(dataset_cfg.batch_size),
                device_mesh=self.model_mesh,
                remote_group='model',
                instance_id=f'{os.getpid()}-{safe_context_key}-',
            )
            feeders.append(
                PromptFeeder(
                    context=context,
                    dataloader=dataloader,
                    rollouter=self.rollouter,
                    max_pending_groups=max_pending_groups,
                ))
        return feeders

    def build_dataset(self, context_cfg):
        from omegaconf import OmegaConf

        return build_prompt_dataset_from_config(
            OmegaConf.to_container(context_cfg, resolve=True),
            OmegaConf.to_container(self.cfg.model.template, resolve=True),
        )

    def build_reward_registry(self):
        registry = {}
        for context_cfg in training_context_configs(self.cfg):
            if context_cfg.reward_type == 'gsm8k':
                registry[context_cfg.reward_type] = GSM8KReward()
            else:
                raise ValueError(f'unsupported reward_type for AsyncMultiLoraGRPOPipeline: {context_cfg.reward_type}')
        return registry

    def build_advantage_fn(self):
        return grpo_advantage_fn

    def build_train_partition_fn(self):
        return self.train_partition

    def train_partition(self, context, partition_id: str, dataloader) -> TrainerStepResult:
        batch = list(dataloader)
        mini_batch_size = int(self.cfg.pipeline.mini_batch_size)
        for mb_start in range(0, len(batch), mini_batch_size):
            mini_batch = batch[mb_start:mb_start + mini_batch_size]
            inputs = [model_input_from_training_sample(sample) for sample in mini_batch]
            for model_input in inputs:
                validate_training_input_length(
                    model_input,
                    context=context,
                    partition_id=partition_id,
                    max_length=int(self.cfg.model.template.max_length),
                )
            old_logps = [sample.get('old_logps', []) for sample in mini_batch]
            advantages = [sample.get('advantages', 0.0) for sample in mini_batch]
            self.model.forward_backward(
                inputs=inputs,
                old_logps=old_logps,
                advantages=advantages,
                adapter_name=context.adapter_name,
            )
            self.model.clip_grad_and_step(
                adapter_name=context.adapter_name,
                max_grad_norm=float(self.cfg.pipeline.max_grad_norm),
                norm_type=int(self.cfg.pipeline.norm_type),
            )

        save_name = (f'{self.cfg.pipeline.save_name_prefix}-{context.training_run_id}-'
                     f'{context.adapter_name}-v{context.policy_version + 1}')
        save_result = self.model.save(
            save_name,
            output_dir=self.cfg.model.adapter_checkpoint_dir,
            adapter_name=context.adapter_name,
            save_optimizer=bool(self.cfg.pipeline.save_optimizer),
            is_sampler=bool(self.cfg.pipeline.is_sampler_checkpoint),
        )
        adapter_path = save_result if isinstance(save_result, str) else getattr(save_result, 'twinkle_path', None)
        return TrainerStepResult(adapter_path=adapter_path)


def grpo_advantage_fn(samples: list[dict[str, Any]], context) -> tuple[list[float], list[float]]:
    from twinkle.advantage import GRPOAdvantage

    rewards = [float(sample.get('rewards', sample.get('reward', 0.0))) for sample in samples]
    if not rewards:
        return [], []
    num_generations = max(1, max(int(sample.get('generation_idx', 0)) for sample in samples) + 1)
    advantages = GRPOAdvantage()(rewards, num_generations=num_generations, scale='group').tolist()
    return advantages, rewards
