# Copyright (c) ModelScope Contributors. All rights reserved.
"""Async GRPO cookbook using TransferQueue + MultiLoraTransformersModel.

This is the runnable MVP entrypoint for the async RL pipeline:

  dataset prompts -> rollout -> TransferQueue -> reward -> advantage -> trainer

Training samples are read from TransferQueue by TrainerWorker. The optional
argument passed to `pipeline.run(...)` is only a rollout prompt feed.

Prerequisites:
  1. Twinkle model and sampler services are running.
  2. TransferQueue is installed and reachable.
  3. vLLM sampler service is created with LoRA enabled.

Run:
  python cookbook/client/twinkle/self_host/async_multi_lora_grpo.py \
    --config cookbook/client/twinkle/self_host/async_multi_lora_grpo.yaml
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from omegaconf import OmegaConf
from peft import LoraConfig

from twinkle import get_logger, init_twinkle_client
from twinkle.advantage import GRPOAdvantage
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.preprocessor.llm import GSM8KProcessor
from twinkle.reward import GSM8KAccuracyReward
from twinkle.reward.base import Reward
from twinkle_agentic.async_rl import (
    BaseRLPipeline,
    BaseRLPipelineConfig,
    PromptLoader,
    LoraContext,
    TransferQueueDataPlane,
    TransferQueueRuntimeConfig,
)
from twinkle_client.sampler import vLLMSampler

logger = get_logger()


class GSM8KBrevityReward(Reward):
    """Reward valid, shorter answers."""

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for traj in trajectories:
            messages = traj.get('messages', [])
            completion = ''
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    completion = msg.get('content', '')
                    break
            has_answer = bool(
                re.search(r'\\boxed\{[^}]+\}', completion)
                or re.search(r'####\s*[\-\d,\.]+', completion)
            )
            if not has_answer:
                rewards.append(0.0)
                continue
            length = len(completion)
            rewards.append(1.0 if length <= 300 else max(0.0, 1.0 - (length - 300) / 3000))
        return rewards


class GSM8KReward:

    def __init__(self):
        self.accuracy = GSM8KAccuracyReward()
        self.brevity = GSM8KBrevityReward()

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        accuracy = self.accuracy(trajectories)
        brevity = self.brevity(trajectories)
        return [a + b for a, b in zip(accuracy, brevity)]


class SingleTurnClientRollout:
    """One-turn rollout adapter for twinkle_client vLLMSampler.

    It accepts one prompt group and returns `num_generations` train rows.
    `adapter_path` from LoraAdapterRegistry is mapped to client sampler's
    `adapter_uri` argument.
    """

    def __init__(self, sampler: vLLMSampler, *, sampling_params: Dict[str, Any], num_generations: int):
        self.sampler = sampler
        self.sampling_params = dict(sampling_params)
        self.num_generations = num_generations

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        adapter_uri = kwargs.get('adapter_path') or kwargs.get('adapter_uri')
        adapter_name = kwargs.get('adapter_name', '')
        responses = self.sampler.sample(
            inputs=trajectories,
            sampling_params=self.sampling_params,
            adapter_name=adapter_name,
            adapter_uri=adapter_uri,
            num_samples=self.num_generations,
        )
        rows: list[dict[str, Any]] = []
        for prompt_idx, response in enumerate(responses):
            source = trajectories[prompt_idx]
            group_id = source.get('group_id') or source.get('sample_id') or f'prompt_{prompt_idx}'
            for generation_idx, sequence in enumerate(response.sequences):
                row = dict(sequence.new_input_feature or source)
                row.setdefault('group_id', group_id)
                row['generation_idx'] = generation_idx
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


def load_config(path: str):
    return OmegaConf.load(path)


def lora_context_configs(cfg):
    if cfg.get('lora_contexts'):
        return list(cfg.lora_contexts)
    return [cfg.lora_context]


def primary_lora_context(cfg):
    return lora_context_configs(cfg)[0]


def build_lora_contexts(cfg) -> list[LoraContext]:
    contexts = []
    for context_cfg in lora_context_configs(cfg):
        contexts.append(
            LoraContext(
                tenant_id=context_cfg.tenant_id,
                training_run_id=context_cfg.training_run_id,
                base_model_id=context_cfg.base_model_id,
                adapter_name=context_cfg.adapter_name,
                reward_type=context_cfg.reward_type,
                tool_profile=context_cfg.get('tool_profile', 'default'),
            )
        )
    base_models = {context.base_model_id for context in contexts}
    if len(base_models) != 1:
        raise ValueError(f'one async multi-LoRA job must use one base model, got {sorted(base_models)}')
    return contexts


def maybe_init_ray(cfg) -> None:
    if not cfg.transfer_queue.get('init_ray', False):
        return
    import ray
    if not ray.is_initialized():
        ray.init(namespace=cfg.transfer_queue.get('ray_namespace', 'TransferQueueApp'))


def context_dataset_config(cfg, context_cfg):
    return context_cfg.get('dataset', cfg.dataset)


def create_dataset(cfg, context_cfg):
    dataset_cfg = context_dataset_config(cfg, context_cfg)
    data_slice = range(int(dataset_cfg.data_num)) if dataset_cfg.get('data_num') else None
    dataset = Dataset()
    dataset.add_dataset(
        DatasetMeta(
            dataset_cfg.dataset_id,
            subset_name=dataset_cfg.get('subset_name'),
            split=dataset_cfg.get('split', 'train'),
            data_slice=data_slice,
        )
    )
    template_cfg = cfg.model.template
    dataset.set_template(
        template_cfg.cls,
        model_id=context_cfg.base_model_id,
        max_length=template_cfg.get('max_length', 2048),
        truncation_strategy=template_cfg.get('truncation_strategy', 'delete'),
        enable_thinking=template_cfg.get('enable_thinking', False),
    )
    dataset.map(GSM8KProcessor(system=dataset_cfg.system_prompt))
    dataset.encode(add_generation_prompt=True)
    return dataset


def build_model(cfg):
    from twinkle_client.model import MultiLoraTransformersModel

    primary_context = primary_lora_context(cfg)
    lora_cfg = cfg.model.lora
    lora_config = LoraConfig(
        target_modules=lora_cfg.target_modules,
        r=int(lora_cfg.r),
        lora_alpha=int(lora_cfg.lora_alpha),
        lora_dropout=float(lora_cfg.lora_dropout),
    )
    model = MultiLoraTransformersModel(model_id=primary_context.base_model_id)
    loss_kwargs = {k: v for k, v in cfg.model.loss.items() if k != 'cls'}
    optimizer_params = {'lr': float(cfg.model.optimizer.lr)}
    template_kwargs = {k: v for k, v in cfg.model.template.items() if k != 'cls'}
    for context_cfg in lora_context_configs(cfg):
        adapter_name = context_cfg.adapter_name
        model.add_adapter_to_model(
            adapter_name,
            lora_config,
            gradient_accumulation_steps=int(cfg.model.gradient_accumulation_steps),
        )
        model.set_loss(cfg.model.loss.cls, adapter_name=adapter_name, **loss_kwargs)
        model.set_optimizer(cfg.model.optimizer.cls, adapter_name=adapter_name, **optimizer_params)
        model.set_processor(cfg.model.processor.cls, adapter_name=adapter_name)
        model.set_template(
            cfg.model.template.cls,
            model_id=primary_context.base_model_id,
            adapter_name=adapter_name,
            **template_kwargs,
        )
    return model


def build_sampler(cfg):
    primary_context = primary_lora_context(cfg)
    sampler = vLLMSampler(model_id=primary_context.base_model_id)
    sampler.set_template(
        cfg.sampler.template.cls,
        model_id=primary_context.base_model_id,
        **{k: v for k, v in cfg.sampler.template.items() if k != 'cls'},
    )
    return sampler


def build_data_plane(cfg):
    maybe_init_ray(cfg)
    tq_cfg = cfg.transfer_queue
    return TransferQueueDataPlane(
        tq_config=TransferQueueRuntimeConfig(
            total_storage_size=tq_cfg.get('total_storage_size'),
            max_rows=tq_cfg.get('max_rows'),
            max_rows_per_context=tq_cfg.get('max_rows_per_context'),
            num_data_storage_units=int(tq_cfg.get('num_data_storage_units', 4)),
            storage_backend=tq_cfg.get('storage_backend', 'SimpleStorage'),
        )
    )


def grpo_advantage_fn(samples: List[Dict[str, Any]], context) -> tuple[list[float], list[float]]:
    rewards = [float(sample.get('rewards', sample.get('reward', 0.0))) for sample in samples]
    if not rewards:
        return [], []
    num_generations = max(1, len(samples))
    try:
        num_generations = max(1, max(int(sample.get('generation_idx', 0)) for sample in samples) + 1)
    except ValueError:
        pass
    advantages = GRPOAdvantage()(rewards, num_generations=num_generations, scale='group').tolist()
    return advantages, rewards


class AsyncMultiLoraGRPOPipeline(BaseRLPipeline):

    def __init__(self, cfg):
        self.cfg = cfg
        contexts = build_lora_contexts(cfg)
        primary_context = contexts[0]
        super().__init__(
            config=BaseRLPipelineConfig(
                lora_contexts=contexts,
                tenant_id=primary_context.tenant_id,
                training_run_id=primary_context.training_run_id,
                base_model_id=primary_context.base_model_id,
                adapter_name=primary_context.adapter_name,
                reward_type=primary_context.reward_type,
                tool_profile=primary_context.tool_profile,
                max_staleness=int(cfg.pipeline.max_staleness),
                target_groups_per_partition=int(cfg.pipeline.rollout.batch_size),
                max_concurrent_groups=int(cfg.pipeline.rollout.max_concurrent_groups),
                max_train_partitions=int(cfg.pipeline.max_steps),
                save_name_prefix=cfg.pipeline.save_name_prefix,
                is_sampler_checkpoint=bool(cfg.pipeline.is_sampler_checkpoint),
                save_optimizer=bool(cfg.pipeline.save_optimizer),
            )
        )

    def build_model(self):
        return build_model(self.cfg)

    def build_prompt_loaders(self):
        loaders = []
        max_pending_groups = self.cfg.pipeline.rollout.get('max_pending_groups')
        prompt_batch_size = int(self.cfg.pipeline.rollout.batch_size)
        for context_cfg, context in zip(lora_context_configs(self.cfg), self.contexts):
            dataloader = DataLoader(
                dataset=create_dataset(self.cfg, context_cfg),
                batch_size=prompt_batch_size,
                num_workers=0,
            )
            loaders.append(
                PromptLoader(
                    context=context,
                    dataloader=dataloader,
                    rollouter=self.rollouter,
                    max_pending_groups=max_pending_groups,
                )
            )
        return loaders

    def build_rollout(self):
        return SingleTurnClientRollout(
            build_sampler(self.cfg),
            sampling_params=OmegaConf.to_container(self.cfg.sampler.sampling_params, resolve=True),
            num_generations=int(self.cfg.pipeline.rollout.num_generations),
        )

    def build_data_plane(self):
        return build_data_plane(self.cfg)

    def build_reward_registry(self):
        return {context_cfg.reward_type: GSM8KReward() for context_cfg in lora_context_configs(self.cfg)}

    def build_advantage_fn(self):
        return grpo_advantage_fn


def build_pipeline(cfg):
    return AsyncMultiLoraGRPOPipeline(cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default=Path(__file__).with_suffix('.yaml').as_posix(),
        help='Path to async multi-LoRA GRPO YAML config.',
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    init_twinkle_client(
        base_url=cfg.client.base_url,
        api_key=os.environ.get('TWINKLE_API_KEY', cfg.client.api_key),
    )

    pipeline = build_pipeline(cfg)

    history = pipeline.run_until_idle(max_steps=int(cfg.pipeline.max_steps))
    trained = sum(1 for item in history if item.get('train') is not None)
    logger.info('async_multi_lora_grpo progress: trained_partitions=%s', trained)


if __name__ == '__main__':
    main()
