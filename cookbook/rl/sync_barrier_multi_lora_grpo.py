"""Synchronous barrier baseline for native async multi-LoRA GRPO.

The model, sampler, datasets, rewards, batch semantics, and checkpoint cadence
match ``async_multi_lora_grpo.py``.  The only intentional difference is the
execution schedule: every round finishes rollout for all active contexts
before any context starts training.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from omegaconf import OmegaConf

from twinkle_agentic.async_rl.metrics import JSONLMetricsRecorder, rollout_performance_metrics
from twinkle_agentic.async_rl.pipeline import (_model_attention_implementation, _prompt_batches,
                                                _reward_for_context, _sampler_data_parallel_size,
                                                _sequence_parallel_size, _train_batch,
                                                _validate_context_batch_config, configure_lora_lr_scheduler)
from twinkle_agentic.async_rl.tq_utils import REQUIRED_MODEL_INPUT_FIELDS, columns_to_tq_fields
from twinkle_agentic.async_rl.types import LoraContext, PartitionAdmission
from twinkle_agentic.async_rl.vllm_sampler_tq import (_compute_reward_metrics,
                                                      _sample_responses_to_rollout_rows)


@dataclass
class SyncContextState:
    context: LoraContext
    prompt_batches: Iterator[Sequence[dict[str, Any]]]
    rollout_batch_size: int
    num_generations: int
    sampling_params: Any
    advantage_groups_per_batch: int
    train_groups_per_batch: int
    reward_fn: Any
    adapter_path: str
    adapter_history: list[str]
    partition_step: int = 0
    optimizer_steps: int = 0
    policy_version: int = 0
    exhausted: bool = False


@dataclass
class SyncPartition:
    admission: PartitionAdmission
    state: SyncContextState
    rows: list[dict[str, Any]]
    rewards: list[float]
    advantages: list[float] | None = None


class SyncBarrierMultiLoraGRPO:

    def __init__(self, raw_config: dict[str, Any]):
        import twinkle
        from peft import LoraConfig
        from twinkle import DeviceGroup, DeviceMesh
        from twinkle.data_format import SamplingParams
        from twinkle.model import MultiLoraTransformersModel
        from twinkle.processor import InputProcessor
        from twinkle.sampler import vLLMSampler

        runtime = raw_config['runtime']
        model_config = raw_config['model']
        lora_data = raw_config['lora']
        template_data = raw_config.get('template', {})
        template_cls = template_data.get('cls', 'Qwen3_5Template')
        enable_thinking = bool(template_data.get('enable_thinking', False))
        model_gpus = int(runtime['model_gpus'])
        sampler_gpus = int(runtime['sampler_gpus'])
        sampler_tp = int(runtime['sampler_tp'])
        sampler_dp = _sampler_data_parallel_size(sampler_gpus, sampler_tp)
        sequence_parallel_size = _sequence_parallel_size(
            model_gpus,
            int(model_config['sequence_parallel_size']),
        )
        padding_free = bool(model_config['padding_free'])
        attn_implementation = _model_attention_implementation(
            model_config,
            padding_free=padding_free,
            sequence_parallel_size=sequence_parallel_size,
        )
        model_max_length = int(model_config['max_length'])
        sampler_config = raw_config['sampler']
        total_gpus = model_gpus + sampler_gpus

        twinkle.initialize(
            mode='ray',
            nproc_per_node=total_gpus,
            groups=[
                DeviceGroup('model', list(range(model_gpus)), device_type='GPU'),
                DeviceGroup(
                    'sampler',
                    list(range(model_gpus, total_gpus)),
                    device_type='GPU',
                    gpus_per_worker=sampler_tp,
                ),
            ],
            lazy_collect=False,
        )
        model_mesh = DeviceMesh.from_sizes(
            world_size=model_gpus,
            dp_size=model_gpus,
            ulysses_size=sequence_parallel_size,
        )
        model_data_parallel_size = model_mesh.data_world_size
        sampler_mesh = DeviceMesh.from_sizes(
            world_size=sampler_gpus,
            dp_size=sampler_dp,
            tp_size=sampler_tp,
        )
        model_kwargs = {}
        if attn_implementation is not None:
            model_kwargs['attn_implementation'] = attn_implementation
        self.model = MultiLoraTransformersModel(
            model_id=runtime['model_id'],
            device_mesh=model_mesh,
            remote_group='model',
            max_length=model_max_length,
            **model_kwargs,
        )
        lora_config = LoraConfig(
            target_modules='all-linear',
            r=lora_data['r'],
            lora_alpha=lora_data['alpha'],
            lora_dropout=lora_data['dropout'],
        )
        self.micro_batch_sizes: dict[str, int] = {}
        self.states: list[SyncContextState] = []
        for item in raw_config['lora_contexts']:
            context = LoraContext(
                item['tenant_id'],
                item['training_run_id'],
                runtime['model_id'],
                item['adapter_name'],
                item.get('reward_type', 'gsm8k_accuracy'),
            )
            rollout = item['rollout']
            advantage = item['advantage']
            train = item['train']
            rollout_batch_size = int(rollout['batch_size'])
            num_generations = int(rollout['num_generations'])
            advantage_groups = int(advantage['groups_per_batch'])
            train_groups = int(train['groups_per_batch'])
            micro_batch_size = int(train['micro_batch_size'])
            _validate_context_batch_config(
                context.key,
                rollout_groups=rollout_batch_size,
                num_generations=num_generations,
                advantage_groups=advantage_groups,
                train_groups=train_groups,
                micro_batch_size=micro_batch_size,
                sampler_dp=sampler_dp,
                model_dp=model_data_parallel_size,
            )
            self.model.add_adapter_to_model(context.adapter_name, lora_config, gradient_accumulation_steps=1)
            self.model.set_optimizer('AdamW', lr=lora_data['learning_rate'], adapter_name=context.adapter_name)
            configure_lora_lr_scheduler(self.model, context.adapter_name, lora_data)
            self.model.set_loss('GRPOLoss', epsilon=.2, adapter_name=context.adapter_name)
            self.model.set_processor(
                InputProcessor,
                adapter_name=context.adapter_name,
                padding_free=padding_free,
            )
            self.model.set_template(
                template_cls,
                model_id=runtime['model_id'],
                adapter_name=context.adapter_name,
                enable_thinking=enable_thinking,
                max_length=model_max_length,
            )
            initial_path = self.model.save(
                f'sync-{context.adapter_name}-initial',
                output_dir=runtime['output_dir'],
                adapter_name=context.adapter_name,
            )
            state = SyncContextState(
                context=context,
                prompt_batches=iter(
                    _prompt_batches(
                        item['dataset'],
                        model_id=runtime['model_id'],
                        batch_size=rollout_batch_size,
                        template_cls=template_cls,
                        enable_thinking=enable_thinking,
                    )),
                rollout_batch_size=rollout_batch_size,
                num_generations=num_generations,
                sampling_params=SamplingParams(
                    max_tokens=rollout['max_tokens'],
                    temperature=rollout['temperature'],
                    top_p=rollout['top_p'],
                    repetition_penalty=float(rollout.get('repetition_penalty', 1.0)),
                    logprobs=1,
                    num_samples=1,
                ),
                advantage_groups_per_batch=advantage_groups,
                train_groups_per_batch=train_groups,
                reward_fn=_reward_for_context(context),
                adapter_path=initial_path,
                adapter_history=[initial_path],
            )
            self.states.append(state)
            self.micro_batch_sizes[context.key] = micro_batch_size

        self.sampler = vLLMSampler(
            model_id=runtime['model_id'],
            remote_group='sampler',
            device_mesh=sampler_mesh,
            engine_args={
                'tensor_parallel_size': sampler_tp,
                'enable_lora': True,
                'max_loras': int(runtime['sampler_max_loras']),
                'max_lora_rank': lora_data['r'],
                'max_model_len': int(sampler_config['max_model_len']),
                'gpu_memory_utilization': float(sampler_config['gpu_memory_utilization']),
                'max_num_seqs': int(sampler_config['max_num_seqs']),
                'max_num_batched_tokens': int(sampler_config['max_num_batched_tokens']),
                'enforce_eager': bool(sampler_config['enforce_eager']),
            },
        )
        self.sampler.set_template(
            template_cls,
            model_id=runtime['model_id'],
            enable_thinking=enable_thinking,
            max_length=model_max_length,
        )
        self.output_dir = runtime['output_dir']
        self.max_steps = runtime.get('max_steps')
        self.max_steps = None if self.max_steps is None else int(self.max_steps)
        self.keep_adapter_versions = max(0, int(runtime.get('keep_adapter_versions', 0)))
        self.metrics = JSONLMetricsRecorder(
            runtime['metrics_path'],
            run_id=runtime.get('run_id', 'sync_barrier_multi_lora_grpo'),
            mode='sync_barrier',
        )
        self.completed_partitions = 0
        self._creation_order = 0

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        round_index = 0
        try:
            while self.max_steps is None or self.completed_partitions < self.max_steps:
                self.metrics.record(
                    event='barrier_round_started',
                    metrics={'round_index': round_index, 'trained_partitions': self.completed_partitions},
                )
                partitions = self._rollout_round()
                if not partitions:
                    break
                self._advantage_round(partitions)
                self._train_round(partitions)
                self.metrics.record(
                    event='barrier_round_done',
                    metrics={
                        'round_index': round_index,
                        'trained_partitions': self.completed_partitions,
                        'overlap_ratio': 0.0,
                    },
                )
                round_index += 1
        except Exception as exc:
            self.metrics.flush()
            self.metrics.record(
                event='run_failed',
                metrics={
                    'error': f'{type(exc).__name__}: {exc}',
                    'wall_time_s': time.perf_counter() - started,
                    **self.metrics.stats(),
                },
            )
            self.metrics.close()
            raise
        result = {
            'trained_partitions': self.completed_partitions,
            'wall_time_s': time.perf_counter() - started,
            'per_context': {
                state.context.key: {
                    'optimizer_steps': state.optimizer_steps,
                    'policy_version': state.policy_version,
                    'adapter_path': state.adapter_path,
                }
                for state in self.states
            },
        }
        self.metrics.flush()
        result.update(self.metrics.stats())
        self.metrics.record(event='run_completed', metrics=result)
        self.metrics.close()
        return result

    def _rollout_round(self) -> list[SyncPartition]:
        partitions = []
        for state in self.states:
            if state.exhausted:
                continue
            if self.max_steps is not None and self.completed_partitions + len(partitions) >= self.max_steps:
                break
            prompts = next(state.prompt_batches, None)
            if prompts is None or len(prompts) != state.rollout_batch_size:
                state.exhausted = True
                continue
            admission = PartitionAdmission(
                context=state.context,
                partition_id=state.context.partition_id(state.partition_step),
                step=state.partition_step,
                target_groups=state.rollout_batch_size,
                num_generations=state.num_generations,
                created_order=self._creation_order,
            )
            self._creation_order += 1
            self.metrics.record(
                event='rollout_submitted',
                context=state.context,
                partition_id=admission.partition_id,
                policy_version=state.policy_version,
                metrics={
                    'prompt_count': admission.target_groups,
                    'sample_count': admission.sample_count,
                    'num_generations': admission.num_generations,
                },
            )
            rollout_started = time.perf_counter()
            sources = [{
                **dict(prompt),
                'group_id': f'{admission.partition_id}/group_{group_index}',
                'generation_idx': generation_index,
            } for group_index, prompt in enumerate(prompts)
                       for generation_index in range(state.num_generations)]
            responses = self.sampler.sample(
                [dict(prompt) for prompt in prompts for _ in range(state.num_generations)],
                state.sampling_params,
                adapter_name=state.context.adapter_name,
                adapter_path=state.adapter_path,
            )
            rows = _sample_responses_to_rollout_rows(
                sources,
                responses,
                policy_version=state.policy_version,
            )
            if len(rows) != admission.sample_count:
                raise ValueError(
                    f'{admission.partition_id} expected {admission.sample_count} samples, got {len(rows)}')
            for row in rows:
                row.update({
                    'rollout_adapter_path': state.adapter_path,
                    'rollout_policy_versions': [state.policy_version],
                    'initial_policy_version': state.policy_version,
                    'final_policy_version': state.policy_version,
                    'policy_version_span': 0,
                })
            rewards = [float(value) for value in state.reward_fn(rows, context=state.context)]
            if len(rewards) != len(rows):
                raise ValueError(f'{admission.partition_id} reward count does not match sample count')
            rollout_latency_s = time.perf_counter() - rollout_started
            self._record_rollout_groups(state, admission, rows, rewards, rollout_latency_s)
            self.metrics.record(
                event='rollout_partition_done',
                context=admission.context,
                partition_id=admission.partition_id,
                policy_version=state.policy_version,
                metrics=rollout_performance_metrics(rows, rollout_latency_s=rollout_latency_s),
            )
            partitions.append(SyncPartition(admission, state, rows, rewards))
            state.partition_step += 1
        if partitions:
            self.metrics.record(
                event='rollout_barrier_reached',
                metrics={'partitions': len(partitions), 'trained_partitions': self.completed_partitions},
            )
        return partitions

    def _record_rollout_groups(
        self,
        state: SyncContextState,
        admission: PartitionAdmission,
        rows: list[dict[str, Any]],
        rewards: list[float],
        rollout_latency_s: float,
    ) -> None:
        for group_index in range(admission.target_groups):
            start = group_index * admission.num_generations
            end = start + admission.num_generations
            group_rows = rows[start:end]
            group_rewards = rewards[start:end]
            metrics = {
                **_compute_reward_metrics(
                    {state.context.key: state.reward_fn},
                    state.context,
                    group_rows,
                    group_rewards,
                ),
                'group_id': f'{admission.partition_id}/group_{group_index}',
                'reward': sum(group_rewards) / len(group_rows),
                **rollout_performance_metrics(group_rows),
                'rollout_latency_s': rollout_latency_s,
                'policy_version': state.policy_version,
            }
            self.metrics.record(
                event='rollout_done',
                context=admission.context,
                partition_id=admission.partition_id,
                policy_version=state.policy_version,
                metrics=metrics,
            )

    def _advantage_round(self, partitions: list[SyncPartition]) -> None:
        from twinkle.advantage import GRPOAdvantage

        advantage_fn = GRPOAdvantage()
        for partition in partitions:
            admission = partition.admission
            partition.advantages = advantage_fn(
                partition.rewards,
                num_generations=admission.num_generations,
                scale='group',
            ).tolist()
            groups = partition.state.advantage_groups_per_batch
            samples_per_batch = groups * admission.num_generations
            for start in range(0, len(partition.rows), samples_per_batch):
                self.metrics.record(
                    event='advantage_done',
                    context=admission.context,
                    partition_id=admission.partition_id,
                    policy_version=partition.state.policy_version,
                    metrics={'sample_count': min(samples_per_batch, len(partition.rows) - start)},
                )

    def _train_round(self, partitions: list[SyncPartition]) -> None:
        for partition in partitions:
            admission = partition.admission
            state = partition.state
            assert partition.advantages is not None
            samples_per_batch = state.train_groups_per_batch * admission.num_generations
            for start in range(0, len(partition.rows), samples_per_batch):
                end = start + samples_per_batch
                batch = self._training_batch(
                    partition.rows[start:end],
                    partition.rewards[start:end],
                    partition.advantages[start:end],
                )
                train_started = time.perf_counter()
                metrics = _train_batch(self.model, self.micro_batch_sizes, batch, admission)
                state.optimizer_steps += 1
                metrics.update({
                    'sample_count': end - start,
                    'train_latency_s': time.perf_counter() - train_started,
                    'policy_version_gap_mean': 0.0,
                    'policy_version_gap_p95': 0.0,
                    'policy_version_gap_max': 0,
                    'rollout_policy_span_mean': 0.0,
                    'rollout_policy_span_max': 0,
                    'optimizer_step': state.optimizer_steps,
                })
                self.metrics.record(
                    event='train_step_done',
                    context=state.context,
                    partition_id=admission.partition_id,
                    policy_version=state.policy_version,
                    metrics=metrics,
                )
            finalize_started = time.perf_counter()
            next_policy_version = state.policy_version + 1
            save_started = time.perf_counter()
            state.adapter_path = self.model.save(
                f'sync-{state.context.adapter_name}-v{next_policy_version}',
                output_dir=self.output_dir,
                adapter_name=state.context.adapter_name,
            )
            adapter_save_latency_s = time.perf_counter() - save_started
            publish_started = time.perf_counter()
            state.policy_version = next_policy_version
            policy_publish_latency_s = time.perf_counter() - publish_started
            state.adapter_history.append(state.adapter_path)
            self.metrics.record(
                event='policy_published',
                context=state.context,
                partition_id=admission.partition_id,
                policy_version=state.policy_version,
                metrics={
                    'adapter_path': state.adapter_path,
                    'adapter_save_latency_s': adapter_save_latency_s,
                    'policy_publish_latency_s': policy_publish_latency_s,
                },
            )
            prune_started = time.perf_counter()
            self._prune_adapter_history(state)
            adapter_prune_latency_s = time.perf_counter() - prune_started
            self.completed_partitions += 1
            self.metrics.record(
                event='partition_done',
                context=state.context,
                partition_id=admission.partition_id,
                policy_version=state.policy_version,
                metrics={
                    'adapter_save_latency_s': adapter_save_latency_s,
                    'policy_publish_latency_s': policy_publish_latency_s,
                    'adapter_prune_latency_s': adapter_prune_latency_s,
                    'partition_finalize_latency_s': time.perf_counter() - finalize_started,
                },
            )

    @staticmethod
    def _training_batch(rows: list[dict[str, Any]], rewards: list[float], advantages: list[float]):
        fields = {
            name: [row[name] for row in rows]
            for name in (*REQUIRED_MODEL_INPUT_FIELDS, 'logprobs')
        }
        fields.update({'rewards': rewards, 'advantages': advantages})
        return columns_to_tq_fields(fields, len(rows))

    def _prune_adapter_history(self, state: SyncContextState) -> None:
        retained_count = max(1, self.keep_adapter_versions)
        stale = state.adapter_history[:-retained_count]
        state.adapter_history = state.adapter_history[-retained_count:]
        for path in stale:
            if os.path.isdir(path):
                shutil.rmtree(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='cookbook/rl/sync_barrier_multi_lora_grpo.yaml')
    args = parser.parse_args()
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    print(SyncBarrierMultiLoraGRPO(config).run())


if __name__ == '__main__':
    main()

# MODEL_ID=/path/to/model \
# DATASET_ID=/path/to/gsm8k \
# python cookbook/rl/async_multi_lora_grpo.py
