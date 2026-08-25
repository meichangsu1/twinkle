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

from twinkle.metric import MetricRecord, create_metrics_reporter
from twinkle_agentic.async_rl.metrics import advantage_signal_metrics, rollout_metrics
from twinkle_agentic.async_rl.pipeline import (_prompt_batches, _reward_for_context, _train_batch)
from twinkle_agentic.async_rl.tq_utils import REQUIRED_MODEL_INPUT_FIELDS, columns_to_tq_fields
from twinkle_agentic.async_rl.types import LoraContext, PartitionAdmission
from twinkle_agentic.async_rl.utils import (
    TrainBatchConfig,
    build_native_fsdp_model_kwargs,
    configure_lora_lr_scheduler,
    resolve_context_learning_rate,
    resolve_context_lora_target_modules,
    resolve_context_loss_config,
    resolve_model_attention_implementation,
    resolve_sequence_parallel_size,
    sample_responses_to_rollout_rows,
    sampler_data_parallel_size,
    validate_context_batch_config,
)
from twinkle_agentic.async_rl.vllm_sampler_tq import _compute_reward_metrics


@dataclass
class SyncContextState:
    context: LoraContext
    prompt_batches: Iterator[Sequence[dict[str, Any]]]
    rollout_batch_size: int
    num_generations: int
    sampling_params: Any
    mini_batch_size: int
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

        raw_config = OmegaConf.to_container(OmegaConf.create(raw_config), resolve=True)
        if not isinstance(raw_config, dict):
            raise TypeError('sync RL config must resolve to a mapping')

        runtime = raw_config['runtime']
        model_config = raw_config['model']
        lora_data = raw_config['lora']
        loss_data = raw_config.get('loss')
        template_data = raw_config.get('template', {})
        template_cls = template_data.get('cls', 'Qwen3_5Template')
        enable_thinking = bool(template_data.get('enable_thinking', False))
        model_gpus = int(runtime['model_gpus'])
        sampler_gpus = int(runtime['sampler_gpus'])
        sampler_tp = int(runtime['sampler_tp'])
        sampler_dp = sampler_data_parallel_size(sampler_gpus, sampler_tp)
        sequence_parallel_size = resolve_sequence_parallel_size(
            model_gpus,
            int(model_config['sequence_parallel_size']),
        )
        padding_free = bool(model_config['padding_free'])
        attn_implementation = resolve_model_attention_implementation(
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
        self.model_data_parallel_size = model_data_parallel_size
        sampler_mesh = DeviceMesh.from_sizes(
            world_size=sampler_gpus,
            dp_size=sampler_dp,
            tp_size=sampler_tp,
        )
        model_kwargs = build_native_fsdp_model_kwargs(model_config)
        if attn_implementation is not None:
            model_kwargs['attn_implementation'] = attn_implementation
        self.model = MultiLoraTransformersModel(
            model_id=runtime['model_id'],
            device_mesh=model_mesh,
            remote_group='model',
            max_length=model_max_length,
            **model_kwargs,
        )
        self.train_batch_configs: dict[str, TrainBatchConfig] = {}
        self.states: list[SyncContextState] = []
        self.evaluation_configs: dict[str, dict[str, Any]] = {}
        self._evaluation_batches: dict[str, list[Sequence[dict[str, Any]]]] = {}
        global_evaluation = dict(raw_config.get('evaluation') or {})
        for item in raw_config['lora_contexts']:
            context = LoraContext(
                item['tenant_id'],
                item['training_run_id'],
                runtime['model_id'],
                item['adapter_name'],
            )
            rollout = item['rollout']
            train = item['train']
            rollout_batch_size = int(rollout['batch_size'])
            num_generations = int(rollout['num_generations'])
            train_batch_config = TrainBatchConfig(
                mini_batch_size=int(train['mini_batch_size']),
                micro_batch_size=int(train['micro_batch_size']),
                dynamic_batching=bool(train.get('dynamic_batching', False)),
                max_tokens_per_micro_batch=(
                    int(train['max_tokens_per_micro_batch'])
                    if train.get('max_tokens_per_micro_batch') is not None else None
                ),
                packing_algorithm=str(train.get('packing_algorithm', 'ffd')),
            )
            validate_context_batch_config(
                context.key,
                rollout_groups=rollout_batch_size,
                num_generations=num_generations,
                train=train_batch_config,
                sampler_dp=sampler_dp,
                model_dp=model_data_parallel_size,
            )
            adapter_lora_config = LoraConfig(
                target_modules=resolve_context_lora_target_modules(item, lora_data),
                r=lora_data['r'],
                lora_alpha=lora_data['alpha'],
                lora_dropout=lora_data['dropout'],
            )
            self.model.add_adapter_to_model(
                context.adapter_name,
                adapter_lora_config,
                gradient_accumulation_steps=1,
            )
            self.model.set_optimizer(
                'AdamW',
                lr=resolve_context_learning_rate(train, lora_data),
                adapter_name=context.adapter_name,
            )
            configure_lora_lr_scheduler(self.model, context.adapter_name, lora_data)
            loss_cls, loss_kwargs = resolve_context_loss_config(item, loss_data)
            self.model.set_loss(
                loss_cls,
                adapter_name=context.adapter_name,
                **loss_kwargs,
            )
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
                mini_batch_size=train_batch_config.mini_batch_size,
                reward_fn=_reward_for_context(
                    item.get('reward'),
                    context_key=context.key,
                ),
                adapter_path=initial_path,
                adapter_history=[initial_path],
            )
            self.states.append(state)
            self.train_batch_configs[context.key] = train_batch_config
            if bool(global_evaluation.get('enabled', False)):
                eval_dataset = item.get('eval_dataset')
                if eval_dataset is None:
                    raise ValueError(f'eval_dataset is required for periodic evaluation of {context.key}')
                eval_batch_size = int(global_evaluation.get('batch_size', 16))
                eval_interval = int(global_evaluation.get('interval', 1))
                if eval_batch_size <= 0 or eval_interval <= 0:
                    raise ValueError('evaluation.batch_size and evaluation.interval must be positive')
                eval_sampling = dict(global_evaluation.get('sampling_params') or {})
                self.evaluation_configs[context.key] = {
                    'interval': eval_interval,
                    'dataset_name': eval_dataset.get('name', eval_dataset['dataset_id']),
                    'prompt_batches': _prompt_batches(
                        eval_dataset,
                        model_id=runtime['model_id'],
                        batch_size=eval_batch_size,
                        template_cls=template_cls,
                        enable_thinking=enable_thinking,
                        full_batches_only=False,
                    ),
                    'sampling_params': SamplingParams(
                        max_tokens=int(eval_sampling.get('max_tokens', rollout['max_tokens'])),
                        temperature=float(eval_sampling.get('temperature', 0.0)),
                        top_p=float(eval_sampling.get('top_p', 1.0)),
                        repetition_penalty=float(eval_sampling.get('repetition_penalty', 1.0)),
                        logprobs=0,
                        num_samples=1,
                    ),
                    'reward_fn': _reward_for_context(
                        eval_dataset.get('reward'),
                        context_key=f'{context.key} evaluation',
                    ),
                }

        sampler_engine_args = {
            'tensor_parallel_size': sampler_tp,
            'enable_lora': True,
            'max_loras': int(runtime['sampler_max_loras']),
            'max_lora_rank': lora_data['r'],
            'max_model_len': int(sampler_config['max_model_len']),
            'gpu_memory_utilization': float(sampler_config['gpu_memory_utilization']),
            'max_num_seqs': int(sampler_config['max_num_seqs']),
            'enforce_eager': bool(sampler_config['enforce_eager']),
        }
        if sampler_config.get('max_num_batched_tokens') is not None:
            sampler_engine_args['max_num_batched_tokens'] = int(sampler_config['max_num_batched_tokens'])
        self.sampler = vLLMSampler(
            model_id=runtime['model_id'],
            remote_group='sampler',
            device_mesh=sampler_mesh,
            engine_args=sampler_engine_args,
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
        self.metrics = create_metrics_reporter(
            raw_config.get('metrics'),
            run_id=str(runtime.get('run_id', 'sync_barrier_multi_lora_grpo')),
        )
        self.completed_partitions = 0
        self._creation_order = 0

    def _record_metric(
        self,
        stage: str,
        *,
        admission: PartitionAdmission | None = None,
        context: LoraContext | None = None,
        values: dict[str, Any] | None = None,
        status: str = 'completed',
        attributes: dict[str, Any] | None = None,
        optimizer_step: int | None = None,
        policy_version: int | None = None,
    ) -> None:
        if self.metrics is None:
            return
        self.metrics.record(MetricRecord(
            stage=stage,
            values=dict(values or {}),
            context_key=(
                admission.context.key if admission is not None
                else context.key if context is not None else None
            ),
            partition_id=admission.partition_id if admission is not None else None,
            partition_index=admission.step if admission is not None else None,
            optimizer_step=optimizer_step,
            policy_version=policy_version,
            status=status,
            attributes=dict(attributes or {}),
        ))

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            while self.max_steps is None or self.completed_partitions < self.max_steps:
                partitions = self._rollout_round()
                if not partitions:
                    break
                self._advantage_round(partitions)
                self._train_round(partitions)
        except Exception as exc:
            self._record_metric(
                'run',
                status='failed',
                values={'wall_time_s': time.perf_counter() - started},
                attributes={'error': f'{type(exc).__name__}: {exc}'},
            )
            if self.metrics is not None:
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
        self._record_metric(
            'run',
            values={
                'trained_partitions': result['trained_partitions'],
                'wall_time_s': result['wall_time_s'],
            },
        )
        if self.metrics is not None:
            self.metrics.flush()
            result['metrics_health'] = self.metrics.health()
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
            self._record_metric(
                'rollout',
                admission=admission,
                status='submitted',
                policy_version=state.policy_version,
                values={
                    'prompt_count': admission.target_groups,
                    'sample_count': admission.sample_count,
                    'num_generations': admission.num_generations,
                },
                attributes={'scope': 'partition'},
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
            rows = sample_responses_to_rollout_rows(
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
            self._record_rollout_groups(state, admission, rows, rewards)
            self._record_metric(
                'rollout',
                admission=admission,
                policy_version=state.policy_version,
                values=rollout_metrics(
                    completion_lengths=[int(row['completion_length']) for row in rows],
                    stop_reasons=[row.get('stop_reason') for row in rows],
                    rollout_latency_s=rollout_latency_s,
                ),
                attributes={'scope': 'partition'},
            )
            partitions.append(SyncPartition(admission, state, rows, rewards))
            state.partition_step += 1
        return partitions

    def _record_rollout_groups(
        self,
        state: SyncContextState,
        admission: PartitionAdmission,
        rows: list[dict[str, Any]],
        rewards: list[float],
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
                **rollout_metrics(
                    rewards={'reward': group_rewards},
                    completion_lengths=[int(row['completion_length']) for row in group_rows],
                    stop_reasons=[row.get('stop_reason') for row in group_rows],
                ),
            }
            self._record_metric(
                'rollout',
                admission=admission,
                policy_version=state.policy_version,
                values=metrics,
                attributes={
                    'scope': 'group',
                    'group_id': f'{admission.partition_id}/group_{group_index}',
                },
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
            samples_per_batch = admission.num_generations
            for start in range(0, len(partition.rows), samples_per_batch):
                end = min(start + samples_per_batch, len(partition.rows))
                self._record_metric(
                    'advantage',
                    admission=admission,
                    policy_version=partition.state.policy_version,
                    values={
                        'sample_count': end - start,
                        **advantage_signal_metrics(
                            partition.rewards[start:end],
                            partition.advantages[start:end],
                            num_generations=admission.num_generations,
                        ),
                    },
                )

    def _train_round(self, partitions: list[SyncPartition]) -> None:
        for partition in partitions:
            admission = partition.admission
            state = partition.state
            assert partition.advantages is not None
            samples_per_batch = state.mini_batch_size
            for start in range(0, len(partition.rows), samples_per_batch):
                end = start + samples_per_batch
                batch = self._training_batch(
                    partition.rows[start:end],
                    partition.rewards[start:end],
                    partition.advantages[start:end],
                )
                train_started = time.perf_counter()
                metrics = _train_batch(
                    self.model,
                    self.train_batch_configs,
                    batch,
                    admission,
                    model_data_parallel_size=self.model_data_parallel_size,
                )
                state.optimizer_steps += 1
                metrics.update({
                    'sample_count': end - start,
                    'train_latency_s': time.perf_counter() - train_started,
                    'policy_version_gap_mean': 0.0,
                    'policy_version_gap_p95': 0.0,
                    'policy_version_gap_max': 0,
                    'rollout_policy_span_mean': 0.0,
                    'rollout_policy_span_max': 0,
                })
                self._record_metric(
                    'train',
                    admission=admission,
                    optimizer_step=state.optimizer_steps,
                    policy_version=state.policy_version,
                    values=metrics,
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
            self._record_metric(
                'policy',
                admission=admission,
                optimizer_step=state.optimizer_steps,
                policy_version=state.policy_version,
                values={
                    'adapter_save_latency_s': adapter_save_latency_s,
                    'policy_publish_latency_s': policy_publish_latency_s,
                },
                attributes={'operation': 'publish', 'adapter_path': state.adapter_path},
            )
            self._evaluate_policy(state, admission)
            prune_started = time.perf_counter()
            self._prune_adapter_history(state)
            adapter_prune_latency_s = time.perf_counter() - prune_started
            self.completed_partitions += 1
            self._record_metric(
                'partition',
                admission=admission,
                optimizer_step=state.optimizer_steps,
                policy_version=state.policy_version,
                values={
                    'adapter_save_latency_s': adapter_save_latency_s,
                    'policy_publish_latency_s': policy_publish_latency_s,
                    'adapter_prune_latency_s': adapter_prune_latency_s,
                    'partition_finalize_latency_s': time.perf_counter() - finalize_started,
                },
            )

    def _evaluate_policy(self, state: SyncContextState, admission: PartitionAdmission) -> None:
        config = self.evaluation_configs.get(state.context.key)
        if config is None or state.policy_version % int(config['interval']):
            return
        if state.context.key not in self._evaluation_batches:
            self._evaluation_batches[state.context.key] = list(config['prompt_batches'])

        started = time.perf_counter()
        rewards: list[float] = []
        completion_lengths: list[int] = []
        prompt_count = 0
        for batch in self._evaluation_batches[state.context.key]:
            prompts = list(batch)
            responses = self.sampler.sample(
                prompts,
                config['sampling_params'],
                adapter_name=state.context.adapter_name,
                adapter_path=state.adapter_path,
            )
            rows = sample_responses_to_rollout_rows(
                prompts,
                responses,
                policy_version=state.policy_version,
            )
            rewards.extend(float(value) for value in config['reward_fn'](rows, context=state.context))
            completion_lengths.extend(int(row['completion_length']) for row in rows)
            prompt_count += len(prompts)
        if not rewards:
            raise ValueError(f'evaluation dataset is empty for {state.context.key}')
        self._record_metric(
            'evaluation',
            admission=admission,
            optimizer_step=state.optimizer_steps,
            policy_version=state.policy_version,
            values={
                'accuracy': sum(rewards) / len(rewards),
                'sample_count': len(rewards),
                'prompt_count': prompt_count,
                'completion_length': sum(completion_lengths) / len(completion_lengths),
                'eval_latency_s': time.perf_counter() - started,
            },
            attributes={'eval_dataset': config['dataset_name']},
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
            self.sampler.unload_adapter_paths([path])
            if os.path.isdir(path):
                shutil.rmtree(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='cookbook/rl/async_rl/compare_multi_lora_gsm8k_sync.yaml',
    )
    args = parser.parse_args()
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    print(SyncBarrierMultiLoraGRPO(config).run())


if __name__ == '__main__':
    main()

# MODEL_ID=/path/to/model \
# TENANT_A_DATASET_ID=/path/to/tenant_a_gsm8k \
# TENANT_B_DATASET_ID=/path/to/tenant_b_gsm8k \
# python cookbook/rl/async_rl/sync_barrier_multi_lora_grpo.py
