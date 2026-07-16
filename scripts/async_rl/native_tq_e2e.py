#!/usr/bin/env python3
"""Local fake-TQ regression for the domain-oriented async RL pipeline."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import yaml
from pathlib import Path
from typing import Any

from twinkle_agentic.async_rl import (AdvantageWorker, AsyncMultiLoraGRPOConfig, AsyncMultiLoraGRPOPipeline,
                                      ContextSchedulePolicy, JSONLMetricsRecorder, LoraContext, LoraContextManager,
                                      RolloutWorker, SchedulerConfig, TQDataPlane, TrainerWorker)
from twinkle_agentic.async_rl.tq_utils import columns_to_tq_fields


class FakeMeta:

    def __init__(self, indexes, client):
        self.global_indexes, self.client, self.size = list(indexes), client, len(indexes)
        self.custom_meta = [dict(client.tags.get(index, {})) for index in indexes]

    def select_samples(self, indexes):
        return FakeMeta([self.global_indexes[index] for index in indexes], self.client)

    def select_fields(self, _fields):
        return self

    def update_custom_meta(self, values):
        for tag, update in zip(self.custom_meta, values):
            tag.update(update)

    def get_all_custom_meta(self):
        return [dict(tag) for tag in self.custom_meta]


class LocalActorHandle:
    """Explicit local equivalent of a Ray actor handle for this fake test."""

    def __init__(self, target):
        self._target = target

    def __getattr__(self, name):
        method = getattr(self._target, name)

        class RemoteMethod:

            async def remote(_, *args, **kwargs):
                result = method(*args, **kwargs)
                return await result if inspect.isawaitable(result) else result

        return RemoteMethod()


class FakeTQ:

    def __init__(self):
        self.rows, self.tags, self.partitions, self.cursors = {}, {}, {}, {}
        self.next_index = 0

    async def async_put(self, data, metadata=None, partition_id=None):
        columns, size = {name: list(values) for name, values in data.items()}, int(data.batch_size[0])
        if metadata is None:
            indexes = list(range(self.next_index, self.next_index + size))
            self.next_index += size
            self.partitions[partition_id] = indexes
            for index in indexes:
                self.rows[index] = {}
            metadata = FakeMeta(indexes, self)
        else:
            indexes = metadata.global_indexes
        for name, values in columns.items():
            for index, value in zip(indexes, values):
                self.rows[index][name] = value
        return metadata

    async def async_get_meta(self,
                             *,
                             data_fields,
                             batch_size,
                             partition_id,
                             task_name=None,
                             sampling_config=None,
                             **_kwargs):
        indexes = self.partitions.get(partition_id, [])
        key, start = (task_name, partition_id), self.cursors.get((task_name, partition_id), 0)
        group_size, selected = int(sampling_config['n_samples_per_prompt']), []
        for offset in range(start, len(indexes), group_size):
            group = indexes[offset:offset + group_size]
            if len(group) != group_size or not all(field in self.rows[index] for index in group
                                                   for field in data_fields):
                break
            if len(selected) + group_size > batch_size:
                break
            selected.extend(group)
        if selected:
            self.cursors[key] = start + len(selected)
        return FakeMeta(selected, self)

    async def async_get_data(self, metadata):
        keys = set().union(*(self.rows[index] for index in metadata.global_indexes))
        return {name: [self.rows[index][name] for index in metadata.global_indexes] for name in keys}

    async def async_set_custom_meta(self, metadata):
        for index, tag in zip(metadata.global_indexes, metadata.custom_meta):
            self.tags[index] = dict(tag)

    async def async_check_consumption_status(self, task_name, partition_id):
        return self.cursors.get((task_name, partition_id), 0) >= len(self.partitions.get(partition_id, []))

    async def async_reset_consumption(self, partition_id, task_name=None):
        self.cursors[(task_name, partition_id)] = 0

    async def async_clear_partition(self, partition_id):
        for index in self.partitions.pop(partition_id, []):
            self.rows.pop(index, None)
            self.tags.pop(index, None)


class FakeSampler:

    def __init__(self, client, context_manager, loop):
        self.client = client
        self.context_manager = context_manager
        self.loop = loop
        self._submissions = set()

    def sample(self, groups, sampling_params, allow_partial_rollout):
        future = asyncio.run_coroutine_threadsafe(self._sample(groups), self.loop)
        self._submissions.add(future)
        future.add_done_callback(self._submissions.discard)
        return {
            'submitted_prompt_groups': len(groups),
            'submitted_samples': sum(group.num_samples for group in groups),
        }

    async def _sample(self, groups):
        for group in groups:
            policy = await self.context_manager.get_rollout_policy.remote(group.context)
            ids = group.storage.global_indexes
            rewards = [float((index + 1) % 3) / 2 for index in ids]
            rollout_rows = [{
                **group.prompt,
                'generation_idx': generation_idx,
                'logprobs': [-0.1 * (index + 1)] * len(group.prompt['labels']),
                'completion_length': index % 4 + 1,
                'rollout_policy_version': policy.version,
                'rollout_policy_versions': [policy.version],
                'initial_policy_version': policy.version,
                'final_policy_version': policy.version,
                'policy_version_span': 0,
                'rollout_adapter_path': policy.adapter_path,
            } for generation_idx, index in enumerate(ids)]
            await TQDataPlane(self.client).complete_rollout_group(
                group,
                rollout_rows=rollout_rows,
                rewards=rewards,
                submission_id='fake',
            )

    def drain_metrics(self):
        return []


def prompt_batches(count, batch_size):
    prompts = [{'input_ids': [index, 1], 'labels': [index, 1], 'attention_mask': [1, 1]} for index in range(count)]
    return [prompts[index:index + batch_size] for index in range(0, len(prompts), batch_size)]


async def run(path: Path):
    config, client = yaml.safe_load(path.read_text()), FakeTQ()
    manager = LocalActorHandle(LoraContextManager(max_staleness=config['runtime']['max_staleness']))
    data_plane, recorder = TQDataPlane(client), JSONLMetricsRecorder(
        Path(config['runtime']['output_dir']) / 'metrics.jsonl', run_id='native_tq_smoke', mode='fake')
    sources, rollout_config = {}, {}
    advantage_groups_per_batch, train_groups_per_batch = {}, {}
    for item in config['contexts']:
        context = LoraContext(item['tenant_id'], item['training_run_id'], config['runtime']['model_id'],
                              item['adapter_name'])
        await manager.register_context.remote(context)
        key, rollout = context.key, item['rollout']
        sources[key] = prompt_batches(item['dataset']['prompt_count'], rollout['batch_size'])
        rollout_config[key] = {
            'context': context,
            'batch_size': rollout['batch_size'],
            'num_generations': rollout['num_generations'],
            'sampling_params': {},
        }
        advantage_groups_per_batch[key] = item['advantage']['groups_per_batch']
        train_groups_per_batch[key] = item['train']['groups_per_batch']
    rollout = LocalActorHandle(
        RolloutWorker(
            context_manager=manager,
            data_plane=data_plane,
            sampler=FakeSampler(client, manager, asyncio.get_running_loop()),
            prompt_batches=sources,
            rollout_config=rollout_config,
            scheduler=SchedulerConfig(ContextSchedulePolicy.ROUND_ROBIN, 1),
            idle_delay_s=.001))
    advantage = LocalActorHandle(
        AdvantageWorker(
            context_manager=manager,
            data_plane=data_plane,
            advantage_fn=lambda data, _lease:
            ([float(x) - sum(map(float, data['rewards'])) / len(data['rewards'])
              for x in data['rewards']], list(map(float, data['rewards']))),
            groups_per_batch=advantage_groups_per_batch,
            scheduler=SchedulerConfig(ContextSchedulePolicy.OLDEST_PARTITION, 1),
            idle_delay_s=.001))
    trainer = LocalActorHandle(
        TrainerWorker(
            context_manager=manager,
            data_plane=data_plane,
            train_fn=lambda data, _admission: {
                'reward': sum(map(float, data['rewards'])) / len(data['rewards']),
                'loss': sum(abs(float(x)) for x in data['advantages']) / len(data['advantages'])
            },
            save_adapter=lambda admission: f'fake/{admission.context.adapter_name}/v{admission.step + 1}',
            groups_per_batch=train_groups_per_batch,
            scheduler=SchedulerConfig(ContextSchedulePolicy.STICKY, None),
            idle_delay_s=.001))
    result = await AsyncMultiLoraGRPOPipeline(
        context_manager=manager,
        rollout_worker=rollout,
        advantage_worker=advantage,
        trainer_worker=trainer,
        metrics=recorder,
        config=AsyncMultiLoraGRPOConfig(metrics_drain_interval_s=.001)).run_async()
    result['remaining_partitions'] = len(client.partitions)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='cookbook/rl/async_native_multi_lora_grpo.yaml')
    result = asyncio.run(run(Path(parser.parse_args().config)))
    if result['remaining_partitions']:
        raise SystemExit('uncleared fake TQ partitions')


if __name__ == '__main__':
    main()
