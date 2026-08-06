"""Client-orchestrated async GRPO built from the low-level component APIs."""
from __future__ import annotations

import asyncio
import inspect
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from peft import LoraConfig

from twinkle.advantage import GRPOAdvantage
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.preprocessor.llm import GSM8KProcessor
from twinkle.reward import GSM8KAccuracyReward
from twinkle_client import DataPlaneClient, init_twinkle_client
from twinkle_client.async_rl import Worker, WorkerPipeline
from twinkle_client.common.serialize import json_safe
from twinkle_client.model import MultiLoraTransformersModel
from twinkle_client.sampler import vLLMSampler

BASE_MODEL = os.environ.get('TWINKLE_MODEL_ID', 'Qwen/Qwen3.5-4B')
MODEL_ID = f'ms://{BASE_MODEL}'
ADAPTER_NAME = os.environ.get('TWINKLE_ADAPTER_NAME', 'client-grpo')
MAX_PARTITIONS = int(os.environ.get('TWINKLE_MAX_PARTITIONS', '100'))
MAX_STALENESS = int(os.environ.get('TWINKLE_MAX_STALENESS', '2'))
ROLLOUT_CONCURRENCY = int(os.environ.get('TWINKLE_ROLLOUT_CONCURRENCY', '8'))
NUM_GENERATIONS = int(os.environ.get('TWINKLE_NUM_GENERATIONS', '4'))
BATCH_SIZE = int(os.environ.get('TWINKLE_BATCH_SIZE', '8'))
TRAIN_MINI_BATCH_SIZE = int(os.environ.get('TWINKLE_TRAIN_MINI_BATCH_SIZE', '8'))
MICRO_BATCH_SIZE = int(os.environ.get('TWINKLE_MICRO_BATCH_SIZE', '4'))
MAX_TOKENS_PER_MICRO_BATCH = int(os.environ.get('TWINKLE_MAX_TOKENS_PER_MICRO_BATCH', '4096'))


@dataclass(frozen=True)
class _Policy:
    version: int
    adapter_uri: str


@dataclass
class _RolloutPartition:
    """One DataLoader batch bound to one immutable policy snapshot."""

    partition_id: int
    policy: _Policy
    rollouts: list[asyncio.Task[Any]]
    ready: asyncio.Queue['_ReadyGroup'] = field(default_factory=asyncio.Queue)


@dataclass
class _ReadyGroup:
    group_index: int
    rows: list[dict[str, Any]]
    tags: list[dict[str, Any]]
    ref: Any
    forward_kwargs: dict[str, Any]


@dataclass
class _RolloutResult:
    partition: _RolloutPartition
    group_index: int
    ref: Any


class _GRPOState:

    def __init__(self, policy: _Policy):
        self.policy = policy
        self.live: deque[_RolloutPartition] = deque()
        self.input_done = False
        self.failure: BaseException | None = None
        self.condition = asyncio.Condition()

    async def wait_for_admission(self) -> _Policy:
        async with self.condition:
            await self.condition.wait_for(
                lambda: self.failure is not None or len(self.live) < MAX_STALENESS + 1)
            if self.failure is not None:
                raise self.failure
            return self.policy

    async def add_partition(self, partition: _RolloutPartition) -> None:
        async with self.condition:
            self.live.append(partition)
            self.condition.notify_all()

    async def finish_input(self) -> None:
        async with self.condition:
            self.input_done = True
            self.condition.notify_all()

    async def fail(self, error: BaseException) -> None:
        async with self.condition:
            if self.failure is None:
                self.failure = error
            self.condition.notify_all()

    async def oldest_partition(self) -> _RolloutPartition | None:
        async with self.condition:
            await self.condition.wait_for(
                lambda: self.failure is not None or bool(self.live) or self.input_done)
            if self.failure is not None:
                raise self.failure
            return self.live[0] if self.live else None

    async def publish(self, partition: _RolloutPartition, policy: _Policy) -> None:
        async with self.condition:
            if not self.live or self.live[0] is not partition:
                raise RuntimeError(f'partition {partition.partition_id} attempted out-of-order publication')
            self.policy = policy
            self.live.popleft()
            self.condition.notify_all()


def create_dataset() -> Dataset:
    dataset = Dataset(DatasetMeta('ms://modelscope/gsm8k', subset_name='main', split='train'))
    dataset.set_template('Qwen3_5Template', model_id=MODEL_ID, max_length=2048, enable_thinking=False)
    dataset.map(GSM8KProcessor(system='Put the final answer within \\boxed{}.'))
    dataset.encode(add_generation_prompt=True)
    return dataset


async def rollout_group(
    sampler: vLLMSampler,
    prompt: dict[str, Any],
    policy: _Policy,
    semaphore: asyncio.Semaphore,
    group_id: str,
) -> Any:
    """Submit one GRPO group and keep its sample-level TQ DataRef alive."""
    async with semaphore:
        result = await _submit(
            sampler.submit_sample,
            [prompt],
            adapter_name=ADAPTER_NAME,
            adapter_uri=policy.adapter_uri,
            policy_version=policy.version,
            group_ids=[group_id],
            sampling_params={
                'max_tokens': 1024,
                'temperature': 1.0,
                'top_p': 0.95,
                'logprobs': 1,
            },
            num_samples=NUM_GENERATIONS,
        )
    if not isinstance(result, dict) or not result.get('output_ref'):
        raise RuntimeError('async sampler did not return a DataPlane output_ref')
    from twinkle_client.types import DataRef
    return DataRef(**result['output_ref'])


def start_partition(
    partition_id: int,
    batch: list[dict[str, Any]],
    policy: _Policy,
    sampler: vLLMSampler,
    semaphore: asyncio.Semaphore,
) -> _RolloutPartition:
    """Capture the snapshot before submitting any rollout in this partition."""
    return _RolloutPartition(
        partition_id=partition_id,
        policy=policy,
        rollouts=[
            asyncio.create_task(
                rollout_group(
                    sampler,
                    prompt,
                    policy,
                    semaphore,
                    f'partition-{partition_id}/group-{group_index}',
                ))
            for group_index, prompt in enumerate(json_safe(batch))
        ],
    )


async def _put_rows(data_plane: DataPlaneClient, rows, *, kind: str, tags):
    """Pass native sample tags while remaining friendly to small cookbook fakes."""
    try:
        return await data_plane.aput(rows, kind=kind, tags=tags)
    except TypeError as error:
        if 'tags' not in str(error):
            raise
        return await data_plane.aput(rows, kind=kind)


async def _submit(method, *args, **kwargs):
    if inspect.iscoroutinefunction(method):
        return await method(*args, **kwargs)
    task = await asyncio.to_thread(method, *args, **kwargs)
    if inspect.isawaitable(task):
        return await task
    return task


class _RolloutWorker(Worker):

    def __init__(self, dataloader, sampler, state: _GRPOState, output: asyncio.Queue, semaphore):
        super().__init__('rollout')
        self.dataloader = dataloader
        self.sampler = sampler
        self.state = state
        self.output = output
        self.semaphore = semaphore

    async def _collect(self, partition, group_index, task):
        try:
            ref = await task
            await self.output.put(_RolloutResult(partition, group_index, ref))
        except BaseException as error:
            await self.state.fail(error)
            raise

    async def run(self) -> None:
        batches = iter(self.dataloader)
        collectors: list[asyncio.Task] = []
        try:
            for partition_id in range(MAX_PARTITIONS):
                policy = await self.state.wait_for_admission()
                batch = next(batches, None)
                if batch is None:
                    break
                prompts = batch if isinstance(batch, list) else [batch]
                if len(prompts) != BATCH_SIZE:
                    print(f'dropping incomplete final batch with {len(prompts)} prompts')
                    break
                partition = start_partition(
                    partition_id, prompts, policy, self.sampler, self.semaphore)
                await self.state.add_partition(partition)
                collectors.extend(
                    asyncio.create_task(self._collect(partition, index, task))
                    for index, task in enumerate(partition.rollouts)
                )
            await self.state.finish_input()
            await asyncio.gather(*collectors)
            await self.output.put(None)
        except BaseException as error:
            await self.state.fail(error)
            for task in collectors:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*collectors, return_exceptions=True)
            raise


class _AdvantageWorker(Worker):

    def __init__(self, data_plane, state: _GRPOState, source: asyncio.Queue):
        super().__init__('advantage')
        self.data_plane = data_plane
        self.state = state
        self.source = source

    async def run(self) -> None:
        try:
            while True:
                result = await self.source.get()
                if result is None:
                    return
                group_id = f'partition-{result.partition.partition_id}/group-{result.group_index}'
                try:
                    batch = await self.data_plane.aget_batch(result.ref)
                    if len(batch.rows) != NUM_GENERATIONS:
                        raise RuntimeError(
                            f'group {group_id} expected {NUM_GENERATIONS} generations, '
                            f'got {len(batch.rows)}')
                    features = [row.get('new_input_feature') for row in batch.rows]
                    if not all(isinstance(feature, dict) for feature in features):
                        raise RuntimeError(f'group {group_id} has no trainable new_input_feature')
                    old_logps = [
                        [position[0][1] for position in (row.get('logprobs') or [])]
                        for row in batch.rows
                    ]
                    rewards = await asyncio.to_thread(GSM8KAccuracyReward(), features)
                    advantages = await asyncio.to_thread(
                        GRPOAdvantage(), rewards, num_generations=NUM_GENERATIONS)
                    train_rows = [dict(feature) for feature in features]
                    if batch.tags and len(batch.tags) != len(train_rows):
                        raise RuntimeError(
                            f'group {group_id} returned {len(batch.tags)} tags for '
                            f'{len(train_rows)} rows')
                    source_tags = batch.tags or [{} for _ in train_rows]
                    tags = [{
                        **tag,
                        'record_type': 'sample',
                        'group_id': group_id,
                        'generation_idx': index,
                        'rollout_status': 'ROLLOUT_DONE',
                        'advantage_status': 'ADVANTAGE_DONE',
                        'rollout_policy_version': result.partition.policy.version,
                        'rollout_adapter_uri': result.partition.policy.adapter_uri,
                    } for index, tag in enumerate(source_tags)]
                    ref = await self.data_plane.aappend(result.ref, train_rows, tags=tags)
                    # The physical TQ rows retain rollout fields, while this
                    # reference selects only the trainable InputFeature.
                    train_ref = ref.model_copy(update={'fields': list(train_rows[0])})
                    await result.partition.ready.put(
                        _ReadyGroup(
                            result.group_index,
                            train_rows,
                            tags,
                            train_ref,
                            {
                                'old_logps': json_safe(old_logps),
                                'advantages': json_safe(advantages),
                            },
                        ))
                except BaseException:
                    await self.data_plane.arelease(result.ref)
                    raise
        except BaseException as error:
            await self.state.fail(error)
            raise


class _TrainerWorker(Worker):

    def __init__(self, model, data_plane, state: _GRPOState):
        super().__init__('trainer')
        self.model = model
        self.data_plane = data_plane
        self.state = state
        self.optimizer_step = 0

    async def _train(self, groups: list[_ReadyGroup]) -> None:
        rows = [row for group in groups for row in group.rows]
        tags = [tag for group in groups for tag in group.tags]
        created_batch = len(groups) > 1
        train_ref = (
            await _put_rows(self.data_plane, rows, kind='grpo-train', tags=tags)
            if created_batch else groups[0].ref
        )
        old_logps = [
            value
            for group in groups
            for value in group.forward_kwargs['old_logps']
        ]
        advantages = [
            value
            for group in groups
            for value in group.forward_kwargs['advantages']
        ]
        try:
            await _submit(
                self.model.submit_forward_backward,
                train_ref,
                old_logps=old_logps,
                advantages=advantages,
                dynamic_batching=True,
                micro_batch_size=MICRO_BATCH_SIZE,
                max_tokens_per_micro_batch=MAX_TOKENS_PER_MICRO_BATCH,
            )
            await _submit(self.model.submit_clip_grad_and_step, max_grad_norm=1.0)
            self.optimizer_step += 1
        finally:
            if created_batch:
                await self.data_plane.arelease(train_ref)
            for group in groups:
                await self.data_plane.arelease(group.ref)

    async def run(self) -> None:
        try:
            while True:
                partition = await self.state.oldest_partition()
                if partition is None:
                    return
                staleness = self.state.policy.version - partition.policy.version
                if staleness > MAX_STALENESS:
                    raise RuntimeError(
                        f'partition {partition.partition_id} staleness {staleness} exceeds {MAX_STALENESS}')
                groups_per_step = TRAIN_MINI_BATCH_SIZE // NUM_GENERATIONS
                ready = []
                for _ in range(len(partition.rollouts)):
                    ready.append(await partition.ready.get())
                    if len(ready) == groups_per_step:
                        await self._train(ready)
                        ready.clear()
                if ready:
                    raise RuntimeError('partition ended with an incomplete train mini-batch')
                publish_version = self.state.policy.version + 1
                saved = await _submit(self.model.submit_save, f'policy-{publish_version}')
                policy = _Policy(publish_version, saved['twinkle_path'])
                await self.state.publish(partition, policy)
                print(
                    f'partition={partition.partition_id} policy={policy.version} '
                    f'optimizer_step={self.optimizer_step} staleness={staleness}')
        except BaseException as error:
            await self.state.fail(error)
            raise


async def run_grpo(
    dataloader: DataLoader,
    model: MultiLoraTransformersModel,
    sampler: vLLMSampler,
    data_plane: DataPlaneClient,
) -> None:
    """Overlap rollout partitions while training and publishing them in FIFO order."""
    if MAX_STALENESS < 0:
        raise ValueError('MAX_STALENESS must be non-negative')
    if min(ROLLOUT_CONCURRENCY, NUM_GENERATIONS, BATCH_SIZE, TRAIN_MINI_BATCH_SIZE) <= 0:
        raise ValueError('rollout concurrency and all batch sizes must be positive')
    if TRAIN_MINI_BATCH_SIZE % NUM_GENERATIONS:
        raise ValueError('TRAIN_MINI_BATCH_SIZE must be divisible by NUM_GENERATIONS')
    groups_per_step = TRAIN_MINI_BATCH_SIZE // NUM_GENERATIONS
    if BATCH_SIZE % groups_per_step:
        raise ValueError('BATCH_SIZE * NUM_GENERATIONS must be divisible by TRAIN_MINI_BATCH_SIZE')

    initial = await _submit(model.submit_save, 'policy-0')
    state = _GRPOState(_Policy(version=0, adapter_uri=initial['twinkle_path']))
    semaphore = asyncio.Semaphore(ROLLOUT_CONCURRENCY)
    rollout_results: asyncio.Queue = asyncio.Queue()
    await WorkerPipeline((
        _RolloutWorker(dataloader, sampler, state, rollout_results, semaphore),
        _AdvantageWorker(data_plane, state, rollout_results),
        _TrainerWorker(model, data_plane, state),
    )).run()


async def train() -> None:
    client = init_twinkle_client(
        base_url=os.environ.get('TWINKLE_SERVER_URL', 'http://localhost:8000'),
        api_key=os.environ.get('TWINKLE_SERVER_TOKEN', 'EMPTY_TOKEN'),
    )
    try:
        model = MultiLoraTransformersModel(MODEL_ID)
        sampler = vLLMSampler(MODEL_ID)
        data_plane = DataPlaneClient()

        model.add_adapter_to_model(
            ADAPTER_NAME,
            LoraConfig(target_modules='all-linear', r=8, lora_alpha=32, lora_dropout=0.05),
        )
        model.set_loss('GRPOLoss', epsilon=0.2, beta=0.0)
        model.set_optimizer('AdamW', lr=2e-5)
        model.set_processor('InputProcessor', padding_free=True)
        model.set_template('Qwen3_5Template', model_id=MODEL_ID)
        sampler.set_template('Qwen3_5Template', model_id=MODEL_ID)

        dataloader = DataLoader(dataset=create_dataset(), batch_size=BATCH_SIZE, num_workers=0)
        await run_grpo(dataloader, model, sampler, data_plane)
    finally:
        client.close()


if __name__ == '__main__':
    asyncio.run(train())
