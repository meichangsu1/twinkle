# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import asyncio
import os
import uuid
from concurrent.futures import Future
from copy import copy
from typing import Any, Iterable

from twinkle import DeviceMesh, get_logger, remote_class, remote_function
from twinkle.data_format import SamplingParams, Trajectory
from twinkle.hub import HubOperation
from twinkle.sampler.vllm_sampler import vLLMSampler

from .data_plane import TransferQueueDataPlane, TransferQueueRuntimeConfig
from .tq_utils import ROLLOUT_TRAIN_FIELDS, rows_to_tq_fields
from .types import LoraContext, PromptGroupMeta, PromptGroupRef, PromptGroupStatus, RolloutOutput

logger = get_logger()


def resolve_adapter_path(adapter_path: str) -> str:
    path = os.path.expanduser(str(adapter_path))
    if os.path.exists(path):
        return os.path.abspath(path)
    if os.path.isabs(path):
        raise FileNotFoundError(f'local LoRA adapter path does not exist: {path}')
    return HubOperation.download_model(model_id_or_path=adapter_path)


class TQSamplerRollout:
    """Async RL rollout adapter that submits fire-and-forget TQ sampler requests."""

    def __init__(self, sampler: Any, *, sampling_params: SamplingParams, default_num_generations: int | None = None):
        self.sampler = sampler
        self.sampling_params = sampling_params
        self.default_num_generations = default_num_generations

    def __call__(self, trajectories: list[Trajectory], **kwargs) -> dict[str, Any]:
        raw_num_generations = kwargs['num_generations'] if 'num_generations' in kwargs else self.default_num_generations
        num_generations = int(raw_num_generations or 0)
        assert num_generations > 0, f'num_generations must be passed to rollout, got {num_generations}'
        request = {
            'context': kwargs['context'],
            'partition_id': kwargs['partition_id'],
            'group_refs': list(kwargs['group_refs']),
            'groups': list(kwargs['groups']),
            'prompt_groups': [dict(item) for item in trajectories],
            'num_generations': num_generations,
            'rollout_policy_version': kwargs['policy_version'],
            'adapter_name': kwargs.get('adapter_name', ''),
            'adapter_path': kwargs.get('adapter_path'),
            'sampling_params': self.sampling_params,
        }
        return self.sampler.sample(request)


@remote_class()
class VLLMSamplerTQ(vLLMSampler):
    """vLLM sampler that writes async RL rollout results directly to TransferQueue.

    ``sample()`` is intentionally fire-and-forget: it schedules generation work
    on the sampler actor's vLLM event loop and returns submission metadata
    without waiting for any prompt group to finish.
    """

    def __init__(
        self,
        model_id: str,
        engine_args: dict[str, Any] | None = None,
        device_mesh: DeviceMesh | None = None,
        *,
        tq_config: TransferQueueRuntimeConfig | None = None,
        tq_client: Any | None = None,
        reward_registry: dict[str, Any] | None = None,
        **kwargs,
    ):
        _assert_single_sampler_dp(device_mesh)
        super().__init__(model_id=model_id, engine_args=engine_args, device_mesh=device_mesh, **kwargs)
        self.data_plane = TransferQueueDataPlane(tq_client=tq_client, tq_config=tq_config)
        self.reward_registry = dict(reward_registry or {})
        self._background_submissions: dict[str, Future] = {}

    @remote_function(execute='first', collect='first', lazy_collect=False)
    def sample(self, request: dict[str, Any]) -> dict[str, Any]:
        normalized = self._normalize_tq_request(request)
        submission_id = str(uuid.uuid4())
        future = self._submit_in_loop(self._run_submission(submission_id, normalized))
        self._background_submissions[submission_id] = future
        future.add_done_callback(self._on_submission_done(submission_id))
        return {
            'submission_id': submission_id,
            'submitted_prompt_groups': len(normalized['group_refs']),
            'submitted_samples': len(normalized['group_refs']) * int(normalized['num_generations']),
        }

    def _submit_in_loop(self, coro) -> Future:
        return asyncio.run_coroutine_threadsafe(coro, self._async_loop)

    def _on_submission_done(self, submission_id: str):

        def callback(future: Future) -> None:
            self._background_submissions.pop(submission_id, None)
            try:
                future.result()
            except Exception as exc:
                logger.warning('VLLMSamplerTQ background submission failed: submission=%s error=%s',
                               submission_id, exc)

        return callback

    def _normalize_tq_request(self, request: dict[str, Any]) -> dict[str, Any]:
        required = {
            'context',
            'partition_id',
            'group_refs',
            'groups',
            'prompt_groups',
            'num_generations',
            'rollout_policy_version',
            'sampling_params',
        }
        missing = sorted(required - set(request))
        if missing:
            raise ValueError(f'VLLMSamplerTQ request missing required fields: {missing}')
        context = request['context']
        if not isinstance(context, LoraContext):
            raise TypeError(f'VLLMSamplerTQ request context must be LoraContext, got {type(context)!r}')
        group_refs = list(request['group_refs'])
        groups = list(request['groups'])
        prompt_groups = [dict(item) for item in request['prompt_groups']]
        if not group_refs:
            raise ValueError('VLLMSamplerTQ request group_refs must not be empty')
        if len(group_refs) != len(groups) or len(group_refs) != len(prompt_groups):
            raise ValueError(f'VLLMSamplerTQ request group_refs/groups/prompt_groups size mismatch: '
                             f'{len(group_refs)} != {len(groups)} != {len(prompt_groups)}')
        if not all(isinstance(group_ref, PromptGroupRef) for group_ref in group_refs):
            bad = [type(group_ref).__name__ for group_ref in group_refs if not isinstance(group_ref, PromptGroupRef)]
            raise TypeError(f'VLLMSamplerTQ request group_refs must be PromptGroupRef objects, got {bad}')
        if not all(isinstance(group, PromptGroupMeta) for group in groups):
            bad = [type(group).__name__ for group in groups if not isinstance(group, PromptGroupMeta)]
            raise TypeError(f'VLLMSamplerTQ request groups must be PromptGroupMeta objects, got {bad}')
        partition_id = str(request['partition_id'])
        if any(group_ref.partition_id != partition_id for group_ref in group_refs):
            raise ValueError(f'VLLMSamplerTQ request groups must all belong to partition {partition_id!r}')
        for group_ref, group in zip(group_refs, groups):
            if group.partition_id != group_ref.partition_id or group.group_id != group_ref.group_id:
                raise ValueError(f'VLLMSamplerTQ request group metadata/ref mismatch: '
                                 f'{group.partition_id}/{group.group_id} != '
                                 f'{group_ref.partition_id}/{group_ref.group_id}')
            if group.context.key != context.key:
                raise ValueError(f'VLLMSamplerTQ request group {group.group_id} belongs to '
                                 f'{group.context.key}, not {context.key}')
        num_generations = int(request['num_generations'])
        if num_generations <= 0:
            raise ValueError(f'num_generations must be positive, got {num_generations}')
        sampling_params = request['sampling_params']
        if isinstance(sampling_params, dict):
            sampling_params = SamplingParams.from_dict(sampling_params)
        if not isinstance(sampling_params, SamplingParams):
            raise TypeError(f'sampling_params must be SamplingParams or dict, got {type(sampling_params)!r}')
        if sampling_params.num_samples != 1:
            raise ValueError('VLLMSamplerTQ requires sampling_params.num_samples == 1; '
                             'use request.num_generations for GRPO fan-out')
        return {
            'context': context,
            'partition_id': partition_id,
            'group_refs': group_refs,
            'groups': groups,
            'prompt_groups': prompt_groups,
            'num_generations': num_generations,
            'rollout_policy_version': int(request['rollout_policy_version']),
            'adapter_name': str(request.get('adapter_name') or ''),
            'adapter_path': request.get('adapter_path'),
            'sampling_params': sampling_params,
        }

    async def _run_submission(self, submission_id: str, request: dict[str, Any]) -> None:
        adapter_path = request.get('adapter_path')
        lora_request = None
        try:
            if adapter_path is not None:
                logger.info(f'Loading LoRA from {adapter_path}')
                local_adapter_path = await asyncio.to_thread(resolve_adapter_path, adapter_path)
                lora_request = await self.engine._get_or_load_lora(local_adapter_path)
                if lora_request is None:
                    logger.warning(f'Failed to pre-load LoRA from {local_adapter_path}, sampling will proceed without LoRA')
        except Exception as exc:
            logger.warning('VLLMSamplerTQ submission failed before group sampling: submission=%s error=%s',
                           submission_id, exc)
            for group_ref, group in zip(request['group_refs'], request['groups']):
                group.status = PromptGroupStatus.FAILED
                await self._mark_group_failed(group=group, submission_id=submission_id, error=str(exc))
            return

        group_tasks = [
            asyncio.create_task(
                self._run_prompt_group(
                    submission_id=submission_id,
                    request=request,
                    group_ref=group_ref,
                    group=group,
                    prompt_group=prompt_group,
                    lora_request=lora_request,
                ))
            for group_ref, group, prompt_group in zip(request['group_refs'], request['groups'], request['prompt_groups'])
        ]
        await asyncio.gather(*group_tasks, return_exceptions=True)

    async def _run_prompt_group(
        self,
        *,
        submission_id: str,
        request: dict[str, Any],
        group_ref: PromptGroupRef,
        group: PromptGroupMeta,
        prompt_group: dict[str, Any],
        lora_request: Any,
    ) -> None:
        context: LoraContext = request['context']
        num_generations = int(request['num_generations'])
        sampling_params: SamplingParams = request['sampling_params']
        adapter_name = str(request.get('adapter_name') or '')
        sources = []
        for generation_idx in range(num_generations):
            source = dict(prompt_group)
            source['group_id'] = group_ref.group_id
            source['generation_idx'] = generation_idx
            sources.append(source)
        try:
            responses = await self._sample_sources(
                sources,
                sampling_params,
                adapter_name=adapter_name,
                lora_request=lora_request,
            )
            rows = sample_responses_to_rollout_rows(
                sources,
                responses,
                policy_version=request['rollout_policy_version'],
            )
            assert len(rows) == num_generations, (
                f'group {group_ref.group_id} expected {num_generations} rollout samples, got {len(rows)}')
            rewards = compute_rewards_for_context(self.reward_registry, context, rows)
            if rewards is None:
                raise ValueError(f'no reward function registered for context {context.key}')
            reward_metrics = compute_reward_metrics_for_context(self.reward_registry, context, rows, rewards)
            sample_keys, sample_fields, sample_tags = build_rollout_group_sample_write(
                group,
                group_ref,
                rows,
                rewards=rewards,
                expected_num_generations=num_generations,
            )
            await self.data_plane.async_write_sample_batch(
                partition_id=group_ref.partition_id,
                keys=sample_keys,
                fields=rows_to_tq_fields(sample_fields),
                tags=sample_tags,
            )
            await self.data_plane.async_mark_prompt_group(
                group,
                PromptGroupStatus.ROLLOUT_DONE,
                sample_keys=sample_keys,
                extra_tag={
                    'submission_id': submission_id,
                    **reward_metrics,
                },
            )
        except Exception as exc:
            logger.warning('VLLMSamplerTQ rollout group failed: partition=%s group=%s error=%s',
                           group_ref.partition_id, group_ref.group_id, exc)
            await self._mark_group_failed(group=group, submission_id=submission_id, error=str(exc))

    async def _sample_sources(
        self,
        sources: list[dict[str, Any]],
        sampling_params: SamplingParams,
        *,
        adapter_name: str,
        lora_request: Any,
    ) -> list[Any]:
        logprobs_only = False
        if sampling_params.max_tokens == 0:
            sampling_params = copy(sampling_params)
            sampling_params.max_tokens = 1
            logprobs_only = True

        is_trajectory = 'input_ids' not in sources[0]
        multi_modal_data_list = [self._extract_multi_modal_data(source) for source in sources]
        if is_trajectory:
            template = self.template
            assert template is not None, 'Use set_template before sampling trajectories'
            encoded_inputs = [
                self.encode_trajectory_for_vllm(source, adapter_name, not logprobs_only) for source in sources
            ]
        else:
            encoded_inputs = sources
        tasks = [
            self._sample_single(
                feat,
                sampling_params,
                lora_request=lora_request,
                multi_modal_data=multi_modal_data,
                logprobs_only=logprobs_only,
            ) for feat, multi_modal_data in zip(encoded_inputs, multi_modal_data_list)
        ]
        return await asyncio.gather(*tasks)

    async def _mark_group_failed(
        self,
        *,
        group: PromptGroupMeta,
        submission_id: str,
        error: str,
    ) -> None:
        await self.data_plane.async_mark_prompt_group(
            group,
            PromptGroupStatus.FAILED,
            extra_tag={
                'submission_id': submission_id,
                'error': error,
            },
        )


def _assert_single_sampler_dp(device_mesh: DeviceMesh | None) -> None:
    if device_mesh is None:
        return
    data_world_size = int(getattr(device_mesh, 'data_world_size', 1) or 1)
    if data_world_size != 1:
        raise ValueError('VLLMSamplerTQ currently supports sampler DP size 1 only. '
                         f'Got sampler data_world_size={data_world_size}. TP is still supported.')


def extract_sampled_token_logps(logprobs: Any) -> list[float]:
    values: list[float] = []
    for item in logprobs or []:
        if not item:
            values.append(0.0)
        else:
            values.append(float(item[0][1]))
    return values


def sample_responses_to_rollout_rows(
    sources: Iterable[Trajectory],
    responses: Iterable[Any],
    *,
    policy_version: int | None,
) -> list[RolloutOutput]:
    rows: list[RolloutOutput] = []
    for source, response in zip(sources, responses):
        for sequence in response.sequences:
            row = dict(source)
            row.update(sequence.new_input_feature or {})
            row.setdefault('group_id', source['group_id'])
            row.setdefault('generation_idx', source['generation_idx'])
            row['logprobs'] = extract_sampled_token_logps(sequence.logprobs)
            row['stop_reason'] = sequence.stop_reason
            row['completion_length'] = len(sequence.tokens)
            row['rollout_policy_version'] = policy_version
            rows.append(row)
    return rows


def compute_rewards_for_context(
    reward_registry: dict[str, Any],
    context: LoraContext,
    rollout_rows: list[RolloutOutput],
) -> list[float] | None:
    reward_fn = reward_registry.get(context.key) or reward_registry.get(context.reward_type)
    if reward_fn is None:
        return None
    return list(reward_fn(rollout_rows, context=context))


def compute_reward_metrics_for_context(
    reward_registry: dict[str, Any],
    context: LoraContext,
    rollout_rows: list[RolloutOutput],
    rewards: list[float] | None,
) -> dict[str, Any]:
    reward_fn = reward_registry.get(context.key) or reward_registry.get(context.reward_type)
    metric_payload = getattr(reward_fn, 'metric_payload', None)
    if metric_payload is None:
        return {}
    return dict(metric_payload(rollout_rows, rewards=rewards, context=context))


def build_rollout_group_sample_write(
    group: PromptGroupMeta,
    group_ref: PromptGroupRef,
    samples: Iterable[RolloutOutput],
    *,
    rewards: list[float] | None = None,
    expected_num_generations: int,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    group_samples = [dict(sample) for sample in samples]
    assert expected_num_generations > 0, f'expected_num_generations must be positive, got {expected_num_generations}'
    assert len(group_samples) == expected_num_generations, (
        f'group {group_ref.group_id} expected {expected_num_generations} rollout samples, '
        f'got {len(group_samples)}')
    if rewards is not None:
        assert len(rewards) == len(group_samples), (
            f'reward count {len(rewards)} does not match sample count {len(group_samples)}')
    if group.partition_id != group_ref.partition_id or group.group_id != group_ref.group_id:
        raise ValueError(f'group metadata/ref mismatch: group={group.partition_id}/{group.group_id}, '
                         f'ref={group_ref.partition_id}/{group_ref.group_id}')
    if group.status not in {PromptGroupStatus.PENDING, PromptGroupStatus.RUNNING}:
        raise ValueError(f'group {group.group_id} is not pending/running: {group.status}')

    sample_keys: list[str] = []
    sample_fields: list[dict[str, Any]] = []
    sample_tags: list[dict[str, Any]] = []
    generation_indices: list[int] = []
    reward_iter = iter(rewards or [])
    for sample_index, trajectory in enumerate(group_samples):
        sample = dict(trajectory)
        if rewards is not None:
            sample['rewards'] = float(next(reward_iter))
        generation_idx = int(sample.get('generation_idx', sample_index))
        key = f'samples/{group_ref.group_id}/{generation_idx}'
        if key in sample_keys:
            raise ValueError(f'duplicate rollout sample key {key!r}')
        generation_indices.append(generation_idx)
        logprobs = require_rollout_logprobs(sample, sample_key=key)
        sample['logprobs'] = logprobs
        fields = rollout_sample_fields(sample)
        tag = sample_tag(
            context=group.context,
            group=group,
            sample=sample,
            sample_key=key,
            generation_idx=generation_idx,
            logprobs=logprobs,
        )
        sample_keys.append(key)
        sample_fields.append(fields)
        sample_tags.append(tag)

    assert generation_indices == list(range(expected_num_generations)), (
        f'group {group.group_id} generation_idx must be 0..{expected_num_generations - 1} in order, '
        f'got {generation_indices}')
    return sample_keys, sample_fields, sample_tags


def require_rollout_logprobs(sample: dict[str, Any], *, sample_key: str) -> list[float]:
    logprobs = sample.get('logprobs')
    if not isinstance(logprobs, list):
        raise TypeError(f'rollout sample {sample_key!r} logprobs must be list[float], got {type(logprobs)!r}')

    values: list[float] = []
    for index, value in enumerate(logprobs):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f'rollout sample {sample_key!r} logprobs[{index}] must be a float, got {type(value)!r}')
        values.append(float(value))

    labels = sample.get('labels')
    if labels is not None:
        trainable_tokens = sum(1 for label in labels if label != -100)
        if len(values) != trainable_tokens:
            raise ValueError(f'rollout sample {sample_key!r} logprobs length must match trainable labels: '
                             f'{len(values)} != {trainable_tokens}')
    return values


def rollout_sample_fields(sample: dict[str, Any]) -> dict[str, Any]:
    return {field_name: sample[field_name] for field_name in ROLLOUT_TRAIN_FIELDS if field_name in sample}


def sample_tag(
    *,
    context: LoraContext,
    group: PromptGroupMeta,
    sample: dict[str, Any],
    sample_key: str,
    generation_idx: int,
    logprobs: list[float],
) -> dict[str, Any]:
    tag = context.metadata()
    tag.update(dict(sample.get('metadata') or {}))
    context.validate_metadata(tag)
    for state_field in ('partition_id', 'partition_status', 'group_status', 'num_samples', 'sample_keys'):
        tag.pop(state_field, None)
    tag.update({
        'record_type': 'sample',
        'sample_status': 'success',
        'sample_id': sample.get('sample_id', sample_key),
        'group_id': group.group_id,
        'generation_idx': generation_idx,
        'rollout_policy_version': group.rollout_policy_version,
        'rollout_adapter_path': group.rollout_adapter_path,
        'logprobs_length': len(logprobs),
    })
    trainable_tokens = _trainable_token_count(sample.get('labels'))
    if trainable_tokens is not None:
        tag['trainable_tokens'] = trainable_tokens
    for sample_field, tag_field in (
        ('input_ids', 'input_length'),
        ('labels', 'label_length'),
        ('attention_mask', 'attention_length'),
    ):
        length = _safe_len(sample.get(sample_field))
        if length is not None:
            tag[tag_field] = length
    for field_name in ('stop_reason', 'truncated', 'turns'):
        if field_name in sample:
            tag[field_name] = sample[field_name]
    return tag


def _trainable_token_count(labels: Any) -> int | None:
    if labels is None:
        return None
    return sum(1 for label in labels if label != -100)


def _safe_len(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return len(value)
    except TypeError:
        return None
