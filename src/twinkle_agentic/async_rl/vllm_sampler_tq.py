# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import asyncio
import os
import uuid
from concurrent.futures import Future
from copy import copy
from dataclasses import dataclass
from queue import Empty, SimpleQueue
from typing import Any

from twinkle import DeviceMesh, get_logger, remote_class, remote_function
from twinkle.data_format import SampledSequence, SampleResponse, SamplingParams
from twinkle.sampler.vllm_sampler import vLLMSampler
from .data_plane import TQDataPlane
from .types import LoraContext, MetricEvent, PromptGroup, RolloutOutput, RolloutPolicy

logger = get_logger()


def _extract_sampled_token_logps(logprobs: Any) -> list[float]:
    return [0.0 if not item else float(item[0][1]) for item in logprobs or []]


def _sample_responses_to_rollout_rows(
    sources: list[dict[str, Any]],
    responses: list[SampleResponse],
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
            row['logprobs'] = _extract_sampled_token_logps(sequence.logprobs)
            row['stop_reason'] = sequence.stop_reason
            row['completion_length'] = len(sequence.tokens)
            row['rollout_policy_version'] = policy_version
            rows.append(row)
    return rows


def _compute_rewards(
    reward_registry: dict[str, Any],
    context: LoraContext,
    rollout_rows: list[RolloutOutput],
) -> list[float] | None:
    reward_fn = reward_registry.get(context.key) or reward_registry.get(context.reward_type)
    if reward_fn is None:
        return None
    return list(reward_fn(rollout_rows, context=context))


def _compute_reward_metrics(
    reward_registry: dict[str, Any],
    context: LoraContext,
    rollout_rows: list[RolloutOutput],
    rewards: list[float],
) -> dict[str, Any]:
    reward_fn = reward_registry.get(context.key) or reward_registry.get(context.reward_type)
    metric_payload = getattr(reward_fn, 'metric_payload', None)
    if metric_payload is None:
        return {}
    return dict(metric_payload(rollout_rows, rewards=rewards, context=context))


@dataclass(frozen=True)
class _GeneratedSample:
    response: SampleResponse
    policies: tuple[RolloutPolicy, ...]
    attempts: int
    was_aborted: bool
    resumed_partial_output: bool

    @property
    def initial_policy(self) -> RolloutPolicy:
        return self.policies[0]

    @property
    def final_policy(self) -> RolloutPolicy:
        return self.policies[-1]

    @property
    def retry_count(self) -> int:
        return self.attempts - 1


def resolve_adapter_path(adapter_path: str) -> str:
    path = os.path.abspath(os.path.expanduser(str(adapter_path)))
    if not os.path.exists(path):
        raise FileNotFoundError(f'local LoRA adapter path does not exist: {path}')
    return path


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
        context_manager: Any,
        reward_registry: dict[str, Any] | None = None,
        rollout_max_retries: int = 2,
        rollout_retry_delay_s: float = 0.5,
        **kwargs,
    ):
        self.context_manager = context_manager
        super().__init__(model_id=model_id, engine_args=engine_args, device_mesh=device_mesh, **kwargs)
        self.data_plane = TQDataPlane()
        self.reward_registry = dict(reward_registry or {})
        self.rollout_max_retries = int(rollout_max_retries)
        self.rollout_retry_delay_s = float(rollout_retry_delay_s)
        if self.rollout_max_retries < 0:
            raise ValueError(f'rollout_max_retries must be non-negative, got {self.rollout_max_retries}')
        if self.rollout_retry_delay_s < 0:
            raise ValueError(f'rollout_retry_delay_s must be non-negative, got {self.rollout_retry_delay_s}')
        self._background_submissions: dict[str, Future] = {}
        self._metrics_events: SimpleQueue[MetricEvent] = SimpleQueue()
        self._failure: str | None = None

    async def _emit(self, event: str, context: LoraContext, partition_id: str, metrics: dict[str, Any]) -> None:
        self._metrics_events.put(
            MetricEvent(event, context, partition_id, dict(metrics), metrics.get('policy_version')))

    @remote_function(dispatch='all', collect='flatten', lazy_collect=False)
    def drain_metrics(self) -> list[MetricEvent]:
        events = []
        while True:
            try:
                events.append(self._metrics_events.get_nowait())
            except Empty:
                break
        return events

    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def check_health(self) -> None:
        if self._failure is not None:
            raise RuntimeError(self._failure)

    @remote_function(dispatch='slice_dp', collect='none', lazy_collect=False)
    def sample(
        self,
        groups: list[PromptGroup],
        sampling_params: SamplingParams,
        allow_partial_rollout: bool = False,
    ) -> dict[str, Any]:
        """Schedule this DP worker's complete prompt groups and return immediately."""
        submission_id = str(uuid.uuid4())
        future = self._submit_in_loop(
            self._sample_prompt_groups(
                submission_id,
                groups,
                sampling_params,
                bool(allow_partial_rollout),
            ))
        self._background_submissions[submission_id] = future
        future.add_done_callback(self._on_submission_done(submission_id))
        return {
            'submission_id': submission_id,
            'submitted_prompt_groups': len(groups),
            'submitted_samples': sum(group.num_samples for group in groups),
        }

    def _submit_in_loop(self, coro) -> Future:
        return asyncio.run_coroutine_threadsafe(coro, self._async_loop)

    def _on_submission_done(self, submission_id: str):

        def callback(future: Future) -> None:
            self._background_submissions.pop(submission_id, None)
            error = future.exception()
            if error is not None:
                self._failure = f'{type(error).__name__}: {error}'
                logger.warning('VLLMSamplerTQ background submission failed: submission=%s error=%s', submission_id,
                               error)

        return callback

    async def _sample_prompt_groups(
        self,
        submission_id: str,
        groups: list[PromptGroup],
        sampling_params: SamplingParams,
        allow_partial_rollout: bool,
    ) -> None:
        results = await asyncio.gather(
            *(self._run_prompt_group(
                submission_id=submission_id,
                group=group,
                sampling_params=sampling_params,
                allow_partial_rollout=allow_partial_rollout,
            ) for group in groups),
            return_exceptions=True)
        failed_group = next(
            ((group, result) for group, result in zip(groups, results) if isinstance(result, Exception)), None)
        if failed_group is not None:
            group, error = failed_group
            await self._emit(
                'rollout_failed',
                group.context,
                group.partition_id,
                {
                    'group_id': group.group_id,
                    'error': str(error)
                },
            )
            raise RuntimeError(f'rollout failed for {group.group_id}: {error}') from error

    async def _run_prompt_group(
        self,
        *,
        submission_id: str,
        group: PromptGroup,
        sampling_params: SamplingParams,
        allow_partial_rollout: bool,
    ) -> None:
        """Sample all generations for one group, then write that group once."""
        started = asyncio.get_running_loop().time()
        num_generations = group.num_samples
        sources = [{
            **group.prompt, 'group_id': group.group_id,
            'generation_idx': generation_idx
        } for generation_idx in range(num_generations)]
        generated_samples = await self._generate_group_samples(
            group.context, sources, sampling_params, allow_partial_rollout=allow_partial_rollout)
        rows = []
        for source, generated in zip(sources, generated_samples):
            sample_rows = _sample_responses_to_rollout_rows([source], [generated.response],
                                                            policy_version=generated.final_policy.version)
            if len(sample_rows) != 1:
                raise ValueError(f'generation {source["generation_idx"]} produced {len(sample_rows)} samples')
            row = sample_rows[0]
            versions = [policy.version for policy in generated.policies]
            row.update({
                'rollout_policy_version': generated.final_policy.version,
                'rollout_adapter_path': generated.final_policy.adapter_path,
                'rollout_policy_versions': versions,
                'initial_policy_version': versions[0],
                'final_policy_version': versions[-1],
                'policy_version_span': max(versions) - min(versions),
            })
            rows.append(row)
        if len(rows) != num_generations:
            raise ValueError(f'group {group.group_id} expected {num_generations} rollout samples, got {len(rows)}')

        rewards = _compute_rewards(self.reward_registry, group.context, rows)
        if rewards is None:
            raise ValueError(f'no reward function registered for context {group.context.key}')
        reward_metrics = _compute_reward_metrics(self.reward_registry, group.context, rows, rewards)
        await self.data_plane.complete_rollout_group(
            group,
            rollout_rows=rows,
            rewards=rewards,
            submission_id=submission_id,
            tag_metrics=reward_metrics,
        )

        completion_lengths = [int(row.get('completion_length', 0)) for row in rows]
        output_tokens = sum(completion_lengths)
        rollout_latency_s = asyncio.get_running_loop().time() - started
        policy_versions = [policy.version for sample in generated_samples for policy in sample.policies]
        ordered_completion_lengths = sorted(completion_lengths)
        completion_p95_index = max(0, (95 * len(ordered_completion_lengths) + 99) // 100 - 1)
        await self._emit(
            'rollout_done',
            group.context,
            group.partition_id,
            {
                'group_id': group.group_id,
                'sample_count': len(rows),
                'completion_length_mean': sum(completion_lengths) / len(completion_lengths),
                'completion_length_p95': ordered_completion_lengths[completion_p95_index],
                'rollout_latency_s': rollout_latency_s,
                'output_tokens': output_tokens,
                'output_tokens_per_s': output_tokens / rollout_latency_s,
                'retry_count': sum(sample.retry_count for sample in generated_samples),
                'aborted_sample_count': sum(sample.was_aborted for sample in generated_samples),
                'partial_resumed_sample_count': sum(sample.resumed_partial_output for sample in generated_samples),
                'policy_version': max(policy_versions),
                'policy_version_min': min(policy_versions),
                'policy_version_max': max(policy_versions),
            },
        )

    async def _load_lora_for_policy(self, policy: RolloutPolicy) -> Any:
        """Load the adapter selected for one group's rollout snapshot."""
        if policy.adapter_path is None:
            return None
        local_path = os.path.abspath(await asyncio.to_thread(resolve_adapter_path, policy.adapter_path))
        lora_request = await self.engine._get_or_load_lora(local_path)
        if lora_request is None:
            raise RuntimeError(f'failed to load LoRA adapter from {local_path}')
        return lora_request

    async def _generate_group_samples(
        self,
        context: LoraContext,
        sources: list[dict[str, Any]],
        sampling_params: SamplingParams,
        *,
        allow_partial_rollout: bool,
    ) -> list[_GeneratedSample]:
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
                self.encode_trajectory_for_vllm(source, context.adapter_name, not logprobs_only) for source in sources
            ]
        else:
            encoded_inputs = sources
        tasks = [
            self._generate_sample(
                context,
                feat,
                sampling_params,
                multi_modal_data=multi_modal_data,
                logprobs_only=logprobs_only,
                allow_partial_rollout=allow_partial_rollout,
            ) for feat, multi_modal_data in zip(encoded_inputs, multi_modal_data_list)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        failures = [result for result in results if isinstance(result, Exception)]
        if failures:
            raise failures[0]
        return results

    async def _generate_sample(
        self,
        context: LoraContext,
        original_input: dict[str, Any],
        sampling_params: SamplingParams,
        *,
        multi_modal_data: dict[str, Any] | None,
        logprobs_only: bool,
        allow_partial_rollout: bool,
    ) -> _GeneratedSample:
        current_input = original_input
        partial_responses: list[SampleResponse] = []
        partial_policies: list[RolloutPolicy] = []
        generated_tokens = 0
        last_error: Exception | None = None
        was_aborted = False
        resumed_partial_output = False

        for attempt in range(self.rollout_max_retries + 1):
            policy = await self.context_manager.get_rollout_policy.remote(context)
            attempt_params = copy(sampling_params)
            if allow_partial_rollout and attempt_params.max_tokens is not None:
                attempt_params.max_tokens -= generated_tokens
            try:
                response = await self._sample_single(
                    current_input,
                    attempt_params,
                    lora_request=await self._load_lora_for_policy(policy),
                    multi_modal_data=multi_modal_data,
                    logprobs_only=logprobs_only,
                )
                sequence = response.sequences[0]
            except Exception as exc:
                last_error = exc
            else:
                if sequence.stop_reason not in {'abort', 'error'}:
                    if not allow_partial_rollout or not partial_responses:
                        return _GeneratedSample(response, (policy, ), attempt + 1, was_aborted, resumed_partial_output)
                    partial_responses.append(response)
                    partial_policies.append(policy)
                    return _GeneratedSample(
                        self._merge_partial_responses(partial_responses), tuple(partial_policies), attempt + 1,
                        was_aborted, resumed_partial_output)

                last_error = RuntimeError(f'generation stopped with {sequence.stop_reason}')
                was_aborted = was_aborted or sequence.stop_reason == 'abort'
                if allow_partial_rollout and sequence.tokens:
                    resumed_partial_output = True
                    partial_responses.append(response)
                    partial_policies.append(policy)
                    generated_tokens += len(sequence.tokens)
                    current_input = sequence.new_input_feature
                    if sampling_params.max_tokens is not None and generated_tokens >= sampling_params.max_tokens:
                        return _GeneratedSample(
                            self._merge_partial_responses(partial_responses, stop_reason='length'),
                            tuple(partial_policies),
                            attempt + 1,
                            was_aborted,
                            resumed_partial_output,
                        )
                elif not allow_partial_rollout:
                    current_input = original_input

            if attempt < self.rollout_max_retries:
                await asyncio.sleep(self.rollout_retry_delay_s)

        error_detail = f'{type(last_error).__name__}: {last_error}'
        error = RuntimeError(
            f'generation failed after {self.rollout_max_retries + 1} attempts; last error: {error_detail}')
        raise error from last_error

    def _merge_partial_responses(
        self,
        responses: list[SampleResponse],
        *,
        stop_reason: str | None = None,
    ) -> SampleResponse:
        sequences = [response.sequences[0] for response in responses]
        tokens = [token for sequence in sequences for token in sequence.tokens]
        logprobs = [logprob for sequence in sequences for logprob in (sequence.logprobs or [])]
        final_sequence = sequences[-1]
        return SampleResponse(
            prompt_token_ids=responses[0].prompt_token_ids,
            sequences=[
                SampledSequence(
                    stop_reason=stop_reason or final_sequence.stop_reason,
                    tokens=tokens,
                    logprobs=logprobs,
                    decoded=self.template.decode(tokens),
                    new_input_feature=final_sequence.new_input_feature,
                    routed_experts=final_sequence.routed_experts,
                )
            ],
        )
