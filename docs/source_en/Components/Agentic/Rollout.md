# Multi-Turn Rollout

The Rollout module provides one multi-turn conversation engine for agentic RLHF training. `MultiTurnRollout` can generate each assistant turn with a local sampler, an OpenAI-compatible API, or a callback that chooses between them.

## Rollout Base Class

```python
from abc import ABC, abstractmethod
from twinkle.data_format import Trajectory

class Rollout(ABC):

    @abstractmethod
    def __call__(self, trajectories: List[Trajectory], **kwargs) -> List[Trajectory]:
        raise NotImplementedError()
```

All rollouts accept a list of trajectories and return the same number of trajectories with additional fields (`messages`, `turns`, `stop_reason`, `truncated`).

## MultiTurnRollout

Multi-turn rollout engine supporting local samplers, external APIs, and per-turn backend selection. Each trajectory runs independently in the rollout thread pool.

### Per-turn Loop

1. Encode each trajectory into an `InputFeature` with a generation prompt
2. Call `response_callback(...)` to obtain one `SampledSequence` from the sampler or API
3. Check termination: `stop_reason == 'length'`, no tool calls, or max turns reached
4. Dispatch tools via `ToolManager`, append tool responses
5. Compute bridge tokens (tool turns + generation prompt) with `labels = -100`
6. Repeat until all trajectories are done

```python
from twinkle_agentic.rollout.multi_turn import MultiTurnRollout
from twinkle_agentic.tools.tool_manager import ToolManager
from twinkle.data_format.sampling import SamplingParams

rollout = MultiTurnRollout(
    sampler=vllm_sampler,
    template=template,
    tool_manager=tool_manager,
    sampling_params=SamplingParams(temperature=0.7, max_tokens=4096),
    max_turns=6,
    max_trajectory_tokens=8192,
    trace_dir='rollout_traces/',
)

# Run rollout
results = rollout(trajectories)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sampler` | Sampler | Local sampler. Used by default when both backends exist. |
| `api` | `API` | Optional external generation API. |
| `template` | `Template` | Required local chat template for encoding every backend's output. |
| `response_callback` | `Callable` | Optional per-turn backend selector returning `SampledSequence`. |
| `api_appended_as` | `str` | API turns are `demonstration` (SFT only) or `context` (no loss). |
| `api_kwargs` | `Dict` | Request fields forwarded to each API call. |
| `tool_manager` | `ToolManager` | Tool dispatcher. Can also be passed per-call. |
| `sampling_params` | `SamplingParams` | Default sampling parameters. |
| `session_affinity` | `bool` | Defaults to True; rollout manages IDs for capable samplers regardless of callback identity. Overridable per call. |
| `max_turns` | `int` | Maximum number of turns per trajectory (default: 6). |
| `max_trajectory_tokens` | `int` | Max total token length; exceeding truncates the trajectory. |
| `trace_dir` | `str` | Directory for per-trajectory JSON trace dumps. |
| `trace_callback` | `Callable` | Decides whether to store a trajectory trace. |
| `success_callback` | `Callable` | Decides filename prefix (`ok-` vs `fail-`). |

### Session Affinity

`MultiTurnRollout` defaults to `session_affinity=True`. With a capable sampler, rollout creates a unique
ID and an independent `adapter_kwargs` dictionary once per episode, including `session_ids=[session_id]`.
All turns and follow-ups reuse these options; callbacks treat them as read-only.

Default and custom callbacks forward options through `sampler.sample(..., **adapter_kwargs)`.
No additional callback parameter or callback identity check is needed. Despite its name, `adapter_kwargs`
contains backend call options, including adapters, partial rollout, and routing. A custom callback that
does not forward these options is not guaranteed to use session affinity.
Dispatch consumes `session_ids` without adding fields to `InputFeature`. Token, logprob, and training
outputs are unchanged. API-only paths and older samplers retain their calls; `APISampler` does not
forward sampler options to external APIs in mixed-backend callbacks.

Clients of the same sampling pool share a routing actor. Existing bindings take priority over load
balancing without reserving instances. Episode exit removes the binding. Sampling releases occupancy
when the call exits; this version does not keep timed-out remote tasks charged until actual completion.

The bounded LRU defaults to 10000 bindings. Idle bindings, including sessions waiting on tools, may
be evicted. If every binding is executing, a new request uses ordinary load balancing without creating
a binding. Affinity improves cache reuse opportunities rather than guaranteeing hits.

```python
rollout = MultiTurnRollout(sampler, template, session_affinity=True)
results = rollout(trajectories)  # No session_ids argument is required.
results = rollout(trajectories, session_affinity=False)
```

Enable vLLM caching separately with `engine_args={"enable_prefix_caching": True}`. Closing bindings
does not clear engine caches or change weight synchronization. This version does not change
`submit_generation()`, TQ, or routing inside PD instances.

### Output Fields

Each output trajectory dict includes:

| Field | Type | Description |
|-------|------|-------------|
| `messages` | `List[Dict]` | Full conversation including tool turns. |
| `input_ids` | `List[int]` | Token IDs of the full sequence. |
| `labels` | `List[int]` | Training labels (`-100` for non-trainable tokens). |
| `completion_mask` | `List[int]` | Policy-generated positions that carry rollout log probabilities. |
| `turns` | `int` | Number of turns performed. |
| `stop_reason` | `str` | `'stop'` / `'length'` |
| `truncated` | `bool` | Whether the trajectory was cut off rather than concluding on its own: generation hit `max_tokens` (`stop_reason='length'`), the turn limit was reached, or a length cap dropped it. |
| `logprobs` | `List` | Per-token log probabilities (if available). |

### Ray Remote Support

`MultiTurnRollout` is decorated with `@remote_class()`, enabling transparent deployment as a Ray actor:

```python
# The rollout can run as a Ray remote actor
rollout_actor = MultiTurnRollout.remote(sampler=sampler, template=template, ...)
results = ray.get(rollout_actor.__call__.remote(trajectories))
```

## API and Mixed-Backend Rollouts

API-only rollout uses the same class and still requires the local template that tokenizes external replies:

```python
from twinkle_agentic.protocol.openai import OpenAI
from twinkle_agentic.rollout import MultiTurnRollout

api = OpenAI(model='qwen3.5-32b', base_url='http://localhost:8000/v1', concurrency=8)
rollout = MultiTurnRollout(
    api,
    template=template,
    tool_manager=tool_manager,
    sampling_params=SamplingParams(temperature=0.7),
    max_turns=6,
    trace_dir='api_traces/',
)
results = rollout(trajectories)
```

When both `sampler` and `api` are supplied, the default is the sampler. Pass `response_callback` to choose per turn; it receives both backends and must return one `SampledSequence`. API turns have no rollout log probabilities, so `api_appended_as='demonstration'` includes them in SFT but excludes them from GRPO. Use `'context'` to exclude them from both.

### Stop Reasons

| Reason | Description |
|--------|-------------|
| `stop` | Assistant responded without tool calls (natural end). |
| `length` | Generation reached its token limit. |
| `max_turns` | Reached the tool-turn limit without a follow-up. |
| `generation_error` | The external endpoint failed before returning a valid response. |
