# vLLMSampler

## Shared Instance Routing

In Ray mode, continuous-work delegates selection to a pool-wide `RequestRouter`. It balances by
in-flight load, breaks ties randomly, and prefers existing session bindings. `sample()` accepts an
optional `session_ids` argument. `MultiTurnRollout` prepares IDs once per episode; default and custom
callbacks pass them through by forwarding `adapter_kwargs` to the sampler.

```python
sampler = vLLMSampler(
    model_id='ms://Qwen/Qwen3.5-4B',
    remote_group='sampler_group',
    device_mesh=DeviceMesh.from_sizes(dp_size=2),
    routing_config={"max_sessions": 10000},
    engine_args={"enable_prefix_caching": True},
)
```

The dispatch layer consumes `routing_config`; it never reaches the engine. Currently only the positive
integer `max_sessions` is supported, defaulting to 10000. Clients with the same actor ID set and Ray
namespace share one router; conflicting configurations are rejected. The creating job retains the
router, which is not detached.

`session_ids` contains one nonempty string or `None` per input, in input order. Dispatch consumes this
argument without modifying `InputFeature` or passing IDs to templates or engines. Omitted IDs and
`None` entries use ordinary balancing.

```python
responses = sampler.sample(inputs, session_ids=["episode-A", "episode-B"])
sampler.release_sessions(["episode-A", "episode-B"])  # When the sessions end.
```

Sessions do not reserve workers;
idle bindings may be evicted by the bounded LRU. A full table with all bindings executing falls back
to ordinary routing for new sessions, without blocking or creating a binding.

`release_sessions(ids)` removes bindings without clearing KV cache and rejects executing sessions.
Sampling releases occupancy in `finally`, and rollout exit removes the binding. This version does not
track remote tasks that outlive the caller; timeouts or errors may underestimate actual running work.
Local single-engine mode accepts the parameter but performs no instance routing and creates no routing actor.

vLLMSampler uses the vLLM engine for efficient inference, supporting high-throughput batch sampling.

## Usage Example

```python
from twinkle.sampler import vLLMSampler
from twinkle.data_format import SamplingParams
from twinkle import DeviceMesh

# Create sampler
sampler = vLLMSampler(
    model_id='ms://Qwen/Qwen3.5-4B',
    device_mesh=DeviceMesh.from_sizes(dp_size=2, tp_size=2),
    remote_group='sampler_group'
)

# Add LoRA
sampler.add_adapter_to_model('my_lora', 'path/to/lora')

# Set sampling parameters
params = SamplingParams(
    max_tokens=512,
    temperature=0.7,
    top_p=0.9,
    top_k=50
)

# Perform sampling
responses = sampler.sample(
    trajectories,
    sampling_params=params,
    adapter_name='my_lora',
    num_samples=4  # Generate 4 samples per prompt
)
```

## Features

- **High Performance**: Achieves high throughput using PagedAttention and continuous batching
- **LoRA Support**: Supports dynamic loading and switching of LoRA adapters
- **Multi-Sample Generation**: Can generate multiple samples per prompt
- **Tensor Parallel**: Supports tensor parallelism to accelerate large model inference

## Remote Execution

vLLMSampler supports the `@remote_class` decorator and can run in Ray clusters:

```python
import twinkle
from twinkle import DeviceGroup, DeviceMesh
from twinkle.sampler import vLLMSampler

# Initialize Ray cluster
device_groups = [
    DeviceGroup(name='sampler', ranks=4, device_type='cuda')
]
twinkle.initialize('ray', groups=device_groups)

# Create remote sampler
sampler = vLLMSampler(
    model_id='ms://Qwen/Qwen3.5-4B',
    device_mesh=DeviceMesh.from_sizes(dp_size=4),
    remote_group='sampler'
)

# sample method executes in remote worker
responses = sampler.sample(trajectories, sampling_params=params)
```

## Environment Variables

- `TWINKLE_VLLM_IPC_TIMEOUT_S`:
  Controls the timeout (in seconds) for the IPC channel (ZMQ REQ/REP)
  between `vLLMSampler` and the vLLM worker extension.
  Default is `300`. This value must be greater than `0`.

> In RLHF training, vLLMSampler is typically separated from the Actor model, using different hardware resources to avoid interference between inference and training.
