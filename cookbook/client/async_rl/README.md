# Client-orchestrated asynchronous RL

This directory demonstrates direct orchestration of the server's Model,
Sampler, and TransferQueue DataPlane components.

The client owns its Dataset, multi-turn rollout, Reward, Advantage, policy
versioning, staleness, and algorithm. There is no central async-RL management,
runtime, tenant submission API, or server-side RL worker involved.

- client_orchestrated_grpo.py maps each DataLoader batch to a private client-side
  rollout partition. Its Rollout, Advantage, and Trainer workers run as
  independent asyncio tasks. Prompt groups stream through TQ independently, so
  ready groups can train while the remaining groups are still sampling. A
  policy is published once, after the whole partition has trained.
- client_orchestrated_dpo.py uses Dataset, Reference, and Trainer workers. It is
  a runnable offline DPO loop and shows that Worker is a role lifecycle rather
  than a fixed RL stage graph.

Start the component server with
cookbook/client/server/transformer/server_config.yaml.

```bash
pip install -e '.[async-rl,client]'
twinkle-server launch -c cookbook/client/server/transformer/server_config.yaml
```

Then start one or more independent client orchestrators:

```bash
python cookbook/client/async_rl/client_orchestrated_grpo.py
```

The training loop composes only the low-level component methods:

- `sampler.sample_to_data_plane(...)` / `sampler.asample_to_data_plane(...)`
- `model.forward_only_from_data_plane(...)`
- `model.forward_backward_from_data_plane(...)`
- `model.clip_grad_and_step(...)`
- `model.save(...)`
- `data_plane.put/get/append/release(...)` and
  `aput/aget/aget_batch/aappend/arelease(...)`

There is no additional RL runtime or orchestration protocol. The GRPO example
uses `asample_to_data_plane()` so the Sampler's output `DataRef` remains in TQ. Each
generation is one row tagged with its group, generation index, rollout policy,
and status. The Advantage worker reads only decoded completions and appends
reward and advantage to the same keys. Token tensors and sampled log-probabilities
remain server-side. The Trainer passes one or more `DataRef` values to
`forward_backward_from_data_plane()` and releases them in a local `finally` block. `asample()`
remains the materialized-response convenience API.

- `ClientMultiTurnRollout.arun()` keeps tool calls and Reward computation in
  the client and accepts an explicit `adapter_uri` policy snapshot.

`_RolloutPartition` is a private client record, not a server resource or SDK
API. The local FIFO limits live DataLoader batches before rollout, ready prompt
groups immediately use the Model primitives above, and the client calls
`save()` once after a whole batch has trained. `WorkerPipeline` only
starts, joins, and fail-fast cancels concrete roles; queues and algorithm state
remain ordinary client Python code.

Different client processes may run GRPO and DPO against the same component
server. Model adapters are session-scoped. DataRefs are opaque capabilities
whose UUID identifies an independent physical TQ partition; DataPlane storage
does not know about tokens or sessions. The client remains the single writer
and algorithm owner for its adapter. Async Sampler requests share vLLM
continuous batching; this example does not promise strict round-robin fairness
between sampler tenants.

The original YAML-managed runtime is still separate and can be started with:

```bash
python cookbook/rl/async_multi_lora_grpo.py \
  --config cookbook/rl/async_multi_lora_grpo.yaml
```
