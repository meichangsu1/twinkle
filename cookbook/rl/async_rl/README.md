# Async Multi-LoRA GRPO

One YAML configuration launches two LoRA tenants over a shared training model,
vLLM sampler, and TransferQueue data plane. Rollout, advantage calculation, and
training run as independent workers, while each tenant keeps its own dataset,
reward, optimizer, scheduler, partitions, and policy versions.

This directory contains the YAML-managed CLI workflow. For client-orchestrated
GRPO over HTTP, see
[`cookbook/client/async_rl`](../../client/async_rl/README.md).

## Resources

The default configuration uses three GPUs:

| Component | GPUs | Purpose |
|---|---:|---|
| Training model | 2 | Native FSDP training for all LoRA tenants |
| vLLM sampler | 1 | Shared rollout generation |

The example has two GSM8K tenants. Each tenant consumes 128 prompts in eight
partitions:

```text
16 prompts × 4 generations = 64 samples per partition
64 samples ÷ mini_batch_size 4 = 16 optimizer steps per partition
8 partitions × 16 optimizer steps = 128 optimizer steps per tenant
```

## Quick start

Install the async-RL dependencies:

```bash
pip install -e '.[async-rl]'
```

Set a model and datasets that both training processes and Ray workers can
access:

```bash
export MODEL_ID=/absolute/path/to/Qwen3.5-4B
export TENANT_A_DATASET_ID=/absolute/path/to/gsm8k
export TENANT_B_DATASET_ID=/absolute/path/to/gsm8k
export CUDA_VISIBLE_DEVICES=0,1,2
```

`MODEL_ID` must be a Hugging Face-compatible directory readable by both
Transformers and vLLM. Dataset values may be local ModelScope-compatible paths
or `ms://...` identifiers. Use local paths when running offline.

Start a local Ray cluster when needed and launch training:

```bash
bash cookbook/rl/async_rl/run_async_multi_lora_grpo.sh
```

If Ray is already running, the script reuses it. To launch the Python entry
point directly:

```bash
python cookbook/rl/async_rl/async_multi_lora_grpo.py \
  --config cookbook/rl/async_rl/async_multi_lora_grpo.yaml
```

## Scheduling and staleness

The workers exchange complete prompt groups through TransferQueue:

```text
RolloutWorker -> TransferQueue -> AdvantageWorker -> TrainerWorker
```

`max_staleness` limits live partitions, not mini-batches within one partition:

- `max_staleness=0` allows one live partition. Training may still overlap with
  rollout inside that partition once enough complete prompt groups form a
  mini-batch.
- `max_staleness=1` allows two live partitions, so training an older partition
  may also overlap with rollout for the next partition.

The default `mini_batch_size=4` equals one complete four-generation prompt
group, allowing the Trainer to consume a group without waiting for all 16
groups in the partition. A policy version is published only after every
mini-batch in the partition has trained.

Batch settings must satisfy:

```text
partition_samples = rollout.batch_size × rollout.num_generations
partition_samples % train.mini_batch_size == 0
train.mini_batch_size % rollout.num_generations == 0
train.mini_batch_size % model_dp == 0
```

## Outputs

| Output | Default path |
|---|---|
| LoRA checkpoints | `output/async_multi_lora_grpo/` |
| Rollout JSONL | `output/async_multi_lora_grpo/rollouts/` |
| Metrics | `outputs/async_rl/metrics.jsonl` |
| Metrics summary | `outputs/async_rl/summary.json` |

Set `rollout_output.enabled: false` when benchmarking throughput to avoid
writing one JSONL file per completed prompt group.

## Files

| File | Role |
|---|---|
| `async_multi_lora_grpo.py` | Load the YAML and run `AsyncMultiLoraGRPOPipeline` |
| `async_multi_lora_grpo.yaml` | Model, sampler, scheduler, LoRA, dataset, reward, and tenant configuration |
| `run_async_multi_lora_grpo.sh` | Validate required environment variables, start local Ray if needed, and launch training |

## Synchronous and asynchronous comparisons

The comparison configurations keep each synchronous/asynchronous pair aligned
on model, data, rollout, LoRA, and train batch settings. The synchronous files
use `sync_barrier_multi_lora_grpo.py`; they are not async runs with
`max_staleness=0`.

| Dataset | Tenants | Synchronous config | Asynchronous config |
|---|---:|---|---|
| GSM8K | 1 | `compare_single_lora_gsm8k_sync.yaml` | `compare_single_lora_gsm8k_async.yaml` |
| GSM8K | 2 | `compare_multi_lora_gsm8k_sync.yaml` | `compare_multi_lora_gsm8k_async.yaml` |
| DAPO-Math | 1 | `compare_single_lora_dapo_sync.yaml` | `compare_single_lora_dapo_async.yaml` |
| DAPO-Math | 2 | `compare_multi_lora_dapo_sync.yaml` | `compare_multi_lora_dapo_async.yaml` |

Prepare two deterministic 500-row GSM8K datasets:

```bash
python cookbook/rl/async_rl/prepare_multi_tenant_gsm8k.py \
  --source /data/gsm8k/train.jsonl \
  --samples-per-tenant 500 \
  --output-dir /data/gsm8k_multi_tenant
```

Run the GSM8K comparisons after exporting their required dataset paths:

```bash
bash cookbook/rl/async_rl/run_compare_single_lora_gsm8k_sync.sh
bash cookbook/rl/async_rl/run_compare_single_lora_gsm8k_async.sh
bash cookbook/rl/async_rl/run_compare_multi_lora_gsm8k_sync.sh
bash cookbook/rl/async_rl/run_compare_multi_lora_gsm8k_async.sh
```

DAPO-Math comparison scripts validate the required dataset environment
variables and select three visible GPUs by default:

```bash
bash cookbook/rl/async_rl/run_compare_single_lora_dapo_sync.sh
bash cookbook/rl/async_rl/run_compare_single_lora_dapo_async.sh
bash cookbook/rl/async_rl/run_compare_multi_lora_dapo_sync.sh
bash cookbook/rl/async_rl/run_compare_multi_lora_dapo_async.sh
```

## Troubleshooting

- **A placement group stays pending** — the default run needs three visible
  GPUs. Check `ray status` and make sure Ray was started with at least three
  GPUs.
- **The process tries to download a model or dataset** — replace every
  `MODEL_ID` and `TENANT_*_DATASET_ID` value with an absolute local path visible
  to all Ray workers.
- **Async is slower with `max_staleness=0`** — this setting disables
  cross-partition overlap but retains Worker, Ray RPC, and TransferQueue
  overhead. Partition-internal overlap also depends on prompt completion
  distribution.
- **Rollout files dominate runtime** — disable `rollout_output.enabled` for
  performance measurements.
