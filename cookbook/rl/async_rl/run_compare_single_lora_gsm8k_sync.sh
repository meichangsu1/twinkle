#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MODEL_ID="${MODEL_ID:-/nas/disk1/Qwen3-4B}"
: "${GSM8K_TRAIN_DATASET_ID:?Set GSM8K_TRAIN_DATASET_ID to a local GSM8K dataset path}"
export GSM8K_TRAIN_DATASET_ID

PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "${PYTHON_BIN}" cookbook/rl/async_rl/sync_barrier_multi_lora_grpo.py \
  --config cookbook/rl/async_rl/compare_single_lora_gsm8k_sync.yaml
