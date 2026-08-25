#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export MODEL_ID="${MODEL_ID:-/nas/disk1/Qwen3-4B}"
: "${DAPO_TRAIN_DATASET_ID:?Set DAPO_TRAIN_DATASET_ID to a local DAPO dataset path}"
export DAPO_TRAIN_DATASET_ID

PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "${PYTHON_BIN}" cookbook/rl/async_multi_lora_grpo.py \
  --config cookbook/rl/compare_single_lora_dapo_async.yaml
