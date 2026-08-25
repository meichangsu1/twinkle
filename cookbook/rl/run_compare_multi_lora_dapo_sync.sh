#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export MODEL_ID="${MODEL_ID:-/nas/disk1/Qwen3-4B}"
: "${TENANT_A_DAPO_DATASET_ID:?Set TENANT_A_DAPO_DATASET_ID to tenant A's local dataset path}"
: "${TENANT_B_DAPO_DATASET_ID:?Set TENANT_B_DAPO_DATASET_ID to tenant B's local dataset path}"
export TENANT_A_DAPO_DATASET_ID TENANT_B_DAPO_DATASET_ID

PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "${PYTHON_BIN}" cookbook/rl/sync_barrier_multi_lora_grpo.py \
  --config cookbook/rl/compare_multi_lora_dapo_sync.yaml
