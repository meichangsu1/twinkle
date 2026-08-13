#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TWINKLE_SERVER_URL="${TWINKLE_SERVER_URL:-http://127.0.0.1:8000}"
export TWINKLE_SERVER_TOKEN="${TWINKLE_SERVER_TOKEN:-EMPTY_TOKEN}"
export TWINKLE_MODEL_ID="${TWINKLE_MODEL_ID:-deepseek-v4-0731-local}"
export DSV4_MODEL_ID="${DSV4_MODEL_ID:-/nas/disk1/random-deepseek-v4-4b}"
export DATASET_ID="${DATASET_ID:-/model/ljl/dataset/self-cognition.jsonl}"
export ADAPTER_NAMES="${ADAPTER_NAMES:-tenant_a,tenant_b}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
export MAX_STEPS="${MAX_STEPS:-10}"
export MAX_LENGTH="${MAX_LENGTH:-2048}"
export LORA_R="${LORA_R:-8}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export LR="${LR:-1e-4}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/twinkle_dsv4_0731_npu_multi_lora}"

test -f "$DATASET_ID"

exec python3 "$SCRIPT_DIR/dsv4_multi_lora_sft.py"
