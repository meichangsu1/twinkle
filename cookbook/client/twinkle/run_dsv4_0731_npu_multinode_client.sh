#!/usr/bin/env bash
set -euo pipefail

# The client is an HTTP orchestrator. Run exactly one client process; it does
# not join Ray and does not need NPU devices.
#
# Example:
#   HEAD_IP=10.0.0.10 \
#   OUTPUT_DIR=/shared/twinkle_output/dsv4-0731-a3-32npu \
#     bash run_dsv4_0731_npu_multinode_client.sh

: "${HEAD_IP:?Set HEAD_IP to the Twinkle/Ray head node IP}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a shared absolute path mounted on both server nodes}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TWINKLE_SERVER_URL="${TWINKLE_SERVER_URL:-http://$HEAD_IP:8000}"
export TWINKLE_SERVER_TOKEN="${TWINKLE_SERVER_TOKEN:-EMPTY_TOKEN}"
export TWINKLE_MODEL_ID="${TWINKLE_MODEL_ID:-deepseek-v4-0731-local}"
# The template/encoding side must resolve the same checkpoint as the server.
export DSV4_MODEL_ID="${DSV4_MODEL_ID:-hf://deepseek-ai/DeepSeek-V4-Flash-0731}"
export DATASET_ID="${DATASET_ID:-/model/ljl/dataset/self-cognition.jsonl}"
export OUTPUT_DIR

# forward_backward dispatches one slice to each of the 32 FSDP ranks.
export BATCH_SIZE="${BATCH_SIZE:-32}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
export MAX_STEPS="${MAX_STEPS:-10}"
export MAX_LENGTH="${MAX_LENGTH:-2048}"
export LORA_R="${LORA_R:-8}"
export LORA_ALPHA="${LORA_ALPHA:-32}"

mkdir -p "$OUTPUT_DIR"
exec python3 cookbook/client/twinkle/dsv4_multi_lora_sft.py
