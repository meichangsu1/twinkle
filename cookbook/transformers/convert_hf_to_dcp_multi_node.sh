# Multi-node debug launch for HF -> target-world-size DCP conversion.
# Run this script on both nodes with the same MASTER_ADDR/MASTER_PORT and the correct NODE_RANK.
#
# Node 0:
#   MASTER_ADDR=<node0_ip> NODE_RANK=0 bash cookbook/transformers/convert_hf_to_dcp_multi_node.sh
# Node 1:
#   MASTER_ADDR=<node0_ip> NODE_RANK=1 bash cookbook/transformers/convert_hf_to_dcp_multi_node.sh

set -euo pipefail

: "${MASTER_ADDR:?Set MASTER_ADDR to the rank-0 node IP before launching.}"

MASTER_PORT="${MASTER_PORT:-29500}"
NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
TARGET_WORLD_SIZE="${TARGET_WORLD_SIZE:-2}"
VISIBLE_DEVICES="${VISIBLE_DEVICES:-0}"
MODEL_ID="${MODEL_ID:-ms://deepseek-ai/DeepSeek-V4-Flash}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/deepseek_v4_dcp_ws${TARGET_WORLD_SIZE}}"
NUM_LAYERS="${NUM_LAYERS:-1}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"

PYTHONPATH=src CUDA_VISIBLE_DEVICES="${VISIBLE_DEVICES}" torchrun \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  cookbook/transformers/convert_hf_to_dcp.py \
  --model-id "${MODEL_ID}" \
  --output-dir "${OUTPUT_DIR}" \
  --target-world-size "${TARGET_WORLD_SIZE}" \
  --num-layers "${NUM_LAYERS}" \
  --torch-dtype "${TORCH_DTYPE}" \
  --trust-remote-code

# MASTER_ADDR=<node0_ip> NODE_RANK=0 bash cookbook/transformers/convert_hf_to_dcp_multi_node.sh

# MASTER_ADDR=<node0_ip> NODE_RANK=1 bash cookbook/transformers/convert_hf_to_dcp_multi_node.sh
