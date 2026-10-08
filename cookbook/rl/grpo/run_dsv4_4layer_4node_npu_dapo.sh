#!/usr/bin/env bash
# Four-layer link test: two 16-NPU actor nodes, two 16-NPU rollout nodes.
set -euo pipefail

export ROLLOUT_DP=4
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
export STEPS=${STEPS:-1}
export BATCH_SIZE=${BATCH_SIZE:-32}
export NUM_GENERATIONS=${NUM_GENERATIONS:-4}

exec bash /opt/twinkle/cookbook/rl/grpo/run_dsv4_4layer_3node_npu_dapo.sh "$@"
