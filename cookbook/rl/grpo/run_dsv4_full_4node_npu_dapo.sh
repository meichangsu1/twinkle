#!/usr/bin/env bash
# Full-model DAPO GRPO: two actor nodes; default two TP8 rollout instances on one node.
set -euo pipefail

export ROLLOUT_DP=${ROLLOUT_DP:-2}
# A 4096-token completion previously exhausted actor memory; allow override
# after measuring full-model memory with the new rollout topology.
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}

exec bash /highcode/shared_data/rl/run_dsv4_full_npu_dapo.sh "$@"
