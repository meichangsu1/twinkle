#!/usr/bin/env bash
# Four-layer DAPO GRPO on three A3 nodes (16 NPUs per node).
set -euo pipefail

role=${1:-}
if [[ ! "$role" =~ ^(head|worker|run)$ ]]; then
  echo "Usage: $0 {head|worker|run}" >&2
  exit 2
fi

cd /opt/twinkle
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TWINKLE_TRUST_REMOTE_CODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LOG_LEVEL=INFO RAY_ROTATION_MAX_BYTES=20971520 RAY_ROTATION_BACKUP_COUNT=1
export NETWORK_IFACE=bond0 GLOO_SOCKET_IFNAME=bond0 HCCL_SOCKET_IFNAME=bond0
export HCCL_CONNECT_TIMEOUT=7200 HCCL_EXEC_TIMEOUT=0
export RAY_TMPDIR=/dev/shm RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HEAD_IP=11.173.4.131
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
test -d /sys/class/net/bond0

if [[ "$role" == head ]]; then
  unset RAY_ADDRESS
  ray start --head --node-ip-address="$HEAD_IP" --port=6379 \
    --resources='{"NPU":16}' --temp-dir=/dev/shm/ray-dsv4-4layer \
    --disable-usage-stats --include-dashboard=false
  exit 0
fi
if [[ "$role" == worker ]]; then
  unset RAY_ADDRESS
  : "${NODE_IP:?Set NODE_IP to 11.173.4.140 or 11.173.4.144}"
  if [[ "$NODE_IP" != 11.173.4.140 && "$NODE_IP" != 11.173.4.144 ]]; then
    echo 'NODE_IP must be 11.173.4.140 or 11.173.4.144' >&2
    exit 2
  fi
  ray start --address="$HEAD_IP:6379" --node-ip-address="$NODE_IP" \
    --resources='{"NPU":16}' --disable-usage-stats
  exit 0
fi

export RAY_ADDRESS="$HEAD_IP:6379"
export DATASET_KIND=dapo DAPO_PATH=${DAPO_PATH:-/highcode/shared_data/DAPO-Math-17}
export DATASET_MAX_ROWS=${DATASET_MAX_ROWS:-2000}
export ACTOR_MODEL=${ACTOR_MODEL:-/highcode/shared_data/DeepSeek-V4-Flash-0731-4layers-bf16}
export ROLLOUT_MODEL=${ROLLOUT_MODEL:-/highcode/shared_data/DeepSeek-V4-Flash-0731-4layers-w8a8-v2}
export NPUS_PER_NODE=16 ACTOR_NPUS=32 ACTOR_EP=32 ROLLOUT_START_RANK=32 ROLLOUT_TP=8
export ACTOR_PRECISION=bf16 LORA_R=8 LORA_ALPHA=32
export MAX_MODEL_LEN=4096 MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=4096
export GPU_MEMORY_UTILIZATION=0.85 TWINKLE_VLLM_BUCKET_SIZE_MB=64
export STEPS=${STEPS:-3} BATCH_SIZE=${BATCH_SIZE:-64} NUM_GENERATIONS=${NUM_GENERATIONS:-2}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024} LR=${LR:-1e-5}

ray status --address="$RAY_ADDRESS"
python - <<'PY'
import os
import ray

ray.init(address=os.environ['RAY_ADDRESS'], ignore_reinit_error=True)
alive = [node for node in ray.nodes() if node['Alive']]
available = ray.cluster_resources().get('NPU', 0)
print(f'Four-layer Ray preflight: alive_nodes={len(alive)}, NPU_resources={available}')
if len(alive) != 3 or available < 48:
    raise RuntimeError('Four-layer run requires three alive Ray nodes and at least 48 NPU resources')
ray.shutdown()
PY

if [[ -d "$DAPO_PATH" ]]; then
  test -f "$DAPO_PATH/dapo-math-17k.parquet"
else
  test -f "$DAPO_PATH"
fi
test -f "$ACTOR_MODEL/config.json"
test -f "$ROLLOUT_MODEL/config.json"
if (( BATCH_SIZE <= ACTOR_NPUS )); then
  echo 'BATCH_SIZE must exceed 32 actor ranks' >&2
  exit 2
fi
if (( BATCH_SIZE * NUM_GENERATIONS % ACTOR_NPUS != 0 )); then
  echo 'BATCH_SIZE * NUM_GENERATIONS must be divisible by ACTOR_NPUS' >&2
  exit 2
fi
if (( DATASET_MAX_ROWS > 0 && STEPS * BATCH_SIZE > DATASET_MAX_ROWS )); then
  echo 'STEPS * BATCH_SIZE exceeds DATASET_MAX_ROWS' >&2
  exit 2
fi

REPORT_ROOT=${REPORT_ROOT:-/highcode/shared_data/dsv4_logs}
mkdir -p "$REPORT_ROOT"
RUN_DIR=$(mktemp -d "$REPORT_ROOT/4layer_XXXXXXXX")
export REPORT_DIR="$RUN_DIR/report"
nohup python -u -m cookbook.rl.grpo.dsv4_lora_npu \
  >"$RUN_DIR/grpo.log" 2>&1 </dev/null &
echo "GRPO PID=$! Report=$REPORT_DIR Log=$RUN_DIR/grpo.log"
