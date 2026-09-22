#!/usr/bin/env bash
# Four-layer DAPO link test in the development environment (two nodes, four NPUs each).
set -euo pipefail

role=${1:-}
if [[ ! "$role" =~ ^(head|worker|run)$ ]]; then
  echo "Usage: $0 {head|worker|run}" >&2
  exit 2
fi

cd /nas/disk6/ljl/project/dsv4-lora-weight-sync/twinkle
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TWINKLE_TRUST_REMOTE_CODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LOG_LEVEL=INFO RAY_ROTATION_MAX_BYTES=20971520 RAY_ROTATION_BACKUP_COUNT=1
export NETWORK_IFACE=eth0 GLOO_SOCKET_IFNAME=eth0 HCCL_SOCKET_IFNAME=eth0
export HCCL_CONNECT_TIMEOUT=7200 HCCL_EXEC_TIMEOUT=0
export RAY_TMPDIR=/dev/shm RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HEAD_IP=172.61.8.191 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
test -d /sys/class/net/eth0

if [[ "$role" == head ]]; then
  unset RAY_ADDRESS
  ray start --head --node-ip-address="$HEAD_IP" --port=6379 \
    --resources='{"NPU":4}' --temp-dir=/dev/shm/ray-dsv4-mini \
    --disable-usage-stats --include-dashboard=false
  exit 0
fi
if [[ "$role" == worker ]]; then
  unset RAY_ADDRESS
  ray start --address="$HEAD_IP:6379" --node-ip-address=172.61.10.150 \
    --resources='{"NPU":4}' --disable-usage-stats
  exit 0
fi

export RAY_ADDRESS="$HEAD_IP:6379"
export DATASET_KIND=dapo DAPO_PATH=${DAPO_PATH:-/model/ljl/project/data/DAPO-Math-17k}
export DATASET_MAX_ROWS=${DATASET_MAX_ROWS:-2000}
export ACTOR_MODEL=${ACTOR_MODEL:-/nas/disk1/DeepSeek-V4-Flash-0731-4layers-bf16}
export ROLLOUT_MODEL=${ROLLOUT_MODEL:-/nas/disk1/DeepSeek-V4-Flash-0731-4layers-w8a8}
export NPUS_PER_NODE=4 ACTOR_NPUS=4 ACTOR_EP=4 ROLLOUT_START_RANK=4 ROLLOUT_TP=4
export ACTOR_PRECISION=bf16 LORA_R=8 LORA_ALPHA=32
export MAX_MODEL_LEN=1024 MAX_NUM_SEQS=2 MAX_NUM_BATCHED_TOKENS=4096
export GPU_MEMORY_UTILIZATION=0.85 TWINKLE_VLLM_BUCKET_SIZE_MB=1
export STEPS=${STEPS:-3} BATCH_SIZE=${BATCH_SIZE:-8} NUM_GENERATIONS=${NUM_GENERATIONS:-2}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512} LR=${LR:-1e-5}

ray status --address="$RAY_ADDRESS"
python - <<'PY'
import os
import ray

ray.init(address=os.environ['RAY_ADDRESS'], ignore_reinit_error=True)
alive = [node for node in ray.nodes() if node['Alive']]
available = ray.cluster_resources().get('NPU', 0)
print(f'Mini Ray preflight: alive_nodes={len(alive)}, NPU_resources={available}')
if len(alive) != 2 or available < 8:
    raise RuntimeError('Mini run requires two alive Ray nodes and at least eight NPU resources')
ray.shutdown()
PY

if [[ -d "$DAPO_PATH" ]]; then
  test -f "$DAPO_PATH/dapo-math-17k.parquet"
else
  test -f "$DAPO_PATH"
fi
test -f "$ACTOR_MODEL/config.json"
test -f "$ROLLOUT_MODEL/config.json"
if (( BATCH_SIZE * NUM_GENERATIONS % ACTOR_NPUS != 0 )); then
  echo 'BATCH_SIZE * NUM_GENERATIONS must be divisible by ACTOR_NPUS' >&2
  exit 2
fi
if (( DATASET_MAX_ROWS > 0 && STEPS * BATCH_SIZE > DATASET_MAX_ROWS )); then
  echo 'STEPS * BATCH_SIZE exceeds DATASET_MAX_ROWS' >&2
  exit 2
fi
REPORT_ROOT=${REPORT_ROOT:-/tmp/dsv4_dapo_reports}
mkdir -p "$REPORT_ROOT"
TEST_ROOT=$(mktemp -d "$REPORT_ROOT/mini_XXXXXXXX")
export REPORT_DIR="$TEST_ROOT/report"
echo "Report: $REPORT_DIR"
python -u -m cookbook.rl.grpo.dsv4_lora_npu 2>&1 | tee "$TEST_ROOT/grpo.log"
