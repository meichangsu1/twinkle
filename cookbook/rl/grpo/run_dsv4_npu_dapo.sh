#!/usr/bin/env bash
# Four-layer smoke test, or three-node full-model DAPO GRPO.
# Run from the matching node; see dsv4_lora_h800_read_me.md.
set -euo pipefail

role=${1:-}
if [[ ! "$role" =~ ^(mini-head|mini-worker|mini|head|worker|full)$ ]]; then
  echo "Usage: $0 {mini-head|mini-worker|mini|head|worker|full}" >&2
  exit 2
fi

if [[ "$role" == mini* ]]; then
  # Development cluster from the four-layer section of the existing readme.
  repo_dir=/nas/disk6/ljl/project/dsv4-lora-weight-sync/twinkle
  network_iface=eth0
  head_ip=172.61.8.191
  default_dapo_path=/model/ljl/project/data/DAPO-Math-17k
  visible_devices=0,1,2,3
else
  # Separate full-model A3 environment.
  repo_dir=/opt/twinkle
  network_iface=bond0
  head_ip=22.6.7.15
  default_dapo_path=/highcode/shared_data/DAPO-Math-17
  visible_devices=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
fi
cd "$repo_dir"
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TWINKLE_TRUST_REMOTE_CODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LOG_LEVEL=INFO RAY_ROTATION_MAX_BYTES=20971520 RAY_ROTATION_BACKUP_COUNT=1
export NETWORK_IFACE="$network_iface" GLOO_SOCKET_IFNAME="$network_iface" HCCL_SOCKET_IFNAME="$network_iface"
export HCCL_CONNECT_TIMEOUT=7200 HCCL_EXEC_TIMEOUT=0
export RAY_TMPDIR=/dev/shm RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HEAD_IP="$head_ip" ASCEND_RT_VISIBLE_DEVICES="$visible_devices"
export DATASET_KIND=dapo DAPO_PATH=${DAPO_PATH:-$default_dapo_path}
export DATASET_MAX_ROWS=${DATASET_MAX_ROWS:-2000}
export ACTOR_PRECISION=bf16 LORA_R=8 LORA_ALPHA=32
export LR=${LR:-1e-5} NUM_GENERATIONS=${NUM_GENERATIONS:-2}
test -d "/sys/class/net/$network_iface"

if [[ "$role" == mini-head || "$role" == mini-worker ]]; then
  unset RAY_ADDRESS
  if [[ "$role" == mini-head ]]; then
    ray start --head --node-ip-address="$HEAD_IP" --port=6379 \
      --resources='{"NPU":4}' --temp-dir=/dev/shm/ray-dsv4-mini \
      --disable-usage-stats --include-dashboard=false
  else
    ray start --address="$HEAD_IP:6379" --node-ip-address=172.61.10.150 \
      --resources='{"NPU":4}' --disable-usage-stats
  fi
  exit 0
elif [[ "$role" == mini ]]; then
  # Readme's two-node, four-NPU-per-node layout.
  export RAY_ADDRESS="$HEAD_IP:6379"
  export ACTOR_MODEL=${ACTOR_MODEL:-/nas/disk1/DeepSeek-V4-Flash-0731-4layers-bf16}
  export ROLLOUT_MODEL=${ROLLOUT_MODEL:-/nas/disk1/DeepSeek-V4-Flash-0731-4layers-w8a8}
  export NPUS_PER_NODE=4 ACTOR_NPUS=4 ACTOR_EP=4 ROLLOUT_START_RANK=4 ROLLOUT_TP=4
  export MAX_MODEL_LEN=1024 MAX_NUM_SEQS=2 MAX_NUM_BATCHED_TOKENS=4096
  export GPU_MEMORY_UTILIZATION=0.85 TWINKLE_VLLM_BUCKET_SIZE_MB=1
  export STEPS=${STEPS:-3} BATCH_SIZE=${BATCH_SIZE:-8} MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
  ray status --address="$RAY_ADDRESS"
  python - <<'PY'
import os
import ray

ray.init(address=os.environ['RAY_ADDRESS'], ignore_reinit_error=True)
alive = [node for node in ray.nodes() if node['Alive']]
available = ray.cluster_resources().get('NPU', 0)
print(f'Ray preflight: alive_nodes={len(alive)}, NPU_resources={available}')
if len(alive) != 2 or available < 8:
    raise RuntimeError('Four-layer run requires two alive Ray nodes and at least eight NPU resources')
ray.shutdown()
PY
elif [[ "$role" == head || "$role" == worker ]]; then
  # Full-model Ray cluster: start on .15, then on .13 and .14 with NODE_IP set.
  unset RAY_ADDRESS
  if [[ "$role" == head ]]; then
    ray start --head --node-ip-address="$HEAD_IP" --port=6379 \
      --resources='{"NPU":16}' --temp-dir=/dev/shm/ray-dsv4-full \
      --disable-usage-stats --include-dashboard=false
  else
    : "${NODE_IP:?Set NODE_IP to 22.6.7.13 or 22.6.7.14 on this worker}"
    if [[ "$NODE_IP" != 22.6.7.13 && "$NODE_IP" != 22.6.7.14 ]]; then
      echo 'NODE_IP must be 22.6.7.13 or 22.6.7.14' >&2
      exit 2
    fi
    ray start --address="$HEAD_IP:6379" --node-ip-address="$NODE_IP" \
      --resources='{"NPU":16}' --disable-usage-stats
  fi
  exit 0
else
  export RAY_ADDRESS="$HEAD_IP:6379"
  export ACTOR_MODEL=/highcode/shared_data/DeepSeek-V4-Flash-0731-BF16/DeepSeek-V4-Flash-0731-BF16/DeepSeek-V4-Flash-0731-BF16_20260806_01
  export ROLLOUT_MODEL=/highcode/shared_data/DeepSeek-V4-Flash-0731-W8A8-HW/DeepSeek-V4-Flash-0731-W8A8-HW_20260803_01
  export NPUS_PER_NODE=16 ACTOR_NPUS=32 ACTOR_EP=32 ROLLOUT_START_RANK=32 ROLLOUT_TP=16
  export MAX_MODEL_LEN=4096 MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=4096
  export GPU_MEMORY_UTILIZATION=0.85 TWINKLE_VLLM_BUCKET_SIZE_MB=64
  export STEPS=${STEPS:-3} BATCH_SIZE=${BATCH_SIZE:-64} MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
  ray status --address="$RAY_ADDRESS"
  python - <<'PY'
import os
import ray

ray.init(address=os.environ['RAY_ADDRESS'], ignore_reinit_error=True)
alive = [node for node in ray.nodes() if node['Alive']]
available = ray.cluster_resources().get('NPU', 0)
print(f'Ray preflight: alive_nodes={len(alive)}, NPU_resources={available}')
if len(alive) != 3 or available < 48:
    raise RuntimeError('Full-model run requires three alive Ray nodes and at least 48 NPU resources')
ray.shutdown()
PY
fi

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
TEST_ROOT=$(mktemp -d "$REPORT_ROOT/${role}_XXXXXXXX")
export REPORT_DIR="$TEST_ROOT/report"
echo "Report: $REPORT_DIR"
python -u -m cookbook.rl.grpo.dsv4_lora_npu 2>&1 | tee "$TEST_ROOT/grpo.log"
