#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
cd "$PROJECT_DIR"

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export TWINKLE_TRUST_REMOTE_CODE=1
export TWINKLE_FAIL_FAST=1
export TOKENIZERS_PARALLELISM=true
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-0}"

MODEL_PATH=/nas/disk1/random-deepseek-v4-4b
CONFIG_PATH="$SCRIPT_DIR/server_config_dsv4_0731_npu.yaml"

test -f "$MODEL_PATH/config.json"
test -f "$MODEL_PATH/tokenizer.json"

python3 - <<'PY'
import os
import torch
import torch_npu  # noqa: F401

visible = [item.strip() for item in os.environ['ASCEND_RT_VISIBLE_DEVICES'].split(',') if item.strip()]
if len(visible) != 2:
    raise SystemExit(f'需要两张 NPU，当前 ASCEND_RT_VISIBLE_DEVICES={visible}')
if not torch.npu.is_available():
    raise SystemExit('torch.npu.is_available() 为 False')
if torch.npu.device_count() < 2:
    raise SystemExit(f'当前进程只能看到 {torch.npu.device_count()} 张 NPU')
print(f'NPU 检查通过：visible={visible}, device_count={torch.npu.device_count()}')
PY

if ! ray status >/dev/null 2>&1; then
    ray start \
        --head \
        --num-cpus="${TWINKLE_RAY_CPUS:-8}" \
        --resources='{"NPU": 2}' \
        --disable-usage-stats \
        --include-dashboard=false
fi

python3 - <<'PY'
import ray

ray.init(address='auto', logging_level='ERROR')
npu_count = float(ray.cluster_resources().get('NPU', 0))
ray.shutdown()
if npu_count < 2:
    raise SystemExit(
        f'当前 Ray 集群只有 {npu_count:g} 个 NPU 资源。'
        '请确认没有其他任务占用后执行 ray stop --force，再重新运行本脚本。'
    )
print(f'Ray NPU 资源检查通过：NPU={npu_count:g}')
PY

python3 -m twinkle.server check-config --config "$CONFIG_PATH"
exec python3 "$SCRIPT_DIR/server_dsv4_0731_npu.py"
