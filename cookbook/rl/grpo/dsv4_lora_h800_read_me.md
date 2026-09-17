# DeepSeek-V4 LoRA RL 操作手册

当前资源：H800 一台；Ascend A3 两台，每台 16 张 NPU。
先用同源四层模型验证权重同步，再跑三轮 GRPO。四层模型不用于评估回答质量。

- H800：单机取 4 张卡，actor 2 卡、rollout TP=2。
- NPU：四层双机测试，actor 4 卡、rollout TP=4，分别放在不同节点。
  当前资源管理按每节点 16 卡预留资源，测试时两台机器均需有足够空闲资源。
- 不需要启动 Twinkle HTTP 服务或单独执行 `vllm serve`，脚本会通过 Ray 创建实例。
- 已有专用 Ray 集群时只连接，不重复启动；不要停止或占用其他训练/推理任务。

以下命令使用 Bash，远程代码目录假定为 `/opt/twinkle`。
`/实际路径/`、IP 和网卡按真实环境替换。同一硬件章节按顺序、在同一会话执行。

## 1. 运行前准备

所有参与节点使用同一份当前源码及兼容的 Python/torch/vLLM 环境。
H800 使用已验证支持 DSV4 专家 LoRA 的 NVIDIA vLLM；NPU 使用已验证支持
W8A8_DYNAMIC 专家 LoRA 和原位更新的 vllm-ascend 环境。不在测试时自动升级依赖。

需同步到远程的运行入口：

- `cookbook/rl/grpo/dsv4_lora_h800.py`
- `cookbook/rl/grpo/dsv4_lora_npu.py`
- `cookbook/rl/grpo/dsv4_lora_sync_audit.py`
- `cookbook/rl/grpo/dsv4_lora_sync_audit_npu.py`

同时需要本次修改的 `src/twinkle` 代码，不能只复制这几个入口。
文件对照另需已有的 `output/convert_twinkle_dsv4_lora_for_vllm.py`；
NPU 对照还需同目录的 `output/diagnose_dsv4_quarot.py`。

在参与节点的终端准备公共环境：

```bash
cd /opt/twinkle
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TWINKLE_TRUST_REMOTE_CODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LOG_LEVEL=INFO
export RAY_ROTATION_MAX_BYTES=20971520
export RAY_ROTATION_BACKUP_COUNT=1
set -o pipefail
```

模型要求：

- actor 使用已验证可训练的 BF16 基座；rollout 使用同源、相同层集合的推理基座。
- 四层模型必须截取相同的原始层，不使用随机初始化模型；tokenizer 保持一致。
- NPU rollout 必须是此前验证过的 W8A8_DYNAMIC + QuaRot 模型；其
  `quant_model_description.json` 指向的 Q 文件必须保留，通常为
  `optional/quarot.safetensors`。训练基座也需在 rollout 节点可读，以读取对应 FFN norm。
- 文件对照的输出目录必须在 driver、actor、rollout 上以同一路径可读写。
  这是测试对照要求；正式在线同步不从 adapter 文件加载。

## 2. H800 单机操作

### 2.1 启动或连接 Ray

仅在该机未运行 Ray 时执行，示例选择物理卡 0–3：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
ray start --head --port=6379 --num-gpus=4 --temp-dir=/dev/shm/ray-dsv4-e2e
```

已有测试集群时跳过启动，并将下列地址改为实际地址：

```bash
export RAY_ADDRESS=127.0.0.1:6379
ray status --address="$RAY_ADDRESS"
```

Ray 运行目录不放 NFS。`/dev/shm` 会消耗 RAM，运行时同时监控其容量和容器内存；
日志轮转不限制整个 Ray session 的总大小。

### 2.2 四层同步对照

```bash
# 父目录暂按 /nas/disk1；如实际位置不同，请修改 MODEL_ROOT。
export MODEL_ROOT=/nas/disk1
export ACTOR_MODEL="$MODEL_ROOT/DeepSeek-V4-Flash-0731-4layers-bf16"
export ROLLOUT_MODEL="$MODEL_ROOT/DeepSeek-V4-Flash-0731-4layers-original"
export GPUS_PER_NODE=4
export ACTOR_GPUS=2
export ACTOR_EP=2
export ROLLOUT_START_RANK=2
export ROLLOUT_TP=2

export ACTOR_PRECISION=bf16
export LORA_R=8
export LORA_ALPHA=32
export MAX_MODEL_LEN=1024
export MAX_NUM_SEQS=2
export GPU_MEMORY_UTILIZATION=0.85

# 小 IPC 桶测试分片、缓冲区复用以及 TP ACK。
export TWINKLE_VLLM_BUCKET_SIZE_MB=1
export OFFLINE_LORA_CONVERTER=/opt/twinkle/output/convert_twinkle_dsv4_lora_for_vllm.py
export SYNC_REPEATS=20

export TEST_ROOT="$(mktemp -d /nas/disk6/ljl/dsv4_h800_e2e_XXXXXXXX)"
export AUDIT_DIR="$TEST_ROOT/sync_audit"
python -u -m cookbook.rl.grpo.dsv4_lora_sync_audit \
  2>&1 | tee "$TEST_ROOT/sync_audit.log"
```

`TEST_ROOT` 的父目录需存在且可写。`AUDIT_DIR` 子目录由脚本创建，不要提前创建。
成功时检查 `$AUDIT_DIR/summary.json` 中 `passed=true`、`backend=nvidia`。

### 2.3 四层三轮 GRPO

同步对照成功退出后，沿用上面的模型及设备配置：

```bash
export GSM8K_PATH=/nas/disk6/ljl/gsm8k
export REPORT_DIR="$TEST_ROOT/grpo"
export STEPS=3
export BATCH_SIZE=2
export NUM_GENERATIONS=2
export MAX_NEW_TOKENS=128
export LR=1e-5

python -u -m cookbook.rl.grpo.dsv4_lora_h800 \
  2>&1 | tee "$TEST_ROOT/grpo.log"
```

`GSM8K_PATH` 可为单个 Parquet 文件，或直接包含 `train*.parquet` 的目录。
本配置至少需要 6 条数据，不重复补齐。`REPORT_DIR` 不要提前创建。

**当前 H800 只有一台，本手册不提供 H800 双机命令。**
单机 8 张卡也不能照搬“actor 8 卡 + 独立 rollout 4 卡”的完整模型配置。
当前测试入口不做 actor/rollout 复用卡或卸载切换；现有资源先完成四层验收，
完整模型需另外确认两份基座与训练/推理开销能同时容纳。

## 3. NPU 双机操作

### 3.1 两台机器准备环境

两台机器均先执行第 1 节公共环境，再执行：

```bash
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

export HEAD_IP=172.61.10.149
export WORKER_IP=172.61.8.184
export NETWORK_IFACE=bond0
test -d "/sys/class/net/$NETWORK_IFACE"

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export GLOO_SOCKET_IFNAME="$NETWORK_IFACE"
export HCCL_SOCKET_IFNAME="$NETWORK_IFACE"
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=0
export RAY_TMPDIR=/dev/shm

python -c 'import torch, torch_npu; print("NPU available:", torch.npu.is_available(), "count:", torch.npu.device_count())'
```

每台应显示 NPU 可用且能看到 16 张卡；两机网卡名称不同时各自设置。
确保 Ray、HCCL 和同步控制端口互通。不要直接使用会额外启动 HTTP 服务的
`run_dsv4_0731_npu_multinode.sh`，这里只需要 Ray。

### 3.2 启动 head 和 worker

仅在对应节点尚未运行 Ray 时执行。

head 节点：

```bash
ray start --head \
  --node-ip-address="$HEAD_IP" --port=6379 \
  --resources='{"NPU":16}' \
  --temp-dir=/dev/shm/ray-dsv4-e2e \
  --disable-usage-stats --include-dashboard=false
```

worker 节点：

```bash
ray start --address="$HEAD_IP:6379" \
  --node-ip-address="$WORKER_IP" \
  --resources='{"NPU":16}' --disable-usage-stats
```

两台加入后，在 head 检查：

```bash
export RAY_ADDRESS="$HEAD_IP:6379"
ray status --address="$RAY_ADDRESS"
```

应有两个节点、合计 32 个 NPU 资源。已有集群必须已注册 `NPU` 自定义资源；
只注册 GPU 资源不适用。后面的 Python 命令只在 head 执行一次，不要两机各启动一份。

### 3.3 四层跨节点同步对照

actor 使用一个节点上的 4 张卡，rollout 使用另一个节点上的 4 张卡。
这是四层模型的跨机测试配置，不沿用完整模型的 rollout TP=8 配置。
两个模型都需准备可用的四层版本；不能拿四层 BF16 actor 配完整 43 层量化 rollout。
本次使用官方四层权重反量化后的 BF16 版本训练、W8A8 版本推理。
下面暂按模型父目录为 `/nas/disk1` 配置；若实际位置不同，修改 `MODEL_ROOT`。
两个节点应能访问以下路径，rollout 适配器还需读取 BF16 基座的 norm 权重。

```bash
export MODEL_ROOT=/nas/disk1
export ACTOR_MODEL="$MODEL_ROOT/DeepSeek-V4-Flash-0731-4layers-bf16"
export ROLLOUT_MODEL="$MODEL_ROOT/DeepSeek-V4-Flash-0731-4layers-w8a8"
test -f "$ACTOR_MODEL/config.json"
test -f "$ROLLOUT_MODEL/config.json"
export NPUS_PER_NODE=16
export ACTOR_NPUS=4
export ACTOR_EP=4
export ROLLOUT_START_RANK=16
export ROLLOUT_TP=4

export ACTOR_PRECISION=bf16
export LORA_R=8
export LORA_ALPHA=32
export MAX_MODEL_LEN=1024
export MAX_NUM_SEQS=2
export MAX_NUM_BATCHED_TOKENS=4096
export GPU_MEMORY_UTILIZATION=0.85

# 此环境变量名称沿用原框架；在 NPU 上也控制 SHM 传输桶。
export TWINKLE_VLLM_BUCKET_SIZE_MB=1
export TWINKLE_VLLM_IPC_TIMEOUT_S=1800
export TWINKLE_CKPT_HCCL_META_TIMEOUT_S=1800
export OFFLINE_LORA_CONVERTER=/opt/twinkle/output/convert_twinkle_dsv4_lora_for_vllm.py
test -f "$OFFLINE_LORA_CONVERTER"
test -f /opt/twinkle/output/diagnose_dsv4_quarot.py
export SYNC_REPEATS=20

export TEST_ROOT="$(mktemp -d /nas/disk6/ljl/dsv4_npu_e2e_XXXXXXXX)"
export AUDIT_DIR="$TEST_ROOT/sync_audit"
python -u -m cookbook.rl.grpo.dsv4_lora_sync_audit_npu \
  2>&1 | tee "$TEST_ROOT/sync_audit.log"
```

脚本选择 Ascend QuaRot 适配器；文件对照也会向原离线脚本传递
`--backend ascend --format 2d --base-model ACTOR_MODEL --quarot-model ROLLOUT_MODEL`，
不会拿未旋转的 NVIDIA adapter 作对照。若模型配方检查失败，应核对模型与 Q/norm，
不要跳过检查或改用 NVIDIA 适配器。

成功时检查 `$AUDIT_DIR/summary.json` 中 `passed=true`、`backend=ascend`。
首次初始化和旋转可能耗时较长；显存记录使用 NPU 接口，不调用 CUDA 统计。

### 3.4 四层三轮 GRPO

同步对照成功退出后，沿用 NPU 环境：

```bash
export GSM8K_PATH=/nas/disk6/ljl/gsm8k
export REPORT_DIR="$TEST_ROOT/grpo"
export STEPS=3
export BATCH_SIZE=2
export NUM_GENERATIONS=2
export MAX_NEW_TOKENS=128
export LR=1e-5

python -u -m cookbook.rl.grpo.dsv4_lora_npu \
  2>&1 | tee "$TEST_ROOT/grpo.log"
```

本配置每轮 4 个样本，匹配 4 个 actor rank。调整 actor 数量时，
`BATCH_SIZE * NUM_GENERATIONS` 必须能被 actor 的数据并行规模整除；
`ACTOR_EP` 还需整除 actor 数量和模型专家数。

**完整模型暂不照搬上述划卡。** 你之前完整 BF16 actor 已需要两台 A3；
若仍占满 32 张卡，就没有独立 rollout 可用的卡。仅启用
`memory_efficient_init` 不能保证运行期空出显存。先验收四层链路；
完整模型需实测 actor 能否缩到留有 rollout 空间的规模，或另行增加资源/支持卸载调度，
不能在同一批卡上直接再启动 rollout。

## 4. 看结果与重跑

### 同步对照

默认 20 次对照，输出 `sync_000.json` 到 `sync_019.json` 和 `summary.json`。

- 同 token 的文件/内存 logprob：`atol=rtol=1e-4`，不自动放宽。
- 非零测试 LoRA v0 → v1 → v0：恢复结果一致，v0/v1 分数不能完全相同。
- 默认检查各 TP rank 最后一次相对第 3 次的 live allocated 增长不超过 64 MiB；
  reserved、peak 和 host peak 也会记录，不能将进程高水位当成单次增量。
- 每轮穿插文件对照，最后再同步一次恢复内存安装。这不是纯在线压力测试，也不是
  已训练 LoRA 的效果评测；显存检查不能证明所有 CPU/GPU/NPU 内存均无泄漏。

需要分析时提供 `summary.json`、`sync_000.json`、`sync_001.json`、
`sync_002.json`、最后一个 `sync_*.json` 及对应日志。

### GRPO

预期生成 `round_0.json`、`round_1.json`、`round_2.json`、`final_sync.json`。
日志查看 `nonzero_advantages` 和训练返回值；JSON 保留采样 token/logprob、奖励及 advantage。

四层模型可能全部奖励相同、advantage 为零。此时只能说链路跑通，不能说明有效更新
或训练质量改善。采样固定 temperature=1、top_p=1、top_k=-1，KL 系数为 0；
只在设置 `FINAL_CHECKPOINT_DIR` 时保存最终 checkpoint，不参与在线同步。

### 失败与资源检查

- 第一处 traceback 优先保留；没有 `summary.json` 不算同步验收通过。
- 同步失败立即停止本轮，不继续采样。通信失败需重建本次测试进程，不自动重试。
- 重跑选择新的 `TEST_ROOT`，不要覆盖或删除旧报告。
- 上下文不足时调高 `MAX_MODEL_LEN`，先确认显存余量。
- 1 MiB 只改变 IPC/SHM 桶，不改变 Checkpoint Engine 原有的每桶 3 GiB 配置；
  完整 adapter、vLLM 缓存及转换副本仍占内存。速度测试可取消
  `TWINKLE_VLLM_BUCKET_SIZE_MB` 恢复原有默认值，但需重新检查显存和共享内存余量。
- 监控 `df -h /dev/shm`、`free -h`，H800 用 `nvidia-smi`，NPU 用 `npu-smi info`。

脚本的 CPU 回归不替代远程硬件结果。H800/NPU 实际传输、加载和训练均以本次远程
报告为准，不因本地测试通过就认定硬件验收完成。
