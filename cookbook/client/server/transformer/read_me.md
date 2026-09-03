# DeepSeek-V4 四层模型：FSDP/EP/NPU GMM 排查手册

本文档用于定位 Twinkle 加载 DeepSeek-V4 后基座生成异常的问题。整套检查只需要从正式 BF16 模型截取的前四层，不依赖模型能够正常回答自然语言。

排查目标是区分：

1. checkpoint 或 Transformers 转换错误；
2. 两个节点加载出的源权重不一致；
3. FSDP node-local 权重分发错误；
4. EP 专家切分或 AllToAll 错误；
5. NPU grouped-matmul 专家权重方向错误。

四层模型无法形成有意义的自然语言回答是正常现象。所有结论必须依据权重诊断日志和最后一个位置的 logits，不能依据生成文本是否可读。

## 本次修改包含的文件

核心修复及诊断开关：

- `src/twinkle/kernel/ops/moe/npu.py`
  - 同时根据 `gate_up_proj` 和 `down_proj` 判断 Transformers `[E,out,in]` 与 GMM `[E,in,out]` 布局。
  - 修复 DeepSeek-V4 中 `hidden_size == 2 * moe_intermediate_size` 导致方阵 `gate_up_proj` 被错误识别的问题。
- `src/twinkle/kernel/ops/ep/__init__.py`
  - 新增 `TWINKLE_EP_FORCE_LOOP=1`，可强制使用逐专家 `F.linear` 参考实现。
- `src/twinkle/model/transformers/moe/expert_parallel.py`
  - 新增首轮 EP 路由、split、专家区间及输出诊断日志。
- `src/twinkle/model/transformers/strategy/native_fsdp.py`
  - 新增 node-local 源权重和 EP 本地专家切片诊断日志。

四层诊断配置及启动脚本：

- `cookbook/client/server/transformer/server_config_dsv4_4layer_diag_no_ep.yaml`
- `cookbook/client/server/transformer/server_config_dsv4_4layer_diag_ep.yaml`
- `cookbook/client/server/transformer/run_dsv4_4layer_ep_diagnostic.sh`
- `cookbook/client/server/transformer/run_dsv4_0731_npu_2node_2npu.sh`
  - 现在支持通过 `TWINKLE_SERVER_CONFIG_PATH` 选择配置。

诊断工具：

- `cookbook/client/server/transformer/diagnostics/test_dsv4_npu_gmm_layout.py`
- `cookbook/client/server/transformer/diagnostics/probe_dsv4_4layer_logits_ray.py`
- `cookbook/client/server/transformer/diagnostics/probe_dsv4_4layer_logits_cli.py`
- `cookbook/client/server/transformer/diagnostics/probe_dsv4_4layer_logits.py`
- `cookbook/client/server/transformer/diagnostics/compare_dsv4_4layer_logits.py`

单元测试：

- `tests/kernel/ops/test_moe.py`

## 1. 四层模型要求

必须使用从正式 DeepSeek-V4-Flash BF16 checkpoint 截取的前四层，不能使用随机初始化模型。配置至少需要保持：

```text
num_hidden_layers=4
hidden_size=4096
moe_intermediate_size=2048
n_routed_experts=256
num_experts_per_tok=6
dtype=bfloat16
```

重点是保留：

```text
hidden_size == 2 * moe_intermediate_size == 4096
```

这正是原实现中方阵 `gate_up_proj` 布局误判的触发条件。

两个节点必须看到相同的绝对模型路径，并使用相同的 Twinkle、Transformers、PyTorch、torch-npu 和 CANN 版本。

启动前在两个节点分别检查：

```bash
PROJECT_DIR=/opt/twinkle
MODEL_DIR=/nas/disk6/ljl/DeepSeek-V4-Flash-0731-BF16-4layers

cd "$PROJECT_DIR"
git rev-parse HEAD
test -f "$MODEL_DIR/config.json"
test -f "$MODEL_DIR/model.safetensors.index.json"
python3 - <<'PY'
import torch
import transformers
import torch_npu

print('torch=', torch.__version__)
print('torch_npu=', torch_npu.__version__)
print('transformers=', transformers.__version__)
PY
```

两个节点的输出必须一致。

## 2. 避免占用 Pod ephemeral-storage

Ray 的 Unix socket 路径必须短，但缓存又不应写入只有 10 GiB 的容器临时盘。已知容器的 `/dev/shm` 有 800 GiB 时，建议把 Ray 临时目录放到共享内存；日志和结果仍写入存储卷：

```bash
mkdir -p /dev/shm/rh
mkdir -p /nas/disk6/ljl/dsv4_ep_diag/tmp
mkdir -p /nas/disk6/ljl/dsv4_ep_diag/logs
mkdir -p /nas/disk6/ljl/dsv4_ep_diag/results
```

之后使用：

```bash
export RAY_TMPDIR=/dev/shm/rh
export TMPDIR=/nas/disk6/ljl/dsv4_ep_diag/tmp
export RAY_ROTATION_MAX_BYTES=20971520
export RAY_ROTATION_BACKUP_COUNT=1
```

不要把 Ray 临时目录设成很长的路径，否则可能再次触发 AF_UNIX 107 字节路径上限。

## 3. 先做单卡 NPU GMM 数值测试

该步骤不启动 Ray，也不加载四层模型。它使用保持 DeepSeek-V4 比例的微型专家，直接比较：

```text
F.linear 参考结果
vs
npu_grouped_matmul 结果
```

在一台 NPU 机器执行：

```bash
cd /opt/twinkle

ASCEND_RT_VISIBLE_DEVICES=0 \
PYTHONPATH=/opt/twinkle/src \
python3 cookbook/client/server/transformer/diagnostics/test_dsv4_npu_gmm_layout.py \
  --device npu:0 \
  --output /nas/disk6/ljl/dsv4_ep_diag/results/npu_gmm_layout.json
```

成功时：

```text
passed=true
max_abs_diff 和 mean_abs_diff 在 BF16 允许范围内
old_bug_max_abs_diff 明显大于修复后的 max_abs_diff
```

如果这里失败，不要启动分布式服务，先处理 NPU GMM 与 `F.linear` 的数值差异。

## 3A. 多机 Ray CLI 基准（推荐）

这里的“CLI”表示在 Head 节点运行一个普通 Python driver，而不是必须使用
torchrun。driver 直接通过 Ray 创建两节点 NPU 模型 Actor：

```text
Python driver
  -> Ray placement group
  -> 两节点、每节点两个 MultiLoraTransformersModel Actor
  -> native_fsdp + HCCL + EP
  -> logits 返回 driver
```

该模式不启动 Ray Serve、GatewayServer 或 ModelManagement，不创建 HTTP session
或租户，因此不受服务端槽位、心跳和接口限流影响。模型内部会创建一个禁用 delta
的诊断 adapter，以复用生产 `MultiLoraTransformersModel` 在 FSDP 包装前安装槽位和
对齐 dtype 的相同初始化路径；该 adapter 不登记到 ServerState 容量中。该模式仍然
覆盖 Ray 多机 Actor、node-local 权重加载、`_broadcast_sharded_state_dict`、FSDP
和 EP。

先创建短临时目录和持久化结果目录：

```bash
mkdir -p /dev/shm/rh
mkdir -p /nas/disk6/ljl/dsv4_ep_diag/tmp
mkdir -p /nas/disk6/ljl/dsv4_ep_diag/logs
mkdir -p /nas/disk6/ljl/dsv4_ep_diag/results
```

Head 节点：

```bash
cd /opt/twinkle

export HEAD_IP=172.61.10.254
export ASCEND_RT_VISIBLE_DEVICES=0,1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export PYTHONPATH=/opt/twinkle/src
export TMPDIR=/nas/disk6/ljl/dsv4_ep_diag/tmp

ray stop --force || true
ray start \
  --head \
  --node-ip-address="$HEAD_IP" \
  --port=6379 \
  --num-cpus="$(nproc)" \
  --resources='{"NPU": 2}' \
  --temp-dir=/dev/shm/rh \
  --disable-usage-stats \
  --include-dashboard=false
```

Worker 节点：

```bash
cd /opt/twinkle

export HEAD_IP=172.61.10.254
export NODE_IP=172.61.12.251
export ASCEND_RT_VISIBLE_DEVICES=0,1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export PYTHONPATH=/opt/twinkle/src
export TMPDIR=/nas/disk6/ljl/dsv4_ep_diag/tmp

ray stop --force || true
ray start \
  --address="$HEAD_IP:6379" \
  --node-ip-address="$NODE_IP" \
  --num-cpus="$(nproc)" \
  --resources='{"NPU": 2}' \
  --temp-dir=/dev/shm/rh \
  --disable-usage-stats
```

Worker 加入后，只在 Head 节点运行 driver：

```bash
cd /opt/twinkle

export DSV4_MODEL_ID=/nas/disk6/ljl/DeepSeek-V4-Flash-0731-BF16-4layers
export ASCEND_RT_VISIBLE_DEVICES=0,1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export PYTHONPATH=/opt/twinkle/src
export TMPDIR=/nas/disk6/ljl/dsv4_ep_diag/tmp

python3 cookbook/client/server/transformer/diagnostics/probe_dsv4_4layer_logits_ray.py \
  --ray-address auto \
  --world-size 4 \
  --nproc-per-node 2 \
  --mode no_ep \
  --output-dir /nas/disk6/ljl/dsv4_ep_diag/results \
  2>&1 | tee /nas/disk6/ljl/dsv4_ep_diag/logs/ray_no_ep.log
```

完成后，Ray 集群保持运行。分别重新执行 driver，只修改模式和日志名：

```bash
python3 cookbook/client/server/transformer/diagnostics/probe_dsv4_4layer_logits_ray.py \
  --ray-address auto --world-size 4 --nproc-per-node 2 \
  --mode ep_loop \
  --output-dir /nas/disk6/ljl/dsv4_ep_diag/results \
  2>&1 | tee /nas/disk6/ljl/dsv4_ep_diag/logs/ray_ep_loop.log

python3 cookbook/client/server/transformer/diagnostics/probe_dsv4_4layer_logits_ray.py \
  --ray-address auto --world-size 4 --nproc-per-node 2 \
  --mode ep_gmm \
  --output-dir /nas/disk6/ljl/dsv4_ep_diag/results \
  2>&1 | tee /nas/disk6/ljl/dsv4_ep_diag/logs/ray_ep_gmm.log
```

每个 driver 结束时只删除自己创建的模型 Actor 和 placement group，不执行
`ray stop`。输出文件为：

```text
ray_no_ep_last_logits.pt/json
ray_ep_loop_last_logits.pt/json
ray_ep_gmm_last_logits.pt/json
```

如果首次 forward 报错：

```text
AssertionError: FSDP expects uniform original parameter dtype but got
{torch.bfloat16, torch.float32}
```

说明运行的是旧版 Ray probe：它实例化了纯 `TransformersModel`，没有进入生产
Multi-LoRA 的槽位安装和 `_ensure_lora_dtype()` 路径。使用更新后的
`probe_dsv4_4layer_logits_ray.py`；它直接构造 `MultiLoraTransformersModel`，先安装
禁用的诊断 adapter，再进入 `_lazy_wrap_model()`。无需修改公共
`TransformersModel._lazy_wrap_model()`，Ray 集群也无需重启。

比较方式：

```bash
python3 cookbook/client/server/transformer/diagnostics/compare_dsv4_4layer_logits.py \
  /nas/disk6/ljl/dsv4_ep_diag/results/ray_no_ep_last_logits.pt \
  /nas/disk6/ljl/dsv4_ep_diag/results/ray_ep_loop_last_logits.pt \
  --output /nas/disk6/ljl/dsv4_ep_diag/results/compare_ray_no_ep_vs_ep_loop.json

python3 cookbook/client/server/transformer/diagnostics/compare_dsv4_4layer_logits.py \
  /nas/disk6/ljl/dsv4_ep_diag/results/ray_ep_loop_last_logits.pt \
  /nas/disk6/ljl/dsv4_ep_diag/results/ray_ep_gmm_last_logits.pt \
  --output /nas/disk6/ljl/dsv4_ep_diag/results/compare_ray_ep_loop_vs_ep_gmm.json
```

## 3B. 可选的本地/torchrun 基准

CLI 模式直接在 torchrun 进程中构造 `MultiLoraTransformersModel`，不启动 Ray、
Gateway 或 ModelManagement，也不会创建 session、租户或登记服务端 LoRA 容量。
它会在每个进程内安装一个禁用 delta 的诊断 adapter，使 dtype 对齐和 FSDP 包装
顺序与生产服务一致。该步骤验证：

```text
Transformers checkpoint 转换
native_fsdp 包装和权重加载
EP 专家切分与通信
loop/GMM 专家计算
```

它不会验证 Ray Actor 和服务端 `_broadcast_sharded_state_dict` 路径，因此 CLI
通过后仍需执行后面的 C/S 对照。

### 单节点四张 NPU

```bash
cd /opt/twinkle

export DSV4_MODEL_ID=/nas/disk6/ljl/DeepSeek-V4-Flash-0731-BF16-4layers
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0
export PYTHONPATH=/opt/twinkle/src

torchrun --standalone --nproc-per-node=4 \
  cookbook/client/server/transformer/diagnostics/probe_dsv4_4layer_logits_cli.py \
  --mode no_ep \
  --output-dir /nas/disk6/ljl/dsv4_ep_diag/results

torchrun --standalone --nproc-per-node=4 \
  cookbook/client/server/transformer/diagnostics/probe_dsv4_4layer_logits_cli.py \
  --mode ep_loop \
  --output-dir /nas/disk6/ljl/dsv4_ep_diag/results

torchrun --standalone --nproc-per-node=4 \
  cookbook/client/server/transformer/diagnostics/probe_dsv4_4layer_logits_cli.py \
  --mode ep_gmm \
  --output-dir /nas/disk6/ljl/dsv4_ep_diag/results
```

每种模式必须启动一组新的 torchrun 进程，不能在同一 Python 进程内切换
`TWINKLE_EP_FORCE_LOOP`。

### 两节点、每节点两张 NPU

以下示例中 Head 为 `172.61.10.254`，Worker 为 `172.61.12.251`。先在 Head
执行 rank 0 命令，它会等待 Worker；然后在 Worker 执行 rank 1 命令。

Head：

```bash
cd /opt/twinkle

export DSV4_MODEL_ID=/nas/disk6/ljl/DeepSeek-V4-Flash-0731-BF16-4layers
export ASCEND_RT_VISIBLE_DEVICES=0,1
export HCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export PYTHONPATH=/opt/twinkle/src

torchrun \
  --nnodes=2 \
  --nproc-per-node=2 \
  --node-rank=0 \
  --master-addr=172.61.10.254 \
  --master-port=29610 \
  cookbook/client/server/transformer/diagnostics/probe_dsv4_4layer_logits_cli.py \
  --mode no_ep \
  --output-dir /nas/disk6/ljl/dsv4_ep_diag/results
```

Worker：

```bash
cd /opt/twinkle

export DSV4_MODEL_ID=/nas/disk6/ljl/DeepSeek-V4-Flash-0731-BF16-4layers
export ASCEND_RT_VISIBLE_DEVICES=0,1
export HCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export PYTHONPATH=/opt/twinkle/src

torchrun \
  --nnodes=2 \
  --nproc-per-node=2 \
  --node-rank=1 \
  --master-addr=172.61.10.254 \
  --master-port=29610 \
  cookbook/client/server/transformer/diagnostics/probe_dsv4_4layer_logits_cli.py \
  --mode no_ep \
  --output-dir /nas/disk6/ljl/dsv4_ep_diag/results
```

完成 `no_ep` 后，把两个节点命令中的 `--mode no_ep` 同时改成
`--mode ep_loop` 再运行一次，然后改成 `--mode ep_gmm`。上一组 torchrun 已经
完全退出后可以复用端口 `29610`；如果仍有残留进程，则先终止残留进程或换一个
未占用端口。

只使用一张 NPU 时只能验证 `no_ep`：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
PYTHONPATH=/opt/twinkle/src \
python3 cookbook/client/server/transformer/diagnostics/probe_dsv4_4layer_logits_cli.py \
  --mode no_ep \
  --model-id /nas/disk6/ljl/DeepSeek-V4-Flash-0731-BF16-4layers \
  --output-dir /nas/disk6/ljl/dsv4_ep_diag/results
```

CLI 输出：

```text
cli_no_ep_last_logits.pt/json
cli_ep_loop_last_logits.pt/json
cli_ep_gmm_last_logits.pt/json
```

报告中的 `max_rank_diff` 应接近 0。使用现有比较脚本比较 CLI 三组结果：

```bash
python3 cookbook/client/server/transformer/diagnostics/compare_dsv4_4layer_logits.py \
  /nas/disk6/ljl/dsv4_ep_diag/results/cli_no_ep_last_logits.pt \
  /nas/disk6/ljl/dsv4_ep_diag/results/cli_ep_loop_last_logits.pt \
  --output /nas/disk6/ljl/dsv4_ep_diag/results/compare_cli_no_ep_vs_ep_loop.json

python3 cookbook/client/server/transformer/diagnostics/compare_dsv4_4layer_logits.py \
  /nas/disk6/ljl/dsv4_ep_diag/results/cli_ep_loop_last_logits.pt \
  /nas/disk6/ljl/dsv4_ep_diag/results/cli_ep_gmm_last_logits.pt \
  --output /nas/disk6/ljl/dsv4_ep_diag/results/compare_cli_ep_loop_vs_ep_gmm.json
```

### 比较 memory_efficient_init 开关后的实际权重

最终 logits 不同只能证明初始化路径影响了结果。要定位具体参数，使用相同的
`ep_loop` 计算路径分别开启和关闭 `memory_efficient_init`，并增加
`--dump-weight-fingerprints`。诊断只读取每个 rank 的本地参数，并从每个参数均匀
采样最多 4096 个值，不会将完整权重保存到磁盘，也不会为了展平非连续参数而复制
完整权重。

开启：

```bash
torchrun --standalone --nproc-per-node=4 \
  cookbook/client/server/transformer/diagnostics/probe_dsv4_4layer_logits_cli.py \
  --mode ep_loop \
  --dump-weight-fingerprints \
  --output-dir /nas/disk6/ljl/dsv4_ep_diag/results/memory_efficient_on
```

关闭：

```bash
torchrun --standalone --nproc-per-node=4 \
  cookbook/client/server/transformer/diagnostics/probe_dsv4_4layer_logits_cli.py \
  --mode ep_loop \
  --disable-memory-efficient-init \
  --dump-weight-fingerprints \
  --output-dir /nas/disk6/ljl/dsv4_ep_diag/results/memory_efficient_off
```

每次运行会按 rank 生成：

```text
cli_ep_loop_weight_fingerprints_rank0.json
cli_ep_loop_weight_fingerprints_rank1.json
cli_ep_loop_weight_fingerprints_rank2.json
cli_ep_loop_weight_fingerprints_rank3.json
```

以关闭状态作为 reference 进行比较：

```bash
python3 cookbook/client/server/transformer/diagnostics/compare_dsv4_weight_fingerprints.py \
  /nas/disk6/ljl/dsv4_ep_diag/results/memory_efficient_off \
  /nas/disk6/ljl/dsv4_ep_diag/results/memory_efficient_on \
  --mode ep_loop \
  --output /nas/disk6/ljl/dsv4_ep_diag/results/compare_memory_init_weights.json \
  || true
```

重点查看比较报告中的以下字段：

```text
metadata_mismatches    shape、stride、dtype 或连续性不同
value_mismatches       参数采样值不同
first_difference       第一个不同采样值的本地 flat index 和多维坐标
rank                   不同权重所在的 EP/FSDP rank
name                   不同权重的完整参数名
```

如果第一个不一致参数是 `embed_tokens`、attention、norm 或 `lm_head`，优先检查
普通 FSDP DTensor 重建。如果只有 `mlp.experts.gate_up_proj/down_proj` 不一致，优先
检查 pre-EP state capture 和 `_scatter_ep_expert_tensor()` 的专家切分。

还可以将 `cli_<mode>_last_logits.pt` 与 C/S 产生的
`<mode>_last_logits.pt` 比较。CLI 一致而 C/S 不一致时，重点排查 Ray/服务端权重
分发；CLI 本身已经不一致时，重点排查 FSDP/EP/GMM。

## 4. 三组服务端对照模式

三组模式只改变 EP 和专家计算方式：

| 模式 | FSDP | EP | 专家计算 |
|---|---:|---:|---|
| `no_ep` | 开启 | 关闭 | Transformers 原始前向 |
| `ep_loop` | 开启 | 开启 | 强制逐专家 `F.linear` |
| `ep_gmm` | 开启 | 开启 | NPU grouped-matmul |

两份 YAML 除 EP 设置外保持一致：

```text
两节点
每节点 2 张 NPU
world_size=4
fsdp_size=4
ep_size=4（仅 EP 模式）
memory_efficient_init=true
max_loras=1
max_length=512
```

EP=4 时，256 个专家的预期区间为：

```text
rank 0 / ep_rank 0: experts   0..63
rank 1 / ep_rank 1: experts  64..127
rank 2 / ep_rank 2: experts 128..191
rank 3 / ep_rank 3: experts 192..255
```

## 5. 两个节点的公共环境

以下示例使用：

```text
Head:   172.61.10.254
Worker: 172.61.12.251
网卡:   eth0
```

如果实际环境使用 `bond0` 或其他地址，只修改环境变量，不需要修改脚本或 YAML。

Head 节点执行：

```bash
cd /opt/twinkle

export DSV4_MODEL_ID=/nas/disk6/ljl/DeepSeek-V4-Flash-0731-BF16-4layers
export HEAD_IP=172.61.10.254
export WORKER_IP=172.61.12.251
export NODE_IP=172.61.10.254
export NETWORK_IFACE=eth0
export ASCEND_RT_VISIBLE_DEVICES=0,1
export RESET_RAY=1
export RAY_TMPDIR=/dev/shm/rh
export TMPDIR=/nas/disk6/ljl/dsv4_ep_diag/tmp
export RAY_ROTATION_MAX_BYTES=20971520
export RAY_ROTATION_BACKUP_COUNT=1
```

Worker 节点执行：

```bash
cd /opt/twinkle

export DSV4_MODEL_ID=/nas/disk6/ljl/DeepSeek-V4-Flash-0731-BF16-4layers
export HEAD_IP=172.61.10.254
export WORKER_IP=172.61.12.251
export NODE_IP=172.61.12.251
export NETWORK_IFACE=eth0
export ASCEND_RT_VISIBLE_DEVICES=0,1
export RESET_RAY=1
export RAY_TMPDIR=/dev/shm/rh
export TMPDIR=/nas/disk6/ljl/dsv4_ep_diag/tmp
export RAY_ROTATION_MAX_BYTES=20971520
export RAY_ROTATION_BACKUP_COUNT=1
```

必须确保两个节点选择相同模式。

## 6. 模式一：启动 no-EP 基准

先在 Head 节点启动；脚本会等待 Worker 加入：

```bash
nohup bash cookbook/client/server/transformer/run_dsv4_4layer_ep_diagnostic.sh head no_ep \
  >/nas/disk6/ljl/dsv4_ep_diag/logs/no_ep_head.log 2>&1 </dev/null &

echo "head pid=$!"
```

然后在 Worker 节点执行：

```bash
nohup bash cookbook/client/server/transformer/run_dsv4_4layer_ep_diagnostic.sh worker no_ep \
  >/nas/disk6/ljl/dsv4_ep_diag/logs/no_ep_worker.log 2>&1 </dev/null &
```

服务启动完成后，在 Head 节点保存基准 logits：

```bash
cd /opt/twinkle

PYTHONPATH=/opt/twinkle/src \
python3 cookbook/client/server/transformer/diagnostics/probe_dsv4_4layer_logits.py \
  --mode no_ep \
  --server-url http://127.0.0.1:8000 \
  --output-dir /nas/disk6/ljl/dsv4_ep_diag/results
```

诊断配置的 `max_loras` 为 1。一次 probe 结束后，客户端只会停止 session
心跳；服务端会在 `adapter_timeout: 60` 到期并经过下一轮清理检查后释放临时
adapter，通常需要最多约 70 秒。probe 默认会等待空闲槽位 90 秒，因此连续执行
不同模式时不需要重启服务。可按需调整：

```bash
--capacity-wait-seconds 120 --capacity-poll-seconds 5
```

只有等待超时且容量仍为 `used_loras=1` 时，才需要检查 ModelManagement 的
adapter 清理日志或重启诊断服务。

如果旧版 probe 持续输出 `used=0/0`，表示 ModelManagement 尚未通过首个模型请求
惰性注册容量，并不表示 LoRA 已占满。使用修复后的 probe：它会先调用模型
`/create`，再查询容量，正常应看到 `used=0/1` 或在上一个 adapter 未清理时看到
`used=1/1`。

四层诊断配置使用 `rps_limit: 100` 和 `tps_limit: 100000`。probe 会连续调用
`/create`、`add_adapter`、`set_processor` 和 `forward_only`；若仍使用旧配置中的
`rps_limit: 1`，这些串行初始化请求也会触发每用户滑动窗口限流。修改 YAML 后必须
重启诊断服务，运行中的 ModelManagement 不会动态重载 `queue_config`。

生成：

```text
no_ep_last_logits.pt
no_ep_last_logits.json
```

## 7. 模式二：启动 EP-loop 基准

切换模式前，两个节点都要停止旧 Ray。不能复用旧模型进程，因为 NPU 专家权重存在进程级缓存。

两个节点分别执行：

```bash
ray stop --force
```

Head：

```bash
nohup bash cookbook/client/server/transformer/run_dsv4_4layer_ep_diagnostic.sh head ep_loop \
  >/nas/disk6/ljl/dsv4_ep_diag/logs/ep_loop_head.log 2>&1 </dev/null &
```

Worker：

```bash
nohup bash cookbook/client/server/transformer/run_dsv4_4layer_ep_diagnostic.sh worker ep_loop \
  >/nas/disk6/ljl/dsv4_ep_diag/logs/ep_loop_worker.log 2>&1 </dev/null &
```

服务启动后，在 Head 保存 logits：

```bash
cd /opt/twinkle

PYTHONPATH=/opt/twinkle/src \
python3 cookbook/client/server/transformer/diagnostics/probe_dsv4_4layer_logits.py \
  --mode ep_loop \
  --server-url http://127.0.0.1:8000 \
  --output-dir /nas/disk6/ljl/dsv4_ep_diag/results
```

日志中必须出现：

```text
TWINKLE_EP_FORCE_LOOP=1
forcing the per-expert F.linear loop
```

## 8. 模式三：启动 EP-GMM

再次在两个节点执行：

```bash
ray stop --force
```

Head：

```bash
nohup bash cookbook/client/server/transformer/run_dsv4_4layer_ep_diagnostic.sh head ep_gmm \
  >/nas/disk6/ljl/dsv4_ep_diag/logs/ep_gmm_head.log 2>&1 </dev/null &
```

Worker：

```bash
nohup bash cookbook/client/server/transformer/run_dsv4_4layer_ep_diagnostic.sh worker ep_gmm \
  >/nas/disk6/ljl/dsv4_ep_diag/logs/ep_gmm_worker.log 2>&1 </dev/null &
```

服务启动后保存 logits：

```bash
cd /opt/twinkle

PYTHONPATH=/opt/twinkle/src \
python3 cookbook/client/server/transformer/diagnostics/probe_dsv4_4layer_logits.py \
  --mode ep_gmm \
  --server-url http://127.0.0.1:8000 \
  --output-dir /nas/disk6/ljl/dsv4_ep_diag/results
```

日志中必须出现：

```text
EP experts compute: using NPU grouped matmul
```

不能出现强制 loop 日志。

## 9. 比较三组 logits

在任意能访问共享结果目录、并安装 PyTorch 的节点执行：

```bash
cd /opt/twinkle

python3 cookbook/client/server/transformer/diagnostics/compare_dsv4_4layer_logits.py \
  /nas/disk6/ljl/dsv4_ep_diag/results/no_ep_last_logits.pt \
  /nas/disk6/ljl/dsv4_ep_diag/results/ep_loop_last_logits.pt \
  --output /nas/disk6/ljl/dsv4_ep_diag/results/compare_no_ep_vs_ep_loop.json
```

然后比较 loop 与 GMM：

```bash
python3 cookbook/client/server/transformer/diagnostics/compare_dsv4_4layer_logits.py \
  /nas/disk6/ljl/dsv4_ep_diag/results/ep_loop_last_logits.pt \
  /nas/disk6/ljl/dsv4_ep_diag/results/ep_gmm_last_logits.pt \
  --output /nas/disk6/ljl/dsv4_ep_diag/results/compare_ep_loop_vs_ep_gmm.json
```

默认判定阈值：

```text
rtol=1e-2
atol=2e-2
```

不要仅根据 SHA256 判断。BF16 的并行执行顺序可能导致极小数值差异，应以 `allclose`、`max_abs_diff`、`mean_abs_diff` 和 top-20 重合度为准。

## 10. 结论判定

| 对比结果 | 结论 |
|---|---|
| `no_ep ≈ ep_loop`，`ep_loop ≉ ep_gmm` | NPU GMM 权重布局或计算错误 |
| `no_ep ≉ ep_loop`，`ep_loop ≈ ep_gmm` | EP 专家切分、AllToAll 或回排错误 |
| 三组都不一致 | EP 与 GMM 可能同时有问题 |
| `no_ep` 自身不稳定 | checkpoint 转换或 FSDP 权重分发错误 |
| 三组都一致 | 四层 FSDP/EP/GMM 链路通过，可以验证完整模型 |

脚本在 `allclose=false` 时返回非零退出码，这是预期的诊断行为，不表示脚本自身崩溃。

## 11. 查看权重分发和 EP 日志

诊断配置默认设置：

```bash
TWINKLE_EP_DIAGNOSTICS=1
```

查看日志：

```bash
grep -h '\[EP_DIAG\]' /nas/disk6/ljl/dsv4_ep_diag/logs/ep_*_head.log \
  >/nas/disk6/ljl/dsv4_ep_diag/results/ep_diagnostics.txt
```

重点检查以下日志。

### 11.1 两个节点的源权重

每个 node-local source rank 都会输出：

```text
local_source=<rank>
param=<参数名>
full_shape=<完整专家形状>
source_preview=<采样值>
```

相同参数在两个 node-local source 上必须满足：

```text
full_shape 相同
source_preview 相同
```

如果不同，优先检查两个节点：

- 是否使用相同模型目录；
- 模型文件是否一致；
- Transformers/torch版本是否一致；
- 是否实际运行同一个 Twinkle commit。

### 11.2 EP 专家范围

每个rank会输出：

```text
rank=<global rank>
ep_rank=<EP rank>
expert_range=[start,end)
local_shape=<本地权重形状>
local_preview=<本地切片采样值>
```

EP=4时必须严格对应：

```text
rank0 -> [0,64)
rank1 -> [64,128)
rank2 -> [128,192)
rank3 -> [192,256)
```

本地权重预期形状：

```text
gate_up_proj: [64,4096,4096]
down_proj:    [64,4096,2048]
```

### 11.3 Router与AllToAll split

每层第一次前向会输出：

```text
selected_experts
routing_weights
input_splits
output_splits
output_finite
```

固定输入下，各rank在AllToAll前的 `selected_experts` 和 `routing_weights` 应一致。代码还会验证：

```text
sum(input_splits) == token_count * num_experts_per_tok
```

如果该条件不成立，诊断模式会直接报错退出。

## 12. 单元测试和静态检查

在安装了项目依赖的环境执行：

```bash
cd /opt/twinkle

python3 -m pytest -q tests/kernel/ops/test_moe.py

python3 -m py_compile \
  cookbook/client/server/transformer/diagnostics/test_dsv4_npu_gmm_layout.py \
  cookbook/client/server/transformer/diagnostics/probe_dsv4_4layer_logits_ray.py \
  cookbook/client/server/transformer/diagnostics/probe_dsv4_4layer_logits_cli.py \
  cookbook/client/server/transformer/diagnostics/probe_dsv4_4layer_logits.py \
  cookbook/client/server/transformer/diagnostics/compare_dsv4_4layer_logits.py \
  cookbook/client/server/transformer/diagnostics/compare_dsv4_weight_fingerprints.py

bash -n \
  cookbook/client/server/transformer/run_dsv4_0731_npu_2node_2npu.sh \
  cookbook/client/server/transformer/run_dsv4_4layer_ep_diagnostic.sh
```

## 13. 完整模型的最终验证

只有满足以下条件后才启动43层完整模型：

```text
单卡 NPU GMM 与 F.linear 一致
no_ep logits ≈ ep_loop logits
ep_loop logits ≈ ep_gmm logits
两个 node-local source 的权重采样一致
EP 专家范围正确
路由 split 不变量通过
```

完整模型启动时应关闭诊断开关：

```bash
export TWINKLE_EP_FORCE_LOOP=0
export TWINKLE_EP_DIAGNOSTICS=0
```

然后按顺序验证：

1. 不训练、不注册旧LoRA，先检查基座生成；
2. 注册全零LoRA，确认它与 `disable_lora=True` 逐token一致；
3. 修复前在错误EP前向下训练的LoRA不能作为正确性基准；
4. 基座确认正常后重新训练LoRA；
5. 对比重新训练后的 base/logprob/生成结果。

## 14. 清理

诊断结束后在两个节点执行：

```bash
ray stop --force
```

结果位于共享卷：

```text
/nas/disk6/ljl/dsv4_ep_diag/results
/nas/disk6/ljl/dsv4_ep_diag/logs
```

确认不再需要后可手动清理共享卷中的诊断日志和结果。`/dev/shm/rh` 仅用于当前 Pod 的 Ray 临时文件，Pod 删除后不会保留。

## 15. 路由 Management 类型 NameError

如果应用注册路由时出现以下任意错误：

```text
NameError: name 'GatewayServer' is not defined
NameError: name 'ModelManagement' is not defined
NameError: name 'SamplerManagement' is not defined
NameError: name 'ProcessorManagement' is not defined
```

说明运行版本中的 handler 只在 `TYPE_CHECKING` 分支导入了对应的 Management
类型，但 FastAPI 在运行时解析了路由参数的类型注解。这发生在服务配置加载阶段，
与模型权重、FSDP 和 EP 无关。

当前源码已在 Gateway、Model、Sampler 和 Processor 的 handler 中提供无循环导入
的运行时类型别名，包括：

```text
src/twinkle/server/gateway/openai_handlers.py
src/twinkle/server/gateway/tinker_handlers.py
src/twinkle/server/gateway/twinkle_handlers.py
src/twinkle/server/model/tinker_handlers.py
src/twinkle/server/model/twinkle_handlers.py
src/twinkle/server/sampler/tinker_handlers.py
src/twinkle/server/sampler/twinkle_handlers.py
src/twinkle/server/processor/twinkle_handlers.py
```

将修复后的源码同步到 head 和 worker 使用的相同 Twinkle 目录，然后重新启动服务。
可先做静态检查：

```bash
cd /opt/twinkle
python3 -m compileall -q \
  src/twinkle/server/gateway \
  src/twinkle/server/model \
  src/twinkle/server/sampler \
  src/twinkle/server/processor
```

如果错误仍然存在，先确认容器实际导入的源码位置：

```bash
cd /opt/twinkle
PYTHONPATH=/opt/twinkle/src python3 - <<'PY'
import inspect
import twinkle.server.gateway.openai_handlers as module

print(inspect.getfile(module))
print('GatewayServer runtime alias:', module.GatewayServer)
PY
```

输出文件应来自 `/opt/twinkle/src/twinkle/server/gateway/`，且不应再次出现
`NameError`。
