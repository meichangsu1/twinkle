# Sticky session 基准

`benchmark_sticky_session.py` 重放固定的多轮 token 输入，复用生产代码的
`_run_continous_work`、共享 `RequestRouter`、lease 和结果收集。
每个 Ray replica 内部运行本地 `vLLMSampler.sample()`，使用真实引擎；不经过训练器、工具或 partial rollout。
这用于隔离路由与 prefix cache 的交互，不是完整 RL 训练性能测试。

## GPU 实验

需要安装当前项目的 vLLM/Ray 环境，以及至少两个独立采样副本所需的 GPU。
同一个副本内的 TP 不算多个可路由副本。建议先使用已缓存、能单卡运行的模型。

```bash
python scripts/benchmark_sticky_session.py \
  --model /path/to/cached/model \
  --replicas 2 --tensor-parallel-size 1 \
  --episodes 64 --turns 6 \
  --prompt-tokens 2048 --append-tokens 128 --max-tokens 64 \
  --concurrency 8 32 --repeats 5 \
  --output-dir /tmp/sticky-session-gpu
```

连接已有 Ray 集群时加 `--ray-address auto`。默认启动本地 Ray；资源不足会报错。
输出目录必须不存在，避免覆盖实验结果。

## NPU 实验

使用已能运行 Twinkle `vLLMSampler` 的昇腾环境，包含配套的 CANN、torch_npu、vLLM 和
vLLM-Ascend；无需安装训练后端。环境准备见[项目 NPU 指南](../docs/source_zh/使用指引/NPU的支持.md)。
先准备一个当前环境支持、能单卡运行的模型，并使用其本地路径。

最小配置是 **2 张 NPU、2 个副本、每副本 TP=1**。四组依次运行，共用这些设备；
若每副本 TP=2，则需要 4 张 NPU，每个副本的 TP 设备必须在同一节点。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1

python scripts/benchmark_sticky_session.py \
  --device npu --model /path/to/cached/model \
  --replicas 2 --tensor-parallel-size 1 \
  --episodes 32 --turns 4 \
  --prompt-tokens 2048 --append-tokens 128 --max-tokens 64 \
  --concurrency 4 8 --repeats 3 --groups B D \
  --output-dir /tmp/sticky-session-npu
```

该命令先比较开启 prefix cache 时的普通路由 B 和 sticky 路由 D；去掉 `--groups B D`
即可跑完整四组。`--gpu-memory-utilization` 沿用 vLLM 的参数名称，在 NPU 上同样表示设备内存比例。

本地启动时根据可见 NPU 数注册 Ray 的 `NPU` 资源，每个副本占用 TP 数量的完整资源，
并在加载引擎前使用 Ray 分配的设备 ID 设置 `ASCEND_RT_VISIBLE_DEVICES`。
连接已有集群时仍使用 `--ray-address auto`，各节点需提前注册相应的 `NPU` 资源，
可见设备范围应在启动 Ray 节点前设置；此时脚本不会重新注册集群资源。
`summary.json` 的 `replica_pools` 记录节点、设备类型及设备 ID，可核对同节点副本是否使用不同的卡。
引擎指标缺失仍记录为 `null`，不会以 mock 数据代替 NPU 测量。

## 对照组

| 组 | session affinity | prefix cache |
|---|---|---|
| A | 关闭 | 关闭 |
| B | 关闭 | 开启 |
| C | 开启 | 关闭 |
| D | 开启 | 开启 |

默认跑四组，重点比较 D/B。仅跑主对照可加 `--groups B D`。
相同 cache 配置复用一个实例池，每次测量前等待缓存重置完成；不同 cache 配置重建实例池。
启动与预热不计入测量，同一 cache 配置内的组运行顺序按 repeat 打乱。
所有组重放完全相同的输入，模型和引擎 seed 固定，输出长度上限相同，实际输出 token 数单独记录。
异步任务到达顺序和普通负载均衡不保证确定性，需要观察多次 repeat 的波动。

## CPU 功能检查

```bash
python scripts/benchmark_sticky_session.py \
  --backend mock --episodes 4 --turns 3 \
  --prompt-tokens 64 --append-tokens 32 --max-tokens 4 \
  --concurrency 2 --repeats 1 \
  --output-dir /tmp/sticky-session-smoke
```

mock 使用真实 Ray actor 和生产分发/路由代码，缓存命中与计算延迟均为合成值。
输出标记 `synthetic: true`、`metrics_source: synthetic_mock`，只能证明基准流程有效。

## 输出与解读

- `config.json`：参数、包版本（包括 torch_npu、vLLM-Ascend）、实验类型。
- `workload.jsonl`：每条 episode 的固定 token 输入；每轮只追加，且在靠前位置区分 episode。
- `requests.jsonl`：group/repeat/concurrency、episode/turn、输入 hash、实际实例、延迟、token 数及引擎观测。
- `summary.json`：每次测量的 p50/p95、吞吐、缓存命中、实例切换、各实例工作量、路由清理状态，以及逐 repeat 的 D/B 比值。

`session_ids` 在一条 episode 内保持一致，episode 退出时解除绑定。
无绑定淘汰时 sticky 组发生跨实例切换、出现请求错误或路由状态泄漏，脚本退出失败并保留结果。
如果有超时，测量停止，不把仍执行的后台任务带入下一轮实验。
`--max-sessions` 默认 10000；调小时可以观察绑定容量退化及实例切换。

缓存命中按 token 加权，单独统计第二轮及以后的命中率。缺少 `num_cached_tokens` 时记录 `null`，
不会伪造为零；聚合命中率只有在对应请求指标完整时才计算。
prefill/queue/TTFT 根据部署版本支持的 `RequestOutput.metrics` 提取，缺失时同样为 `null`，
汇总中的 `count` 表示指标覆盖情况。prefill 是引擎统计口径的调度至首 token 时间，不是设备 kernel 独占耗时。
`first_output_s` 是观测到首次非空输出的 worker 时长，不冒充不可用的引擎 TTFT；
`latency_s` 是调用端整个同步采样时长，包含路由、Ray、排队和完整生成。

本脚本的累计输入与当前生成结果解耦：后续轮次使用预先构造的 token 前缀，不拼接本次返回的输出，
保证对照输入相同。`workload_hash` 可用于核对数据一致性。
真实训练验证应另外使用业务轨迹、真实工具等待和实际权重更新频率。
缓存命中提高不等于吞吐必然提高；同时检查尾延迟和 `per_instance` 的 token 工作量、前端在途请求峰值。
