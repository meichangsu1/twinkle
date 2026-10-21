# 当前 CW sticky session 的端到端测试

此脚本从 `sticky-session` 分支的固定权重重放基准适配而来，使用当前分支的
`remote_function → _run_continous_work → _sticky_session_cw_plan/_cw_plan → Ray worker`。
真实后端的 worker 内运行本地 `vLLMSampler → VLLMEngine → vLLM`，不走 HTTP SDK。
这里验证路由和推理收益；模型训练、权重同步和实际工具执行不在此重放实验内。

## 1. CPU 检查测试链路

```bash
.venv/bin/python scripts/benchmark_sticky_session.py \
  --backend mock --replicas 2 --episodes 8 --turns 3 \
  --prompt-tokens 64 --append-tokens 32 --max-tokens 4 \
  --tool-wait-ms 1 10 --concurrency 2 4 --repeats 2 \
  --groups A B D --output-dir /tmp/sticky-cw-smoke
```

输出目录必须不存在。mock 使用真实 Ray actors、生产 CW 分发和 session 绑定，
但缓存与推理耗时是模拟值，只用于检查流程。`summary.json` 中 `synthetic=true`。

本地 cookbook 的 session 自动生成和跨轮次透传另有集成测试：

```bash
.venv/bin/python -m pytest -q tests/infra/test_sampler_admission_protocol.py \
  -k local_rollout_keeps_each_episode_on_one_ray_sampler_worker
```

它从 `MultiTurnRollout(sticky_session=True)` 经过实际 Ray 分发到两个探针 worker，
检查每个 episode 的多轮调用保持 session 和 worker。探针不执行模型推理。

## 2. 两张 GPU/NPU 上测真实效果

先在运行环境安装本项目、Ray 和匹配设备的 vLLM，准备本地模型。
两副本、每副本 TP=1 至少需要两张卡；TP>1 时需要 `replicas × TP` 张卡。

```bash
python scripts/benchmark_sticky_session.py \
  --backend vllm --device cuda --model /path/to/model \
  --replicas 2 --tensor-parallel-size 1 \
  --max-model-len 8192 --episodes 64 --turns 6 \
  --prompt-tokens-range 512 4096 --append-tokens-range 64 256 \
  --output-tokens-range 16 128 --tool-wait-ms 50 500 \
  --concurrency 4 8 --repeats 5 --groups B D \
  --output-dir /tmp/sticky-cw-vllm
```

NPU 将 `--device cuda` 改为 `--device npu`，并使用已配置 torch_npu、CANN、
vllm-ascend 的环境。运行已有 Ray 集群时可传 `--ray-address auto`；各节点需要
相同代码和模型路径，NPU 集群还需已向 Ray 注册相应资源。

| 组 | Sticky | Prefix cache | 用途 |
| --- | --- | --- | --- |
| A | 关 | 关 | 可选的无缓存基线 |
| B | 关 | 开 | 普通负载均衡基线 |
| C | 开 | 关 | 可选的路由开销对照 |
| D | 开 | 开 | 与 B 比较 sticky 带来的变化 |

所有组使用同一 seed 冻结的追加式 token 输入、输出长度上限和工具等待计划；
输入开头含独立 episode 标记，避免不同 episode 共享前缀掩盖路由效果。
生成结果不拼入下一轮，以保持组间输入相同；真实输出长度仍可能因 EOS 不同。
每个 trial 前清空引擎 prefix cache 和 driver 的 session 绑定，B/D 顺序按 seed
打乱。warmup 发生在清空缓存之前。应在空闲设备上重复测量。

## 3. 判断结果

- `requests.jsonl`：逐请求的 session ID、实际 actor ID、输入 hash、缓存 token、引擎指标和端到端耗时。
- `workload.jsonl`：冻结输入及每轮输出上限、工具等待计划。
- `summary.json`：设备分配、软件版本、输入 hash、每个 trial 和 D/B 成对比较。

先检查 `errors=0`、`completed_episodes=episodes`、CW 的 `total_inflight=0`、
`pending_calls=0`、`pending_methods={}`；静态 worker 池中，未超过 30 分钟空闲
过期时间的 D 组应有 `worker_switches=0`。脚本对采样错误、未释放的在途请求和
sticky 切换直接报错。生产 session 绑定会在请求完成后保留，空闲过期时回收；
所以 D 组的 `session_count` 非零是预期行为，基准只在 trial 边界显式清空它。

随后比较 B/D 的 `continuation_cache_hit_fraction`、`engine_prefill_s`、
`engine_ttft_s`、`episodes_per_s` 和 `episode_latency_p95_s`。
`comparisons_D_over_B` 按相同 concurrency/repeat 配对：吞吐比大于 1、p95 比小于 1
分别表示吞吐和延迟改善。应同时看每副本请求/输入/输出分布，判断是否出现负载倾斜。
同一个 worker 不保证缓存仍驻留；缓存压力、权重更新和多轮输入变化都会影响收益。

部分 vLLM 版本不提供所有请求指标，缺失值保留为 `null`，并报告
`cache_metric_coverage`，不能把缺失当作零命中。`first_output_s` 是生成器首次输出
的观察时间；若引擎只返回最终结果，它不等同于 TTFT，以引擎请求指标为准。

## 与旧分支 GRPO 基准的关系

`benchmark_grpo_sticky_session.py` 已适配当前 CW 实现，覆盖权重同步、采样、
reward/advantage、backward 和 optimizer。完整训练收益使用其 `--task synthetic`
多轮模式测试，至少准备一张训练卡和两张独立采样卡，详见
[GRPO 测试说明](README_grpo_sticky_session.md)。
