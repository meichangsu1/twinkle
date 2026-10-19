# GRPO 端到端 Prefix Cache / Sticky Session 实验

本实验执行真实 GSM8K 单轮 GRPO：采样、评分、组内 advantage、前后向、优化器更新，以及下一批前的权重同步。它与 `benchmark_sticky_session.py` 的固定权重多轮重放实验独立。

同一 prompt 的 G 个候选使用相同路由 ID，在采样结束的 `finally` 中解除绑定。ID 不进入输入特征或引擎请求参数；生产 sampler、router 和训练实现保持原样。

| 组别 | Prefix cache | 路由 |
| --- | --- | --- |
| A | 关闭 | 普通最小在途负载 |
| B | 开启 | 普通最小在途负载 |
| D | 开启 | 同一个 GRPO prompt 组绑定同一实例 |

## 启动

在仓库根目录运行，使用已安装 Twinkle、Ray、vLLM、训练依赖的环境。NPU 还需要匹配的 torch_npu / CANN / vllm-ascend；脚本接入已有的 NPU 设备分配和 HCCL 权重同步路径，尚未在真实 NPU/GPU 上验证。

默认 1 张训练卡、2 张采样卡（TP=1），共 3 张。同一时刻只运行一个实验组；各组和各 seed 串行运行，不会同时启动九套模型。训练卡使用 LoRA + bf16，`--model-devices` 可增加训练 DP 卡数。实际显存和主机内存需求取决于安装版本、模型和配置，脚本不保证单卡能容纳任意模型。

先验证小规模完整训练链路：

```bash
python scripts/benchmark_grpo_sticky_session.py \
  --device npu --model /path/to/Qwen3.5-0.8B \
  --groups A B D --seeds 42 \
  --steps 3 --warmup-steps 1 \
  --batch-size 4 --num-generations 8 --mini-batch-size 16 \
  --max-prompt-tokens 2048 --max-tokens 512 --max-model-len 4096 \
  --max-num-seqs 32 --language-model-only \
  --output-dir output/grpo-sticky-smoke
```

正式对照示例：

```bash
python scripts/benchmark_grpo_sticky_session.py \
  --device npu --model /path/to/Qwen3.5-0.8B \
  --groups A B D --seeds 42 43 44 \
  --steps 50 --warmup-steps 5 \
  --batch-size 16 --num-generations 8 --mini-batch-size 32 \
  --micro-batch-size 1 --max-prompt-tokens 4096 \
  --max-tokens 512 --max-model-len 8192 --max-num-seqs 64 \
  --temperature 1.0 --top-p 0.95 --language-model-only \
  --output-dir output/grpo-sticky
```

CUDA 使用 `--device cuda`。`--language-model-only` 适合支持此参数的多模态模型引擎，例如 Qwen3.5 的纯文本实验；其他模型可省略。`--enforce-eager` 可用于调试引擎图编译问题，比较组应使用相同设置。

`--steps` 是**完整 rollout 批次数**。默认每批 16 个问题 × 8 个候选 = 128 条轨迹，mini-batch=32，因此每批进行 4 次优化器更新。预热批次也真实训练，只在统计时排除，不能跨实验组复用预热后的权重。每组均从相同模型文件和新的 LoRA/优化器状态开始；应使用固定的本地模型快照，避免运行期间上游模型更新。

`--max-prompt-tokens` 是长度上限，超长数据整条过滤；不会把短 GSM8K 问题填充到此长度。`max_prompt_tokens + max_tokens` 必须不超过 `max_model_len`，模板和引擎使用一致的上下文长度。答案输出过短可能造成任务失败，应检查实际输出长度、答案正确率和 `length_stop_fraction` 后再确定预算。

## 数据与随机性

运行前在独立 CPU 准备进程中加载 `ms://modelscope/gsm8k`、`main/train`，使用现有 `GSM8KProcessor` 和 `Template(enable_thinking=False)`，保存一次编码后的 `inputs.jsonl`，包含评分所需的 ground truth。各组、各 seed 读取同一份文件，相同批次使用相同 prompt 顺序。数据不足以覆盖所有批次时直接报错，不循环补齐。

`--dataset-id` / `--dataset-revision` / `--subset` 可以固定 GSM8K 来源；处理器要求 `question`、`answer` 的 GSM8K 格式。`--input-plan /previous/run/inputs.jsonl` 可以跨运行重用编码输入，要求行数恰好等于 steps × batch-size，且使用相同 tokenizer、模板、模型和长度设置。

训练和 vLLM 引擎设置相同 seed，但**不会给 G 个 n=1 请求设置相同的请求级采样 seed**，避免生成完全相同的候选、使 GRPO advantage 退化。实际完成内容、长度、后续策略权重可能因调度及数值差异而改变；本实验保证输入和训练预算一致，不声称完整轨迹重放或逐步权重一致。

引擎显式使用并检查 `logprobs_mode=raw_logprobs`，与训练端计算的原始模型 logprob 对齐；temperature / top-p 控制采样分布，不把截断、重归一化后的 logprob 当作原始模型概率。安装的 vLLM 需要支持此模式。

## 指标与结果

```text
output/grpo-sticky/
  inputs.jsonl              # 所有实验共享的编码输入和评分标签
  config.json, prepare.log
  summary.json              # 所有组/seed 的汇总与配对比值
  A-seed-42/
    config.json, run.log
    instances.json          # 实际实例、设备可见性、原生 prefix cache 开关
    requests.jsonl          # 每个原生请求的 token 数、缓存数、实例、输入 hash、引擎 ID
    steps.jsonl             # 每批计时、reward、正确率、长度、路由状态
    summary.json
  B-seed-42/ ...
  D-seed-42/ ...
```

每批阶段：权重同步、清缓存、采样（含组绑定清理）、reward/advantage、训练更新，及其他观测/数据处理开销。训练计时包含设备同步和训练指标收集，避免把异步设备任务的排队时间当作完成耗时；`model_metrics` 保存现有训练组件返回的 loss 等指标。总计时包含观测 RPC 和数据处理等实验开销；初始化、输入准备和可选的最终 checkpoint 保存单独于这段计时，不计入稳定训练吞吐。使用 `--save-checkpoint` 保存每组最终 checkpoint，默认不保存以避免额外磁盘占用。

每次权重同步后清空 prefix cache，三组保持相同语义。命中率使用原生 `num_cached_tokens` 的总和 / 输入 token 总和；缺失时为 null，保留 coverage，不以 0 冒充。每条请求单独记录，因此重复 prompt 不会互相覆盖观测。D 组验证同组只使用一个实例，逐批检查 lease 和绑定均已释放。批内有完全重复的问题时，按 prompt hash 无法区分其独立组，相关组副本数量统计会排除并记录 `ambiguous_prompt_groups`。

父 summary 单独记录准备输入的 `prepare_wall_s` 和每组进程完整生命周期的 `process_wall_s`。每组 summary 另含 `startup_s`、预热阶段 `warmup_totals_s`、含预热的 `training_wall_s_including_warmup` 和可选的 `checkpoint_save_s`，便于区分启动成本与稳定训练吞吐。

同时提交的 G 个请求是否能复用缓存取决于引擎调度，脚本不刻意先串行执行一个候选来预热其余候选。普通路由下同组分散到两个实例，也可能在每个实例内复用相同 prompt。D 不保证更快：组内集中还可能带来负载不均衡。先检查缓存命中和组分布，再解释耗时变化。

自动配对 B/A、D/B、D/A，先对每个 seed 算比值：

- 采样吞吐比 = candidate 的 trajectories/s ÷ baseline 的 trajectories/s。
- 完整训练吞吐比 = candidate 的轨迹数/完整批次耗时 ÷ baseline 对应值。
- 总耗时降低 = 1 − candidate 总耗时 ÷ baseline 总耗时。

同时提供实际输出 token 比和答案正确率差，辅助判断是否只是生成量或任务完成情况改变。`nonzero_advantage_fraction` 可帮助识别整组 reward 一致导致的零更新信号。训练 rollout 正确率不是独立验证集成绩。

绘图不需要训练设备、Ray 或 vLLM，仅需 matplotlib：

```bash
python scripts/plot_grpo_sticky_session.py output/grpo-sticky --formats png svg
```

生成 `plots/overview.*`、`stage_times.*`、`paired_ratios.*`。独立 seed 为散点，主图柱为中位数，误差线为 min-max，非置信区间；阶段堆叠图使用平均值。失败时保留日志、已完成批次和 error.json，父 summary 标记 failed 并停止后续实验。绘制未完成实验会标记 INCOMPLETE RUN，不把失败记录当作有效测量。
