# 多 LoRA 异步 RL 数据加载设计

## 1. 目标

多租户 multi-LoRA 异步 RL 中有两类“加载数据”的动作，必须明确区分：

```text
rollout 前:
  从原始 dataset 读取 prompt group，送给 AsyncRollouter。

trainer 前:
  从 TransferQueue 读取已经完成 rollout / reward / advantage 的 train_k rows。
```

因此本设计不把所有数据读取都叫做 DataLoader，而是拆成：

```text
twinkle.dataloader.DataLoader:
  Twinkle 现有原始 dataset 加载器。

PromptLoader:
  pipeline source component，包装 twinkle.dataloader.DataLoader，
  把原始 prompt groups 按 LoraContext 投递给 AsyncRollouter。

TransferQueue-backed dataloader:
  TrainerWorker 通过 TransferQueueDataPlane 构建的训练数据读取器。
```

## 2. 组件边界

### 2.1 twinkle.dataloader.DataLoader

`twinkle.dataloader.DataLoader` 保持现有职责：

```text
Dataset / LazyDataset
  -> template
  -> processor
  -> encode
  -> twinkle.dataloader.DataLoader
  -> prompt batch
```

它服务 rollout 前的原始数据加载。它读的是用户数据集，不读 TransferQueue。

典型输出是：

```text
messages
question
image
task metadata
group_id
```

### 2.2 PromptLoader

`PromptLoader` 是异步 RL pipeline 中的 source component，职责很薄：

```text
PromptLoader(context, twinkle_dataloader, rollouter)
  -> next(dataloader)
  -> normalize batch into prompt groups
  -> AsyncRollouter.enqueue_prompt_groups(context, prompt_groups)
```

它不做 rollout 调度，不计算 reward，不读 TQ，也不训练模型。

第一版接口：

```python
class PromptLoader:
    context: LoraContext
    dataloader: Iterable
    rollouter: AsyncRollouter
    max_pending_groups: int | None
    exhausted: bool
    submitted_groups: int

    def can_load(self) -> bool: ...
    def step(self) -> ComponentResult | None: ...
    def is_idle(self) -> bool: ...
    def shutdown(self) -> None: ...
```

`max_pending_groups` 用于限制单个 LoRA 在 rollout pending queue 中堆积过多原始 prompt，避免某个 LoRA 的数据源过快导致内存或调度偏置。

### 2.3 AsyncRollouter

`AsyncRollouter` 只接收已经绑定 `LoraContext` 的 prompt groups：

```text
pending_prompt_groups_by_context:
  context.key -> queue[prompt_group]
active_rollout_tasks:
  task -> RolloutTaskState
completed_rollout_results:
  queue[completed TQ write result]
```

它根据：

```text
StalenessManager capacity
LoraAdapterRegistry state
TransferQueueDataPlane.check_capacity(context)
rollout policy
max_concurrent_groups
```

决定下一个要 rollout 的 LoRA。选中 context 后，`AsyncRollouter` 会从该 context 的 pending queue 中取出一个 prompt group，创建一个 rollout task，并用 `asyncio.create_task` 放入 `active_rollout_tasks` 后立即返回。已经完成的 task 会进入 `completed_rollout_results`，下一轮 `step()` 返回 `kind="rollout"`，对应数据已经写入 TQ。也就是说，`PromptLoader` 只负责“喂数据”，真正的 LoRA rollout 调度、异步提交和完成回收仍然在 `AsyncRollouter` 内部完成。

### 2.4 TrainerWorker 的 dataloader

Trainer 侧不使用原始 dataset，也不使用 rollout-side `PromptLoader`。

Trainer 只能读取 TransferQueue 中已经完成处理的训练数据：

```text
TransferQueue train_k
  rollout fields
  reward fields
  advantage fields
  row-level policy_version / old_logps
```

读取入口：

```python
dataloader = TransferQueueDataPlane.build_streaming_dataloader(context, partition_id)
```

这个 dataloader 的语义是 TQ-backed train batch iterator，不是 `twinkle.dataloader.DataLoader`。

## 3. 多 LoRA 数据加载方式

每个 LoRA / `LoraContext` 可以绑定自己的 dataset 配置：

```yaml
lora_contexts:
  - tenant_id: tenant_a
    training_run_id: run_a
    base_model_id: ms://Qwen/Qwen3.5-4B
    adapter_name: lora_a
    reward_type: gsm8k
    loss_type: grpo
    dataset:
      dataset_id: ms://modelscope/gsm8k
      subset_name: main
      split: train
      data_num: 2000
      batch_size: 4
      system_prompt: "You are a helpful math assistant."

  - tenant_id: tenant_b
    training_run_id: run_b
    base_model_id: ms://Qwen/Qwen3.5-4B
    adapter_name: lora_b
    reward_type: gsm8k
    loss_type: grpo
    dataset:
      dataset_id: /data/custom_math.jsonl
      split: train
      data_num: 1000
      batch_size: 2
      system_prompt: "Solve the problem."
```

如果某个 context 没有单独声明 `dataset`，可以回退到全局 `dataset` 配置。

初始化逻辑：

```text
BaseRLPipeline.build_lora_contexts()
  -> context_a, context_b

BaseRLPipeline.create_roles()
  -> PromptLoader
  -> AsyncRollouter
  -> RewardWorker
  -> AdvantageWorker
  -> TrainerWorker
  -> build_prompt_loaders()
  -> build_pipeline_components()

build_prompt_loaders()
  -> DataLoader(dataset_a) -> PromptLoader(context_a)
  -> DataLoader(dataset_b) -> PromptLoader(context_b)
```

运行时：

```text
PromptLoader(context_a).step()
  -> AsyncRollouter.enqueue_prompt_groups(context_a, prompt_groups_a)

PromptLoader(context_b).step()
  -> AsyncRollouter.enqueue_prompt_groups(context_b, prompt_groups_b)

AsyncRollouter.step()
  -> pick_next_rollout_context()
  -> submit rollout task
  -> later collect completed task
  -> TransferQueueDataPlane.put_rollout_batch(context, train_k)
```

## 4. 为什么不让 Trainer 直接读 twinkle.dataloader.DataLoader

Trainer 的输入必须包含 rollout 和 RL 训练所需字段：

```text
messages / input_ids / labels
old_logps
rewards
advantages / returns
policy_version
adapter_revision
group_id / generation_idx
```

这些字段不是原始 dataset 能直接提供的，而是在以下链路中产生：

```text
twinkle.dataloader.DataLoader
  -> PromptLoader
  -> AsyncRollouter
  -> TransferQueueDataPlane
  -> RewardWorker
  -> AdvantageWorker
  -> TrainerWorker
```

因此 trainer 直接读取原始 dataset 会绕过 rollout/reward/advantage，破坏异步 RL 的数据闭环。

## 5. 第一版落地约束

```text
1. PromptLoader 使用 twinkle.dataloader.DataLoader，不自建新的 dataset 系统。
2. 每个 LoraContext 可以有一个独立 DataLoader。
3. PromptLoader 只向 AsyncRollouter 入队，不直接写 TQ。
4. AsyncRollouter 统一做 multi-LoRA rollout 调度。
5. TrainerWorker 只通过 TransferQueueDataPlane 读取 train_k。
6. 一个 train_k 不混 adapter；可以包含多个 policy_version rows。
7. 某个 PromptLoader exhausted 只影响对应 LoRA，不影响其他 LoRA。
```

## 6. 与当前代码的对应关系

```text
src/twinkle_agentic/async_rl/prompt_loader.py
  PromptLoader:
    pipeline source component
    step() -> ComponentResult(kind="prompt")

src/twinkle_agentic/async_rl/pipeline.py
  BaseRLPipeline.build_prompt_loaders()
  BaseRLPipeline.components
  BaseRLPipeline.run_async()

src/twinkle_agentic/async_rl/workers.py
  AsyncRollouter.enqueue_prompt_groups()
  AsyncRollouter.pending_prompt_group_count()
  AsyncRollouter.step()
  RewardWorker.step()
  AdvantageWorker.step()
  TrainerWorker.run_once()
  TrainerWorker.step()

cookbook/rl/async_multi_lora_grpo.py
  server-side multi-LoRA 示例：
    每个 LoraContext 创建一个 twinkle.dataloader.DataLoader
    每个 DataLoader 包成一个 PromptLoader

cookbook/client/twinkle/self_host/async_multi_lora_grpo.py
  client/self-host 示例：
    同样使用 PromptLoader 封装 twinkle.dataloader.DataLoader
```

## 7. Pipeline Component 视角

第一版不引入复杂进程管理器，但所有热路径组件都按统一的轻量接口组织：

```python
step() -> ComponentResult | None
is_idle() -> bool
shutdown() -> None
```

组件职责：

| 组件 | 类型 | 输入 | 输出 | multi-LoRA 支撑方式 |
|---|---|---|---|---|
| `PromptLoader` | source component | `twinkle.dataloader.DataLoader` | `AsyncRollouter` pending queue | 每个 `LoraContext` 一个 loader |
| `AsyncRollouter` | rollout producer | pending prompt groups | TQ `train_k` rollout rows | `pending_by_context` + rollout policy |
| `RewardWorker` | TQ transformer | rollout-ready partition | reward fields | 按 context 轮询 claim |
| `AdvantageWorker` | TQ transformer | reward-ready partition | advantages / returns | 按 context 轮询 claim |
| `TrainerWorker` | TQ consumer / optimizer | `TRAIN_READY train_k` | adapter weights / clear partition | `TrainerScheduler` 选择 partition |

`BaseRLPipeline` 只负责根据 `algorithm` 创建角色图，并按顺序调用 `component.step()` 推进系统。组件之间的数据交换不通过 pipeline 传递业务数据：

```text
PromptLoader -> AsyncRollouter pending queue
AsyncRollouter / RewardWorker / AdvantageWorker / TrainerWorker -> TransferQueueDataPlane
TrainerWorker -> LoraAdapterRegistry / receive_weights
```

默认 GRPO 组件图是：

```text
PromptLoader
  -> AsyncRollouter
  -> RewardWorker
  -> AdvantageWorker
  -> MultiLoraGRPOTrainerWorker
```

如果算法链路不同，不应在 `BaseRLPipeline` 中增加训练细节分支，而是由对应算法的 pipeline 子类覆盖 `create_roles()`。例如 DPO 可以使用：

```text
PairFeeder
  -> DPOTrainerWorker
```

这样 `PromptLoader` 仍然是正式 pipeline component；只是它只属于需要 rollout prompt source 的算法链路。
