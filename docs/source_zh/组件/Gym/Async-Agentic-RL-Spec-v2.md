# 基于 TransferQueue 的 Agentic RL Fully Async 设计规格

## 目标

本规格定义 Twinkle 在 agentic 多轮 RL 中接入 TransferQueue 的目标架构。设计重点不是新增一个 `AsyncMultiTurnRollout`，而是在现有 `MultiTurnRollout`、`vLLMSampler`、reward、advantage、trainer 之外增加一层异步生产和训练编排：

```text
BaseRLPipeline 负责整个 RL 生命周期编排。
AsyncRollouter 负责 rollout 数据生产侧。
TransferQueue 只作为数据容器。
StatelessManager 负责背压和参数版本同步策略。
TrainerWorker 持续从 TransferQueue 消费可训练数据。
```

第一版落地目标是 GRPO。设计以当前 `cookbook/exp/grpo_baseline.py` 的同步流程为基线，但把内存中的 `all_trajectories / rewards / advantages / old_logps` handoff 改为通过 TransferQueue 传递，并让 rollout 和 trainer 可以重叠执行。

## 当前基线

`grpo_baseline.py` 的核心流程是同步 batch-blocking：

```python
expand_prompts = [p for prompt in batch for p in [prompt] * NUM_GENERATIONS]

ckpt_manager.sync_weights(merge_and_sync=False)
sampler.reset_prefix_cache()

all_trajectories = rollout(expand_prompts)
total_rewards, f1_rewards, cot_rewards = compute_rewards(all_trajectories)
advantages = GRPOAdvantage()(total_rewards, num_generations=NUM_GENERATIONS, scale="group")

for mini_batch in all_trajectories:
    ref_logps = model.forward_only(..., disable_lora=True)
    model.forward_backward(
        inputs=mini_batch,
        old_logps=old_logps,
        advantages=advantages,
        ref_logps=ref_logps,
    )
    model.clip_grad_and_step()
```

这个流程有三个问题：

- `rollout(expand_prompts)` 要等整个 expanded batch 完成后才返回，快完成的 prompt group 不能提前进入训练数据系统。
- rollout、reward、advantage、trainer 在同一个 driver loop 中串行运行，无法自然 overlap。
- 权重同步、staleness、长尾任务取消等控制逻辑没有独立抽象，难以切换同步/异步模式。

## 核心概念

### Prompt Group

GRPO 的最小语义单元不是单条 trajectory，而是一个 prompt 的多条 generation：

```text
prompt_i
  -> NUM_GENERATIONS 条 trajectories
  -> 一个 prompt group
```

原因是 `GRPOAdvantage` 需要按 group 计算 relative advantage。第一版 fully async 的 rollout 调度单位应是 prompt group，而不是单条 trajectory。

### Transfer Batch

Rollout 不应等完整 rollout step 所有 group 就绪后才写入 TransferQueue，也不应每条 sample 写一次。

推荐写入粒度是 transfer batch：

```text
若干 ready prompt groups
  -> batch_to_transfer
  -> flatten 成 samples
  -> put 到同一个 train_k partition
```

这样可以把长尾影响范围从完整 rollout step 缩小到 prompt group。

### Partition

TransferQueue partition 是数据生命周期和权重版本同步单位：

```text
partition_id = train_{rollout_id}
```

一个 `train_k` 可以被多次 append。rollout side 按 transfer batch 往 `train_k` 追加 ready groups；trainer side 从 `train_k` 持续读取字段 ready 的 samples。`train_k` 达到目标样本量并训练完成后，触发权重同步并清理 partition。

### Policy Version

`policy_version` 表示 rollout 使用的参数版本。第一版建议令：

```text
policy_version = 当前 trainer 已同步给 sampler 的参数版本
```

每个 trajectory 必须记录：

```text
rollout_id
partition_id
group_id
generation_idx
policy_version
```

trainer 根据 `current_policy_version - partition.policy_version` 判断 staleness。第一版不做 sample-level staleness filtering。

## 总体架构

```mermaid
flowchart LR
    Dataset["DataLoader<br/>Prompt batches"] --> Pipeline["BaseRLPipeline<br/>全局编排"]

    Pipeline --> Manager["StatelessManager<br/>控制面<br/>背压 / staleness / sync 策略"]
    Pipeline --> Rollouter["AsyncRollouter<br/>rollout producer"]
    Pipeline --> Reward["RewardWorker<br/>任务 reward"]
    Pipeline --> Advantage["AdvantageWorker<br/>GRPOAdvantage"]
    Pipeline --> Trainer["TrainerWorker<br/>forward_backward / step"]
    Pipeline --> Ckpt["CheckpointEngineManager<br/>权重同步"]

    TQ["TransferQueue<br/>train_k partitions"] -. "get_metadata()" .-> Manager
    Manager -. "rollout permits<br/>partition / policy_version" .-> Rollouter
    Manager -. "reward permits" .-> Reward
    Manager -. "advantage permits" .-> Advantage
    Manager -. "train permits" .-> Trainer
    Manager -. "sync / clear decisions" .-> Ckpt

    Rollouter -->|"ready groups append"| TQ
    TQ --> Reward
    Reward -->|"append rewards"| TQ
    TQ --> Advantage
    Advantage -->|"append advantages"| TQ
    TQ --> Trainer
    Trainer -->|"train events / metrics"| Manager
    Ckpt -->|"update sampler weights"| Rollouter

    Rollouter --> Rollout["MultiTurnRollout<br/>现有多轮/tool loop"]
    Rollout --> Sampler["vLLMSampler<br/>vLLM AsyncLLM backend"]
```

组件边界：

- `BaseRLPipeline`：训练入口和总编排，负责组件初始化、资源分配、角色创建、worker 启停、日志、checkpoint、异常处理。
- `AsyncRollouter`：只负责 rollout 数据生产侧，不负责 reward、advantage、loss、optimizer。
- `TransferQueue`：只负责存储 sample、field、metadata，不决定同步还是异步。
- `TransferQueueClient`：Twinkle 侧封装，屏蔽 TransferQueue native API、KV API、StreamingDataset 的具体差异。
- `StatelessManager`：控制面核心。它只根据 TransferQueue metadata 和少量版本计数返回控制决策，不持有 sample payload。
- `RewardWorker`：读取 rollout-ready samples，追加 reward 字段。
- `AdvantageWorker`：按 group 等齐 rewards，追加 advantages。
- `TrainerWorker`：读取 train-ready samples，执行 ref forward、forward_backward、optimizer step。
- `CheckpointEngineManager`：复用现有 trainer -> sampler 权重同步路径。

## 组件架构图

### 控制面与数据面

```mermaid
flowchart TB
    subgraph Driver["BaseRLPipeline"]
        Config["load_config()"]
        Build["build_components()"]
        Dispatch["dispatch(ControlDecision)"]
        Lifecycle["shutdown()"]
    end

    subgraph ControlPlane["控制面"]
        TQMeta["TransferQueueClient.get_metadata()"]
        Manager["StatelessManager.decide()"]
        Decision["ControlDecision"]
    end

    subgraph Producers["生产侧"]
        Rollouter["AsyncRollouter"]
        Rollout["MultiTurnRollout"]
        Sampler["vLLMSampler"]
    end

    subgraph DataPlane["数据面"]
        TQ["TransferQueue<br/>train_k partition"]
        RolloutFields["rollout fields"]
        RewardFields["reward fields"]
        AdvantageFields["advantage fields"]
        TrainState["trained / dropped state"]
    end

    subgraph Consumers["消费侧"]
        RewardWorker["RewardWorker"]
        AdvantageWorker["AdvantageWorker"]
        TrainerWorker["TrainerWorker"]
        Ckpt["CheckpointEngineManager"]
    end

    Config --> Build --> Dispatch
    TQ --> TQMeta --> Manager --> Decision --> Dispatch

    Dispatch --> Rollouter
    Rollouter --> Rollout --> Sampler
    Rollouter --> RolloutFields --> TQ

    Dispatch --> RewardWorker
    TQ --> RewardWorker --> RewardFields --> TQ

    Dispatch --> AdvantageWorker
    TQ --> AdvantageWorker --> AdvantageFields --> TQ

    Dispatch --> TrainerWorker
    TQ --> TrainerWorker --> TrainState --> TQ
    TrainerWorker --> Ckpt --> Sampler

    Dispatch --> Lifecycle
```

控制面只处理 metadata 和 decision；数据面只处理 sample payload、field 追加和状态更新。`BaseRLPipeline` 不直接实现算法逻辑，它只把 `StatelessManager.decide()` 产出的 `ControlDecision` 分发给对应组件。

### AsyncRollouter 内部结构

```mermaid
flowchart LR
    Dataset["DataLoader"] --> Feed["feed_pending_queue()"]
    Feed --> Pending["pending_queue<br/>RolloutGroupRequest"]
    Pending --> Submit["submit_until_capacity()"]
    Submit --> Active["active_tasks<br/>group tasks"]

    Active --> RunGroup["run_one_group()"]
    RunGroup --> MTR["MultiTurnRollout.__call__()"]
    MTR --> Sampler["vLLMSampler.sample()"]
    Sampler --> MTR
    MTR --> Result["RolloutGroupResult"]

    Result --> Drain["drain_completed_groups()"]
    Drain --> Buffer["transfer_buffer<br/>ready groups"]
    Buffer --> Flush["flush_transfer_buffer()"]
    Flush --> TQPut["TransferQueueClient.put_rollout_batch()"]
    TQPut --> Partition["train_k partition"]

    Manager["ControlDecision"] -. "set_rollout_budget()" .-> Submit
    Manager -. "pause_submit()" .-> Submit
```

`AsyncRollouter` 的核心状态是 `pending_queue`、`active_tasks` 和 `transfer_buffer`。它按 prompt group 生产数据，group 完成后先进入 transfer buffer，达到 `transfer_batch_size_groups` 后追加到当前 `train_k`。

### TransferQueue 消费侧结构

```mermaid
flowchart TB
    TQ["TransferQueue<br/>train_k"]

    subgraph Fields["partition fields"]
        F1["rollout fields<br/>input_ids / labels / old_logps / messages"]
        F2["reward fields<br/>rewards / raw_rewards"]
        F3["advantage fields<br/>advantages / returns"]
        F4["train state<br/>trained / dropped"]
    end

    subgraph Workers["workers"]
        RW["RewardWorker"]
        AW["AdvantageWorker"]
        TW["TrainerWorker"]
    end

    TQ --> F1
    F1 --> RW
    RW --> F2
    F2 --> TQ

    TQ --> F2
    F2 --> AW
    AW --> F3
    F3 --> TQ

    TQ --> F3
    F3 --> TW
    TW --> F4
    F4 --> TQ

    TQ -. "get_metadata()" .-> Meta["QueueMetadata / PartitionMetadata"]
    Meta -. "decide()" .-> Manager["StatelessManager"]
```

TransferQueue 中的 field 是分阶段追加的。`RewardWorker`、`AdvantageWorker`、`TrainerWorker` 都通过 `TransferQueueClient` claim 自己需要的 ready 数据；`StatelessManager` 只读取 metadata，不读取 payload。

## 类关系

```mermaid
classDiagram
    class BaseRLPipeline {
        +config
        +TransferQueueClient tq
        +StatelessManager manager
        +build_components()
        +allocate_resources()
        +create_roles()
        +run()
        +shutdown()
    }

    class AsyncRollouter {
        +pending_queue
        +active_tasks
        +transfer_buffer
        +run()
        +submit_groups()
        +run_one_group()
        +flush_transfer_buffer()
        +abort_stale_tasks()
    }

    class StatelessManager {
        +current_policy_version
        +decide(metadata)
        +on_parameter_synced()
        +on_partition_trained()
    }

    class TransferQueueClient {
        +put_rollout_batch()
        +append_rewards()
        +append_advantages()
        +claim_train_batch()
        +mark_partition_done()
        +clear_partition()
        +get_metadata()
    }

    class RewardWorker {
        +run()
        +compute_rewards()
    }

    class AdvantageWorker {
        +run()
        +compute_advantages()
    }

    class TrainerWorker {
        +run()
        +train_batch()
        +sync_weights()
    }

    class MultiTurnRollout {
        +__call__(trajectories)
    }

    class vLLMSampler {
        +sample(inputs)
        +astream_one(trajectory)
        +receive_weights()
    }

    BaseRLPipeline --> AsyncRollouter
    BaseRLPipeline --> RewardWorker
    BaseRLPipeline --> AdvantageWorker
    BaseRLPipeline --> TrainerWorker
    BaseRLPipeline --> StatelessManager
    BaseRLPipeline --> TransferQueueClient
    AsyncRollouter --> MultiTurnRollout
    MultiTurnRollout --> vLLMSampler
    RewardWorker --> TransferQueueClient
    AdvantageWorker --> TransferQueueClient
    TrainerWorker --> TransferQueueClient
    StatelessManager --> TransferQueueClient : reads metadata only
```

## AsyncRollouter

`AsyncRollouter` 是 rollout side 的异步生产运行时。它不是整个 RL pipeline，也不负责训练。它存在的原因是当前接口：

```python
all_trajectories = rollout(expand_prompts)
```

是 batch-blocking。要实现 fully async，需要把数据生产改成：

```text
DataLoader prompts
  -> pending_queue
  -> active prompt group tasks
  -> ready group transfer_buffer
  -> TransferQueue train_k
```

### 内部状态

```python
@dataclass
class RolloutGroupRequest:
    prompt: Trajectory
    rollout_id: int
    partition_id: str
    group_id: str
    num_generations: int
    policy_version: int


@dataclass
class RolloutGroupResult:
    request: RolloutGroupRequest
    trajectories: list[Trajectory]
    status: str  # ok / timeout / aborted / failed
    error: str | None = None
```

`AsyncRollouter` 持有：

```text
pending_queue: asyncio.Queue[RolloutGroupRequest]
active_tasks: set[asyncio.Task]
transfer_buffer: list[list[Trajectory]]
queue_size
concurrency
transfer_batch_size_groups
group_timeout_s
```

### 运行逻辑

```python
class AsyncRollouter:
    async def run(self):
        while not self.stopped:
            await self.feed_pending_queue()
            await self.submit_until_capacity()
            await self.drain_completed_groups()
            await self.flush_if_ready()
            await self.apply_manager_backpressure()

    async def run_one_group(self, request: RolloutGroupRequest) -> RolloutGroupResult:
        group_inputs = [request.prompt] * request.num_generations
        trajectories = await self.call_rollout(group_inputs)
        for generation_idx, traj in enumerate(trajectories):
            traj["partition_id"] = request.partition_id
            traj["rollout_id"] = request.rollout_id
            traj["group_id"] = request.group_id
            traj["generation_idx"] = generation_idx
            traj["policy_version"] = request.policy_version
        return RolloutGroupResult(request=request, trajectories=trajectories, status="ok")
```

第一版可以复用现有 `MultiTurnRollout`：

```text
一个 group task 调用一次:
  rollout([prompt] * NUM_GENERATIONS)
```

这样 group 内部仍然复用 `MultiTurnRollout` 的 batched sampler 调用；group 之间由 `AsyncRollouter` 做并发。

后续如果要更接近 verl，可把 `MultiTurnRollout` 内部局部状态拆成可调度的 agent loop state，让同一 group 内的 `NUM_GENERATIONS` 条 trajectory 也以独立 task 并发执行。

### 和 vLLMSampler 的关系

当前 `vLLMSampler` 足够作为第一版 fully async 的底层 sampler：

- `sample()` 支持 `List[Trajectory]` / `List[InputFeature]`。
- sampler 内部使用 vLLM `AsyncLLM`，可以并发提交多个请求。
- `SampleResponse` 带 `tokens`、`logprobs`、`decoded`、`new_input_feature`。
- `receive_weights()` 支持 trainer 到 sampler 的权重同步。
- `astream_one()` 可作为后续更细粒度 streaming rollout runtime 的基础。

不足之处在上层 runtime，而不是 sampler 本身。第一版不要求新增 vLLM engine。

## 数据流

### 数据线和权重线

```mermaid
flowchart LR
    subgraph DataLine["数据线"]
        D["DataLoader"] --> AR["AsyncRollouter"]
        AR -->|"ready groups / transfer batch"| TQ["TransferQueue train_k"]
        TQ --> RW["RewardWorker"]
        RW --> TQ
        TQ --> AW["AdvantageWorker"]
        AW --> TQ
        TQ --> TW["TrainerWorker"]
        TW -->|"clear train_k"| TQ
    end

    subgraph WeightLine["权重线"]
        TW -->|"optimizer steps"| Model["Actor Model"]
        Model -->|"sync weights"| CKPT["CheckpointEngineManager"]
        CKPT -->|"receive_weights"| Sampler["vLLMSampler"]
        Sampler --> AR
    end
```

### rollout 写入 TQ

一个 rollout step 创建一个 partition：

```text
rollout_id = k
partition_id = train_k
target_groups = partition.target_groups
target_samples_per_partition = partition.target_groups * rollout.num_generations
```

这里的 `partition.target_groups` 表示一个 `train_k` 目标收集多少个 prompt group，不表示 rollouter 一次必须生成多少个 group。rollouter 仍然是 streaming producer，group ready 后分批写入同一个 `train_k`。

`AsyncRollouter` 持续生成 group。每当一个 group 完成：

```text
RolloutGroupResult
  -> optional group filter
  -> transfer_buffer.append(group)
```

当 `transfer_buffer` 达到阈值：

```text
len(transfer_buffer) >= transfer_batch_size_groups
```

则 flatten group 并写入同一个 partition：

```python
tq.put_rollout_batch(
    partition_id=f"train_{rollout_id}",
    samples=flatten(transfer_buffer),
)
```

推荐默认值：

```text
transfer_batch_size_groups =
  train_global_batch_size // num_iters_per_train_update // rollout.num_generations
```

若没有 `num_iters_per_train_update`，默认：

```text
transfer_batch_size_groups =
  train_global_batch_size // rollout.num_generations
```

一次 put 的 sample 数：

```text
transfer_batch_size_samples =
  transfer_batch_size_groups * rollout.num_generations
```

### TQ 字段

rollout 初始写入字段：

```text
input_ids
attention_mask
position_ids
labels
old_logps              # 从 trajectory["logprobs"] 提取
messages
turns
stop_reason
truncated
policy_version
rollout_id
partition_id
group_id
generation_idx
```

reward 追加字段：

```text
rewards
raw_rewards
reward_breakdown       # 例如 f1/cot/tool 等
```

advantage 追加字段：

```text
advantages
returns                # 如果算法需要
```

trainer 可追加或临时计算：

```text
ref_logps              # KL 开启时可以在 trainer 内计算，也可以拆 RefWorker
train_metrics
```

第一版可以把 `ref_logps` 留在 `TrainerWorker` 内部，复用 `grpo_baseline.py` 的逻辑：

```python
ref_outputs = model.forward_only(inputs=mb_inputs, disable_lora=True)
```

## GRPO 处理

GRPO 的 ready 条件是 group 完整：

```text
同一个 (partition_id, group_id)
  有 NUM_GENERATIONS 条 trajectories
  且每条都有 reward
```

`AdvantageWorker` 读取完整 group：

```python
total_rewards = [sample["rewards"] for sample in group]
advantages = GRPOAdvantage()(
    total_rewards,
    num_generations=NUM_GENERATIONS,
    scale="group",
)
```

然后按 `generation_idx` 写回：

```text
sample_0.advantages = advantages[0]
sample_1.advantages = advantages[1]
...
```

Trainer 不需要知道 reward 如何计算，只需要读取字段 ready 的 samples：

```text
required fields:
  input_ids
  labels
  old_logps
  advantages
```

## 控制流

控制流由 `BaseRLPipeline` 驱动，`StatelessManager` 是控制决策中心。`BaseRLPipeline` 负责启动 worker、周期性读取 TransferQueue metadata、调用 `StatelessManager.decide()` 并分发 `ControlDecision`；`StatelessManager` 决定哪个阶段可以推进、推进多少、是否同步权重。

```mermaid
sequenceDiagram
    participant P as BaseRLPipeline
    participant M as StatelessManager
    participant Q as TransferQueueClient
    participant R as AsyncRollouter
    participant L as MultiTurnRollout
    participant W as RewardWorker
    participant A as AdvantageWorker
    participant T as TrainerWorker
    participant C as CheckpointEngineManager

    P->>P: load_config()
    P->>P: build_components()
    P->>P: allocate_resources()
    P->>P: create_roles()
    P->>Q: init_transfer_queue()
    P->>M: start()
    P->>R: start()
    P->>W: start()
    P->>A: start()
    P->>T: start()

    loop run()
        P->>Q: get_metadata()
        P->>M: decide()

        alt rollout permits > 0
            P->>R: set_rollout_budget()
            R->>R: feed_pending_queue()
            R->>R: submit_until_capacity()
            R->>L: __call__()
            R->>R: drain_completed_groups()
            R->>R: flush_transfer_buffer()
            R->>Q: put_rollout_batch()
        else rollout_permits == 0
            P->>R: pause_submit()
        end

        alt reward permits > 0
            P->>W: run_reward()
            W->>Q: claim_reward_batch()
            W->>W: compute_rewards()
            W->>Q: append_rewards()
        end

        alt advantage permits > 0
            P->>A: run_advantage()
            A->>Q: claim_reward_ready_groups()
            A->>A: compute_advantages()
            A->>Q: append_advantages()
        end

        alt train_budget > 0
            P->>T: train()
            T->>Q: claim_train_batch()
            T->>T: forward_only()
            T->>T: forward_backward()
            T->>T: clip_grad_and_step()
            T->>Q: mark_trained()
            T-->>P: emit_train_event()
        end

        alt sync_weights
            P->>C: sync_weights()
            C->>R: receive_weights()
            P->>M: on_parameter_synced()
        end

        alt clear_partition
            P->>Q: clear_partition()
            P->>M: on_partition_cleared()
        end

        P->>P: log_metrics()
        P->>P: maybe_save_checkpoint()
    end

    P->>R: shutdown()
    P->>W: shutdown()
    P->>A: shutdown()
    P->>T: shutdown()
    P->>P: shutdown()
```

`ControlDecision` 是 manager 对本轮控制面的输出。它可以是 dataclass，也可以拆成多项 permit API；关键是 worker 不绕过 manager 自行推进关键阶段。

```python
@dataclass
class ControlDecision:
    rollout_permits: int
    reward_permits: int
    advantage_permits: int
    train_budget: int
    partition_id: str
    policy_version: int
    max_staleness: int
    sync_weights: bool = False
    clear_partitions: list[str] = field(default_factory=list)
```

`BaseRLPipeline` 是总入口，但它的主循环应围绕 `get_metadata()` 和 `decide()` 展开：

```python
class BaseRLPipeline:
    def run(self):
        self.load_config()
        self.build_components()
        self.allocate_resources()
        self.create_roles()
        self.init_transfer_queue()

        self.manager.start()
        self.rollouter.start()
        self.reward_worker.start()
        self.advantage_worker.start()
        self.trainer_worker.start()

        while not self.should_stop():
            metadata = self.tq.get_metadata()
            decision = self.manager.decide(
                metadata=metadata,
                policy_version=self.policy_version,
            )
            self.dispatch(decision)
            self.log_metrics(metadata)
            self.handle_failures(metadata)
            self.maybe_save_checkpoint()

        self.shutdown()
```

`dispatch(decision)` 只负责把 manager 的决策传给对应 worker：

```text
rollout_permits > 0    -> AsyncRollouter 可以继续提交 group task
reward_permits > 0     -> RewardWorker 可以 claim rollout-ready samples
advantage_permits > 0  -> AdvantageWorker 可以 claim reward-ready groups
train_budget > 0       -> TrainerWorker 可以 claim train-ready samples
sync_weights           -> CheckpointEngineManager 执行权重同步
clear_partitions       -> TransferQueueClient 清理已训练 partition
```

## 背压策略

fully async 的背压分三层。

### pending_queue 背压

限制待 rollout 的 prompt group 数：

```python
pending_queue = asyncio.Queue(maxsize=queue_size)
```

`queue_size` 是用户可见参数，表示最多缓存多少个待 rollout 的 prompt group。dataset feeding 太快时，`pending_queue.put()` 会阻塞。

### active_tasks 背压

限制同时运行的 prompt group 数：

```text
concurrency = auto
  由系统根据 rollout replicas 和每个 replica 的建议并发数推导

concurrency = 64
  用户显式指定最多同时运行 64 个 prompt group
```

提交任务前：

```python
max_concurrent_groups = resolve_concurrency(concurrency)

while len(active_tasks) >= max_concurrent_groups:
    done, active_tasks = await asyncio.wait(
        active_tasks,
        return_when=asyncio.FIRST_COMPLETED,
    )
    handle_done(done)
```

### TransferQueue / staleness 背压

限制 rollout 领先 trainer 的数据量和参数版本差距。

按照 Relax-style 第一版设计，只暴露一个用户参数：

```text
max_staleness
```

`max_staleness` 是整数，单位是 `train_k` partition / policy version：

```text
max_staleness = 0
  最多只有当前未完成 train_k；rollout 不领先 trainer。

max_staleness = 1
  rollout 可以比 trainer 多保留 1 个未完成 train_{k+1}。

max_staleness = 2
  rollout 可以再多保留 1 个未完成 train_{k+2}。
```

Manager 内部派生：

```text
max_live_partitions = max_staleness + 1
```

第一版不支持 verl-style 的非整数 stale sample budget，也不支持 mixed-version train batch。trainer 只能从同一个 `train_k` / 同一个 `policy_version` claim 训练数据。

当满足下面条件时，`StatelessManager.decide()` 生成的 `ControlDecision.rollout_permits` 应为 `0`：

```text
active_partitions >= max_live_partitions
or tq_bytes >= max_tq_bytes
```

`max_tq_bytes` 是系统保护参数，默认可以不暴露给算法用户。

### 版本一致性

Relax-style 第一版使用 partition-consistent batch：

```text
claim_train_batch(partition_id=train_k)
  -> batch 内所有 samples 来自同一个 train_k
  -> batch 内所有 samples 来自同一个 policy_version
```

这意味着：

```text
policy_version(train_k) = k 对应的 rollout 参数版本
trainer train train_k
trainer sync weights -> policy_version = k + 1
clear train_k
```

如果一个 partition 因为 `max_staleness` 超限而已经不可训练，manager 应整体 drop 或 clear 该 partition，而不是从多个 partition 混合凑 batch。mixed-version batch 不属于第一版目标。

## 权重同步

权重同步由 trainer 控制，rollout 只接收新版本。

第一版推荐策略：

```text
Trainer 每训练完一个 train_k partition:
  1. 完成该 partition 内所有 optimizer steps
  2. checkpoint engine 同步权重到 sampler
  3. current_policy_version += 1
  4. clear train_k
```

一个 partition 内 optimizer step 数：

```text
num_steps_per_partition =
  target_samples_per_partition // train_global_batch_size
```

第一版不把权重同步频率暴露成独立参数，默认策略固定为：

```text
sync_on_partition_done = true
```

也就是每个 `train_k` 训练完成后同步一次权重。`trigger_parameter_sync_step`、partial rollout、mixed-version batch 都不属于第一版目标。

## 同步与异步模式

同一套 pipeline 通过一个高层 `mode` 切换模式。用户不直接配置 `max_live_partitions`、`sync_on_partition_done` 这类底层参数。

推荐用户可见参数：

```yaml
async_training:
  mode: sync                 # sync / async
  max_staleness: 0           # rollout 最多领先 trainer 几个未完成 train_k
```

默认行为：

| mode | 默认 `max_staleness` | 版本一致性 | 语义 |
|---|---:|---|---|
| `sync` | `0` | partition-consistent | 等当前 `train_k` 完整训练、同步、清理后再进入下一轮 |
| `async` | `1` | partition-consistent | rollout 可以领先 trainer 一个 `train_k`，trainer batch 仍来自单一版本 |

Manager 派生参数：

```text
sync_on_partition_done = true
max_live_partitions = max_staleness + 1
```

### on-policy sync

```yaml
async_training:
  mode: sync
```

语义：

```text
rollout train_k
reward train_k
advantage train_k
trainer train_k
sync weights
clear train_k
进入 train_{k+1}
```

这等价于当前 `grpo_baseline.py` 的严格同步流程，只是数据通过 TQ 传递。

### group-level async

```yaml
async_training:
  mode: async
  max_staleness: 1
```

语义：

```text
多个 prompt group 并发 rollout
ready groups 分批 append 到 train_k
reward/advantage/trainer 持续消费 TQ
trainer 完成 train_k 后同步权重
```

这是第一版推荐 fully async 模式。

## TransferQueueClient 接口

```python
class TransferQueueClient:
    def put_rollout_batch(
        self,
        *,
        partition_id: str,
        samples: list[dict],
    ) -> Metadata:
        ...

    def claim_reward_batch(
        self,
        *,
        partition_id: str,
        max_samples: int,
    ) -> list[dict]:
        ...

    def append_rewards(
        self,
        *,
        metadata: Metadata,
        rewards: dict,
    ) -> None:
        ...

    def claim_reward_ready_groups(
        self,
        *,
        partition_id: str,
        num_generations: int,
        max_groups: int,
    ) -> list[list[dict]]:
        ...

    def append_advantages(
        self,
        *,
        group_metadata: list[Metadata],
        advantages: list[float],
    ) -> None:
        ...

    def claim_train_batch(
        self,
        *,
        partition_id: str,
        required_fields: list[str],
        batch_size: int,
    ) -> list[dict]:
        ...

    def mark_trained(self, *, metadata: Metadata) -> None:
        ...

    def clear_partition(self, *, partition_id: str) -> None:
        ...

    def get_metadata(self) -> QueueMetadata:
        ...
```

`TransferQueueClient` 的职责是封装底层 TQ API，不让 pipeline 依赖具体实现。高吞吐 tensor 数据优先走 native `put/get_meta/get_data`；细粒度状态更新可以走 KV API。对上层只暴露 sample/group/partition 语义。

控制面 metadata 不包含 trajectory/token/reward payload，只包含 manager 做决策需要的状态：

```python
@dataclass
class QueueMetadata:
    active_partitions: list[PartitionMetadata]
    total_bytes: int | None
    trainer_step: int
    policy_version: int


@dataclass
class PartitionMetadata:
    partition_id: str
    rollout_id: int
    policy_version: int
    target_groups: int
    rollout_done_groups: int
    reward_done_groups: int
    advantage_done_groups: int
    trained_groups: int
    dropped_groups: int
    is_rollout_done: bool
    is_train_done: bool
```

## StatelessManager 接口

```python
class StatelessManager:
    def decide(
        self,
        *,
        metadata: QueueMetadata,
        policy_version: int,
    ) -> ControlDecision:
        """控制面唯一入口。根据 TQ metadata 和参数版本返回执行决策。"""
        ...

    def rollout_partition(self) -> str:
        """返回当前 rollout 应写入的 train_k partition。"""
        ...

    def on_parameter_synced(self, *, new_policy_version: int) -> None:
        ...

    def on_partition_trained(self, *, partition_id: str) -> None:
        ...

    def on_partition_cleared(self, *, partition_id: str) -> None:
        ...
```

`decide()` 内部可以拆 helper，例如：

```text
compute_rollout_permits(metadata)
compute_reward_permits(metadata)
compute_advantage_permits(metadata)
compute_train_budget(metadata)
compute_sync_decision(metadata)
compute_clear_decision(metadata)
```

Manager 可以保存少量运行时计数，例如当前 `rollout_id`、`policy_version`、`trainer_step`。它不能保存 sample payload，也不能成为 replay buffer。

## 用户定制钩子

本设计需要让任务用户只改任务相关逻辑，不改 TransferQueue 数据面和基础调度。推荐把可定制点分成六层：pipeline、rollout/env/tool、AsyncRollouter、reward/advantage/filter、trainer、manager。

### Pipeline 级钩子

面向需要新增算法流程或替换组件组合的用户。默认通过 YAML 配置组件；高级用户可以继承 `BaseRLPipeline`。

```python
class BaseRLPipeline:
    def build_components(self) -> None:
        """创建 model、sampler、rollout、reward、advantage、trainer、manager。"""

    def allocate_resources(self) -> None:
        """创建 DeviceMesh、remote groups、TransferQueue backend。"""

    def create_roles(self) -> None:
        """创建 AsyncRollouter / RewardWorker / AdvantageWorker / TrainerWorker。"""

    def dispatch(self, decision: ControlDecision) -> None:
        """把 StatelessManager.decide() 的结果分发给各 worker。"""

    def should_stop(self) -> bool:
        """控制训练停止条件，例如 max_steps、max_epochs、外部 stop signal。"""
```

定制边界：

- 可以替换 worker 类型、reward/advantage 算法、日志和 checkpoint 策略。
- 不建议在 pipeline 中直接读写 TQ payload；payload 读写应走 worker 和 `TransferQueueClient`。

### Rollout / Env / Tool 钩子

面向 agentic 任务用户。用户可以定义多轮交互、tool 调用、tool 结果解析、环境状态更新。

```python
class BaseInteractionEnv:
    def reset(self, prompt: Trajectory) -> tuple[dict, dict]:
        """初始化环境状态，返回 observation 和 reset_info。"""

    def step(self, response_text: str) -> tuple[dict, bool, dict]:
        """处理模型 response，执行 tool/env 逻辑，返回 observation、done、info。"""

    def format_observation(self, observation: dict) -> list[dict]:
        """把环境 observation 转成可追加到 messages 的内容。"""
```

对于当前代码路径，第一版可以先通过 `ToolManager` 和 `MultiTurnRollout` 定制：

```python
class CustomToolManager(ToolManager):
    def __call__(self, tool_call: dict) -> str:
        """执行 tool call，并返回 tool message content。"""
```

推荐约定：

- env/tool 负责交互过程和 observation。
- reward 不放在 env 内，reward 独立由 `RewardWorker` 计算。
- env/tool 可以把诊断信息写入 trajectory metadata，例如 `tool_calls`、`env_infos`、`truncated_reason`。

### AsyncRollouter 钩子

面向需要控制 rollout 生产行为的用户。默认按 prompt group 生产；用户可以定制 group 构造、过滤、超时和重试。

```python
class AsyncRollouter:
    def build_group_request(self, prompt: Trajectory, group_idx: int) -> RolloutGroupRequest:
        """从 prompt 构造 RolloutGroupRequest。"""

    def should_keep_group(self, result: RolloutGroupResult) -> bool:
        """判断 ready group 是否进入 transfer_buffer。"""

    def on_group_timeout(self, request: RolloutGroupRequest) -> str:
        """返回 retry / drop。"""

    def annotate_trajectory(
        self,
        traj: Trajectory,
        request: RolloutGroupRequest,
        generation_idx: int,
    ) -> Trajectory:
        """写入 partition_id、group_id、generation_idx、policy_version 等字段。"""
```

定制边界：

- 可以调整 group filter、timeout、retry 和 trace 逻辑。
- 不应在这里计算 reward/advantage，也不应执行 optimizer step。

### Reward / Advantage / Filter 钩子

面向算法和任务评价定制。Reward 与 Advantage 通过 worker 接入 TQ。

```python
class RewardFn:
    def __call__(self, trajectories: list[Trajectory]) -> dict[str, list[float]]:
        """返回 total reward 和 reward breakdown。"""


class AdvantageFn:
    def __call__(
        self,
        rewards: list[float],
        *,
        num_generations: int,
        group_ids: list[str],
    ) -> list[float]:
        """按 group 计算 advantage。GRPO 默认要求 group 完整。"""
```

可选 filter：

```python
class GroupFilter:
    def __call__(self, group: list[Trajectory], rewards: dict[str, list[float]]) -> bool:
        """过滤无训练信号、格式错误、超长或无效 group。"""
```

推荐约定：

- reward 输出 `rewards` 和 `reward_breakdown`。
- advantage 输出按 sample 对齐的 `advantages`。
- homogeneous group 过滤应是可配置策略，不写死在 TransferQueue。

### Trainer 钩子

面向算法训练细节定制。第一版 trainer batch 是 partition-consistent，因此 trainer 只从单个 `train_k` claim 数据。

```python
class TrainerWorker:
    def build_train_inputs(self, samples: list[dict]) -> dict:
        """从 TQ sample 构造 model.forward_backward 输入。"""

    def compute_ref_logps(self, inputs: list[dict]) -> Any:
        """KL 开启时计算 ref_logps；默认可复用 model.forward_only(disable_lora=True)。"""

    def train_batch(self, samples: list[dict]) -> dict:
        """执行 forward_backward、clip_grad_and_step，并返回 metrics。"""

    def should_skip_batch(self, samples: list[dict]) -> bool:
        """可选跳过无训练信号 batch。"""
```

定制边界：

- 可以替换 loss、metric、ref_logps 路径和 batch filter。
- 不应绕过 `TransferQueueClient.mark_trained()` 更新训练状态。

### StatelessManager 钩子

面向控制策略定制。第一版默认 Relax-style：整数 `max_staleness`、partition-consistent batch、train_k 完成后 sync。

```python
class StatelessManager:
    def decide(self, *, metadata: QueueMetadata, policy_version: int) -> ControlDecision:
        """根据 metadata 返回 permits、sync、clear 决策。"""

    def compute_rollout_permits(self, metadata: QueueMetadata) -> int:
        """根据 max_staleness 和 active partition 数控制 rollout 背压。"""

    def compute_train_budget(self, metadata: QueueMetadata) -> int:
        """只为单个可训练 partition 发放 train budget。"""

    def compute_sync_decision(self, metadata: QueueMetadata) -> bool:
        """train_k 完成后触发权重同步。"""
```

定制边界：

- 可以调整 permits、清理策略、停止条件和 metric 上报。
- 不应持有 sample payload，不应实现 replay buffer。
- 第一版不提供 mixed-version batch 和 sample-level staleness，自定义 manager 也不应绕过这个约束。

### 用户最小接入面

普通任务用户通常只需要提供：

```text
1. dataset / preprocessor
2. rollout prompt template
3. ToolManager 或 BaseInteractionEnv
4. RewardFn
5. YAML 配置：partition.target_groups、rollout.num_generations、async_training.max_staleness
```

算法用户才需要扩展：

```text
AdvantageFn
TrainerWorker
StatelessManager
BaseRLPipeline
```

## 失败处理

### rollout group timeout

如果 group 超时：

```text
status = timeout
```

默认策略：

```text
abort group
重新入队
超过 max_retries 后 drop group
```

如果 drop group 导致 partition 样本不足，AsyncRollouter 继续从 dataloader 取新的 group 补齐 target。

### reward homogeneous group

当前 `grpo_baseline.py` 会跳过全对或全错比例过高的 batch。异步后建议把 filter 下沉到 group 或 transfer batch：

```text
group rewards 全同且无训练信号:
  标记 filtered
  不进入 train-ready 状态
```

是否过滤应由算法配置决定，不应写死在 TQ。

### stale partitions

如果 partition 的版本差距超过阈值：

```text
current_policy_version - partition.policy_version > max_staleness
```

默认：

```text
mark partition dropped
不进入 trainer
```

第一版不在同一个 trainer batch 内混合多个 policy version；stale 判断以 partition 为单位。

## YAML 示例

```yaml
pipeline:
  class: AsyncAgenticGRPOPipeline

model:
  class: MegatronModel
  model_id: ms://Qwen/Qwen3.5-4B
  loss: GRPOLoss
  metric: GRPOMetric

sampler:
  class: vLLMSampler
  engine_args:
    max_model_len: 32768
    enable_lora: true
    max_lora_rank: 32

rollout:
  class: MultiTurnRollout
  max_turns: 6
  num_generations: 8
  max_trajectory_tokens: null

partition:
  target_groups: 128

async_rollout:
  queue_size: 128
  concurrency: auto
  transfer_batch_size_groups: auto
  group_timeout_s: 600
  max_retries: 1

transfer_queue:
  partition_prefix: train

async_training:
  mode: async
  max_staleness: 1

reward:
  class:
    - F1Reward
    - CoTReward
  weights:
    f1: 1.0
    cot: 0.2

advantage:
  class: GRPOAdvantage
  scale: group

trainer:
  train_global_batch_size: 64
  micro_batch_size: 2
  gradient_accumulation_steps: 1
  kl_beta: 0.02
```

## 非目标

第一版不做：

- token-level partial generation resume。
- 混 policy version trajectory 训练。
- 在 TransferQueue 内实现 replay buffer 或采样策略。
- 重写 vLLM engine。
- 把 reward 逻辑放入 env；env/tool 交互由 rollout 或用户自定义 agent loop 负责，reward 独立计算。

## 结论

Twinkle 的 fully async agentic RL 应采用三层结构：

```text
BaseRLPipeline:
  全局训练编排

AsyncRollouter:
  prompt group 级异步 rollout producer

TransferQueue:
  rollout / reward / advantage / trainer 之间的数据容器
```

第一版不需要修改 `vLLMSampler` 的核心能力，也不需要新增 `AsyncMultiTurnRollout`。关键改动是把当前 batch-blocking 的 rollout handoff 改成：

```text
prompt group ready
  -> transfer buffer
  -> append train_k
  -> trainer rolling consume
```

这样可以保留 GRPO group 语义，同时显著缓解 rollout step 内样本间长尾，并为后续 sample-level/trajectory-level streaming runtime 留出扩展空间。
