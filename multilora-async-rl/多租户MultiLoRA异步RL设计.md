# 多租户 Multi-LoRA 异步 RL 设计

## 1. 背景与目标

客户端多租户异步 RL 的核心问题不是简单支持多个 LoRA adapter，而是要在同一套训练服务中同时隔离和路由：

```text
权重:
  base model 可以共享，LoRA adapter 必须隔离。

数据:
  TransferQueue 中的 partition、row、metadata、claim/ack/clear 必须隔离。

环境:
  不同租户或任务可能使用不同 env、tool、sandbox、browser、simulator。

Reward / loss:
  不同任务可能使用不同 reward_fn、advantage 规则和 loss 计算方式。

资源:
  rollout 并发、工具并发、TQ 容量、trainer slots 需要按租户和训练任务限额。
```

本设计面向“客户端提交多租户训练任务，服务端共享基础模型和异步 RL 基础设施”的场景。用户只需要声明训练身份、工具 profile、reward、loss 和配置；底层 pipeline 负责 TransferQueue 数据面、staleness 控制、权重同步和资源隔离。

第一版交互能力收敛到 `ToolManager` 级别：

```text
支持:
  native tool:
    平台内置、受信任的本地工具，例如 extract_condensed、受控 sandbox wrapper。

  remote tool:
    平台内置 RemoteTool wrapper 通过 HTTP/gRPC 调用用户自有工具服务。

暂不支持:
  server 直接 import 用户自定义 Env Python 代码。
  通用 EnvFactory / reset / step / close 协议作为热路径能力。
```

`Env` 和 `ToolManager` 不是互斥关系。长期设计中，Env 是真实环境交互层，ToolManager 是工具分发层；Env 可以持有 ToolManager。但第一版为了降低实现复杂度，只落地 `MultiTurnRollout + ToolManager`，把浏览器、游戏、复杂 simulator 这类强状态 Env 放到后续版本。

## 2. 核心概念

### 2.1 LoraContext

`LoraContext` 是一次训练任务在多租户系统中的路由和隔离身份。它不保存训练样本，只用于决定当前请求应该访问哪份数据、哪份 LoRA 权重、哪个环境、哪个 reward/loss 逻辑。

```python
@dataclass
class LoraContext:
    tenant_id: str
    training_run_id: str
    base_model_id: str
    adapter_name: str
    adapter_revision: str | None
    policy_version: int

    env_type: str
    tool_profile: str
    reward_type: str
    loss_type: str
    algorithm: str
```

字段含义：

```text
tenant_id:
  租户、客户、业务方或项目空间。用于权限、quota、计费和租户级清理。

training_run_id:
  tenant 下的一次具体训练任务。用于 checkpoint、恢复、取消和 run 级生命周期。
  本方案约定一个 training_run 固定只训练一个 LoRA adapter。

base_model_id:
  共享基础模型标识。多个 tenant 可以共享同一个 base model。

adapter_name:
  当前训练任务使用的 LoRA adapter 名称。它与 training_run_id 一对一绑定。

adapter_revision:
  可选，用于区分已发布或已恢复的 adapter 版本。

policy_version:
  rollout 使用的策略版本。每次 train_k 完成训练并同步权重后递增。
  policy_version 是 sample 级元数据，不是 partition 唯一键。
  一个 train_k 可以包含不同 policy_version 生成的 samples，但每条 sample 必须记录自己的 policy_version 和 old_logps。

env_type:
  环境类型。第一版只作为兼容和预留字段，默认可设为 tool_calling。

tool_profile:
  当前训练任务允许使用的工具集合配置，例如 code_agent、condense_agent。
  ToolManagerFactory 根据该字段创建 tenant/run/adapter 作用域内的 ToolManager。

reward_type:
  reward 实现类型，例如 unit_test_reward、format_reward、web_task_reward。

loss_type / algorithm:
  训练算法和 loss 选择，例如 grpo、ppo、dapo。
```

### 2.2 LoraAdapterRegistry 与 AdapterRecord

`LoraContext` 和 `LoraAdapterRegistry` 的边界必须分清：

```text
LoraContext:
  静态身份与路由信息。
  用来回答“这条数据/请求属于哪个 tenant、哪个 training_run、哪个 adapter、哪个 reward/loss”。

LoraAdapterRegistry:
  运行时状态表。
  用来回答“这个 adapter 当前能不能 rollout、能不能 train、当前权重版本是多少、有哪些 live partitions”。
```

二者通过 `LoraContext.key` 建立关系：

```python
LoraContext.key = "{tenant_id}/{training_run_id}/{adapter_name}"

LoraAdapterRegistry.records[LoraContext.key] -> AdapterRecord
```

`LoraContext` 不应该承载可变运行状态。下面这些字段由 `AdapterRecord` 管理：

```text
policy_version:
  当前 adapter 最新可 rollout 的策略版本。训练并同步完一个 train_k 后递增。

adapter_revision:
  当前 adapter 最新 checkpoint / adapter_path。

live_partitions:
  当前尚未 clear 的 train_k partitions。

in_flight_rollouts:
  当前正在生成、尚未写完的 rollout prompt group 数。

training_partition:
  当前正在训练的 train_k。一个 adapter 同一时刻最多训练一个 partition。

sync_in_progress:
  当前 adapter 是否正在权重同步。同步期间阻塞该 adapter 的新 rollout / train。

state:
  adapter 生命周期状态，例如 ACTIVE / FAILED / CANCELLED。
```

运行时典型访问方式：

```python
context = LoraContext(...)
record = adapter_registry.register(context)

current = adapter_registry.get(context)
rollout_context = context.with_policy_version(
    current.policy_version,
    current.adapter_revision,
)
```

### 2.3 AdapterRecord 状态转换

`AdapterRecord.state` 只表示 adapter 的高层生命周期；`in_flight_rollouts`、`training_partition`、`sync_in_progress` 和 `live_partitions` 是更细的运行时状态。当前第一版核心状态转换如下：

```mermaid
stateDiagram-v2
    [*] --> ACTIVE: register(context)

    ACTIVE --> ACTIVE: on_rollout_started / in_flight_rollouts += 1
    ACTIVE --> ACTIVE: on_rollout_finished / in_flight_rollouts -= 1
    ACTIVE --> ACTIVE: on_partition_created / live_partitions.add(train_k)
    ACTIVE --> ACTIVE: on_partition_cleared / live_partitions.remove(train_k)

    ACTIVE --> TRAINING: on_train_started(train_k)
    TRAINING --> ACTIVE: on_train_finished(train_k)

    ACTIVE --> SYNCING: on_weight_sync_started()
    SYNCING --> ACTIVE: on_weight_sync_finished(adapter_revision)\npolicy_version += 1

    ACTIVE --> FAILED: mark_failed(error)
    TRAINING --> FAILED: mark_failed(error)
    SYNCING --> FAILED: mark_failed(error)
    FAILED --> [*]
```

实现上 `TRAINING` / `SYNCING` 不一定是 `LoraAdapterState` 枚举值，而是由字段表达：

```text
TRAINING:
  record.training_partition is not None

SYNCING:
  record.sync_in_progress == True

FAILED:
  record.state == FAILED
```

因此调度判断不是只看一个 state 字段，而是组合判断：

```text
can_accept_rollout(context):
  record.state == ACTIVE
  and not record.sync_in_progress

can_train(context):
  record.state == ACTIVE
  and not record.sync_in_progress
  and record.training_partition is None
```

事件与字段更新关系：

| 事件 | 更新字段 | 作用 |
|---|---|---|
| `register(context)` | 创建 `AdapterRecord` | 注册一个 LoRA 训练上下文 |
| `on_rollout_started(context)` | `in_flight_rollouts += 1` | 标记有 rollout task 在途 |
| `on_rollout_finished(context)` | `in_flight_rollouts -= 1` | 释放 rollout 在途计数 |
| `on_partition_created(context, train_k)` | `live_partitions.add(train_k)` | 参与 staleness / 容量控制 |
| `on_partition_cleared(context, train_k)` | `live_partitions.remove(train_k)` | 释放 staleness / TQ 容量 |
| `on_train_started(context, train_k)` | `training_partition = train_k` | 阻止同 adapter 并发训练多个 partition |
| `on_train_finished(context, train_k)` | `training_partition = None` | 释放训练占用 |
| `on_weight_sync_started(context)` | `sync_in_progress = True` | 阻塞该 adapter 新 rollout / train |
| `on_weight_sync_finished(context, adapter_revision)` | `policy_version += 1`, `adapter_revision = ...`, `sync_in_progress = False` | 推进权重版本 |
| `mark_failed(context, error)` | `state = FAILED`, `last_error = error` | 从调度中剔除该 adapter |

### 2.4 namespace

TransferQueue 中的数据必须按 `LoraContext` 生成 namespace：

```text
{tenant_id}/{training_run_id}/{adapter_name}/train_{k}
```

示例：

```text
tenant_a/code_grpo_001/code_lora/train_7
tenant_b/browser_rl_003/browser_lora/train_2
```

文档中可以继续使用 `train_k` 描述逻辑生命周期，但底层写入、读取、claim、ack、clear 都必须带完整 namespace。

第一版采用严格的 train_k 与 TQ partition 一一对应关系：

```text
train_k == TQ partition suffix

train_id = 3
  -> partition_id = {tenant_id}/{training_run_id}/{adapter_name}/train_3
```

因此同一个 `(tenant_id, training_run_id, adapter_name, train_k)` 只能存在一个 live partition。该 partition 处于 `OPEN` 时，AsyncRollouter 可以继续向其中追加 prompt groups；一旦达到 `pipeline.rollout.batch_size` 并 seal，不能继续追加 rollout 数据。

`policy_version` 不决定 partition id。它记录每条 sample 由哪个 rollout 权重版本生成。同一个 `train_k` 可以混合多个 `policy_version`，但 partition 内仍然不能混 tenant、training_run、adapter、reward_type、loss_type 或 algorithm。

### 2.5 隔离边界

第一版不支持跨 context 混合训练：

```text
同一个 train_k:
  只能属于一个 tenant_id / training_run_id / adapter_name。
  只能对应一个 TransferQueue partition。
  可以包含多个 policy_version 的 samples，但每条 sample 必须保留自己的 policy_version / adapter_revision / old_logps。

同一个 GRPO group:
  只能来自同一个 prompt、同一个 adapter。是否允许混 policy_version 由 advantage 规则决定；
  GRPO 默认建议同一个 group 内保持同一个 policy_version。

同一个 optimizer batch:
  只能来自同一个 loss schema 和 reward schema。

同一个 ToolManager:
  不能跨 tenant / run 复用。ToolManager 内的 native/remote tool 必须按当前 context 做权限和 quota 校验。
```

## 3. 总体架构

![alt text](多租户异步rl.png)

| 序号 | 调用 | 说明 |
|---:|---|---|
| 1 | `submit_training_job(config)` | `Client` 提交训练任务配置。配置里包含租户、基础模型、LoRA adapter、数据源、tool profile、reward/loss、异步训练参数等。 |
| 2 | `register_adapter(context)` | `BaseRLPipeline` 构造 `LoraContext` 后，把当前训练任务的 LoRA 注册到 `LoraAdapterRegistry`。一个 `training_run_id` 固定对应一个 `adapter_name`。 |
| 3 | `init_namespace(context)` | `BaseRLPipeline` 让 `TransferQueueDataPlane` 初始化 TQ namespace，例如 `{tenant_id}/{training_run_id}/{adapter_name}/train_k`，并写入基础 metadata 约束。 |
| 4 | `get_metadata()` | `TransferQueueDataPlane` 向 `StalenessManager` 提供当前 live partitions、oldest partition、partition 状态和 policy_version 等容量事实。 |
| 5 | `capacity / throttle hint` | `StalenessManager` 按当前 adapter 的 `max_staleness` 计算还能继续提交多少 rollout，以及是否需要 throttle 或 sleep。 |
| 6 | `can_accept_rollout(context)` | `AsyncRollouter` 询问 `LoraAdapterRegistry` 当前 adapter 是否处于可 rollout 状态，并检查 `in_flight_rollouts`、`live_partitions`、同步状态等运行时状态。 |
| 7 | `check_capacity(context)` | `AsyncRollouter` 在提交前向 `TransferQueueDataPlane` 检查目标 namespace 的 TQ 容量是否还能接收新的 rollout rows。 |
| 8 | `run_rollout(context)` | `AsyncRollouter` 选择一个 `LoraContext` 后启动 rollout task。一次 submit batch 内只包含同一个 tenant/run/adapter，但可以包含多个 prompt groups；`policy_version` 作为每条 sample 的生成版本写入 metadata。 |
| 9 | `call native / remote tool` | `MultiTurnRollout` 在多轮交互中通过 `ToolManager` 调用 native tool 或 remote tool API。第一版不 import 用户 Env Python 代码。 |
| 10 | `sample(adapter_name, policy_version)` | `MultiTurnRollout` 调用 `vLLMSampler` 生成模型回复。请求必须携带 `adapter_name` 和 `policy_version`，用于多 LoRA 路由和版本追踪。 |
| 11 | `put_rollout_batch(context, train_k)` | rollout 完成一批 trajectory group 后，由 `AsyncRollouter` 写入对应 `train_k` partition。写入时附带 sample metadata。 |
| 12 | `native TQ ops` | `TransferQueueDataPlane` 将 put/claim/append/clear 等操作转换为底层 TransferQueue backend 操作。 |
| 13 | `claim / append reward` | `RewardWorker` claim rollout-ready 数据，按 `reward_type` 计算 reward，并追加 reward 字段。 |
| 14 | `claim / append advantage` | `AdvantageWorker` claim reward-ready 数据，按算法计算 advantage/return，并追加字段。满足训练条件后将 partition 标记为 `TRAIN_READY`。 |
| 15 | `list_train_ready_partitions()` | `TrainerScheduler` 从 `TransferQueueDataPlane` 查询可训练 partition 候选集合。候选必须已经完成 rollout、reward、advantage。 |
| 16 | `next_partition(train_k)` | `TrainerScheduler` 选择下一个训练 partition。选择结果必须保持 `train_k` 内 adapter、loss/reward schema 一致；允许 rows 来自不同 `policy_version`。 |
| 17 | `iter(train_k)` | `TrainerWorker` 针对选中的 `train_k` 构建 `StreamingDataset / StreamingDataLoader`，开始按 batch 读取训练数据。 |
| 18 | `read / ack rows` | `StreamingDataset / StreamingDataLoader` 通过 `TransferQueueDataPlane` 从 TQ 读取 rows，并对已消费数据做 ack/progress 更新。 |
| 19 | `train_k done` | `TrainerWorker` 完成当前 partition 的全部 optimizer steps。`BaseRLPipeline` 不直接执行训练逻辑，只观察组件 step 结果。 |
| 20 | `sync_weights(adapter)` | `TrainerWorker` 在 `train_k` 边界触发当前 adapter 权重保存/同步回调；具体实现由 trainer 组件决定，例如 GRPO trainer 保存 LoRA 到 `adapter_path`。 |
| 21 | `receive_weights(adapter, version)` | `vLLMSampler` 接收新 adapter 权重，并更新 rollout 侧可用的 `policy_version`。 |
| 22 | `clear_partition(train_k)` | 权重同步完成后，`TrainerWorker` 通过 `TransferQueueDataPlane` 清理已训练完成的 `train_k`，释放 TQ 容量并推进 staleness 窗口。 |

关键约束：

```text
rollout submit batch:
  只能包含一个 LoraContext。

train_k:
  只能包含一个 adapter_name / reward_type / loss_type / algorithm。
  可以包含多个 policy_version 的 rows。

权重同步:
  训练完一个 train_k 后同步一次 adapter 权重。

partition 清理:
  必须在训练完成并且 rollout 侧权重同步成功后执行。
```

### 3.1 Pipeline 与组件图

`BaseRLPipeline` 的定位是运行时控制面，而不是算法逻辑容器。它负责：

```text
1. 初始化共享资源：LoraContext / TransferQueueDataPlane / LoraAdapterRegistry / StalenessManager
2. 创建组件：PromptLoader / AsyncRollouter / RewardWorker / AdvantageWorker / TrainerWorker
3. 创建角色图：默认 GRPO 使用 `create_grpo_roles()`；其他算法由子类覆盖 `create_roles()`
4. 按顺序调用 component.step()
5. 关闭组件和共享资源
```

`BaseRLPipeline` 的构造入口只接收 `config`。模型、rollout、TQ 数据面、registry、staleness manager、reward/advantage 函数、调度策略等都由 `build_*()` 方法根据 config 创建：

```text
BaseRLPipeline(config)
  -> build_model()
  -> build_rollout()
  -> build_data_plane()
  -> build_reward_registry()
  -> build_advantage_fn()
  -> build_rollout_policy()
  -> build_train_policy()
  -> build_prompt_loaders()
```

测试或特殊任务如果需要 fake 资源，不应通过构造函数额外传入一组 runtime 属性，而应通过子类覆盖对应 `build_*()` 方法。这样生产初始化路径始终是 config-driven 的。

默认 GRPO 组件图是：

```text
PromptLoader
  -> AsyncRollouter
  -> RewardWorker
  -> AdvantageWorker
  -> MultiLoraGRPOTrainerWorker
```

第一版 server-side GRPO 具体实现类为：

```python
twinkle_agentic.async_rl.AsyncMultiLoraGRPOPipeline
```

该类继承 `BaseRLPipeline`，构造入口为 `cfg/model_mesh/sampler_mesh`，内部通过 `build_model()`、`build_rollout()`、`build_data_plane()`、`build_prompt_loaders()`、`build_reward_registry()` 和 `build_advantage_fn()` 从 YAML 配置创建资源。

不同算法不应该在 `BaseRLPipeline` 中增加大量训练细节分支。`BaseRLPipeline.create_roles()` 只提供默认 GRPO 角色图；DPO、SFT 或新的 RL 算法应通过子类覆盖 `create_roles()`，在该方法里创建自己的角色图。

```text
DPO:
  PairFeeder -> DPOTrainerWorker

GRPO:
  PromptLoader -> AsyncRollouter -> RewardWorker -> AdvantageWorker -> GRPOTrainerWorker
```

因此 `BaseRLPipeline` 学习 Relax Controller 的地方是“控制面负责启动角色，角色自治运行”，但不把具体算法训练细节放进控制面。

### 3.2 时序图

```mermaid
sequenceDiagram
    participant C as Client
    participant P as BaseRLPipeline
    participant AR as LoraAdapterRegistry
    participant Q as TransferQueueDataPlane
    participant SM as StalenessManager
    participant R as AsyncRollouter
    participant MR as MultiTurnRollout
    participant TF as ToolManagerFactory
    participant TM as ToolManager
    participant V as vLLMSampler
    participant RW as RewardWorker
    participant AW as AdvantageWorker
    participant TS as TrainerScheduler
    participant TW as TrainerWorker
    participant SD as StreamingDataset
    participant CK as CheckpointEngineManager

    C->>P: 1 submit_training_job(config)
    P->>AR: 2 register_adapter(context)
    P->>Q: 3 init_namespace(context)

    loop rollout producer loop
        Q->>SM: 4 get_metadata()
        SM-->>R: 5 capacity / throttle hint
        R->>AR: 6 can_accept_rollout(context)
        R->>Q: 7 check_capacity(context)
        alt capacity available
            R->>MR: 8 run_rollout(context)
            MR->>TF: create(sample, context)
            TF-->>MR: tool_manager
            MR->>TM: 9 call native / remote tool
            TM-->>MR: tool result
            MR->>V: 10 sample(adapter_name, policy_version)
            V-->>MR: response
            R->>Q: 11 put_rollout_batch(context, train_k)
            Q->>Q: 12 native TQ ops
        else throttle or sleep
            R->>R: sleep(throttle_hint)
        end
    end

    loop reward / advantage workers
        RW->>Q: 13 claim / append reward
        AW->>Q: 14 claim / append advantage
        Q-->>Q: mark_train_ready(train_k)
    end

    loop trainer loop
        TS->>Q: 15 list_train_ready_partitions()
        TS->>TW: 16 next_partition(train_k)
        TW->>SD: 17 iter(train_k)
        SD->>Q: 18 read / ack rows
        TW->>TW: 19 train_k done
        TW->>CK: 20 sync_weights(adapter)
        CK-->>TW: adapter_weights / checkpoint_path
        TW->>V: 21 receive_weights(adapter, version)
        TW->>Q: 22 clear_partition(train_k)
    end
```

这个时序图里有两个关键决策点：

```text
Rollout 侧:
  LoraAdapterRegistry 判断 adapter 是否 ACTIVE；
  StalenessManager 判断当前 context 是否还有 rollout capacity；
  AsyncRollouter 才决定 submit / throttle / sleep。

Trainer 侧:
  TransferQueueDataPlane 提供 TRAIN_READY partitions；
  LoraAdapterRegistry 过滤当前可训练 adapter；
  TrainerScheduler 选择下一个 train_k；
  TrainerWorker 根据 train_k.context.adapter_name 在 partition 边界切换 LoRA。
```

## 4. 多 LoRA 调度策略

多租户多 LoRA 的调度目标不是固定的。不同场景下可能需要不同策略：

```text
吞吐优先:
  尽量减少 rollout / reward / advantage / trainer / weight sync 的空泡。

公平优先:
  让不同 tenant / training_run / LoRA 按权重获得 rollout 和 train 机会。
```

因此调度逻辑应拆成两层：

```text
gating:
  判断某个 LoraContext 是否有资格参与调度。

policy:
  在所有有资格的候选中选择下一个 LoraContext 或 train_k。
```

所有策略都必须先经过 gating。公平策略不能绕过 staleness，吞吐策略也不能绕过租户隔离。

### 4.1 Rollout 调度

`AsyncRollouter` 负责多 LoRA rollout 侧调度。它内部维护 pending、active 和 completed 三类队列，不需要单独引入 `RolloutScheduler` 组件。

```text
pending_prompt_groups_by_context:
  context.key -> queue[prompt_group]

active_rollout_tasks:
  asyncio.Task -> RolloutTaskState(context, prompt_groups, group_count, submitted_at)

completed_rollout_results:
  queue[ComponentResult(kind="rollout")]
```

`AsyncRollouter.step()` 是非阻塞驱动：

```text
1. 回收 finished tasks。
2. 将已完成结果放入 completed_rollout_results。
3. 在 capacity 允许时继续 submit 新 rollout tasks。
4. 如果有 completed result，返回 kind="rollout"。
5. 如果本轮只提交了 task，返回 kind="rollout_submit"。
6. 如果无 pending、无完成结果、无可提交容量，返回 None。
```

rollout task 完成后立即通过 `TransferQueueDataPlane.put_rollout_batch()` 写入对应 `train_k` partition；不等待同一轮提交的其他 rollout task 全部完成。因此短任务可以先进入 TQ，长尾任务只拖慢自己。

推荐状态：

```python
@dataclass
class RolloutContextState:
    context: LoraContext
    pending_groups: int
    in_flight_rollouts: int
    live_partitions: int
    open_partitions: int
    train_ready_partitions: int
    rollout_capacity: int
    last_submit_time: float
    submitted_groups: int
    weight: float = 1.0
```

`AsyncRollouter.pick_next_rollout_context()` 的 gating 顺序：

```text
1. 当前 context 有 pending prompt group。
2. LoraAdapterRegistry 判断 adapter 处于 ACTIVE。
3. adapter 不在 sync_in_progress / draining / cancelled 状态。
4. StalenessManager 判断当前 context 仍有 rollout capacity。
5. TransferQueueDataPlane.check_capacity(context) 通过。
6. AsyncRollouter 全局 active prompt groups 未超过 `max_concurrent_groups`。
```

通过 gating 后再进入策略选择。

选中 context 后，`AsyncRollouter` 会提交一个 prompt group 对应的 rollout task。每轮可提交多少 task 由以下值共同限制：

```text
pending prompt groups
StalenessManager.rollout_capacity
当前 context active prompt groups
全局剩余 active prompt group capacity
```

默认建议：

```text
pipeline.rollout.batch_size 表示一个 train_k 收集多少个 prompt group。
一个 rollout task 始终只处理一个 prompt group。
```

这样可以让多个 prompt group 通过异步 task 并发进入 sampler / vLLM，同时不会在单个 task 内混多个 LoRA context。

#### 4.1.1 吞吐优先策略

吞吐优先策略是 work-conserving 的：只要系统还有容量，就尽量提交 rollout task。

推荐优先级：

```text
1. 优先选择没有 OPEN partition 的 context。
2. 再选择 live_partitions 最少的 context。
3. 再选择 in_flight_rollouts 最少的 context。
4. 同分时 round-robin。
```

这样可以让 trainer 未来更容易拿到 `TRAIN_READY` 数据，减少训练侧空等。

伪代码：

```python
def pick_work_conserving(candidates: list[RolloutContextState]):
    candidates = [c for c in candidates if c.rollout_capacity > 0]
    if not candidates:
        return None

    return min(
        candidates,
        key=lambda c: (
            c.open_partitions > 0,
            c.live_partitions,
            c.in_flight_rollouts,
            c.last_submit_time,
        ),
    ).context
```

#### 4.1.2 公平策略

公平策略用于平台多租户场景。目标是让不同 tenant / adapter 长期获得接近权重比例的 rollout 机会。

推荐使用 deficit round-robin，而不是简单 round-robin。原因是不同任务的 rollout 时长可能差异很大，简单轮转容易被长尾任务拖慢。

```python
class WeightedFairRolloutPolicy:
    def __init__(self, quantum: int):
        self.quantum = quantum
        self.deficit: dict[str, float] = defaultdict(float)

    def pick_next_context(self, candidates: list[RolloutContextState]):
        for state in self.round_robin(candidates):
            key = state.context_key
            self.deficit[key] += state.weight * self.quantum

            cost = 1  # 第一版按一个 prompt group 计费
            if self.deficit[key] >= cost:
                self.deficit[key] -= cost
                return state.context

        return None
```

第一版可以先把 `cost` 固定为 `1 prompt group`。后续如果要更精细，可以改为 token 数、预计 rollout 时长或实际资源消耗。

### 4.2 Trainer 调度

`TrainerScheduler` 负责训练侧调度。它只从 `TRAIN_READY` partition 中选择下一个 `train_k`，不负责 rollout 生产。

gating 顺序：

```text
1. partition.status == TRAIN_READY。
2. partition 内 metadata 同属一个 LoraContext。
3. LoraAdapterRegistry.can_train(context) 通过。
4. adapter 不在 sync_in_progress / cancelled 状态。
5. train_k 的 reward_type / loss_type / algorithm 与 trainer 可执行配置匹配。
```

通过 gating 后再进入策略选择。

#### 4.2.1 吞吐优先策略

吞吐优先策略要同时减少 trainer 空泡和 LoRA 切换成本：

```text
1. 如果当前 adapter 有 TRAIN_READY partition，继续训练当前 adapter。
2. 如果当前 adapter 没有 TRAIN_READY partition，立即切到其他 ready adapter。
3. 多个 adapter 都 ready 时，选择 ready_partition_count 最多的 adapter。
4. 同分时选择 oldest train_k。
```

伪代码：

```python
def pick_prefer_current(candidates, current_context):
    same = ready_for_context(candidates, current_context)
    if same:
        return oldest_partition(same)

    grouped = group_by_context(candidates)
    return max(
        grouped.items(),
        key=lambda item: (len(item[1]), -oldest_partition_id(item[1])),
    )[1][0]
```

该策略不会为了等待当前 LoRA 而让 trainer 空转。

#### 4.2.2 公平策略

公平训练策略用于要求不同 tenant / LoRA 按权重获得训练机会的场景。

第一版可以按 `train_k` 粒度做 weighted fair scheduling：

```python
class WeightedFairTrainPolicy:
    def __init__(self, quantum: int):
        self.quantum = quantum
        self.deficit: dict[str, float] = defaultdict(float)

    def pick_next_partition(self, candidates, current_context):
        grouped = group_by_context(candidates)

        for context, partitions in self.round_robin(grouped):
            key = context.key
            self.deficit[key] += context.weight * self.quantum

            cost = 1  # 第一版按一个 train_k 计费
            if self.deficit[key] >= cost:
                self.deficit[key] -= cost
                return oldest_partition(partitions)

        return None
```

后续可以把 `cost` 改为 `num_rows`、token 数或 optimizer step 数。

### 4.3 策略配置

推荐在配置里显式声明 rollout 和 trainer 两侧策略：

```yaml
multi_lora:
  rollout_schedule:
    policy: work_conserving   # work_conserving | fair
    fairness_unit: group
    weights:
      tenant_a/code_lora: 1.0
      tenant_b/math_lora: 1.0

  train_schedule:
    policy: prefer_current    # prefer_current | fair | fifo
    fairness_unit: partition
    switch_penalty: 0.0
    weights:
      tenant_a/code_lora: 1.0
      tenant_b/math_lora: 1.0
```

配置建议：

```text
追求吞吐、减少空泡:
  rollout_schedule.policy = work_conserving
  train_schedule.policy = prefer_current

追求租户公平:
  rollout_schedule.policy = fair
  train_schedule.policy = fair

高优租户:
  使用 fair 策略并提高该租户或 adapter 的 weight。
```

### 4.4 减少空泡的边界

减少空泡不能破坏数据一致性：

```text
允许:
  不同 submit batch 使用不同 LoRA。
  vLLMSampler 内部并发处理多个 LoRA 请求。
  adapter_a 权重同步时，adapter_b 继续 rollout 或 train。

不允许:
  一个 submit batch 混多个 LoRA。
  一个 train_k 混多个 LoRA。
  trainer 在 train_k 中途切换 adapter。
  sync_in_progress 的 adapter 继续提交新 rollout 或新 train_k。
```

`sync_in_progress` 是 per-adapter 状态，不是 global 状态。某个 adapter 同步权重时，只暂停该 adapter 的新 rollout / train，不阻塞其他 adapter。

## 5. Worker 租户隔离原则

### 5.0 Pipeline Component 模型

第一版将热路径执行单元统一建模为 pipeline component：

```python
step() -> ComponentResult | None
is_idle() -> bool
shutdown() -> None
```

`BaseRLPipeline` 是控制面，只负责初始化资源、创建组件、按顺序调用 `component.step()` 和关闭组件；它不直接搬运 reward/advantage/train 数据。

| 组件 | 类型 | 调度单位 | 数据入口 | 数据出口 | multi-LoRA 适配方式 |
|---|---|---|---|---|---|
| `PromptLoader` | source component | prompt batch | `twinkle.dataloader.DataLoader` | `AsyncRollouter` pending queue | 每个 `LoraContext` 一个 loader |
| `AsyncRollouter` | rollout producer | prompt group | pending queue | TQ `train_k` rollout rows | `pending_by_context` + staleness gating |
| `RewardWorker` | TQ transformer | rollout-ready partition | `TransferQueueDataPlane` | reward fields | 按 context 轮询 claim |
| `AdvantageWorker` | TQ transformer | reward-ready partition | `TransferQueueDataPlane` | advantages / returns | 按 context 轮询 claim |
| `TrainerWorker` | TQ consumer / optimizer | `TRAIN_READY train_k` | TQ-backed dataloader | adapter weights / partition clear | `TrainerScheduler` 选择 partition |

因此 multi-LoRA 支撑不是放在单个 scheduler 里，而是每个组件都遵守同一组约束：

```text
1. 输入必须绑定 LoraContext。
2. 内部状态必须按 context.key 隔离。
3. 输出必须写回同一个 context namespace。
4. 一个 train_k 不混 adapter。
```

多租户模式下，worker 可以共享进程池，但每次执行必须以 `LoraContext` 为隔离边界。隔离不是必须“一租户一个 worker”，而是必须保证 claim、compute、append、ack 都不能跨 namespace。

### 5.1 通用隔离要求

所有处理 TQ 数据的 worker 都必须满足：

```text
claim:
  只能从当前 context 对应 namespace claim 数据。

compute:
  只能使用当前 context 指定的 reward_type / loss_type / tool_profile / algorithm。

append:
  写回字段时必须校验 sample metadata 和 context 一致。

state:
  不能跨 tenant / training_run 复用有状态 client、缓存、sandbox、token 或 ToolManager。

permission:
  访问 remote tool、remote reward、secret、文件、数据库时，只能使用当前 tenant 授权。
```

写回前至少校验：

```text
sample.tenant_id == context.tenant_id
sample.training_run_id == context.training_run_id
sample.adapter_name == context.adapter_name
sample.reward_type == context.reward_type
sample.loss_type == context.loss_type
```

`sample.policy_version` 不要求等于当前 `context.policy_version`。它表示该 row 由哪个 rollout 权重版本生成，trainer/reward/advantage 只能把它作为元数据和 loss 输入的一部分保留，不能用当前最新版本覆盖它。

### 5.2 RewardWorker 隔离

`RewardWorker` 必须做租户隔离。第一版推荐共享 worker pool，但每个 batch 必须是 context-homogeneous：

```text
同一个 reward batch 内只能有一个:
  tenant_id
  training_run_id
  adapter_name
  reward_type
  policy_version
  train_k
```

推荐流程：

```python
class RewardWorker:
    async def run_once(self):
        batch = await tq.claim_reward_batch()
        context = batch.context

        reward_fn = reward_registry.get(context.reward_type)
        reward_client = reward_client_factory.create(context)

        rewards = await reward_fn.compute(batch.samples, context, reward_client)

        await tq.append_rewards(context, batch.partition_id, rewards)
```

关键要求：

```text
1. 不允许跨 namespace claim rollout-ready 数据。
2. 不允许一个 reward_fn 处理其他 tenant 的数据。
3. 不允许跨 tenant 复用有状态 reward client。
4. 不允许 server import 用户自定义 reward Python 代码。
5. append_rewards 时必须校验 context metadata。
```

如果 reward 逻辑不可信、需要代码执行、浏览器、远程 verifier 或强资源隔离，应将 reward 逻辑放到 sandbox 或 user-owned remote reward service 中。

### 5.3 AdvantageWorker 隔离

`AdvantageWorker` 的隔离要求与 `RewardWorker` 类似。它只能处理同一个 `LoraContext` 下 reward-ready 的 rows，并且 advantage 计算不能跨 adapter 或 policy_version 聚合。

```text
GRPO group:
  必须在同一个 tenant/run/adapter/policy_version 内计算。

advantage / return:
  只能写回原 namespace 下的 train_k。
```

### 5.4 TrainerWorker 隔离

`TrainerWorker` 的隔离边界是 `train_k`：

```text
1. 一个 train_k 只能属于一个 adapter。
2. 一个 optimizer batch 只能来自一个 loss_type / reward_type / algorithm。
3. LoRA 只能在 train_k 边界切换。
4. train_k 完成后才允许触发该 adapter 的权重同步。
```

`TrainerScheduler` 可以在多个 adapter 的 `TRAIN_READY` partition 之间选择，但不能把多个 adapter 混进一个训练 partition。

### 5.5 ToolManager 隔离

`ToolManager` 必须按 `tenant_id / training_run_id / adapter_name / tool_profile` 创建或绑定：

```text
native tool:
  只能使用平台内置、受信任工具。

remote tool:
  只能调用当前 tenant 配置中允许的 endpoint 和 auth_ref。

stateful tool:
  不能跨 tenant 或 training_run 复用状态。
```

第一版不支持 server 直接 import 用户 Env Python 代码。需要自定义逻辑时，用户逻辑必须运行在 sandbox 或 user-owned remote tool / env service 中。

## 6. 异常与恢复

```text
worker 崩溃:
  通过 lease_deadline 发现，未完成 claim 重新进入可 claim 状态或标记 failed。

tenant 取消训练:
  clear_namespace(tenant_id/training_run_id)，停止 rollout/tool/reward/train worker。

tool 超时:
  当前 trajectory 标记 stop_reason=tool_timeout 或 tool_error，是否参与训练由 reward/loss 决定。

quota 超限:
  暂停对应 tenant/run/adapter 的 rollout producer，不删除未训练数据。

partition 过期:
  非终态只进入 recovery，不直接 hard delete。

权重同步失败:
  train_k 保持 TRAIN_DONE 或 SYNC_FAILED，禁止 clear_partition，等待 retry。
```

## 7. 第一版约束

第一版建议明确不支持：

```text
1. 一个 train_k 混多个 adapter。
2. 一个 GRPO group 混多个 policy_version。
3. 一个 optimizer batch 混多个 loss_type / reward_type。
4. ToolManager 或带状态 native/remote tool 跨 tenant 复用。
5. TTL 自动删除非终态训练数据。
6. TQ backend offload 替代 staleness/backpressure。
7. server 直接 import 并执行用户自定义 Env Python 代码。
8. 第一版实现通用 EnvFactory / reset / step / close 协议。
```

后续如果要支持 mixed-version batch，需要额外设计 per-sample importance correction、版本级 loss mask、trainer batch grouping 和权重版本追踪，不建议放入第一版。
