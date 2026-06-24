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
  rollout 并发、环境并发、TQ 容量、trainer slots 需要按租户和训练任务限额。
```

本设计面向“客户端提交多租户训练任务，服务端共享基础模型和异步 RL 基础设施”的场景。用户只需要声明训练身份、环境、reward、loss 和配置；底层 pipeline 负责 TransferQueue 数据面、staleness 控制、权重同步和资源隔离。

## 2. 核心概念

### 2.1 TrainingContext

`TrainingContext` 是一次训练任务在多租户系统中的路由和隔离身份。它不保存训练样本，只用于决定当前请求应该访问哪份数据、哪份 LoRA 权重、哪个环境、哪个 reward/loss 逻辑。

```python
@dataclass
class TrainingContext:
    tenant_id: str
    training_run_id: str
    base_model_id: str
    adapter_name: str
    adapter_revision: str | None
    policy_version: int

    env_type: str
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

base_model_id:
  共享基础模型标识。多个 tenant 可以共享同一个 base model。

adapter_name:
  当前训练任务使用的 LoRA adapter 名称。

adapter_revision:
  可选，用于区分已发布或已恢复的 adapter 版本。

policy_version:
  rollout 使用的策略版本。每次 train_k 完成训练并同步权重后递增。

env_type:
  环境类型，例如 code_sandbox、browser、math、search、game。

reward_type:
  reward 实现类型，例如 unit_test_reward、format_reward、web_task_reward。

loss_type / algorithm:
  训练算法和 loss 选择，例如 grpo、ppo、dapo。
```

### 2.2 namespace

TransferQueue 中的数据必须按 `TrainingContext` 生成 namespace：

```text
{tenant_id}/{training_run_id}/{adapter_name}/train_{k}
```

示例：

```text
tenant_a/code_grpo_001/code_lora/train_7
tenant_b/browser_rl_003/browser_lora/train_2
```

文档中可以继续使用 `train_k` 描述逻辑生命周期，但底层写入、读取、claim、ack、clear 都必须带完整 namespace。

### 2.3 隔离边界

第一版不支持跨 context 混合训练：

```text
同一个 train_k:
  只能属于一个 tenant_id / training_run_id / adapter_name / policy_version。

同一个 GRPO group:
  只能来自同一个 prompt、同一个 adapter、同一个 policy_version。

同一个 optimizer batch:
  只能来自同一个 loss schema 和 reward schema。

同一个 env 实例:
  不能跨 tenant / run 复用。
```

## 3. 总体架构

```text
Client
  |
  | submit TrainingJobConfig
  v
BaseRLPipeline / Controller
  |
  | build TrainingContext
  | init TransferQueueDataPlane
  | create workers
  v
TransferQueueDataPlane  <-------------------------------+
  |                                                      |
  | namespace: tenant/run/adapter/train_k                |
  | metadata / rows / fields / claim / ack / clear       |
  |                                                      |
  +--> AsyncRollouter ----> EnvFactory ----> User Env    |
  |        |                  |              tools/sandbox/browser
  |        +--> vLLMSampler(adapter_name, policy_version)
  |
  +--> RewardWorker ----> RewardRegistry[reward_type]
  |
  +--> AdvantageWorker -> algorithm-specific grouping
  |
  +--> TrainerWorker ---> StreamingDataset/DataLoader
  |        |
  |        +--> LossRegistry[loss_type]
  |
  +--> CheckpointEngineManager
           |
           +--> vLLMSampler.receive_weights(adapter_name, policy_version)
```

关键原则：

```text
TransferQueueDataPlane:
  是唯一 TQ 数据面入口，负责 namespace 拼接、metadata、claim/ack、append、clear 和容量保护。

StalenessManager:
  只按 context 计算 rollout capacity / throttle / sleep hint，不直接写 TQ，不做权重同步。

AsyncRollouter:
  根据 StalenessManager 返回的 capacity、配置最大并发、active_tasks 和 pending_queue 决定是否提交 rollout task。

EnvFactory:
  根据 context.env_type 或 sample.env_type 创建租户隔离的环境实例。

RewardRegistry / LossRegistry:
  根据 context.reward_type / loss_type 路由到对应实现。
```

## 4. 数据面设计

### 4.1 sample metadata

每条写入 TQ 的 trajectory sample 至少包含：

```text
tenant_id
training_run_id
base_model_id
adapter_name
adapter_revision
policy_version
partition_id
group_id
generation_idx
env_type
reward_type
loss_type
algorithm
```

这些字段用于消费侧校验，不能只依赖 partition path。`TransferQueueDataPlane` 在写入和读取时都应校验 metadata 与当前 `TrainingContext` 一致。

### 4.2 partition metadata

`PartitionMetadata` 需要记录生命周期和恢复信息：

```python
@dataclass
class PartitionMetadata:
    context: TrainingContext
    partition_id: str
    policy_version: int
    target_groups: int
    ready_groups: int
    status: str
    created_at: float
    updated_at: float
    owner_worker_id: str | None
    lease_deadline: float | None
```

推荐状态：

```text
OPEN:
  rollout 还在追加样本。

SEALED:
  train_k rollout 完成，不再接收新样本。

REWARD_DONE:
  reward 已完成。

ADVANTAGE_DONE:
  advantage 已完成，可以训练。

TRAINING:
  trainer 正在消费。

TRAIN_DONE:
  trainer 完成 train_k。

CLEARED:
  权重同步完成，partition 已清理。

FAILED / CANCELLED:
  异常或用户取消。
```

### 4.3 TTL 与异常清理

TTL 不能作为正常清理机制。正常清理路径必须是：

```text
train_k rollout done
-> reward/advantage done
-> trainer consumed train_k
-> weight sync done
-> clear_partition(train_k)
```

TTL 只用于异常检测和终态回收：

```text
非终态 partition 超时:
  标记 expired / needs_recovery，不直接删除。

终态 partition 超时:
  CLEARED / CANCELLED / FAILED_CONFIRMED 可以 hard delete。
```

## 5. LoRA 权重路由与切换

### 5.1 rollout 请求绑定 adapter

每个 rollout 请求必须显式绑定 `adapter_name` 和 `policy_version`：

```python
response = await sampler.sample(
    messages=messages,
    adapter_name=context.adapter_name,
    policy_version=context.policy_version,
)
```

一个 trajectory 在正常情况下只能使用一个 `(adapter_name, policy_version)`。

### 5.2 切换时机

允许切换 LoRA 的时机：

```text
新 rollout request 提交前:
  AsyncRollouter 为新 task 读取当前 context.policy_version。

train_k 训练完成并同步后:
  vLLMSampler.receive_weights(adapter_name, new_policy_version)。

partial rollout 被 abort 后恢复:
  已生成 token 可按配置 mask 掉，后续 token 使用新 policy_version。
```

不允许切换的时机：

```text
一个正在生成的 trajectory 中途。

一个未 abort 的多轮 agent trajectory 内。

一个 train_k 内混入多个 adapter。
```

### 5.3 权重同步

每个 `train_k` 训练完成后同步一次当前 adapter 权重：

```text
TrainerWorker train_k done
-> BaseRLPipeline.sync_weights(context, train_k)
-> CheckpointEngineManager.export_adapter(context.adapter_name)
-> vLLMSampler.receive_weights(context.adapter_name, new_policy_version)
-> TransferQueueDataPlane.clear_partition(context, train_k)
```

权重同步不由 `StalenessManager` 控制。`StalenessManager` 只控制 rollout 生产速度。

## 6. 环境隔离与切换

多租户下环境必须通过 `EnvFactory` 路由，不能全局共享一个 env 实例。

```python
class EnvFactory:
    def register(self, env_type: str, env_cls: type[BaseInteractionEnv]) -> None:
        ...

    def create(self, sample: dict, context: TrainingContext) -> BaseInteractionEnv:
        env_type = sample.get("env_type") or context.env_type
        env_config = sample.get("env_config", {})
        return self._registry[env_type](**env_config)
```

环境生命周期建议：

```text
per trajectory:
  隔离最强，适合 code sandbox、browser、外部 simulator。

per prompt group:
  适合同一个 prompt 的 num_generations 共享只读上下文。

per worker:
  只允许无状态 env 或带强隔离 session 的 env。
```

环境权限和资源也要按 tenant/run 控制：

```text
tool whitelist
sandbox image whitelist
network policy
file system policy
timeout
max concurrent envs
max external API QPS
```

## 7. Reward / Advantage / Loss 路由

### 7.1 RewardRegistry

RewardWorker 根据 `context.reward_type` 选择 reward 实现：

```python
reward_fn = reward_registry.get(context.reward_type)
reward = reward_fn(trajectory, context=context)
```

约束：

```text
reward 只能写回当前 context namespace。
reward schema 必须写入 metadata，trainer 侧校验一致。
不同 reward_type 的样本不能混入同一个 train_k。
```

### 7.2 Advantage

GRPO 默认按 prompt group 分组：

```text
group key = tenant_id / training_run_id / adapter_name / policy_version / group_id
```

第一版不支持 mixed-version group。只有同一个 `policy_version` 的 `num_generations` 条 trajectory 都 ready 后，才计算 advantage。

### 7.3 LossRegistry

TrainerWorker 根据 `context.loss_type` / `algorithm` 选择 loss：

```python
loss_fn = loss_registry.get(context.loss_type)
loss = loss_fn(batch, context=context)
```

Trainer batch 必须校验：

```text
tenant_id 一致
training_run_id 一致
adapter_name 一致
policy_version 一致
reward_type 一致
loss_type 一致
algorithm 一致
```

## 8. 容量、Quota 与 Staleness

### 8.1 容量初始化

单训练流容量：

```text
max_live_partitions_per_run = max_staleness + 1
samples_per_partition = target_prompt_groups_per_partition * rollout.num_generations
max_rows_per_run = samples_per_partition * max_live_partitions_per_run
```

多租户容量需要分层：

```text
global_max_rows / global_max_bytes:
  整个 TQ backend 的总容量。

tenant_max_rows / tenant_max_bytes:
  单租户最大占用。

run_max_rows / run_max_bytes:
  单 training_run 最大占用。

adapter_max_live_partitions:
  单 adapter 最多存活 train_k 数量，通常等于 max_staleness + 1。
```

### 8.2 StalenessManager 决策

`StalenessManager` 按 context 计算 capacity：

```python
capacity = staleness_manager.get_rollout_capacity(
    metadata=tq_data_plane.get_metadata(context),
    context=context,
)
```

决策规则：

```text
adapter live partitions > max_staleness + 1:
  暂停该 adapter rollout。

tenant quota 超限:
  暂停该 tenant 的新 rollout。

global quota 超限:
  所有 rollout throttle/sleep，只允许消费侧继续处理。

接近 throttle_watermark:
  AsyncRollouter 降低提交速率或减小 transfer batch。
```

`StalenessManager` 不删除数据。容量压力的优先动作是限制 producer，而不是丢弃未训练样本。

## 9. 客户端接入方式

### 9.1 用户代码

用户通常只需要提供环境、reward 和配置：

```python
from twinkle_agentic.pipeline import AsyncAgenticRLPipeline
from twinkle_agentic.env import EnvFactory
from my_project.envs import CodeSandboxEnv
from my_project.rewards import UnitTestReward


env_factory = EnvFactory()
env_factory.register("code_sandbox", CodeSandboxEnv)

pipeline = AsyncAgenticRLPipeline.from_yaml(
    "configs/tenant_a_code_async_rl.yaml",
    env_factory=env_factory,
    reward_registry={
        "unit_test_reward": UnitTestReward(),
    },
)

pipeline.run()
```

如果用户要完全自定义 rollout 流程，可以继承 rollout：

```python
class CodeAgentRollout(MultiTurnRollout):
    async def run_group(self, sample, *, context, sampler, env_factory):
        env = env_factory.create(sample, context)
        ...
        return trajectories
```

如果用户要自定义整体编排流程，才继承 pipeline：

```python
class CustomRLPipeline(AsyncAgenticRLPipeline):
    def build_workers(self):
        ...

    async def on_train_partition_done(self, context, partition_id):
        ...
```

### 9.2 YAML 示例

```yaml
tenant:
  tenant_id: tenant_a
  training_run_id: code_agent_grpo_001

model:
  base_model_id: Qwen/Qwen3.5-4B

adapter:
  multi_lora: true
  adapter_name: tenant_a_code_lora
  adapter_revision: null

rollout:
  type: multi_turn
  num_generations: 8
  max_turns: 6
  max_concurrent_groups: 64
  transfer_batch_groups: 4

env:
  type: code_sandbox
  config:
    image: python:3.11
    timeout: 10
    network: disabled
    max_files: 64

reward:
  type: unit_test_reward

trainer:
  algorithm: grpo
  loss_type: grpo
  global_batch_size: 128
  mini_batch_size: 32

async_rl:
  max_staleness: 2
  partial_rollout: false

transfer_queue:
  namespace: "{tenant_id}/{training_run_id}/{adapter_name}"
  storage_backend: SimpleStorage
  capacity:
    global_max_bytes: 200GiB
    tenant_max_bytes: 40GiB
    run_max_bytes: 20GiB
    safety_factor: 1.2
```

## 10. 正常流程

```text
1. Client 提交 TrainingJobConfig。
2. BaseRLPipeline 构造 TrainingContext。
3. TransferQueueDataPlane 初始化 namespace 和容量限制。
4. StalenessManager 根据 context metadata 计算 rollout capacity。
5. AsyncRollouter 根据 capacity 和 max_concurrent_groups 提交 rollout task。
6. EnvFactory 根据 env_type 创建租户隔离 env。
7. vLLMSampler 使用 adapter_name / policy_version 生成。
8. AsyncRollouter 将完成的 trajectory group 写入 context/train_k。
9. RewardWorker 根据 reward_type 计算 reward 并写回。
10. AdvantageWorker 按 context + group_id 计算 advantage。
11. TrainerWorker 通过 StreamingDataLoader 读取 train_k。
12. TrainerWorker 使用 loss_type 训练当前 adapter。
13. train_k 完成后同步 adapter 权重。
14. vLLMSampler 接收新 adapter 权重。
15. TransferQueueDataPlane 清理 context/train_k。
```

## 11. 异常与恢复

```text
worker 崩溃:
  通过 lease_deadline 发现，未完成 claim 重新进入可 claim 状态或标记 failed。

tenant 取消训练:
  clear_namespace(tenant_id/training_run_id)，停止 rollout/env/reward/train worker。

env 超时:
  当前 trajectory 标记 stop_reason=env_timeout，是否参与训练由 reward/loss 决定。

quota 超限:
  暂停对应 tenant/run/adapter 的 rollout producer，不删除未训练数据。

partition 过期:
  非终态只进入 recovery，不直接 hard delete。

权重同步失败:
  train_k 保持 TRAIN_DONE 或 SYNC_FAILED，禁止 clear_partition，等待 retry。
```

## 12. 第一版约束

第一版建议明确不支持：

```text
1. 一个 train_k 混多个 adapter。
2. 一个 GRPO group 混多个 policy_version。
3. 一个 optimizer batch 混多个 loss_type / reward_type。
4. env 实例跨 tenant 复用。
5. TTL 自动删除非终态训练数据。
6. TQ backend offload 替代 staleness/backpressure。
```

后续如果要支持 mixed-version batch，需要额外设计 per-sample importance correction、版本级 loss mask、trainer batch grouping 和权重版本追踪，不建议放入第一版。
