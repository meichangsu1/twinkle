# TransferQueueDataPlane Spec

## 1. 定位与命名映射

### 1.1 定位

`TransferQueueDataPlane` 是异步 RL pipeline 中 **唯一的数据面边界**。所有对 TransferQueue (TQ) 后端的读写操作都必须经过这个类，上层组件（`AsyncRollouter`、`RewardWorker`、`AdvantageWorker`、`TrainerWorker`、`BaseRLPipeline`）不直接调用 TQ API。

```text
位置: src/twinkle_agentic/async_rl/data_plane.py
依赖: transfer_queue (pip install TransferQueue)
导出: twinkle_agentic.async_rl.TransferQueueDataPlane
```

### 1.2 概念层 → 代码层 → 物理层 命名映射

本系统涉及三层命名：

```text
概念层:  详细设计文档中使用的业务概念名称（面向开发者沟通）
代码层:  src/twinkle_agentic/async_rl/ 中的类名、变量名、方法名（面向实现）
物理层:  TransferQueue 后端中实际存储的 key、partition_id、field name、tag（面向存储）
```

#### 1.2.1 核心概念映射

| 概念层（设计文档） | 代码层（实现） | 物理层（TQ 存储） | 说明 |
|---|---|---|---|
| `TrainingContext` | `TrainingContext` | 无独立存储，字段分散在 TQ tag 中 | 训练作用域身份。包含 `tenant_id`、`training_run_id`、`base_model_id`、`adapter_name`、`adapter_revision`、`policy_version`、`env_type`、`tool_profile`、`reward_type`、`loss_type`、`algorithm` 字段 |
| `scope` | `context` | - | 变量名。设计文档用 `scope`，代码用 `context` |
| `scope.key` | `context.key` | TQ partition_id 的前缀部分 | 值为 `{tenant_id}/{training_run_id}/{adapter_name}`，是 scope 的唯一字符串标识 |
| `train_k`（逻辑名） | `partition_id`（物理名） | TQ `partition_id` 参数 | 逻辑名 `train_k` 只出现在文档中；代码和 TQ 中统一使用完整物理名 `{tenant_id}/{training_run_id}/{adapter_name}/train_{k}` |
| `rollout_id` | `_next_train_id[context.key]` | partition_id 中的 `train_{k}` 后缀 | 自增计数器，per-context，从 0 开始 |
| `QueueMetadata` | `list[PartitionMetadata]` | 通过 `tq.kv_list()` 重建 | 设计文档定义了 `QueueMetadata` 类，代码中 `get_metadata()` 直接返回 `list[PartitionMetadata]` |
| `Trajectory` | `SampleRecord` (`Dict[str, Any]`) | TQ 中的一个 key 对应的 fields + tag | 一条完整的训练样本 |
| `prompt group` | `ready_groups` 计数单位 | 同一个 `group_id` 的多个 sample | 一个 prompt 对应 `num_generations` 条 trajectory |
| `transfer batch` | `put_rollout_batch()` 的 `trajectories` 参数 | 一次 `_kv_batch_put` 调用 | AsyncRollouter 聚合后一次写入 TQ 的数据批次 |

#### 1.2.2 Partition 状态映射

| 概念层（设计文档） | 代码层（实现） | 物理层（TQ tag） |
|---|---|---|
| rollout done | `PartitionStatus.ROLLOUT_DONE` | `tag['status'] = 'ROLLOUT_DONE'` |
| reward done | `PartitionStatus.REWARD_DONE` | `tag['status'] = 'REWARD_DONE'` |
| advantage done / train ready | `PartitionStatus.TRAIN_READY` | `tag['status'] = 'TRAIN_READY'` |
| training | `PartitionStatus.TRAINING` | `tag['status'] = 'TRAINING'` |
| train done | `PartitionStatus.TRAIN_DONE` | `tag['status'] = 'TRAIN_DONE'` |
| cleared | `PartitionStatus.CLEARED` | key 已从 TQ 中删除 |
| failed | `PartitionStatus.FAILED` | `tag['status'] = 'FAILED'` |
| cancelled | `PartitionStatus.CANCELLED` | `tag['status'] = 'CANCELLED'` |
| is_rollout_done (bool) | `meta.status == PartitionStatus.ROLLOUT_DONE` | - |
| is_train_done (bool) | `meta.status == PartitionStatus.TRAIN_DONE` | - |

#### 1.2.3 PartitionMetadata 字段映射

| 概念层（设计文档） | 代码层（实现） | 物理层（TQ tag） | 说明 |
|---|---|---|---|
| `scope` | `context: TrainingContext` | 展开为 `tenant_id`、`training_run_id` 等独立 tag 字段 | 设计文档用 `scope`，代码用 `context` |
| `partition_id` | `partition_id: str` | `tag['partition_id']` | 完整物理名 |
| `rollout_id` | 从 `partition_id` 解析：`meta.logical_train_id` | partition_id 中的 `train_{k}` 部分 | 代码中没有独立字段，通过 `rsplit('/', 1)[-1]` 提取 |
| `policy_version` | `policy_version: int` | `tag['policy_version']` | rollout 使用的模型版本 |
| `target_groups` | `target_groups: int` | `tag['target_groups']` | 目标 prompt group 数 |
| `rollout_done_groups` | 无独立字段 | - | 代码用 `ready_groups` 统一计数 |
| `reward_done_groups` | 无独立字段 | - | 代码用状态机替代（`REWARD_DONE` 表示全部完成） |
| `advantage_done_groups` | 无独立字段 | - | 代码用状态机替代（`TRAIN_READY` 表示全部完成） |
| `trained_groups` | 无独立字段 | - | 代码用状态机替代（`TRAIN_DONE` 表示全部完成） |
| `dropped_groups` | 无 | - | 第一版未实现 |
| - | `ready_groups: int` | `tag['ready_groups']` | 已写入的 prompt group 数（代码独有） |
| - | `status: PartitionStatus` | `tag['status']` | 当前状态（代码独有，设计文档用多个 bool 和 counter 表达） |
| - | `num_rows: int` | `tag['num_rows']` | partition 内的 sample 总数（代码独有） |
| - | `created_at: float` | - | 创建时间戳（代码独有，仅内存） |
| - | `updated_at: float` | - | 最后更新时间戳（代码独有，仅内存） |
| - | `owner_worker_id: str` | - | 当前租约持有者（代码独有，仅内存） |
| - | `lease_deadline: float` | - | 租约过期时间（代码独有，仅内存） |

**设计差异说明**：设计文档用多个独立计数器（`rollout_done_groups`、`reward_done_groups` 等）追踪每个阶段的完成进度；代码实现用单一状态机（`PartitionStatus`）+ 单一计数器（`ready_groups`）简化。第一版假设一个 partition 的所有 group 在同一个阶段一起完成，不需要 per-stage 计数。

#### 1.2.4 Sample 字段映射（TQ fields vs tag）

每个 sample 在 TQ 中分为 **fields**（数据）和 **tag**（元数据）两层存储：

**fields（数据层，大对象，按 stage 追加）：**

| 阶段 | 字段名 | 类型 | 写入者 | 说明 |
|------|--------|------|--------|------|
| rollout | `messages` | `list[dict]` | AsyncRollouter | 完整对话历史（system + user + assistant + tool calls） |
| rollout | `group_id` | `str` | AsyncRollouter | prompt group 标识，同一个 prompt 的 `num_generations` 条 trajectory 共享 |
| rollout | `generation_idx` | `int` | AsyncRollouter | 在 group 内的序号（0 ~ num_generations-1） |
| rollout | `old_logps` | `list[float]` | AsyncRollouter | rollout 时的 log probabilities |
| reward | `rewards` | `float` | RewardWorker | 计算后的 reward 值 |
| advantage | `advantages` | `float` | AdvantageWorker | 计算后的 advantage 值 |
| advantage | `returns` | `float` | AdvantageWorker | 计算后的 return 值 |

**tag（元数据层，小对象，用于过滤和状态追踪）：**

| 字段名 | 类型 | 来源 | 说明 |
|--------|------|------|------|
| `tenant_id` | `str` | `TrainingContext.tenant_id` | 租户标识 |
| `training_run_id` | `str` | `TrainingContext.training_run_id` | 训练任务标识 |
| `base_model_id` | `str` | `TrainingContext.base_model_id` | 基础模型标识 |
| `adapter_name` | `str` | `TrainingContext.adapter_name` | LoRA adapter 名称 |
| `adapter_revision` | `str \| None` | `TrainingContext.adapter_revision` | adapter 权重版本 |
| `policy_version` | `int` | `TrainingContext.policy_version` | rollout 使用的模型版本 |
| `env_type` | `str` | `TrainingContext.env_type` | 环境类型 |
| `tool_profile` | `str` | `TrainingContext.tool_profile` | 工具配置 |
| `reward_type` | `str` | `TrainingContext.reward_type` | reward 实现类型 |
| `loss_type` | `str` | `TrainingContext.loss_type` | loss 类型 |
| `algorithm` | `str` | `TrainingContext.algorithm` | 训练算法 |
| `partition_id` | `str` | `PartitionMetadata.partition_id` | 所属 partition |
| `status` | `str` | `PartitionMetadata.status.value` | partition 当前状态 |
| `target_groups` | `int` | `PartitionMetadata.target_groups` | 目标 group 数 |
| `ready_groups` | `int` | `PartitionMetadata.ready_groups` | 已就绪 group 数 |
| `num_rows` | `int` | `PartitionMetadata.num_rows` | partition 内 sample 总数 |

#### 1.2.5 task_name 映射

`task_name` 是 TQ 内部区分数据处理阶段的逻辑名字（详细设计 3.3）。代码中的使用方式：

| task_name | 写入阶段 | 消费阶段 | 代码中的使用位置 |
|---|---|---|---|
| `rollout` | `put_rollout_batch()` | `claim_reward_batch()` 读取 | 隐式：rollout 写入的数据是 reward 的输入 |
| `reward` | `append_rewards()` | `claim_advantage_batch()` 读取 | 隐式：reward 写入的数据是 advantage 的输入 |
| `advantage` | `append_advantages()` | `build_streaming_dataloader()` 读取 | 隐式：advantage 写入的数据是 trainer 的输入 |
| `train` | - | `ack_rows(task_name='train')` | 显式：`ack_rows()` 和 `build_streaming_dataloader(task_name=)` 的参数 |

**当前实现说明**：第一版通过 `PartitionStatus` 状态机隐式控制阶段流转（`ROLLOUT_DONE` → `REWARD_DONE` → `TRAIN_READY`），没有在每个操作中显式传递 `task_name`。只有 `ack_rows()` 和 `build_streaming_dataloader()` 显式使用 `task_name` 参数区分消费任务。后续集成 TQ 原生 `StreamingDataset` 时，需要在所有操作中显式传递 `task_name`。

#### 1.2.6 容量与调度概念映射

| 概念层（设计文档） | 代码层（实现） | 说明 |
|---|---|---|
| `max_live_partitions` | `max_staleness + 1` | TQ 中最多同时存在的未清理 partition 数 |
| `available_partition_slots` | `RolloutCapacity.available_groups` | 设计文档按 partition 计数，代码按 group 计数 |
| `available_group_slots` | `RolloutCapacity.available_groups` | 代码合并为一个字段 |
| `should_throttle` | `RolloutCapacity.action == 'submit' and reason == 'near_staleness_limit'` | 代码用 action + reason 组合表达 |
| `should_sleep` | `RolloutCapacity.action == 'sleep'` | 代码用 action 字段表达 |
| `global_max_rows` | `TransferQueueRuntimeConfig.max_rows` | 全局 row 上限 |
| `scope_max_rows` | `TransferQueueRuntimeConfig.max_rows_per_context` | 单 scope row 上限 |
| `scope_max_live_partitions` | `TransferQueueRuntimeConfig.max_live_partitions_per_context` | 单 scope live partition 上限 |

#### 1.2.7 组件映射

| 概念层（设计文档） | 代码层（实现） | 文件位置 |
|---|---|---|
| `BaseRLPipeline` | `BaseRLPipeline` | `async_rl/pipeline.py` |
| `TransferQueueDataPlane` | `TransferQueueDataPlane` | `async_rl/data_plane.py` |
| `StalenessManager` | `StalenessManager` | `async_rl/staleness.py` |
| `AsyncRollouter` | `AsyncRollouter` | `async_rl/workers.py` |
| `RewardWorker` | `RewardWorker` | `async_rl/workers.py` |
| `AdvantageWorker` | `AdvantageWorker` | `async_rl/workers.py` |
| `TrainerWorker` | `TrainerWorker` | `async_rl/workers.py` |
| `TrainerScheduler` | `TrainerScheduler` | `async_rl/workers.py` |
| `AdapterRegistry` | `AdapterRegistry` | `async_rl/registry.py` |
| `StreamingDataset` | `build_streaming_dataloader()` 返回 `list` | `async_rl/data_plane.py` |
| `StreamingDataLoader` | 未实现（第一版用 list 替代） | - |
| `CheckpointEngineManager` | 外部组件，通过 `receive_weights_fn` 回调 | `twinkle.checkpoint_engine` |
| `MultiTurnRollout` | `MultiTurnRollout` | `twinkle_agentic/rollout/multi_turn.py` |
| `vLLMSampler` | `vLLMSampler` | `twinkle.sampler` |
| `ToolManager` | `ToolManager` | `twinkle_agentic/tools/tool_manager.py` |
| `ToolManagerFactory` | `ToolManagerFactory` | `async_rl/workers.py` |

#### 1.2.8 方法映射

| 概念层（设计文档） | 代码层（实现） | 说明 |
|---|---|---|
| `init_transfer_queue()` | `init_namespace(context)` | 设计文档叫 init_transfer_queue，代码叫 init_namespace |
| `put_rollout_batch(scope, partition_id, samples)` | `put_rollout_batch(context, partition_id, trajectories)` | 参数名 scope→context, samples→trajectories |
| `claim_reward_batch(scope, partition_id, batch_size)` | `claim_reward_batch(context, batch_size)` | 代码不传 partition_id，自动找第一个 ROLLOUT_DONE |
| `append_rewards(metadata, rewards)` | `append_rewards(context, partition_id, rewards)` | 代码传 context + partition_id 而非 metadata 对象 |
| `claim_reward_ready_groups(scope, partition_id, num_generations, max_groups)` | `claim_advantage_batch(context, batch_size)` | 设计文档按 group claim，代码按 sample batch claim |
| `append_advantages(group_metadata, advantages)` | `append_advantages(context, partition_id, advantages, returns)` | 代码多了 returns 参数 |
| `build_streaming_dataset(scope, partition_id, required_fields, batch_size, sampler)` | `build_streaming_dataloader(context, partition_id, task_name=None)` | 代码简化为返回 list，不集成 TQ StreamingDataset |
| `read_train_batch(metadata)` | `build_streaming_dataloader()` + `ack_rows()` | 代码拆分为读取和确认两步 |
| `ack_train_batch(metadata)` | `ack_rows(context, partition_id, sample_ids, task_name)` | 代码需要显式传 sample_ids |
| `mark_trained(metadata)` | `mark_trained(context, partition_id)` | 代码传 context + partition_id |
| `clear_partition(scope, partition_id)` | `clear_partition(context, partition_id)` | 参数名 scope→context |
| `get_metadata()` | `get_metadata(context=None)` | 代码可按 context 过滤 |
| `sync_weights(scope, adapter_name, train_k)` | 通过 `receive_weights_fn` 回调 | 不在 DataPlane 中，由 BaseRLPipeline 调用 |

#### 1.2.9 TQ 物理存储结构

```text
TransferQueue 物理存储:

  partition_id = "tenant_a/run_001/lora/train_0"
  │
  ├── key = "tenant_a/run_001/lora/train_0/sample_0"
  │   ├── fields: {messages: [...], group_id: "g0", generation_idx: 0, old_logps: [...], rewards: 0.8, advantages: 0.3, returns: 0.8}
  │   └── tag:    {tenant_id: "tenant_a", training_run_id: "run_001", adapter_name: "lora", policy_version: 0, status: "ROLLOUT_DONE", target_groups: 2, ready_groups: 2, num_rows: 16, ...}
  │
  ├── key = "tenant_a/run_001/lora/train_0/sample_1"
  │   ├── fields: {messages: [...], group_id: "g0", generation_idx: 1, old_logps: [...], rewards: 1.0, advantages: 0.5, returns: 1.0}
  │   └── tag:    {tenant_id: "tenant_a", ..., status: "ROLLOUT_DONE", ...}
  │
  └── key = "tenant_a/run_001/lora/train_0/sample_15"
      ├── fields: {messages: [...], group_id: "g1", generation_idx: 7, old_logps: [...], rewards: 0.5, advantages: -0.2, returns: 0.5}
      └── tag:    {tenant_id: "tenant_a", ..., status: "ROLLOUT_DONE", ...}
```

**关键规则：**

1. **key 格式**：`{partition_id}/sample_{idx}`，其中 `idx` 是 partition 内的全局递增序号。如果 sample 自带 `sample_id`，则使用 `sample_id` 作为 key。
2. **fields 追加**：fields 按 stage 追加写入。rollout 阶段写入 `messages`、`group_id` 等；reward 阶段追加 `rewards`；advantage 阶段追加 `advantages`、`returns`。TQ 的 `kv_put` 对已有 key 是 merge 语义（新 fields 合并到旧 fields）。
3. **tag 同步**：每次 partition 状态变化时，`_sync_partition_status()` 会把最新的 `PartitionMetadata.tag()` 写回该 partition 下所有 key 的 tag。
4. **partition 隔离**：不同 `context.key` 的 partition_id 前缀不同，物理上不会冲突。

## 2. 依赖: TransferQueue KV API 合约

`TransferQueueDataPlane` 只使用 TQ 的 KV 层 API，不使用底层 Client API 或 StreamingDataLoader。以下是本类依赖的精确 TQ 接口：

### 2.1 初始化

```python
import transfer_queue as tq

# config 由 OmegaConf 构建
tq.init(config)   # config.init == True 时调用
```

config 结构：
```python
{
    'controller': { ... },       # 透传自 TransferQueueRuntimeConfig.controller
    'backend': {
        'storage_backend': 'SimpleStorage',
        'SimpleStorage': {
            'num_data_storage_units': 4,
            'total_storage_size': ...,  # 可选
        },
    },
}
```

### 2.2 KV 读写

```python
# 写入/更新单个 key
tq.kv_put(key: str, partition_id: str, fields: dict | None, tag: dict | None)

# 批量读取
tq.kv_batch_get(keys: list[str], partition_id: str, select_fields=None) -> dict[str, list] | TensorDict

# 列出 key 和 tag
tq.kv_list(partition_id: str | None) -> dict[str, dict[str, dict]]
#   partition_id 不为 None: {partition_id: {key: tag_dict}}
#   partition_id 为 None:    {partition_id: {key: tag_dict}, ...}

# 清除
tq.kv_clear(keys: list[str], partition_id: str)
```

### 2.3 FakeTransferQueueClient 测试合约

测试使用 `tests/twinkle_agentic/async_rl/fakes.py` 中的 `FakeTransferQueueClient`，它实现了上述四个方法的内存版本。构造时传入 `tq_client=FakeTransferQueueClient()` 即可绕过真实 TQ 依赖。

## 3. 配置

### 3.1 TransferQueueRuntimeConfig 完整定义

```python
@dataclass
class TransferQueueRuntimeConfig:
    """TransferQueue 初始化与运行时容量控制配置。

    容量规划原则（详见详细设计 3.1）：
      TQ 容量按允许同时存活的 train_k partition 数规划，不按单个 sample 动态扩容。
      max_live_partitions = max_staleness + 1
      samples_per_partition = target_groups * num_generations
      max_rows = samples_per_partition * max_live_partitions
      max_tq_bytes = estimate_bytes_per_sample * max_rows * safety_factor
    """

    # ── TQ 后端初始化 ──────────────────────────────────────────────
    total_storage_size: Optional[int] = None
    num_data_storage_units: int = 4
    storage_backend: str = 'SimpleStorage'
    controller: Dict[str, Any] = field(default_factory=dict)
    backend: Dict[str, Any] = field(default_factory=dict)
    init: bool = True

    # ── 容量规划输入参数 ──────────────────────────────────────────
    target_groups: int = 128
    num_generations: int = 8
    max_staleness: int = 1
    estimate_bytes_per_sample: Optional[int] = None
    safety_factor: float = 1.2

    # ── 容量保护阈值 ──────────────────────────────────────────────
    max_rows: Optional[int] = None
    max_rows_per_context: Optional[int] = None
    max_tq_bytes: Optional[int] = None
    max_live_partitions_per_context: Optional[int] = None

    # ── 运行时控制 ────────────────────────────────────────────────
    lease_timeout: float = 300.0
```

### 3.2 字段详细说明

#### 3.2.1 TQ 后端初始化

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `total_storage_size` | `int \| None` | `None` | TQ SimpleStorage 总容量（bytes）。为 `None` 时由容量公式自动计算。对应 TQ config 中 `backend.SimpleStorage.total_storage_size`。 |
| `num_data_storage_units` | `int` | `4` | TQ storage unit 数量。每个 unit 独立管理一部分数据。建议与 DP（数据并行）数对齐。对应 TQ config 中 `backend.SimpleStorage.num_data_storage_units`。 |
| `storage_backend` | `str` | `'SimpleStorage'` | 存储后端类型。可选值：`'SimpleStorage'`（内存）、`'Yuanrong'`（分布式）、`'MooncakeStore'`（RDMA）。对应 TQ config 中 `backend.storage_backend`。 |
| `controller` | `Dict[str, Any]` | `{}` | TQ controller 配置透传。可配置 `sampler`（如 `RankAwareSampler`）、`polling_mode` 等。对应 TQ config 中 `controller` 节点。 |
| `backend` | `Dict[str, Any]` | `{}` | TQ backend 配置透传。用于传递非 SimpleStorage 后端的专有配置（如 Yuanrong 的 `worker_port`、MooncakeStore 的 `metadata_server`）。 |
| `init` | `bool` | `True` | 是否在 `TransferQueueDataPlane` 构造时调用 `tq.init(config)`。测试时传 `False` 可跳过初始化。 |

#### 3.2.2 容量规划输入参数

这些参数用于自动计算 TQ 容量。设计文档公式（详细设计 3.1）：

```text
samples_per_partition = target_groups * num_generations
max_live_partitions = max_staleness + 1
max_rows = samples_per_partition * max_live_partitions
max_tq_bytes = estimate_bytes_per_sample * max_rows * safety_factor
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `target_groups` | `int` | `128` | 一个 `train_k` 目标收集多少个 prompt group。对应 YAML 中 `partition.target_groups`。每个 prompt group 包含 `num_generations` 条 trajectory。 |
| `num_generations` | `int` | `8` | 每个 prompt 生成多少条 trajectory sample。GRPO advantage 默认以 prompt group 为单位计算。对应 YAML 中 `rollout.num_generations`。 |
| `max_staleness` | `int` | `1` | rollout 最多领先 trainer 的未完成 `train_k` 数。`max_staleness=0` 表示严格同步（最多 1 个 live partition）；`max_staleness=1` 允许 rollout 和 trainer 重叠（最多 2 个 live partition）。对应 YAML 中 `staleness.max_staleness`。 |
| `estimate_bytes_per_sample` | `int \| None` | `None` | 单个 sample 的预估字节数。为 `None` 时不计算 `max_tq_bytes`。典型值：短文本 ~2KB，长文本（4K tokens）~32KB，多轮 agent trajectory ~64KB。 |
| `safety_factor` | `float` | `1.2` | 容量安全系数。`1.2` 表示预留 20% 余量。用于 `max_tq_bytes` 计算。 |

**容量计算示例**：

```text
场景: target_groups=128, num_generations=8, max_staleness=1

samples_per_partition = 128 * 8 = 1024
max_live_partitions = 1 + 1 = 2
max_rows = 1024 * 2 = 2048

如果 estimate_bytes_per_sample = 32768 (32KB):
  max_tq_bytes = 32768 * 2048 * 1.2 = 80,530,636 bytes ≈ 76.8 MB
```

#### 3.2.3 容量保护阈值

这些是 DataPlane 层面的运行时保护，防止单个 tenant/adapter 占满整个 TQ。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_rows` | `int \| None` | `None` | **全局** row 上限。整个 TQ backend 的总 row 数不超过此值。为 `None` 时由容量公式自动计算（`samples_per_partition * max_live_partitions`）。`check_capacity()` 检查此值。 |
| `max_rows_per_context` | `int \| None` | `None` | **单 context** row 上限。单个 `(tenant_id, training_run_id, adapter_name)` 的 row 数不超过此值。用于多租户场景防止一个 tenant 占满 TQ。为 `None` 时等于 `max_rows`（不做 per-context 限制）。 |
| `max_tq_bytes` | `int \| None` | `None` | TQ 总 bytes 保护阈值。为 `None` 时由容量公式自动计算（`estimate_bytes_per_sample * max_rows * safety_factor`）。如果 TQ backend 支持按 bytes 初始化，优先使用此值；如果只支持 row 数，使用 `max_rows`。 |
| `max_live_partitions_per_context` | `int \| None` | `None` | **单 context** 的 live partition 上限。为 `None` 时等于 `max_staleness + 1`。用于多租户场景限制单个 adapter 的并发 partition 数。 |

**多租户容量模型**（详细设计 3.1）：

```text
全局上限:
  max_rows:                整个 TQ backend 的总 row 上限
  max_tq_bytes:            整个 TQ backend 的总 bytes 上限

单 context (tenant_id, training_run_id, adapter_name) 上限:
  max_rows_per_context:    单个 scope 的 row 上限
  max_live_partitions_per_context:  单个 scope 的 live partition 上限
                                  默认 = max_staleness + 1
```

`StalenessManager` 使用 scope 内的 live partitions 计算 capacity；`TransferQueueDataPlane.check_capacity()` 负责执行全局和 scope 级容量保护。

#### 3.2.4 运行时控制

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `lease_timeout` | `float` | `300.0` | partition 租约超时时间（秒）。worker claim partition 后获得排他租约，超时后自动释放，允许其他 worker 重新 claim。用于 worker 崩溃恢复。 |

### 3.3 配置优先级与覆盖规则

```text
显式值 > 自动计算 > 默认值

示例:
  如果用户显式设置 max_rows=5000，则不使用 target_groups * num_generations * (max_staleness + 1) 的计算结果。
  如果用户未设置 max_rows 但设置了 target_groups=64, num_generations=4, max_staleness=0，
  则 max_rows = 64 * 4 * (0 + 1) = 256。
```

### 3.4 task_name 配置

`task_name` 是 TQ 内部区分数据处理阶段的逻辑名字（详细设计 3.3）。第一版固定使用四类：

| task_name | 写入/消费阶段 | 主要字段 |
|---|---|---|
| `rollout` | `AsyncRollouter` 写入初始 trajectory rows | `messages`、`group_id`、`generation_idx`、`policy_version`、`old_logps` |
| `reward` | `RewardWorker` 消费 rollout-ready rows 并写回 reward | `rewards`、`raw_rewards`、`reward_breakdown` |
| `advantage` | `AdvantageWorker` 消费 reward-ready groups 并写回 advantage | `advantages`、`returns` |
| `train` | `StreamingDataset / StreamingDataLoader` 消费 train-ready rows | training read state、ack state |

典型数据流：

```text
put(partition_id=train_k, task_name=rollout, fields=rollout_fields)
claim(partition_id=train_k, task_name=reward, required_fields=rollout_fields)
append(partition_id=train_k, task_name=reward, fields=reward_fields)
claim(partition_id=train_k, task_name=advantage, required_fields=reward_fields)
append(partition_id=train_k, task_name=advantage, fields=advantage_fields)
read(partition_id=train_k, task_name=train, required_fields=train_required_fields)
ack(partition_id=train_k, task_name=train)
clear_partition(train_k)
```

所有 `claim / read / ack / append / clear` 必须携带 `TrainingContext`（`tenant_id / training_run_id / adapter_name`）过滤条件。禁止跨 scope claim 数据。

### 3.5 storage_backend 配置

#### SimpleStorage（默认，内存）

```python
TransferQueueRuntimeConfig(
    storage_backend='SimpleStorage',
    num_data_storage_units=4,
    total_storage_size=100000,  # bytes，可选
)
```

#### Yuanrong（分布式）

```python
TransferQueueRuntimeConfig(
    storage_backend='Yuanrong',
    backend={
        'Yuanrong': {
            'auto_init': True,
            'worker_port': 31501,
            'metastore_port': 2379,
            'enable_yr_npu_transport': False,
            'enable_rdma': False,
            'worker_args': '--shared_memory_size_mb 8192',
        },
    },
)
```

#### MooncakeStore（RDMA）

```python
TransferQueueRuntimeConfig(
    storage_backend='MooncakeStore',
    backend={
        'MooncakeStore': {
            'auto_init': True,
            'metadata_server': 'localhost:50050',
            'master_server_address': 'localhost:50051',
            'protocol': 'tcp',
        },
    },
)
```

### 3.6 典型配置示例

#### 最小配置（单租户，同步模式）

```python
config = TransferQueueRuntimeConfig(
    target_groups=1,
    num_generations=4,
    max_staleness=0,
)
# 自动计算: max_rows = 1 * 4 * 1 = 4, max_live_partitions = 1
```

#### 标准 GRPO 配置（单租户，异步模式）

```python
config = TransferQueueRuntimeConfig(
    target_groups=128,
    num_generations=8,
    max_staleness=1,
    estimate_bytes_per_sample=32768,
    safety_factor=1.2,
    num_data_storage_units=4,
    lease_timeout=300.0,
)
# 自动计算:
#   samples_per_partition = 128 * 8 = 1024
#   max_live_partitions = 1 + 1 = 2
#   max_rows = 1024 * 2 = 2048
#   max_tq_bytes = 32768 * 2048 * 1.2 ≈ 76.8 MB
```

#### 多租户配置（多 adapter 共享 TQ）

```python
config = TransferQueueRuntimeConfig(
    target_groups=64,
    num_generations=8,
    max_staleness=1,
    max_rows=8192,                    # 全局上限
    max_rows_per_context=2048,        # 单 adapter 上限
    max_live_partitions_per_context=2, # 单 adapter 最多 2 个 live partition
    num_data_storage_units=8,         # 与 DP 数对齐
    lease_timeout=600.0,              # 长超时，适应慢 worker
)
```

#### 测试配置（mock，不初始化 TQ）

```python
config = TransferQueueRuntimeConfig(init=False)
data_plane = TransferQueueDataPlane(tq_client=FakeTransferQueueClient(), tq_config=config)
```

## 4. 构造

```python
class TransferQueueDataPlane:
    def __init__(self, tq_client=None, tq_config=None):
```

| 参数 | 说明 |
|---|---|
| `tq_client` | 可选。传入符合 KV API 合约的对象。仅用于测试。 |
| `tq_config` | 可选。`TransferQueueRuntimeConfig`，默认值即可。 |

构造逻辑：
1. 保存 `tq_config`。
2. 如果 `tq_client` 为 None，`import transfer_queue as tq` 并按 `config.init` 决定是否调用 `tq.init()`。
3. 初始化内部状态：`_meta: Dict[str, PartitionMetadata]`、`_next_train_id: Dict[str, int]`、`_lock: RLock`。

**失败条件**：`tq_client=None` 且 `transfer_queue` 未安装时，抛出 `RuntimeError`。

## 5. 内部状态

```python
self._meta: Dict[str, PartitionMetadata]     # partition_id -> metadata (内存缓存)
self._next_train_id: Dict[str, int]          # context.key -> 自增 train_id
self._lock: threading.RLock                  # 线程安全
```

`_meta` 是 partition 状态的 source of truth。TQ 后端只存储 KV 数据和 tag，partition 状态机在内存中维护。`_load_partition_meta()` 在 `list_partitions()` 时从 TQ tag 重建 `_meta`。

## 6. Namespace 与 Partition ID

### 6.1 Namespace 格式

```text
{tenant_id}/{training_run_id}/{adapter_name}/train_{k}
```

由 `TrainingContext.partition_id(train_id)` 生成。

### 6.2 init_namespace

```python
def init_namespace(self, context: TrainingContext) -> None:
```

验证 context metadata 可序列化。当前实现只调用 `context.metadata()` 做校验，不在 TQ 中创建显式 namespace 对象。namespace 通过写入时的 `partition_id` 和 `tag` 隐式建立。

### 6.3 next_partition_id

```python
def next_partition_id(self, context: TrainingContext) -> str:
```

线程安全地递增 `context.key` 对应的 train_id，返回 `context.partition_id(train_id)`。

## 7. Partition 生命周期

### 7.1 Partition 是什么

**Partition 是一个 rollout step 的数据容器，代表从数据产生到训练消费再到清理的完整生命周期。**

在异步 RL 训练中，系统需要持续产生新数据（rollout）并训练模型。每次产生的一批数据称为一个 rollout step，对应一个 `train_k`（k 是递增的整数）。Partition 就是这个 `train_k` 在 TransferQueue 中的物理载体。

**为什么需要 Partition？**

1. **数据隔离**：不同 rollout step 的数据不能混合。`train_0` 的数据和 `train_1` 的数据必须分开存储、分开处理、分开训练。
2. **状态追踪**：每批数据从产生到训练完成要经历多个阶段（rollout → reward → advantage → train → clear）。Partition 是状态变化的主体，所有 worker 的操作都以它为边界。
3. **生命周期管理**：Partition 有明确的起点（创建）和终点（清理）。清理后释放 TQ 容量，允许新的 rollout step 创建新 partition。
4. **多租户隔离**：Partition ID 包含完整的 scope 信息（`tenant_id/training_run_id/adapter_name/train_k`），确保不同租户、不同 adapter 的数据不会交叉。

**Partition 与 Sample 的关系：**

- 一个 Partition 包含多个 Sample（trajectory）。
- Sample 是数据的最小单位（一条完整的 trajectory）。
- Partition 是状态管理的最小单位（所有 sample 共享同一个状态）。

**Partition 与 Policy Version 的关系：**

- 一个 Partition 对应一个 `policy_version`（rollout 时使用的模型版本）。
- 同一个 partition 内的所有 sample 必须来自同一个 policy version。
- 训练完成后同步权重，policy version 递增，下一个 partition 使用新版本。

### 7.2 Partition ID 命名规则

**逻辑名 vs 物理名：**

```text
逻辑名:   train_k          （文档中使用的简写）
物理名:   {tenant_id}/{training_run_id}/{adapter_name}/train_k  （TQ 中实际使用的 key）
```

**生成逻辑（`data_plane.py:78-82`）：**

```python
def next_partition_id(self, context: TrainingContext) -> str:
    with self._lock:
        train_id = self._next_train_id[context.key]  # 自增计数器，从 0 开始
        self._next_train_id[context.key] += 1
        return context.partition_id(train_id)
        # 例: tenant_a/run_001/lora/train_0
        #     tenant_a/run_001/lora/train_1
        #     tenant_a/run_001/lora/train_2
```

**关键点：**

- `context.key` = `tenant_id/training_run_id/adapter_name`，是 scope 的唯一标识。
- `_next_train_id` 是 per-context 的自增计数器，确保同一个 scope 内的 partition ID 不重复。
- 不同 scope 的 partition ID 天然不同（因为 `context.key` 不同），不会在 TQ 中冲突。

**示例：**

```text
Scope A: tenant_a/run_001/lora_a
  → train_0, train_1, train_2, ...

Scope B: tenant_b/run_002/lora_b
  → train_0, train_1, train_2, ...  （与 Scope A 的 train_0 不冲突，因为完整 ID 不同）
```

### 7.3 状态机

Partition 的状态机定义了数据从产生到消亡的完整生命周期：

```text
OPEN ──(seal/ready_groups >= target_groups)──> ROLLOUT_DONE
ROLLOUT_DONE ──(append_rewards)──────────────> REWARD_DONE
REWARD_DONE ──(append_advantages)────────────> TRAIN_READY
TRAIN_READY ──(mark_training)────────────────> TRAINING
TRAINING ──(mark_trained)────────────────────> TRAIN_DONE
TRAIN_DONE ──(clear_partition)───────────────> CLEARED

任意状态 ──(异常)──> FAILED
任意状态 ──(取消)──> CANCELLED
```

**状态含义：**

| 状态 | 含义 | 谁触发 | 触发条件 |
|------|------|--------|----------|
| `OPEN` | 初始状态，正在接收 rollout 数据 | `create_partition()` | 创建新 partition |
| `ROLLOUT_DONE` | Rollout 完成，等待 reward 计算 | `put_rollout_batch()` | `seal=True` 或 `ready_groups >= target_groups` |
| `REWARD_DONE` | Reward 计算完成，等待 advantage 计算 | `append_rewards()` | RewardWorker 写回 reward 字段 |
| `TRAIN_READY` | Advantage 计算完成，可以开始训练 | `append_advantages()` | AdvantageWorker 写回 advantage/returns 字段 |
| `TRAINING` | 正在训练 | `mark_training()` | TrainerWorker 开始读取数据 |
| `TRAIN_DONE` | 训练完成，等待权重同步和清理 | `mark_trained()` | TrainerWorker 完成所有 optimizer steps |
| `CLEARED` | 已清理，生命周期结束 | `clear_partition()` | BaseRLPipeline 完成权重同步后调用 |
| `FAILED` | 异常终止 | 任何操作 | 运行时异常（如 worker 崩溃） |
| `CANCELLED` | 用户取消 | `clear_namespace()` | 用户主动取消训练任务 |

**状态转换的原子性：**

- 每次状态转换都会同步更新 TQ 中所有 sample 的 tag（`_sync_partition_status()`）。
- 这确保多进程场景下，任何 worker 通过 `kv_list()` 都能读到最新状态。

### 7.4 创建 Partition

```python
def create_partition(
    self,
    context: TrainingContext,
    *,
    target_groups: int,
    partition_id: Optional[str] = None,
) -> PartitionMetadata:
```

**参数说明：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `context` | `TrainingContext` | 是 | 训练上下文，包含 scope 信息 |
| `target_groups` | `int` | 是 | 目标 prompt group 数量。一个 prompt group 包含 `num_generations` 条 trajectory。 |
| `partition_id` | `str \| None` | 否 | 指定 partition ID。为 `None` 时自动调用 `next_partition_id()` 生成。 |

**返回值：**

| 类型 | 说明 |
|------|------|
| `PartitionMetadata` | 新创建的 partition 元数据，`status == OPEN` |

**内部行为：**

1. 如果 `partition_id` 为 `None`，调用 `next_partition_id(context)` 生成新 ID。
2. 创建 `PartitionMetadata` 对象，初始状态为 `OPEN`。
3. 写入 `_meta` 缓存（线程安全）。
4. 返回 metadata。

**调用时机：**

- `AsyncRollouter._select_or_create_partition()`：当没有 `OPEN` 状态的 partition 时创建新 partition。
- 也可以手动调用（测试或特殊场景）。

**示例：**

```python
# 自动分配 ID
meta = data_plane.create_partition(context, target_groups=128)
# meta.partition_id = "tenant_a/run_001/lora/train_0"
# meta.status = OPEN
# meta.target_groups = 128

# 手动指定 ID
meta = data_plane.create_partition(context, target_groups=64, partition_id="custom/train_5")
```

### 7.5 写入 Rollout 数据

```python
def put_rollout_batch(
    self,
    context: TrainingContext,
    partition_id: str,
    trajectories: List[SampleRecord],
    *,
    ready_groups: int = 1,
    seal: bool = False,
) -> PartitionMetadata:
```

**参数说明：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `context` | `TrainingContext` | 是 | 训练上下文，用于校验 partition 归属和写入 metadata |
| `partition_id` | `str` | 是 | 目标 partition ID |
| `trajectories` | `List[SampleRecord]` | 是 | 要写入的 trajectory 列表。每个 trajectory 是一个 dict，包含 `messages`、`group_id`、`generation_idx` 等字段。 |
| `ready_groups` | `int` | 否 | 本次写入的 prompt group 数量，默认 1。累加到 `meta.ready_groups`。 |
| `seal` | `bool` | 否 | 是否立即密封 partition（状态转为 `ROLLOUT_DONE`），默认 `False`。 |

**返回值：**

| 类型 | 说明 |
|------|------|
| `PartitionMetadata` | 更新后的 partition 元数据 |

**内部行为：**

1. **校验归属**：检查 `meta.context.key == context.key`，防止跨 scope 写入。
2. **自动创建**：如果 partition 不存在，调用 `create_partition()` 创建。
3. **遍历 trajectories**：对每个 sample：
   - 合并 `context.metadata()` 到 sample metadata（确保每个 sample 都带完整的 scope 信息）。
   - 调用 `context.validate_metadata()` 校验一致性。
   - 生成 sample key（优先用 `sample_id`，否则 `{partition_id}/sample_{idx}`）。
   - 提取 fields（排除 `metadata` 和 `sample_id`）和 tag（metadata + partition tag）。
4. **批量写入 TQ**：调用 `_kv_batch_put(keys, partition_id, fields, tags)`。
5. **更新统计**：`meta.num_rows += len(keys)`，`meta.ready_groups += ready_groups`。
6. **状态转换**：如果 `seal=True` 或 `ready_groups >= target_groups`，状态转为 `ROLLOUT_DONE`。
7. **同步状态**：调用 `_sync_partition_status()` 把状态写回 TQ 所有 sample 的 tag。

**调用时机：**

- `AsyncRollouter.run_one_group()`：rollout 完成后写入结果。

**示例：**

```python
# 写入一个 prompt group（包含 8 条 trajectory）
trajectories = [
    {'sample_id': 's0', 'messages': [...], 'group_id': 'g0', 'generation_idx': 0},
    {'sample_id': 's1', 'messages': [...], 'group_id': 'g0', 'generation_idx': 1},
    # ... 共 8 条
]
meta = data_plane.put_rollout_batch(context, "tenant_a/run_001/lora/train_0", trajectories, ready_groups=1)
# meta.num_rows = 8
# meta.ready_groups = 1
# meta.status = OPEN（如果 target_groups > 1）

# 写入最后一个 group 并密封
meta = data_plane.put_rollout_batch(context, partition_id, trajectories, ready_groups=1, seal=True)
# meta.status = ROLLOUT_DONE
```

**多次写入 vs 一次写入：**

- 一个 partition 可以接收多次 `put_rollout_batch`（多个 prompt group），直到 `ready_groups >= target_groups`。
- 也可以一次性写入所有数据并 `seal=True`。
- 两种方式最终状态都是 `ROLLOUT_DONE`。

### 7.6 Claim / Append 模式（Reward 和 Advantage）

Reward 和 Advantage worker 使用 **claim-append** 模式处理数据：

```text
1. Claim: 找到目标状态的 partition，读取 samples
2. Compute: 在 worker 内部计算 reward 或 advantage
3. Append: 写回计算结果，推进状态
```

#### 7.6.1 Claim Reward Batch

```python
def claim_reward_batch(
    self,
    context: TrainingContext,
    batch_size: int,
    *,
    worker_id: Optional[str] = None,
) -> tuple[PartitionMetadata, list[SampleRecord]]:
```

**参数说明：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `context` | `TrainingContext` | 是 | 指定从哪个 scope 的 `ROLLOUT_DONE` partition 中 claim |
| `batch_size` | `int` | 是 | 最多返回的 sample 数量 |
| `worker_id` | `str \| None` | 否 | 如果提供，获取排他租约（防止多 worker 重复处理） |

**返回值：**

| 类型 | 说明 |
|------|------|
| `tuple[PartitionMetadata, list[SampleRecord]]` | (被 claim 的 partition 元数据, 最多 `batch_size` 个 sample) |

**内部行为：**

1. 调用 `list_partitions(context, statuses=[ROLLOUT_DONE])` 找到候选 partition。
2. 取第一个（按 `created_at` 排序，最老的优先）。
3. 如果 `worker_id` 不为 `None`，调用 `claim_partition_with_lease()` 获取租约。
4. 调用 `_get_samples()` 从 TQ 读取数据。
5. 返回 `(meta, samples[:batch_size])`。

**异常：**

- `LookupError`: 没有 `ROLLOUT_DONE` 状态的 partition。
- `RuntimeError`: 如果指定了 `worker_id` 且 partition 已被其他 worker 租约。

#### 7.6.2 Append Rewards

```python
def append_rewards(
    self,
    context: TrainingContext,
    partition_id: str,
    rewards: list[float],
    *,
    field_name: str = 'rewards',
) -> PartitionMetadata:
```

**参数说明：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `context` | `TrainingContext` | 是 | 用于校验 sample metadata 一致性 |
| `partition_id` | `str` | 是 | 目标 partition ID |
| `rewards` | `list[float]` | 是 | reward 值列表，长度必须等于 partition 内 sample 数量 |
| `field_name` | `str` | 否 | 写入 TQ 的字段名，默认 `'rewards'` |

**返回值：**

| 类型 | 说明 |
|------|------|
| `PartitionMetadata` | 更新后的 partition 元数据，`status == REWARD_DONE` |

**内部行为：**

1. 从 TQ 读取 partition 内所有 samples。
2. 校验 `len(rewards) == len(samples)`。
3. 对每个 sample 调用 `context.validate_metadata()` 确保一致性。
4. 批量写入 reward 字段到 TQ（`_batch_update_samples()`）。
5. 更新 partition 状态为 `REWARD_DONE`。
6. 同步状态到 TQ tag。

**异常：**

- `ValueError`: reward 数量不匹配或 metadata 校验失败。

#### 7.6.3 Claim Advantage Batch

```python
def claim_advantage_batch(
    self,
    context: TrainingContext,
    batch_size: int,
    *,
    worker_id: Optional[str] = None,
) -> tuple[PartitionMetadata, list[SampleRecord]]:
```

**行为：** 与 `claim_reward_batch` 相同，但查找 `REWARD_DONE` 状态的 partition。

#### 7.6.4 Append Advantages

```python
def append_advantages(
    self,
    context: TrainingContext,
    partition_id: str,
    advantages: list[float],
    returns: Optional[list[float]] = None,
) -> PartitionMetadata:
```

**参数说明：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `context` | `TrainingContext` | 是 | 用于校验 sample metadata 一致性 |
| `partition_id` | `str` | 是 | 目标 partition ID |
| `advantages` | `list[float]` | 是 | advantage 值列表 |
| `returns` | `list[float] \| None` | 否 | return 值列表；为 `None` 时使用 `advantages` 作为 `returns` |

**返回值：**

| 类型 | 说明 |
|------|------|
| `PartitionMetadata` | 更新后的 partition 元数据，`status == TRAIN_READY` |

**内部行为：** 与 `append_rewards` 类似，但写入 `advantages` 和 `returns` 字段，状态转为 `TRAIN_READY`。

### 7.7 训练阶段

#### 7.7.1 Mark Training

```python
def mark_training(self, context: TrainingContext, partition_id: str) -> PartitionMetadata:
```

**行为：** 将 partition 状态从 `TRAIN_READY` 转为 `TRAINING`。TrainerWorker 开始读取数据前调用。

#### 7.7.2 Build Streaming Dataloader

```python
def build_streaming_dataloader(
    self,
    context: TrainingContext,
    partition_id: str,
    *,
    task_name: Optional[str] = None,
) -> list[SampleRecord]:
```

**参数说明：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `context` | `TrainingContext` | 是 | 用于校验 partition 归属 |
| `partition_id` | `str` | 是 | 要读取的 partition ID |
| `task_name` | `str \| None` | 否 | 消费任务名称（如 `'train'`）；指定时自动过滤已 ack 的 sample |

**返回值：**

| 类型 | 说明 |
|------|------|
| `list[SampleRecord]` | partition 内所有（或未消费的）sample，每个包含 `sample_id`、`metadata`、以及所有 fields（messages、rewards、advantages、returns 等） |

**内部行为：**

1. 校验 `meta.context.key == context.key`。
2. 从 TQ 读取 partition 内所有 samples。
3. 如果 `task_name` 不为 `None`，过滤掉已 ack 的 sample（`_consumed[partition_id][task_name]`）。

**注意：** 当前实现返回 `list[SampleRecord]`，后续应集成 TQ 原生的 `StreamingDataset` / `StreamingDataLoader`。

#### 7.7.3 Mark Trained

```python
def mark_trained(self, context: TrainingContext, partition_id: str) -> PartitionMetadata:
```

**行为：** 将 partition 状态从 `TRAINING` 转为 `TRAIN_DONE`。TrainerWorker 完成所有 optimizer steps 后调用。

#### 7.7.4 Clear Partition

```python
def clear_partition(self, context: TrainingContext, partition_id: str) -> None:
```

**行为：**

1. 校验 partition 归属。
2. 从 TQ 获取 partition 内所有 key。
3. 批量清理 TQ 数据（`kv_clear`）。
4. 更新 `_meta` 中该 partition 的状态为 `CLEARED`。
5. 清理 `_consumed` 和 `_leases` 中该 partition 的记录。

**调用时机：** BaseRLPipeline 完成权重同步后调用，释放 TQ 容量，允许新的 rollout step 创建新 partition。

### 7.8 Partition 生命周期完整示例

```python
# 1. 创建 partition
meta = data_plane.create_partition(context, target_groups=2)
# partition_id = "tenant_a/run_001/lora/train_0"
# status = OPEN, target_groups = 2, ready_groups = 0

# 2. 写入第一个 prompt group
trajectories_1 = [...]  # 8 条 trajectory
meta = data_plane.put_rollout_batch(context, partition_id, trajectories_1, ready_groups=1)
# status = OPEN, ready_groups = 1

# 3. 写入第二个 prompt group 并密封
trajectories_2 = [...]  # 8 条 trajectory
meta = data_plane.put_rollout_batch(context, partition_id, trajectories_2, ready_groups=1, seal=True)
# status = ROLLOUT_DONE, ready_groups = 2, num_rows = 16

# 4. Reward worker claim 并计算
meta, samples = data_plane.claim_reward_batch(context, batch_size=1024)
rewards = reward_fn(samples)  # worker 内部计算
meta = data_plane.append_rewards(context, partition_id, rewards)
# status = REWARD_DONE

# 5. Advantage worker claim 并计算
meta, samples = data_plane.claim_advantage_batch(context, batch_size=1024)
advantages, returns = advantage_fn(samples)  # worker 内部计算
meta = data_plane.append_advantages(context, partition_id, advantages, returns)
# status = TRAIN_READY

# 6. Trainer worker 开始训练
meta = data_plane.mark_training(context, partition_id)
# status = TRAINING

dataloader = data_plane.build_streaming_dataloader(context, partition_id, task_name='train')
for batch in dataloader:
    model.forward_backward(batch)
    model.optimizer_step()

meta = data_plane.mark_trained(context, partition_id)
# status = TRAIN_DONE

# 7. 权重同步（BaseRLPipeline 调用 CheckpointEngineManager）
ckpt_manager.sync_weights(context.adapter_name, context.policy_version)

# 8. 清理 partition
data_plane.clear_partition(context, partition_id)
# status = CLEARED, TQ 数据已删除

# 9. 下一个 rollout step 创建新 partition
meta = data_plane.create_partition(context, target_groups=2)
# partition_id = "tenant_a/run_001/lora/train_1"
# 生命周期重新开始
```

### 7.9 Partition 与 Staleness 控制

**StalenessManager** 使用 partition 数量计算 rollout capacity：

```text
max_live_partitions = max_staleness + 1

live_partition_count = len([p for p in partitions if p.status != CLEARED])

if live_partition_count >= max_live_partitions:
    available_partition_slots = 0  # 不允许新 rollout
else:
    available_partition_slots = max_live_partitions - live_partition_count
```

**示例：**

```text
max_staleness = 1
max_live_partitions = 2

当前状态:
  train_0: TRAINING
  train_1: ROLLOUT_DONE

live_partition_count = 2
available_partition_slots = 0  # rollout 必须等待 train_0 完成并清理
```

**关键点：**

- Partition 清理后，`live_partition_count` 减少，`available_partition_slots` 增加。
- 这形成了自然的背压机制：rollout 不能无限领先于 trainer。
- `max_staleness=0` 表示严格同步（最多 1 个 live partition）。
- `max_staleness=1` 允许 rollout 和 trainer 重叠（最多 2 个 live partition）。

## 8. 查询与容量

### 8.1 list_partitions

```python
def list_partitions(self, context=None, *, statuses=None) -> list[PartitionMetadata]:
```

先调用 `_load_partition_meta()` 从 TQ 重建 `_meta`，再按 context 和 statuses 过滤，按 `(created_at, partition_id)` 排序。

### 8.2 get_metadata

```python
def get_metadata(self, context=None) -> list[PartitionMetadata]:
```

`list_partitions` 的别名，供 `StalenessManager` 使用。

### 8.3 check_capacity

```python
def check_capacity(self, context: TrainingContext) -> bool:
```

检查 `max_rows`（全局）和 `max_rows_per_context`（单 context）是否还有余量。

### 8.4 list_train_ready_partitions

```python
def list_train_ready_partitions(self) -> list[PartitionMetadata]:
```

等价于 `list_partitions(statuses=[PartitionStatus.TRAIN_READY])`。

## 9. 内部方法

| 方法 | 说明 |
|---|---|
| `_init_transfer_queue(config)` | `import transfer_queue`，可选调用 `tq.init()` |
| `_build_tq_config(config)` | 用 OmegaConf 构建 TQ config dict |
| `_claim_samples(context, batch_size, statuses, task_name)` | 找到目标 partition 并读取 samples |
| `_mark_status(context, partition_id, status)` | 校验归属后更新状态 |
| `_get_samples(partition_id)` | 从 TQ 读取 partition 全部 samples |
| `_update_samples(partition_id, updates)` | 对已有 samples 追加/更新 fields |
| `_sync_partition_status(meta)` | 把 partition 状态写回 TQ 所有 key 的 tag |
| `_load_partition_meta()` | 从 TQ tag 重建 `_meta` 缓存 |
| `_rows_from_tq_data(data, size)` | 将 TQ 返回的 dict-of-lists 转为 list-of-dicts |
| `_split_field(value, size)` | 处理单值/tensor/list 的字段拆分 |
| `_meta_from_tag(partition_id, tag, num_rows)` | 从 TQ tag 重建 `PartitionMetadata` |

## 10. 线程安全

所有写操作（`create_partition`、`put_rollout_batch`、`_mark_status`、`clear_partition`）都在 `self._lock` (RLock) 内执行。读操作（`list_partitions`、`check_capacity`）也在锁内读取 `_meta` 快照。

## 11. 隔离约束

1. **跨 context 写入拒绝**：`put_rollout_batch`、`append_rewards`、`append_advantages`、`mark_training`、`mark_trained`、`build_streaming_dataloader`、`clear_partition` 都校验 `meta.context.key == context.key`。
2. **metadata 一致性**：写入时调用 `context.validate_metadata(sample_meta)` 确保 sample 的 tag 与 context 一致。
3. **namespace 隔离**：不同 `TrainingContext` 的 partition_id 天然不同（包含 `tenant_id/training_run_id/adapter_name`），不会在 TQ 中冲突。

## 12. 与上层组件的交互

### 12.1 具体交互类信息

```text
BaseRLPipeline:
  __init__:     data_plane.init_namespace(context)
  step_async:   通过 worker 间接调用

AsyncRollouter:
  add_pending:           data_plane.init_namespace(context)
  pick_next_training_context: data_plane.check_capacity(context)
  _state_for:            data_plane.get_metadata(context)
  _select_or_create_partition: data_plane.list_partitions / create_partition
  run_one_group:         data_plane.put_rollout_batch(context, partition_id, trajectories)

RewardWorker:
  run_once:   data_plane.claim_reward_batch -> append_rewards

AdvantageWorker:
  run_once:   data_plane.claim_advantage_batch -> append_advantages

TrainerScheduler:
  next_partition: 使用 data_plane.list_train_ready_partitions() 的结果

TrainerWorker:
  run_once:   data_plane.mark_training -> build_streaming_dataloader -> mark_trained -> clear_partition

StalenessManager:
  get_rollout_capacity: 使用 data_plane.get_metadata(context) 的结果
```

### 12.2 与外部交互接口

一共8个对外接口。由于DataPlane是管理TQ的，所以最终的交互需要区分metadata和data，也就是是否对TQ组件有操作，还是在DataPlane与元数据交互。

| 对象                | 调用                            | TQ操作                                    | 说明                                                         |
| :------------------ | ------------------------------- | ----------------------------------------- | ------------------------------------------------------------ |
| BaseRLPipeline      | `init_namespace(context)`       | 启动TQ并调用put操作写入metadata信息到TQ中 | `BaseRLPipeline` 让 `TransferQueueDataPlane` 初始化 TQ namespace，例如 `{tenant_id}/{training_run_id}/{adapter_name}/train_k`，并写入基础metadata 约束。 |
| StalenessManager    | `get_metadata()`                |                                           | `TransferQueueDataPlane` 向 `StalenessManager` 提供当前 live partitions、oldest partition、partition 状态和 policy_version 等容量事实。 |
| TransferQueue       | `native TQ ops`                 |                                           | `TransferQueueDataPlane` 将 put/claim/append/clear 等操作转换为底层 TransferQueue backend 操作。 |
| RewardWorker        | `claim / append reward`         |                                           | `RewardWorker` claim rollout-ready 数据，按 `reward_type` 计算 reward，并追加 reward 字段。 |
| AdvantageWorker     | `claim / append advantage`      |                                           | `AdvantageWorker` claim reward-ready 数据，按算法计算 advantage/return，并追加字段。满足训练条件后将 partition 标记为 `TRAIN_READY`。 |
| TrainerScheduler    | `list_train_ready_partitions()` |                                           | `TrainerScheduler` 从 `TransferQueueDataPlane` 查询可训练 partition 候选集合。候选必须已经完成 rollout、reward、advantage。 |
| StreamingDataLoader | `read / ack rows`               |                                           | `StreamingDataset / StreamingDataLoader` 通过 `TransferQueueDataPlane` 从 TQ 读取 rows，并对已消费数据做 ack/progress 更新。 |
| BaseRLPipeline      | `clear_partition(train_k)`      |                                           | 权重同步完成后，`BaseRLPipeline` 通过 `TransferQueueDataPlane` 清理已训练完成的 `train_k`，释放 TQ 容量并推进 staleness 窗口。 |

### 12.3 init_namespace(context)

**调用方**: `BaseRLPipeline.__init__()`  
**设计文档步骤**: 步骤 3  
**操作类型**: 元数据（不直接操作 TQ）

```python
def init_namespace(self, context: TrainingContext) -> None:
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `context` | `TrainingContext` | 是 | 训练上下文，包含 tenant_id/training_run_id/adapter_name 等路由身份 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| 无 | `None` | 仅做校验，不返回数据 |

**内部行为**:
- 调用 `context.metadata()` 验证 TrainingContext 可序列化为 dict
- 不在 TQ 中创建显式 namespace 对象（namespace 通过后续写入时的 partition_id 和 tag 隐式建立）

**异常**:
- `ValueError`: TrainingContext 字段缺失或类型错误时抛出

**调用时机**:
- `BaseRLPipeline.__init__()` 中调用一次
- `AsyncRollouter.add_pending()` 中每次提交新 context 时调用

---

### 12.4 get_metadata(context)

**调用方**: `StalenessManager.get_rollout_capacity()`  
**设计文档步骤**: 步骤 4  
**操作类型**: 元数据 + TQ 读取

```python
def get_metadata(self, context: Optional[TrainingContext] = None) -> list[PartitionMetadata]:
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `context` | `TrainingContext \| None` | 否 | 为 None 时返回所有 partition；指定时只返回该 context 的 partition |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| partitions | `list[PartitionMetadata]` | 按 `(created_at, partition_id)` 排序的 partition 列表 |

**内部行为**:
1. 调用 `tq.kv_list()` 从 TQ 重建 `_meta` 缓存（`_load_partition_meta()`）
2. 按 context.key 过滤（如果指定了 context）
3. 返回排序后的 partition 列表

**TQ 操作**:
- `tq.kv_list()` → 获取所有 partition 的 key 和 tag

**下游使用**:
- `StalenessManager` 用返回的 partition 列表计算 `live_partitions`、`oldest_partition`、`available_groups` 等容量事实
- `AsyncRollouter._state_for()` 用返回值构建 `RolloutContextState`

---

### 12.5 claim / append reward

**调用方**: `RewardWorker.run_once()`  
**设计文档步骤**: 步骤 13  
**操作类型**: TQ 读写（claim 读 + append 写）

#### 12.5.1 claim_reward_batch

```python
def claim_reward_batch(
    self,
    context: TrainingContext,
    batch_size: int,
    *,
    worker_id: Optional[str] = None,
) -> tuple[PartitionMetadata, list[SampleRecord]]:
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `context` | `TrainingContext` | 是 | 指定从哪个 context 的 ROLLOUT_DONE partition 中 claim |
| `batch_size` | `int` | 是 | 最多返回的 sample 数量 |
| `worker_id` | `str \| None` | 否 | 如果提供，获取排他租约（lease） |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| meta | `PartitionMetadata` | 被 claim 的 partition 元数据 |
| samples | `list[SampleRecord]` | 最多 `batch_size` 个 sample，每个包含 `sample_id`、`metadata`、以及 rollout 写入的 fields |

**内部行为**:
1. 调用 `list_partitions(context, statuses=[ROLLOUT_DONE])` 找到候选 partition
2. 取第一个（按 created_at 排序）
3. 如果 `worker_id` 不为 None，调用 `claim_partition_with_lease()` 获取租约
4. 调用 `_get_samples()` 从 TQ 读取数据

**TQ 操作**:
- `tq.kv_list(partition_id)` → 获取 partition 内所有 key 和 tag
- `tq.kv_batch_get(keys, partition_id)` → 批量读取 sample fields

**异常**:
- `LookupError`: 没有 ROLLOUT_DONE 状态的 partition 时抛出
- `RuntimeError`: 如果指定了 worker_id 且 partition 已被其他 worker 租约

#### 12.5.2 append_rewards

```python
def append_rewards(
    self,
    context: TrainingContext,
    partition_id: str,
    rewards: list[float],
    *,
    field_name: str = 'rewards',
) -> PartitionMetadata:
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `context` | `TrainingContext` | 是 | 用于校验 sample metadata 一致性 |
| `partition_id` | `str` | 是 | 目标 partition ID |
| `rewards` | `list[float]` | 是 | reward 值列表，长度必须等于 partition 内 sample 数量 |
| `field_name` | `str` | 否 | 写入 TQ 的字段名，默认 `'rewards'` |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| meta | `PartitionMetadata` | 更新后的 partition 元数据，`status == REWARD_DONE` |

**内部行为**:
1. 从 TQ 读取 partition 内所有 samples
2. 校验 `len(rewards) == len(samples)`
3. 对每个 sample 调用 `context.validate_metadata()` 确保一致性
4. 批量写入 reward 字段到 TQ
5. 更新 partition 状态为 `REWARD_DONE`
6. 同步状态到 TQ tag

**TQ 操作**:
- `tq.kv_list(partition_id)` → 获取 key 和 tag
- `tq.kv_batch_get(keys, partition_id)` → 读取 samples
- `tq.kv_batch_put(keys, partition_id, fields, tags)` → 批量写入 reward
- `tq.kv_batch_put(keys, partition_id, tags)` → 同步状态

**异常**:
- `ValueError`: reward 数量不匹配或 metadata 校验失败

---

### 12.6 claim / append advantage

**调用方**: `AdvantageWorker.run_once()`  
**设计文档步骤**: 步骤 14  
**操作类型**: TQ 读写（claim 读 + append 写）

#### 12.6.1 claim_advantage_batch

```python
def claim_advantage_batch(
    self,
    context: TrainingContext,
    batch_size: int,
    *,
    worker_id: Optional[str] = None,
) -> tuple[PartitionMetadata, list[SampleRecord]]:
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `context` | `TrainingContext` | 是 | 指定从哪个 context 的 REWARD_DONE partition 中 claim |
| `batch_size` | `int` | 是 | 最多返回的 sample 数量 |
| `worker_id` | `str \| None` | 否 | 如果提供，获取排他租约 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| meta | `PartitionMetadata` | 被 claim 的 partition 元数据 |
| samples | `list[SampleRecord]` | 最多 `batch_size` 个 sample，每个包含 `sample_id`、`metadata`、rollout fields、以及 `rewards` 字段 |

**内部行为**: 与 `claim_reward_batch` 相同，但查找 `REWARD_DONE` 状态的 partition

**TQ 操作**: 同 12.5.1

**异常**:
- `LookupError`: 没有 REWARD_DONE 状态的 partition

#### 12.6.2 append_advantages

```python
def append_advantages(
    self,
    context: TrainingContext,
    partition_id: str,
    advantages: list[float],
    returns: Optional[list[float]] = None,
) -> PartitionMetadata:
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `context` | `TrainingContext` | 是 | 用于校验 sample metadata 一致性 |
| `partition_id` | `str` | 是 | 目标 partition ID |
| `advantages` | `list[float]` | 是 | advantage 值列表 |
| `returns` | `list[float] \| None` | 否 | return 值列表；为 None 时使用 advantages 作为 returns |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| meta | `PartitionMetadata` | 更新后的 partition 元数据，`status == TRAIN_READY` |

**内部行为**:
1. 从 TQ 读取 samples
2. 校验 `len(advantages) == len(samples)`
3. 对每个 sample 校验 metadata
4. 批量写入 `advantages` 和 `returns` 字段
5. 更新状态为 `TRAIN_READY`

**TQ 操作**: 同 12.5.2

**异常**:
- `ValueError`: advantage 数量不匹配或 metadata 校验失败

---

### 12.7 list_train_ready_partitions()

**调用方**: `TrainerScheduler.next_partition()`  
**设计文档步骤**: 步骤 15  
**操作类型**: 元数据 + TQ 读取

```python
def list_train_ready_partitions(self) -> list[PartitionMetadata]:
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| 无 | - | - | 无参数 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| partitions | `list[PartitionMetadata]` | 所有 `status == TRAIN_READY` 的 partition，按 `(created_at, partition_id)` 排序 |

**内部行为**:
- 等价于 `list_partitions(statuses=[PartitionStatus.TRAIN_READY])`
- 从 TQ 重建 `_meta`，过滤 TRAIN_READY 状态

**TQ 操作**:
- `tq.kv_list()` → 获取所有 partition 的 key 和 tag

**下游使用**:
- `TrainerScheduler` 对返回的 candidates 做 gating（adapter 状态、sync 状态）和 policy 选择（prefer_current / fair）
- 返回的 partition 必须已经完成 rollout → reward → advantage 全流程

---

### 12.8 read / ack rows

**调用方**: `TrainerWorker.run_once()` → `build_streaming_dataloader()` + `ack_rows()`  
**设计文档步骤**: 步骤 17-18  
**操作类型**: TQ 读取 + 元数据更新

#### 12.8.1 build_streaming_dataloader

```python
def build_streaming_dataloader(
    self,
    context: TrainingContext,
    partition_id: str,
    *,
    task_name: Optional[str] = None,
) -> list[SampleRecord]:
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `context` | `TrainingContext` | 是 | 用于校验 partition 归属 |
| `partition_id` | `str` | 是 | 要读取的 partition ID |
| `task_name` | `str \| None` | 否 | 消费任务名称（如 `'train'`、`'eval'`）；指定时自动过滤已 ack 的 sample |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| samples | `list[SampleRecord]` | partition 内所有（或未消费的）sample，每个包含 `sample_id`、`metadata`、以及所有 fields（messages、rewards、advantages、returns 等） |

**内部行为**:
1. 校验 `meta.context.key == context.key`
2. 从 TQ 读取 partition 内所有 samples
3. 如果 `task_name` 不为 None，过滤掉已 ack 的 sample（`_consumed[partition_id][task_name]`）

**TQ 操作**:
- `tq.kv_list(partition_id)` → 获取 key 和 tag
- `tq.kv_batch_get(keys, partition_id)` → 批量读取 sample fields

**异常**:
- `ValueError`: partition 不属于指定 context

#### 12.8.2 ack_rows

```python
def ack_rows(
    self,
    context: TrainingContext,
    partition_id: str,
    sample_ids: List[str],
    *,
    task_name: str = 'train',
) -> int:
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `context` | `TrainingContext` | 是 | 用于校验 partition 归属 |
| `partition_id` | `str` | 是 | 目标 partition ID |
| `sample_ids` | `list[str]` | 是 | 已消费的 sample ID 列表 |
| `task_name` | `str` | 否 | 消费任务名称，默认 `'train'`；不同 task 独立追踪 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| new_acked | `int` | 本次新增 ack 的 sample 数量（排除已 ack 的） |

**内部行为**:
1. 校验 partition 归属
2. 将 `sample_ids` 加入 `_consumed[partition_id][task_name]`（set 操作，幂等）
3. 返回新增 ack 数量

**TQ 操作**: 无（纯内存操作）

**异常**:
- `ValueError`: partition 不属于指定 context

**配套查询**:
```python
def get_consumed_count(self, partition_id: str, *, task_name: str = 'train') -> int:
```
返回指定 partition 和 task 的已 ack sample 数量。

---

### 12.9 clear_partition(train_k)

**调用方**: `TrainerWorker.run_once()` → `BaseRLPipeline` 权重同步后  
**设计文档步骤**: 步骤 22  
**操作类型**: TQ 写入 + 元数据清理

```python
def clear_partition(self, context: TrainingContext, partition_id: str) -> None:
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `context` | `TrainingContext` | 是 | 用于校验 partition 归属 |
| `partition_id` | `str` | 是 | 要清理的 partition ID |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| 无 | `None` | 清理操作无返回值 |

**内部行为**:
1. 校验 partition 归属（如果 `_meta` 中存在该 partition）
2. 从 TQ 获取 partition 内所有 key
3. 批量清理 TQ 数据
4. 更新 `_meta` 中该 partition 的状态为 `CLEARED`
5. 清理 `_consumed` 和 `_leases` 中该 partition 的记录

**TQ 操作**:
- `tq.kv_list(partition_id)` → 获取所有 key
- `tq.kv_clear(keys, partition_id)` → 批量清理

**异常**:
- `ValueError`: partition 不属于指定 context

**配套批量清理**:
```python
def clear_namespace(self, context: TrainingContext) -> int:
```
清理 context 下所有 partition（用于租户取消训练）。返回实际清理的 partition 数量。

---

### 12.10 native TQ ops（内部实现）

**调用方**: `TransferQueueDataPlane` 内部方法  
**设计文档步骤**: 步骤 12  
**操作类型**: TQ 后端操作

DataPlane 将所有上层操作转换为以下 TQ KV API 调用：

| DataPlane 方法 | TQ 操作 | 说明 |
|---------------|---------|------|
| `put_rollout_batch` | `kv_batch_put` + `kv_put` | 批量写入 sample fields 和 tags |
| `append_rewards` | `kv_batch_put` | 批量更新 reward 字段 |
| `append_advantages` | `kv_batch_put` | 批量更新 advantage/returns 字段 |
| `_sync_partition_status` | `kv_batch_put` | 批量更新所有 key 的 tag（状态同步） |
| `_get_samples` | `kv_list` + `kv_batch_get` | 列出 key 后批量读取 |
| `list_partitions` | `kv_list` | 获取所有 partition 的 key 和 tag |
| `clear_partition` | `kv_list` + `kv_clear` | 列出 key 后批量清理 |

**TQ API 合约**:
- `kv_put(key: str, partition_id: str, fields: dict, tag: dict)` → 写入单个 key
- `kv_batch_put(keys: list[str], partition_id: str, fields: None, tags: list[dict])` → 批量写入 tags
- `kv_batch_get(keys: list[str], partition_id: str)` → 返回 `TensorDict` 或 `dict[str, list]`
- `kv_list(partition_id: str | None)` → 返回 `{partition_id: {key: tag_dict}}`
- `kv_clear(keys: list[str], partition_id: str)` → 批量清理

**数据转换**:
- TQ 返回的 `TensorDict` 通过 `_rows_from_tq_data()` 转换为 `list[SampleRecord]`
- 支持 `to_dict()` 方法（TensorDict）和 dict-of-lists 格式

### 12.11 数据流图：Metadata vs Data 分层架构

#### 12.11.1 TQ 数据模型

TQ 中每个 sample 有两个独立的存储层：

```text
TQ Storage (partition_id = "tenant_a/run_001/lora/train_0"):

  key: "sample_0"
  ├── fields (DATA):  {messages: [...], rewards: 0.8, advantages: 0.3, returns: 0.8}
  └── tag (METADATA): {tenant_id: "tenant_a", status: "ROLLOUT_DONE", num_rows: 3}

  key: "sample_1"
  ├── fields (DATA):  {messages: [...], rewards: 1.0, advantages: 0.5, returns: 1.0}
  └── tag (METADATA): {tenant_id: "tenant_a", status: "ROLLOUT_DONE", num_rows: 3}

  key: "sample_2"
  ├── fields (DATA):  {messages: [...], rewards: 0.5, advantages: -0.2, returns: 0.5}
  └── tag (METADATA): {tenant_id: "tenant_a", status: "ROLLOUT_DONE", num_rows: 3}
```

| 层 | TQ 术语 | 存什么 | 特点 |
|---|---|---|---|
| **fields** | data | 训练数据：messages、rewards、advantages、returns、old_logps | 大，可能是 TensorDict/tensor，按 stage 分阶段追加 |
| **tag** | metadata | 路由和状态：tenant_id、adapter_name、status、policy_version、num_rows | 小，纯 dict，轻量级 |

#### 12.11.2 三层架构数据流图

```text
┌─────────────────────────────────────────────────────────────────────┐
│  上层组件 (Pipeline / Workers / Scheduler)                          │
│  只跟 DataPlane 交互，不直接碰 TQ                                   │
│                                                                     │
│  BaseRLPipeline ──init_namespace()──┐                               │
│  AsyncRollouter ──put_rollout_batch()──┐                            │
│  RewardWorker ──claim/append_reward()──┤                            │
│  AdvantageWorker ──claim/append_advantage()──┤                      │
│  TrainerScheduler ──list_train_ready()──┤                           │
│  TrainerWorker ──read/ack/clear()──┤                                │
│  StalenessManager ──get_metadata()──┤                               │
└─────────────────────────────────────┼───────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  DataPlane 管理层 (TransferQueueDataPlane)                          │
│                                                                     │
│  ┌─ Partition 级 metadata (DataPlane 自己维护) ──────────────────┐  │
│  │                                                                │  │
│  │  self._meta: Dict[str, PartitionMetadata]                      │  │
│  │    → 状态机: OPEN → ROLLOUT_DONE → REWARD_DONE → TRAIN_READY  │  │
│  │            → TRAINING → TRAIN_DONE → CLEARED                   │  │
│  │    → 归属: context.key (tenant_id/run_id/adapter_name)         │  │
│  │    → 统计: num_rows, target_groups, ready_groups               │  │
│  │    → 时间: created_at, updated_at                              │  │
│  │                                                                │  │
│  │  self._leases: Dict[str, lease_info]                           │  │
│  │    → worker_id, deadline (租约互斥)                            │  │
│  │                                                                │  │
│  │  self._consumed: Dict[str, Dict[str, Set[str]]]                │  │
│  │    → partition_id → task_name → set of sample_ids (ack 追踪)   │  │
│  │                                                                │  │
│  │  self._next_train_id: Dict[str, int]                           │  │
│  │    → context.key → 自增计数器 (partition ID 生成)              │  │
│  │                                                                │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              │ 写入方向: _sync_partition_status()   │
│                              │ 读取方向: _load_partition_meta()     │
│                              ▼                                      │
│  ┌─ Sample 级 data + metadata (委托给 TQ) ──────────────────────┐   │
│  │                                                               │   │
│  │  写入: _kv_batch_put(keys, partition_id, fields, tags)        │   │
│  │  读取: _get_samples(partition_id) → kv_list + kv_batch_get    │   │
│  │  清理: clear_partition() → kv_list + kv_clear                 │   │
│  │                                                               │   │
│  └───────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────┬──────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TransferQueue 后端 (tq.kv_*)                                       │
│                                                                     │
│  ┌─ fields (DATA) ──────────────────────────────────────────────┐   │
│  │  messages, rewards, advantages, returns, old_logps, ...      │   │
│  │  → 大对象，TensorDict，按 stage 分阶段追加                    │   │
│  │  → kv_put / kv_batch_put 写入                                │   │
│  │  → kv_batch_get 读取                                         │   │
│  │  → kv_clear 清理                                             │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─ tag (SAMPLE METADATA) ──────────────────────────────────────┐   │
│  │  tenant_id, adapter_name, policy_version, status, num_rows   │   │
│  │  → 小对象，dict，用于 kv_list 过滤和 _load_partition_meta    │   │
│  │  → kv_put / kv_batch_put 写入                                │   │
│  │  → kv_list 读取                                              │   │
│  └───────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

#### 12.11.3 Metadata vs Data 区分原则

| 信息 | 谁维护 | 存在哪 | 为什么 |
|------|--------|--------|--------|
| partition 状态机 (OPEN→...→CLEARED) | **DataPlane** | `_meta` + TQ tag 冗余 | DataPlane 是状态机 owner，TQ tag 是持久化副本 |
| partition 归属 (context.key) | **DataPlane** | `_meta` | 路由逻辑，TQ 只存 tenant_id 等原始字段 |
| lease / 租约 | **DataPlane** | `_leases` | TQ 不感知 worker 调度 |
| ack / 消费追踪 | **DataPlane** | `_consumed` | 当前是内存 set，后续可迁移到 TQ 的 task_name 消费机制 |
| train_id 自增 | **DataPlane** | `_next_train_id` | 纯逻辑计数器 |
| sample fields (messages, rewards, ...) | **TQ** | TQ fields | 大对象，TQ 负责存储和检索 |
| sample tag (tenant_id, status, ...) | **TQ** | TQ tag | DataPlane 写入，TQ 持久化，用于 `_load_partition_meta` 重建 |

**一句话总结**：DataPlane 是 partition 级 metadata 的 owner（状态机、租约、消费追踪），TQ 是 sample 级 data 的 owner（fields + tag）。DataPlane 把 partition 状态冗余写入 TQ tag，是为了多进程场景下 `_load_partition_meta()` 能从 TQ 重建内存状态。

#### 12.11.4 同步机制

```text
写入方向（DataPlane → TQ）:
  put_rollout_batch()  → _sync_partition_status() → kv_batch_put(tag) 写回所有 key
  append_rewards()     → _sync_partition_status() → kv_batch_put(tag) 写回所有 key
  append_advantages()  → _sync_partition_status() → kv_batch_put(tag) 写回所有 key
  mark_training()      → _sync_partition_status() → kv_batch_put(tag) 写回所有 key
  mark_trained()       → _sync_partition_status() → kv_batch_put(tag) 写回所有 key

读取方向（TQ → DataPlane）:
  list_partitions()    → _load_partition_meta()   → kv_list() 读 tag 重建 _meta
```

#### 12.11.5 一致性风险

| 场景 | `_meta`（内存） | TQ tag（后端） | 谁是对的 |
|------|----------------|---------------|---------|
| 单进程正常流程 | ✅ 一致 | ✅ 一致 | 都行 |
| 多进程（Ray worker） | ❌ 各自独立 | ✅ 共享 | TQ |
| worker 崩溃重启 | ❌ 丢失 | ✅ 持久化 | TQ |
| `_sync` 和 `_load` 之间 | 可能过期 | 可能过期 | 都不保证 |

**缓解措施**：
- `list_partitions()` 每次调用都先 `_load_partition_meta()` 从 TQ 重建
- lease 过期自动恢复（`_recover_expired_leases()`）
- ack 当前是纯内存，后续应迁移到 TQ 的 `task_name` 消费机制

## 13. 测试策略

### 13.1 测试用 Fake

`tests/twinkle_agentic/async_rl/fakes.py` 提供 `FakeTransferQueueClient`，实现 `kv_put`、`kv_batch_get`、`kv_list`、`kv_clear` 的内存版本。

### 13.2 核心测试用例

| 测试 | 覆盖点 |
|---|---|
| `test_default_data_plane_requires_real_transfer_queue_when_not_installed` | 无 TQ 时构造失败 |
| `test_data_plane_rollout_reward_advantage_and_clear` | 完整生命周期 |
| `test_data_plane_rejects_cross_context_append` | 跨 context 隔离 |
| `test_data_plane_check_capacity_by_row_limits` | 容量守卫 |
| `test_async_rollouter_and_trainer_worker_mvp_flow` | 端到端集成 |
| `test_base_pipeline_runs_one_multilora_grpo_partition` | pipeline 集成 |

### 13.3 运行测试

```bash
pytest tests/twinkle_agentic/async_rl/test_async_rl_core.py
pytest tests/twinkle_agentic/async_rl/test_base_pipeline.py
```

## 14. 安装 TransferQueue

```bash
pip install TransferQueue
```

带 Yuanrong 后端：
```bash
pip install "TransferQueue[yuanrong]"
```

源码开发：
```bash
git clone <TransferQueue repo>
cd TransferQueue
pip install -e .
```

`TransferQueueDataPlane` 在 `_init_transfer_queue()` 中做 `import transfer_queue as tq`，未安装时抛出明确的 `RuntimeError` 提示。

## 15. 验证要求

实现完成后必须通过以下四层验证。

### 15.1 TQ 接口调用正确性

验证 `TransferQueueDataPlane` 对 `transfer_queue` 的调用是否符合 TQ KV API 合约（参见第 2 节）。

检查项：

| 检查点 | 要求 |
|---|---|
| `tq.init(config)` | config 结构必须包含 `controller` 和 `backend` 两个 key；`backend.SimpleStorage` 必须包含 `num_data_storage_units` |
| `tq.kv_put(key, partition_id, fields, tag)` | `key` 为 str；`partition_id` 为 str；`fields` 为 dict（不含 `metadata` 和 `sample_id`）；`tag` 为 dict |
| `tq.kv_batch_get(keys, partition_id)` | `keys` 为 list[str]；返回值为 dict-of-lists 或 TensorDict，需经 `_rows_from_tq_data` 转换 |
| `tq.kv_list(partition_id)` | `partition_id` 可为 str 或 None；返回 `{partition_id: {key: tag_dict}}` |
| `tq.kv_clear(keys, partition_id)` | `keys` 为 list[str]；`partition_id` 为 str |

验证方法：

```python
# 用 FakeTransferQueueClient 拦截所有 TQ 调用，断言参数类型和结构
fake = FakeTransferQueueClient()
dp = TransferQueueDataPlane(tq_client=fake)

# 写入后检查 fake.fields 和 fake.tags 的内容
dp.put_rollout_batch(context, partition_id, [sample])
assert partition_id in fake.fields
assert all(isinstance(k, str) for k in fake.fields[partition_id])
```

### 15.2 代码正确性

验证 `data_plane.py` 本身无语法错误、接口问题和类型错误。

检查项：

| 检查点 | 要求 |
|---|---|
| 语法 | `python -c "import ast; ast.parse(open('data_plane.py').read())"` 无报错 |
| import | 所有 import 路径存在且可解析 |
| 类型注解 | 参数和返回值类型与调用方期望一致 |
| 公开接口签名 | 与 `workers.py`、`pipeline.py` 中的调用方式完全匹配 |
| 线程安全 | 所有写 `_meta` 的操作在 `self._lock` 内 |

验证方法：

```bash
# 语法检查
python -c "import ast; ast.parse(open('src/twinkle_agentic/async_rl/data_plane.py').read())"

# import 检查
python -c "from twinkle_agentic.async_rl.data_plane import TransferQueueDataPlane, TransferQueueRuntimeConfig"

# 接口匹配：确认 workers.py 和 pipeline.py 的调用签名与 data_plane.py 一致
pytest tests/twinkle_agentic/async_rl/ -v
```

### 15.3 设计文档一致性

验证代码逻辑是否与 `multilora-async-rl/多租户MultiLoRA异步RL设计.md` 的设计意图一致。

检查项：

| 设计文档要求 | 代码实现 | 验证方法 |
|---|---|---|
| namespace 格式 `{tenant_id}/{training_run_id}/{adapter_name}/train_{k}` | `TrainingContext.partition_id()` 生成 | 断言 `context.partition_id(3) == 'tenant/run/lora/train_3'` |
| 同一 train_k 只属于一个 context | `put_rollout_batch` 校验 `meta.context.key == context.key` | 跨 context 写入抛出 `ValueError` |
| rollout 写入后标记 ROLLOUT_DONE | `seal=True` 或 `ready_groups >= target_groups` 时转状态 | 断言 `meta.status == PartitionStatus.ROLLOUT_DONE` |
| reward claim/append 推进到 REWARD_DONE | `append_rewards` 设置状态 | 断言 `meta.status == PartitionStatus.REWARD_DONE` |
| advantage claim/append 推进到 TRAIN_READY | `append_advantages` 设置状态 | 断言 `meta.status == PartitionStatus.TRAIN_READY` |
| 训练完成后清理 partition | `clear_partition` 调用 `tq.kv_clear` 并设置 CLEARED | 断言 `meta.status == PartitionStatus.CLEARED` |
| 容量守卫 | `check_capacity` 检查 `max_rows` 和 `max_rows_per_context` | 超限时返回 False |
| metadata 一致性校验 | `context.validate_metadata()` 在写入时调用 | 不匹配的 metadata 抛出 `ValueError` |

验证方法：

```bash
# 运行现有单元测试，覆盖完整生命周期
pytest tests/twinkle_agentic/async_rl/test_async_rl_core.py -v
pytest tests/twinkle_agentic/async_rl/test_base_pipeline.py -v
```

### 15.4 端到端模型验证

使用真实模型和数据集验证 `TransferQueueDataPlane` 在完整 pipeline 中的行为。

配置：

| 项目 | 值 |
|---|---|
| 模型 | `Qwen/Qwen3.5-0.8B`（或本地路径） |
| 数据集 | `gsm8k`（HuggingFace `gsm8k` 或 ModelScope 镜像） |
| 训练模式 | GRPO，单 LoRA adapter |
| TQ 后端 | SimpleStorage（默认） |
| rollout | `MultiTurnRollout` 或 `EchoRollout`（快速验证时用 mock） |
| reward | `F1Reward` 或 constant reward |

验证流程：

```python
from twinkle_agentic.async_rl import (
    BaseRLPipeline, BaseRLPipelineConfig, TransferQueueDataPlane, TransferQueueRuntimeConfig,
)

# 1. 初始化 TQ（需要 pip install TransferQueue）
data_plane = TransferQueueDataPlane(tq_config=TransferQueueRuntimeConfig(
    num_data_storage_units=4,
    total_storage_size=100000,
))

# 2. 构建 pipeline
config = BaseRLPipelineConfig(
    tenant_id='test_tenant',
    training_run_id='gsm8k_run',
    base_model_id='Qwen/Qwen3.5-0.8B',
    adapter_name='gsm8k_lora',
    reward_type='f1',
    max_staleness=0,
    target_groups_per_partition=1,
    max_train_partitions=2,
)

pipeline = BaseRLPipeline(
    config=config,
    model=model,           # MultiLoraTransformersModel 实例
    rollout=rollout,       # MultiTurnRollout 实例
    reward_registry={'f1': f1_reward_fn},
    data_plane=data_plane,
)

# 3. 提交 prompt samples 并运行
pipeline.run(prompt_samples, max_steps=2)

# 4. 验证
partitions = data_plane.list_partitions(pipeline.context)
assert all(p.status == PartitionStatus.CLEARED for p in partitions)
assert pipeline.adapter_registry.get(pipeline.context).policy_version == 2
```

验证通过标准：

1. pipeline 完整运行 rollout → reward → advantage → train → clear 流程，无异常。
2. 所有 partition 最终状态为 `CLEARED`。
3. `policy_version` 按预期递增。
4. TQ 中无残留数据（`tq.kv_list()` 对应 partition 为空）。
5. 模型权重文件实际生成（`adapter_revision` 路径存在）。

运行命令：

```bash
# 需要 GPU 和 TransferQueue 已安装
pytest tests/twinkle_agentic/async_rl/ -v -k "e2e"
# 或在 cookbook 中运行完整示例
python cookbook/rl/grpo.py --model Qwen/Qwen3.5-0.8B --dataset gsm8k
```

## 16. 真实 TransferQueue 集成测试

### 16.1 环境要求

真实 TQ 测试（不使用 `FakeTransferQueueClient`）需要：

| 依赖 | 说明 |
|---|---|
| `TransferQueue >= 0.1.8` | `pip install TransferQueue` |
| `ray >= 2.10` | TQ 依赖 Ray 作为分布式运行时 |
| NPU 环境 | Ascend 910B3 + CANN 9.0 + torch_npu（或 CPU-only 环境） |
| Ray 初始化 | `num_gpus=0`（NPU 环境不暴露 CUDA），`include_dashboard=False` |

NPU 环境特殊处理：
- 设置 `CUDA_VISIBLE_DEVICES=""` 避免 Ray 尝试 CUDA 路径
- 设置 `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0` 消除 GPU 检测警告
- `conftest.py` 中 session-scoped fixture 统一初始化 Ray

### 16.2 当前验证状态

| 验证层 | 状态 | 说明 |
|---|---|---|
| 15.1 TQ 接口调用正确性 | ✅ 通过 | 8 个测试，使用 `RecordingFakeTQ` 拦截所有调用 |
| 15.2 代码正确性 | ✅ 通过 | 语法、import、类型注解无错误 |
| 15.3 设计文档一致性 | ✅ 通过 | 28 个测试，状态机、隔离约束、容量守卫全部符合 |
| 15.4 端到端模型验证（mock） | ✅ 通过 | 5 个测试，使用 `FakeTransferQueueClient` + mock model/rollout |
| 15.4 端到端模型验证（真实 TQ） | ✅ 通过 | 11 个测试，NPU (Ascend 910B3) + Ray 2.55 + TransferQueue 0.1.8 |

**总计：64 passed, 1 skipped**

### 16.3 真实 TQ 测试文件

`tests/twinkle_agentic/async_rl/test_real_tq.py` 包含 11 个真实 TQ 集成测试：

- `test_init_namespace` - 初始化 namespace
- `test_create_partition` - 创建 partition
- `test_put_rollout_batch_and_read_back` - 写入并读回
- `test_full_lifecycle_with_real_tq` - 完整生命周期
- `test_reward_and_advantage_workers_with_real_tq` - worker 集成
- `test_multi_partition_with_real_tq` - 多 partition
- `test_multi_context_isolation_with_real_tq` - 多 context 隔离
- `test_check_capacity_with_real_tq` - 容量检查
- `test_staleness_manager_with_real_tq` - staleness 管理
- `test_clear_partition_removes_tq_data` - 清理验证
- `test_cross_context_rejected_with_real_tq` - 跨 context 拒绝

运行命令（需要稳定 Ray 环境）：

```bash
pytest tests/twinkle_agentic/async_rl/test_real_tq.py -v --tb=short
```

### 16.4 NPU 环境适配记录

| 问题 | 原因 | 解决方案 |
|---|---|---|
| Ray 初始化超时 (SIGTERM) | torch 为 CPU 版本，Ray 尝试 CUDA 路径失败 | `CUDA_VISIBLE_DEVICES=""` + `num_gpus=0` |
| `kv_batch_get` 返回 TensorDict | 真实 TQ 返回 `tensordict.TensorDict`，非 dict-of-lists | `_rows_from_tq_data` 已处理（`hasattr(data, 'to_dict')`） |
| 跨测试 partition 残留 | 真实 TQ 共享状态，`kv_list()` 返回所有 partition | 每个测试使用唯一 context（tenant/run/adapter 名不同） |
| `max_rows` 容量检查误判 | 全局 `max_rows` 受其他测试残留数据影响 | 改用 `max_rows_per_context`（按 context 隔离） |

## 17. 已知限制与后续演进

1. **状态缓存一致性**：`_meta` 是内存缓存，多进程场景下不同进程的 `_meta` 可能不一致。`_load_partition_meta()` 每次 `list_partitions` 时从 TQ 重建，但不保证原子性。
2. **无 lease/claim 机制**：当前 `claim_reward_batch` / `claim_advantage_batch` 只是找到第一个匹配的 partition 并读取，没有 lease deadline 或并发 claim 互斥。设计文档中提到的 `lease_deadline` 和 `owner_worker_id` 字段已在 `PartitionMetadata` 中预留但未使用。
3. **StreamingDataLoader 未集成**：`build_streaming_dataloader` 当前返回 `list[SampleRecord]`，不是 TQ 的 `StreamingDataset`/`StreamingDataLoader`。后续需要支持 TQ 原生流式消费。
4. **无异步 KV API**：当前所有 TQ 调用都是同步的。TQ 提供了 `async_kv_put` 等异步接口，后续可考虑在异步 worker 中使用。
5. **partition 清理不级联**：`clear_partition` 只清理 TQ 数据和更新 `_meta` 状态，不清理 `_next_train_id` 计数器。