# TransferQueueDataPlane API 文档

**版本**: 1.0  
**日期**: 2026-06-26  
**分支**: multilora-async-rl-tq  
**文件**: `src/twinkle_agentic/async_rl/data_plane.py`  
**导入**: `from twinkle_agentic.async_rl import TransferQueueDataPlane`

---

## 目录

1. [概述](#1-概述)
2. [数据类型](#2-数据类型)
3. [配置](#3-配置)
4. [构造与生命周期](#4-构造与生命周期)
5. [API 1: init_namespace](#5-api-1-init_namespace)
6. [API 2: get_metadata](#6-api-2-get_metadata)
7. [API 3: put_rollout_batch](#7-api-3-put_rollout_batch)
8. [API 4: claim_reward_batch / claim_reward_ready_groups / append_rewards](#8-api-4-claim--append-reward)
9. [API 5: claim_advantage_batch / append_advantages](#9-api-5-claim--append-advantage)
10. [API 6: list_train_ready_partitions](#10-api-6-list_train_ready_partitions)
11. [API 7: build_streaming_dataloader / ack_rows](#11-api-7-read--ack-rows)
12. [API 8: clear_partition / clear_namespace](#12-api-8-clear_partition)
13. [辅助 API](#13-辅助-api)
14. [常量](#14-常量)
15. [状态机](#15-状态机)
16. [完整使用示例](#16-完整使用示例)

---

## 1. 概述

`TransferQueueDataPlane` 是异步 RL pipeline 中唯一的数据面边界。所有对 TransferQueue (TQ) 后端的读写操作都必须经过此类。

**8 个对外接口**（对应 8 个上游组件）：

| # | 接口 | 调用方 | 说明 |
|---|------|--------|------|
| 1 | `init_namespace` | BaseRLPipeline | 初始化 namespace |
| 2 | `get_metadata` | StalenessManager | 查询 partition 聚合信息 |
| 3 | `put_rollout_batch` | AsyncRollouter | 写入 rollout 数据 |
| 4 | `claim_reward_batch` / `append_rewards` | RewardWorker | claim + 追加 reward |
| 5 | `claim_advantage_batch` / `append_advantages` | AdvantageWorker | claim + 追加 advantage |
| 6 | `list_train_ready_partitions` | TrainerScheduler | 查询可训练 partition |
| 7 | `build_streaming_dataloader` / `ack_rows` | TrainerWorker | 读取训练数据 + 确认消费 |
| 8 | `clear_partition` | BaseRLPipeline | 清理已完成 partition |

---

## 2. 数据类型

### 2.1 TrainingContext

训练上下文，标识一个训练任务的路由身份。

```python
@dataclass(frozen=True)
class TrainingContext:
    tenant_id: str                           # 租户 ID
    training_run_id: str                     # 训练任务 ID
    base_model_id: str                       # 基础模型 ID
    adapter_name: str                        # LoRA adapter 名称
    adapter_revision: Optional[str] = None   # adapter 权重版本
    policy_version: int = 0                  # rollout 使用的模型版本
    env_type: str = 'tool_calling'           # 环境类型
    tool_profile: str = 'default'            # 工具配置
    reward_type: str = 'default'             # reward 实现类型
    loss_type: str = 'default'               # loss 类型
    algorithm: str = 'grpo'                  # 训练算法
```

**关键属性**：

| 属性/方法 | 返回值 | 说明 |
|-----------|--------|------|
| `context.key` | `str` | `"{tenant_id}/{training_run_id}/{adapter_name}"`，scope 唯一标识 |
| `context.partition_id(train_id)` | `str` | `"{key}/train_{k}"`，完整 partition ID |
| `context.metadata()` | `dict` | 所有字段展开为 dict，用于写入 TQ tag |
| `context.validate_metadata(dict)` | `None` | 校验 dict 与 context 一致，不一致抛 `ValueError` |
| `context.with_policy_version(v)` | `TrainingContext` | 返回新 context，更新 policy_version |

### 2.2 PartitionStatus

```python
class PartitionStatus(StrEnum):
    OPEN = 'OPEN'
    ROLLOUT_DONE = 'ROLLOUT_DONE'
    REWARD_DONE = 'REWARD_DONE'
    TRAIN_READY = 'TRAIN_READY'
    TRAINING = 'TRAINING'
    TRAIN_DONE = 'TRAIN_DONE'
    CLEARED = 'CLEARED'
    FAILED = 'FAILED'
    CANCELLED = 'CANCELLED'
```

### 2.3 PartitionMetadata

```python
@dataclass
class PartitionMetadata:
    context: TrainingContext          # 所属 scope
    partition_id: str                 # 完整 partition ID
    policy_version: int               # rollout 使用的模型版本
    target_groups: int = 0            # 目标 prompt group 数
    ready_groups: int = 0             # 已写入的 prompt group 数
    status: PartitionStatus = OPEN    # 当前状态
    created_at: float                 # 创建时间戳
    updated_at: float                 # 最后更新时间戳
    owner_worker_id: Optional[str]    # 当前租约持有者
    lease_deadline: Optional[float]   # 租约过期时间
    num_rows: int = 0                 # partition 内 sample 总数
```

### 2.4 QueueMetadata

```python
@dataclass
class QueueMetadata:
    context: TrainingContext                       # 所属 scope
    active_partitions: list[PartitionMetadata]     # 未清理的 partition 列表
    total_rows: int                                # active partition 内的总 sample 数
    trainer_step: int                              # 已完成的训练步数
    current_policy_version: int                    # 当前 policy 版本
```

**属性**：

| 属性 | 返回值 | 说明 |
|------|--------|------|
| `live_partition_count` | `int` | `len(active_partitions)` |
| `oldest_partition` | `PartitionMetadata \| None` | 最老的 active partition |

**兼容性**：实现了 `__iter__` 和 `__len__`，可直接作为 `list[PartitionMetadata]` 使用。

### 2.5 SampleRecord

```python
SampleRecord = Dict[str, Any]
```

每个 sample 是一个 dict，至少包含：

| 字段 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `sample_id` | `str` | DataPlane 生成或 sample 自带 | sample 唯一标识 |
| `metadata` | `dict` | DataPlane 注入 | 包含 context 所有字段 + partition 状态 |
| `messages` | `list[dict]` | rollout 写入 | 对话历史 |
| `group_id` | `str` | rollout 写入 | prompt group 标识 |
| `generation_idx` | `int` | rollout 写入 | group 内序号 |
| `old_logps` | `list[float]` | rollout 写入 | log probabilities |
| `rewards` | `float` | RewardWorker 追加 | reward 值 |
| `advantages` | `float` | AdvantageWorker 追加 | advantage 值 |
| `returns` | `float` | AdvantageWorker 追加 | return 值 |

---

## 3. 配置

### 3.1 TransferQueueRuntimeConfig

```python
@dataclass
class TransferQueueRuntimeConfig:
    # TQ 后端
    total_storage_size: Optional[int] = None
    num_data_storage_units: int = 4
    storage_backend: str = 'SimpleStorage'
    controller: Dict[str, Any] = field(default_factory=dict)
    backend: Dict[str, Any] = field(default_factory=dict)
    init: bool = True

    # 容量规划输入
    target_groups: int = 128
    num_generations: int = 8
    max_staleness: int = 1
    estimate_bytes_per_sample: Optional[int] = None
    safety_factor: float = 1.2

    # 容量保护阈值
    max_rows: Optional[int] = None
    max_rows_per_context: Optional[int] = None
    max_tq_bytes: Optional[int] = None
    max_live_partitions_per_context: Optional[int] = None

    # 运行时
    lease_timeout: float = 300.0
```

### 3.2 容量自动计算

```text
samples_per_partition = target_groups * num_generations
max_live_partitions   = max_staleness + 1
max_rows              = samples_per_partition * max_live_partitions
max_tq_bytes          = estimate_bytes_per_sample * max_rows * safety_factor
```

优先级：**显式值 > 自动计算 > 默认值**

### 3.3 配置示例

```python
# 标准 GRPO
config = TransferQueueRuntimeConfig(
    target_groups=128,
    num_generations=8,
    max_staleness=1,
    estimate_bytes_per_sample=32768,
)
# 自动: max_rows=2048, max_live_partitions=2

# 多租户
config = TransferQueueRuntimeConfig(
    target_groups=64, num_generations=8, max_staleness=1,
    max_rows=8192,                     # 全局上限
    max_rows_per_context=2048,         # 单 adapter 上限
    max_live_partitions_per_context=2, # 单 adapter partition 上限
)

# 测试（mock）
config = TransferQueueRuntimeConfig(init=False)
dp = TransferQueueDataPlane(tq_client=FakeTransferQueueClient(), tq_config=config)
```

---

## 4. 构造与生命周期

### 4.1 构造

```python
def __init__(
    self,
    tq_client: Optional[Any] = None,
    tq_config: Optional[TransferQueueRuntimeConfig] = None,
)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `tq_client` | `Any \| None` | TQ 客户端。为 `None` 时自动 `import transfer_queue` 并初始化。测试时传 mock。 |
| `tq_config` | `TransferQueueRuntimeConfig \| None` | 配置。为 `None` 时使用默认配置。 |

**异常**：`tq_client=None` 且 `transfer_queue` 未安装时抛 `RuntimeError`。

### 4.2 关闭

```python
def close(self) -> None
```

调用 `tq.close()` 释放资源。

---

## 5. API 1: init_namespace

**调用方**: `BaseRLPipeline.__init__()`  
**操作类型**: 元数据校验

```python
def init_namespace(self, context: TrainingContext) -> None
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `context` | `TrainingContext` | 是 | 训练上下文 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| 无 | `None` | - |

**行为**：校验 `context.metadata()` 可序列化。namespace 通过后续写入时的 `partition_id` 和 `tag` 隐式建立。

**调用时机**：pipeline 初始化时调用一次，每次提交新 context 时也可调用。

```python
dp = TransferQueueDataPlane(tq_config=config)
dp.init_namespace(context)
```

---

## 6. API 2: get_metadata

**调用方**: `StalenessManager.get_rollout_capacity()`  
**操作类型**: TQ 读取 + 元数据聚合

```python
def get_metadata(
    self,
    context: Optional[TrainingContext] = None,
) -> QueueMetadata | list[PartitionMetadata]
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `context` | `TrainingContext \| None` | 否 | 指定 scope。为 `None` 时返回所有 partition 列表。 |

| 返回值 | 条件 | 类型 | 说明 |
|--------|------|------|------|
| `QueueMetadata` | `context` 不为 `None` | 聚合信息 | 包含 `active_partitions`、`total_rows`、`trainer_step`、`current_policy_version` |
| `list[PartitionMetadata]` | `context` 为 `None` | partition 列表 | 向后兼容 |

**QueueMetadata 属性**：

| 属性 | 类型 | 说明 |
|------|------|------|
| `context` | `TrainingContext` | 所属 scope |
| `active_partitions` | `list[PartitionMetadata]` | 未清理的 partition（排除 CLEARED/CANCELLED） |
| `total_rows` | `int` | active partition 内的总 sample 数 |
| `trainer_step` | `int` | 已完成的训练步数 |
| `current_policy_version` | `int` | 当前 policy 版本 |
| `live_partition_count` | `int` | `len(active_partitions)` |
| `oldest_partition` | `PartitionMetadata \| None` | 最老的 active partition |

**TQ 操作**：`tq.kv_list()` → 重建 `_meta` 缓存

```python
qm = dp.get_metadata(context)
print(qm.live_partition_count)    # 2
print(qm.oldest_partition.status) # ROLLOUT_DONE
print(qm.total_rows)              # 2048

# 兼容 list 用法
for partition in qm:
    print(partition.partition_id)
```

---

## 7. API 3: put_rollout_batch

**调用方**: `AsyncRollouter.run_one_group()`  
**操作类型**: TQ 写入

```python
def put_rollout_batch(
    self,
    context: TrainingContext,
    partition_id: str,
    trajectories: List[SampleRecord],
    *,
    ready_groups: int = 1,
    seal: bool = False,
) -> PartitionMetadata
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `context` | `TrainingContext` | 是 | 训练上下文 |
| `partition_id` | `str` | 是 | 目标 partition ID |
| `trajectories` | `List[SampleRecord]` | 是 | trajectory 列表，每个是 dict |
| `ready_groups` | `int` | 否 | 本次写入的 prompt group 数，默认 1 |
| `seal` | `bool` | 否 | 是否立即密封，默认 `False` |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `PartitionMetadata` | 更新后的 partition 元数据 |

**行为**：
1. 校验 partition 归属（跨 scope 写入抛 `ValueError`）
2. 如果 partition 不存在，自动创建
3. 为每个 sample 注入 `context.metadata()` 到 tag
4. 批量写入 TQ（`kv_batch_put`）
5. 更新 `num_rows` 和 `ready_groups`
6. 如果 `seal=True` 或 `ready_groups >= target_groups`，状态转为 `ROLLOUT_DONE`

**异常**：
- `ValueError`: partition 不属于指定 context，或 metadata 校验失败

```python
# 写入一个 prompt group（8 条 trajectory）
trajectories = [
    {'sample_id': 's0', 'messages': [...], 'group_id': 'g0', 'generation_idx': 0, 'old_logps': [...]},
    # ... 共 8 条
]
meta = dp.put_rollout_batch(context, partition_id, trajectories, ready_groups=1)

# 写入并密封
meta = dp.put_rollout_batch(context, partition_id, trajectories, seal=True)
# meta.status == PartitionStatus.ROLLOUT_DONE
```

---

## 8. API 4: claim / append reward

**调用方**: `RewardWorker.run_once()`  
**操作类型**: TQ 读取 + 写入

### 8.1 claim_reward_batch

```python
def claim_reward_batch(
    self,
    context: TrainingContext,
    batch_size: int,
    *,
    worker_id: Optional[str] = None,
) -> tuple[PartitionMetadata, list[SampleRecord]]
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `context` | `TrainingContext` | 是 | 指定从哪个 scope 的 `ROLLOUT_DONE` partition 中 claim |
| `batch_size` | `int` | 是 | 最多返回的 sample 数量 |
| `worker_id` | `str \| None` | 否 | 获取排他租约，防止多 worker 重复 claim |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `(PartitionMetadata, list[SampleRecord])` | tuple | (被 claim 的 partition, 最多 batch_size 个 sample) |

**异常**：
- `LookupError`: 没有 `ROLLOUT_DONE` 状态的 partition
- `RuntimeError`: worker_id 指定且 partition 已被其他 worker 租约

### 8.2 claim_reward_ready_groups

按 group 级别 claim（详细设计 7.1 接口）。

```python
def claim_reward_ready_groups(
    self,
    context: TrainingContext,
    num_generations: int,
    max_groups: int,
    *,
    worker_id: Optional[str] = None,
) -> tuple[PartitionMetadata, list[list[SampleRecord]]]
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `context` | `TrainingContext` | 是 | 指定 scope |
| `num_generations` | `int` | 是 | 每个 group 的 trajectory 数 |
| `max_groups` | `int` | 是 | 最多返回的 group 数 |
| `worker_id` | `str \| None` | 否 | 获取排他租约 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `(PartitionMetadata, list[list[SampleRecord]])` | tuple | (partition, group 列表，每个 group 是 `num_generations` 个 sample) |

**行为**：按 `group_id` 分组，只返回完整 group（>= `num_generations` 条）。

### 8.3 append_rewards

```python
def append_rewards(
    self,
    context: TrainingContext,
    partition_id: str,
    rewards: list[float],
    *,
    field_name: str = 'rewards',
) -> PartitionMetadata
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `context` | `TrainingContext` | 是 | 用于校验 sample metadata |
| `partition_id` | `str` | 是 | 目标 partition |
| `rewards` | `list[float]` | 是 | reward 值列表，长度必须等于 sample 数 |
| `field_name` | `str` | 否 | TQ 字段名，默认 `'rewards'` |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `PartitionMetadata` | 更新后元数据 | `status == REWARD_DONE` |

**异常**：
- `ValueError`: reward 数量不匹配或 metadata 校验失败

```python
# RewardWorker 典型用法
meta, samples = dp.claim_reward_batch(context, batch_size=1024, worker_id='rw1')
rewards = reward_fn([s.get('trajectory', s) for s in samples])
meta = dp.append_rewards(context, meta.partition_id, rewards)
dp.release_lease(meta.partition_id, worker_id='rw1')
# meta.status == PartitionStatus.REWARD_DONE
```

---

## 9. API 5: claim / append advantage

**调用方**: `AdvantageWorker.run_once()`  
**操作类型**: TQ 读取 + 写入

### 9.1 claim_advantage_batch

```python
def claim_advantage_batch(
    self,
    context: TrainingContext,
    batch_size: int,
    *,
    worker_id: Optional[str] = None,
) -> tuple[PartitionMetadata, list[SampleRecord]]
```

参数和返回值与 `claim_reward_batch` 相同，但查找 `REWARD_DONE` 状态的 partition。

### 9.2 append_advantages

```python
def append_advantages(
    self,
    context: TrainingContext,
    partition_id: str,
    advantages: list[float],
    returns: Optional[list[float]] = None,
) -> PartitionMetadata
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `context` | `TrainingContext` | 是 | 用于校验 sample metadata |
| `partition_id` | `str` | 是 | 目标 partition |
| `advantages` | `list[float]` | 是 | advantage 值列表 |
| `returns` | `list[float] \| None` | 否 | return 值列表。为 `None` 时使用 `advantages` |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `PartitionMetadata` | 更新后元数据 | `status == TRAIN_READY` |

```python
# AdvantageWorker 典型用法
meta, samples = dp.claim_advantage_batch(context, batch_size=1024, worker_id='aw1')
advantages, returns = advantage_fn(samples)
meta = dp.append_advantages(context, meta.partition_id, advantages, returns)
dp.release_lease(meta.partition_id, worker_id='aw1')
# meta.status == PartitionStatus.TRAIN_READY
```

---

## 10. API 6: list_train_ready_partitions

**调用方**: `TrainerScheduler.next_partition()`  
**操作类型**: TQ 读取 + 元数据过滤

```python
def list_train_ready_partitions(self) -> list[PartitionMetadata]
```

| 参数 | 无 |
|------|-----|

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `list[PartitionMetadata]` | 所有 `TRAIN_READY` 状态的 partition | 按 `(created_at, partition_id)` 排序 |

```python
ready = dp.list_train_ready_partitions()
for p in ready:
    print(f'{p.partition_id}: {p.context.adapter_name} v{p.policy_version}')
```

---

## 11. API 7: read / ack rows

**调用方**: `TrainerWorker.run_once()`  
**操作类型**: TQ 读取 + 元数据更新

### 11.1 build_streaming_dataloader

```python
def build_streaming_dataloader(
    self,
    context: TrainingContext,
    partition_id: str,
    *,
    task_name: Optional[str] = None,
    required_fields: Optional[frozenset] = None,
) -> list[SampleRecord]
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `context` | `TrainingContext` | 是 | 用于校验 partition 归属 |
| `partition_id` | `str` | 是 | 要读取的 partition |
| `task_name` | `str \| None` | 否 | 消费任务名（如 `'train'`）；指定时自动过滤已 ack 的 sample |
| `required_fields` | `frozenset \| None` | 否 | 必需字段集合；指定时校验每个 sample 字段完整性 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `list[SampleRecord]` | sample 列表 | 每个包含 `sample_id`、`metadata`、以及所有 fields |

**异常**：
- `ValueError`: partition 不属于指定 context，或 sample 缺少必需字段

### 11.2 ack_rows

```python
def ack_rows(
    self,
    context: TrainingContext,
    partition_id: str,
    sample_ids: List[str],
    *,
    task_name: str = 'train',
) -> int
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `context` | `TrainingContext` | 是 | 用于校验 partition 归属 |
| `partition_id` | `str` | 是 | 目标 partition |
| `sample_ids` | `list[str]` | 是 | 已消费的 sample ID 列表 |
| `task_name` | `str` | 否 | 消费任务名，默认 `'train'`。不同 task 独立追踪 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `int` | 本次新增 ack 的 sample 数 | 排除已 ack 的（幂等操作） |

```python
# TrainerWorker 典型用法
dp.mark_training(context, partition_id)
samples = dp.build_streaming_dataloader(
    context, partition_id,
    task_name='train',
    required_fields=TRAIN_REQUIRED_FIELDS,
)
for batch in batches(samples):
    model.forward_backward(batch)
    model.optimizer_step()
    dp.ack_rows(context, partition_id, [s['sample_id'] for s in batch], task_name='train')
dp.mark_trained(context, partition_id)
```

---

## 12. API 8: clear_partition

**调用方**: `BaseRLPipeline`（权重同步后）  
**操作类型**: TQ 清理 + 元数据更新

### 12.1 clear_partition

```python
def clear_partition(self, context: TrainingContext, partition_id: str) -> None
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `context` | `TrainingContext` | 是 | 用于校验 partition 归属 |
| `partition_id` | `str` | 是 | 要清理的 partition |

| 返回值 | 无 | `None` |
|--------|-----|--------|

**行为**：
1. 校验 partition 归属
2. 批量清理 TQ 数据（`kv_clear`）
3. 状态转为 `CLEARED`
4. 清理 `_consumed` 和 `_leases`

### 12.2 clear_namespace

批量清理一个 scope 下所有 partition（用于租户取消训练）。

```python
def clear_namespace(self, context: TrainingContext) -> int
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `context` | `TrainingContext` | 是 | 要清理的 scope |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `int` | 实际清理的 partition 数 |

```python
# 权重同步后清理
dp.clear_partition(context, partition_id)

# 租户取消训练
cleared = dp.clear_namespace(context)
print(f'Cleared {cleared} partitions')
```

---

## 13. 辅助 API

### 13.1 create_partition

```python
def create_partition(
    self,
    context: TrainingContext,
    *,
    target_groups: int,
    partition_id: Optional[str] = None,
) -> PartitionMetadata
```

创建新 partition，状态 `OPEN`。`partition_id` 为 `None` 时自动生成。

### 13.2 next_partition_id

```python
def next_partition_id(self, context: TrainingContext) -> str
```

生成下一个 partition ID（`{context.key}/train_{k}`），`k` 自增。

### 13.3 list_partitions

```python
def list_partitions(
    self,
    context: Optional[TrainingContext] = None,
    *,
    statuses: Optional[Iterable[PartitionStatus]] = None,
) -> list[PartitionMetadata]
```

列出 partition，可按 context 和状态过滤。

### 13.4 check_capacity

```python
def check_capacity(self, context: TrainingContext) -> bool
```

检查是否还有容量接收新数据。检查三个维度：全局 rows、per-context rows、per-context live partitions。

### 13.5 mark_training / mark_trained

```python
def mark_training(self, context: TrainingContext, partition_id: str) -> PartitionMetadata
def mark_trained(self, context: TrainingContext, partition_id: str) -> PartitionMetadata
```

状态推进：`TRAIN_READY → TRAINING → TRAIN_DONE`。`mark_trained` 自动递增 `trainer_step`。

### 13.6 Lease 管理

```python
def claim_partition_with_lease(
    self, context, partition_id, *, worker_id: str, timeout: Optional[float] = None,
) -> PartitionMetadata

def release_lease(self, partition_id: str, *, worker_id: str) -> None

def renew_lease(self, partition_id: str, *, worker_id: str, timeout: Optional[float] = None) -> None
```

排他租约机制。`timeout` 默认使用 `config.lease_timeout`（300 秒）。过期自动释放。

### 13.7 get_consumed_count

```python
def get_consumed_count(self, partition_id: str, *, task_name: str = 'train') -> int
```

返回指定 partition 和 task 的已 ack sample 数。

---

## 14. 数据规范

以下常量定义了 TransferQueue 中各阶段的数据字段规范，用于文档化和测试基准。这些规范不在代码中导出，但调用方应遵循以确保数据完整性。

### 14.1 Row Field Schema

每个 sample 在 TQ 中分为 **fields**（数据）和 **tag**（元数据）两层。fields 按 stage 追加写入，tag 由 DataPlane 统一管理。

#### Rollout 阶段写入的 fields

```python
ROLLOUT_FIELDS = frozenset({
    'messages',          # list[dict] - 对话历史
    'group_id',          # str - prompt group 标识
    'generation_idx',    # int - group 内序号
    'old_logps',         # list[float] - rollout 时的 log probabilities
})
```

#### Reward 阶段追加的 fields

```python
REWARD_FIELDS = frozenset({
    'rewards',           # float - reward 值
})
```

#### Advantage 阶段追加的 fields

```python
ADVANTAGE_FIELDS = frozenset({
    'advantages',        # float - advantage 值
    'returns',           # float - return 值
})
```

#### 训练阶段必需的 fields

训练前可使用此集合校验 sample 字段完整性：

```python
TRAIN_REQUIRED_FIELDS = frozenset(
    ROLLOUT_FIELDS | REWARD_FIELDS | ADVANTAGE_FIELDS
)
```

**使用示例**：

```python
# 训练前校验所有必需字段
samples = dp.build_streaming_dataloader(
    ctx, partition_id, 
    required_fields=TRAIN_REQUIRED_FIELDS
)
# 如果缺少字段会抛出 ValueError
```

### 14.2 Sample Isolation Tag Fields

每个 sample 的 tag 必须包含以下隔离字段，用于多租户和多 LoRA 隔离：

```python
SAMPLE_ISOLATION_TAG_FIELDS = frozenset({
    'tenant_id',         # 业务租户
    'training_run_id',   # 训练任务 ID
    'base_model_id',     # 基础模型 ID
    'adapter_name',      # LoRA adapter 名称
    'adapter_revision',  # adapter 权重版本（可选）
    'policy_version',    # rollout 使用的模型版本
    'partition_id',      # 所属 partition
    'group_id',          # prompt group 标识
    'generation_idx',    # group 内序号
})
```

### 14.3 Task Names

TQ 内部区分数据处理阶段的逻辑名字：

```python
TASK_NAMES = frozenset({
    'rollout',           # AsyncRollouter 写入
    'reward',            # RewardWorker 消费和追加
    'advantage',         # AdvantageWorker 消费和追加
    'train',             # TrainerWorker 消费
})
```

**说明**：`task_name` 用于 `ack_rows()` 和 `build_streaming_dataloader(task_name=)` 参数，区分不同阶段的消费状态。

---

## 15. 状态机

```text
OPEN ──(seal / ready_groups >= target_groups)──> ROLLOUT_DONE
ROLLOUT_DONE ──(append_rewards)────────────────> REWARD_DONE
REWARD_DONE ──(append_advantages)──────────────> TRAIN_READY
TRAIN_READY ──(mark_training)──────────────────> TRAINING
TRAINING ──(mark_trained)──────────────────────> TRAIN_DONE
TRAIN_DONE ──(clear_partition)─────────────────> CLEARED
```

---

## 16. 完整使用示例

```python
from twinkle_agentic.async_rl import (
    TransferQueueDataPlane,
    TransferQueueRuntimeConfig,
    TrainingContext,
    PartitionStatus,
    TRAIN_REQUIRED_FIELDS,
)

# ── 1. 初始化 ──
config = TransferQueueRuntimeConfig(
    target_groups=128, num_generations=8, max_staleness=1,
)
dp = TransferQueueDataPlane(tq_config=config)

context = TrainingContext(
    tenant_id='tenant_a',
    training_run_id='run_001',
    base_model_id='Qwen/Qwen3.5-0.8B',
    adapter_name='gsm8k_lora',
    reward_type='gsm8k',
    loss_type='grpo',
    algorithm='grpo',
)

dp.init_namespace(context)

# ── 2. Rollout 写入 ──
partition_id = dp.next_partition_id(context)
partition = dp.create_partition(context, target_groups=1)

trajectories = [
    {'sample_id': f's{i}', 'messages': [...], 'group_id': 'g0',
     'generation_idx': i, 'old_logps': [...]}
    for i in range(8)
]
meta = dp.put_rollout_batch(context, partition_id, trajectories, ready_groups=1, seal=True)
assert meta.status == PartitionStatus.ROLLOUT_DONE

# ── 3. Reward ──
meta, samples = dp.claim_reward_batch(context, batch_size=1024, worker_id='rw1')
rewards = [1.0] * len(samples)
meta = dp.append_rewards(context, partition_id, rewards)
dp.release_lease(partition_id, worker_id='rw1')
assert meta.status == PartitionStatus.REWARD_DONE

# ── 4. Advantage ──
meta, samples = dp.claim_advantage_batch(context, batch_size=1024, worker_id='aw1')
advantages = [0.0] * len(samples)
meta = dp.append_advantages(context, partition_id, advantages)
dp.release_lease(partition_id, worker_id='aw1')
assert meta.status == PartitionStatus.TRAIN_READY

# ── 5. Train ──
ready = dp.list_train_ready_partitions()
assert len(ready) == 1

dp.mark_training(context, partition_id)
samples = dp.build_streaming_dataloader(
    context, partition_id,
    task_name='train',
    required_fields=TRAIN_REQUIRED_FIELDS,
)
# ... model.forward_backward(samples) ...
dp.ack_rows(context, partition_id, [s['sample_id'] for s in samples])
dp.mark_trained(context, partition_id)

# ── 6. 权重同步 + 清理 ──
# ckpt_manager.sync_weights(context.adapter_name, context.policy_version)
dp.clear_partition(context, partition_id)
assert dp.list_partitions(context)[0].status == PartitionStatus.CLEARED

# ── 7. 查询 ──
qm = dp.get_metadata(context)
print(f'live={qm.live_partition_count}, rows={qm.total_rows}, step={qm.trainer_step}')

dp.close()
```
