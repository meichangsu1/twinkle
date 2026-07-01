# TrainerScheduler API 文档

## 概述

`TrainerScheduler` 是异步 RL pipeline 中训练侧的调度决策组件。它负责从 `TransferQueueDataPlane` 查询 `TRAIN_READY` 状态的 partition，应用五层 gating 过滤，并通过可插拔的调度策略选择下一个要训练的 `train_k`。

**位置**: `src/twinkle_agentic/async_rl/scheduler.py`

**导出**: `from twinkle_agentic.async_rl import TrainerScheduler`

---

## 公开 API

### 1. `__init__`

```python
def __init__(
    self,
    *,
    adapter_registry: AdapterRegistry,
    data_plane: Optional[TransferQueueDataPlane] = None,
    train_policy: Optional[Any] = None,
    config: Optional[TrainerSchedulerConfig] = None,
)
```

**构造参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `adapter_registry` | `AdapterRegistry` | 是 | Adapter 运行时状态表，用于 gating 判断 |
| `data_plane` | `TransferQueueDataPlane \| None` | 否 | TQ 数据面引用。提供后可调用 `list_train_ready_partitions()` 和 `next_partition()` 无参版本 |
| `train_policy` | `Any \| None` | 否 | 调度策略实例。优先于 `config.build_policy()` |
| `config` | `TrainerSchedulerConfig \| None` | 否 | 配置对象。用于构建策略（当 `train_policy` 为 None 时） |

**示例**:

```python
# 基础用法（无 data_plane，需显式传入 candidates）
scheduler = TrainerScheduler(adapter_registry=registry)

# 完整用法（有 data_plane，可自动查询 TQ）
scheduler = TrainerScheduler(
    adapter_registry=registry,
    data_plane=data_plane,
    config=TrainerSchedulerConfig(policy='lru'),
)
```

---

### 2. `list_train_ready_partitions`

```python
def list_train_ready_partitions(self) -> list[PartitionMetadata]
```

**用途**: 查询 `TransferQueueDataPlane`，返回所有 `TRAIN_READY` 状态的 partition。

**返回值**: `list[PartitionMetadata]` — 可训练的 partition 列表。

**行为**:
- 如果构造时未提供 `data_plane`，返回空列表。
- 内部调用 `data_plane.list_train_ready_partitions()`。

**示例**:

```python
ready_partitions = scheduler.list_train_ready_partitions()
print(f"Found {len(ready_partitions)} trainable partitions")
```

**设计文档引用**: 多租户设计 3.0 步骤 15。

---

### 3. `next_partition`

```python
def next_partition(
    self,
    candidates: Optional[List[PartitionMetadata]] = None,
    current_context: Optional[TrainingContext] = None,
) -> Optional[PartitionMetadata]
```

**用途**: 选择下一个要训练的 partition。

**参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `candidates` | `list[PartitionMetadata] \| None` | 否 | 候选 partition 列表。为 `None` 时自动调用 `list_train_ready_partitions()` |
| `current_context` | `TrainingContext \| None` | 否 | 当前正在训练的 context（用于 `prefer_current` 等策略） |

**返回值**: `PartitionMetadata | None` — 选中的 partition，或 `None`（无合法候选）。

**行为**:
1. 如果 `candidates` 为 `None`，调用 `list_train_ready_partitions()` 查询 TQ。
2. 应用五层 gating 过滤（见下文）。
3. 委托给 `train_policy.pick_next_partition(eligible, current_context)` 选择。

**示例**:

```python
# 自动查询 TQ（推荐用法）
partition = scheduler.next_partition(current_context=current_ctx)

# 显式传入候选（测试或特殊场景）
partition = scheduler.next_partition(candidates=my_list, current_context=current_ctx)
```

**设计文档引用**: 多租户设计 3.0 步骤 16。

---

### 4. `list_eligible_partitions`

```python
def list_eligible_partitions(
    self,
    candidates: List[PartitionMetadata],
    current_context: Optional[TrainingContext] = None,
) -> list[PartitionMetadata]
```

**用途**: 只执行 gating 过滤，不执行策略选择。返回所有通过 gating 的合法候选。

**参数**: 同 `next_partition`。

**返回值**: `list[PartitionMetadata]` — 通过 gating 的 partition 列表。

**用途场景**:
- 调试：查看哪些 partition 通过了 gating。
- Metrics：统计 eligible 数量。
- 自定义策略：先获取 eligible，再用自定义逻辑选择。

**示例**:

```python
eligible = scheduler.list_eligible_partitions(candidates)
print(f"{len(eligible)} partitions passed gating")
```

---

### 5. `get_schedule_decision`

```python
def get_schedule_decision(
    self,
    candidates: List[PartitionMetadata],
    current_context: Optional[TrainingContext] = None,
) -> ScheduleDecision
```

**用途**: 执行完整调度并返回决策详情（用于日志和 metrics）。

**参数**: 同 `next_partition`。

**返回值**: `ScheduleDecision` — 包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `selected` | `PartitionMetadata \| None` | 选中的 partition |
| `reason` | `str` | 决策原因：`'selected'` / `'no_candidates'` / `'all_rejected'` / `'policy_returned_none'` |
| `total_candidates` | `int` | 输入的候选总数 |
| `eligible_count` | `int` | 通过 gating 的数量 |
| `rejected` | `list[RejectedPartition]` | 被拒绝的 partition 及原因 |
| `elapsed_ms` | `float` | 调度耗时（毫秒） |

**示例**:

```python
decision = scheduler.get_schedule_decision(candidates)
if decision.selected:
    print(f"Selected {decision.selected.partition_id} in {decision.elapsed_ms:.2f}ms")
else:
    print(f"No selection: {decision.reason}")
    for r in decision.rejected:
        print(f"  Rejected {r.partition_id}: {r.reason}")
```

---

### 6. `is_compatible`

```python
def is_compatible(self, partition: PartitionMetadata) -> bool
```

**用途**: 判断 partition 的 `reward_type` / `loss_type` / `algorithm` 是否与当前 trainer 兼容。

**默认行为**: 返回 `True`（第一版支持所有类型）。

**扩展方式**: 子类覆盖此方法以实现自定义过滤。

**示例**:

```python
class StrictScheduler(TrainerScheduler):
    def is_compatible(self, partition):
        return partition.context.algorithm == 'ppo'

scheduler = StrictScheduler(adapter_registry=registry)
```

**设计文档引用**: 多租户设计 4.2 gating 第 5 条。

---

## Gating 机制

`next_partition`、`list_eligible_partitions`、`get_schedule_decision` 都会执行五层 gating 过滤：

| 层级 | 条件 | 拒绝原因 |
|------|------|---------|
| G1 | `partition.status == TRAIN_READY` | `'not_train_ready'` |
| G2 | `partition.context.key` 合法 | `'unknown_adapter'` |
| G3 | `adapter.state not in (FAILED, CANCELLED, DRAINING)` | `'adapter_terminal_state'` |
| G4 | `adapter_registry.can_train(context)` 通过 | `'cannot_train'` |
| G5 | `is_compatible(partition)` 返回 `True` | `'incompatible'` |

**`can_train` 条件**:
- `adapter.state == ACTIVE`
- `not adapter.sync_in_progress`
- `adapter.training_partition is None`

---

## 关联类型

### `TrainerSchedulerConfig`

配置驱动的策略构建：

```python
@dataclass
class TrainerSchedulerConfig:
    policy: str = 'prefer_current'
    switch_penalty: float = 0.0
    switch_cost: float = 1.0
    fairness_quantum: float = 1.0
    aging_rate: float = 0.1
    max_staleness: int = 1
    adaptive_high_load_threshold: float = 3.0
    adaptive_switch_rate_threshold: float = 0.7
    weights: Dict[str, float] = field(default_factory=dict)
    fairness_unit: str = 'partition'

    def build_policy(self, adapter_registry: AdapterRegistry) -> Any:
        """根据 policy 字段构建对应的调度策略实例。"""
```

**支持的 policy**:

| policy | 策略类 | 说明 |
|--------|--------|------|
| `'prefer_current'` | `PreferCurrentTrainPolicy` | 优先当前 adapter |
| `'cost_aware'` | `CostAwareTrainPolicy` | 显式建模切换成本 |
| `'sjf'` | `SJFTrainPolicy` | 最短作业优先 |
| `'fair'` | `DeficitFairTrainPolicy` | 加权公平轮转 |
| `'stride'` | `StrideTrainPolicy` | 确定性比例份额 |
| `'wfq'` | `WeightedFairQueueTrainPolicy` | 加权公平排队 |
| `'lru'` | `LRUTrainPolicy` | 最近最少使用 |
| `'edf'` | `EDFTrainPolicy` | 最早截止时间优先 |
| `'priority'` | `PriorityTrainPolicy` | 优先级 + aging |
| `'adaptive'` | `AdaptiveTrainPolicy` | 自适应混合策略 |
| `'fifo'` | `FIFOTrainPolicy` | 先进先出 |

### `ScheduleDecision`

调度决策快照（frozen dataclass）：

```python
@dataclass(frozen=True)
class ScheduleDecision:
    selected: Optional[PartitionMetadata]
    reason: str
    total_candidates: int
    eligible_count: int
    rejected: list[RejectedPartition]
    elapsed_ms: float
```

### `RejectedPartition`

被拒绝的 partition 记录（frozen dataclass）：

```python
@dataclass(frozen=True)
class RejectedPartition:
    partition_id: str
    context_key: str
    reason: str
```

---

## 调用方

### `TrainerWorker.step()`

热路径调用方式：

```python
class TrainerWorker:
    def step(self) -> Optional[ComponentResult]:
        partition = self.scheduler.next_partition(current_context=self.current_context)
        if partition is None:
            return None
        # ... 训练流程 ...
```

### `BaseRLPipeline.build_trainer_scheduler()`

构造方式：

```python
def build_trainer_scheduler(self, *, train_policy: Optional[Any]) -> TrainerScheduler:
    config = TrainerSchedulerConfig(
        policy=self.config.train_schedule_policy,
        switch_penalty=self.config.train_schedule_switch_penalty,
        # ... 其他配置 ...
    )
    return TrainerScheduler(
        adapter_registry=self.adapter_registry,
        data_plane=self.data_plane,
        train_policy=train_policy,
        config=config,
    )
```

---

## 完整示例

```python
from twinkle_agentic.async_rl import (
    AdapterRegistry,
    TrainerScheduler,
    TrainerSchedulerConfig,
    TransferQueueDataPlane,
    TrainingContext,
)

# 1. 初始化依赖
registry = AdapterRegistry()
data_plane = TransferQueueDataPlane(tq_config=config)

# 2. 注册 adapter
ctx = TrainingContext(
    tenant_id='tenant_a',
    training_run_id='run_001',
    base_model_id='Qwen/Qwen3.5-0.8B',
    adapter_name='gsm8k_lora',
)
registry.register(ctx)
data_plane.init_namespace(ctx)

# 3. 创建 scheduler
scheduler = TrainerScheduler(
    adapter_registry=registry,
    data_plane=data_plane,
    config=TrainerSchedulerConfig(policy='lru'),
)

# 4. 查询可训练 partition
ready = scheduler.list_train_ready_partitions()
print(f"Found {len(ready)} trainable partitions")

# 5. 选择下一个 partition（自动查询 TQ）
partition = scheduler.next_partition(current_context=ctx)
if partition:
    print(f"Selected {partition.partition_id}")

# 6. 获取完整决策 trace（用于调试）
decision = scheduler.get_schedule_decision(
    candidates=ready,
    current_context=ctx,
)
print(f"Decision: {decision.reason}, elapsed: {decision.elapsed_ms:.2f}ms")
```

---

## 设计文档引用

- **总体架构**: `multilora-async-rl/多租户MultiLoRA异步RL设计.md` 3.0 步骤 15-16
- **Gating 条件**: `multilora-async-rl/多租户MultiLoRA异步RL设计.md` 4.2
- **调度策略**: `multilora-async-rl/多租户MultiLoRA异步RL设计.md` 4.2.1 - 4.2.2
- **Spec 文档**: `multilora-async-rl/TrainerScheduler_spec.md`

---

## 测试覆盖

83 个测试用例覆盖：

- **Gating**: 9 个测试（5 层过滤 + `is_compatible` 钩子）
- **策略**: 58 个测试（11 种策略的行为验证）
- **配置**: 14 个测试（`TrainerSchedulerConfig.build_policy`）
- **决策 trace**: 5 个测试（`ScheduleDecision` 字段验证）
- **集成**: 7 个测试（`list_train_ready_partitions` + `next_partition` 无参版本）

运行测试：

```bash
pytest tests/twinkle_agentic/async_rl/test_scheduler.py -v
```
