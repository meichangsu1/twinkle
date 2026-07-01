# TransferQueueDataPlane 开发日志 - 2026-06-30

**分支**: multilora-async-rl-scheduler  
**环境**: NPU (Ascend 910B3) + TransferQueue 0.1.8 + Ray 2.55.1

---

## 1. TrainerScheduler 特性开发

### 1.1 设计文档

编写了完整的 TrainerScheduler 设计文档 `TrainerScheduler_spec.md`，包含：

- **5 层 Gating 机制**：TRAIN_READY 状态检查、AdapterRegistry 查询、终态过滤、can_train 检查、is_compatible 兼容性检查
- **11 种调度策略**：涵盖吞吐优先、公平调度、新鲜度优先、优先级、自适应等类别
- **配置驱动**：通过 `TrainerSchedulerConfig` 支持 YAML 配置
- **可观测性**：`ScheduleDecision` 提供完整调度决策 trace

### 1.2 代码实现

#### 新增文件

- `src/twinkle_agentic/async_rl/scheduler.py` - TrainerScheduler 核心实现
- `src/twinkle_agentic/async_rl/rollout_scheduling.py` - Rollout 调度策略
- `src/twinkle_agentic/async_rl/train_scheduling.py` - Train 调度策略
- `tests/twinkle_agentic/async_rl/test_scheduler.py` - 83 个测试用例

#### 修改文件

- `src/twinkle_agentic/async_rl/workers.py` - 移除 TrainerScheduler，从 scheduler.py 导入
- `src/twinkle_agentic/async_rl/pipeline.py` - 添加 train_schedule_* 配置字段
- `src/twinkle_agentic/async_rl/__init__.py` - 导出新类

### 1.3 11 种调度策略

| 策略 | 类别 | 核心思想 |
|------|------|---------|
| `prefer_current` | 吞吐优先 | 优先当前 adapter，减少 LoRA 切换 |
| `cost_aware` | 吞吐优先 | 显式建模切换成本，批量训练同 adapter |
| `sjf` | 吞吐优先 | 最短作业优先，最小化平均完成时间 |
| `fair` | 公平调度 | 加权 deficit round-robin |
| `stride` | 公平调度 | 确定性比例份额（Xen 风格） |
| `wfq` | 公平调度 | 加权公平排队，延迟隔离 |
| `lru` | 新鲜度优先 | 最近最少使用，防止 adapter 饥饿 |
| `edf` | 新鲜度优先 | 最早截止时间优先，防 staleness 越界 |
| `priority` | 优先级 | 静态优先级 + aging 防饥饿 |
| `adaptive` | 自适应 | 根据负载动态切换策略 |
| `fifo` | 简单 | 先进先出，无状态 |

### 1.4 TrainerScheduler API

```python
class TrainerScheduler:
    def __init__(
        self,
        *,
        adapter_registry: AdapterRegistry,
        data_plane: Optional[TransferQueueDataPlane] = None,
        train_policy: Optional[Any] = None,
        config: Optional[TrainerSchedulerConfig] = None,
        supported_reward_type: Optional[str] = None,
        supported_loss_type: Optional[str] = None,
        supported_algorithm: Optional[str] = None,
    )
    
    def list_train_ready_partitions(self) -> list[PartitionMetadata]
    def next_partition(
        self,
        candidates: Optional[list[PartitionMetadata]] = None,
        current_context: Optional[TrainingContext] = None,
    ) -> Optional[PartitionMetadata]
    def list_eligible_partitions(
        self,
        candidates: list[PartitionMetadata],
        current_context: Optional[TrainingContext] = None,
    ) -> list[PartitionMetadata]
    def get_schedule_decision(
        self,
        candidates: list[PartitionMetadata],
        current_context: Optional[TrainingContext] = None,
    ) -> ScheduleDecision
    def is_compatible(self, partition: PartitionMetadata) -> bool
```

**关键特性**：

- `list_train_ready_partitions()` - 主动查询 TQ，返回 TRAIN_READY 状态的 partition
- `next_partition()` - 支持无参调用（自动查询 TQ）或传入候选列表
- `is_compatible()` - 检查 partition 的 reward_type/loss_type/algorithm 是否匹配 trainer 配置

### 1.5 策略拆分

将调度策略按用途拆分为两个文件：

- `rollout_scheduling.py` - Rollout 策略（`WorkConservingRolloutPolicy`, `DeficitFairRolloutPolicy`）
- `train_scheduling.py` - Train 策略（11 种）
- `scheduling.py` - 向后兼容的重导出层

---

## 2. Upstream 合并与适配

### 2.1 第一次合并

**Commit**: `db7ff31`  
**来源**: https://github.com/meichangsu1/twinkle/tree/multilora-async-rl

**冲突文件**：

- `__init__.py` - 保留 scheduler.py 导入
- `data_plane.py` - 保留 QueueMetadata、claim 方法的 worker_id 参数、`_batch_update_samples`
- `pipeline.py` - 保留 train_schedule_* 配置字段
- `scheduling.py` - 保留 11 种 train 策略
- `workers.py` - 移除 TrainerScheduler（已迁移到 scheduler.py），保留新的 step() API

**Upstream 新增内容**：

- `AsyncRollouter` 批量提交机制（`submit_rollout_tasks()`, `collect_finished_rollout_tasks()`）
- `RolloutTaskState` 状态跟踪
- `max_submit_groups` 参数
- `RolloutCallable` Protocol
- `PromptFeeder` 延迟初始化
- `grpo_pipeline.py` 大量改动（MultiLoraTransformersModel、DataLoader factory）

### 2.2 第二次合并

**Commit**: `3a38464`  
**来源**: 同一 upstream 分支的更新

**无冲突**，自动合并成功。

**Upstream 变更**：

- `cookbook/rl/async_multi_lora_grpo.yaml` - 配置调整
- `src/twinkle/model/multi_lora.py` - MultiLoRA 模型增强
- `src/twinkle_agentic/async_rl/grpo_pipeline.py` - 移除 2 行代码
- `tests/twinkle/test_multi_lora.py` - 新增 21 行测试

---

## 3. DataPlane 适配 Upstream

### 3.1 问题分析

Upstream 大幅简化了 DataPlane，删除了本分支添加的多个特性：

| 被删除的特性 | 说明 |
|-------------|------|
| `QueueMetadata` 类 | `get_metadata()` 改为直接返回 `list[PartitionMetadata]` |
| `TaskName` 枚举 | 改用字符串常量 |
| `close()` | 资源清理方法 |
| `claim_reward_ready_groups()` | 按 group 级别 claim |
| `build_streaming_dataset()` | `_StreamingDatasetWrapper` 整个类 |
| `build_streaming_dataloader()` 参数 | 删除 `task_name`/`required_fields` |
| `ack_rows()` | 消费确认机制 |
| `get_consumed_count()` | 消费计数 |
| `claim_partition_with_lease()` | 租约机制 |
| `release_lease()` / `renew_lease()` | 租约管理 |
| `clear_namespace()` | 批量清理 |
| `_consumed` / `_leases` / `_trainer_steps` | 内部状态 |
| `_batch_update_samples()` | 批量更新（改回逐条 `kv_put`） |
| `_kv_batch_put()` | 批量写入辅助方法 |
| `TransferQueueRuntimeConfig.__post_init__()` | 容量自动计算逻辑 |

### 3.2 适配方案

**原则**：以 upstream 为准，直接替换 DataPlane。

**操作**：

```bash
cp /tmp/upstream_data_plane.py src/twinkle_agentic/async_rl/data_plane.py
```

### 3.3 测试更新

#### 删除的测试文件

- `test_data_plane_new_features.py` - 所有测试都依赖已删除的特性（ack、lease、streaming dataset、close）

#### 重写的测试文件

- `test_developer_a_acceptance.py` - 移除容量计算、租约、流式数据集等测试，保留基础配置和生命周期测试

#### 更新的测试文件

- `test_data_plane_verification.py` - 错误消息从 "belongs to" 改为 "metadata mismatch"
- `test_real_tq.py` - 删除 10 个依赖已删除特性的测试：
  - `test_field_state_flow_with_real_tq` - 依赖 ack_rows
  - `test_lease_claim_mutual_exclusion_with_real_tq` - 依赖租约机制
  - `test_ack_rows_with_real_tq` - 依赖 ack_rows
  - `test_claim_reward_ready_groups_with_real_tq` - 依赖 claim_reward_ready_groups
  - `test_clear_namespace_with_real_tq` - 依赖 clear_namespace
  - `test_close_with_real_tq` - 依赖 close()
  - `test_required_fields_validation_with_real_tq` - 依赖 required_fields 参数
  - `test_queue_metadata_with_real_tq` - 依赖 QueueMetadata
  - `test_metadata_stage_progress_with_real_tq` - 依赖 QueueMetadata
  - `test_capacity_auto_calculation_with_real_tq` - 依赖容量自动计算

### 3.4 简化后的 DataPlane API

```python
class TransferQueueDataPlane:
    def __init__(self, tq_client=None, tq_config=None)
    def init_namespace(self, context: TrainingContext)
    def next_partition_id(self, context: TrainingContext) -> str
    def create_partition(self, context, *, target_groups, partition_id=None) -> PartitionMetadata
    def put_rollout_batch(self, context, partition_id, trajectories, *, ready_groups=1, seal=False) -> PartitionMetadata
    def list_partitions(self, context=None, *, statuses=None) -> list[PartitionMetadata]
    def get_metadata(self, context=None) -> list[PartitionMetadata]
    def check_capacity(self, context: TrainingContext) -> bool
    def claim_reward_batch(self, context, batch_size) -> tuple[PartitionMetadata, list[SampleRecord]]
    def append_rewards(self, context, partition_id, rewards, *, field_name='rewards') -> PartitionMetadata
    def claim_advantage_batch(self, context, batch_size) -> tuple[PartitionMetadata, list[SampleRecord]]
    def append_advantages(self, context, partition_id, advantages, returns=None) -> PartitionMetadata
    def list_train_ready_partitions(self) -> list[PartitionMetadata]
    def mark_training(self, context, partition_id) -> PartitionMetadata
    def mark_trained(self, context, partition_id) -> PartitionMetadata
    def build_streaming_dataloader(self, context, partition_id) -> list[SampleRecord]
    def clear_partition(self, context, partition_id)
```

### 3.5 简化后的 TransferQueueRuntimeConfig

```python
@dataclass
class TransferQueueRuntimeConfig:
    total_storage_size: int | None = None
    max_rows: int | None = None
    max_rows_per_context: int | None = None
    num_data_storage_units: int = 4
    storage_backend: str = 'SimpleStorage'
    controller: dict[str, Any] = field(default_factory=dict)
    backend: dict[str, Any] = field(default_factory=dict)
    init: bool = True
```

**删除的字段**：

- `target_groups`, `num_generations`, `max_staleness` - 容量计算输入
- `estimate_bytes_per_sample`, `safety_factor` - 容量估算
- `max_tq_bytes`, `max_live_partitions_per_context` - 容量保护阈值
- `lease_timeout` - 租约超时

**删除的逻辑**：

- `__post_init__()` 容量自动计算
- `check_capacity()` 中的 `max_live_partitions_per_context` 检查

---

## 4. QueueMetadata 移除

### 4.1 变更内容

**删除**：

- `types.py` 中的 `QueueMetadata` 类定义
- `__init__.py` 中的 `QueueMetadata` 导出

**修改**：

- `data_plane.py` 的 `get_metadata()` 返回类型从 `QueueMetadata | list[PartitionMetadata]` 改为 `list[PartitionMetadata]`
- `get_metadata()` 实现简化为直接调用 `list_partitions(context)`

### 4.2 影响范围

**调用方更新**：

- `workers.py` - `AsyncRollouter.build_rollout_state()` 直接使用 `list[PartitionMetadata]`
- `pipeline.py` - `BaseRLPipeline._is_drained()` 直接迭代 `list[PartitionMetadata]`

**测试更新**：

- 所有依赖 `QueueMetadata` 属性的测试改为直接操作 `list[PartitionMetadata]`
- 例如：`qm.live_partition_count` → `len(partitions)`
- 例如：`qm.total_rows` → `sum(p.num_rows for p in partitions)`
- 例如：`qm.oldest_partition` → `min(partitions, key=lambda p: (p.created_at, p.partition_id))`

---

## 5. is_compatible 实现

### 5.1 方法签名

```python
def is_compatible(self, partition: PartitionMetadata) -> bool:
    """Check if partition's reward_type/loss_type/algorithm matches trainer config."""
```

### 5.2 实现逻辑

```python
def is_compatible(self, partition: PartitionMetadata) -> bool:
    ctx = partition.context
    if self.supported_reward_type is not None and ctx.reward_type != self.supported_reward_type:
        return False
    if self.supported_loss_type is not None and ctx.loss_type != self.supported_loss_type:
        return False
    if self.supported_algorithm is not None and ctx.algorithm != self.supported_algorithm:
        return False
    return True
```

### 5.3 构造函数参数

```python
def __init__(
    self,
    *,
    adapter_registry: AdapterRegistry,
    data_plane: Optional[TransferQueueDataPlane] = None,
    train_policy: Optional[Any] = None,
    config: Optional[TrainerSchedulerConfig] = None,
    supported_reward_type: Optional[str] = None,
    supported_loss_type: Optional[str] = None,
    supported_algorithm: Optional[str] = None,
)
```

### 5.4 使用场景

Gating 第 5 层过滤：

```python
def _apply_gating(self, candidates):
    for partition in candidates:
        # G1: TRAIN_READY 状态检查
        # G2: AdapterRegistry 查询
        # G3: 终态过滤（FAILED/CANCELLED/DRAINING）
        # G4: can_train 检查
        # G5: is_compatible 兼容性检查
        if not self.is_compatible(partition):
            rejected.append(RejectedPartition(
                partition_id=partition.partition_id,
                context_key=partition.context.key,
                reason='incompatible',
            ))
            continue
        eligible.append(partition)
    return eligible, rejected
```

---

## 6. 测试结果

### 6.1 最终测试结果

```
181 passed, 1 skipped, 8 warnings
```

### 6.2 测试分布

| 测试文件 | 测试数 | 说明 |
|---------|--------|------|
| `test_async_rl_core.py` | 17 | 核心功能测试（1 skipped） |
| `test_base_pipeline.py` | 5 | Pipeline 集成测试 |
| `test_data_plane_verification.py` | 39 | DataPlane 接口验证 |
| `test_developer_a_acceptance.py` | 21 | 开发者验收测试（重写） |
| `test_e2e_gsm8k.py` | 5 | 端到端 GSM8K 测试 |
| `test_real_tq.py` | 12 | 真实 TQ 集成测试（删除 10 个） |
| `test_scheduler.py` | 83 | TrainerScheduler 测试（新增） |
| **总计** | **182** | **181 passed, 1 skipped** |

### 6.3 删除的测试

| 测试文件 | 删除数 | 原因 |
|---------|--------|------|
| `test_data_plane_new_features.py` | 29 | 整个文件删除，依赖 ack/lease/streaming/close |
| `test_real_tq.py` | 10 | 依赖已删除的 DataPlane 特性 |
| **总计** | **39** | |

---

## 7. 文件变更清单

### 7.1 新增文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `src/twinkle_agentic/async_rl/scheduler.py` | 245 | TrainerScheduler 核心实现 |
| `src/twinkle_agentic/async_rl/rollout_scheduling.py` | 60 | Rollout 调度策略 |
| `src/twinkle_agentic/async_rl/train_scheduling.py` | 447 | Train 调度策略（11 种） |
| `tests/twinkle_agentic/async_rl/test_scheduler.py` | 958 | TrainerScheduler 测试 |
| `multilora-async-rl/TrainerScheduler_spec.md` | 2004 | 设计文档 |
| `multilora-async-rl/TrainerScheduler_API.md` | 约 500 | API 文档 |
| `multilora-async-rl/TransferQueue-20260630.md` | 本文档 | 开发日志 |

### 7.2 修改文件

| 文件 | 变更 | 说明 |
|------|------|------|
| `src/twinkle_agentic/async_rl/data_plane.py` | 替换 | 用 upstream 版本替换（779 → 366 行） |
| `src/twinkle_agentic/async_rl/workers.py` | 修改 | 移除 TrainerScheduler，从 scheduler.py 导入 |
| `src/twinkle_agentic/async_rl/pipeline.py` | 修改 | 添加 train_schedule_* 配置字段 |
| `src/twinkle_agentic/async_rl/__init__.py` | 修改 | 导出新类，移除 QueueMetadata |
| `src/twinkle_agentic/async_rl/types.py` | 修改 | 移除 QueueMetadata 类 |
| `src/twinkle_agentic/async_rl/scheduling.py` | 重写 | 改为向后兼容的重导出层 |
| `tests/twinkle_agentic/async_rl/test_developer_a_acceptance.py` | 重写 | 移除已删除特性的测试 |
| `tests/twinkle_agentic/async_rl/test_data_plane_verification.py` | 修改 | 更新错误消息匹配 |
| `tests/twinkle_agentic/async_rl/test_real_tq.py` | 修改 | 删除 10 个测试 |

### 7.3 删除文件

| 文件 | 行数 | 原因 |
|------|------|------|
| `tests/twinkle_agentic/async_rl/test_data_plane_new_features.py` | 399 | 所有测试依赖已删除的特性 |

---

## 8. Commit 记录

| Commit | 说明 |
|--------|------|
| `7ce27b7` | feat: implement TrainerScheduler with 11 scheduling policies |
| `db7ff31` | Merge upstream/multilora-async-rl into multilora-async-rl-scheduler |
| `3a38464` | Merge remote-tracking branch 'upstream/multilora-async-rl' into multilora-async-rl-scheduler |
| 未提交 | feat: adapt DataPlane to upstream version |

---

## 9. 关键决策

### 9.1 以 Upstream 为准

**决策**：当 upstream 删除了本分支添加的特性时，直接替换 DataPlane 为 upstream 版本，而不是保留扩展。

**理由**：

- Upstream 是主线，本分支是特性分支
- 保持与 upstream 的一致性，避免后续合并冲突
- Upstream 的简化版本更轻量，删除的特性（ack、lease、streaming dataset）可能在后续版本中以不同方式重新引入

### 9.2 策略拆分

**决策**：将调度策略按用途拆分为 `rollout_scheduling.py` 和 `train_scheduling.py`，保留 `scheduling.py` 作为向后兼容层。

**理由**：

- Rollout 和 Train 策略的接口不同（`pick_next_context` vs `pick_next_partition`）
- 职责分离，便于维护
- `scheduling.py` 重导出保证现有代码不需要修改导入路径

### 9.3 is_compatible 实现

**决策**：在 `TrainerScheduler` 中实现 `is_compatible()` 方法，检查 partition 的 reward_type/loss_type/algorithm 是否匹配 trainer 配置。

**理由**：

- 设计文档要求 Gating 第 5 层检查兼容性
- 支持多租户场景下不同 trainer 处理不同类型的 partition
- 通过构造函数参数配置，灵活且可扩展

---

## 10. 后续工作

### 10.1 待提交

当前工作尚未提交，包含：

- DataPlane 适配到 upstream 版本
- 测试更新（删除 39 个测试，重写 2 个测试文件）
- QueueMetadata 移除

### 10.2 待验证

- 真实 TQ 环境下的集成测试（需要 NPU 环境）
- TrainerScheduler 与 AsyncRollouter 的端到端集成
- 11 种调度策略在不同负载场景下的性能表现

### 10.3 待讨论

- 是否需要重新引入 upstream 删除的特性（ack、lease、streaming dataset）
- 是否需要在 upstream 基础上扩展 DataPlane 功能
- TrainerScheduler 的配置是否需要进一步简化

---

## 11. 总结

本日完成了以下工作：

1. **TrainerScheduler 特性开发**：实现 11 种调度策略，5 层 gating 机制，配置驱动，可观测性
2. **两次 Upstream 合并**：成功合并 upstream 的最新改动，解决冲突
3. **DataPlane 适配**：以 upstream 为准，替换 DataPlane，删除 39 个测试，更新 2 个测试文件
4. **QueueMetadata 移除**：简化 `get_metadata()` 返回类型
5. **is_compatible 实现**：完成 Gating 第 5 层兼容性检查
6. **策略拆分**：按用途拆分为 rollout_scheduling.py 和 train_scheduling.py

**最终状态**：

- 181 个测试通过，1 个 skipped
- DataPlane 与 upstream 保持一致
- TrainerScheduler 功能完整，测试覆盖全面
- 代码已准备好提交
