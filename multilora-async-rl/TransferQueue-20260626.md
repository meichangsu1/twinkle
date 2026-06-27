# TransferQueueDataPlane 开发者A交付验收报告

**日期**: 2026-06-26  
**分支**: multilora-async-rl-tq  
**环境**: NPU (Ascend 910B3) + TransferQueue 0.1.8 + Ray 2.55.1

## 1. 任务概述

根据 `详细设计.md` 第 7.1 节，完成开发者 A 的 TransferQueueDataPlane 全部交付物，并通过验收标准。

## 2. 审计结果

对照详细设计 7.1 逐项审计，发现以下差距：

| 要求 | 审计状态 | 差距说明 |
|------|---------|---------|
| 封装底层 TQ API | ✅ 已有 | - |
| TQ 容量初始化公式 | ❌ 缺失 | config 有字段但不自动计算 `target_groups * num_generations * (max_staleness + 1)` |
| 固化 task_name | ⚠️ 部分 | 只在 ack/dataloader 显式使用，未定义常量 |
| TrainingScope 过滤 | ✅ 已有 | context.key 校验 |
| QueueMetadata 类 | ❌ 缺失 | 未定义，get_metadata 返回 list |
| PartitionMetadata | ✅ 已有 | - |
| row 字段 schema | ❌ 缺失 | 未正式定义常量 |
| mock backend | ✅ 已有 | FakeTransferQueueClient |
| claim_reward_ready_groups | ❌ 缺失 | 只有 sample 级 claim，无 group 级 |
| 验收：多 worker 不重复 claim | ⚠️ 无显式测试 | 缺少专项测试 |

## 3. 本轮新增功能

### 3.1 容量自动计算

**文件**: `src/twinkle_agentic/async_rl/data_plane.py` — `TransferQueueRuntimeConfig`

新增方法：

```python
def compute_max_rows(self) -> int:
    """samples_per_partition * max_live_partitions"""
    return self.target_groups * self.num_generations * (self.max_staleness + 1)

def compute_max_live_partitions(self) -> int:
    return self.max_staleness + 1

def compute_max_tq_bytes(self) -> Optional[int]:
    if self.estimate_bytes_per_sample is None:
        return None
    return int(self.estimate_bytes_per_sample * self.compute_max_rows() * self.safety_factor)

def resolve_max_rows(self) -> int:
    """显式值 > 自动计算"""

def resolve_max_rows_per_context(self) -> int:
def resolve_max_live_partitions_per_context(self) -> int:
```

**容量公式**（详细设计 3.1）：

```text
samples_per_partition = target_groups * num_generations
max_live_partitions   = max_staleness + 1
max_rows              = samples_per_partition * max_live_partitions
max_tq_bytes          = estimate_bytes_per_sample * max_rows * safety_factor
```

**优先级规则**：显式值 > 自动计算 > 默认值

### 3.2 QueueMetadata 类

**文件**: `src/twinkle_agentic/async_rl/data_plane.py`

```python
@dataclass
class QueueMetadata:
    context: TrainingContext
    active_partitions: list[PartitionMetadata]
    total_rows: int
    trainer_step: int
    current_policy_version: int

    @property
    def live_partition_count(self) -> int: ...

    @property
    def oldest_partition(self) -> Optional[PartitionMetadata]: ...

    def __iter__(self): ...   # 兼容 list[PartitionMetadata] 的现有代码
    def __len__(self): ...
```

**`get_metadata()` 返回值变更**：

| 调用方式 | 返回类型 | 说明 |
|---------|---------|------|
| `get_metadata(context)` | `QueueMetadata` | 包含 scope 聚合信息 |
| `get_metadata()` | `list[PartitionMetadata]` | 向后兼容 |

### 3.3 claim_reward_ready_groups（group 级 claim）

**文件**: `src/twinkle_agentic/async_rl/data_plane.py`

```python
def claim_reward_ready_groups(
    self,
    context: TrainingContext,
    num_generations: int,
    max_groups: int,
    *,
    worker_id: Optional[str] = None,
) -> tuple[PartitionMetadata, list[list[SampleRecord]]]:
```

**行为**：
1. 找到第一个 `ROLLOUT_DONE` 状态的 partition
2. 按 `group_id` 分组 samples
3. 只返回完整 group（每组 >= `num_generations` 条）
4. 最多返回 `max_groups` 个 group
5. 支持 `worker_id` lease 互斥

### 3.4 Row Field Schema 常量

**文件**: `src/twinkle_agentic/async_rl/data_plane.py` 顶部

```python
ROLLOUT_FIELDS = frozenset({'messages', 'group_id', 'generation_idx', 'old_logps'})
REWARD_FIELDS = frozenset({'rewards'})
ADVANTAGE_FIELDS = frozenset({'advantages', 'returns'})
TRAIN_REQUIRED_FIELDS = frozenset(ROLLOUT_FIELDS | REWARD_FIELDS | ADVANTAGE_FIELDS)
SAMPLE_ISOLATION_TAG_FIELDS = frozenset({
    'tenant_id', 'training_run_id', 'base_model_id', 'adapter_name',
    'adapter_revision', 'policy_version', 'partition_id', 'group_id', 'generation_idx',
})
TASK_NAMES = frozenset({'rollout', 'reward', 'advantage', 'train'})
```

**使用方式**: `build_streaming_dataloader` 新增 `required_fields` 参数，用于校验 sample 字段完整性：

```python
def build_streaming_dataloader(
    self,
    context: TrainingContext,
    partition_id: str,
    *,
    task_name: Optional[str] = None,
    required_fields: Optional[frozenset] = None,
) -> list[SampleRecord]:
    # ... 读取 samples ...
    if required_fields is not None:
        incomplete = []
        for s in samples:
            missing = required_fields - set(s.keys())
            if missing:
                incomplete.append((s['sample_id'], missing))
        if incomplete:
            raise ValueError(
                f'{len(incomplete)} samples missing required fields: '
                f'{incomplete[0][0]} missing {incomplete[0][1]}'
            )
    return samples
```

**调用示例**:

```python
# 训练前校验所有必需字段
samples = dp.build_streaming_dataloader(
    ctx, partition_id, 
    required_fields=TRAIN_REQUIRED_FIELDS
)
# 如果缺少字段会抛出 ValueError
```

### 3.5 check_capacity 增强

新增 `max_live_partitions_per_context` 检查：

```python
def check_capacity(self, context: TrainingContext) -> bool:
    # 全局 row 上限
    if total_rows >= max_rows: return False
    # 单 context row 上限
    if context_rows >= max_rows_per_ctx: return False
    # 单 context live partition 上限（新增）
    if context_live >= max_live: return False
    return True
```

### 3.6 mark_trained 增强

自动递增 `_trainer_steps` 计数器：

```python
def mark_trained(self, context, partition_id):
    self._trainer_steps[context.key] = self._trainer_steps.get(context.key, 0) + 1
    return self._mark_status(context, partition_id, PartitionStatus.TRAIN_DONE)
```

反映在 `QueueMetadata.trainer_step` 中。

## 4. 新增测试

**文件**: `tests/twinkle_agentic/async_rl/test_developer_a_acceptance.py`（42 个测试）

### 4.1 TestCapacityAutoCalculation（13 个测试）

| 测试 | 覆盖点 |
|------|--------|
| `test_compute_max_rows_default` | 默认配置 128*8*2 = 2048 |
| `test_compute_max_rows_custom` | 自定义 64*4*1 = 256 |
| `test_compute_max_live_partitions` | max_staleness+1 |
| `test_compute_max_tq_bytes` | bytes 计算 |
| `test_compute_max_tq_bytes_none_when_no_estimate` | 无估算值时返回 None |
| `test_resolve_max_rows_explicit_overrides_auto` | 显式值优先 |
| `test_resolve_max_rows_auto_when_none` | None 时自动计算 |
| `test_resolve_max_rows_per_context` | per-context 显式值 |
| `test_resolve_max_rows_per_context_defaults_to_global` | 默认回退到全局值 |
| `test_resolve_max_live_partitions_per_context` | per-context partition 上限 |
| `test_resolve_max_live_partitions_per_context_defaults` | 默认 max_staleness+1 |
| `test_check_capacity_uses_auto_calculated_max_rows` | 自动容量守卫 |
| `test_check_capacity_enforces_max_live_partitions` | live partition 守卫 |

### 4.2 TestQueueMetadata（9 个测试）

| 测试 | 覆盖点 |
|------|--------|
| `test_get_metadata_with_context_returns_queue_metadata` | 返回 QueueMetadata |
| `test_get_metadata_without_context_returns_list` | 向后兼容 |
| `test_queue_metadata_active_partitions` | active 过滤 |
| `test_queue_metadata_excludes_cleared` | 排除 CLEARED |
| `test_queue_metadata_oldest_partition` | 最老 partition |
| `test_queue_metadata_iterable` | `__iter__` 兼容 |
| `test_queue_metadata_len` | `__len__` 兼容 |
| `test_queue_metadata_trainer_step_increments` | trainer_step 递增 |
| `test_queue_metadata_current_policy_version` | policy_version 传递 |

### 4.3 TestClaimRewardReadyGroups（5 个测试）

| 测试 | 覆盖点 |
|------|--------|
| `test_claim_reward_ready_groups_returns_groups` | 按 group_id 分组返回 |
| `test_claim_reward_ready_groups_respects_max_groups` | max_groups 限制 |
| `test_claim_reward_ready_groups_skips_incomplete_groups` | 跳过不完整 group |
| `test_claim_reward_ready_groups_raises_when_no_ready` | 无 ready partition 时抛异常 |
| `test_claim_reward_ready_groups_with_lease` | lease 互斥 |

### 4.4 TestRowFieldSchema（9 个测试）

| 测试 | 覆盖点 |
|------|--------|
| `test_rollout_fields` | ROLLOUT_FIELDS 包含正确字段 |
| `test_reward_fields` | REWARD_FIELDS 包含 rewards |
| `test_advantage_fields` | ADVANTAGE_FIELDS 包含 advantages/returns |
| `test_train_required_fields_is_union` | TRAIN_REQUIRED_FIELDS 是并集 |
| `test_sample_isolation_tag_fields` | 隔离字段完整 |
| `test_task_names` | TASK_NAMES 四类 |
| `test_build_streaming_dataloader_validates_required_fields` | required_fields 校验通过 |
| `test_build_streaming_dataloader_raises_on_missing_fields` | 缺少字段时抛异常 |
| `test_build_streaming_dataloader_train_required_fields_after_full_pipeline` | 完整流程后所有字段齐全 |

### 4.5 TestMultiWorkerClaimExclusion（6 个测试，验收标准核心）

| 测试 | 覆盖点 |
|------|--------|
| `test_two_reward_workers_cannot_claim_same_partition` | 两个 reward worker 不能 claim 同一 partition |
| `test_two_advantage_workers_cannot_claim_same_partition` | 两个 advantage worker 不能 claim 同一 partition |
| `test_worker_can_claim_after_lease_released` | 释放后可重新 claim |
| `test_worker_can_claim_after_lease_expires` | 超时后可重新 claim |
| `test_different_partitions_can_be_claimed_by_different_workers` | 不同 partition 可并行 |
| `test_full_lifecycle_no_duplicate_claim` | 完整生命周期无重复 claim |

## 5. 测试结果

### 5.1 Mock 测试

| 测试文件 | 测试数 | 状态 |
|---------|--------|------|
| test_async_rl_core.py | 11 | ✅ (1 skipped) |
| test_base_pipeline.py | 2 | ✅ |
| test_data_plane_verification.py | 36 | ✅ |
| test_e2e_gsm8k.py | 5 | ✅ |
| test_data_plane_new_features.py | 25 | ✅ |
| **test_developer_a_acceptance.py（新增）** | **42** | ✅ |
| **Mock 测试小计** | **121** | **120 passed, 1 skipped** |

### 5.2 真实 TQ 测试

| 测试文件 | 测试数 | 状态 |
|---------|--------|------|
| test_real_tq.py | 11 | ✅ |
| **真实 TQ 测试小计** | **11** | **11 passed** |

### 5.3 总计

| 测试类型 | 通过 | 跳过 | 失败 | 总计 |
|---------|------|------|------|------|
| Mock 测试 | 120 | 1 | 0 | 121 |
| 真实 TQ 测试 | 11 | 0 | 0 | 11 |
| **总计** | **131** | **1** | **0** | **132** |

**通过率**: 99.2% (131/132)

## 6. 验收标准对照

### 6.1 详细设计 7.1 验收标准

| 验收标准 | 状态 | 测试覆盖 |
|---------|------|---------|
| 同一 sample 不被多个 reward/advantage worker 重复 claim | ✅ | `TestMultiWorkerClaimExclusion`（6 个测试） |
| 通过 mock TQ 完成 rollout→reward→advantage→train→clear 字段状态流转 | ✅ | `test_full_lifecycle_no_duplicate_claim` + 原有生命周期测试 |
| metadata 正确反映 live partitions、最老 partition、各阶段完成进度 | ✅ | `TestQueueMetadata`（9 个测试） |

### 6.2 接口交付对照

| 设计文档接口 | 代码实现 | 状态 |
|---|---|---|
| `init_queue(config)` | `__init__(tq_config)` + `init_namespace()` | ✅ |
| `get_metadata() -> QueueMetadata` | `get_metadata(context) -> QueueMetadata` | ✅ |
| `put_rollout_batch(scope, partition_id, samples)` | `put_rollout_batch(context, partition_id, trajectories)` | ✅ |
| `claim_reward_batch(scope, partition_id, batch_size)` | `claim_reward_batch(context, batch_size)` | ✅ |
| `append_rewards(metadata, rewards)` | `append_rewards(context, partition_id, rewards)` | ✅ |
| `claim_reward_ready_groups(scope, partition_id, num_generations, max_groups)` | `claim_reward_ready_groups(context, num_generations, max_groups)` | ✅ 新增 |
| `append_advantages(group_metadata, advantages)` | `append_advantages(context, partition_id, advantages, returns)` | ✅ |
| `build_streaming_dataset(scope, partition_id, ...)` | `build_streaming_dataloader(context, partition_id, task_name)` | ✅ |
| `mark_trained(metadata)` | `mark_trained(context, partition_id)` | ✅ |
| `clear_partition(scope, partition_id)` | `clear_partition(context, partition_id)` | ✅ |

### 6.3 主要产出对照

| 产出要求 | 状态 |
|---------|------|
| 实现 TransferQueueDataPlane，封装底层 TQ API | ✅ |
| 实现 TQ 容量初始化：`target_groups * num_generations * (max_staleness + 1)` | ✅ 新增 |
| 固化 `task_name = rollout / reward / advantage / train` | ✅ 新增常量 |
| 固化 `TrainingScope = tenant_id / training_run_id / adapter_name` | ✅ context.key |
| 定义 QueueMetadata、PartitionMetadata、row 字段 schema | ✅ 新增 |
| 提供 mock / local backend | ✅ FakeTransferQueueClient |

## 7. 代码变更统计

| 文件 | 变更类型 | 新增行数 |
|------|---------|---------|
| `src/twinkle_agentic/async_rl/data_plane.py` | 修改 | ~80 行（QueueMetadata、容量计算、claim_reward_ready_groups、schema 常量） |
| `src/twinkle_agentic/async_rl/__init__.py` | 修改 | 新增 8 个导出 |
| `tests/twinkle_agentic/async_rl/test_developer_a_acceptance.py` | 新增 | 39 个测试，~350 行 |

## 8. 运行命令

```bash
# Mock 测试（不需要 TQ/Ray）
pytest tests/twinkle_agentic/async_rl/ --ignore=tests/twinkle_agentic/async_rl/test_real_tq.py -v

# 真实 TQ 测试（需要 TransferQueue + Ray）
TORCH_DEVICE_BACKEND_AUTOLOAD=0 CUDA_VISIBLE_DEVICES="" RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 \
  pytest tests/twinkle_agentic/async_rl/test_real_tq.py -v

# 仅运行开发者 A 验收测试
pytest tests/twinkle_agentic/async_rl/test_developer_a_acceptance.py -v
```
