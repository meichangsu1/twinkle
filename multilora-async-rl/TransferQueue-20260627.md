# TransferQueueDataPlane 开发日志 - 2026-06-27

**分支**: multilora-async-rl-tq  
**环境**: NPU (Ascend 910B3) + TransferQueue 0.1.8 + Ray 2.55.1

---

## 1. 容量计算修复

**问题**：`max_rows` 每次调用 `resolve_max_rows()` 都重新计算，且没有 `self.max_rows` 的赋值。

**修复**：使用 `__post_init__` 在初始化时一次性计算所有容量字段。

**Commit**: `bd1a886`

### 修改前

```python
def resolve_max_rows(self) -> int:
    if self.max_rows is not None:
        return self.max_rows
    return self.compute_max_rows()  # 每次调用都重新计算
```

### 修改后

```python
def __post_init__(self):
    samples_per_partition = self.target_groups * self.num_generations
    max_live_partitions = self.max_staleness + 1

    if self.max_rows is None:
        self.max_rows = samples_per_partition * max_live_partitions

    if self.max_rows_per_context is None:
        self.max_rows_per_context = self.max_rows

    if self.max_tq_bytes is None and self.estimate_bytes_per_sample is not None:
        self.max_tq_bytes = int(self.estimate_bytes_per_sample * self.max_rows * self.safety_factor)

    if self.max_live_partitions_per_context is None:
        self.max_live_partitions_per_context = max_live_partitions
```

### 测试更新

- 删除 `compute_max_rows()`、`resolve_max_rows()` 等方法
- 测试改为直接检查属性值：`config.max_rows == 128 * 8 * (1 + 1)`
- 13 个容量测试全部通过

---

## 2. TaskName 枚举化

**问题**：`task_name` 使用硬编码字符串 `'train'`、`'reward'`、`'advantage'`、`'rollout'`，容易拼写错误。

**修复**：在 `types.py` 中定义 `TaskName` StrEnum，与 `PartitionStatus` 风格一致。

**Commit**: `d1a0461`

### 新增枚举

```python
class TaskName(StrEnum):
    ROLLOUT = 'rollout'
    REWARD = 'reward'
    ADVANTAGE = 'advantage'
    TRAIN = 'train'
```

### 修改范围

| 文件 | 修改内容 |
|------|---------|
| `types.py` | 新增 `TaskName` StrEnum |
| `data_plane.py` | 删除类内常量，所有引用改为 `TaskName.REWARD` 等 |
| `__init__.py` | 导出 `TaskName` |
| `test_data_plane_new_features.py` | 使用 `TaskName.TRAIN` |
| `test_developer_a_acceptance.py` | 使用 `TaskName.TRAIN` |
| `test_real_tq.py` | 使用 `TaskName.TRAIN` |

---

## 3. build_streaming_dataset() 封装 TQ 原生流式

**问题**：设计文档要求 `StreamingDataset / StreamingDataLoader`，但代码中未实现。TQ 原生提供这两个类，需要在 DataPlane 中封装调用。

**修复**：新增 `build_streaming_dataset()` 方法和 `_StreamingDatasetWrapper` 类。

**Commit**: `8d83f86`

### 新增 API

```python
def build_streaming_dataset(
    self,
    context: TrainingContext,
    partition_id: str,
    *,
    batch_size: int = 32,
    data_fields: Optional[List[str]] = None,
    task_name: Optional[str] = None,
    dp_rank: int = 0,
) -> '_StreamingDatasetWrapper'
```

### _StreamingDatasetWrapper 实现

- **Client API 路径**：使用 `client.get_meta()` + `client.get_data()` 流式读取，自动 ack
- **KV API 降级路径**：Client API 不可用时，使用 `_get_samples()` 全量读取后按 batch 切分
- 每次迭代自动调用 `ack_rows()` 确认消费
- `total_acked` 属性跟踪已确认样本数

### 新增测试（3 个）

| 测试 | 验证内容 |
|------|---------|
| `test_streaming_dataset_yields_batches` | 10 个样本按 batch_size=3 切分为 4 个 batch |
| `test_streaming_dataset_auto_acks` | 读取后自动 ack，consumed_count 正确 |
| `test_streaming_dataset_rejects_cross_context` | 跨 context 访问抛 ValueError |

---

## 4. append_rewards / append_advantages 跨 context 隔离

**问题**：`append_rewards` 和 `append_advantages` 没有检查 partition 是否属于当前 context，可能导致跨任务数据污染。

**修复**：在两个方法开头添加 `meta.context.key != context.key` 检查。

**Commit**: `4519b0c`

### 修改内容

```python
# append_rewards (line 294-296)
meta = self._meta.get(partition_id)
if meta is not None and meta.context.key != context.key:
    raise ValueError(f'partition {partition_id} belongs to {meta.context.key}, not {context.key}')

# append_advantages (同样逻辑)
```

### 新增测试（3 个）

| 测试 | 验证内容 |
|------|---------|
| `test_append_rewards_rejects_cross_context` | 用错误 context 调 append_rewards 抛 ValueError |
| `test_append_advantages_rejects_cross_context` | 用错误 context 调 append_advantages 抛 ValueError |
| `test_cross_context_data_isolation` | 端到端：两个 context 各自创建 partition，验证错误 context 无法修改对方数据 |

---

## 5. 多 Worker 防重复 Claim 验证

**验收标准**：同一个 sample 不会被多个 reward / advantage worker 重复 claim。

### 验证结果：6/6 通过

| 测试 | 场景 | 结果 |
|------|------|------|
| `test_two_reward_workers_cannot_claim_same_partition` | worker_1 claim 后 worker_2 尝试 claim | ✅ RuntimeError |
| `test_two_advantage_workers_cannot_claim_same_partition` | adv_worker_1 claim 后 adv_worker_2 尝试 claim | ✅ RuntimeError |
| `test_worker_can_claim_after_lease_released` | worker_1 release 后 worker_2 可以 claim | ✅ 成功 claim |
| `test_worker_can_claim_after_lease_expires` | worker_1 租约过期后 worker_2 可以 claim | ✅ 成功 claim |
| `test_different_partitions_can_be_claimed_by_different_workers` | 不同 partition 可以被不同 worker 同时 claim | ✅ 并发正常 |
| `test_full_lifecycle_no_duplicate_claim` | 完整生命周期 rollout→reward→advantage→train→clear | ✅ 无重复 |

### 防重复机制

```text
claim_reward_batch(ctx, batch_size, worker_id='rw1')
  → _claim_samples()
    → list_partitions(ctx, statuses=[ROLLOUT_DONE])  # 按 context 过滤
    → claim_partition_with_lease(ctx, partition_id, worker_id='rw1')
      → _recover_expired_leases()                     # 清理过期租约
      → 检查 _leases[partition_id]                    # 是否已被其他 worker 持有
      → 设置 _leases[partition_id] = {worker_id, deadline}

claim_reward_batch(ctx, batch_size, worker_id='rw2')  # 第二个 worker
  → claim_partition_with_lease()
    → 发现 _leases[partition_id] 已被 rw1 持有
    → raise RuntimeError("partition is leased by rw1")
```

---

## 6. 字段状态流转验证（真实 TQ）

**问题**：需要验证通过 DataPlane 操作真实 TQ 时，字段在 `rollout -> reward -> advantage -> train -> clear` 各阶段的状态流转是否正确。

**修复**：新增 `test_field_state_flow_with_real_tq` 测试，逐阶段验证字段状态。

**Commit**: `1f2cc9d`

### 验证场景

```python
# Stage 1: Rollout - 写入 messages, group_id, generation_idx, old_logps
sample = {
    'sample_id': 'sample_0',
    'messages': [{'role': 'user', 'content': 'What is 2+2?'}],
    'group_id': 'group_0',
    'generation_idx': 0,
    'old_logps': [0.1, 0.2, 0.3],
}
meta = real_dp.put_rollout_batch(ctx, partition.partition_id, [sample], seal=True)
assert meta.status == PartitionStatus.ROLLOUT_DONE

# 验证：只有 rollout 字段，没有 rewards/advantages/returns
samples = real_dp.build_streaming_dataloader(ctx, partition.partition_id)
assert 'messages' in samples[0]
assert 'group_id' in samples[0]
assert 'old_logps' in samples[0]
assert 'rewards' not in samples[0]
assert 'advantages' not in samples[0]
assert 'returns' not in samples[0]

# Stage 2: Reward - 追加 rewards 字段
meta = real_dp.append_rewards(ctx, partition.partition_id, [1.5])
assert meta.status == PartitionStatus.REWARD_DONE

# 验证：rewards 字段已添加，值正确
samples = real_dp.build_streaming_dataloader(ctx, partition.partition_id)
assert 'rewards' in samples[0]
assert samples[0]['rewards'] == 1.5
assert 'advantages' not in samples[0]
assert 'returns' not in samples[0]

# Stage 3: Advantage - 追加 advantages 和 returns 字段
meta = real_dp.append_advantages(ctx, partition.partition_id, [0.8], returns=[1.2])
assert meta.status == PartitionStatus.TRAIN_READY

# 验证：advantages 和 returns 字段已添加，值正确
samples = real_dp.build_streaming_dataloader(ctx, partition.partition_id)
assert 'advantages' in samples[0]
assert samples[0]['advantages'] == 0.8
assert 'returns' in samples[0]
assert samples[0]['returns'] == 1.2

# Stage 4: Train - 标记训练中，读取数据，ack，标记训练完成
meta = real_dp.mark_training(ctx, partition.partition_id)
assert meta.status == PartitionStatus.TRAINING

# 验证：所有字段都存在
samples = real_dp.build_streaming_dataloader(ctx, partition.partition_id)
assert 'messages' in samples[0]
assert 'rewards' in samples[0]
assert 'advantages' in samples[0]
assert 'returns' in samples[0]

# ack 并验证消费计数
acked = real_dp.ack_rows(ctx, partition.partition_id, ['sample_0'], task_name=TaskName.TRAIN)
assert acked == 1
assert real_dp.get_consumed_count(partition.partition_id, task_name=TaskName.TRAIN) == 1

meta = real_dp.mark_trained(ctx, partition.partition_id)
assert meta.status == PartitionStatus.TRAIN_DONE

# Stage 5: Clear - 验证数据从 TQ 中删除
real_dp.clear_partition(ctx, partition.partition_id)
partitions = real_dp.list_partitions(ctx)
assert partitions[0].status == PartitionStatus.CLEARED

# 验证 TQ 数据已清除
tq_data = real_dp.tq.kv_list(partition_id=partition.partition_id)
assert len(tq_data.get(partition.partition_id, {})) == 0
```

### 验证结果

✅ **所有阶段字段状态正确**
- Rollout 阶段：只有 rollout 字段
- Reward 阶段：rewards 字段正确追加
- Advantage 阶段：advantages 和 returns 字段正确追加
- Train 阶段：所有字段可读，ack 机制工作正常
- Clear 阶段：数据从 TQ 中完全删除

---

## 8. Metadata 阶段进度验证（真实 TQ）

**问题**：需要验证 metadata 是否能正确反映 live partitions、最老 partition、各阶段完成进度。

**修复**：新增 `test_metadata_stage_progress_with_real_tq` 测试，验证 metadata 在多阶段并行场景下的正确性。

**Commit**: `8b8649d`

### 验证场景

```python
# 创建 4 个 partition，分别处于不同阶段
p1: ROLLOUT_DONE
p2: REWARD_DONE
p3: TRAIN_READY
p4: TRAINING

# 验证 metadata 正确反映各阶段状态
qm = real_dp.get_metadata(ctx)
assert qm.live_partition_count == 4
assert qm.total_rows == 4
assert qm.oldest_partition.partition_id == p1.partition_id

# 统计各阶段 partition 数量
partitions = list(qm)
status_counts = {}
for p in partitions:
    status_counts[p.status] = status_counts.get(p.status, 0) + 1

assert status_counts[PartitionStatus.ROLLOUT_DONE] == 1
assert status_counts[PartitionStatus.REWARD_DONE] == 1
assert status_counts[PartitionStatus.TRAIN_READY] == 1
assert status_counts[PartitionStatus.TRAINING] == 1

# p4 完成训练，验证状态变化
real_dp.mark_trained(ctx, p4.partition_id)
qm = real_dp.get_metadata(ctx)
status_counts = count_statuses(qm)
assert status_counts.get(PartitionStatus.TRAINING, 0) == 0
assert status_counts[PartitionStatus.TRAIN_DONE] == 1

# 清除 p4，验证 live count 减少
real_dp.clear_partition(ctx, p4.partition_id)
qm = real_dp.get_metadata(ctx)
assert qm.live_partition_count == 3
assert qm.total_rows == 3  # p4 数据已删除

# 验证 oldest 仍然是 p1
assert qm.oldest_partition.partition_id == p1.partition_id

# 清除 p1，验证 oldest 变为 p2
real_dp.clear_partition(ctx, p1.partition_id)
qm = real_dp.get_metadata(ctx)
assert qm.live_partition_count == 2
assert qm.oldest_partition.partition_id == p2.partition_id
```

### 验证结果

✅ **Metadata 正确反映各阶段进度**
- live_partition_count：正确统计活跃 partition 数量
- total_rows：正确统计总行数，清除后自动减少
- oldest_partition：正确识别最老 partition，清除后自动更新
- 各阶段状态计数：正确统计 ROLLOUT_DONE、REWARD_DONE、TRAIN_READY、TRAINING、TRAIN_DONE 各阶段数量

---

## 9. 全量测试结果

```
======================= 156 passed, 1 skipped in 67.83s ========================
```

| 测试文件 | 测试数 | 状态 |
|---------|--------|------|
| test_async_rl_core.py | 16 | ✅ (1 skipped) |
| test_base_pipeline.py | 5 | ✅ |
| test_data_plane_verification.py | 39 | ✅ |
| test_data_plane_new_features.py | 28 | ✅ |
| test_developer_a_acceptance.py | 42 | ✅ |
| test_e2e_gsm8k.py | 5 | ✅ |
| test_real_tq.py | 22 | ✅ |
| **总计** | **134** | **134 passed, 1 skipped** |

---

## 10. Commit 记录

| Commit | 说明 |
|--------|------|
| `bd1a886` | fix: compute capacity fields once in __post_init__ |
| `d1a0461` | refactor: use StrEnum for TaskName |
| `8d83f86` | feat: add build_streaming_dataset() wrapping TQ native streaming |
| `4519b0c` | fix: add context.key validation to append_rewards and append_advantages |
| `1f2cc9d` | test: add field state flow verification with real TQ |
| `837fa66` | docs: update daily log with field state flow verification |
| `8b8649d` | test: add metadata stage progress verification with real TQ |
