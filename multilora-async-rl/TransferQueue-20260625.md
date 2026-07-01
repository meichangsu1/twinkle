# TransferQueue DataPlane 验证实验报告

**日期**: 2026-06-25  
**分支**: multilora-async-rl-tq  
**环境**: NPU (Ascend 910B3) + TransferQueue 0.1.8 + Ray 2.55.1

## 1. 实验目标

验证 `TransferQueueDataPlane` 实现的正确性，包括：
1. TQ 接口调用正确性
2. 代码语法和类型正确性
3. 与设计文档的一致性
4. 端到端集成测试（mock + 真实 TQ）

## 2. 环境配置

### 2.1 硬件环境
- **NPU**: 8x Ascend 910B3
- **CANN**: 9.0.0
- **架构**: aarch64

### 2.2 软件环境
- **Python**: 3.11.15
- **PyTorch**: 2.10.0+cpu
- **torch_npu**: 2.10.0rc2
- **Ray**: 2.55.1
- **TransferQueue**: 0.1.8
- **tensordict**: 0.13.0

### 2.3 关键环境变量
```bash
CUDA_VISIBLE_DEVICES=""
RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
```

## 3. 测试结果汇总

### 3.1 总体结果

#### Mock 测试（使用 FakeTransferQueueClient）

| 测试文件 | 通过 | 跳过 | 失败 | 总计 |
|---------|------|------|------|------|
| test_async_rl_core.py | 10 | 1 | 0 | 11 |
| test_base_pipeline.py | 2 | 0 | 0 | 2 |
| test_data_plane_verification.py | 36 | 0 | 0 | 36 |
| test_e2e_gsm8k.py | 5 | 0 | 0 | 5 |
| **Mock 测试小计** | **53** | **1** | **0** | **54** |

#### 真实 TQ 测试（使用 TransferQueue 0.1.8 + Ray 2.55.1）

| 测试文件 | 通过 | 跳过 | 失败 | 总计 |
|---------|------|------|------|------|
| test_real_tq.py | 11 | 0 | 0 | 11 |
| **真实 TQ 测试小计** | **11** | **0** | **0** | **11** |

#### 总计

| 测试类型 | 通过 | 跳过 | 失败 | 总计 |
|---------|------|------|------|------|
| Mock 测试 | 53 | 1 | 0 | 54 |
| 真实 TQ 测试 | 11 | 0 | 0 | 11 |
| **总计** | **64** | **1** | **0** | **65** |

**通过率**: 98.5% (64/65)，1 个测试因 TQ 已安装而被跳过（预期行为）

**测试环境差异**：
- **Mock 测试**：使用 `FakeTransferQueueClient` 内存实现，无需 Ray/TQ 依赖，运行速度快
- **真实 TQ 测试**：使用真实 TransferQueue SimpleStorage 后端，需要 Ray 初始化，验证真实环境行为

### 3.2 Mock 测试详情

使用 `FakeTransferQueueClient` 和 `RecordingFakeTQ` 进行测试，无需真实 TransferQueue 依赖。

#### 3.2.1 原有核心测试 (11 passed)
- `test_training_context_namespace_and_metadata_validation`
- `test_data_plane_rollout_reward_advantage_and_clear`
- `test_data_plane_rejects_cross_context_append`
- `test_data_plane_check_capacity_by_row_limits`
- `test_adapter_registry_blocks_current_adapter_during_sync_only`
- `test_staleness_capacity_by_live_partitions`
- `test_work_conserving_rollout_policy_prefers_less_live_work`
- `test_deficit_fair_rollout_policy_alternates_candidates`
- `test_prefer_current_train_policy_keeps_current_then_switches`
- `test_async_rollouter_and_trainer_worker_mvp_flow`
- `test_default_data_plane_requires_real_transfer_queue_when_not_installed` (skipped - TQ 已安装)

#### 3.2.2 Pipeline 集成测试 (2 passed)
- `test_base_pipeline_runs_one_multilora_grpo_partition`
- `test_base_pipeline_uses_latest_adapter_revision_for_next_rollout`

#### 3.2.3 TQ 接口正确性验证 (8 passed)
- `test_kv_put_parameter_types_on_rollout_write`
- `test_kv_put_tag_contains_context_metadata`
- `test_kv_batch_get_parameter_types`
- `test_kv_list_parameter_types`
- `test_kv_clear_parameter_types`
- `test_kv_put_tag_only_update_does_not_pass_fields`
- `test_kv_put_on_append_rewards_writes_correct_fields`
- `test_kv_put_on_append_advantages_writes_correct_fields`

#### 3.2.4 设计文档一致性验证 (27 passed)
- Namespace 格式验证
- 多租户隔离验证
- 状态机转换验证（OPEN → ROLLOUT_DONE → REWARD_DONE → TRAIN_READY → TRAINING → TRAIN_DONE → CLEARED）
- 容量守卫验证（max_rows, max_rows_per_context）
- Metadata 校验验证
- GRPO group 隔离验证
- Auto-seal 机制验证
- 多 partition 并发验证
- Reward/Advantage 计数校验
- Streaming dataloader 隔离验证
- Claim/Append 工作流验证

#### 3.2.5 E2E GSM8K 集成测试 (5 passed)
使用 mock model 和 rollout，验证完整 pipeline 生命周期：
- `test_full_pipeline_runs_multiple_partitions`
- `test_pipeline_context_metadata_matches_gsm8k_config`
- `test_pipeline_weight_sync_increments_policy_version`
- `test_pipeline_partition_lifecycle_is_complete`
- `test_pipeline_no_residual_tq_data_after_clear`

### 3.3 真实 TQ 测试详情

使用真实 TransferQueue 后端（SimpleStorage），在 NPU 环境运行。

#### 3.3.1 测试列表 (11 passed)
1. `test_init_namespace` - 命名空间初始化
2. `test_create_partition` - Partition 创建
3. `test_put_rollout_batch_and_read_back` - 写入和读回验证
4. `test_full_lifecycle_with_real_tq` - 完整生命周期（7 个状态转换）
5. `test_reward_and_advantage_workers_with_real_tq` - Worker 集成
6. `test_multi_partition_with_real_tq` - 多 partition 管理
7. `test_multi_context_isolation_with_real_tq` - 多租户隔离
8. `test_check_capacity_with_real_tq` - 容量检查（使用 max_rows_per_context）
9. `test_staleness_manager_with_real_tq` - Staleness 管理
10. `test_clear_partition_removes_tq_data` - 数据清理验证
11. `test_cross_context_rejected_with_real_tq` - 跨 context 拒绝

#### 3.3.2 关键技术发现

**TensorDict 兼容性**
- 真实 TQ 的 `kv_batch_get` 返回 `tensordict.TensorDict` 而非 `dict`
- `_rows_from_tq_data` 已正确处理：`hasattr(data, 'to_dict')` 分支
- 验证通过：数据正确转换为 list of dicts

**Ray 初始化**
- NPU 环境需要 `num_gpus=0` 避免 CUDA 检测
- 使用 session-scoped fixture 统一管理 Ray 生命周期
- 初始化耗时约 1-2 秒

**跨测试状态隔离**
- 真实 TQ 共享全局状态，`kv_list()` 返回所有 partition
- 解决方案：每个测试使用唯一 context（tenant/run/adapter 名不同）
- 容量检查改用 `max_rows_per_context` 避免全局 `max_rows` 误判

## 4. 问题与解决方案

### 4.1 Ray 初始化超时（已解决）
**问题**: 初始尝试使用 `num_gpus=None` 导致 Ray 尝试 CUDA 路径，30 秒后 SIGTERM

**解决方案**:
```python
ray.init(
    namespace='TQDataPlaneTest',
    ignore_reinit_error=True,
    num_gpus=0,  # NPU 环境不暴露 CUDA
    include_dashboard=False,
)
```

### 4.2 跨测试 Partition 残留（已解决）
**问题**: 真实 TQ 共享状态，前面测试的 partition 影响后续 `check_capacity` 和 `staleness_manager`

**解决方案**:
```python
# 每个测试使用唯一 context
ctx = make_context('cap_lora', tenant='cap_tenant', run='cap_run')
ctx = make_context('stale_lora', tenant='stale_tenant', run='stale_run')
```

### 4.3 TQ 已安装时跳过测试（已解决）
**问题**: `test_default_data_plane_requires_real_transfer_queue_when_not_installed` 预期 TQ 未安装时抛错

**解决方案**:
```python
@pytest.mark.skipif(
    __import__('importlib').util.find_spec('transfer_queue') is not None,
    reason='transfer_queue is installed',
)
def test_default_data_plane_requires_real_transfer_queue_when_not_installed():
    ...
```

## 5. 验证覆盖度

### 5.1 Spec 15.1 - TQ 接口调用正确性 ✅
- 验证所有 TQ API 调用的参数类型和结构
- 覆盖：`kv_put`, `kv_batch_get`, `kv_list`, `kv_clear`
- 测试数：8

### 5.2 Spec 15.2 - 代码正确性 ✅
- 语法检查：通过
- Import 检查：通过
- 类型注解：通过
- 接口签名：与 workers.py/pipeline.py 一致

### 5.3 Spec 15.3 - 设计文档一致性 ✅
- 状态机转换：完整验证
- 隔离约束：多租户、多 adapter、多 policy_version
- 容量守卫：全局和 per-context
- Metadata 校验：写入时验证
- 测试数：27

### 5.4 Spec 15.4 - 端到端验证 ✅
- Mock 版本：5 个测试，验证完整 pipeline 生命周期
- 真实 TQ 版本：11 个测试，验证真实后端行为

## 6. 本地数据准备

### 6.1 模型
- **路径**: `/data/model/Qwen3.5-0.8B`
- **来源**: ModelScope `Qwen/Qwen3.5-0.8B`
- **大小**: 1.7GB
- **格式**: safetensors

### 6.2 数据集
- **路径**: `/data/gsm8k_train.parquet`
- **来源**: ModelScope `modelscope/gsm8k`
- **样本数**: 7473
- **字段**:
  - `question`: 原始问题文本
  - `answer`: 完整解答（含 `#### <数字>`）
  - `messages`: Twinkle 格式消息列表（system + user）
  - `user_data`: `[('ground_truth', '<数字>')]`
  - `ground_truth`: 提取的答案数字

### 6.3 数据预处理
```python
from modelscope import MsDataset
from twinkle.preprocessor.llm import GSM8KProcessor

dataset = MsDataset.load('modelscope/gsm8k', subset_name='main', split='train')
dataset.map(GSM8KProcessor())  # 生成 messages 和 user_data
dataset.to_parquet('/data/gsm8k_train.parquet')
```

## 7. 测试命令

### 7.1 运行所有测试
```bash
pytest tests/twinkle_agentic/async_rl/ -v
```

### 7.2 仅运行 Mock 测试
```bash
pytest tests/twinkle_agentic/async_rl/test_async_rl_core.py -v
pytest tests/twinkle_agentic/async_rl/test_base_pipeline.py -v
pytest tests/twinkle_agentic/async_rl/test_data_plane_verification.py -v
pytest tests/twinkle_agentic/async_rl/test_e2e_gsm8k.py -v
```

### 7.3 仅运行真实 TQ 测试
```bash
pytest tests/twinkle_agentic/async_rl/test_real_tq.py -v
```

### 7.4 运行真实端到端脚本
```bash
# 使用本地模型和数据
python cookbook/rl/async_rl_grpo_gsm8k.py

# 自定义路径
MODEL_ID=/path/to/model GSM8K_DATA=/path/to/data.parquet \
  python cookbook/rl/async_rl_grpo_gsm8k.py
```

## 8. 结论

### 8.1 验证结果
✅ **TransferQueueDataPlane 实现正确**
- 所有 64 个测试通过（1 个预期跳过）
- TQ 接口调用符合规范
- 状态机转换正确
- 多租户隔离有效
- 容量守卫工作正常

### 8.2 真实 TQ 兼容性
✅ **与 TransferQueue 0.1.8 完全兼容**
- TensorDict 返回类型正确处理
- Ray 初始化在 NPU 环境正常工作
- 数据清理和状态同步正确

### 8.3 设计文档符合度
✅ **完全符合 multilora-async-rl 设计文档**
- Namespace 格式：`{tenant_id}/{training_run_id}/{adapter_name}/train_{k}`
- 状态机：7 个状态，6 个转换
- 隔离边界：tenant、run、adapter、policy_version
- 容量控制：max_rows、max_rows_per_context、max_staleness

### 8.4 待完成项
- ⏳ 真实模型端到端测试（需要 vLLM NPU 版本）
- ⏳ 多 NPU 分布式测试
- ⏳ 长时间稳定性测试

## 9. 附录

### 9.1 测试文件清单
- `tests/twinkle_agentic/async_rl/test_async_rl_core.py` (11 tests)
- `tests/twinkle_agentic/async_rl/test_base_pipeline.py` (2 tests)
- `tests/twinkle_agentic/async_rl/test_data_plane_verification.py` (36 tests)
- `tests/twinkle_agentic/async_rl/test_e2e_gsm8k.py` (5 tests)
- `tests/twinkle_agentic/async_rl/test_real_tq.py` (11 tests)

### 9.2 相关文件
- `src/twinkle_agentic/async_rl/data_plane.py` - TransferQueueDataPlane 实现
- `multilora-async-rl/TransferQueueDataPlane_spec.md` - 详细规格说明
- `multilora-async-rl/多租户MultiLoRA异步RL设计.md` - 设计文档
- `cookbook/rl/async_rl_grpo_gsm8k.py` - 端到端示例脚本

### 9.3 环境依赖
```bash
pip install TransferQueue
pip install -e ".[transformers,ray,test]"
```

---

## 10. 功能增强（2026-06-25 下午）

### 10.1 新增功能概述

基于设计文档审查，发现 `TransferQueueDataPlane` 缺少 5 项关键功能，已全部实现并通过测试：

| 功能 | 设计文档要求 | 优先级 | 状态 |
|------|------------|--------|------|
| kv_batch_put 批量写入 | TQ 文档要求热路径不用循环 kv_put | 高 | ✅ |
| ack 机制 | 步骤 18: read/ack rows | 高 | ✅ |
| lease/claim 互斥 | 异常恢复: worker 崩溃后 claim 自动释放 | 高 | ✅ |
| clear_namespace | 异常恢复: 租户取消训练 | 中 | ✅ |
| close | 资源清理 | 中 | ✅ |

### 10.2 新增 API 详情

#### 10.2.1 Ack 机制（训练消费追踪）

```python
# 标记已消费样本
ack_rows(context, partition_id, sample_ids, *, task_name='train') -> int

# 查询已消费数量
get_consumed_count(partition_id, *, task_name='train') -> int

# 构建 dataloader 时自动过滤已 ack 样本
build_streaming_dataloader(context, partition_id, *, task_name=None) -> list
```

**特性**：
- 支持多 task 隔离（train/eval 独立追踪）
- `clear_partition` 自动重置消费状态
- 幂等操作（重复 ack 不会重复计数）

#### 10.2.2 Lease/Claim 互斥机制（worker 崩溃恢复）

```python
# 获取排他租约
claim_partition_with_lease(context, partition_id, *, worker_id, timeout=None) -> PartitionMetadata

# 释放租约
release_lease(partition_id, *, worker_id) -> None

# 续租
renew_lease(partition_id, *, worker_id, timeout=None) -> None
```

**特性**：
- 防止多 worker 同时处理同一 partition
- 自动过期恢复：`_recover_expired_leases()` 在 claim 时清理超时租约
- 默认租约超时：300 秒（可通过 `TransferQueueRuntimeConfig.lease_timeout` 配置）
- `clear_partition` 自动释放租约

#### 10.2.3 Namespace 管理

```python
# 清理 context 下所有 partition
clear_namespace(context) -> int
```

**特性**：
- 不影响其他 context 的数据
- 返回实际清理的 partition 数量
- 用于租户取消训练场景

#### 10.2.4 资源清理

```python
# 关闭 TQ 系统
close() -> None
```

**特性**：
- 调用 `tq.close()` 关闭 TQ 系统
- 兼容无 `close` 方法的 mock client

#### 10.2.5 批量写入优化

**内部方法**：
```python
_kv_batch_put(keys, partition_id, fields_list, tags_list) -> None
_batch_update_samples(partition_id, updates) -> None
```

**优化点**：
- `put_rollout_batch`: 改用 `kv_batch_put` 批量写入，替代循环 `kv_put`
- `_sync_partition_status`: 批量更新 tag
- `_batch_update_samples`: 批量更新 fields
- 性能提升：减少 TQ API 调用次数

### 10.3 代码变更统计

**文件**: `src/twinkle_agentic/async_rl/data_plane.py`

| 指标 | 变更前 | 变更后 |
|------|--------|--------|
| 总行数 | 354 | ~420 |
| 公开方法数 | 15 | 20 |
| 内部方法数 | 7 | 9 |
| 新增配置项 | 0 | 1 (`lease_timeout`) |

**新增配置项**：
```python
@dataclass
class TransferQueueRuntimeConfig:
    # ... 原有配置 ...
    lease_timeout: float = 300.0  # 默认租约超时（秒）
```

### 10.4 新增测试

**文件**: `tests/twinkle_agentic/async_rl/test_data_plane_new_features.py` (25 tests)

| 测试类别 | 测试数 | 覆盖内容 |
|---------|--------|---------|
| TestKVBatchPutOptimization | 3 | 验证批量写入优化 |
| TestAckMechanism | 7 | ack 机制、消费追踪、task 隔离 |
| TestLeaseClaimMechanism | 9 | 租约获取/释放/续租/过期恢复 |
| TestClearNamespace | 4 | namespace 清理、跨 context 隔离 |
| TestClose | 2 | 资源清理、mock 兼容 |

**关键测试用例**：
- `test_put_rollout_batch_uses_kv_batch_put`: 验证批量写入
- `test_ack_rows_tracks_consumed_samples`: 验证 ack 追踪
- `test_build_streaming_dataloader_filters_acked_rows`: 验证过滤已消费样本
- `test_lease_blocks_other_workers`: 验证租约互斥
- `test_expired_lease_auto_recovered`: 验证自动过期恢复
- `test_clear_namespace_clears_all_partitions_for_context`: 验证 namespace 清理

### 10.5 更新后的测试结果

**总计: 89 passed, 1 skipped**

#### Mock 测试（使用 FakeTransferQueueClient）

| 测试文件 | 测试数 | 状态 | 说明 |
|---------|--------|------|------|
| test_async_rl_core.py | 11 | ✅ (1 skipped) | 核心功能测试 |
| test_base_pipeline.py | 2 | ✅ | Pipeline 集成测试 |
| test_data_plane_verification.py | 36 | ✅ | TQ 接口 + 设计文档一致性验证 |
| test_e2e_gsm8k.py | 5 | ✅ | GSM8K 端到端集成测试 |
| **test_data_plane_new_features.py** | **25** | ✅ | **新增功能测试（批量写入、ack、lease、namespace）** |
| **Mock 测试小计** | **79** | **78 passed, 1 skipped** | |

#### 真实 TQ 测试（使用 TransferQueue 0.1.8 + Ray 2.55.1）

| 测试文件 | 测试数 | 状态 | 说明 |
|---------|--------|------|------|
| test_real_tq.py | 11 | ✅ | 真实 TQ 后端集成测试 |
| **真实 TQ 测试小计** | **11** | **11 passed** | |

**通过率**: 98.9% (89/90)，1 个测试因 TQ 已安装而被跳过（预期行为）

**测试环境差异**：
- **Mock 测试**：使用 `FakeTransferQueueClient` 内存实现，无需 Ray/TQ 依赖，运行速度快
- **真实 TQ 测试**：使用真实 TransferQueue SimpleStorage 后端，需要 Ray 初始化，验证真实环境行为

### 10.6 设计文档覆盖度更新

| 设计文档要求 | 步骤 | 实现状态 |
|------------|------|---------|
| init_namespace | 3 | ✅ 已实现 |
| get_metadata | 4 | ✅ 已实现 |
| check_capacity | 7 | ✅ 已实现 |
| put_rollout_batch | 11 | ✅ 已实现（批量优化） |
| claim/append reward | 13 | ✅ 已实现（批量优化） |
| claim/append advantage | 14 | ✅ 已实现（批量优化） |
| list_train_ready_partitions | 15 | ✅ 已实现 |
| iter(train_k) | 17 | ✅ 已实现 |
| **read/ack rows** | **18** | **✅ 新增实现** |
| clear_partition | 22 | ✅ 已实现 |
| **lease/claim 互斥** | **异常恢复** | **✅ 新增实现** |
| **clear_namespace** | **异常恢复** | **✅ 新增实现** |
| **close** | **资源清理** | **✅ 新增实现** |

### 10.7 本地数据准备

#### 10.7.1 模型下载

**模型**: Qwen3.5-0.8B  
**来源**: ModelScope  
**路径**: `/data/model/Qwen3.5-0.8B`  
**大小**: 1.7GB  
**格式**: safetensors

```bash
# 下载命令
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3.5-0.8B', cache_dir='/data/model')"
```

#### 10.7.2 数据集下载与预处理

**数据集**: GSM8K  
**来源**: ModelScope  
**路径**: `/data/gsm8k_train.parquet`  
**样本数**: 7473  
**格式**: Parquet

**字段结构**：
- `question`: 原始问题文本
- `answer`: 完整解答（含 `#### <数字>`）
- `messages`: Twinkle 格式消息列表（system + user）
- `user_data`: `[('ground_truth', '<数字>')]`
- `ground_truth`: 提取的答案数字

**预处理脚本**：
```python
from modelscope import MsDataset
from twinkle.preprocessor.llm import GSM8KProcessor
import pyarrow as pa
import pyarrow.parquet as pq

dataset = MsDataset.load('modelscope/gsm8k', subset_name='main', split='train')
dataset.map(GSM8KProcessor())  # 生成 messages 和 user_data

# 转换为 parquet
table = pa.table({
    'question': dataset['question'],
    'answer': dataset['answer'],
    'messages': dataset['messages'],
    'user_data': dataset['user_data'],
    'ground_truth': [ud[0][1] for ud in dataset['user_data']]
})
pq.write_table(table, '/data/gsm8k_train.parquet')
```

### 10.8 Cookbook 脚本更新

**文件**: `cookbook/rl/async_rl_grpo_gsm8k.py`

**变更**：
- 使用本地模型路径：`/data/model/Qwen3.5-0.8B`
- 使用本地数据路径：`/data/gsm8k_train.parquet`
- 修正参数名：`model_name_or_path` → `model_id`

**运行命令**：
```bash
# 需要先安装 vllm-ascend
pip install vllm-ascend==0.18.0

# 运行端到端训练
python cookbook/rl/async_rl_grpo_gsm8k.py
```

**依赖**：
- vllm-ascend >= 0.18.0（NPU 推理引擎）
- TransferQueue >= 0.1.8
- twinkle (本地安装)

### 10.9 待完成项

| 项目 | 原因 | 下一步 |
|------|------|--------|
| vllm-ascend 安装 | 用户取消了安装 | `pip install vllm-ascend==0.18.0` |
| 真实模型端到端运行 | 依赖 vllm-ascend | 安装后运行 `python cookbook/rl/async_rl_grpo_gsm8k.py` |
| StreamingDataset 集成 | TQ 原生流式消费 | 后续版本 |
| 异步 KV API | `async_kv_put` 等 | 后续版本 |
| 多 NPU 分布式测试 | 需要多卡环境 | 后续版本 |
| 长时间稳定性测试 | 需要生产环境验证 | 后续版本 |

### 10.10 总结

今日下午完成了 `TransferQueueDataPlane` 的 5 项关键功能增强：

1. **kv_batch_put 批量写入优化** - 提升写入性能
2. **ack 机制** - 实现训练消费追踪，支持多 task 隔离
3. **lease/claim 互斥机制** - 实现 worker 崩溃恢复，防止数据重复处理
4. **clear_namespace** - 支持租户取消训练场景
5. **close** - 实现资源清理

所有功能均已通过测试（25 个新增测试），总测试数达到 89 passed, 1 skipped。

代码从 354 行扩展到 ~420 行，新增 5 个公开 API 和 2 个内部方法，完全覆盖设计文档要求。
