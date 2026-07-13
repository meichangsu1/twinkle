# TransferQueue Client API 文档

本文专门整理 [transfer_queue/client.py](transfer_queue/client.py) 中 `AsyncTransferQueueClient` 和 `TransferQueueClient` 的能力、调用顺序和示例。多数用户推荐通过 `tq.init()` + `tq.get_client()` 获取同步 `TransferQueueClient`；只有在框架内部或需要自己管理事件循环/ZMQ 连接时，才直接使用 `AsyncTransferQueueClient`。

## 1. Client 能做什么

`TransferQueueClient` 是 TransferQueue 的底层用户入口。它连接两个面：

```text
Client
  -> Controller：获取/更新 metadata、生产状态、消费状态、partition 信息
  -> StorageManager：写入、读取、清理真实 TensorDict 数据
```

能力分组：

- 数据写入：`put()` / `async_put()`
- 元数据获取：`get_meta()` / `async_get_meta()`
- 数据读取：`get_data()` / `async_get_data()`
- 元数据更新：`set_custom_meta()` / `async_set_custom_meta()`
- 数据清理：`clear_samples()`、`clear_partition()`
- 状态查询：production / consumption status
- 消费状态重置：`reset_consumption()`
- partition 列表查询：`get_partition_list()`
- KV 元数据桥接：`kv_retrieve_meta()`、`kv_retrieve_keys()`、`kv_list()`

## 2. 获取 Client

### 推荐方式：`tq.get_client()`

```python
import ray
import transfer_queue as tq

ray.init(namespace="TransferQueueApp")
tq.init()

client = tq.get_client()
```

`tq.init()` 会初始化 controller、storage backend，并创建当前进程的 client。其他进程再次调用 `tq.init()` 会连接已有 controller，然后创建自己的 client。

### 直接构造同步 Client

一般不建议业务代码这样写，除非你已经拿到了 controller 的 `ZMQServerInfo`，并知道 backend 配置。

```python
from transfer_queue.client import TransferQueueClient

client = TransferQueueClient(
    client_id="my_client",
    controller_info=controller_zmq_info,
)
client.initialize_storage_manager(
    manager_type="SimpleStorage",
    config=simple_storage_config,
)
```

### 直接构造异步 Client

```python
from transfer_queue.client import AsyncTransferQueueClient

client = AsyncTransferQueueClient(
    client_id="my_async_client",
    controller_info=controller_zmq_info,
)
client.initialize_storage_manager(
    manager_type="SimpleStorage",
    config=simple_storage_config,
)
```

异步 client 的公开方法都需要 `await`。

## 3. 基本数据流

### 首次写入数据

没有 `BatchMeta` 时，`put(data, partition_id=...)` 会内部用 `mode="insert"` 创建 metadata，并写入 storage。

```python
import torch
from tensordict import TensorDict

data = TensorDict(
    {
        "input_ids": torch.arange(24).reshape(4, 6),
        "attention_mask": torch.ones(4, 6),
    },
    batch_size=4,
)

meta = client.put(data=data, partition_id="train")
```

异步版本：

```python
meta = await client.async_put(data=data, partition_id="train")
```

### 消费数据

先拿 metadata，再按 metadata 读真实数据。

```python
meta = client.get_meta(
    data_fields=["input_ids", "attention_mask"],
    batch_size=2,
    partition_id="train",
    task_name="update_actor",
)

batch = client.get_data(meta)
```

异步版本：

```python
meta = await client.async_get_meta(
    data_fields=["input_ids", "attention_mask"],
    batch_size=2,
    partition_id="train",
    task_name="update_actor",
)
batch = await client.async_get_data(meta)
```

### 给已有样本追加字段

传入已有 `BatchMeta` 时，`put()` 会把新字段写到这些样本上。

```python
meta = client.get_meta(
    data_fields=["input_ids"],
    batch_size=2,
    partition_id="train",
    task_name="generate",
)

response = TensorDict(
    {"response": torch.arange(8).reshape(2, 4)},
    batch_size=2,
)

updated_meta = client.put(data=response, metadata=meta)
```

## 4. `get_meta`

### `client.get_meta(...) -> BatchMeta`

签名：

```python
client.get_meta(
    data_fields: list[str],
    batch_size: int,
    partition_id: str,
    mode: str = "fetch",
    task_name: str | None = None,
    sampling_config: dict | None = None,
) -> BatchMeta
```

参数：

- `data_fields`：需要读取的字段列表。
- `batch_size`：请求样本数。
- `partition_id`：数据分区。
- `mode`：
  - `"fetch"`：默认，只返回 ready 且未被当前 `task_name` 消费的样本。
  - `"force_fetch"`：绕过 ready / consumption / sampler 过滤，可能拿到未 ready 或已消费样本。
  - `"insert"`：内部写入使用，业务代码通常不要直接用。
- `task_name`：消费状态命名空间。同一个 task 通常不会重复拿已消费样本。
- `sampling_config`：传给 sampler 的运行时参数，例如 `dp_rank`、`batch_index`。

返回：`BatchMeta`。

示例：

```python
meta = client.get_meta(
    data_fields=["input_ids"],
    batch_size=8,
    partition_id="train",
    task_name="train_actor",
)

print(meta.global_indexes)
print(meta.field_names)
print(meta.is_ready)
```

带 `RankAwareSampler` 参数：

```python
meta = client.get_meta(
    data_fields=["input_ids"],
    batch_size=8,
    partition_id="train",
    task_name="train_actor",
    sampling_config={
        "dp_rank": 0,
        "batch_index": 0,
    },
)
```

异步版本：

```python
meta = await client.async_get_meta(...)
```

## 5. `put`

### `client.put(...) -> BatchMeta`

签名：

```python
client.put(
    data: TensorDict,
    metadata: BatchMeta | None = None,
    partition_id: str | None = None,
    data_parser: Callable | None = None,
) -> BatchMeta
```

参数：

- `data`：要写入的 `TensorDict`。
- `metadata`：目标样本 metadata。提供时表示写入这些样本；不提供时会创建新样本。
- `partition_id`：不提供 `metadata` 时必填。
- `data_parser`：写入前解析引用数据，仅 SimpleStorage 支持。

返回：写入后的 `BatchMeta`。源码中会在写入后执行 `metadata.add_fields(data)`，所以返回 meta 会包含新增字段。

首次插入：

```python
meta = client.put(data=data, partition_id="train")
```

追加字段：

```python
meta = client.put(data=response, metadata=meta)
```

注意：

- `data.batch_size[0]` 应该与 `metadata.size` 对齐。
- 如果 `metadata=None`，必须传 `partition_id`。
- 多 worker 并发写入时，源码注释提醒可能存在数据顺序不一致风险，业务侧要保证 partition 和数据组织方式明确。

异步版本：

```python
meta = await client.async_put(...)
```

## 6. `get_data`

### `client.get_data(metadata) -> TensorDict`

按 `BatchMeta` 从 storage backend 读取真实数据。

```python
batch = client.get_data(meta)
```

只读部分字段：

```python
input_meta = meta.select_fields(["input_ids"])
input_batch = client.get_data(input_meta)
```

如果传入空 `BatchMeta` 或没有字段的 metadata，返回空 `TensorDict`。

异步版本：

```python
batch = await client.async_get_data(meta)
```

## 7. custom_meta

### `client.set_custom_meta(metadata) -> None`

把 `BatchMeta.custom_meta` 写回 controller。`custom_meta` 是样本级轻量元数据，KV API 里的 `tag` 底层就是它。

```python
meta.update_custom_meta(
    [
        {"uid": "u0", "score": 0.9},
        {"uid": "u1", "score": 0.8},
    ]
)
client.set_custom_meta(meta)
```

后续 `get_meta()` 返回的 `BatchMeta` 可以取回这些 custom meta。

异步版本：

```python
await client.async_set_custom_meta(meta)
```

## 8. 清理接口

### `client.clear_samples(metadata) -> None`

清理指定样本。会同时清 controller 里的 metadata 和 storage 里的真实数据。

```python
meta = client.get_meta(
    data_fields=["input_ids"],
    batch_size=4,
    partition_id="train",
    task_name="cleanup",
)

client.clear_samples(meta)
```

异步版本：

```python
await client.async_clear_samples(meta)
```

### `client.clear_partition(partition_id) -> None`

清理整个 partition。推荐按 rollout / step / dataset split 切 partition，然后生命周期结束后整批清理。

```python
client.clear_partition("rollout_100")
```

异步版本：

```python
await client.async_clear_partition("rollout_100")
```

## 9. 状态查询接口

### `client.get_production_status(data_fields, partition_id)`

查看 partition 中指定字段的生产状态。

```python
global_indexes, production_status = client.get_production_status(
    data_fields=["input_ids", "attention_mask"],
    partition_id="train",
)

print(global_indexes)
print(production_status)
```

返回：

- `global_indexes: torch.Tensor | None`
- `production_status: torch.Tensor | None`

`production_status` 中 `1` 表示 ready，`0` 表示 not ready。

异步版本：

```python
global_indexes, production_status = await client.async_get_production_status(...)
```

### `client.check_production_status(data_fields, partition_id) -> bool`

判断指定字段是否全部 ready。

```python
ready = client.check_production_status(
    data_fields=["input_ids"],
    partition_id="train",
)
```

异步版本：

```python
ready = await client.async_check_production_status(...)
```

### `client.get_consumption_status(task_name, partition_id)`

查看某个 task 在 partition 上的消费状态。

```python
global_indexes, consumption_status = client.get_consumption_status(
    task_name="update_actor",
    partition_id="train",
)

print(global_indexes)
print(consumption_status)
```

返回：

- `global_indexes: torch.Tensor | None`
- `consumption_status: torch.Tensor | None`

`consumption_status` 中 `1` 表示已消费，`0` 表示未消费。

异步版本：

```python
global_indexes, consumption_status = await client.async_get_consumption_status(...)
```

### `client.check_consumption_status(task_name, partition_id) -> bool`

判断该 task 是否已经消费完整个 partition。

```python
done = client.check_consumption_status(
    task_name="update_actor",
    partition_id="train",
)
```

源码行为：

- 如果 `consumption_status is None` 或为空，返回 `False`。
- 否则所有元素都为 `1` 时返回 `True`。

异步版本：

```python
done = await client.async_check_consumption_status(...)
```

### `client.reset_consumption(partition_id, task_name=None) -> bool`

重置消费状态，让数据可以被重新消费。

```python
# 只重置一个 task
ok = client.reset_consumption(
    partition_id="train",
    task_name="update_actor",
)

# 重置这个 partition 下所有 task
ok = client.reset_consumption(partition_id="train")
```

异步版本：

```python
ok = await client.async_reset_consumption(...)
```

### `client.get_partition_list() -> list[str]`

列出 controller 当前管理的 partition。

```python
partitions = client.get_partition_list()
print(partitions)
```

异步版本：

```python
partitions = await client.async_get_partition_list()
```

## 10. KV 元数据接口

这些是 KV 高层 API 使用的底层桥接接口。普通业务优先用 `tq.kv_put()` / `tq.kv_batch_put()` / `tq.kv_batch_get()` / `tq.kv_list()` / `tq.kv_clear()`。

### `client.kv_retrieve_meta(keys, partition_id, create=False) -> BatchMeta`

按用户 key 获取对应的 `BatchMeta`。`create=True` 时，不存在的 key 会被注册。

```python
meta = client.kv_retrieve_meta(
    keys=["sample_0", "sample_1"],
    partition_id="train",
    create=False,
)
```

注册新 key：

```python
meta = client.kv_retrieve_meta(
    keys=["sample_2"],
    partition_id="train",
    create=True,
)
```

异步版本：

```python
meta = await client.async_kv_retrieve_meta(...)
```

### `client.kv_retrieve_keys(global_indexes, partition_id) -> list[str]`

根据 `global_index` 查回 KV key。

```python
keys = client.kv_retrieve_keys(
    global_indexes=[0, 1, 2],
    partition_id="train",
)
```

异步版本：

```python
keys = await client.async_kv_retrieve_keys(...)
```

### `client.kv_list(partition_id=None) -> dict[str, dict[str, Any]]`

列出一个或全部 partition 的 key 和 custom_meta/tag。

```python
info = client.kv_list(partition_id="train")
print(info)
```

返回结构：

```python
{
    "train": {
        "sample_0": {"score": 0.9},
        "sample_1": {"score": 0.8},
    }
}
```

异步版本：

```python
info = await client.async_kv_list(partition_id="train")
```

## 11. 同步 Client 与异步 Client 对照

| 同步 `TransferQueueClient` | 异步 `AsyncTransferQueueClient` | 说明 |
|---|---|---|
| `put()` | `async_put()` | 写入数据 |
| `get_meta()` | `async_get_meta()` | 获取 BatchMeta |
| `get_data()` | `async_get_data()` | 读取真实 TensorDict |
| `set_custom_meta()` | `async_set_custom_meta()` | 写入样本级 custom_meta |
| `clear_samples()` | `async_clear_samples()` | 清理部分样本 |
| `clear_partition()` | `async_clear_partition()` | 清理整个 partition |
| `get_production_status()` | `async_get_production_status()` | 查询生产状态 |
| `check_production_status()` | `async_check_production_status()` | 判断是否全部 ready |
| `get_consumption_status()` | `async_get_consumption_status()` | 查询消费状态 |
| `check_consumption_status()` | `async_check_consumption_status()` | 判断是否全部 consumed |
| `reset_consumption()` | `async_reset_consumption()` | 重置消费状态 |
| `get_partition_list()` | `async_get_partition_list()` | 列出 partition |
| `kv_retrieve_meta()` | `async_kv_retrieve_meta()` | key -> BatchMeta |
| `kv_retrieve_keys()` | `async_kv_retrieve_keys()` | global_index -> key |
| `kv_list()` | `async_kv_list()` | 列出 key/tag |

同步 `TransferQueueClient` 内部会启动一个后台事件循环线程，把 async 方法封装成同步方法。使用完可调用：

```python
client.close()
```

如果 client 是通过 `tq.get_client()` 获取，通常由 `tq.close()` 统一清理。

## 12. 常见问题和建议

### 如何得到一个 partition 有多少行？

KV 写入场景：

```python
info = client.kv_list(partition_id="train")
row_count = len(info.get("train", {}))
```

底层 Client 场景：

```python
global_indexes, _ = client.get_production_status(
    data_fields=["input_ids"],
    partition_id="train",
)
row_count = 0 if global_indexes is None else len(global_indexes)
```

### `task_name` 有什么影响？

`task_name` 是消费状态命名空间。同一个 `task_name` 再次 `get_meta()` 时，通常不会拿到已标记 consumed 的样本；不同 `task_name` 可以独立消费同一批样本。

```python
meta_a = client.get_meta(..., task_name="generate")
meta_b = client.get_meta(..., task_name="train")
```

### `force_fetch` 什么时候用？

`mode="force_fetch"` 会绕过 ready 状态、消费状态和 sampler 过滤，可能拿到未 ready 或已消费样本。适合调试，不建议作为常规训练读取路径。

### Client 是否负责自动过期？

不负责自动 TTL 过期。需要业务侧调用：

```python
client.clear_samples(meta)
client.clear_partition(partition_id)
```

推荐按数据生命周期组织 partition，例如 `rollout_100`，用完后 `clear_partition()`。
