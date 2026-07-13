# TransferQueue API 文档

本文面向调用者，按“能直接照着写代码”的方式整理 TransferQueue 暴露出来的主要接口。Python 包名是 `transfer_queue`，通常写作 `import transfer_queue as tq`。

## 1. 安装与初始化

### 安装

```bash
pip install TransferQueue
```

源码开发模式：

```bash
pip install -e .
```

带 Yuanrong 后端依赖：

```bash
pip install "TransferQueue[yuanrong]"
# 或
pip install -e ".[yuanrong]"
```

### `tq.init(conf=None)`

初始化 TransferQueue。第一次调用会创建 `TransferQueueController`、初始化 storage backend、创建当前进程的 client；后续进程调用会连接已有 controller，并创建本进程自己的 client。

```python
import ray
import transfer_queue as tq

ray.init(namespace="TransferQueueApp")
tq.init()
```

带配置：

```python
from omegaconf import OmegaConf
import transfer_queue as tq

conf = OmegaConf.create(
    {
        "backend": {
            "storage_backend": "SimpleStorage",
            "SimpleStorage": {
                "num_data_storage_units": 4,
                "total_storage_size": 200000,
            },
        },
    }
)
tq.init(conf)
```

StreamingDataLoader 场景常用：

```python
from omegaconf import OmegaConf
from transfer_queue import RankAwareSampler
import transfer_queue as tq

conf = OmegaConf.create(
    {
        "controller": {
            "sampler": RankAwareSampler,
            "polling_mode": True,
        },
        "backend": {
            "SimpleStorage": {
                "num_data_storage_units": 4,
            }
        },
    },
    flags={"allow_objects": True},
)
tq.init(conf)
```

### `tq.close()`

关闭当前 TransferQueue 系统，清理 client、controller 和当前进程初始化的 storage 资源。

```python
tq.close()
ray.shutdown()
```

### `tq.get_metrics_endpoint() -> str | None`

启用 metrics 后，返回 Prometheus `/metrics` 地址。

```python
endpoint = tq.get_metrics_endpoint()
if endpoint:
    print(f"http://{endpoint}/metrics")
```

## 2. KV API

KV API 是最高层接口，用用户自定义 `key` 访问样本。适合快速接入、外部 replay buffer 或已有 controller 管理调度的场景。

### 数据模型

```text
partition_id  -> 逻辑命名空间，例如 train / val / rollout_100
key           -> 用户定义的样本 ID，例如 sample_001
fields        -> TensorDict 字段列，例如 input_ids / attention_mask
tag           -> 样本级轻量元数据，例如 step / score / status
```

### `tq.kv_put(key, partition_id, fields=None, tag=None, data_parser=None) -> KVBatchMeta`

写入或更新单个 key。`fields` 和 `tag` 至少提供一个。

参数：

- `key: str`：样本 key。
- `partition_id: str`：逻辑分区。
- `fields: TensorDict | dict | None`：要写入的字段。单样本 dict 会自动加 batch 维。
- `tag: dict | None`：样本级元数据。
- `data_parser: Callable | None`：写入前解析引用数据，仅 SimpleStorage 支持。

返回：`KVBatchMeta`，包含 `keys`、`tags`、`partition_id`、`fields`。

```python
import torch
import transfer_queue as tq

tq.init()

meta = tq.kv_put(
    key="sample_0",
    partition_id="train",
    fields={"input_ids": torch.tensor([1, 2, 3])},
    tag={"step": 0, "status": "ready"},
)

print(meta.keys)
print(meta.fields)
```

只更新 tag：

```python
tq.kv_put(
    key="sample_0",
    partition_id="train",
    tag={"status": "finished"},
)
```

### `tq.kv_batch_put(keys, partition_id, fields=None, tags=None, data_parser=None) -> KVBatchMeta`

批量写入或更新多个 key。热路径优先使用这个接口，不要循环 `kv_put()`。

参数：

- `keys: list[str]`：样本 key 列表。
- `partition_id: str`：逻辑分区。
- `fields: TensorDict | None`：批量字段，`fields.batch_size[0] == len(keys)`。
- `tags: list[dict] | None`：每个 key 一个 tag，长度等于 `len(keys)`。
- `data_parser: Callable | None`：写入前解析引用数据，仅 SimpleStorage 支持。

```python
import torch
from tensordict import TensorDict
import transfer_queue as tq

keys = ["sample_0", "sample_1", "sample_2"]
fields = TensorDict(
    {
        "input_ids": torch.arange(18).reshape(3, 6),
        "attention_mask": torch.ones(3, 6),
    },
    batch_size=3,
)
tags = [{"step": 0, "status": "ready"} for _ in keys]

meta = tq.kv_batch_put(
    keys=keys,
    partition_id="train",
    fields=fields,
    tags=tags,
)
```

增量追加字段：

```python
response = TensorDict(
    {"response": torch.arange(12).reshape(3, 4)},
    batch_size=3,
)
tq.kv_batch_put(keys=keys, partition_id="train", fields=response)
```

### `tq.kv_batch_get(keys, partition_id, select_fields=None) -> TensorDict`

按 key 读取数据。可只读取部分字段。

参数：

- `keys: str | list[str]`：单个 key 或多个 key。
- `partition_id: str`：逻辑分区。
- `select_fields: str | list[str] | None`：要读取的字段；`None` 表示读取该 key 已有的全部字段。

返回：`TensorDict`。

```python
data = tq.kv_batch_get(
    keys=["sample_0", "sample_1"],
    partition_id="train",
    select_fields=["input_ids", "attention_mask"],
)

input_only = tq.kv_batch_get(
    keys="sample_0",
    partition_id="train",
    select_fields="input_ids",
)
```

注意：请求字段未 ready 时会抛错。分阶段追加字段时，先确认对应字段已经写入。

### `tq.kv_batch_get_by_meta(meta, select_fields=None) -> TensorDict`

使用 `kv_put()` / `kv_batch_put()` 返回的 `KVBatchMeta` 读取数据。

```python
meta = tq.kv_batch_put(keys=keys, partition_id="train", fields=fields)
data = tq.kv_batch_get_by_meta(meta, select_fields="input_ids")
```

### `tq.kv_list(partition_id=None) -> dict[str, dict[str, Any]]`

列出 partition 中的 key 和 tag。

```python
train_info = tq.kv_list(partition_id="train")
print(train_info["train"].keys())

all_info = tq.kv_list()
for pid, key_to_tag in all_info.items():
    print(pid, len(key_to_tag))
```

返回结构：

```python
{
    "train": {
        "sample_0": {"step": 0, "status": "ready"},
        "sample_1": {"step": 0, "status": "ready"},
    }
}
```

### `tq.kv_clear(keys, partition_id) -> None`

删除指定 key 的 controller 元数据和 storage 数据。

```python
tq.kv_clear(keys="sample_0", partition_id="train")
tq.kv_clear(keys=["sample_1", "sample_2"], partition_id="train")
```

### 异步 KV API

异步接口与同步 KV API 参数一致，只是需要 `await`：

```python
await tq.async_kv_put(...)
await tq.async_kv_batch_put(...)
data = await tq.async_kv_batch_get(...)
data = await tq.async_kv_batch_get_by_meta(...)
info = await tq.async_kv_list(...)
await tq.async_kv_clear(...)
```

## 3. 底层 Client API

底层 client 适合需要直接控制 `BatchMeta`、消费状态、字段选择和 sampler 参数的场景。

### `tq.get_client() -> TransferQueueClient`

获取当前进程的 client。调用前需要先 `tq.init()`。

```python
client = tq.get_client()
```

### `client.put(data, metadata=None, partition_id=None, data_parser=None) -> BatchMeta`

写入 `TensorDict`。

两种用法：

1. 没有 metadata：首次插入新样本，需要提供 `partition_id`。
2. 有 metadata：给已有样本追加或更新字段。

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

给已有样本追加字段：

```python
meta = client.get_meta(
    data_fields=["input_ids"],
    batch_size=4,
    partition_id="train",
    task_name="generate",
)

response = TensorDict(
    {"response": torch.arange(16).reshape(4, 4)},
    batch_size=4,
)

client.put(data=response, metadata=meta)
```

### `client.get_meta(data_fields, batch_size, partition_id, mode="fetch", task_name=None, sampling_config=None) -> BatchMeta`

从 controller 获取一批可消费样本的元数据。

参数：

- `data_fields: list[str]`：需要的字段。
- `batch_size: int`：请求样本数。
- `partition_id: str`：分区。
- `mode: str`：通常用 `"fetch"`；`"force_fetch"` 会绕过 readiness 和消费过滤；`"insert"` 是内部写入使用。
- `task_name: str | None`：消费状态命名空间。
- `sampling_config: dict | None`：传给 sampler 的动态参数。

```python
meta = client.get_meta(
    data_fields=["input_ids", "attention_mask"],
    batch_size=2,
    partition_id="train",
    task_name="update_actor",
)

print(meta.global_indexes)
print(meta.field_names)
print(meta.is_ready)
```

带 sampler 参数：

```python
meta = client.get_meta(
    data_fields=["input_ids"],
    batch_size=8,
    partition_id="train",
    task_name="update_actor",
    sampling_config={"dp_rank": 0, "batch_index": 0},
)
```

### `client.get_data(metadata) -> TensorDict`

根据 `BatchMeta` 从 storage backend 读取真实数据。

```python
batch = client.get_data(meta)
print(batch["input_ids"])
```

字段级读取：

```python
input_meta = meta.select_fields(["input_ids"])
input_batch = client.get_data(input_meta)
```

### `client.set_custom_meta(metadata) -> None`

把 `BatchMeta.custom_meta` 写回 controller。

```python
meta.update_custom_meta(
    [{"score": 0.9}, {"score": 0.8}]
)
client.set_custom_meta(meta)
```

### `client.clear_samples(metadata) -> None`

删除一批样本的 metadata 和真实数据。

```python
meta = client.get_meta(
    data_fields=["input_ids"],
    batch_size=2,
    partition_id="train",
    task_name="cleanup",
)
client.clear_samples(meta)
```

### `client.clear_partition(partition_id) -> None`

删除整个 partition。推荐按数据生命周期切 partition，用完后整批清理。

```python
client.clear_partition("rollout_100")
```

### 状态查询接口

```python
global_indexes, status = client.get_production_status(
    data_fields=["input_ids", "attention_mask"],
    partition_id="train",
)

global_indexes, status = client.get_consumption_status(
    task_name="update_actor",
    partition_id="train",
)

all_ready = client.check_production_status(
    data_fields=["input_ids"],
    partition_id="train",
)

all_consumed = client.check_consumption_status(
    task_name="update_actor",
    partition_id="train",
)

partitions = client.get_partition_list()
```

### 重置消费状态

让同一份数据可以被重新消费。

```python
client.reset_consumption(partition_id="train", task_name="update_actor")

# 重置该 partition 下所有 task 的消费状态
client.reset_consumption(partition_id="train")
```

### Client KV 元数据接口

这些是 KV 高层 API 内部使用的低层接口，一般用户优先用 `tq.kv_*`。

```python
meta = client.kv_retrieve_meta(
    keys=["sample_0", "sample_1"],
    partition_id="train",
    create=False,
)

keys = client.kv_retrieve_keys(
    global_indexes=meta.global_indexes,
    partition_id="train",
)

info = client.kv_list(partition_id="train")
```

## 4. StreamingDataLoader API

适合训练 worker 持续流式消费生产者写入的数据。

### `StreamingDataset(...)`

构造参数：

- `config: DictConfig`：运行时配置，通常从 controller 取 `get_config()`。
- `batch_size: int`：每次从 TransferQueue 拉取的大 batch。
- `micro_batch_size: int`：每次 yield 给训练循环的小 batch。
- `data_fields: list[str]`：要读取的字段。
- `partition_id: str`：消费哪个 partition。
- `task_name: str`：消费状态命名空间。
- `dp_rank: int`：数据并行组 ID；同一 `dp_rank` 的 worker 拿一致样本。
- `should_check_consumption_status: bool`：`False` 表示无限流；`True` 表示有限数据集消费完后停止。
- `fetch_batch_fn: Callable | None`：自定义拉取函数。
- `process_batch_fn: Callable | None`：自定义 batch 后处理/切分函数。

### `StreamingDataLoader(dataset, num_workers=0, prefetch_factor=None, ...)`

包装 `StreamingDataset`，返回 PyTorch DataLoader 风格迭代器。输出是 `(TensorDict, BatchMeta)`。

完整示例：

```python
import ray
import transfer_queue as tq
from transfer_queue import RankAwareSampler, StreamingDataset, StreamingDataLoader
from omegaconf import OmegaConf

ray.init(namespace="TransferQueueApp")

conf = OmegaConf.create(
    {
        "controller": {
            "sampler": RankAwareSampler,
            "polling_mode": True,
        },
        "backend": {
            "SimpleStorage": {
                "num_data_storage_units": 4,
            }
        },
    },
    flags={"allow_objects": True},
)
tq.init(conf)

controller = ray.get_actor("TransferQueueController", namespace="transfer_queue")
runtime_conf = ray.get(controller.get_config.remote())

dataset = StreamingDataset(
    config=runtime_conf,
    batch_size=64,
    micro_batch_size=8,
    data_fields=["input_ids", "attention_mask"],
    partition_id="train",
    task_name="update_actor",
    dp_rank=0,
    should_check_consumption_status=False,
)

dataloader = StreamingDataLoader(
    dataset=dataset,
    num_workers=2,
    prefetch_factor=2,
)

for batch, batch_meta in dataloader:
    loss = train_step(batch)
```

注意：

- 流式场景建议 `controller.polling_mode=True`。
- `num_workers > 0` 时不要在 DataLoader worker 内调用 `tq.init()`；`StreamingDataset` 会直接用 ZMQ 信息创建 client。
- 无限流模式下 actor 退出前显式 `del dataloader`，避免 worker 子进程继续等待数据。

## 5. BatchMeta API

`BatchMeta` 是底层核心元数据对象。它不保存真实数据，只描述“哪些样本、哪些字段、状态如何”。

### 主要属性

```python
meta.global_indexes      # list[int]
meta.partition_ids       # list[str]
meta.field_schema        # dict[str, dict]
meta.production_status   # numpy int8 array, 1 ready / 0 not ready
meta.custom_meta         # list[dict], 每个样本一个
meta.extra_info          # batch 级额外信息
meta.field_names         # list[str]
meta.size                # 样本数
meta.is_ready            # 是否所有样本 ready
```

### 常用操作

```python
# 只保留部分字段
input_meta = meta.select_fields(["input_ids"])

# 只保留部分样本位置
partial_meta = meta.select_samples([0, 2, 4])

# 切成 N 份
chunks = meta.chunk(4)

# 多个 batch 追加行
merged = BatchMeta.concat([meta_a, meta_b])

# 同一批样本追加字段
unioned = meta_a.union(meta_b)

# 更新样本级自定义元数据
meta.update_custom_meta([{"score": 0.9} for _ in range(meta.size)])
client.set_custom_meta(meta)
```

### 手动构造

通常不需要手动构造，除非写测试或高级扩展。

```python
import numpy as np
import torch
from transfer_queue import BatchMeta

meta = BatchMeta(
    global_indexes=[0, 1],
    partition_ids=["train", "train"],
    field_schema={
        "input_ids": {
            "dtype": torch.int64,
            "shape": (512,),
            "is_nested": False,
            "is_non_tensor": False,
        }
    },
    production_status=np.array([1, 1], dtype="int8"),
)
```

## 6. KVBatchMeta API

KV API 返回的轻量元数据。

属性：

```python
meta.keys          # list[str]
meta.tags          # list[dict]
meta.partition_id  # str | None
meta.fields        # list[str] | None
meta.extra_info    # dict
meta.size          # int
```

常用：

```python
meta = tq.kv_batch_put(keys=keys, partition_id="train", fields=fields)
data = tq.kv_batch_get_by_meta(meta)

subset = meta.select_keys(["sample_0", "sample_2"])
```

## 7. Sampler API

Sampler 决定 controller 从 ready 样本里返回哪些样本，以及哪些样本标记为已消费。

### 内置 sampler

```python
from transfer_queue import (
    SequentialSampler,
    RankAwareSampler,
    GRPOGroupNSampler,
    SeqlenBalancedSampler,
)
```

- `SequentialSampler`：默认顺序采样，`consumed_indexes = sampled_indexes`。
- `RankAwareSampler`：分布式训练中按 `dp_rank` / `batch_index` 缓存采样结果，让同一数据并行组拿一致样本。
- `GRPOGroupNSampler`：按连续 N 个样本组成 group，完整 ready 后返回。
- `SeqlenBalancedSampler`：在 GRPO group 基础上按序列长度做负载均衡。

配置 sampler：

```python
from omegaconf import OmegaConf
from transfer_queue import RankAwareSampler
import transfer_queue as tq

conf = OmegaConf.create(
    {
        "controller": {
            "sampler": RankAwareSampler,
            "polling_mode": True,
        }
    },
    flags={"allow_objects": True},
)
tq.init(conf)
```

### 自定义 sampler

继承 `BaseSampler` 并实现 `sample()`。

```python
from typing import Any
import numpy as np
from transfer_queue import BaseSampler

class RandomReuseSampler(BaseSampler):
    def __init__(self, seed: int = 0):
        super().__init__()
        self.rng = np.random.RandomState(seed)

    def sample(
        self,
        ready_indexes: list[int],
        batch_size: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[int], list[int]]:
        if len(ready_indexes) < batch_size:
            return [], []

        sampled = self.rng.choice(
            ready_indexes,
            size=batch_size,
            replace=False,
        ).tolist()

        consumed = []  # 不标记 consumed，允许之后再次采样
        return sampled, consumed
```

使用：

```python
from omegaconf import OmegaConf

conf = OmegaConf.create(
    {"controller": {"sampler": RandomReuseSampler(seed=42)}},
    flags={"allow_objects": True},
)
tq.init(conf)
```

`sample()` 返回值含义：

```text
sampled_indexes   本次返回给用户的 global_index
consumed_indexes  本次标记为已消费的 global_index
```

如果想无放回消费，通常两者相同；如果想复用样本，`consumed_indexes` 可以为空或只包含部分样本。

## 8. Storage Backend 配置

默认后端是 `SimpleStorage`。

```python
conf = OmegaConf.create(
    {
        "backend": {
            "storage_backend": "SimpleStorage",
            "SimpleStorage": {
                "total_storage_size": 100000,
                "num_data_storage_units": 4,
            },
        }
    }
)
```

Yuanrong：

```python
conf = OmegaConf.create(
    {
        "backend": {
            "storage_backend": "Yuanrong",
            "Yuanrong": {
                "auto_init": True,
                "worker_port": 31501,
                "metastore_port": 2379,
                "enable_yr_npu_transport": False,
                "enable_rdma": False,
                "worker_args": "--shared_memory_size_mb 8192",
            },
        }
    }
)
```

MooncakeStore：

```python
conf = OmegaConf.create(
    {
        "backend": {
            "storage_backend": "MooncakeStore",
            "MooncakeStore": {
                "auto_init": True,
                "metadata_server": "localhost:50050",
                "master_server_address": "localhost:50051",
                "protocol": "tcp",
            },
        }
    }
)
```

## 9. 常见调用模式

### 模式 A：KV 快速读写

```python
import ray
import torch
from tensordict import TensorDict
import transfer_queue as tq

ray.init(namespace="TransferQueueApp")
tq.init()

keys = ["a", "b"]
fields = TensorDict({"x": torch.ones(2, 4)}, batch_size=2)

tq.kv_batch_put(keys=keys, partition_id="train", fields=fields)
data = tq.kv_batch_get(keys=keys, partition_id="train", select_fields="x")
tq.kv_clear(keys=keys, partition_id="train")

tq.close()
ray.shutdown()
```

### 模式 B：生产者写、消费者按 task 消费

```python
producer_client = tq.get_client()
producer_client.put(data=fields, partition_id="rollout_0")

consumer_client = tq.get_client()
meta = consumer_client.get_meta(
    data_fields=["x"],
    batch_size=2,
    partition_id="rollout_0",
    task_name="train",
)
batch = consumer_client.get_data(meta)
consumer_client.clear_samples(meta)
```

### 模式 C：按 partition 做生命周期管理

```python
partition_id = f"rollout_{global_step}"
client.put(data=rollout_data, partition_id=partition_id)

# 多个任务消费 ...

client.clear_partition(partition_id)
```
