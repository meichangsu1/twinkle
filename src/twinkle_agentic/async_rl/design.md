`VLLMSamplerTQ` 的核心异步原理是：

> `sample()` 不等待模型生成完成，只把生成协程提交到 vLLM 专用事件循环，然后立即返回；生成结果完成后，由后台协程计算 Reward 并写入 TransferQueue。

整体流程：

```text
RolloutWorker
    │ 提交 PromptGroup
    ▼
VLLMSamplerTQ.sample()
    │ 立即返回 submission_id
    │
    └── 后台 asyncio event loop
          ├── 并发处理多个 PromptGroup
          │     ├── 并发生成多个 generation
          │     ├── 计算 Reward
          │     └── 写入 TransferQueue
          │
          └── AdvantageWorker 从 TQ 获取完整数据
```

### 1. vLLMSampler 创建专用线程和事件循环

父类 `vLLMSampler` 初始化时创建：

```python
self._async_loop = asyncio.new_event_loop()
self._async_thread = threading.Thread(
    target=self._run_event_loop,
    daemon=True,
)
self._async_thread.start()
```

后台线程持续运行：

```python
def _run_event_loop(self):
    asyncio.set_event_loop(self._async_loop)
    self._async_loop.run_forever()
```

之所以使用独立事件循环，是因为：

- vLLM AsyncLLM 要求异步方法在创建 Engine 的同一个事件循环中运行
- Ray Worker 本身可能已经运行 uvloop
- 不能在一个正在运行的 loop 中再次调用 `asyncio.run()`
- 专用线程可以让所有 vLLM 异步操作固定在同一个 loop 中

### 2. `sample()` 是同步方法，但采用 fire-and-forget

虽然 `sample()` 不是 `async def`：

```python
def sample(
    self,
    groups,
    sampling_params,
    allow_partial_rollout=False,
):
```

它内部会把协程提交到专用事件循环：

```python
future = self._submit_in_loop(
    self._sample_prompt_groups(...)
)
```

实际实现：

```python
def _submit_in_loop(self, coro) -> Future:
    return asyncio.run_coroutine_threadsafe(
        coro,
        self._async_loop,
    )
```

`run_coroutine_threadsafe()` 将协程安全地从当前 Ray 线程提交到 vLLM 后台线程，返回一个 `concurrent.futures.Future`。

`sample()` 不调用：

```python
future.result()
```

而是保存 Future：

```python
self._background_submissions[submission_id] = future
```

然后立即返回：

```python
return {
    'submission_id': submission_id,
    'submitted_prompt_groups': len(groups),
    'submitted_samples': ...,
}
```

所以这里真正的异步边界是：

```text
sample() 返回
≠ rollout 已完成

sample() 返回
= rollout 已成功提交到后台事件循环
```

### 3. RolloutWorker 提交后立即处理下一批

`RolloutWorker` 中：

```python
await asyncio.to_thread(
    self.sampler.sample,
    list(prepared.groups),
    prepared.sampling_params,
    self.allow_partial_rollout,
)
```

这里只等待 `sample()` 完成“提交”，不会等待模型生成完成。

随后立即：

```python
self._start_next_batch(key)
```

因此 RolloutWorker 可以继续：

- 预取下一批 prompt
- 为其他 context 创建 partition
- 向 sampler 提交更多 rollout

模型生成和数据生产在后台继续运行。

### 4. 多个 PromptGroup 并发执行

后台入口是：

```python
async def _sample_prompt_groups(...):
    results = await asyncio.gather(
        *(
            self._run_prompt_group(...)
            for group in groups
        ),
        return_exceptions=True,
    )
```

一个 submission 中的多个 `PromptGroup` 会并发运行：

```text
submission
├── group_0
├── group_1
├── group_2
└── group_3
```

这些协程最终共享同一个 vLLM Engine。vLLM 会在内部进行 continuous batching，将不同请求动态组合成 GPU batch。

### 5. 同一 Group 的多个 generation 也并发

每个 group 包含同一个 prompt 的多次生成：

```python
sources = [
    {
        **group.prompt,
        'generation_idx': generation_idx,
    }
    for generation_idx in range(num_generations)
]
```

随后 `_generate_group_samples()` 为每个 generation 创建任务：

```python
tasks = [
    self._generate_sample(...)
    for feat in encoded_inputs
]
```

并发等待：

```python
results = await asyncio.gather(
    *tasks,
    return_exceptions=True,
)
```

所以并发结构是两层：

```text
多个 PromptGroup 并发
└── 每个 Group 的多个 generation 并发
```

再加上 Ray 的 sampler DP 分片，完整并行层次是：

```text
多个 Ray sampler DP worker
└── 每个 worker 多个 submission
    └── 每个 submission 多个 group
        └── 每个 group 多个 generation
```

### 6. 单次 generation 是真正的异步 vLLM 请求

最终执行：

```python
response = await self._sample_single(...)
```

父类中的 `_sample_single()` 调用：

```python
response = await self.engine.sample(...)
```

等待期间不会阻塞专用事件循环，其他 generation 可以继续提交或处理结果。

### 7. 完成后直接写入 TransferQueue

一个 group 的所有 generation 完成后：

```python
rewards = _compute_rewards(...)
```

然后：

```python
await self.data_plane.complete_rollout_group(
    group,
    rollout_rows=rows,
    rewards=rewards,
    submission_id=submission_id,
)
```

它会将：

- 模型训练字段
- logprobs
- rewards
- policy version
- rollout 状态

写回预分配的 TQ 样本。

因此生成结果不会先返回给 `RolloutWorker`，数据流是：

```text
VLLMSamplerTQ
    ↓
TransferQueue
    ↓
AdvantageWorker
    ↓
TrainerWorker
```

TransferQueue 是不同异步阶段之间的数据交接和就绪判断机制。

### 8. AdvantageWorker 不等待指定 Future

AdvantageWorker 不持有 sampler 的 Future。它通过 TQ 查询完整 batch：

```python
fetch_ready_batch(...)
```

只有相应 group 的字段已经写完，TQ 才会返回可消费的 metadata。

这实现了生产者和消费者解耦：

```text
Sampler 负责生产
AdvantageWorker 负责轮询和消费
两者不直接互相 await
```

### 9. 后台异常通过健康检查传播

因为 `sample()` 已经提前返回，后台异常不能直接抛给调用者，所以注册完成回调：

```python
future.add_done_callback(
    self._on_submission_done(submission_id)
)
```

如果后台任务失败：

```python
self._failure = f'{type(error).__name__}: {error}'
```

Pipeline 周期性调用：

```python
self.sampler.check_health()
```

`check_health()` 再把后台错误抛出来：

```python
if self._failure is not None:
    raise RuntimeError(self._failure)
```

因此错误传播路径是：

```text
后台 rollout 失败
    ↓
Future callback 保存 _failure
    ↓
Pipeline 调用 check_health()
    ↓
整个训练任务失败
```

总结来说，`VLLMSamplerTQ` 的“异步”不是简单地把 `sample()` 改成 `async def`，而是由四部分共同实现：

- Ray sampler DP worker 提供进程/GPU 级并行
- 专用线程和 asyncio loop 承载 vLLM 异步 Engine
- fire-and-forget 提交让 RolloutWorker 不等待生成完成
- TransferQueue 让 rollout、advantage 和 training 阶段异步流水化