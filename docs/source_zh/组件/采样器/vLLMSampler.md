# vLLMSampler

vLLMSampler 使用 vLLM 引擎进行高效推理,支持高吞吐量的批量采样。

## 共享实例路由

Ray 模式下，continuous-work 通过采样池共享的 `RequestRouter` 选择实例，而不是在每个调用端
单独维护负载计数。默认按最小在途负载分配，平局随机选择；携带有效会话绑定时优先复用实例。
`sample()` 支持独立的可选参数 `session_ids`；`MultiTurnRollout` 为每条 episode 准备一次会话参数，
默认或自定义回调通过 `**adapter_kwargs` 转发给 sampler。

```python
sampler = vLLMSampler(
    model_id='ms://Qwen/Qwen3.5-4B',
    remote_group='sampler_group',
    device_mesh=DeviceMesh.from_sizes(dp_size=2),
    routing_config={"max_sessions": 10000},
    engine_args={"enable_prefix_caching": True},
)
```

`routing_config` 由分发基础设施消费，不传入引擎；当前支持正整数 `max_sessions`，默认 10000。
同一 Ray namespace、相同实例 actor ID 集合共享路由器；配置不一致时报错。
路由器由创建采样池的作业持有，不使用 detached 生命周期。

`session_ids` 按输入顺序逐条对应，每个元素是非空字符串或 `None`。分发层消费该参数，
不修改 `InputFeature`，也不将 ID 传入模板或引擎。省略参数或使用 `None` 的输入按普通负载均衡处理。

```python
responses = sampler.sample(inputs, session_ids=["episode-A", "episode-B"])
sampler.release_sessions(["episode-A", "episode-B"])  # 会话结束时调用
```

同一实例仍可并发处理多个会话；空闲绑定可能被 LRU 淘汰。
满表且所有绑定都有在途请求时，新请求照常执行，本次不建立绑定。

`release_sessions(ids)` 解除绑定，不清空 KV cache；有在途请求时拒绝释放。
采样调用在 `finally` 中释放占用，rollout 在轨迹退出时解除绑定。第一版不跟踪调用退出后仍运行的
远程任务；超时或异常时，负载计数可能低于实际执行数量。本地单引擎模式接收该参数，
但不创建路由 actor，也不执行实例路由。

## 使用示例

```python
from twinkle.sampler import vLLMSampler
from twinkle.data_format import SamplingParams
from twinkle import DeviceMesh

# 创建采样器
sampler = vLLMSampler(
    model_id='ms://Qwen/Qwen3.5-4B',
    device_mesh=DeviceMesh.from_sizes(dp_size=2, tp_size=2),
    remote_group='sampler_group'
)

# 添加 LoRA
sampler.add_adapter_to_model('my_lora', 'path/to/lora')

# 设置采样参数
params = SamplingParams(
    max_tokens=512,
    temperature=0.7,
    top_p=0.9,
    top_k=50
)

# 进行采样
responses = sampler.sample(
    trajectories,
    sampling_params=params,
    adapter_name='my_lora',
    num_samples=4  # 每个 prompt 生成 4 个样本
)
```

## 特性

- **高性能**: 使用 PagedAttention 和连续批处理实现高吞吐量
- **LoRA 支持**: 支持动态加载和切换 LoRA 适配器
- **多样本生成**: 可以为每个 prompt 生成多个样本
- **Tensor Parallel**: 支持张量并行加速大模型推理

## 远程执行

vLLMSampler 支持 `@remote_class` 装饰器,可以在 Ray 集群中运行:

```python
import twinkle
from twinkle import DeviceGroup, DeviceMesh
from twinkle.sampler import vLLMSampler

# 初始化 Ray 集群
device_groups = [
    DeviceGroup(name='sampler', ranks=4, device_type='cuda')
]
twinkle.initialize('ray', groups=device_groups)

# 创建远程采样器
sampler = vLLMSampler(
    model_id='ms://Qwen/Qwen3.5-4B',
    device_mesh=DeviceMesh.from_sizes(dp_size=4),
    remote_group='sampler'
)

# sample 方法会在 remote worker 中执行
responses = sampler.sample(trajectories, sampling_params=params)
```

## 环境变量

- `TWINKLE_VLLM_IPC_TIMEOUT_S`:
  控制 `vLLMSampler` 与 vLLM worker extension 之间 IPC 通道（ZMQ REQ/REP）的超时时间（秒）。
  默认值为 `300`。该值必须大于 `0`。

> vLLMSampler 在 RLHF 训练中通常与 Actor 模型分离,使用不同的硬件资源,避免推理和训练相互干扰。
