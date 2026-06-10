# Async Agentic RL With TransferQueue - 架构设计与实现详解

## 1. 概述

Twinkle 的 **Async Agentic RL** 是一套面向多轮工具调用、环境交互和异步强化学习训练的设计方案。它将 **Rollout 生成**、**环境交互**、**样本传输** 和 **Trainer 更新** 解耦，使 rollout worker 可以持续生产多轮 agent 轨迹，trainer 可以持续消费训练样本并更新模型。

核心数据通道是 **TransferQueue**。Rollout 侧将完成的多轮 episode 转换为 `RolloutSample` 后写入队列，Trainer 侧从队列异步读取样本，计算 advantage 并执行 `forward_backward`。权重同步继续复用 Twinkle 现有的 `CheckpointEngineManager`。

### 1.1 核心设计理念

| 维度 | 同步 GRPO 流程 | Async Agentic RL 流程 |
| --- | --- | --- |
| 执行模型 | Rollout 完成一批后 Trainer 才开始训练 | Rollout 和 Trainer 并行运行 |
| 数据传输 | Python 变量直接传递一批 sample | TransferQueue 异步传输样本 |
| 交互模式 | 单轮 completion 为主 | 多轮 assistant/tool/env 交互 |
| 环境抽象 | reward 函数直接处理 trajectory | `Env.reset/step` 管理交互，`Reward` 独立打分 |
| 样本新鲜度 | 默认严格 on-policy | 通过 `policy_version` 和 `max_staleness` 控制 |
| 权重同步 | 每轮或每 step 同步 sampler 权重 | Trainer 按 `sync_interval` 异步发布新权重 |
| 用户扩展点 | 主要替换 reward / dataset | 可替换 Tool、ActionParser、Env、Reward、SampleBuilder、AgentLoop |

> 注意：本 spec 的目标不是替换当前同步 GRPO cookbook，而是提供一条支持多轮 agentic rollout 的异步训练路径。MVP 可以先复用现有 `MultiTurnRollout`，后续再演进到流式 sampler。

### 1.2 关键优势

1. **提升 GPU 利用率**：Rollout worker 和 Trainer 分离，推理与训练可以重叠执行。
2. **支持复杂多轮环境**：通过 `Env/Gym` 抽象支持工具调用、浏览器、代码执行、检索、游戏等环境。
3. **控制 off-policy 程度**：通过 `policy_version`、`max_staleness` 和队列策略限制样本陈旧度。
4. **降低模块耦合**：Trainer 不关心工具和环境，Rollout worker 不关心 optimizer。
5. **保留现有训练组件**：继续使用 `GRPOLoss`、`GRPOAdvantage`、`vLLMSampler` 和 checkpoint engine。

______________________________________________________________________

## 2. 系统架构

### 2.1 整体架构图

```text
┌────────────────────────────────────────────────────────────────────┐
│                         Controller / Driver                        │
│                                                                    │
│  - create rollout workers                                          │
│  - create trainer                                                  │
│  - create transfer queue                                           │
│  - manage policy_version / sync schedule                           │
└────────────────────────────────────────────────────────────────────┘
          │                         │                         │
          ▼                         ▼                         ▼
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│ RolloutWorker ×N │       │  TransferQueue   │       │     Trainer      │
│                  │       │                  │       │                  │
│ PromptSource     │       │ bounded buffer   │       │ model            │
│ AgentLoop        │ ───►  │ backpressure     │ ───►  │ GRPO/PPO loss    │
│ Sampler          │       │ stale filtering  │       │ optimizer        │
│ Env/Gym          │       │ stats            │       │ metrics          │
└────────┬─────────┘       └──────────────────┘       └────────┬─────────┘
         │                                                       │
         │ rollout with policy_version=N                        │
         │                                                       │ weights
         ▼                                                       ▼
┌──────────────────┐                                  ┌──────────────────┐
│ Environment      │                                  │ CheckpointEngine │
│ ToolManager      │                                  │ weight sync      │
│ External APIs    │                                  │ version publish  │
└──────────────────┘                                  └──────────────────┘
```

### 2.2 角色定义

| 角色 | 主要职责 | 是否持有模型权重 | 是否执行环境 |
| --- | --- | --- | --- |
| `RolloutWorker` | 采样、多轮交互、构造 `RolloutSample` | 持有 sampler / inference 权重 | 是 |
| `TransferQueue` | 缓冲 rollout 样本、提供背压、暴露统计 | 否 | 否 |
| `Trainer` | 消费样本、计算 advantage、反向传播、优化 | 持有 training 权重 | 否 |
| `Env/Gym` | reset/step，执行环境逻辑、工具调用、结果解析 | 否 | 是 |
| `Reward` | 对完成或中间 trajectory 打分 | 否 | 否 |
| `CheckpointEngine` | 将 trainer 权重同步给 sampler | 传输权重 | 否 |

### 2.3 与现有组件的关系

| 现有组件 | 在本方案中的用途 |
| --- | --- |
| `Trajectory` | episode 状态和 message history |
| `SamplingParams` | rollout 采样配置 |
| `vLLMSampler` | rollout worker 的推理后端 |
| `MultiTurnRollout` | MVP 可复用的多轮 tool-call rollout |
| `ToolManager` | 默认 `ToolEnv` 的工具执行器 |
| `Reward` | 独立 reward 模块，负责 final/dense reward |
| `GRPOLoss` | trainer 的 policy optimization loss |
| `GRPOAdvantage` | trainer 侧或 queue consumer 侧计算 group advantage |
| `CheckpointEngineManager` | trainer 到 sampler 的权重同步 |

______________________________________________________________________

## 3. 数据流详解：TransferQueue

### 3.1 TransferQueue 的定位

`TransferQueue` 是 rollout 生产者和 trainer 消费者之间的数据平面。它不执行模型推理，也不执行 optimizer step，只负责样本传输和流控。

```text
RolloutWorker.put(RolloutSample[]) ──► TransferQueue ──► Trainer.get_batch()
```

### 3.2 RolloutSample 数据结构

队列中不建议传裸 `Trajectory`，而应传标准化的 `RolloutSample`。

```python
@dataclass
class RolloutSample:
    sample_id: str
    prompt_id: str
    policy_version: int
    created_at: float

    input_feature: dict
    messages: list[dict]
    old_logps: list[float]
    reward: float
    advantage: float | None

    stop_reason: str
    turns: int
    truncated: bool
    metrics: dict
```

字段分工：

| 字段 | 用途 |
| --- | --- |
| `input_feature` | trainer `model.forward_backward(inputs=...)` |
| `old_logps` | GRPO/PPO importance ratio |
| `reward` | advantage 计算 |
| `advantage` | 可选，若上游已经计算 advantage |
| `messages` | trace、debug、reward 复查 |
| `policy_version` | 样本新鲜度控制 |
| `turns/stop_reason/truncated` | 过滤和统计 |
| `metrics` | 任务自定义指标 |

关键不变量：

```text
len(old_logps) == count(input_feature["labels"] != -100)
```

这个不变量必须在 `RolloutSampleBuilder.validate()` 中检查，否则 `GRPOLoss` 的 old-logp 对齐会被破坏。

### 3.3 TransferQueue 接口

```python
class TransferQueue:

    def put(
        self,
        samples: list[RolloutSample],
        block: bool = True,
        timeout: float | None = None,
    ) -> int:
        """Insert samples and return the number accepted."""

    def get_batch(
        self,
        batch_size: int,
        timeout: float | None = None,
    ) -> list[RolloutSample]:
        """Return up to batch_size samples."""

    def qsize(self) -> int:
        """Return approximate queue size."""

    def stats(self) -> dict:
        """Return queue metrics."""
```

### 3.4 Backpressure 策略

Backpressure 解决的问题是：rollout 生产速度大于 trainer 消费速度时，如何避免内存无限增长和样本过旧。

MVP 推荐策略：

1. **有界队列**：`max_size` 必须显式配置。
2. **默认阻塞生产者**：`put(block=True)`，队列满时 rollout worker 等待。
3. **Trainer 侧丢 stale sample**：`current_version - sample.policy_version > max_staleness` 的样本不训练。
4. **暴露监控指标**：队列大小、生产者等待时间、消费者等待时间、丢弃样本数、平均 policy lag。

可选增强：

| 策略 | 适用场景 |
| --- | --- |
| `put(block=False)` | rollout 成本低、允许丢样本 |
| stale-first eviction | 队列满时优先清理旧版本样本 |
| priority queue | 成功样本或高价值样本优先训练 |
| max trajectory size | 超长多轮轨迹直接丢弃或截断 |

推荐默认配置：

```python
queue_max_size = train_batch_size * max_staleness * 2
put_block = True
max_staleness = 2
```

______________________________________________________________________

## 4. Agentic 环境交互抽象

### 4.1 为什么需要可扩展交互接口

不同 agentic 任务的环境交互差异很大：

- 普通工具调用：assistant 生成 `tool_calls`，系统执行函数并返回 `tool` message。
- 浏览器环境：assistant 生成点击、输入、导航等 action，环境返回 DOM/截图/URL。
- 代码环境：assistant 生成代码或命令，环境执行并返回 stdout/stderr。
- 检索环境：assistant 生成 query，环境返回文档。
- 游戏环境：assistant 生成 action，环境推进状态并返回 observation。

因此 rollout loop 不应该硬编码 tool-call 流程。更合理的边界是：`MultiTurnRollout` 或 `AgentLoop` 继续负责“驱动多轮生成”，原本写在 rollout 内部的工具调用、调用结果解析和 observation 构造下沉到 `Env/Gym`。

### 4.2 组件分层

| 组件 | 作用 | 用户是否常替换 |
| --- | --- | --- |
| `MultiTurnRollout` / `AgentLoop` | 通用 episode 控制流和模型采样驱动 | 少量高级用户 |
| `ActionParser` | assistant 输出转 action | 常见 |
| `Env/Gym` | 环境状态推进、工具调用、调用结果解析、observation 渲染 | 最常见 |
| `Reward` | 对完成或中间 trajectory 打分 | 常见 |
| `RolloutSampleBuilder` | trajectory 转训练样本 | 中等 |
| `Tool` | 单个函数工具 | 最常见 |

```text
MultiTurnRollout / AgentLoop
     │
     ▼
sampler output / response_text
     │
     ▼
ActionParser or Env internal parser
     │
     ▼
Env.step / Env.step_text
     │
     ▼
AgentLoop.should_stop
     │
     ▼
Reward
     │
     ▼
RolloutSampleBuilder
     │
     ▼
TransferQueue
```

### 4.3 MultiTurnRollout 与 Env 的关系

`Env` 和 `MultiTurnRollout + Tool` 不是互斥关系。`MultiTurnRollout` 仍然可以作为多轮 rollout driver；区别是它不应该直接持有“如何执行工具、如何解析工具结果、如何构造 observation”的业务逻辑，而是把这一段交给 Env。

当前 `MultiTurnRollout` 中这段逻辑：

```python
tool_messages = [
    {
        "role": "tool",
        "content": tool_managers[global_idx](tc),
    }
    for tc in tool_calls
]
```

可以演进为：

```python
obs, done, info = env.step(response_text=seq.decoded, tool_calls=tool_calls)
tool_messages = env.render_messages(obs, info)
```

这样：

- `MultiTurnRollout` 负责采样、维护 pif、拼接 bridge tokens、收集 old logps；
- `Env` 负责工具调用、环境状态、结果解析、done 判断；
- `Reward` 负责 episode 结束后的训练打分；
- `ToolManager` 仍然可以作为 `ToolEnv` 内部的工具注册和分发机制。

### 4.4 AgentLoop

`AgentLoop` 是未来可抽象出来的默认多轮控制器。它负责调用 sampler、parser、env、reward 和 sample builder，并暴露 hook 给用户。MVP 可以先让 `AgentLoop` wrapper 当前 `MultiTurnRollout`。

```python
class AgentLoop:

    def __init__(
        self,
        sampler,
        template,
        env: "Gym",
        reward: "Reward",
        action_parser: "ActionParser",
        sample_builder: "RolloutSampleBuilder",
        sampling_params: SamplingParams,
        max_turns: int = 6,
        max_trajectory_tokens: int | None = None,
    ):
        ...

    def run(
        self,
        prompts: list[Trajectory],
        *,
        policy_version: int,
        **kwargs,
    ) -> list[RolloutSample]:
        ...

    def before_episode(self, trajectories: list[Trajectory], **kwargs) -> None:
        """Called after gym.reset()."""

    def before_model_step(self, active: list[Trajectory], turn: int, **kwargs) -> None:
        """Called before sampler.sample()."""

    def after_model_step(
        self,
        trajectories: list[Trajectory],
        responses: list[SampleResponse],
        turn: int,
        **kwargs,
    ) -> None:
        """Called after sampler.sample()."""

    def should_stop(
        self,
        trajectory: Trajectory,
        *,
        turn: int,
        stop_reason: str | None,
        info: dict,
    ) -> bool:
        """Return True if this episode should stop."""

    def after_episode(self, trajectories: list[Trajectory], **kwargs) -> None:
        """Called before sample building."""
```

默认 `AgentLoop` 行为应与当前 `MultiTurnRollout` 一致，但 tool 调用部分委托给 Env：

1. 编码初始 `Trajectory`。
2. 对 active trajectory 执行一次 batched `sampler.sample()`。
3. 将 assistant 输出追加到 message history。
4. 用 `ActionParser` 解析 action。
5. 调用 `Env.step()` 或 `Env.step_text()`，由 Env 执行工具/环境动作并解析调用结果。
6. 判断是否继续下一轮。
7. episode 结束后调用独立 `Reward` 模块打分。
8. 用 `RolloutSampleBuilder` 构造样本并写入 TransferQueue。

### 4.5 ActionParser

`ActionParser` 负责把 assistant 输出转换为环境 action。

```python
class ActionParser:

    def parse(
        self,
        trajectory: Trajectory,
        assistant_message: dict,
        *,
        decoded: str | None = None,
        template=None,
        turn: int,
        **kwargs,
    ) -> list[dict]:
        ...
```

默认实现：

```python
class ToolCallActionParser(ActionParser):

    def parse(self, trajectory, assistant_message, *, decoded=None, template=None, turn=0, **kwargs):
        tool_calls = assistant_message.get("tool_calls") or []
        if not tool_calls and decoded and template is not None:
            tool_calls = template.parse_tool_call(decoded) or []
        return [
            {
                "type": "tool_call",
                "tool_call": tc,
                "turn": turn,
            }
            for tc in tool_calls
        ]
```

用户可以替换为：

- `BrowserActionParser`
- `CodeActionParser`
- `JsonActionParser`
- `SearchActionParser`
- `GameActionParser`

### 4.6 Env / Gym

`Env/Gym` 是环境交互的主扩展点。它不负责最终训练打分；打分应放在独立 `Reward` 模块中。Env 的职责是维护环境状态、执行 tool/environment action、解析调用结果，并把结果渲染回下一轮模型可读的 observation message。

```python
class BaseInteractionEnv:

    def reset(self, **kwargs) -> tuple[dict, dict]:
        """Reset one environment instance.

        Returns:
            observation: initial observation payload
            info: reset metadata
        """

    def step(self, response_text: str, **kwargs) -> tuple[dict, bool, dict]:
        """Process one assistant response.

        Returns:
            observation: payload to be rendered into the next model-visible message
            done: whether this episode should stop
            info: environment metadata
        """
```

Batch rollout 可以用一个 adapter 管理多个 env instance：

```python
class BatchedEnv:

    def reset(self, prompts: list[Trajectory], **kwargs) -> list[Trajectory]:
        """Create one BaseInteractionEnv per trajectory and return initial trajectories."""

    def step(
        self,
        trajectories: list[Trajectory],
        response_texts: list[str],
        actions: list[list[dict]] | None = None,
        **kwargs,
    ) -> tuple[list[Trajectory], list[bool], list[dict]]:
        """Apply one assistant response per active trajectory."""
```

`BaseInteractionEnv` 可以选择自己解析 response text，也可以接收 `ActionParser` 已经解析好的 actions。

以图像工具环境为例：

```python
class DeepeyesEnv(BaseInteractionEnv):
    """Environment for Deepeyes with zoom-in and rotate tools."""

    MIN_DIMENSION = 28

    def __init__(self, *, max_turns: int | None = None, image=None):
        self.max_turns = max_turns
        self.turn = 0
        self.tool_calls: list[dict[str, Any]] = []
        self.current_image = image
        self.origin_image = image

    def reset(self):
        self.turn = 0
        self.tool_calls.clear()
        observation: dict[str, Any] = {}
        reset_info = {"has_image": self.current_image is not None}
        return observation, reset_info

    def step(self, response_text: str):
        self.turn += 1

        if ANSWER_RE.search(response_text):
            return self._build_obs_text(text="Answer received."), True, {"final_answer": True}

        tool_call = self._extract_tool_call(response_text)
        if not tool_call:
            obs = self._build_obs_text(text="No tool call detected; ending the episode.")
            return obs, True, {"tool_executed": False}

        obs, done, info = self._apply_tool(tool_call)
        if self.max_turns is not None and self.turn >= self.max_turns:
            done = True
        return obs, done, info
```

对于 OpenAI-shape tool call，可以提供 `ToolEnv`：

```python
class ToolEnv(BaseInteractionEnv):

    def __init__(self, tool_manager: ToolManager, *, max_turns: int | None = None):
        self.tool_manager = tool_manager
        self.max_turns = max_turns
        self.turn = 0

    def call_tool(
        self,
        action: dict,
        **kwargs,
    ) -> object:
        """Invoke one tool/environment action and return a raw result."""

    def parse_tool_result(
        self,
        action: dict,
        result: object,
        **kwargs,
    ) -> object:
        """Parse a raw tool result into a structured observation."""

    def on_tool_error(
        self,
        action: dict,
        error: Exception,
        **kwargs,
    ) -> dict:
        """Convert environment errors into observation messages."""

    def render_observation(
        self,
        observation: object,
        **kwargs,
    ) -> dict:
        """Convert raw environment output into a message."""
```

默认 `ToolEnv`：

- 接收 `type="tool_call"` action；
- 通过 `ToolManager` 执行工具；
- 通过 `parse_tool_result()` 将工具返回值解析为 observation；
- 通过 `render_observation()` 将 observation 渲染成 `role="tool"` message；
- 工具异常通过 `on_tool_error()` 转成 observation；
- 不计算最终 reward。

### 4.7 Reward

`Reward` 是独立打分模块。它可以复用现有 `twinkle.reward.Reward` 接口，也可以在 agentic 场景下扩展为支持 final reward 和 dense reward。

推荐接口：

```python
class Reward:

    def __call__(
        self,
        trajectories: list[Trajectory],
        *,
        infos: list[dict] | None = None,
        **kwargs,
    ) -> list[float]:
        """Compute final scalar rewards for completed trajectories."""
```

可选 dense reward 接口：

```python
class StepReward:

    def step_reward(
        self,
        trajectories: list[Trajectory],
        actions: list[list[dict]],
        next_trajectories: list[Trajectory],
        infos: list[dict],
        **kwargs,
    ) -> list[float]:
        """Compute per-step rewards if the algorithm needs dense feedback."""
```

Reward 模块可以由用户自由组合：

```python
reward = RewardPipeline([
    F1Reward(),
    ToolExploreReward(),
    FormatReward(),
])
```

职责边界：

| 模块 | 应该负责 | 不应该负责 |
| --- | --- | --- |
| `Env/Gym` | tool 调用、环境推进、结果解析、observation 渲染、done 判断 | policy loss、最终训练 reward |
| `Reward` | final/dense reward、任务指标、reward shaping | 执行外部工具、修改环境状态 |

### 4.8 RolloutSampleBuilder

`RolloutSampleBuilder` 负责把完成的 episode 转成 queue item。

```python
class RolloutSampleBuilder:

    def build(
        self,
        trajectories: list[Trajectory],
        rewards: list[float],
        *,
        policy_version: int,
        **kwargs,
    ) -> list[RolloutSample]:
        ...

    def trainable_mask(self, trajectory: Trajectory, **kwargs) -> list[bool] | None:
        """Select trainable tokens or rounds."""

    def validate(self, sample: RolloutSample) -> bool:
        """Return False to drop invalid samples before queue insertion."""
```

默认 builder：

- 使用 rollout 产生的 `input_ids`、`labels`、`logprobs`；
- 检查 old-logp/label 对齐；
- 保存 final scalar reward；
- 不提前计算 advantage，让 trainer 统一计算。

高级用户可以实现：

- 只训练最终 assistant turn；
- 只训练成功 tool-call turn；
- 只训练 `trajectory["user_data"]["key_rounds"]`；
- 存储 dense reward；
- 附加任务 trace metadata。

### 4.9 Hook 调用顺序

```text
env.reset(prompts)
agent_loop.before_episode()

for turn in range(max_turns):
    agent_loop.before_model_step(active_trajectories, turn)
    sampler.sample(active_input_features)
    agent_loop.after_model_step(active_trajectories, responses, turn)

    response_texts = decode responses
    actions = action_parser.parse(...)  # optional; Env may parse response_text itself
    next_trajectories, dones, infos = env.step(response_texts=response_texts, actions=actions)
    agent_loop.should_stop(...) per trajectory

rewards = reward(final_trajectories, infos=final_infos)
agent_loop.after_episode()
sample_builder.build(..., rewards=rewards)
transfer_queue.put(samples)
```

### 4.10 用户自定义层级

用户应选择最小可行扩展点：

1. **只替换 `Tool`**：普通函数工具调用。
2. **替换 `ActionParser`**：模型 action 语法不同。
3. **替换 `Env/Gym`**：环境有状态、tool 调用方式、observation 解析逻辑不同。
4. **替换 `Reward`**：任务打分、reward shaping 或 dense reward 逻辑不同。
5. **替换 `RolloutSampleBuilder`**：训练 token 选择或样本 metadata 不同。
6. **继承 `AgentLoop`**：episode 控制流本身不同。

______________________________________________________________________

## 5. Rollout Worker 流程

### 5.1 主循环

```python
while running:
    prompts = prompt_source.next_batch(prompt_batch_size)
    version = policy_state.current_version()

    samples = agent_loop.run(
        prompts,
        policy_version=version,
    )

    accepted = transfer_queue.put(samples, block=True)
    metrics.log({"accepted": accepted, **transfer_queue.stats()})
```

### 5.2 与现有 MultiTurnRollout 的关系

MVP 有两种实现路径：

1. **Wrapper 路径**：`AgentLoop` 内部直接调用现有 `MultiTurnRollout`，然后用 `RolloutSampleBuilder` 转换样本。
2. **重构路径**：把 `MultiTurnRollout` 中的 tool loop 拆成 `AgentLoop + ToolCallActionParser + ToolEnv + Reward`。

推荐先采用 wrapper 路径，降低改动风险；后续再逐步重构。

______________________________________________________________________

## 6. Trainer 流程

### 6.1 主循环

```python
while global_step < max_steps:
    samples = transfer_queue.get_batch(train_batch_size, timeout=batch_timeout)

    samples = [
        s for s in samples
        if current_version - s.policy_version <= max_staleness
    ]
    if not samples:
        continue

    advantages = advantage_fn(
        [s.reward for s in samples],
        num_generations=num_generations,
        scale="group",
    )

    model.forward_backward(
        inputs=[s.input_feature for s in samples],
        old_logps=[s.old_logps for s in samples],
        advantages=advantages.tolist(),
        micro_batch_size=micro_batch_size,
    )
    model.clip_grad_and_step()

    global_step += 1

    if global_step % sync_interval == 0:
        ckpt_manager.sync_weights(merge_and_sync=False)
        current_version += 1
        policy_state.publish_version(current_version)
```

### 6.2 policy version 与 staleness

异步 rollout 会产生旧策略样本。每条样本记录生成时的 `policy_version`，trainer 消费时检查：

```python
policy_lag = current_version - sample.policy_version
if policy_lag > max_staleness:
    drop(sample)
```

推荐配置：

| 模式 | `max_staleness` | `sync_interval` | 说明 |
| --- | --- | --- | --- |
| 严格 on-policy | 0 | 1 | 最稳定，吞吐最低 |
| 默认 async | 2 | 1-8 | 平衡吞吐和新鲜度 |
| 高吞吐 async | 4+ | 8+ | 更 off-policy，需要监控 KL/ratio |

______________________________________________________________________

## 7. 异步权重同步

### 7.1 同步对象

Trainer 更新后，需要将权重同步给 rollout sampler。

```text
Trainer model ── CheckpointEngineManager.sync_weights() ──► vLLMSampler
```

### 7.2 同步时机

推荐由 trainer 控制同步：

```python
if global_step % sync_interval == 0:
    ckpt_manager.sync_weights(merge_and_sync=False)
    policy_state.publish_version(current_version + 1)
```

### 7.3 与 sample staleness 的关系

`policy_version` 推荐在 sampler 权重同步成功后递增，而不是每次 optimizer step 后递增。这样 rollout worker 看到的版本与 sampler 实际权重一致。

______________________________________________________________________

## 8. Reward 设计

### 8.1 Final Reward

MVP 只要求 final scalar reward：

```text
final trajectory ── Reward.__call__() ──► reward
```

适合 GRPO：

```python
advantages = GRPOAdvantage()(rewards, num_generations=N, scale="group")
```

### 8.2 Dense Reward

后续可支持 step reward：

```text
Env.step() -> next trajectories / infos
StepReward.step_reward() -> step rewards
Reward.__call__() -> final rewards
```

需要进一步定义 credit assignment：

- per-episode scalar；
- per-turn reward；
- per-token reward；
- final reward + tool-use shaping reward。

MVP 不强制支持 dense reward。

______________________________________________________________________

## 9. 失败处理与样本过滤

### 9.1 Rollout 侧

Rollout 失败不应导致 trainer 崩溃。

推荐策略：

- 工具异常转成 `role="tool"` 的错误 observation；
- 环境/API 异常设置 `stop_reason="error"`；
- 超过 `max_turns` 设置 `truncated=True`；
- 超过 `max_trajectory_tokens` 截断或丢弃；
- 构造 sample 前执行 `sample_builder.validate()`。

### 9.2 Trainer 侧

Trainer 应丢弃：

- 缺少 `old_logps` 的样本；
- 缺少 `labels` 的样本；
- old-logp/label 不对齐的样本；
- 超过 `max_staleness` 的样本；
- task-specific invalid sample。

______________________________________________________________________

## 10. 推荐模块布局

```text
src/twinkle/transfer_queue/
├── __init__.py
├── base.py              # RolloutSample, TransferQueue
├── local.py             # LocalTransferQueue for tests
└── ray_queue.py         # RayTransferQueue for distributed workers

src/twinkle/gym/
├── base.py              # Env/Gym reset/step and hooks
├── action.py            # ActionParser, ToolCallActionParser
├── sample_builder.py    # RolloutSampleBuilder
└── tool_env.py          # ToolManager-backed Env/Gym

src/twinkle_agentic/reward/
├── __init__.py
├── base.py              # Agentic Reward / StepReward adapters if needed
├── pipeline.py          # RewardPipeline
└── f1.py                # Existing task rewards

src/twinkle_agentic/runner/
├── agent_loop.py        # Generic multi-turn agent loop
├── rollout_worker.py    # Producer
└── async_rl_trainer.py  # Consumer / trainer
```

______________________________________________________________________

## 11. MVP 实施计划

1. 新增 `RolloutSample` 和 `TransferQueue` base interface。
2. 实现 `LocalTransferQueue`，用于单进程单测。
3. 实现 `RayTransferQueue`，用于多 worker 异步 rollout。
4. 扩展 `Gym` 为 `reset/step`，移除环境内置 score 职责。
5. 新增 `ActionParser` 和默认 `ToolCallActionParser`。
6. 新增 `ToolEnv`，复用 `ToolManager`，并提供 tool result parsing hooks。
7. 新增独立 `Reward` / `RewardPipeline`，复用现有 reward 模块。
8. 新增 `RolloutSampleBuilder`，实现 old-logp/label 对齐校验。
9. 新增 `AgentLoop`，先 wrapper 当前 `MultiTurnRollout`。
10. 新增 `RolloutWorker`，持续生产 sample 并写 queue。
11. 新增 `AsyncGRPOTrainer`，持续消费 queue 并训练。
12. 加入 `policy_version`、`max_staleness`、queue stats。
13. 添加 cookbook：

```text
ToolEnv + RewardPipeline + AgentLoop/MultiTurnRollout + RayTransferQueue + AsyncGRPOTrainer
```

______________________________________________________________________

## 12. Open Questions

| 问题 | MVP 建议 |
| --- | --- |
| `policy_version` 何时递增？ | sampler 权重同步成功后递增 |
| stale sample 在哪里丢弃？ | trainer 丢弃，queue 只负责传输 |
| 是否需要 priority queue？ | MVP 不需要 |
| 是否支持 dense reward？ | MVP 只支持独立 `Reward` 的 final scalar reward |
| `Env.step()` 接 raw message 还是 action？ | 两者都支持：默认接 `response_text`，也可接收 `ActionParser` 输出的 action |
| Env 是否包含 tool 调用？ | 是，Env/ToolEnv 负责 tool 调用和结果解析 |
| Reward 是否放在 Env 中？ | 否，Reward 独立于 Env |
| 是否重构 `MultiTurnRollout`？ | MVP 先 wrapper，后续再拆分 |
