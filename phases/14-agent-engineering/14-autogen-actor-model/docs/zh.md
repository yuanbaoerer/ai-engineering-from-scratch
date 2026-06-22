# AutoGen v0.4：Actor 模型与 Agent 框架

> AutoGen v0.4（Microsoft Research，2025年1月）围绕 actor 模型重新设计了 agent 编排。异步消息交换、事件驱动 agent、故障隔离、天然并发。该框架目前已进入维护模式，Microsoft Agent Framework（2025年10月公开预览）将接替其成为继任者。

**类型：** Learn + Build
**语言：** Python (stdlib)
**前置课程：** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**时间：** ~75 分钟

## 学习目标

- 描述 actor 模型：agent 作为 actor，消息作为唯一的 IPC，每个 actor 独立的故障隔离。
- 列出 AutoGen v0.4 的三个 API 层——Core、AgentChat、Extensions——以及各自的用途。
- 解释为什么将消息投递与处理解耦可以带来故障隔离和天然并发。
- 用 Python stdlib 实现一个 actor 运行时，并将双 agent 代码审查流程移植到其上。

## 问题所在

大多数 agent 框架是同步的：一个 agent 生产，一个 agent 消费，在同一个调用栈中。失败会导致整个栈崩溃。并发是后加的。分布式则需要重写。

AutoGen v0.4 的答案：actor 模型。每个 agent 都是一个拥有私有收件箱的 actor。消息是唯一的交互方式。运行时将投递与处理解耦。故障隔离在单个 actor 内。并发是原生的。分布式只是换了传输方式。

## 核心概念

### Actor

一个 actor 拥有：

- 私有状态（外部永远不能直接访问）。
- 收件箱（消息队列）。
- 处理函数：`receive(message) -> effects`，其中 effects 可以是"回复"、"发送给其他 actor"、"生成新 actor"、"更新状态"、"停止自身"。

两个 actor 不能共享内存。它们只能发送消息。

### AutoGen v0.4 的三个 API 层

1. **Core。** 底层 actor 框架。`AgentRuntime`、`Agent`、`Message`、`Topic`。异步消息交换，事件驱动。
2. **AgentChat。** 任务驱动的高级 API（替代 v0.2 的 `ConversableAgent`）。`AssistantAgent`、`UserProxyAgent`、`RoundRobinGroupChat`、`SelectorGroupChat`。
3. **Extensions。** 集成——OpenAI、Anthropic、Azure、工具、记忆。

### 为什么解耦很重要

在 v0.2 模型中，调用 `agent_a.chat(agent_b)` 会同步阻塞 agent_a，直到 agent_b 返回。在 v0.4 中，`send(agent_b, msg)` 将消息放入 agent_b 的收件箱并立即返回。运行时稍后投递。有三个后果：

- **故障隔离。** Agent B 崩溃不会导致 Agent A 崩溃——运行时捕获 B 处理函数中的失败，并决定如何处理（记录日志、重试、死信队列）。
- **天然并发。** 多条消息同时在途；actor 并发处理各自的收件箱。
- **分布式就绪。** 收件箱 + 传输层的抽象是相同的，无论 actor 是在进程中还是在另一台主机上。

### 拓扑结构

- **RoundRobinGroupChat。** Agent 按固定轮转顺序依次发言。
- **SelectorGroupChat。** 选择器 agent 根据对话上下文决定谁下一个发言。
- **Magentic-One。** 面向网页浏览、代码执行、文件处理的参考多 agent 团队。基于 AgentChat 构建。

### 可观测性

内置 OpenTelemetry 支持。每条消息都发出一个 span；工具调用携带 `gen_ai.*` 属性，遵循 2026 年 OTel GenAI 语义约定（第 23 课）。

### 状态：维护模式

2026 年初：AutoGen v0.7.x 对于研究和原型开发是稳定的。Microsoft 已将活跃开发转向 Microsoft Agent Framework（2025 年 10 月公开预览；1.0 GA 目标在 2026 年 Q1 末）。AutoGen 的模式可以平滑移植——actor 模型是一个持久的理念。

## 动手构建

`code/main.py` 实现了一个 stdlib actor 运行时：

- `Message` — 带有 `sender`、`recipient`、`topic`、`body` 的类型化消息载荷。
- `Actor` — 抽象类，包含 `receive(message, runtime)` 方法。
- `Runtime` — 事件循环，带有共享队列、投递机制、故障隔离。
- 一个双 actor 演示：`ReviewerAgent` 审查代码，`ChecklistAgent` 运行检查清单；它们交换消息直到达成共识。

运行：

```
python3 code/main.py
```

追踪输出展示了消息投递、一个 actor 中的模拟故障（未导致另一个 actor 崩溃），以及最终达成共识的过程。

## 实际应用

- **AutoGen v0.4/v0.7**（维护中）——适用于研究、原型开发、多 agent 模式。
- **Microsoft Agent Framework**（公开预览）——未来方向；相同的 actor 模型理念，全新的 API。
- **LangGraph swarm 拓扑**（第 13 课）——通过共享工具切换实现类似模式。
- **自定义 actor 运行时**——当你需要特定传输方式（NATS、RabbitMQ、gRPC）时使用。

## 交付物

`outputs/skill-actor-runtime.md` 为给定的多 agent 任务生成一个最小 actor 运行时以及团队模板（RoundRobin 或 Selector）。

## 练习

1. 添加死信队列：当处理函数抛出异常时，将失败消息暂存以供人工检查。你的小测试中 DLQ 被触发的频率如何？
2. 实现 `SelectorGroupChat`：一个选择器 actor 根据对话状态选择谁处理下一条消息。
3. 添加分布式传输：将进程内队列替换为基于 HTTP 的 JSON 服务器，使 actor 可以在不同进程中运行。
4. 为每条消息接入 OTel span（或空操作替代）。按第 23 课发出 `gen_ai.agent.name`、`gen_ai.operation.name`。
5. 阅读 AutoGen v0.4 的架构文章。将你的小测试移植到真实的 `autogen_core` API。你跳过了哪些在生产中很重要的东西？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Actor | "Agent" | 私有状态 + 收件箱 + 处理函数；不共享内存 |
| Message | "Event" | 类型化载荷；actor 之间唯一的交互方式 |
| Inbox | "Mailbox" | 每个 actor 的待处理消息队列 |
| Runtime | "Agent host" | 路由消息并隔离故障的事件循环 |
| Topic | "Channel" | actor 之间的命名发布-订阅路由 |
| Fault isolation | "Let it crash" | 一个 actor 失败不会导致其他 actor 崩溃 |
| RoundRobinGroupChat | "Fixed-rotation team" | Agent 按顺序轮流发言 |
| SelectorGroupChat | "Context-routed team" | 选择器决定谁下一个发言 |
| Magentic-One | "Reference team" | 面向网页 + 代码 + 文件的多 agent 小队 |

## 延伸阅读

- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 重新设计文章
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 图状替代方案
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — AutoGen 默认发出的 span
