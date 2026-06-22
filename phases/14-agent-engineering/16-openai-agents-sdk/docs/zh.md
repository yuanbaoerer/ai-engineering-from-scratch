# OpenAI Agents SDK: 交接、护栏、追踪

> OpenAI Agents SDK 是一个基于 Responses API 的轻量级多智能体框架。包含五个基本元素：Agent（智能体）、Handoff（交接）、Guardrail（护栏）、Session（会话）、Tracing（追踪）。交接是以 `transfer_to_<agent>` 命名的工具。护栏在输入或输出时触发。追踪默认开启。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置条件：** 阶段 14 · 01（智能体循环）、阶段 14 · 06（工具使用）
**时间：** 约 75 分钟

## 学习目标

- 说出 OpenAI Agents SDK 的五个基本元素。
- 解释交接：为什么它们被建模为工具、模型看到的名称形式以及上下文如何转移。
- 区分输入护栏、输出护栏和工具护栏；解释 `run_in_parallel` 与阻塞模式。
- 实现一个带有交接、护栏和 span 风格追踪的标准库运行时。

## 问题

无法清晰委托的智能体最终会将所有内容塞进一个提示中。没有护栏的智能体会泄露个人身份信息、违反政策的输出，或无限循环。OpenAI 的 SDK 将使多智能体工作可行的三个基本元素规范化。

## 概念

### 五个基本元素

1. **智能体。** LLM + 指令 + 工具 + 交接。
2. **交接。** 向另一个智能体的委托。在模型中表示为名为 `transfer_to_<agent_name>` 的工具。
3. **护栏。** 在输入（仅第一个智能体）、输出（仅最后一个智能体）或工具调用（每个函数工具）上进行验证。
4. **会话。** 跨轮次的自动对话历史。
5. **追踪。** 内置 span 用于 LLM 生成、工具调用、交接、护栏。

### 交接作为工具

模型在其工具列表中看到 `transfer_to_billing_agent`。调用它会通知运行时：

1. 复制对话上下文（或通过 `nest_handoff_history` beta 功能折叠它）。
2. 使用其指令初始化目标智能体。
3. 使用目标智能体继续运行。

这是监督者模式（第 13 课 / 第 28 课）的产品化。

### 护栏

三种类型：

- **输入护栏。** 在第一个智能体的输入上运行。在任何 LLM 调用之前拒绝不安全或超出范围的请求。
- **输出护栏。** 在最后一个智能体的输出上运行。捕获个人身份信息泄露、政策违规、格式错误的响应。
- **工具护栏。** 在每个函数工具上运行。验证参数、检查权限、审计执行。

模式：

- **并行**（默认）。护栏 LLM 与主 LLM 并行运行。尾部延迟较低。如果触发，主 LLM 的工作将被丢弃（浪费 token）。
- **阻塞**（`run_in_parallel=False`）。护栏 LLM 先运行。如果触发，不会在主调用上浪费 token。

触发器抛出 `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`。

### 追踪

默认开启。每个 LLM 生成、工具调用、交接和护栏都会发出一个 span。`OPENAI_AGENTS_DISABLE_TRACING=1` 可退出。`add_trace_processor(processor)` 将 span 同时发送到您自己的后端和 OpenAI。

### 会话

`Session` 将对话历史存储在后端（SQLite、Redis、自定义）。`Runner.run(agent, input, session=session)` 自动加载并追加。

### 此模式可能出现的问题

- **交接漂移。** 智能体 A 交接给智能体 B，智能体 B 又交还给智能体 A。添加跳数计数器。
- **护栏绕过。** 工具护栏仅在函数工具上触发；内置工具（文件读取器、网络获取）需要单独的策略。
- **过度追踪。** span 中的敏感内容。配合 OTel GenAI 内容捕获规则（第 23 课）——外部存储，通过 ID 引用。

## 构建它

`code/main.py` 在标准库中实现了 SDK 形状：

- `Agent`、`FunctionTool`、`Handoff`（作为具有传递语义的函数工具）。
- `Runner` 带有输入/输出/工具护栏、交接调度和跳数计数器。
- 一个简单的 span 发射器以显示追踪形状。
- 一个分类智能体，根据用户的查询交接给计费或支持；护栏在一个输入上触发。

运行它：

```bash
python3 code/main.py
```

追踪显示两次成功的交接，一次输入护栏触发，以及一个反映真实 SDK 发出的 span 树。

## 使用它

- **OpenAI Agents SDK** 用于 OpenAI 优先的产品。
- **Claude Agent SDK**（第 17 课）用于 Claude 优先的产品。
- **LangGraph**（第 13 课）当您需要显式状态和持久恢复时。
- **自定义**当您需要精确控制（语音、多提供商、联邦部署）时。

## 发布它

`outputs/skill-agents-sdk-scaffold.md` 为一个带有分类智能体、交接、输入/输出/工具护栏、会话存储和跟踪处理器的 Agents SDK 应用提供了脚手架。

## 练习

1. 添加交接跳数计数器：在 N 次转移后拒绝。追踪行为。
2. 将 `nest_handoff_history` 实现为一个选项——在转移之前将先前的消息折叠成一个摘要。
3. 编写一个阻塞输出护栏。比较会触发它的提示与通过的提示的延迟。
4. 将 `add_trace_processor` 连接到 JSON 日志记录器。它每个 span 发出什么形状？
5. 阅读 SDK 文档。将您的标准库玩具移植到 `openai-agents-python`。您建模错了什么？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Agent（智能体） | "LLM + 指令" | SDK 中的智能体类型；拥有工具和交接 |
| Handoff（交接） | "转移" | 模型调用以委托给另一个智能体的工具 |
| Guardrail（护栏） | "策略检查" | 对输入/输出/工具调用的验证 |
| Tripwire（触发器） | "护栏触发" | 护栏拒绝时抛出的异常 |
| Session（会话） | "历史存储" | 在运行之间持久化的对话记忆 |
| Tracing（追踪） | "span" | 内置的可观测性，覆盖 LLM + 工具 + 交接 + 护栏 |
| Blocking guardrail（阻塞护栏） | "顺序检查" | 护栏先运行；触发时不会浪费 token |
| Parallel guardrail（并行护栏） | "并发检查" | 护栏并行运行；延迟较低，触发时会浪费 token |

## 进一步阅读

- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/) — 基本元素、交接、护栏、追踪
- [Claude Agent SDK 概述](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude 风格的对应物
- [Anthropic，构建有效的智能体](https://www.anthropic.com/research/building-effective-agents) — 何时需要交接
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — Agents SDK span 映射的标准