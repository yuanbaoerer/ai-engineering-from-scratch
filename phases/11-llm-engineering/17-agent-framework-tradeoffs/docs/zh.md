# Agent 框架权衡 — LangGraph vs CrewAI vs AutoGen vs Agno

> 每个框架都卖同样的演示（研究 agent 生成报告），也都藏着同样的 bug（状态模式与编排层打架）。选那个抽象模型与你的问题形态相匹配的框架；其余一切都是你写两遍的胶水代码。

**类型：** Learn
**语言：** Python
**前置要求：** Phase 11 · 09 (Function Calling)、Phase 11 · 16 (LangGraph)
**时间：** 约 45 分钟

## 问题

你有一个需要不止一次 LLM 调用才能完成的任务。它可能是一个研究工作流（规划、搜索、总结、引用）。可能是一个代码审查管道（解析 diff、批评、打补丁、验证）。可能是一个多轮助手，能预订机票、撰写邮件、提交报销单。你选了一个框架。

三天后，你发现框架的抽象漏了。 CrewAI 给你角色，但当"研究员"需要把一个结构化计划交给"写作者"时，它跟你对着干。 AutoGen 给你 agent 之间的聊天，但没有原生状态，所以你的检查点是一个对话日志的 pickle 快照。 LangGraph 给你状态图，但逼你在知道 agent 会做什么之前就给每个转换命名。 Agno 给你一个单 agent 抽象，但当你试图分叉到三个并发 worker 时，它就尖叫。

解决方案不是"选最好的框架"。而是让框架的核心抽象与你问题的形态相匹配。本课绘制那张地图。

## 概念

![Agent 框架矩阵：核心抽象 vs 问题形态](../assets/framework-matrix.svg)

2026 年的格局中，四个框架占据主导地位。它们的核心理念不同。

| 框架 | 核心抽象 | 最适合 | 最不适合 |
|------------|------------------|----------|-----------|
| **LangGraph** | `StateGraph` — 类型化状态、节点、条件边、检查点。 | 有显式状态和人在回路中断的工作流；需要时间旅行调试的生产级 agent。 | 松散的、角色驱动的头脑风暴，拓扑未知。 |
| **CrewAI** | `Crew` — 角色（目标、背景故事）、任务、流程（顺序或层级）。 | 角色扮演或以人物形象驱动的短线性/层级计划工作流。 | Crew 轮次历史之外的有状态操作；复杂分支。 |
| **AutoGen** | `ConversableAgent` 对 — 两个或多个 agent 轮流对话直到满足退出条件。 | 多 agent *对话*（师生、提议者-批评者、演员-评论者），思维从聊天中涌现。 | 已知 DAG 的确定性工作流；需要跨重启持久状态的操作。 |
| **Agno** | `Agent` — 单个 LLM + 工具 + 内存，可组合成团队。 | 快速构建的单 agent 和轻量级团队；强大的多模态能力和内置存储驱动。 | 带自定义 reducer 的深度显式分支图。 |

### "抽象"实际上意味着什么

框架的核心抽象，就是你在演示架构时在白板上画的那个东西。

- **LangGraph** → 你画一个图。节点是步骤，边是转换，每个点的状态对象都是类型化的。思维模型是一个状态机。
- **CrewAI** → 你画一张组织图。每个角色有工作描述，管理者路由任务。思维模型是一组专家小团队。
- **AutoGen** → 你画一个 Slack 私信。两个 agent 互相发消息；如果需要moderator，第三个加入。思维模型是聊天。
- **Agno** → 你画一个带有挂载工具的单个盒子。把盒子并排放置就成为一个团队。思维模型是"带电池的 agent"。

### 状态问题

状态是大多数框架选择在生产中崩溃的地方。

- **LangGraph。** 类型化状态（`TypedDict` 或 Pydantic 模型），逐字段 reducer，原生检查点（SQLite/Postgres/Redis）。恢复、中断和时间旅行是开箱即用的。*（见 Phase 11 · 16。）*
- **CrewAI。** 状态通过 `context` 字段在任务之间作为字符串流动，或通过 `output_pydantic` 结构化。内置没有持久化每个 crew 的存储；如果 crew 需要在重启后存活，你自己接上。
- **AutoGen。** 状态是聊天历史和任何用户定义的 `context`。对话记录持久化；任意工作流状态不会持久化，除非你写适配器。
- **Agno。** 内置存储驱动（SQLite、Postgres、Mongo、Redis、DynamoDB）通过 `storage=` 附加到 `Agent` — 对话会话和用户记忆自动持久化。不是完整的图检查点；而是会话存储。

### 分支问题

每个非平凡的 agent 都会分支。谁决定分支很重要。

- **LangGraph** — 由你决定，通过条件边。路由是一个带命名分支的 Python 函数。分支是编译图中的一等公民；检查点记录了走了哪个分支。
- **CrewAI** — 在层级模式下由管理者决定；在顺序模式下你在构建时决定。路由隐含在任务列表中；在管理者的 prompt 之外没有原生的"if"。
- **AutoGen** — 由 agent 通过聊天决定。分支从谁下一个发言中涌现。`GroupChatManager` 选择下一个发言者；你可以手写 `speaker_selection_method`，但默认是 LLM 驱动的。
- **Agno** — 由 agent 决定下一步调用哪个工具决定。团队有 coordinator/router/collaborator 模式；除此之外的分支由开发者负责。

### 可观测性问题

- **LangGraph** — 通过 LangSmith 或任何 OTel 导出器实现 OpenTelemetry。每个节点转换都是一个 trace span；检查点同时也是可重放的 trace。LangSmith 是第一方选项；Langfuse/Phoenix 也有适配器。
- **CrewAI** — 自 2025 年底起原生支持 OpenTelemetry；与 Langfuse、Phoenix、Opik、AgentOps 集成。
- **AutoGen** — 通过 `autogen-core` 集成 OpenTelemetry；AgentOps 和 Opik 有连接器。追踪粒度是每个 agent 消息级别，不是每个节点。
- **Agno** — 内置 `monitoring=True` 标志加 OpenTelemetry 导出器；与 Langfuse 深度集成用于会话追踪。

### 成本和延迟

所有四个框架都增加了每次调用的开销（框架逻辑、验证、序列化）。开销大致递增顺序：Agno ≈ LangGraph < CrewAI ≈ AutoGen。差异主要取决于框架做了多少额外的 LLM 路由。 CrewAI 的层级管理者花费 token 决定谁下一个；AutoGen 的 `GroupChatManager` 同样如此。 LangGraph 只在你写 `llm.invoke` 的地方花 token。 Agno 的单 agent 路径很薄。

当每次运行的成本重要时，优先选择显式路由（LangGraph 边、AutoGen `speaker_selection_method`）而非 LLM 选择的路由。

### 互操作性

- **LangGraph** ↔ **LangChain** 工具、检索器、LLM。原生的 MCP 适配器（工具作为 MCP 服务器导入）。
- **CrewAI** ↔ 工具继承自 `BaseTool`；LangChain 工具、LlamaIndex 工具和 MCP 工具都可以适配接入。 Crew-to-crew 委托通过 `allow_delegation=True`。
- **AutoGen** → `FunctionTool` 包装任何 Python 可调用对象；MCP 适配器可用。与 AG2 生态系统深度耦合，用于 agent-to-agent 模式。
- **Agno** → `@tool` 装饰器或 BaseTool 子类；MCP 适配器；工具可以在 agent 和团队之间共享。

## 技能

> 你能用一句话解释，为什么一个给定的框架适合一个给定的 agent 问题。

预构建检查清单：

1. **画出形态。** 这是一个图（类型化状态、命名转换）？一场角色扮演（专家交接工作）？一场聊天（agent 对话直到完成）？还是一个带工具的单 agent？
2. **决定谁分支。** 开发者决定分支 → LangGraph。管理 agent 决定 → CrewAI 层级。聊天涌现 → AutoGen。工具调用决定 → Agno。
3. **检查状态预算。** 你需要从检查点恢复？时间旅行？人在运行中途中断？如果是，LangGraph 是默认选项；Agno 会话覆盖会话作用域的状态。
4. **检查成本预算。** LLM 选择的路由每次轮次花费额外 token。如果 agent 每天运行数千次，优先选择显式路由。
5. **计算框架开销。** 每个框架都是另一个依赖。如果任务只是两次 LLM 调用和一个工具，写 30 行纯 Python；没有框架比没有框架更便宜。

在你画不出图、组织图、聊天或 agent 盒子之前，不要伸手拿框架。不要选一个强迫你为实际需要的东西与它的状态模型搏斗的框架。

## 决策矩阵

| 问题形态 | 推荐框架 | 原因 |
|---------------|---------------------|-----|
| 带类型化状态、人工审批、长期运行的 DAG 工作流 | LangGraph | 原生状态、检查点、中断、时间旅行。 |
| 带明确角色的研究/写作管道 | CrewAI（顺序）或 LangGraph 子图 | CrewAI 中角色-per-任务 表达成本低；当分支变复杂时用 LangGraph 扩展。 |
| 提议者-批评者或师生对话 | AutoGen | 双 agent 聊天是其原生形态。 |
| 带工具、会话、记忆的单 agent | Agno | 最薄的设置，内置存储和记忆。 |
| 带 reducer 的数千个并行分叉 | LangGraph + `Send` | 唯一一个有一等公民并行调度 API 的。 |
| 快速原型，无框架承诺 | 纯 Python + provider SDK | 没有框架是最快的框架。 |

## 练习

1. **简单。** 用同一个任务 — "研究 Anthropic 总部，写一份 200 字的简报，引用来源" — 分别用 LangGraph（四个节点：plan、search、write、cite）和 CrewAI（三个角色：researcher、writer、editor）实现它。报告每次运行的 token 成本和代码行数。
2. **中等。** 用 AutoGen（researcher ↔ writer 聊天，editor 通过 `GroupChat` 加入）和 Agno（带 `search_tools` 和 `write_tools` 的单 agent，加会话存储）构建相同任务。将四个实现按以下维度排名：(a) 每次运行的成本，(b) 崩溃后恢复能力，(c) 在 write 步骤前注入人工审批的能力。
3. **困难。** 构建一个决策树脚本 `pick_framework.py`，接受简短的问题描述（JSON：`{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`）并返回一个建议和一句话的理由。在你自己设计的六个案例上验证它。

## 关键术语

| 术语 | 人们怎么说 | 实际意味着什么 |
|------|-----------------|-----------------------|
| 编排 | "Agent 如何协调" | 决定哪个节点/角色/agent 下一个运行的那一层。 |
| 持久状态 | "重启后恢复" | 能跨进程死亡存活的状态，附属于检查点或会话存储。 |
| LLM 选择的路由 | "让模型决定" | 一个规划 LLM 每轮选择下一步；灵活但每次决策都要花 token。 |
| 显式路由 | "开发者决定" | 一个 Python 函数或静态边选择下一步；便宜且可审计。 |
| Crew | "一个 CrewAI 团队" | 角色 + 任务 + 流程（顺序或层级）绑定成单个可运行单元。 |
| GroupChat | "AutoGen 的多 agent 聊天" | 在 N 个 agent 之间由发言者选择器管理的会话。 |
| Team (Agno) | "多 agent Agno" | 在一组 agent 之上的路由 / 协调 / 协作模式。 |
| StateGraph | "LangGraph 的图" | 类型化状态、节点、条件边、检查点抽象。 |

## 延伸阅读

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) — StateGraph、检查点、中断、时间旅行。
- [CrewAI 文档](https://docs.crewai.com/) — Crews、Flows、Agents、Tasks、Processes。
- [AutoGen 文档](https://microsoft.github.io/autogen/) — ConversableAgent、GroupChat、teams、tools。
- [Agno 文档](https://docs.agno.com/) — Agent、Team、Workflow、storage、memory。
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — 模式库（prompt chaining、routing、parallelization、orchestrator-workers、evaluator-optimizer），框架无关。
- [Yao et al., "ReAct: Synergizing Reasoning and Acting" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — 每个框架都在包装的循环。
- [Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (2023)](https://arxiv.org/abs/2308.08155) — AutoGen 的设计论文。
- [Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (UIST 2023)](https://arxiv.org/abs/2304.03442) — CrewAI 风格 persona 栈所基于的角色扮演基础。
- Phase 11 · 16 (LangGraph) — 本课对比的框架。
- Phase 11 · 19 (Reflexion) — 一个能干净地映射到 LangGraph 但别扭地映射到 CrewAI 的模式。
- Phase 11 · 22 (生产可观测性) — 如何为你选择的框架添加工具。