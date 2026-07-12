# Agno 与 Mastra 生产运行时

> 日期: 2026-07-12

## 1. 课程定位：为什么需要生产运行时

Lesson 01 手写了 500 行 stdlib ReAct loop，Lesson 18 告诉你：Agno 把这 500 行压缩到 20 行。框架省掉的是 **MessageBuffer、ToolRegistry、Provider 序列化、停止条件检测、trace 记录** — 这些是工程样板，不是业务逻辑。

先学过 Lesson 01 再学这节，才能判断框架省的是"该省的"还是"不该省的"。

## 2. Agno — Python 极速 Agent 运行时

| 特性 | 详情 |
|---|---|
| 前身 | Phidata（2025 年初改名重定位） |
| 核心理念 | 无图、无链、无 DSL，纯 Python 对象 |
| 性能 | ~2μs 实例化、~6.5 KiB 内存/agent、23+ 模型提供商 |
| 生产路径 | **无状态 FastAPI 后端** — 每个请求创建新 agent，session 状态存 DB |
| 多模态 | 原生支持文本、图像、音频、视频输入输出 |
| 多 Agent | `Team` 对象 — 无需画图，直接 `members=[agent1, agent2]` |
| 运行时 | AgentOS（FastAPI 封装），提供 `/chat`、`/sessions`、`/health` |

**适用场景**：Python 后端、FastAPI 技术栈、需要大量短命 agent 的高并发场景。

## 3. Mastra — TypeScript 全栈 Agent 框架

| 特性 | 详情 |
|---|---|
| 底层 | 构建在 **Vercel AI SDK** 之上 |
| 三大原语 | **Agents**（自主推理）、**Tools**（Zod 类型校验）、**Workflows**（确定性多步编排） |
| 模型路由 | 统一路由器 — 3,300+ 模型、94 个提供商 |
| Server 适配 | Express、Hono、Fastify、Koa；一等公民 Next.js / Astro 集成 |
| 调试 | Mastra Studio（localhost:4111）— 本地交互式 UI |
| 许可证 | Apache 2.0 + `ee/` 目录下的 source-available 企业版 |

**适用场景**：TypeScript 后端、Next.js/Vercel 部署、需要统一多 provider 路由和 Zod 类型工具的团队。

## 4. 核心区别对比

| 维度 | Agno | Mastra |
|---|---|---|
| 语言 | Python | TypeScript |
| 核心模型 | 无状态 agent 对象 | Agent + Tool + Workflow 三原语 |
| 性能卖点 | ~2μs 实例化（短命 agent） | Vercel 生态集成（全栈体验） |
| 生产部署 | FastAPI (AgentOS) | Express/Hono/Next.js 适配器 |
| 状态管理 | session → DB | Composite Storage（可插拔） |
| 可观测性 | 集成 Langfuse/Phoenix | Mastra Studio（一等公民） |

## 5. 移植 ReAct Loop 到 Agno 后发生了什么

### 消失了（Agno 内部处理）

- `MessageBuffer` + provider 序列化 → Agent 内部管理
- `ToolRegistry.register()` → `tools=[fn]` 列表直传
- `AgentLoop` while 循环 → `agent.run()` 内部循环
- 手动 stop condition 检测 → 内置（model 结束 / 无 tool call / max turns）
- `TraceEntry` 手动记录 → 内置 tracing，可对接 Langfuse
- Anthropic/OpenAI schema 区分 → 统一接口，模型层抽象
- 手动 `tool_use_id` 关联 → 框架自动处理

### 留下了（你自己仍需关心）

- **工具函数**：仍然是普通 Python 函数，签名不变
- **系统 prompt / instructions**：你定义 agent 行为
- **session_id**：你决定何时创建/复用 session
- **选模型**：你挑 provider + model ID
- **评估输出**：你读 response.content 并判断

## 6. 选型决策：不止二选一

课程的决策表列了 8 条路径，Agno 和 Mastra 只是其中两个选项：

| 条件 | 选择 |
|---|---|
| Python + FastAPI + 大量短命 agent | Agno |
| TypeScript + Next.js/Vercel + 多 provider | Mastra |
| 需要持久化状态 + 显式图 | LangGraph |
| Claude-first + 最大控制权 | Claude Agent SDK |
| OpenAI-first + guardrails | OpenAI Agents SDK |

**选型本质**：根据技术栈、性能需求、运维形态从多个框架里挑一个，而不是在 Agno 和 Mastra 之间二选一。

## 7. 课程三层教学逻辑

1. **为什么需要生产运行时**：省掉工程样板，专注业务逻辑
2. **框架帮你做了什么，藏了什么**：理解便利背后的抽象代价
3. **选型不是二选一，而是八选一**：根据实际场景匹配最佳方案

## 8. 关键概念速查

| 术语 | 含义 |
|---|---|
| Agno | 无状态 session-scoped Python agent 运行时 |
| Mastra | Agents + Tools + Workflows + Model Router 的 TypeScript 框架 |
| Vercel | 前端云平台：部署托管 + Serverless 运行时 + AI SDK |
| Langfuse | LLM 可观测性平台：tracing、evals、prompt 管理、成本监控 |
| Composite Storage | Mastra 的 memory/workflows/observability 各自连接不同后端 |
| Mastra Studio | localhost:4111 UI，本地调试 agent 的交互式工具 |
| Source-available | 许可证允许阅读源码但限制商业使用（非 Apache 2.0） |

## 9. 待消化的问题

- Agno 的 2μs 实例化在真实 LLM 调用（数百毫秒到秒级）面前是否真的有意义？什么场景下框架开销才是瓶颈？
- Mastra 的 `ee/` 许可证对开源 fork 的具体限制是什么？
- 如果已有 CrewAI 代码，迁移到 Agno 会断裂哪些东西？
