# Agno 和 Mastra：生产级运行时

> Agno（Python）和 Mastra（TypeScript）是 2026 年的生产级运行时组合。Agno 旨在实现微秒级 Agent 实例化和无状态 FastAPI 后端。Mastra 基于 Vercel AI SDK 提供 Agent、工具、工作流、统一模型路由和组合式存储。

**类型：** 学习
**语言：** Python、TypeScript
**前置条件：** 第 14 阶段 · 01（Agent 循环）、第 14 阶段 · 13（LangGraph）
**时间：** 约 45 分钟

## 学习目标

- 了解 Agno 的性能目标及其适用场景。
- 说出 Mastra 的三个原语——Agent、工具、工作流——以及支持的服务器适配器。
- 解释为什么无状态会话作用域 FastAPI 后端是 Agno 推荐的生产路径。
- 在给定技术栈（Python 优先 vs TypeScript 优先）的情况下选择 Agno 还是 Mastra。

## 问题背景

LangGraph、AutoGen、CrewAI 属于重量级框架。希望"只要快速的 Agent 循环，在我的运行时中运行"的团队会转向 Agno（Python）或 Mastra（TypeScript）。两者都牺牲了一些框架内置的原语，以换取更快的速度和与周边技术栈更紧密的契合。

## 核心概念

### Agno

- Python 运行时，前身是 Phi-data。
- "没有图、链或复杂的模式——只有纯粹的 Python。"
- 来自官方文档的性能目标：约 2 微秒的 Agent 实例化、每个 Agent 约 3.75 KiB 内存、约 23 个模型提供商。
- 生产路径：无状态会话作用域 FastAPI 后端。每个请求启动一个新的 Agent；会话状态存储在数据库中。
- 原生多模态（文本、图像、音频、视频、文件）和 Agentic RAG。

速度目标在你每秒有数千个短生命周期 Agent 时（聊天汇聚、评估流水线）很重要。当一个 Agent 运行 10 分钟时，它就没那么重要了。

### Mastra

- 基于 TypeScript，构建在 Vercel AI SDK 之上。
- 三个原语：**Agent**、**工具**（Zod 类型化）、**工作流**。
- 统一模型路由器——2026 年 3 月已支持 94 个提供商的 3,300+ 模型。
- 组合式存储：内存、工作流、可观测性分别连接不同的后端；大规模可观测性推荐使用 ClickHouse。
- Apache 2.0 许可证，`ee/` 目录在源码可用的企业许可证下。
- 服务器适配器支持 Express、Hono、Fastify、Koa；一等公民的 Next.js 和 Astro 集成。
- 内置 Mastra Studio（localhost:4111）用于调试。
- 2026 年 1 月 1.0 版本发布时，GitHub 星标 22k+，每周 npm 下载 300k+。

### 定位对比

两者都不是要成为 LangGraph。它们在以下方面竞争：

- **语言契合度。** Agno 面向 Python 优先团队；Mastra 面向 TypeScript 优先团队。
- **运行时体验。** Agno = 近乎零开销；Mastra = 与 Vercel 生态系统深度集成。
- **可观测性。** 两者都与 Langfuse/Phoenix/Opik（第 24 课）集成，但 Mastra Studio 是官方原生的。

### 何时选择哪个

- **Agno** — Python 后端、大量短生命周期 Agent、强性能要求、FastAPI 技术栈。
- **Mastra** — TypeScript 后端、Next.js / Vercel 部署、统一多提供商模型路由、Zod 类型化工具。
- **LangGraph**（第 13 课）— 当持久化状态和显式图推理比原始速度更重要时。
- **OpenAI / Claude Agent SDK** — 当你想要提供商的标准化形态时（第 16–17 课）。

### 常见错误

- **为性能而性能。** 选择 Agno 仅因为"2 微秒"听起来不错，而实际负载是每个请求一次慢速 Agent 调用。开销不是瓶颈。
- **生态系统锁定。** Mastra 的 Vercel 风格集成在 Vercel 上是优势，在其他地方则是劣势。
- **企业许可证混淆。** Mastra 的 `ee/` 目录是源码可用，而非 Apache 2.0。如果打算分叉，请阅读许可证。

## 构建实践

本课主要是对比性质——没有单一代码产物能同时公正地展示两个框架。参见 `code/main.py` 中的并行示例：一个最小化的"运行 Agent、流式输出、持久化会话"流程，分别用 Agno 风格和 Mastra 风格实现两次。

运行方式：

```
python3 code/main.py
```

两个结构不同但功能等价的追踪。

## 实际应用

- **Agno** — 需要速度和 FastAPI 形态的 Python 后端。
- **Mastra** — 拥有多个提供商和工作流原语的 TypeScript 后端。
- 两者都提供官方原生可观测性钩子。两者都与 Langfuse 集成。

## 交付上线

`outputs/skill-runtime-picker.md` 根据技术栈、延迟预算和运维形态来选择 Agno、Mastra、LangGraph 或提供商 SDK。

## 练习

1. 阅读 Agno 的文档。将标准库 ReAct 循环（第 01 课）移植到 Agno。哪些东西消失了？哪些保留了？
2. 阅读 Mastra 的文档。将同样的循环移植到 Mastra。工具类型化方面有什么变化（Zod vs 无）？
3. 基准测试：在你的技术栈上测量 Agent 实例化延迟。Agno 的 2 微秒对你的负载重要吗？
4. 设计一个迁移方案：如果你一直在 Python 中使用 CrewAI，迁移到 Agno 会破坏什么？
5. 阅读 Mastra 的 `ee/` 许可条款。哪些限制会影响开源分叉？

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|----------|----------|
| Agno | "快速 Python Agent" | 无状态会话作用域 Agent 运行时 |
| Mastra | "基于 Vercel AI SDK 的 TypeScript Agent" | Agent + 工具 + 工作流 + 模型路由器 |
| 统一模型路由器 | "多提供商访问" | 单一客户端访问 94 个提供商的 3,300+ 模型 |
| 组合式存储 | "多个后端" | 内存/工作流/可观测性各自连接不同的存储 |
| Mastra Studio | "本地调试器" | localhost:4111 用于自省 Agent 的 UI |
| 源码可用 | "非开源" | 许可证允许查看源码但限制商业使用 |

## 延伸阅读

- [Agno Agent Framework 文档](https://www.agno.com/agent-framework) — 性能目标、FastAPI 集成
- [Mastra 文档](https://mastra.ai/docs) — 原语、服务器适配器、模型路由器
- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview) — 有状态图的替代方案
- [Comet Opik](https://www.comet.com/site/products/opik/) — Mastra 集成引用的可观测性对比
