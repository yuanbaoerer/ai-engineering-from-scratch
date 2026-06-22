# 生产运行时：队列、事件、定时任务

> 生产级智能体有六种运行时形态：请求-响应、流式、持久化执行、基于队列的后台、事件驱动和定时调度。先选形态，再选框架。在每种形态中，可观测性都是关键支撑。

**类型：** 学习
**语言：** Python（标准库）
**前置课程：** 第14阶段 · 13（LangGraph）、第14阶段 · 22（语音）
**时间：** ~60分钟

## 学习目标

- 列出六种生产运行时形态，并将每种形态与框架/产品模式对应起来。
- 解释为什么持久化执行（LangGraph）对长周期任务很重要。
- 描述事件驱动运行时以及Claude Managed Agents的适用场景。
- 解释可观测性对多步骤智能体而言为何是关键支撑。

## 问题背景

生产级智能体会以Jupyter notebook无法暴露的方式失败：第37步网络超时、用户在语音通话中途挂断、定时任务在机器重启时崩溃、后台工作进程内存耗尽。运行时形态决定了哪些故障是可以恢复的。

## 核心概念

### 请求-响应

- 同步HTTP。用户等待完成。
- 仅适用于短任务（<30秒）。
- 技术栈：Agno（Python + FastAPI）、Mastra（TypeScript + Express/Hono/Fastify/Koa）。
- 可观测性：标准HTTP访问日志 + OTel span。

### 流式

- 通过SSE或WebSocket实现渐进式输出。
- LiveKit将其扩展为WebRTC，用于语音/视频（第22课）。
- 技术栈：任何支持流式的框架 + 处理SSE/WS的前端。
- 可观测性：逐块计时、首token延迟、尾部延迟。

### 持久化执行

- 每步之后状态自动检查点；失败时自动恢复。
- AutoGen v0.4的Actor模型将故障隔离到单个智能体（第14课）。
- LangGraph的核心差异化能力（第13课）。
- 当步骤数未知且恢复代价高昂时，这是必需的。

### 基于队列的后台执行

- 任务进入队列，工作进程领取执行，结果通过webhook或pub/sub回传。
- 对长周期智能体至关重要（每个任务数十到数百步，参见Anthropic computer use公告）。
- 技术栈：Celery（Python）、BullMQ（Node）、SQS + Lambda（AWS）、自定义方案。
- 可观测性：队列深度、每任务延迟分布、死信队列大小。

### 事件驱动

- 智能体订阅触发器：新邮件、PR创建、定时触发。
- Claude Managed Agents开箱支持此模式（第17课）。
- CrewAI Flows（第15课）构建事件驱动的确定性工作流。
- 可观测性：触发源、事件到启动的延迟、智能体执行延迟。

### 定时调度

- Cron形态的智能体，周期性运行。
- 结合持久化执行，使失败的夜间任务在下次调度时恢复。
- 技术栈：Kubernetes CronJob + 持久化框架；托管方案（Render cron、Vercel cron）。

### 2026年部署模式

- **CrewAI Flows** 用于事件驱动的生产环境。
- **Agno** 无状态FastAPI用于Python微服务。
- **Mastra** 服务器适配器（Express、Hono、Fastify、Koa）用于嵌入式部署。
- **Pipecat Cloud / LiveKit Cloud** 用于托管语音（第22课）。
- **Claude Managed Agents** 用于托管的长周期异步执行。

### 可观测性是关键支撑

如果没有OpenTelemetry GenAI span（第23课）加上Langfuse/Phoenix/Opik后端（第24课），你无法调试一个在第40步失败的多步骤智能体。在生产环境中这不是可选项。它决定了"快速调试"和"加更多日志从头重跑"之间的区别。

### 生产运行时常见故障

- **形态选择错误。** 对一个5分钟的任务选择了请求-响应。用户挂断；工作进程堆积；重试不断叠加。
- **无死信队列。** 队列工作进程没有死信机制。失败的任务凭空消失。
- **后台任务不透明。** 后台智能体运行但不导出trace。故障不可见，直到用户报告。
- **跳过持久化状态。** 任何超过30秒且无法承受重启的任务都需要持久化执行。

## 动手构建

`code/main.py` 是一个基于标准库的多形态演示：

- 请求-响应端点（普通函数）。
- 流式处理器（生成器）。
- 带死信队列的基于队列的工作进程。
- 事件触发器注册表。
- Cron形态的调度器。

运行方式：

```bash
python3 code/main.py
```

输出：五条trace，展示每种形态在同一任务上的行为。相同的智能体逻辑，不同的外壳。持久化执行（第六种形态）有意在第13课中通过LangGraph检查点来覆盖。

## 实际应用

- **请求-响应** 用于聊天式用户体验。
- **流式** 用于渐进式响应。
- **持久化** 用于长周期任务。
- **队列** 用于批处理/异步/长时间运行。
- **事件** 用于智能体的响应式行为。
- **定时任务** 用于日常维护（记忆整合、评估、成本报告）。

## 交付物

`outputs/skill-runtime-shape.md` 为任务选择运行时形态并配置可观测性要求。

## 练习

1. 将你的第01课ReAct循环移植到你技术栈中的所有六种形态。哪种形态适合哪种产品界面？
2. 为基于队列的演示添加死信队列。模拟10%的任务失败；显示死信队列大小。
3. 编写一个定时触发的评估智能体，每晚对当天的前20条trace运行评估。
4. 实现带背压的流式传输：如果客户端响应慢，暂停智能体。这与turn预算如何交互？
5. 阅读Claude Managed Agents文档。何时应将自托管的长周期智能体迁移到托管方案？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| 请求-响应 | "同步" | 用户等待；仅适用于短任务 |
| 流式 | "SSE / WS" | 渐进式输出；更好的用户体验；可逐块观测延迟 |
| 持久化执行 | "从失败中恢复" | 检查点化的状态；从最后一步重启 |
| 基于队列 | "后台任务" | 生产者 / 工作进程池 / 死信队列 |
| 事件驱动 | "基于触发器" | 智能体对外部事件作出响应 |
| 死信队列 | "Dead-letter queue" | 失败任务的停放区 |
| Claude Managed Agents | "托管运行环境" | Anthropic托管的长周期异步执行，支持缓存+压缩 |

## 延伸阅读

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 持久化执行细节
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — 托管长周期异步执行
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — "每个任务数十到数百步"
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — Actor模型故障隔离
