# OpenTelemetry GenAI — 端到端追踪工具调用

> 一个智能体调用了五个工具、三个 MCP 服务器和两个子智能体。你需要一条贯穿所有环节的追踪。OpenTelemetry GenAI 语义约定（semantic conventions，v1.37 及以上版本中的稳定属性）是 2026 年的标准，并被 Datadog、Langfuse、Arize Phoenix、OpenLLMetry 和 AgentOps 原生支持。本课会列出必需属性，讲解 span 层级（agent → LLM → tool），并提供一个可插入任意 OTel exporter 的标准库 span 发射器。

**类型：** 构建
**语言：** Python（标准库，OTel span 发射器）
**先修要求：** Phase 13 · 07（MCP server），Phase 13 · 08（MCP client）
**时间：** 约 75 分钟

## 学习目标

- 说出 LLM span 和工具执行 span 所需的 OTel GenAI 属性。
- 构建覆盖智能体循环、LLM 调用、工具调用和 MCP client 分发的追踪层级。
- 决定要捕获哪些内容（选择启用）以及要隐去哪些内容（默认行为）。
- 在不重写工具代码的情况下，将 spans 发射到本地 collector（Jaeger、Langfuse）。

## 问题

来自 2026 年 2 月的一次调试：用户报告“我的智能体有时需要 30 秒才响应；有时只要 3 秒。”没有 traces。日志显示了 LLM 调用，但没有工具分发、没有 MCP server 往返、也没有子智能体。你只能猜。最终你发现：某个 MCP server 偶尔会在冷启动时挂起。

没有端到端 tracing，你找不到这个问题。OTel GenAI 解决了它。

这些约定在 2025-2026 年间由 OpenTelemetry semantic-conventions 小组确定。它们定义了稳定的属性名称，因此 Datadog、Langfuse、Phoenix、OpenLLMetry 和 AgentOps 都能解析同样的 spans。只需埋点一次，就能发送到任意后端。

## 核心概念

### Span 层级

```
agent.invoke_agent  (top, INTERNAL span)
 ├── llm.chat       (CLIENT span)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT span)
 ├── llm.chat       (CLIENT span)
 └── subagent.invoke (INTERNAL)
```

整个结构嵌套在同一个 trace id 之下。Span ids 用来连接父子关系。

### 必需属性

根据 2025-2026 semconv：

- `gen_ai.operation.name` — `"chat"`、`"text_completion"`、`"embeddings"`、`"execute_tool"`、`"invoke_agent"`。
- `gen_ai.provider.name` — `"openai"`、`"anthropic"`、`"google"`、`"azure_openai"`。
- `gen_ai.request.model` — 请求的模型字符串（例如 `"gpt-4o-2024-08-06"`）。
- `gen_ai.response.model` — 实际服务该请求的模型。
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`。
- `gen_ai.response.id` — 用于关联的 provider response id。

对于工具 spans：

- `gen_ai.tool.name` — 工具标识符。
- `gen_ai.tool.call.id` — 具体的调用 id。
- `gen_ai.tool.description` — 工具描述（可选）。

对于智能体 spans：

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`。

### Span kinds

- 对跨越进程边界的调用（LLM provider、MCP server）使用 `SpanKind.CLIENT`。
- 对智能体自身的循环步骤和工具执行使用 `SpanKind.INTERNAL`。

### 选择启用内容捕获

默认情况下，spans 携带指标和计时信息，而不是 prompts 或 completions。大 payload 和 PII 默认关闭。设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 以及特定的内容捕获环境变量即可包含内容。在生产环境启用前请仔细审查。

### Span 上的事件

可以将 token 级事件添加为 span events：

- `gen_ai.content.prompt` — 输入消息。
- `gen_ai.content.completion` — 输出消息。
- `gen_ai.content.tool_call` — 记录下来的工具调用。

事件在 span 内按时间排序，便于详细回放。

### Exporters

OTel spans 可以导出到：

- **Jaeger / Tempo。** OSS，本地部署。
- **Langfuse。** 专用于 LLM observability；可视化 token usage。
- **Arize Phoenix。** Evals + tracing 结合。
- **Datadog。** 商业产品；原生解析 `gen_ai.*` 属性。
- **Honeycomb。** 面向列；便于查询。

它们都使用 OTLP 这种线缆格式。你的代码不需要关心。

### 跨 MCP 传播

当 MCP client 调用 server 时，将 W3C traceparent header 注入请求。Streamable HTTP 支持标准 headers。Stdio 本身不携带 HTTP headers；该规范的 2026 roadmap 正在讨论为 JSON-RPC 调用添加 `_meta.traceparent` 字段。

在该功能发布之前：手动把 traceparent 放进每个请求的 `_meta`。Server 记录 trace id。

### Metrics

除 spans 外，GenAI semconv 还定义了 metrics：

- `gen_ai.client.token.usage` — histogram。
- `gen_ai.client.operation.duration` — histogram。
- `gen_ai.tool.execution.duration` — histogram。

将它们用于不需要单次调用细节的 dashboards。

### AgentOps 层

AgentOps（创立于 2024 年）专注于 GenAI observability。它包装流行框架（LangGraph、Pydantic AI、CrewAI），自动发射 OTel spans。如果你的技术栈使用受支持的框架，它很有用；否则使用手动埋点。

## 使用它

`code/main.py` 会为一个调用 LLM、分发两个工具并进行一次 MCP 往返的智能体，将 OTel 形态的 spans 发射到 stdout（格式类似 OTLP-JSON）。没有真实 exporter —— 本课聚焦于 span 形态和属性集合。你可以把输出粘贴到 OTLP 兼容的查看器中，或者直接阅读。

需要关注：

- Trace id 在所有 spans 中共享。
- 父子链接通过 `parentSpanId` 编码。
- 必需的 `gen_ai.*` 属性已填充。
- 内容捕获默认关闭；一个场景会通过环境变量将其打开。

## 交付它

本课会生成 `outputs/skill-otel-genai-instrumentation.md`。给定一个智能体代码库，该 skill 会生成一份埋点计划：在哪里添加 spans、填充哪些属性，以及目标 exporters 是哪些。

## 练习

1. 运行 `code/main.py`。统计 spans 数量，并识别哪些是 CLIENT，哪些是 INTERNAL。

2. 打开内容捕获（环境变量），并确认出现 `gen_ai.content.prompt` 和 `gen_ai.content.completion` 事件。注意这对 PII 的影响。

3. 添加工具执行 metric `gen_ai.tool.execution.duration`，并为每次调用将其作为一个 histogram sample 发射。

4. 将 traceparent 从父 agent span 传播到 MCP 请求的 `_meta.traceparent` 字段。验证 MCP server 会看到相同的 trace id。

5. 阅读 OTel GenAI semconv spec。找出一个 semconv 中列出但本课代码没有发射的属性。添加它。

## 关键术语

| 术语 | 人们常说 | 它实际表示 |
|------|----------------|------------------------|
| OTel | “OpenTelemetry” | traces、metrics、logs 的开放标准 |
| GenAI semconv | “GenAI semantic conventions” | LLM / tool / agent spans 的稳定属性名称 |
| `gen_ai.*` | “The attribute namespace” | 所有 GenAI 属性共享此前缀 |
| Span | “Timed operation” | 具有开始、结束和属性的工作单元 |
| Trace | “Cross-span ancestry” | 共享同一个 trace id 的 spans 树 |
| SpanKind | “CLIENT / SERVER / INTERNAL” | 关于 span 方向的提示 |
| OTLP | “OpenTelemetry Line Protocol” | exporters 使用的线缆格式 |
| Opt-in content | “Prompt / completion capture” | 默认关闭；通过环境变量启用 |
| traceparent | “W3C header” | 跨服务传播 trace context |
| Exporter | “Backend-specific shipper” | 将 spans 发送到 Jaeger / Datadog / 等后端的组件 |

## 延伸阅读

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — GenAI spans、metrics 和 events 的权威约定
- [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — LLM 和工具执行 span 的属性列表
- [OpenTelemetry — GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — agent 级 `invoke_agent` span
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — GitHub 上托管的事实来源
- [Datadog — LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — 生产集成 walkthrough
