# OpenTelemetry GenAI 语义约定

> OpenTelemetry 的 GenAI SIG（2024 年 4 月启动）定义了 agent 遥测的标准模式。Span 名称、属性和内容捕获规则在各厂商间趋于统一，使得 agent 追踪在 Datadog、Grafana、Jaeger 和 Honeycomb 中含义一致。

**类型：** 学习 + 实践
**语言：** Python（stdlib）
**前置知识：** 阶段 14 · 13（LangGraph），阶段 14 · 24（可观测性平台）
**时间：** ~60 分钟

## 学习目标

- 说出 GenAI span 的三个类别：model/client、agent、tool。
- 区分 `invoke_agent` CLIENT 与 INTERNAL span 及各自的适用场景。
- 列出顶级 GenAI 属性：provider name、request model、data-source ID。
- 解释内容捕获契约：opt-in 机制、`OTEL_SEMCONV_STABILITY_OPT_IN`、外部引用建议。

## 问题背景

每个厂商都自定义自己的 span 名称，运维团队不得不为每个框架单独搭建仪表盘。OpenTelemetry 的 GenAI SIG 通过定义一个标准来解决这个问题，使整个生态系统有统一的目标。

## 核心概念

### Span 类别

1. **Model / client span。** 涵盖原始 LLM 调用。由 provider SDK（Anthropic、OpenAI、Bedrock）和框架模型适配器发出。
2. **Agent span。** `create_agent`（agent 构造时）和 `invoke_agent`（agent 运行时）。
3. **Tool span。** 每次工具调用一个；通过父子关系连接到 agent span。

### Agent span 命名

- Span 名称：如果已命名则为 `invoke_agent {gen_ai.agent.name}`；否则回退为 `invoke_agent`。
- Span kind：
  - **CLIENT** — 用于远程 agent 服务（OpenAI Assistants API、Bedrock Agents）。
  - **INTERNAL** — 用于进程内 agent 框架（LangChain、CrewAI、本地 ReAct）。

### 关键属性

- `gen_ai.provider.name` — `anthropic`、`openai`、`aws.bedrock`、`google.vertex`。
- `gen_ai.request.model` — 模型 ID。
- `gen_ai.response.model` — 解析后的模型（可能因路由而与请求模型不同）。
- `gen_ai.agent.name` — agent 标识符。
- `gen_ai.operation.name` — `chat`、`completion`、`invoke_agent`、`tool_call`。
- `gen_ai.data_source.id` — 用于 RAG：查询了哪个语料库或存储。

存在针对 Anthropic、Azure AI Inference、AWS Bedrock、OpenAI 的特定约定。

### 内容捕获

默认规则：instrumentation 默认不应捕获输入/输出。捕获通过以下属性 opt-in：

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

推荐的生产模式：将内容存储在外部（S3、日志存储），在 span 上记录引用（指针 ID，而非完整文本）。这是第 27 课中内容投毒防御机制在可观测性中的落地。

### 稳定性

截至 2026 年 3 月，大多数约定仍为实验性。通过以下方式启用稳定预览：

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ 原生将 GenAI 属性映射到其 LLM Observability 模式。其他后端（Grafana、Honeycomb、Jaeger）支持原始属性。

### 常见错误

- **在 span 中捕获完整 prompt。** 追踪中包含 PII、密钥、客户数据，运维人员可直接读取。应存储到外部。
- **缺少 `gen_ai.provider.name`。** 多 provider 仪表盘在归属信息缺失时无法工作。
- **Span 没有父链接。** 孤立的 tool span。务必传播上下文。
- **未设置稳定性 opt-in。** 后端升级时属性可能被重命名。

## 实践

`code/main.py` 实现了一个符合 GenAI 约定的 stdlib span 发送器：

- `Span` 包含 GenAI 属性模式。
- `Tracer` 支持 `start_span` 和嵌套上下文。
- 一个脚本化的 agent 运行，发出：`create_agent`、`invoke_agent`（INTERNAL）、每个工具的 span、LLM 调用的 `chat` span。
- 一个内容捕获模式，将 prompt 存储到外部并在 span 上记录 ID。

运行方式：

```
python3 code/main.py
```

输出：一个包含所有必需 GenAI 属性的 span 树，以及展示 opt-in 内容引用的"外部存储"。

## 应用场景

- **Datadog LLM Observability**（v1.37+）原生映射属性。
- **Langfuse / Phoenix / Opik**（第 24 课）— 自动检测生态系统。
- **Jaeger / Honeycomb / Grafana Tempo** — 原始 OTel 追踪；基于 GenAI 属性搭建仪表盘。
- **自托管** — 运行带 GenAI 处理器的 OTel Collector。

## 交付物

`outputs/skill-otel-genai.md` 将 OTel GenAI span 接入现有 agent，包含内容捕获默认配置和外部引用存储。

## 练习

1. 用 `invoke_agent`（INTERNAL）+ 每个工具的 span 对第 01 课的 ReAct 循环进行检测。发送到 Jaeger 实例。
2. 添加"仅引用"模式的内容捕获：prompt 存入 SQLite，span 属性只携带行 ID。
3. 阅读 `gen_ai.data_source.id` 规范，将其接入第 09 课的 Mem0 搜索。
4. 设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`，验证你的属性不会被 collector 重命名。
5. 搭建一个仪表盘：仅从 GenAI 属性中分析"哪些工具错误与哪些模型相关"。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| GenAI SIG | "OpenTelemetry GenAI 小组" | 定义模式的 OTel 工作组 |
| invoke_agent | "Agent span" | 表示 agent 运行的 span 名称 |
| CLIENT span | "远程调用" | 调用远程 agent 服务的 span |
| INTERNAL span | "进程内" | 进程内 agent 运行的 span |
| gen_ai.provider.name | "Provider" | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | "RAG 来源" | 检索命中的语料库/存储 |
| 内容捕获 | "Prompt 记录" | opt-in 捕获消息；生产环境存储到外部 |
| 稳定性 opt-in | "预览模式" | 用于固定实验性约定的环境变量 |

## 延伸阅读

- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 规范文档
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 默认包含 GenAI span
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 内置 OTel span
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — W3C trace context 传播
