# LLM 可观测性栈选择

> 2026 年的可观测性市场分为两类。开发平台（LangSmith、Langfuse、Comet Opik）将监控与评估、prompt 管理、会话回放捆绑在一起。网关/插桩工具（Helicone、SigNoz、OpenLLMetry、Phoenix）专注于遥测。Langfuse 核心采用 MIT 许可，在开源和商业之间保持良好平衡（免费云版本每月 50K 事件）。Phoenix 基于 OpenTelemetry，采用 Elastic License 2.0——非常适合漂移/RAG 可视化，但不是持久化生产后端。Arize AX 使用零拷贝 Iceberg/Parquet 集成，声称比单一可观测性方案便宜 100 倍。LangSmith 在 LangChain/LangGraph 生态中领先，$39/用户/月，仅 Enterprise 版本支持自托管。Helicone 基于代理，15-30 分钟即可设置，每月 100K 请求免费，但对 agent trace 的深度不足。常见的生产模式：网关（Helicone/Portkey）+ 评估平台（Phoenix/TruLens），通过 OpenTelemetry 粘合。

**类型：** 学习
**语言：** Python（stdlib，用于模拟 trace 采样的玩具模拟器）
**前置要求：** 阶段 17 · 08（推理指标）、阶段 14（Agent 工程）
**时间：** 约 60 分钟

## 学习目标

- 区分开发平台（捆绑：评估 + prompt + 会话）与网关/遥测工具（仅 trace + 指标）。
- 将六个主要工具（Langfuse、LangSmith、Phoenix、Arize AX、Helicone、Opik）映射到其许可、定价和最佳用例。
- 解释 OpenTelemetry 粘合模式，让你可以将网关工具与独立评估平台结合使用。
- 列出 2026 年的成本差异化因素（Arize AX 的零拷贝方案 vs 单一数据摄入），并给出大约 100 倍的倍数。

## 问题背景

你发布了一个 LLM 功能。它能工作。但你对 prompt 失败、工具循环、延迟回归、成本飙升或 prompt-cache 命中率毫无可见性。你搜索"LLM observability"，得到八个工具，都声称解决相同的问题，但价格各不相同。

它们解决的不是同一个问题。LangSmith 回答"这次 LangGraph 运行为何失败？"Phoenix 回答"我的 RAG 管道是否在漂移？"Helicone 回答"哪个应用在烧 token？"Langfuse 回答"我能否自托管整套方案？"不同的工具，不同的受众。

选择涉及四个维度：技术栈（LangChain？原始 SDK？多厂商？）、许可容忍度（仅 MIT？Elastic 也可以？商业许可？）、预算（免费层？$100/月？$1000/月？）、自托管（必须？可选？永远不需要？）。

## 核心概念

### 两大类别

**开发平台**将可观测性与评估、prompt 管理、数据集版本控制、会话回放捆绑在一起。你运行实验，看哪个 prompt 有效，用数据集回归测试新 prompt 与旧的优胜方案。LangSmith、Langfuse、Comet Opik。

**网关/遥测工具**对推理调用进行插桩——prompt、响应、token、延迟、模型、成本。Helicone、SigNoz、OpenLLMetry、Phoenix。极简主义。可通过 OpenTelemetry 与独立评估工具组合使用。

### Langfuse — 开源平衡

- 核心 Apache / MIT 许可；通过 Docker 自托管。
- 云免费层：每月 50K 事件。付费：团队版 $29/月。
- 评估、prompt 管理、trace、数据集。对开发平台四大功能有合理覆盖。
- 最佳场景：你需要 LangSmith 级别的功能但必须自托管或保持开源许可。

### Phoenix（Arize）— 遥测优先，OpenTelemetry 原生

- Elastic License 2.0；自托管非常简单。
- 在 RAG 和漂移可视化方面表现出色。嵌入空间散点图作为一等产物发布。
- 非设计为持久化生产后端——主要是开发时可观测性。
- 最佳场景：RAG 管道开发、漂移调试，与独立网关配合用于生产。

### Arize AX — 规模化方案

- 商业产品。通过 Iceberg/Parquet 实现零拷贝数据湖集成。
- 声称在规模化时比单一可观测性方案（Datadog 级别）便宜约 100 倍。数学原理：你在 S3 上用自己的 Parquet 存储 trace；Arize 直接读取。
- 最佳场景：每天 >10M trace，已有数据湖，需要 LLM 专用仪表板而不想支付 Datadog 的价格。

### LangSmith — LangChain/LangGraph 优先

- 商业产品，$39/用户/月。仅 Enterprise 版本支持自托管。
- 对 LangChain 和 LangGraph 栈是最佳选择。如果你不使用其中任何一个，则吸引力较低。
- 最佳场景：团队已投入 LangChain，愿意付费。

### Helicone — 基于代理的最小可行方案

- 15-30 分钟设置，只需将 `OPENAI_API_BASE` 换成 Helicone 代理。
- MIT 许可；每月 100K 请求免费，付费 $20/月起。
- 包含故障转移、缓存、速率限制——同时也充当网关。
- 对 agent / 多步 trace 的深度不足。
- 最佳场景：快速启动，单一技术栈应用，需要网关 + 可观测性合二为一。

### Opik（Comet）— 开源开发平台

- Apache 2.0，完全开源。
- 与 Langfuse 类似功能集，带有 Comet 血统。
- 最佳场景：已在使用 Comet 的 ML 团队，想要在同一面板中获得 LLM 可观测性。

### SigNoz — OpenTelemetry 优先的全栈 APM

- Apache 2.0。通过 OpenTelemetry 处理通用 APM 加上 LLM。
- 最佳场景：跨服务和 LLM 调用的统一可观测性。

### 粘合剂：OpenTelemetry + GenAI 语义约定

OpenTelemetry 在 2025 年底发布了 GenAI 语义约定（`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`）。消费 OTel 的工具可以互操作。正在涌现的生产模式：

1. 从每个 LLM 调用发出带 GenAI 约定的 OTel。
2. 路由到网关（Helicone / Portkey）用于日常使用。
3. 双写到评估平台（Phoenix / Langfuse）用于回归测试。
4. 归档到数据湖（Iceberg）用于通过 Arize AX 或 DuckDB 进行长期分析。

### 陷阱：在错误的层插桩

在你的 agent 框架内部插桩（例如添加 LangSmith trace）会将你与该框架耦合。在 HTTP/OpenAI-SDK 层插桩（通过 OpenLLMetry 或你的网关）是可移植的。

### 采样——你无法保留一切

每天 >1M 请求时，全 trace 保留的成本超过 LLM 调用本身。按规则采样：100% 错误、100% 高成本、5% 成功。始终保留聚合数据；仅对长尾保留原始数据。

### 需要记住的数据

- Langfuse 免费云版：每月 50K 事件。
- LangSmith：$39/用户/月。
- Helicone 免费版：每月 100K 请求。
- Arize AX 声称：规模化时比单一方案便宜约 100 倍。
- OpenTelemetry GenAI 约定：2025 年发布，2026 年广泛采用。

## 使用

`code/main.py` 模拟一天 1M trace 在不同保留策略（100% 摄入、采样、采样 + 错误）下的表现。报告存储成本和每种策略的损失。

## 产出

本课产出 `outputs/skill-observability-stack.md`。根据技术栈、规模、预算、许可偏好，选择工具。

## 练习

1. 你的 LangChain 团队想要开源自托管可观测性。选择 Langfuse 或 Opik 并说明理由。
2. 每天 5M trace，Datadog 报价 $150K/月，计算 Arize AX 的盈亏平衡点。
3. 设计一套 OpenTelemetry GenAI 属性集，作为你组织指南中每次 LLM 调用的强制要求。
4. 论述 Phoenix 单独是否足以满足生产需求。什么时候不够？
5. Helicone 有 20ms 代理开销。在 P99 TTFT 300 ms 下，这是否可接受？如果 SLA 是 100 ms 呢？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| OpenLLMetry | "LLM 的 OTel" | 用于 LLM 的开源 OpenTelemetry 插桩 |
| GenAI conventions | "OTel 属性" | LLM 调用的标准 OTel 属性名 |
| LangSmith | "LangChain 可观测性" | 与 LangChain 生态捆绑的商业平台 |
| Langfuse | "开源 LangSmith" | MIT 开源，功能集类似 |
| Phoenix | "Arize 开发工具" | OpenTelemetry 原生的开发/评估平台 |
| Arize AX | "规模化可观测性" | 商业零拷贝 Iceberg/Parquet 可观测性 |
| Helicone | "代理可观测性" | HTTP 代理收集 LLM 遥测 + 网关功能 |
| Opik | "Comet LLM" | 来自 Comet 的 Apache 2.0 开源开发平台 |
| Session replay | "trace 重放" | 重放包含工具调用的完整 agent 会话 |
| Eval | "离线测试" | 在标注数据集上运行候选模型/prompt |

## 延伸阅读

- [SigNoz — Top LLM Observability Tools 2026](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX Alternative analysis](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — Setting Up Langfuse, LangSmith, Helicone, Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix docs](https://docs.arize.com/phoenix)
- [Helicone docs](https://docs.helicone.ai/)
