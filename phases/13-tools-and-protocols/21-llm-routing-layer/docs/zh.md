# LLM 路由层 — LiteLLM、OpenRouter、Portkey

> 供应商锁定（provider lock-in）代价高昂。不同的工具调用（tool-calling）工作负载适合不同模型。路由网关提供统一的 API 表面、重试、故障转移、成本跟踪和护栏。2026 年主要有三类代表：LiteLLM（开源自托管）、OpenRouter（托管 SaaS）、Portkey（生产级，2026 年 3 月开源）。本课说明决策标准，并演示一个基于标准库的路由网关。

**类型:** 学习
**语言:** Python（标准库，路由 + 故障转移 + 成本跟踪器）
**先修:** 第 13 阶段 · 02（函数调用），第 13 阶段 · 17（网关）
**时间:** ~45 分钟

## 学习目标

- 区分自托管、托管和生产级路由方案。
- 实现一个回退链，按定义好的优先级顺序在供应商失败时重试。
- 跨供应商跟踪每个请求的成本和 token 用量。
- 针对给定的生产约束，在 LiteLLM、OpenRouter 和 Portkey 之间做选择。

## 问题背景

供应商路由很重要的场景：

1. **成本。** Claude Sonnet 的成本是 Haiku 的 3 倍。对于分流任务，Haiku 足够；对于综合归纳任务，Sonnet 值得使用。按请求路由。

2. **故障转移。** OpenAI 某个小时状态很差。所有请求都失败。你希望无需重新部署，就能自动回退到 Anthropic。

3. **延迟。** 实时聊天 UI 需要很快的首 token 延迟（time-to-first-token）。批量摘要器则不需要。按延迟 SLA 路由。

4. **合规。** 欧盟用户必须留在欧盟区域。按区域路由。

5. **实验。** 在同一工作负载上对两个模型做 A/B 测试。按测试桶路由。

为每个集成手写这些逻辑很重复。路由网关提供一个 OpenAI 兼容 API，并处理其余部分。

## 核心概念

### OpenAI 兼容代理形态

大家都在说 OpenAI 形态。路由网关暴露 `/v1/chat/completions`，接受 OpenAI schema，并在内部代理到 Anthropic / Gemini / Cohere / Ollama / 任意后端。客户端无需关心。

### 模型别名

你的代码不写 `claude-3-5-sonnet-20251022`，而是写 `our_smart_model`。网关把别名映射到真实模型。当 Anthropic 发布 Claude 4 时，你在服务端修改别名；代码完全不用动。

### 回退链

```
primary: openai/gpt-4o
on 5xx: anthropic/claude-3-5-sonnet
on 5xx: google/gemini-1.5-pro
on 5xx: refuse
```

网关在配置中定义这一点。重试会计入预算，避免回退级联导致成本爆炸。

### 语义缓存

相同或近似相同的提示词会命中缓存，而不是调用供应商。对重复的 agent 循环，节省幅度可达 30% 到 60%。键基于 embedding；近似相同的提示词共享一个缓存槽位。

### 护栏

网关层面：

- **PII 脱敏。** 在发送提示词前进行基于正则或 ML 的处理。
- **策略违规。** 拒绝包含禁止内容的提示词。
- **输出过滤。** 清理补全结果中的泄漏信息。

Portkey 和 Kong 都内置带有明确取向的护栏。LiteLLM 则把它们作为可选项。

### 按 key 限速

一个 API key = 一个团队。按 key 的预算可以防止某个团队耗尽共享配额。大多数网关都支持这一点。

### 自托管与托管的取舍

| 因素 | LiteLLM（自托管） | OpenRouter（托管） | Portkey（生产） |
|--------|----------------------|----------------------|----------------------|
| 代码 | 开源，Python | 托管 SaaS | 开源（2026 年 3 月）+ 托管 |
| 设置 | 部署一个代理 | 注册 | 两者皆可 |
| 供应商 | 100+ | 300+ | 100+ |
| 计费 | 你自己的 key | OpenRouter credits | 你自己的 key |
| 可观测性 | OpenTelemetry | 仪表盘 | 完整 OTel + PII 脱敏 |
| 最适合 | 想要完全控制的团队 | 快速原型开发 | 有合规要求的生产环境 |

当你有 SRE 团队并希望数据主权时，LiteLLM 胜出。当你想要单一订阅且不想维护基础设施时，OpenRouter 胜出。当你需要开箱即用的护栏和合规能力时，Portkey 胜出。

### 成本跟踪

每个请求都携带 `provider`、`model`、`input_tokens`、`output_tokens`。将它们乘以按模型、按 token 的价格（来自网关维护的价格表）。按用户 / 团队 / 项目聚合。

### MCP 加路由

网关可以同时路由 LLM 调用和 MCP 采样请求。当采样请求的 modelPreferences 偏好某个特定模型时，网关会转换到正确的后端。这也是第 13 阶段 · 17（MCP 网关）和本课路由网关有时会合并为一个服务的地方。

### 路由策略

- **静态优先级。** 列表中的第一个；出错时回退。
- **负载均衡。** 轮询或加权。
- **成本感知。** 选择满足延迟 / 质量要求的最便宜模型。
- **延迟感知。** 选择最近 N 分钟内最快的模型。
- **任务感知。** 提示词分类器把编码任务路由到一个模型，把摘要任务路由到另一个模型。

## 使用它

`code/main.py` 用约 150 行实现了一个路由网关：接受 OpenAI 形态的请求，转换到各供应商 stub，运行优先级回退链，跟踪每个请求的成本，并对输入应用 PII 脱敏处理。用三个场景运行它：正常请求、主供应商宕机触发回退、PII 泄漏被脱敏捕获。

需要关注：

- `ROUTES` 字典：别名 -> 按优先级排序的具体供应商列表。
- 回退循环会在 5xx 上重试。
- 成本跟踪器将 token 用量乘以各模型费率。
- PII 脱敏器会在转发前清理形似 SSN 的模式。

## 交付它

本课会产出 `outputs/skill-routing-config-designer.md`。给定一个工作负载画像（延迟、成本、合规），该 skill 会选择 LiteLLM / OpenRouter / Portkey，并生成路由配置。

## 练习

1. 运行 `code/main.py`。触发宕机场景；确认回退落到第二个供应商，并且成本归因正确。

2. 添加语义缓存：提示词的 SHA256 作为查找键；缓存命中会立即返回。测量重复调用节省的成本。

3. 添加一个提示词分类器，把 "code ..." 提示词路由到偏重智能的别名，把 "summarize ..." 提示词路由到偏重速度的别名。

4. 设计按团队预算：每个团队有月度支出上限；达到上限后网关拒绝请求。选择一种执行粒度（按请求或按时间窗口）。

5. 并排阅读 LiteLLM、OpenRouter 和 Portkey 文档。说出每个产品提供、而另外两个没有的一项功能。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| 路由网关 | “LLM proxy” | 位于多个供应商前方的统一 API 表面层 |
| OpenAI 兼容 | “Speaks the OpenAI schema” | 接受 `/v1/chat/completions` 形态，并转换到任意后端 |
| 模型别名 | “our_smart_model” | 代码中的名称，由网关映射到具体模型 |
| 回退链 | “Retry list” | 失败时按顺序尝试的供应商列表 |
| 语义缓存 | “Prompt-embedding cache” | 键是提示词的 embedding；近重复内容共享一次缓存命中 |
| 护栏 | “Input/output filters” | 脱敏 PII，拒绝策略违规 |
| 按 key 限速 | “Team budget” | 作用域限定在一个 API key 上的配额 |
| 成本跟踪 | “Per-request spend” | 聚合 token 用量 x 每个模型的价格 |
| LiteLLM | “The open proxy” | 可自托管的 OSS 路由网关 |
| OpenRouter | “The managed SaaS” | 基于 credit 计费的托管网关 |
| Portkey | “The production option” | 开源 + 托管，并内置护栏 |

## 延伸阅读

- [LiteLLM — docs](https://docs.litellm.ai/) — 自托管路由网关
- [OpenRouter — quickstart](https://openrouter.ai/docs/quickstart) — 托管路由 SaaS
- [Portkey — docs](https://portkey.ai/docs) — 带护栏的生产路由
- [TrueFoundry — LiteLLM vs OpenRouter](https://www.truefoundry.com/blog/litellm-vs-openrouter) — 决策指南
- [Relayplane — LLM gateway comparison 2026](https://relayplane.com/blog/llm-gateway-comparison-2026) — 供应商调研
