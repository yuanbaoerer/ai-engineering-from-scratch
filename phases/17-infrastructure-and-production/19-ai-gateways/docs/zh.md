# AI 网关 — LiteLLM、Portkey、Kong AI Gateway、Bifrost

> 网关位于你的应用和模型提供商之间。核心功能是提供商路由、降级、重试、速率限制、密钥引用、可观测性、防护栏。2026 年市场格局：**LiteLLM** 是 MIT 开源，支持 100+ 提供商，OpenAI 兼容，但在约 2000 RPS 时崩溃（8 GB 内存，公开基准测试中出现级联故障）；最适合 Python 应用，<500 RPS，开发/原型阶段。**Portkey** 定位为控制平面（防护栏、PII 脱敏、越狱检测、审计追踪），2026 年 3 月转为 Apache 2.0 开源，20-40 ms 延迟开销，生产版 $49/月。**Kong AI Gateway** 构建在 Kong Gateway 之上——Kong 在相同 12 CPU 上的基准测试：比 Portkey 快 228%，比 LiteLLM 快 859%；$100/模型/月定价（Plus 层最多 5 个）；如果你已在使用 Kong，适合企业场景。**Bifrost**（Maxim AI）——自动重试配合可配置退避，OpenAI 429 时降级到 Anthropic。**Cloudflare / Vercel AI 网关**——托管，零运维，基础重试。数据驻留驱动自托管决策；Portkey 和 Kong 处于中间位置，提供开源 + 可选托管。

**类型：** 学习
**语言：** Python（stdlib，玩具网关路由模拟器）
**前置课程：** 第 17 阶段 · 01（托管 LLM 平台），第 17 阶段 · 16（模型路由）
**时间：** 约 60 分钟

## 学习目标

- 枚举六大网关核心功能（路由、降级、重试、速率限制、密钥、可观测性、防护栏）。
- 将四个 2026 年网关（LiteLLM、Portkey、Kong AI、Bifrost）映射到规模上限和用例。
- 引用 Kong 基准测试（比 Portkey 快 228%，比 LiteLLM 快 859%）并解释其对 >500 RPS 场景的重要性。
- 根据数据驻留和运维预算选择自托管还是托管。

## 问题

你的产品调用 OpenAI、Anthropic 和自托管的 Llama。每个提供商有不同的 SDK、错误模型、速率限制和认证方案。你想要故障转移（如果 OpenAI 返回 429，尝试 Anthropic）、统一的凭据存储、统一的可观测性和每个租户的速率限制。

在应用层重新发明这些会将每个服务耦合到每个提供商。网关层将其整合为一个进程，提供一个 API（通常是 OpenAI 兼容），然后扇出到各个提供商。

## 概念

### 六大核心功能

1. **提供商路由**——OpenAI、Anthropic、Gemini、自托管等在一个 API 后面。
2. **降级**——429、5xx 或质量故障时，在其他地方重试。
3. **重试**——指数退避，有界尝试次数。
4. **速率限制**——每个租户、每个密钥、每个模型。
5. **密钥引用**——运行时从保险库拉取凭据（永不放在应用中）。
6. **可观测性**——OTel + GenAI 属性（第 17 阶段 · 13）+ 成本归属。
7. **防护栏**——PII 脱敏、越狱检测、允许主题过滤器。

### LiteLLM — MIT 开源，Python

- 100+ 提供商，OpenAI 兼容，路由器配置，降级，基础可观测性。
- 在 Kong 基准测试中约 2000 RPS 时崩溃；8 GB 内存占用，持续负载下级联故障。
- 最适合：Python 应用，<500 RPS，开发/预发布网关，实验性路由。
- 成本：开源 $0；有云免费层。

### Portkey — 控制平面定位

- 截至 2026 年 3 月为 Apache 2.0 开源。防护栏、PII 脱敏、越狱检测、审计追踪。
- 每请求 20-40 ms 延迟开销。
- 生产版 $49/月，含数据保留 + SLA。
- 最适合：需要防护栏 + 可观测性捆绑的受监管行业。

### Kong AI Gateway — 规模之选

- 构建在 Kong Gateway 之上（成熟的 API 网关产品，lua+OpenResty）。
- Kong 在 12 CPU 等效环境上的基准测试：比 Portkey 快 228%，比 LiteLLM 快 859%。
- 定价：$100/模型/月，Plus 层最多 5 个。
- 最适合：已在使用 Kong；>1000 RPS；愿意付费许可。

### Bifrost（Maxim AI）

- 自动重试配合可配置退避。
- OpenAI 429 时降级到 Anthropic 是典型方案。
- 新进入者；商业产品。

### Cloudflare AI Gateway / Vercel AI Gateway

- 托管，零运维。基础重试和可观测性。
- 最适合：Cloudflare/Vercel 上的边缘 JavaScript 应用。
- 在防护栏和速率限制方面相比 Kong/Portkey 有限。

### 自托管 vs 托管

数据驻留是决定因素。医疗和金融默认自托管（LiteLLM 或 Portkey 开源或 Kong）。消费产品默认托管（Cloudflare AI Gateway）或中间层（Portkey 托管）。混合方案：受监管租户自托管，其他托管。

### 延迟预算

- LiteLLM：典型 5-15 ms 开销。
- Portkey：20-40 ms 开销。
- Kong：3-8 ms 开销。
- Cloudflare/Vercel：1-3 ms 开销（边缘优势）。

网关延迟直接加到 TTFT 上。对于 TTFT P99 < 100 ms SLA，选择 Kong 或 Cloudflare。对于 P99 < 500 ms，任何都行。

### 速率限制语义很重要

简单令牌桶（token-bucket）适用于中等规模。多租户需要滑动窗口 + 突发允许 + 每租户分级。LiteLLM 提供令牌桶；Kong 提供滑动窗口；Portkey 提供分级。

### 网关 + 可观测性 + 路由组合

第 17 阶段 · 13（可观测性）+ 16（模型路由）+ 19（网关）在生产中是同一层。选择一个覆盖所有三者的工具，或仔细连接它们：大多数 2026 年部署将 Helicone（可观测性）或 Portkey（防护栏）与 Kong（规模）组合使用。

### 你应该记住的数字

- LiteLLM：约 2000 RPS 时崩溃，8 GB 内存。
- Portkey：20-40 ms 开销；2026 年 3 月起 Apache 2.0。
- Kong：比 Portkey 快 228%，比 LiteLLM 快 859%。
- Kong 定价：$100/模型/月，Plus 层最多 5 个。
- Cloudflare/Vercel：边缘 1-3 ms 开销。

## 使用

`code/main.py` 模拟在 429/5xx 注入下跨 3 个提供商的网关路由及降级。报告延迟、重试率和降级命中率。

## 交付

本课产出 `outputs/skill-gateway-picker.md`。给定规模、运维态势、合规性和延迟预算，选择网关。

## 练习

1. 运行 `code/main.py`。配置从 OpenAI→Anthropic→自托管的降级。在 5% 提供商错误率下预期命中率是多少？
2. 你的 SLA 是在 300 ms 基线上 TTFT P99 < 200 ms。哪些网关在预算内？
3. 一名医疗客户需要自托管 + PII 脱敏 + 审计。选择 Portkey 开源还是 Kong。
4. 比较 LiteLLM vs Kong：在什么 RPS 上限时团队应该迁移？
5. 为多租户 SaaS 设计速率限制策略：免费层、试用层、付费层。令牌桶还是滑动窗口？

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|----------|
| 网关 | "API 代理" | 位于应用和提供商之间的进程 |
| LiteLLM | "MIT 那个" | Python 开源，100+ 提供商，2K RPS 时崩溃 |
| Portkey | "防护栏网关" | 控制平面 + 可观测性，Apache 2.0 |
| Kong AI Gateway | "规模那个" | 构建在 Kong Gateway 之上，基准测试领先 |
| Bifrost | "Maxim 的网关" | 重试 + Anthropic 降级方案 |
| Cloudflare AI Gateway | "边缘托管" | 边缘部署的托管网关，零运维 |
| PII 脱敏 | "数据清洗" | 发送到模型前的正则 + NER 掩码 |
| 越狱检测 | "提示注入防护" | 用户输入分类器 |
| 审计追踪 | "受监管日志" | 每次 LLM 调用的不可变记录 |
| 令牌桶 | "简单速率限制" | 基于填充的速率限制器 |
| 滑动窗口 | "精确速率限制" | 时间窗口速率限制器；更公平 |

## 延伸阅读

- [Kong AI Gateway Benchmark](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — AI Gateways 2026 Comparison](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — Top LLM Gateway Tools 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway docs](https://docs.konghq.com/gateway/latest/ai-gateway/)
