# Batch API — 50% 折扣成为行业标准

> 每个主要提供商都提供带 50% 折扣和约 24 小时周转时间的异步 batch API。OpenAI、Anthropic、Google 以及大多数推理平台（Fireworks batch 层、Together batch）都实现了相同的模式。将 batch 与 prompt caching 组合，夜间管道可降至同步无缓存成本的约 10%。规则非常简单：如果不是交互式的，就应该用 batch。内容生成管道、文档分类、数据提取、报告生成、批量标注、目录打标——任何能容忍 24 小时延迟的任务在迁移到 batch 之前都是在浪费钱。2026 年的生产模式是将每个新的 LLM 工作负载分类到三个车道：交互式（同步 + 缓存）、半交互式（异步队列 + 回退）、批量（夜间、叠加缓存输入）。假装是交互式但能容忍几分钟延迟的工作负载浪费最多。

**类型：** 学习
**语言：** Python（stdlib，用于模拟 batch vs 同步成本的玩具模拟器）
**前置要求：** 阶段 17 · 14（Prompt 与语义缓存）
**时间：** 约 45 分钟

## 学习目标

- 列出三个提供商的 batch API（OpenAI、Anthropic、Google）以及共同的 50% 折扣 + 24 小时周转保证。
- 计算将 batch + 缓存输入叠加在夜间分类工作负载上的成本，并与同步无缓存基线对比。
- 将工作负载分类为交互式 / 半交互式 / 批量并说明车道选择理由。
- 列出两个陷阱：部分交互性（用户期望比 24 小时更快）和输出 schema 漂移（batch 文件格式因提供商而异）。

## 问题背景

你的团队发布了夜间报告生成管道。50,000 份文档，总结每份，聚类摘要，起草执行简报。同步运行需要 4 小时，每晚 $2,000。你听说了 batch API。

Batch 给你 50% 折扣。你还在 system prompt（所有 50k 调用共享）上启用了 prompt caching。叠加后，账单降至每晚 $180——约为基线的 9%。相同的管道，三次配置变更。

Batch 是 LLM 成本工具箱中最便宜但没人拉的杠杆。原因主要是组织性的：团队在 SLA 实际上是"明早之前"时想的是"实时"。本课是关于不要把 90% 的账单留在桌上。

## 核心概念

### 三个 Batch API

**OpenAI Batch API**：JSONL 文件上传，包含请求列表。承诺 24 小时周转（实践中通常 2-8 小时）。输入和输出 token 均享 50% 折扣。`/v1/batches` 端点。符合条件的输入还可额外享受缓存输入定价。

**Anthropic Message Batches**：JSONL 上传。24 小时周转。50% 折扣。支持 `cache_control`——缓存写入是显式的，batch 内的读取自动发生。

**Google Vertex AI Batch Prediction**：BigQuery 或 GCS 输入。Gemini 享有类似的 50% 折扣。与 Vertex pipelines 集成。

### 语义：异步，而非慢

Batch 是"我承诺在 24 小时内返回"——而非"这需要 24 小时"。典型的 P50 是 2-6 小时。提供商在 GPU 利用率低的非高峰时段调度你的 batch。

### 与缓存叠加

一个 50k 文档的总结任务，使用相同的 4K token system prompt：

- 同步无缓存：50000 ×（$input × 4000 + $output × 200）全价。
- 同步有缓存：system prompt 在首次写入后缓存；剩余 49999 次获得 10 倍便宜的输入。
- Batch 有缓存：以上全部加上读写均享 50% 折扣。

组合：batch + cache = 同步无缓存账单的约 10%。任何夜间运行且有共享 system prompt 的工作都应该使用此组合。

### 工作负载分类

**交互式** — 用户等待响应。TTFT 很重要。使用 prompt caching 的同步调用。不能用 batch。

**半交互式** — 用户提交任务，几分钟后回来检查。异步队列，batch 不可用时回退到同步。中等量级的 RAG 索引即属此类。

**批量** — 用户期望"明早"或"下个小时"看到结果。内容管道、大规模分类、离线分析。始终用 batch，始终叠加缓存。

常见错误：因为管道是生产环境就将所有东西分类为交互式。生产环境不是延迟规格——SLA 才是。

### 部分交互性陷阱

有些功能看起来是交互式的但能容忍 5-10 分钟。例如：带有"刷新"按钮的夜间客户健康报告。用户点击刷新；等 10 分钟没问题。团队将其作为同步实现。50 个并发刷新的成本是 batch + 邮件交付的 10 倍。

要问的问题："24 小时对这个用户意味着什么？"如果答案是"他们不会注意到"，就用 batch。

### 输出 schema 陷阱

Batch 文件格式因提供商而异：

- OpenAI：JSONL，每行一个请求。
- Anthropic：JSONL，每行一条消息；响应格式内嵌。
- Vertex：BigQuery 表或带 TFRecord 的 GCS 前缀。

编写"一个 batch 客户端"跨提供商意味着每个提供商一个适配器代码。声称支持多提供商 batch 的网关（Portkey、LiteLLM 某些层）仍然只是薄封装原始格式。

### 需要记住的数据

- 各提供商的 batch 折扣：输入 + 输出统一 50%。
- 周转 SLA：承诺 24 小时，典型 P50 为 2-6 小时。
- 叠加 batch + 缓存输入：约为同步无缓存成本的 10%。
- 工作负载分类规则：如果 24 小时延迟可接受，始终用 batch。

## 使用

`code/main.py` 为 50k 文档工作负载计算同步、同步+缓存、batch、batch+缓存四种模式的成本。报告以美元和百分比表示的节省。

## 产出

本课产出 `outputs/skill-batch-triager.md`。根据工作负载特征，分类为交互式/半交互式/批量并估算节省。

## 练习

1. 运行 `code/main.py`。对于 100k 文档管道，3K token system prompt 和 500 token 输出，计算完整组合（batch + cache）相比同步基线的节省。
2. 在你熟悉的真实产品中选择三个功能。将每个分类为交互式/半交互式/批量。
3. 一个用户抱怨他们的报告花了 3 小时。这是 batch 分类错误还是合理的交互式？写出判定标准。
4. 你的 batch API 返回 SLA 是 24 小时但 P99 是 20 小时。你如何向用户传达这一点——边缘情况下的下游系统行为是什么？
5. 计算盈亏平衡点：在什么共享前缀长度下，batch + cache 比在你自己的预留 GPU 上夜间运行更便宜？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Batch API | "异步折扣" | 50% 折扣，24 小时周转 |
| JSONL | "batch 格式" | 每行一个 JSON 请求；OpenAI/Anthropic 标准 |
| Message Batches | "Anthropic batch" | Anthropic 的 batch API 产品名称 |
| Batch prediction | "Vertex batch" | Vertex AI 的 batch API 产品 |
| Turnaround SLA | "24 小时承诺" | 是保证而非典型；典型为 2-6 小时 |
| Workload triage | "交互性决策" | 交互式 / 半交互式 / 批量路由决策 |
| Output schema | "响应格式" | 每个提供商的 JSONL 布局；不可移植 |
| Stacked discount | "batch + cache" | 两者叠加时约为无缓存同步账单的 10% |

## 延伸阅读

- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) — JSONL 格式和 `/v1/batches` 语义。
- [Anthropic Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) — batch 格式和 `cache_control` 交互。
- [Vertex AI Batch Prediction](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/batch-prediction) — Gemini batch 语义。
- [Finout — OpenAI vs Anthropic API Pricing 2026](https://www.finout.io/blog/openai-vs-anthropic-api-pricing-comparison)
- [Zen Van Riel — LLM API Cost Comparison 2026](https://zenvanriel.com/ai-engineer-blog/llm-api-cost-comparison-2026/)
