# 托管 LLM 平台 — Bedrock、Vertex AI、Azure OpenAI

> 三大超大规模平台，三种截然不同的策略。AWS Bedrock 是模型市场——Claude、Llama、Titan、Stability、Cohere，一个 API 即可调用。Azure OpenAI 是与 OpenAI 的独家合作，加上 Provisioned Throughput Units (PTUs) 提供专用容量。Vertex AI 以 Gemini 为核心，拥有最强的长上下文和多模态能力。2026 年 Artificial Analysis 测得 Azure OpenAI 在 Llama 3.1 405B 等效模型上的中位延迟约为 50 ms，Bedrock 约为 75 ms——PTUs 解释了这一差距，因为专用容量优于共享按需实例。决策标准不是"哪个最快"，而是"哪个模型目录和 FinOps 接口最匹配我的产品"。本课教你如何带着权衡分析来选择，而非凭感觉。

**类型：** 学习
**语言：** Python（stdlib，玩具级成本与延迟比较器）
**前置课程：** 阶段 11（LLM 工程），阶段 13（工具与协议）
**时间：** 约 60 分钟

## 学习目标

- 说出三种平台策略（市场模式 vs 独家合作 vs Gemini 优先），并将每种策略匹配到对应的产品使用场景。
- 解释 Azure OpenAI 中 Provisioned Throughput Units (PTUs) 的作用，以及为什么按需 Bedrock 在 405B 规模下通常慢约 25 ms。
- 画出每个平台的 FinOps 归属接口图（Bedrock Application Inference Profiles vs Vertex 项目-团队映射 vs Azure 作用域 + PTU 预留）。
- 写出"双提供商最低要求"策略，并解释为什么单一供应商锁定在 2026 年是昂贵的错误。

## 问题背景

你为产品选择了 Claude 3.7 Sonnet，现在需要部署它。你可以直接调用 Anthropic API，也可以通过 AWS Bedrock 调用，或者通过网关。直接 API 最简单；Bedrock 增加了 BAAs、VPC 端点、IAM 和 CloudWatch 归属；网关增加了故障转移、统一计费和跨提供商的速率限制。

更深层的问题是模型目录。如果你需要在同一产品中使用 Claude、Llama 和 Gemini，你无法从单一来源全部购买——除非那个来源同时是 Bedrock 加 Vertex 加 Azure OpenAI。超大规模平台之间不可互换——它们各自在谁拥有模型层这个问题上下了不同的赌注。

本课梳理这三种赌注、延迟差距、FinOps 差距和锁定风险。

## 核心概念

### 三种策略

**AWS Bedrock** — 市场模式。Claude（Anthropic）、Llama（Meta）、Titan（AWS 自研）、Stability（图像）、Cohere（嵌入向量）、Mistral，以及图像和嵌入模型子目录。一个 API、一个 IAM 接口、一个 CloudWatch 导出。Bedrock 的赌注是客户更看重选择多样性，而非单一模型。

**Azure OpenAI** — 独家合作。你可以在 Azure 数据中心使用 GPT-4 / 4o / 5 / o 系列、DALL·E、Whisper，以及对 OpenAI 模型进行微调。"Azure OpenAI Service" 目录中没有非 OpenAI 模型——那些放在 Azure AI Foundry（独立产品）中。Azure 的赌注是 OpenAI 仍然是前沿，客户希望对这一特定合作关系拥有企业级控制。

**Vertex AI** — Gemini 优先，其他其次。Gemini 1.5 / 2.0 / 2.5 Flash 和 Pro，加上 Model Garden（第三方模型）。Vertex 的赌注是多模态长上下文——1M token 的 Gemini 上下文是其差异化优势。

### 规模下的延迟差距

Artificial Analysis 运行持续基准测试。在等效的 Llama 3.1 405B 部署（共享按需）上，Azure OpenAI 中位首 token 延迟约为 50 ms；Bedrock 约为 75 ms。这一差距不是 AWS 的失败——而是容量模型的差异。Azure 销售 PTUs（Provisioned Throughput Units），为你的租户预留 GPU 容量。Bedrock 的等效产品（Provisioned Throughput）存在但起价约 $21/小时/单位，大多数客户仍使用共享按需。

按需共享容量与其他客户的流量竞争。专用容量不会。如果你的产品 SLA 要求 P99 TTFT < 100 ms，你需要在 Azure 上购买 PTU，购买 Bedrock Provisioned Throughput，或者接受默认的方差。

### Provisioned Throughput 经济学

Azure PTUs：预留的推理计算块。对于可预测的工作负载，相比按需可节省高达约 70%。无论流量如何，按固定小时成本——即使空闲也需要付费。盈亏平衡点通常在约 40-60% 的持续利用率。

Bedrock Provisioned Throughput：$21-$50/小时，取决于模型和区域。类似的数学——盈亏平衡点约为峰值利用率的一半。需要按月承诺。

Vertex 预留容量按 Gemini SKU 销售；定价因模型和区域而异，公开宣传较少。

### FinOps 接口——真正的差异化因素

**Bedrock Application Inference Profiles** 是市场中最干净的归属机制。用 `team`、`product`、`feature` 标记配置文件；通过它路由所有模型调用；CloudWatch 按配置文件拆分成本，无需后处理。2025 年推出，仍是超大规模平台中最细粒度的原生方案。

**Vertex** 归属是按团队建项目加全局标签。你将每个团队建模为一个 GCP 项目，在每个资源上打标签，使用 BigQuery Billing Export + DataStudio 进行汇总。工作量更大，但 BigQuery 允许你对成本数据执行任意 SQL。

**Azure** 依赖订阅/资源组作用域加标签，PTU 预留作为一等成本对象。标签从资源组继承而非请求，因此逐请求归属需要 Application Insights 自定义指标或在网关上添加头部。

规律：Bedrock 原生最干净，Vertex 通过 BigQuery 最灵活，Azure 最不透明除非你做埋点。

### 锁定是 2026 年的风险

当单一模型主导时，单一超大规模平台承诺没问题。2026 年前沿每月都在变化——一个季度 Claude 3.7，下个季度 Gemini 2.5，再下个季度 GPT-5。锁定一个平台意味着你被排除在三分之二的前沿之外。

有效团队采用的模式：任何产品关键 LLM 调用最低要求两个提供商。Bedrock 加 Azure OpenAI 是常见组合——一个提供 Claude，另一个提供 GPT，两者之间故障转移，同一网关。成本增加可忽略，因为网关路由最优；在故障期间（如 2025 年 1 月 Azure OpenAI 事件、AWS us-east-1 故障）的可用性提升是决定性的。

### 数据驻留、BAAs 和受监管行业

Bedrock：大多数区域提供 BAAs；VPC 端点；护栏。常见金融科技默认选择。
Azure OpenAI：HIPAA、SOC 2、ISO 27001；EU 数据驻留；企业受监管的默认选择。
Vertex：HIPAA、GDPR、按区域的数据驻留；Google Cloud 合规体系。

三者都满足基本合规要求。差异在于数据保留策略、日志处理方式，以及滥用监控是否读取你的流量（大多数默认启用；企业版可选退出）。

### 你应该记住的数字

- Azure OpenAI 在 Llama 3.1 405B 等效模型上的中位 TTFT：约 50 ms（使用 PTU）。
- Bedrock 按需中位 TTFT：约 75 ms。
- Bedrock Provisioned Throughput：$21-$50/小时/单位。
- Azure PTU 盈亏平衡点：约 40-60% 持续利用率。
- 高利用率下 PTU 相比按需的节省：高达 70%。

## 使用

`code/main.py` 在合成工作负载上比较三个平台——建模按需 vs PTU 经济学、TTFT 方差和成本归属性保真度。运行它可以看到 PTU 何时划算，以及市场模式的模型广度何时超过 TTFT 差距。

## 交付

本课产出 `outputs/skill-managed-platform-picker.md`。给定工作负载配置（所需模型、TTFT SLA、每日量、合规要求），它推荐主平台、备选平台和 FinOps 埋点方案。

## 练习

1. 运行 `code/main.py`。Azure PTU 在什么持续利用率下优于按需实例（70B 级模型）？计算盈亏平衡点并与 40-60% 的宣传区间比较。
2. 你的产品需要 Claude 3.7 Sonnet 和 GPT-4o。设计双提供商部署——哪个分配给哪个超大规模平台，前面放什么网关，故障转移策略是什么？
3. 一个受监管的医疗客户要求 BAAs、US-East 数据驻留和 P99 TTFT < 100ms。选择一个平台并用三个具体特性论证。
4. 你发现 Bedrock 账单本月涨了 4 倍但流量没变。没有 Application Inference Profiles，你如何找到原因？有 profiles 的话，需要多久？
5. 阅读 Azure OpenAI 和 Bedrock 定价页面。对于每月 100M token 的 Claude 工作负载，哪个更便宜——直接 Anthropic API、Bedrock 按需，还是 Bedrock Provisioned Throughput？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| Bedrock | "AWS LLM 服务" | 跨 Claude、Llama、Titan、Mistral、Cohere 的模型市场 |
| Azure OpenAI | "Azure 的 ChatGPT" | Azure 数据中心中的独家 OpenAI 模型，带企业控制 |
| Vertex AI | "Google 的 LLM" | Gemini 优先平台，带 Model Garden 用于第三方模型 |
| PTU | "专用容量" | Provisioned Throughput Unit——预留推理 GPU，按小时计费 |
| Application Inference Profile | "Bedrock 标签" | 按产品的成本/使用配置文件，带标签，CloudWatch 原生 |
| Model Garden | "Vertex 目录" | Vertex AI 的第三方模型区，与 Gemini 分开 |
| 双提供商最低要求 | "LLM 冗余" | 在 ≥2 个超大规模平台上运行每个关键 LLM 路径的策略 |
| BAA | "HIPAA 文书" | Business Associate Agreement；处理 PHI 必需；三者都提供 |
| 深入阅读 | — | — |

## 延伸阅读

- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/) — 权威费率卡和 Provisioned Throughput 定价。
- [Azure OpenAI Service Pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) — PTU 经济学和费率卡。
- [Vertex AI Generative AI Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing) — Gemini 层级和 Model Garden 附加费。
- [Artificial Analysis LLM Leaderboard](https://artificialanalysis.ai/) — 跨提供商的持续延迟和吞吐量基准测试。
- [The AI Journal — AWS Bedrock vs Azure OpenAI CTO Guide 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) — 企业决策框架。
- [Finout — Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) — 归属机制对比。
