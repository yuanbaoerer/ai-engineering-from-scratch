# 推理平台经济学 — Fireworks、Together、Baseten、Modal、Replicate、Anyscale

> 2026 年的推理市场不再是 GPU 时间租赁。它分化为定制芯片（Groq、Cerebras、SambaNova）、GPU 平台（Baseten、Together、Fireworks、Modal）和 API 优先市场（Replicate、DeepInfra）。Fireworks 于 2026 年 5 月 1 日将 GPU 租用价格提高 $1/hr，$4B 估值对应每天 10T+ token 处理量，说明量驱动模式有效。Baseten 在 2026 年 1 月以 $5B 估值完成 $300M Series E。竞争定位规则很简单：Fireworks 优化延迟，Together 优化目录广度，Baseten 优化企业打磨，Modal 优化 Python 原生 DX，Replicate 优化多模态覆盖，Anyscale 优化分布式 Python。本课给你一个可以直接交给创始人的矩阵。

**类型：** 学习
**语言：** Python（stdlib，玩具级单次调用经济学比较器）
**前置课程：** 阶段 17 · 01（托管 LLM 平台），阶段 17 · 04（vLLM 推理内部机制）
**时间：** 约 60 分钟

## 学习目标

- 说出三个市场细分（定制芯片、GPU 平台、API 优先），并将每个供应商映射到对应细分。
- 解释为什么"按 token" API 定价模型趋向推理引擎的成本曲线压缩，而非硬件成本曲线。
- 计算至少三个供应商的有效单次请求成本，并解释按分钟计费（Baseten、Modal）何时优于按 token 计费。
- 识别给定工作负载（无服务器突发、稳定高吞吐、微调变体、多模态）的正确默认平台。

## 问题背景

你评估了托管超大规模平台，决定需要更窄、更快的提供商——Fireworks 要延迟，Together 要广度，Baseten 要微调定制模型。现在你有六个真实选择，但定价页面对不上。Fireworks 显示 $/M tokens；Baseten 显示 $/minute；Modal 显示 $/second；Replicate 显示 $/prediction。不建模工作负载就无法直接比较。

更糟的是，每个定价页面背后的商业模式不同。Fireworks 在共享 GPU 上运行自己的定制引擎（FireAttention）；按 token 费率反映其利用率曲线。Baseten 提供 Truss + 专用 GPU；按分钟反映独占性。Modal 是真正的 Python 无服务器——按秒计费，亚秒级冷启动。相同的输出（LLM 响应），三个不同的成本函数。

本课建模这六个平台，并告诉你每个何时胜出。

## 核心概念

### 三个细分

**定制芯片** — Groq（LPU）、Cerebras（WSE）、SambaNova（RDU）。在相同模型上，解码速度通常比 GPU 集群快 5-10 倍。按 token 价格更高（Groq 在 2025 年底 Llama-70B 上约 $0.99/M），但延迟敏感场景无可匹敌。Groq 是语音代理和实时翻译的生产选择。

**GPU 平台** — Baseten、Together、Fireworks、Modal、Anyscale。运行在 NVIDIA（H100、H200、2026 年的 B200）或有时 AMD 上。介于"原始 GPU 租赁"（RunPod、Lambda）和"超大规模托管服务"（Bedrock）之间的经济层。

**API 优先市场** — Replicate、DeepInfra、OpenRouter、Fal。广泛目录，按预测或按秒付费，强调首次调用时间。

### Fireworks — 延迟优化 GPU 平台

- FireAttention 引擎（定制）；宣称在等效配置上比 vLLM 低 4 倍延迟。
- 批处理层级约为基础无服务器费率的 50%，适用于非交互工作负载。
- 微调模型与基础模型同价——相比对 LoRA 收费的提供商是真正的差异化。
- 2026 年中：2026 年 5 月 1 日起将按需 GPU 租用价格提高 $1/hr。批量定价可协商。
- 财务信号：$4B 估值，每天处理 10T+ token。

### Together — 广度优化

- 200+ 模型，包括上游发布后数天内上线的开源模型。
- 在等效 LLM 模型上比 Replicate 便宜 50-70%——"AI Native Cloud" 定位是量和目录。
- 推理 + 微调 + 训练一个 API。

### Baseten — 企业打磨优化

- Truss 框架：模型打包，依赖、密钥、推理配置一个清单。
- GPU 范围从 T4 到 B200。按分钟计费，合理的冷启动缓解。
- SOC 2 Type II、HIPAA 就绪。常见金融科技和医疗选择。
- $5B 估值，2026 年 1 月 Series E（$300M，来自 CapitalG、IVP、NVIDIA）。

### Modal — Python 原生优化

- 纯 Python 的基础设施即代码。用 `@modal.function(gpu="A100")` 装饰函数，一条命令部署。
- 按秒计费。冷启动 2-4 秒（预热后）；小模型 <1 秒。
- $87M Series B，$1.1B 估值（2025）。独立调查中开发者体验评分最高。

### Replicate — 多模态广度

- 按预测付费。图像、视频和音频模型的默认平台。
- 集成生态（Zapier、Vercel、CMS 插件）。
- LLM 按 token 费率竞争力较弱，但在多模态种类上胜出。

### Anyscale — Ray 原生

- 基于 Ray 构建；RayTurbo 是 Anyscale 的专有推理引擎（与 vLLM 竞争）。
- 最适合分布式 Python 工作负载，其中推理步骤是更大图中的一个节点。
- 托管 Ray 集群；与 Ray AIR 和 Ray Serve 紧密集成。

### 按 token vs 按分钟——何时各胜

按 token 适用于延迟不敏感且突发的工作负载——只为使用量付费。按分钟适用于利用率高且可预测的场景——一旦 GPU 饱和就优于按 token。

粗略规则：持续利用率超过约 30% 的专用 GPU 工作负载，按分钟（Baseten、Modal）开始优于按 token（Fireworks、Together）。低于此值，按 token 胜出，因为避免为空闲付费。

### 定制引擎是真正的护城河

上述所有超过 vLLM 和 SGLang 的平台都声称有定制引擎。FireAttention、RayTurbo、Baseten 的推理栈。定制引擎声明偏向营销——诚实的框架是 vLLM + SGLang 占生产开源推理的约 80%，平台层的差异化在于 DX、归属和 SLA。

### 你应该记住的数字

- Fireworks GPU 租用：2026 年 5 月 1 日起提高 $1/hr。
- Fireworks 声称：在等效配置上比 vLLM 低 4 倍延迟。
- Together：LLM 比 Replicate 便宜 50-70%。
- Baseten 估值：$5B（Series E，2026 年 1 月，$300M 轮次）。
- Modal 估值：$1.1B（Series B，2025）。
- 持续利用率超过约 30% 时按分钟优于按 token。

```figure
cost-per-token
```

## 使用

`code/main.py` 在合成工作负载上跨定价模型比较六个供应商。报告 $/天 和有效 $/M tokens。运行它找到按 token 和按分钟的盈亏平衡点。

## 交付

本课产出 `outputs/skill-inference-platform-picker.md`。给定工作负载配置、SLA 和预算，选择主推理平台并列出备选。

## 练习

1. 运行 `code/main.py`。Baseten（按分钟）在什么持续利用率下优于 Fireworks（按 token），针对 70B 模型在一块 H100 上？自己推导交叉点并与经验法则比较。
2. 你的产品需要图像生成 + 聊天 + 语音转文字。为每种模态选择平台，并命名统一它们的网关模式。
3. Fireworks 将你的主要模型价格提高 $1/hr。如果 40% 流量转到批处理层级（50% 折扣），建模混合成本影响。
4. 一个受监管客户要求 SOC 2 Type II + HIPAA + 专用 GPU。哪三个平台可行，哪个在 FinOps 上胜出？
5. 比较 Fireworks 无服务器、Together 按需、Baseten 专用和 Replicate API 上 Llama 3.1 70B 每 1,000 次预测的成本。每天 10 次预测时哪个最便宜？每天 10,000 次呢？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| 定制芯片 | "非 GPU 芯片" | Groq LPU、Cerebras WSE、SambaNova RDU——为解码优化 |
| FireAttention | "Fireworks 引擎" | 定制注意力内核；宣称比 vLLM 低 4 倍延迟 |
| Truss | "Baseten 的格式" | 模型打包清单；依赖 + 密钥 + 推理配置 |
| 按 token | "API 定价" | 按消耗的 token 计费；不为空闲付费 |
| 按分钟 | "专用定价" | 按墙钟 GPU 时间计费；高利用率时胜出 |
| 按预测 | "Replicate 定价" | 按模型调用计费；常见于图像/视频 |
| RayTurbo | "Anyscale 引擎" | Ray 上的专有推理；在 Ray 集群上与 vLLM 竞争 |
| 批处理层级 | "50% 折扣" | 降价的非交互队列；Fireworks、OpenAI 常见 |
| 微调模型基础费率 | "Fireworks LoRA" | 按基础模型费率收费 LoRA 服务请求（差异化因素） |

## 延伸阅读

- [Fireworks Pricing](https://fireworks.ai/pricing) — 按 token 费率、批处理层级、GPU 租用。
- [Baseten Pricing](https://www.baseten.co/pricing/) — 按分钟费率、承诺容量、企业层级。
- [Modal Pricing](https://modal.com/pricing) — 按秒 GPU 费率和免费层级。
- [Together AI Pricing](https://www.together.ai/pricing) — 模型目录和按 token 费率。
- [Anyscale Pricing](https://www.anyscale.com/pricing) — RayTurbo 和托管 Ray 定价。
- [Northflank — Fireworks AI Alternatives](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) — 对比评估。
- [Infrabase — AI Inference API Providers 2026](https://infrabase.ai/blog/ai-inference-api-providers-compared) — 供应商全景。
