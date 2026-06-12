# 生产中的 EAGLE-3 投机解码

> 投机解码将快速 draft 模型与目标模型配对。draft 提出 K 个 token；目标在一次前向传播中验证；被接受的 token 是免费的。2026 年，EAGLE-3 是生产级变体——它在目标模型的隐藏状态（而非原始 token）上训练 draft head，将接受率 alpha 推到通用聊天的 0.6-0.8 区间。正确的问题不是"draft 有多快"，而是"我的流量上 alpha 是多少？"如果 alpha 降到约 0.55 以下，投机解码在高并发下是净负面的，因为每个被拒绝的 draft 都要付出第二次目标前向传播的代价。本课教你先测量 alpha，再开开关。

**类型：** 学习
**语言：** Python（stdlib，玩具级接受率模拟器）
**前置课程：** 阶段 17 · 04（vLLM 推理内部机制），阶段 10 · 18（多 Token 预测）
**时间：** 约 60 分钟

## 学习目标

- 说出投机解码的三代演进，并解释 EAGLE-3 相比 EAGLE-2 和经典 draft model 的变化。
- 定义接受率 alpha，从 alpha 和 K（draft 长度）计算预期加速比，并识别你目标并发下的盈亏平衡 alpha。
- 解释为什么投机解码在 vLLM 2026 中是可选的（非默认），以及为什么不开测 alpha 就打开是生产反模式。
- 写一个测量方案：用哪个基准、哪个提示分布、哪个并发点、以哪个指标为门控。

## 问题背景

解码是内存受限的。在 H100 上运行 Llama 3.3 70B FP8，每个解码 token 读取约 140 GB/s 的权重并发出一个 token。解码期间 GPU 计算几乎空闲——瓶颈是 HBM 带宽，而非 matmul 吞吐。

投机解码利用了这个差距。用便宜的 draft 模型生成 K 个候选 token，然后让目标模型在一次前向传播中验证全部 K 个。每个验证过的 token 实际上是免费的（摊销到目标本来就要做的 K-batch 前向传播中）。

经典的 draft model 方法使用同系列的较小模型（Llama 3.2 1B 为 Llama 3.3 70B 起草）。可行但接受率平庸——较小模型的分布与目标偏离。EAGLE，然后 EAGLE-2，然后 EAGLE-3 直接在目标模型的内部状态上训练轻量 draft head，使 draft 的分布更紧密地跟踪目标。这就是为什么 alpha 从 draft model 的 0.4 提升到 EAGLE-3 的 0.6-0.8。

关键点：EAGLE-3 在 vLLM 2026 中是可选的。必须显式设置 `speculative_config`。不设标志就没有加速。不测量真实流量上的 alpha 就打开的团队，往往看到尾部延迟变差而非变好。

## 核心概念

### 投机解码实际买到了什么

没有 spec decode 时，每 token 成本是一次目标前向传播。有 spec decode 时，draft 长度 K，接受率 alpha，每次目标前向传播的预期 token 数是 `1 + K * alpha`。加速比是 `(1 + K * alpha) / (1 + epsilon)`，其中 epsilon 是 draft 加验证的开销。K=5，alpha=0.7：`(1 + 5*0.7) / (1 + 0.1) = 4.5 / 1.1 = 4.1x`。实际数字聚集在 2-3x，因为 alpha 在生产流量上很少那么高，且 epsilon 在大批大小时增长。

### 为什么 alpha 是唯一重要的指标

被拒绝的 token 不会消失——它们强制对第一个被拒绝 token 做第二次目标前向传播。在 alpha 降到 0.4 的工作负载上，你付出 draft 开销加验证加重试。在高并发（比如 256 并发）下，decode 批次已经足够大，"目标单独"和"目标加验证"之间的内存带宽差距缩小。在大多数 2026 硬件上 alpha 低于 0.55 时，spec decode 是净负面的。

Alpha 因工作负载而异。在 ShareGPT 风格的通用聊天上，EAGLE-3 在 ShareGPT 上训练达到 0.6-0.8。在领域特定流量（代码、医疗、法律）上，通用数据训练的 draft head 降到 0.4-0.6。训练领域特定 draft head 恢复 alpha——与目标微调相比，这是轻量、快速的训练任务。

### EAGLE 各代一览

- **经典 draft model**：同系列小模型。Alpha 0.3-0.5。基础设施简单——加载两个模型，draft 每次目标前向传播做 K 次前向。
- **EAGLE-1（2024）**：在目标隐藏状态（最后一层）上训练的单个 draft head。Alpha 约 0.5-0.6。目标之上的小参数开销。
- **EAGLE-2（2025）**：自适应 draft 长度和基于树的 draft（在一次目标前向传播中验证多个分支）。Alpha 约 0.6-0.7。更复杂的 draft 调度器。
- **EAGLE-3（2025-2026）**：在多个目标层（不仅是最后一层）上训练的 draft head，更好的对齐。通用聊天上 Alpha 约 0.6-0.8。

### 2026 年的生产方案

1. 部署目标模型原版。测量基线 TTFT、ITL、目标并发下的吞吐量。
2. 通过 vLLM `speculative_config` 启用 EAGLE-3 draft。重新运行基准。
3. 记录接受率 alpha。vLLM V1 将其报告为 `spec_decode_metrics.accepted_tokens_per_request`。除以请求的 draft 长度得到 alpha。
4. 如果在生产流量分布上 alpha < 0.55，禁用 spec decode 或训练领域特定 EAGLE-3 draft。
5. 在生产并发下重新运行。确认 P99 ITL 没有变差。

### 生产陷阱：P99 尾部

平均 ITL 随 spec decode 下降。如果不调优，P99 可能变差。被拒绝的 draft 触发两遍序列（draft + 验证失败 + 重试）。在满批次下，这两遍串行化。关注 P99 ITL，而非 P50。

### EAGLE-3 已部署在哪里

Google 在 2025 年将投机解码部署在 AI Overviews 中（相同质量，更快响应）。vLLM V1 将 `speculative_config` 作为文档接口；V1 中的 N-gram GPU 投机解码是与 chunked prefill 兼容的变体。SGLang 支持 EAGLE-3 作为前缀密集工作负载的推荐 draft 路径。

### 盈亏平衡公式一行

预期加速比：`S(alpha, K) = (1 + K*alpha) / (1 + verify_overhead)`。设 `S = 1` 解出 alpha：`alpha_breakeven = verify_overhead / K`。典型 verify_overhead 约 0.15，K=5：`alpha_breakeven = 0.03`。但这是原始解码数学。在高并发下，验证开销上升，decode 批次已经在序列间摊销内存读取，因此有效的 alpha_breakeven 在实践中攀升到约 0.45-0.55。

### 何时不使用投机解码

- 批次 1 离线生成，延迟不重要。用原版目标。
- 非常短的输出（50 token 以下）。draft 开销和验证成本占主导。
- 没有领域训练 draft head 的专业领域。Alpha 太低。
- vLLM v0.18.0 加 draft model spec decode 加 `--enable-chunked-prefill`。此组合不能编译。文档记录的例外是 V1 中的 N-gram GPU spec decode。

## 使用

`code/main.py` 在一系列 alpha 值和 draft 长度 K 上模拟有无投机解码的解码循环。它打印盈亏平衡 alpha、测量的加速比和尾部行为。在多个 (alpha, K) 组合上运行它，看看投机解码何时不再划算。

## 交付

本课产出 `outputs/skill-eagle3-rollout.md`。给定目标模型、流量分布描述和并发目标，产出分阶段 EAGLE-3 上线方案——基准基线、启用配置、测量 alpha、以 alpha >= 0.55 为门控、监控 P99 ITL。

## 练习

1. 运行 `code/main.py`。K=5 时，你需要什么 alpha 才能达到 2x 加速？3x 呢？这对 verify_overhead 有多敏感？
2. 假设生产流量 70% 通用聊天，30% 代码。通用聊天在 ShareGPT 训练的 EAGLE-3 上 alpha 0.7；代码 alpha 0.4。混合 alpha 是多少？spec decode 净正吗？
3. 阅读 vLLM `speculative_config` 文档。说出三种模式（draft model、EAGLE、N-gram），以及哪个与 chunked prefill 兼容。
4. 你看到启用 EAGLE-3 后平均 ITL 降 25%，但 P99 ITL 升了 15%。诊断并提出缓解方案。
5. 计算 EAGLE-3 draft head 在 Llama 3.3 70B 上的内存成本。与运行 Llama 3.2 1B 作为经典 draft 相比如何？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| 投机解码 | "draft 加验证" | 用便宜模型提出 K 个 token，在一次目标前向中验证全部 K 个 |
| 接受率 alpha | "spec 接受率" | draft token 被目标接受的比例；唯一重要的指标 |
| Draft 长度 K | "spec k" | draft 每次目标前向传播提出多少 token；典型 4-8 |
| 验证开销 epsilon | "spec 开销" | 验证和重试相比原版目标前向的额外成本；随批次增长 |
| EAGLE-3 | "最新 EAGLE" | 2025-2026 变体；在多个目标层上训练 draft head；通用聊天 alpha 0.6-0.8 |
| `speculative_config` | "vLLM spec 配置" | vLLM V1 中的显式可选加入；无默认意味着无加速 |
| N-gram spec decode | "N-gram draft" | GPU 端使用 prompt 中 N-gram 查找的 draft；与 chunked prefill 兼容 |
| 盈亏平衡 alpha | "无效 alpha" | spec decode 给出零加速的 alpha；在生产并发下关注此值 |
| 被拒 draft 两遍 | "重试成本" | draft 被拒时的两次目标前向；驱动 P99 尾部 |

## 延伸阅读

- [vLLM — Speculative Decoding docs](https://docs.vllm.ai/en/latest/features/spec_decode/) — 关于 `speculative_config` 和 V1 中 chunked-prefill 兼容性的权威来源。
- [vLLM Speculative Config API](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/) — 确切的字段集。
- [EAGLE paper (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) — 原始 EAGLE draft head 公式。
- [EAGLE-2 paper (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — 自适应 draft 和树。
- [UC Berkeley EECS-2025-224](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html) — 带投机解码的高效 LLM 系统。
- [BentoML — Speculative Decoding](https://bentoml.com/llm/inference-optimization/speculative-decoding) — 生产上线清单。
