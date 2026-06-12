# 推理指标 — TTFT、TPOT、ITL、Goodput、P99

> 四个指标决定了推理部署是否正常工作。TTFT 是预填充加排队加网络。TPOT（等同于 ITL）是每个 token 的内存受限解码成本。端到端延迟是 TTFT 加上 TPOT 乘以输出长度。吞吐量是整个集群聚合的每秒 token 数。但对产品真正重要的是 goodput——同时满足所有 SLO 的请求比例。高吞吐量但低 goodput 意味着你正在处理永远无法及时到达用户的 token。2026 年 Llama-3.1-8B-Instruct 在 TRT-LLM 上的参考数据：平均 TTFT 162 ms，平均 TPOT 7.33 ms，平均 E2E 1,093 ms。始终报告 P50、P90、P99——永远不要只报告平均值。还要注意度量陷阱：GenAI-Perf 在 ITL 计算中排除 TTFT，LLMPerf 包含 TTFT；两个工具对同一运行的 TPOT 给出不同的结果。

**类型：** 学习
**语言：** Python（stdlib，玩具百分位计算器和 goodput 报告器）
**前置课程：** 阶段 17 · 04（vLLM 推理内部机制）
**时间：** 约 60 分钟

## 学习目标

- 精确定义 TTFT、TPOT、ITL、E2E、吞吐量和 goodput，并说出每个指标衡量的组件。
- 解释为什么平均值不是 LLM 服务的正确统计量，以及如何解读 P50/P90/P99。
- 构建 SLO 多约束条件（例如 TTFT<500 ms 且 TPOT<15 ms 且 E2E<2 s）并计算对应的 goodput。
- 列出两个在同一运行中对 TPOT 给出不同结果的基准测试工具，并解释原因。

## 问题描述

"我们的吞吐量是每秒 15,000 token。"所以呢？如果 40% 的请求端到端超过 2 秒，用户会放弃会话。仅凭吞吐量无法告诉你产品是否正常工作。

推理有多个延迟维度，每个维度的失败模式不同。Prefill 是计算受限的，与 prompt 长度成比例。Decode 是内存受限的，与批大小成比例。排队延迟是运维问题。网络是物理距离问题。你需要为每个维度设定不同的指标，需要百分位数，还需要一个综合指标来回答"用户是否得到了他们期望的体验"——这就是 goodput。

## 核心概念

### TTFT — 首个 token 时间

`TTFT = queue_time + network_request + prefill_time`

当 prompt 较长时，prefill 主导。在 H100 上的 Llama-3.3-70B FP8 中，32k prompt 需要约 800 ms 的纯预填充时间。队列时间是负载下的调度器行为。网络请求是包含 TLS 的线路时间。TTFT 是用户在任何内容流式返回之前看到的延迟。

### TPOT / ITL — token 间延迟

多个名称指代同一个量。`TPOT`（time per output token）、`ITL`（inter-token latency）、`decode latency per token`——都是同一个东西。它是第一个 token 之后连续流式 token 之间的时间。

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

在同一 Llama-3.3-70B H100 栈上使用分块 prefill 时，TPOT 平均约 7 ms。不使用分块 prefill 时，在相邻序列的长 prefill 期间，TPOT 可能飙升到 50 ms。关注 P99，而非平均值。

### E2E 延迟

`E2E = TTFT + TPOT * output_tokens + network_response`

对于长输出（>500 token），E2E 由 TPOT 主导。对于短输出和长 prompt，E2E 由 TTFT 主导。报告以输出长度为条件的 E2E。

### 吞吐量

`throughput = total_output_tokens / elapsed_time`

聚合指标。告诉你集群效率。不能告诉你单个请求的健康状况。

### Goodput — 你真正关心的指标

`goodput = fraction of requests meeting (TTFT <= a) AND (TPOT <= b) AND (E2E <= c)`

SLO 是一个多约束条件。只有当每个约束都满足时，请求才被视为"良好"。Goodput 是这个比例。60% goodput 下的高吞吐量是失败。99% goodput 下的较低吞吐量才是目标。

2026 年，goodput 是 MLPerf Inference v6.0 提交中使用的指标，也是 AI 平台提供商内部 SLA 跟踪的指标。

### 为什么平均值是错误的统计量

LLM 延迟分布是右偏的。一个 decode 批次中有一个长 prefill 的邻居时，可能 500 个 token 的 TPOT 约 7 ms，而 20 个 token 的 TPOT 约 60 ms。平均 TPOT 是 9 ms。P99 TPOT 是 65 ms。用户经常遇到 P99——这就是他们离开的原因。

始终报告三元组（P50、P90、P99）。对于用户体验，P99 是你需要优化的指标。

### 参考数据 — Llama-3.1-8B-Instruct on TRT-LLM, 2026

- 平均 TTFT：162 ms
- 平均 TPOT：7.33 ms
- 平均 E2E：1,093 ms
- P99 TPOT：因分块 prefill 配置而异，10-25 ms。

这些是 NVIDIA 发布的参考点。它们随模型大小（70B 会显示 3-5 倍）、硬件（H100 vs B200 约 3 倍）和负载而变化。

### 度量陷阱

2026 年两个最常用的基准测试工具对同一运行的 TPOT 给出不同的结果：

- **NVIDIA GenAI-Perf**：在 ITL 计算中排除 TTFT。ITL 从第 2 个 token 开始。
- **LLMPerf**：包含 TTFT。ITL 从第 1 个 token 开始。

对于一个 TTFT 为 500 ms、总解码 700 ms 内产生 100 个输出 token 的请求，GenAI-Perf 报告 `ITL = 700/99 = 7.07 ms`，LLMPerf 报告 `ITL = 1200/100 = 12.00 ms`。工具的选择改变了数字。

始终声明使用哪个工具。始终发布定义。

### 构建 SLO

2026 年面向消费者的 70B 聊天模型的合理 SLO：

- TTFT P99 <= 800 ms。
- TPOT P99 <= 25 ms。
- E2E P99 <= 3 s（针对 <300 token 输出）。
- Goodput 目标 >= 99%。

企业 SLO 会收紧 TTFT（200-400 ms）并放宽 E2E。关键是要写下来，测量所有三个指标，并将 goodput 作为单一综合指标进行跟踪。

### 如何度量

- 运行真实流量或逼真的合成流量（LLMPerf 使用 `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`）。
- 基准测试运行以 2 倍峰值并发为目标。
- 运行 30-50 次迭代，对合并样本取百分位数。
- 发布时注明工具名称、工具版本、模型、硬件、并发数、prompt 分布。

```figure
throughput-latency
```

## 使用

`code/main.py` 是一个玩具 goodput 计算器。生成合成延迟分布，应用 SLO，并计算 goodput。还展示了 GenAI-Perf vs LLMPerf 在同一 trace 上的 TPOT 差异。

## 交付

本课产出 `outputs/skill-slo-goodput-gate.md`。给定工作负载和 SLO，生成一个 CI/CD 就绪的基准测试方案，用 goodput（而非吞吐量）来把关部署。

## 练习

1. 运行 `code/main.py`。生成一个带有 1% 尾部尖峰的分布。当 P99 TPOT 从 30 ms 收紧到 15 ms 时，goodput 如何变化？
2. 一位供应商报价"Llama 3.3 70B H100 上 15,000 tok/s"。在相信之前，列出三个要问的问题。
3. 为什么分块 prefill 能保护 P99 TPOT 但不能保护平均 TPOT？
4. 为语音助手构建一个消费级 SLO（首个 token 是被听到的，而不是被读到的）。哪个指标对用户最可见？
5. 阅读 LLMPerf README 和 GenAI-Perf 文档。找出工具之间存在分歧的另外三个指标。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| TTFT | "首个 token 时间" | 队列 + 网络 + prefill；长 prompt 时由 prefill 主导 |
| TPOT | "每个输出 token 的时间" | 首个 token 之后的每 token 内存受限解码成本 |
| ITL | "token 间延迟" | 大多数工具中等同于 TPOT（并非所有——见 GenAI-Perf） |
| E2E | "端到端" | TTFT + TPOT * output_len；加上响应侧网络 |
| 吞吐量 | "tok/s" | 集群效率；没有延迟百分位数就毫无意义 |
| Goodput | "SLO 达标率" | 同时满足所有 SLO 约束的请求比例 |
| P99 | "尾部" | 1% 最差情况延迟；用户体验指标 |
| SLO 多约束 | "联合约束" | 所有三个延迟边界的 AND；任何一个被违反即算失败 |
| GenAI-Perf vs LLMPerf | "工具陷阱" | 工具对 ITL 是否包含 TTFT 存在分歧 |

## 延伸阅读

- [NVIDIA NIM — LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — TTFT、ITL、TPOT 的规范定义。
- [Anyscale — LLM Serving Benchmarking Metrics](https://docs.anyscale.com/llm/serving/benchmarking/metrics) — 替代定义和度量方案。
- [BentoML — LLM Inference Metrics](https://bentoml.com/llm/inference-optimization/llm-inference-metrics) — 真实部署上的应用度量。
- [LLMPerf](https://github.com/ray-project/llmperf) — 基于 Ray 的开源基准测试。
- [GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c++/perf_analyzer/genai-perf/README.html) — NVIDIA 的基准测试工具。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 行业公认的基于 goodput 的基准测试。
