# LLM API 负载测试 —— 为什么 k6 和 Locust 会误导你

> 传统的负载测试工具并非为流式响应、可变输出长度、Token 级指标或 GPU 饱和而设计。大多数团队会掉入两个陷阱。GIL 陷阱：Locust 的 Token 级测量在 Python GIL 下运行分词（Tokenization）过程，在高并发下与请求生成竞争资源；分词积压会夸大报告的 Token 间延迟（Inter-Token Latency）——成为瓶颈的是您的客户端，而非服务器。提示词同质化陷阱：在循环中使用完全相同的提示词，只测试了 Token 分布上的一个点；而真实流量具有可变长度和多样化的前缀匹配。LLMPerf 通过 `--mean-input-tokens` + `--stddev-input-tokens` 解决了这个问题。2026 年的工具映射：LLM 专用工具（GenAI-Perf、LLMPerf、LLM-Locust、guidellm）用于确保 Token 级精度；**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）** —— 具备流式感知能力，通过 TestRun/PrivateLoadZone CRD 实现 Kubernetes 原生分布式测试，最适合 CI/CD 门禁；Vegeta 用于 Go 的恒定速率饱和测试；Locust 2.43.3 原生版本仅能通过 LLM-Locust 扩展支持流式测试。负载模式：稳态（Steady-state）、爬坡（Ramp）、突刺（Spike，用于自动扩缩容测试）、长稳（Soak，用于检测内存泄漏）。

**类型：** 构建
**语言：** Python（标准库，玩具级真实提示词生成器 + 延迟收集器）
**前置知识：** 第 17 阶段 · 08（推理指标），第 17 阶段 · 03（GPU 自动扩缩容）
**时间：** 约 75 分钟

## 学习目标

- 解释为什么通用负载测试工具会对 LLM API 产生误导的两种反模式（GIL 陷阱、提示词同质化陷阱）。
- 根据不同目的选择合适的工具：LLMPerf（基准性能运行）、k6 + 流式扩展（CI 门禁）、guidellm（大规模合成测试）、GenAI-Perf（NVIDIA 官方基准）。
- 设计四种负载模式（稳态、爬坡、突刺、长稳）并说明每种模式捕获的故障类型。
- 使用输入 Token 的均值和标准差（Mean + Stddev）构建真实的提示词分布，而非使用固定长度。

## 问题所在

您使用 k6 在 500 个并发用户下测试了 LLM 端点。测试通过了。您上线了。在生产环境中，仅有 200 个真实用户时服务就崩溃了——P99 TTFT 暴涨，GPU 资源耗尽。

发生了两件事。第一，k6 发送了 500 个完全相同的提示词——您的请求合并（Request Coalescing）和前缀缓存（Prefix Caching）让系统看起来像是在处理 500 个并发解码，但实际上只处理了一个请求。第二，k6 像 HTTP 一样看待流式响应，而不是像人眼那样体验它；它看到的是一个 HTTP 连接，而不是 500 个以不同间隔到达的 Token。

LLM 的负载测试是一门独立的学科。

## 核心概念

### GIL 陷阱（Locust）

Locust 使用 Python 编写，并在 GIL 下运行客户端分词（Tokenization）过程。在高并发下，分词器会在请求生成后排队等待。报告的 Token 间延迟包含了客户端的分词积压时间。您认为服务器很慢；其实是测试工具本身成为了瓶颈。

解决方案：LLM-Locust 扩展将分词移至独立进程，或使用编译语言的测试工具（如 k6、使用 tokenizers.rs 的 LLMPerf）。

### 提示词同质化陷阱

所有已知的负载测试工具都允许您配置单个提示词。在循环执行 10,000 次的测试中，每次发送的都是完全相同的提示词。服务器每次都看到相同的前缀——前缀缓存命中率接近 100%，吞吐量看起来非常出色。

解决方案：从提示词分布中进行采样。LLMPerf 使用 `--mean-input-tokens 500 --stddev-input-tokens 150`——生成多样化的长度和内容。

### 四种负载模式

1. **稳态（Steady-state）** —— 以恒定 RPS 持续 30-60 分钟。捕获：基线性能回归。
2. **爬坡（Ramp）** —— 在 15 分钟内将 RPS 从 0 线性增加到目标值。捕获：容量断点、预热异常。
3. **突刺（Spike）** —— 突然将 RPS 提高 3-10 倍，持续 2 分钟后恢复。捕获：自动扩缩容延迟、队列饱和、冷启动影响。
4. **长稳（Soak）** —— 以稳态运行 4-8 小时。捕获：内存泄漏、连接池漂移、可观测性系统溢出。

### 2026 年工具映射

**LLMPerf** (Anyscale) —— 基于 Python 但使用 Rust 支持的分词器。支持均值/标准差提示词。具备流式感知能力。性能运行的最佳默认选择。

**NVIDIA GenAI-Perf** —— NVIDIA 官方参考基准。使用 Triton 客户端；指标覆盖全面。请注意其 ITL 指标不包含 TTFT；而 LLMPerf 包含。对同一服务器，两个工具会得出不同的 TPOT 结果。

**LLM-Locust** (TrueFoundry) —— 解决了 GIL 陷阱的 Locust 扩展。提供熟悉的 Locust DSL 语法和流式指标。

**guidellm** —— 大规模合成基准测试工具。

**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**：
- k6 本身（Go 语言编写，编译型，无 GIL）增加了流式感知指标。
- k6 Operator 使用 TestRun / PrivateLoadZone CRD 实现 Kubernetes 原生分布式测试。
- 最适合 CI/CD 门禁和 SLA 测试。

**Vegeta** —— Go 语言编写，比 k6 更简单。恒定速率 HTTP 饱和测试。不具备 LLM 感知能力，但适合用于网关/速率限制测试。

**Locust 2.43.3 原生版本** —— 存在 LLM 的 GIL 陷阱。仅可通过 LLM-Locust 扩展使用。

### CI 中的 SLA 门禁

在 PR 上运行 k6：

- 在基线 RPS 下运行 30-50 次迭代。
- 门禁条件：P50/P95 TTFT，5xx 错误率 < 5%，TPOT 低于阈值。
- 若不满足条件则构建失败。

### 真实的提示词分布

从真实流量样本（如果您有的话）或已发布的分布数据（例如用于聊天的 ShareGPT 提示词、用于代码的 HumanEval 数据集）构建。将均值和标准差（Mean + Stddev）输入 LLMPerf。应不惜一切代价避免使用单一提示词进行循环测试。

### 您应该记住的数字

- k6 Operator 1.0 GA：2025 年 9 月。
- k6 v2026.1.0：具备流式感知指标。
- 典型 LLMPerf 运行：在并发度 X 下运行 100-1000 个请求。
- 典型 CI 门禁：每个 PR 运行 30-50 次迭代。
- 四种模式：稳态、爬坡、突刺、长稳。

## 使用它

`code/main.py` 模拟了一个具有真实提示词分布的负载测试，测量了有效的 TPOT，并展示了单一提示词陷阱。

## 交付它

本课程产出 `outputs/skill-load-test-plan.md`。根据工作负载和 SLA，选择工具并设计四种负载模式。

## 练习

1. 运行 `code/main.py`。比较均匀分布与真实分布——差距在哪里？
2. 编写一个用于 CI 门禁的 k6 脚本：在 100 个并发下，TTFT P95 < 800 ms，运行时间 5 分钟。
3. 您的长稳测试显示内存以每小时 50 MB 的速度增长。列出三个可能的原因以及用于区分这些原因的监测方法。
4. 进行一次从 10 RPS 到 100 RPS 的突刺测试。如果 Karpenter + vLLM 生产技术栈已就位（第 17 阶段 · 03 + 18），预期的恢复时间是多少？
5. GenAI-Perf 报告 TPOT=6ms；LLMPerf 在同一服务器上报告 TPOT=11ms。解释原因。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| LLMPerf | “LLM 测试工具” | Anyscale 的基准测试工具，具备流式感知能力 |
| GenAI-Perf | “NVIDIA 工具” | NVIDIA 官方参考基准工具 |
| LLM-Locust | “面向 LLM 的 Locust” | 修复了 GIL 陷阱的 Locust 扩展 |
| guidellm | “合成基准测试” | 大规模合成测试工具 |
| k6 Operator | “K8s k6” | 基于 CRD 的分布式 k6 |
| GIL 陷阱 | “Python 客户端开销” | 分词积压夸大了报告的延迟 |
| 提示词同质化陷阱 | “单一提示词谎言” | 循环使用相同提示词会命中缓存，夸大吞吐量 |
| 稳态（Steady-state） | “恒定负载” | 以恒定 RPS 持续 N 分钟 |
| 爬坡（Ramp） | “线性增加” | 在指定时长内从 0 增加到目标值 |
| 突刺（Spike） | “突发测试” | 突然的倍数增加然后恢复 |
| 长稳（Soak） | “长时间测试” | 持续数小时用于泄漏检测 |

## 延伸阅读

- [TianPan — LLM 应用程序的负载测试](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — 2026 年 LLM 负载测试](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — LLM 推理基准测试入门](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)