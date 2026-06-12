# TensorRT-LLM 在 Blackwell 上的 FP8 和 NVFP4

> TensorRT-LLM 仅限 NVIDIA，但它在 Blackwell 上表现出色。在 GB200 NVL72 上使用 Dynamo 编排，SemiAnalysis InferenceX 在 2026 年 Q1-Q2 测得 120B 模型每百万 token 成本 $0.012，而 H100 + vLLM 为 $0.09——7 倍的经济差距。该技术栈叠加了三种浮点格式：FP8 因其动态范围仍然是 KV cache 和 attention 内核的关键选择；NVFP4（4-bit 微缩放）用于权重和激活值；多 token 预测（MTP）和分离式 prefill/decode 在此基础上再增加 2-3 倍。Day-0 模型支持直接加载 FP4 权重，无需训练后转换。2026 年工程团队需要注意的是：TRT-LLM 是封闭的 NVIDIA 栈，采用它意味着用可移植性换取吞吐量。在投入之前，根据你的模型和硬件组合算一笔账。

**类型：** 学习
**语言：** Python（stdlib，玩具 FP8/NVFP4 内存和成本计算器）
**前置课程：** 阶段 17 · 04（vLLM 推理内部机制），阶段 10 · 13（量化）
**时间：** 约 75 分钟

## 学习目标

- 解释为什么即使权重使用 NVFP4，FP8 对于 KV cache 和 attention 仍然至关重要。
- 计算前沿模型在 BF16、FP8 和 NVFP4 下的 HBM 占用，并分析节省来自哪里。
- 列出 TRT-LLM 利用的 Blackwell 特有功能（Day-0 FP4、MTP、分离式服务、all-to-all 原语）。
- 判断在什么情况下 TRT-LLM 的 NVIDIA 锁定值得 vs vLLM 在 Hopper 上的 7 倍成本差距。

## 问题描述

2026 年推理经济的前沿问题是"每美元能产生多少 token"。答案取决于四个叠加选择：硬件代次（Hopper H100/H200 vs Blackwell B200/GB200）、精度（BF16 → FP8 → NVFP4）、服务引擎（vLLM vs SGLang vs TRT-LLM）和编排（普通 vs 分离式 vs Dynamo）。

在 Hopper 上使用 vLLM，120B MoE 的运行成本约为每百万 token $0.09。在 Blackwell 上使用 TRT-LLM + Dynamo，同一模型的运行成本约为 $0.012——便宜 7 倍。差距的一部分来自硬件（Blackwell 的单 GPU LLM 吞吐量是 Hopper 的 11-15 倍）。另一部分来自技术栈：FP4 权重、MTP 草稿、分离式 prefill/decode 以及用于 MoE 专家通信的 NVLink 5 all-to-all。

你无法在 NVIDIA 栈之外复制这一点。这就是权衡——用可移植性换取经济性。理解哪些技术栈选择贡献了差距的多少，是本课的重点。

## 核心概念

### 为什么 FP8 仍然是 KV cache 的底线

2026 年的一个常见误区：假设 NVFP4 适用于所有场景。事实并非如此。KV cache 需要 FP8（8-bit 浮点数），因为它存储的 attention keys 和 values 覆盖了很大的动态范围。将 KV 量化到 FP4 会导致灾难性的精度损失——分布的尾部会丢失，attention scores 会崩溃。FP8 的指数位为 KV cache 提供了所需的范围。

NVFP4（2025-2026）适用于权重和激活值。微缩放：每个权重块有自己的缩放因子，使小块可以跨越不同的动态范围而不会损失逐张量缩放。对于激活值，FP4 可以胜任，因为激活值在单层内的范围较小。

典型的 Blackwell 配置：

- 权重：NVFP4（4-bit 微缩放）。
- 激活值：NVFP4。
- KV cache：FP8。
- Attention 累加器：FP32（softmax 稳定性）。

### TRT-LLM 使用的 Blackwell 特有原语

- **Day-0 FP4 权重**：模型提供者直接发布 FP4 权重；TRT-LLM 加载无需训练后转换。FP4 无需 AWQ / GPTQ 步骤。
- **多 token 预测（MTP）**：与 EAGLE（阶段 17 · 05）相同的思路，但集成到 TRT-LLM 构建中。
- **分离式服务**：prefill 和 decode 使用独立的 GPU 池，KV cache 通过 NVLink 或 InfiniBand 传输。与 Dynamo（阶段 17 · 20）相同的思路。
- **All-to-all 通信原语**：NVLink 5 将 MoE 专家通信延迟降低了 3 倍（相比 Hopper）。TRT-LLM 的 MoE 内核针对此进行了优化。
- **NVFP4 + MXFP8 微缩放**：Blackwell Tensor Cores 上的硬件加速缩放因子处理。

### 需要记住的数据

- HGX B200 通过 TRT-LLM 在 GPT-OSS-120B 上达到 $0.02/M tokens。
- GB200 NVL72 通过 Dynamo（编排 TRT-LLM）达到 $0.012/M tokens。
- H100 + vLLM 在类似工作负载上约 $0.09/M tokens。
- TRT-LLM 更新在三个月内带来 2.8 倍吞吐量提升（2026 年）。
- Blackwell vs Hopper：单 GPU LLM 吞吐量 11-15 倍。
- MLPerf Inference v6.0（2026 年 4 月）：Blackwell 在所有提交的任务中占据主导。

### FP4 在质量上的实际代价

NVFP4 是激进的选择。在推理密集型工作负载（思维链、数学、长上下文代码生成）上，FP4 权重的精度退化明显。逐块校准可以缓解但无法消除。发布推理模型的团队通常使用 FP8 权重 + FP4 激活值作为折衷方案，或坚持在 H200 上全面使用 FP8。

原则：在投入 NVFP4 权重之前，始终在你的评估集上验证任务质量。

### 为什么这是一个 NVIDIA 锁定决策

TRT-LLM 是 C++ + CUDA + 闭源内核。模型需要为特定 GPU SKU 编译。不支持 AMD、Intel、ARM。如果你的基础设施策略是多厂商，TRT-LLM 在 TRT-LLM 服务层不可行——你仍然可以在混合硬件上使用 vLLM 服务。如果你只用 NVIDIA，7 倍的差距足以弥补锁定成本。

### 2026 年实践方案

对于年推理预算超过 1 亿美元的情况，在 Hopper + vLLM 上运行意味着浪费 7-10 倍。将成本主导的工作负载迁移到 Blackwell + TRT-LLM + Dynamo。将实验层保留在 H100 + vLLM 上以保持模型迭代速度。在投产前对每个 NVFP4 转换的模型进行质量验证。

### 分离式的附加收益

TRT-LLM 的分离式服务（prefill 和 decode 使用独立池）在阶段 17 · 20 中有深入介绍。在 Blackwell 上，乘数叠加：FP4 权重 × MTP 加速 × 分离式部署 × 缓存感知路由。7 倍的数字假设使用了完整的技术栈。

```figure
pipeline-parallel
```

## 使用

`code/main.py` 计算模型在三种技术栈下的 HBM 占用、解码吞吐量（内存受限场景）和 $/M-tokens：H100 + BF16 + vLLM、H100 + FP8 + vLLM、B200 + NVFP4/FP8 + TRT-LLM。运行它以查看叠加效应以及每个变更贡献的差距份额。

## 交付

本课产出 `outputs/skill-trtllm-blackwell-advisor.md`。给定工作负载、模型大小和年度 token 量，判断 Blackwell + TRT-LLM 栈是否值得 NVIDIA 锁定。

## 练习

1. 运行 `code/main.py`。对于一个 30% 激活参数的 120B MoE，计算 H100 BF16、H100 FP8 和 B200 NVFP4/FP8 下的内存带宽受限解码吞吐量。最大的跃升来自哪里？
2. 一位客户每年在 H100 + vLLM 上花费 200 万美元。假设 7 倍经济差距，他们需要购买多少块 Blackwell GPU 才能在 12 个月内摊平迁移成本？
3. 你在 NVFP4 权重转换后看到 MATH 上精度下降了 3 个百分点。列出两个恢复路径：一个质量优先（保持 FP8 权重），一个成本优先（用领域内数据校准）。
4. 阅读 MLPerf v6.0 推理结果。哪个任务的 Blackwell-over-Hopper 差距最小？为什么？
5. 计算 405B 模型在 NVFP4 权重 + FP8 KV cache、128k 上下文下所需的 HBM。能否放入单个 GB200 NVL72 节点？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| FP8 | "8-bit 浮点" | 8 位浮点数；因动态范围用于 KV cache 和 attention |
| NVFP4 | "4-bit 微缩放" | NVIDIA 的 4-bit 微缩放 FP 格式；Blackwell 上的权重和激活值 |
| MXFP8 | "MX eight" | 微缩放 FP8 变体；Blackwell Tensor Cores 上的硬件加速 |
| Day-0 FP4 | "发布 FP4 权重" | 模型提供者直接发布 FP4 格式的权重；无需训练后转换 |
| MTP | "多 token 预测" | TRT-LLM 集成的推测解码草稿（阶段 17 · 05） |
| 分离式服务 | "分离 prefill/decode" | prefill 和 decode 使用独立 GPU 池；KV 通过 NVLink/IB 传输 |
| All-to-all | "MoE 专家通信" | 将 token 路由到专家 GPU 的通信模式；NVLink 5 降低 3 倍延迟 |
| InferenceX | "SemiAnalysis 推理基准" | 2026 年行业公认的每 token 成本基准 |

## 延伸阅读

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — 2026 年 4 月 MLPerf 结果。
- [NVIDIA — MoE Inference on Blackwell](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) — NVLink 5 all-to-all 和 MoE 内核。
- [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html) — 官方引擎文档。
- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) — TRT-LLM 之上的分离式编排。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 发布 Blackwell 数据的基准测试套件。
