# 生产量化 — AWQ、GPTQ、GGUF K-quants、FP8、MXFP4/NVFP4

> 量化格式不是通用选择——它是硬件、服务引擎和工作负载的函数。GGUF Q4_K_M 或 Q5_K_M 主导 CPU 和边缘设备，通过 llama.cpp 和 Ollama 交付。GPTQ 在 vLLM 中需要同一基础模型上的多 LoRA 时胜出。AWQ 配合 Marlin-AWQ 内核在 7B 级模型上以 INT4 实现最佳 Pass@1，达到约 741 tok/s——2026 年数据中心生产的默认选择。FP8 在 Hopper、Ada 和 Blackwell 上保持中间路线——近无损且广泛支持。NVFP4 和 MXFP4（Blackwell 微缩放）是激进的选择，需要逐块验证。两个常见的坑：校准数据集必须匹配部署领域，以及 KV cache 与权重量化是独立的——AWQ 课中"我的模型现在只有 4 GB"的说法忽略了生产批大小下 10-30 GB 的 KV cache。

**类型：** 学习
**语言：** Python（stdlib，玩具格式间内存和吞吐量比较）
**前置知识：** Phase 10 · 13（Quantization 基础），Phase 17 · 04（vLLM Serving Internals）
**时间：** 约 75 分钟

## 学习目标

- 列出六种生产量化格式及其在 2026 年的最佳适用场景。
- 根据硬件（CPU vs GPU，Hopper vs Blackwell）、引擎（vLLM、TRT-LLM、llama.cpp）和工作负载（日常聊天、推理、多 LoRA）选择格式。
- 计算所选格式节省的权重内存和未受影响的 KV cache 大小。
- 说出导致量化模型在领域流量上精度退化的校准数据集陷阱。

## 问题描述

量化减少内存和 HBM 带宽，这正是解码所需要的。一个 FP16 的 70B 模型有 140 GB 权重。将权重量化到 INT4（AWQ 或 GPTQ），模型变为 35 GB——可以放入一块 H100 并为 KV cache 留出空间，这在 128 个并发序列、2k 上下文时很重要，此时仅 KV cache 就有 20-30 GB。

但量化并非免费。激进的量化会降低质量，尤其是在推理密集型任务上。不同格式适配不同引擎。不同硬件原生支持不同精度。2026 年的格式动物园是真实存在的，你不能照搬别人的选择——必须根据你的技术栈来选择。

## 核心概念

### 六种格式

| 格式 | 位宽 | 最佳场景 | 引擎 |
|------|------|----------|------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU、边缘设备、笔记本 | llama.cpp, Ollama |
| GPTQ | 4-8 | vLLM 上的多 LoRA | vLLM, TGI |
| AWQ | 4 | 数据中心 GPU 生产 | vLLM (Marlin-AWQ), TGI |
| FP8 | 8 | Hopper/Ada/Blackwell 数据中心 | vLLM, TRT-LLM, SGLang |
| MXFP4 | 4 | Blackwell 多用户 | TRT-LLM |
| NVFP4 | 4 | Blackwell 多用户 | TRT-LLM |

### GGUF — CPU/边缘默认选择

GGUF 是一种文件格式，本身不是量化方案——它在一个容器中打包 K-quant 变体（Q2_K、Q3_K_M、Q4_K_M、Q5_K_M、Q6_K、Q8_0）。Q4_K_M 和 Q5_K_M 是生产默认选择——在 4-5 位下接近 BF16 质量。是 CPU 或边缘服务的最佳选择，因为 llama.cpp 是目前最快的 CPU 推理引擎。

vLLM 中的吞吐量损失：7B 上约 93 tok/s——该格式未针对 GPU 内核优化。仅在部署目标为 CPU/边缘时使用 GGUF。其他情况不推荐。

### GPTQ — vLLM 中的多 LoRA

GPTQ 是一种训练后量化算法，带有校准过程。Marlin 内核使其在 GPU 上快速运行（比非-Marlin GPTQ 快 2.6 倍）。7B 上约 712 tok/s。

独特优势：GPTQ-Int4 在 vLLM 中支持 LoRA 适配器。如果你服务的基础模型加上 10-50 个微调变体（每个作为 LoRA），GPTQ 是你的选择。截至 2026 年初，NVFP4 尚不支持 LoRA。

### AWQ — 数据中心 GPU 默认选择

激活感知权重量化（Activation-aware Weight Quantization）。在量化过程中保护约 1% 最显著的权重。Marlin-AWQ 内核：比朴素方法快 10.9 倍。7B 上约 741 tok/s，在 INT4 格式中具有最佳 Pass@1。

选择 AWQ 作为新的 GPU 服务方案，除非你需要多 LoRA（GPTQ）或 Blackwell 激进的 FP4（NVFP4）。

### FP8 — 可靠的中间路线

8-bit 浮点数。近无损。广泛支持。Hopper Tensor Cores 原生加速 FP8。Blackwell 继承。FP8 是 2026 年在质量不可妥协（推理、医疗、代码生成）时的安全默认选择。内存节省是 INT4 的一半，但质量风险低得多。

### MXFP4 / NVFP4 — Blackwell 激进选择

微缩放 FP4。每个权重块有自己的缩放因子。激进但在 Blackwell Tensor Cores 上有硬件加速。与 FP8 相比每 token 字节数减半——这就是 Phase 17 · 07 中的经济优势。

注意事项：
- 尚不支持 LoRA（2026 年初）。
- 在推理密集型工作负载上精度下降明显。
- 在每个模型的评估集上进行验证。

### 校准陷阱

AWQ 和 GPTQ 需要校准数据集——通常是 C4 或 WikiText。对于领域模型（代码、医疗、法律），在通用网页文本上校准会让算法对哪些权重需要保护做出错误决策。HumanEval 上的 Pass@1 可能下降几个百分点。

解决方案：在领域内数据上校准。通常几百个领域样本就够了。在发布前在评估集上测试。

### KV cache 陷阱

AWQ 将权重压缩到 4 位。KV cache 是独立的，保持 FP16/FP8。对于一个使用 AWQ 的 70B 模型：

- 权重：约 35 GB（从 140 GB 量化到 INT4）。
- KV cache 在 128 并发 × 2k 上下文下：约 20 GB。
- 激活值：约 5 GB。
- 总计：约 60 GB——可以放入 H100 80GB。

天真地认为"我把模型量化到了 4 GB"忽略了其他 30-50 GB。要全面预算 HBM。

另外，KV cache 量化（FP8 KV 或 INT8 KV）是一个独立的选择，有其自身的权衡——它直接影响 attention 精度，不是免费的收益。

### AWQ INT4 对推理有风险

思维链、数学、长上下文代码生成——这些会受到激进量化的明显影响。AWQ INT4 在 MATH 上损失约 3-5 个百分点。对于推理密集型工作负载，使用 FP8 或 BF16 发布；接受内存成本。

### 2026 年选择指南

- CPU/边缘服务：GGUF Q4_K_M。搞定。
- GPU 服务，日常聊天，无 LoRA：AWQ。
- GPU 服务，多 LoRA：使用 Marlin 的 GPTQ。
- 推理工作负载：FP8。
- Blackwell 数据中心，已验证质量：NVFP4 + FP8 KV。
- 不确定：在每种候选格式上运行 1,000 样本评估。

```figure
gpu-memory-breakdown
```

## 使用方法

`code/main.py` 计算六种格式在不同模型大小下的内存占用（权重 + KV + 激活值）和相对吞吐量。展示 KV cache 主导的场景、权重压缩有收益的场景以及 FP8 是安全选择的场景。

## 实践产出

本课产出 `outputs/skill-quantization-picker.md`。给定硬件、模型大小、工作负载类型和质量容忍度，选择格式并生成校准/验证计划。

## 练习

1. 运行 `code/main.py`。对于 128 并发、2k 上下文的 70B 模型，计算每种格式的总 HBM。哪种格式可以放入一块 H100 80GB？
2. 你有一个 7B 的编码模型。选择一种格式并说明理由。如果你对质量容忍度判断错误，恢复路径是什么？
3. 计算为医疗领域模型校准 AWQ 所需的校准数据集大小。为什么更多数据并不总是更好？
4. 阅读 Marlin-AWQ 内核论文或发布说明。用三句话解释为什么 AWQ 在 7B 上达到 741 tok/s 而原始 GPTQ 约 712。
5. 在什么情况下将 AWQ 权重与 FP8 KV cache 结合使用有意义，与保持 KV 为 BF16 相比？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| GGUF | "llama.cpp 格式" | 打包 K-quant 变体的文件格式；CPU/边缘默认 |
| Q4_K_M | "Q4 K M" | 4-bit K-quant medium；生产级 GGUF 默认 |
| GPTQ | "gee pee tee q" | 带校准的训练后 INT4；在 vLLM 中支持 LoRA |
| AWQ | "a w q" | 激活感知 INT4；Marlin 内核；INT4 中最佳 Pass@1 |
| Marlin 内核 | "快速 INT4 内核" | Hopper 上 INT4 的定制 CUDA 内核；10 倍加速 |
| FP8 | "8-bit 浮点" | Hopper/Ada/Blackwell 上的安全精度默认 |
| MXFP4 / NVFP4 | "微缩放四位" | Blackwell 4-bit FP，逐块缩放因子 |
| 校准数据集 | "校准数据" | 用于选择量化参数的输入文本；必须匹配领域 |
| KV cache 量化 | "KV INT8" | 与权重独立的选择；影响 attention 精度 |

## 延伸阅读

- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — 比较基准测试。
- [Jarvis Labs — vLLM Quantization Complete Guide](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) — 按格式列出的吞吐量数据。
- [PremAI — GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/) — 逐格式选择指南。
- [vLLM docs — Quantization](https://docs.vllm.ai/en/latest/features/quantization/index.html) — 支持的格式和标志。
- [AWQ 论文 (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978) — 原始 AWQ 公式。
- [GPTQ 论文 (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323) — 原始 GPTQ 公式。
