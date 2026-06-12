# 自托管推理引擎选型 — llama.cpp、Ollama、TGI、vLLM、SGLang

> 2026 年四大引擎主导自托管推理。根据硬件、规模和生态选择。**llama.cpp** 在 CPU 上最快 — 最广的模型支持、完全控制量化和线程。**Ollama** 是开发笔记本上的一键安装，比 llama.cpp 慢约 15-30%（Go + CGo + HTTP 序列化），在类生产负载下吞吐量差距达 3 倍。**TGI 于 2025 年 12 月 11 日进入维护模式** — 仅修复 bug，原始吞吐量比 vLLM 慢约 10%，但历史可观测性和 HF 生态集成最好。维护状态使其成为长期风险选项 — 新项目用 SGLang 或 vLLM 更安全。**vLLM** 是通用生产默认 — v0.15.1（2026 年 2 月）增加 PyTorch 2.10、RTX Blackwell SM120、H200 优化。**SGLang** 是多轮 Agent / 前缀密集场景的专家 — 生产环境 400,000+ GPU（xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS）。硬件约束：仅 CPU → 只能用 llama.cpp。AMD / 非 NVIDIA → 只能用 vLLM（TRT-LLM 仅限 NVIDIA）。2026 年管线模式：开发 = Ollama，预发布 = llama.cpp，生产 = vLLM 或 SGLang。全程使用相同的 GGUF/HF 权重。

**类型：** 学习
**语言：** Python（stdlib，引擎决策树遍历器）
**前置课程：** 阶段 17 涵盖引擎的所有课程（04、06、07、09、18）
**时间：** 约 45 分钟

## 学习目标

- 根据硬件（CPU / AMD / NVIDIA Hopper / Blackwell）、规模（1 用户 / 100 / 10,000）和工作负载（通用聊天 / Agent / 长上下文）选择引擎。
- 说出 2026 年 TGI 维护模式状态（2025 年 12 月 11 日）以及为什么它使新项目偏向 vLLM 或 SGLang。
- 描述使用相同 GGUF 或 HF 权重的开发/预发布/生产管线。
- 解释为什么"仅 CPU"强制使用 llama.cpp 而"AMD"排除 TRT-LLM。

## 问题背景

你的团队开始一个新的自托管 LLM 项目。一个工程师说 Ollama，另一个说 vLLM，第三个说"TGI 不是开箱即用吗？"三者在不同场景下都对。没有一个在所有场景下都对。

2026 年选择树很重要：硬件第一，规模第二，工作负载第三。而一个特定的 2025 年事件 — TGI 于 12 月 11 日进入维护模式 — 改变了新项目的默认选择。

## 核心概念

### 五大引擎

| 引擎 | 最佳用途 | 备注 |
|------|---------|------|
| **llama.cpp** | CPU / 边缘 / 最小依赖 / 最广模型支持 | CPU 上最快，完全控制 |
| **Ollama** | 开发笔记本、单用户、一键安装 | 比 llama.cpp 慢 15-30%；生产吞吐量差距 3 倍 |
| **TGI** | HF 生态、受监管行业 | **2025 年 12 月 11 日进入维护模式** |
| **vLLM** | 通用生产、100+ 用户 | 广泛生产默认；v0.15.1 2026 年 2 月 |
| **SGLang** | 多轮 Agent、前缀密集工作负载 | 生产环境 400,000+ GPU |

### 硬件优先决策

**仅 CPU** → llama.cpp。Ollama 也能用但更慢。其他引擎在 CPU 上没有竞争力。

**AMD GPU** → vLLM（AMD ROCm 支持）。SGLang 也能用。TRT-LLM 仅限 NVIDIA，排除。

**NVIDIA Hopper（H100 / H200）** → vLLM 或 SGLang 或 TRT-LLM。三者都是顶级。

**NVIDIA Blackwell（B200 / GB200）** → TRT-LLM 是吞吐量领先者（阶段 17 · 07）。vLLM 和 SGLang 紧随其后。

**Apple Silicon（M 系列）** → llama.cpp（Metal）。Ollama 封装了这个。

### 规模第二决策

**1 用户 / 本地开发** → Ollama。一条命令，首 token 秒级响应。

**10-100 用户 / 小团队** → vLLM 单 GPU。

**100-10k 用户 / 生产** → vLLM 生产栈（阶段 17 · 18）或 SGLang。

**10k+ 用户 / 企业** → vLLM 生产栈 + 分离部署（阶段 17 · 17）+ LMCache（阶段 17 · 18）。

### 工作负载第三决策

**通用聊天 / QA** → vLLM 通用默认胜出。

**多轮 Agent（工具、规划、记忆）** → SGLang 的 RadixAttention（阶段 17 · 06）主导。

**前缀重用密集的 RAG** → SGLang。

**代码生成** → vLLM 可以；SGLang 在缓存上稍好。

**长上下文（128K+）** → vLLM + 分块预填充；SGLang + 分层 KV。

### TGI 维护陷阱

Hugging Face TGI 于 2025 年 12 月 11 日进入维护模式 — 之后仅修复 bug。历史上：顶级可观测性、一流的 HF 生态集成（模型卡、安全工具），原始吞吐量略落后于 vLLM。

2026 年新项目：默认避开 TGI。现有 TGI 部署可以继续，但最终应迁移。SGLang 和 vLLM 是更安全的默认。

### 管线模式

开发（Ollama）→ 预发布（llama.cpp）→ 生产（vLLM）。全程使用相同的 GGUF 或 HF 权重。工程师在笔记本上快速迭代；预发布镜像生产量化；生产是推理目标。

### Ollama 注意事项

Ollama 很适合开发。不适合共享生产：Go HTTP 序列化增加开销，并发管理比 vLLM 简单，OpenTelemetry 支持落后。在 Ollama 擅长的地方使用 — 单用户、一条命令 — 共享时切换到 vLLM。

### 自托管 vs 托管是独立决策

阶段 17 · 01（托管超大规模）、· 02（推理平台）涵盖托管。本课假设你已决定自托管。自托管原因：数据驻留、自定义微调、规模化总拥有成本、托管上没有的领域模型。

### 需要记住的数字

- TGI 维护模式：2025 年 12 月 11 日。
- vLLM v0.15.1：2026 年 2 月；PyTorch 2.10；Blackwell SM120 支持。
- SGLang 生产规模：400,000+ GPU。
- Ollama 吞吐量差距 vs llama.cpp：慢 15-30%；生产负载下 3 倍差距。

```figure
data-parallel
```

## 使用

`code/main.py` 是一个决策树遍历器：给定硬件 + 规模 + 工作负载，选择引擎并解释原因。

## 交付

本课产出 `outputs/skill-engine-picker.md`。给定约束，选择引擎并写出迁移计划。

## 练习

1. 使用你的硬件 / 规模 / 工作负载运行 `code/main.py`。输出是否符合你的直觉？
2. 你的基础设施是 12 台 H100 和 8 台 MI300X AMD。选哪个引擎？为什么 TRT-LLM 不在选项中？
3. 一个团队想在 2026 年使用 TGI，因为"这是我们熟悉的"。论证迁移的理由。
4. Ollama 开发到 vLLM 生产：量化、配置和可观测性有什么变化？
5. P99 前缀长度 8K 且跨租户高重用的 RAG 产品。选择一个引擎并与阶段 17 · 11 + 18 叠加。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| llama.cpp | "那个 CPU 的" | 最广模型支持，CPU 上最快 |
| Ollama | "那个笔记本的" | 一键安装，开发级吞吐量 |
| TGI | "HF 的推理" | 2025 年 12 月起维护模式 |
| vLLM | "那个默认的" | 2026 年广泛生产基线 |
| SGLang | "那个 Agent 的" | 前缀密集，RadixAttention |
| TRT-LLM | "仅限 NVIDIA 的" | Blackwell 吞吐量领先者，仅 NVIDIA |
| GGUF | "llama.cpp 格式" | 捆绑 K-quant 变体 |
| 生产栈 | "vLLM K8s" | 阶段 17 · 18 参考部署 |
| 管线模式 | "开发→预发布→生产" | 相同权重下 Ollama → llama.cpp → vLLM |

## 延伸阅读

- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morph — llama.cpp vs Ollama 2026](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai — Comprehensive LLM Inference Engine Comparison](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI — 10 Best vLLM Alternatives 2026](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [TGI maintenance announcement](https://github.com/huggingface/text-generation-inference) — 发布说明。
- [vLLM v0.15.1 release notes](https://github.com/vllm-project/vllm/releases)
