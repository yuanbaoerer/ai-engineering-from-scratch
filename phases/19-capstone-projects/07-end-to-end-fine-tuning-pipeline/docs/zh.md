# 综合项目 07 — 端到端微调流水线（数据 → SFT → DPO → 服务）

> 一个 8B 参数模型，在你自己的数据上训练，使用你自己的偏好进行 DPO 对齐，量化后进行推测解码，并以可衡量的 $/1M tokens 进行服务。2026 年的开源技术栈是 Axolotl v0.8、TRL 0.15、Unsloth 用于迭代、GPTQ/AWQ/GGUF 用于量化、vLLM 0.7 配合 EAGLE-3 用于服务。本综合项目的目标是可复现地运行整个流水线 — 输入 YAML，输出服务端点 — 并按照 2026 模型开放框架发布模型卡片。

**类型：** 综合项目
**语言：** Python（流水线）、YAML（配置）、Bash（脚本）
**前置课程：** 阶段 2（机器学习）、阶段 3（深度学习）、阶段 7（Transformer）、阶段 10（从零构建 LLM）、阶段 11（LLM 工程）、阶段 17（基础设施）、阶段 18（安全）
**涉及阶段：** P2 · P3 · P7 · P10 · P11 · P17 · P18
**时间：** 35 小时

## 问题

2026 年每个严肃的 AI 团队都会维护一条微调流水线。不是因为他们要发布前沿基座模型，而是因为下游适配 — 领域 SFT、基于标注偏好的 DPO、用于推测解码的蒸馏草稿模型、配合 EAGLE-3 的服务化 — 才是可衡量收益所在。Axolotl v0.8 处理多 GPU SFT 配置。TRL 0.15 处理 DPO 和 GRPO。Unsloth 实现快速单 GPU 迭代。vLLM 0.7 配合 EAGLE-3 可将解码吞吐量提升 2-3 倍且不损失质量。工具已经就绪；关键在于 YAML 配置、数据卫生和评估规范。

你将对一个 8B 基座模型（Llama 3.3、Qwen3 或 Gemma 3）进行 SFT 然后 DPO 训练（使用特定任务数据），量化后进行服务，并使用 lm-evaluation-harness、RewardBench-2、MT-Bench-v2 和 MMLU-Pro 对比衡量增益。你将按照 2026 模型开放框架生成模型卡片。重点是可复现性 — 一条命令即可端到端重跑整个流水线。

## 概念

流水线包含五个阶段。**数据**：去重（MinHash / Datatrove）、质量过滤（Nemotron-CC 风格分类器）、PII 清洗、数据划分卫生检查（防止公共基准污染）。**SFT**：Axolotl YAML 配置、8xH100 上 ZeRO-3、余弦调度、序列打包、2-3 个 epoch。**DPO 或 GRPO**：TRL 配置、1 个 epoch、偏好对可以是人工标注或模型判定、beta 调优。**量化**：GPTQ + AWQ + GGUF 以提供部署灵活性。**服务**：vLLM 0.7 配合 EAGLE-3 推测头（或 SGLang 配合 SpecForge）、K8s 部署、基于队列等待时间的 HPA。

消融实验是交付物：SFT-only 对比 SFT+DPO 对比 SFT+GRPO 在三个特定任务基准上的表现。服务指标：batch 1 / 8 / 32 下的 tokens/s、EAGLE-3 接受率、$/1M tokens。安全评估：Llama Guard 4 通过率。模型卡片：偏差评估、可复现性种子、数据许可证。

## 架构

```
raw data (HF datasets + internal)
    |
    v
Datatrove dedup + Nemotron-CC quality filter + PII scrub
    |
    v
split hygiene (MMLU-Pro contamination check)
    |
    v
Axolotl SFT config (YAML)  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO config       ---> 4xH100, 1 epoch
    |
    v
GPTQ + AWQ + GGUF quantize
    |
    v
vLLM 0.7 + EAGLE-3 speculative decoding
    |
    v
K8s deployment, HPA on queue-wait
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
model card (2026 MOF) + safety eval (Llama Guard 4)
```

## 技术栈

- 数据：Datatrove 用于去重，Nemotron-CC 分类器用于质量过滤，Presidio 用于 PII 清洗
- 基座模型：Llama 3.3 8B、Qwen3 14B 或 Gemma 3 12B
- SFT：Axolotl v0.8 配合 ZeRO-3、Flash Attention 3、序列打包
- 偏好调优：TRL 0.15 用于 DPO 或 GRPO；Unsloth 用于单 GPU 迭代
- 量化：GPTQ（Marlin）、AWQ、GGUF（通过 llama.cpp）
- 服务：vLLM 0.7 配合 EAGLE-3 推测解码（或 SGLang 0.4 + SpecForge）
- 评估：lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro
- 安全评估：Llama Guard 4、ShieldGemma-2
- 基础设施：Kubernetes + NVIDIA device plugin、基于队列等待时间的 HPA
- 可观测性：W&B 用于训练，Langfuse 用于推理

## 构建步骤

1. **数据流水线。** 对原始语料运行 Datatrove 去重。应用 Nemotron-CC 风格的质量分类器。使用 Presidio 清洗 PII。使用显式种子写入训练/验证数据划分。

2. **污染检查。** 对每个验证划分，使用 MinHash 与 MMLU-Pro、MT-Bench-v2、RewardBench-2 测试集进行对比计算。拒绝任何重叠。

3. **Axolotl SFT。** 使用 ZeRO-3、FA3、序列打包的 YAML 配置。在 8xH100 上训练 2-3 个 epoch。记录到 W&B。

4. **TRL DPO / GRPO。** 拿 SFT 检查点，在偏好对上运行一个 epoch 的 DPO（或使用可验证奖励的 GRPO 处理数学/代码任务）。扫描 beta 值。

5. **量化。** 生成三种量化版本：GPTQ-INT4-Marlin、AWQ-INT4、GGUF-Q4_K_M（用于 llama.cpp）。记录模型大小和名义吞吐量。

6. **使用推测解码进行服务。** vLLM 0.7 配置 EAGLE-3 草稿头（通过 Red Hat Speculators 训练）。测量 batch 1 / 8 / 32 下的接受率和尾延迟。对比 Anthropic / OpenAI 在相同评估上的 $/1M tokens。

7. **评估矩阵。** 在基座、SFT-only、SFT+DPO、SFT+GRPO 上运行 lm-eval-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro。生成对比表格。

8. **安全评估。** Llama Guard 4 在开发集上的通过率。ShieldGemma-2 输出过滤器。

9. **模型卡片。** MOF 2026 模板：数据、训练、评估、安全、许可证、可复现性章节（包含 YAML 配置和 commit SHA）。

## 使用方式

```
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k deduped, 12k filtered, 280k accepted (seed=7)
[SFT]     3 epochs, 8xH100, 6h12m, val loss 1.42 -> 1.03
[DPO]     1 epoch, beta=0.08, 4xH100, 1h40m
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 acceptance 0.74, p99 126ms @ bs=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    model-card.md generated under 2026 MOF
```

## 交付说明

`outputs/skill-finetuning-pipeline.md` 描述了交付物。一条命令即可运行从数据到 SFT 到 DPO 到量化到服务到评估的完整流程，并输出模型卡片和服务端点。

| 权重 | 标准 | 衡量方式 |
|:--:|---|---|
| 25 | 与基座模型的评估增益 | 在目标任务（MMLU-Pro、MT-Bench-v2、特定任务）上的衡量增益 |
| 20 | 流水线可复现性 | 一条命令使用相同种子端到端重跑 |
| 20 | 数据卫生 | 去重率、PII 清洗覆盖率、污染检查通过 |
| 20 | 服务效率 | bs=1/8/32 下的 tokens/s、EAGLE-3 接受率、$/1M tokens |
| 15 | 模型卡片 + 安全评估 | 2026 MOF 完整性 + Llama Guard 4 通过率 |
| **100** | | |

## 练习

1. 在同一特定任务基准上运行 SFT-only 对比 SFT+DPO 对比 SFT+GRPO。报告哪种偏好方法胜出以及优势幅度。

2. 将 Llama 3.3 8B 替换为 Qwen3 14B。在匹配质量下衡量 $/1M tokens。

3. 测量 EAGLE-3 在领域数据与通用 ShareGPT 上的接受率。报告差异及其对延迟预算的影响。

4. 注入 1% 的污染（将 MMLU-Pro 答案泄露到训练数据中）并重跑评估。观察 MMLU-Pro 准确率不现实地飙升。构建一个能捕获此类问题的污染检查 CI 门禁。

5. 添加 LoRA SFT 作为全量微调的替代方案。在内存降低 10 倍的情况下衡量质量差距。

## 关键术语

| 术语 | 人们通常的说法 | 实际含义 |
|------|---------------|----------|
| Axolotl | "SFT 训练器" | 基于 YAML 配置的统一训练器，支持 SFT、DPO 和蒸馏 |
| TRL | "偏好调优器" | Hugging Face 库，用于在 LLM 上进行 DPO、GRPO、PPO |
| GRPO | "群组相对策略优化" | DeepSeek R1 的强化学习方案，使用可验证奖励 |
| EAGLE-3 | "推测解码草稿" | 提前预测 N 个 token 的草稿头；vLLM 使用目标模型进行验证 |
| MOF | "模型开放框架" | 2026 年标准，根据数据、代码、许可证对模型发布进行评级 |
| 污染检查 | "数据划分卫生" | 基于 MinHash 的方法，检测测试集泄露到训练集中 |
| 接受率 | "EAGLE / MTP 指标" | 目标模型接受的草稿 token 的比例 |

## 延伸阅读

- [Axolotl 文档](https://axolotl-ai-cloud.github.io/axolotl/) — 参考 SFT / DPO 训练器
- [TRL 文档](https://huggingface.co/docs/trl) — DPO 和 GRPO 参考实现
- [Unsloth](https://github.com/unslothai/unsloth) — 单 GPU 迭代参考
- [DeepSeek R1 论文 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — GRPO 方法论
- [vLLM + EAGLE-3 文档](https://docs.vllm.ai) — 参考服务栈
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 替代推测解码训练器
- [模型开放框架 2026](https://isocpp.org/) — 开放发布评级标准
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — 规范评估运行器
