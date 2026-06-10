# Janus-Pro：解耦编码器用于统一多模态模型

> 统一多模态模型存在一种不可避免的内在张力。理解任务需要语义特征——SigLIP 或 DINOv2 输出的向量富含概念级信息。生成任务需要重建友好的编码——VQ token 能够组合回清晰的像素。这两个目标在单一编码器中无法兼容。Janus（DeepSeek，2024 年 10 月）和 Janus-Pro（DeepSeek，2025 年 1 月）提出的解决方案是：停止试图兼顾，将两个编码器解耦。在不同任务间共享 transformer 主干，但将理解任务路由至 SigLIP，将生成任务路由至 VQ tokenizer。在 7B 参数规模下，Janus-Pro 在 GenEval 上超越 DALL-E 3，同时在 MMMU 上与 LLaVA 持平。本课将分析为何两个编码器能奏效，而单一编码器不行。

**类型：** 构建
**语言：** Python（标准库，双编码器路由 + 共享主干信号）
**前置知识：** Phase 12 · 13（Transfusion），Phase 12 · 14（Show-o）
**时间：** 约 120 分钟

## 学习目标

- 解释为何单一共享编码器会在理解质量或生成质量上做出妥协。
- 描述 Janus-Pro 的路由机制：输入侧使用 SigLIP 特征进行理解，输入和输出均使用 VQ token 进行生成。
- 追溯数据混合扩展策略，分析 Janus-Pro 在 Janus 基础上成功的原因。
- 比较解耦架构（Janus-Pro）、耦合连续架构（Transfusion）和耦合离散架构（Show-o）。

## 问题所在

统一模型在理解和生成任务间共享 transformer 主干。此前的尝试（Chameleon、Show-o、Transfusion）都使用单一视觉 tokenizer 同时服务于两个方向。这个 tokenizer 本身就是一种妥协：

- 针对重建优化（生成）：VQ-VAE 能捕捉细粒度像素细节，但生成的 token 语义连贯性较弱。
- 针对语义优化（理解）：SigLIP 嵌入将"猫"的图像聚集在"猫"的 token 附近，但不支持良好的重建。

Show-o 和 Transfusion 为此付出了明显的质量代价——某一方向上的表现受损。Janus-Pro 提出疑问：当两个任务的需求不同时，为何非要强制使用一个 tokenizer？

## 核心概念

### 解耦视觉编码

Janus-Pro 的架构将两个编码器分离：

- **理解路径。** 输入图像 → SigLIP-SO400m → 2 层 MLP → transformer 主干。
- **生成路径。** 输入图像（若基于现有图像进行条件生成）→ VQ tokenizer → token ID → transformer 主干。
- **输出生成。** Transformer 预测的图像 token → VQ 解码器 → 像素。

Transformer 主干是共享的。主干上游和下游的所有组件都是任务专属的。

输入通过 prompt 格式进行区分：`<understand>` 标签路由至 SigLIP；`<generate>` 路由至 VQ。或者路由根据任务隐式确定。

### 为何有效

理解损失使用 SigLIP 特征，这些特征经过 CLIP 风格预训练的调整，适合语义相似性任务。由于输入特征更适合该任务，模型的感知基准测试表现优于 Show-o / Transfusion。

生成损失使用 VQ token，这些 token 经过 tokenizer 的调整，适合重建任务。由于 VQ 编码能干净地组合回像素，图像质量优于 Show-o。

共享的 transformer 主干看到两种输入分布（SigLIP 和 VQ），并学会同时处理两者。其核心主张是：只要有足够的数据和足够的参数，主干就能吸收这种切换能力。

### 数据扩展——Janus 与 Janus-Pro 的对比

Janus（原始版本，arXiv 2410.13848）提出了解耦思想，但规模较小（1.3B 参数，数据有限）。Janus-Pro（arXiv 2501.17811）进行了扩展：

- 7B 参数（对比 1.3B）。
- 第一阶段（对齐）使用 9000 万图像-文本对，高于之前的 7200 万。
- 第二阶段（统一训练）使用 7200 万对，高于之前的 2600 万。
- 第三阶段新增 20 万图像生成指令样本。

结果是：Janus-Pro-7B 在 MMMU 上与 LLaVA 持平（60.3 对比约 58），在 GenEval 上超越 DALL-E 3（0.80 对比 0.67）。一个开源模型，在统一光谱的两端都具有竞争力。

### JanusFlow——rectified flow 变体

JanusFlow（arXiv 2411.07975）将 VQ 生成路径替换为 rectified-flow 生成路径（连续）。拆分变为 SigLIP 负责理解 + rectified flow 负责生成。质量上限进一步提升。架构仍保持解耦编码器-共享主干的设计。

### 共享主干的职责

Transformer 主干处理统一的序列，但面对两种输入分布。它的职责是：

- **理解任务：** 消费 SigLIP 特征 + 文本 token → 自回归地输出文本。
- **生成任务：** 消费文本 token +（可选的图像 VQ token）→ 自回归地输出图像 VQ token。

主干在每个块中没有模态专属的权重。它就是你在 Qwen 或 Llama 内部预期看到的那种文本风格 transformer，加上两个输入适配器。

有趣的是，这意味着 Janus-Pro 的主干可以从预训练 LLM 初始化。Janus-Pro 确实从 DeepSeek-MoE-7B 初始化。这一选择很重要：LLM 贡献了推理能力，而纯粹从零开始训练的统一模型难以达到这种水平。

### 与 InternVL-U 的对比

InternVL-U（第 12.10 课）是 2026 年的后续工作。它结合了：

- 原生多模态预训练（InternVL3 主干）。
- 解耦编码器路由（SigLIP 输入，VQ + 扩散头输出）。
- 统一的理解 + 生成 + 编辑能力。

InternVL-U 将 Janus-Pro 的架构选择纳入一个更大的框架中。解耦编码器的理念现在已成为大规模统一模型的默认方案。

### 局限性

解耦编码器增加了架构复杂性。需要训练两个 tokenizer，维护两条输入路径，应对两套故障模式。对于不需要生成能力的产品，Janus-Pro 是过度设计——选择 LLaVA 家族的理解模型即可。

对于不需要理解能力的产品，Janus-Pro 是大材小用——选择 Stable Diffusion 3 / Flux 模型即可。

对于两者都需要的产品，Janus-Pro 现在是参考性的开源架构。

## 动手实践

`code/main.py` 模拟了 Janus-Pro 的路由：

- 两个模拟编码器：类 SigLIP（生成 256 维语义向量）和类 VQ（生成整数编码）。
- 一个基于任务标签选择编码器的 prompt 路由器。
- 一个共享主干（占位符），处理 token 序列，无论由哪个编码器生成。
- 从第一阶段（对齐）到第三阶段（指令微调）的加权采样调度切换。

打印 3 个示例的路由路径：图像问答、文生图、图像编辑。

## 交付成果

本课产出 `outputs/skill-decoupled-encoder-picker.md`。针对一个希望在接近前沿质量水平上同时实现统一生成和理解的产品，从 Janus-Pro、JanusFlow 或 InternVL-U 中进行选择，并给出具体的数据规模建议。

## 练习题

1. Janus-Pro-7B 在 GenEval 上超越 DALL-E 3。解释为何一个 7B 开源模型能在生成任务上匹敌前沿闭源模型，但在理解任务上却做不到。

2. 实现一个路由器函数：给定 prompt 文本，分类为 `understand` 或 `generate`。你如何处理模糊 prompt，例如"描述然后画出来"？

3. JanusFlow 将 VQ 路径替换为 rectified flow。Transformer 主干现在输出什么，损失函数有何变化？

4. 为 Janus-Pro 架构提议第四项任务，可通过增加一个解耦编码器来处理。例如：图像分割（DINO 风格）、深度估计（MiDaS 风格）。

5. 阅读 Janus-Pro 论文第 4.2 节关于数据扩展的内容。哪个数据阶段对 T2I 质量提升（相对于 Janus）贡献最大？

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|----------|----------|
| 解耦编码 | "两个视觉编码器" | 每个方向使用独立的 tokenizer 或编码器：理解用语义型，生成用重建型 |
| 共享主干 | "一个 transformer" | 单一 transformer 处理任一编码器的输出；没有模态专属权重 |
| SigLIP 用于理解 | "语义特征" | CLIP 家族视觉塔提供丰富的概念特征，但重建能力较差 |
| VQ 用于生成 | "重建编码" | 向量量化 token，能干净地解码回像素 |
| JanusFlow | "Rectified-flow 变体" | Janus-Pro 使用连续的 flow-matching 生成头替代 VQ |
| 路由标签 | "任务标签" | Prompt 标记（`<understand>` / `<generate>`），用于选择输入编码器 |

## 延伸阅读

- [Wu et al. — Janus (arXiv:2410.13848)](https://arxiv.org/abs/2410.13848)
- [Chen et al. — Janus-Pro (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)
- [Ma et al. — JanusFlow (arXiv:2411.07975)](https://arxiv.org/abs/2411.07975)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Dong et al. — DreamLLM (arXiv:2309.11499)](https://arxiv.org/abs/2309.11499)
