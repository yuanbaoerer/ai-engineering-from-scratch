# Flamingo 与 Gated Cross-Attention 在 Few-Shot VLM 中的应用

> DeepMind 的 Flamingo（2022）率先实现了两项突破。它证明了单个模型可以处理任意交错排列的图像、视频和文本序列。同时，它展示了 VLM 可以进行上下文学习——在 few-shot prompt 中给出三个示例（图像，标题）对后，模型无需任何梯度更新即可为新图像生成标题。其核心机制是 gated cross-attention 层，这些层被插入到冻结的 LLM 的现有层之间，并通过一个初始化为零的 tanh gate 来保证初始化时 LLM 的文本能力不受影响。本课将深入讲解 Flamingo 的 Perceiver resampler 和 gated cross-attention 架构——它是 Gemini 交错输入和 Idefics2 视觉 token 的先驱。

**类型：** 学习
**语言：** Python（标准库，gated cross-attention + Perceiver resampler 演示）
**前置知识：** Phase 12 · 03（BLIP-2 Q-Former）
**时间：** 约 120 分钟

## 学习目标

- 解释 gated cross-attention 如何通过 tanh(gate) = 0 在初始化时保留冻结 LLM 的文本能力。
- 详细讲解 Perceiver resampler：通过 cross-attention 将 N 个图像 patch 映射为 K 个固定的"latent" query。
- 描述 Flamingo 如何处理交错的图像-文本序列，并通过因果掩码（causal masking）来尊重图像的位置。
- 复现 few-shot 多模态 prompt 结构（3 个图像-标题示例，然后是一个查询图像）。

## 问题背景

BLIP-2 将 32 个视觉 token 输入到冻结 LLM 的输入层。对于每个 prompt 中只有一张图像的情况，这很有效。但如果你想输入*多张*与文本交错的图像，例如"这是图像 A，给它加标题；这是图像 B，给它加标题；现在是图像 C，给它加标题"呢？此时 LLM 的 self-attention 需要在同一个序列中同时处理图像 token 和文本 token，而哪些位置可以关注哪些图像的问题会变得复杂。

Flamingo 的解决方案是：完全不改变 LLM 的输入流。在现有的 LLM 块之间插入额外的 cross-attention 层。文本 token 仍然像往常一样流经 LLM 的 causal self-attention。每隔几个 LLM 块，文本 token 还会通过一个新增的 gated 层来 cross-attend 到图像特征。gate（初始化为零）意味着在训练开始时，这些新层是无操作的——模型表现得完全像预训练的 LLM。随着训练进行，gate 逐渐打开，视觉信息开始流入。

Flamingo 解决的第二个问题是：如何处理每个 prompt 中数量可变的图像（0 张、1 张或多张）？答案是 Perceiver resampler——一个小型的 cross-attention 模块，它接收任意数量的 patch，并输出固定数量的视觉 latent token。无论 prompt 中有多少张图像，LLM 的 cross-attention 层看到的都是相同的形状。

## 核心概念

### 冻结的 LLM

Flamingo 从一个冻结的 Chinchilla 70B LLM 开始。全部 700 亿参数保持不变。现有的文本 self-attention 和 FFN 正常运作。

### Perceiver resampler

对于 prompt 中的每张图像，ViT 会生成 N 个 patch token。Perceiver resampler 有 K 个固定的可学习 latent（Flamingo 使用 K=64）。每个 resampler 块包含两个子步骤：

1. Cross-attention：K 个 latent 对 N 个 patch token 进行关注（Q 来自 latent，K/V 来自 patch）。
2. Latent 内部的 self-attention + FFN。

经过 6 个 resampler 块后，无论 ViT 产生了多少个 patch，输出都是维度为 1024 的 K=64 个视觉 token。一张 224x224 的图像（196 个 patch）和一张 480x480 的图像（900 个 patch），最终都输出 64 个 resampler token。

对于视频，resampler 会按时间维度应用：每帧的 patch 产生 64 个 latent，时间位置编码让模型能够区分 t=0 和 t=N。整个视频变成 T * 64 个视觉 token。

### Gated cross-attention

在冻结 LLM 的每 M 层之间（Flamingo 使用 M=4），插入一个新的 gated cross-attention 块：

```
x_after_llm_block = llm_block(x_before)
cross = cross_attn(x_after, resampler_output)
gated = tanh(alpha) * cross + x_after
x_before_next_block = gated
```

- `alpha` 是一个初始化为零的可学习标量。
- `tanh(0) = 0`，因此在初始化时 gated 分支的贡献为零。
- 随着 `alpha` 偏离零，cross-attention 的贡献平滑增长。
- 残差连接意味着即使 gate 完全打开，也不会覆盖 LLM 的文本表示；它只是在文本表示之上添加视觉信息。

这是 Flamingo 中最重要的设计决策：视觉条件是加性的、带 gate 的，并且在初始化时为零。在 step 0 时，Flamingo 在纯文本输入上就是一个完美的 Chinchilla 70B。

### 交错输入的 masked cross-attention

在一个 prompt 如"<image A> caption A <image B> caption B <image C> ?"中，每个文本 token 应该只能看到序列中出现在它之前的图像。Cross-attention mask 强制规定：位置 `t` 的文本 token 只能关注图像索引 `i < i_t` 的图像 resampler token，其中 `i_t` 是位置 `t` 之前最近的一张图像。"只看最近的前一张图像"或"看所有前面的图像"都是可行的选择；Flamingo 选择了前者。

### 上下文 few-shot 学习

Flamingo 的 prompt 看起来像这样：

```
<image1> A photo of a cat. <image2> A photo of a dog. <image3> A photo of a
```

模型看到补全模式后输出"bird"（或 image3 所展示的内容）。无需梯度更新。冻结 LLM 的上下文学习能力通过 gated cross-attention 得以延续——这是论文的核心亮点，也是其重要意义所在。

### 训练数据

Flamingo 在三个数据集上训练：

1. MultiModal MassiveWeb（M3W）：4300 万个带有交错图像和文本的网页，按阅读顺序重建。
2. Image-Text Pairs（ALIGN + LTIP）：44 亿对。
3. Video-Text Pairs（VTP）：2700 万个短视频片段。

OBELICS（2023）是交错网页语料库的开源复现版本，Idefics、Idefics2 以及大多数开源"Flamingo-like"模型都在其上训练。

### OpenFlamingo 与 Otter

OpenFlamingo（2023）是开源复现版本。架构完全相同（Perceiver resampler + 在冻结的 LLaMA 或 MPT 上的 gated cross-attention）。提供 3B、4B、9B 的 checkpoint。由于基础 LLM 更小且数据更少，质量略逊于 Flamingo。

Otter（2023）在 OpenFlamingo 基础上增加了在 MIMIC-IT（一个多模态指令数据集）上的指令微调，证明了 gated cross-attention 也适用于指令遵循任务。

### 后继者

- Idefics / Idefics2 / Idefics3：Hugging Face 的 gated cross-attention 系列， progressively 简化（Idefics2 放弃了 resampler，转而使用直接 patch token 加 adaptive pooling）。
- Flamingo-to-Chameleon 过渡：到 2024 年，许多团队转向了 early-fusion（Lesson 12.11）；Flamingo 风格的 gated cross-attention 在需要冻结 backbone 的生产环境中仍有应用。
- Gemini 的交错输入：概念上继承了 Flamingo 的交错格式灵活性，尽管具体机制是专有的。

### 与 BLIP-2 的对比

| | BLIP-2 | Flamingo |
|---|---|---|
| 视觉桥接 | Q-Former，仅在输入层一次 | 在每 M 层进行 gated cross-attention |
| 视觉 token | 每张图像 32 个 | 每个 cross-attn 层每张图像 64 个 |
| 冻结 LLM | 是 | 是 |
| Few-shot 上下文学习 | 较弱 | 强——论文的核心亮点 |
| 交错输入 | 无原生支持 | 支持，是设计目标 |
| 训练数据 | 1.3 亿对 | 13 亿对 + 4300 万交错页面 |
| 参数量 | 1.88 亿训练参数 | 约 100 亿训练参数（cross-attn 层） |
| 计算资源 | 8 张 A100，数天 | 数千个 TPUv4，数周 |

预算有限时做单图 VQA 选 BLIP-2。做交错输入、few-shot 或多图推理选 Flamingo/Idefics2。

## 使用它

`code/main.py` 演示了以下内容：

1. 一个 Perceiver resampler，处理 36 个 fake patch token，使用 8 个可学习 latent（纯 Python cross-attention）。
2. 一个 gated cross-attention 步骤：`alpha = 0` 时输出等于输入（LLM 不变），然后 `alpha = 2.0` 时视觉贡献被混合进来。
3. 一个交错 mask 构建器，为"(image 1) (text 1) (image 2) (text 2)"序列生成二维 attention mask。

## 交付它

本课产出 `outputs/skill-gated-bridge-diagnostic.md`。给定一个开放 VLM 的配置（resampler 有/无、cross-attn 频率、gate 方案），它识别 Flamingo 谱系的元素并解释冻结策略。对于调试为什么微调后文本性能下降很有用（答案：gate 开得太快太宽）。

## 练习题

1. 计算 Flamingo-9B 的视觉参数量：9B LLM + 14 亿 gated cross-attention 层 + 6400 万 resampler。训练参数占总参数的比例是多少？

2. 在 PyTorch 中实现 gated 残差 `y = tanh(alpha) * cross + x`。通过实验证明 `alpha=0` 时，在初始化时 `y==x` 精确成立。

3. 阅读 OpenFlamingo 第 3.2 节（arXiv:2308.01390），了解他们如何处理 batch 中每个 prompt 图像数量不同的情况。描述 padding 策略。

4. 为什么 Flamingo 的 cross-attention mask 让文本 token 只能关注*最近的*前一张图像，而不是所有前面的图像？阅读 Flamingo 论文第 2.4 节并解释权衡。

5. 上下文 few-shot：为一个新的 Flamingo 变体构造一个包含 4 个"图像 → 主要物体颜色"示例的 prompt。描述当示例数量从 0 变化到 8 时，预期准确率的变化模式。

## 关键术语

| 术语 | 通常的说法 | 实际含义 |
|------|------------|----------|
| Perceiver resampler | "Fixed-latent cross-attention" | 从可变数量的输入 patch 产生 K 个固定 token 的模块 |
| Gated cross-attention | "Tanh-gated bridge" | 残差层 `y = tanh(alpha)*cross + x`，可学习 alpha，初始化为 0 |
| Interleaved input | "Mixed sequence" | 图像和文本按阅读顺序自由混合的 prompt 格式 |
| Frozen LLM | "No LLM gradients" | 文本 LLM 的权重不更新；只有 resampler + cross-attn 层参与训练 |
| Few-shot | "In-context examples" | 在 prompt 中给出少量（图像，答案）对；模型无需微调即可泛化 |
| OBELICS | "Interleaved web corpus" | 包含 1.41 亿个按阅读顺序排列图像和文本的网页的开源数据集 |
| Chinchilla | "70B frozen base" | Flamingo 的冻结文本 LLM，来自 DeepMind 的 Chinchilla 论文 |
| Gate schedule | "How alpha moves" | Cross-attention gate 在训练过程中打开的速度 |
| Cross-attn frequency | "Every M layers" | Gated cross-attention 块的插入频率；Flamingo 使用 M=4 |
| OpenFlamingo | "Open reproduction" | MosaicML/LAION 的开源 checkpoint，3-9B；架构与 Flamingo 相同 |

## 延伸阅读

- [Alayrac et al. — Flamingo (arXiv:2204.14198)](https://arxiv.org/abs/2204.14198) — 原始论文。
- [Awadalla et al. — OpenFlamingo (arXiv:2308.01390)](https://arxiv.org/abs/2308.01390) — 开源复现。
- [Laurençon et al. — OBELICS (arXiv:2306.16527)](https://arxiv.org/abs/2306.16527) — 交错网页语料库。
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — 通用 Perceiver 架构。
- [Li et al. — Otter (arXiv:2305.03726)](https://arxiv.org/abs/2305.03726) — 经过指令微调的 Flamingo 后继者。
- [Laurençon et al. — Idefics2 (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246) — Flamingo 方法的现代简化版本。
