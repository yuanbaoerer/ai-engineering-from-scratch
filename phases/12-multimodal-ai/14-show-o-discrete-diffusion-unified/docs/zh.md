# Show-o 与 Discrete-Diffusion 统一模型

> Transfusion 混合了连续和离散表示。Show-o（Xie 等人，2024 年 8 月）则走了另一条路：文本 token 使用因果 next-token prediction，图像 token 使用受 MaskGIT 启发的 masked discrete diffusion。两者共处于一个 transformer 中，并采用 hybrid attention mask。其结果是在单一骨干网络、每种模态一个 tokenizer、单一损失函数（将 next-token 扩展为 masked prediction）上统一了 VQA、text-to-image、inpainting 和混合模态生成。本课将深入讲解 Show-o 的设计——为什么 masked discrete diffusion 是一种并行的、少步数的图像生成器——并与 Transfusion 和 Emu3 进行对比。

**类型：** 学习
**语言：** Python（标准库，masked-discrete-diffusion sampler）
**前置知识：** Phase 12 · 13（Transfusion）
**时间：** ~120 分钟

## 学习目标

- 解释 masked discrete diffusion：按照 schedule 均匀 mask token，然后让 transformer 恢复它们。
- 比较并行图像解码（Show-o、MaskGIT）与自回归图像解码（Chameleon、Emu3）在速度和质量上的差异。
- 说出 Show-o 在一个 checkpoint 中处理的三个任务：T2I、VQA、image inpainting。
- 选择一种 masking schedule（cosine、linear、truncated），并分析其对采样质量的影响。

## 问题背景

Transfusion 的双损失训练虽然有效，但动态较复杂——连续扩散损失与离散 NTP 损失的数值尺度不同。平衡损失权重需要进行超参数搜索。该架构有效但复杂。

Show-o 的解决方案：保持两种模态均为离散表示（如 Chameleon），但通过 masked discrete diffusion 并行生成图像，而非顺序生成。训练目标变为单一的 masked-token-prediction，它自然地泛化了 next-token-prediction。

## 核心概念

### Masked discrete diffusion（MaskGIT）

Chang 等人（2022）提出的原始 MaskGIT 技巧非常优雅。从完全 mask 的图像开始（每个 token 都是特殊的 `<MASK>` id）。在每一步中，并行预测所有被 mask 的 token，然后保留置信度最高的 top-K 个预测，将其余的重新 mask。经过约 8-16 次迭代后，所有 token 都被填充完毕。每一步解除 mask 的 token 数量 schedule 需要调优——cosine schedule 效果较好。

训练很简单：从 [0, 1] 中均匀采样一个 masking ratio，将其应用于图像的 VQ token，训练 transformer 恢复被 mask 的 token。这正是 BERT 对文本所做的，只是扩展到了图像生成。

### Show-o：一个 transformer，hybrid mask

Show-o 将 MaskGIT 置于因果语言模型 transformer 中。Attention mask 的规则如下：

- 文本 token：因果的（标准 LLM）。
- 图像 token：在图像块内完全双向（因此被 mask 的 token 在预测时可以看到其他所有图像 token）。
- 文本到图像：文本关注之前的图像，图像关注之前的文本。

训练在以下三种样本之间交替进行：
1. 文本序列上的标准 NTP。
2. T2I 样本：文本 → 图像，图像 token 被 mask，使用 masked-token-prediction loss。
3. VQA 样本：图像 → 文本，文本 token 被 mask（实际上就是 NTP）。

统一的损失是 `<MASK>` token 上的交叉熵，它同时覆盖了文本 NTP（只有最后一个 token 被"mask"）和图像 masked-diffusion（随机子集被 mask）。

### 并行采样

Show-o 生成图像仅需约 16 步，而非约 1000 步（每个 token 自回归）或约 20 步（diffusion）。每一步并行预测所有被 mask 的 token；提交 top-K 个高置信度的；重复。

对比：
- Chameleon / Emu3（token 自回归）：N_tokens 次前向传播，通常每张图像 1024-4096 次。
- Transfusion（连续 diffusion）：约 20 步，每步一次完整 transformer 前向传播。
- Show-o（masked discrete diffusion）：约 16 步，每步一次完整 transformer 前向传播。

在相似规模的模型上，Show-o 比 Chameleon 更快，与 Transfusion 的步数大致相当，但每步成本更低（离散 vocab logits 对比连续 MSE loss）。

### 一个 checkpoint 支持多种任务

Show-o 在推理时支持四种任务，通过 prompt 格式选择：

- 文本生成：标准自回归文本输出。
- VQA：图像输入，文本输出。
- T2I：文本输入，通过 masked discrete diffusion 输出图像。
- Inpainting：图像中部分 token 被 mask，进行填充。

Inpainting 能力来自 masked-prediction 训练，无需额外代价。将 VQ-token 网格的某个区域 mask，输入其余部分加文本 prompt，预测被 mask 的 token。

### Masking schedule

每一步解除 mask 的 token 数量 schedule 会影响质量。Show-o 推荐 cosine：

```
mask_ratio(t) = cos(pi * t / (2 * T))   # t = 0..T
```

在第 0 步，所有 token 被 mask（ratio 1.0）。在第 T 步，没有 token 被 mask。Cosine 将质量集中在中间范围的 ratio 上，这些位置的预测信息量最大。Linear schedule 也能工作，但更快进入平台期。

### Show-o2

Show-o2（2025 年后续工作，arXiv 2506.15564）对 Show-o 进行了扩展：更大的 LLM 基础模型、更好的 tokenizer、改进的 mask schedule。架构模式相同。

### Show-o 的定位

在 2026 年的分类体系中：

- 离散 token + NTP：Chameleon、Emu3。简单但推理慢。
- 离散 token + masked diffusion：Show-o、MaskGIT、LlamaGen、Muse。并行采样，但仍受 tokenizer 损失影响。
- 连续 + diffusion：Transfusion、MMDiT、DiT。质量最高，训练更复杂。
- 连续 + flow matching in a VLM：JanusFlow、InternVL-U。最新方向。

按任务选择：当你需要一个开放权重模型，同时支持 T2I + inpainting + VQA 且速度合理时，选择 Show-o；当质量至上且能接受双损失工程的复杂性时，选择 Transfusion。

## 动手实践

`code/main.py` 模拟了 Show-o 的采样过程：

- 一个包含 16 个 VQ token 的 toy grid。
- 一个 mock "transformer"，根据 prompt 和当前已 unmask 的 token 预测 logits。
- 使用 cosine schedule 进行 8 步并行 masked sampling。
- 打印中间状态（mask 模式的演变）和最终 token。

运行它，观察 mask 如何一步步溶解。

## 项目产出

本课的产出是 `outputs/skill-unified-gen-model-picker.md`。给定一个需要同时支持理解（VQA、captioning）和生成（T2I、inpainting）且受限于开放权重的产品，在 Show-o 家族、Transfusion/MMDiT 家族和 Emu3 / Chameleon 家族之间做出选择，并给出具体的权衡分析。

## 练习题

1. Masked discrete diffusion 约需 16 步采样。为什么不能 1 步？如果在第 0 步就全部 unmask，会发生什么？

2. Inpainting 是 masked diffusion 的免费附赠能力。请提出一个真实或假设的产品用例，说明 Show-o 的 inpainting 优于专用模型。

3. Cosine schedule 与 linear schedule：对 T=8，追踪每步 unmask 的 token 数量。哪种更均衡？

4. 一张 512x512 的 Show-o 图像对应 1024 个 token。在 vocab K=16384 时，模型输出 1024 * log2(16384) = 14,336 bits（约 1.75 KiB）的数据。Stable Diffusion 输出 512*512*24 bits = 6,291,456 bits（约 768 KiB）的原始像素。压缩比是多少，它换来了什么质量？

5. 阅读 LlamaGen（arXiv:2406.06525）。LlamaGen 的 class-conditional 自回归图像模型与 Show-o 的 masked 方法有何不同？

## 关键术语

| 术语 | 通常的说法 | 实际含义 |
|------|-----------|---------|
| Masked discrete diffusion | "MaskGIT-style" | 训练预测被 mask 的 token；推理时迭代地 unmask 最自信的预测 |
| Cosine schedule | "Unmask schedule" | 推理过程中 mask ratio 的衰减；将置信度增长集中在中间范围 |
| Parallel decoding | "All tokens at once" | 每一步在一个前向传播中预测所有被 mask 的 token 的完整序列，然后提交 top-K |
| Hybrid attention | "Causal + bidirectional" | 对文本 token 为因果 mask，在图像块内为双向 mask |
| Inpainting | "Fill-in generation" | 以部分 token 被 mask 的图像为条件，预测缺失的 token；来自训练目标的免费能力 |
| Commitment rate | "Top-K per step" | 每次迭代被宣布"完成"的 token 数量；控制推理与质量的权衡 |

## 延伸阅读

- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
- [Show-o2 (arXiv:2506.15564)](https://arxiv.org/abs/2506.15564)
- [Chang et al. — MaskGIT (arXiv:2202.04200)](https://arxiv.org/abs/2202.04200)
- [Sun et al. — LlamaGen (arXiv:2406.06525)](https://arxiv.org/abs/2406.06525)
- [Chang et al. — Muse (arXiv:2301.00704)](https://arxiv.org/abs/2301.00704)
