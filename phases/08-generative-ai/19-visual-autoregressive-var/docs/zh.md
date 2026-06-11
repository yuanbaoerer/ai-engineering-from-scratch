# 视觉自回归建模（VAR）：下一尺度预测

> 扩散模型在时间上迭代采样（去噪步骤）。VAR 在尺度上迭代采样——它先预测一个 1x1 token，然后预测 2x2、4x4，直到最终分辨率，每个尺度都以之前尺度为条件。2024 年的论文表明，VAR 在图像生成上符合 GPT 风格的 scaling laws，并在相同计算预算下击败 DiT。本课构建其核心机制。

**类型：** 构建
**语言：** Python（使用 PyTorch）
**先修要求：** 第 7 阶段第 03 课（多头注意力），第 8 阶段第 06 课（DDPM）
**时间：** 约 90 分钟

## 问题

自回归生成之所以主导语言建模，是因为它能可预测地扩展：更多计算、更多参数、更低困惑度、更好输出。在 2024 年之前，图像生成主要有两类 AR 尝试：PixelRNN/PixelCNN（逐像素）和 DALL-E 1 / Parti / MuseGAN（在 VQ-VAE code 上逐 token）。

两者都受困于生成顺序问题。像素和 token 排列在 2D 网格中，但 AR 模型必须按 1D 光栅顺序访问它们。早期的角落像素并不知道图像最终会变成什么。生成质量的扩展效果比 GPT-on-text 更差，并且在匹配计算量下从未达到扩散模型质量。

VAR 通过改变被生成的对象来解决生成顺序问题。它不是在空间中一个接一个预测图像 token，而是以逐渐提高的分辨率预测整张图像。第 1 步：预测一个 1x1 token（整体图像“摘要”）。第 2 步：预测一个 2x2 token 网格（更粗的特征）。第 3 步：预测一个 4x4 网格。第 K 步：预测最终的 (H/8)x(W/8) 网格。

每个尺度都会注意力关注所有先前尺度（在“尺度顺序”上因果），并在自身尺度内部并行。顺序问题消失了：尺度 k 上的整张图像在一次 transformer 前向中生成。

## 概念

### VQ-VAE 多尺度分词器

VAR 需要一个**多尺度离散分词器**。对于图像 x，它会产生一系列逐渐更高分辨率的 token 网格：

```
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

每个 z_k 使用同一个 codebook（典型大小 4096–16384）。每个尺度的分词并不是独立的——它被训练为：将每个尺度的残差相加即可重建 f：

```
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

这是一个**残差 VQ**（residual VQ）变体。尺度 k 捕获尺度 1..k-1 遗漏的内容。解码器接收所有尺度嵌入之和并生成图像。

多尺度 VQ 分词器只训练一次（类似 VQGAN），然后冻结。所有生成工作都由其上的自回归模型完成。

### 下一尺度预测

生成模型是一个 transformer，它看到所有先前尺度的 token，并预测下一尺度的 token。

输入序列结构：
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

位置嵌入同时编码尺度索引和该尺度内的空间位置。注意力在尺度顺序上是因果的：尺度 k、位置 (i, j) 的 token 可以关注尺度 1..k 的所有 token，以及尺度 k 内按某种内部顺序更早出现的 token（VAR 使用固定位置注意力，没有尺度内因果性——同一尺度内的所有位置并行预测）。

训练损失：在每个尺度 k，基于所有先前尺度的 token 预测 token z_k。对离散 VQ code 使用交叉熵损失。结构与 GPT 相同，只是这里的“序列”具有尺度结构。

### 生成

推理时：
```
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

对于 K = 10 个尺度，生成需要 10 次 transformer 前向。每次前向并行产生整个尺度——尺度内没有逐 token 自回归。对于 256x256 图像，这大约是 10 次前向，而 DiT 需要 28–50 次。

### 为什么下一尺度胜过下一 token

三个结构性优势：
1. **由粗到细符合自然图像统计。** 人类视觉感知和图像数据集都呈现尺度相关的规律：低频结构稳定且可预测；高频细节以低频内容为条件。下一尺度预测利用了这一点。
2. **尺度内并行生成。** 不同于 GPT 风格的 token AR，VAR 一步生成某个尺度上的所有 token。有效生成长度是对数尺度，而不是线性尺度。
3. **没有生成顺序偏置。** 尺度 k 的 token 可以看到完整的尺度 k-1；不存在“左侧”或“上方”偏置，迫使早期 token 在后续上下文可用之前过早承诺。

### Scaling Law

Tian et al. 证明，VAR 在 ImageNet 上的 FID 遵循幂律 scaling 曲线——就像 GPT 的困惑度一样。参数量或计算量翻倍，误差会可靠地减半。这是第一个展现出像语言模型那样干净 scaling 行为的图像生成模型。结果是，VAR 尺度预测可以从计算量中预测，而不是对每种架构做经验猜测。

### 与扩散的关系

VAR 和扩散共享同一个数据压缩叙事：二者都把生成问题拆成一串更容易的子问题。

- 扩散：逐渐加噪，学习撤销一步。
- VAR：逐渐增加分辨率，学习预测下一尺度。

它们是穿过同一问题的不同轴线。二者都产生可处理的条件分布。经验上，VAR 推理更快（更少前向，且每个尺度内完全并行），并且在类别条件 ImageNet 上匹配或超过 DiT。文本条件 VAR（VARclip、HART）是一个活跃研究方向。

## 构建它

在 `code/main.py` 中，你将：
1. 在合成“图像”数据（2D 高斯环）上构建一个小型**多尺度 VQ 分词器**。
2. 训练一个 **VAR 风格 transformer** 来做下一尺度 token 预测。
3. 通过调用 transformer 4 次（4 个尺度）并解码来采样。
4. 验证按尺度顺序训练使得尺度内生成可以并行。

这是一个玩具实现。重点是看到尺度结构注意力 mask，以及尺度内并行生成确实在工作。

## 交付它

本课会产出 `outputs/skill-var-tokenizer-designer.md`——一个用于设计多尺度分词器的技能：尺度数量、尺度比例、codebook 大小、残差共享、解码器架构。

## 练习

1. **尺度数量消融。** 使用 4、6、8、10 个尺度训练 VAR。测量重建质量与自回归前向次数的关系。尺度越多 = 残差越细 = 质量越好，但前向次数更多。

2. **Codebook 大小。** 使用 512、4096、16384 的 codebook 大小训练分词器。更大的 codebook 会带来更好的重建，但预测更困难。找到拐点。

3. **尺度内并行检查。** 对一个训练好的 VAR，显式测量注意力模式。在尺度 k 内，模型是否关注跨尺度位置而不关注尺度内位置？验证 mask 实现。

4. **VAR vs DiT scaling。** 对同一个 ImageNet 类别条件任务，在匹配参数预算下训练 VAR 和 DiT（例如 33M、130M、458M）。绘制 FID vs compute。VAR 应在每个规模上领先 DiT——在小规模上复现论文结果。

5. **文本条件。** 扩展 VAR，使其接收文本嵌入（CLIP pooled）作为通过 adaLN 注入的额外条件输入。这是 HART 配方。它对文本对齐采样的 FID 能改善多少？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| VAR | “Visual AutoRegressive” | 通过在 VQ token 网格金字塔上做下一尺度预测来生成图像 |
| 下一尺度预测 | “先预测粗，再预测细” | 模型在逐渐提高的分辨率尺度上预测 token，并以所有先前尺度为条件 |
| 多尺度 VQ 分词器 | “残差 VQ” | 产生 K 个逐渐提高分辨率 token 网格的 VQ-VAE，解码器将所有尺度相加 |
| 尺度 k | “金字塔第 k 层” | K 个分辨率层级之一，从 k=1 的 1x1 到 k=K 的 (H/p)x(W/p) |
| 尺度内并行 | “每个尺度一次前向” | 尺度 k 上的所有 token 在一次 transformer 前向中预测，而不是自回归预测 |
| 跨尺度因果 | “尺度顺序注意力” | 尺度 k 的 token 可以关注尺度 1..k 的全部内容，但不能关注尺度 k+1..K |
| 残差 VQ | “加性分词” | 每个尺度的 token 编码低尺度遗留的残差；解码器求和所有尺度嵌入 |
| VAR scaling law | “Image GPT scaling” | FID 随计算量遵循可预测幂律，类似语言模型的困惑度 |
| HART | “Hybrid VAR + text” | 文本条件 VAR 变体，将 MaskGIT 风格迭代解码与 VAR 的尺度结构结合 |
| 尺度位置嵌入 | “(scale, row, col) 三元组” | 位置编码同时携带尺度索引和该尺度内的空间坐标 |

## 延伸阅读

- [Tian et al., 2024 — “Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction”](https://arxiv.org/abs/2404.02905) — VAR 论文，规范参考
- [Peebles and Xie, 2022 — “Scalable Diffusion Models with Transformers”](https://arxiv.org/abs/2212.09748) — DiT，扩散对比基线
- [Esser et al., 2021 — “Taming Transformers for High-Resolution Image Synthesis”](https://arxiv.org/abs/2012.09841) — VQGAN，VAR 多尺度分词器所扩展的分词器家族
- [van den Oord et al., 2017 — “Neural Discrete Representation Learning”](https://arxiv.org/abs/1711.00937) — VQ-VAE，离散图像分词的基础
- [Tang et al., 2024 — “HART: Efficient Visual Generation with Hybrid Autoregressive Transformer”](https://arxiv.org/abs/2410.10812) — 文本条件 VAR
