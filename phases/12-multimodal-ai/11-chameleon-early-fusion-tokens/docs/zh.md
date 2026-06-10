# Chameleon 与早期融合（Early-Fusion）纯 Token 多模态模型

> 到目前为止，我们见过的所有 VLM 都将图像和文本分开处理。视觉 token 来自视觉编码器，流经一个投影器，然后在 LLM 内部与文本相遇。视觉和文本的词表从不重叠。Chameleon（Meta，2024 年 5 月）提出了一个问题：如果它们重叠会怎样？训练一个 VQ-VAE，将图像转换为来自共享词表的离散 token 序列。现在，每一份多模态文档都是一个序列——文本 token 和图像 token 交错排列，使用单一的自回归损失。附带效果是：该模型可以生成混合模态的输出——在单次推理调用中交替生成文本和图像 token。本课阅读早期融合的相关论文，并从头到尾构建一个简化版本。

**类型：** 构建
**语言：** Python（标准库，VQ-VAE tokenizer + 交错解码器）
**前置知识：** Phase 12 · 05, Phase 8（生成式 AI）
**时间：** 约 180 分钟

## 学习目标

- 解释共享词表 + 单一损失如何改变模型的能力。
- 描述 VQ-VAE 如何将图像 token 化为与 transformer 的 next-token 目标兼容的离散序列。
- 说出 Chameleon 的训练稳定性技巧：QK-Norm、dropout 的放置位置、LayerNorm 的顺序。
- 比较 Chameleon 与 BLIP-2 的 Q-Former 方法，并描述各自适用的场景。

## 问题所在

基于适配器的 VLM（LLaVA、BLIP-2、Qwen-VL）将文本和图像视为两种不同的事物。文本 token 经过 `embed(text_token)`；图像则经过 `visual_encoder(image) → projector → ... pseudo_tokens`。模型有两条输入路径，在中途汇合。

这带来了三个后果：

1. LLM 只能消费图像，不能生成图像。输出只能是文本。
2. 混合模态文档（如文章中交替出现的段落和图像）处理起来很别扭——你要么在模型外部解析多模态输入，要么串联多次生成。
3. 分布不匹配。视觉 token 和文本 token 存在于隐藏空间的不同区域，造成微妙的对齐问题。

Chameleon 否定了这一前提：图像只是来自共享词表的离散 token 序列。在交错的文档上训练模型，一个损失函数，一个自回归解码器，你就能免费解锁混合模态生成能力。

## 核心概念

### 作为图像 tokenizer 的 VQ-VAE

该 tokenizer 是一个向量量化变分自编码器（Vector-Quantized Variational Autoencoder）。其架构如下：

- 编码器（Encoder）：CNN + ViT，将图像映射为空间特征图，例如 32x32 的特征，维度为 256。
- 码本（Codebook）：一个包含 K 个向量的可学习词表（Chameleon 使用 8192 个），维度同样为 256。
- 量化（Quantization）：对于每个空间特征，通过 L2 距离查找最近的码本条目。将连续特征替换为对应的整数索引。
- 解码器（Decoder）：CNN，将量化后的特征还原为像素。

训练目标：VAE 重建损失 + commitment loss + codebook loss。码本索引构成了图像的离散字母表。

对于 Chameleon：一张图像变为 32*32 = 1024 个 token，来自一个包含 8192 个条目的词表。与文本 token（来自 LLM 的 BPE 词表，例如 32000 个）拼接。最终词表大小为：40192。Transformer 看到的是同一个序列，使用同一个损失函数。

### 共享词表

Chameleon 的词表结合了文本 token、图像 token 和模态分隔符。每个 token 都有唯一的 ID。输入嵌入层将每个 ID 映射为一个 D 维的隐藏向量。输出投影将隐藏向量映射回词表的 logits。Softmax 选择下一个 token，无论其模态如何。

分隔符很重要：`<image>` 和 `</image>` 标签包裹图像 token 序列。在生成时，如果模型输出了 `<image>`，下游软件就知道接下来的 1024 个 token 是 VQ 索引，需要发送给解码器进行像素渲染。

### 混合模态生成

推理过程就是在共享词表中进行 next-token 预测。示例提示词："Draw a cat and describe it."（画一只猫并描述它。）Chameleon 输出：

```
<image> 4821 1029 2891 ... (1024 个图像 token) </image>
The cat is orange, sitting on a windowsill...
```

模型自主决定顺序——它可能先生成图像再生成文本，也可能先文本后图像，或者交错生成。同一个解码器，同一个损失函数。

与只能生成文本的适配器 VLM 相比，Chameleon 重新开启了模型输出模态的可能性。

### 训练稳定性 —— QK-Norm、dropout、LayerNorm 顺序

早期融合训练在大规模下是不稳定的。Chameleon 的论文记录了三个技巧：

- **QK-Norm**。在注意力机制内部，对 query 和 key 的投影应用 LayerNorm，再进行点积运算。防止在深层网络中 logits 的幅度爆炸。多个 2024 年后的大型模型都采用了这一方法。
- **Dropout 的放置位置**。在每次残差相加（residual-add）之后都应用 dropout，而不仅仅是在 attention 和 MLP 之后。当图像 token 的梯度可能占主导时，需要更多的正则化。
- **LayerNorm 的顺序**。在残差分支上使用 Pre-LN（标准做法），并在最后一个块的跳跃连接（skip connection）上额外加一个 LN。稳定最终层的梯度流。

没有这些技巧，340 亿参数的 Chameleon 训练在多个检查点处都会发散。有了这些技巧，它就能收敛。训练配方与架构本身一样，都是该工作的核心贡献。

### Tokenizer 的重建上限

VQ-VAE 是有损的。在码本条目为 8192、每张 512x512 图像使用 1024 个 token 的情况下，重建 PSNR 上限约为 26-28 dB。这对于可识别的图像生成来说已经足够，但明显比连续空间中的扩散模型差（Stable Diffusion 3 可达到 32+ dB）。

Tokenizer 是瓶颈。更好的 tokenizer（MAGVIT-v2、IBQ、SBER-MoVQGAN）可以提升这个上限。Emu3（第 12.12 课）仅凭一个更好的 tokenizer 就达到了 SDXL 级别的生成质量。

### Chameleon vs BLIP-2 / LLaVA

**Chameleon（早期融合，共享词表）：**
- 一个损失函数，一个解码器。
- 生成混合模态输出。
- Tokenizer 是质量上限。
- 成本高：推理路径上每张生成的图像都需要 VQ-VAE 解码器。

**BLIP-2 / LLaVA（晚期融合，独立塔）：**
- 视觉输入，仅文本输出。
- 复用预训练的 LLM。
- 理解任务没有 tokenizer 瓶颈。
- 成本低：单次前向传播。

根据任务选择。如果你需要图像生成，选 Chameleon 家族。如果你只需要理解，适配器-VLM 更简单，能复用更多预训练计算资源。

### Fuyu 和 AnyGPT

**Fuyu**（Adept，2023）是一种相关方法：完全跳过独立的视觉编码器，将原始图像块通过 LLM 的输入投影层输入，就像它们是 token 一样，没有 tokenizer。比 Chameleon 更简单，但失去了共享词表的输出生成能力。

**AnyGPT**（Zhan et al., 2024）将 Chameleon 扩展到四种模态：文本、图像、语音、音乐。每种模态都使用相同的 VQ-VAE 技巧，共享同一个 transformer。任意模态到任意模态的生成。在第 12.16 课中有更详细的介绍。

## 动手实践

`code/main.py` 构建了一个端到端的简化早期融合模型：

- 一个微型 VQ-VAE 风格的量化器，将 8x8 的图像块映射为码本索引（K=16）。
- 一个共享词表，包含（文本 ID 0..31）+（图像 ID 32..47）+（分隔符 48, 49）。
- 一个玩具自回归解码器（bigram 表），在合成标题 + 图像 token 序列上训练。
- 采样循环，给定提示词后输出交替的文本 + 图像 token。

代码有意将 transformer 保持得极小（bigram），以便你可以从头到尾追踪信号流。

## 交付成果

本课产出 `outputs/skill-tokenizer-vs-adapter-picker.md`。给定一份产品规格（仅理解 vs 理解+生成、所需的图像质量、成本预算），它在 Chameleon 家族（早期融合）和 LLaVA 家族（晚期融合）之间做出选择，并用定量的经验法则进行论证。

## 练习题

1. Chameleon 使用 K=8192 个码本条目，每张 512x512 图像使用 1024 个 token。估算与 24 位 RGB 图像相比的压缩比。它是有损的吗？有多有损？

2. 一张 4K 图像（3840x2160）在相同的 VQ-VAE 密度下会产生多少图像 token？Chameleon 风格的模型能在一次推理调用中生成 4K 图像吗？什么会先崩溃——上下文长度、tokenizer 质量，还是 KV cache？

3. 用纯 Python 实现 QK-Norm。给定一个 64 维的 query 和 key，展示在 LayerNorm 前后的点积结果。为什么在深层网络中控制幅度很重要？

4. 阅读 Chameleon 论文的第 2.3 节关于训练稳定性的内容。描述在没有 QK-Norm 的情况下，340 亿参数模型观察到的确切失效模式。"范数爆炸"（norm explosion）的特征是什么？

5. 扩展玩具解码器，使其在仅给定文本提示词的情况下输出混合模态响应。测量在训练数据分布为 60% 文本优先 / 40% 图像优先时，模型选择图像优先与文本优先的频率。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| 早期融合（Early fusion） | "统一 token" | 图像被转换为离散 token，从一开始就与 transformer 的词表共享 |
| VQ-VAE | "图像 tokenizer" | CNN + ViT + 码本，将图像映射为 transformer 可以预测的整数索引 |
| 共享词表（Shared vocabulary） | "一本字典" | 一个覆盖文本 + 图像 + 模态分隔符的单一 token ID 空间 |
| QK-Norm | "注意力稳定器" | 在 query 和 key 进行点积之前对其应用 LayerNorm，防止范数爆炸 |
| 混合模态生成（Mixed-modality generation） | "文本 + 图像输出" | 推理过程自主地在一次前向传播中产生交错的文本和图像 token |
| 码本大小（Codebook size） | "K 个条目" | VQ-VAE 可以量化到的离散向量数量；在压缩率和保真度之间权衡 |
| Tokenizer 上限（Tokenizer ceiling） | "重建极限" | 解码 VQ token 所能达到的最佳 PSNR；限制了模型的图像质量 |

## 延伸阅读

- [Chameleon Team — Chameleon: Mixed-Modal Early-Fusion Foundation Models (arXiv:2405.09818)](https://arxiv.org/abs/2405.09818)
- [Aghajanyan et al. — CM3 (arXiv:2201.07520)](https://arxiv.org/abs/2201.07520)
- [Yu et al. — CM3Leon (arXiv:2309.02591)](https://arxiv.org/abs/2309.02591)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Adept — Fuyu-8B blog (adept.ai)](https://www.adept.ai/blog/fuyu-8b)
