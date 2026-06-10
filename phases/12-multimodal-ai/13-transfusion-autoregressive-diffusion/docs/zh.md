# Transfusion：一个 Transformer 中实现自回归文本 + 扩散图像

> Chameleon 和 Emu3 将全部赌注押在离散 token 上。它们确实可行，但量化瓶颈显而易见——图像质量在低于连续空间扩散模型的水平上就遇到了瓶颈。Transfusion（Meta，Zhou 等人，2024 年 8 月）采取了相反的路线：保持图像连续，完全放弃 VQ-VAE，用一个 Transformer 配合两个损失函数进行训练。文本 token 使用 next-token-prediction（NTP）。图像 patch 使用 flow-matching / diffusion 损失。两个目标函数优化同一套权重。Stable Diffusion 3（MMDiT）的底层架构与其是近亲。本节课将研读 Transfusion 论文，构建一个玩具级的双损失训练器，并追踪让单个 Transformer 同时完成两项任务的注意力掩码。

**类型：** Build
**语言：** Python（标准库，MNIST 规模的玩具双损失训练器）
**前置知识：** Phase 12 · 11（Chameleon），Phase 8（Generative AI）
**时间：** 约 180 分钟

## 学习目标

- 搭建一个同时运行两个损失的 Transformer（文本 token 用 NTP，图像 patch 用 diffusion MSE），共享同一个 backbone。
- 解释为什么图像 patch 之间使用双向注意力、文本 token 之间使用因果注意力是正确的掩码选择。
- 从计算量、质量和代码复杂度三个维度，比较 Transfusion 风格（连续图像，diffusion 损失）与 Chameleon 风格（离散图像，NTP）。
- 说明 MMDiT 的贡献：每个 block 中的模态特定权重，以及在残差流上的联合注意力。

## 问题背景

离散与连续图像 token 的争论比 LLM 出现得更早。连续表示（原始像素、VAE 隐变量）保留细节。离散 token（VQ 索引）适配 Transformer 的原生词表，但在量化步骤中丢失细节。

Chameleon / Emu3 选择了离散路线：一个损失、一个架构，但图像保真度受限于 tokenizer 质量。

扩散模型选择了连续路线：图像质量出色，但与 LLM 是分开的模型，需要复杂的噪声调度工程，且与文本生成无法干净地集成。

Transfusion 提出的问题是：能否两者兼得？保持图像连续，仍然只训练一个模型，将两个损失缝合进同一个梯度步中。

## 核心概念

### 双损失架构

单个 decoder-only Transformer 处理一个序列，其中包含：

- 文本 token（离散的，来自 BPE 词表）。
- 图像 patch（连续的，16x16 像素块通过线性嵌入投影到 hidden dim——与 ViT 编码器的输入层相同）。
- `<image>` 和 `</image>` 标签标记连续 patch 所在的位置。

前向传播只运行一次。损失根据 token 类型选择两个头之一：

- 对于文本 token：在 vocab-logits 头上使用标准交叉熵。
- 对于图像 patch：在连续 patch 上使用 diffusion 损失——预测加到每个 patch 上的噪声。

梯度流经共享的 Transformer 主体。两个损失同时改善共享权重。

### 注意力掩码：因果文本 + 双向图像

文本 token 必须是因果的——不能让一个文本 token 关注到未来的文本，否则 teacher forcing 会失效。然而，图像 patch 代表的是同一个快照；它们应该在同一个图像块内双向地相互关注。

掩码规则：

```
M[i, j] = 1 如果满足：
  (i 是文本 且 j 是文本 且 j <= i)   # 文本之间因果
  或 (i 是图像 且 j 是图像 且 same_image_block(i, j))   # 图像块内双向
  或 (i 是文本 且 j 是图像 且 j < i_image_end)   # 文本关注前面的图像
  或 (i 是图像 且 j 是文本 且 j < i_image_start)   # 图像关注前面的文本
```

在训练和推理时实现为 block-triangular mask。

### Transformer 内部的 Diffusion 损失

Diffusion 损失是标准的：向图像 patch 添加噪声，让模型预测噪声（或等价地预测干净 patch）。Transfusion 的版本使用 flow matching——预测从噪声到干净数据的 velocity field。

训练过程：
1. 对于每个图像 patch x0，采样一个随机时间步 t。
2. 采样噪声 ε，计算 xt = (1-t) * x0 + t * ε（flow matching 的线性插值）。
3. Transformer 预测 v_theta(xt, t)；损失 = MSE(v_theta(xt, t), ε - x0)。
4. 与同一序列中的文本 NTP 损失一起反向传播。

推理时，生成过程为：
- 文本 token：标准自回归采样。
- 图像 patch：diffusion 采样循环（通常 10-30 步），以前面的文本 token 为条件。

### MMDiT：Stable Diffusion 3 的变体

Stable Diffusion 3（Esser 等人，2024 年 3 月）与 Transfusion 大约同期发布了 MMDiT（Multimodal Diffusion Transformer）。两者架构是近亲关系。

MMDiT 的关键区别：

- 每个 block 有模态特定的权重。每个 Transformer block 为文本 token 和图像 patch 分别配备独立的 Q、K、V 和 MLP 权重。注意力是联合的（跨模态）；其余部分是模态特定的。
- Rectified flow 训练。一种特定的 flow-matching 变体，采样方式已知，数学比 DDPM 更简单。
- 规模。MMDiT 是 SD3 的 backbone（20 亿和 80 亿参数变体）。Transfusion 论文扩展到 70 亿参数。

两者都收敛到同一个核心思想：一个 Transformer 在文本上运行 NTP，在连续图像表示上运行 diffusion。

### 为什么这优于 Chameleon 风格

连续扩散与离散 NTP 在图像生成上的质量差距是可量化的。Transfusion 论文报告：

- 在 70 亿参数下，FID 比同规模的 Chameleon 风格模型低 3-5 分。
- 不需要训练 tokenizer——图像编码器更简单（线性投影到 hidden dim，与 ViT 的输入层相同）。
- 推理时可以并行化图像 patch 去噪，而自回归图像 token 无法做到。

缺点：Transfusion 是双损失模型，训练动态更复杂。损失权重需要调参。NTP 和 diffusion 之间的调度不匹配可能导致一个头主导训练。

### 后续发展

Janus-Pro（第 12.15 课）通过将视觉编码器解耦为理解和生成两个部分——SigLIP 用于理解，VQ 用于生成——同时共享 Transformer 主体，进一步发展了 Transfusion 的思想。Show-o（第 12.14 课）将 diffusion 替换为离散 diffusion（masked prediction）。统一生成家族在 Transfusion 之后迅速分化。

2026 年生产级的能生成图像的 VLM——Gemini 3 Pro、GPT-5、Claude Opus 4.7 的图像生成路径——几乎肯定使用了这一家族的某种后代。具体细节是专有的。

## 动手实践

`code/main.py` 在一个类似 MNIST 的玩具问题上构建了一个 Transfusion：

- 文本 caption 是描述数字（0-9）的短整数序列。
- 图像是 4x4 的字节网格。
- 一对共享权重的线性投影充当 Transformer 的替身；文本用 NTP 损失，噪声 patch 用 MSE 损失。
- 训练循环交替使用两个损失，注意力掩码是显式构造的。
- 生成过程在一个前向传播中同时产生文本 caption 和 4x4 图像。

这个 Transformer 是玩具级的。真正的价值在于双损失的管线搭建、注意力掩码构造和推理循环。

## 产出交付

本节课产出 `outputs/skill-two-loss-trainer-designer.md`。给定一个新的多模态训练任务（文本 + 图像、文本 + 音频、文本 + 视频），它设计双损失调度方案（损失权重、掩码形状、共享 vs 模态特定 block）并标记实现风险。

## 练习题

1. 一个 Transfusion 风格模型训练时 70% 是文本 token，30% 是图像 patch。图像 diffusion 损失的幅度约为文本 NTP 损失的 10 倍。应该设置什么损失权重来平衡它们？

2. 为序列 `[T, T, <image>, P, P, P, P, </image>, T]` 实现 block-triangular mask。将每个条目标记为 0 或 1。

3. MMDiT 有模态特定的 QKV 权重。与 Transfusion 的完全共享 Transformer 相比，这增加了多少参数开销？在 70 亿参数下，是否值得？

4. 生成过程：给定一个文本提示，模型运行 NTP 生成 50 个 token，然后遇到 `<image>`，然后在 20 个去噪步中对 256 个 patch 运行 diffusion。总共需要多少次前向传播？

5. 阅读 SD3 论文第 3 节。描述 rectified flow 以及为什么它比 DDPM 在更少的推理步数内收敛。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| Two-loss training | "NTP + diffusion" | 单个 Transformer 在同一个梯度步中同时优化文本 token 的交叉熵和连续图像 patch 的 MSE |
| Flow matching | "Rectified flow" | Diffusion 变体，预测从噪声到干净数据的 velocity field；数学比 DDPM 更简单 |
| MMDiT | "Multimodal DiT" | Stable Diffusion 3 的架构：联合注意力，模态特定的 MLP 和 norm |
| Block-triangular mask | "Causal text + bidirectional image" | 注意力掩码，在文本之间是因果的，在图像区域内是双向的 |
| Continuous image representation | "No VQ" | 图像 patch 作为实值向量，而非整数码本索引 |
| Velocity prediction | "v-parameterization" | 网络输出是噪声与数据之间的 velocity field，而非噪声本身 |

## 延伸阅读

- [Zhou et al. — Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser et al. — Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao et al. — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
