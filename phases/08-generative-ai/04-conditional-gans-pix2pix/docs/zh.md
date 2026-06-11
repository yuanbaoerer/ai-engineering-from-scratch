# 条件 GAN 与 Pix2Pix

> 2014-2017 年生成模型的第一个重大突破，是能够控制 GAN 生成什么。附加一个标签、一张图像或一句话。Pix2Pix 做的是图像版本，而且在狭窄的图像到图像任务上，它至今仍然胜过所有通用文本到图像模型。

**类型：** 构建
**语言：** Python
**先修：** 阶段 8 · 03（GANs）、阶段 4 · 06（U-Net）、阶段 3 · 07（CNNs）
**时间：** 约 75 分钟

## 问题

无条件 GAN 会采样任意人脸。做演示有用，生产中没用。你想要的是：*把草图映射成照片*、*把地图映射成航拍图*、*把白天场景映射成夜晚*、*给灰度图上色*。在所有这些任务中，你都会得到一张输入图像 `x`，并且必须输出与其存在某种语义对应关系的 `y`。每个 `x` 都可能对应许多合理的 `y`。均方误差会把它们压成一团糊。对抗损失不会，因为“看起来真实”是尖锐的。

条件 GAN（Mirza & Osindero，2014）把条件 `c` 作为输入同时加入 `G` 和 `D`。Pix2Pix（Isola 等，2017）对此做了专门化：条件是一整张输入图像，生成器是 U-Net，判别器是一个*基于 patch 的*分类器（PatchGAN），损失是对抗损失 + L1。即使在 2026 年，这个配方在狭窄的图像到图像领域也胜过从零训练的文本到图像模型，因为它是在*配对数据*上训练的——你恰好拥有所需的信号。

## 概念

![Pix2Pix: U-Net generator, PatchGAN discriminator](../assets/pix2pix.svg)

**条件 G。** `G(x, z) → y`。在 Pix2Pix 中，`z` 是 G 内部的 dropout（没有输入噪声——Isola 发现显式噪声会被忽略）。

**条件 D。** `D(x, y) → [0, 1]`。输入是*成对的*（条件，输出）。这是关键差异：D 必须判断 `y` 是否与 `x` 一致，而不只是判断 `y` 看起来是否真实。

**U-Net 生成器。** 带有跨瓶颈跳跃连接（skip connections）的编码器-解码器。对于输入和输出共享低层结构（边缘、轮廓）的任务至关重要。没有这些跳连，高频细节会消失。

**PatchGAN 判别器。** D 不输出单个真/假分数，而是输出一个 `N×N` 网格，每个单元判断约 70×70 像素感受野的真实性，然后取平均。这是一种马尔可夫随机场假设：真实感是局部的。它训练更快、参数更少、输出更锐利。

**损失。**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

L1 项稳定训练，并把 G 推向已知目标。L1 比 L2 给出更锐利的边缘（中位数，而不是均值）。`λ = 100` 是 Pix2Pix 的默认值。

## CycleGAN——当你没有配对数据时

Pix2Pix 需要配对的 `(x, y)` 数据。CycleGAN（Zhu 等，2017）通过增加一个额外损失来去掉这个要求：*循环一致性*（cycle consistency）损失。两个生成器 `G: X → Y` 和 `F: Y → X`。训练它们，使得 `F(G(x)) ≈ x` 且 `G(F(y)) ≈ y`。这样你就可以在没有配对样本的情况下，把马变成斑马、把夏天变成冬天。

在 2026 年，非配对图像到图像任务大多通过扩散模型（ControlNet、IP-Adapter）而不是 CycleGAN 完成，但循环一致性的思想几乎保留在每篇非配对领域自适应论文中。

## 构建它

`code/main.py` 在 1-D 数据上实现了一个微型条件 GAN。条件 `c` 是类别标签（0 或 1）。任务：为给定类别生成来自条件分布的样本。

### 步骤 1：把条件追加到 G 和 D 的输入中

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

独热编码（one-hot encoding）是最简单的方式。更大的模型会使用可学习嵌入、FiLM 调制或交叉注意力（cross-attention）。

### 步骤 2：进行条件训练

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

生成器必须匹配*给定条件下*的真实分布，而不是边缘分布。

### 步骤 3：验证每个类别的输出

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## 常见陷阱

- **条件被忽略。** G 学会边缘化，D 从不惩罚，因为条件信号太弱。修复：更强地给 D 注入条件（早期层，而不只是后期层），使用投影判别器（projection discriminator，Miyato & Koyama 2018）。
- **L1 权重太低。** G 漂移到任意看起来真实但不忠实的输出。Pix2Pix 风格任务从 λ≈100 开始。
- **L1 权重太高。** G 产生模糊输出，因为 L1 仍然是一个 L_p 范数。训练稳定后逐步降低。
- **D 中泄漏真实标签。** 把 `(x, y)` 拼接为 D 的输入，而不只是 `y`。否则 D 无法检查一致性。
- **每个类别的模式崩塌。** 每个类别都可能独立崩塌。运行按类别划分的多样性检查。

## 使用它

2026 年图像到图像任务的状态：

| 任务 | 最佳方法 |
|------|---------------|
| 草图 → 照片，同一领域，配对数据 | Pix2Pix / Pix2PixHD（依然快速，依然锐利） |
| 草图 → 照片，非配对 | 使用 Scribble 条件模型的 ControlNet |
| 语义分割 → 照片 | SPADE / GauGAN2 或 SD + ControlNet-Seg |
| 风格迁移 | 带 IP-Adapter 或 LoRA 的扩散模型；GAN 方法已是遗留方案 |
| 深度 → 照片 | Stable Diffusion 上的 ControlNet-Depth |
| 超分辨率 | Real-ESRGAN（GAN）、ESRGAN-Plus 或 SD-Upscale（扩散） |
| 上色 | ColTran、基于扩散的上色器，或 Pix2Pix-color |
| 白天 → 夜晚、季节、天气 | CycleGAN 或基于 ControlNet 的方法 |

当 (a) 你有数千个配对样本，(b) 任务狭窄且可重复，且 (c) 你需要快速推理时，Pix2Pix 仍然是正确工具。在通用开放域任务上，扩散模型胜出。

## 交付它

保存 `outputs/skill-img2img-chooser.md`。该技能接收任务描述、数据可用性（配对 vs 非配对，N 个样本）以及延迟/质量预算，然后输出：方法（Pix2Pix、CycleGAN、ControlNet 变体、SDXL + IP-Adapter）、训练数据需求、推理成本和评估协议（LPIPS、FID、任务特定指标）。

## 练习

1. **简单。** 修改 `code/main.py`，添加第三个类别。确认 G 仍然把每个类别的噪声映射到正确模式。
2. **中等。** 在 1-D 设置中用感知风格损失替换 L1（例如，把一个小型冻结 D 当作特征提取器）。它会改变条件分布的锐利程度吗？
3. **困难。** 在 1-D 设置中勾勒一个 CycleGAN：两个分布、两个生成器、循环损失。展示它能在没有配对数据的情况下学会二者之间的映射。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Conditional GAN | “带标签的 GAN” | G(z, c), D(x, c)。两个网络都看到条件。 |
| Pix2Pix | “图像到图像 GAN” | 带 U-Net G 和 PatchGAN D + L1 损失的配对 cGAN。 |
| U-Net | “带跳连的编码器-解码器” | 对称卷积网络；跳连保留高频。 |
| PatchGAN | “局部真实感分类器” | D 输出每个 patch 的分数，而不是全局分数。 |
| CycleGAN | “非配对图像翻译” | 两个 G + 循环一致性损失；无配对数据。 |
| SPADE | “GauGAN” | 用语义图归一化中间激活；分割到图像。 |
| FiLM | “Feature-wise linear modulation” | 来自条件的逐特征仿射变换；廉价条件化。 |

## 生产说明：Pix2Pix 作为受延迟约束的基线

当你有配对数据和狭窄任务（草图 → 渲染、语义图 → 照片、白天 → 夜晚）时，Pix2Pix 的一次性推理在延迟上比扩散模型快一个数量级。生产中的比较通常是：

| 路径 | 步数 | 单张 L4 上 512² 的典型延迟 |
|------|-------|----------------------------------------|
| Pix2Pix（U-Net 前向） | 1 | ~30 ms |
| SD-Inpaint 或 SD-Img2Img | 20 | ~1.2 s |
| SDXL-Turbo Img2Img | 1-4 | ~0.15-0.35 s |
| ControlNet + SDXL base | 20-30 | ~3-5 s |

Pix2Pix 在静态批处理中赢得吞吐量（每个请求都是相同 FLOPs）。扩散模型在质量和泛化上胜出。现代玩法通常是为狭窄任务上线一个 Pix2Pix 风格的蒸馏模型，并为长尾输入提供扩散模型回退。

## 延伸阅读

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — cGAN 论文。
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — Pix2Pix。
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — CycleGAN。
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) — Pix2PixHD。
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) — SPADE / GauGAN。
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) — 投影 D。
