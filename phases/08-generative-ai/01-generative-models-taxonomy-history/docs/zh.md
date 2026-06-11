# 生成模型 —— 分类法与历史

> 每一种图像模型、文本模型、视频模型和 3D 模型都可以归入五个桶之一。选错桶，你会和数学苦战数周；选对桶，过去十二年这个领域的进展就会在你脑中清晰地层层堆叠起来。

**类型：** 学习
**语言：** Python
**先修要求：** 第 2 阶段（机器学习基础）、第 3 阶段（深度学习核心）、第 7 阶段 · 14（Transformer）
**时间：** 约 45 分钟

## 问题

生成模型只做一件事：给定从某个未知分布 `p_data(x)` 中抽取的训练样本，输出看起来像是来自同一分布的新样本。人脸、句子、MIDI 文件、蛋白质结构——如果眯起眼看，它们都是同一个问题。

麻烦在于，`p_data` 存在于一个拥有数百万维度的空间中（512x512 RGB 图像大约有 786k 个维度），样本位于这个空间内部一张很薄的流形（manifold）上，而你可能只有 1000 万个样本。暴力拟合密度没有希望。每一种生成模型都是一种折中：把一个难题换成一个稍微没那么难的问题。

过去十二年里，有五大家族存活下来。理解每个家族做出的折中，就能理解为什么它在某些任务上胜出，而在另一些任务上崩溃。

## 概念

![Five families of generative models — taxonomy by what they model](../assets/taxonomy.svg)

**1. 显式密度，可精确计算。** 把 `log p(x)` 写成一个你确实可以求值的和式。自回归模型（Autoregressive models，如 PixelCNN、WaveNet、GPT）将 `p(x) = ∏ p(x_i | x_<i)` 分解。归一化流（Normalizing flows，如 RealNVP、Glow）把 `p(x)` 构建为简单基分布的可逆变换。优点：精确似然，训练损失干净。缺点：自回归推理是顺序的（长序列很慢），流模型需要可逆架构（架构限制很强）。

**2. 显式密度，近似计算。** 从下方界定 `log p(x)`（ELBO），并优化这个下界。VAE（Kingma 2013）使用带变分后验的编码器-解码器。扩散模型（DDPM，Ho 2020）训练一个去噪器，隐式地优化加权 ELBO。到 2026 年，扩散是图像、视频和 3D 的主导骨干。

**3. 隐式密度。** 完全跳过密度；学习一个生成器 `G(z)` 来产生样本，再学习一个判别器 `D(x)` 来判断真实与伪造。GAN（Goodfellow 2014）属于这一类。推理很快（一次前向传播），但训练出了名地不稳定。即使在 2026 年，StyleGAN 1/2/3 在固定领域的照片级真实感（人脸、卧室）上仍然是最先进水平。

**4. 基于分数 / 连续时间。** 直接学习对数密度的梯度 `∇_x log p(x)`（即分数，score）。Song & Ermon（2019）表明，分数匹配可以把扩散推广为随机微分方程（SDE）。流匹配（Flow matching，Lipman 2023）是 2024-2026 年的热门方向：无需模拟的训练、更直的路径、比 DDPM 快 4-10 倍的采样。Stable Diffusion 3、Flux、AudioCraft 2 都使用流匹配。

**5. 对离散编码做基于 token 的自回归建模。** 用 VQ-VAE 或残差量化器（residual quantizer）把高维数据压缩成较短的离散 token 序列，然后用 Transformer 建模这个 token 序列。Parti、MuseNet、AudioLM、VALL-E、Sora 的 patch tokenizer 都使用这一思路。这本质上是第 1 类外加一个学习得到的 tokenizer。

## 简史

| 年份 | 模型 | 为什么重要 |
|------|-------|-----------------|
| 2013 | VAE (Kingma) | 第一个拥有可用训练损失的深度生成模型。 |
| 2014 | GAN (Goodfellow) | 隐式密度、没有似然——样本清晰得令人震惊。 |
| 2015 | DRAW, PixelCNN | 顺序式图像生成。 |
| 2017 | Glow, RealNVP | 可逆流；通过深度获得精确似然。 |
| 2017 | Progressive GAN | 首次生成百万像素级人脸。 |
| 2019 | StyleGAN / StyleGAN2 | 在人脸这个单一领域，照片级真实感仍然难以超越。 |
| 2020 | DDPM (Ho) | 扩散变得实用。 |
| 2021 | CLIP, DALL-E 1, VQGAN | 文生图进入主流。 |
| 2022 | Imagen, Stable Diffusion 1, DALL-E 2 | 潜空间扩散 + 文本条件 = 商品化。 |
| 2022 | ControlNet, LoRA | 对预训练扩散模型进行精细控制。 |
| 2023 | SDXL, Midjourney v5, Flow matching | 规模 + 更好的训练动态。 |
| 2024 | Sora, Stable Diffusion 3, Flux.1 | 视频扩散；流匹配胜出。 |
| 2025 | Veo 2, Kling 1.5, Runway Gen-3, Nano Banana | 生产级视频。 |
| 2026 | Consistency + Rectified Flow | 从扩散骨干实现一步采样。 |

## 五问分诊法

当一篇新的生成模型论文出现时，在阅读方法部分之前先回答这五个问题。

1. **建模的是什么？** 像素、潜变量、离散 token、3D 高斯、网格、波形？
2. **密度是显式还是隐式？** 他们有没有写出 `log p(x)`？
3. **采样：一次生成还是迭代生成？** 迭代意味着推理更慢；一次生成通常意味着对抗式或蒸馏式。
4. **条件：无条件、类别、文本、图像、姿态？** 这决定损失和架构脚手架。
5. **评估：FID、CLIP 分数、IS、人类偏好、任务准确率？** 每一种都有已知失效模式（见第 14 课）。

本阶段的每一课你都会重新回答这五个问题。到最后，它们会变成你的本能反应。

## 构建它

本课的代码是一个轻量级可视化：从样本中用三种玩具方法（核密度、离散直方图，以及最近样本的“类 GAN”生成器）拟合一维高斯混合，这样你可以在一个屏幕能打印出来的问题上，看清显式密度与隐式密度的区别。

运行 `code/main.py`。它会从一个双峰高斯混合中抽取 2000 个样本，然后打印：

```
explicit density (histogram): p(x in [-0.5, 0.5]) ≈ 0.38
approximate density (KDE):     p(x in [-0.5, 0.5]) ≈ 0.41
implicit (nearest-sample gen): 20 new samples printed, no p(x)
```

注意：前两种允许你问“这个点有多可能？”第三种不能。这就是*显式 vs 隐式*的区别，它会影响之后每一课。

## 使用它

2026 年，哪类任务该用哪个家族？

| 任务 | 最佳家族 | 原因 |
|------|-------------|-----|
| 照片级人脸、窄领域 | StyleGAN 2/3 | 仍然最清晰，推理最快。 |
| 通用文生图 | 潜空间扩散 + 流匹配 | SD3、Flux.1、DALL-E 3。 |
| 快速文生图 | Rectified flow + 蒸馏 | SDXL-Turbo、SD3-Turbo、LCM。 |
| 文生视频 | Diffusion Transformer + 流匹配 | Sora、Veo 2、Kling。 |
| 语音 + 音乐 | 基于 token 的 AR（AudioLM、VALL-E、MusicGen）或流匹配（AudioCraft 2） | 离散 token 扩展成本低。 |
| 3D 场景 | Gaussian Splatting 拟合、扩散先验 | 3D-GS 用于重建，扩散用于新视角。 |
| 密度估计（不采样） | 流模型 | 唯一拥有精确 `log p(x)` 的家族。 |
| 仿真 / 物理 | 流匹配、score SDE | 直线路径，平滑向量场。 |

## 交付它

保存为 `outputs/skill-model-chooser.md`。

该技能接受一个任务描述，并输出：（1）应使用哪个家族，（2）三个开源选项和三个托管选项的排序列表，（3）你应该留意的可能失效模式，以及（4）计算/时间预算。

## 练习

1. **简单。** 对下面五个产品分别识别其家族和骨干：ChatGPT image、Midjourney v7、Sora、Runway Gen-3、ElevenLabs。证据应来自公开技术报告。
2. **中等。** 你明天要读的一篇论文声称采样比扩散快 100 倍。写下三个问题，用来检查这种加速在条件生成和高分辨率下是否仍然成立。
3. **困难。** 选择一个你关心的领域（例如蛋白质结构、CAD、分子、轨迹）。对该领域当前 SOTA 模型回答五问分诊法，并勾勒一个更好的模型会改变什么。

## 关键术语

| 术语 | 人们通常怎么说 | 它实际意味着什么 |
|------|-----------------|-----------------------|
| 生成模型 | “它能造新东西” | 学习 `p_data(x)` 的采样器，可选地暴露 `log p(x)`。 |
| 显式密度 | “你可以求值” | 模型提供闭式或可 tractable 计算的 `log p(x)`。 |
| 隐式密度 | “GAN 风格” | 只有采样器——无法评估给定点的 `p(x)`。 |
| ELBO | “证据下界” | `log p(x)` 的可计算下界；VAE 和扩散都优化它。 |
| Score | “对数密度的梯度” | `∇_x log p(x)`；扩散和 SDE 模型学习这个场。 |
| 流形假设 | “数据活在一个曲面上” | 高维数据集中在低维流形上；这是降维有效的原因。 |
| 自回归 | “预测下一个片段” | 把联合分布分解为条件分布的乘积。 |
| 潜变量 | “压缩编码” | 解码器可用来重建输入的低维表示。 |

## 生产说明：五大家族，五种推理形态

每个家族都对应不同的推理服务器成本曲线。生产推理文献把 LLM 推理表述为 prefill + decode；同样的分解也适用于这里：

- **自回归（第 1 和第 5 类）。** 顺序解码主导延迟；KV-cache、连续批处理（continuous batching）和投机解码（speculative decoding）都可以直接应用。
- **VAE / 扩散 / 流匹配（第 2 和第 4 类）。** 这里没有 LLM 意义上的 decode。成本 = `num_steps × step_cost`，而 `step_cost` 是在完整潜空间分辨率上做一次 Transformer 或 U-Net 前向传播。生产旋钮是步数（DDIM / DPM-Solver / 蒸馏）、批大小和精度（bf16 / fp8 / int4）。
- **GAN（第 3 类）。** 一次前向传播。没有 schedule，没有 KV-cache。TTFT ≈ 总延迟。这就是为什么 StyleGAN 在窄领域 UX 上仍然胜出的原因。

当你在论文摘要里看到“比扩散更快”时，把它翻译成“更少步数 × 相同步成本”或“相同步数 × 更低步成本”。除此之外都是营销。

## 延伸阅读

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — GAN 论文。
- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — VAE 论文。
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM 论文。
- [Song et al. (2021). Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) — 作为 SDE 的扩散。
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — 流匹配论文。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — Stable Diffusion 3。
