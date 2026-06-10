# Emu3: 用于图像和视频生成的下一 Token 预测

> BAAI 的 Emu3（Wang 等人，2024 年 9 月）是 2024 年本应终结扩散模型与自回归模型之争的成果。一个单一的 Llama 风格仅解码器 Transformer，仅通过下一 Token 预测目标进行训练，在一个统一的词表中处理文本 + VQ 图像 Token + 3D VQ 视频 Token，在图像生成上击败了 SDXL，在感知任务上击败了 LLaVA-1.6。没有 CLIP 损失。没有扩散调度。推理时使用 classifier-free guidance（CFG）来提升质量，但核心训练目标是带有 teacher forcing 的下一 Token 预测。发表于 Nature。本节课解读 Emu3 的核心论点——更好的 tokenizer 加上规模就是你所需要的一切——并与扩散方法进行对比。

**类型：** 学习
**语言：** Python（标准库，3D 视频 tokenizer 数学 + 自回归采样器骨架）
**前置知识：** Phase 12 · 11（Chameleon）
**时间：** 约 120 分钟

## 学习目标

- 解释为什么 Emu3 的单损失下一 Token 目标能够奏效，尽管长期以来人们认为图像质量需要扩散模型。
- 描述 3D 视频 tokenizer：时空 VQ 码本是什么样的，为什么 patch 要跨越时间维度。
- 比较 Emu3 与 Stable Diffusion XL（训练计算量、推理成本、质量上限）。
- 说出同一个 Emu3 模型扮演的三个角色：Emu3-Gen（图像生成）、Emu3-Chat（感知）、Emu3-Stage2（视频生成）。

## 问题背景

2024 年之前的传统观念：图像生成需要扩散模型。论据是：离散图像 Token 会丢失太多信息，无法重建细节，而且自回归采样会在数千个 Token 上累积误差。Stable Diffusion、DALL-E 3、Imagen、Midjourney 都使用某种形式的扩散。Chameleon（第 12.11 课）在小规模上部分反驳了这一点，但在质量上未能匹敌 SDXL。

Emu3 正面挑战了这一论点。其主张是：更好的视觉 tokenizer + 足够的规模 + 下一 Token 损失 = 在同一个模型中击败扩散模型的图像生成能力，同时该模型还能做感知任务。

这一赌注在发表时颇具争议。两年后，开源统一生成家族（Emu3、Show-o、Janus-Pro、Transfusion）已成为研究的默认路径；生产级前沿模型似乎也在使用某种变体。

## 核心概念

### Emu3 的 tokenizer

关键要素是视觉 tokenizer。Emu3 训练了一个自定义的 IBQ 类 tokenizer（Inverse Bottleneck Quantizer，SBER-MoVQGAN 家族），每个 Token 实现 8x8 的分辨率缩减。一张 512x512 的图像变为 64x64 = 4096 个 Token，码本大小为 32768。

这比 Chameleon 的每 512x512 图像 1024 个 Token（K=8192）更多，但每个 Token 更便宜（更小的码本查找、更简单的编解码器）。关键指标：重建 PSNR 达到 30.5 dB，与 Stable Diffusion 的连续隐空间 32 dB 相当。

对于视频：一个 3D VQ tokenizer 将一个时空 patch（4x4x4 像素）编码为一个整数。一段 4 秒、8 FPS 的片段有 32 帧；在 256x256 分辨率下，空间缩减 4 倍、时间缩减 4 倍，Token 数量为 (256/4) * (256/4) * (32/4) = 64 * 64 * 8 = 32,768 个 Token。

Tokenizer 质量是上限。Emu3 的贡献部分在于"我们训练了一个非常好的 tokenizer"。

### 单损失训练

Emu3 使用一个目标：在共享词表上对文本 Token、2D 图像 Token 和 3D 视频 Token 进行下一 Token 预测。训练期间权重会乘以模态特定的因子以平衡贡献，但损失函数是相同的。

训练数据混合包括：
- 图像生成：`<text caption> <image> image_tokens </image>`
- 图像感知：`<image> image_tokens </image> <question> text_tokens`
- 视频生成：`<text caption> <video> video_tokens </video>`
- 视频感知：类似上述格式。
- 纯文本：标准 NTP。

模型从数据分布中学习何时生成图像 Token 或文本 Token。生成能力来自于模型在 `<image>` 标签后预测图像 Token。

### Classifier-free guidance 与 temperature

自回归图像生成在推理时使用 classifier-free guidance（CFG）会获得显著提升。Emu3 使用了它：生成两次，一次使用完整 caption，一次使用空 caption，用 guidance weight（典型值 3.0-7.0）混合 logits。这与扩散模型使用的 CFG 技巧相同，只是借用到自回归设置中。

Temperature 很重要：太高会产生伪影；太低会导致模式崩溃。Emu3 推荐的 temperature 为感知任务 1.0，图像生成 0.8。

### 三种角色，一个模型

Emu3 以三个功能不同的 API 发布，但底层是同一套权重：

- Emu3-Gen。图像生成。输入文本，输出图像 Token。
- Emu3-Chat。VQA 和图像描述。输入图像（Token），输出文本。
- Emu3-Stage2。视频生成和视频 VQA。输入文本或视频，输出文本或视频。

没有任务特定的头。只是不同的 prompt 模板。同一个 checkpoint。

### 基准测试

来自 Emu3 论文（2024 年 9 月）：

- 图像生成：在 MJHQ-30K FID 上击败 SDXL（5.4 vs 5.6），GenEval 总体得分（0.54 vs 0.55——统计平局），Deep-Eval 的综合得分也大致持平。
- 图像感知：在 VQAv2 上击败 LLaVA-1.6（75.1 vs 72.4），在 MMMU 上大致持平。
- 视频生成：4 秒片段质量在 FVD 上与 Sora 时代的公开基准模型具有竞争力。

这些数字并非总是领先——Emu3 在这里让一分，在那里得一分——但"下一 Token 预测就是你所需要的一切"这一主张在跨模态上是可辩护的。

### 计算成本

Emu3 使用一个 7B 参数的模型，在大约 3000 亿个多模态 Token 上训练。GPU 小时数大致与 Llama-2-7B 的预训练相当（在 A100 级芯片上 2000-4000 GPU 年）。像 Stable Diffusion 3 这样的扩散模型在类似的预算下训练，但需要单独的文本编码器和更复杂的流水线。

在推理时，Emu3 比 SDXL 慢：4096 个图像 Token 以 30 tok/s 的速度生成，每张 512x512 图像约需 2 分钟，而 SDXL 只需 2-5 秒。推测解码（speculative decoding）和 KV-cache 优化可以缩小差距，但无法完全弥补。自回归图像生成计算量大；这是固有的权衡。

### 为什么重要

Emu3 的深层贡献是概念性的。如果下一 Token 预测可以扩展到匹敌扩散模型的图像生成能力，那么统一模型路径（一个损失、一个骨干网络、任意模态）就是可行的。未来的模型不需要单独的文本编码器、单独的扩散调度器、单独的 VAE。一个 Transformer，每个模态一个 tokenizer，然后扩展规模。

Show-o、Janus-Pro 和 InternVL-U 都建立在这一论点之上或对其提出挑战。中国实验室（BAAI、DeepSeek）在 2025 年之前比美国实验室更积极地朝这个方向发表成果。

## 动手实践

`code/main.py` 构建了两个玩具示例：

- 一个 2D 与 3D VQ tokenizer 数量计算器：给定（分辨率、patch 大小、片段长度、FPS），计算图像与视频的 Token 数量。
- 一个带有 classifier-free guidance 和 temperature 的自回归图像 Token 采样器。

CFG 实现遵循 Emu3 的配方——用 guidance weight 混合条件和非条件 logits。

## 产出交付

本节课产出 `outputs/skill-token-gen-cost-analyzer.md`。给定一个生成产品规格（图像或视频、目标分辨率、质量等级、延迟预算），它计算 Token 数量、推理成本，并在 Emu3 家族与扩散模型之间做出选择。

## 练习题

1. Emu3 在 8x8 缩减下每张 512x512 图像生成 4096 个 Token。计算 1024x1024 和 2048x2048 的等效值。推理延迟会发生什么变化？

2. 阅读 Emu3 论文第 3.3 节关于视频 tokenizer 的内容。描述 3D VQ patch 的形状，以及为什么它是 4x4x4 而不是 8x8x1。

3. Classifier-free guidance weight 5.0 与 3.0：有什么视觉效果差异？在 `code/main.py` 中追踪数学原理。

4. 计算 Emu3-7B 在 300B Token 上的训练 FLOPs，并与 Stable Diffusion 3 比较。哪个训练成本更高？

5. Emu3 在 FID 上击败 SDXL，但在 VQAv2 上不如专门的 VLM。解释为什么统一损失方法在不同基准上相对于专家模型表现出不同的优势。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Next-token prediction | "NTP" | 标准自回归损失：给定 token[0..i] 预测 token[i+1]；当数据被 tokenized 后适用于任何模态 |
| IBQ tokenizer | "Inverse bottleneck quantizer" | 一类 VQ-VAE，具有更大的码本（32768+）和比 Chameleon 更好的重建质量 |
| 3D VQ | "Spatiotemporal quantizer" | 由（时间、行、列）索引的码本；一个 Token 覆盖一个 4x4x4 像素立方体 |
| Classifier-free guidance | "CFG" | 用权重 gamma 混合条件和非条件 logits；在推理时提升图像质量 |
| Unified vocabulary | "Shared tokens" | 文本 + 图像 + 视频都使用同一个整数空间；模型预测下一个出现的任意模态 |
| MJHQ-30K | "Image gen benchmark" | 具有 30k 个 prompt 的 Midjourney 质量基准；Emu3 在此报告 FID |

## 延伸阅读

- [Wang et al. — Emu3: Next-Token Prediction is All You Need (arXiv:2409.18869)](https://arxiv.org/abs/2409.18869)
- [Sun et al. — Emu: Generative Pretraining in Multimodality (arXiv:2307.05222)](https://arxiv.org/abs/2307.05222)
- [Liu et al. — LWM (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Yu et al. — MAGVIT-v2 (arXiv:2310.05737)](https://arxiv.org/abs/2310.05737)
- [Tian et al. — VAR (arXiv:2404.02905)](https://arxiv.org/abs/2404.02905)
