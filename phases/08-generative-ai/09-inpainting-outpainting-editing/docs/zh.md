# 修复、扩图与图像编辑

> 文生图创造新事物。修复（inpainting）修补旧事物。在生产中，70% 可计费的图像工作都是编辑——替换背景、移除 logo、扩展画布、重新生成一只手。修复正是扩散模型真正体现价值的地方。

**类型：** 构建
**语言：** Python
**先修要求：** 第 8 阶段 · 07（潜在扩散）、第 8 阶段 · 08（ControlNet 与 LoRA）
**时间：** 约 75 分钟

## 问题

客户发来一张完美的产品照，但背景里有一块分散注意力的招牌。你想擦掉招牌，同时让其他所有像素保持完全一致。你不能从零运行文生图——结果会有不同颜色、不同光照、不同产品角度。你只想重新生成*被遮罩的区域*，并且希望重新生成的内容尊重周围上下文。

这就是 inpainting。它的变体包括：

- **Inpainting（修复）。** 在遮罩内部重新生成，保留外部像素。
- **Outpainting（扩图）。** 在遮罩外部（或画布之外）重新生成，保留内部。
- **Image editing（图像编辑）。** 重新生成整张图，但保留原图的语义或结构保真度（SDEdit、InstructPix2Pix）。

2026 年，每个扩散 pipeline 都会提供 inpainting 模式。Flux.1-Fill、Stable Diffusion Inpaint、SDXL-Inpaint、DALL-E 3 Edit。它们的工作原理相同。

## 概念

![Inpainting：带遮罩感知的去噪与上下文保持式回注入](../assets/inpainting.svg)

### 朴素方法（以及为什么它错）

带着遮罩运行标准文生图。在每个采样步骤，把未遮罩区域的噪声潜变量替换为干净图像经过正向扩散后的版本。它能用……但效果很差。边界伪影会渗出来，因为模型并不知道遮罩区域里应该有什么。

### 正确的 inpainting 模型

训练一个修改过的 U-Net，让它接受 9 个输入通道而不是 4 个：

```
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

额外通道是一份 VAE 编码后的源图像，以及一个单通道遮罩。训练时，你随机遮住图像中的区域，并训练模型只对遮罩区域去噪，同时把未遮罩区域作为干净的条件信号提供给模型。推理时，模型可以“看见”遮罩周围是什么，并生成连贯的补全内容。

SD-Inpaint、SDXL-Inpaint、Flux-Fill 都使用这种 9 通道（或类似）输入。Diffusers 中对应 `StableDiffusionInpaintPipeline`、`FluxFillPipeline`。

### SDEdit（Meng 等人，2022）——免费编辑

给源图像加噪到某个中间时间 `t`，然后用新提示词从 `t` 反向运行到 0。不需要重新训练。起始 `t` 的选择在保真度与创作自由之间做权衡：

- `t/T = 0.3` → 与源图几乎相同，仅有小的风格变化
- `t/T = 0.6` → 中等编辑，保留粗略结构
- `t/T = 0.9` → 几乎从噪声生成，源图保留很少

### InstructPix2Pix（Brooks 等人，2023）

在 `(input_image, instruction, output_image)` 三元组上微调扩散模型。推理时，同时以输入图像和文本指令（“让它变成日落”、“添加一条龙”）为条件。它有两个 CFG scale：图像 scale 和文本 scale。

### RePaint（Lugmayr 等人，2022）

保留一个标准无条件扩散模型。在每个反向步骤中，进行重采样——偶尔跳回到更有噪声的状态并重新生成。这样可以避免边界伪影。适用于你没有训练好的 inpainting 模型时。

## 构建它

`code/main.py` 在 5 维数据上实现一个玩具版 1-D inpainting 方案。我们在 5-D 混合数据上训练一个 DDPM，每个样本是来自两个簇之一的 5 个浮点数。推理时，我们“遮罩”5 个维度中的 2 个，在每一步注入未遮罩 3 个维度的正向加噪版本，并只重新生成被遮罩维度。

### 步骤 1：5-D DDPM 数据

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### 步骤 2：在全部 5 个维度上训练去噪器

标准 DDPM。网络对 5-D 噪声输入输出 5-D 噪声预测。

### 步骤 3：推理时，遮罩感知反向过程

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # 用干净源图的重新加噪版本替换未遮罩维度
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...然后在 x_t 上运行正常反向步骤
```

这是朴素方法，但它在玩具 1-D 数据上有效。真实图像 inpainting 使用 9 通道输入，因为纹理连贯性更重要。

### 步骤 4：outpainting

Outpainting 是遮罩反转后的 inpainting：遮住新的（原本不存在的）画布区域，用原图填充其余部分。训练目标完全相同。

## 陷阱

- **接缝。** 朴素方法会留下可见边界，因为梯度信息不会跨遮罩流动。修复：把遮罩膨胀 8-16 像素，或使用真正的 inpainting 模型。
- **遮罩泄漏。** 如果条件图像的未遮罩区域质量低或有噪声，它会污染遮罩内的生成。先去噪或轻微模糊。
- **CFG 与遮罩大小相互作用。** 小遮罩 + 高 CFG = 饱和补丁。小编辑应降低 CFG。
- **SDEdit 保真悬崖。** 从 `t/T = 0.5` 增到 `t/T = 0.6`，可能会丢失主体身份。要做 sweep 并保存 checkpoint。
- **提示词不匹配。** 提示词应该描述*整张*图，而不只是新内容。用“一只猫坐在椅子上”，不要只写“一只猫”。

## 使用它

| 任务 | Pipeline |
|------|----------|
| 移除物体，小遮罩 | SD-Inpaint 或 Flux-Fill，标准提示词 |
| 替换天空 | SD-Inpaint + “blue sky at sunset” |
| 扩展画布 | SDXL outpaint 模式（8px feather）或带 outpaint mask 的 Flux-Fill |
| 重新生成手 / 脸 | SD-Inpaint，并用提示词重新描述主体 + ControlNet-Openpose |
| 改变某个区域的风格 | 在遮罩区域上用 `t/T=0.5` 的 SDEdit |
| “让它变成日落” | InstructPix2Pix 或 Flux-Kontext |
| 背景替换 | SAM mask → SD-Inpaint |
| 超高保真 | 对最难案例使用 Flux-Fill 或 GPT-Image（托管） |

SAM（Meta 的 Segment Anything，2023）+ diffusion inpaint 是 2026 年的背景移除 pipeline。SAM 2（2024）可用于视频。

## 交付它

保存 `outputs/skill-editing-pipeline.md`。该技能接收原始图像 + 编辑描述 + 可选遮罩（或 SAM 提示），并输出：遮罩生成方法、基座模型、CFG scales（图像 + 文本）、SDEdit-t 或 inpainting 模式，以及 QA 检查清单。

## 练习

1. **简单。** 在 `code/main.py` 中，把被遮罩维度比例从 0.2 改到 0.8。在什么比例下，inpaint 质量（遮罩维度中的残差）等于无条件生成？
2. **中等。** 实现 RePaint：每 10 个反向步骤，跳回 5 步（加噪）并重新去噪。测量它是否降低遮罩边缘的边界残差。
3. **困难。** 使用 Hugging Face diffusers 比较：SD 1.5 Inpaint + ControlNet-Openpose 与 Flux.1-Fill，在 20 个面部重生成任务上测试。分别给姿态遵循度和身份保持度打分。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Inpainting | “填洞” | 在遮罩内部重新生成；保留外部像素。 |
| Outpainting | “扩展画布” | 在画布外部重新生成；保留内部。 |
| 9-channel U-Net | “真正的 inpainting 模型” | 输入为 `noisy \| encoded-source \| mask` 的 U-Net。 |
| SDEdit | “带噪声级别的 img2img” | 加噪到时间 `t`，再用新提示词去噪。 |
| InstructPix2Pix | “纯文本编辑” | 在（图像、指令、输出）三元组上微调的扩散模型。 |
| RePaint | “无需重训” | 反向过程中周期性重新加噪，以减少接缝。 |
| SAM | “Segment Anything” | 通过点击或框生成遮罩；与 inpaint 搭配使用。 |
| Flux-Kontext | “带上下文编辑” | 接受参考图像 + 指令来做编辑的 Flux 变体。 |

## 生产注记：编辑 pipeline 对延迟敏感

用户编辑图像时，期望 5 秒以内的往返。在 L4 上，1024² 的 30 步 SDXL-Inpaint 需要 3-4 秒，再加上 SAM 遮罩生成（约 200 ms）和 VAE 编码/解码（合计约 500 ms）。从生产视角看，这更受 TTFT 限制，而不是吞吐量限制——batch 1、低并发、每个阶段都要最小化：

- **SAM-H 是慢的那个。** 1024² 下 SAM-H 约 200 ms；SAM-ViT-B 约 40 ms，质量损失很小。SAM 2（视频）会增加时间维度开销；不要用于单图编辑。
- **能跳过编码就跳过。** `pipe.image_processor.preprocess(img)` 会编码到潜变量。如果你已经有上一次生成的潜变量（迭代式编辑 UI 中很常见），直接通过 `latents=...` 传入，跳过一次 VAE 编码。
- **遮罩膨胀也影响吞吐。** 小遮罩意味着 U-Net 前向的大部分计算被浪费了（未遮罩像素反正会被钳住）。`diffusers` 的 `StableDiffusionInpaintPipeline` 无论如何都会运行完整 U-Net；只有 9 通道的真正 inpaint 变体才能利用遮罩计算。
- **Flux-Kontext 是 2025 年的答案。** 对 `(source_image, instruction)` 做一次前向传播——没有单独遮罩，没有 SDEdit 噪声 sweep。在 H100 上，它约 1.5 秒就能交付一次编辑。架构层面的经验是：合并阶段。

## 延伸阅读

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) — 无训练 inpainting。
- [Meng et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — SDEdit。
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) — 文本指令编辑。
- [Kirillov et al. (2023). Segment Anything](https://arxiv.org/abs/2304.02643) — SAM，遮罩来源。
- [Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) — 视频 SAM。
- [Hertz et al. (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) — 注意力层级编辑。
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) — 2024 年工具。
