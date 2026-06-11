# 视频生成

> 图像是一个二维张量。视频则是三维张量。理论相同；计算难度却高出 10-100 倍。OpenAI 的 Sora（2024 年 2 月）证明了这件事可行。到 2026 年，Veo 2、Kling 1.5、Runway Gen-3、Pika 2.0 和 WAN 2.2 已经能从文本生成 1080p 生产级视频，而开源权重栈（CogVideoX、HunyuanVideo、Mochi-1、WAN 2.2）大约落后 12 个月。

**类型：** 构建
**语言：** Python
**先修要求：** 第 8 阶段 · 07（潜空间扩散，Latent Diffusion）、第 7 阶段 · 09（ViT）、第 8 阶段 · 06（DDPM）
**时间：** 约 45 分钟

## 问题

一个 10 秒、1080p、24fps 的视频包含 240 帧，每帧 1920×1080×3 像素。每个片段约有 1.5 GB 原始数据。在像素空间做扩散不可行。你需要：

1. **时空压缩。** 一个把视频而不是单帧编码为时空补丁序列的 VAE。
2. **时间一致性。** 多秒内的帧需要共享内容、光照和对象身份。网络必须对运动建模。
3. **计算预算。** 在相同模型规模下，视频训练比图像贵 10-100 倍。
4. **条件控制。** 文本、图像（首帧）、音频或另一个视频。大多数生产模型都接受这四类条件。

解决这个问题的架构是把 **扩散 Transformer（Diffusion Transformer，DiT）** 应用于时空补丁，并在海量（提示词、字幕、视频）数据集上训练。扩散损失与第 06 课相同。

## 核心概念

![视频扩散：切分补丁、DiT、解码](../assets/video-generation.svg)

### Patchify

用 3D VAE（学习到的时空压缩）编码视频。潜变量形状为 `[T_latent, H_latent, W_latent, C_latent]`。把它切分成大小为 `[t_p, h_p, w_p]` 的补丁。对 Sora 风格模型，`t_p = 1`（逐帧补丁）或 `t_p = 2`（每两帧一个补丁）。一个 10 秒 1080p 视频会压缩成约 20,000-100,000 个补丁。

### 时空 DiT

Transformer 处理扁平化后的补丁序列。每个补丁都有 3D 位置嵌入（时间 + y + x）。注意力通常被因式分解：

- **空间注意力** 在每一帧的补丁内部计算。
- **时间注意力** 在相同空间位置的不同帧之间计算。
- **完整 3D 注意力** 昂贵 16-100 倍；只在低分辨率或研究场景中使用。

### 文本条件

通过大型文本编码器做交叉注意力（Sora 使用 T5-XXL，CogVideoX-5B 也使用 T5-XXL）。长提示词很重要——Sora 的训练集使用 GPT 生成的密集重标注字幕，平均每个片段 200 个 token。

### 训练

在时空潜变量上使用标准扩散损失（ε 或 v 预测）。数据：网络视频 + 约 1 亿个精选片段 + 合成文本字幕。计算量：即使是小型研究实验也需要 10,000+ GPU 小时；Sora 规模则是 100,000+。

## 2026 年生产格局

| 模型 | 日期 | 最长时长 | 最高分辨率 | 开源权重？ | 亮点 |
|-------|------|--------------|---------|---------------|---------|
| Sora (OpenAI) | 2024-02 | 60s | 1080p | 否 | 首个在规模化条件下展现世界模拟器属性的模型 |
| Sora Turbo | 2024-12 | 20s | 1080p | 否 | 推理速度快 5 倍的生产版 Sora |
| Veo 2 (Google) | 2024-12 | 8s | 4K | 否 | 2025 年最高质量 + 最佳物理表现 |
| Veo 3 | 2025 Q3 | 15s | 4K | 否 | 原生音频和更强的镜头控制 |
| Kling 1.5 / 2.1 (Kuaishou) | 2024-2025 | 10s | 1080p | 否 | 2025 年 Q1 最佳人体运动 |
| Runway Gen-3 Alpha | 2024-06 | 10s | 768p | 否 | 上层配套专业视频工具 |
| Pika 2.0 | 2024-10 | 5s | 1080p | 否 | 最强角色一致性 |
| CogVideoX (THUDM) | 2024 | 10s | 720p | 是（2B、5B） | 首个开放的 5B 规模视频模型 |
| HunyuanVideo (Tencent) | 2024-12 | 5s | 720p | 是（13B） | 2024 年末开源 SOTA |
| Mochi-1 (Genmo) | 2024-10 | 5.4s | 480p | 是（10B） | 许可证最宽松 |
| WAN 2.2 (Alibaba) | 2025-07 | 5s | 720p | 是 | 2025 年中最强开源模型 |

开源权重正在比图像领域更快地缩小差距：到 2026 年中，HunyuanVideo + WAN 2.2 LoRA 已经支撑了大多数开源工作流。

## 动手构建

`code/main.py` 模拟核心时空 DiT 思路：把一个小型合成视频切成补丁，添加逐补丁位置嵌入，并用类似 Transformer 的补丁注意力对整个序列去噪。不使用 numpy；纯 Python。我们会展示：即使在一维中，当相邻帧补丁共享去噪器和位置嵌入时，也会出现时间一致性。

### 第 1 步：把合成一维“视频”切成补丁

```python
def make_video(T_frames=8, rng=None):
    # a "video" is a sequence of 1-D values following a smooth trajectory
    base = rng.gauss(0, 1)
    return [base + 0.3 * t + rng.gauss(0, 0.1) for t in range(T_frames)]
```

### 第 2 步：每帧的位置嵌入

```python
def pos_embed(t, dim):
    return sinusoidal(t, dim)
```

### 第 3 步：去噪器看到整个序列

我们的微型网络不是独立地对每一帧去噪，而是拼接所有帧值及其位置嵌入，并联合预测所有帧的噪声。

### 第 4 步：时间一致性测试

训练后采样一个视频。测量逐帧差值。如果模型学到了时间结构，这些差值会小于独立采样每一帧时的差值。

## 常见陷阱

- **逐帧独立采样 = 闪烁。** 如果你对每一帧分别运行图像扩散，输出会闪烁，因为每帧噪声都是独立的。视频扩散通过注意力或共享噪声耦合帧来修复这一点。
- **朴素 3D 注意力 = 显存爆炸。** 对 10 秒 1080p 潜变量做完整 3D 注意力需要数千亿次操作。应因式分解为空间 + 时间。
- **数据标注比数据规模更重要。** Sora 相比早期工作的主要升级，是使用了详细约 10 倍的字幕（GPT-4 重新标注片段）。OpenAI 技术报告对此说得很明确。
- **首帧条件。** 大多数生产模型也接受一张图像作为首帧。这是“图生视频”（image-to-video）模式；训练中也包含这种变体。
- **物理漂移。** 长片段（>10s）会累积细微不一致。滑动窗口生成 + 关键帧锚定会有帮助。

## 使用它

| 用例 | 2026 年选择 |
|----------|-----------|
| 最高质量托管文本生成视频 | Veo 3 或 Sora |
| 可控镜头的电影感视频 | 带 motion brushes 的 Runway Gen-3 |
| 跨片段角色一致性 | Pika 2.0 或 Kling 2.1 |
| 开源权重、快速微调 | WAN 2.2 + LoRA |
| 图生视频 | WAN 2.2-I2V、Kling 2.1 I2V 或 Runway |
| 音频到视频唇形同步 | Veo 3（原生音频）或专用唇形同步模型 |
| 视频编辑 | Runway Act-Two、Kling Motion Brush、Flux-Kontext（静帧） |

在质量相当的情况下，视频每秒成本从 2024 到 2026 年下降了 20 倍。

## 交付它

保存 `outputs/skill-video-brief.md`。该技能接收一份视频简报（时长、宽高比、风格、镜头计划、主体一致性、音频），并输出：模型 + 托管方案、提示词脚手架（镜头语言、主体描述、运动描述符）、种子 + 可复现协议，以及帧级 QA 检查清单。

## 练习

1. **简单。** 在 `code/main.py` 中，比较 (a) 逐帧独立采样、(b) 联合序列采样 的逐帧差值。报告差值的均值和方差。
2. **中等。** 添加首帧条件：把第 0 帧固定为给定值并采样其余帧。测量固定值如何传播。
3. **困难。** 使用 HuggingFace diffusers 在本地 GPU 上运行 CogVideoX-2B。对一个 6 秒、720p 片段计时 20 个推理步。分析时空注意力以找出瓶颈。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Video VAE | “3-D VAE” | 把 `(T, H, W, C)` 压缩为时空潜变量的编码器。 |
| Patches | “这些 token” | 潜变量中的固定大小 3D 块；输入给 DiT。 |
| Factorized attention | “空间 + 时间” | 先在空间上做注意力，再在时间上做注意力；跳过完整 3D 注意力。 |
| Image-to-video (I2V) | “让这张照片动起来” | 模型接收图像 + 文本，输出从该图像开始的视频。 |
| Keyframe conditioning | “锚定帧” | 固定特定帧来控制视频走向。 |
| Motion brush | “方向提示” | 用户在图像上绘制运动向量的 UI 输入。 |
| Re-captioning | “密集字幕” | 使用 LLM 用详细提示重新标注训练片段。 |
| Flicker | “时间伪影” | 帧间不一致；通过耦合去噪修复。 |

## 生产备注：视频潜变量是内存带宽问题

一个 10 秒、1080p、24 fps 的片段是 240 帧 × 1920 × 1080 × 3 ≈ 1.5 GB 原始像素。经过 4× 视频 VAE 压缩（`2 × 空间 × 2 × 时间`）后，每个请求的潜变量约为 100 MB。把它在 batch 1 下通过时空 DiT 跑 30 步，你每步都要在 HBM 中搬运约 3 GB 数据——瓶颈是内存带宽，而不是 FLOPs。

三个生产调节旋钮，都直接来自生产推理文献中的推理章节：

- **跨 DiT 的 TP。** 文本生成视频模型通常 ≥10B 参数。在 4 张 H100 上做 TP=4 是标准做法；405B 级模型使用 PP=2 × TP=2。每步延迟随 TP 大致线性下降，直到触及 all-reduce 墙。
- **帧批处理 = 连续批处理。** 生成时，视频在概念上是一批由注意力连接的帧。连续批处理（in-flight scheduling）适用：如果模型架构允许滑动窗口生成，可以在返回帧 `t-1` 的同时开始渲染帧 `t+1`。
- **片段级预填充缓存。** 对图生视频来说，首帧条件类似 LLM 的 prompt prefill：计算一次，在时间解码器多次传递中复用。这本质上是视频的 KV-cache。

## 延伸阅读

- [Brooks et al. (2024). Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) — Sora 技术报告。
- [Yang et al. (2024). CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072) — CogVideoX。
- [Kong et al. (2024). HunyuanVideo: A Systematic Framework for Large Video Generative Models](https://arxiv.org/abs/2412.03603) — HunyuanVideo。
- [Genmo (2024). Mochi-1 Technical Report](https://www.genmo.ai/blog/mochi) — Mochi-1。
- [Alibaba (2025). WAN 2.2](https://wanvideo.io/) — 2025 年中开源 SOTA。
- [Ho, Salimans, Gritsenko et al. (2022). Video Diffusion Models](https://arxiv.org/abs/2204.03458) — 开创性视频扩散论文。
- [Blattmann et al. (2023). Align your Latents (Video LDM)](https://arxiv.org/abs/2304.08818) — Stable Video Diffusion 的前身。
