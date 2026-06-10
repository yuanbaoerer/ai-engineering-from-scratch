# 视频-语言模型：时间 Token 与时序定位

> 视频不是照片的堆叠。一段5秒的视频包含因果顺序、动作动词和事件时间信息，这些是图像模型无法表达的。Video-LLaMA（Zhang 等，2023年6月）发布了首个开源视频大语言模型，具备音视频时序定位能力。VideoChat 和 Video-LLaVA 扩展了这一模式。到2025年，Qwen2.5-VL 的 TMRoPE 缩小了与前沿闭源模型的差距。每个系统采用了不同的时间 Token 方案——按片段的 Q-former、按帧的 concat-pool、按 Token 的 TMRoPE。本节课将解读这些模式，构建均匀采样与动态采样帧采样器，并在时序定位任务上进行评估。

**类型：** 构建
**语言：** Python（标准库，帧采样器 + 时序定位评估器）
**前置条件：** Phase 12 · 08（LLaVA-OneVision）
**时间：** 约180分钟

## 学习目标

- 解释为什么时间位置编码能够独立于视觉编码器改变视频 VLM 的性能。
- 比较均匀采样、动态 FPS 采样和事件驱动帧采样在每秒 Token 数与时序定位准确率上的差异。
- 描述按片段的 Q-former（Video-LLaMA）vs 按帧的池化（Video-LLaVA）vs 按 Token 的 M-RoPE（Qwen2.5-VL）三种设计。
- 说出四个视频基准测试：VideoMME、TempCompass、EgoSchema、Video-MMMU。

## 问题背景

一段1分钟的视频，30 FPS，共1800帧。按每帧196个视觉 Token（ViT-B @ 224）计算，总共352k个 Token——超过了任何2024年大语言模型的上下文长度。

三种降维策略：

1. 子采样帧（根据内容选择1-8 FPS）。
2. 对每帧的 patch token 进行激进池化（3x3 或 4x4 双线性池化）。
3. 通过 Q-former 压缩，将16帧片段压缩为64个 Token。

每种策略的权衡不同。子采样丢失时间细节。池化丢失空间细节。Q-former 两者都略有损失，但节省 Token。

时间位置编码是另一个维度：模型如何知道第5帧在第6帧之前？可选方案包括简单的1D时间 RoPE（Video-LLaMA）、可学习的时间嵌入（Video-LLaVA）和 TMRoPE（Qwen2.5-VL，完整3D）。

## 核心概念

### Video-LLaMA：按片段的 Q-former + 音频分支

Video-LLaMA（2023）是首个开源视频大语言模型。架构：

- 16帧片段，2 FPS（即8秒）。
- 每帧 ViT 特征 -> Video Q-former，对所有16帧进行交叉注意力 -> 32个可学习查询 -> 大语言模型。
- 并行音频分支：波形 -> ImageBind 音频编码器 -> Audio Q-former -> 32个查询 -> 大语言模型。

优势：音视频联合推理。劣势：固定片段长度，无法进行任意时间定位。

### VideoChat 和 Video-LLaVA

VideoChat 保留了 Video-LLaMA 的思路，但去掉了音频并做了简化。Video-LLaVA（Lin 等，2023）在图像和视频帧上训练了统一的视觉编码器（"投影前对齐"），获得了统一的表示。两者都是冻结 CLIP 编码器 + MLP + 大语言模型。

两者都无法处理长视频。都是8-16帧的系统。

### Qwen2.5-VL 和 TMRoPE

Qwen2.5-VL 引入了 TMRoPE——Temporal-Modality Rotary Position Embedding（时间-模态旋转位置编码）。每个 patch token 携带 (t, h, w) 位置，其中 t 是实际时间戳（而非帧索引）。

与简单时间嵌入的关键区别：

- 绝对时间，而非索引。模型看到的是"在4.2秒"而非"在第15帧"。
- 按 Token 旋转，而非按片段。每个视觉 Token 根据其时间戳独立旋转。
- 兼容动态 FPS。如果此处采样2 FPS、彼处采样4 FPS，TMRoPE 原生处理不均匀间隔。

TMRoPE 支持"猫在什么时候跳？"这类查询。模型可以输出"在4.2秒"。Video-LLaMA 只能说"在片段早期"。

### 帧采样策略

均匀采样：在视频时长内均匀采样 N 帧。简单，丢失运动峰值。

动态 FPS：根据运动强度自适应采样。光流或帧差分在高运动段选择更密集采样。Qwen2.5-VL 采用此策略训练。

事件驱动：运行轻量级检测器，在动作发生处采样更多。VideoAgent 使用此策略。

关键帧 + 上下文：在镜头边界处采样 + 少量相邻帧。用于电影内容。

### 每帧池化

1 FPS、每帧576个 Token 的情况下，5分钟片段共172,800个 Token。Qwen2.5-VL-72B 的128k上下文可以处理，但代价高昂。

3x3 双线性池化减少到每帧64个 Token -> 5分钟共19,200个 Token。大多数任务的甜蜜点。

对于空间细节不那么重要的 agent 工作流，可以更激进地池化（6x6 -> 每帧16个 Token）。

### 四个视频基准测试

- VideoMME：全面的视频理解，短 + 中 + 长视频。
- TempCompass：细粒度时间推理，"之前"/"之后"问题。
- EgoSchema：长程第一人称视频。
- Video-MMMU：多模态多学科视频问题。

完整的视频 VLM 评估需要覆盖全部四个。它们侧重不同维度——TempCompass 关注顺序，EgoSchema 关注3分钟以上推理，VideoMME 覆盖不同时长。

### 时序定位输出格式

时序定位的输出格式：

- 自由文本："猫大约在第4秒跳起。"易于解析但不精确。
- 结构化 JSON：`{"event": "jump", "start": 4.1, "end": 4.3}`。Qwen2.5-VL 训练此格式。
- 基于 Token：特殊 `<time>4.1</time>` Token 与答案交错。Qwen2.5-VL 的内部格式。

基于 Token 的格式对下游使用最精确。Qwen2.5-VL 的 JSON 输出格式可直接解析。

### 2026年最佳实践

2026年视频 VLM 的最佳实践：

- 编码器：SigLIP 2 配合 M-RoPE 或 TMRoPE（Qwen2.5-VL）。
- 帧采样：动态 FPS（根据运动选择1-4 FPS），带最大帧数上限。
- 每帧池化：3x3 双线性。
- 输出：带时间和事件字段的结构化 JSON。
- 基准测试：VideoMME + TempCompass 用于通用评估；EgoSchema 用于长程评估。

## 动手实践

`code/main.py` 包含：

- 均匀采样和动态 FPS 帧采样器。
- 一个简易时序定位评估器：给定时间 T 的"真值"事件和模型输出，在容差范围内评分准确率。
- Video-LLaMA（16帧，Q-former）、Video-LLaVA（8帧，MLP）、Qwen2.5-VL（动态 FPS + TMRoPE）的对比。

## 交付成果

本节课产出 `outputs/skill-video-vlm-frame-planner.md`。给定一个视频任务（监控、动作识别、时序定位、摘要），它会选择帧采样器、池化因子、输出格式和预期准确率等级。

## 练习题

1. 对于一段3分钟的烹饪演示，选择均匀采样还是动态 FPS。用 Token 数量论证。

2. TMRoPE 相比简单的时间嵌入表，具体增加了什么能力？

3. 写一个 VLM 可以学习输出的时序定位 JSON schema。包含错误情况。

4. 阅读 Video-LLaVA 第3节"Alignment Before Projection"。为什么这比训练独立的图像和视频编码器更好？

5. 根据 VideoMME 排行榜，截至2026年，顶级开源模型与顶级闭源模型的差距是多少？其中多少差距可归因于时间编码 vs 基础大语言模型规模？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Temporal grounding | "时间定位的答案" | VLM 输出事件发生的具体时间戳范围 |
| TMRoPE | "Time-Multimodal RoPE" | 带绝对时间戳的3D旋转位置编码，Qwen2.5-VL 使用 |
| Dynamic FPS | "运动感知采样" | 在高运动段采样更多帧，在静态段采样更少 |
| Frame pooling | "每帧空间压缩" | 在进入大语言模型前用双线性插值减少每帧的 patch 数 |
| Video Q-former | "片段压缩器" | 交叉注意力瓶颈，将 N 帧映射到 K 个可学习查询 |
| VideoMME | "视频基准" | 全面的短/中/长视频基准测试，2500+ 样本 |

## 延伸阅读

- [Zhang 等 — Video-LLaMA (arXiv:2306.02858)](https://arxiv.org/abs/2306.02858)
- [Li 等 — VideoChat (arXiv:2305.06355)](https://arxiv.org/abs/2305.06355)
- [Lin 等 — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Qwen Team — Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Lin 等 — VILA-1.5 (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
