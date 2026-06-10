# Qwen-VL 系列与 Dynamic-FPS 视频

> Qwen-VL 系列 —— Qwen-VL (2023)、Qwen2-VL (2024)、Qwen2.5-VL (2025)、Qwen3-VL (2025) —— 是 2026 年最具影响力的开源视觉语言模型家族。每一代都做出了一个决定性的架构赌注，而开源生态系统的其余部分在十二个月内纷纷效仿：通过 M-RoPE 实现原生动态分辨率、dynamic-FPS 采样配合绝对时间对齐、ViT 中的窗口注意力，以及结构化智能体输出格式。到了 Qwen3-VL，这一配方已经稳定下来：一个支持原生宽高比输入的 2D-RoPE-ViT 编码器、一个将特征投影到大型 Qwen3 语言基座模型的 MLP projector，以及将 OCR、grounding 和智能体行为作为一等目标的训练阶段。本课程按时间顺序阅读这个家族，以便你理解每一个旋钮为何设置在此处。

**类型：** 学习
**语言：** Python（标准库，M-RoPE 编码器 + dynamic-FPS 采样器）
**前置条件：** Phase 12 · 06 (patch-n'-pack)
**时间：** ~120 分钟

## 学习目标

- 计算 M-RoPE 的三轴旋转（时间、高度、宽度），并解释为何三者都是必需的。
- 为视频选择一种 dynamic-FPS 采样策略，并推理每秒 token 数与事件检测准确性之间的权衡。
- 按顺序说出 Qwen-VL 四代的升级内容，以及每一代实现了什么。
- 编写 Qwen2.5-VL 风格的 JSON 智能体输出格式，并从 VLM 响应中解析结构化工具调用。

## 问题背景

Qwen-VL 于 2023 年 8 月发布，直接回应了 LLaVA-1.5 和 BLIP-2。Qwen 团队瞄准的差距有三方面：分辨率、视频和结构化输出。

分辨率：LLaVA-1.5 运行在 336x336。对于照片尚可，但对于中文发票或密集的电子表格截图则毫无用处。Qwen-VL 的首个创新是 448x448 分辨率以及 grounded bounding-box 输出，让模型能够指向物体。

视频：Video-LLaMA 堆叠了逐帧编码器并将它们输入 LLM。对于短视频片段有效，但对于时间轴才是信号的多分钟视频则不行。Qwen 团队想要一个能理解时间的单一编码器。

结构化输出：LLaVA 输出自由格式文本。智能体需要 JSON。Qwen-VL 在显式的 JSON 输出格式上进行训练，包括将边界框坐标作为文本输出。

每一代 Qwen-VL 都沿这三个轴之一进行扩展。

## 核心概念

### Qwen-VL (2023 年 8 月)

第一代：OpenCLIP ViT-bigG/14 作为编码器（25 亿参数），兼容 LLaMA 的 Q-Former（1 步，256 个 query），Qwen-7B 基座模型。贡献：

- 448x448 分辨率（当时开源 VLM 的 SOTA）。
- Grounding：在带有显式坐标 token 输出的图文对上进行训练。"The cat is at <box>(112, 204), (280, 344)</box>"。
- 从一开始就进行中英双语训练。

当时的基准测试：在英语上与 GPT-4V 竞争，在中文上占据主导地位。Grounding 监督才是真正的头条。

### Qwen2-VL (2024 年 9 月) —— M-RoPE 与原生分辨率

Qwen2-VL 用原生动态分辨率 ViT 编码器取代了固定分辨率 + Q-Former 的堆叠结构。关键变化：

- 原生动态分辨率。ViT 接受任何能被 28 整除的 HxW（patch 为 14，2x 空间合并）。一张 1120x672 的图像（40x24 个合并 patch）产生 960 个视觉 token。无需 resize、无需 tiling、无需缩略图。
- M-RoPE（Multimodal RoPE）。每个 token 携带 3D 位置 (t, h, w) 而非 1D。对于图像 t=0，对于视频 t = frame_index。RoPE 按每个轴的频率旋转 query/key 向量。没有位置嵌入表。
- MLP projector。舍弃 Q-Former；在合并的 patch token 上使用 2 层 MLP。
- 支持 dynamic FPS 的视频。视频默认以 1-2 FPS 采样，但模型接受任意帧数。

结果：Qwen2-VL-7B 在多个多模态基准测试上匹敌 GPT-4o，并在 DocVQA 上击败它（94.5 对 88.4）。架构变化是决定性的一步。

### Qwen2.5-VL (2025 年 2 月) —— dynamic FPS + 绝对时间

Qwen2.5-VL 的重大转变是视频。Dynamic FPS 不仅仅是"在需要时采样更多帧"。论文将其形式化为：

- 绝对时间 token。不使用位置索引（第 0、1、2 帧……），而是使用实际时间戳。"At 0:04, the cat jumps." 模型看到 `<time>0.04</time>` token 与帧 token 交错排列。
- Dynamic FPS。对于慢速镜头以 1 FPS 采样，对于动作场景以 4+ FPS 采样。用户或训练者选择；M-RoPE 自适应。
- ViT 中的窗口注意力。空间注意力是窗口化的（在块内局部）以提高吞吐量；每隔几层添加全局注意力。
- 显式 JSON 输出格式。在工具调用数据上训练："{\"tool\": \"click\", \"coords\": [380, 220]}" 。开箱即用的智能体就绪。
- MRoPE-v2 缩放。位置随最大输入尺寸缩放，因此 10 分钟的视频不会耗尽频率范围。

基准测试：Qwen2.5-VL-72B 在大多数视频基准测试上击败 GPT-4o，在文档上与 Gemini 2.0 匹敌，并在 GUI grounding 上创下开源模型 SOTA（ScreenSpot：准确率 84%，GPT-4o 为 38%）。

### Qwen3-VL (2025 年 11 月)

Qwen3-VL 是一次增量升级，重在巩固而非重新发明：更大的 LLM 骨干网络（Qwen3-72B）、扩展的训练数据、改进的 OCR、通过 Qwen3 "thinking mode" 实现的更强推理能力。ViT 和 M-RoPE 保持不变。论文侧重于数据和训练改进，而非架构。

家族启示：到 2025 年，Qwen-VL 架构已经稳定。后续世代扩展的是计算和数据，而非基础组件。

### M-RoPE 的数学原理

经典 RoPE 使用配对坐标按位置 `m` 旋转维度为 `d` 的 query `q`：

```
q_rot[2i]   = q[2i]   * cos(m * theta_i) - q[2i+1] * sin(m * theta_i)
q_rot[2i+1] = q[2i]   * sin(m * theta_i) + q[2i+1] * cos(m * theta_i)
theta_i     = 10000^(-2i/d)
```

M-RoPE 将隐藏维度分成三个频带。假设 `d = 96`。将 32 维分配给时间，32 维分配给高度，32 维分配给宽度。每个频带按其自身轴的位置旋转。位于 (t=5, h=10, w=20) 的 patch 在其三个频带上分别应用旋转 `R_t(5)`、`R_h(10)`、`R_w(20)`。

文本 token 使用 `t = text_index, h = 0, w = 0`（或归一化选择），保持兼容性。视频帧使用 `t = frame_time, h = row, w = col`。单张图像使用 `t = 0`。

其优势在于：一种位置编码即可处理文本、图像和视频，无需分支代码或不同的位置表。

### Dynamic-FPS 采样逻辑

给定一段时长为 `T` 秒的视频和目标 token 预算 `B`：

1. 计算你能负担的最大 FPS：`fps_max = B / (T * tokens_per_frame)`。
2. 从 `{1, 2, 4, 8}` 中选择一个满足 `fps <= fps_max` 的目标 FPS。
3. 如果运动量大（光流启发式或用户显式请求），选择更高的 FPS。如果运动量小，选择更低的。
4. 以选定的 FPS 均匀采样；在帧之间插入 `<time>t</time>` token。

Qwen2.5-VL 隐式训练这一逻辑；在推理时用户通过 `fps` 参数控制。一段 60 秒的动作序列以 4 FPS 采样，每帧 81 个 token = 19440 个 token，在 32k 上下文中是可管理的。

### 结构化智能体输出

Qwen2.5-VL 的智能体训练明确针对结构化工具调用：

```
{
  "tool": "mouse_click",
  "coords": [1024, 512],
  "button": "left",
  "modifier": null
}
```

解析是确定性的：对模型输出进行 JSON.parse。相比之下，自由格式的 "click at (1024, 512)" 需要正则表达式和歧义处理。这一转变正是 Qwen2.5-VL 的 ScreenSpot 分数从 Qwen2-VL 的 55% 跃升至 84% 的原因。

## 动手实践

`code/main.py` 实现了：

- 对混合文本、图像 patch 和视频帧的打包序列进行 M-RoPE 位置计算。
- Dynamic-FPS 采样器：给定（时长、预算、运动级别），选择 FPS 并输出帧时间戳。
- 一个处理带有坐标字段的工具调用响应的玩具级 Qwen2.5-VL JSON 输出解析器。

运行它，然后感受在 5 分钟视频上将固定 FPS 替换为 dynamic-FPS 时的差异。

## 产出交付

本课程产出 `outputs/skill-qwen-vl-pipeline-designer.md`。给定一个视频任务（监控、智能体、动作识别、无障碍辅助），它输出 Qwen2.5-VL 配置（帧预算、FPS 策略、窗口注意力标志、智能体输出模式）和延迟估计。每当你为视频产品部署 Qwen-VL 家族模型时，都可以使用它。

## 练习题

1. 计算一个位于 (t=3, h=5, w=7) 的 patch 的 M-RoPE 旋转，隐藏维度为 48（每个频带 16，基础 theta 为 10000）。展示每个频带前三对配对的旋转角度。

2. 一段 10 分钟的监控摄像头录像以 1 FPS 采样会产生多少帧？在 384 分辨率下经过 3x pool，总共有多少 token？Qwen2.5-VL 默认的 32k 上下文能否处理？

3. 为一段 30 秒的网球回合、一段 30 秒的食谱演示、一段 30 秒的 UI 智能体录像分别选择 FPS。用 dynamic-FPS 逻辑为每个选择提供理由。

4. Qwen2.5-VL 完全舍弃了 Q-Former。为什么一个简单的 MLP 在 2025 年能工作，而在 2023 年不行？（提示：数据规模和编码器质量。）

5. 将三个 Qwen2.5-VL JSON 工具调用输出解析为 Python 字典。对于格式错误的 JSON 什么会失败，Qwen cookbook 推荐什么恢复策略？

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| M-RoPE | "Multimodal RoPE" | 三维旋转位置嵌入，在隐藏维度中划分时间、高度和宽度频带 |
| Dynamic FPS | "智能采样" | 根据运动、时长和 token 预算为每个视频选择帧采样率 |
| Absolute time token | "时间戳 token" | `<time>t</time>` 交错在序列中，使模型看到实际秒数而非帧索引 |
| Window attention | "局部注意力" | 空间自注意力限制在小窗口内以提高速度；定期添加全局注意力 |
| Structured agent output | "JSON 模式" | 训练数据监督，教导 VLM 输出可解析的 JSON，包含坐标和工具名称 |
| min_pixels / max_pixels | "分辨率边界" | Qwen2.5-VL 的每次请求控制，限制总像素数从而限制 token 数 |
| Grounding | "指向它" | 将边界框坐标作为文本 token 输出；自 Qwen-VL v1 起使用 |

## 延伸阅读

- [Bai et al. — Qwen-VL (arXiv:2308.12966)](https://arxiv.org/abs/2308.12966)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Qwen Team — Qwen3-VL (arXiv:2511.21631)](https://arxiv.org/abs/2511.21631)
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
