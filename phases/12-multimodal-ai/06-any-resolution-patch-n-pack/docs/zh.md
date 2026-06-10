# Any-Resolution Vision: Patch-n'-Pack 与 NaFlex

> 真实图像并非 224x224 的正方形。一张收据是 9:16，一张图表是 16:9，一张医学扫描可能是 4096x4096，一张手机截图是 9:19.5。2024 年之前的 VLM 解决方案——将所有图像缩放到固定正方形——丢弃了使 OCR、文档理解和高分辨率场景解析得以实现的关键信息。NaViT（Google，2023）展示了如何将可变分辨率的 patch 打包到单个 transformer batch 中，并使用块对角掩码（block-diagonal masking）。Qwen2-VL 的 M-RoPE（2024）完全摒弃了绝对位置表。LLaVA-NeXT 的 AnyRes 将高分辨率图像切分为基础图 + 子图。SigLIP 2 的 NaFlex 变体（2025）现在是希望用单个 checkpoint 服务所有宽高比的开源 VLM 的默认编码器。本节课完整实现 patch-n'-pack。

**类型：** Build
**语言：** Python（标准库，patch packer + block-diagonal mask）
**前置条件：** Phase 12 · 01（ViT patches），Phase 12 · 05（LLaVA）
**时间：** ~120 分钟

## 学习目标

- 将一批可变分辨率图像的 patch 打包到一个序列中，并构建块对角注意力掩码。
- 针对给定任务在 AnyRes tiling（LLaVA-NeXT）、NaFlex（SigLIP 2）和 M-RoPE（Qwen2-VL）之间做出选择。
- 在不调整大小的情况下为 OCR、图表和摄影计算 token 预算。
- 说出正方形缩放的三种失败模式：文字被压扁、内容被裁剪、padding 浪费 token。

## 问题所在

Transformer 期望一个序列。一个 batch 是相同长度的序列堆叠。如果你的图像是 224x224，你每次得到 196 个 patch token，不需要 padding，任务完成。用 224 训练，用 224 推理，再也不用考虑分辨率。

现实世界并不配合。文档是竖向的（8.5x11 英寸，约 2:3）。图表截图是横向的（16:9）。收据又高又窄（1:3）。医学影像以 2048x2048 或更高分辨率传输。移动设备截图是 1170x2532（0.46:1）。

三种 2024 年之前的选项及其失败原因：

1. 缩放到固定正方形（224x224 或 336x336）。挤压会扭曲文字和人脸。降采样会破坏图表标签和 OCR 内容。这是 LLaVA-1.5 之前的标准做法。
2. 裁剪到固定宽高比。你丢弃了图像的大部分内容，而选择裁剪位置本身就是一个视觉问题。
3. padding 到最长边。避免了扭曲，但对于竖向图像会浪费 50% 以上的 token 在 padding 上。所有这些 pad token 的注意力成本是二次的。

2024-2025 年的答案：让 transformer 以图像的原始分辨率消费 patch，并想办法将异构 batch 打包到一个序列中而不浪费计算。

## 核心概念

### NaViT 与 patch-n'-pack

NaViT（Dehghani 等，2023）是展示这种方法在大规模上有效的论文。其思路是机械性的：

1. 对于 batch 中的每张图像，在选定的 patch size（例如 14）下计算其原生 patch 网格。
2. 将每张图像的 patch 展平为其自身可变长度的序列。
3. 将所有图像的 patch 拼接成一个长序列作为 batch。
4. 构建块对角注意力掩码，使图像 A 的 patch 只在图像 A 内部进行注意力计算。
5. 携带每个 patch 的位置信息（2D RoPE 或分数位置嵌入）。

一个包含三张图像的 batch：336x336（576 token）、224x224（256 token）和 448x336（768 token），变成一个 1600 token 的序列，带有 1600x1600 的块对角掩码。没有 padding。没有计算浪费。Transformer 可以处理任意宽高比。

NaViT 还引入了训练期间的分数 patch 丢弃——在 batch 中随机丢弃 50% 的 patch——这既起到正则化作用又加速了训练。SigLIP 2 继承了这一点。

### AnyRes（LLaVA-NeXT）

LLaVA-NeXT 的 AnyRes 是务实的替代方案。给定一张高分辨率图像和一个固定编码器（336 的 CLIP 或 SigLIP），对图像进行切分：

1. 从预定义的网格布局集合中——(1x1)、(1x2)、(2x1)、(1x3)、(3x1)、(2x2) 等——选择最符合图像宽高比的那个。
2. 将完整图像切分到网格中；每个 tile 成为一个 336x336 的裁剪。
3. 同时生成一个缩略图：将整个图像缩放到 336x336 作为全局上下文 token。
4. 通过冻结的 336 编码器编码每个 tile。拼接 tile token + 缩略图 token。

对于 672x672 的图像，使用 2x2 网格加缩略图：4 * 576 + 576 = 2880 个视觉 token。昂贵但有效——LLM 既能看到局部细节又能看到全局上下文。

当你的编码器是冻结的且只支持一种分辨率时，AnyRes 是首选方案。对于大图像，token 数量会爆炸（1344x1344 的图像在 4x4 网格下是 9216 + 576 ≈ 9800 个 token，这几乎占满了 8k LLM 上下文的大部分）。

### M-RoPE（Qwen2-VL）

Qwen2-VL 引入了 Multimodal Rotary Position Embedding。与 NaViT 的分数位置或 AnyRes 的 tile-and-thumbnail 不同，每个 patch 携带一个 3D 位置（时间、高度、宽度）。query/key 的旋转处理任意的 H、W 和时间长度。

M-RoPE 原生支持动态分辨率，无需重新训练。在推理时，你输入任意 HxW 的图像，patch embedder 生成 H/14 x W/14 个 token，每个 token 获得其 (t=0, r=row, c=col) 位置，RoPE 以正确的频率旋转注意力，完成。Qwen2.5-VL 和 Qwen3-VL 延续了这一点。InternVL3 的 V2PE 是相同的思路，针对不同模态使用可变编码。

与 AnyRes 不同，M-RoPE 在原生分辨率下是 O(H x W / P^2) 个 token——没有乘性的 tile 开销。与 NaViT 不同，它仍然期望每次前向传播只有一张图像。跨分辨率 batching 仍然需要在之上叠加 patch-n'-pack。

### NaFlex（SigLIP 2）

NaFlex 是 SigLIP 2 checkpoint 的 native-flex 模式。单个模型在推理时服务多种序列长度（256、729、1024 token）。内部它在训练期间使用 NaViT 风格的 patch-n'-pack 和每个 patch 的绝对分数位置。卖点：一个 checkpoint，根据任务在推理时选择你的 token 预算。

对于语义任务（分类、检索），256 token。对于 OCR 或图表理解，1024 token。无需重新训练。

### 打包掩码

块对角掩码是大多数实现容易出错的地方。对于长度为 `N_total` 的打包序列，覆盖图像 `i=0..B-1`，各图像长度为 `n_i`，掩码 `M` 的形状为 `(N_total, N_total)`，当两个索引落在同一张图像的块内时为 1，否则为 0。你可以从累积长度列表构建它：

```
offsets = [0, n_0, n_0+n_1, ..., N_total]
M[i, j] = 1 当且仅当存在 b 使得 offsets[b] <= i < offsets[b+1] 且 offsets[b] <= j < offsets[b+1]
```

在 PyTorch 中，使用 `torch.block_diag` 或显式 gather 只需一行代码。FlashAttention 的可变长度路径（`cu_seqlens`）完全跳过掩码，直接使用累积长度张量在同序列内进行注意力计算——对于典型 batch 比密集掩码快约 10 倍。

### Token 预算

根据任务选择策略：

- OCR / 文档：1024-4096 token。SigLIP 2 NaFlex 在 1024，或 AnyRes 3x3 + 缩略图。
- 图表和 UI：384-448 原生分辨率下的 729-1024 token。Qwen2.5-VL 动态分辨率并设置最大像素上限。
- 自然照片：256-576 token 就足够了。下游 LLM 能看到足够的信息。在内容密度高的地方为 token 付费。
- 视频：空间池化后每帧 64-128 token，2-8 FPS。第 12.17 课涵盖此内容。

2026 年生产规则：选择一个按任务设定的最大像素上限，在该上限内以原生宽高比编码，打包 batch，并跳过 padding。Qwen2.5-VL 暴露的 `min_pixels` 和 `max_pixels` 正是为此而设。

## 使用它

`code/main.py` 为一批具有整数像素坐标的异构图像实现了 patch-n'-pack。它：

- 接收一个 (H, W) 图像尺寸列表。
- 在 patch size 为 14 的情况下计算每张图像的 patch 序列长度。
- 将它们打包到一个总长度为 `sum(n_i)` 的序列中。
- 构建块对角注意力掩码（密集形式，为了清晰）。
- 比较打包成本与正方形缩放和 AnyRes tiling。
- 为混合 batch（收据、图表、截图、照片）打印 token 预算表。

运行它。输出的数字就是每个 2026 年开源 VLM 都使用 patch-n'-pack 的原因。

## 交付它

本节课产出 `outputs/skill-resolution-budget-planner.md`。给定一个混合宽高比的工作负载（OCR、图表、照片、视频帧）和一个总 token 预算，它选择正确的策略（NaFlex、AnyRes、M-RoPE 或固定正方形）并输出每个请求的配置。在为产品调整 VLM 大小时使用此技能——它能防止悄无声息的 10 倍 token 膨胀，那会让延迟预算崩溃。

## 练习题

1. 一张收据是 600x1500（1:2.5）。在 patch size 为 14 时，原生分辨率下有多少 token？缩放到 336 正方形后有多少？实践中哪种会损失更多 OCR 精度？

2. 为四张图像的 batch 构建块对角掩码，长度分别为 256、576、729、1024。验证注意力矩阵是 2585x2585，并且恰好有 `256^2 + 576^2 + 729^2 + 1024^2` 个非零项。

3. 对于 patch 14 下的 1792x896 图像，比较：(a) 缩放到 336 正方形然后编码，(b) AnyRes 2x1 + 缩略图，(c) M-RoPE 在原生分辨率。哪种使用最少的 token？哪种保留最多细节？

4. 实现分数 patch 丢弃：给定一个打包序列，随机均匀丢弃 50% 的 token，并相应地更新块对角掩码。测量掩码稀疏度的变化。

5. 阅读 Qwen2-VL 论文（arXiv:2409.12191）的第 3.2 节。用两句话描述 `min_pixels` 和 `max_pixels` 控制什么，以及为什么两个边界都很重要。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Patch-n'-pack | "NaViT-style packing" | 将不同图像的可变长度 patch 序列拼接到同一个 batch 维度中 |
| Block-diagonal mask | "Packing mask" | 注意力掩码，将每张图像的 patch 限制为只关注自身，而非打包中的邻居 |
| AnyRes | "LLaVA-NeXT tiling" | 将高分辨率图像切分为固定大小的 tile 网格加一个全局缩略图；用固定编码器编码每个 tile |
| NaFlex | "SigLIP 2 native-flex" | 单个 SigLIP 2 checkpoint，在推理时无需重新训练即可服务 256/729/1024 token 预算 |
| M-RoPE | "Multimodal RoPE" | 3D 旋转位置编码（时间、行、列），无需位置表即可处理任意 H、W、T |
| cu_seqlens | "FlashAttention packing" | FlashAttention 可变长度路径使用的累积长度张量，替代密集块对角掩码 |
| min_pixels / max_pixels | "Resolution bounds" | Qwen2.5-VL 的每个请求旋钮，限制非常小或非常大输入的 token 数量 |
| Visual token budget | "How many tokens per image" | 每张图像发出的 patch token 粗略计数；设置 LLM 的 prompt 预算和注意力成本 |

## 延伸阅读

- [Dehghani et al. — Patch n' Pack: NaViT (arXiv:2307.06304)](https://arxiv.org/abs/2307.06304)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Laurençon et al. — What matters when building vision-language models? (Idefics2, arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
