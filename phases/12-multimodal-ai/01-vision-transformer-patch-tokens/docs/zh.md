# Vision Transformer 与 Patch-Token 原语

> 在涉及任何多模态内容之前，图像必须先变成 transformer 可以处理的 token 序列。2020 年的 ViT 论文用 16x16 像素 patch、线性投影和位置嵌入回答了这个问题。五年后，每一款 2026 年的前沿模型（Claude Opus 4.7 原生 2576px、Gemini 3.1 Pro、Qwen3.5-Omni）仍然以此开始——编码器从 ViT 演变为 DINOv2 再到 SigLIP 2，加入了 register token，位置编码方案变成了 2D-RoPE，但原语始终未变。本课从头到尾讲解 patch-token 流水线，并用标准库 Python 实现它，以便 Phase 12 的其余部分对"视觉 token"有一个具体的心智模型。

**类型：** 学习
**语言：** Python（标准库，patch tokenizer + 几何计算器）
**前置知识：** Phase 7（Transformers）、Phase 4（Computer Vision）
**时间：** 约 120 分钟

## 学习目标

- 将 HxWx3 图像转换为带有正确位置编码的 patch token 序列。
- 为给定配置（patch size、分辨率、hidden dim、depth）的 ViT 计算序列长度、参数量和 FLOPs。
- 说出将 ViT 从 2020 年研究推向 2026 年生产的三大升级：自监督预训练（DINO / MAE）、register token 和原生分辨率打包（native-resolution packing）。
- 针对下游任务在 CLS pooling、mean pooling 和 register token 之间做出选择。

## 问题背景

Transformers 操作的是向量序列。文本已经是一个序列（字节或 token）。图像则是带有三个颜色通道的二维像素网格——不是一个序列。如果你把每个像素都展平，一张 224x224 的 RGB 图像会变成 150,528 个 token，在这个长度上做自注意力是完全不可行的（与序列长度呈二次关系）。

2020 年之前的做法是在前面接一个 CNN 特征提取器：ResNet 产生一个 7x7 的 2048 维特征图，将这 49 个 token 喂给 transformer。这能工作，但继承了 CNN 的偏置（平移等变性、局部感受野），并且失去了 transformer 的规模化优势。

Dosovitskiy 等人（2020）提出了一个直率的问题：如果我们跳过 CNN 呢？将图像切分成固定大小的 patch（比如 16x16 像素），将每个 patch 线性投影成一个向量，加上位置嵌入，然后将序列喂给一个普通的 transformer。在当时这被视为异端——没有卷积的视觉处理。但在足够的数据（JFT-300M，然后是 LAION）上，它在 ImageNet 上击败了 ResNet，并且持续改进。

到 2026 年，ViT 原语已是毋庸置疑的基础。每一个开放权重的 VLM 的视觉塔都是某种后代（DINOv2、SigLIP 2、CLIP、EVA、InternViT）。问题不再是"我们应该用 patch 吗？"而是"用什么 patch size、什么分辨率策略、什么预训练目标、什么位置编码"。

## 核心概念

### Patch 作为 token

给定一个形状为 `(H, W, 3)` 的图像 `x` 和一个 patch size `P`，你将图像切割成一个 `(H/P) x (W/P)` 的互不重叠的 patch 网格。每个 patch 是一个 `P x P x 3` 的像素立方体。将每个立方体展平为一个 `3 P^2` 的向量。应用一个共享的线性投影 `W_E`，其形状为 `(3 P^2, D)`，将每个 patch 映射到模型的隐藏维度 `D`。

对于 ViT-B/16 的标准配置：
- 分辨率 224，patch size 16 → 网格 14x14 → 196 个 patch token。
- 每个 patch 是 `16 x 16 x 3 = 768` 个像素值，投影到 `D = 768`。
- 添加一个可学习的 `[CLS]` token → 序列长度 197。

Patch 投影在数学上等同于一个 2D 卷积，其 kernel size 为 `P`，stride 为 `P`，输出通道数为 `D`。生产代码实际上就是这样实现的——`nn.Conv2d(3, D, kernel_size=P, stride=P)`。"线性投影"的视角是概念性的；kernel 的视角是高效的。

### 位置嵌入

Patch 本身没有固有的顺序——transformer 将它们视为一个集合（bag）。早期的 ViT 添加了可学习的 1D 位置嵌入（每个位置一个 768 维的向量，共 197 个）。这能工作，但将模型绑定到了训练分辨率：在推理时如果改变网格大小，就必须对位置表进行插值。

现代视觉主干网络使用 2D-RoPE（Qwen2-VL 的 M-RoPE、SigLIP 2 的默认方案）或分解式 2D 位置编码。2D-RoPE 根据 patch 的（行，列）索引旋转 query 和 key 向量，因此模型从旋转角度推断出相对的 2D 位置。没有位置表。模型在推理时可以处理任意网格大小。

### CLS token、池化输出和 register token

什么是图像级别的表示？三种选择共存：

1. `[CLS]` token。在 patch 序列前添加一个可学习的向量。经过所有 transformer block 后，CLS token 的隐藏状态就是图像表示。继承自 BERT。原始 ViT、CLIP 使用。
2. Mean pool。对 patch token 的输出隐藏状态取平均。SigLIP、DINOv2、大多数现代 VLM 使用。
3. Register token。Darcet 等人（2023）观察到，没有显式 sink token 的 ViT 会发展出高范数的"伪影" patch，这些 patch 会劫持自注意力。添加 4–16 个可学习的 register token 可以吸收这种负载，并改善密集预测质量（分割、深度）。DINOv2 和 SigLIP 2 都配备了 register token。

这个选择对下游任务很重要。CLS 对分类来说足够好。对于将 patch token 输入 LLM 的 VLM，你完全跳过池化——每个 patch 都变成一个 LLM 输入 token。Register token 在交接前被丢弃（它们是脚手架，不是内容）。

### 预训练：监督、对比、掩码、自蒸馏

2020 年的 ViT 使用 JFT-300M 上的监督分类进行预训练。很快被以下方法取代：

- CLIP（2021）：在 4 亿对图像-文本上进行对比学习。第 12.02 课。
- MAE（2021，He 等人）：掩码 75% 的 patch，重建像素。自监督，纯图像即可工作。
- DINO（2021）/ DINOv2（2023）：使用学生-教师架构的自蒸馏，无需标签，无需 caption。2023 年的 DINOv2 ViT-g/14 是最强的纯视觉主干网络，也是"密集特征"用例的默认选择。
- SigLIP / SigLIP 2（2023、2025）：使用 sigmoid loss 和 NaFlex 实现原生宽高比的 CLIP。2026 年开放 VLM（Qwen、Idefics2、LLaVA-OneVision）中的主导视觉塔。

你选择的预训练决定了主干网络擅长什么：CLIP/SigLIP 用于与文本的语义匹配，DINOv2 用于密集视觉特征，MAE 作为下游微调的起点。

### 缩放定律

ViT 缩放（Zhai 等人，2022）确立了 ViT 的质量在模型大小、数据大小和计算量上遵循可预测的规律。在固定计算量下：
- 更大的模型 + 更多数据 → 更好的质量。
- Patch size 是序列长度与保真度之间的杠杆。Patch 14（DINOv2/SigLIP SO400m 的典型值）每张图像产生更多 token；对 OCR 和密集任务更好，对速度更差。
- 分辨率是另一个大杠杆。从 224 到 384 再到 512 几乎总是有帮助的，但 FLOPs 呈二次增长。

ViT-g/14（10 亿参数，patch 14，分辨率 224 → 256 个 token）和 SigLIP SO400m/14（4 亿参数，patch 14）是 2026 年开放 VLM 的两个主力编码器。

### ViT 的参数量

完整计算在 `code/main.py` 中。对于 ViT-B/16 @ 224：

```
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

在加载 checkpoint 之前，用这种方式粗略估算每个 ViT 的参数量。主干网络的大小决定了任何下游 VLM 的 VRAM 下限。

### 2026 年生产配置

2026 年大多数开放 VLM 配备的编码器是原生分辨率（NaFlex）下的 SigLIP 2 SO400m/14。它具有：
- 4 亿参数。
- Patch size 14，默认分辨率 384 → 每张图像 729 个 patch token。
- 图像级任务使用 mean pool；所有 729 个 patch 流入 LLM 进行 VQA。
- 4 个 register token，在交给 LLM 前丢弃。
- 带有图像级缩放的 2D-RoPE，用于原生宽高比。

该配置中的每一个决定都可以追溯到一篇你可以阅读的论文。

## 使用它

`code/main.py` 是一个 patch tokenizer 和几何计算器。给定（图像 H、W，patch P，隐藏维度 D，深度 L），它会报告：

- Patch 后的网格形状和序列长度。
- 合成 8x8 像素 toy 图像的 token 序列（走一遍展平 + 投影路径）。
- 按 patch embed、position embed、transformer block 和 head 分解的参数量。
- 目标分辨率下每次前向传播的 FLOPs。
- ViT-B/16 @ 224、ViT-L/14 @ 336、DINOv2 ViT-g/14 @ 224、SigLIP SO400m/14 @ 384 的对比表。

运行它。将参数量与公布的数字匹配。尝试不同的 patch size 和分辨率，感受 token 数量的成本。

## 交付物

本课产出 `outputs/skill-patch-geometry-reader.md`。给定一个 ViT 配置（patch size、分辨率、hidden dim、depth），它产出 token 数量、参数量和 VRAM 估算及其依据。每当你为 VLM 选择视觉主干网络时使用这个技能——它可以防止"token 爆炸，LLM 上下文被填满"的意外。

## 练习

1. 计算 Qwen2.5-VL 在原生 1280x720 输入、patch size 14 时的 patch-token 序列长度。与仅使用 CLS 表示相比如何？

2. 一帧 1080p 画面（1920x1080）在 patch 14 下产生多少个 token？以 30 FPS 播放 5 分钟视频，总共有多少个视觉 token？哪种方式节省成本最多：pooling、帧采样还是 token merging？

3. 用纯 Python 实现对 patch token 的 mean pooling。验证对 DINOv2 输出的 196 个 token 做 mean-pool 是否与模型 `forward` 返回的 pooled embedding 一致。

4. 阅读 "Vision Transformers Need Registers"（arXiv:2309.16588）的第三节。用两句话描述 register 吸收了什么伪影，以及为什么这对下游密集预测很重要。

5. 修改 `code/main.py` 以支持 patch-n'-pack：给定一组不同分辨率的图像，生成一个打包后的单一序列和块对角注意力掩码。学到第 12.06 课时与其进行验证。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| Patch | "16x16 像素方块" | 输入图像的固定大小互不重叠区域；变成一个 token |
| Patch embedding | "线性投影" | 一个共享的学习矩阵（或 stride=P 的 Conv2d），将展平的 patch 像素映射到 D 维向量 |
| CLS token | "Class token" | 前置的可学习向量，其最终隐藏状态代表整张图像；2026 年已变为可选 |
| Register token | "Sink token" | 额外的可学习 token，吸收 ViT 在预训练过程中发展出的高范数注意力伪影 |
| Position embedding | "位置信息" | 使序列具有顺序感知的逐位置向量或旋转；2D-RoPE 是现代默认方案 |
| Grid | "Patch grid" | 给定分辨率和 patch size 下的 (H/P) x (W/P) 二维 patch 数组 |
| NaFlex | "原生灵活分辨率" | SigLIP 2 特性：单一模型无需重新训练即可服务多种宽高比和分辨率 |
| Backbone | "视觉塔" | 预训练的图像编码器，其 patch-token 输出在 VLM 中喂给 LLM |
| Pooling | "图像级摘要" | 将 patch token 变成一个向量的策略：CLS、mean、attention pool 或基于 register 的方式 |
| Patch 14 vs 16 | "更细 vs 更粗的网格" | Patch 14 每张图像产生更多 token，对 OCR 保真度更好，更慢；patch 16 是经典默认值 |

## 延伸阅读

- [Dosovitskiy 等人 — An Image is Worth 16x16 Words (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929) — 原始 ViT。
- [He 等人 — Masked Autoencoders Are Scalable Vision Learners (arXiv:2111.06377)](https://arxiv.org/abs/2111.06377) — MAE，自监督预训练。
- [Oquab 等人 — DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193) — 大规模自蒸馏，无需标签。
- [Darcet 等人 — Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588) — register token 与伪影分析。
- [Tschannen 等人 — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 2026 年的默认视觉塔。
- [Zhai 等人 — Scaling Vision Transformers (arXiv:2106.04560)](https://arxiv.org/abs/2106.04560) — 经验缩放定律。
