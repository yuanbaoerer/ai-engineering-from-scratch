# 视觉 Transformer (ViT)

> 图像是 patch 的网格。句子是 token 的网格。同一个 Transformer 吃掉两者。

**类型:** 构建
**语言:** Python
**前置知识:** 第 7 阶段 · 05（完整 Transformer）、第 4 阶段 · 03（CNN）、第 4 阶段 · 14（视觉 Transformer 入门）
**时间:** 约 45 分钟

## 问题所在

2020 年之前，计算机视觉意味着卷积。ImageNet、COCO 和检测基准上的每个 SOTA 都使用 CNN 骨干。Transformer 用于语言。

Dosovitskiy et al. (2020)——"An Image is Worth 16x16 Words"——展示了可以完全抛弃卷积。将图像切成固定大小的 patch，线性投影每个 patch 到嵌入，将序列送入普通 Transformer 编码器。在足够规模下（ImageNet-21k 预训练或更大），ViT 匹配或击败基于 ResNet 的模型。

ViT 是 2026 年更广泛模式的开始：一种架构，多种模态。Whisper 将音频 token 化。ViT 将图像 token 化。机器人的动作 token。视频的像素 token。Transformer 不在乎——给它一个序列，它就学习。

到 2026 年，ViT 及其后代（DeiT、Swin、DINOv2、ViT-22B、SAM 3）拥有大部分视觉领域。CNN 在边缘设备和延迟敏感任务上仍然胜出。其他一切在栈中某处都有一个 ViT。

## 核心概念

![图像 → patch → token → transformer](../assets/vit.svg)

### 第一步——分 patch

将 `H × W × C` 图像分成 `N × (P·P·C)` 的扁平 patch 序列。典型设置：`224 × 224` 图像，`16 × 16` patch → 196 个 patch，每个 768 个值。

```
图像 (224, 224, 3) → 14 × 14 网格的 16x16x3 patch → 196 个长度 768 的向量
```

Patch 大小是杠杆。更小的 patch = 更多 token，更好的分辨率，二次注意力成本。更大的 patch = 更粗糙，更便宜。

### 第二步——线性嵌入

单个可学习矩阵将每个扁平 patch 投影到 `d_model`。等价于核大小 `P`、步长 `P` 的卷积。在 PyTorch 中这字面上是 `nn.Conv2d(C, d_model, kernel_size=P, stride=P)`——2 行实现。

### 第三步——添加 `[CLS]` token，添加位置嵌入

- 添加可学习的 `[CLS]` token。其最终隐藏状态是用于分类的图像表示。
- 添加可学习位置嵌入（ViT 原始）或正弦 2D（后续变体）。
- 2024+ RoPE 扩展到 2D 用于位置，有时无显式嵌入。

### 第四步——标准 Transformer 编码器

堆叠 L 个 `LayerNorm → Self-Attention → + → LayerNorm → MLP → +` 块。与 BERT 相同。无视觉特定层。这是论文的教学亮点。

### 第五步——头

分类：取 `[CLS]` 隐藏状态 → 线性 → softmax。对于 DINOv2 或 SAM，丢弃 `[CLS]`，直接使用 patch 嵌入。

### 重要的变体

| 模型 | 年份 | 变化 |
|------|------|------|
| ViT | 2020 | 原始。固定 patch 大小，完整全局注意力。 |
| DeiT | 2021 | 蒸馏；仅在 ImageNet-1k 上可训练。 |
| Swin | 2021 | 层次化带移位窗口。固定次二次成本。 |
| DINOv2 | 2023 | 自监督（无标签）。最佳通用视觉特征。 |
| ViT-22B | 2023 | 22B 参数；扩展定律适用。 |
| SigLIP | 2023 | ViT + 语言配对，sigmoid 对比损失。 |
| SAM 3 | 2025 | 分割一切；ViT-Large + 可提示掩码解码器。 |

### 为什么花了一段时间

ViT 需要*大量*数据才能匹配 CNN，因为它没有 CNN 归纳偏置（平移不变性、局部性）。没有 >1 亿标注图像或强自监督预训练，CNN 在匹配计算下仍然胜出。DeiT 在 2021 年用蒸馏技巧修复了这个问题；DINOv2 在 2023 年用自监督永久修复了它。

## 动手实现

参见 `code/main.py`。纯标准库的分 patch + 线性嵌入 + 完整性检查。无训练——任何现实规模的 ViT 都需要 PyTorch 和数小时 GPU 时间。

### 第一步：假图像

24 × 24 RGB 图像作为 `(R, G, B)` 元组的行列表。我们使用 6×6 patch → 16 个 patch，每个 108 维嵌入向量。

### 第二步：分 patch

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

光栅顺序：行优先跨网格。每个 ViT 使用这种排序。

### 第三步：线性嵌入

每个扁平 patch 乘以随机 `(patch_flat_size, d_model)` 矩阵。验证添加 `[CLS]` 后输出形状为 `(N_patches + 1, d_model)`。

### 第四步：计算现实 ViT 的参数量

打印 ViT-Base 的参数计数：12 层、12 头、d=768、patch=16。与 ResNet-50（~25M）比较。ViT-Base 约 86M。ViT-Large ~307M。ViT-Huge ~632M。

## 使用场景

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patch
cls_emb = out[:, 0]                       # 图像表示
```

**DINOv2 嵌入是 2026 年图像特征的默认选择。** 冻结骨干，训练小型头。适用于分类、检索、检测、描述。Meta 的 DINOv2 检查点在每个非文本视觉任务上优于 CLIP。

**Patch 大小选择。** 小型模型使用 16×16 (ViT-B/16)。密集预测（分割）使用 8×8 或 14×14 (SAM, DINOv2)。非常大的模型使用 14×14。

## 交付使用

参见 `outputs/skill-vit-configurator.md`。该技能根据数据集大小、分辨率和计算预算为新视觉任务选择 ViT 变体和 patch 大小。

## 练习

1. **简单。** 运行 `code/main.py`。验证 patch 数等于 `(H/P) * (W/P)`，扁平 patch 维度等于 `P*P*C`。
2. **中等。** 实现 2D 正弦位置嵌入——对每个 patch 的 `row` 和 `col` 使用两个独立的正弦编码，拼接。将它们送入微型 PyTorch ViT，在 CIFAR-10 上比较可学习位置嵌入的准确率。
3. **困难。** 构建 3 层 ViT (PyTorch)，在 1,000 张 MNIST 图像上用 4×4 patch 训练。测量测试准确率。然后在相同的 1,000 张图像上添加 DINOv2 预训练（简化版：仅训练编码器从掩码 patch 预测 patch 嵌入）。准确率是否提高？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Patch | "视觉 Transformer 的 token" | 图像 `P × P × C` 区域的像素值扁平向量。 |
| 分 patch | "切 + 扁平化" | 将图像切成不重叠的 patch，每个扁平化为向量。 |
| `[CLS]` token | "图像摘要" | 添加的可学习 token；其最终嵌入是图像表示。 |
| 归纳偏置 | "模型假设什么" | ViT 比 CNN 有更少的先验；需要更多数据弥补差距。 |
| DINOv2 | "自监督 ViT" | 使用图像增强 + 动量教师无标签训练。2026 年最佳通用图像特征。 |
| SigLIP | "CLIP 的继任者" | ViT + 文本编码器用 sigmoid 对比损失训练；在匹配计算下优于 CLIP。 |
| Swin | "窗口化 ViT" | 层次化 ViT，局部注意力 + 移位窗口；次二次。 |
| 注册 token | "2023 年技巧" | 几个额外可学习 token 吸收注意力汇聚；改善 DINOv2 特征。 |

## 延伸阅读

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — ViT 论文。
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT。
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin。
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2。
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — DINOv2 的注册 token 修复。
