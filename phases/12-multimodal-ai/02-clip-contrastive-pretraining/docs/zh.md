# CLIP 与对比式视觉-语言预训练

> OpenAI 的 CLIP（2021）证明了一个足以驱动此后五年的核心思想：仅使用来自网络的噪声图像-标题对和一个对比损失，将图像编码器和文本编码器对齐到同一个向量空间中。零个有监督标签。4亿对数据。由此得到的嵌入空间能够执行零样本分类、图像-文本检索，并作为视觉塔嵌入到每一个 2026 年的 VLM 中。SigLIP 2（2025）用 sigmoid 替代了 softmax，并以更低的成本超越了 CLIP。本节课从 InfoNCE 的数学推导讲到 sigmoid 成对损失，并用 stdlib Python 构建训练步骤。

**类型：** 构建
**语言：** Python（stdlib，InfoNCE + sigmoid 损失实现）
**前置知识：** Phase 12 · 01（ViT 分块），Phase 7（Transformers）
**时间：** 约 180 分钟

## 学习目标

- 从互信息出发推导 InfoNCE 损失，并实现一个数值稳定的向量化版本。
- 解释为什么 sigmoid 成对损失（SigLIP）能够在 batch 32768+ 的规模下扩展，而没有 softmax 所需的 all-gather 开销。
- 通过构建文本模板（`a photo of a {class}`）并在余弦相似度上取 argmax，运行零样本 ImageNet 分类。
- 说出 CLIP / SigLIP 预训练提供的四个可调杠杆：batch size、temperature、prompt template、data quality。

## 问题背景

CLIP 之前的视觉模型是有监督的。收集标注数据集（ImageNet：120万张图像，1000个类别），训练 CNN，发布。标注成本高昂，标注会偏向标注者能够达成一致的范畴，而且标注无法迁移到新任务，除非进行微调。

网络上的图像-标题对有十亿以上的松散标注，完全免费。一张金毛寻回犬的照片，附带 alt 文本 "my dog Max in the park"，就携带了一个监督信号——文本描述了图像。问题是：你能把它转化为有用的训练信号吗？

CLIP 的答案：将图像-标题对视为匹配任务。给定 N 张图像和 N 个标题，学习将每张图像与其自身的标题匹配，而不是与另外 N-1 个干扰项匹配。监督信号是"这两个东西属于一起；那 N-1 个不属于"。没有类别标签。没有人工标注。只有一个对比损失。

由此得到的嵌入空间能做的远超 CLIP 训练时的目标。ImageNet 零样本有效，是因为 "a photo of a cat" 的嵌入会靠近那些从未被显式标注为 "cat" 的猫的图片。正是这一赌注催生了每一个 2026 年的 VLM。

## 核心概念

### 双编码器（Dual encoder）

CLIP 有两个塔：

- 图像编码器 `f`：ViT 或 ResNet，每张图像输出一个 D 维向量。
- 文本编码器 `g`：小型 transformer，每个标题输出一个 D 维向量。

两个塔都将输出归一化为单位长度。相似度为 `cos(f(x), g(y)) = f(x)^T g(y)`，因为两者都是单位范数。

对于 N 个（图像，标题）对的 batch，构建形状为 `(N, N)` 的相似度矩阵 `S`：

```
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

其中 `tau` 是一个可学习的 temperature（CLIP 初始化为 0.07；在 log 空间中学习）。

### InfoNCE 损失

CLIP 使用对行和列对称的交叉熵：

```
loss_i2t = CE(S, labels=identity)     # 每张图像的正样本是它自己的标题
loss_t2i = CE(S^T, labels=identity)   # 每个标题的正样本是它自己的图像
loss = (loss_i2t + loss_t2i) / 2
```

这就是 InfoNCE。CE 中的 softmax 迫使每张图像与其标题的匹配度高于 batch 中所有其他标题。"负样本"就是 batch 中的所有其他项。更大的 batch = 更多的负样本 = 更强的信号。CLIP 在 batch 32k 下训练；规模至关重要。

### Temperature

`tau` 控制 softmax 的锐度。低 tau → 尖锐分布，具有 hard negative mining 效果。高 tau → 柔和，所有样本都参与贡献。CLIP 学习 `log(1/tau)`，并进行裁剪以防止 collapse。SigLIP 2 固定初始 tau，改用可学习的 bias。

### 为什么 sigmoid 扩展性更好（SigLIP）

Softmax 需要整个相似度矩阵同步。在分布式训练中，你必须 all-gather 每个 embedding 到每个副本，然后做 softmax。这在通信量上是 world size 的二次方。

SigLIP 用逐元素的 sigmoid 替代 softmax：对于每个对 `(i, j)`，损失是一个二分类"这两个是匹配的对吗？"正类标签是对角线，其余都是负类。损失为：

```
L = -1/N sum over (i, j) [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]
```

`y_ij = 1` 当 `i == j`，否则为 0。每个对的损失是独立的。不需要 all-gather。每个 GPU 计算自己的局部块并求和。SigLIP 2 可以廉价地将 batch 扩展到 32k-512k，而 CLIP 需要成比例地增加通信量。

### 零样本分类

给定 N 个类别名称，为每个类别构建一个文本模板：

```
"a photo of a {class}"
```

用文本编码器嵌入每个模板。用图像编码器嵌入你的图像。Argmax 余弦相似度 = 预测类别。不需要在目标类别上训练。

Prompt template 很重要。CLIP 原始论文每个类别使用了 80 个模板（plain、artistic、photo、painting 等）并对 embedding 取平均。ImageNet 提升 3 个点。现代用法通常选择一到两个模板。

### Linear probe 与微调

零样本是一个基线。Linear probe（在冻结的 CLIP 特征之上为你的目标类别训练一个线性层）在域内任务上优于零样本。完整微调在域内优于 linear probe，但可能损害零样本迁移能力。三种方案，三种权衡。

### SigLIP 2：NaFlex 与 dense features

SigLIP 2（2025）增加了：
- NaFlex：单一模型处理可变宽高比和分辨率。
- 更好的 dense features，用于分割和深度估计，目标是作为 VLM 中的冻结 backbone。
- 多语言：在 100+ 种语言上训练，而 CLIP 仅支持英语。
- 10 亿参数规模，而 CLIP 最高为 4 亿。

在 2026 年的开源 VLM 中，SigLIP 2 SO400m/14 是默认的视觉塔。CLIP 仍然是纯图像-文本检索的默认选择，前提是 LAION-2B 训练分布与你的查询模式匹配。

### ALIGN、BASIC、OpenCLIP、EVA-CLIP

ALIGN（Google，2021）：与 CLIP 相同的思路，18 亿对规模，90% 噪声数据。证明了噪声数据可以扩展。OpenCLIP（LAION）：在 LAION-400M / 2B 上开源复现 CLIP，多个规模，是首选的开源 checkpoint。EVA-CLIP：从 masked image modeling 初始化；VLM 的强 backbone。BASIC：Google 的 CLIP+ALIGN 混合体。都是同一个家族，不同的数据和调优。

### 零样本天花板

CLIP 类模型的 ImageNet 零样本上限约为 76%（CLIP-G、OpenCLIP-G）。超越这一水平需要更大的数据（SigLIP 2 达到 80%+）或架构变更（有监督头、更多参数）。基准正在饱和；真正的价值在于下游 VLM 所消耗的嵌入空间。

## 动手实践

`code/main.py` 实现了：

1. 一个玩具双编码器（基于 hash 的图像特征、文本字符特征），让你无需 numpy 就能看到 InfoNCE 的形状。
2. 纯 Python 实现的 InfoNCE 损失（通过 log-sum-exp 保证数值稳定性）。
3. 用于对比的 sigmoid 成对损失。
4. 一个零样本分类流程：计算与一组文本 prompt 的余弦相似度，取 argmax 进行预测。

运行它并观察损失曲线。绝对数值是玩具级别的；但曲线形状与真实 CLIP 训练器输出的一致。

## 交付成果

本节课产出 `outputs/skill-clip-zero-shot.md`。给定一组图像（通过路径）和一个目标类别列表，它使用 CLIP 模板构建文本 prompt，用指定的 checkpoint（例如 `openai/clip-vit-large-patch14`）对两边进行嵌入，并返回 top-1 / top-5 预测及相似度分数。该 skill 不会对 prompt 列表之外的类别做出断言。

## 练习题

1. 手动为 4 个对的 batch 实现 InfoNCE。构建 4x4 相似度矩阵，运行 softmax，提取对角线，计算交叉熵。用这个手算结果验证你的 Python 实现。

2. SigLIP 除了 temperature 还使用了一个 bias 参数 `b`：`S'[i,j] = S[i,j]/tau + b`。当 batch 中存在严重的类别不平衡（每行负样本远多于正样本）时，`b` 起什么作用？阅读 SigLIP 第 3 节（arXiv:2303.15343）。

3. 为猫 vs 狗构建一个零样本分类器。尝试两个 prompt 模板：`a photo of a {class}` 和 `a picture of a {class}`。在 100 张测试图像上测量准确率。模板 ensemble 是否优于单个模板？

4. 计算在 512 GPU、batch 32k 下，softmax InfoNCE 与 sigmoid 成对损失的通信成本。哪个是 O(N) 扩展，哪个是 O(N^2)？引用 SigLIP 第 4 节。

5. 阅读 OpenCLIP scaling-laws 论文（arXiv:2212.07143，Cherti 等）。从图表中复现他们关于数据扩展的结论：在固定模型大小下，ImageNet 零样本准确率与训练数据大小之间的对数线性关系是什么？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| InfoNCE | "对比损失" | 在 batch 的相似度矩阵上做交叉熵；每个 item 的正样本是其配对 item，负样本是其他所有 |
| Sigmoid loss | "SigLIP 损失" | 逐对二元交叉熵；没有 softmax，没有 all-gather，在分布式训练中廉价扩展 |
| Temperature | "tau" | 在 softmax/sigmoid 前缩放 logits 的标量；控制分布的锐度 |
| Zero-shot | "无需微调分类" | 使用文本 prompt 构建类别 embedding，通过余弦相似度分类；不在目标类别上训练 |
| Prompt template | "a photo of a ..." | 围绕类别名称的文本脚手架；对零样本准确率影响 1-5 个点 |
| Dual encoder | "双塔" | 一个图像编码器 + 一个文本编码器，输出在共享的 D 维空间中 |
| Hard negative | "困难干扰项" | 与正样本足够相似的负样本，模型必须努力才能将它们分开 |
| Linear probe | "冻结 + 一层" | 仅在冻结特征之上训练一个线性分类器；衡量特征质量 |
| NaFlex | "原生灵活分辨率" | SigLIP 2 的能力，可以在不调整大小的情况下以任意宽高比和分辨率摄入图像 |
| Temperature scaling | "log 参数化 tau" | CLIP 参数化 `log(1/tau)` 以使梯度表现良好；裁剪以防止 collapse 到接近零的 tau |

## 延伸阅读

- [Radford et al. — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — CLIP 论文。
- [Zhai et al. — Sigmoid Loss for Language Image Pre-Training (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343) — SigLIP。
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 多语言 + NaFlex。
- [Jia et al. — ALIGN (arXiv:2102.05918)](https://arxiv.org/abs/2102.05918) — 用噪声网络数据扩展规模。
- [Cherti et al. — Reproducible scaling laws for contrastive language-image learning (arXiv:2212.07143)](https://arxiv.org/abs/2212.07143) — OpenCLIP 扩展规律。
