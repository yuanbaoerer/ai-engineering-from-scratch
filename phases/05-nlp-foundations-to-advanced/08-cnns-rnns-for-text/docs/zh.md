# 文本的 CNN 与 RNN

> 卷积学习 n-gram，循环记忆上下文。两者都被注意力机制超越。但在受限硬件上，两者仍然重要。

**类型：** 构建
**语言：** Python
**前置课程：** 第 3 阶段 · 11（PyTorch 入门），第 5 阶段 · 03（词嵌入），第 4 阶段 · 02（从零实现卷积）
**时间：** 约 75 分钟

## 问题所在

TF-IDF 和 Word2Vec 生成的扁平向量忽略了词序。基于它们的分类器无法区分 `dog bites man` 和 `man bites dog`。而词序有时正是关键信号。

在 Transformer 出现之前，有两大类架构填补了这一空白。

**文本卷积网络（TextCNN）。** 对词嵌入（Word Embedding）序列应用一维卷积。宽度为 3 的滤波器就是一个可学习的三元组检测器：它覆盖三个词并输出一个分数。叠加不同宽度（2、3、4、5）以检测多尺度模式。最大池化到固定大小的表示。扁平、并行、快速。

**循环网络（RNN、LSTM、GRU）。** 逐个处理 token，维护一个向前传递信息的隐藏状态（Hidden State）。顺序执行、有记忆能力、支持灵活输入长度。2014 年到 2017 年主导了序列建模，然后注意力机制出现了。

本课将构建这两种架构，然后说明促生注意力机制的关键缺陷。

## 核心概念

**TextCNN**（Kim，2014）。Token 被嵌入后，宽度为 `k` 的一维卷积在连续的 `k` 元组嵌入上滑动滤波器，生成特征图（Feature Map）。对该特征图进行全局最大池化（Global Max Pooling）以选取最强激活。将多个滤波器宽度的最大池化输出拼接，送入分类头。

为什么有效。滤波器是可学习的 n-gram。最大池化具有位置不变性，因此"not good"无论出现在评论开头还是中间，都会激活相同的特征。三个滤波器宽度各配 100 个滤波器，就得到 300 个学习到的 n-gram 检测器。训练是并行的，没有顺序依赖。

**RNN。** 在每个时间步 `t`，隐藏状态 `h_t = f(W * x_t + U * h_{t-1} + b)`。在各时间步共享 `W`、`U`、`b`。时间步 `T` 的隐藏状态是整个前缀的摘要。用于分类时，对 `h_1 ... h_T` 进行池化（最大池化、平均池化或取最后一个）。

普通 RNN 会遭受梯度消失问题。**LSTM** 添加了门控机制（Gating Mechanism）来决定遗忘什么、存储什么、输出什么，从而在长序列中稳定梯度。**GRU** 将 LSTM 简化为两个门；参数更少，性能相当。

**双向 RNN** 运行一个前向 RNN 和一个后向 RNN，拼接隐藏状态。每个 token 的表示同时看到左右上下文。对标注任务至关重要。

## 动手实现

### 步骤 1：用 PyTorch 实现 TextCNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)` 将 `[batch, seq_len, embed_dim]` 重塑为 `[batch, embed_dim, seq_len]`，因为 `nn.Conv1d` 将中间轴视为通道维度。池化后的输出是固定大小的，与输入长度无关。

### 步骤 2：LSTM 分类器

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

对序列进行最大池化，而非取最后一个状态。对于分类任务，最大池化通常优于取最后隐藏状态，因为长序列末尾的信息往往会主导最后的状态。

### 步骤 3：梯度消失演示（直觉理解）

没有门控的普通 RNN 无法学习长距离依赖。考虑一个简单的任务：预测 token `A` 是否出现在序列中的任何位置。如果 `A` 在位置 1 而序列长度为 100，那么损失的梯度必须经过 99 次循环权重乘法才能回传。如果权重小于 1，梯度消失；如果大于 1，梯度爆炸。

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# 在权重=0.9、100 步的条件下：
#   0.9 ^ 100 ≈ 2.7e-5
# 从第 100 步到第 1 步的梯度实际上为零。
```

LSTM 通过**细胞状态（Cell State）**解决了这个问题，细胞状态在网络上运行时只有加法交互（遗忘门对其进行乘法缩放，但梯度仍沿"高速公路"流动）。GRU 以更少的参数实现了类似的效果。两者都能在 100+ 步的序列上实现稳定训练。

### 步骤 4：为什么这仍然不够

即使使用 LSTM，仍然存在三个问题。

1. **顺序瓶颈。** 对长度为 1000 的序列训练 RNN 需要 1000 次串行前向/反向传播。无法在时间维度上并行化。
2. **编码器-解码器（Encoder-Decoder）设置中的固定大小上下文向量（Context Vector）。** 解码器只能看到编码器的最终隐藏状态，整个输入被压缩到一个向量中。长输入会丢失细节。第 09 课会直接讨论这一点。
3. **远距离依赖精度上限。** LSTM 优于普通 RNN，但仍然难以在 200+ 步的距离上传播特定信息。

注意力机制解决了以上三个问题。Transformer 完全抛弃了循环结构。第 10 课是转折点。

## 实际使用

PyTorch 的 `nn.LSTM`、`nn.GRU` 和 `nn.Conv1d` 都是生产就绪的。训练代码是标准的。

Hugging Face 提供了可直接作为输入层使用的预训练嵌入：

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

适用场景清单。

- **边缘/设备端推理。** 使用 GloVe 嵌入的 TextCNN 比 Transformer 小 10-100 倍。如果你的部署目标是手机，这就是你的技术栈。
- **流式/在线分类。** RNN 每次处理一个 token；Transformer 需要完整序列。对于实时传入的文本，LSTM 仍然胜出。
- **用于基线的小模型。** 在新任务上快速迭代。5 分钟在 CPU 上训练一个 TextCNN。
- **有限数据下的序列标注。** BiLSTM-CRF（第 06 课）仍然是适用于 1k-10k 标注句子的生产级 NER 架构。

其他场景都交给 Transformer。

## 交付使用

保存为 `outputs/prompt-text-encoder-picker.md`：

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## 练习

1. **简单。** 在一个三分类玩具数据集上训练 TextCNN（自行构造数据）。验证滤波器宽度（2, 3, 4）在平均 F1 上优于单一宽度（3）。
2. **中等。** 为 LSTM 分类器实现最大池化、平均池化和取最后状态三种池化方式。在小数据集上比较；记录哪种池化胜出并假设原因。
3. **困难。** 构建一个 BiLSTM-CRF NER 标注器（结合第 06 课和本课）。在 CoNLL-2003 上训练。与第 06 课的 CRF 基线和 BERT 微调进行比较。报告训练时间、内存和 F1。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| TextCNN | 用于文本的 CNN | 在词嵌入上叠加一维卷积并进行全局最大池化。Kim（2014）。 |
| RNN | 循环网络 | 每个时间步更新隐藏状态：`h_t = f(W x_t + U h_{t-1})`。 |
| LSTM | 门控 RNN | 添加输入/遗忘/输出门和细胞状态。在长序列上稳定训练。 |
| GRU | 简化版 LSTM | 两个门而非三个。准确率相近，参数更少。 |
| Bidirectional | 双向 | 前向 + 后向 RNN 拼接。每个 token 同时看到两侧上下文。 |
| Vanishing gradient | 梯度消失 | 普通 RNN 中反复乘以小于 1 的权重，导致早期步骤的梯度实际上为零。 |

## 延伸阅读

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — TextCNN 论文。八页。可读性强。
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — LSTM 论文。出人意料地清晰易懂。
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — 让 LSTM 为所有人所理解的图解。
