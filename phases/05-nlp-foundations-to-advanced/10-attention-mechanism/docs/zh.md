# 注意力机制 — 突破性进展

> 解码器不再眯着眼睛看压缩后的摘要，而是开始审视整个源序列。此后的一切都是注意力加上工程实践。

**类型：** 构建
**语言：** Python
**前置知识：** 第 5 阶段 · 09（序列到序列模型）
**时间：** 约 45 分钟

## 问题所在

第 09 课以一个有节制的失败结束。在玩具复制任务上训练的 GRU 编码器-解码器，准确率从长度 5 时的 89% 下降到长度 80 时接近随机猜测。原因是结构性的，不是训练 bug：编码器提取的所有信息都必须塞进一个固定大小的隐藏状态中，而解码器看不到任何其他内容。

Bahdanau、Cho 和 Bengio 在 2014 年发表了一个三行代码的修复方案。不是只给解码器最终的编码器状态，而是保留每个编码器状态。在解码器的每一步，计算编码器状态的加权平均，其中权重表示"解码器现在需要多大程度地关注编码器位置 `i`？"这个加权平均就是上下文向量，它在每个解码步骤都会变化。

这就是整个核心思想。Transformer 对其进行了扩展。自注意力（Self-Attention）将其应用于单个序列。多头注意力（Multi-Head Attention）并行运行多个注意力头。但 2014 年的版本已经打破了瓶颈，一旦你理解了它，转向 Transformer 就是工程问题，而不是概念问题。

## 核心概念

![Bahdanau 注意力：解码器查询所有编码器状态](../assets/attention.svg)

在每个解码步骤 `t`：

1. 使用前一个解码器隐藏状态 `s_{t-1}` 作为**查询（query）**。
2. 将其与每个编码器隐藏状态 `h_1, ..., h_T` 进行评分。每个编码器位置一个标量分数。
3. 对分数进行 softmax，得到注意力权重 `α_{t,1}, ..., α_{t,T}`，这些权重之和为 1。
4. 上下文向量 `c_t = Σ α_{t,i} * h_i`。编码器状态的加权平均。
5. 解码器接收 `c_t` 加上前一个输出词元，生成下一个词元。

加权平均是关键。当解码器需要将 "Je" 翻译成 "I" 时，它会给予 "Je" 对应的编码器状态高权重，其他位置低权重。当需要 "not" 时，它会给予 "pas" 高权重。上下文向量在每一步都会重新塑造。

## 形状（每个人都踩过的坑）

这是每个注意力实现第一次都会出错的地方。请仔细阅读。

| 事物 | 形状 | 说明 |
|------|------|------|
| 编码器隐藏状态 `H` | `(T_enc, d_h)` | 如果是 BiLSTM，`d_h = 2 * d_hidden` |
| 解码器隐藏状态 `s_{t-1}` | `(d_s,)` | 一个向量 |
| 注意力分数 `e_{t,i}` | 标量 | 每个编码器位置一个 |
| 注意力权重 `α_{t,i}` | 标量 | 对所有 `i` 进行 softmax 后 |
| 上下文向量 `c_t` | `(d_h,)` | 与编码器状态形状相同 |

**Bahdanau（加性）评分。** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`。

- `s_{t-1}` 形状为 `(d_s,)`，`h_i` 形状为 `(d_h,)`。
- `W_a` 形状为 `(d_attn, d_s)`。`U_a` 形状为 `(d_attn, d_h)`。
- tanh 内部的求和形状为 `(d_attn,)`。
- `v_α` 形状为 `(d_attn,)`。与 `v_α` 的内积将结果压缩为标量。**这就是 `v_α` 的作用。** 它不是魔法，而是将注意力维度向量投影为标量分数的操作。

**Luong（乘性）评分。** 三种变体：

- `dot`：`e_{t,i} = s_t^T * h_i`。要求 `d_s == d_h`。硬约束。如果你的编码器是双向的，请跳过。
- `general`：`e_{t,i} = s_t^T * W * h_i`，其中 `W` 形状为 `(d_s, d_h)`。消除了等维约束。
- `concat`：本质上是 Bahdanau 形式。由于前两种更廉价，所以很少使用。

**一个值得命名的 Bahdanau/Luong 陷阱。** Bahdanau 使用 `s_{t-1}`（生成当前词*之前*的解码器状态）。Luong 使用 `s_t`（*之后*的状态）。混淆它们会产生微妙的错误梯度，极难调试。选择一篇论文并坚持其约定。

```figure
attention-heatmap
```

## 构建实现

### 步骤 1：加性（Bahdanau）注意力

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

对照上面的表格检查你的形状。`encoder_states` 形状为 `(T_enc, d_h)`。`projected_enc` 形状为 `(T_enc, d_attn)`。`projected_dec` 形状为 `(d_attn,)` 并进行广播。`combined` 形状为 `(T_enc, d_attn)`。`scores` 形状为 `(T_enc,)`。`weights` 形状为 `(T_enc,)`。`context` 形状为 `(d_h,)`。可以发布了。

### 步骤 2：Luong 点积和通用评分

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

每个函数三行代码。这就是 Luong 论文成功的原因。在大多数任务上准确率相同，但代码量少得多。

### 步骤 3：一个详细的数值示例

给定三个编码器状态（大致对应 "cat"、"sat"、"mat"）和一个与第一个编码器状态最匹配的解码器状态，注意力分布集中在位置 0。如果解码器状态转移到与最后一个匹配，注意力就会移到位置 2。上下文向量随之追踪。

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

第一行胜出。然后将解码器状态移近第三个编码器状态，观察权重如何变化。就是这样。注意力就是显式的对齐。

### 步骤 4：为什么这是通向 Transformer 的桥梁

将上述语言翻译成 Q/K/V：

- **查询（Query）** = 解码器状态 `s_{t-1}`
- **键（Key）** = 编码器状态（我们用来评分的对象）
- **值（Value）** = 编码器状态（我们加权求和的对象）

在经典注意力中，键和值是同一个东西。自注意力将它们分开：你可以用序列查询自身，对 K 和 V 使用不同的学习投影。多头注意力使用不同的学习投影并行运行。Transformer 多次堆叠整个阶段并去掉了 RNN。

数学是一样的。形状是一样的。从 Bahdanau 注意力到缩放点积注意力的教学跳跃主要是符号的变化。

## 使用它

PyTorch 和 TensorFlow 直接提供了注意力实现。

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

这就是一个 Transformer 注意力层。查询批次 5 个位置，键/值批次 10 个位置，每个 128 维，8 个头。`output` 是新的上下文增强查询。`weights` 是你可以可视化的 5x10 对齐矩阵。

### 经典注意力仍然重要的场景

- 教学。单头、单层、基于 RNN 的版本让每个概念都清晰可见。
- Transformer 无法适配的设备端序列任务。
- 2014-2017 年的任何论文。不了解 Bahdanau 的约定，你会误读这些论文。
- 机器翻译中的细粒度对齐分析。即使在 Transformer 模型上，原始注意力权重也是可解释性工具，阅读它们需要知道它们是什么。

### 注意力权重即解释的陷阱

注意力权重看起来可解释。它们是跨位置求和为一的权重；你可以绘制它们；高权重意味着"关注了这个位置"。审稿人喜欢它们。

它们并不像看起来那么可解释。Jain 和 Wallace（2019）表明，在某些任务上，注意力分布可以被任意排列和替换，而不会改变模型预测。在没有消融或反事实检查的情况下，永远不要将注意力权重作为推理的证据。

## 交付

保存为 `outputs/prompt-attention-shapes.md`：

```markdown
---
name: attention-shapes
description: Debug shape bugs in attention implementations.
phase: 5
lesson: 10
---

Given a broken attention implementation, you identify the shape mismatch. Output:

1. Which matrix has the wrong shape. Name the tensor.
2. What its shape should be, derived from (d_s, d_h, d_attn, T_enc, T_dec, batch_size).
3. One-line fix. Transpose, reshape, or project.
4. A test to catch regressions. Typically: assert `output.shape == (batch, T_dec, d_h)` and `weights.shape == (batch, T_dec, T_enc)` and `weights.sum(dim=-1) close to 1`.

Refuse to recommend fixes that silently broadcast. Broadcast-hiding bugs surface later as silent accuracy degradation, the worst kind of attention bug.

For Bahdanau confusion, insist the decoder input is `s_{t-1}` (pre-step state). For Luong, `s_t` (post-step state). For dot-product, flag dimension mismatch between query and key as the most common first-time error.
```

## 练习

1. **简单。** 实现 `softmax` 掩码，使编码器中的填充词元获得零注意力权重。在具有可变长度序列的批次上测试。
2. **中等。** 将多头注意力添加到 Luong `general` 形式中。将 `d_h` 分成 `n_heads` 组，每个头运行注意力，然后拼接。验证单头情况与你之前的实现匹配。
3. **困难。** 在第 09 课的玩具复制任务上训练带有 Bahdanau 注意力的 GRU 编码器-解码器。绘制准确率与序列长度的关系图。与无注意力基线进行比较。你应该看到随着长度增加，差距扩大，证实注意力解除了瓶颈。

## 关键术语

| 术语 | 人们怎么说 | 它的实际含义 |
|------|-----------|-------------|
| 注意力（Attention） | 看东西 | 值序列的加权平均，权重由查询-键相似度计算得出。 |
| 查询、键、值（Query, Key, Value） | QKV | 三个投影：Q 提问，K 是匹配对象，V 是返回对象。 |
| 加性注意力（Additive attention） | Bahdanau | 前馈评分：`v^T tanh(W q + U k)`。 |
| 乘性注意力（Multiplicative attention） | Luong 点积/通用 | 评分是 `q^T k` 或 `q^T W k`。更廉价，在大多数任务上准确率相同。 |
| 对齐矩阵（Alignment matrix） | 那张漂亮的图 | 注意力权重的 `(T_dec, T_enc)` 网格。阅读它可以看到模型关注了什么。 |

## 延伸阅读

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 原始论文。
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) — 三种评分变体及其比较。
- [Jain and Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186) — 可解释性注意事项。
- [Dive into Deep Learning — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) — 使用 PyTorch 的可运行教程。
