# 完整 Transformer — 编码器 + 解码器

> 注意力是主角。其他一切——残差、归一化、前馈、交叉注意力——都是让你能将其深层堆叠的脚手架。

**类型:** 构建
**语言:** Python
**前置知识:** 第 7 阶段 · 02（自注意力）、第 7 阶段 · 03（多头注意力）、第 7 阶段 · 04（位置编码）
**时间:** 约 75 分钟

## 问题所在

单个注意力层是特征提取器，不是模型。每层一次矩阵乘法对语言来说容量不够。你需要深度——而没有正确的管道，深度会崩溃。

2017 年 Vaswani 论文打包了六个设计决策，将一个注意力层变成了可堆叠的块。此后每个 Transformer——仅编码器（BERT）、仅解码器（GPT）、编码器-解码器（T5）——都继承相同的骨架。2026 年，这些块已被精炼（RMSNorm、SwiGLU、pre-norm、RoPE），但骨架相同。

本课讲骨架。后续课程将其专门化——06 用于编码器，07 用于解码器，08 用于编码器-解码器。

## 核心概念

![编码器和解码器块内部连线](../assets/full-transformer.svg)

### 六个组件

1. **嵌入 + 位置信号。** Token → 向量。位置通过 RoPE（现代）或正弦（经典）注入。
2. **自注意力。** 每个位置关注其他所有位置。在解码器中被掩码。
3. **前馈网络 (FFN)。** 逐位置两层 MLP：`W_2 · activation(W_1 · x)`。默认扩展比 4 倍。
4. **残差连接。** `x + sublayer(x)`。没有这个，梯度在约 6 层后消失。
5. **层归一化。** `LayerNorm` 或 `RMSNorm`（现代）。稳定残差流。
6. **交叉注意力（仅解码器）。** 查询来自解码器，键和值来自编码器输出。

观察一个向量流过一个块：注意力跨位置混合，残差将它向前传递，FFN 变换它，归一化保持流稳定。

```figure
transformer-block
```

### 编码器块（BERT、T5 编码器使用）

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── 残差 ──────┘
```

编码器是双向的。无掩码。所有位置看到所有位置。

### 解码器块（GPT、T5 解码器使用）

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

解码器每块有三个子层。中间的——交叉注意力——是信息从编码器流向解码器的唯一位置。在纯解码器架构（GPT）中，交叉注意力被省略，只有掩码自注意力 + FFN。

### Pre-norm vs post-norm

原始论文：`x + sublayer(LN(x))` vs `LN(x + sublayer(x))`。Post-norm 在 2019 年左右失宠——没有精心预热很难深度训练。Pre-norm（子层*之前*的 `LN`）是 2026 年默认：Llama、Qwen、GPT-3+、Mistral 全部使用。

### 2026 年现代化块

Vaswani 2017 发布了 LayerNorm + ReLU。现代栈替换了两者。生产块的实际样子：

| 组件 | 2017 | 2026 |
|------|------|------|
| 归一化 | LayerNorm | RMSNorm |
| FFN 激活 | ReLU | SwiGLU |
| FFN 扩展 | 4 倍 | 2.6 倍（SwiGLU 使用三个矩阵，总参数匹配） |
| 位置 | 正弦绝对 | RoPE |
| 注意力 | 完整 MHA | GQA（或 MLA） |
| 偏置项 | 有 | 无 |

RMSNorm 去掉了 LayerNorm 的均值中心化（少一次减法），节省计算且经验上至少同样稳定。SwiGLU (`Swish(W1 x) ⊙ W3 x`) 在 Llama、PaLM 和 Qwen 论文中一致优于 ReLU/GELU FFN 约 0.5 点 ppl。

### 参数量

对于一个 `d_model = d` 且 FFN 扩展 `r` 的块：

- MHA: `4 · d²` (Q、K、V、O 投影)
- FFN (SwiGLU): `3 · d · (r · d)` ≈ `3rd²`
- 归一化：可忽略

在 `d = 4096, r = 2.6, layers = 32`（大约 Llama 3 8B）下，总计：`32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B 参数每层 × 32 ≈ 7B`（加嵌入和头）。与已发布计数匹配。

## 动手实现

### 第一步：构建块

使用第 03 课的微型 `Matrix` 类（复制到此文件以保持独立）：

- `layer_norm(x, eps=1e-5)` — 减均值，除以标准差。
- `rms_norm(x, eps=1e-6)` — 除以 RMS。无均值减法。
- `gelu(x)` 和 `silu(x) * W3 x` (SwiGLU)。
- `ffn_swiglu(x, W1, W2, W3)`。
- `encoder_block(x, params)` 和 `decoder_block(x, enc_out, params)`。

参见 `code/main.py` 的完整连线。

### 第二步：连线一个 2 层编码器和 2 层解码器

堆叠它们。将编码器输出传入每个解码器交叉注意力。在输出投影之前添加最终 LN。

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### 第三步：在玩具示例上运行前向传播

传入 6 个 token 的源和 5 个 token 的目标。验证输出形状为 `(5, vocab)`。无训练——本课是关于架构，不是损失。

### 第四步：替换为 RMSNorm + SwiGLU

用 RMSNorm 和 SwiGLU 替换 LayerNorm 和 ReLU-FFN。确认形状仍然匹配。这是通过一次函数替换实现的 2026 年现代化。

## 使用场景

PyTorch/TF 参考实现：`nn.TransformerEncoderLayer`、`nn.TransformerDecoderLayer`。但大多数 2026 年生产代码自行编写块，因为：

- Flash Attention 在注意力内部调用，而非通过 `nn.MultiheadAttention`。
- GQA / MLA 不在标准库参考中。
- RoPE、RMSNorm、SwiGLU 不是 PyTorch 默认。

HF `transformers` 有你应该阅读的干净参考块：`modeling_llama.py` 是 2026 年标准的仅解码器块。约 500 行，值得通读一次。

**编码器 vs 解码器 vs 编码器-解码器——何时选择：**

| 需求 | 选择 | 示例 |
|------|------|------|
| 分类、嵌入、文本问答 | 仅编码器 | BERT、DeBERTa、ModernBERT |
| 文本生成、聊天、代码、推理 | 仅解码器 | GPT、Llama、Claude、Qwen |
| 结构化输入 → 结构化输出（翻译、摘要） | 编码器-解码器 | T5、BART、Whisper |

解码器赢得了语言领域，因为它扩展最干净且同时处理理解和生成。当输入有清晰的"源序列"身份（翻译、语音识别、结构化任务）时，编码器-解码器仍然是最佳选择。

## 交付使用

参见 `outputs/skill-transformer-block-reviewer.md`。该技能对照 2026 年默认值审查新的 Transformer 块实现，标记缺失部分（pre-norm、RoPE、RMSNorm、GQA、FFN 扩展比）。

## 练习

1. **简单。** 在 `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True` 下计算 encoder_block 的参数量。通过实现块并使用 `sum(p.numel() for p in block.parameters())` 验证。
2. **中等。** 从 post-norm 切换到 pre-norm。初始化两者，在随机输入上堆叠 12 层后测量激活范数。Post-norm 的激活应该爆炸；pre-norm 的应该保持有界。
3. **困难。** 在玩具复制任务（复制反转的 `x`）上实现 4 层编码器-解码器。训练 100 步。报告损失。替换为 RMSNorm + SwiGLU + RoPE——损失是否下降？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 块 (Block) | "一个 Transformer 层" | norm + 注意力 + norm + FFN 的堆叠，包裹在残差连接中。 |
| 残差 (Residual) | "跳跃连接" | `x + f(x)` 输出；使梯度能流过深层堆叠。 |
| Pre-norm | "归一化在前，不在后" | 现代: `x + sublayer(LN(x))`。无需预热即可训练更深。 |
| RMSNorm | "去掉均值的 LayerNorm" | 除以 RMS；少一次操作，同样稳定。 |
| SwiGLU | "大家都切换到的 FFN" | `Swish(W1 x) ⊙ W3 x → W2`。在 LM ppl 上优于 ReLU/GELU。 |
| 交叉注意力 | "解码器如何看到编码器" | Q 来自解码器、K/V 来自编码器输出的 MHA。 |
| FFN 扩展 | "中间 MLP 有多宽" | 隐藏大小与 d_model 的比值，通常为 4（LayerNorm）或 2.6（SwiGLU）。 |
| 无偏置 | "去掉 +b 项" | 现代栈省略线性层的偏置；轻微 ppl 改善，更小模型。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原始块规范。
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — pre-norm 深层优于 post-norm 的原因。
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm。
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — SwiGLU 论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 2026 年标准仅解码器块。
