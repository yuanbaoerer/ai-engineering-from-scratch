# 从零构建 Transformer — 毕业项目

> 十三课。一个模型。没有捷径。

**类型:** 构建
**语言:** Python
**前置知识:** 第 7 阶段 · 01 到 13。不要跳过。
**时间:** 约 120 分钟

## 问题所在

你已经读了每篇论文。你实现了注意力、多头分割、位置编码、编码器和解码器块、BERT 和 GPT 损失、MoE、KV 缓存。现在让它们在真实任务上协同工作。

毕业项目：在字符级语言建模任务上端到端训练小型仅解码器 Transformer。它读莎士比亚。它生成新的莎士比亚。它小到可以在笔记本电脑上 10 分钟内训练。它正确到替换更大数据集和更长训练能得到真正的 LM。

这是课程的"nanoGPT"。它不是原创的——Karpathy 2023 年的 nanoGPT 教程是每个学生至少写一次的参考实现。我们借鉴其形状并围绕我们所学的内容重新设计。

## 核心概念

![从零构建 Transformer 块图](../assets/capstone.svg)

架构，带注释：

```
输入 token (B, N)
   │
   ▼
token 嵌入 + 位置嵌入  ◀── 第 04 课 (RoPE 选项)
   │
   ▼
┌──── 块 × L ────────────────────┐
│  RMSNorm                          │  ◀── 第 05 课
│  MultiHeadAttention (causal)      │  ◀── 第 03 + 07 课 (因果掩码)
│  残差                              │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── 第 05 课
│  残差                              │
└────────────────────────────────── ┘
   │
   ▼
最终 RMSNorm
   │
   ▼
lm_head (绑定到 token 嵌入)
   │
   ▼
logits (B, N, V)
   │
   ▼
偏移一位交叉熵                    ◀── 第 07 课
```

### 我们交付什么

- `GPTConfig` — 一个配置所有超参数的地方。
- `MultiHeadAttention` — 因果、批处理，带可选 Flash 风格路径（PyTorch 的 `scaled_dot_product_attention`）。
- `SwiGLUFFN` — 现代 FFN。
- `Block` — pre-norm，残差包裹的注意力 + FFN。
- `GPT` — 嵌入、堆叠块、LM 头、generate()。
- 训练循环带 AdamW、余弦 LR、梯度裁剪。
- 莎士比亚文本上的字符级分词器。

### 我们不交付什么

- RoPE — 在第 04 课概念性实现。这里我们使用可学习位置嵌入以保持简单。练习要求你替换为 RoPE。
- 生成时的 KV 缓存 — 每个生成步对整个前缀重新计算注意力。更慢但更简单。练习要求你添加 KV 缓存。
- Flash Attention — PyTorch 2.0+ 在输入匹配时自动调度；我们使用 `F.scaled_dot_product_attention`。
- MoE — 每块单个 FFN。你在第 11 课见过 MoE。

### 目标指标

在 Mac M2 笔记本上，4 层、4 头、d_model=128 的 GPT 在 `tinyshakespeare.txt` 上训练 2,000 步：

- 训练损失从约 4.2（随机）收敛到约 1.5，约 6 分钟。
- 采样输出看起来像莎士比亚：古词、换行、"ROMEO:" 等专有名词涌现。
- 验证损失（保留文本最后 10%）紧密跟踪训练损失；此规模/预算下无过拟合。

## 动手实现

本课使用 PyTorch。安装 `torch`（CPU 版即可）。参见 `code/main.py`。脚本处理：

- 如果缺失则下载 `tinyshakespeare.txt`（或读取本地副本）。
- 字节级字符分词器。
- 90/10 训练/验证分割。
- 支持硬件上的 bf16 自动混合精度训练循环。
- 训练完成后的采样。

### 第一步：数据

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 个唯一字符。微型词表。4 字节 vocab_size。无 BPE，无分词器戏剧。

### 第二步：模型

参见 `code/main.py`。块是第 05 课的教科书——pre-norm、RMSNorm、SwiGLU、因果 MHA。4/4/128 的参数计数：约 800K。

### 第三步：训练循环

获取长度 256 token 窗口的随机批次。前向。偏移一位交叉熵。反向。AdamW 步。记录。重复。

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### 第四步：采样

给定提示，重复前向，从 top-p logits 采样，追加，继续。500 token 后停止。

### 第五步：读取输出

2,000 步后：

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

不是莎士比亚。但形似莎士比亚。约 800K 参数和笔记本上 6 分钟的明显胜利。

## 使用场景

这个毕业项目是参考架构。三个扩展以将其推向真实：

1. **替换分词器。** 使用 BPE（如 `tiktoken.get_encoding("cl100k_base")`）。词表大小从 65 跳到约 50,000。模型容量需要相应扩大。
2. **在更大语料上训练。** 使用 `OpenWebText` 或 `fineweb-edu` (HuggingFace)。单 A100 上 125M 参数 GPT 的 10B token 约需 24 小时。
3. **添加 RoPE + KV 缓存 + Flash Attention。** 下面的练习逐步引导你。

最终变成 125M 参数的 GPT，生成流畅英语。不是前沿模型。但相同的代码路径——只是更大——是 Karpathy、EleutherAI 和 Allen Institute 在 2026 年用来训练研究检查点的。

## 交付使用

参见 `outputs/skill-transformer-review.md`。该技能审查从零构建的 Transformer 实现，检查所有 13 前课的正确性。

## 练习

1. **简单。** 运行 `code/main.py`。验证你训练模型的最终步验证损失低于 2.0。将 `max_steps` 从 2,000 改到 5,000——验证损失是否持续改善？
2. **中等。** 用 RoPE 替换可学习位置嵌入。在 `MultiHeadAttention` 内部对 Q 和 K 应用旋转。训练并验证验证损失至少同样低。
3. **中等。** 在采样循环中实现 KV 缓存。有和无缓存下生成 500 token。笔记本上墙钟应改善 5-20 倍。
4. **困难。** 为模型添加第二个头预测下一个加一 token（MTP——来自 DeepSeek-V3 的多 token 预测）。联合训练。有帮助吗？
5. **困难。** 用 4 专家 MoE 替换每块的单个 FFN。路由器 + top-2 路由。在匹配活跃参数下观察验证损失变化。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| nanoGPT | "Karpathy 的教程 repo" | 最小仅解码器 Transformer 训练代码，约 300 行；标准参考。 |
| tinyshakespeare | "标准玩具语料" | 约 1.1 MB 文本；2015 年以来每个字符 LM 教程使用。 |
| 绑定嵌入 | "共享输入/输出矩阵" | LM 头权重 = token 嵌入矩阵的转置；节省参数，改善质量。 |
| bf16 自动混合精度 | "训练精度技巧" | 前向/反向用 bf16，优化器状态用 fp32；2021 年以来的标准。 |
| 梯度裁剪 | "阻止尖峰" | 全局梯度范数上限 1.0；防止训练爆炸。 |
| 余弦 LR 调度 | "2020+ 默认" | LR 线性升温（预热）然后余弦衰减到峰值的 10%。 |
| MFU | "模型 FLOP 利用率" | 实现 FLOPs / 理论峰值；2026 年密集 40%、MoE 30% 算强。 |
| 验证损失 | "保留损失" | 模型从未见过的数据上的交叉熵；过拟合检测器。 |

## 延伸阅读

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) — 经典带注释实现。
