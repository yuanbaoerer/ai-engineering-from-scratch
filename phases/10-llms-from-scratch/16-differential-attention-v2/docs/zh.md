# 差分注意力 (V2)

> Softmax 注意力会将少量概率分散到每一个不相关的词元上。在 10 万个词元的上下文中，这些噪声会累积起来并淹没信号。差分 Transformer (Differential Transformer, Ye 等, ICLR 2025) 通过计算两个 softmax 的差值来解决这个问题，从而减去共享的噪声基底。DIFF V2 (Microsoft, 2026 年 1 月) 是面向生产环境的重写版本：解码延迟与基线 Transformer 持平，无需自定义内核，兼容 FlashAttention。本节课从 V1 到 V2 完整讲解，并提供一个可在标准库 Python 中运行的差分操作玩具实现。

**类型:** 构建
**语言:** Python (标准库)
**前置知识:** Phase 7 · 02 (自注意力), Phase 7 · 15 (注意力变体), Phase 10 · 14 (架构概览)
**时间:** ~60 分钟

## 学习目标

- 准确阐述 softmax 注意力为何存在噪声基底，以及它为何随上下文长度增长。
- 推导差分注意力公式，并解释减法为何能消除共享的噪声分量同时保留信号。
- 梳理 V1 到 V2 的差异：什么变快了、什么简化了、什么更稳定了，以及每项改动为何对生产级预训练是必要的。
- 用纯 Python 从零实现差分注意力，并在合成信号加噪声查询上实证验证噪声消除特性。

## 问题背景

标准 softmax 注意力有一个数学特性，在规模化时会变成运营上的麻烦。对于查询 `q`，注意力权重为 `softmax(qK^T / sqrt(d))`。Softmax 永远无法产生精确的零值——每一个不相关的词元都会获得一些正的质量。这部分残余质量就是噪声，并且它会随上下文长度而缩放。在 128k 词元时，即使每个不相关的词元只获得 0.001% 的概率，127,999 个词元合计也贡献了约 12% 的总量。模型必须学会绕过这个随上下文增长的噪声基底。

从经验上看，这表现为注意力头之间的干扰：长上下文 RAG 中的幻觉引用、10 万词元检索任务中的"中间丢失" (lost-in-the-middle) 失败，以及超过 32k 后的"大海捞针" (needle-in-haystack) 基准测试中微妙的准确率下降。Differential Transformer 论文 (arXiv:2410.05258, ICLR 2025) 测量了这一差距：DIFF Transformer 在同等规模基线上实现了更低的困惑度、更高的长上下文准确率和更少的幻觉。

DIFF V1 有三个问题使其无法进入前沿预训练流水线。它的值缓存 (value cache) 在每个解码步骤中需要加载两次，它需要打破 FlashAttention 兼容性的自定义 CUDA 内核，并且它的逐头 RMSNorm 在 70B 以上规模的长时间训练中导致不稳定。DIFF V2 (Microsoft unilm 博客, 2026 年 1 月 20 日) 修复了全部三个问题。本节课将讲解两个版本，构建差分操作符，并在玩具查询上基准测试噪声消除效果。

## 核心概念

### Softmax 的噪声基底

对于查询 `q` 和键 `K = [k_1, ..., k_N]`，注意力权重为：

```
w_i = exp(q . k_i / sqrt(d)) / sum_j exp(q . k_j / sqrt(d))
```

没有任何 `w_i` 会真正为零。如果 `k_i` 与 `q` 完全无关，分数 `q . k_i` 不为 0——它在零附近波动，方差为 `||q||^2 / d`。经过 softmax 归一化后，每个无关词元仍然对加权和贡献 `O(1/N)`。无关词元的总贡献为 `O((N-1)/N) = O(1)`——这不是一个小量。

模型想要的是类似硬 top-k 的效果：在匹配词元上权重高，其他地方接近零。Softmax 本身太平滑，无法直接做到这一点。

### 差分的思想

将每个头的 Q 和 K 投影分成两部分：Q = (Q_1, Q_2) 和 K = (K_1, K_2)。计算两个注意力图：

```
A_1 = softmax(Q_1 K_1^T / sqrt(d))
A_2 = softmax(Q_2 K_2^T / sqrt(d))
```

输出：

```
DiffAttn = (A_1 - lambda * A_2) V
```

减法消除了两个图共享的任何噪声分布。如果两个图在 127k 个无关词元上都有大致均匀的权重（在随机初始化时它们会如此），这些就会相互抵消。信号——在少数真正相关词元上的尖峰权重——只有当它以相同幅度同时出现在两个图中时才会抵消，而一旦模型训练完成，这种情况就不会发生。

`lambda` 是每个头的可学习标量，参数化为 `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`。它可以是负数。`lambda_init` 默认为一个小的正数，如 0.8。

### 为什么这类似于头戴式降噪

想象两个有噪声的麦克风录制同一个声音。两者都拾取了说话者加上相关的背景噪声。将一个减去另一个，共享的噪声就会消失。声音得以保留，因为两个信号在相位或幅度上差异足够大，不会完全抵消。逐头 `lambda` 学习的就是这种平衡。

### V1 vs V2：差异对比

V1 保持了与基线 Transformer 相同的参数量。为了每个头获得两个查询，它将头维度减半。这损失了头的表达能力——更痛苦的是——将每个头的值缓存减半。解码时每个步骤必须加载值缓存两次（每个 softmax 分支一次）。结果：尽管参数量相同，解码速度却比基线慢。

V2 将查询头数量翻倍，保持 KV 头数量不变（从上投影中借用参数）。头维度与基线相同。减法之后，额外的维度被投影回降至与基线 Transformer 的 O_W 投影匹配。三件事同时发生：

1. 解码速度与基线持平（KV 缓存只加载一次）。
2. FlashAttention 无需改动即可运行（无需自定义内核）。
3. 解码时的算术强度上升（每从 HBM 加载一字节进行更多计算）。

V2 还移除了 V1 用于稳定减法的逐头 RMSNorm。在 70B 级别的预训练规模下，该 RMSNorm 会在训练后期导致不稳定。V2 用一个更简单的初始化方案替代它，无需额外模块即可保持训练稳定。

### 何时选用它

| 工作负载 | 收益 |
|----------|------|
| 长上下文 RAG (64k+) | 更干净的注意力图，更少的幻觉引用 |
| 大海捞针基准测试 | 超过 32k 后准确率显著提升 |
| 多文档问答 | 更少的跨文档干扰 |
| 8k 代码补全 | 边际收益，不值得架构改动 |
| 短对话 (< 4k) | 与基线基本无法区分 |

价值随上下文长度增长。在 4k 词元时，噪声基底足够小，标准注意力即可胜任。在 128k 时，它正在伤害你。

### 它如何与其他 2026 年的技术搭配

| 特性 | 与 DIFF V2 兼容？ |
|------|------------------|
| GQA | 是 (V2 增加 Q 头数量，而非 KV 头) |
| MLA (DeepSeek) | 原则上可以，尚无已发表论文将两者结合 |
| MoE | 是 (注意力与 MLP 块独立) |
| RoPE | 是 (保持不变) |
| YaRN / 长上下文缩放 | 是 (正是 DIFF 最能发挥作用的地方) |
| FlashAttention | V2 可以 (V1 不行) |
| 投机解码 | 是 (注意力改动对投机解码循环不可见) |

## 动手实现

`code/main.py` 用纯 Python 实现了差分注意力。一个具有已知信号加噪声结构的玩具查询让你可以直接测量噪声消除比率。

### 步骤 1：标准 softmax 注意力

标准库矩阵操作：列表的列表，手动矩阵乘法，通过减去最大值实现数值稳定性的 softmax。

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### 步骤 2：将 Q, K 分成两半

V1 风格：将头维度减半。V2 风格：保持头维度，将头数量翻倍。玩具实现采用 V1 风格以保证教学清晰——数学完全相同，只是簿记方式不同。

### 步骤 3：两个 softmax 分支 + 减法

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

注意：输出权重可以为负数。这没有问题——值缓存仍然可以处理带符号的贡献。随后的 V 投影会吸收符号。

### 步骤 4：噪声消除测量

构建一个长度为 1024 的合成序列。在已知位置放置信号词元，其余填充噪声。计算 (a) 标准 softmax 注意力在信号位置上的权重和 (b) 差分注意力权重。测量两者的信噪比。差分注意力可靠地产生高出 3 倍到 10 倍的信噪比，具体取决于两个分支被训练到多大程度的不同。

### 步骤 5：V1 vs V2 参数量核算

给定一个配置 (hidden=4096, heads=32, d_head=128)，打印：

- 基线 Transformer：Q, K, V 每个大小为 `hidden * hidden`，MLP 为 4 * hidden。
- DIFF V1：Q, K 每个大小为 `hidden * hidden`，V 大小为 `hidden * hidden` (不变)，内部头维度减半。增加逐头 `lambda` 参数 (O(heads * d_head))。
- DIFF V2：Q 大小为 `2 * hidden * hidden`，K 大小为 `hidden * hidden`，V 大小为 `hidden * hidden`。额外维度在 O_W 之前投影回降。增加相同的 `lambda` 参数。

玩具实现会测量 V2 的额外参数成本（每个注意力块大约 `hidden * hidden` 额外参数）并打印出来。

## 如何使用

截至 2026 年 4 月，DIFF V2 尚未在所有生产推理服务器中上线，但正在 vLLM 和 SGLang 中进行集成。与此同时，该模式已出现在：

- Microsoft 内部长上下文生产模型中。
- 多个面向 256k+ 上下文的开源模型训练运行的研究复现中。
- 将 DIFF 注意力与滑动窗口注意力在交替层中结合的混合架构中。

在 2026 年，你何时会考虑使用它：

- 从零开始训练一个目标有效上下文为 64k+ 的新模型。从一开始就添加差分注意力；后期重新训练成本高昂。
- 微调一个长上下文模型，其中"中间丢失"失败主导了你的评估。对 Q 投影进行 LoRA 可以近似 DIFF 结构。

何时不应使用：

- 你正在服务一个预训练的稠密模型，其长上下文性能稳定。在现有权重上，重新训练的成本很少能回本。
- 你的上下文始终在 16k 以下。噪声基底可以忽略不计。

## 交付物

本节课产出 `outputs/skill-diff-attention-integrator.md`。给定一个模型架构、目标上下文长度、幻觉特征和训练预算，它生成一份将差分注意力添加到新预训练运行或 LoRA 微调的集成计划。

## 练习题

1. 运行 `code/main.py`。验证差分注意力在合成查询上报告的信噪比高于标准 softmax 注意力。改变噪声幅度，展示标准注意力变得不可用的交叉点。

2. 计算从基线到 DIFF V1 以及从基线到 DIFF V2 的参数量差异，针对一个 7B 级别模型 (hidden=4096, heads=32, d_head=128, 32 层)。展示哪些组件增加了参数，哪些保持不变。

3. 阅读 DIFF V1 论文 (arXiv:2410.05258) 的第三节和 DIFF V2 Hugging Face 博客的第二节。用两句话解释为什么 V1 的逐头 RMSNorm 是必要的，以及为什么 V2 可以在不导致训练发散的情况下移除它。

4. 实现一个消融实验：计算 `lambda = 0` (纯第一个 softmax) 和 `lambda = 1` (完全减法) 时的差分注意力。在合成查询上，测量信噪比在扫描过程中的变化。找出使信噪比最大化的 `lambda`。

5. 将玩具实现扩展到 GQA + DIFF V2。选择 8 个 KV 头和 32 个 Q 头。展示 KV 缓存大小与具有相同 (8, 32) 配置的基线 GQA 模型相匹配。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| 差分注意力 (Differential attention) | "两个 softmax 相减" | 将 Q, K 分成两半，计算两个 softmax 图，将第二个（按 lambda 缩放）从第一个中减去，然后乘以 V |
| 噪声基底 (Noise floor) | "Softmax 的非零尾部" | Softmax 放在每个无关词元上的 O(1/N) 权重，在长上下文中累加为 O(1) |
| lambda | "减法缩放因子" | 逐头可学习标量，参数化为 `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`；可以为负数 |
| DIFF V1 | "ICLR 2025 版本" | 原始 Differential Transformer；将头维度减半以保持参数量，需要自定义内核，解码更慢 |
| DIFF V2 | "2026 年 1 月的修复版" | 将 Q 头翻倍而保持 KV 头；匹配基线解码速度，兼容 FlashAttention |
| 逐头 RMSNorm (Per-head RMSNorm) | "V1 的稳定器" | V1 在差分后应用的额外归一化；V2 将其移除以防止训练后期不稳定 |
| 信噪比 (Signal-to-noise ratio) | "多少注意力被浪费了" | 真正信号位置上的权重与无关位置平均权重的比率 |
| 中间丢失 (Lost in the middle) | "长上下文失败模式" | 检索准确率对长上下文中间文档下降的经验现象——差分注意力减少了这一问题 |
| 算术强度 (Arithmetic intensity) | "每加载一字节的 FLOPs" | V2 通过每次 KV 加载加倍查询来提升的解码比率；对内存受限的解码很重要 |

## 延伸阅读

- [Ye 等 — Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) — 原始论文，包含噪声消除理论和长上下文消融实验
- [Microsoft unilm — Differential Transformer V2 (Hugging Face 博客, 2026 年 1 月)](https://huggingface.co/blog/microsoft/diff-attn-v2) — 面向生产环境的重写，匹配基线解码，兼容 FlashAttention
- [Understanding Differential Transformer Unchains Pretrained Self-Attentions (arXiv:2505.16333)](https://arxiv.org/abs/2505.16333) — 理论分析，解释减法为何能恢复预训练注意力结构
- [Shared DIFF Transformer (arXiv:2501.17900)](https://arxiv.org/html/2501.17900) — 参数共享变体
- [Vaswani 等 — Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) — DIFF 所减去的基线 Transformer
- [Liu 等 — Lost in the Middle (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172) — 差分注意力所针对的长上下文基准测试
