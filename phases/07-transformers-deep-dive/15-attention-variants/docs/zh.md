# 注意力变体 — 滑动窗口、稀疏、差分

> 完整注意力是一个圆。每个 token 看到每个 token，内存付出代价。四种变体弯曲圆的形状，回收一半成本。

**类型:** 构建
**语言:** Python
**前置知识:** 第 7 阶段 · 02（自注意力）、第 7 阶段 · 03（多头）、第 7 阶段 · 12（KV 缓存 / Flash Attention）
**时间:** 约 60 分钟

## 问题所在

完整注意力在序列长度上花费 `O(N²)` 内存和 `O(N²)` 计算。对于 128K 上下文的 Llama 3 70B，那是每层 160 亿个注意力条目，乘以 80 层。Flash Attention（第 12 课）隐藏了 `O(N²)` 激活内存，但不改变算术成本——每个 token 仍然关注其他每个 token。

三类变体改变了注意力矩阵本身的拓扑：

1. **滑动窗口注意力 (SWA)。** 每个 token 只关注固定窗口的邻居，而非整个前缀。内存和计算降到 `O(N · W)`，W 是窗口大小。Gemma 2/3、Mistral 7B 的第一层、Phi-3-Long。
2. **稀疏/块注意力。** 只有选定的对 `(i, j)` 被评分；其余强制零权重。Longformer、BigBird、OpenAI 稀疏 Transformer。
3. **差分注意力。** 用独立的 Q/K 投影计算两个注意力图，减去一个。消除将权重泄漏到前几个 token 的"注意力汇聚"。微软的 DIFF Transformer (2024)。

这些共存。2026 年前沿模型通常混合它们：大多数层是 SWA-1024，每第五层是全局完整注意力，少数是差分头清理检索。Gemma 3 的 5:1 SWA 到全局比率是当前教科书默认。

## 核心概念

### 滑动窗口注意力 (SWA)

位置 `i` 的每个查询只关注 `[i - W, i]`（因果 SWA）或 `[i - W/2, i + W/2]`（双向）中的位置。窗口外的 token 在分数矩阵中得 `-inf`。

```
完整因果:           滑动窗口 (W=4):
位置 0-7            位置 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

对于 `N = 8192` 和 `W = 1024`，分数矩阵期望有 1024 × 8192 非零行——8 倍减少。

**KV 缓存随 SWA 缩小。** 每层只需保留最后 `W` 个 token 的 K 和 V。对于 Gemma-3 配置（1024 窗口，128K 上下文），KV 缓存缩小 128 倍。

**质量成本。** 纯 SWA Transformer 在长距离检索上挣扎。修复：SWA 层与全注意力层交错。Gemma 3 使用 5:1 SWA:全局。Mistral 7B 使用因果 SWA 栈，信息通过重叠窗口"向前流动"——每层将有效感受野扩展 `W`，`L` 层后模型可以关注 `L × W` 个 token。

### 稀疏/块注意力

提前选取 `N × N` 稀疏模式。三种标准形状：

- **局部 + 步幅（OpenAI 稀疏 Transformer）。** 关注最后 `W` 个 token 加上之前每 `stride` 个 token。在 `O(N · sqrt(N))` 计算下捕获局部和远距离。
- **Longformer / BigBird。** 局部窗口 + 一小部分全局 token（如 `[CLS]`）关注所有人并被所有人关注 + 随机稀疏链接。经验上匹配质量下 2 倍上下文。
- **原生稀疏注意力（DeepSeek，2025）。** 学习哪些 `(Q, K)` 块重要；在核级别跳过零块。FlashAttention 兼容。

稀疏注意力是核工程故事。数学简单（掩码分数矩阵）；胜利来自从不将零条目加载到 SRAM。FlashAttention-3 和 2026 年 FlexAttention API 使自定义稀疏模式在 PyTorch 中成为一等公民。

### 差分注意力 (DIFF Transformer, 2024)

常规注意力有"注意力汇聚"问题：softmax 强制每行和为 1，所以不想关注任何东西的 token 将权重倾倒在第一个 token（或前几个）上。这偷走了本应给真实内容的容量。

差分注意力通过计算**两个**注意力图并相减来修复：

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

其中 `λ` 是可学习标量（通常 0.5-0.8）。A1 捕获真实内容权重；A2 捕获汇聚。相减抵消汇聚，将权重重新分配给相关 token。

报告结果（微软 2024）：困惑度降低 5-10%，相同训练长度下有效上下文延长 1.5-2 倍，更锐利的针在干草堆检索。

### 变体比较

| 变体 | 计算 | KV 缓存 | 质量 vs 完整 | 生产使用 |
|------|------|---------|-------------|----------|
| 完整注意力 | O(N²) | O(N) 每层 | 基线 | 每个模型的默认层 |
| SWA (窗口 1024) | O(N·W) | O(W) 每层 | -0.1 ppl，配全局层好 | Gemma 2/3、Phi-3-Long |
| 局部 + 步幅稀疏 | O(N·√N) | 混合 | 类似 SWA | OpenAI 稀疏 Transformer、Longformer |
| BigBird (局部 + 全局 + 随机) | O(N) 近似 | 混合 | 匹配 2 倍上下文下的完整 | 早期长上下文 BERT |
| 原生稀疏 (DeepSeek-V3.2) | O(N · 活跃比例) | O(N) | 0.05 ppl 以内 | DeepSeek-V3.2, 2025 |
| 差分 | O(2·N²) | O(2N) | -5 到 -10% ppl | DIFF Transformer, 2026 早期模型 |

## 动手实现

参见 `code/main.py`。我们实现因果掩码比较器，在玩具序列上并排展示完整、SWA、局部+步幅和差分注意力。

### 第一步：完整因果掩码（基线）

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

第 07 课基线。下三角；对角线上方零权重。

### 第二步：滑动窗口因果掩码

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

一个参数——`window`。当 `window >= n`，你恢复完整因果注意力。当 `window = 1`，每个 token 只关注自身。

### 第三步：局部 + 步幅稀疏掩码

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

密集局部窗口加上回到序列开头的每 `stride` 个 token。额外层的感受野以对数步增长。

### 第四步：差分注意力

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

两次注意力传递，用可学习混合系数相减。代码中我们比较单注意力 vs 差分的注意力汇聚热力图，观察汇聚坍缩。

### 第五步：KV 缓存大小

在 `N = 131072` 下打印每种变体每层的缓存大小。SWA 和稀疏变体下降 10-100 倍。差分翻倍。有意识地支付你的内存账单。

## 使用场景

2026 年生产模式：

```python
from transformers import AutoModelForCausalLM
# Gemma 3 混合 SWA (window=1024) 和全局层，5:1。
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

PyTorch 2.5+ 中的 FlexAttention 接受掩码函数：

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

这编译成自定义 Triton 核。对常见模式在 FlashAttention-3 速度的 10% 以内，掩码函数是 Python 可调用对象。

**何时选择各方案：**

- **纯完整注意力** — 每层最多约 16K 上下文，或检索质量至关重要时。
- **SWA + 全局混合** — 长上下文 (>32K)，训练和推理内存受限。32K 以上的 2026 年默认。
- **稀疏块注意力** — 自定义核，自定义模式。保留给专门工作负载（检索、音频）。
- **差分注意力** — 任何注意力汇聚污染有害的工作负载（长上下文 RAG、针在干草堆）。

## 交付使用

参见 `outputs/skill-attention-variant-picker.md`。该技能根据目标上下文长度、检索需求和训练/推理计算配置为新模型选择注意力拓扑。

## 练习

1. **简单。** 运行 `code/main.py`。验证 `window=4` 的 SWA 将每行最后 4 个 token 之外的一切清零。验证 `window=n` 逐位重现完整因果注意力。
2. **中等。** 在第 07 课毕业项目上实现 `window=1024` 的因果 SWA。在 tinyshakespeare 上训练 1,000 步。验证损失 vs 完整注意力回归多少？峰值内存下降多少？
3. **困难。** 在毕业项目模型中实现 Gemma-3 风格的 5:1 层混合（5 SWA，1 全局）。在匹配参数下比较损失、内存和生成质量 vs 纯 SWA 和纯全局基线。
4. **困难。** 实现带每头可学习 `λ` 的差分注意力。在合成检索任务（一个针，2,000 个干扰项）上训练。在匹配参数下测量检索准确率 vs 单注意力基线。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 滑动窗口注意力 (SWA) | "局部注意力" | 每个查询关注其最后 `W` 个 token；KV 缓存缩小到 `O(W)`。 |
| 有效感受野 | "模型看到多远" | 在窗口 `W` 的 `L` 层 SWA 栈中，最多 `L × W` 个 token。 |
| Longformer / BigBird | "局部 + 全局 + 随机" | 带少数始终参与的全局 token 的稀疏模式；早期长上下文方法。 |
| 原生稀疏注意力 | "DeepSeek 的核技巧" | 学习块级稀疏性；在核级别跳过零块同时保持质量。 |
| 差分注意力 | "两个图，一个减去" | DIFF Transformer：减去可学习 `λ` 乘以第二个注意力图以消除注意力汇聚。 |
| 注意力汇聚 | "权重泄漏到 token 0" | softmax 归一化强制行和为 1；无信息查询将权重倾倒在位置 0。 |
| FlexAttention | "掩码即 Python" | PyTorch 2.5+ API，将任意掩码函数编译成 FlashAttention 形状核。 |
| 层类型混合 | "5:1 SWA 到全局" | 在栈中交错稀疏和全注意力层，以更低内存保持质量。 |

## 延伸阅读

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — 标准滑动窗口 + 全局 token 论文。
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — 局部 + 全局 + 随机。
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — OpenAI 的局部+步幅模式。
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) — 1:1 SWA:全局混合。
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) — 现在教科书默认的 5:1 混合，window=1024。
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) — DIFF Transformer 论文。
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) — DeepSeek-V3.2 的可学习稀疏注意力。
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) — "使用场景"中掩码即可调用模式的 API 参考。
