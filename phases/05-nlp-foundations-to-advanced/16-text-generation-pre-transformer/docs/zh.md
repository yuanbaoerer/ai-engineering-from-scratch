# Transformer 之前的文本生成 — N-gram 语言模型

> 如果一个词令人意外，说明模型很差。困惑度（Perplexity）将意外量化为数字。平滑技术使它保持有限。

**类型：** 构建
**语言：** Python
**前置知识：** 第 5 阶段 · 01（文本处理），第 2 阶段 · 14（朴素贝叶斯）
**时间：** 约 45 分钟

## 问题背景

在 Transformer 之前，在 RNN 之前，在词嵌入之前，语言模型通过统计前 `n-1` 个词后面出现某个词的频率来预测下一个词。统计 "the cat" → "sat" 出现 47 次，"the cat" → "jumped" 出现 12 次，"the cat" → "refrigerator" 出现 0 次。归一化后得到概率分布。

这就是 n-gram 语言模型。从 1980 年到 2015 年，它驱动着每一个语音识别器、每一个拼写检查器和每一个基于短语的机器翻译系统。当你需要轻量级的设备端语言建模时，它至今仍在使用。

有趣的问题在于如何处理未见过的 n-gram。原始的基于计数的模型对任何未见过的序列赋予零概率，这是灾难性的，因为句子很长，几乎每个长句都包含至少一个未见过的序列。五十年的平滑研究解决了这个问题。Kneser-Ney 平滑是其成果，现代深度学习继承了其实证传统。

## 核心概念

![N-gram 模型：计数、平滑、生成](../assets/ngram.svg)

**N-gram 概率：** `P(w_i | w_{i-n+1}, ..., w_{i-1})`。固定 `n`（通常三元组用 3，四元组用 4）。根据计数计算：

```text
P(w | context) = count(context, w) / count(context)
```

**零计数问题。** 训练中未出现的任何 n-gram 概率为零。2007 年一项对 Brown 语料库的研究发现，即使是四元组模型，也有 30% 的测试集四元组在训练中未出现。不使用平滑就无法在任何真实文本上进行评估。

**平滑方法，按复杂度递增排列：**

1. **拉普拉斯（加一平滑）。** 每个计数加 1。简单，但对罕见事件效果很差。
2. **Good-Turing。** 基于频率的频率，将概率质量从高频事件重新分配给未见过的事件。
3. **插值（Interpolation）。** 将 n-gram、(n-1)-gram 等的估计值用可调权重组合。
4. **回退。** 如果 n-gram 计数为零，则回退到 (n-1)-gram。Katz 回退对其进行了形式化。
5. **绝对折扣（Absolute Discounting）。** 从所有计数中减去固定折扣 `D`，将概率重新分配给未见过的事件。
6. **Kneser-Ney。** 绝对折扣加上对低阶模型的巧妙选择：使用*延续概率*（一个词出现在多少种上下文中）而不是原始频率。

Kneser-Ney 的洞察非常深刻。"San Francisco" 是一个常见的二元组。一元组 "Francisco" 主要出现在 "San" 之后。朴素的绝对折扣给 "Francisco" 很高的一元组概率（因为计数高）。Kneser-Ney 注意到 "Francisco" 只出现在一种上下文中，因此相应地降低了其延续概率。结果：以 "Francisco" 结尾的新二元组得到了适当的低概率。

**评估：困惑度。** 在测试集上每个词的平均负对数似然的指数。越低越好。困惑度为 100 意味着模型在 100 个词中均匀选择时一样困惑。

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

```figure
ngram-backoff
```

## 动手实现

### 步骤 1：三元组计数

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

输入是分词后的句子列表。输出是 n-gram 计数和上下文计数。`<s>` 和 `</s>` 是句子边界标记。

### 步骤 2：拉普拉斯平滑

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

每个计数加 1。起到平滑作用，但过度将概率质量分配给未见过的事件，同时也会损害已知罕见事件的概率。

### 步骤 3：Kneser-Ney（二元组，插值版）

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

三个关键部分。`continuation_prob` 捕获"这个词出现在多少种不同的上下文中？"（Kneser-Ney 的创新）。`lambda_prev` 是折扣释放出的质量，用于加权回退。最终概率是折扣后的主项加上加权的延续项。

### 步骤 4：使用采样生成文本

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

按概率比例采样。每次使用不同种子会得到不同输出。要获得类似束搜索的输出，在每一步选择 argmax（贪心）并添加一个小的随机性控制（温度）。

### 步骤 5：困惑度

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

越低越好。对于 Brown 语料库，调优良好的四元组 KN 模型困惑度约为 140。Transformer 语言模型在同一测试集上困惑度为 15-30。差距约为 10 倍。这就是该领域转向的原因。

## 应用场景

- **经典 NLP 教学。** 这是理解平滑、最大似然估计（MLE）和困惑度最清晰的方式。
- **KenLM。** 生产级 n-gram 库。用于语音和机器翻译系统中的重打分，适用于低延迟场景。
- **设备端自动补全。** 键盘中的三元组模型。至今仍在使用。
- **基线。** 在宣称你的神经语言模型效果好之前，始终先计算 n-gram 语言模型的困惑度。如果你的 Transformer 没有大幅超越 KN，说明有问题。

## 交付成果

保存为 `outputs/prompt-lm-baseline.md`：

```markdown
---
name: lm-baseline
description: Build a reproducible n-gram language model baseline before training a neural LM.
phase: 5
lesson: 16
---

Given a corpus and target use (next-word prediction, rescoring, perplexity baseline), output:

1. N-gram order. Trigram for general English, 4-gram if corpus is large, 5-gram for speech rescoring.
2. Smoothing. Modified Kneser-Ney is the default; Laplace only for teaching.
3. Library. `kenlm` for production, `nltk.lm` for teaching, roll your own only to learn.
4. Evaluation. Held-out perplexity with consistent tokenization between train and test sets.

Refuse to report perplexity computed with different tokenization between systems being compared — perplexity numbers are comparable only under identical tokenization. Flag OOV rate in test set; KN handles OOV poorly unless you reserve a special <UNK> token during training.
```

## 练习

1. **简单。** 在 1000 句莎士比亚语料库上训练三元组语言模型。生成 20 个句子。它们在局部看起来合理，但全局上不连贯。这是经典演示。
2. **中等。** 在留出的莎士比亚测试集上实现 KN 模型的困惑度计算。与拉普拉斯比较。你应该看到 KN 将困惑度降低了 30-50%。
3. **困难。** 构建一个三元组拼写纠正器：给定一个拼写错误的词及其上下文，生成纠正候选并按语言模型下的上下文概率排序。在 Birkbeck 拼写语料库（公开）上评估。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| N-gram | 词序列 | `n` 个连续标记的序列。 |
| 平滑（Smoothing） | 避免零概率 | 重新分配概率质量，使未见过的事件获得非零概率。 |
| 困惑度（Perplexity） | 语言模型质量指标 | 在留出数据上的 `exp(-average log-prob)`。越低越好。 |
| 回退（Backoff） | 退回到更短的上下文 | 如果三元组计数为零，则使用二元组。Katz 回退将其形式化。 |
| Kneser-Ney | n-gram 的最佳平滑方法 | 绝对折扣加上低阶模型的延续概率。 |
| 延续概率（Continuation probability） | KN 特有 | `P(w)` 按 `w` 出现的上下文数量加权，而非按原始计数加权。 |

## 扩展阅读

- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — n-gram 语言模型和平滑的经典教材。
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) — 奠定了 Kneser-Ney 作为最佳 n-gram 平滑器地位的论文。
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) — KN 的原始论文。
- [KenLM](https://kheafield.com/code/kenlm/) — 快速的生产级 n-gram 语言模型，2026 年仍用于延迟敏感的应用场景。
