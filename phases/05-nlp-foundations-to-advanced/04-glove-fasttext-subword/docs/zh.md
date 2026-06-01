# GloVe、FastText 和子词嵌入

> Word2Vec 为每个词训练一个嵌入向量（Embedding Vector）。GloVe 对共现矩阵（Co-occurrence Matrix）进行分解。FastText 对词的组成部分进行嵌入。BPE 则架起了通往 Transformer 的桥梁。

**类型：** 构建
**语言：** Python
**前置课程：** Phase 5 · 03（从零实现 Word2Vec）
**时间：** 约 45 分钟

## 问题背景

Word2Vec 留下了两个未解的问题。

首先，有一条平行的研究路线直接对共现矩阵进行分解（LSA、HAL），而不是像 Word2Vec 那样进行在线的 skip-gram 更新。Word2Vec 的迭代方法是否本质上更好，还是两者之间的差异仅仅是因为它们处理计数的方式不同？**GloVe** 回答了这个问题：使用精心设计的损失函数进行矩阵分解，能够匹配甚至超越 Word2Vec 的效果，而且训练成本更低。

其次，这两种方法都无法处理从未见过的词。`Zoomer-approved`、`dogecoin`、上周才造出来的专有名词、某个罕见词根的所有屈折形式——这些词都不在词表（Vocabulary）中。**FastText** 通过嵌入字符 n-gram 来解决这个问题：一个词是其各部分（包括词素）的总和，因此即使是词表外的词也能获得一个合理的向量。

第三，当 Transformer 到来之后，问题再次发生了变化。词级词表的上限大约在一百万个条目左右；而真实语言的开放性远不止于此。**字节对编码（BPE）** 及其相关方法通过学习一个由高频子词单元组成的词表来解决这个问题，这个词表可以覆盖所有文本。每个现代大语言模型使用的分词器（Tokenizer）都是子词分词器。

本课将逐一讲解这三种方法，然后说明在什么情况下该选择哪一种。

## 核心概念

**GloVe（全局向量）。** 构建词-词共现矩阵 `X`，其中 `X[i][j]` 表示词 `j` 出现在词 `i` 上下文中的频率。训练向量使得 `v_i · v_j + b_i + b_j ≈ log(X[i][j])`。对损失函数加权，使高频词对不会主导整个损失。就这样。

**FastText。** 一个词等于其字符 n-gram 之和加上该词本身。`where` 变成 `<wh, whe, her, ere, re>, <where>`。词向量是这些组成部分向量的总和。训练方式与 Word2Vec 相同。好处在于：未见过的词（如 `whereupon`）可以由已知的 n-gram 组合而成。

**BPE（字节对编码）。** 从单个字节（或字符）的词表开始。统计语料库中所有相邻词对的频率。将出现频率最高的词对合并为一个新 token。重复 `k` 次迭代。结果：得到一个包含 `k + 256` 个 token 的词表，其中高频序列（如 `ing`、`tion`、`the`）成为单个 token，而罕见词则被拆分为熟悉的部分。任何句子都能被分词为某种 token 序列。

## 动手实现

### GloVe：分解共现矩阵

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

有两个关键部分值得说明。加权函数 `f(x) = (x/x_max)^alpha` 对高频词对（如 `(the, and)`）进行降权，使其不会主导损失函数。最终的嵌入是 `W`（中心词）和 `W_tilde`（上下文词）两个表的和。将两者相加是一个已发表的技巧，效果通常优于只使用其中一个。

### FastText：子词感知嵌入

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

每个词由其 n-gram 集合（通常为 3 到 6 个字符）表示。词嵌入是其所有 n-gram 嵌入的总和。在 skip-gram 训练中，将 Word2Vec 原本使用单个向量的地方替换为这种 n-gram 求和的方式。

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

对于一个未见过的词，只要它的部分 n-gram 是已知的，你仍然可以得到一个向量。`whereupon` 与 `where` 共享 `<wh`、`her`、`ere` 和 `<where` 这些 n-gram，因此这两个词的向量会彼此接近。

### BPE：学习得到的子词词表

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

第一次迭代合并出现频率最高的相邻词对。经过足够多的迭代后，高频子串（如 `low`、`est`、`tion`）成为单个 token，而罕见词也能被干净地拆分。

真正的 GPT / BERT / T5 分词器会学习 3 万到 10 万次合并。结果：任何文本都能被分词（Tokenization）为一个长度有界的已知 ID 序列，永远不会出现词表外（OOV）的情况。

## 实际使用

在实际应用中，你很少会自己训练这些模型。你通常加载预训练的检查点。

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

在 Transformer 时代使用 BPE 风格的子词分词：

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

`Ġ` 前缀标记词边界（这是 GPT-2 的约定）。每个现代分词器都是 BPE、WordPiece（BERT）或 SentencePiece（T5、LLaMA）的变体。

### 如何选择

| 场景 | 选择 |
|------|------|
| 预训练的通用词向量，不需要处理词表外词 | GloVe 300d |
| 预训练的通用词向量，需要处理拼写错误 / 新词 / 形态丰富的语言 | FastText |
| 任何输入 Transformer 的任务（训练或推理） | 使用模型自带的分词器，切勿更换 |
| 从零开始训练自己的语言模型 | 先在语料库上训练 BPE 或 SentencePiece 分词器 |
| 使用线性模型的生产环境文本分类 | 仍然是 TF-IDF，参见第 02 课 |

## 交付使用

保存为 `outputs/skill-embeddings-picker.md`：

```markdown
---
name: tokenizer-picker
description: Pick a tokenization approach for a new language model or text pipeline.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

Given a task and dataset description, you output:

1. Tokenization strategy (word-level, BPE, WordPiece, SentencePiece, byte-level). One-sentence reason.
2. Vocabulary size target (e.g., 32k for an English-only LM, 64k-100k for multilingual).
3. Library call with the exact training command. Name the library. Quote the arguments.
4. One reproducibility pitfall. Tokenizer-model mismatch is the single most common silent production bug; call out which pair must be used together.

Refuse to recommend training a custom tokenizer when the user is fine-tuning a pretrained LLM. Refuse to recommend word-level tokenization for any model targeting production inference. Flag non-English / multi-script corpora as needing SentencePiece with byte fallback.
```

## 练习

1. **简单。** 运行 `char_ngrams("playing")` 和 `char_ngrams("played")`。计算这两个 n-gram 集合的 Jaccard 重叠度。你应该能看到大量共享的部分（`pla`、`lay`、`play`），这就是 FastText 在不同形态变体之间迁移效果好的原因。
2. **中等。** 扩展 `learn_bpe` 以跟踪词表增长情况。绘制每个语料库字符对应的 token 数量随合并次数变化的曲线。你应该能看到初期压缩速度很快，最终趋于约每 token 2-3 个字符。
3. **困难。** 在莎士比亚全集上训练一个 1000 次合并的 BPE。比较常见词与罕见专有名词的分词结果。测量分词前后每个词的平均 token 数。写下让你感到意外的发现。

## 关键术语

| 术语 | 人们怎么说 | 它的实际含义 |
|------|-----------|-------------|
| 共现矩阵（Co-occurrence Matrix） | 词-词频率表 | `X[i][j]` = 词 `j` 出现在词 `i` 上下文窗口中的频率。 |
| 子词（Subword） | 词的一部分 | 字符 n-gram（FastText）或学习得到的 token（BPE/WordPiece/SentencePiece）。 |
| BPE | 字节对编码 | 迭代合并出现频率最高的相邻词对，直到词表达到目标大小。 |
| OOV | 词表外 | 模型从未见过的词。Word2Vec/GloVe 无法处理。FastText 和 BPE 可以处理。 |
| 嵌入（Embedding） | 向量表示 | 将离散符号（词、子词、token）映射到连续向量空间的过程或结果。 |
| 字节级 BPE（Byte-level BPE） | 基于原始字节的 BPE | GPT-2 的方案。词表从 256 个字节开始，因此永远不会出现 OOV。 |

## 延伸阅读

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — GloVe 论文，仅七页，至今仍是关于损失函数推导的最佳文献。
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — FastText 论文。
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — 将 BPE 引入现代 NLP 的开创性论文。
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) — BPE、WordPiece 和 SentencePiece 在实际应用中的区别。
