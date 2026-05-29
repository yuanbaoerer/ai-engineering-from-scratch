# 词袋模型、TF-IDF 与文本表示

> 先计数，后思考。2026 年，TF-IDF 在定义清晰的任务上仍然优于词嵌入。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 5 · 01（文本处理）、阶段 2 · 02（从零实现线性回归）
**时长：** 约 75 分钟

## 问题

模型需要数字，你只有字符串。

每个 NLP 流水线都必须回答同一个问题：如何将一个可变长度的 token 序列转换为分类器可以消费的固定大小向量。学术界给出的第一个答案是"最简单粗暴但能用的方法"——数词，造向量。

这个向量承载的生产级 NLP 比任何词嵌入模型都多。垃圾邮件过滤器、主题分类器、日志异常检测、搜索排序（BM25 之前）、第一波情感分析、学术 NLP 基准测试的第一个十年。2026 年的从业者在窄分类任务上仍然首选它。它快速、可解释，在"词出现与否决定结果"的任务上，往往与 4 亿参数的词嵌入模型无异。

本节从零构建词袋模型，然后是 TF-IDF。接着展示 scikit-learn 三行代码完成同样的工作。然后指出它的失败模式，这就是你转向词嵌入的原因。

## 概念

**词袋模型（BoW）** 丢弃顺序。对每个文档，统计每个词汇表词项出现的次数。向量长度等于词汇表大小。位置 `i` 是词 `i` 的计数。

**TF-IDF** 重新加权 BoW。出现在每个文档中的词没有信息量，降低权重。罕见于语料库但在一个文档中频繁出现的词是信号，提高权重。

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

其中 `TF` 是文档中的词频，`df` 是文档频率（包含该词的文档数），`N` 是总文档数。`log` 保持 ubiquitous 词的权重有界。

关键特性：两者都产生稀疏向量，轴可解释。你可以查看训练好的分类器权重，读出哪些词将文档推向哪个类别。用 768 维 BERT 词嵌入就无法这样做。

## 动手构建

### 步骤 1：构建词汇表

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

输入：token 化文档列表（任何词级分词器都可以；本节的 `code/main.py` 使用简化的 lowercase 变体）。输出：`{word: index}` 字典。稳定的插入顺序意味着词索引 0 是第一个文档中看到的第一个词。惯例各有不同；scikit-learn 按字母顺序排序。

### 步骤 2：词袋模型

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

行是文档，列是词汇表索引。条目 `[i][j]` 是"词 `j` 在文档 `i` 中出现多少次"。文档 1 中 `cat` 出现了两次，因为它确实出现了。文档 0 中 `ran` 出现零次，因为它确实没有出现。

### 步骤 3：词频与文档频率

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

有两个值得指出的平滑技巧。`(n+1)/(d+1)` 避免 `log(x/0)`。末尾的 `+1` 确保出现在每个文档中的词 IDF 仍为 1（而非 0），与 scikit-learn 默认值一致。其他实现使用原始的 `log(N/df)`。两者都行；平滑版本更友好。

### 步骤 4：TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

三个文档，五个词汇表词（`the`, `cat`, `sat`, `dog`, `ran`）。`the` 出现在全部三个文档中，所以它的 IDF 低。`dog` 只出现在一个文档中，所以它的 IDF 高。向量是稀疏的（大多数条目很小），有区分度的词脱颖而出。

### 步骤 5：L2 归一化行

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

没有归一化，较长的文档得到更大的向量，主导相似度得分。L2 归一化将每个文档放到单位超球面上。现在行之间的余弦相似度就是点积。

## 使用它

scikit-learn 提供生产级版本。

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` 一步完成 token 化、词汇表构建和 BoW。`TfidfVectorizer` 加上 IDF 加权和 L2 归一化。两者都返回稀疏矩阵。对于 10 万个文档，稠密版本放不进内存；保持稀疏，直到分类器要求稠密。

改变一切的旋钮：

| 参数 | 效果 |
|-----|------|
| `ngram_range=(1, 2)` | 包含二元组。通常能提升分类效果。 |
| `min_df=2` | 丢弃出现在少于 2 个文档中的词。在有噪声的数据上剪裁词汇表。 |
| `max_df=0.95` | 丢弃出现在超过 95% 文档中的词。在没有硬编码停用词列表的情况下近似停用词移除。 |
| `stop_words="english"` | scikit-learn 内置停用词列表。任务相关——情感分析*不应该*丢弃否定词。 |
| `sublinear_tf=True` | 使用 `1 + log(tf)` 而非原始 `tf`。当一个词在一个文档中重复多次时有用。 |

### TF-IDF 仍然胜出的场景（2026 年）

- 垃圾邮件检测、主题标注、日志异常标记。词出现与否是关键；语义细微差别不重要。
- 低数据场景（数百个标注样本）。TF-IDF 加上逻辑回归没有预训练成本。
- 延迟敏感系统。TF-IDF 加上线性模型在微秒级响应。 transformer 嵌入一个文档需要 10-100ms。
- 必须解释预测的系统。检查分类器的系数。正向权重最高的词就是原因。

### TF-IDF 失败的场景

语义盲区失败。考虑这两个文档：

- "The movie was not good at all."
- "The movie was excellent."

一个差评，一个好评。它们的 TF-IDF 重叠恰好是 `{the, movie, was}`。词袋分类器必须死记硬背 `not` 靠近 `good` 会翻转标签。它可以在足够数据上学会这个，但永远不如理解句法的模型优雅。

另一个失败：推理时的未登录词。在 IMDb 影评上训练的 BoW 模型不知道如何处理 `Zoomer-approved`，因为这个 token 从未在训练中出现。子词嵌入（第 4 节）处理这个问题。TF-IDF 不行。

### 混合方法：TF-IDF 加权嵌入

2026 年中等数据分类的务实默认值：使用 TF-IDF 权重作为词嵌入上的注意力。

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

你从嵌入中获得语义能力，从 TF-IDF 中获得罕见词强调。分类器在汇聚向量上训练。在约 5 万标注样本以下的情感、主题和意图分类任务上，这比单独的任一方法表现更好。

## 交付

保存为 `outputs/prompt-vectorization-picker.md`：

```markdown
---
name: vectorization-picker
description: Given a text-classification task, recommend BoW, TF-IDF, embeddings, or a hybrid.
phase: 5
lesson: 02
---

You recommend a text-vectorization strategy. Given a task description, output:

1. Representation (BoW, TF-IDF, transformer embeddings, or a hybrid). Explain why in one sentence.
2. Specific vectorizer configuration. Name the library. Quote the arguments (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`).
3. One failure mode to test before shipping.

Refuse to recommend embeddings when the user has under 500 labeled examples unless they show evidence of semantic failure in a TF-IDF baseline. Refuse to remove stopwords for sentiment analysis (negations carry signal). Flag class imbalance as needing more than a vectorizer change.

Example input: "Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

Example output:

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.
```

## 练习

1. **简单。** 在 L2 归一化的 TF-IDF 输出上实现 `cosine_similarity(doc_vec_a, doc_vec_b)`。验证相同文档得分为 1.0，词汇表完全不同的文档得分为 0.0。
2. **中等。** 为 `bag_of_words` 添加 `n-gram` 支持。参数 `n` 生成 `n` 元组的计数。测试 `n=2` 在 `["the", "cat", "sat"]` 上产生 `["the cat", "cat sat"]` 的二元组计数。
3. **困难。** 使用 GloVe 100d 向量（下载一次，缓存）构建上述 TF-IDF 加权嵌入混合模型。在 20 Newsgroups 数据集上比较分类准确率：纯 TF-IDF、纯平均池化嵌入、混合模型。报告各自在什么情况下胜出。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| BoW | 词频向量 | 一个文档中词汇表词项的计数。丢弃顺序。 |
| TF | 词频 | 一个词在文档中的计数，可选地按文档长度归一化。 |
| DF | 文档频率 | 至少包含该词一次的文档数量。 |
| IDF | 逆文档频率 | `log(N / df)` 平滑版。降低无处不在的词的权重。 |
| 稀疏向量 | 大多为零 | 词汇表通常有 1 万到 10 万个词；大多数在任何给定文档中都不存在。 |
| 余弦相似度 | 向量夹角 | L2 归一化向量的点积。1 表示相同，0 表示正交。 |

## 延伸阅读

- [scikit-learn — feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — 权威 API 参考，外加每个旋钮的说明。
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) — 这篇论文让 TF-IDF 统治了十年。
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — 2026 年的视角：老方法何时胜出以及为什么。