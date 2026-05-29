# 词嵌入 —— 从零实现 Word2Vec

> 一个词由它所处的环境决定。在这一思想基础上训练一个浅层网络，几何关系自然涌现。

**类型：** 实战项目
**语言：** Python
**前置知识：** 第五阶段 · 02（BoW + TF-IDF）、第三阶段 · 03（从零实现反向传播）
**预计时间：** 约 75 分钟

## 问题背景

TF-IDF 知道 `dog` 和 `puppy` 是不同的单词。它不知道它们的意思几乎相同。在 `dog` 上训练的分类器无法泛化到关于 `puppy` 的评论。你可以通过列出同义词来弥补这一点，但这在罕见词、领域术语以及每一种你未预料到的语言面前都会失效。

你需要一个表示方式，让 `dog` 和 `puppy` 在向量空间中位置相近。 让 `king - man + woman` 接近 `queen`。让在 `dog` 上训练的模型能够将部分信号免费迁移到 `puppy`。

Word2Vec 给了我们这个向量空间。两层神经网络，万亿 token 的训练规模，2013 年发表。架构简单得近乎尴尬。但它彻底改变了 NLP 十年。

## 核心概念

**分布假说**（Firth, 1957）："观其伴，知其义。" 如果两个词出现在相似的上下文中，它们很可能意思相近。

Word2Vec 有两种变体，都利用了这一思想。

- **Skip-gram。** 给定中心词，预测周围的词。窗口大小为 2 时，`cat -> (the, sat, on)`。
- **CBOW（连续词袋）。** 给定周围的词，预测中心词。`(the, sat, on) -> cat`。

Skip-gram 训练较慢，但对罕见词的处理更好。它成为了默认选择。

网络有一个隐藏层，没有非线性激活函数。输入是词汇表上的 one-hot 向量。输出是词汇表上的 softmax。训练完成后，丢弃输出层。隐藏层的权重就是词嵌入。

```
one-hot(中心词) ── W ──▶ 隐藏层 (d维) ── W' ──▶ softmax(词汇表)
                          ^
                          这个就是嵌入
```

关键技巧：对 10 万个词做 softmax 开销大得离谱。Word2Vec 使用**负采样**将其转化为二分类任务。预测"这个上下文词是否出现在这个中心词附近，是或否"。每个训练样本采样几个负样本（非共现词），而不是对整个词汇表计算 softmax。

## 从零构建

### 步骤 1：从语料库生成训练对

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

窗口内每个（中心词，上下文）对都是正训练样本。

### 步骤 2：嵌入表

两个矩阵。`W` 是中心词嵌入表（你要保留的那个）。`W'` 是上下文词嵌入表（通常被丢弃，有时会与 `W` 做平均）。

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

用小的随机值初始化。词汇表 1 万、维度 100 是现实可行的配置；教学中，词汇表 50、维度 16 就足够看到几何关系了。

### 步骤 3：负采样目标函数

对于每个正样本对 `(center, context)`，从词汇表中采样 `k` 个随机词作为负样本。训练模型使得正样本的点积 `W[center] · W'[context]` 较高，负样本的较低。

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

魔法公式：正样本对的逻辑损失（希望 sigmoid 接近 1）加上负样本对的逻辑损失（希望 sigmoid 接近 0）。梯度流向两个表。完整的推导见原始论文；如果想深入理解，建议用笔和纸亲手推导一遍。

### 步骤 4：在玩具语料库上训练

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

在大规模语料上训练足够多的轮次后，在相似上下文中出现的词会有相似的中心嵌入。在玩具语料上，你能隐约看到这个效果。在数十亿 token 上，效果就非常显著了。

### 步骤 5：类比技巧

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

在预训练的 300 维 Google News 向量上：

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`。这并不是因为模型懂得什么是皇室。而是因为向量 `(king - man)` 捕捉了某种类似"皇室"的东西，加上它到 `woman` 就落在皇室-女性的区域附近。

## 实际应用

从零实现 Word2Vec 是为了学习。生产级 NLP 使用 `gensim`。

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

真实工作中，你几乎不需要自己训练 Word2Vec。你直接下载预训练向量。

- **GloVe** — 斯坦福的共现矩阵分解方法。50d、100d、200d、300d 多种规格。通用覆盖效果良好。课程 04 专门讲解 GloVe。
- **fastText** — Facebook 的 Word2Vec 扩展，通过字符 n-gram 进行嵌入。通过组合子词来处理未登录词。课程 04。
- **Google News 预训练 Word2Vec** — 300 维，300 万词表，2013 年发布。至今仍被每日下载。

### 2026 年 Word2Vec 仍有优势的领域

- **轻量级领域专用检索。** 在笔记本电脑上用一小时训练医学摘要，就能获得通用模型无法捕捉的专业向量。
- **类比风格特征工程。** `gender_vector = mean(man - woman pairs)`。从中减去其他词以获得性别中立轴。这仍在公平性研究中使用。
- **可解释性。** 100 维足够小，可以通过 PCA 或 t-SNE 可视化并真正看到聚类的形成。
- **任何需要在设备上运行且没有 GPU 的推理场景。** Word2Vec 查找只是一个单行获取操作。

### Word2Vec 的局限性

多义词壁垒。`bank` 只有一个向量。`river bank` 和 `financial bank` 共享它。`table`（电子表格 vs. 家具）也共享它。下游分类器无法从向量中区分词义。

上下文嵌入（ELMo、BERT、以及之后所有的 transformer）通过为每个词的出现根据周围上下文产生不同的向量来解决这个问题。这就是从 Word2Vec 到 BERT 的跨越：从静态到上下文。第七阶段涵盖 transformer 部分。

另一个局限是未登录词问题。Word2Vec 从未见过 `Zoomer-approved`，如果它不在训练数据中就没有任何后备方案。fastText 通过子词组合解决了这个问题（课程 04）。

## 发布

保存为 `outputs/skill-embedding-probe.md`：

```markdown
---
name: embedding-probe
description: 检查 word2vec 模型。运行类比、查找近邻、诊断质量。
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

你探测训练好的词嵌入来验证它们是否正常工作。给定一个 `gensim.models.KeyedVectors` 对象和词汇表，你运行：

1. 三个经典类比测试。`king : man :: queen : woman`。`paris : france :: tokyo : japan`。`walking : walked :: swimming : ?`。报告 top-1 结果及其余弦相似度。
2. 五个领域专用词的最近邻测试。打印 top-5 近邻及其余弦相似度。
3. 一个对称性检查。`similarity(a, b) == similarity(b, a)` 在浮点精度范围内。
4. 一个退化检查。如果任何嵌入的范数低于 0.01 或高于 100，则模型存在训练 bug。标记它。

不要仅凭类比准确率就宣称模型是好的。类比基准测试可以被"刷分"，且不能迁移到下游任务。建议结合内在评估和下游评估一起使用。
```

## 练习

1. **简单。** 在小规模语料库（20 句关于猫和狗的句子）上运行训练循环。200 轮后，验证 `nearest(vocab, W, W[vocab["cat"]])` 在 top 3 中返回 `dog`。如果没有，增加训练轮次或词汇量。
2. **中等。** 添加高频词二次采样。频率高于 `10^-5` 的词以与其频率成正比的概率从训练对中丢弃。测量对罕见词相似性的影响。
3. **困难。** 在 20 Newsgroups 语料库上训练模型。计算两个偏置轴：`he - she` 和 `doctor - nurse`。将职业词投影到这两个轴上。报告哪个职业有最大的偏置差距。这是公平性研究人员使用的探测方法。

## 核心术语

| 术语 | 人们通常的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 词嵌入 | 词作为向量 | 从上下文中学习得到的稠密、低维（通常 100-300）表示。 |
| Skip-gram | Word2Vec 技巧 | 从中心词预测上下文词。比 CBOW 慢，但对罕见词更好。 |
| 负采样 | 训练捷径 | 用对 `k` 个随机词的二分类替代对整个词汇表的 softmax。 |
| 静态嵌入 | 每个词一个向量 | 不考虑上下文，同一个词始终是同一个向量。在多义词上失效。 |
| 上下文嵌入 | 上下文敏感的向量 | 根据周围词，每个出现都产生不同的向量。这就是 transformer 产生的东西。 |
| OOV | 未登录词 | 训练时未见过的词。Word2Vec 无法为这些词生成向量。 |

## 延伸阅读

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) — 负采样论文。简短易懂。
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) — 最清晰的梯度推导，如果觉得原论文的数学太密集的话。
- [gensim Word2Vec 教程](https://radimrehurek.com/gensim/models/word2vec.html) — 生产级训练设置，实际可用。