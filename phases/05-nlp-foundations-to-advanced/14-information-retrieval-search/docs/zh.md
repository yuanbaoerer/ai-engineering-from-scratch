# 信息检索与搜索

> BM25 精确但脆弱。密集检索撒网广泛但会错过关键词。混合检索是 2026 年的默认方案。其余都是调优。

**类型:** 构建
**语言:** Python
**前置知识:** 第 5 阶段 · 02（BoW + TF-IDF），第 5 阶段 · 04（GloVe、FastText、子词）
**时间:** 约 75 分钟

## 问题

用户输入 "what happens if someone lies to get money"，期望找到实际涵盖该内容的法规："IPC 第 420 条。"关键词搜索完全找不到（没有共享词汇）。如果嵌入模型没有在法律文本上训练过，语义搜索（Semantic Search）也会错过。真正的搜索必须两者兼顾。

信息检索是每个 RAG 系统、每个搜索栏、每个文档站点模糊查找背后的流水线。2026 年在生产中有效的架构不是单一方法，而是一系列互补方法的链式组合，每一种都能捕捉前一种的失败。

本课程构建每个组件，并指出每种方法能捕捉哪些失败。

## 概念

![混合检索：BM25 + 密集检索 + RRF + 交叉编码器重排](../assets/retrieval.svg)

四个层次。按需选择。

1. **稀疏检索（Sparse Retrieval）（BM25）。** 快速、精确匹配准确，但语义能力差。基于倒排索引（Inverted Index）运行。在数百万文档上每次查询低于 10 毫秒。能正确获取法规引用、产品代码、错误消息、命名实体。
2. **密集检索。** 将查询和文档编码为向量。最近邻搜索。捕捉释义和语义相似性。会错过仅差一个字符的精确关键词匹配。使用 FAISS 或向量数据库每次查询 50-200 毫秒。
3. **融合。** 合并稀疏和密集检索的排序列表。倒数排名融合（RRF）是简单的默认方案，因为它忽略原始分数（它们处于不同尺度），只使用排名位置。当你知道某个信号在你的领域占主导时，加权融合是一个选项。
4. **交叉编码器重排。** 取融合后的前 30 个结果。运行交叉编码器（查询 + 文档一起输入，对每对进行评分）。保留前 5 个。交叉编码器每对比双编码器慢，但准确度高得多。你只在前 30 个上运行，从而分摊成本。

三路检索（BM25 + 密集 + 学习型稀疏如 SPLADE）在 2026 年基准测试中优于两路检索，但需要支持学习型稀疏索引的基础设施。对大多数团队来说，两路检索加重排交叉编码器是最佳平衡点。

## 构建

### 第 1 步：从零实现 BM25

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

两个值得了解的参数。`k1=1.5` 控制词频饱和度；越高意味着词重复的权重越大。`b=0.75` 控制长度归一化；0 忽略文档长度，1 完全归一化。默认值是 Robertson 在原始论文中的推荐值，很少需要调优。

### 第 2 步：使用双编码器进行密集检索

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

对嵌入进行 L2 归一化，使点积等于余弦相似度。`all-MiniLM-L6-v2` 是 384 维，速度快，对大多数英文检索足够强大。对于多语言工作，使用 `paraphrase-multilingual-MiniLM-L12-v2`。追求最高精度，使用 `bge-large-en-v1.5` 或 `e5-large-v2`。

### 第 3 步：倒数排名融合

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

常数 `k=60` 来自原始 RRF 论文。`k` 越高，排名差异的贡献越平坦；`k` 越低，顶部排名越占主导。60 是论文发表的默认值，很少需要调优。

### 第 4 步：混合搜索 + 重排

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

三个阶段组合而成。BM25 找词汇匹配。密集检索找语义匹配。RRF 合并两个排名，无需分数校准。交叉编码器使用查询-文档对一起重新评分前 30 个结果，捕捉双编码器遗漏的细粒度相关性。保留前 5 个。

### 第 5 步：评估

| 指标 | 含义 |
|--------|---------|
| Recall@k | 在正确文档存在的查询中，有多少比例出现在前 k 个结果中？ |
| MRR（平均倒数排名） | 第一个相关文档的 1/rank 的平均值。 |
| nDCG@k | 考虑相关性梯度，而非仅仅是二元相关/不相关。 |

对于 RAG 来说，检索器的 **Recall@k** 是最重要的指标。如果正确的段落不在检索结果中，阅读器就无法回答。

调试提示：对于失败的查询，对比稀疏和密集检索的排名。如果一个找到了正确文档而另一个没有，你就有词汇不匹配问题（修复：补充缺失的部分）或语义歧义问题（修复：更好的嵌入或重排器）。

## 使用

2026 年技术栈：

| 规模 | 技术栈 |
|-------|-------|
| 1k-100k 文档 | 内存中 BM25 + `all-MiniLM-L6-v2` 嵌入 + RRF。无需单独数据库。 |
| 100k-10M 文档 | FAISS 或 pgvector 用于密集检索 + Elasticsearch / OpenSearch 用于 BM25。并行运行。 |
| 10M+ 文档 | 支持混合检索的 Qdrant / Weaviate / Vespa / Milvus。在前 30 个结果上进行交叉编码器重排。 |
| 最高质量前沿 | 三路检索（BM25 + 密集 + SPLADE）+ ColBERT 后期交互重排 |

无论你选择什么，都要为评估做预算。在基准测试端到端 RAG 准确率之前，先基准测试检索召回率。阅读器无法修复检索器遗漏的内容。

### 2026 年生产 RAG 中的宝贵经验

- **80% 的 RAG 故障源于数据摄入和分块，而非模型。** 团队花数周时间更换 LLM 和调优提示词，而检索却在每三次查询中静默返回错误的上下文。先修复分块。
- **分块策略比块大小更重要。** 固定大小分割会破坏表格、代码和嵌套标题。句子感知是默认方案；语义或基于 LLM 的分块在技术文档和产品手册上效果显著。
- **父文档模式。** 检索小的 "子" 块以提高精度。当来自同一父级部分的多个子块出现时，替换为父级块以保留上下文。这在不重新训练的情况下持续提升答案质量。
- **k_rerank=3 通常是最优的。** 超过这个数量的每个额外块都会增加 token 成本和生成延迟，而不会提升答案质量。如果 k=8 仍然比 k=3 好，说明重排器表现不佳。
- **HyDE / 查询扩展。** 从查询生成一个假设性答案，嵌入该答案，进行检索。弥合短问题和长文档之间的措辞差距。无需训练即可免费提升精度。
- **上下文预算低于 8K token。** 在该限制下持续命中意味着重排器阈值太宽松。
- **对所有内容进行版本管理。** 提示词、分块规则、嵌入模型、重排器。任何漂移都会静默破坏答案质量。CI 对忠实度、上下文精度和未回答问题率进行门控，在用户看到之前阻断回退。
- **三路检索（BM25 + 密集 + 学习型稀疏如 SPLADE）在 2026 年基准测试中优于两路检索**，特别是对于混合专有名词和语义的查询。当基础设施支持 SPLADE 索引时即可部署。

根据 2026 年行业测量，合理的检索设计可将幻觉减少 70-90%。大多数 RAG 性能提升来自更好的检索，而非模型微调。

## 部署

保存为 `outputs/skill-retrieval-picker.md`：

```markdown
---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.
```

## 练习

1. **简单。** 在一个 500 文档的语料库上实现上述 `hybrid_search`。测试 20 个查询。比较 BM25-only、dense-only 和 hybrid 在前 5 个结果中的召回率。
2. **中等。** 添加 MRR 计算。对于每个有已知正确文档的测试查询，找出正确文档在 BM25、密集和混合排名中的位置。报告每种方法的 MRR。
3. **困难。** 使用 MultipleNegativesRankingLoss（Sentence Transformers）在你的领域微调密集编码器。从 500 个查询-文档对构建训练集。比较微调前后的召回率。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| BM25 | 关键词搜索 | Okapi BM25。根据词频、IDF 和长度对文档评分。 |
| 密集检索（Dense Retrieval） | 向量搜索 | 将查询和文档编码为向量，找最近邻。 |
| 双编码器（Bi-Encoder） | 嵌入模型 | 独立编码查询和文档。查询时速度快。 |
| 交叉编码器（Cross-Encoder） | 重排模型 | 将查询和文档一起编码。慢但准确。 |
| RRF | 排名融合 | 通过求和 `1/(k + rank)` 合并两个排名。 |
| Recall@k | 检索指标 | 相关文档出现在前 k 个结果中的查询比例。 |

## 扩展阅读

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — BM25 的权威论述。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，经典的双编码器。
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) — 缩小与密集检索差距的学习型稀疏检索器。
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — RRF 论文。
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) — 后期交互检索。
