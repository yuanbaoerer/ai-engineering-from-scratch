# 主题建模 — LDA 与 BERTopic

> LDA：文档是主题的混合，主题是词的分布。BERTopic：文档在嵌入空间中聚类，聚类即主题。相同的目标，不同的分解方式。

**类型：** 学习
**语言：** Python
**前置知识：** 第 5 阶段 · 02（BoW + TF-IDF），第 5 阶段 · 03（Word2Vec）
**时间：** 约 45 分钟

## 问题描述

你有 10,000 条客服工单、50,000 篇新闻文章或 200,000 条推文。你需要在不逐一阅读的情况下了解这批文档的内容。你没有标注好的类别标签，甚至不知道存在多少个类别。

主题建模可以在无监督的情况下回答这个问题。给定一个语料库，它会返回一组数量较少且连贯的主题，以及每个文档在这些主题上的分布。

两种算法家族占据主导地位。LDA（2003）将每篇文档视为潜在主题的混合，将每个主题视为词上的分布。推断方法是贝叶斯推断（Bayesian Inference）。在需要混合成员主题分配和可解释的词级概率分布的生产环境中，LDA 仍然被广泛使用。

BERTopic（2020）使用 BERT 编码文档，用 UMAP 降维（Dimensionality Reduction），用 HDBSCAN 聚类，再通过基于类别的 TF-IDF 提取主题词。它在短文本、社交媒体以及语义相似度比词汇重叠更重要的场景中表现更优。每篇文档只分配一个主题，这对长篇内容来说是一个局限。

本课将建立对两种方法的直觉，并指出在给定语料库下应该选择哪一种。

## 核心概念

![LDA 混合模型 vs BERTopic 聚类](../assets/topic-modeling.svg)

**LDA 生成故事。** 每个主题是词上的分布。每篇文档是主题的混合。要生成文档中的一个词，先从文档的主题混合中采样一个主题，再从该主题的分布中采样一个词。推断过程则反过来：给定观测到的词，推断每篇文档的主题分布和每个主题的词分布。坍缩吉布斯采样（Collapsed Gibbs Sampling）或变分贝叶斯（Variational Bayes）完成了数学推导。

LDA 的核心输出：

- `doc_topic`：矩阵 `(n_docs, n_topics)`，每行之和为 1（文档的主题混合）。
- `topic_word`：矩阵 `(n_topics, vocab_size)`，每行之和为 1（主题的词分布）。

**BERTopic 流水线。**

1. 使用句子变换器（如 `all-MiniLM-L6-v2`）编码每篇文档。生成 384 维向量。
2. 使用 UMAP 降维到约 5 维。BERT 嵌入维度过高，不适合直接聚类。
3. 使用 HDBSCAN 聚类。基于密度的方法，产生大小不一的聚类以及一个"离群值"标签。
4. 对每个聚类，基于该聚类内的文档计算基于类别的 TF-IDF，提取排名靠前的词。

输出为每篇文档一个主题（加上一个 -1 离群标签）。可选地，通过 HDBSCAN 的概率向量获得软成员归属。

## 动手实现

### 第 1 步：通过 scikit-learn 实现 LDA

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

注意：停用词（Stopwords）已移除，`min_df` 和 `max_df` 过滤了罕见词和过于常见的词，使用 CountVectorizer（而非 TfidfVectorizer），因为 LDA 要求原始词频。

### 第 2 步：BERTopic（生产环境）

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

过滤 `Topic != -1` 会丢弃 BERTopic 的离群桶（HDBSCAN 无法聚类的文档）。`min_topic_size` 控制 HDBSCAN 的最小聚类大小；BERTopic 库的默认值为 10。本示例为配合课程规模将其显式设置为 15。对于超过 10,000 篇文档的语料库，建议增大到 50 或 100。

### 第 3 步：评估

两种方法都输出主题词。问题在于这些词是否具有连贯性。

- **主题连贯性（c_v）。** 在滑动窗口上下文中计算排名靠前词对的 NPMI（归一化逐点互信息（PMI / Pointwise Mutual Information）），将分数聚合成主题向量，再通过余弦相似度比较这些向量。越高越好。使用 `gensim.models.CoherenceModel` 并设置 `coherence="c_v"`。
- **主题多样性。** 所有主题的排名靠前词中去重词的比例。越高越好（主题之间不重叠）。
- **定性检查。** 阅读每个主题的排名靠前词。它们是否命名了一个真实的事物？人工判断仍然是最后一道防线。

## 何时选择哪种方法

| 场景 | 选择 |
|------|------|
| 短文本（推文、评论、标题） | BERTopic |
| 含主题混合的长文档 | LDA |
| 无 GPU / 计算资源有限 | LDA 或 NMF |
| 需要文档级多主题分布 | LDA |
| 需要 LLM 集成进行主题标注 | BERTopic（直接支持） |
| 资源受限的边缘部署 | LDA |
| 追求最大语义连贯性 | BERTopic |

最实际的考量因素是文档长度。BERT 嵌入有截断限制；LDA 的词频统计则适用于任意长度。对于超过嵌入模型上下文窗口的文档，要么分块后聚合，要么使用 LDA。

## 工具选型

2026 年技术栈：

- **BERTopic。** 短文本和语义敏感场景的默认选择。
- **`gensim.models.LdaModel`。** 经典 LDA 生产实现，成熟且久经考验。
- **`sklearn.decomposition.LatentDirichletAllocation`。** 便于实验的 LDA 实现。
- **NMF。** 非负矩阵分解（NMF / Non-negative Matrix Factorization）。LDA 的快速替代方案，在短文本上质量相当。
- **Top2Vec。** 设计与 BERTopic 类似。社区较小，但在部分基准测试上表现良好。
- **FASTopic。** 较新的方案，在超大语料库上比 BERTopic 更快。
- **基于 LLM 的标注。** 运行任意聚类，然后用模型为每个聚类命名。

## 交付产出

保存为 `outputs/skill-topic-picker.md`：

```markdown
---
name: topic-picker
description: Pick LDA or BERTopic for a corpus. Specify library, knobs, evaluation.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

Given a corpus description (document count, avg length, domain, language, compute budget), output:

1. Algorithm. LDA / NMF / BERTopic / Top2Vec / FASTopic. One-sentence reason.
2. Configuration. Number of topics: `recommended = max(5, round(sqrt(n_docs)))`, clamped to 200 for corpora under 40,000 docs; permit >200 only when the corpus is genuinely large (>40k) and note the increased compute cost. `min_df` / `max_df` filters and embedding model for neural approaches also belong here.
3. Evaluation. Topic coherence (c_v) via `gensim.models.CoherenceModel`, topic diversity, and a 20-sample human read.
4. Failure mode to probe. For LDA, "junk topics" absorbing stopwords and frequent terms. For BERTopic, the -1 outlier cluster swallowing ambiguous documents.

Refuse BERTopic on documents longer than the embedding model's context window without a chunking strategy. Refuse LDA on very short text (tweets, reviews under 10 tokens) as coherence collapses. Flag any n_topics choice below 5 as likely wrong; flag >200 on corpora under 40k docs as likely over-splitting.
```

## 练习

1. **简单。** 在 20 Newsgroups 数据集上使用 5 个主题拟合 LDA。打印每个主题的前 10 个词。手动为每个主题标注标签。算法是否找到了真实的类别？
2. **中等。** 在相同的 20 Newsgroups 子集上拟合 BERTopic。比较两种方法发现的主题数量、排名靠前的词以及定性连贯性。哪种方法更清晰地呈现了真实类别？
3. **困难。** 在你的语料库上计算 LDA 和 BERTopic 的 c_v 连贯性。分别使用 5、10、20、50 个主题运行。绘制连贯性与主题数量的关系图。报告哪种方法在不同主题数量下更稳定。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 主题（Topic） | 语料库涉及的某个事物 | 词上的概率分布（LDA）或相似文档的聚类（BERTopic）。 |
| 混合成员（Mixed membership） | 文档属于多个主题 | LDA 为每篇文档分配一个涵盖所有主题的分布。 |
| UMAP | 降维 | 保留局部结构的流形学习（Manifold Learning）；在 BERTopic 中使用。 |
| HDBSCAN | 密度聚类 | 发现大小不一的聚类；为离群点生成"噪声"标签（-1）。 |
| c_v 连贯性 | 主题质量指标 | 在滑动窗口内，排名靠前主题词的平均逐点互信息。 |

## 扩展阅读

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — LDA 论文。
- [Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) — BERTopic 论文。
- [Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — 提出 c_v 等指标的论文。
- [BERTopic 文档](https://maartengr.github.io/BERTopic/) — 生产环境参考。示例非常丰富。
