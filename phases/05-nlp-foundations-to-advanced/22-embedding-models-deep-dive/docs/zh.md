# 嵌入模型 —— 2026 深度解析

> Word2Vec 给你一个词一个向量。现代嵌入模型给你一段话一个向量，跨语言，支持稀疏（Sparse Embedding）、稠密（Dense Embedding）和多向量（Multi-vector）视图，尺寸可调以适应你的索引。选错了，RAG 就会检索到错误的内容。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 5 · 03（Word2Vec），Phase 5 · 14（信息检索）
**时间：** 约 60 分钟

## 问题所在

你的 RAG 系统有 40% 的概率检索到错误的段落。罪魁祸首很少是向量数据库（Vector Database）或提示词，而是嵌入模型。

在 2026 年选择嵌入模型意味着在五个维度上做出选择：

1. **稠密 vs 稀疏 vs 多向量。** 一个段落一个向量，还是一个 token 一个向量，又或者是稀疏加权词袋（Bag-of-words）。
2. **语言覆盖。** 单语英语模型在纯英语任务上仍然占优。多语言模型在混合语料库时胜出。
3. **上下文长度。** 512 token vs 8,192 vs 32,768 —— 实际有效容量通常是标称最大值的 60-70%。
4. **维度预算。** 全精度 3,072 个浮点数 = 每个向量 12 KB。1 亿个向量时，存储费用为 $1,300/月。Matryoshka 截断可以减少 4 倍。
5. **开源 vs 托管服务。** 开源权重意味着你控制整个技术栈和数据。托管服务意味着你用控制权换取"永远最新"。

本课会列出各项权衡，让你能基于证据做出选择，而非跟随上季度的流行趋势。

## 核心概念

![稠密、稀疏和多向量嵌入](../assets/embedding-modes.svg)

**稠密嵌入（Dense Embedding）。** 每个段落一个向量（通常 384-3,072 维）。余弦相似度（Cosine Similarity）按语义接近程度对段落排序。OpenAI `text-embedding-3-large`、BGE-M3 稠密模式、Voyage-3。默认选择。

**稀疏嵌入（Sparse Embedding）。** SPLADE 风格。Transformer 为词表中每个 token 预测一个权重，然后将大部分置零。结果是一个大小为 |vocab| 的稀疏向量。捕获词法匹配（类似 BM25），但使用学习到的词项权重。在关键词密集的查询中表现强劲。

**多向量（Multi-vector）（后期交互）。** ColBERTv2、Jina-ColBERT。每个 token 一个向量。使用 MaxSim 进行评分：对于查询中的每个 token，找到最相似的文档 token，然后求和。存储和评分成本更高，但在长查询和特定领域语料库上表现更优。

**BGE-M3：三合一。** 单个模型同时输出稠密、稀疏和多向量表示。每种表示可独立查询；分数通过加权求和融合。2026 年当你想从一个检查点获得灵活性时的默认选择。

**Matryoshka 表示学习。** 训练使得向量的前 N 维本身就是一个有用的独立嵌入。将 1,536 维向量截断到 256 维，仅损失约 1% 的精度，却节省 6 倍存储。OpenAI text-3、Cohere v4、Voyage-4、Jina v5、Gemini Embedding 2、Nomic v1.5+ 均支持。

### MTEB 排行榜只讲述部分故事

大规模文本嵌入基准测试 —— 2022 年推出时包含 8 种任务类型共 56 个任务，MTEB v2 扩展到 100+ 个任务。2026 年初，Gemini Embedding 2 在检索任务上领先（67.71 MTEB-R）。Cohere embed-v4 在通用任务上领先（65.2 MTEB）。BGE-M3 在开源多语言上领先（63.0）。排行榜是必要条件但不是充分条件 —— 始终在你自己的领域上进行基准测试。

### 三层模式

| 用例 | 模式 |
|------|------|
| 快速首轮检索 | 稠密双编码器（Bi-encoder）（BGE-M3、text-3-small） |
| 召回率（Recall）提升 | 稀疏（SPLADE、BGE-M3 稀疏）+ RRF 融合 |
| top-50 精度 | 多向量（ColBERTv2）或交叉编码器（Cross-encoder）重排器 |

大多数生产系统三层都用。

## 动手实现

### 步骤 1：基线 —— 使用 Sentence-BERT 的稠密嵌入

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True` 使点积（Dot Product）等于余弦相似度。务必设置此项。

### 步骤 2：Matryoshka 截断

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

截断后重新归一化（Normalization）。Nomic v1.5、OpenAI text-3 和 Voyage-4 的训练使得前几个级别截断是无损的。非 Matryoshka 模型（原始 Sentence-BERT）在截断时会急剧退化。

### 步骤 3：BGE-M3 多功能性

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

三个索引，一次推理调用。分数融合：

```python
dense_score = ... # cosine over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

在你的领域上调优权重。

### 步骤 4：在自定义任务上进行 MTEB 评估

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

在你领域的一个*代表性*子集上运行候选模型。不要只看排行榜排名 —— 你的领域很重要。

### 步骤 5：手写余弦相似度

参见 `code/main.py`。使用平均哈希技巧嵌入（仅标准库）。无法与 Transformer 嵌入竞争，但展示了基本流程：分词 → 向量 → 归一化 → 点积。

## 常见陷阱

- **查询和文档使用同一模型。** 部分模型（Voyage、Jina-ColBERT）使用非对称编码（Asymmetric Encoding） —— 查询和文档经过不同的路径。务必检查模型卡片。
- **缺少前缀。** `bge-*` 模型需要在查询前加上 `"Represent this sentence for searching relevant passages: "`。忘记这一点会导致 3-5 个百分点的召回率（Recall）差距。
- **Matryoshka 截断过度。** 1,536 → 256 通常是安全的。1,536 → 64 则不然。在你的评估集上验证。
- **上下文截断。** 大多数模型会静默截断超过最大长度的输入。长文档需要分块（参见第 23 课）。
- **忽略延迟尾部。** MTEB 分数隐藏了 p99 延迟。600M 参数模型可能比 335M 模型高出 2 分，但每次查询成本高 3 倍。

## 实际应用

2026 年技术栈：

| 场景 | 选择 |
|------|------|
| 纯英语、快速、API | `text-embedding-3-large` 或 `voyage-3-large` |
| 开源权重、英语 | `BAAI/bge-large-en-v1.5` |
| 开源权重、多语言 | `BAAI/bge-m3` 或 `Qwen3-Embedding-8B` |
| 长上下文（32k+） | Voyage-3-large、Cohere embed-v4、Qwen3-Embedding-8B |
| 纯 CPU 部署 | Nomic Embed v2（137M 参数，MoE） |
| 存储受限 | Matryoshka 截断 + int8 量化 |
| 关键词密集查询 | 添加 SPLADE 稀疏，与稠密 RRF 融合 |

2026 年模式：从 BGE-M3 或 text-3-large 开始，用 MTEB 在你的领域上评估，如果领域专用模型领先超过 3 分则替换。

## 交付产物

保存为 `outputs/skill-embedding-picker.md`：

```markdown
---
name: embedding-picker
description: Pick embedding model, dimension, and retrieval mode for a given corpus and deployment.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

Given a corpus (size, languages, domain, avg length), deployment target (cloud / edge / on-prem), latency budget, and storage budget, output:

1. Model. Named checkpoint or API. One-sentence reason.
2. Dimension. Full / Matryoshka-truncated / int8-quantized. Reason tied to storage budget.
3. Mode. Dense / sparse / multi-vector / hybrid. Reason.
4. Query prefix / template if required by the model card.
5. Evaluation plan. MTEB tasks relevant to domain + held-out domain eval with nDCG@10.

Refuse recommendations that truncate Matryoshka to <64 dims without domain validation. Refuse ColBERTv2 for corpora under 10k passages (overhead not justified). Flag long-document corpora (>8k tokens) routed to models with 512-token windows.
```

## 练习

1. **简单。** 使用 `bge-small-en-v1.5` 以全维度（384）编码 100 个句子，然后以 Matryoshka 128 维编码。测量 10 个查询上的 MRR 下降。
2. **中等。** 在你领域的 500 个段落上比较 BGE-M3 的稠密、稀疏和 colbert。哪个在 recall@10 上胜出？RF 融合是否优于最佳单一模式？
3. **困难。** 在你排名前 2 的领域任务上，用 MTEB 评估三个候选模型。报告 MTEB 分数、100 个查询批量的 p99 延迟以及每百万查询的成本。选择帕累托最优（Pareto Optimal）的模型。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 稠密嵌入（Dense Embedding） | "那个向量" | 每段文本一个固定大小的向量。用余弦相似度排序。 |
| 稀疏嵌入（Sparse Embedding） | "学习到的 BM25" | 词表中每个 token 一个权重；大部分为零；端到端训练。 |
| 多向量（Multi-vector） | "ColBERT 风格" | 每个 token 一个向量；MaxSim 评分；索引更大，召回率更高。 |
| Matryoshka | "俄罗斯套娃技巧" | 前 N 维本身就是一个有效的较小嵌入。 |
| MTEB | "那个基准测试" | 大规模文本嵌入基准测试 —— 推出时 56 个任务，v2 中 100+。 |
| BEIR | "检索基准测试" | 18 个零样本检索任务；常用于引用跨领域鲁棒性。 |
| 非对称编码（Asymmetric Encoding） | "查询 ≠ 文档路径" | 模型对查询和文档使用不同的投影。 |

## 延伸阅读

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) —— 双编码器论文。
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) —— 排行榜论文。
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) —— 统一三模式模型。
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) —— 维度阶梯训练目标。
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) —— 生产环境中的后期交互。
- [MTEB 排行榜（Hugging Face）](https://huggingface.co/spaces/mteb/leaderboard) —— 实时排名。
