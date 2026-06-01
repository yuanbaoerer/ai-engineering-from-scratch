# 文本摘要

> 抽取式（Extractive）系统告诉你文档说了什么。生成式（Abstractive）系统告诉你作者想表达什么。不同的任务，不同的陷阱。

**类型：** 构建
**语言：** Python
**前置知识：** 第 5 阶段 · 02（词袋 + TF-IDF），第 5 阶段 · 11（机器翻译）
**时间：** 约 75 分钟

## 问题描述

一篇 2,000 字的新闻文章出现在你的信息流中。你需要用 120 个字概括它。你可以从文章中挑选三个最重要的句子（抽取式），或者用自己的话重写内容（生成式）。两者都被称为摘要。但它们是完全不同的问题。

抽取式摘要本质上是一个排序问题。给每个句子打分，返回得分最高的 `top-k` 个句子。输出总是语法正确的，因为它是原文逐字提取的。风险在于可能会遗漏分布在文章各处的内容。

生成式摘要本质上是一个生成问题。Transformer 根据输入生成新的文本。输出流畅且具有压缩性，但可能会幻觉（Hallucination）出源文档中不存在的事实。风险在于自信地编造内容。

本课将同时构建这两种方法，并分析各自的失败模式。

## 核心概念

![抽取式 TextRank 与生成式 Transformer](../assets/summarization.svg)

**抽取式。** 将文章视为一个图，节点是句子，边是相似度。在图上运行 PageRank（或类似算法），根据句子与其他句子的连接程度来打分。得分最高的句子就是摘要。经典实现是 **TextRank**（Mihalcea 和 Tarau，2004）。

**生成式。** 在文档-摘要对上微调 Transformer 编码器-解码器（BART、T5、Pegasus）。推理时，模型读取文档并通过交叉注意力（Cross-Attention）逐个 token 生成摘要。Pegasus 特别使用了 gap-sentence 预训练目标，使其无需大量微调就能出色地完成摘要任务。

使用 **ROUGE**（Recall-Oriented Understudy for Gisting Evaluation）进行评估。ROUGE-1 和 ROUGE-2 评估 unigram 和 bigram 重叠度。ROUGE-L 评估最长公共子序列。分数越高越好，但 40 ROUGE-L 算"良好"，50 算"卓越"。每篇论文都会报告这三个指标。使用 `rouge-score` 包。

## 构建

### 步骤 1：TextRank（抽取式）

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

有两点值得注意。相似度函数使用了对数归一化的词重叠度，这是原始 TextRank 的变体。TF-IDF 向量的余弦相似度也可以。阻尼因子 0.85 和迭代次数是 PageRank 的默认值。

### 步骤 2：使用 BART 进行生成式摘要

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN 在 CNN/DailyMail 语料库上进行了微调。它开箱即用即可生成新闻风格的摘要。对于其他领域（科学论文、对话、法律），请使用相应的 Pegasus 检查点或在目标数据上进行微调。

### 步骤 3：ROUGE 评估

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

始终使用词干提取（Stemming）。如果不使用，"running"和"run"会被算作不同的词，导致 ROUGE 低估分数。

### ROUGE 之外（2026 年摘要评估）

ROUGE 作为主导性摘要指标已有二十年历史，但在 2026 年单独使用已不足够。一项对 NLG 论文的大规模元分析表明：

- **BERTScore**（上下文嵌入相似度）在 2023 年前持续增长，现在大多数摘要论文都会同时报告 ROUGE 和 BERTScore。
- **BARTScore** 将评估视为生成问题：通过预训练 BART 对摘要在给定源文档下的可能性进行打分。
- **MoverScore**（基于上下文嵌入的推土机距离）在 2025 年摘要基准测试中登顶，因为它比 ROUGE 更好地捕捉语义重叠。
- **FactCC** 和 **基于 QA 的忠实度评估** 在 2021-2023 年较为常见，现在通常被 **G-Eval**（一种使用 GPT-4 提示链对连贯性、一致性、流畅性、相关性进行思维链推理评分的方法）所取代。
- **G-Eval** 和类似的 LLM 评判方法在评分标准设计良好时，与人类判断的一致性约为 80%。

生产建议：报告 ROUGE-L 用于历史对比，BERTScore 用于语义重叠，G-Eval 用于连贯性和事实性。使用 50-100 个人工标注的摘要进行校准。

### 步骤 4：事实性问题

生成式摘要容易产生幻觉。抽取式摘要的幻觉风险要低得多，因为输出是逐字从源文档提取的，不过如果源句子被断章取义、过时或引用顺序错误，抽取式摘要仍可能产生误导。这是生产系统在合规相关内容中仍然偏好抽取式方法的最大原因。

需要了解的幻觉类型：

- **实体替换。** 源文档说的是"John Smith"。摘要说的是"John Brown"。
- **数字漂移。** 源文档说的是"25,000"。摘要说的是"25 million"。
- **极性翻转。** 源文档说的是"rejected the offer"。摘要说的是"accepted the offer"。
- **事实捏造。** 源文档没有提到 CEO。摘要说 CEO 批准了。

有效的评估方法：

- **FactCC。** 一个在源句子和摘要句子之间的蕴含关系上训练的二元分类器。预测事实/非事实。
- **基于 QA 的事实性评估。** 向 QA 模型提出答案在源文档中的问题。如果摘要支持不同的答案，则标记为可疑。
- **实体级 F1。** 比较源文档和摘要中的命名实体（Named Entity）。仅出现在摘要中的实体是可疑的。

对于任何面向用户且事实性重要的场景（新闻、医疗、法律、金融），抽取式是更安全的默认选择。生成式需要在流程中加入事实性检查。

## 应用

2026 年技术栈：

| 用例 | 推荐方案 |
|------|----------|
| 新闻，3-5 句摘要，英文 | `facebook/bart-large-cnn` |
| 科学论文 | `google/pegasus-pubmed` 或调优的 T5 |
| 多文档，长文本 | 任何具有 32k+ 上下文的 LLM，使用提示 |
| 对话摘要 | `philschmid/bart-large-cnn-samsum` |
| 抽取式，构造上低幻觉风险 | TextRank 或 `sumy` 的 LSA / LexRank |

在 2026 年，当计算资源不是限制因素时，具有长上下文的 LLM 通常能超越专用模型。权衡在于成本和可复现性；专用模型能提供更一致的输出。

## 交付

保存为 `outputs/skill-summary-picker.md`：

```markdown
---
name: summary-picker
description: 选择抽取式或生成式，指定库，事实性检查。
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

给定一个任务（文档类型、合规要求、长度、计算预算），输出：

1. 方法。抽取式或生成式。用一句话解释原因。
2. 起始模型/库。指定名称。`sumy.TextRankSummarizer`、`facebook/bart-large-cnn`、`google/pegasus-pubmed` 或 LLM 提示。
3. 评估计划。ROUGE-1、ROUGE-2、ROUGE-L（使用带词干提取的 rouge-score）。如果是生成式，还需事实性检查。
4. 一个需要探查的失败模式。在生成式新闻摘要中，实体替换是最常见的；标记源实体未出现在摘要中的样本。

对于医疗、法律、金融或受监管内容，如果没有事实性门控，拒绝使用生成式摘要。当输入超过模型上下文窗口时，标记为需要分块 map-reduce 摘要（而非简单截断）。
```

## 练习

1. **简单。** 对 5 篇新闻文章运行 TextRank。将 top-3 句子与参考摘要进行比较。测量 ROUGE-L。你应该能在 CNN/DailyMail 风格的文章上看到 30-45 的 ROUGE-L。
2. **中等。** 实现实体级事实性评估：从源文档和摘要中提取命名实体（spaCy），计算源实体在摘要中的召回率和摘要实体相对于源文档的精确率。高精确率和低召回率意味着安全但简短；低精确率意味着存在幻觉实体。
3. **困难。** 在 50 篇 CNN/DailyMail 文章上比较 BART-large-CNN 与 LLM（Claude 或 GPT-4）。报告 ROUGE-L、事实性（通过实体 F1）以及每个摘要的成本。记录各自的优势领域。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| 抽取式（Extractive） | 挑选句子 | 从源文档逐字返回句子。永远不会产生幻觉。 |
| 生成式（Abstractive） | 重写 | 根据源文档生成新文本。可能产生幻觉。 |
| ROUGE | 摘要指标 | 系统输出与参考之间的 N-gram/LCS 重叠度。 |
| TextRank | 基于图的抽取式 | 在句子相似度图上运行 PageRank。 |
| 事实性（Factuality / Faithfulness） | 是否正确 | 摘要中的陈述是否得到源文档支持。 |
| 幻觉（Hallucination） | 编造的内容 | 摘要中源文档不支持的内容。 |

## 延伸阅读

- [Mihalcea 和 Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) —— 抽取式经典论文。
- [Lewis 等 (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) —— BART 论文。
- [Zhang 等 (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) —— Pegasus 和 gap-sentence 目标。
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) —— ROUGE 论文。
- [Maynez 等 (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) —— 事实性综述论文。
