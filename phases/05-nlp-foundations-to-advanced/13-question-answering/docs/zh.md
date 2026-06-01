# 问答系统

> 三种系统塑造了现代问答技术。抽取式系统找到文本片段。检索增强系统将其扎根于文档。生成式系统直接产生答案。如今的每一个现代AI助手都是这三种方式的混合体。

**类型：** 构建
**语言：** Python
**前置知识：** 第5阶段 · 第11课（机器翻译），第5阶段 · 第10课（注意力机制）
**时间：** 约75分钟

## 问题所在

用户输入"第一代iPhone是什么时候发布的？"，期望得到"2007年6月29日"。而不是"Apple的历史悠久而丰富"。也不是孤立的"2007"没有上下文。用户需要的是一个直接、有依据、正确的答案。

过去十年间，三种架构主导了问答领域。

- **抽取式问答。** 给定一个问题和一段已知包含答案的段落，找到答案片段在段落中的起始和结束位置索引。SQuAD是该领域的标准基准测试。
- **开放域问答。** 段落不是预先给定的。先检索相关段落，然后提取或生成答案。这是当今每一个RAG流水线的基石。
- **生成式/闭卷问答。** 大型语言模型从其参数记忆中回答问题。无需检索。推理速度最快，但在事实可靠性方面最差。

2026年的趋势是混合方式：检索最佳的几段文本，然后提示生成模型基于这些段落生成答案。这就是RAG（检索增强生成），第14课将深入讲解检索部分，本课构建问答部分。

## 核心概念

![问答架构：抽取式、检索增强式、生成式](../assets/qa.svg)

**抽取式。** 使用Transformer（BERT系列）同时编码问题和段落。训练两个预测头来预测答案片段的起始和结束token索引。损失函数是在有效位置上的交叉熵。输出是来自段落的文本片段。按设计不会产生幻觉，但按设计也无法处理段落无法回答的问题。

**检索增强（RAG）。** 两个阶段。首先，检索器从语料库中找到前k个相关段落。然后，阅读器（抽取式或生成式）利用这些段落生成答案。检索器-阅读器（Reader）的分离设计使两者可以独立训练和评估。现代RAG通常在两者之间增加一个重排序器（Reranker）。

**生成式。** 仅解码器的LLM（GPT、Claude、Llama）从学习到的权重中回答问题。没有检索步骤。在常见知识上表现出色，在罕见或近期事实方面表现灾难性。幻觉率与预训练数据中事实出现频率呈负相关。

## 动手构建

### 步骤1：使用预训练模型的抽取式问答

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2`在SQuAD 2.0上训练，该数据集包含无法回答的问题。默认情况下，`question-answering`流水线即使模型的空答案得分最高时也会返回得分最高的片段——它*不会*自动返回空答案。要获得明确的"无答案"行为，请在流水线调用时传入`handle_impossible_answer=True`：此时流水线仅在空答案得分超过所有片段得分时才返回空答案。无论如何，务必始终检查`score`字段。

### 步骤2：检索增强流水线（示意图）

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

两阶段流水线。密集检索器（Dense Retriever）（Sentence-BERT）通过语义相似度找到相关段落。抽取式阅读器（RoBERTa-SQuAD）从合并的前k段文本中提取答案片段。适用于小型语料库。对于百万级文档语料库，请使用FAISS或向量数据库。

### 步骤3：结合RAG的生成式问答

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

提示模式很重要。明确告诉模型基于上下文回答，并在上下文不足时返回"我不知道"，与朴素提示相比可将幻觉率降低40-60%。更精细的模式还可添加引用、置信度分数和结构化提取。

### 步骤4：反映真实世界的评估

SQuAD使用**精确匹配（EM）**和**token级F1**。EM是在规范化（小写、去除标点、删除冠词）后的严格匹配——预测结果要么完全匹配，要么得0分。F1基于预测和参考答案之间的token重叠计算，给予部分分数。两者对改写答案的评价都不足："June 29, 2007"与"June 29th, 2007"通常EM为0（序数词破坏了规范化），但由于token重叠仍能获得可观的F1分数。

对于生产环境的问答系统：

- **答案准确率**（LLM评判或人工评判，因为指标无法捕捉语义等价性）。
- **引用准确率。** 引用的段落是否真的支持该答案？通过生成引用与检索到的段落之间的字符串匹配即可自动检查。
- **拒答校准。** 当答案不在检索到的段落中时，系统是否正确地说"我不知道"？衡量虚假自信率。
- **检索召回率。** 在评估阅读器之前，先衡量检索器是否将正确的段落放入了前k个结果中。阅读器无法弥补缺失的段落。

### RAGAS：2026年的生产评估框架

`RAGAS`专为RAG系统设计，是2026年的默认生产评估工具。它无需标准参考答案即可对四个维度进行评分：

- **忠实度（Faithfulness）。** 答案中的每个主张是否都来自检索到的上下文？通过基于NLI的蕴含关系来衡量。这是你的主要幻觉指标。
- **答案相关性（Answer Relevance）。** 答案是否针对了问题？通过从答案生成假设性问题并与真实问题比较来衡量。
- **上下文精确率（Context Precision）。** 在检索到的文本块中，有多少比例是真正相关的？精确率低意味着提示中有噪声。
- **上下文召回率（Context Recall）。** 检索到的集合是否包含所有所需信息？召回率低意味着阅读器无法成功。

无需参考答案的评分让你可以在真实的生产流量上进行评估，无需精心策划的标准答案。对于精确匹配指标无能为力的开放式问题，可以在其上层叠加LLM作为评判。

`pip install ragas`。接入你的检索器+阅读器。每个查询得到四个标量值。对回归进行告警。

## 使用场景

2026年技术栈。

| 使用场景 | 推荐方案 |
|---------|-------------|
| 给定段落，找到答案片段 | `deepset/roberta-base-squad2` |
| 针对固定语料库，不接受闭卷方式 | RAG：密集检索器 + LLM阅读器 |
| 针对文档存储的实时问答 | 带混合检索器（BM25 + 密集）+ 重排序器的RAG（第14课） |
| 对话式问答（追问） | 带对话历史的LLM + 每轮RAG |
| 高度事实性、受监管的领域 | 基于权威语料库的抽取式；绝不单独使用生成式 |

抽取式问答在2026年不再流行，因为结合LLM的RAG能处理更多场景。但在需要字面引用的场景中仍然在使用：法律研究、合规审查、审计工具。

## 交付使用

保存为 `outputs/skill-qa-architect.md`：

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

## 练习

1. **简单。** 在10段Wikipedia文章上搭建上述SQuAD抽取式流水线。手动编写10个问题。衡量答案正确的频率。如果段落和问题质量良好，你应该能看到7-9个正确答案。
2. **中等。** 添加拒答分类器。当最高检索得分低于阈值（比如0.3余弦相似度）时，返回"我不知道"而不是调用阅读器。在验证集上调整阈值。
3. **困难。** 在你选择的10,000文档语料库上构建RAG流水线。实现混合检索（BM25 + 密集）与RRF融合（参见第14课）。衡量有无混合步骤时的答案准确率。记录哪些类型的问题受益最大。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| 抽取式问答（Extractive QA） | 找到答案片段 | 预测给定段落中答案的起始和结束位置索引。 |
| 开放域问答（Open-Domain QA） | 针对语料库的问答 | 没有给定段落；必须先检索再回答。 |
| RAG | 检索后生成 | 检索增强生成。检索器+阅读器流水线。 |
| SQuAD | 标准基准测试 | 斯坦福问答数据集。EM + F1指标。 |
| 幻觉（Hallucination） | 编造的答案 | 阅读器输出未被检索到的上下文所支持。 |
| 拒答校准（Refusal Calibration） | 知道何时闭嘴 | 系统在无法回答时正确地说"我不知道"。 |

## 延伸阅读

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — 该基准测试的论文。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，问答领域标准的密集检索器。
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — 命名了RAG的论文。
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — 全面的RAG综述。
