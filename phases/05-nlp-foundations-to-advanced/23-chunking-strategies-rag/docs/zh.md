# RAG 的分块策略

> 分块配置对检索质量的影响与嵌入模型的选择一样大（Vectara NAACL 2025）。分块搞错了，再多的重排序也救不了你。

**类型：** 实践
**语言：** Python
**前置知识：** 第 5 阶段 · 14（信息检索），第 5 阶段 · 22（嵌入模型）
**时间：** 约 60 分钟

## 问题所在

你把一份 50 页的合同放进 RAG 系统。用户问："终止条款是什么？"检索器返回的是封面页。为什么？因为模型是基于 512 token 的块训练的，而终止条款在第 20 页，跨越了分页，且没有本地关键词将其与查询关联。

解决方法不是"买一个更好的嵌入模型"。解决方法是分块。多大？重叠多少？在哪里分割？是否包含上下文？

2026 年 2 月的基准测试显示了令人惊讶的结果：

- Vectara 的 2026 年研究：递归 512 token 分块以 69% → 54% 的准确率击败了语义分块。
- SPLADE + Mistral-8B 在 Natural Questions 上：重叠没有带来任何可衡量的收益。
- 上下文悬崖：响应质量在约 2,500 token 上下文时急剧下降。

"显而易见"的答案（语义分块，20% 重叠，1000 token）通常是错误的。本课建立对六种策略的直觉，并告诉你何时使用哪种。

## 概念

![在一段文本上可视化六种分块策略](../assets/chunking.svg)

**固定分块（Fixed Chunking）。** 每 N 个字符或 token 分割一次。最简单的基线。在句子中间断开。压缩率好，连贯性差。

**递归分块（Recursive Chunking）。** LangChain 的 `RecursiveCharacterTextSplitter`。先尝试按 `\n\n` 分割，然后是 `\n`，然后是 `.`，然后是空格。优雅地回退。2026 年的默认选择。

**语义分块。** 对每个句子进行嵌入。计算相邻句子之间的余弦相似度。在相似度低于阈值处分割。保持主题连贯性。速度较慢；有时会产生 40 token 的小片段，损害检索效果。

**句子分块（Sentence Chunking）。** 按句子边界分割。每个块一个句子或 N 个句子的窗口。在约 5k token 范围内与语义分块效果相当，但成本仅为一小部分。

**父子文档分块。** 存储小的子块用于检索*和*更大的父块用于上下文。通过子块检索；返回父块。优雅降级：即使子块检索不好，仍能返回合理的父块。

**后期分块（2024）。** 先在 token 级别对整个文档进行嵌入，然后将 token 嵌入池化为块嵌入。保留跨块上下文。适用于长上下文嵌入器（BGE-M3, Jina v3）。计算量更大。

**上下文检索（Anthropic, 2024）。** 在每个块前加上 LLM 生成的文档位置摘要（"此块是终止条款第 3.2 节..."）。在 Anthropic 自己的基准测试中，检索效果提升了 35-50%。索引成本较高。

### 击败所有默认配置的规则

将块大小与查询类型匹配：

| 查询类型 | 块大小 |
|---------|--------|
| 事实型（"CEO 的名字是什么？"） | 256-512 token |
| 分析型/多跳型 | 512-1024 token |
| 整节理解 | 1024-2048 token |

NVIDIA 的 2026 年基准测试。块应该大到足以包含答案加上本地上下文，小到足以让检索器的 top-K 结果聚焦于答案而非上下文噪声。

## 构建它

### 步骤 1：固定和递归分块

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### 步骤 2：语义分块

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

在你的领域调整 `threshold`。太高 → 碎片化。太低 → 一个巨大的块。

### 步骤 3：父子文档分块

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

关键洞察：去重父块。多个子块可能映射到同一个父块；全部返回会浪费上下文。

### 步骤 4：上下文检索（Anthropic 模式）

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

索引上下文化后的块。在查询时，检索受益于额外的上下文信号。

### 步骤 5：评估

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

始终进行基准测试。你的语料库"最佳"策略可能与任何博客文章都不匹配。

## 常见陷阱

- **仅在事实型查询上评估分块。** 多跳查询会揭示截然不同的赢家。使用按查询类型分层的评估集。
- **语义分块没有最小大小限制。** 会产生 40 token 的碎片，损害检索效果。始终强制执行 `min_tokens`。
- **重叠如同货物崇拜。** 2026 年研究发现，重叠通常没有任何收益，却使索引成本翻倍。要测量，不要假设。
- **没有最小/最大限制。** 5 token 或 5000 token 的块都会破坏检索。要进行限制。
- **跨文档分块。** 绝不要让一个块跨越两个文档。始终按文档分块，然后合并。

## 使用它

2026 年的技术栈：

| 场景 | 策略 |
|------|------|
| 首次构建，未知语料库 | 递归，512 token，无重叠 |
| 事实型问答 | 递归，256-512 token |
| 分析型/多跳型 | 递归，512-1024 token + 父子文档分块 |
| 大量交叉引用（合同、论文） | 后期分块或上下文检索 |
| 对话/对话语料库 | 轮次级块 + 说话者元数据 |
| 短文本（推文、评论） | 一个文档 = 一个块 |

从递归 512 开始。在 50 个查询的评估集上测量 recall@5。从那里调整。

## 交付它

保存为 `outputs/skill-chunker.md`：

```markdown
---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
```

## 练习

1. **简单。** 用 fixed(512, 0)、recursive(512, 0) 和 recursive(512, 100) 分块一个 20 页的文档。比较块数量和边界质量。
2. **中等。** 在 5 个文档上构建一个 30 个查询的评估集。测量递归、语义和父子文档分块的 recall@5。哪个胜出？与博客文章匹配吗？
3. **困难。** 实现上下文检索。测量相对于基线递归的 MRR 改进。报告索引成本（LLM 调用）与准确率增益。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 块（Chunk） | 文档的一块 | 被嵌入、索引和检索的子文档单元。 |
| 重叠（Overlap） | 安全边际 | 相邻块之间共享的 N 个 token；在 2026 年基准测试中通常无用。 |
| 语义分块（Semantic Chunking） | 智能分块 | 在相邻句子嵌入相似度下降处分割。 |
| 父子文档（Parent-child Document） | 两级检索 | 检索小子块，返回大父块。 |
| 后期分块（Late Chunking） | 嵌入后再分块 | 在 token 级别对完整文档进行嵌入，然后池化为块向量。 |
| 上下文检索（Contextual Retrieval） | Anthropic 的技巧 | 索引前在每个块前加上 LLM 生成的摘要。 |
| 上下文悬崖（Context Cliff） | 2500 token 墙 | 2026 年 1 月在 RAG 中观察到的约 2.5k 上下文 token 时的质量下降。 |

## 扩展阅读

- [Yepes 等人 / LangChain — 递归字符分割文档](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — 生产环境中的默认选择。
- [Vectara (2024, NAACL 2025). 分块配置分析](https://arxiv.org/abs/2410.13070) — 分块与嵌入选择同样重要。
- [Jina AI — 长上下文嵌入模型中的后期分块 (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — 后期分块论文。
- [Anthropic — 上下文检索](https://www.anthropic.com/news/contextual-retrieval) — 使用 LLM 生成的上下文前缀，检索效果提升 35-50%。
- [NVIDIA 2026 块大小基准测试 — Premai 摘要](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — 按查询类型的块大小。
