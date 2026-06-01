# 关系抽取与知识图谱构建

> NER 找到了实体。实体链接锚定了它们。关系抽取（Relation Extraction）找到了它们之间的边。知识图谱（Knowledge Graph）是节点、边及其来源的总和。

**类型：** 构建
**语言：** Python
**前置课程：** Phase 5 · 06（NER），Phase 5 · 25（实体链接）
**时间：** 约 60 分钟

## 问题

一位分析师读到："Tim Cook 于 2011 年成为 Apple 的 CEO。"这里有四个事实：

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

关系抽取（RE）将自由文本转化为结构化三元组 `(subject, relation, object)`。在语料库中进行聚合，你就得到了一个知识图谱。聚合并查询，你就得到了一个用于 RAG、分析或合规审计的推理基础。

2026 年的问题：LLM 很热心地抽取关系。太热心了。它们会编造源文本不支持的三元组。没有溯源信息，你无法区分真实三元组和看似合理的虚构内容。2026 年的答案是 AEVS 风格的锚定-验证流水线。

## 核心概念

![文本 → 三元组 → 知识图谱](../assets/relation-extraction.svg)

**三元组形式。** `(subject_entity, relation_type, object_entity)`。关系来自封闭本体（Wikidata 属性、FIBO、UMLS）或开放集合（OpenIE 风格，什么都可以）。

**三种抽取方法。**

1. **基于规则/模式（Pattern）。** Hearst 模式："X such as Y" → `(Y, isA, X)`。加上手工编写的正则表达式。脆弱但精确、可解释。
2. **监督分类器。** 给定句子中的两个实体提及，从固定集合中预测关系。在 TACRED、ACE、KBP 上训练。2015-2022 年的标准方法。
3. **生成式 LLM。** 提示模型输出三元组。开箱即用。需要溯源信息，否则会编造看似合理的垃圾内容。

**AEVS（锚定-抽取-验证-补充，2026）。** 当前的幻觉（Hallucination）缓解框架：

- **锚定（Anchor）。** 用精确位置识别每个实体片段和关系短语片段。
- **抽取（Extract）。** 生成与锚定片段关联的三元组。
- **验证（Verify）。** 将每个三元组元素匹配回源文本；拒绝任何不被支持的内容。
- **补充（Supplement）。** 覆盖率检查确保没有锚定片段被遗漏。

幻觉大幅减少。需要更多计算资源，但可审计。

**开放与封闭的权衡。**

- **封闭本体。** 固定属性列表（如 Wikidata 的 11,000+ 属性）。可预测。可查询。难以凭空捏造。
- **开放 IE。** 任何动词短语都可以成为关系。高召回率（Recall）。低精确率（Precision）。查询起来很混乱。

生产环境的知识图谱通常是混合的：用开放 IE 进行发现，然后在合并到主图之前将关系规范化到封闭本体上。

## 动手实现

### 第 1 步：基于模式的抽取

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

完整的玩具抽取器参见 `code/main.py`。Hearst 模式仍然出现在特定领域的流水线中，因为它们易于调试。

### 第 2 步：监督关系分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL 是一个 seq2seq 关系抽取器：输入文本，输出三元组，已经是 Wikidata 属性 ID。在远程监督数据上微调。标准的开源权重基线。

### 第 3 步：带锚定的 LLM 提示抽取

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

验证返回的每个片段是否与源文本匹配。拒绝任何 `text[start:end] != triple_entity` 的内容。这是 AEVS "验证"步骤的最简形式。

### 第 4 步：规范化到封闭本体

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # drop unmapped open relations or route to manual review
```

规范化通常占工程工作量的 60-80%。做好预算。

### 第 5 步：构建小型图谱并查询

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

这是每个基于知识图谱的 RAG 系统的基本单元。可以用 RDF 三元组存储（Blazegraph、Virtuoso）、属性图（Neo4j）或向量增强图存储来扩展它。

## 常见陷阱

- **在 RE 之前做共指消解。** "He founded Apple" — RE 需要知道 "he" 是谁。先运行共指消解（第 24 课）。
- **实体规范化。** "Apple Inc" 和 "Apple" 必须解析到同一个节点。先做实体链接（第 25 课）。
- **幻觉三元组。** LLM 会输出文本不支持的三元组。强制执行片段验证。
- **关系规范化漂移。** 开放 IE 的关系不一致（"was born in"、"came from"、"is a native of"）。合并到规范 ID，否则图谱无法查询。
- **时间错误。** "Tim Cook is CEO of Apple" — 现在是真的，2005 年是假的。许多关系有时间边界。使用限定符（Qualifier）（Wikidata 中的 `P580` 开始时间、`P582` 结束时间）。
- **领域不匹配。** REBEL 在 Wikipedia 上训练。法律、医学和科学文本通常需要领域微调的 RE 模型。

## 使用指南

2026 年的技术栈：

| 场景 | 选择 |
|------|------|
| 快速生产，通用领域 | REBEL 或 LlamaPred 配合 Wikidata 规范化 |
| 特定领域（生物医学、法律） | SciREX 风格的领域微调 + 自定义本体 |
| LLM 提示，可审计输出 | AEVS 流水线：锚定 → 抽取 → 验证 → 补充 |
| 大量新闻 IE | 基于模式 + 监督的混合方法 |
| 从零构建知识图谱 | 开放 IE + 手动规范化 |
| 时序知识图谱 | 带限定符抽取（开始/结束时间、时间点） |

集成模式：NER → 共指消解 → 实体链接 → 关系抽取 → 本体映射 → 图谱加载。每个阶段都是潜在的质量关口。

## 交付

保存为 `outputs/skill-re-designer.md`：

```markdown
---
name: re-designer
description: Design a relation extraction pipeline with provenance and canonicalization.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Given a corpus (domain, language, volume) and downstream use (KG-RAG, analytics, compliance), output:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Reason tied to precision vs recall target.
2. Ontology. Closed property list (Wikidata / domain) or open IE with canonicalization pass.
3. Provenance. Every triple carries source char-span + doc id. Non-negotiable for audit.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. Precision / recall on 200 hand-labelled triples + hallucination-rate on LLM-extracted sample.

Refuse any LLM-based RE pipeline without span verification (source provenance). Refuse open-IE output flowing into a production graph without canonicalization. Flag pipelines with no temporal qualifier on time-bounded relations (employer, spouse, position).
```

## 练习

1. **简单。** 在 5 个新闻句子上运行 `code/main.py` 中的模式抽取器。手动检查精确率。
2. **中等。** 在相同句子上使用 REBEL（或小型 LLM）。比较三元组。哪个抽取器精确率更高？召回率更高？
3. **困难。** 构建 AEVS 流水线：用 LLM 抽取 + 验证片段与源文本的匹配。在 50 个 Wikipedia 风格的句子上测量验证步骤前后的幻觉率。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 三元组（Triple） | 主语-关系-宾语 | `(s, r, o)` 元组，知识图谱的基本单元。 |
| 开放 IE | 抽取任何内容 | 开放词汇关系短语；高召回率，低精确率。 |
| 封闭本体（Closed Ontology） | 固定模式 | 有限的关系类型集合（Wikidata、UMLS、FIBO）。 |
| 规范化（Canonicalization） | 统一一切 | 将表面名称/关系映射到规范 ID。 |
| AEVS | 基于锚定的抽取 | 锚定-抽取-验证-补充流水线（2026）。 |
| 溯源（Provenance） | 来源链接 | 每个三元组携带文档 ID + 字符片段指向其来源。 |
| 远程监督（Distant supervision） | 廉价标签 | 将文本与现有知识图谱对齐以创建训练数据。 |

## 扩展阅读

- [Mintz 等人 (2009). 无标注数据的关系抽取远程监督](https://www.aclweb.org/anthology/P09-1113.pdf) — 远程监督论文。
- [Huguet Cabot, Navigli (2021). REBEL: 基于端到端语言生成的关系抽取](https://aclanthology.org/2021.findings-emnlp.204.pdf) — seq2seq RE 主力模型。
- [Wadden 等人 (2019). 基于上下文化片段表示的实体、关系和事件抽取 (DyGIE++)](https://arxiv.org/abs/1909.03546) — 联合 IE。
- [AEVS — 锚定-抽取-验证-补充框架](https://www.mdpi.com/2073-431X/15/3/178) — 2026 年幻觉缓解设计。
- [Wikidata SPARQL 教程](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — 规范图谱查询。
