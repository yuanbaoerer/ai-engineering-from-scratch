# 实体链接与消歧

> NER 识别出"Paris"。实体链接决定：Paris, France？Paris Hilton？Paris, Texas？Paris（特洛伊王子）？不做链接，你的知识图谱就始终是模糊的。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 5 · 06（NER），Phase 5 · 24（共指消解）
**时间：** 约 60 分钟

## 问题所在

一句话是："Jordan beat the press." 你的 NER 将"Jordan"标记为 PERSON。很好。但*哪个* Jordan？

- Michael Jordan（篮球运动员）？
- Michael B. Jordan（演员）？
- Michael I. Jordan（伯克利 ML 教授——没错，这种混淆在 ML 论文中真实存在）？
- Jordan（国家）？
- Jordan（希伯来语名字）？

实体链接（EL）将每个提及解析为知识库中的唯一条目：Wikidata、Wikipedia、DBpedia 或你的领域知识库。包含两个子任务：

1. **候选生成。** 给定"Jordan"，哪些知识库条目是合理的？
2. **消歧（Disambiguation）。** 给定上下文，哪个候选是正确的？

两个步骤都是可学习的。两个步骤都有基准测试。整个组合流水线已经稳定了十年——变化的是消歧器的质量。

## 核心概念

![实体链接流水线：提及 → 候选 → 消歧后实体](../assets/entity-linking.svg)

**候选生成。** 给定提及的表面形式（"Jordan"），在别名索引中查找候选。Wikipedia 别名词典覆盖了大多数命名实体："JFK" → John F. Kennedy、Jacqueline Kennedy、JFK 机场、JFK（电影）。典型索引每个提及返回 10-30 个候选。

**消歧：三种方法。**

1. **先验（Prior） + 上下文（Milne & Witten, 2008）。** `P(entity | mention) × context-similarity(entity, text)`。效果好，速度快，无需训练。
2. **基于嵌入（ESS / REL / Blink）。** 编码提及 + 上下文。编码每个候选的描述。选取最大余弦相似度（Cosine Similarity）。2020-2024 年的默认方案。
3. **生成式（GENRE, 2021；基于 LLM, 2023+）。** 逐个 token 解码实体的规范名称。受有效实体名称的 trie 树约束，保证输出为有效的知识库 ID。

**端到端 vs 流水线。** 现代模型（ELQ、BLINK、ExtEnD、GENRE）在一次前向传播中运行 NER + 候选生成 + 消歧。流水线系统在生产环境中仍然占主导，因为你可以替换组件。

### 两个度量指标

- **提及召回率（Mention Recall）（候选生成）。** 在所有标准提及中，正确的知识库条目出现在候选列表中的比例。整个流水线的下限。
- **消歧准确率（Disambiguation Accuracy） / F1。** 给定正确候选，top-1 命中的比例。

始终报告两者。一个在 80% 候选召回率上达到 99% 消歧准确率的系统，是一个 80% 召回率（Recall）的流水线。

## 构建过程

### 第 1 步：从 Wikipedia 重定向构建别名索引

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Wikipedia 别名数据：约 1800 万（别名，实体）对。从 Wikidata 转储下载。存储为倒排索引（Inverted Index）。

### 第 2 步：基于上下文的消歧

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Jaccard 重叠只是一个示例。替换为嵌入的余弦相似度（参见 `code/main.py` step-2 的 Transformer 版本）。

### 第 3 步：基于嵌入的方法（BLINK 风格）

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

在索引阶段，对每个知识库实体嵌入一次。在查询阶段，对提及 + 上下文嵌入一次，与候选池做点积（Dot Product），选取最大值。

### 第 4 步：生成式实体链接（概念）

GENRE 逐字符解码实体的 Wikipedia 标题。受限解码（参见第 20 课）确保只能输出有效的标题。与基于知识库的 trie 树紧密集成。现代后继者是 REL-GEN 和带有结构化输出的 LLM 提示 EL。

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

结合白名单（Outlines `choice`），这是 2026 年最简单的可部署 EL 流水线。

### 第 5 步：在 AIDA-CoNLL 上评估

AIDA-CoNLL 是标准 EL 基准测试：1,393 篇路透社文章，34k 个提及，Wikipedia 实体。报告知识库内准确率（`P@1`）和知识库外 NIL 检测率。

## 常见陷阱

- **NIL 处理。** 一些提及不在知识库中（新兴实体、小众人物）。系统必须预测 NIL 而非猜测错误实体。需要单独衡量。
- **提及边界错误。** 上游 NER 遗漏部分跨度（"Bank of America" 仅被标记为"Bank"）。EL 召回率下降。
- **流行度偏差。** 训练过的系统过度预测高频实体。ML 论文中提及"Michael I. Jordan"时，往往链接到篮球运动员 Jordan。
- **跨语言 EL。** 将中文文本中的提及映射到英文 Wikipedia 实体。需要多语言编码器或翻译步骤。
- **知识库过时。** 新公司、事件、人物不在去年的 Wikipedia 转储中。生产流水线需要刷新循环。

## 使用建议

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 通用英文 + Wikipedia | BLINK 或 REL |
| 跨语言，知识库 = Wikipedia | mGENRE |
| LLM 友好，每天少量提及 | 用候选列表 + 受限 JSON 提示 Claude/GPT-4 |
| 领域特定知识库（医疗、法律） | 自定义 BERT + 知识库感知检索 + 在领域 AIDA 风格数据集上微调 |
| 极低延迟 | 仅使用精确匹配先验（Milne-Witten 基线） |
| 研究 SOTA | GENRE / ExtEnD / 生成式 LLM-EL |

2026 年可部署的生产模式：NER → 共指消解 → 对每个提及做 EL → 将聚类折叠为每个聚类一个规范实体。输出：文档中每个实体一个知识库 ID，而非每个提及一个。

## 交付成果

保存为 `outputs/skill-entity-linker.md`：

```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

## 练习

1. **简单。** 在 `code/main.py` 中实现先验 + 上下文消歧器，处理 10 个歧义提及（Paris、Jordan、Apple）。手动标注正确实体。衡量准确率。
2. **中等。** 用 sentence transformer 编码 50 个歧义提及。嵌入每个候选的描述。将基于嵌入的消歧与 Jaccard 上下文重叠进行比较。
3. **困难。** 构建一个 1k 实体的领域知识库（例如你公司的员工 + 产品）。实现端到端 NER + EL。在 100 个留出句子上衡量精确率（Precision）和召回率（Recall）。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| 实体链接（EL） | 链接到 Wikipedia | 将提及映射到知识库的唯一条目。 |
| 候选生成（Candidate Generation） | 可能是谁？ | 为提及返回一组合理的知识库条目候选。 |
| 消歧 | 选对的那个 | 使用上下文对候选打分，选出胜者。 |
| 别名索引（Alias Index） | 查找表 | 从表面形式映射到候选实体。 |
| NIL | 不在知识库中 | 明确预测没有知识库条目匹配。 |
| KB | 知识库 | Wikidata、Wikipedia、DBpedia 或你的领域知识库。 |
| AIDA-CoNLL | 基准测试 | 1,393 篇带有标准实体链接的路透社文章。 |

## 延伸阅读

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) — 先验 + 上下文的奠基性方法。
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) — 基于嵌入的主力方案。
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) — 带受限解码的生成式 EL。
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) — 基准测试论文。
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) — 开源生产级技术栈。
