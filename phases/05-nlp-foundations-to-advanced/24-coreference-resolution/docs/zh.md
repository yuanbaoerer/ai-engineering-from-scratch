# 共指消解

> "她给他打了电话。他没有接。医生在吃午饭。"三处指代涉及两个人，却没有提到任何名字。共指消解（Coreference Resolution）能弄清楚谁是谁。

**类型：** 学习
**语言：** Python
**前置知识：** 第5阶段 · 06（命名实体识别）、第5阶段 · 07（词性标注与句法分析）
**时间：** 约60分钟

## 问题描述

从一篇300词的文章中提取所有提及苹果公司的内容。当文章直接写"Apple"时很容易，但当它写"the company"、"they"、"Cupertino's technology giant"或"Jobs's firm"时就很困难。如果不将这些指代解析到同一实体，你的命名实体识别管道会遗漏60-80%的提及。

共指消解将所有指向同一真实世界实体的表达链接到一个簇中。它是表层NLP（命名实体识别、句法分析）与下游语义（信息抽取、问答、摘要、知识图谱）之间的粘合剂。

2026年它为何重要：

- 摘要："The CEO announced..."与"Tim Cook announced..."——摘要应该指名道姓。
- 问答："Who did she call?"需要解析"she"指代谁。
- 信息抽取：知识图谱中把"PER1 founded Apple"和"Jobs founded Apple"作为两条独立条目是错误的。
- 跨文档信息抽取：合并关于同一事件的不同文章中的提及就是跨文档共指消解。

## 概念

![共指聚类：提及 → 实体](../assets/coref.svg)

**任务描述。** 输入：一篇文档。输出：提及（文本片段）的聚类，其中每个簇对应一个实体。

**提及类型。**

- **命名实体。** "Tim Cook"
- **名词性。** "the CEO"、"the company"
- **代词性。** "he"、"she"、"they"、"it"
- **同位语。** "Tim Cook, Apple's CEO,"

**架构。**

1. **基于规则（Hobbs, 1978）。** 基于语法树的代词解析，使用语法规则。良好的基线。在代词上出人意料地难以超越。
2. **提及对分类器。** 对每一对提及 (m_i, m_j)，预测它们是否共指。通过传递闭包（Transitive Closure）进行聚类。2016年之前的标准方法。
3. **提及排序。** 对每个提及，对候选先行词进行排序（包括"无先行词"）。选择排名最高的。
4. **基于片段的端到端（End-to-end）（Lee et al., 2017）。** Transformer编码器。枚举所有候选片段（达到长度上限）。预测提及分数。为每个片段预测先行词概率。贪心聚类。现代默认方法。
5. **生成式（2024+）。** 用提示词让大语言模型回答："列出此文本中的每个代词及其先行词。"在简单案例上效果不错，在长文档和罕见指代上表现欠佳。

**评估指标。** 五种标准指标（MUC、B³、CEAF、BLANC、LEA），因为没有单一指标能完美捕捉聚类质量。将前三种的平均值作为CoNLL F1报告。2026年在CoNLL-2012上的最先进水平：约83 F1。

**已知的困难案例。**

- 指代几页之前引入的实体的定指描述（Definite Description）。
- 桥接回指（"the wheels" -> 前面提到的一辆车）。
- 中文、日文等语言中的零回指（Zero Anaphora）。
- 预指（先行词之前的代词）：When **she** walked in, Mary smiled.

## 动手实现

### 第1步：预训练神经网络共指消解（AllenNLP / spaCy-experimental）

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # 实验性模型
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

在较长的文档上，你会得到类似结果：
- 簇 1：[Apple, The company, they]
- 簇 2：[new products]

### 第2步：基于规则的代词解析器（教学用途）

参见 `code/main.py` 获取一个仅使用标准库的实现：

1. 提取提及：命名实体（大写片段）、代词（字典查找）、定指描述（"the X"）。
2. 对每个代词，查看前K个提及，按以下标准打分：
   - 性别/数一致（启发式规则）
   - 相近度（更近的优先）
   - 句法角色（主语优先）
3. 链接到得分最高的先行词。

无法与神经网络模型竞争。但它展示了搜索空间以及端到端模型必须做出的决策。

### 第3步：使用大语言模型进行共指消解

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

需要注意的两种失败模式。第一，大语言模型过度合并（"him"和"her"指代两个不同的人）。第二，大语言模型在长文档中会悄悄遗漏提及。务必通过片段偏移检查来验证。

### 第4步：评估

标准的 conll-2012 脚本计算 MUC、B³、CEAF-φ4 并报告平均值。对于内部评估，先在标注的测试集上计算片段级别的精确率（Precision）和召回率（Recall），然后增加提及链接 F1。

## 常见陷阱

- **单例爆炸。** 有些系统将每个提及报告为独立的簇。B³对此较为宽容。MUC会惩罚这种情况。务必检查所有三个指标。
- **长上下文中的代词。** 在超过2000个token的文档上，性能下降约15 F1。请谨慎分块。
- **性别假设。** 硬编码的性别规则在非二元指代者、组织、动物身上会失效。使用学习模型或中性打分。
- **大语言模型在长文档上的漂移。** 单次API调用无法可靠地在50+段落的文档中聚类提及。使用滑动窗口（Sliding Window）+合并策略。

## 应用选择

2026年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 英文、单文档 | `en_coreference_web_trf`（spaCy-experimental）或 AllenNLP 神经网络共指 |
| 多语言 | SpanBERT / XLM-R，在OntoNotes或多语言CoNLL上训练 |
| 跨文档事件共指 | 专用端到端模型（2025-2026年最先进水平） |
| 快速大语言模型基线 | GPT-4o / Claude，使用结构化输出的共指提示词 |
| 生产级对话系统 | 基于规则的后备方案 + 神经网络主模型 + 关键槽位人工审查 |

2026年投入使用的集成模式：先运行命名实体识别，再运行共指消解，将共指簇合并到命名实体中。下游任务看到的是每个簇一个实体，而不是每个提及一个实体。

## 交付物

保存为 `outputs/skill-coref-picker.md`：

```markdown
---
name: coref-picker
description: Pick a coreference approach, evaluation plan, and integration strategy.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

Given a use case (single-doc / multi-doc, domain, language), output:

1. Approach. Rule-based / neural span-based / LLM-prompted / hybrid. One-sentence reason.
2. Model. Named checkpoint if neural.
3. Integration. Order of operations: tokenize → NER → coref → downstream task.
4. Evaluation. CoNLL F1 (MUC + B³ + CEAF-φ4 average) on held-out set + manual cluster review on 20 documents.

Refuse LLM-only coref for documents over 2,000 tokens without sliding-window merge. Refuse any pipeline that runs coref without a mention-level precision-recall report. Flag gender-heuristic systems deployed in demographically diverse text.
```

## 练习

1. **简单。** 在5个手工编写的段落上运行 `code/main.py` 中的基于规则的解析器。根据真实标注衡量提及链接准确率。
2. **中等。** 在一篇新闻文章上使用预训练的神经网络共指模型。将聚类结果与你自己的人工标注进行比较。失败在哪里？
3. **困难。** 构建一个共指增强的命名实体识别管道：先运行命名实体识别，然后通过共指簇合并。在100篇文章上衡量实体覆盖率相比仅使用命名实体识别的提升。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| 提及（Mention） | 一处引用 | 指向一个实体的文本片段（名称、代词、名词短语）。 |
| 先行词（Antecedent） | "it"指代的对象 | 后续提及与之共指的较早提及。 |
| 簇（Cluster） | 实体的所有提及 | 全部指向同一真实世界实体的提及集合。 |
| 回指（Anaphora） | 向后引用 | 后续提及指向较早提及（"he" -> "John"）。 |
| 预指（Cataphora） | 向前引用 | 较早提及指向后续提及（"When he arrived, John..."）。 |
| 桥接（Bridging） | 隐式引用 | "I bought a car. The wheels were bad."（那辆车的轮子。） |
| CoNLL F1 | 排行榜上的数字 | MUC、B³、CEAF-φ4 F1分数的平均值。 |

## 延伸阅读

- [Jurafsky & Martin, SLP3 第26章 — 共指消解与实体链接](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — 经典教材章节。
- [Lee et al. (2017). 端到端神经网络共指消解](https://arxiv.org/abs/1707.07045) — 基于片段的端到端方法。
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) — 提升共指消解效果的预训练方法。
- [Pradhan et al. (2012). CoNLL-2012 共享任务](https://aclanthology.org/W12-4501/) — 基准评测。
- [Hobbs (1978). 代词引用消解](https://www.sciencedirect.com/science/article/pii/0024384178900064) — 基于规则的经典方法。
