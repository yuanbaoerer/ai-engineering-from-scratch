# 多语言自然语言处理

> 一个模型，100+ 种语言，大多数语言零训练数据。跨语言迁移是 2020 年代的实用奇迹。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 5 · 04（GloVe、FastText、子词），Phase 5 · 11（机器翻译）
**时间：** 约 45 分钟

## 问题

英语有数十亿标注样本。乌尔都语有数千个。迈蒂利语几乎一个都没有。任何服务于全球用户的实用 NLP 系统，都必须能够在缺乏任务特定训练数据的语言长尾上工作。

多语言模型通过在多种语言上同时训练一个模型来解决这个问题。共享的表示让模型能够将在高资源语言中学到的技能迁移到低资源语言上。在英语情感分析上微调模型，它就能直接在乌尔都语上产生令人惊讶的好情感预测结果。这就是零样本跨语言迁移，它重塑了 NLP 服务全球的方式。

本课程将说明权衡取舍、经典模型，以及一个让刚接触多语言工作的团队最容易犯错的决策：选择用于迁移的源语言。

## 核心概念

![通过共享多语言嵌入空间实现跨语言迁移](../assets/multilingual.svg)

**共享词汇表。** 多语言模型使用在所有目标语言文本上训练的 SentencePiece 或 WordPiece 分词器。词汇表是共享的：同一个子词单元在相关语言中表示同一个语素。英语和意大利语中的 `anti-` 获得相同的 token。

**共享表示。** 在多种语言上通过掩码语言建模（Masked Language Modeling / MLM）预训练的 Transformer 学会了这样一个规律：不同语言中语义相似的句子会产生相似的隐藏状态。mBERT、XLM-R 和 NLLB 都表现出这种特性。英语中"cat"的嵌入在向量空间中靠近法语的"chat"和西班牙语的"gato"，完整的句子嵌入也是如此。

**零样本迁移。** 在一种语言（通常是英语）的标注数据上微调模型。推理时，在模型支持的任何其他语言上运行它。不需要目标语言的标签。对于类型学（Typology）上相关的语言效果很强，对于距离较远的语言则较弱。

**少样本微调。** 添加 100-500 个目标语言的标注样本。在分类任务上，准确率跃升至英语基线的 95-98%。这是多语言 NLP 中最具成本效益的单一杠杆。

## 模型

| 模型 | 年份 | 覆盖范围 | 备注 |
|------|------|----------|------|
| mBERT | 2018 | 104 种语言 | 在维基百科上训练。第一个实用的多语言语言模型。低资源语言表现较弱。 |
| XLM-R | 2019 | 100 种语言 | 在 CommonCrawl 上训练（远大于维基百科）。设定跨语言基线。Base 270M，Large 550M。 |
| XLM-V | 2023 | 100 种语言 | XLM-R 配备 100 万 token 词汇表（对比 250k）。在低资源语言上表现更好。 |
| mT5 | 2020 | 101 种语言 | 用于多语言生成的 T5 架构。 |
| NLLB-200 | 2022 | 200 种语言 | Meta 的翻译模型；包含 55 种低资源语言。 |
| BLOOM | 2022 | 46 种语言 + 13 种编程语言 | 多语言训练的开源 176B 大语言模型。 |
| Aya-23 | 2024 | 23 种语言 | Cohere 的多语言大语言模型。在阿拉伯语、印地语、斯瓦希里语上表现强劲。 |

根据用例选择。分类任务使用 XLM-R-base 作为合理的默认选择。生成任务选择 mT5 或 NLLB，取决于翻译还是开放式生成。大语言模型风格的工作搭配 Aya-23 或使用显式多语言提示的 Claude。

## 源语言决策（2026 年研究）

大多数团队默认使用英语作为微调源语言。最新研究（2026 年）表明这通常是错误的。

语言相似性比原始语料库大小更能预测迁移质量。对于斯拉夫语系目标，德语或俄语通常优于英语。对于印度语系目标，印地语通常优于英语。**qWALS** 相似度指标（2026 年，基于世界语言结构图集特征）对此进行了量化。**LANGRANK**（Lin 等人，ACL 2019）是一种独立的、更早的方法，它从语言学相似性、语料库大小和谱系关系的组合中对候选源语言进行排序。

实用规则：如果你的目标语言在类型学上有一个近亲高资源语言，先尝试在该语言上微调，然后与英语微调进行比较。

## 动手实践

### 步骤 1：零样本跨语言分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

一个模型，三种语言，相同的 API。在 NLI 数据上训练的 XLM-R 通过蕴含技巧很好地迁移到分类任务。

### 步骤 2：多语言嵌入空间

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

翻译结果在嵌入空间中彼此接近。一个不同的英语句子则距离更远。这正是跨语言检索、聚类和相似度计算的基础。

### 步骤 3：少样本微调策略

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

对于 100-500 个目标语言样本，`num_train_epochs=5` 和 `learning_rate=2e-5` 是安全的默认值。更高的学习率会导致多语言对齐崩溃，你最终会得到一个只支持英语的模型。

## 真正有效的评估

- **按语言在留出集上的准确率。** 不要聚合。聚合指标会掩盖长尾问题。
- **与单语基线（Monolingual Baseline）对比。** 对于有足够数据的语言，从头训练的单语模型有时会击败多语言模型。需要测试。
- **实体级测试。** 目标语言中的命名实体。多语言模型对于远离拉丁字母的书写系统往往分词能力较弱。
- **跨语言一致性。** 两种语言中相同含义的内容应产生相同的预测。衡量其中的差距。

## 使用指南

2026 年技术栈：

| 任务 | 推荐方案 |
|------|----------|
| 分类，100 种语言 | 微调 XLM-R-base（约 270M） |
| 零样本文本分类 | `joeddav/xlm-roberta-large-xnli` |
| 多语言句子嵌入 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 翻译，200 种语言 | `facebook/nllb-200-distilled-600M`（参见第 11 课） |
| 生成式多语言 | Claude、GPT-4、Aya-23、mT5-XXL |
| 低资源语言 NLP | XLM-V 或在相关高资源语言上进行领域特定微调 |

如果性能很重要，始终要为目标语言的微调预留预算。零样本只是一个起点，而非最终答案。

### 分词税（Tokenization Tax）（低资源语言的问题所在）

多语言模型在所有语言之间共享一个分词器。该词汇表是在以英语、法语、西班牙语、中文、德语为主的语料库上训练的。对于主流语言集之外的任何语言，三种隐性代价会叠加：

- **生育率税（Fertility Tax）。** 低资源语言文本每个词被分词为的 token 数量远多于英语。一个印地语句子可能需要等效英语句子 3-5 倍的 token。这 3-5 倍的开销会消耗你的上下文窗口、训练效率和延迟。
- **变体恢复税。** 每个拼写错误、变音符号变体、Unicode 规范化不匹配或大小写变化，在嵌入空间中都会成为一个冷启动的无关序列。模型无法学习母语者认为理所当然的正字法对应关系。
- **容量溢出税（Capacity Overflow Tax）。** 代价 1 和 2 消耗了上下文位置、层深度和嵌入维度。留给实际推理的资源系统性地少于高资源语言从同一模型中获得的资源。

实际症状：你的模型在印地语上正常训练，损失曲线看起来正确，评估困惑度看起来合理，但生产输出微妙地出错。词形在句子中途崩溃。罕见的屈折变化始终无法恢复。**你无法通过扩大数据规模来修复一个破碎的分词器。**

缓解措施：选择对目标语言覆盖良好的分词器（XLM-V 的 100 万 token 词汇表是直接的解决方案）；在训练前验证目标文本的分词生育率；对真正的长尾书写系统使用字节级回退（Byte-level Fallback）（SentencePiece 的 `byte_fallback=True`，GPT-2 风格的字节级 BPE），这样就不会有任何词超出词汇表。

## 部署

保存为 `outputs/skill-multilingual-picker.md`：

```markdown
---
name: multilingual-picker
description: Pick source language, target model, and evaluation plan for a multilingual NLP task.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Given requirements (target languages, task type, available labeled data per language), output:

1. Source language for fine-tuning. Default English; check LANGRANK or qWALS if target language has a typologically close high-resource language.
2. Base model. XLM-R (classification), mT5 (generation), NLLB (translation), Aya-23 (generative LLM).
3. Few-shot budget. Start with 100-500 target-language examples if available. Zero-shot only if labeling is infeasible.
4. Evaluation plan. Per-language accuracy (not aggregate), cross-lingual consistency, entity-level F1 on non-Latin scripts.

Refuse to ship a multilingual model without per-language evaluation — aggregate metrics hide long-tail failures. Flag scripts with low tokenization coverage (Amharic, Tigrinya, many African languages) as needing a model with byte-fallback (SentencePiece with byte_fallback=True, or byte-level tokenizer like GPT-2).
```

## 练习

1. **简单。** 在英语、法语、印地语和阿拉伯语上各运行 10 句零样本分类流程。报告每种语言的准确率。你应该会看到法语表现强劲，印地语尚可，阿拉伯语参差不齐。
2. **中等。** 使用 `paraphrase-multilingual-MiniLM-L12-v2` 在一个小型多语言语料库上构建跨语言检索器。用英语查询，检索任何语言的文档。衡量 recall@5。
3. **困难。** 比较英语源和印地语源微调在印地语分类任务上的表现。在两种方案下各使用 500 个目标语言样本进行少样本微调。报告哪种源语言产生更好的印地语准确率以及差距有多大。这是 LANGRANK 论点的小规模验证。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 多语言模型（Multilingual Model） | 一个模型，多种语言 | 跨语言共享词汇表和参数。 |
| 跨语言迁移（Cross-lingual Transfer） | 在一种语言上训练，在另一种语言上运行 | 在源语言上微调，在没有目标语言标签的情况下在目标语言上评估。 |
| 零样本（Zero-shot） | 无需目标语言标签 | 不在目标语言上微调即进行迁移。 |
| 少样本（Few-shot） | 少量目标语言标签 | 使用 100-500 个目标语言样本进行微调。 |
| mBERT | 第一个多语言语言模型 | 在维基百科上预训练的 104 语言 BERT。 |
| XLM-R | 标准跨语言基线 | 在 CommonCrawl 上预训练的 100 语言 RoBERTa。 |
| NLLB | Meta 的 200 语言机器翻译 | No Language Left Behind（不让任何语言掉队）。包含 55 种低资源语言。 |

## 延伸阅读

- [Conneau 等人 (2019)。大规模无监督跨语言表示学习](https://arxiv.org/abs/1911.02116) — XLM-R 论文。
- [Pires, Schlinger, Garrette (2019)。多语言 BERT 到底有多"多语言"？](https://arxiv.org/abs/1906.01502) — 开启跨语言迁移研究方向的分析论文。
- [Costa-jussa 等人 (2022)。不让任何语言掉队](https://arxiv.org/abs/2207.04672) — NLLB-200 论文。
- [Üstün 等人 (2024)。Aya 模型：一个经过指令微调的开源多语言语言模型](https://arxiv.org/abs/2402.07827) — Aya，Cohere 的多语言大语言模型。
- [语言相似性预测跨语言迁移学习性能 (2026)](https://www.mdpi.com/2504-4990/8/3/65) — qWALS / LANGRANK 源语言论文。
