# 自然语言推理 -- 文本蕴含

> "t 蕴含 h" 意味着一个阅读 t 的人会得出 h 为真的结论。NLI 是预测蕴含/矛盾/中立的任务。表面上看很无聊，但在生产环境中却不可或缺。

**类型:** 学习
**语言:** Python
**前置条件:** 第5阶段 · 05（情感分析），第5阶段 · 13（问答系统）
**时间:** 约60分钟

## 问题

你构建了一个摘要生成器。它生成了摘要。你怎么知道摘要中不包含幻觉？

你构建了一个聊天机器人。它回答了"是"。你怎么知道这个回答有检索到的段落作为支撑？

你需要对10,000篇新闻文章按主题分类。你没有训练标签。你能复用一个模型吗？

这三个问题都可以归结为自然语言推理。NLI 的问题是：给定一个前提（Premise）`t` 和一个假设（Hypothesis）`h`，`h` 是被 `t` 蕴含（Entailment）、矛盾（Contradiction），还是中立（Neutral）（无关）？

- **幻觉检测:** `t` = 源文档，`h` = 摘要声明。非蕴含 = 幻觉。
- **基于检索的问答:** `t` = 检索到的段落，`h` = 生成的回答。非蕴含 = 捏造。
- **零样本分类（Zero-shot Classification）:** `t` = 文档，`h` = 文字化标签（"这是关于体育的"）。蕴含 = 预测标签。

一个任务，三种生产用途。这就是为什么每个 RAG 评估框架都在底层内置了一个 NLI 模型。

## 概念

![NLI：三分类，前提与假设](../assets/nli.svg)

**三个标签。**

- **蕴含。** `t` → `h`。"猫在垫子上" 蕴含 "有一只猫。"
- **矛盾。** `t` → ¬`h`。"猫在垫子上" 矛盾于 "没有猫。"
- **中立。** 无法从任一方向推断。"猫在垫子上" 与 "猫饿了" 是中立关系。

**不是逻辑蕴含。** NLI 是*自然*语言推理——是一个普通读者会做出的推断，而不是严格的逻辑。"John 遛了他的狗" 在 NLI 中蕴含 "John 有一只狗"，但严格的一阶逻辑（First-order Logic）只有在你将所有权公理化后才会承认这一点。

**数据集。**

- **SNLI**（2015）。57万个人工标注的对，以图像描述作为前提。领域较窄。
- **MultiNLI**（2017）。43.3万个对，涵盖10种体裁。2026年的标准训练语料库。
- **ANLI**（2019）。对抗性（Adversarial）NLI。人类专门编写了旨在击败现有模型的样本。难度更高。
- **DocNLI, ConTRoL**（2020-21）。文档长度的前提。测试多跳和长距离推理。

**架构。** 一个 Transformer 编码器（BERT、RoBERTa、DeBERTa）读取 `[CLS] premise [SEP] hypothesis [SEP]`。`[CLS]` 表示馈入一个三路 softmax。在 MNLI 上训练，在留出的基准上评估，在分布内对上可获得 90%+ 的准确率。

**通过 NLI 实现零样本分类。** 给定一个文档和候选标签，将每个标签转化为假设（"这段文本是关于体育的"）。计算每个标签的蕴含概率。选择最大的那个。这就是 Hugging Face 的 `zero-shot-classification` 管道背后的机制。

## 动手实践

### 步骤 1：运行预训练的 NLI 模型

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # return all labels; replaces deprecated return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

对于生产环境中的 NLI，`facebook/bart-large-mnli` 和 `microsoft/deberta-v3-large-mnli` 是开源的默认选择。DeBERTa-v3 在排行榜上名列前茅。

### 步骤 2：零样本分类

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

默认模板是 "This example is about {label}."。可以通过 `hypothesis_template` 自定义。不需要训练数据。不需要微调。开箱即用。

### 步骤 3：RAG 忠实度（Faithfulness）检查

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

这是 RAGAS 忠实度的核心。将生成的回答分解为原子声明。针对检索到的上下文检查每个声明。报告被蕴含的比例。

### 步骤 4：手写 NLI 分类器（概念性）

参见 `code/main.py`，一个仅使用标准库的玩具示例：前提和假设通过词汇重叠（Lexical Overlap）+否定检测进行比较。无法与 Transformer 模型竞争——但它展示了任务的形式：两个文本输入，三分类标签输出，损失 = 在 `{entail, contradict, neutral}` 上的交叉熵（Cross-entropy）。

## 常见陷阱

- **仅基于假设的捷径。** 模型可以仅从假设预测标签，在 SNLI 上达到约60%的准确率，因为 "not"、"nobody"、"never" 与矛盾相关。这是检测标签泄露的强大基线。
- **词汇重叠启发式。** 子序列启发式（"每个子序列都被蕴含"）在 SNLI 上通过，但在 HANS/ANLI 上失败。请使用对抗性基准。
- **文档长度退化。** 单句 NLI 模型在文档长度的前提上 F1 下降 20+。对于长上下文，请使用 DocNLI 训练的模型。
- **零样本模板敏感性。** "This example is about {label}" vs "{label}" vs "The topic is {label}" 可以导致准确率波动 10+ 个百分点。请调整模板。
- **领域不匹配。** MNLI 在通用英语上训练。法律、医学和科学文本需要领域特定的 NLI 模型（例如 SciNLI、MedNLI）。

## 使用场景

2026 年的技术栈：

| 使用场景 | 模型 |
|---------|------|
| 通用 NLI | `microsoft/deberta-v3-large-mnli` |
| 快速/边缘部署 | `cross-encoder/nli-deberta-v3-base` |
| 零样本分类（轻量级） | `facebook/bart-large-mnli` |
| 文档级 NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| 多语言 | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| RAG 幻觉检测 | RAGAS / DeepEval 内置的 NLI 层 |

2026 年的元模式：NLI 是文本理解的万能胶。每当你需要 "A 是否支持 B？" 或 "A 是否与 B 矛盾？"——在调用另一个 LLM 之前，先考虑使用 NLI。

## 交付

保存为 `outputs/skill-nli-picker.md`：

```markdown
---
name: nli-picker
description: Pick an NLI model, label template, and evaluation setup for a classification / faithfulness / zero-shot task.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

Given a use case (faithfulness check, zero-shot classification, document-level inference), output:

1. Model. Named NLI checkpoint. Reason tied to domain, length, language.
2. Template (if zero-shot). Verbalization pattern. Example.
3. Threshold. Entailment cutoff for the decision rule. Reason based on calibration.
4. Evaluation. Accuracy on held-out labeled set, hypothesis-only baseline, adversarial subset.

Refuse to ship zero-shot classification without a 100-example labeled sanity check. Refuse to use a sentence-level NLI model on document-length premises. Flag any claim that NLI solves hallucination — it reduces it; it does not eliminate it.
```

## 练习

1. **简单。** 在 20 个手工制作的（前提、假设、标签）三元组上运行 `facebook/bart-large-mnli`，覆盖所有三个类别。测量准确率。添加对抗性的"子序列启发式"陷阱（"我没有吃蛋糕" vs "我吃了蛋糕"），看看它是否会失败。
2. **中等。** 在 100 条 AG News 标题上比较零样本模板 `"This text is about {label}"`、`"The topic is {label}"` 和 `"{label}"`。报告准确率波动。
3. **困难。** 构建一个 RAG 忠实度检查器：原子声明分解 + 对每个声明进行 NLI。在 50 个带有黄金上下文的 RAG 生成回答上进行评估。与人工标签相比，测量误报率和漏报率。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| NLI | 自然语言推理 | 前提-假设关系的三分类。 |
| RTE | 识别文本蕴含 | NLI 的旧名称；同一个任务。 |
| 蕴含（Entailment） | "t 蕴含 h" | 普通读者在给定 t 的情况下会得出 h 为真的结论。 |
| 矛盾（Contradiction） | "t 排除 h" | 普通读者在给定 t 的情况下会得出 h 为假的结论。 |
| 中立（Neutral） | "未决定" | 无法从 t 到 h 做出任何方向的推断。 |
| 零样本分类（Zero-shot Classification） | NLI 作为分类器 | 将标签文字化为假设，选择最大蕴含概率。 |
| 忠实度（Faithfulness） | 回答是否有支撑？ | 对（检索到的上下文，生成的回答）进行 NLI。 |

## 延伸阅读

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — SNLI。
- [Williams, Nangia, Bowman (2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) — MultiNLI。
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) — ANLI 基准。
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) — NLI 作为分类器。
- [He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) — 2026 年 NLI 的主力模型。
