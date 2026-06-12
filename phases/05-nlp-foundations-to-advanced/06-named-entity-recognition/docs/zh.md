# 命名实体识别

> 把名字提取出来。听起来简单，直到你遇到模糊边界、嵌套实体和领域术语。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 5 · 02（词袋 + TF-IDF），阶段 5 · 03（词嵌入）
**时间：** 约 75 分钟

## 问题

"Apple sued Google over its iPhone search deal in the US." 五个实体：Apple（ORG），Google（ORG），iPhone（PRODUCT），search deal（也许是），US（GPE）。一个好的 NER 系统能提取所有实体并给出正确类型。一个差的系统会漏掉 iPhone，把苹果公司和苹果水果搞混，把 "US" 标注为人名。

NER 是每一条结构化提取管线背后的核心组件。简历解析、合规日志扫描、医疗记录匿名化、搜索查询理解、聊天机器人回复的依据定位、法律合同提取。你几乎看不到它，但你一直依赖它。

本课程沿着经典路径（基于规则、HMM、CRF）走向现代路径（BiLSTM-CRF，然后是 Transformer）。每一步都解决了前一步的特定局限。这个模式本身就是课程的核心。

## 概念

**BIO 标注**（或 BILOU）将实体提取转化为序列标注问题。为每个 token 标注 `B-TYPE`（实体开头）、`I-TYPE`（实体内部）或 `O`（不属于任何实体）。

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

多 token 实体形成链：`New B-GPE`、`York I-GPE`、`City I-GPE`。理解 BIO 的模型可以提取任意跨度。

架构演进：

- **基于规则。** 正则表达式 + 词典查找。对已知实体精确率高，对新实体零覆盖。
- **HMM。** 隐马尔可夫模型。给定标签的 token 发射概率，标签到标签的转移概率。维特比解码。在标注数据上训练。
- **CRF。** 条件随机场。类似 HMM 但是判别式的，因此你可以混合任意特征（词形、大小写、相邻词）。截至 2026 年，仍然是低资源部署场景下的经典生产主力。
- **BiLSTM-CRF。** 神经网络特征取代手工特征。LSTM 双向读取句子，CRF 层在顶层强制标签序列的一致性。
- **基于 Transformer。** 微调 BERT 并加上 token 分类头。精度最高。计算量最大。

```figure
ner-bio-tagging
```

## 构建

### 步骤 1：BIO 标注辅助函数

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### 步骤 2：手工特征

对于经典（非神经网络）NER，特征是关键。常用的特征有：

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` 返回 `xXxxxx`。`word_shape("USA-2024")` 返回 `XXX-dddd`。大小写模式对专有名词具有很高的信号价值。

### 步骤 3：简单的基于规则 + 词典的基线

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

生产环境的词典通常包含从 Wikipedia 和 DBpedia 抓取的数百万条目。覆盖率很好。但消歧（`Apple` 是公司还是水果）非常糟糕。这就是统计模型胜出的原因。

### 步骤 4：CRF 步骤（概要，非完整实现）

在没有概率论基础的情况下，用 50 行代码从头实现完整 CRF 并不能带来多少启发。改用 `sklearn-crfsuite`：

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` 和 `c2` 是 L1 和 L2 正则化。`all_possible_transitions=True` 让模型学习到非法序列（例如 `I-ORG` 出现在 `O` 之后）的概率很低，这就是 CRF 无需你编写约束就能强制 BIO 一致性的方法。

### 步骤 5：BiLSTM-CRF 带来了什么

特征变成了可学习的。输入：token 嵌入（GloVe 或 fastText）。LSTM 从左到右和从右到左读取。拼接后的隐藏状态通过 CRF 输出层。CRF 仍然强制标签序列的一致性；LSTM 用学习到的特征取代了手工特征。

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

CRF 层使用 `torchcrf.CRF`（pip install pytorch-crf）。相比手工 CRF 的提升是可衡量的，但除非你有数万个标注句子，否则提升幅度比你预期的要小。

## 使用

spaCy 开箱即用地提供生产级 NER。

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

注意 `iPhone` 被标注为 `ORG` 而不是 `PRODUCT`——spaCy 的小模型对产品实体的覆盖较弱。大模型（`en_core_web_lg`）表现更好。Transformer 模型（`en_core_web_trf`）表现更好。

Hugging Face 的基于 BERT 的 NER：

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` 将连续的 B-X、I-X token 合并为一个跨度。不使用它的话，你会得到 token 级别的标签，需要自己合并。

### 基于 LLM 的 NER（2026 年的选项）

零样本和少样本 LLM NER 现在在许多领域已经可以与微调模型相媲美，而且在标注数据稀缺时表现显著更好。

- **零样本提示。** 给 LLM 一个实体类型列表和示例 schema。要求 JSON 输出。开箱即用；在新领域上精度中等。
- **ZeroTuneBio 风格的提示。** 将任务分解为候选提取 -> 语义解释 -> 判断 -> 重新检查。多阶段提示（非一次性）在生物医学 NER 上显著提升了精度。同样的模式适用于法律、金融和科学领域。
- **结合 RAG 的动态提示。** 每次推理调用时，从小型标注种子集中检索最相似的标注示例；动态构建少样本提示。在 2026 年的基准测试中，这将 GPT-4 生物医学 NER 的 F1 提升了 11-12%，超过了静态提示。
- **按实体类型分解。** 对于长文档，单次调用提取所有实体类型会随着长度增加而降低召回率。对每种实体类型运行一次提取。推理成本更高，但精度显著提升。这是临床笔记和法律合同的标准模式。

2026 年的生产建议：在收集训练数据之前，先用 LLM 零样本作为基线。通常 F1 已经足够好，你不需要微调。

### 经典 NER 仍然胜出的场景

即使有 LLM 可用，经典 NER 在以下场景中仍然胜出：

- 延迟预算低于 50ms。
- 你有数千个标注样本，需要 98% 以上的 F1。
- 领域有稳定的本体，预训练的 CRF 或 BiLSTM 迁移效果好。
- 监管约束要求使用本地部署的非生成式模型。

### 失效的场景

- **领域偏移。** 在 CoNLL 上训练的 NER 用于法律合同的效果比词典还差。在你的领域上微调。
- **嵌套实体。** "Bank of America Tower" 同时是一个组织机构和一个设施。标准 BIO 无法表示重叠跨度。你需要嵌套 NER（多轮或基于跨度的模型）。
- **长实体。** "United States Federal Deposit Insurance Corporation。" Token 级别模型有时会将其拆分。使用 `aggregation_strategy` 或后处理。
- **稀疏类型。** 医学 NER 标签如 DRUG_BRAND、ADVERSE_EVENT、DOSE。通用模型完全不了解。Scispacy 和 BioBERT 是那里的起点。

## 交付

保存为 `outputs/skill-ner-picker.md`：

```markdown
---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.
```

## 练习

1. **简单。** 实现 `bio_to_spans`（`spans_to_bio` 的逆操作），并在 10 个句子上验证往返一致性。
2. **中等。** 在 CoNLL-2003 英文 NER 数据集上训练上述 sklearn-crfsuite CRF。使用 `seqeval` 报告每个实体的 F1。典型结果：约 84 F1。
3. **困难。** 在特定领域的 NER 数据集（医疗、法律或金融）上微调 `distilbert-base-cased`。与 spaCy 小模型进行对比。记录数据泄漏检查，并写下让你感到意外的发现。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| NER | 提取名字 | 为 token 跨度标注类型（PERSON、ORG、GPE、DATE 等）。 |
| BIO | 标注方案 | `B-X` 开头，`I-X` 延续，`O` 外部。 |
| BILOU | 改进的 BIO | 增加 `L-X`（最后一个）、`U-X`（单个），使边界更清晰。 |
| CRF | 结构化分类器 | 建模标签之间的转移，而不仅仅是发射。强制有效序列。 |
| 嵌套 NER | 重叠实体 | 一个跨度是与其子跨度不同的实体。BIO 无法表达这一点。 |
| 实体级别 F1（Entity-level F1） | 正确的 NER 指标 | 预测跨度必须与真实跨度完全匹配。Token 级别 F1 会高估精度。 |

## 延伸阅读

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — BiLSTM-CRF 论文。经典之作。
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — 引入了后来成为标准的 token 分类模式。
- [spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities) — `Doc.ents` 和 `Span` 上每个属性的实用参考。
- [seqeval](https://github.com/chakki-works/seqeval) — 正确的指标库。始终使用它。
