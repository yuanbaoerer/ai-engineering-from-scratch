# 文本处理 — 分词、词干提取、词形还原

> 语言是连续的。模型是离散的。预处理是桥梁。

**类型：** 构建
**语言：** Python
**前置知识：** 第二阶段 · 14（朴素贝叶斯）
**时间：** 约 45 分钟

## 问题所在

模型无法读取 "The cats were running."。它只能读取整数。

每个 NLP 系统都从三个相同的问题开始。单词从哪里开始？单词的词根是什么？如何将 "run"、"running"、"ran" 在有帮助时视为同一个词，在无帮助时视为不同的词？

分词错误，模型学到的就是垃圾。如果你的分词器把 `don't` 作为一个 token，但把 `do n't` 作为两个 token，训练分布就会分裂。如果你的词干提取器把 `organization` 和 `organ` 压缩到同一个词干，主题建模就会失效。如果你的词形还原器需要词性上下文，但你没有传入，动词就会被当作名词处理。

本节课从头构建三个预处理步骤，然后展示 NLTK 和 spaCy 如何完成相同的工作，让你看到其中的权衡。

## 核心概念

三个操作。每个都有其职责和失败模式。

**分词（Tokenization）** 将字符串分割成 token。"Token" 是故意模糊的，因为正确的粒度取决于任务。经典 NLP 用词级。Transformer 用子词级。无声息分隔的语言用字符级。

**词干提取（Stemming）** 用规则切除后缀。快、激进、笨。`running -> run`。`organization -> organ`。第二个就是失败模式。

**词形还原（Lemmatization）** 利用语法知识将单词还原为词典形式。更慢、准确、需要查表或形态分析器。`ran -> run`（需要知道 "ran" 是 "run" 的过去式）。`better -> good`（需要知道比较级形式）。

经验法则。当速度很重要且可以容忍噪声时用词干提取（搜索索引、粗略分类）。当意义很重要时用词形还原（问答、语义搜索、用户会阅读的任何内容）。

```figure
edit-distance
```

## 构建

### 步骤 1：正则表达式分词器

最简单的实用分词器在非字母数字字符处分割，同时将标点符号作为独立 token。并不完美，也不是最终版本，但一行代码就能运行。

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

三个按优先级排列的模式。带可选内部撇号的单词（`don't`、`it's`）。纯数字。任何单个非空白、非字母数字字符作为独立 token（标点符号）。

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

需要注意的失败模式。`3pm` 会分割成 `['3', 'pm']`，因为我们在字母序列和数字序列之间切换。对大多数任务来说够用了。URL、电子邮件、话题标签都会出问题。生产环境需要在通用模式之前添加特定模式。

### 步骤 2：Porter 词干提取器（第 1a 步）

完整的 Porter 算法有五个阶段的规则。仅第 1a 步就涵盖了最常见的英语后缀，并教会你模式。

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

从上到下阅读规则。`ies -> i` 规则是为什么 `ponies -> poni` 而不是 `pony` 的原因。真正的 Porter 有第 1b 步可以修复这个问题。规则会竞争。先出现的规则获胜。顺序比任何单一规则都重要。

### 步骤 3：基于查表的词形还原器

真正的词形还原需要形态学。一个可教学的简化版本使用小型词元表和回退机制。

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

最后一种情况是关键的教学时刻。`watched` 不在我们的表中，我们的回退只处理 `ing`。真正的词形还原覆盖 `ed`、不规则动词、形容词比较级、音变复数（`children -> child`）。这就是为什么生产系统使用 WordNet、spaCy 的形态分析器或完整的形态分析器。

### 步骤 4：将它们串联起来

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

缺失的部分是 POS 标注器。第五阶段 · 07（词性标注）会构建一个。现在，暂时将所有内容默认为 `NOUN` 并承认这个限制。

## 使用

NLTK 和 spaCy 都提供了生产级版本。各只需几行代码。

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` 处理了缩写、Unicode 以及你的正则表达式会错过的边缘情况。`PorterStemmer` 运行全部五个阶段。`WordNetLemmatizer` 需要将 POS 标签从 NLTK 的 Penn Treebank 标注方案转换到 WordNet 的缩写集。上面的转换连接代码是大多数教程跳过的部分。

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy 将整个管道隐藏在 `nlp(text)` 后面。分词、POS 标注和词形还原全部运行。比 NLTK 在规模上更快。开箱即用更准确。代价是你不能轻易交换单个组件。

### 何时选择哪个

| 场景 | 选择 |
|------|------|
| 教学、研究、需要交换组件 | NLTK |
| 生产、多语言、速度重要 | spaCy |
| Transformer 管道（你反正会用模型的 tokenizer 进行分词） | 使用 `tokenizers` / `transformers`，跳过经典预处理 |

### 没人警告你的两个失败模式

大多数教程只教算法然后就停止了。两件事会困扰真正的预处理管道，而且几乎从不涉及。

**可复现性漂移。** NLTK 和 spaCy 在不同版本之间会改变分词和词形还原行为。spaCy 2.x 中产生 `['do', "n't"]` 的结果可能在 3.x 中产生 `["don't"]`。你的模型是用一种分布训练的。现在推理运行在不同的分布上。准确率悄悄下降，没人知道为什么。在 `requirements.txt` 中固定库版本。编写一个预处理回归测试，冻结 20 个样本句子的预期分词结果。每次升级时运行它。

**训练/推理不匹配。** 用激进的预处理（lowercase、停用词去除、词干提取）训练，部署在原始用户输入上，看着性能崩溃。这是生产级 NLP 最常见的失败案例。如果你在训练时进行了预处理，推理时必须运行完全相同的函数。将预处理作为函数放在模型包内，而不是作为笔记本单元格让服务团队重写。

## 发布

一个可重用的提示词，帮助工程师无需阅读三本教科书就能选择预处理策略。

保存为 `outputs/prompt-preprocessing-advisor.md`：

```markdown
---
name: preprocessing-advisor
description: Recommends a tokenization, stemming, and lemmatization setup for an NLP task.
phase: 5
lesson: 01
---

You advise on classical NLP preprocessing. Given a task description, you output:

1. Tokenization choice (regex, NLTK word_tokenize, spaCy, or transformer tokenizer). Explain why.
2. Whether to stem, lemmatize, both, or neither. Explain why.
3. Specific library calls. Name the functions. Quote the POS-tag translation if NLTK is involved.
4. One failure mode the user should test for.

Refuse to recommend stemming for user-visible text. Refuse to recommend lemmatization without POS tags. Flag non-English input as needing a different pipeline.
```

## 练习

1. **简单。** 扩展 `tokenize` 将 URL 保留为单个 token。测试：`tokenize("Visit https://example.com today.")` 应该产生一个 URL token。
2. **中等。** 实现 Porter 第 1b 步。如果单词包含元音且以 `ed` 或 `ing` 结尾，则删除它。处理双辅音规则（`hopping -> hop`，而不是 `hopp`）。
3. **困难。** 构建一个使用 WordNet 作为查表但在 WordNet 没有条目时回退到 Porter 词干提取器的词形还原器。在带标注的语料库上测量准确率，对比纯 WordNet 和纯 Porter。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------|----------|
| Token | 一个单词 | 模型消耗的任何单位。可以是单词、子词、字符或字节。 |
| Stem | 词根 | 基于规则去除后缀的结果。不一定是真实单词。 |
| Lemma | 词典形式 | 你查词典时用的形式。需要语法上下文才能正确计算。 |
| POS tag | 词性 | NOUN、VERB、ADJ 等类别。准确词形还原需要它。 |
| Morphology | 词形规则 | 单词如何根据时态、数、格改变形式。词形还原依赖它。 |

## 延伸阅读

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — 原始论文，五页，仍然是最清晰的解释。
- [spaCy 101 — 语言学特征](https://spacy.io/usage/linguistic-features) — 真正的管道如何连接。
- [NLTK 第三章](https://www.nltk.org/book/ch03.html) — 你还没想到的分词边缘情况。
