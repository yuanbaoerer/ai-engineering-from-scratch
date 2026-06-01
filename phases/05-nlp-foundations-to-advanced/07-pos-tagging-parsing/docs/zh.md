# 词性标注与句法分析

> 语法曾经一度不再流行。后来每个 LLM 流水线都需要验证结构化抽取，于是语法又回来了。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 5 · 01（文本处理），Phase 2 · 14（朴素贝叶斯）
**时间：** 约 45 分钟

## 问题所在

第 01 课承诺过，词形还原需要词性标签。如果不把 `running` 标记为动词，词形还原器就无法将其还原为 `run`；如果不把 `better` 标记为形容词，就无法还原为 `good`。

这个承诺背后隐藏着一整个子领域。词性标注（Part-of-Speech Tagging / POS Tagging）为每个词分配语法类别。句法分析（Syntactic Parsing）则恢复句子的树结构：哪个词修饰哪个词，哪个动词支配哪些论元。经典 NLP 花了二十年时间精炼这两项任务。然后深度学习将它们压缩成了预训练 Transformer 之上的一个 token 分类任务，研究界也随之转移了方向。

但应用界没有。每条结构化抽取流水线的底层仍然使用词性标注和依存树。LLM 生成的 JSON 会根据语法约束进行校验。问答系统利用依存分析（Dependency Parsing）来分解查询。机器翻译质量评估器检查分析树的对齐情况。

值得了解。本课介绍标签集、基线方法，以及你何时该停止从头实现，转而调用 spaCy。

## 核心概念

**词性标注**为每个 token 标注一个语法类别。**Penn Treebank（PTB）**标签集是英语的默认选择。包含 36 个标签，区分之细致令普通读者觉得繁琐：`NN` 单数名词，`NNS` 复数名词，`NNP` 专有名词单数，`VBD` 动词过去式，`VBZ` 动词第三人称单数现在时，等等。**Universal Dependencies（UD）**标签集更粗粒度（17 个标签），且语言无关，已成为跨语言工作的默认选择。

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**句法分析**生成一棵树。两种主要风格：

- **成分分析。** 名词短语、动词短语、介词短语层层嵌套。输出是一棵非终结符类别（NP、VP、PP）的树，词语是叶节点。
- **依存分析。** 每个词有一个它所依赖的中心词，并标注语法关系。输出是一棵树，每条边是一个（中心词、依存词、关系）三元组。

依存分析在 2010 年代胜出，因为它能很好地跨语言泛化，尤其是对语序自由的语言。

```
running 是 ROOT
cats 是 running 的 nsubj
were 是 running 的 aux
at 是 running 的 prep
3pm 是 at 的 pobj
```

## 动手构建

### 第 1 步：最高频标签基线

最简单但有效的词性标注器。对每个词，预测它在训练数据中出现次数最多的标签。

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

在 Brown 语料库上，这个基线的准确率约为 85%。不算好，但这是任何正式模型都不应低于的底线。

### 第 2 步：二元 HMM 标注器

对序列的联合概率建模：

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

需要两张表：转移概率（Transition Probability，给定前一个标签的当前标签概率）和发射概率（Emission Probability，给定标签的词的概率）。两者都通过计数加拉普拉斯平滑来估计。解码使用维特比算法（在标签格上做动态规划）。

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

二元 HMM 在 Brown 语料库上的准确率约为 93%。从 85% 到 93% 的提升主要来自转移概率——模型学到了 `DET NOUN` 很常见而 `NOUN DET` 很罕见。

### 第 3 步：现代标注器为何更优

转移概率和发射概率是局部的。它们无法捕捉 `saw` 在 "I bought a saw" 中是名词，而在 "I saw the movie" 中是动词这一事实。使用任意特征（后缀、词形、前后词、词本身）的 CRF 可达到约 97% 的准确率。BiLSTM-CRF 或 Transformer 可达到约 98% 以上。

这项任务的上限由标注者之间的分歧决定。人类标注者在 Penn Treebank 上的一致率约为 97%。准确率超过 98% 的模型可能是在测试集上过拟合了。

### 第 4 步：依存分析概述

从头实现完整的依存分析超出本课范围；权威教材的讲解见 Jurafsky 和 Martin 的著作。需要了解两个经典方法族：

- **基于转移的（Transition-based）**分析器（arc-eager、arc-standard）行为类似移进-归约分析器：读取 token，将其压入栈，然后执行归约操作来创建弧。贪心解码很快。经典实现是 MaltParser。现代神经版本：Chen 和 Manning 的基于转移的分析器。
- **基于图的（Graph-based）**分析器（Eisner 算法、Dozat-Manning 双仿射）为每条可能的中心词-依存词边打分，然后选取最大生成树。更慢但更准确。

对于大多数实际应用，直接调用 spaCy：

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

从下往上读 `dep` 列，句子的语法结构便一目了然。

## 实际应用

每个生产级 NLP 库都将词性标注和依存分析器作为标准流水线的一部分提供。

- **spaCy**（`en_core_web_sm` / `md` / `lg` / `trf`）。快速、准确，与分词 + 命名实体识别 + 词形还原集成。`token.tag_`（Penn 标签），`token.pos_`（UD 标签），`token.dep_`（依存关系）。
- **Stanford NLP（stanza）**。Stanford CoreNLP 的继任者。在 60 多种语言上达到最先进的效果。
- **trankit**。基于 Transformer，UD 准确率很高。
- **NLTK**。`pos_tag`。可用，速度较慢，较老。适合教学使用。

### 2026 年仍然重要的应用场景

- **词形还原。** 第 01 课需要词性标注才能正确进行词形还原。始终需要。
- **从 LLM 输出中进行结构化抽取。** 验证生成的句子是否符合语法约束（例如主谓一致、必要的修饰语）。
- **基于方面的情感分析。** 依存分析告诉你哪个形容词修饰哪个名词。
- **查询理解。** "movies directed by Wes Anderson starring Bill Murray" 通过分析可以分解为结构化约束。
- **跨语言迁移。** UD 标签和依存关系是语言无关的，能够对新语言进行零样本结构化分析。
- **低算力流水线。** 如果无法部署 Transformer，词性标注 + 依存分析 + 地名辞典也能让你走得很远。

## 交付成果

保存为 `outputs/skill-grammar-pipeline.md`：

```markdown
---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

给定一个下游任务（信息抽取、改写验证、查询分解、词形还原），你需要输出：

1. 标签集选择。仅英语的遗留流水线使用 Penn Treebank，多语言或跨语言场景使用 Universal Dependencies。
2. 库选择。大多数生产环境用 spaCy，学术级多语言用 stanza，最高 UD 准确率用 trankit。注明具体的模型 ID。
3. 集成模式。展示调用库并使用所需属性（`.pos_`、`.dep_`、`.head`）的 3-5 行代码。
4. 需要测试的失败模式。名词-动词歧义（`saw`、`book`、`can`）和介词短语附着歧义是经典陷阱。抽取 20 个输出样本进行人工检查。

拒绝推荐自行构建分析器。从头构建分析器是研究项目，不是应用任务。对于任何使用词性标签但不处理大小写变体的流水线，标记为脆弱。
```

## 练习

1. **简单。** 在一个小型标注语料库（例如 NLTK 的 Brown 子集）上使用最高频标签基线，在留出句子上测量准确率。验证约 85% 的结果。
2. **中等。** 训练上面的二元 HMM 并报告每个标签的精确率/召回率。HMM 最容易混淆哪些标签？
3. **困难。** 使用 spaCy 的依存分析从 1000 个句子样本中提取主谓宾三元组。在 50 个手动标注的三元组上进行评估。记录抽取失败的场景（通常是被动语态、并列结构和省略主语）。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 词性标签（POS Tag） | 词的类型 | 语法类别。PTB 有 36 个；UD 有 17 个。 |
| Penn Treebank | 标准标签集 | 英语专用。细粒度的动词时态和名词数。 |
| Universal Dependencies | 多语言标签集 | 比 PTB 更粗粒度；语言中性；跨语言工作的默认选择。 |
| 依存分析（Dependency Parsing） | 句子树 | 每个词有一个中心词，每条边有一个语法关系。 |
| 维特比算法（Viterbi Algorithm） | 动态规划 | 在给定发射和转移概率的情况下，找到概率最高的标签序列。 |

## 延伸阅读

- [Jurafsky 和 Martin —— 《语音与语言处理》，第 8 章和第 18 章](https://web.stanford.edu/~jurafsky/slp3/) —— 词性标注和句法分析的权威教材。
- [Universal Dependencies 项目](https://universaldependencies.org/) —— 每个多语言分析器使用的跨语言标签集和树库集合。
- [spaCy 语言学特性指南](https://spacy.io/usage/linguistic-features) —— Token 上暴露的每个属性的实用参考。
- [Chen 和 Manning（2014）。《使用神经网络的快速准确依存分析器》](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) —— 将神经分析器引入主流的论文。
