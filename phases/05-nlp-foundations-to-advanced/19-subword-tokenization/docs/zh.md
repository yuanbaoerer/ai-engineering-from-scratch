# 子词分词 — BPE、WordPiece、Unigram、SentencePiece

> 词级分词器遇到未见过的词就束手无策。字符级分词器让序列长度爆炸。子词分词器折中取优。每一个现代大语言模型都基于其中之一。

**类型：** 学习
**语言：** Python
**前置知识：** 第 5 阶段 · 01（文本处理），第 5 阶段 · 04（GloVe / FastText / 子词）
**时间：** 约 60 分钟

## 问题所在

你的词表有 50,000 个词。用户输入了 "untokenizable"。你的分词器返回 `[UNK]`。模型现在对这个词毫无信号。更糟的是：你的语料库中第 90 百分位的文档有 40 个稀有词，这意味着每个文档丢失 40 比特的信息。

子词分词解决了这个问题。常用词保持为单个词元。稀有词分解为有意义的片段：`untokenizable` → `un`、`token`、`izable`。训练数据覆盖一切，因为任何字符串最终都是字节序列。

2026 年每一个前沿大语言模型都基于三种算法之一（BPE、Unigram、WordPiece），封装在三个库之一（tiktoken、SentencePiece、HF Tokenizers）中。你无法在不选择其中之一的情况下发布语言模型。

## 核心概念

![BPE vs Unigram vs WordPiece，逐字符对比](../assets/subword-tokenization.svg)

**BPE（字节对编码）。** 从字符级词表开始。统计每对相邻字符。将频率最高的字符对合并为一个新词元。重复直到达到目标词表大小。主导算法：GPT-2/3/4、Llama、Gemma、Qwen2、Mistral。

**字节级 BPE。** 算法相同，但在原始字节（256 个基础词元）而非 Unicode 字符上操作。保证零 `[UNK]` 词元——任何字节序列都能编码。GPT-2 使用 50,257 个词元（256 字节 + 50,000 次合并 + 1 个特殊词元）。

**Unigram。** 从一个庞大的词表开始。为每个词元分配一个一元概率。迭代地剪除移除后对语料库对数似然影响最小的词元。推理时具有概率性：可以采样分词结果（对通过子词正则化进行数据增强很有用）。T5、mBART、ALBERT、XLNet、Gemma 使用此方法。

**WordPiece。** 合并使训练语料库似然最大化（而非原始频率最高）的词对。BERT、DistilBERT、ELECTRA 使用此方法。

**SentencePiece vs tiktoken。** SentencePiece 是一个直接在原始 Unicode 文本上*训练*词表（BPE 或 Unigram）的库，将空格编码为 `▁`。tiktoken 是 OpenAI 的快速*编码器*，基于预构建词表工作；它不进行训练。

经验法则：

- **训练新词表：** SentencePiece（多语言，无需预分词）或 HF Tokenizers。
- **对 GPT 词表进行快速推理：** tiktoken（cl100k_base、o200k_base）。
- **两者兼备：** HF Tokenizers——一个库，训练 + 服务一体化。

```figure
bpe-merge
```

## 动手实现

### 第 1 步：从零实现 BPE

参见 `code/main.py`。核心循环：

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

算法编码了三个事实。`</w>` 标记词尾，因此 "low"（后缀）和 "lower"（前缀）保持区分。频率加权使高频词对优先胜出。合并列表是有序的——推理时按训练顺序应用合并。

### 第 2 步：使用学到的合并进行编码

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

朴素实现的时间复杂度为 O(n·|merges|)。生产级实现（tiktoken、HF Tokenizers）使用带优先队列的合并排名查找，运行时间接近线性。

### 第 3 步：实际使用 SentencePiece

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # 或 "unigram"
    character_coverage=0.9995, # CJK 语言应降低（例如英语 0.9995，日语 0.995）
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

注意：无需预分词，空格编码为 `▁`，`character_coverage` 控制稀有字符被保留还是映射到 `<unk>` 的积极程度。

### 第 4 步：使用 tiktoken 兼容 OpenAI 词表

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

仅编码。速度快（Rust 后端）。与 GPT-4/5 分词精确匹配，用于字节统计、成本估算、上下文窗口预算。

## 2026 年仍在出现的陷阱

- **分词器漂移（Tokenizer drift）。** 在词表 A 上训练，在词表 B 上部署。词元 ID 不同；模型输出垃圾。在 CI 中检查 `tokenizer.json` 的哈希值。
- **空格歧义。** BPE 对 "hello" 和 " hello" 产生不同的词元。务必显式指定 `add_special_tokens` 和 `add_prefix_space`。
- **多语言训练不足。** 以英语为主的语料库产生的词表将非拉丁文字分割成多 5-10 倍的词元。在 GPT-3.5 上，相同的提示在日语/阿拉伯语中的成本高 5-10 倍。o200k_base 部分解决了这个问题。
- **表情符号拆分。** 单个表情符号可能占用 5 个词元。在预算上下文时检查表情符号处理。

## 实际应用

2026 年的技术栈：

| 场景 | 选择 |
|------|------|
| 从零训练单语模型 | HF Tokenizers（BPE） |
| 训练多语言模型 | SentencePiece（Unigram，`character_coverage=0.9995`） |
| 提供兼容 OpenAI 的 API | tiktoken（GPT-4+ 使用 `o200k_base`） |
| 领域专用词表（代码、数学、蛋白质） | 在领域语料库上训练自定义 BPE，与基础词表合并 |
| 端侧推理、小模型 | Unigram（较小词表效果更好） |

词表大小是一个扩展决策，而非常量。粗略经验法则：小于 1B 参数用 32k，1-10B 用 50-100k，多语言/前沿模型用 200k+。

## 输出技能卡

保存为 `outputs/skill-bpe-vs-wordpiece.md`：

```markdown
---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
```

## 练习

1. **简单。** 在 `code/main.py` 的小型语料库上训练一个 500 次合并的 BPE。编码三个留出词。有多少词恰好产生 1 个词元，有多少产生 >1 个词元？
2. **中等。** 在 100 个英文维基百科句子上比较 `cl100k_base`、`o200k_base` 和你训练的 vocab=32k 的 SentencePiece BPE 的词元数量。报告每种方法的压缩比。
3. **困难。** 用 BPE、Unigram 和 WordPiece 训练同一个语料库。在一个小型情感分类器上测量使用每种方法时的下游准确率。选择的差异是否使 F1 分数移动超过 1 个点？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| BPE | 字节对编码 | 贪婪合并最频繁的字符对，直到达到目标词表大小。 |
| 字节级 BPE | 永远没有未知词元 | 在原始 256 字节上进行 BPE；GPT-2 / Llama 使用此方法。 |
| Unigram | 概率分词器 | 从大型候选集使用对数似然进行剪枝；T5、Gemma 使用此方法。 |
| SentencePiece | 处理空格的那个 | 在原始文本上训练 BPE/Unigram 的库；空格编码为 `▁`。 |
| tiktoken | 快的那个 | OpenAI 的 Rust 后端 BPE 编码器，用于预构建词表。不训练。 |
| 合并列表（Merge list） | 魔法数字 | 有序的 `(a, b) → ab` 合并列表；推理时按顺序应用。 |
| 字符覆盖度（Character coverage） | 多稀有才算太稀有？ | 训练语料库中分词器必须覆盖的字符比例；通常约 0.9995。 |

## 延伸阅读

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — BPE 论文。
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) — Unigram 论文。
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) — 该库论文。
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) — 简明参考。
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) — 使用手册 + 编码列表。
