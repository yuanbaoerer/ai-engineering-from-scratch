# 从零构建分词器

> 第 01 课给了你一件玩具。这一课给你一件武器。

**类型：** 构建
**语言：** Python
**前置知识：** 第 10 阶段，第 01 课（分词器：BPE、WordPiece、SentencePiece）
**时间：** 约 90 分钟

## 学习目标

- 构建一个生产级的 BPE 分词器，支持 Unicode、空白符归一化和特殊词元
- 实现字节级回退机制，使分词器能够编码任何输入（包括表情符号、中日韩文字和代码）而不产生未知词元
- 添加预分词正则表达式模式，在应用 BPE 合并之前于词边界处切分文本
- 在语料库上训练自定义分词器，并在多语言文本上评估其压缩率，与 tiktoken 进行对比

## 问题所在

你在第 01 课中实现的 BPE 分词器对英文文本有效。现在试试日文。或者表情符号。或者混用制表符和空格的 Python 代码。

它会崩溃。

不是因为 BPE 本身有误——而是因为实现不完整。一个生产级分词器需要处理任意编码的原始字节，在切分前归一化 Unicode，管理那些永远不会被合并的特殊词元，将预分词与子词切分串联起来，并且所有操作都要足够快，不至于在处理 15 万亿词元的训练流水线中成为瓶颈。

GPT-2 的分词器有 50,257 个词元。Llama 3 有 128,256 个。GPT-4 大约有 100,000 个。这些都不是玩具数字。这些词表背后的合并表是在数百 GB 的文本上训练出来的，而周围的配套机制——归一化、预分词、特殊词元注入、对话模板格式化——才是区分一个只能处理 "hello world" 的分词器和一个能处理整个互联网的分词器的关键。

你将构建这些机制。

## 核心概念

### 完整流水线

生产级分词器不是单一算法。它是一个包含五个阶段的流水线，每个阶段解决不同的问题。

```mermaid
graph LR
    A[原始文本] --> B[归一化]
    B --> C[预分词]
    C --> D[BPE 合并]
    D --> E[特殊词元]
    E --> F[词元 ID]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

每个阶段都有明确的职责：

| 阶段 | 功能 | 重要性 |
|------|-------------|----------------|
| 归一化 | NFKC Unicode，可选小写，可选去除重音符号 | "fi" 合字 (U+FB01) 变成 "fi"（两个字符）。不做这一步，同一个词会得到不同的词元。 |
| 预分词 | 在 BPE 之前将文本切分成块 | 防止 BPE 跨词边界合并。"the cat" 绝不应产生词元 "e c"。 |
| BPE 合并 | 对字节序列应用学习到的合并规则 | 核心压缩。将原始字节转换为子词词元。 |
| 特殊词元 | 注入 [BOS]、[EOS]、[PAD]、对话模板标记 | 这些词元有固定 ID。它们从不参与 BPE 合并。模型需要它们来理解结构。 |
| ID 映射 | 将词元字符串转换为整数 ID | 模型看到的是整数，不是字符串。 |

### 字节级 BPE

第 01 课的分词器操作的是 UTF-8 字节。这是正确的选择。但我们跳过了重要的一点：当这些字节不是有效的 UTF-8 时会发生什么？

字节级 BPE 通过将每个可能的字节值（0-255）都视为有效词元来解决这个问题。你的基础词表恰好是 256 个条目。任何文件——文本、二进制、损坏的数据——都可以被分词而不产生未知词元。

GPT-2 添加了一个技巧：将每个字节映射到一个可打印的 Unicode 字符，使词表保持人类可读。字节 0x20（空格）在他们的映射中变成字符 "G"。这纯粹是装饰性的。算法本身并不关心。

真正的威力在于：字节级 BPE 能处理地球上的所有语言。中文字符每个占 3 个 UTF-8 字节。日文可以是 3-4 字节。阿拉伯文、天城文、表情符号——都只是字节序列。BPE 算法在这些字节序列中寻找模式的方式，与它在英文 ASCII 字节中寻找模式的方式完全相同。

### 预分词

在 BPE 接触你的文本之前，你需要将其切分成块。这可以防止合并算法创建跨越词边界的词元。

GPT-2 使用一个正则表达式模式来切分文本：

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

这个模式在缩略形式处切分（"don't" 变成 "don" + "'t"），将带可选前导空格的单词、数字、标点符号和空白符分别切分。前导空格保留并附加在单词上——所以 "the cat" 变成 [" the", " cat"]，而不是 ["the", " ", "cat"]。

Llama 使用 SentencePiece，它完全跳过了正则表达式。它将原始字节流视为一个长序列，让 BPE 算法自己找出边界。这更简单，但给了 BPE 更多创建跨词词元的自由。

选择很重要。GPT-2 的正则表达式防止分词器学习到将一个词末尾的 "the" 和下一个词开头的 "the" 合并。SentencePiece 允许这样做，有时能产生更高效的压缩，但词元的可解释性更差。

### 特殊词元

每个生产级分词器都会为结构标记预留词元 ID：

| 词元 | 用途 | 使用者 |
|-------|---------|---------|
| `[BOS]` / `<s>` | 序列开始 | Llama 3、GPT |
| `[EOS]` / `</s>` | 序列结束 | 所有模型 |
| `[PAD]` | 批次对齐填充 | BERT、T5 |
| `[UNK]` | 未知词元（字节级 BPE 消除了这个需求） | BERT、WordPiece |
| `<\|im_start\|>` | 对话消息边界开始 | ChatGPT、Qwen |
| `<\|im_end\|>` | 对话消息边界结束 | ChatGPT、Qwen |
| `<\|user\|>` | 用户轮次标记 | Llama 3 |
| `<\|assistant\|>` | 助手轮次标记 | Llama 3 |

特殊词元不会被 BPE 拆分。它们在合并算法运行之前被精确匹配，替换为固定 ID，周围的文本则正常分词。

### 对话模板

这是大多数人感到困惑、大多数实现容易出错的地方。

当你向对话模型发送消息时，API 接收的是一个消息列表：

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

模型看到的不是 JSON。它看到的是扁平的词元序列。对话模板使用特殊词元将消息转换为这个扁平序列。每个模型的做法都不同：

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

模板弄错，模型就会输出垃圾。它是在一种精确的格式上训练的。任何偏差——少一个换行、交换词元、多空格——都会让输入偏离训练分布。

### 速度

Python 对于生产级分词来说太慢了。

tiktoken（OpenAI）是用 Rust 写的，带有 Python 绑定。HuggingFace tokenizers 也是 Rust。SentencePiece 是 C++。这些相比纯 Python 能实现 10-100 倍的加速。

作为参考：以每秒 100 万词元（快速的 Python）的速度为 Llama 3 预训练分词 15 万亿词元，需要 174 天。以每秒 1 亿词元（Rust）的速度，只需要 1.7 天。

你用 Python 构建是为了理解算法。在生产环境中，你会使用编译后的实现，只接触 Python 包装层。

```figure
weight-tying
```

## 动手构建

### 步骤 1：字节级编码

基础部分。将任意字符串转换为字节序列，将每个字节映射到一个可打印字符以便显示，并支持反向转换。

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

用多语言文本测试，观察字节数：

```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

"hello" 是 5 字节。"你好" 是 6 字节（每个字符 3 字节）。火焰表情符号是 4 字节。字节级分词器不在乎是什么语言。字节就是字节。

### 步骤 2：基于正则的预分词器

使用 GPT-2 正则表达式模式将文本切分成块。每个块由 BPE 独立分词。

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

`regex` 模块支持 Unicode 属性转义（`\p{L}` 表示字母，`\p{N}` 表示数字）。标准库的 `re` 模块不支持，因此我们回退到 ASCII 字符类。对于生产级多语言分词器，请安装 `regex`。

试一下：

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

前导空格保留并附加在单词上。缩略形式在撇号处切分。标点符号成为独立的块。BPE 永远不会跨这些边界合并词元。

### 步骤 3：字节序列上的 BPE

第 01 课的核心算法，但现在独立地对预分词后的块进行操作。

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### 步骤 4：特殊词元处理

特殊词元需要精确匹配和固定 ID。它们完全绕过 BPE。

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### 步骤 5：完整的分词器类

将所有环节串联起来：归一化、按特殊词元切分、预分词、BPE 合并、映射为 ID。

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### 步骤 6：多语言测试

真正的考验。用英文、中文、表情符号和代码来测试它。

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

中文字符每个产生 3 字节。表情符号产生 4 字节。这些都不会让分词器崩溃。都不会产生未知词元。这就是字节级 BPE 的威力。

## 使用它

### 对比真实分词器

加载 Llama 3、GPT-4 和 Mistral 的真实分词器。看看它们各自如何处理同一段多语言文本。

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

你会看到同一段文本产生不同的词元数量。拥有 128K 词表的 Llama 3 更积极地合并常见模式。拥有 100K 词表的 GPT-4 居中。拥有 32K 词表的 Mistral 产生更多词元，但嵌入层更小。

权衡始终相同：更大的词表意味着更短的序列，但更多的参数。

## 交付

本课产出一个用于构建和调试生产级分词器的提示词。详见 `outputs/prompt-tokenizer-builder.md`。

## 练习题

1. **简单：** 添加一个 `get_token_bytes(id)` 方法，显示任意词元 ID 的原始字节。用它检查你最常见的合并词元实际代表什么。
2. **中等：** 实现 Llama 风格的预分词器，它在空白符和数字处切分但保留前导空格。在相同语料库上将其词表与 GPT-2 正则表达式方法进行对比。
3. **困难：** 添加一个对话模板方法，接收 `{"role": ..., "content": ...}` 消息列表，并为 Llama 3 对话格式生成正确的词元序列。与 HuggingFace 的实现进行测试对比。

## 关键术语

| 术语 | 人们通常的说法 | 实际含义 |
|------|----------------|----------------------|
| 字节级 BPE (Byte-level BPE) | "基于字节的分词器" | 基础词表为 256 个字节值的 BPE——能处理任何输入而不产生未知词元 |
| 预分词 (Pre-tokenization) | "BPE 之前的切分" | 基于正则或规则的切分，防止 BPE 跨词边界合并 |
| NFKC 归一化 (NFKC normalization) | "Unicode 清理" | 先规范分解再做兼容组合——"fi" 合字变成 "fi"，全角 "A" 变成 "A" |
| 对话模板 (Chat template) | "消息如何变成词元" | 将角色/内容消息列表转换为扁平词元序列的精确格式——模型特定，必须匹配训练格式 |
| 特殊词元 (Special tokens) | "控制词元" | 绕过 BPE 的预留词元 ID——[BOS]、[EOS]、[PAD]、对话标记——在合并前精确匹配 |
| 生育率 (Fertility) | "每词词元数" | 输出词元与输入词数的比率——GPT-4 英文为 1.3，韩文为 2-3，越高意味着上下文浪费越多 |
| tiktoken | "OpenAI 分词器" | 带 Python 绑定的 Rust BPE 实现——比纯 Python 快 10-100 倍 |
| 合并表 (Merge table) | "词表" | 训练期间学习到的字节对合并的有序列表——这就是分词器学到的知识 |

## 延伸阅读

- [OpenAI tiktoken 源码](https://github.com/openai/tiktoken) —— GPT-3.5/4 使用的 Rust BPE 实现
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) —— 支持 BPE、WordPiece、Unigram 的 Rust 分词器库
- [Llama 3 论文 (Meta, 2024)](https://arxiv.org/abs/2407.21783) —— 128K 词表和分词器训练的详细信息
- [SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226) —— 语言无关的分词
- [GPT-2 分词器源码](https://github.com/openai/gpt-2/blob/master/src/encoder.py) —— 原始字节到 Unicode 的映射
