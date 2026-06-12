# BERT — 掩码语言建模

> GPT 预测下一个词。BERT 预测缺失的词。一句话的差异——影响了五年来所有嵌入相关的工作。

**类型:** 构建
**语言:** Python
**前置知识:** 第 7 阶段 · 05（完整 Transformer）、第 5 阶段 · 02（文本表示）
**时间:** 约 45 分钟

## 问题所在

2018 年，每个 NLP 任务——情感分析、NER、QA、蕴含——都在自己的标注数据上从头训练自己的模型。没有预训练的"理解英语"检查点可以微调。ELMo (2018) 展示了可以用双向 LSTM 预训练上下文嵌入；有帮助但不能泛化。

BERT (Devlin et al. 2018) 提出：如果我们取一个 Transformer 编码器，在互联网上的每个句子上训练它，强制它从两侧上下文预测缺失的词呢？然后在下游任务上微调一个头。参数效率令人惊叹。

结果：18 个月内，BERT 及其变体（RoBERTa、ALBERT、ELECTRA）主导了当时存在的每个 NLP 排行榜。到 2020 年，地球上的每个搜索引擎、内容审核管道和语义搜索系统内部都有一个 BERT。

2026 年，仅编码器模型仍然是分类、检索和结构化提取的正确工具——它们每 token 运行速度比解码器快 5-10 倍，它们的嵌入是每个现代检索栈的骨干。ModernBERT (2024年12月) 通过 Flash Attention + RoPE + GeGLU 将架构推到了 8K 上下文。

## 核心概念

![掩码语言建模：选取 token、掩码、预测原始](../assets/bert-mlm.svg)

### 训练信号

取一个句子：`the quick brown fox jumps over the lazy dog`。

随机掩码 15% 的 token：

```
输入:  the [MASK] brown fox jumps [MASK] the lazy dog
目标: the quick brown fox jumps over the lazy dog
```

训练模型预测掩码位置的原始 token。因为编码器是双向的，预测位置 1 的 `[MASK]` 可以使用位置 2+ 的 `brown fox jumps`。这是 GPT 做不到的。

### BERT 掩码规则

被选中预测的 15% token 中：

- 80% 被替换为 `[MASK]`。
- 10% 被替换为随机 token。
- 10% 保持不变。

为什么不总是 `[MASK]`？因为 `[MASK]` 在推理时从不出现。训练模型在 100% 掩码位置期望 `[MASK]` 会在预训练和微调之间产生分布偏移。10% 随机 + 10% 不变让模型保持诚实。

### 下一句预测 (NSP)——以及被抛弃的原因

原始 BERT 还训练 NSP：给定两个句子 A 和 B，预测 B 是否跟随 A。RoBERTa (2019) 消融了它，表明 NSP 有害而非有益。现代编码器跳过它。

### 2026 年的变化：ModernBERT

2024 年 ModernBERT 论文用 2026 年原语重建了块：

| 组件 | 原始 BERT (2018) | ModernBERT (2024) |
|------|------------------|-------------------|
| 位置 | 可学习绝对 | RoPE |
| 激活 | GELU | GeGLU |
| 归一化 | LayerNorm | Pre-norm RMSNorm |
| 注意力 | 完整密集 | 交替局部 (128) + 全局 |
| 上下文长度 | 512 | 8192 |
| 分词器 | WordPiece | BPE |

与 2018 年的栈不同，它是 Flash Attention 原生的。在序列长度 8K 下推理速度比 DeBERTa-v3 快 2-3 倍，GLUE 分数更好。

### 2026 年仍然选择编码器的场景

| 任务 | 为什么编码器优于解码器 |
|------|----------------------|
| 检索 / 语义搜索嵌入 | 双向上下文 = 每 token 更好的嵌入质量 |
| 分类（情感、意图、毒性） | 一次前向传播；无生成开销 |
| NER / token 标注 | 每位置输出，原生双向 |
| 零样本蕴含 (NLI) | 编码器顶部的分类头 |
| RAG 重排序器 | 交叉编码器评分，比 LLM 重排序器快 10 倍 |

```figure
transformer-residual
```

## 动手实现

### 第一步：掩码逻辑

参见 `code/main.py`。函数 `create_mlm_batch` 接收 token ID 列表、词表大小和掩码概率。返回输入 ID（应用掩码后）和标签（仅在掩码位置，其他位置为 -100——PyTorch 的忽略索引约定）。

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: 保持原样
    return input_ids, labels
```

### 第二步：在微型语料上运行 MLM 预测

在 20 个词的词表、200 个句子上训练 2 层编码器 + MLM 头。无梯度——我们做前向传播完整性检查。完整训练需要 PyTorch。

### 第三步：比较掩码类型

展示三方规则如何让模型在没有 `[MASK]` 时仍可用。在未掩码句子和掩码句子上预测。两者都应该产生合理的 token 分布，因为模型在训练中见过两种模式。

### 第四步：微调头

在玩具情感数据集上用分类头替换 MLM 头。只有头训练；编码器冻结。这是每个 BERT 应用遵循的模式。

## 使用场景

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**嵌入模型是微调过的 BERT。** `sentence-transformers` 模型如 `all-MiniLM-L6-v2` 是用对比损失训练的 BERT。编码器相同。损失变了。

**交叉编码器重排序器也是微调过的 BERT。** 在 `[CLS] query [SEP] doc [SEP]` 上的配对分类。查询和文档之间的双向注意力正是交叉编码器比双编码器质量更高的原因。

**2026 年何时不选 BERT。** 任何生成性任务。编码器没有合理的方式自回归生成 token。还有：任何 1B 参数以下，小型解码器可以用更多灵活性匹配质量的场景（Phi-3-Mini、Qwen2-1.5B）。

## 交付使用

参见 `outputs/skill-bert-finetuner.md`。该技能为新分类或提取任务确定 BERT 微调范围（骨干选择、头规范、数据、评估、停止）。

## 练习

1. **简单。** 运行 `code/main.py`，在 10,000 个 token 上打印掩码分布。确认约 15% 被选中，其中约 80% 变成 `[MASK]`。
2. **中等。** 实现全词掩码：如果一个词被分词为子词，要么全部掩码要么全部不掩码。测量这是否在 500 句语料上提高 MLM 准确率。
3. **困难。** 在公开数据集的 10,000 个句子上训练微型（2 层、d=64）BERT。为 SST-2 情感微调 `[CLS]` token。在匹配参数下与仅解码器基线比较——谁赢？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| MLM | "掩码语言建模" | 训练信号：随机将 15% token 替换为 `[MASK]`，预测原始。 |
| 双向 | "两边都看" | 编码器注意力无因果掩码——每个位置看到其他所有位置。 |
| `[CLS]` | "池化 token" | 添加到每个序列开头的特殊 token；其最终嵌入用作句子级表示。 |
| `[SEP]` | "分段分隔符" | 分隔配对序列（如 query/doc、句子 A/B）。 |
| NSP | "下一句预测" | BERT 的第二个预训练任务；RoBERTa 证明无用，2019 年后废弃。 |
| 微调 | "适配任务" | 编码器基本冻结；在顶部为下游任务训练小型头。 |
| 交叉编码器 | "重排序器" | 将查询和文档都作为输入的 BERT，输出相关性分数。 |
| ModernBERT | "2024 年刷新" | 用 RoPE、RMSNorm、GeGLU、交替局部/全局注意力、8K 上下文重建的编码器。 |

## 延伸阅读

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — 原始论文。
- [Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — 如何正确训练 BERT；废弃 NSP。
- [Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) — 替换 token 检测在匹配计算下优于 MLM。
- [Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) — ModernBERT 论文。
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) — 标准编码器参考。
