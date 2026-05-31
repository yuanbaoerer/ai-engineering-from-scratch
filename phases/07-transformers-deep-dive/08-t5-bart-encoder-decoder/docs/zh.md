# T5、BART — 编码器-解码器模型

> 编码器理解。解码器生成。将它们重新组合，你就得到了一个为输入 → 输出任务构建的模型：翻译、摘要、改写、转录。

**类型:** 学习
**语言:** Python
**前置知识:** 第 7 阶段 · 05（完整 Transformer）、第 7 阶段 · 06（BERT）、第 7 阶段 · 07（GPT）
**时间:** 约 45 分钟

## 问题所在

仅解码器的 GPT 和仅编码器的 BERT 各自为不同目标拆解了 2017 年架构。但许多任务天然是输入-输出的：

- 翻译：英语 → 法语。
- 摘要：5,000 token 文章 → 200 token 摘要。
- 语音识别：音频 token → 文本 token。
- 结构化提取：散文 → JSON。

对于这些，编码器-解码器是最合适的。编码器产生源的密集表示。解码器生成输出，在每一步交叉关注该表示。训练在输出端是偏移一位的。与 GPT 相同的损失，只是以编码器输出为条件。

两篇论文定义了现代剧本：

1. **T5** (Raffel et al. 2019)。"文本到文本迁移 Transformer。"每个 NLP 任务被重新定义为文本输入、文本输出。单一架构、单一词表、单一损失。在掩码跨度预测上预训练（破坏输入中的跨度，在输出中解码它们）。
2. **BART** (Lewis et al. 2019)。"双向和自回归 Transformer。"去噪自编码器：以多种方式破坏输入（打乱、掩码、删除、旋转），让解码器重建原始。

2026 年，编码器-解码器格式在输入结构重要的地方继续存在：

- Whisper（语音 → 文本）。
- 谷歌的翻译栈。
- 一些具有不同上下文和编辑结构的代码补全/修复模型。
- 用于结构化推理任务的 Flan-T5 及变体。

解码器赢得了聚光灯，但编码器-解码器从未消失。

## 核心概念

![带交叉注意力的编码器-解码器](../assets/encoder-decoder.svg)

### 前向循环

```
源 token ─▶ 编码器 ─▶ (N_src, d_model)  ──┐
                                           │
目标 token ─▶ 解码器块                     │
                 ├─▶ 掩码自注意力          │
                 ├─▶ 交叉注意力 ◀──────────┘
                 └─▶ FFN
                ↓
              下一个 token logits
```

关键是编码器对每个输入运行一次。解码器自回归运行，但在每一步交叉关注*相同的*编码器输出。缓存编码器输出对长输入是免费的加速。

### T5 预训练——跨度破坏

选取输入的随机跨度（平均长度 3 token，总计 15%）。用唯一哨兵替换每个跨度：`<extra_id_0>`、`<extra_id_1>` 等。解码器仅输出带哨兵前缀的被破坏跨度：

```
源: The quick <extra_id_0> fox jumps <extra_id_1> dog
目标: <extra_id_0> brown <extra_id_1> over the lazy
```

比预测整个序列更便宜的信号。在 T5 论文的消融中与 MLM (BERT) 和 prefix-LM (UniLM) 竞争。

### BART 预训练——多噪声去噪

BART 尝试五种噪声函数：

1. Token 掩码。
2. Token 删除。
3. 文本填充（掩码一个跨度，解码器插入正确长度）。
4. 句子排列。
5. 文档旋转。

组合文本填充 + 句子排列产生了最佳下游数字。解码器始终重建原始。BART 的输出是完整序列，不仅是被破坏的跨度——因此预训练计算高于 T5。

### 推理

与 GPT 相同的自回归生成。贪心/束搜索/top-p 采样适用。束搜索（宽度 4-5）是翻译和摘要的标准，因为输出分布比聊天更窄。

### 2026 年何时选择各变体

| 任务 | 编码器-解码器？ | 原因 |
|------|----------------|------|
| 翻译 | 通常是 | 清晰的源序列；固定输出分布；束搜索有效 |
| 语音转文本 | 是 (Whisper) | 输入模态与输出不同；编码器塑造音频特征 |
| 聊天 / 推理 | 否，仅解码器 | 无持久"输入"——对话就是序列 |
| 代码补全 | 通常否 | 仅解码器长上下文胜出；代码模型如 Qwen 2.5 Coder 是仅解码器 |
| 摘要 | 都可以 | BART、PEGASUS 胜过早期仅解码器基线；现代仅解码器 LLM 匹配 |
| 结构化提取 | 都可以 | T5 很干净因为"文本 → 文本"吸收任何输出格式 |

约 2022 年以来的趋势：仅解码器接管了编码器-解码器曾经拥有的任务，因为 (a) 指令微调的仅解码器 LLM 通过提示泛化到任何事，(b) 一种架构比两种更容易扩展，(c) RLHF 假设解码器。编码器-解码器在输入模态不同（语音、图像）或束搜索质量重要的地方坚守。

## 动手实现

参见 `code/main.py`。我们为玩具语料实现 T5 风格的跨度破坏——本课最有用的部分，因为它出现在此后每个编码器-解码器预训练配方中。

### 第一步：跨度破坏

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """选取总和约为 mask_rate 的 token 的跨度。返回 (破坏后输入, 目标)。"""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

目标格式是 T5 约定：`<sent0> span0 <sent1> span1 ...`。破坏后的输入在跨度位置交替不变 token 和哨兵 token。

### 第二步：验证往返

给定破坏后的输入和目标，重建原始句子。如果破坏是可逆的，前向传播是明确定义的。这是完整性检查——真实训练从不做这个，但测试便宜且能捕获跨度记账中的偏移错误。

### 第三步：BART 噪声

五个函数：`token_mask`、`token_delete`、`text_infill`、`sentence_permute`、`document_rotate`。组合其中两个并展示结果。

## 使用场景

HuggingFace 参考：

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

T5 的技巧：任务名称放入输入文本。同一个模型处理数十种任务，因为每种任务都是文本输入、文本输出。2026 年这种模式已被指令微调的仅解码器模型泛化，但 T5 首先将其编纂。

## 交付使用

参见 `outputs/skill-seq2seq-picker.md`。该技能根据输入-输出结构、延迟和质量目标，在编码器-解码器和仅解码器之间为新任务选择。

## 练习

1. **简单。** 运行 `code/main.py`，对 30 个 token 的句子应用跨度破坏，验证拼接非哨兵源 token 与解码目标跨度能重现原始。
2. **中等。** 实现 BART 的 `text_infill` 噪声：用单个 `<mask>` token 替换随机跨度，解码器必须推断正确的跨度长度加内容。展示一个例子。
3. **困难。** 在微型英语 → 猪拉丁语料（200 对）上微调 `flan-t5-small`。在保留的 50 对集上测量 BLEU。与在相同数据上用相同计算微调 `Llama-3.2-1B` 比较。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 编码器-解码器 | "Seq2seq transformer" | 两个栈：用于输入的双向编码器，带交叉注意力的因果解码器用于输出。 |
| 交叉注意力 | "源与目标对话的地方" | 解码器的 Q × 编码器的 K/V。编码器信息进入解码器的唯一位置。 |
| 跨度破坏 | "T5 的预训练技巧" | 用哨兵 token 替换随机跨度；解码器输出跨度。 |
| 去噪目标 | "BART 的玩法" | 对输入应用噪声函数，训练解码器重建干净序列。 |
| 哨兵 token | "`<extra_id_N>` 占位符" | 在源中标记被破坏跨度并在目标中重新标记的特殊 token。 |
| Flan | "指令微调的 T5" | 在 >1,800 个任务上微调的 T5；使编码器-解码器在指令跟随上具有竞争力。 |
| 束搜索 | "解码策略" | 每步保留 top-k 部分序列；翻译/摘要的标准。 |
| 教师强制 | "训练时输入" | 训练时将真实前一个输出 token 传给解码器，而非采样的。 |

## 延伸阅读

- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — T5。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) — BART。
- [Chung et al. (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — Flan-T5。
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper，2026 年标准的编码器-解码器。
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) — 参考实现。
