# 机器翻译

> 翻译是为NLP研究买单三十年、至今仍在买单的任务。

**类型：** 实战构建
**语言：** Python
**前置课程：** 第5阶段 · 第10课（注意力机制），第5阶段 · 第4课（GloVe、FastText、子词）
**时长：** 约75分钟

## 问题描述

模型读取一种语言的句子，输出另一种语言的句子。句子长度各不相同。词序各不相同。某些源语言词映射到多个目标语言词，反之亦然。习语无法进行一对一映射。"I miss you"用法语说是"tu me manques"——字面意思是"你对我是缺失的"。没有词级对齐能经受住这种考验。

机器翻译（Machine Translation）是迫使NLP发明编码器-解码器（Encoder-Decoder）、注意力机制、Transformer，乃至整个大语言模型范式的任务。每一步进展的出现，都是因为翻译质量是可衡量的，而人与机器之间的差距始终顽固地存在。

本课跳过历史回顾，直授2026年的工作流程：预训练多语言编码器-解码器（NLLB-200或mBART）、子词分词、束搜索、BLEU和chrF评估，以及那些仍然悄无声息地溜进生产环境的少数失败模式。

## 核心概念

![机器翻译流程：分词 → 编码 → 带注意力的解码 → 去分词](../assets/mt-pipeline.svg)

现代机器翻译是在平行文本上训练的Transformer编码器-解码器。编码器使用源语言的分词方式读取源文本。解码器通过交叉注意力（Cross-Attention，第10课）利用编码器的输出，逐个子词地生成目标文本。解码使用束搜索（Beam Search）以避免贪心解码的陷阱。输出经过去分词、去大写处理，并与参考译文进行评分。

三个操作层面的选择决定了实际机器翻译的质量。

- **分词器。** 在混合语料库上训练的SentencePiece BPE。跨语言共享词表是NLLB实现零样本语言对翻译的关键。
- **模型大小。** NLLB-200蒸馏版600M可在笔记本电脑上运行。NLLB-200 3.3B是发布的生产默认版本。54.5B是研究天花板。
- **解码策略。** 通用内容使用束宽4-5。使用长度惩罚以避免输出过短。需要术语一致性时使用约束解码。

## 动手实现

### 第1步：调用预训练机器翻译模型

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

这里有三个关键点。`src_lang`告诉分词器使用哪种文字系统和分词方式。`forced_bos_token_id`告诉解码器生成哪种语言。两者都是NLLB特有的技巧；mBART和M2M-100使用各自的约定，且不可互换。

### 第2步：BLEU和chrF

BLEU衡量输出与参考译文之间的n-gram重叠度。使用四个参考n-gram大小（1-4），精度的几何平均值，以及对过短输出的简短惩罚。分数范围为[0, 100]。使用广泛。但解读令人沮丧：30 BLEU表示"可用"；40表示"良好"；50表示"卓越"；低于1 BLEU的差异属于噪声。

chrF衡量字符级F分数。对形态丰富的语言更敏感，因为BLEU会低估匹配数。通常与BLEU一起报告。

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

始终使用`sacrebleu`。它会标准化分词，使得不同论文之间的分数可比较。自己编写BLEU计算是产生误导性基准测试的根源。

### 三层评估体系（2026年）

现代机器翻译评估使用三个互补的指标家族。发布时至少使用其中两种。

- **启发式指标**（BLEU、chrF）。快速、基于参考、可解释、对释义不敏感。用于遗留对比和回归检测。
- **学习型指标**（COMET、BLEURT、BERTScore）。基于人类判断训练的神经模型；比较翻译与源文和参考译文的语义相似度。自2023年以来，COMET与机器翻译研究的关联度最高，是2026年质量敏感场景的生产默认选择。
- **大语言模型作为裁判（LLM-as-a-Judge）**（无参考）。提示大型模型对翻译的流畅性、充分性、语气、文化适当性进行评分。当评分标准设计良好时，GPT-4作为裁判与人类判断的一致性约为80%。用于没有参考译文的开放式内容。

实用的2026年技术栈：`sacrebleu`用于BLEU和chrF，`unbabel-comet`用于COMET，提示式大语言模型用于最终面向用户的信号。在将任何指标应用于生产数据之前，请用50-100个人工标注样本进行校准。

无参考指标（COMET-QE、BLEURT-QE、大语言模型作为裁判）允许你在没有参考译文的情况下评估翻译，这对于参考译文不存在的长尾语言对尤为重要。

### 第3步：生产环境中的故障

上述工作流程在80%的情况下能流畅翻译，在剩余20%的情况下会悄无声息地失败。已知的失败模式：

- **幻觉（Hallucination）。** 模型编造源文中没有的内容。在不熟悉的领域词汇中常见。症状：输出流畅但陈述了源文未提及的事实。缓解措施：对领域术语使用约束解码，对受监管内容进行人工审核，监控输出长度远超输入的情况。
- **目标语言错误。** 模型翻译成了错误的语言。NLLB在稀有语言对上出人意料地容易出现此问题。缓解措施：验证`forced_bos_token_id`，并始终使用语言ID模型对输出进行解码检查。
- **术语漂移。** "Sign up"在文档1中变成"s'inscrire"，在文档2中变成"créer un compte"。对于UI文本和面向用户的字符串，一致性比原始质量更重要。缓解措施：术语表约束解码或译后编辑词典。
- **语体不匹配。** 法语的"tu"与"vous"、日语的敬语级别。模型会选择训练数据中更常见的形式。对于面向客户的内容，这通常是错误的。缓解措施：如果模型支持，使用带语体标记的提示前缀，或仅在正式语料上微调小型模型。
- **短输入长度爆炸。** 非常短的输入句子经常产生过长的翻译，因为长度惩罚在约5个源词以下会急剧失效。缓解措施：设置与源文长度成比例的硬性最大长度限制。

### 第4步：领域微调

预训练模型是通才。法律、医学或游戏对话翻译可以从领域平行数据的微调中获得显著收益。方法并不复杂：

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

几千个高质量的平行样本胜过几十万个嘈杂的网络爬取样本。训练数据的质量是生产环境中最大的杠杆。

## 实际应用

2026年机器翻译生产技术栈：

| 使用场景 | 推荐起点 |
|---------|---------------------------|
| 任意语言互译，200种语言 | `facebook/nllb-200-distilled-600M`（笔记本）或`nllb-200-3.3B`（生产） |
| 以英语为中心，高质量，50种语言 | `facebook/mbart-large-50-many-to-many-mmt` |
| 短文本，低成本推理，英法/德/西 | Helsinki-NLP / Marian模型 |
| 延迟敏感的浏览器端 | ONNX量化的Marian（约50 MB） |
| 最高质量，愿意付费 | GPT-4 / Claude / Gemini + 翻译提示 |

截至2026年，大语言模型在多个语言对上已经超越了专用机器翻译模型，尤其是在习语内容和长上下文方面。权衡在于每token的成本和延迟。当上下文长度、风格一致性或通过提示进行领域适配比吞吐量更重要时，选择大语言模型。

## 交付使用

保存为`outputs/skill-mt-evaluator.md`：

```markdown
---
name: mt-evaluator
description: Evaluate a machine translation output for shipping.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Given a source text and a candidate translation, output:

1. Automatic score estimate. BLEU and chrF ranges you would expect. State whether a reference is available.
2. Five-point human-verifiable check list: (a) content preservation (no hallucinations), (b) correct language, (c) register / formality match, (d) terminology consistency with glossary if provided, (e) no truncation or length explosion.
3. One domain-specific issue to probe. E.g., for legal: named entities and statute citations. For medical: drug names and dosages. For UI: placeholder variables `{name}`.
4. Confidence flag. "Ship" / "Ship with review" / "Do not ship". Tie to the severity of issues found in step 2.

Refuse to ship a translation without a language-ID check on output. Refuse to evaluate without a reference unless the user explicitly opts in to reference-free scoring (COMET-QE, BLEURT-QE). Flag any content over 1000 tokens as likely needing chunked translation.
```

## 练习

1. **简单。** 使用`nllb-200-distilled-600M`将一段5句的英文段落翻译成法语，再翻译回英文。衡量往返翻译与原文的接近程度。你应该会看到语义保持但用词发生漂移。
2. **中等。** 使用`fasttext lid.176`或`langdetect`对翻译输出实现语言ID检查。集成到机器翻译调用中，以便在返回之前捕获目标语言错误的生成。
3. **困难。** 在你选择的5000对领域语料库上微调`nllb-200-distilled-600M`。在微调前后分别在留出集上测量BLEU。报告哪些类型的句子有所改善，哪些发生了退化。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| BLEU | 翻译分数 | 带简短惩罚的n-gram精度。[0, 100]。 |
| chrF | 字符F分数 | 字符级F分数。对形态丰富的语言更敏感。 |
| NMT | 神经机器翻译 | 在平行文本上训练的Transformer编码器-解码器。2017年以来的默认方法。 |
| NLLB | 不落下任何语言 | Meta的200种语言机器翻译模型家族。 |
| 约束解码（Constrained Decoding） | 受控输出 | 强制特定token或n-gram在输出中出现或不出现。 |
| 幻觉（Hallucination） | 编造内容 | 模型输出中不被源文支持的内容。 |
| 无参考指标（Reference-free Metrics） | 无需参考译文 | 在没有参考译文的情况下评估翻译质量的指标，如COMET-QE和BLEURT-QE。 |
| 术语漂移（Terminology Drift） | 用词不一致 | 同一源文术语在不同文档或上下文中被翻译为不同目标词的现象。 |
| 语体不匹配（Register Mismatch） | 正式程度不当 | 翻译输出的正式程度（如法语tu/vous）与预期使用场景不匹配。 |

## 延伸阅读

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) — NLLB论文。
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) — 为什么`sacrebleu`是报告BLEU的唯一正确方式。
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) — chrF论文。
- [Hugging Face MT guide](https://huggingface.co/docs/transformers/tasks/translation) — 实用的微调教程。
