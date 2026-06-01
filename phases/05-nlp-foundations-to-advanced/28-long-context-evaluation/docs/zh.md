# 长上下文评估 — NIAH、RULER、LongBench、MRCR

> Gemini 3 Pro 宣称支持 1000 万 token 上下文。但在 100 万 token 时，8 针 MRCR 的得分降至 26.3%。宣称的 ≠ 实际可用的。长上下文评估告诉你所用模型的真实容量。

**类型：** 学习
**语言：** Python
**前置要求：** Phase 5 · 13（问答系统）、Phase 5 · 23（分块策略）
**时间：** 约 60 分钟

## 问题所在

你有一份 200 页的合同。模型宣称支持 100 万 token 上下文。你把合同粘贴进去，问："终止条款是什么？"模型给出了回答——但回答的是封面页的内容，因为终止条款位于 12 万 token 深处，超出了模型实际能关注到的范围。

这就是 2026 年的上下文容量差距。规格书上写着 100 万或 1000 万。实际上只有 60-70% 是可用的，而且"可用"取决于具体任务。

- **检索（单针大海捞针）：** 在前沿模型上，直到宣称的最大长度都接近完美。
- **多跳/聚合：** 在大多数模型上，超过约 12.8 万 token 后急剧下降。
- **对分散事实的推理：** 最先失败的任务。

长上下文评估衡量这些维度。本课介绍各基准测试、每个测试实际衡量什么，以及如何为你的领域构建自定义针测试。

## 核心概念

![NIAH 基线、RULER 多任务、LongBench 全面评估](../assets/long-context-eval.svg)

**大海捞针（NIAH，2023）。** 将一个事实（"魔法词是 pineapple"）放在长上下文中的可控深度位置。要求模型检索它。遍历深度 × 长度。这是最初的长上下文基准测试。前沿模型现在已在此测试上饱和；它是必要但不充分的基线。

**RULER（Nvidia，2024）。** 4 个类别、13 种任务类型：检索（单键/多键/多值）、多跳追踪（变量跟踪）、聚合（常见词频率）、问答。可配置上下文长度（4k 到 128k+）。能揭示在 NIAH 上饱和但在多跳上失败的模型。在 2024 年的发布中，17 个宣称支持 32k+ 上下文的模型中，只有一半在 32k 时保持了质量。

**LongBench v2（2024）。** 503 道多选题，8k-200 万词上下文，六个任务类别：单文档问答、多文档问答、长上下文学习、长对话、代码仓库、长结构化数据。真实世界长上下文行为的生产基准。

**MRCR（多轮共指消解）。** 大规模多轮共指。8 针、24 针、100 针变体。暴露模型在注意力退化前能同时处理多少事实。

**NoLiMa。** "非词汇针。"针和查询没有字面重叠；检索需要一步语义推理。比 NIAH 更难。

**HELMET。** 拼接多篇文档，提问其中任意一篇的问题。测试选择性注意力。

**BABILong。** 将 bAbI 推理链嵌入无关的干草堆中。测试"干草堆中的推理"，而不仅仅是检索。

### 实际应该报告什么

- **宣称的上下文窗口。** 规格书上的数字。
- **有效检索长度。** 在某个阈值（如 90%）下通过 NIAH 的长度。
- **有效推理长度。** 在该阈值下通过多跳或聚合的长度。
- **退化曲线。** 按任务类型绘制准确率 vs 上下文长度的图表。

给你的规格书两个数字：检索有效长度和推理有效长度。通常推理有效长度是宣称窗口的 25-50%。

## 动手实现

### 步骤 1：为你的领域构建自定义 NIAH

参见 `code/main.py`。骨架代码：

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # 重复填充文本直到足够填满干草堆主体。
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

遍历 `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}。绘制热力图。这就是你目标模型的 NIAH 卡片。

### 步骤 2：多针变体

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

类似"三个魔法词分别是什么？"的问题需要检索全部三个。单针成功不能预测多针成功。

### 步骤 3：多跳变量追踪（RULER 风格）

```python
haystack = """X1 = 42. ... （填充） ... X2 = X1 + 10. ... （填充） ... X3 = X2 * 2."""
question = "What is X3?"
```

答案需要链式三次赋值。前沿模型在 128k 时通常降至 50-70% 准确率。

### 步骤 4：在你的技术栈上运行 LongBench v2

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

按类别报告准确率。聚合分数会掩盖任务级别的巨大差异。

## 常见陷阱

- **仅用 NIAH 评估。** 在 100 万 token 通过 NIAH 不能说明多跳能力。务必运行 RULER 或自定义多跳测试。
- **均匀深度采样。** 许多实现只测试 depth=0.5。测试 depth=0, 0.25, 0.5, 0.75, 1.0——"中间丢失"效应是真实存在的。
- **与填充文本的词汇重叠。** 如果针与填充文本共享关键词，检索就变得微不足道。使用 NoLiMa 风格的无重叠针。
- **忽略延迟。** 100 万 token 的提示需要 30-120 秒预填充。同时测量首次 token 时间和准确率。
- **厂商自报数据。** OpenAI、Google、Anthropic 都发布自己的得分。务必在你的用例上独立重新运行。

## 使用指南

2026 年的技术栈：

| 场景 | 基准测试 |
|------|---------|
| 快速健全性检查 | 自定义 NIAH，3 深度 × 3 长度 |
| 生产模型选择 | RULER（13 种任务），在你的目标长度上运行 |
| 真实世界问答质量 | LongBench v2 单文档问答子集 |
| 多跳推理 | BABILong 或自定义变量追踪 |
| 对话/聊天 | MRCR 8 针，在你的目标长度上运行 |
| 模型升级回归测试 | 固定的内部 NIAH + RULER 测试套件，每个新模型都运行 |

生产环境经验法则：在你打算使用的长度上，至少通过 NIAH + 1 个推理任务，才能信任一个上下文窗口。

## 交付产出

保存为 `outputs/skill-long-context-eval.md`：

```markdown
---
name: long-context-eval
description: 为给定模型和用例设计长上下文评估测试套件。
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

给定目标模型、目标上下文长度和用例，输出：

1. 测试。NIAH 深度 × 长度网格；RULER 多跳；自定义领域任务。
2. 采样。每个长度的深度 0, 0.25, 0.5, 0.75, 1.0。
3. 指标。检索通过率；推理通过率；首次 token 时间；每查询成本。
4. 截止点。有效检索长度（90% 通过）和有效推理长度（70% 通过）。两者都要报告。
5. 回归。固定测试套件，每次模型升级都重新运行，呈现差异。

拒绝仅从模型卡片信任上下文窗口。拒绝对任何多跳工作负载仅用 NIAH 评估。拒绝将厂商自报的长上下文得分作为独立证据。
```

## 练习

1. **简单。** 构建一个 NIAH，3 深度（0.25, 0.5, 0.75）× 3 长度（1k, 4k, 16k）。在任意模型上运行。将通过率绘制为 3×3 热力图。
2. **中等。** 添加一个 3 针变体。在每个长度上测量所有 3 针的检索率。与相同长度的单针通过率对比。
3. **困难。** 构建一个变量追踪任务（X1 → X2 → X3，3 跳），嵌入 64k 填充文本中。在 3 个前沿模型上测量准确率。报告每个模型的有效推理长度。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| NIAH | 大海捞针 | 在填充文本中植入一个事实，要求模型检索。 |
| RULER | 加强版 NIAH | 4 个类别、13 种任务类型：检索/多跳/聚合/问答。 |
| 有效上下文 | 真实容量 | 准确率仍高于阈值的长度。 |
| 中间丢失 | 深度偏差 | 模型对长输入中间位置的内容注意力不足。 |
| 多针 | 同时多个事实 | 多个植入；测试注意力协调能力，而非单纯检索。 |
| MRCR | 多轮共指 | 8、24 或 100 针共指；暴露注意力饱和。 |
| NoLiMa | 非词汇针 | 针和查询没有字面重叠的 token；需要推理。 |

## 延伸阅读

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — 原始 NIAH 仓库。
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — 多任务基准。
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — 真实世界长上下文评估。
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — 更难的针。
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — 干草堆中的推理。
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — 深度偏差论文。
