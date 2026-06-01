# LLM 评估 — RAGAS、DeepEval、G-Eval

> Exact-match 和 F1 会漏掉语义等价的情况。人工审查无法规模化。LLM-as-judge 是生产环境的答案——但需要足够的校准才能信任这个数字。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 5 · 13（问答系统）、Phase 5 · 14（信息检索）
**时间：** 约 75 分钟

## 问题

你的 RAG 系统回答："2007 年 6 月 29 日。"
标准参考是："June 29, 2007."
Exact Match 得分 0。F1 得分约 75%。人类会打 100%。

现在乘以 10,000 个测试用例。再乘以检索器、分块、prompt 或模型的每次变更。你需要一个能理解语义、大规模廉价运行、不会对回归说谎、且能暴露正确失败模式的评估器。

2026 年有三个框架解决这个问题。

- **RAGAS。** Retrieval-Augmented Generation ASsessment。四个 RAG 指标（faithfulness、answer-relevance、context-precision、context-recall），使用 NLI + LLM-judge 后端。有研究支撑，轻量级。
- **DeepEval。** LLM 的 pytest。G-Eval、task-completion、hallucination、bias 指标。CI/CD 原生。
- **G-Eval。** 一种方法（也是 DeepEval 的一个指标）：带思维链的 LLM-as-judge，自定义评分标准，0-1 分。

三者都依赖 LLM-as-judge。本课构建对这种方法以及围绕它的信任层的直觉。

## 核心概念

![四个评估维度，LLM-as-judge 架构](../assets/llm-evaluation.svg)

**LLM-as-judge。** 用一个 LLM 根据评分标准为输出打分，替代静态指标。给定 `(query, context, answer)`，向 judge LLM 发送 prompt："在 faithfulness 上打 0-1 分。"返回分数。

为什么有效：LLM 以极小的成本近似人类判断。GPT-4o-mini 每个评分用例约 $0.003，使得 1000 个样本的回归评估运行成本低于 $5。

为什么它会静默失败：

1. **Judge 偏差。** Judge 偏好更长的回答、来自同一模型家族的回答、与 prompt 风格匹配的回答。
2. **JSON 解析失败。** 坏 JSON → NaN 分数 → 从聚合中静默排除。RAGAS 用户都体验过这种痛苦。用 try/except + 显式失败模式进行门控。
3. **跨模型版本漂移。** 升级 judge 会改变所有指标。冻结 judge 模型 + 版本。

**RAG 四项指标。**

| 指标 | 问题 | 后端 |
|------|------|------|
| Faithfulness | 回答中的每个声明是否来自检索到的上下文？ | 基于 NLI 的蕴含关系 |
| Answer relevance | 回答是否回应了问题？ | 从回答生成假设性问题；与真实问题比较 |
| Context precision | 在检索到的分块中，有多大比例是相关的？ | LLM-judge |
| Context recall | 检索是否返回了所有需要的内容？ | LLM-judge 对照标准答案 |

**G-Eval。** 定义一个自定义标准："回答是否引用了正确的来源？"框架会自动扩展为思维链评估步骤，然后打 0-1 分。适用于 RAGAS 未覆盖的领域特定质量维度。

**校准。** 在获得与人工标注的相关性之前，永远不要信任原始 judge 分数。运行 100 个手工标注的样本。绘制 judge vs 人类的对比图。计算 Spearman rho。如果 rho < 0.7，你的 judge 评分标准需要改进。

## 动手构建

### 步骤 1：使用 NLI 的 faithfulness（RAGAS 风格）

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` 是任意可调用对象：prompt str -> 生成的 str。
# 示例：llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

将回答分解为原子声明。对每个声明用 NLI 对照检索到的上下文进行检查。Faithfulness = 被支持的比例。

### 步骤 2：answer relevance

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: 任何实现 .encode(texts, normalize_embeddings=True) -> ndarray 的模型
# 例如，encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

如果回答暗示的问题与所问的问题不同，relevance 就会下降。

### 步骤 3：G-Eval 自定义指标

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

评估步骤就是评分标准。显式的步骤比隐式的"打 0-1 分"prompt 更稳定。

### 步骤 4：CI 门控

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

作为 pytest 文件发布。在每个 PR 上运行。回归时阻止合并。

### 步骤 5：从零构建简易评估

参见 `code/main.py`。使用纯标准库实现的 faithfulness（回答声明与上下文的重叠）和 relevance（回答 token 与问题 token 的重叠）近似。不是生产级的。展示的是结构。

## 常见陷阱

- **没有校准。** 与人工标注相关性只有 0.3 的 judge 就是噪声。在发布前要求进行校准运行。
- **自我评估。** 使用同一个 LLM 生成和评判会将分数膨胀 10-20%。judge 应使用不同的模型家族。
- **成对评判中的位置偏差。** Judge 偏好先展示的选项。始终随机化顺序并双向运行。
- **原始聚合隐藏失败。** 平均分 0.85 往往隐藏了 5% 的灾难性失败。始终检查底部百分位。
- **黄金数据集腐烂。** 随时间漂移的无版本评估集会破坏纵向对比。每次变更都标记数据集。
- **LLM 成本。** 大规模运行时，judge 调用主导成本。使用满足校准阈值的最便宜模型。GPT-4o-mini、Claude Haiku、Mistral-small。

## 使用场景

2026 年的技术栈：

| 使用场景 | 框架 |
|---------|------|
| RAG 质量监控 | RAGAS（4 个指标） |
| CI/CD 回归门控 | DeepEval + pytest |
| 自定义领域标准 | DeepEval 中的 G-Eval |
| 在线实时流量监控 | RAGAS 的无参考模式 |
| 人工在环抽检 | LangSmith 或 Phoenix（带标注 UI） |
| 红队测试 / 安全评估 | Promptfoo + DeepEval |

典型组合：RAGAS 用于监控，DeepEval 用于 CI，G-Eval 用于新维度。三者都运行；它们的分歧是有用的。

## 交付产出

保存为 `outputs/skill-eval-architect.md`：

```markdown
---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

## 练习

1. **简单。** 在 10 个已知存在幻觉的 RAG 示例上使用 RAGAS。验证 faithfulness 指标能捕获每一个。
2. **中等。** 手工标注 50 个 QA 回答的 0-1 正确性。用 G-Eval 打分。衡量 judge 与人类之间的 Spearman rho。
3. **困难。** 用 DeepEval 构建 pytest CI 门控。故意使检索器回归。验证门控会失败。通过对最低 10% 的阈值检查添加底部百分位告警。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| LLM-as-judge | "用 LLM 打分" | 向 judge 模型发送 prompt，根据评分标准为输出打 0-1 分 |
| RAGAS | "RAG 指标库" | 开源评估框架，包含 4 个无参考的 RAG 指标 |
| Faithfulness | "回答有依据吗？" | 回答中被检索上下文蕴含的声明比例 |
| Context precision | "检索到的分块相关吗？" | 实际有用的 top-K 分块比例 |
| Context recall | "检索找到了所有内容吗？" | 标准答案中被检索分块支持的声明比例 |
| G-Eval | "自定义 LLM judge" | 评分标准 + 思维链评估步骤 + 0-1 分 |
| Calibration | "信任但验证" | judge 分数与人类分数之间的 Spearman 相关性 |

## 延伸阅读

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — RAGAS 论文
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — G-Eval 论文
- [DeepEval 文档](https://deepeval.com/docs/metrics-introduction) — 开源生产级技术栈
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — 偏差、校准、局限性
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — 统一框架，集成 RAGAS、DeepEval、Phoenix
