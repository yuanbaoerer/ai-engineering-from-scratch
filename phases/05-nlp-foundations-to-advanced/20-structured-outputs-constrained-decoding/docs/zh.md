# 结构化输出与约束解码

> 让 LLM 返回 JSON，大多数时候能得到 JSON。在生产环境中，"大多数"就是问题所在。约束解码通过在采样前编辑 logits，将"大多数"变成"始终"。

**类型：** 构建
**语言：** Python
**前置课程：** 第 5 阶段 · 第 17 课（聊天机器人），第 5 阶段 · 第 19 课（子词分词）
**时间：** 约 60 分钟

## 问题

一个分类器向 LLM 发出提示："返回 {positive, negative, neutral} 中的一个。"模型返回："The sentiment is positive — this review is overwhelmingly favorable because the customer explicitly states that they ..."。你的解析器崩溃了。你分类器的 F1 分数为 0.0。

自由形式生成不是契约，而是一种建议。生产系统需要的是契约。

2026 年存在三个层次。

1. **提示工程。** 好好说。"只返回 JSON 对象。"在前沿模型上约 80% 的情况下有效，在较小的模型上效果更差。
2. **原生结构化输出 API。** OpenAI `response_format`、Anthropic 工具调用、Gemini JSON 模式。在支持的模式上可靠。供应商锁定。
3. **约束解码。** 在每个生成步骤修改 logits，使模型*无法*输出无效 token。从构造上保证 100% 有效。适用于任何本地模型。

本课为这三种方式建立直觉，并说明何时该选用哪种。

## 概念

![约束解码在每一步屏蔽无效 token](../assets/constrained-decoding.svg)

**约束解码的工作原理。** 在每个生成步骤中，LLM 在整个词表（约 10 万个 token）上生成一个 logit 向量。一个*logit 处理器*位于模型和采样器之间。它根据当前在目标语法（JSON Schema、正则表达式、上下文无关文法）中的位置计算哪些 token 是有效的，并将所有无效 token 的 logit 设为负无穷。对剩余 logit 的 softmax 只在有效续写上分配概率质量。

2026 年的实现方式：

- **Outlines。** 将 JSON Schema 或正则表达式编译为有限状态机。每个 token 都有 O(1) 的有效下一 token 查询。基于 FSM，因此递归 schema 需要展平。
- **XGrammar / llguidance。** 上下文无关文法引擎。处理递归 JSON Schema。解码开销接近于零。OpenAI 在其 2025 年结构化输出实现中引用了 llguidance。
- **vLLM 引导解码。** 内置 `guided_json`、`guided_regex`、`guided_choice`、`guided_grammar`，通过 Outlines、XGrammar 或 lm-format-enforcer 后端实现。
- **Instructor。** 基于 Pydantic 的任意 LLM 封装器。在验证失败时重试。跨供应商，但不修改 logits——它依赖重试 + 结构化输出感知提示。

### 反直觉的结果

约束解码通常比无约束生成*更快*。有两个原因。首先，它缩小了下一个 token 的搜索空间。其次，巧妙的实现会为强制 token（如 `{"name": "` — 每个字节都是确定的）跳过 token 生成。

### 会让你付出代价的陷阱

字段顺序很重要。把 `answer` 放在 `reasoning` 之前，模型会在思考之前就确定答案。JSON 是有效的，但答案是错误的。没有任何验证能捕获这个问题。

```json
// 不好
{"answer": "yes", "reasoning": "because ..."}

// 好
{"reasoning": "... therefore ...", "answer": "yes"}
```

Schema 字段顺序是逻辑问题，不是格式问题。

## 动手构建

### 第 1 步：从零实现正则约束生成

参见 `code/main.py` 获取独立的 FSM 实现。核心思想用 30 行代码即可表达：

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM 跟踪到目前为止已经满足了语法的哪些部分。`valid_tokens(state, tokenizer)` 计算哪些词表 token 可以推进 FSM 而不离开可接受路径。

### 第 2 步：使用 Outlines 处理 JSON Schema

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

零验证错误。永远不会出错。FSM 使无效输出不可达。

### 第 3 步：使用 Instructor 实现供应商无关的 Pydantic

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

机制不同。Instructor 不触及 logits。它将 schema 格式化到提示中，解析输出，并在验证失败时重试（默认 3 次）。适用于任何供应商。重试会增加延迟和成本。跨供应商可移植性是其卖点。

### 第 4 步：使用原生供应商 API

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

服务端约束解码。对于支持的 schema，可靠性与 Outlines 持平。无需本地模型管理。但会锁定到该供应商。

## 陷阱

- **递归 schema。** Outlines 将递归展平为固定深度。树形结构输出（嵌套评论、AST）需要 XGrammar 或 llguidance（基于 CFG）。
- **大型枚举。** 10,000 个选项的枚举编译缓慢或超时。切换到检索器：先预测 top-k 候选项，再约束到这些候选项。
- **语法过于严格。** 强制 `date: "YYYY-MM-DD"` 正则表达式，模型就无法对缺失的日期输出 `"unknown"`。模型会通过编造日期来补偿。允许 `null` 或哨兵值。
- **过早确定。** 参见上面的字段顺序陷阱。始终将推理放在前面。
- **供应商 JSON 模式不带 schema。** 纯 JSON 模式只保证有效的 JSON，不保证对你的用例*有效*。始终提供完整的 schema。

## 使用指南

2026 年技术栈：

| 场景 | 选择 |
|------|------|
| OpenAI/Anthropic/Google 模型，简单 schema | 原生供应商结构化输出 |
| 任意供应商，Pydantic 工作流，可容忍重试 | Instructor |
| 本地模型，需要 100% 有效性，扁平 schema | Outlines (FSM) |
| 本地模型，递归 schema | XGrammar 或 llguidance |
| 自托管推理服务器 | vLLM 引导解码 |
| 可接受重试的批处理 | Instructor + 最便宜的模型 |

## 交付

保存为 `outputs/skill-structured-output-picker.md`：

```markdown
---
name: structured-output-picker
description: Choose a structured output approach, schema design, and validation plan.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

Given a use case (provider, latency budget, schema complexity, failure tolerance), output:

1. Mechanism. Native vendor structured output, Instructor retries, Outlines FSM, or XGrammar CFG. One-sentence reason.
2. Schema design. Field order (reasoning first, answer last), nullable fields for "unknown", enum vs regex, required fields.
3. Failure strategy. Max retries, fallback model, graceful `null` handling, out-of-distribution refusal.
4. Validation plan. Schema compliance rate (target 100%), semantic validity (LLM-judge), field-coverage rate, latency p50/p99.

Refuse any design that puts `answer` or `decision` before reasoning fields. Refuse to use bare JSON mode without a schema. Flag recursive schemas behind an FSM-only library.
```

## 练习

1. **简单。** 不使用约束解码，提示一个小型开源模型（如 Llama-3.2-3B）输出 `Review(sentiment, confidence, evidence_span)`。在 100 条评论上测量可解析为有效 JSON 的比例。
2. **中等。** 使用 Outlines JSON 模式处理相同的语料。比较合规率、延迟和语义准确性。
3. **困难。** 从零实现一个正则约束解码器，用于电话号码（`\d{3}-\d{3}-\d{4}`）。在 1000 个样本上验证 0 个无效输出。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 约束解码 (Constrained decoding) | 强制有效输出 | 在每个生成步骤屏蔽无效 token 的 logit。 |
| Logit 处理器 (Logit processor) | 约束的执行者 | 函数：`(logits, state) -> masked_logits`。 |
| FSM (有限状态机) | 有限状态机 | 编译后的语法表示；O(1) 有效下一 token 查询。 |
| CFG (上下文无关文法) | 上下文无关文法 | 处理递归的语法；比 FSM 慢但表达力更强。 |
| Schema 字段顺序 | 有关系吗？ | 有——第一个字段就确定了；始终将推理放在答案之前。 |
| 引导解码 (Guided decoding) | vLLM 的叫法 | 同一概念，集成到推理服务器中。 |
| JSON 模式 (JSON mode) | OpenAI 的早期版本 | 保证 JSON 语法；不保证 schema 匹配。 |

## 扩展阅读

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) — Outlines 论文。
- [XGrammar 论文 (2024)](https://arxiv.org/abs/2411.15100) — 基于 CFG 的快速约束解码。
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — 推理服务器集成。
- [OpenAI — Structured Outputs 指南](https://platform.openai.com/docs/guides/structured-outputs) — API 参考 + 注意事项。
- [Instructor 库](https://python.useinstructor.com/) — 跨供应商的 Pydantic + 重试机制。
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) — 6 个约束解码框架的基准测试。
