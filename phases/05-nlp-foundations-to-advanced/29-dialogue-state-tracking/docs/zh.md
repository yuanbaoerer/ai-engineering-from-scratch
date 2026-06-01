# 对话状态跟踪（Dialogue State Tracking / DST）

> "我想要一家北部的便宜餐厅......改成中等价位的......再来个意大利菜。"三个话轮（Turn），三次状态更新。DST 保持槽（Slot）-值字典同步，确保预订顺利完成。

**类型：** 构建
**语言：** Python
**前置知识：** 第5阶段 · 17（聊天机器人）、第5阶段 · 20（结构化输出（Structured Output））
**时间：** 约75分钟

## 问题

在面向任务的对话系统中，用户的目标被编码为一组槽-值对：`{cuisine: italian, area: north, price: moderate}`。每个用户话轮都可能添加、更改或删除一个槽。系统必须读取整个对话并正确输出当前状态。

搞错一个槽，系统就会预订错误的餐厅、安排错误的航班或扣错信用卡。DST 是连接用户表述和后端执行的枢纽。

为什么在 2026 年 LLM 时代它仍然重要：

- 合规敏感领域（银行、医疗、航空预订）需要确定性的槽值，而非自由生成。
- 工具调用代理在调用 API 之前仍然需要槽解析。
- 多轮纠正（Correction）比看起来更难："不对，改成周四。"

现代流程：经典 DST 概念 + LLM 提取器 + 结构化输出护栏。

## 概念

![DST：对话历史 → 槽-值状态](../assets/dst.svg)

**任务结构。** 模式（Schema）定义了域（Domain）（餐厅、酒店、出租车）及其槽（菜系、区域、价格、人数）。每个槽可以为空、从封闭集合中填充值（价格：{cheap, moderate, expensive}），或填入自由形式的值（名称："The Copper Kettle"）。

**两种 DST 公式化方法。**

- **分类。** 对每个（槽，候选值）对，预测是/否。适用于封闭词汇槽。2020 年前的标准方法。
- **生成。** 给定对话，以自由文本形式生成槽值。适用于开放词汇槽。现代默认方法。

**评估指标。** 联合目标准确率（Joint Goal Accuracy / JGA）——每个槽都正确的话轮比例。全有或全无。2026 年 MultiWOZ 2.4 排行榜最高约 83%。

**架构。**

1. **基于规则（槽正则表达式（Regex） + 关键词）。** 针对窄域的强基线。可调试。
2. **TripPy / BERT-DST。** 基于 BERT 编码的复制式生成。LLM 前的标准。
3. **LDST（LLaMA + LoRA）。** 指令微调 LLM，带域-槽提示。在 MultiWOZ 2.4 上达到 ChatGPT 级别质量。
4. **无本体（2024–26）。** 跳过模式；直接生成槽名和值。支持开放域。
5. **提示 + 结构化输出（2024–26）。** LLM 配合 Pydantic 模式 + 约束解码。5 行代码，生产可用。

### 经典失败模式

- **跨话轮共指消解（Coreference Resolution）。** "就选第一个吧。"需要解析指的是哪个选项。
- **覆盖（Override） vs 追加（Append）。** 用户说"加上意大利菜"。你是替换菜系还是追加？
- **隐式确认（Implicit Confirmation）。** "好的"——这是接受了系统提供的预订吗？
- **纠正。** "改成晚上 7 点。"必须更新时间而不清除其他槽。
- **对先前系统话语的共指。** "对，就那个。"哪个"那个"？

## 构建

### 步骤 1：基于规则的槽提取器

参见 `code/main.py`。正则表达式 + 同义词字典覆盖窄域中 70% 的典型话语：

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

在典型词汇之外很脆弱。适用于确定性槽确认。

### 步骤 2：状态更新循环

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

三个不变量：

- 永远不要重置用户未涉及的槽。
- 显式否定（"算了，不要菜系了"）必须清除。
- 用户纠正（"不对......"）必须覆盖，而非追加。

### 步骤 3：LLM 驱动的结构化输出 DST

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic 保证输出有效的状态对象。没有正则表达式，没有模式不匹配，没有幻觉槽。

### 步骤 4：JGA 评估

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

校准：系统在多少比例的话轮中正确预测了所有槽？对于 MultiWOZ 2.4，2026 年顶级系统：80-83%。你的领域内系统应该在窄词汇上超过这个水平，否则 LLM 基线会击败你。

### 步骤 5：处理纠正

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

检测到纠正时，覆盖最近更新的槽而非追加。没有 LLM 帮助很难做好。现代模式：总是让 LLM 根据历史重新生成整个状态，而不是增量更新——这自然地处理了纠正。

## 常见陷阱

- **全历史重新生成的成本。** 每个话轮让 LLM 重新生成状态，总 token 开销为 O(n²)。限制历史长度或总结较早的话轮。
- **模式漂移。** 事后添加新槽会破坏旧的训练数据。为你的模式加上版本号。
- **大小写敏感。** "Italian" vs "italian" vs "ITALIAN"——到处都要规范化。
- **隐式继承。** 如果用户之前指定了"4 个人"，对不同时间的新请求不应清除人数。始终传递完整历史。
- **自由形式 vs 封闭集合。** 名称、时间和地址需要自由形式的槽；菜系和区域是封闭的。在模式中混合使用两者。

## 应用

2026 年技术栈：

| 场景 | 方法 |
|------|------|
| 窄域（一两个意图） | 基于规则 + 正则表达式 |
| 宽域，有标注数据 | LDST（LLaMA + LoRA 在 MultiWOZ 类数据上微调） |
| 宽域，无标注，生产就绪 | LLM + Instructor + Pydantic 模式 |
| 语音 | ASR + 规范化 + LLM-DST |
| 多域预订流程 | 模式引导的 LLM，每域一个 Pydantic 模型 |
| 合规敏感 | 基于规则为主，LLM 兜底并带确认流程 |

## 交付

保存为 `outputs/skill-dst-designer.md`：

```markdown
---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.
```

## 练习

1. **简单。** 在 `code/main.py` 中为 3 个槽（菜系、区域、价格）构建基于规则的状态跟踪器。在 10 个手工构建的对话上测试。测量 JGA。
2. **中等。** 同一数据集，使用 Instructor + Pydantic + 小型 LLM。比较 JGA。检查最难的话轮。
3. **困难。** 实现两者并路由：基于规则为主，当规则方法输出 <2 个置信槽时回退到 LLM。测量组合 JGA 和每轮推理成本。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| DST | 对话状态跟踪 | 跨对话话轮维护槽-值字典。 |
| 槽（Slot） | 用户意图单元 | 后端需要的命名参数（菜系、日期）。 |
| 域（Domain） | 任务领域 | 餐厅、酒店、出租车——槽的集合。 |
| JGA | 联合目标准确率 | 每个槽都正确的话轮比例。全有或全无。 |
| MultiWOZ | 基准数据集 | 多域 WOZ 数据集；标准 DST 评估。 |
| 无本体 DST | 无模式 | 直接生成槽名和值，无固定列表。 |
| 纠正（Correction） | "不对......" | 覆盖先前已填槽的话轮。 |

## 延伸阅读

- [Budzianowski 等人 (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — 权威基准。
- [Feng 等人 (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — LLaMA + LoRA 指令微调用于 DST。
- [Heck 等人 (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — 基于复制的 DST 主力方法。
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) — 基于 EM 的无监督任务导向对话。
- [MultiWOZ 排行榜](https://github.com/budzianowski/multiwoz) — 权威 DST 结果。
