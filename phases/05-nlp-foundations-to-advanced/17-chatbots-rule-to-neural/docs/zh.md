# 聊天机器人 -- 从规则到神经网络再到 LLM 智能体

> ELIZA 用模式匹配来回复。DialogFlow 映射意图。GPT 从权重中生成回答。Claude 运行工具并进行验证。每个时代都解决了上一个时代最严重的失败。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 5 · 13（问答系统），Phase 5 · 14（信息检索）
**时间：** 约 75 分钟

## 问题所在

用户说"我想改签航班"。系统需要弄清楚他们想要什么、缺少什么信息、如何获取这些信息，以及如何完成操作。然后用户说"等等，如果我取消呢？"，系统需要记住上下文、切换任务并保持状态。

对话对机器学习系统来说是困难的。输入是开放式的。输出必须在多个轮次中保持连贯。系统可能需要对现实世界采取行动（改签航班、扣款）。每一步错误都会被用户看到。

聊天机器人架构经历了四个范式的循环，每个范式的引入都是因为前一个范式的失败过于明显。本课按顺序讲解它们。2026 年的生产环境是后两种范式的混合体。

## 核心概念

![聊天机器人演进：规则式 → 检索式 → 神经网络 → 智能体](../assets/chatbot.svg)

**规则式（ELIZA、AIML、DialogFlow）。** 手工编写的模式匹配用户输入并生成回复。意图分类器（Intent Classifier）路由到预定义的流程。槽填充（Slot Filling）状态机收集所需信息。在其设计的狭窄范围内运行良好。超出范围立即失效。目前仍在安全关键领域（银行认证、机票预订）中使用，因为在这些领域不容忍幻觉。

**检索式。** 一种 FAQ 风格的系统。编码每一对（用户话语，回复）。在运行时，编码用户的消息并检索最接近的已存储回复。类似于 Zendesk 经典的"相似文章"功能。比规则式更好地处理改述。没有生成过程，因此不存在幻觉。

**神经网络（seq2seq）。** 在对话日志上训练的编码器-解码器（Encoder-Decoder）。从零开始生成回复。流畅但容易产生通用输出（"我不知道"）和事实漂移。从未可靠地保持话题。这是 Google、Facebook 和 Microsoft 在 2016-2019 年期间聊天机器人表现令人失望的原因。

**LLM 智能体。** 一个语言模型被包装在一个循环中，能够规划、调用工具并验证结果。不是一个带有长提示词的聊天机器人。而是一个智能体循环：规划 → 调用工具 → 观察结果 → 决定下一步。检索优先的接地（Grounding）策略（RAG）防止其产生幻觉。工具调用让它能够真正执行操作。这就是 2026 年的架构。

这四种范式并非依次替代的关系。2026 年的生产级聊天机器人会经过所有四种路径：规则式用于认证和破坏性操作，检索式用于 FAQ，神经网络生成用于自然表述，LLM 智能体用于模糊的开放式查询。

## 动手构建

### 第 1 步：基于规则的模式匹配

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

20 行代码实现的 ELIZA。反射技巧（"我感到难过" → "你为什么感到难过"）是 Weizenbaum 1966 年经典的心理治疗师演示。至今仍有教学意义。

### 第 2 步：检索式（FAQ）

此示例代码片段需要 `pip install sentence-transformers`（会拉取 torch）。本课的可运行代码 `code/main.py` 使用标准库的 Jaccard 相似度代替，因此本课无需外部依赖即可运行。

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

基于阈值的拒绝是关键的设计选择。如果最佳匹配不够接近，返回 `None` 并让系统升级处理。

### 第 3 步：神经网络生成（基线）

使用一个小型指令微调的编码器-解码器（FLAN-T5）或一个微调过的对话模型。在 2026 年单独使用在生产中不可行（矛盾、偏离主题、事实错误），但在混合系统中用于自然表述。DialoGPT 风格的仅解码器模型需要显式的轮次分隔符和 EOS 处理才能生成连贯的回复；FLAN-T5 的 text2text 管道可以开箱即用作为教学示例。

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### 第 4 步：LLM 智能体循环

2026 年生产环境的形态：

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

三个要点。工具是 LLM 可以调用的可执行函数。当 LLM 返回最终答案而非工具调用时，循环终止。步骤预算防止在模糊任务上陷入无限循环。

实际生产中还需添加：检索优先的接地策略（在每次 LLM 调用前注入相关文档）、防护栏（Guardrails，未经确认拒绝执行破坏性操作）、可观测性（Observability，记录每一步）、以及评估（自动检查智能体行为是否符合规范）。

### 第 5 步：混合路由

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

模式是：破坏性操作使用确定性规则，FAQ 使用检索，其他所有内容使用 LLM 智能体。这就是 2026 年客户支持系统的实际部署方式。

## 使用场景

2026 年的技术栈：

| 使用场景 | 架构 |
|---------|---------------|
| 预订、支付、认证 | 规则式状态机 + 槽填充 |
| 客户支持 FAQ | 基于精选答案的检索 |
| 开放式帮助对话 | 带 RAG + 工具调用的 LLM 智能体 |
| 内部工具 / IDE 助手 | 带工具调用（搜索、读取、写入）的 LLM 智能体 |
| 陪伴 / 角色聊天机器人 | 微调 LLM + 人格系统提示词 + 知识检索 |

生产环境中始终使用混合路由。没有单一架构能很好地处理每一种请求。路由层本身通常是一个小型意图分类器。

## 仍然会出现的失败模式

- **自信地编造。** LLM 智能体声称它完成了某个操作，但实际上并没有。缓解措施：验证结果、记录工具调用、绝不让 LLM 声称已完成某事而没有成功的工具返回。
- **提示词注入。** 用户插入覆盖系统提示词的文本。在 OWASP 2025 年 LLM 应用 Top 10 中排名第一（LLM01）。两种形式：直接注入（粘贴到聊天中）和间接注入（隐藏在智能体读取的文档、电子邮件或工具输出中）。

  攻击成功率因场景而异。在通用工具使用和编程基准测试中，前沿模型的实测成功率约为 0.5-8.5%。特定高风险设置（针对 AI 编程智能体的自适应攻击、脆弱的编排）已达到约 84%。生产环境中的 CVE 包括 EchoLeak（CVE-2025-32711，CVSS 9.3）—— Microsoft 365 Copilot 中的一个零点击数据窃取漏洞，由攻击者控制的电子邮件触发。

  缓解措施：在整个循环中将用户输入视为不可信；在工具调用前进行清理；将工具输出与主提示词隔离；使用"规划-验证-执行"（PVE）模式，智能体先规划，然后在执行前验证每个操作是否符合计划（这可以防止工具结果注入新的未计划操作）；对破坏性操作要求用户确认；对工具权限应用最小权限原则。

  再多的提示词工程也无法完全消除这一风险。需要外部运行时防御层（LLM Guard、白名单验证、语义异常检测）。
- **范围蔓延。** 智能体偏离任务，因为工具调用返回了与主题相关但偏离的信息。缓解措施：缩小工具契约；保持系统提示词聚焦；为偏离任务率添加评估。
- **无限循环。** 智能体不断调用同一工具。缓解措施：步骤预算、工具调用去重、LLM 评判"我们是否在取得进展"。
- **上下文窗口耗尽。** 长对话将最早的轮次推出上下文窗口。缓解措施：总结较早的轮次、通过相似度检索相关的历史轮次，或使用长上下文模型。

## 交付使用

保存为 `outputs/skill-chatbot-architect.md`：

```markdown
---
name: chatbot-architect
description: Design a chatbot stack for a given use case.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

Given a product context (user need, compliance constraints, available tools, data volume), output:

1. Architecture. Rule-based, retrieval, neural, LLM agent, or hybrid (specify which paths go where).
2. LLM choice if applicable. Name the model family (Claude, GPT-4, Llama-3.1, Mixtral). Match to tool-use quality and cost.
3. Grounding strategy. RAG sources, retrieval method (see lesson 14), tool contracts.
4. Evaluation plan. Task success rate, tool-call correctness, off-task rate, hallucination rate on held-out dialogs.

Refuse to recommend a pure-LLM agent for any destructive action (payments, account deletion, data modification) without a structured confirmation flow. Refuse to skip the prompt-injection audit if the agent has write access to anything.
```

## 练习

1. **简单。** 使用 10 个模式实现上述基于规则的回复，为咖啡店点单机器人服务。测试边界情况：重复下单、修改、取消、意图不明确。
2. **中等。** 构建一个混合 FAQ + LLM 后备系统。为一个 SaaS 产品准备 50 条 FAQ 条目，使用文档站点的检索作为 LLM 后备。在 100 个真实支持问题上测量拒绝率和准确率。
3. **困难。** 使用三个工具（搜索、读取用户数据、发送邮件）实现上述智能体循环。使用 50 个测试场景运行评估，包括提示词注入尝试。报告偏离任务率、失败任务率以及任何注入成功的情况。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 意图（Intent） | 用户想要什么 | 分类标签（book_flight、reset_password）。路由到处理器。 |
| 槽位（Slot） | 一条信息 | 机器人需要的参数（日期、目的地）。槽填充是连续询问的过程。 |
| RAG | 检索加生成 | 检索相关文档，然后将 LLM 的回复接地。 |
| 工具调用（Tool call） | 函数调用 | LLM 发出带有名称和参数的结构化调用。运行时执行并返回结果。 |
| 智能体循环（Agent loop） | 规划、行动、验证 | 控制器运行 LLM 调用并与工具调用交织，直到任务完成。 |
| 提示词注入（Prompt injection） | 用户攻击提示词 | 尝试覆盖系统提示词的恶意输入。 |

## 延伸阅读

- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — 原始的基于规则的聊天机器人论文。
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) — Google 在 LLM 智能体接管之前的神经聊天机器人论文。
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — 命名了智能体循环模式的论文。
- [Anthropic's guide on building effective agents](https://www.anthropic.com/research/building-effective-agents) — 2024 年的生产指南，在 2026 年仍然适用。
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) — 提示词注入论文。
- [OWASP Top 10 for LLM Applications 2025 — LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) — 将提示词注入列为首要安全威胁的排名。
- [AWS — Securing Amazon Bedrock Agents against Indirect Prompt Injections](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) — 实际的编排层防御，包括"规划-验证-执行"和用户确认流程。
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) — 间接提示词注入导致的典型零点击数据窃取 CVE。这是写入权限智能体需要运行时防御的参考案例。
