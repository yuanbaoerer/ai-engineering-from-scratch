# 提示词缓存与上下文缓存

> 你的系统提示词是 4,000 个 token。你的 RAG 上下文是 20,000 个 token。你每次请求都发送这两者，而且每次都要为两者付费。提示词缓存让提供商在他们的服务端保持这个前缀热数据，再次使用时只收取正常费用的 10%。使用得当，可以将推理成本降低 50%–90%，并将首 token 延迟降低 40%–85%。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 11 · 01（提示词工程）、阶段 11 · 05（上下文工程）、阶段 11 · 11（缓存与成本）
**时长：** 约 60 分钟

## 问题背景

一个编码智能体在每次对话轮次中都向 Claude 发送相同的 15,000 token 系统提示词。20 轮对话，按 $3/M 输入 token 计算，仅输入成本就达 $0.90 — 这还不包括用户的任何实际消息。乘以每日 10,000 次对话，账单就达到 $9,000/天，而这些都是从未改变的文本。

你无法在不损害质量的情况下缩小提示词。你也无法避免发送它 — 模型每轮都需要它。唯一的办法是不要再为提供商已经见过的前缀支付全额费用。

这就是提示词缓存。Anthropic 于 2024 年 8 月推出（2025 年推出了 1 小时扩展 TTL 变体），OpenAI 在同年晚些时候实现了自动化，Google 在 Gemini 1.5 旁边推出了显式上下文缓存，三家现在都在其前沿模型上将此作为一等特性提供。

## 核心概念

![提示词缓存：一次写入，多次廉价读取](../assets/prompt-caching.svg)

**原理。** 当一个请求的前缀与最近请求的前缀匹配时，提供商使用前一次运行的 KV-cache 来服务，而不是重新编码 token。第一次支付少量写入溢价，之后每次读取都享受大幅折扣。

**2026 年三大提供商的实现方式。**

| 提供商 | API 风格 | 命中折扣 | 写入溢价 | 默认 TTL | 最小可缓存 |
|--------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | 在内容块上使用显式 `cache_control` 标记 | 输入降低 90% | 溢价 25% | 5 分钟（可延长至 1 小时） | 1,024 token（Sonnet/Opus），2,048（Haiku） |
| OpenAI | 自动前缀检测 | 输入降低 50% | 无 | 最长 1 小时（尽力而为） | 1,024 token |
| Google（Gemini） | 显式 `CachedContent` API | 按存储计费；读取约正常价格的 25% | 按 token·小时计存储费 | 用户设置（默认 1 小时） | 4,096 token（Flash），32,768（Pro） |

**不变性原则。** 三家都只缓存前缀。如果请求之间有任何 token 不同，从第一个不同 token 开始之后全部未命中。把**稳定**部分放在顶部，**可变**部分放在底部。

### 缓存友好的布局

```
[系统提示词]          <-- 缓存这个
[工具定义]           <-- 缓存这个
[少样本示例]          <-- 缓存这个
[检索到的文档]        <-- 如果重用就缓存，否则不要
[对话历史]           <-- 缓存到上一轮
[当前用户消息]        <-- 永不缓存（每次都不同）
```

违反这个顺序 — 把用户消息放在系统提示词上面、在少样本之间插入动态检索 — 缓存就永远无法命中。

### 盈亏平衡计算

Anthropic 的 25% 写入溢价意味着一个缓存块必须被读取至少两次才能净省成本。1 次写入 + 1 次读取平均每次请求成本为 0.675 倍（节省 32%）；1 次写入 + 10 次读取平均每次为 0.205 倍（节省 80%）。经验法则：在 TTL 内预计至少重用 3 次的都要缓存。

## 构建实现

### 步骤 1：使用显式标记的 Anthropic 提示词缓存

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

`cache_control` 标记告诉 Anthropic 将该块存储 5 分钟。在这个时间窗口内重用会命中；过期后重用会重新写入。

**响应 usage 字段：**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # 按 1.25 倍计费
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # 按 0.1 倍计费
```

在 CI 中检查这两个字段 — 如果 `cache_read_input_tokens` 在多个请求中一直为零，说明你的缓存键在漂移。

### 步骤 2：一小时扩展 TTL

对于长时间运行的批处理作业，5 分钟默认值会在作业之间过期。设置 `ttl`：

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1 小时 TTL 的写入溢价是 2 倍（比基线高 50% 而不是 25%），但如果批处理在 5 次以上重用该前缀，就能快速回本。

### 步骤 3：OpenAI 自动缓存

OpenAI 没有任何可配置项。任何超过 1,024 token 且与最近请求匹配的前缀都会自动获得 50% 折扣。

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # 长且稳定
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # 折扣部分
```

同样的缓存友好布局规则适用。有两件事会杀死 OpenAI 的缓存但不会杀死 Anthropic 的：更改 `user` 字段（用作缓存键组件）和重新排序工具。

### 步骤 4：Gemini 显式上下文缓存

Gemini 将缓存视为一个一等对象，你需要创建并命名：

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Gemini 按 token·小时计收存储费，缓存存活期间都计费，读取约按正常输入价格的 25%。当你需要在数天内跨多个会话重用同一个巨大的提示词时，这是正确的形态。

### 步骤 5：在生产环境中测量命中率

参见 `code/main.py`，它有一个模拟的三提供商会计器，追踪写入/读取/未命中计数并计算每 1K 请求的混合成本。以目标命中率为部署门槛 — 大多数生产级 Anthropic 配置在预热后应看到 >80% 的读取比例。

## 2026 年仍然会出现的陷阱

- **动态时间戳在顶部。** 系统提示词顶部的 `"Current time: 2026-04-22 15:30:02"`。每次请求都未命中。把时间戳移到缓存断点之下。
- **工具重新排序。** 以稳定的顺序序列化工具 — 部署之间的字典重排会破坏每次命中。
- **自由文本近似重复。** "You are helpful." vs "You are a helpful assistant." — 一个字节的差异 = 完全未命中。
- **太小的块。** Anthropic 强制执行 1,024 token 的下限（Haiku 为 2,048）。更小的块静默地不会缓存。
- **盲目的成本仪表板。** 把"输入 token"拆分为缓存的和未缓存的。否则流量下降看起来就像缓存收益。

## 使用场景

2026 年缓存技术栈：

| 场景 | 选择 |
|-----------|------|
| 具有稳定 10k+ 系统提示词、多轮对话的智能体 | Anthropic `cache_control`，5 分钟 TTL |
| 重用前缀 30+ 分钟的批处理作业 | Anthropic `ttl: "1h"` |
| GPT-5 上的无服务器端点，无自定义基础设施 | OpenAI 自动（只需让你的前缀稳定且长） |
| 多天重用巨大的代码/文档语料库 | Gemini 显式 `CachedContent` |
| 跨提供商回退 | 保持可缓存前缀布局在提供商之间相同，这样任何命中都能生效 |

与语义缓存（阶段 11 · 11）结合用于用户消息层：提示词缓存处理**token 完全相同**的重用，语义缓存处理**语义相同**的重用。

## 上线检查清单

保存 `outputs/skill-prompt-caching-planner.md`：

```markdown
---
name: prompt-caching-planner
description: Design a cache-friendly prompt layout and pick the right provider caching mode.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

Given a prompt (system + tools + few-shot + retrieval + history + user) and a usage profile (requests per hour, TTL needed, provider), output:

1. Layout. Reordered sections with a single cache breakpoint marked; explain which sections are stable, which are volatile.
2. Provider mode. Anthropic cache_control, OpenAI automatic, or Gemini CachedContent. Justify from TTL and reuse pattern.
3. Break-even. Expected reads per write within TTL; net cost vs no-cache with math.
4. Verification plan. CI assertion that cache_read_input_tokens > 0 on the second identical request; dashboard split by cached vs uncached tokens.
5. Failure modes. List the three most likely reasons the cache will miss in this setup (dynamic timestamp, tool reorder, near-duplicate text) and how you will prevent each.

Refuse to ship a cache plan that places a dynamic field above the breakpoint. Refuse to enable 1h TTL without a reuse count that makes the 2x write premium pay back.
```

## 练习

1. **简单。** 用一个 5,000 token 的系统提示词对 Claude 进行 10 轮对话。先不用 `cache_control` 运行，然后使用。报告每次的输入 token 账单。
2. **中等。** 编写一个测试工具，给定一个提示词模板和一个请求日志，计算每个提供商（Anthropic 5 分钟、Anthropic 1 小时、OpenAI 自动、显式 Gemini）的预期命中率和成本节省。
3. **困难。** 构建一个布局优化器：给定一个提示词和一个标记了 `stable=True/False` 的字段列表，重写提示词将单个缓存断点放在最大缓存友好的位置而不丢失信息。在真实的 Anthropic 端点上验证。

## 关键术语

| 术语 | 人们通常说的 | 实际含义 |
|------|-----------------|-----------------------|
| 提示词缓存 | "让长提示词变便宜" | 重用提供商端的 KV-cache 以获得匹配的前缀；重复输入 token 享受 50-90% 折扣。 |
| `cache_control` | "Anthropic 的标记" | 内容块属性，声明"之前的所有内容都是可缓存的"；`{"type": "ephemeral"}`。 |
| 缓存写入 | "支付溢价" | 填充缓存的第一次请求；在 Anthropic 上按约 1.25 倍输入价格计费，OpenAI 上免费。 |
| 缓存读取 | "折扣" | 前缀匹配的后续请求；Anthropic 按 10% 计费，OpenAI 按 50%，Gemini 约 25%。 |
| TTL | "存活时间" | 缓存保持热数据的时间；Anthropic 默认 5 分钟（可延长至 1 小时），OpenAI 尽力而为最长 1 小时，Gemini 由用户设置。 |
| 扩展 TTL | "Anthropic 的 1 小时缓存" | `{"type": "ephemeral", "ttl": "1h"}`；写入溢价 2 倍，但对批处理重用值得。 |
| 前缀匹配 | "为什么我的缓存未命中" | 只有从开头到断点之间的每个 token 字节完全相同时，缓存才会命中。 |
| 上下文缓存（Gemini） | "显式的那种" | Google 的命名、按存储计费的缓存对象；适用于大型语料库的多天重用。 |

## 延伸阅读

- [Anthropic — 提示词缓存](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — `cache_control`、1 小时 TTL、盈亏平衡表。
- [OpenAI — 提示词缓存](https://platform.openai.com/docs/guides/prompt-caching) — 自动前缀匹配。
- [Google — 上下文缓存](https://ai.google.dev/gemini-api/docs/caching) — `CachedContent` API 和存储定价。
- [Anthropic 工程博客 — 面向长上下文工作负载的提示词缓存](https://www.anthropic.com/news/prompt-caching) — 原始发布文章，包含延迟数据。
- 阶段 11 · 05（上下文工程） — 如何切分提示词以使缓存能够落地。
- 阶段 11 · 11（缓存与成本） — 在用户消息上配对提示词缓存与语义缓存。
- [Pope 等，《高效扩展 Transformer 推理》(2022)](https://arxiv.org/abs/2211.05102) — 提示词缓存向用户暴露的 KV-cache 内存模型；解释为什么缓存前缀重新读取比重新计算便宜约 10 倍。
- [Agrawal 等，《SARATHI：通过_chunked prefill_背负解码实现高效 LLM 推理》(2023)](https://arxiv.org/abs/2308.16369) — prefill 是提示词缓存缩短的阶段；本文解释了为什么在缓存命中时 TTFT 大幅下降而 TPOT 不受影响。
- [Leviathan 等，《通过投机解码实现 Transformer 快速推理》(2023)](https://arxiv.org/abs/2211.17192) — 提示词缓存与投机解码、Flash Attention 和 MQA/GQA 并列，是弯曲推理成本曲线的杠杆；阅读本文以了解其他三个。