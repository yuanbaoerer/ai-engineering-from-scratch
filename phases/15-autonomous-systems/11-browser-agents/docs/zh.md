# 浏览器代理与长时域 Web 任务

> ChatGPT agent（2025 年 7 月）将 Operator 和 deep research 合并为一个浏览器/终端代理，在 BrowseComp 上达到 SOTA 68.9%。OpenAI 于 2025 年 8 月 31 日关闭了 Operator——产品层的整合。Anthropic 的 Vercept 收购将 Claude Sonnet 在 OSWorld 上从低于 15% 提升到 72.5%。WebArena-Verified（ServiceNow，ICLR 2026）修复了原始 WebArena 中 11.3 个百分点的假阴性率，并发布了 258 任务的 Hard 子集。数字是真实的。攻击面也是真实的：OpenAI 的 preparedness 负责人公开表示，对浏览器代理的间接提示注入"不是可以完全修补的漏洞"。记录的 2025-2026 年攻击：Tainted Memories（Atlas CSRF）、HashJack（Cato Networks）和 Perplexity Comet 中的一键劫持。

**类型：** 学习
**语言：** Python（标准库，间接提示注入攻击面模型）
**前置条件：** 第 15 阶段 · 10（权限模式），第 15 阶段 · 01（长时域代理）
**时间：** ~45 分钟

## 问题

浏览器代理是读取不受信任内容并采取重要动作的长时域代理。代理访问的每个页面都是用户未编写的输入。每个页面上的每个表单都是潜在的命令通道。2025-2026 年的攻击语料库表明这不是假设：Tainted Memories 让攻击者通过精心制作的页面将恶意指令绑定到代理的记忆；HashJack 在代理访问的 URL 片段中隐藏命令；Perplexity Comet 的一键劫持。

防御形势令人不安。OpenAI 的 preparedness 负责人说出了不该说的部分：间接提示注入"不是可以完全修补的漏洞"。这是因为攻击存在于代理的读取-行动边界，该边界在架构上是模糊的——模型读取的每个 token 原则上都可以被读作指令。

本课命名攻击面、命名基准全景（BrowseComp、OSWorld、WebArena-Verified），并建模最小间接提示注入场景，以便你能在第 14 和 18 课中推理真实防御。

## 概念

### 2026 年全景，每个系统一段话

**ChatGPT agent（OpenAI）。** 2025 年 7 月推出。统一了 Operator（浏览）和 Deep Research（多小时研究）。2025 年 8 月 31 日关闭独立 Operator。BrowseComp SOTA 68.9%；OSWorld 和 WebArena-Verified 上的强数字。

**Claude Sonnet + Vercept（Anthropic）。** Anthropic 的 Vercept 收购专注于计算机使用能力。将 Claude Sonnet 在 OSWorld 上从 <15% 提升到 72.5%。Claude Computer Use 作为工具 API 发布。

**Gemini 3 Pro with Browser Use（DeepMind）。** Browser Use 集成发布计算机使用控制；FSF v3（2026 年 4 月，第 20 课）专门跟踪 ML R&D 领域的自主性。

**WebArena-Verified（ServiceNow，ICLR 2026）。** 修复了一个记录良好的问题：原始 WebArena 有约 11.3% 的假阴性率（标记失败的任务实际已解决）。Verified 版本用人工策划的成功标准重新评分，并添加了 258 任务的 Hard 子集（ICLR 2026 论文，openreview.net/forum?id=94tlGxmqkN）。

### BrowseComp vs OSWorld vs WebArena

| 基准 | 测量什么 | 时域 |
|---|---|---|
| BrowseComp | 在时间压力下在开放网络上查找特定事实 | 分钟级 |
| OSWorld | 代理操作完整桌面（鼠标、键盘、shell） | 数十分钟 |
| WebArena-Verified | 模拟站点中的事务性 Web 任务 | 分钟级 |
| Hard 子集 | WebArena-Verified 中具有多页状态转换的任务 | 数十分钟 |

不同的轴。高 BrowseComp 分数说明代理能找到事实；它不能说明代理能预订航班。OSWorld 分数更接近"它在我的桌面上能工作吗"。WebArena-Verified 更接近"它能完成一个流程吗"。任何生产决策都需要匹配任务分布的基准。

### 攻击面，命名

1. **间接提示注入。** 不受信任的页面内容包含指令。代理读取它们。代理执行它们。公开示例：2024 年 Kai Greshake 等人、2025 年 Tainted Memories 论文、2026 年 HashJack（Cato Networks）。
2. **URL 片段/查询注入。** 爬取 URL 的 `#fragment` 或查询字符串包含命令。从不渲染可见；仍在代理上下文内。
3. **记忆绑定攻击。** 页面指示代理写入持久记忆（第 12 课涵盖持久状态）。下一次会话，记忆在没有可见触发器的情况下触发负载。
4. **对已认证会话的 CSRF 形状攻击。** Tainted Memories 类：代理在某处登录；攻击者的页面发出状态更改请求，代理用用户的 cookie 执行。
5. **一键劫持。** 视觉上无害的按钮承载代理跟随的负载。Comet 类。
6. **代理主机表面的内容安全策略漏洞。** 渲染和工具层本身可以是攻击向量；浏览器中的浏览器代理栈很宽。

### 为什么"不可完全修补"

攻击与代理的能力同构。代理必须读取不受信任的内容来完成工作。代理读取的任何内容都可能包含指令。代理遵循的任何指令都可能与用户的实际请求不对齐。防御（信任边界、分类器、工具白名单、重要动作的 HITL）提高了攻击成本并减少了其爆炸半径。它们不关闭该类别。

这与 Lob 定理（第 8 课）是相同的推理模式：代理无法证明下一个 token 是安全的；它只能建立一个系统，使不安全 token 更可检测。

### 实际部署的防御态势

- **读/写边界。** 读取永远不是决定性的。写入（提交表单、发布内容、调用有副作用的工具）如果发起内容来自信任边界外，需要新鲜人类批准。
- **每任务工具白名单。** 代理可以浏览；除非该工具已为任务明确启用，否则不能发起电汇。第 13 课涵盖预算。
- **会话隔离。** 浏览器代理会话仅使用范围化凭据运行。无生产认证，无个人邮箱。保留每个 HTTP 请求的日志用于审计。
- **内容清理器。** 获取的 HTML 在连接到模型上下文之前剥离已知有模式。（减少简单攻击；不阻止复杂负载。）
- **重要动作的 HITL。** 先提议后提交模式（第 15 课）。
- **记忆的金丝雀令牌。** 如果记忆条目触发，用户看到它（第 14 课）。

## 使用它

`code/main.py` 模拟一个微型浏览器代理对三个合成页面的运行。一个页面良性，一个在可见文本中有直接提示注入 blob，一个有 URL 片段注入（不可见但在代理上下文中）。脚本显示（a）朴素代理会做什么，（b）读/写边界捕获什么，（c）清理器捕获什么，（d）两者都不捕获什么。

## 交付它

`outputs/skill-browser-agent-trust-boundary.md` 限定拟议浏览器代理部署：它接触的信任区、它被授权写入什么、首次运行前必须有哪些防御。

## 练习

1. 运行 `code/main.py`。识别清理器捕获但读/写边界不捕获的攻击，以及只有读/写边界捕获的攻击。

2. 扩展清理器以检测一类 HashJack 风格的 URL 片段注入。在带有合法片段的良性 URL 上测量假阳性率。

3. 选择一个你了解的真实浏览器代理工作流（例如"预订航班"）。列出每次读和每次写。标记哪些写需要 HITL 以及为什么。

4. 阅读 WebArena-Verified ICLR 2026 论文。识别原始 WebArena 评分不可靠的一个任务类别，并解释 Verified 子集如何解决。

5. 为浏览器代理设置设计记忆金丝雀。你会存储什么、在哪里、什么触发警报？

## 关键术语

 | 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 间接提示注入 | "坏页面文本" | 代理读取的页面中的不受信任内容包含代理执行的指令 |
| Tainted Memories | "记忆攻击" | 代理将攻击者提供的指令写入持久记忆；下一次会话触发 |
| HashJack | "URL 片段攻击" | 隐藏在 URL 片段/查询字符串中的负载在代理上下文中但不可见渲染 |
| 一键劫持 | "坏按钮" | 可见的功能承载代理执行的后续负载 |
| BrowseComp | "Web 搜索基准" | 在开放网络上查找特定事实；分钟级时域 |
| OSWorld | "桌面基准" | 完整 OS 控制；多步 GUI 任务 |
| WebArena-Verified | "修复的 Web 任务基准" | ServiceNow 重新评分的 WebArena，带 Hard 子集 |
| 读/写边界 | "副作用门" | 读取永远不是决定性的；写入如果内容超出信任需要新鲜批准 |

## 延伸阅读

- [OpenAI — Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) — Operator 和 deep research 合并；BrowseComp SOTA。
- [OpenAI — Computer-Using Agent](https://openai.com/index/computer-using-agent/) — Operator 血统和成为 ChatGPT agent 的架构。
- [Zhou 等人 — WebArena](https://webarena.dev/) — 原始基准。
- [WebArena-Verified（OpenReview）](https://openreview.net/forum?id=94tlGxmqkN) — ICLR 2026 修复子集论文。
- [Anthropic — 实践中测量代理自主性](https://www.anthropic.com/research/measuring-agent-autonomy) — 包含计算机使用代理的攻击面讨论。
