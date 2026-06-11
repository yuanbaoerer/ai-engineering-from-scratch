# 工具接口 — 为什么 Agent 需要结构化 I/O

> 语言模型生成的是 token。程序执行的是动作。二者之间的缺口，就是工具接口（tool interface）：一份让模型请求动作、让宿主执行动作的契约。每一种 2026 年的技术栈 — OpenAI、Anthropic 和 Gemini 上的 function calling；MCP 的 `tools/call`；A2A 的 task parts — 都是在用不同编码表达同一个四步循环。本课会命名这个循环，并展示运行它所需的最小机制。

**类型：** 学习
**语言：** Python（stdlib，无 LLM）
**先修要求：** 第 11 阶段（LLM completion APIs）
**时间：** 约 45 分钟

## 学习目标

- 解释为什么一个只能生成文本的 LLM，无法独自对真实世界采取行动。
- 画出四步工具调用循环（描述 → 决策 → 执行 → 观察），并说明每一步由谁负责。
- 将工具描述写成三部分：名称、JSON Schema 输入，以及确定性的执行器函数。
- 区分纯工具和有副作用的工具，并说明为什么这种划分对安全很重要。

## 问题

LLM 输出的是下一个 token 的概率分布。这就是它的全部输出表面。如果你问聊天模型“班加罗尔现在天气怎么样”，它可以写出一句看起来合理的话，但它无法拨打天气 API。这句话可能碰巧正确，也可能已经过时三天。

弥合这个缺口，就是工具接口的目的。宿主程序 — 你的 agent runtime、Claude Desktop、ChatGPT、Cursor，或一个自定义脚本 — 会向模型公布一组可调用工具。当模型判断需要一个动作时，它会发出一个结构化载荷，写明工具名称和参数。宿主解析这个载荷，真正运行工具，再把结果反馈回来。这个循环会持续进行，直到模型判断不再需要更多调用。

这份契约的第一个版本于 2023 年 6 月以 OpenAI 的 “functions” 参数形式发布。Anthropic 随后在 Claude 2.1 中加入了 `tool_use` blocks。Gemini 几个月后加入了 `functionDeclarations`。现在每个 provider 都暴露同样的形状：输入是一份由 JSON Schema 标注类型的工具列表，输出是一个 JSON 载荷形式的工具调用。Model Context Protocol（2024 年 11 月）把这份契约泛化为一个工具注册表服务于每个模型。A2A（2026 年 4 月，v1.0）又把同一个原语叠加到 agent-to-agent 委派上。

四步循环是所有这些机制底下的不变量。第 13 阶段的其余内容，都是对它的展开。

## 概念

### 第一步：描述

宿主用三个字段声明每个工具。

- **名称。** 稳定、机器可读的标识符。用 `get_weather`，而不是 “weather thing”。
- **描述。** 一段自然语言简介。“当用户询问某个具体城市的当前天气状况时使用。不要用于历史数据。”
- **输入 schema。** 一个 JSON Schema 对象（draft 2020-12），描述工具的参数。

模型会收到这份列表。现代 provider 会用各自特定的模板，把这些声明序列化到 system prompt 中，所以作为调用方，你只需要处理结构化形式。

### 第二步：决策

给定用户消息和可用工具，模型会选择三种行为之一。

1. **直接用文本回答**。不调用工具。
2. **调用一个或多个工具。** 发出结构化调用对象。在 `parallel_tool_calls: true` 下（OpenAI 和 Gemini 默认开启，Anthropic 需要选择启用），模型可以在一个 turn 中发出多个调用。
3. **拒绝。** 严格模式的结构化输出可以产生一个带类型的 `refusal` block，而不是调用。

工具调用载荷有三个稳定字段：调用 `id`、工具 `name`，以及 JSON `arguments` 对象。id 的存在是为了让宿主能把后续结果和具体调用关联起来；当并行调用乱序返回时，这一点很重要。

### 第三步：执行

宿主收到调用后，会根据声明的 schema 验证参数，然后运行执行器。无效参数意味着模型幻觉出了某个字段，或使用了错误类型 — 这是弱模型上非常常见的失败模式。生产级宿主在遇到无效参数时通常会做三件事之一：快速失败并把错误暴露给模型，用受约束的解析器修复 JSON，或在 prompt 中包含验证错误后重试模型。

执行器本身就是普通代码。Python、TypeScript、shell 命令、数据库查询都可以。它会产生一个结果，通常是字符串，但也可以是任意 JSON 值，或结构化内容块（在 MCP 中可以是文本、图片或资源引用）。结果必须可序列化。

### 第四步：观察

宿主把工具结果追加到对话中（作为带匹配 `id` 的 `tool` role message），然后再次调用模型。模型现在在上下文中拥有工具输出，可以生成最终答案，也可以请求更多调用。这个过程会持续到模型停止发出调用，或宿主达到迭代次数的安全上限。

### 信任边界划分

从安全角度看，工具分为两类。

- **纯工具。** 只读、确定性、无副作用。`get_weather`、`search_docs`、`get_current_time`。可以安全地进行推测性调用。
- **后果性工具。** 会改变状态、花钱、触碰用户数据。`send_email`、`delete_file`、`execute_trade`。必须加闸门。

Meta 在 2026 年提出的 agent 安全 “Rule of Two” 规定，一个 turn 最多只能同时组合以下三者中的两者：不受信任的输入、敏感数据、后果性动作。工具接口正是你执行这条规则的地方 — 通过拒绝调用、要求用户确认，或提升作用域。完整安全章节见第 13 阶段 · 15；agent 级权限策略见第 14 阶段 · 09。

### 循环存在于哪里

| Context | Who describes | Who decides | Who executes |
|---------|---------------|-------------|--------------|
| Single-turn function calling (OpenAI/Anthropic/Gemini) | App developer | LLM | App developer |
| MCP | MCP server | LLM via MCP client | MCP server |
| A2A | Agent Card publisher | Calling agent | Called agent |
| Web browser (function-calling agent) | Browser extension / WebMCP | LLM | Browser runtime |

无论在哪里，都是相同的四步。列名会变；结构不会变。

### 为什么不只是 prompt 模型输出 JSON？

“要求模型用 JSON 回复”是 function calling 之前的模式。它在前沿模型上大约有 5% 到 15% 的失败率，在更小模型上失败率更高。失败模式包括缺少花括号、尾随逗号、幻觉字段和错误类型。然后你就需要一次 JSON 修复、一次重试，或一个受约束的解码器。

原生 function calling 更好，原因有三点。第一，provider 会围绕确切调用形状对模型做端到端训练，所以严格模式下有效 JSON 比例会上升到 98% 到 99%。第二，调用载荷位于自己的协议槽位中，而不是自由文本内部 — 因此工具调用永远不会泄漏到用户可见的回复中。第三，provider 会用受约束解码强制 schema 合规（OpenAI 的 strict mode、Anthropic 的 `tool_use`、Gemini 的 `responseSchema`）。输出保证可以通过验证。

第 13 阶段 · 02 会并排讲解三个 provider API。第 13 阶段 · 04 会深入讲结构化输出。

### 熔断器

当模型停止发出调用，或宿主达到最大 turn 数时，循环终止。生产级宿主通常把这个上限设置在 5 到 20 个 turn 之间。超过这个范围，你几乎肯定已经进入了模型无法自行退出的循环。Claude Code 默认 20；OpenAI Assistants 默认 10；Cursor 的 agent mode 默认 25。

另一种选择 — 无界循环 — 每六个月就会以“agent 一夜之间花掉 400 美元 API 调用费”的事后复盘形式出现一次。没有边界，不要上线。

第 14 阶段 · 12 会深入讲错误恢复和自愈；第 17 阶段会讲生产级速率限制。

### 第 13 阶段接下来走向哪里

- 第 02 到 05 课会打磨 provider 级工具调用表面。
- 第 06 到 14 课会把这个循环泛化到 MCP。
- 第 15 到 18 课会防御恶意服务器、对抗性用户，以及未经认证的远程 auth 表面。
- 第 19 到 22 课会把这个模式扩展到 agent-to-agent 协作、可观测性、路由和打包。
- 第 23 课会发布一个使用每个原语的完整生态系统。

其余每一课都是对这个四步循环的展开。请把它作为不变量记在心里。

## 使用它

`code/main.py` 在没有 LLM 的情况下运行四步循环。一个假的“决策器”函数通过对用户消息做模式匹配来模拟模型；执行器、schema 验证器和观察步骤的 harness 都是真实的。运行它，查看完整的请求/响应编排以及可打印的中间状态；然后在后续课程中，把假的决策器替换为任意真实 provider。

需要关注：

- 工具注册表为每个工具保存三个字段：名称、描述、schema，以及一个执行器引用。
- 验证器是一个最小 JSON Schema 子集（types、required、enum、min/max），只用 stdlib 编写。第 13 阶段 · 04 会提供更完整的版本。
- 循环把迭代次数限制在五次。生产级 agent 正需要这种熔断器。

## 交付它

本课会产出 `outputs/skill-tool-interface-reviewer.md`。给定一份工具定义草案（名称 + 描述 + schema + 执行器大纲），这个 skill 会审计它是否适合该循环：名称是否机器稳定，描述是否是一份完整的使用简介，schema 是否正确使用 JSON Schema 2020-12，以及纯工具 vs 后果性工具的分类是否明确。

## 练习

1. 给 `code/main.py` 添加第四个工具，名为 `get_stock_price(ticker)`。把它的描述写成 “Use when the user asks for a current stock price by ticker. Do not use for historical prices or market summaries.” 运行 harness，并确认假的决策器会把提到 ticker 的查询路由到新工具。

2. 破坏 schema 验证器。传入一个 `arguments` 对象缺少必需字段的调用，并确认宿主会在执行前拒绝它。然后传入一个带有额外未知字段的调用。做出决定：宿主应该拒绝还是忽略？用安全论证说明你的选择。

3. 将 harness 中的每个工具分类为纯工具或后果性工具。给需要的注册表条目添加 `consequential: true` 标志，并修改循环，使其在选择后果性工具时打印一行 “would confirm with user”。这是每个生产级宿主都需要的确认闸门形状。

4. 在纸上画出四步循环，并用上面的 provider 列表格，为你最喜欢的客户端（Claude Desktop、Cursor、ChatGPT，或自定义栈）填写进去。与第 13 阶段 · 06 中 MCP 特定的变体交叉参照。

5. 从头到尾阅读 OpenAI 的 function-calling guide。找出一个位于请求中、但不在本文呈现的四步循环中的字段。解释它增加了什么，以及为什么它方便但不是必要。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Tool | “模型可以调用的东西” | 名称 + JSON-Schema 标注类型的输入 + 执行器函数三元组 |
| Function calling | “原生工具使用” | provider 级 API 支持发出结构化工具调用，而不是散文 |
| Tool call | “模型发出的行动请求” | 模型发出的带有 `id`、`name`、`arguments` 的 JSON 载荷 |
| Tool result | “工具返回的内容” | 执行器的输出，被包裹在带匹配 id 的 `tool` role message 中 |
| Parallel tool calls | “一次多个调用” | 一个模型 turn 中的多个调用对象，彼此独立，并可按 id 排序 |
| Strict mode | “保证 JSON” | 受约束解码，强制模型输出根据声明的 schema 通过验证 |
| Pure tool | “只读工具” | 无副作用；可以安全重跑 |
| Consequential tool | “动作工具” | 改变外部状态；需要闸门、审计或用户确认 |
| Four-step loop | “工具调用周期” | 描述 → 决策 → 执行 → 观察 |
| Host | “Agent runtime” | 持有工具注册表、调用模型并运行执行器的程序 |

## 延伸阅读

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — OpenAI 风格工具声明和调用形状的权威参考
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — Claude 的 `tool_use` / `tool_result` block 格式
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini 中的 `functionDeclarations` 和并行调用语义
- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 工具接口与 provider 无关的泛化
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — 每个现代工具 API 都使用的 schema 方言
