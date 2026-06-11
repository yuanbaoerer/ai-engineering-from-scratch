# Function Calling 深入解析 — OpenAI、Anthropic、Gemini

> 三家前沿模型提供商在 2024 年收敛到了同一种工具调用循环（tool-call loop），随后又在其他所有细节上分道扬镳。OpenAI 使用 `tools` 和 `tool_calls`。Anthropic 使用 `tool_use` 和 `tool_result` 块。Gemini 使用 `functionDeclarations` 和唯一 id 关联。本课将三者并排对比，这样你在一个提供商上上线的代码，迁移到另一个提供商时不会因为形状差异而崩掉。

**类型：** 构建
**语言：** Python（stdlib、schema translators）
**先修要求：** Phase 13 · 01（工具接口）
**时间：** 约 75 分钟

## 学习目标

- 说清 OpenAI、Anthropic 和 Gemini 函数调用载荷（function-calling payload）之间的三类形状差异（声明、调用、结果）。
- 将一个工具声明翻译到三种提供商格式，并预测严格模式（strict mode）约束会在哪里不同。
- 在每个提供商中使用 `tool_choice` 来强制、禁止或自动选择工具调用。
- 了解各提供商的硬限制（工具数量、schema 深度、参数长度）以及违反限制时各自发出的错误特征。

## 问题

函数调用请求的形状因提供商而异。下面是 2026 年生产栈中的三个具体例子：

**OpenAI Chat Completions / Responses API。** 你传入 `tools: [{type: "function", function: {name, description, parameters, strict}}]`。模型响应包含 `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`，其中 `arguments` 是必须由你解析的 JSON 字符串。严格模式（`strict: true`）通过受约束解码（constrained decoding）来强制 schema 合规。

**Anthropic Messages API。** 你传入 `tools: [{name, description, input_schema}]`。响应以 `content: [{type: "text"}, {type: "tool_use", id, name, input}]` 的形式返回。`input` 已经解析好（是对象，不是字符串）。你用一条新的 `user` 消息回复，其中包含 `{type: "tool_result", tool_use_id, content}` 块。

**Google Gemini API。** 你传入 `tools: [{functionDeclarations: [{name, description, parameters}]}]`（嵌套在 `functionDeclarations` 下）。响应以 `candidates[0].content.parts: [{functionCall: {name, args, id}}]` 的形式到达，其中 `id` 在 Gemini 3 及以上版本中是唯一的，用于并行调用关联。你用 `{functionResponse: {name, id, response}}` 回复。

同一个循环。不同的字段名、不同的嵌套方式、不同的字符串与对象约定、不同的关联机制。一个团队在 OpenAI 上写好天气 agent，迁移到 Anthropic 只为接线代码就要付出两天成本，再迁移到 Gemini 又要一天。

本课会构建一个翻译器，把三种格式统一成一个规范工具声明，并在边界处路由。Phase 13 · 17 会把同样的模式推广成一个 LLM gateway。

## 概念

### 共同结构

每个提供商都需要五样东西：

1. **工具列表。** 每个工具的名称、描述和输入 schema。
2. **工具选择。** 强制使用某个工具、禁止工具，或让模型自行决定。
3. **调用发出。** 结构化输出，说明工具名和参数。
4. **调用 id。** 将结果关联回正确的调用（并行时很重要）。
5. **结果注入。** 一条消息或一个块，把结果绑定回调用。

### 逐字段形状差异

| Aspect | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| 声明外壳 | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Schema 字段 | `parameters` | `input_schema` | `parameters` |
| 响应容器 | assistant 消息上的 `tool_calls[]` | 类型为 `tool_use` 的 `content[]` | 类型为 `functionCall` 的 `parts[]` |
| 参数类型 | 字符串化 JSON | 已解析对象 | 已解析对象 |
| Id 格式 | `call_...`（OpenAI 生成） | `toolu_...`（Anthropic） | UUID（Gemini 3+） |
| 结果块 | role `tool`、`tool_call_id` | 带 `tool_result`、`tool_use_id` 的 `user` | 带匹配 `id` 的 `functionResponse` |
| 强制工具 | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| 禁止工具 | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| 严格 schema | `strict: true` | schema 就是 schema（始终执行） | 请求级 `responseSchema` |

### 你实际会撞到的限制

- **OpenAI。** 每个请求最多 128 个工具。Schema 深度 5。参数字符串 <= 8192 字节。严格模式要求不能有 `$ref`，不能有带重叠的 `oneOf`/`anyOf`/`allOf`，每个属性都必须列在 `required` 中。
- **Anthropic。** 每个请求最多 64 个工具。Schema 深度理论上不受限制，但实际建议上限为 10。没有严格模式标志；schema 是一种契约，模型通常会遵守。
- **Gemini。** 每个请求最多 64 个函数。Schema 类型是 OpenAPI 3.0 子集（与 JSON Schema 2020-12 略有差异）。从 Gemini 3 开始，并行调用有唯一 id。

### `tool_choice` 行为

每家都支持三种模式，只是命名不同。

- **Auto。** 模型选择工具或文本。默认。
- **Required / Any。** 模型必须至少调用一个工具。
- **None。** 模型不得调用工具。

另外，每个提供商还有一种独有模式：

- **OpenAI。** 按名称强制使用某个特定工具。
- **Anthropic。** 按名称强制使用某个特定工具；`disable_parallel_tool_use` 标志区分单工具与多工具。
- **Gemini。** `mode: "VALIDATED"` 会无论模型意图如何，都把每个响应路由过 schema validator。

### 并行调用

OpenAI 的 `parallel_tool_calls: true`（默认）会在一条 assistant 消息中发出多个调用。你运行所有调用，并回复一批 tool-role 消息，其中每个条目对应一个 `tool_call_id`。Anthropic 历史上是单调用；`disable_parallel_tool_use: false`（Claude 3.5 起默认）启用多调用。Gemini 2 允许并行调用，但没有提供稳定 id；Gemini 3 添加 UUID，因此乱序响应也能干净关联。

### 流式传输

三者都支持流式工具调用。线路格式不同：

- **OpenAI。** `tool_calls[i].function.arguments` 的 delta chunk 会增量到达。你持续累积，直到 `finish_reason: "tool_calls"`。
- **Anthropic。** block-start / block-delta / block-stop 事件。`input_json_delta` chunk 携带部分参数。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 新增）会发出带 `functionCallId` 的 chunk，因此多个并行调用可以交错。

Phase 13 · 03 会深入讲并行 + 流式重组。本课聚焦声明和单调用形状。

### 错误与修复

无效参数错误看起来也不同。

- **OpenAI（非严格）。** 模型返回 `arguments: "{bad json}"`，你的 JSON 解析失败，于是你注入一条错误消息并重新调用。
- **OpenAI（严格）。** 验证发生在解码期间；不可能出现无效 JSON，但可能出现 `refusal`。
- **Anthropic。** `input` 可能包含意外字段；schema 偏建议性。需要在服务端验证。
- **Gemini。** OpenAPI 3.0 的怪异点：对象字段上的 `enum` 会被静默忽略；要自己验证。

### 翻译器模式

代码中的规范工具声明可以长这样（形状由你决定）：

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

三个很小的函数会把它翻译成三种提供商形状。`code/main.py` 中的 harness 正是这样做的，然后把一个假的工具调用通过每个提供商的响应形状来回转换。无需网络——本课教的是形状，不是 HTTP。

生产团队会把这个翻译器封装在 `AbstractToolset`（Pydantic AI）、`UniversalToolNode`（LangGraph）或 `BaseTool`（LlamaIndex）中。Phase 13 · 17 会交付一个 gateway，在三者中的任意一个前面暴露 OpenAI 形状的 API。

## 使用它

`code/main.py` 定义了一个规范 `Tool` dataclass，以及三个会发出 OpenAI、Anthropic 和 Gemini 声明 JSON 的翻译器。然后它把每种形状的手写提供商响应解析成同一个规范调用对象，展示皮肤之下语义完全相同。运行它，并并排 diff 三个声明。

要关注的点：

- 三个声明块只在外壳和字段名上不同。
- 三个响应块的差异在于调用位于哪里（顶层 `tool_calls`、`content[]` 块、`parts[]` 条目）。
- 一个 `canonical_call()` 函数会从三种响应形状中提取 `{id, name, args}`。

## 交付它

本课会产出 `outputs/skill-provider-portability-audit.md`。给定一个面向某个提供商的函数调用集成，这个 skill 会生成可移植性审计：它依赖了哪些提供商限制，哪些字段需要重命名，以及迁移到其他每个提供商时会坏在哪里。

## 练习

1. 运行 `code/main.py`，验证三种提供商声明 JSON 都序列化了同一个底层 `Tool` 对象。修改规范工具，添加一个 enum 参数，并确认只有 Gemini 翻译器需要处理 OpenAPI 怪异点。

2. 为每个提供商添加一个 `ListToolsResponse` 解析器，用于提取模型在 `list_tools` 或 discovery 调用后返回的工具列表。OpenAI 原生没有这个能力；记录这个不对称性。

3. 实现 `tool_choice` 转换：把规范的 `ToolChoice(mode="force", tool_name="x")` 映射成三种提供商形状。然后映射 `mode="any"` 和 `mode="none"`。检查本课的差异表。

4. 从三家提供商中任选一家，完整阅读其函数调用指南。找出其 schema spec 中有一个字段是另外两家不支持的。候选项：OpenAI `strict`、Anthropic `disable_parallel_tool_use`、Gemini `function_calling_config.allowed_function_names`。

5. 写一个测试向量：一个参数违反声明 schema 的工具调用。把它跑过每个提供商的 validator（Lesson 01 中的 stdlib validator 可作为代理），并记录哪些错误会触发。说明你会在生产中为了严格性选择哪个提供商。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Function calling | “Tool use” | 用于发出结构化工具调用的提供商级 API |
| Tool declaration | “Tool spec” | 名称 + 描述 + JSON Schema 输入载荷 |
| `tool_choice` | “Force / forbid” | auto / required / none / specific-name 模式 |
| Strict mode | “Schema enforcement” | OpenAI 用于约束解码以匹配 schema 的标志 |
| `tool_use` block | “Anthropic's call shape” | 带 id、name、input 的内联 content block |
| `functionCall` part | “Gemini's call shape” | 包含 name、args 和 id 的 `parts[]` 条目 |
| Arguments-as-string | “Stringified JSON” | OpenAI 以 JSON 字符串而非对象返回 args |
| Parallel tool calls | “Fan-out in one turn” | 一条 assistant 消息中的多个工具调用 |
| Refusal | “Model declines” | 严格模式下代替调用出现的 refusal block |
| OpenAPI 3.0 subset | “Gemini schema quirk” | Gemini 使用的类 JSON Schema 方言，存在少量差异 |

## 延伸阅读

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — 包含严格模式和并行调用的规范参考
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — `tool_use` 与 `tool_result` 块语义
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — 并行调用、唯一 id 和 OpenAPI 子集
- [Vertex AI — Function calling reference](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — Gemini 的企业级接口
- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — 严格模式 schema 强制执行细节
