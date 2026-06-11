# 并行工具调用与工具流式处理

> 三个相互独立的天气查询如果串行执行，就是三次往返。并行运行后，总耗时会压缩到最慢的单次调用。如今所有前沿提供商都能在单轮中发出多个工具调用。收益是真实的；管道细节却很微妙。本课会讲清两部分：并行扇出（parallel fan-out）和流式参数重组（streamed-argument reassembly），重点强调 id 关联陷阱（id-correlation trap）。

**Type:** 构建
**Languages:** Python（标准库、线程池 + 流式处理框架）
**Prerequisites:** 第 13 阶段 · 02（函数调用深度解析）
**Time:** ~75 分钟

## 学习目标

- 解释为什么存在 `parallel_tool_calls: true`，以及什么时候应该禁用它。
- 在并行扇出期间，将流式参数片段关联到正确的工具调用 id。
- 将部分 `arguments` 字符串重组成完整 JSON，且不要过早解析。
- 运行一个三城市天气基准，展示串行与并行的延迟差异。

## 问题

如果没有并行调用，智能体在回答 “what is the weather in Bengaluru, Tokyo, and Zurich” 时会这样做：

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> run executor, reply with result
LLM -> call get_weather(Tokyo)
host -> run executor, reply with result
LLM -> call get_weather(Zurich)
host -> run executor, reply with result
LLM -> final text answer
```

三次 LLM 往返，而且每次还要支付执行器延迟。粗略来说，墙钟时间约为理想情况的 4 倍。

使用并行调用时：

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer
```

只需要一次 LLM 往返。执行器时间取三者最大值，而不是三者之和。OpenAI、Anthropic 和 Gemini 上的生产基准显示，在扇出型工作负载上，墙钟时间可减少 60% 到 70%。

代价是关联复杂度。当三个调用以乱序完成时，你的结果必须携带匹配的 `tool_call_id`，这样模型才能对齐它们。当结果以流式方式到达时，你必须先把部分参数片段组装成完整 JSON，再执行调用。Gemini 3 加入唯一 id，部分原因正是为了解决一个现实问题：对同一个工具的两次并行调用无法区分。

## 概念

### 启用并行

- **OpenAI.** `parallel_tool_calls: true` 默认开启。设为 `false` 可强制串行。
- **Anthropic.** 通过 `disable_parallel_tool_use: false` 实现并行（Claude 3.5 及以上默认开启）。设为 `true` 可串行。
- **Gemini.** 始终具备并行能力；`tool_config.function_calling_config.mode = "AUTO"` 让模型自行决定。

当工具之间存在顺序依赖（`create_file` 然后 `write_file`）、某次调用的输出会影响另一次调用的输入，或限流器无法承受扇出时，应禁用并行。

### Id 关联

模型发出的每个调用都有一个 `id`。宿主返回的每个结果都必须包含相同的 id。没有它，结果就是歧义的。

- **OpenAI.** 每条 tool-role 消息上有 `tool_call_id`。
- **Anthropic.** 每个 `tool_result` 块上有 `tool_use_id`。
- **Gemini.** 每个 `functionResponse` 上有 `id`（Gemini 3 及以上；Gemini 2 按名称匹配，这会在同名并行调用时失效）。

### 并发运行调用

宿主会在各自的线程、协程或远程 worker 上运行每个调用的执行器。最简单的框架使用线程池；生产环境会使用 asyncio 搭配 `asyncio.gather` 或结构化并发（structured concurrency）。完成顺序不可预测——id 才是标识符。

一个常见 bug：按调用列表顺序回复结果，而不是按完成顺序回复。由于模型通常只关心 `tool_call_id`，这通常也能工作；但如果某个结果被丢弃或重复，乱序提交会让调试更困难。更推荐按完成顺序回复，并显式带上 id。

### 流式工具调用

当模型流式输出时，`arguments` 会分片到达。三个并行调用的三路片段流会在传输线上交错。你需要为每个 id 准备一个累加器（accumulator）。

按提供商看，形态如下：

- **OpenAI.** 每个 chunk 是 `choices[0].delta.tool_calls[i].function.arguments`（部分字符串）。chunk 携带 `index`（调用列表中的位置）。你按 index 累加，在 `id` 首次出现时读取它，并在 `finish_reason = "tool_calls"` 时解析 JSON。
- **Anthropic.** 流事件先是 `message_start`，然后每个块都有一个 `content_block_start`，类型为 `tool_use`（包含 id、name、空 input）。`content_block_delta` 事件携带 `input_json_delta` 片段。`content_block_stop` 关闭每个块。
- **Gemini.** `streamFunctionCallArguments`（Gemini 3 及以上）会发出带有 `functionCallId` 的片段，因此调用可以干净地交错。在 Gemini 3 之前，流式处理一次只返回一个完整调用。

### 部分 JSON 与过早解析陷阱

在 `arguments` 完整之前，你不能解析它。像 `{"city": "Beng` 这样的部分 JSON 不是有效 JSON，会抛出异常。正确的闸门是提供商的调用结束信号：OpenAI 的 `finish_reason = "tool_calls"`、Anthropic 的 `content_block_stop`，或 Gemini 的流结束事件。只有到那时才尝试 `json.loads`。更稳健的方法是使用增量 JSON 解析器，在结构完成时产出事件；OpenAI 的流式处理指南建议为显示实时 “thinking” 指示器的 UX 使用这种方式。用花括号计数判断完整性并不可靠（带引号字符串或转义内容里的花括号会造成误判），只应作为非正式调试启发。

### 乱序完成

```
call_A: fast API, returns first
call_B: slow API, returns second
call_C: median API, returns third
```

宿主回复仍必须引用这些 id：

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

在 OpenAI 或 Anthropic 上，回复中的顺序不影响正确性。Gemini 也接受任意顺序，只要 id 匹配即可。

### 基准：串行 vs 并行

`code/main.py` 中的框架模拟了三个执行器，延迟分别为 400、600 和 800 ms。串行运行总共需要 1800 ms。并行运行耗时为 max(400, 600, 800) = 800 ms。差值是常数而非比例，因此工具数量越多，节省越明显。

现实世界中的注意事项：并行调用会给下游 API 施加压力。对限流服务做 10 路扇出会失败。第 13 阶段 · 17 会介绍网关层面的反压（backpressure）；重试语义计划在未来阶段讲解。

### 流式扇出的墙钟时间

如果模型本身在流式输出，你可以在某个调用的参数一完成时就开始执行，而不必等待所有调用都最终完成。这是 OpenAI 文档中说明的一种优化，但并非所有 SDK 都暴露。本课的框架实现了它：一旦模拟流产生一个完整的参数对象，宿主就会启动该调用。

## 使用它

`code/main.py` 分为两半。第一半使用 `concurrent.futures.ThreadPoolExecutor` 串行和并行运行三个模拟天气调用，并打印墙钟时间。第二半回放一个假的流式响应——三个并行调用的 `arguments` 片段交错在同一条流上——并使用 `StreamAccumulator` 按 id 重组它们。不用 LLM，不用网络，只练习重组逻辑。

需要观察的点：

- 在相同的假延迟下，串行计时器达到 1.8 秒，并行计时器达到 0.8 秒。
- 累加器通过按 id 缓冲并且只在每个调用的 JSON 完整时解析，来处理乱序到达的片段。
- 执行器会在某个 id 的参数完成后立刻启动，而不是等所有流都结束。

## 交付它

本课会产出 `outputs/skill-parallel-call-safety-check.md`。给定一个工具注册表，该 skill 会审计哪些工具可以安全并行化、哪些存在顺序依赖、哪些会压垮下游限流——并返回一个带有逐工具 `parallel_safe` 标志的修订版注册表。

## 练习

1. 运行 `code/main.py`，并改变模拟延迟。确认并行与串行的比例大约是 `max/sum`（真实运行会因为线程调度、序列化和框架开销而略微偏离理想值）。在什么样的延迟分布下，并行不再重要？

2. 扩展累加器，使其能处理 “call was cancelled mid-stream” 的情况：丢弃它的缓冲区并发出一个 `cancelled` 事件。哪个提供商明确记录了这种情况？检查 Anthropic 的 `content_block_stop` 语义和 OpenAI 的 `finish_reason: "length"` 行为。

3. 用 `asyncio.gather` 替换线程池。对两者做基准测试。你应该会看到 async 带来小幅收益，因为上下文切换成本更低；但只有当执行器做真实 I/O 时才会如此。

4. 选择两个不应该并行化的工具（例如先 `create_file` 再 `write_file`）。向注册表添加一个 `ordering_dependency` 图，并基于该图为并行扇出加闸。这是依赖感知调度（dependency-aware scheduling）的最小机制，未来的智能体工程阶段会将其形式化。

5. 阅读 OpenAI 的 parallel-function-calling 小节和 Anthropic 的 `disable_parallel_tool_use` 文档。找出 Anthropic 建议禁用并行性的一个真实世界工具类型。（提示：对同一资源的后果性变更。）

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Parallel tool calls | “单轮扇出” | 模型在单条 assistant 消息中发出多个工具调用 |
| `parallel_tool_calls` | “OpenAI 的标志” | 启用或禁用多调用发出 |
| `disable_parallel_tool_use` | “Anthropic 的反向开关” | 退出并行的标志；默认启用并行 |
| Tool call id | “关联句柄” | 结果消息必须回显的逐调用标识符 |
| Accumulator | “流缓冲区” | 面向部分 `arguments` 片段的逐 id 字符串缓冲区 |
| Out-of-order completion | “最快先完成” | 并行调用以不可预测的顺序完成；id 是粘合剂 |
| Dependency graph | “顺序约束” | 输出会喂给其他工具输入的工具；不能并行化 |
| Parse-early trap | “JSON.parse 炸了” | 尝试解析不完整的 `arguments` 字符串 |
| `streamFunctionCallArguments` | “Gemini 3 功能” | 每个调用都有唯一 id 的流式参数片段 |
| Completion-order reply | “不要等全部完成” | 结果到达即按 id 回复 |

## 延伸阅读

- [OpenAI — Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — 默认行为和退出标志
- [Anthropic — Tool use: implementing tool use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) — `disable_parallel_tool_use` 和结果批处理
- [Google — Gemini function calling parallel section](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini 3 起基于 id 关联的并行调用
- [OpenAI — Streaming responses with tools](https://platform.openai.com/docs/api-reference/responses-streaming) — OpenAI 流中的分片参数重组
- [Anthropic — Streaming messages](https://docs.anthropic.com/en/api/messages-streaming) — 带有 `input_json_delta` 的 `content_block_delta`
