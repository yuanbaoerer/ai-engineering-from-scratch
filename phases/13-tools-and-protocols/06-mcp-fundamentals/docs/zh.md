# MCP 基础 — 原语、生命周期、JSON-RPC 基础

> MCP 之前的每个集成都是一次性的。Model Context Protocol（模型上下文协议，MCP）最早由 Anthropic 于 2024 年 11 月发布，如今由 Linux Foundation 旗下的 Agentic AI Foundation 负责管理，它标准化了发现与调用机制，让任何客户端都能与任何服务器通信。2025-11-25 规范定义了六个原语（三个服务器端、三个客户端）、三阶段生命周期，以及 JSON-RPC 2.0 线格式。掌握这些内容后，本阶段 MCP 章节的其余部分就只是阅读理解了。

**类型：** 学习
**语言：** Python（stdlib，JSON-RPC 解析器）
**先修要求：** Phase 13 · 01 到 05（工具接口与函数调用）
**时间：** 约 45 分钟

## 学习目标

- 说出全部六个 MCP 原语（服务器端的 tools、resources、prompts；客户端的 roots、sampling、elicitation），并分别给出一个使用场景。
- 走查三阶段生命周期（initialize、operation、shutdown），并说明每个阶段由谁发送哪条消息。
- 解析并发出 JSON-RPC 2.0 的 request、response 和 notification 信封。
- 解释 `initialize` 阶段的能力协商（capability negotiation）是什么，以及没有它会破坏什么。

## 问题

MCP 之前，每个使用工具的 agent 都有自己的协议。Cursor 有一个形似 MCP 但不兼容的工具系统。Claude Desktop 随带的是另一个系统。VS Code 的 Copilot 扩展又有第三套。一个团队构建了“Postgres query”工具，却要把同一个工具写三遍，每一遍都对接不同宿主的 API。复用它意味着复制代码。

结果就是一次性集成的寒武纪大爆发，以及生态系统发展速度的天花板。

MCP 通过标准化线格式解决了这个问题。一个 MCP server 可以在每个 MCP client 中工作：Claude Desktop、ChatGPT、Cursor、VS Code、Gemini、Goose、Zed、Windsurf，到 2026 年 4 月已有 300+ 客户端。SDK 月下载量 1.1 亿。公开 server 10,000+。Linux Foundation 于 2025 年 12 月在新的 Agentic AI Foundation 之下接管了治理。

本阶段使用的规范版本是 **2025-11-25**。它新增了异步 Tasks（SEP-1686）、URL 模式 elicitation（SEP-1036）、带 tools 的 sampling（SEP-1577）、增量范围同意（SEP-835），以及 OAuth 2.1 resource-indicator 语义。Phase 13 · 09 到 16 会覆盖这些扩展。本课只讲基础部分。

## 概念

### 三个服务器端原语

1. **Tools。** 可调用的动作。与 Phase 13 · 01 中相同的四步循环。
2. **Resources。** 暴露的数据。可通过 URI 寻址的只读内容：`file:///path`、`db://query/...`、自定义 scheme。
3. **Prompts。** 可复用模板。宿主 UI 中的斜杠命令；服务器提供模板，客户端填充参数。

### 三个客户端原语

4. **Roots。** 服务器被允许触碰的 URI 集合。客户端声明它们；服务器遵守它们。
5. **Sampling。** 服务器请求客户端的模型执行一次补全。支持服务器托管的 agent 循环，而无需服务器端 API key。
6. **Elicitation。** 服务器在执行过程中向客户端用户请求结构化输入。表单或 URL（SEP-1036）。

MCP 中的每项能力都精确属于这六个原语之一。Phase 13 · 10 到 14 会深入讲解每一个。

### 线格式：JSON-RPC 2.0

每条消息都是一个包含这些字段的 JSON 对象：

- Requests: `{jsonrpc: "2.0", id, method, params}`。
- Responses: `{jsonrpc: "2.0", id, result | error}`。
- Notifications: `{jsonrpc: "2.0", method, params}` — 没有 `id`，也不期望响应。

基础规范有约 15 个方法，按原语分组。重要的方法包括：

- `initialize` / `initialized`（握手）
- `tools/list`, `tools/call`
- `resources/list`, `resources/read`, `resources/subscribe`
- `prompts/list`, `prompts/get`
- `sampling/createMessage`（服务器到客户端）
- `notifications/tools/list_changed`, `notifications/resources/updated`, `notifications/progress`

### 三阶段生命周期

**Phase 1: initialize.**

客户端发送 `initialize`，携带它的 `capabilities` 和 `clientInfo`。服务器响应自己的 `capabilities`、`serverInfo`，以及它使用的规范版本。客户端在消化响应后发送 `notifications/initialized`。从这里开始，任一方都可以根据协商出的能力发送请求。

**Phase 2: operation.**

双向。客户端调用 `tools/list` 来发现工具，然后调用 `tools/call` 来调用工具。如果服务器声明了相应能力，它可以发送 `sampling/createMessage`。当服务器的工具集发生变化时，它可以发送 `notifications/tools/list_changed`。当用户改变 root 范围时，客户端可以发送 `notifications/roots/list_changed`。

**Phase 3: shutdown.**

任一方关闭传输。MCP 中没有结构化 shutdown 方法；传输层（stdio 或 Streamable HTTP，Phase 13 · 09）承载连接结束信号。

### 能力协商

`initialize` 握手中的 `capabilities` 就是契约。下面是来自服务器的示例：

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

服务器声明它可以发出 `tools/list_changed` 通知，并支持 `resources/subscribe`。客户端通过声明自己的能力来表示同意：

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

如果客户端没有声明 `sampling`，服务器就不得调用 `sampling/createMessage`。对称地：如果服务器没有声明 `resources.subscribe`，客户端就不得尝试订阅。

这正是防止生态漂移的机制。不支持 sampling 的客户端仍然是有效的 MCP client；不调用 `sampling` 的服务器仍然是有效的 MCP server。它们只是不会一起使用这个功能。

### 结构化内容与错误形状

`tools/call` 返回由类型化块组成的 `content` 数组：`text`、`image`、`resource`。Phase 13 · 14 会把 MCP Apps（`ui://` 交互式 UI）加入这个列表。

错误使用 JSON-RPC 错误码。规范定义的新增项包括：`-32002` "Resource not found"，`-32603` "Internal error"，以及作为 `error.data` 的 MCP 专用错误数据。

### 客户端能力 vs 工具调用细节

一个常见混淆是：`capabilities.tools` 表示客户端是否支持工具列表变更通知。客户端是否会调用特定工具，是由其模型在运行时驱动的选择，而不是一个能力标志。能力标志是规范层面的契约。模型的选择与它正交。

### 为什么是 JSON-RPC 而不是 REST？

JSON-RPC 2.0（2010）是一种轻量级双向协议。REST 是客户端发起的。MCP 需要服务器发起的消息（sampling、notifications），所以具备对称 request/response 形状的 JSON-RPC 自然适合。JSON-RPC 也可以干净地组合在 stdio 和 WebSocket/Streamable HTTP 之上，而不必重新发明 HTTP 的请求形状。

## 使用它

`code/main.py` 提供了一个最小 JSON-RPC 2.0 解析器和发射器，然后手动走查 `initialize` → `tools/list` → `tools/call` → `shutdown` 序列，打印每条消息。没有真实传输；只有消息形状。对照 Further Reading 中链接的规范，验证每个信封。

要关注的内容：

- `initialize` 双向声明能力；响应包含 `serverInfo` 和 `protocolVersion: "2025-11-25"`。
- `tools/list` 返回一个 `tools` 数组；每个条目都有 `name`、`description`、`inputSchema`。
- `tools/call` 使用 `params.name` 和 `params.arguments`。
- 响应 `content` 是 `{type, text}` 块组成的数组。

## 交付它

本课会产出 `outputs/skill-mcp-handshake-tracer.md`。给定一份 pcap 风格的 MCP 客户端-服务器交互 transcript，这个 skill 会为每条消息标注它属于哪个原语、哪个生命周期阶段，以及依赖哪项能力。

## 练习

1. 运行 `code/main.py`。找出能力协商发生在哪一行，并描述如果服务器没有声明 `tools.listChanged` 会发生什么变化。

2. 扩展解析器以处理 `notifications/progress`。消息形状：`{method: "notifications/progress", params: {progressToken, progress, total}}`。在长时间运行的 `tools/call` 进行中发出它，并确认客户端处理器会显示进度条。

3. 从头到尾阅读 MCP 2025-11-25 规范 — 整份文档大约 80 页。找出大多数服务器都不需要的那一个能力标志。提示：它与资源订阅有关。

4. 在纸上画出一个假想的“cron job”功能会属于哪个原语。（提示：服务器希望客户端在计划时间调用它。今天六个原语都不适合。）MCP 的 2026 roadmap 有一个针对它的 SEP 草案。

5. 从 GitHub 上一个开放 MCP server 解析一份 session log。统计 request、response、notification 消息的数量。计算 lifecycle 流量与 operation 流量各占多少比例。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MCP | "Model Context Protocol" | 面向模型到工具发现与调用的开放协议 |
| Server primitive | “服务器暴露的东西” | tools（动作）、resources（数据）、prompts（模板） |
| Client primitive | “客户端允许服务器使用的东西” | roots（范围）、sampling（LLM 回调）、elicitation（用户输入） |
| JSON-RPC 2.0 | “线格式” | 对称的 request/response/notification 信封 |
| `initialize` handshake | “能力协商” | 第一组消息；服务器和客户端声明它们支持的功能 |
| `tools/list` | “发现” | 客户端向服务器请求其当前工具集 |
| `tools/call` | “调用” | 客户端请求服务器用参数执行一个工具 |
| `notifications/*_changed` | “变更事件” | 服务器告诉客户端其原语列表已经变化 |
| Content block | “类型化结果” | 工具结果中的 `{type: "text" \| "image" \| "resource" \| "ui_resource"}` |
| SEP | “Spec Evolution Proposal” | 命名的规范草案提案（例如用于异步 Tasks 的 SEP-1686） |

## 延伸阅读

- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 权威规范文档
- [Model Context Protocol — Architecture concepts](https://modelcontextprotocol.io/docs/concepts/architecture) — 六原语心智模型
- [Anthropic — Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) — 2024 年 11 月发布文章
- [MCP blog — First MCP anniversary](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 一周年回顾与 2025-11-25 规范变更
- [WorkOS — MCP 2025-11-25 spec update](https://workos.com/blog/mcp-2025-11-25-spec-update) — SEP-1686、1036、1577、835 和 1724 摘要
