# MCP 基础 — 原语、生命周期、JSON-RPC 基座

> 在 MCP 出现之前，每一个集成都是一次性的。模型上下文协议（Model Context Protocol，MCP）最初由 Anthropic 于 2024 年 11 月发布，现由 Linux 基金会的 Agentic AI Foundation 管理，它标准化了发现与调用机制，使得任何客户端都能与任何服务器通信。2025-11-25 版规范定义了六个原语（三个服务器端、三个客户端端）、三阶段生命周期，以及 JSON-RPC 2.0 传输格式。掌握了这些，本阶段 MCP 章节的其余内容就只是延伸阅读了。

**类型：** 学习
**语言：** Python（标准库，JSON-RPC 解析器）
**前置条件：** 第 13 阶段 · 01 到 05（工具接口与函数调用）
**时间：** 约 45 分钟

## 学习目标

- 列出所有六个 MCP 原语（服务器端的 tools、resources、prompts；客户端端的 roots、sampling、elicitation），并为每个给出一个用例。
- 走过三阶段生命周期（initialize、operation、shutdown），说明每个阶段由谁发送哪条消息。
- 解析和生成 JSON-RPC 2.0 的请求（request）、响应（response）和通知（notification）信封。
- 解释 `initialize` 时的能力协商（capability negotiation）是什么，以及没有它会出什么问题。

## 问题

在 MCP 之前，每个使用工具的 Agent 都有自己的协议。Cursor 有一个形似 MCP 但不兼容的工具系统。Claude Desktop 搭载了另一种。VS Code 的 Copilot 扩展又有第三种。一个构建了"Postgres 查询"工具的团队不得不将同一个工具写三遍，每遍对接不同宿主的 API。复用它需要复制代码。

结果是一次寒武纪大爆发般的一次性集成，以及生态系统增速的天花板。

MCP 通过标准化传输格式来解决这个问题。一个 MCP 服务器能在每一个 MCP 客户端中工作：Claude Desktop、ChatGPT、Cursor、VS Code、Gemini、Goose、Zed、Windsurf，到 2026 年 4 月已有 300+ 客户端。每月 SDK 下载量 1.1 亿次。10,000+ 公开服务器。Linux 基金会于 2025 年 12 月在新的 Agentic AI Foundation 下接过了管理权。

本阶段使用的规范版本为 **2025-11-25**。它新增了异步任务（async Tasks，SEP-1686）、URL 模式征询（URL-mode elicitation，SEP-1036）、带工具的采样（sampling with tools，SEP-1577）、增量范围授权（incremental scope consent，SEP-835），以及 OAuth 2.1 资源指示器语义。第 13 阶段 · 09 到 16 覆盖这些扩展。本课停留在基础部分。

## 概念

### 三个服务器原语

1. **Tools（工具）。** 可调用的动作。与第 13 阶段 · 01 中的四步循环相同。
2. **Resources（资源）。** 暴露的数据。通过 URI 寻址的只读内容：`file:///path`、`db://query/...`、自定义 scheme。
3. **Prompts（提示模板）。** 可复用的模板。宿主 UI 中的斜杠命令；服务器提供模板，客户端填充参数。

### 三个客户端原语

4. **Roots（根目录）。** 服务器被允许触碰的 URI 集合。客户端声明它们；服务器遵守它们。
5. **Sampling（采样）。** 服务器请求客户端的模型执行一次补全。使得服务器端可以托管 Agent 循环而无需服务器端 API 密钥。
6. **Elicitation（征询）。** 服务器在运行中向客户端用户请求结构化输入。表单或 URL（SEP-1036）。

MCP 中的每个能力都恰好属于这六个原语之一。第 13 阶段 · 10 到 14 将逐一深入讲解。

### 传输格式：JSON-RPC 2.0

每条消息都是一个 JSON 对象，包含以下字段：

- 请求：`{jsonrpc: "2.0", id, method, params}`。
- 响应：`{jsonrpc: "2.0", id, result | error}`。
- 通知：`{jsonrpc: "2.0", method, params}` — 没有 `id`，不期望收到响应。

基础规范约有 15 个方法，按原语分组。重要的有：

- `initialize` / `initialized`（握手）
- `tools/list`、`tools/call`
- `resources/list`、`resources/read`、`resources/subscribe`
- `prompts/list`、`prompts/get`
- `sampling/createMessage`（服务器到客户端）
- `notifications/tools/list_changed`、`notifications/resources/updated`、`notifications/progress`

### 三阶段生命周期

**阶段 1：initialize（初始化）。**

客户端发送 `initialize`，附带其 `capabilities` 和 `clientInfo`。服务器以其自身的 `capabilities`、`serverInfo` 和其支持的规范版本作为响应。客户端消化响应后发送 `notifications/initialized`。从此以后，双方都可以按照协商的能力发送请求。

**阶段 2：operation（运行）。**

双向的。客户端调用 `tools/list` 发现工具，然后调用 `tools/call` 执行。如果服务器声明了 sampling 能力，它可以发送 `sampling/createMessage`。当服务器的工具集发生变化时，它可以发送 `notifications/tools/list_changed`。当用户更改根目录范围时，客户端可以发送 `notifications/roots/list_changed`。

**阶段 3：shutdown（关闭）。**

任一方关闭传输层。MCP 中没有结构化的关闭方法；传输层（stdio 或 Streamable HTTP，第 13 阶段 · 09）承载连接结束信号。

### 能力协商

`initialize` 握手中的 `capabilities` 就是契约。服务器端示例：

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

如果客户端没有声明 `sampling`，服务器不得调用 `sampling/createMessage`。对称地：如果服务器没有声明 `resources.subscribe`，客户端不得尝试订阅。

这就是防止生态漂移的机制。不支持采样的客户端仍然是有效的 MCP 客户端；不调用 `sampling` 的服务器仍然是有效的 MCP 服务器。它们只是不会一起使用那个功能。

### 结构化内容与错误形状

`tools/call` 返回一个 `content` 数组，包含类型化块：`text`、`image`、`resource`。第 13 阶段 · 14 将 MCP 应用（MCP Apps，`ui://` 交互式 UI）添加到该列表。

错误使用 JSON-RPC 错误码。规范定义的扩展：`-32002` "Resource not found"、`-32603` "Internal error"，加上作为 `error.data` 的 MCP 特定错误数据。

### 客户端能力 vs 工具调用细节

一个常见的混淆：`capabilities.tools` 是指客户端是否支持 tool-list-changed 通知。客户端是否会调用特定工具是由其模型驱动的运行时选择，而非能力标志。能力标志是规范级别的契约。模型的选择是正交的。

### 为什么是 JSON-RPC 而不是 REST？

JSON-RPC 2.0（2010）是一种轻量级双向协议。REST 是客户端发起的。MCP 需要服务器端发起的消息（sampling、notifications），因此具有对称请求/响应形状的 JSON-RPC 是自然而然的选择。JSON-RPC 还可以干净地组合在 stdio 和 WebSocket/Streamable HTTP 之上，无需重新发明 HTTP 的请求形状。

```figure
mcp-tool-call
```

## 使用它

`code/main.py` 提供了一个最小化的 JSON-RPC 2.0 解析器和生成器，然后手动走一遍 `initialize` → `tools/list` → `tools/call` → `shutdown` 序列，打印每条消息。没有真正的传输层；只有消息形状。与延伸阅读中链接的规范对比，验证每个信封。

重点关注：

- `initialize` 双向声明能力；响应包含 `serverInfo` 和 `protocolVersion: "2025-11-25"`。
- `tools/list` 返回一个 `tools` 数组；每个条目包含 `name`、`description`、`inputSchema`。
- `tools/call` 使用 `params.name` 和 `params.arguments`。
- 响应的 `content` 是一个 `{type, text}` 块的数组。

## 交付它

本课产出 `outputs/skill-mcp-handshake-tracer.md`。给定一份 MCP 客户端-服务器交互的 pcap 风格转录，该技能标注每条消息所属的原语、生命周期阶段，以及它依赖的能力。

## 练习

1. 运行 `code/main.py`。找出能力协商发生的行，描述如果服务器没有声明 `tools.listChanged` 会有什么变化。

2. 扩展解析器以处理 `notifications/progress`。消息形状：`{method: "notifications/progress", params: {progressToken, progress, total}}`。在一个长时间运行的 `tools/call` 进行中发出它，并确认客户端处理器会显示进度条。

3. 从头到尾阅读 MCP 2025-11-25 规范 — 整个文档大约 80 页。找出大多数服务器不需要的那一个能力标志。提示：它与资源订阅有关。

4. 在纸上勾勒一个假想的"cron job"功能应该属于哪个原语。（提示：服务器希望客户端在预定时间调用它。现有的六个原语都不适合。）MCP 的 2026 路线图中有一个此功能的草案 SEP。

5. 从 GitHub 上的一个开源 MCP 服务器解析一次会话日志。统计请求 vs 响应 vs 通知消息的数量。计算生命周期 vs 运行阶段流量占总流量的比例。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| MCP | "模型上下文协议" | 用于模型到工具发现与调用的开放协议 |
| 服务器原语 | "服务器暴露的东西" | tools（动作）、resources（数据）、prompts（模板） |
| 客户端原语 | "客户端让服务器使用的东西" | roots（范围）、sampling（LLM 回调）、elicitation（用户输入） |
| JSON-RPC 2.0 | "传输格式" | 对称的请求/响应/通知信封 |
| `initialize` 握手 | "能力协商" | 第一对消息；服务器和客户端声明它们支持的功能 |
| `tools/list` | "发现" | 客户端向服务器请求其当前工具集 |
| `tools/call` | "调用" | 客户端要求服务器带参数执行一个工具 |
| `notifications/*_changed` | "变更事件" | 服务器告诉客户端其原语列表已更改 |
| 内容块 | "类型化结果" | 工具结果中的 `{type: "text" \| "image" \| "resource" \| "ui_resource"}` |
| SEP | "规范演进提案" | 命名的草案提案（例如 SEP-1686 对应异步任务） |

## 延伸阅读

- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 权威规范文档
- [Model Context Protocol — Architecture concepts](https://modelcontextprotocol.io/docs/concepts/architecture) — 六原语心智模型
- [Anthropic — Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) — 2024 年 11 月发布博文
- [MCP blog — First MCP anniversary](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 一周年回顾及 2025-11-25 规范变更
- [WorkOS — MCP 2025-11-25 spec update](https://workos.com/blog/mcp-2025-11-25-spec-update) — SEP-1686、1036、1577、835 和 1724 的摘要
