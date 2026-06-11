# 构建 MCP Server — Python + TypeScript SDKs

> 大多数 MCP 教程只展示 stdio 版的 hello-world。真正的 server 会暴露 tools、resources 和 prompts，处理能力协商（capability negotiation），发出结构化错误，并且在不同 SDK 中行为一致。本课将端到端构建一个 notes server：stdlib stdio transport、JSON-RPC dispatch、三种 server primitives，以及一种纯函数风格，等你进阶时可以直接迁移到 Python SDK 的 FastMCP 或 TypeScript SDK。

**类型：** 构建
**语言：** Python（stdlib，stdio MCP server）
**前置要求：** Phase 13 · 06（MCP 基础）
**时间：** ~75 分钟

## 学习目标

- 实现 `initialize`、`tools/list`、`tools/call`、`resources/list`、`resources/read`、`prompts/list` 和 `prompts/get` 方法。
- 编写一个 dispatch loop，从 stdin 读取 JSON-RPC messages，并把 responses 写到 stdout。
- 按照 JSON-RPC 2.0 规范和 MCP 的附加错误码发出结构化 error responses。
- 在不重写 tool 逻辑的前提下，把 stdlib 实现升级到 FastMCP（Python SDK）或 TypeScript SDK。

## 问题

在你能使用 remote transport（Phase 13 · 09）或 auth layer（Phase 13 · 16）之前，需要先有一个干净的本地 server。本地意味着 stdio：client 会把 server 作为子进程启动，messages 通过 stdin/stdout 以换行分隔的方式流动。

2025-11-25 规范规定，stdio messages 被编码为带有显式 `\n` 分隔符的 JSON objects。这里没有 SSE；SSE 是旧的 remote mode，并将在 2026 年中移除（Atlassian 的 Rovo MCP server 已在 2026 年 6 月 30 日弃用它；Keboola 在 2026 年 4 月 1 日弃用）。对于 stdio，每行一个 JSON object 就是完整的 wire format。

notes server 是一个很好的形状，因为它会练习全部三种 server primitives。Tools 执行变更（`notes_create`）。Resources 暴露数据（`notes://{id}`）。Prompts 提供模板（`review_note`）。本课的形状可以泛化到任何领域。

## 概念

### Dispatch loop

```
loop:
  line = stdin.readline()
  msg = json.loads(line)
  if has id:
    handle request -> write response
  else:
    handle notification -> no response
```

三条规则：

- 不要向 stdout 打印任何不是 JSON-RPC envelope 的内容。Debug logs 应写到 stderr。
- 每个 request 都 MUST 匹配一个携带相同 `id` 的 response。
- Notifications MUST NOT 被响应。

### 实现 `initialize`

```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }
```

只声明你支持的能力。client 会依赖 capability set 来控制功能是否可用。

### 实现 `tools/list` 和 `tools/call`

`tools/list` 返回 `{tools: [...]}`，其中每个条目都有 `name`、`description`、`inputSchema`。`tools/call` 接收 `{name, arguments}` 并返回 `{content: [blocks], isError: bool}`。

Content blocks 是带类型的。最常见的有：

```json
{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

Tool errors 有两种形状。Protocol-level errors（未知 method、错误 params）是 JSON-RPC errors。Tool-level errors（调用合法，但 tool 失败）会作为 `{content: [...], isError: true}` 返回。这样 model 就能在自己的 context 中看到失败。

### 实现 resources

Resources 按设计是只读的。`resources/list` 返回 manifest；`resources/read` 返回内容。URIs 可以是 `file://...`、`http://...`，也可以是像 `notes://` 这样的自定义 scheme。

当你把数据暴露为 resource 而不是 tool 时：

- model 不会“调用”它；client 可以在用户请求时把它注入 context。
- Subscriptions 允许 server 在 resource 变化时推送更新（Phase 13 · 10）。
- Phase 13 · 14 会用 `ui://` 为 interactive resources 扩展这一点。

### 实现 prompts

Prompts 是带命名参数的模板。host 会把它们呈现为 slash-commands。一个 `review_note` prompt 可能接收 `note_id` 参数，并生成一个多 message 的 prompt template，client 再把它喂给自己的 model。

### Stdio transport subtleties

- 换行分隔的 JSON。没有 length-prefixed framing。
- 不要缓冲。每次写入后调用 `sys.stdout.flush()`。
- client 控制生命周期。当 stdin 关闭（EOF）时，干净退出。
- 不要静默处理 SIGPIPE；记录日志并退出。

### Annotations

每个 tool 都可以携带 `annotations` 来描述安全属性：

- `readOnlyHint: true` — 纯读取，适合重试。
- `destructiveHint: true` — 不可逆副作用；client 应该确认。
- `idempotentHint: true` — 相同输入产生相同输出。
- `openWorldHint: true` — 与外部系统交互。

client 会用这些来决定 UX（确认对话框、状态指示器）和路由（Phase 13 · 17）。

### Graduation path

`code/main.py` 中的 stdlib server 大约 180 行。FastMCP（Python）把相同逻辑压缩成 decorator-style：

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

TypeScript SDK 也有等价形状。准备好时，graduation path 可以直接迁移；核心概念（capabilities、dispatch、content blocks）是相同的。

## 使用它

`code/main.py` 是一个完整的 notes MCP server，通过 stdio 通信，只使用 stdlib。它为三个 tools（`notes_list`、`notes_search`、`notes_create`）处理 `initialize`、`tools/list`、`tools/call`，为每条 note 处理 `resources/list` 和 `resources/read`，并提供一个 `review_note` prompt。你可以通过管道传入 JSON-RPC messages 来驱动它：

```
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

需要关注的点：

- dispatcher 是一个以 method name 为键的 `dict[str, Callable]`。
- 每个 tool executor 返回 content blocks 列表，而不是裸字符串。
- 当 executor 抛出异常时，会设置 `isError: true`。

## 交付它

本课会产出 `outputs/skill-mcp-server-scaffolder.md`。给定一个领域（notes、tickets、files、database），这个 skill 会 scaffolds 一个 MCP server，包含正确的 tools / resources / prompts 划分和 SDK graduation path。

## 练习

1. 运行 `code/main.py`，并用手写 JSON-RPC messages 驱动它。练习 `notes_create`，然后用 `resources/read` 取回新 note。

2. 添加一个带有 `annotations: {destructiveHint: true}` 的 `notes_delete` tool。验证 client 会显示确认对话框（这需要真实 host；Claude Desktop 可以做到）。

3. 实现 `resources/subscribe`，让 server 在 note 被修改时推送 `notifications/resources/updated`。添加一个 keepalive task。

4. 把 server 移植到 FastMCP。Python 文件应该缩小到 80 行以内。wire behavior 必须完全一致；用同一个 JSON-RPC test harness 验证。

5. 阅读规范中的 `server/tools` 章节，并找出本课 server 未实现的 tool definition 字段。（提示：有好几个；挑一个并加上。）

## 关键术语

| 术语 | 人们常说的说法 | 实际含义 |
|------|----------------|----------|
| MCP server | “暴露 tools 的东西” | 通过 stdio 或 HTTP 讲 MCP JSON-RPC 的进程 |
| stdio transport | “子进程模型” | server 由 client 启动；通过 stdin/stdout 通信 |
| Dispatcher | “Method router” | JSON-RPC method name 到 handler function 的映射 |
| Content block | “Tool result chunk” | tool response 的 `content` array 中的带类型元素 |
| `isError` | “Tool-level failure” | 表示 tool 失败；区别于 JSON-RPC error |
| Annotations | “Safety hints” | readOnly / destructive / idempotent / openWorld flags |
| FastMCP | “Python SDK” | 构建在 MCP protocol 之上的 decorator-based 高阶框架 |
| Resource URI | “Addressable data” | 标识 resource 的 `file://`、`db://` 或自定义 scheme |
| Prompt template | “Slash-command brief” | server 提供的模板，带有给 host UIs 使用的参数槽位 |
| Capability declaration | “Feature toggle” | 在 `initialize` 中声明的 per-primitive flags |

## 延伸阅读

- [Model Context Protocol — Python SDK](https://github.com/modelcontextprotocol/python-sdk) — reference Python implementation
- [Model Context Protocol — TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) — parallel TS implementation
- [FastMCP — server framework](https://gofastmcp.com/) — MCP servers 的 decorator-style Python API
- [MCP — Quickstart server guide](https://modelcontextprotocol.io/quickstart/server) — 使用任一 SDK 的端到端教程
- [MCP — Server tools spec](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) — tools/* messages 的完整参考
