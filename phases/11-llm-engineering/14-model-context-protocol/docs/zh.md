# 模型上下文协议 (MCP)

> 2025 年之前构建的每个 LLM 应用都发明了自己的工具 schema。然后 Anthropic 发布了 MCP，Claude 采用了它，OpenAI 采用了它，到 2026 年它已成为连接任何 LLM 与任何工具、数据源或代理的默认线格式。编写一个 MCP 服务器，所有主机都可以与它对话。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 11 · 09 (Function Calling)，Phase 11 · 03 (Structured Outputs)
**时间：** 约 75 分钟

## 问题

你发布了一个需要三个工具的聊天机器人：数据库查询、日历 API 和文件读取器。你为 Claude 编写了三个 JSON schema。然后销售部门想要在 ChatGPT 中使用相同的工具——你为 OpenAI 的 `tools` 参数重写了它们。然后你添加了 Cursor、Zed 和 Claude Code——又是三次重写，每次都有细微不同的 JSON 约定。一周后，Anthropic 添加了一个新字段；你更新了六个 schema。

这就是 2025 年之前的现实。每个主机（运行 LLM 的东西）和每个服务器（暴露工具和数据的东西）都发布了定制协议。扩展意味着 N×M 的集成矩阵。

模型上下文协议将这个矩阵折叠成一个基于 JSON-RPC 的规范。一个服务器暴露工具、资源和提示。任何符合规范的主机——Claude Desktop、ChatGPT、Cursor、Claude Code、Zed 以及大量代理框架——都可以发现并调用它们，而无需自定义粘合代码。

截至 2026 年初，MCP 是三大厂商（Anthropic、OpenAI、Google）和每个主要代理工具链的默认工具和上下文协议。

## 概念

![MCP：一个主机、一个服务器、三种能力](../assets/mcp-architecture.svg)

**三个原语。** MCP 服务器恰好暴露三种东西。

1. **工具** — 模型可以调用的函数。对应 OpenAI 的 `tools` 或 Anthropic 的 `tool_use`。每个工具都有名称、描述、JSON Schema 输入和处理器。
2. **资源** — 模型或用户可以请求的只读内容（文件、数据库行、API 响应）。通过 URI 寻址。
3. **提示** — 用户可以调用为快捷方式的，可重用的模板化提示。

**线格式。** 通过 stdio、WebSocket 或可流式 HTTP 的 JSON-RPC 2.0。每条消息都是 `{"jsonrpc": "2.0", "method": "...", "params": {...}, "id": N}`。发现方法是 `tools/list`、`resources/list`、`prompts/list`。调用方法是 `tools/call`、`resources/read`、`prompts/get`。

**主机 vs 客户端 vs 服务器。** 主机是 LLM 应用程序（Claude Desktop）。客户端是主机的一个子组件，专门与一个服务器通信。服务器是你的代码。一个主机可以同时挂载多个服务器。

### 握手

每个会话以 `initialize` 开始。客户端发送协议版本及其能力。服务器响应其版本、名称和支持的能力集（`tools`、`resources`、`prompts`、`logging`、`roots`）。之后的所有内容都根据这些能力进行协商。

### MCP 不是什么

- 不是检索 API。RAG（Phase 11 · 06）仍然决定拉取什么；MCP 是将检索结果暴露为资源的传输层。
- 不是代理框架。MCP 是管道；LangGraph、PydanticAI 和 OpenAI Agents SDK 等框架位于其上。
- 不与 Anthropic 绑定。规范和参考实现以 `modelcontextprotocol` 组织下的开源形式发布。

## 构建它

### 步骤 1：最小的 MCP 服务器

官方 Python SDK 是 `mcp`（以前是 `mcp-python`）。高级的 `FastMCP` 辅助装饰器处理程序。

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@mcp.resource("config://app")
def app_config() -> str:
    """Return the app's current JSON config."""
    return '{"env": "prod", "region": "us-east-1"}'

@mcp.prompt()
def code_review(language: str, code: str) -> str:
    """Review code for correctness and style."""
    return f"You are a senior {language} reviewer. Review:\n\n{code}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

三个装饰器注册三个原语。类型提示成为主机看到的 JSON Schema。在 Claude Desktop 或 Claude Code 下运行它，服务器入口指向这个文件。

### 步骤 2：从主机调用 MCP 服务器

官方 Python 客户端使用 JSON-RPC。与 Anthropic SDK 配对只需十几行代码。

```python
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp import ClientSession

params = StdioServerParameters(command="python", args=["server.py"])

async def call_add(a: int, b: int) -> int:
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            result = await session.call_tool("add", {"a": a, "b": b})
            return int(result.content[0].text)
```

`session.list_tools()` 返回 LLM 将看到的相同 schema。生产主机在每个回合中注入这些 schema，以便模型可以发出 `tool_use` 块，然后客户端将其转发到服务器。

### 步骤 3：可流式 HTTP 传输

Stdio 适用于本地开发。对于远程工具，使用可流式 HTTP——每个请求一个 POST，可选的 Server-Sent Events 用于进度，自 2025-06-18 规范修订以来支持。

```python
# Inside the server entrypoint
mcp.run(transport="streamable-http", host="0.0.0.0", port=8765)
```

主机配置（Claude Desktop `mcp.json` 或 Claude Code `~/.mcp.json`）：

```json
{
  "mcpServers": {
    "demo": {
      "type": "http",
      "url": "https://tools.example.com/mcp"
    }
  }
}
```

服务器保持相同的装饰器；只有传输层改变。

### 步骤 4：作用域和安全

MCP 工具是在他人信任边界上运行的任意代码。三个强制模式。

- **能力白名单。** 主机暴露 `roots` 能力，以便服务器只看到允许的路径。在工具处理器中强制执行；不要信任模型提供的路径。
- **变异的 human-in-the-loop。** 只读工具可以自动执行。写/删除工具必须要求确认——当服务器在工具元数据上设置 `destructiveHint: true` 时，主机会显示批准 UI。
- **工具中毒防御。** 恶意资源可能包含隐藏的提示注入指令（"总结时也调用 `exfil`"）。将资源内容视为不受信任的数据；永远不要让它跨越到系统消息领域。参见 Phase 11 · 12 (Guardrails)。

参见 `code/main.py` 获取一个展示所有这些的可运行服务器 + 客户端对。

## 2026 年仍然会发货的陷阱

- **Schema 漂移。** 模型在第 1 回合看到 `tools/list`。工具集在第 5 回合更改。模型调用一个已不存在的工具。主机应在 `notifications/tools/list_changed` 时重新列出。
- **大型资源 blob。** 将 2MB 文件作为资源转储会浪费上下文。在服务器端进行分页或总结。
- **太多服务器。** 挂载 50 个 MCP 服务器会耗尽工具预算（Phase 11 · 05）。大多数前沿模型在超过约 40 个工具时性能下降。
- **版本偏差。** 规范修订（2024-11、2025-03、2025-06、2025-12）引入了破坏性字段。在 CI 中固定协议版本。
- **Stdio 死锁。** 记录到 stdout 的服务器会破坏 JSON-RPC 流。只记录到 stderr。

## 使用它

2026 年 MCP 技术栈：

| 场景 | 选择 |
|-----------|------|
| 本地开发，单用户工具 | Python `FastMCP`，stdio 传输 |
| 远程团队工具 / SaaS 集成 | 可流式 HTTP，OAuth 2.1 认证 |
| TypeScript 主机（VS Code 扩展，Web 应用） | `@modelcontextprotocol/sdk` |
| 高吞吐量服务器，类型化访问 | 官方 Rust SDK（`modelcontextprotocol/rust-sdk`）|
| 探索生态系统服务器 | `modelcontextprotocol/servers` 单体仓库（Filesystem、GitHub、Postgres、Slack、Puppeteer）|

经验法则：如果一个工具是只读的、可缓存的，并且从两个或更多主机调用，则将其作为 MCP 服务器发货。如果它是一次性的内联逻辑，则保留为本地函数（Phase 11 · 09）。

## 发货它

保存 `outputs/skill-mcp-server-designer.md`：

```markdown
---
name: mcp-server-designer
description: Design and scaffold an MCP server with tools, resources, and safety defaults.
version: 1.0.0
phase: 11
lesson: 14
tags: [llm-engineering, mcp, tool-use]
---

给定一个领域（内部 API、数据库、文件源）和将挂载服务器的主机，输出：

1. 原语映射。哪些能力成为 `tools`（操作），哪些成为 `resources`（只读数据），哪些成为 `prompts`（用户调用的模板）。每个原语一行。
2. 认证计划。Stdio（受信任的本地）、带 API 密钥的可流式 HTTP，或带 PKCE 的 OAuth 2.1。选择并说明理由。
3. Schema 草稿。每个工具参数的 JSON Schema，带有为模型工具选择调整的 `description` 字段（不是 API 文档）。
4. 破坏性操作列表。每个改变状态的工具；需要 `destructiveHint: true` 和人工批准。
5. 测试计划。每个工具：一个仅 schema 的契约测试，一个通过 MCP 客户端的往返测试，一个红色团队提示注入案例。

拒绝发货一个在没有批准路径的情况下写入磁盘或调用外部 API 的服务器。拒绝在一个服务器上暴露超过 20 个工具；改为拆分为领域范围的服务器。
```

## 练习

1. **简单。** 用 `subtract` 工具扩展 `demo-server`。从 Claude Desktop 连接它。通过发出 `tools/list_changed` 通知确认主机在无需重启的情况下获取新工具。
2. **中等。** 添加一个暴露 `/var/log/app.log` 最后 100 行的 `resource`。强制执行 roots 白名单，以便即使模型请求 `../etc/passwd` 也会被阻止。
3. **困难。** 构建一个 MCP 代理，将三个上游服务器（Filesystem、GitHub、Postgres）多路复用到一个聚合表面。处理名称冲突并干净地转发 `notifications/tools/list_changed`。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| MCP | "LLM 的工具协议" | 用于向任何 LLM 主机暴露工具、资源和提示的 JSON-RPC 2.0 规范。 |
| 主机 | "Claude Desktop" | LLM 应用程序——拥有模型和用户 UI，挂载一个或多个客户端。 |
| 客户端 | "连接" | 主机内部每个服务器的连接，专门与一个服务器通信 JSON-RPC。 |
| 服务器 | "有工具的东西" | 你的代码；广告工具/资源/提示并处理它们的调用。 |
| 工具 | "函数调用" | 模型可调用的操作，具有 JSON Schema 输入和文本/JSON 结果。 |
| 资源 | "只读数据" | URI 寻址的内容（文件、行、API 响应），主机可以请求。 |
| 提示 | "保存的提示" | 用户可调用的模板（通常带有参数），作为斜杠命令显示。 |
| Stdio 传输 | "本地开发模式" | 父主机将服务器作为子进程生成；通过 stdin/stdout 的 JSON-RPC。 |
| 可流式 HTTP | "2025-06 远程传输" | 请求用 POST，可选的 SSE 用于服务器发起的消息；取代了旧的仅 SSE 传输。 |

## 进一步阅读

- [模型上下文协议规范](https://modelcontextprotocol.io/specification) — 规范参考，按日期版本化。
- [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers) — Filesystem、GitHub、Postgres、Slack、Puppeteer 参考服务器。
- [Anthropic — 介绍 MCP (2024 年 11 月)](https://www.anthropic.com/news/model-context-protocol) — 发布帖子与设计原理。
- [Python SDK](https://github.com/modelcontextprotocol/python-sdk) — 本课程使用的官方 SDK。
- [MCP 安全注意事项](https://modelcontextprotocol.io/docs/concepts/security) — roots、破坏性提示、工具中毒。
- [Google A2A 规范](https://google.github.io/A2A/) — Agent2Agent 协议；与 MCP 的 agent-to-tool 范围互补的 agent-to-agent 通信的兄弟标准。
- [Anthropic — 构建有效代理 (2024 年 12 月)](https://www.anthropic.com/research/building-effective-agents) — MCP 在更广泛的代理设计模式库中的位置（增强型 LLM、工作流、自主代理）。