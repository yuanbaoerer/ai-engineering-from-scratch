# 构建 MCP 客户端 — 发现、调用、会话管理

> 大多数 MCP 内容都会提供服务器教程，却对客户端一笔带过。真正困难的编排工作都在客户端代码里：进程启动、能力协商、跨多个服务器合并工具列表、采样回调（sampling callback）、重连，以及命名空间冲突解决。本课将构建一个多服务器客户端，把三个不同的 MCP 服务器提升到一个扁平的工具命名空间中，供模型使用。

**类型：** 构建
**语言：** Python（标准库，多服务器 MCP 客户端）
**先修要求：** 第 13 阶段 · 07（构建 MCP 服务器）
**时间：** 约 75 分钟

## 学习目标

- 将 MCP 服务器作为子进程启动，完成 `initialize`，并发送 `notifications/initialized`。
- 维护每个服务器的会话状态（能力、工具列表、最近看到的通知 id）。
- 将多个服务器的工具列表合并到一个命名空间中，并处理冲突。
- 将工具调用路由到拥有该工具的服务器，并重新组装响应。

## 问题

真实的 agent host（Claude Desktop、Cursor、Goose、Gemini CLI）会一次加载多个 MCP 服务器。用户可能同时运行文件系统服务器、Postgres 服务器和 GitHub 服务器。客户端的工作是：

1. 启动每个服务器。
2. 分别完成每个服务器的握手。
3. 对每个服务器调用 `tools/list`，并将结果扁平化。
4. 当模型发出 `notes_search` 时，在合并后的命名空间中查找它，并路由到正确的服务器。
5. 处理来自任意服务器的通知（`tools/list_changed`），且不能阻塞。
6. 在传输失败时重连。

手写这一整套逻辑，正是“玩具项目”和“可用系统”的分水岭。官方 SDK 会封装这些细节，但心智模型必须掌握在你自己手里。

## 概念

### 子进程启动

使用 `subprocess.Popen`，并设置 `stdin=PIPE, stdout=PIPE, stderr=PIPE`。设置 `bufsize=1`，并使用文本模式逐行读取。每个服务器都是一个进程；客户端为每个服务器持有一个 `Popen` 句柄。

### 每个服务器的会话状态

每个服务器对应一个 `Session` 对象，其中包含：

- `process` — Popen 句柄。
- `capabilities` — 服务器在 `initialize` 时声明的能力。
- `tools` — 最近一次 `tools/list` 的结果。
- `pending` — 从请求 id 到正在等待响应的 promise/future 的映射。

请求天生是异步的；当服务器 B 正在调用过程中时，发往服务器 A 的 `tools/call` 不能被阻塞。可以使用带队列的线程，也可以使用 asyncio。

### 合并命名空间

当客户端看到聚合后的工具列表时，名称可能会冲突。两个服务器可能都暴露 `search`。客户端有三种选择：

1. **按服务器名称加前缀。** `notes/search`、`files/search`。清晰但不够美观。
2. **静默先到先得。** 后加载服务器的 `search` 覆盖先前的。风险很高；会隐藏冲突。
3. **拒绝冲突。** 拒绝加载第二个服务器；通知用户。对安全敏感的宿主来说最安全。

Claude Desktop 使用按服务器加前缀。Cursor 使用冲突拒绝，并给出清晰错误。VS Code MCP 也采用按服务器加前缀。

### 路由

合并之后，调度表会将 `tool_name -> session` 映射起来。模型按名称发出调用；客户端找到对应 session，并向该服务器的 stdin 写入一条 `tools/call` 消息，然后等待响应。

### 采样回调

如果服务器在 `initialize` 时声明了 `sampling` 能力，它可能会发送 `sampling/createMessage`，要求客户端运行自己的 LLM。客户端必须：

1. 在采样完成之前阻塞对该服务器的后续请求，或者在实现支持并发时进行流水线处理。
2. 调用自己的 LLM provider。
3. 将响应发回服务器。

第 11 课会端到端覆盖采样。本课为了完整性只提供桩实现。

### 通知处理

`notifications/tools/list_changed` 表示需要重新调用 `tools/list`。`notifications/resources/updated` 表示如果该资源正在使用中，就需要重新读取它。通知不能产生响应——不要尝试确认（ack）它们。

一个常见的客户端 bug：在 `tools/call` 上阻塞读取循环，而通知还停留在流里。使用一个后台读取线程，把 stdout 上的每一行都推入队列；主线程出队并分派处理。

### 重连

传输可能失败：服务器崩溃、进程被操作系统杀掉、stdio 管道断开。客户端在 stdout 上检测到 EOF，并将该会话视为已死亡。可选策略：

- 静默重启服务器并重新握手。适合纯只读服务器。
- 将失败暴露给用户。适合带有用户可见会话的有状态服务器。

第 13 阶段 · 09 会覆盖 Streamable HTTP 的重连语义；stdio 更简单。

### Keepalive 和 session id

Streamable HTTP 使用 `Mcp-Session-Id` header。Stdio 没有 session id——进程身份本身就是会话。keepalive ping 是可选的；stdio 管道不会因为空闲而断开。

## 使用它

`code/main.py` 会把三个模拟 MCP 服务器作为子进程启动，分别完成握手，合并它们的工具列表，并把工具调用路由到正确的服务器。这些“服务器”其实是运行玩具响应器的其他 Python 进程（没有真实 LLM）。运行它可以看到：

- 三次初始化，每个服务器都有自己的能力集。
- 三个 `tools/list` 结果被合并为一个包含 7 个工具的命名空间。
- 基于工具名称做出的路由决策。
- 通过命名空间前缀防止了一次冲突。

需要重点查看：

- `Session` dataclass 干净地保存了每个服务器的状态。
- 后台读取线程会在不阻塞主线程的情况下出队 stdout 上的每一行。
- 调度表只是一个简单的 `dict[str, Session]`。
- 冲突处理是显式的：当两个服务器声明同名工具时，后者会用前缀重命名。

## 交付它

本课会产出 `outputs/skill-mcp-client-harness.md`。给定一份声明式 MCP 服务器列表（名称、命令、参数），该 skill 会生成一个 harness：启动这些服务器、合并工具列表，并交付一个带冲突解决的路由函数。

## 练习

1. 运行 `code/main.py` 并观察服务器启动日志。用 SIGTERM 杀掉其中一个模拟服务器进程，观察客户端如何检测 EOF 并将该会话标记为死亡。

2. 实现命名空间前缀。当两个服务器暴露 `search` 时，将第二个重命名为 `<server>/search`。更新调度表，并验证工具调用能正确路由。

3. 为服务器重启添加类似连接池的退避机制：连续失败时使用指数退避，上限为 30 秒，三次失败后向用户发出通知。

4. 草拟一个支持 100 个并发 MCP 服务器的客户端。什么数据结构会替代简单的调度 dict？（提示：用于前缀命名空间的 trie，再加上每个服务器工具数量的指标。）

5. 将客户端移植到官方 MCP Python SDK。SDK 封装了 `stdio_client` 和 `ClientSession`。在保留多服务器路由的同时，代码应该从约 200 行缩减到约 40 行。

## 关键术语

| 术语 | 人们通常的说法 | 实际含义 |
|------|----------------|----------|
| MCP client | “agent host” | 启动服务器并编排工具调用的进程 |
| Session | “每个服务器的状态” | 能力、工具列表和待处理请求的记账信息 |
| Merged namespace | “一个工具列表” | 跨所有活跃服务器的扁平工具名称集合 |
| Namespace collision | “两个服务器同名工具” | 客户端必须对重复项加前缀、拒绝，或先到先得 |
| Routing | “这个调用给谁？” | 从工具名称调度到拥有它的服务器 |
| Background reader | “非阻塞 stdout” | 将服务器 stdout 排空到队列中的线程或任务 |
| Sampling callback | “LLM-as-a-service” | 客户端对服务器发出的 `sampling/createMessage` 的处理器 |
| `notifications/*_changed` | “原语发生变化” | 表示客户端必须重新发现或重新读取的信号 |
| Reconnection policy | “服务器死亡时” | 传输失败时的重启语义 |
| Stdio session | “进程 = 会话” | 没有 session id；子进程生命周期就是会话 |

## 延伸阅读

- [Model Context Protocol — Client spec](https://modelcontextprotocol.io/specification/2025-11-25/client) — 规范性的客户端行为
- [MCP — Quickstart client guide](https://modelcontextprotocol.io/quickstart/client) — 使用 Python SDK 的 hello-world 客户端教程
- [MCP Python SDK — client module](https://github.com/modelcontextprotocol/python-sdk) — `ClientSession` 和 `stdio_client` 参考
- [MCP TypeScript SDK — Client](https://github.com/modelcontextprotocol/typescript-sdk) — TS 对应实现
- [VS Code — MCP in extensions](https://code.visualstudio.com/api/extension-guides/ai/mcp) — VS Code 如何在单个编辑器宿主中复用多个 MCP 服务器
