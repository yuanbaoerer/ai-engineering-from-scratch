# MCP 资源与提示词 — 超越工具的上下文暴露

> 工具（Tools）拿走了 MCP 90% 的关注度。另外两个服务器原语（server primitives）解决的是不同问题。资源（Resources）暴露可读取的数据；提示词（Prompts）把可复用模板暴露为斜杠命令（slash-commands）。许多服务器应该使用资源，而不是把读取操作包装成工具；也应该使用提示词，而不是把工作流硬编码进客户端提示词。本课给出决策规则，并走查 `resources/*` 与 `prompts/*` 消息。

**类型：** 构建
**语言：** Python（stdlib，资源 + 提示词处理器）
**先修：** 第 13 阶段 · 07（MCP 服务器）
**时间：** 约 45 分钟

## 学习目标

- 针对给定领域，判断应将能力暴露为工具、资源还是提示词。
- 实现 `resources/list`、`resources/read`、`resources/subscribe`，并处理 `notifications/resources/updated`。
- 使用参数模板实现 `prompts/list` 和 `prompts/get`。
- 识别主机（host）何时将提示词呈现为斜杠命令，而不是自动注入的上下文。

## 问题

一个为笔记应用编写的朴素 MCP 服务器会把所有东西都暴露成工具：`notes_read`、`notes_list`、`notes_search`。这会把每一次数据访问都包装成由模型驱动的工具调用。后果包括：

- 对于每个可能受益于上下文的查询，模型都必须决定是否调用 `notes_read`。
- 只读内容无法被订阅，也无法流式显示到主机的侧边面板。
- 客户端 UI（Claude Desktop 的资源附件面板、Cursor 的 “Include file” 选择器）无法呈现这些数据。

正确的拆分方式：把数据暴露为资源，把会修改状态或需要计算的动作暴露为工具，把可复用的多步骤工作流暴露为提示词。每种原语都有自己的 UX 能力和访问模式。

## 概念

### 工具 vs 资源 vs 提示词 — 决策规则

| 能力 | 原语 |
|------------|-----------|
| 用户想搜索、过滤或转换数据 | tool |
| 用户想让主机把这些数据作为上下文纳入 | resource |
| 用户想要一个可重复运行的模板化工作流 | prompt |

指导原则：如果模型会在每个相关查询中受益于调用它，那么它就是工具。如果用户会受益于把它附加到一段对话中，那么它就是资源。如果用户想复用的单位是整个多步骤工作流，那么它就是提示词。

### 资源

`resources/list` 返回 `{resources: [{uri, name, mimeType, description?}]}`。`resources/read` 接收 `{uri}` 并返回 `{contents: [{uri, mimeType, text | blob}]}`。

URI 可以是任何可寻址的内容：

- `file:///Users/alice/notes/mcp.md`
- `postgres://my-db/query/SELECT ...`
- `notes://note-14`（自定义 scheme）
- `memory://session-2026-04-22/recent`（服务器特定）

`contents[]` 同时支持文本和二进制。二进制使用 `blob` 作为 base64 编码字符串，并带有 `mimeType`。

### 资源订阅

在 capabilities 中声明 `{resources: {subscribe: true}}`。客户端调用 `resources/subscribe {uri}`。资源变化时，服务器发送 `notifications/resources/updated {uri}`。客户端重新读取。

使用场景：一个笔记服务器的资源是磁盘上的文件；文件监听器触发更新通知；当文件在主机外部被编辑时，Claude Desktop 会把该文件重新拉取进上下文。

### 资源模板（2025-11-25 新增）

`resourceTemplates` 让你暴露一个参数化 URI 模式：`notes://{id}`，其中 `id` 是补全目标。客户端可以在资源选择器中自动补全 ids。

### 提示词

`prompts/list` 返回 `{prompts: [{name, description, arguments?}]}`。`prompts/get` 接收 `{name, arguments}` 并返回 `{description, messages: [{role, content}]}`。

提示词是一个模板，会填充成一组消息，主机再把这些消息喂给其模型。例如，`code_review` 提示词接收 `file_path` 参数，并返回三条消息序列：一条 system 消息、一条带有文件正文的 user 消息，以及一条带有推理模板的 assistant 启动消息。

### 主机与提示词

Claude Desktop、VS Code 和 Cursor 会在聊天 UI 中把提示词暴露为斜杠命令。用户输入 `/code_review`，并从表单中选择参数。服务器的提示词就是“用户快捷方式”和“发送给模型的完整提示词”之间的契约。

并非每个客户端目前都支持提示词——请检查能力协商（capability negotiation）。如果服务器声明了提示词能力，但客户端不支持提示词，那么客户端就不会看到这些斜杠命令。

### “列表已变化”通知

资源和提示词都会在集合发生变化时发出 `notifications/list_changed`。一个刚导入 20 条新笔记的笔记服务器会发出 `notifications/resources/list_changed`；客户端重新调用 `resources/list` 来获取新增项。

### 内容类型约定

对于文本：`mimeType: "text/plain"`、`text/markdown`、`application/json`。
对于二进制：`image/png`、`application/pdf`，再加上 `blob` 字段。
对于 MCP Apps（第 14 课）：在 `ui://` URI 中使用 `text/html;profile=mcp-app`。

### 动态资源

资源 URI 不一定对应某个静态文件。`notes://recent` 可以在每次读取时返回最新的五条笔记。`db://query/users/active` 可以执行参数化查询。服务器可以自由地动态计算内容。

规则：如果客户端可以按 URI 缓存，那么 URI 必须稳定。如果计算是一次性的，那么 URI 应该包含时间戳或 nonce，避免客户端缓存变旧。

### 订阅 vs 轮询

支持订阅的客户端通过 `notifications/resources/updated` 获得服务器推送。订阅之前的客户端，或不支持订阅的主机，则通过重新读取来轮询。两者都符合规范。服务器的能力声明会告诉客户端它支持哪一种。

订阅的成本：服务器上需要维护每个会话的状态（谁订阅了什么）。保持订阅集合有界；已断开的客户端应该超时。

### 提示词 vs 系统提示词

MCP 中的提示词不是系统提示词。主机的系统提示词（它自己的运行指令）和 MCP 提示词（用户调用的服务器提供模板）并行存在。行为良好的客户端绝不会让服务器提示词覆盖它自己的系统提示词；它会把它们分层叠加。

## 使用它

`code/main.py` 在第 07 课的笔记服务器基础上扩展了：

- 每条笔记对应的资源（`notes://note-1` 等），并支持 `resources/subscribe`。
- 一个 `review_note` 提示词，会渲染成三条消息的模板。
- 一个文件监听器模拟：当笔记被修改时发出 `notifications/resources/updated`。
- 一个 `notes://recent` 动态资源，总是返回最新的五条笔记。

运行演示以查看完整流程。

## 交付它

本课会产出 `outputs/skill-primitive-splitter.md`。给定一个拟议的 MCP 服务器，这个 skill 会把每项能力分类为 tool / resource / prompt，并给出理由。

## 练习

1. 运行 `code/main.py`。观察初始资源列表，然后触发一次笔记编辑，并验证 `notifications/resources/updated` 事件被触发。

2. 添加一个 `resources/list_changed` 发射器：创建新笔记时发送通知，让客户端重新发现。

3. 为 GitHub MCP 服务器设计三个提示词：`summarize_pr`、`triage_issue`、`release_notes`。每个都带有参数 schema。提示词正文应无需进一步编辑即可运行。

4. 取第 07 课服务器中的一个现有工具，并判断它应该继续保留为工具，还是拆分成资源 + 工具的组合。用一句话说明理由。

5. 阅读规范中的 `server/resources` 和 `server/prompts` 章节。找出 `resources/read` 中很少被填充、但规范支持的那个字段。提示：查看资源内容上的 `_meta`。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| Resource | “暴露的数据” | 主机可以读取的、可通过 URI 寻址的内容 |
| Resource URI | “指向数据的指针” | 带 scheme 前缀的标识符（`file://`、`notes://` 等） |
| `resources/subscribe` | “监听变化” | 客户端选择订阅的、针对特定 URI 的服务器推送更新 |
| `notifications/resources/updated` | “资源已变化” | 通知客户端某个已订阅资源有新内容的信号 |
| Resource template | “参数化 URI” | 带有主机选择器补全提示的 URI 模式 |
| Prompt | “斜杠命令模板” | 带参数槽位的、命名的多消息模板 |
| Prompt arguments | “模板输入” | 主机在渲染前收集的类型化参数 |
| `prompts/get` | “渲染模板” | 服务器返回填充后的消息列表 |
| Content block | “类型化片段” | `{type: text \| image \| resource \| ui_resource}` |
| Slash-command UX | “用户快捷方式” | 主机把提示词呈现为以 `/` 开头的命令 |

## 延伸阅读

- [MCP — Concepts: Resources](https://modelcontextprotocol.io/docs/concepts/resources) — 资源 URI、订阅和模板
- [MCP — Concepts: Prompts](https://modelcontextprotocol.io/docs/concepts/prompts) — 提示词模板和斜杠命令集成
- [MCP — Server resources spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/resources) — 完整的 `resources/*` 消息参考
- [MCP — Server prompts spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts) — 完整的 `prompts/*` 消息参考
- [MCP — Protocol info site: resources](https://modelcontextprotocol.info/docs/concepts/resources/) — 对官方文档进行扩展的社区指南
