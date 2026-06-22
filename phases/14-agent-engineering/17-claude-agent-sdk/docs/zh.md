# Claude Agent SDK：子代理与会话存储

> Claude Agent SDK 是 Claude Code 运行时的库形式。内置工具、用于上下文隔离的子代理、钩子、W3C trace 传播、会话存储对等。Claude Managed Agents 是托管的长时异步工作替代方案。

**类型：** 学习 + 构建
**语言：** Python（stdlib）
**前置条件：** 阶段 14 · 01（代理循环）、阶段 14 · 10（技能库）
**时间：** ~75 分钟

## 学习目标

- 解释 Anthropic Client SDK（原始 API）与 Claude Agent SDK（运行时形式）之间的区别。
- 描述子代理——并行化和上下文隔离——以及何时使用它们。
- 命名 Python SDK 的会话存储接口（`append`、`load`、`list_sessions`、`delete`、`list_subkeys`）以及 `--session-mirror` 的作用。
- 实现一个具有内置工具、子代理生成（使用隔离上下文）、生命周期钩子和会话存储的标准库运行时。

## 问题背景

原始的 LLM API 只提供一次往返交互。生产级代理需要工具执行、MCP 服务器、生命周期钩子、子代理生成、会话持久化和 trace 传播。Claude Agent SDK 将这种形式作为库提供——与 Claude Code 使用的相同运行时，暴露给自定义代理。

## 核心概念

### Client SDK 与 Agent SDK

- **Client SDK（`anthropic`）。** 原始 Messages API。你拥有循环、工具和状态。
- **Agent SDK（`claude-agent-sdk`）。** 内置工具执行、MCP 连接、钩子、子代理生成、会话存储。Claude Code 循环作为库。

### 内置工具

SDK 开箱即提供 10+ 个工具：文件读写、shell、grep、glob、网页抓取等。自定义工具通过标准的 tool-schema 接口注册。

### 子代理

Anthropic 记录的两种用途：

1. **并行化。** 并发运行独立工作。"为这 20 个模块中的每一个找到测试文件"就是 20 个并行子代理任务。
2. **上下文隔离。** 子代理使用自己的上下文窗口；只有结果返回给协调器。协调器的预算得到保留。

Python SDK 最近新增：`list_subagents()` 和 `get_subagent_messages()`，用于读取子代理转录。

### 会话存储

与 TypeScript 的协议对等：

- `append(session_id, message)` — 添加一轮对话。
- `load(session_id)` — 恢复对话。
- `list_sessions()` — 枚举会话。
- `delete(session_id)` — 级联删除子代理会话。
- `list_subkeys(session_id)` — 列出子代理键。

`--session-mirror`（CLI 标志）将转录实时镜像到外部文件，用于调试。

### 钩子

可注册的生命周期钩子：

- `PreToolUse`、`PostToolUse` — 控制或审计工具调用。
- `SessionStart`、`SessionEnd` — 设置和清理。
- `UserPromptSubmit` — 在模型看到用户输入之前处理。
- `PreCompact` — 在上下文压缩之前运行。
- `Stop` — 代理退出时清理。
- `Notification` — 旁路警报。

钩子是 pro-workflow（阶段 14 课程参考）和类似系统添加横切关注点行为的方式。

### W3C trace 上下文

调用方上活跃的 OTel span 通过 W3C trace context 头传播到 CLI 子进程。整个多进程 trace 在你的后端显示为一条 trace。

### Claude Managed Agents

托管替代方案（beta header `managed-agents-2026-04-01`）。长时异步工作、内置提示缓存、内置压缩。用管理基础设施换取控制权。

### 该模式出错的地方

- **子代理过度生成。** 为 100 个小任务生成 100 个子代理。开销占主导。应批量处理。
- **钩子蔓延。** 每个团队都添加钩子；启动时间膨胀。每季度审查钩子。
- **会话膨胀。** 会话累积；大小增长。使用 `list_sessions` + 过期策略。

## 构建它

`code/main.py` 在标准库中实现了 SDK 形式：

- `Tool`、`ToolRegistry`，带有内置的 `read_file`、`write_file`、`list_dir`。
- `Subagent` — 私有上下文、隔离运行、返回结果。
- `SessionStore` — append、load、list、delete、list_subkeys。
- `Hooks` — `pre_tool_use`、`post_tool_use`、`session_start`、`session_end`。
- 一个演示：主代理并行生成 3 个子代理（每个隔离），聚合结果，持久化会话。

运行它：

```
python3 code/main.py
```

trace 显示了子代理上下文隔离（协调器上下文大小保持有界）、钩子执行和会话持久化。

## 使用它

- **Claude Agent SDK** 用于以 Claude 为中心的产品，需要 Claude Code 运行时形式。
- **Claude Managed Agents** 用于托管的长时异步工作。
- **OpenAI Agents SDK**（第 16 课）用于以 OpenAI 为中心的对应方案。
- **LangGraph + 自定义工具** 如果你想要图形式的状态机。

## 交付它

`outputs/skill-claude-agent-scaffold.md` 提供了一个 Claude Agent SDK 应用的脚手架，包含子代理、钩子、会话存储、MCP 服务器附加和 W3C trace 传播。

## 练习

1. 添加一个子代理生成器，将 20 个任务批量分成每组 5 个并行子代理。测量协调器上下文大小与每个任务一个子代理的对比。
2. 实现一个 `PreToolUse` 钩子，对 `write_file` 调用进行速率限制（每个会话每分钟 5 次）。跟踪行为。
3. 连接 `list_subkeys` 以渲染子代理树。深度嵌套是什么样子的？
4. 将这个玩具实现移植到真实的 `claude-agent-sdk` Python 包。工具注册有什么变化？
5. 阅读 Claude Managed Agents 文档。什么时候你会从自托管切换到托管？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| Agent SDK | "Claude Code 作为库" | 运行时形式：工具、MCP、钩子、子代理、会话存储 |
| 子代理 | "子代理" | 独立上下文、自己的预算；结果向上传递 |
| 会话存储 | "对话数据库" | 持久化、加载、列表、删除对话轮次，带子代理级联 |
| 钩子 | "生命周期回调" | 工具前后、会话、提示提交、压缩、停止 |
| W3C trace context | "跨进程 trace" | 父 span 传播到 CLI 子进程 |
| Managed Agents | "托管运行时" | Anthropic 托管的长时异步工作 |
| `--session-mirror` | "转录镜像" | 在对话轮次流式传输时写入外部文件 |
| MCP 服务器 | "工具表面" | 附加到代理的外部工具/资源源 |

## 延伸阅读

- [Claude Agent SDK 概览](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude Code 的库形式
- [Anthropic，使用 Claude Agent SDK 构建代理](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — 生产级模式
- [Claude Managed Agents 概览](https://platform.claude.com/docs/en/managed-agents/overview) — 托管替代方案
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 对应方案
