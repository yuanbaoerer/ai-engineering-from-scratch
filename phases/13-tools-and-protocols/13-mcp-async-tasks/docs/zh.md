# 异步任务（SEP-1686）——长时间运行工作的“先调用、后获取”模式

> 真正的智能体工作往往需要几分钟到几小时：CI 运行、深度研究综合、批量导出。同步工具调用会断开连接、超时，或阻塞 UI。SEP-1686 于 2025-11-25 合并，新增了 Tasks 原语：任何请求都可以被增强为一个任务，结果可以稍后获取，也可以通过状态通知流式接收。漂移风险提示：Tasks 到 2026 年上半年仍处于实验阶段；SDK 表面仍在围绕规范设计中。

**类型：** 构建
**语言：** Python（stdlib，异步任务状态机）
**先修要求：** Phase 13 · 07（MCP server），Phase 13 · 09（transports）
**时间：** 约 75 分钟

## 学习目标

- 识别何时应将工具从同步调用升级为任务增强（服务端工作 >30 秒）。
- 走通任务生命周期：`working` → `input_required` → `completed` / `failed` / `cancelled`。
- 持久化任务状态，确保崩溃不会丢失进行中的工作。
- 正确轮询 `tasks/status` 并获取 `tasks/result`。

## 问题

一个 `generate_report` 工具会运行耗时数分钟的抽取流水线。在同步模型下可选方案包括：

1. 让连接保持打开三分钟。远程传输会断开；客户端会超时；UI 会冻结。
2. 立即返回一个占位结果；要求客户端轮询自定义端点。这会破坏 MCP 的统一性。
3. 即发即忘；没有结果。

这些都不理想。SEP-1686 增加了第四种方案：任务增强（task augmentation）。任何请求（通常是 `tools/call`）都可以被标记为任务。服务器会立即返回一个任务 id。客户端轮询 `tasks/status`，并在完成后获取 `tasks/result`。服务端状态在重启后仍会保留。

## 概念

### 任务增强

通过设置 `params._meta.task.required: true`（或 `optional: true`，由服务器决定），请求会变成一个任务。服务器立即响应：

```json
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "_meta": {
      "task": {
        "id": "tsk_9f7b...",
        "state": "working",
        "ttl": 900000
      }
    }
  }
}
```

`ttl` 是服务器承诺保留状态的时间；超过 ttl 后，任务结果会被丢弃。

### 按工具选择启用

工具注解可以声明任务支持：

- `taskSupport: "forbidden"` —— 该工具始终同步运行。适合快速工具。
- `taskSupport: "optional"` —— 客户端可以请求任务增强。
- `taskSupport: "required"` —— 客户端必须使用任务增强。

`generate_report` 工具应设为 `required`。`notes_search` 工具应设为 `forbidden`。

### 状态

```
working  -> input_required -> working  (loop via elicitation)
working  -> completed
working  -> failed
working  -> cancelled
```

状态机是只追加的：一旦进入 `completed`、`failed` 或 `cancelled`，任务就是终态。

### 方法

- `tasks/status {taskId}` —— 返回当前状态和进度提示。
- `tasks/result {taskId}` —— 如果尚未完成，则阻塞或返回 404。
- `tasks/cancel {taskId}` —— 幂等；终态会被忽略。
- `tasks/list` —— 可选；枚举活跃和最近完成的任务。

### 流式传输状态变更

当服务器支持时，客户端可以订阅状态通知：

```
server -> notifications/tasks/updated {taskId, state, progress?}
```

相比轮询，使用流式传输的客户端能获得更好的用户体验。轮询始终作为最小能力面被支持。

### 持久化状态

规范要求声明任务支持的服务器必须持久化状态。崩溃不应导致 ttl 内已完成的结果丢失。存储可以从 SQLite、Redis 到文件系统不等。Lesson 13 harness 使用文件系统。

### 取消语义

`tasks/cancel` 是幂等的。如果任务正在执行中，服务器会尝试停止它（检查执行器协作式取消）。如果已经处于终态，该请求就是空操作。

### 崩溃恢复

当服务器进程重启时：

1. 加载所有已持久化的任务状态。
2. 将任何进程已死亡的 `working` 任务标记为 `failed`，错误为 `CRASH_RECOVERY`。
3. 在各自 ttl 内保留 `completed` / `failed` / `cancelled`。

### 异步任务加采样

任务本身也可以调用 `sampling/createMessage`。这正是长时间运行的研究任务的工作方式：服务器的任务线程按需对客户端模型进行采样，而客户端 UI 将任务显示为 `working`，并展示周期性进度更新。

### 为什么这仍是实验性的

SEP-1686 已于 2025-11-25 发布，但更广泛的路线图指出了三个未解决问题：持久订阅原语、子任务（父子任务关系）以及结果 TTL 标准化。预计该规范会在 2026 年持续演进。生产代码应只把 Tasks 的常见场景视为稳定，并针对未来 SDK 在子任务方面的变化做好防护。

## 使用它

`code/main.py` 实现了一个持久化任务存储（基于文件系统），以及一个在后台线程中运行的 `generate_report` 工具。客户端调用该工具后会立即获得任务 id，在 worker 更新进度时轮询 `tasks/status`，并在完成后获取 `tasks/result`。取消可用；通过杀死 worker 线程并重新加载状态来模拟崩溃恢复。

需要关注的内容：

- 任务状态 JSON 会持久化到 `/tmp/lesson-13-tasks/<id>.json`。
- Worker 线程会更新 `progress` 字段；轮询会显示它在推进。
- 客户端侧取消会设置一个 event；worker 检查后提前退出。
- “崩溃”时重新加载状态，会把进行中的任务标记为 `failed`，并带有 `CRASH_RECOVERY`。

## 交付它

本课会产出 `outputs/skill-task-store-designer.md`。给定一个长时间运行的工具（research、build、export），该 skill 会设计任务存储（状态形状、ttl、持久性），选择正确的 taskSupport 标志，并草拟进度通知。

## 练习

1. 运行 `code/main.py`。启动一个 `generate_report` 任务，轮询状态，然后获取结果。

2. 在运行中途添加一次 `tasks/cancel` 调用。验证 worker 会遵守它，并且状态变为 `cancelled`。

3. 模拟崩溃恢复：杀死 worker 线程，重启加载器，并观察 `CRASH_RECOVERY` 失败模式。

4. 将存储扩展到 SQLite。持久性收益相同；查询选项会变多（列出 session X 的所有任务）。

5. 阅读 2026 年 MCP 路线图文章。找出最可能在下一年影响 SDK API 设计的一个 Tasks 相关开放问题。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Task | “长时间运行的工具调用” | 使用 `_meta.task` 增强以进行异步执行的请求 |
| SEP-1686 | “Tasks spec” | 在 2025-11-25 添加 Tasks 的 Spec Evolution Proposal |
| `_meta.task` | “Task envelope” | 包含 id、state、ttl 的逐请求元数据 |
| taskSupport | “Tool flag” | 每个工具的 `forbidden` / `optional` / `required` |
| `tasks/status` | “Poll method” | 获取当前状态和可选进度提示 |
| `tasks/result` | “Fetch result” | 返回已完成 payload；如果尚未完成则返回 404 |
| `tasks/cancel` | “Stop it” | 幂等取消请求 |
| ttl | “Retention budget” | 服务器承诺保留任务状态的毫秒数 |
| `notifications/tasks/updated` | “State push” | 服务器发起的状态变更事件 |
| Durable store | “Crash-safe state” | 文件系统 / SQLite / Redis 持久化层 |

## 延伸阅读

- [MCP — GitHub SEP-1686 issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1686) —— 原始提案和完整讨论
- [WorkOS — MCP async tasks for AI agent workflows](https://workos.com/blog/mcp-async-tasks-ai-agent-workflows) —— 带有设计理由的设计 walkthrough
- [DeepWiki — MCP task system and async operations](https://deepwiki.com/modelcontextprotocol/modelcontextprotocol/2.7-task-system-and-async-operations) —— 机制和状态机
- [FastMCP — Tasks](https://gofastmcp.com/servers/tasks) —— SDK 层面的任务实现模式
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) —— 开放问题和 2026 年优先事项，包括子任务
