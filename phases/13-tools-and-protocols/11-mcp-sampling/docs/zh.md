# MCP Sampling — 服务器请求的 LLM 补全与 Agent 循环

> 大多数 MCP 服务器都是“傻执行器”：接收参数、运行代码、返回内容。Sampling 让服务器可以反转方向：它请求客户端的 LLM 做决策。这使得服务器托管的 Agent 循环成为可能，而服务器不需要持有任何模型凭证。SEP-1577 于 2025-11-25 合并，向 sampling 请求中加入了工具，使循环可以包含更深入的推理。漂移风险提示：SEP-1577 的 tool-in-sampling 形态在 2026 年第一季度仍属实验性，并且 SDK API 仍在稳定中。

**Type:** Build
**Languages:** Python (stdlib, sampling harness)
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 10 (resources and prompts)
**Time:** ~75 minutes

## 学习目标

- 解释 `sampling/createMessage` 解决的问题（没有服务器端 API key 的服务器托管循环）。
- 实现一个服务器：它要求客户端对多轮提示进行 sampling，并返回补全结果。
- 使用 `modelPreferences`（成本 / 速度 / 智能优先级）来指导客户端选择模型。
- 构建一个 `summarize_repo` 工具，它通过 sampling 在内部迭代，而不是硬编码行为。

## 问题

用于代码摘要工作流的实用 MCP 服务器需要：遍历文件树、选择要读取的文件、综合生成摘要并返回。LLM 推理应该发生在哪里？

选项 A：服务器调用自己的 LLM。需要 API key，在服务器端计费，并且对每个用户都很昂贵。

选项 B：服务器返回原始内容；客户端的 Agent 负责推理。可行，但会把服务器逻辑移动到客户端提示中，这很脆弱。

选项 C：服务器通过 `sampling/createMessage` 请求客户端的 LLM。服务器保留算法（读哪些文件、做多少轮），客户端保留计费和模型选择权。服务器完全没有凭证。

Sampling 就是选项 C。它是一种机制，让受信任的服务器可以托管 Agent 循环，而自身不必成为完整的 LLM 托管方。

## 概念

### `sampling/createMessage` 请求

服务器发送：

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "sampling/createMessage",
  "params": {
    "messages": [{"role": "user", "content": {"type": "text", "text": "..."}}],
    "systemPrompt": "...",
    "includeContext": "none",
    "modelPreferences": {
      "costPriority": 0.3,
      "speedPriority": 0.2,
      "intelligencePriority": 0.5,
      "hints": [{"name": "claude-3-5-sonnet"}]
    },
    "maxTokens": 1024
  }
}
```

客户端运行它的 LLM，返回：

```json
{"jsonrpc": "2.0", "id": 42, "result": {
  "role": "assistant",
  "content": {"type": "text", "text": "..."},
  "model": "claude-3-5-sonnet-20251022",
  "stopReason": "endTurn"
}}
```

### `modelPreferences`

三个相加为 1.0 的浮点数：

- `costPriority`：偏好更便宜的模型。
- `speedPriority`：偏好更快的模型。
- `intelligencePriority`：偏好能力更强的模型。

再加上 `hints`：服务器偏好的具名模型。客户端可能会也可能不会遵循 hints；客户端的用户配置始终优先。

### `includeContext`

三个取值：

- `"none"` — 只使用服务器提供的消息。默认值。
- `"thisServer"` — 包含来自此服务器会话的先前消息。
- `"allServers"` — 包含所有会话上下文。

截至 2025-11-25，`includeContext` 已被软弃用，因为它会泄露跨服务器上下文，这是一个安全问题。优先使用 `"none"`，并在 messages 中传递显式上下文。

### 带工具的 Sampling（SEP-1577）

2025-11-25 新增：sampling 请求可以包含一个 `tools` 数组。客户端使用这些工具运行完整的工具调用循环。这让服务器可以通过客户端模型托管 ReAct 风格的 Agent 循环。

```json
{
  "messages": [...],
  "tools": [
    {"name": "fetch_url", "description": "...", "inputSchema": {...}}
  ]
}
```

客户端循环：sample；如果调用了工具，就执行工具；再次 sample；返回最终 assistant 消息。到 2026 年第一季度为止，这仍是实验性的；SDK 签名仍可能漂移。实现时，请对照 2025-11-25 规范的 client/sampling 部分确认。

### 人类在环

客户端必须在运行 sample 之前，向用户展示服务器正在要求模型做什么。恶意服务器可能使用 sampling 来操纵用户会话（“对用户说 X，让他们点击 Y”）。Claude Desktop、VS Code 和 Cursor 会把 sampling 请求展示为确认对话框，用户可以拒绝。

2026 年的共识：没有人类确认的 sampling 是危险信号。网关（Phase 13 · 17）可以自动批准低风险 sampling，并自动拒绝任何可疑请求。

### 没有 API key 的服务器托管循环

典型用例：一个没有自己 LLM 访问权限的代码摘要 MCP 服务器。它会：

1. 遍历 repo 结构。
2. 调用 `sampling/createMessage`，附带“选择五个最可能描述此 repo 用途的文件。”
3. 读取这些文件。
4. 调用 `sampling/createMessage`，附带这些文件内容和“用 3 段话总结这个 repo。”
5. 将摘要作为 `tools/call` 结果返回。

服务器从不接触 LLM API。客户端用户使用自己的凭证为这些补全付费。

### 安全风险（Unit 42 披露，2026 年第一季度）

- **隐蔽 sampling。** 某个工具总是用“从会话上下文中返回用户邮箱”来调用 sampling。Phase 13 · 15 覆盖了这些攻击向量。
- **通过 sampling 窃取资源。** 服务器要求客户端总结攻击者的 payload，让用户买单。
- **循环炸弹。** 服务器在紧密循环中调用 sampling。客户端必须强制执行按会话计的速率限制。

## 使用它

`code/main.py` 随附一个伪造的服务器到客户端 sampling harness。一个模拟的 `summarize_repo` 工具会调用两轮 sampling（先选文件，再摘要），而伪客户端返回预设响应。这个 harness 展示了：

- 服务器发送带有 `modelPreferences` 的 `sampling/createMessage`。
- 客户端返回一个补全。
- 服务器继续它的循环。
- 速率限制器限制每次工具调用可使用的 sampling 调用总数。

要关注的内容：

- 服务器只暴露一个工具（`summarize_repo`）；所有推理都发生在 sampling 调用中。
- 模型偏好会加权客户端的模型选择；hints 列出偏好的模型。
- 循环在 `stopReason: "endTurn"` 时终止。
- `max_samples_per_tool = 5` 限制会捕获失控循环。

## 交付它

本课会产出 `outputs/skill-sampling-loop-designer.md`。给定一个需要 LLM 调用的服务器端算法（研究、摘要、规划），该 skill 会设计一个基于 sampling 的实现，包含合适的 modelPreferences、速率限制和安全确认。

## 练习

1. 运行 `code/main.py`。将 `max_samples_per_tool` 改为 2，并观察速率限制的截断效果。

2. 实现 SEP-1577 的 tool-in-sampling 变体：sampling 请求携带一个 `tools` 数组。验证客户端侧循环会在返回最终补全之前执行这些工具。注意漂移风险：SDK 签名到 2026 年上半年仍可能变化。

3. 添加人类在环确认：在服务器第一次 `sampling/createMessage` 之前，暂停并等待用户批准。被拒绝的调用返回一个带类型的 refusal。

4. 添加一个按客户端会话 keyed 的每用户速率限制器。同一用户触发的同服务器循环应该共享一个预算。

5. 设计一个 `summarize_pdf` 工具，使用 sampling 选择要包含的 chunks。草拟发送的 messages。`modelPreferences.intelligencePriority` 在 0.1 与 0.9 时会如何改变行为？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Sampling | “服务器到客户端的 LLM 调用” | 服务器请求客户端模型生成补全 |
| `sampling/createMessage` | “这个方法” | 用于 sampling 请求的 JSON-RPC 方法 |
| `modelPreferences` | “模型优先级” | 成本 / 速度 / 智能权重，加上名称 hints |
| `includeContext` | “跨会话泄露” | 已软弃用的上下文包含模式 |
| SEP-1577 | “Sampling 中的工具” | 允许在 sampling 中放入工具，用于服务器托管的 ReAct |
| Human-in-the-loop | “用户确认” | 客户端在运行前向用户展示 sampling 请求 |
| Loop bomb | “失控 sampling” | 服务器端无限 sampling 循环；客户端必须限速 |
| Covert sampling | “隐藏推理” | 恶意服务器在 sampling prompts 中隐藏意图 |
| Resource theft | “使用用户的 LLM 预算” | 服务器强迫客户端为其不想要的 sampling 消费付费 |
| `stopReason` | “生成为何停止” | `endTurn`、`stopSequence` 或 `maxTokens` |

## 延伸阅读

- [MCP — Concepts: Sampling](https://modelcontextprotocol.io/docs/concepts/sampling) — sampling 的高层概览
- [MCP — Client sampling spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/client/sampling) — 标准 `sampling/createMessage` 形态
- [MCP — GitHub SEP-1577](https://github.com/modelcontextprotocol/modelcontextprotocol) — sampling 中工具的 Spec Evolution Proposal（实验性）
- [Unit 42 — MCP attack vectors](https://unit42.paloaltonetworks.com/model-context-protocol-attack-vectors/) — 隐蔽 sampling 与资源窃取模式
- [Speakeasy — MCP sampling core concept](https://www.speakeasy.com/mcp/core-concepts/sampling) — 带客户端代码示例的讲解
