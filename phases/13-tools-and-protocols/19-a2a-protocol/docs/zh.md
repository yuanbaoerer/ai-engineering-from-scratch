# A2A — Agent-to-Agent 协议

> MCP 是 agent-to-tool（智能体到工具）。A2A（Agent2Agent）是 agent-to-agent（智能体到智能体）——一个开放协议，用于让基于不同框架构建的黑盒智能体协同工作。它由 Google 于 2025 年 4 月发布，2025 年 6 月捐赠给 Linux Foundation，并在 2026 年 4 月达到 v1.0，获得包括 AWS、Cisco、Microsoft、Salesforce、SAP 和 ServiceNow 在内的 150 多家支持者。它吸收了 IBM 的 ACP，并加入了 AP2 支付扩展。本课将介绍 Agent Card、Task 生命周期，以及两种传输绑定。

**类型：** 构建
**语言：** Python（标准库，Agent Card + Task 测试框架）
**先修要求：** Phase 13 · 06（MCP 基础），Phase 13 · 08（MCP 客户端）
**时间：** 约 75 分钟

## 学习目标

- 区分 agent-to-tool（MCP）与 agent-to-agent（A2A）的使用场景。
- 在 `/.well-known/agent.json` 发布包含技能和端点元数据的 Agent Card。
- 走通 Task 生命周期（submitted → working → input-required → completed / failed / canceled / rejected）。
- 使用带有 Parts（text、file、data）的 Messages，并将 Artifacts 作为输出。

## 问题

一个客服智能体需要把报告撰写委托给一个专门的写作智能体。在 A2A 出现之前，有这些选择：

- 自定义 REST API。可行，但每一组配对都是一次性的。
- 共享代码库。要求两个智能体运行在同一个框架上。
- MCP。不合适：MCP 用于调用工具，而不是让两个智能体在保留各自黑盒内部推理的同时进行协作。

A2A 填补了这个空白。它把交互建模为一个智能体向另一个智能体发送一个 Task，并包含生命周期、消息和产物。被调用智能体的内部状态保持不透明——调用方只能看到任务状态转换和最终输出。

A2A 是“让跨框架智能体彼此通信”的协议。它不会取代 MCP；两者是互补的。

## 核心概念

### Agent Card

每个符合 A2A 的智能体都会在 `/.well-known/agent.json` 发布一张卡片：

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

发现机制基于 URL：获取这张卡片，了解 A2A 端点的 URL，并枚举技能。

### 签名 Agent Cards（AP2）

AP2 扩展（2025 年 9 月）为 Agent Cards 增加了密码学签名。发布者用 JWT 为自己的卡片签名；消费者进行验证。这可以防止冒充。

### Task 生命周期

```
submitted -> working -> completed | failed | canceled | rejected
            -> input_required -> working (loop via message)
```

客户端用 `tasks/send` 发起请求。被调用智能体会在各状态之间转换；客户端通过 SSE 订阅状态更新，或进行轮询。

### Messages 和 Parts

一条 message 携带一个或多个 Parts：

- `text` — 普通内容。
- `file` — 带 mimeType 的 base64 blob。
- `data` — 类型化 JSON 载荷（给被调用智能体的结构化输入）。

示例：

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### Artifacts

输出是 Artifacts，而不是原始字符串。Artifact 是一个具名、带类型的输出：

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

Artifacts 可以以分块形式流式传输。调用方负责累积这些分块。

### 两种传输绑定

1. **HTTP 上的 JSON-RPC。** `/a2a` 端点，POST 用于请求，可选 SSE 用于流式传输。默认绑定。
2. **gRPC。** 用于原生采用 gRPC 的企业环境。

两种绑定承载相同的逻辑消息形状。

### 保持不透明性

一个关键设计原则：被调用智能体的内部状态是不透明的。调用方看到的是任务状态和 artifacts。被调用智能体的 chain-of-thought、它的工具调用、它对子智能体的委托——全部不可见。这不同于 MCP，在 MCP 中工具调用是透明的。

理由：A2A 允许竞争对手在不暴露内部实现的情况下协作。A2A 可以表达“调用这个客服智能体”，而调用方无需知道该智能体如何实现服务。

### 时间线

- **2025-04-09。** Google 宣布 A2A。
- **2025-06-23。** 捐赠给 Linux Foundation。
- **2025-08。** 吸收 IBM 的 ACP。
- **2025-09。** AP2 扩展（Agent Payments）发布。
- **2026-04。** v1.0 发布，获得 150 多家组织支持。

### 与 MCP 的关系

| 维度 | MCP | A2A |
|-----------|-----|-----|
| 使用场景 | Agent-to-tool | Agent-to-agent |
| 不透明性 | 透明的工具调用 | 不透明的内部推理 |
| 典型调用方 | 智能体运行时 | 另一个智能体 |
| 状态 | 工具调用结果 | 带生命周期的 Task |
| 授权 | OAuth 2.1（Phase 13 · 16） | JWT 签名的 Agent Cards（AP2） |
| 传输 | Stdio / Streamable HTTP | HTTP 上的 JSON-RPC / gRPC |

当你想调用一个特定工具时，使用 MCP。当你想把一个完整任务委托给另一个智能体时，使用 A2A。许多生产系统会同时使用两者：一个智能体用 MCP 构建它的工具层，用 A2A 构建它的协作层。

## 使用它

`code/main.py` 实现了一个最小 A2A 测试框架：一个研究智能体发布自己的卡片，一个写作智能体接收带有 parts 的 `tasks/send`，其中包括一份 PDF 和一条文本指令，然后经历 working → input_required → working → completed，并返回一个文本 artifact。全部使用标准库；它使用内存传输以便聚焦消息形状。

需要关注的内容：

- Agent Card JSON 形状。
- Task id 分配和状态转换。
- 包含混合类型 parts 的 Messages。
- 任务中途的 input-required 分支。
- 完成时返回的 Artifact。

## 交付它

本课会产出 `outputs/skill-a2a-agent-spec.md`。给定一个应当可被其他智能体调用的新智能体，该技能会产出 Agent Card JSON、skills schema 和端点蓝图。

## 练习

1. 运行 `code/main.py`。追踪完整的 Task 生命周期，包括被调用智能体请求澄清时出现的 input-required 暂停。

2. 添加一个签名 Agent Card。对卡片的规范 JSON 使用 HMAC 签名。编写一个验证器，并确认它在卡片被篡改时会失败。

3. 实现任务流式传输：写作智能体通过 SSE 发出三个增量 artifact 分块，调用方累积它们。

4. 设计一个封装 MCP 服务器的 A2A 智能体。把每个 MCP tool 映射为一个 A2A skill。说明其中的权衡——会丢失什么不透明性？

5. 阅读 A2A v1.0 公告，找出截至 2026 年 4 月尚未被任何框架实现的一个功能。（提示：它与多跳任务委托有关。）

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| A2A | “Agent-to-Agent protocol” | 用于黑盒智能体协作的开放协议 |
| Agent Card | “`.well-known/agent.json`” | 描述智能体技能和端点的已发布元数据 |
| Skill | “A callable unit” | 智能体支持的具名操作（类似 MCP tool） |
| Task | “Unit of delegation” | 带生命周期和最终 artifact 的工作项 |
| Message | “Task input” | 携带 Parts（text、file、data） |
| Part | “Typed chunk” | message 中的 `text` / `file` / `data` 元素 |
| Artifact | “Task output” | 完成时返回的具名、带类型输出 |
| AP2 | “Agent Payments Protocol” | 用于信任和支付的签名 Agent Cards 扩展 |
| Opacity | “Black-box collaboration” | 被调用智能体的内部细节对调用方隐藏 |
| Input-required | “Task pause” | 智能体需要更多信息时的生命周期状态 |

## 延伸阅读

- [a2a-protocol.org](https://a2a-protocol.org/latest/) — A2A 权威规范
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) — 参考实现和 SDK
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) — 2025 年 6 月治理权转移
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) — 路线图和合作伙伴势能
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) — v1.0 发布说明和向后兼容指导
