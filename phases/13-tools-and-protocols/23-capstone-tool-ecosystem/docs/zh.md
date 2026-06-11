# 结业项目 — 构建完整的工具生态系统

> Phase 13 已经讲授了每个组成部分。本结业项目会把它们串成一个具备生产系统形态的整体：一个包含工具 + 资源 + 提示词 + 任务 + UI 的 MCP 服务器、位于边缘的 OAuth 2.1、RBAC 网关、多服务器客户端、一次 A2A 子智能体调用、进入 collector 的 OTel 追踪、CI 中的工具投毒检测，以及一套 AGENTS.md + SKILL.md 包。完成后，你将能够为每个架构选择给出充分论证。

**类型：** 构建
**语言：** Python（stdlib，端到端生态系统 harness）
**先修要求：** Phase 13 · 01 through 21
**时间：** 约 120 分钟

## 学习目标

- 组合一个 MCP 服务器，对外暴露工具、资源、提示词，以及一个带有 `ui://` app 的任务。
- 在服务器前接入一个 OAuth 2.1 网关，用于强制执行 RBAC 和固定哈希（pinned hashes）。
- 编写一个多服务器客户端，使用 OTel GenAI 属性进行端到端追踪。
- 将部分工作负载委派给 A2A 子智能体；验证不透明性（opacity）得以保留。
- 用 AGENTS.md + SKILL.md 打包整个栈，让其他智能体也能驱动它。

## 问题

交付这个“研究与报告”系统：

- 用户请求：“summarize the three most-cited 2026 arXiv papers on agent protocols.”
- 系统：通过 MCP 搜索 arXiv；通过 A2A 将论文摘要工作委派给专门的 writer agent；聚合结果；将交互式报告渲染为 MCP Apps 的 `ui://` 资源；将每一步记录到 OTel。

Phase 13 的所有原语都会出现。这不是玩具示例——Anthropic（Claude Research 产品）、OpenAI（带 Apps SDK 的 GPTs）以及第三方在 2026 年交付的生产级研究助手系统，正是这种形态。

## 概念

### 架构

```
[user] -> [client] -> [gateway (OAuth 2.1 + RBAC)] -> [research MCP server]
                                                      |
                                                      +- MCP tool: arxiv_search (pure)
                                                      +- MCP resource: notes://recent
                                                      +- MCP prompt: /research_topic
                                                      +- MCP task: generate_report (long)
                                                      +- MCP Apps UI: ui://report/current
                                                      +- A2A call: writer-agent (tasks/send)
                                                      |
                                                      +- OTel GenAI spans
```

### Trace hierarchy

```
agent.invoke_agent
 ├── llm.chat (kick off)
 ├── mcp.call -> tools/call arxiv_search
 ├── mcp.call -> resources/read notes://recent
 ├── mcp.call -> prompts/get research_topic
 ├── a2a.tasks/send -> writer-agent
 │    └── task transitions (opaque internals)
 ├── mcp.call -> tools/call generate_report (task-augmented)
 │    └── tasks/status polling
 │    └── tasks/result (completed, returns ui:// resource)
 └── llm.chat (final synthesis)
```

一个 trace id。每个 span 都带有正确的 `gen_ai.*` 属性。

### 安全姿态

- OAuth 2.1 + PKCE，并通过资源指示器（resource indicator）将 audience 固定到网关。
- 网关持有上游凭证；用户永远看不到它们。
- RBAC：`alice` 拥有 `research:read`、`research:write`，可以调用所有工具。`bob` 拥有 `research:read`，不能调用 `generate_report`。
- 固定描述清单（pinned description manifest）：丢弃任何工具哈希发生变化的服务器。
- Rule of Two 审计：没有任何工具同时组合不可信输入、敏感数据和有后果的动作。

### 渲染

最终的 `generate_report` 任务会返回内容块以及一个 `ui://report/current` 资源。客户端的宿主（Claude Desktop 等）会在沙箱 iframe 中渲染这个交互式 dashboard。dashboard 包含排序后的论文列表、引用次数，以及一个按钮；当用户点击任意论文时，它会调用 `host.callTool('summarize_paper', {arxiv_id})`。

### 打包

整个系统以如下形式交付：

```
research-system/
  AGENTS.md                     # project conventions
  skills/
    run-research/
      SKILL.md                  # the top-level workflow
  servers/
    research-mcp/               # the MCP server
      pyproject.toml
      src/
  agents/
    writer/                     # the A2A agent
  gateway/
    config.yaml                 # RBAC + pinned manifest
```

用户使用 `docker compose up` 部署。Claude Code、Cursor、Codex 和 opencode 用户都可以通过调用 `run-research` skill 来驱动该系统。

### 每节 Phase 13 课程贡献了什么

| Lesson | What the capstone uses |
|--------|------------------------|
| 01-05 | 工具接口、提供商可移植性、并行调用、schema、linting |
| 06-10 | MCP 原语、服务器、客户端、传输、资源 + 提示词 |
| 11-14 | 采样、roots + elicitation、异步任务、`ui://` apps |
| 15-17 | 工具投毒、OAuth 2.1、网关 + registry |
| 18 | A2A 子智能体委派 |
| 19 | OTel GenAI 追踪 |
| 20 | LLM 层的路由网关 |
| 21 | SKILL.md + AGENTS.md 打包 |

## 使用它

`code/main.py` 会把前面课程中的模式串成一个可运行 demo。它完全使用 stdlib，并且全部在进程内运行，因此你可以端到端阅读。它会为研究与报告场景运行完整流程：与网关握手、模拟 OAuth 2.1、合并 tools/list、将 generate_report 作为任务、对 writer 发起 A2A 调用、返回 ui:// 资源，并发出 OTel spans。

观察重点：

- 每一跳都使用同一个 trace id。
- 网关策略会阻止第二个用户执行写入。
- 任务生命周期从 working → completed，并同时返回文本和 ui:// 内容。
- A2A 调用的内部状态对编排器保持不透明。
- AGENTS.md 和 SKILL.md 是另一个智能体复现该工作流所需的唯一文件。

## 交付它

本课会生成 `outputs/skill-ecosystem-blueprint.md`。给定一个产品需求（研究、摘要、自动化），该 skill 会产出完整架构：哪些 MCP 原语、哪些网关控制、哪些 A2A 调用、哪些遥测，以及如何打包。

## 练习

1. 运行 `code/main.py`。注意单个 trace id 以及 spans 如何嵌套。数一数这个 demo 触达了 Phase 13 中的多少个原语。

2. 扩展 demo：添加第二个后端 MCP 服务器（例如 `bibliography`），并确认网关将其工具合并到同一个命名空间中。

3. 将假的 A2A writer agent 替换为在子进程中运行的真实 agent。使用 Lesson 19 的 harness。

4. 在编排器与 LLM 之间的路由网关中添加一个 PII 脱敏步骤。确认用户查询中的 email 会被 scrubbed。

5. 为将要维护该系统的队友编写一份 AGENTS.md。它应该能在五分钟内读完，并提供他们在 Cursor 或 Codex 中驱动这个结业项目所需的一切信息。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Capstone | “Phase-13 集成 demo” | 使用每个原语的端到端系统 |
| Research and report | “场景” | 搜索、摘要、渲染模式 |
| Ecosystem | “所有组件放在一起” | 服务器 + 客户端 + 网关 + 子智能体 + 遥测 + 包 |
| Trace hierarchy | “单个 trace id” | 每一跳的 span 共享该 trace；通过 span ids 建立父子关系 |
| Gateway-issued token | “传递式认证” | 客户端只看到网关的 token；网关持有上游凭证 |
| Merged namespace | “一个扁平列表中的所有工具” | 在网关处进行多服务器合并，冲突时加前缀 |
| Opacity boundary | “A2A 调用隐藏内部细节” | 子智能体的推理对编排器不可见 |
| Three-layer stack | “AGENTS.md + SKILL.md + MCP” | 项目上下文 + 工作流 + 工具 |
| Defense-in-depth | “多层安全防护” | 固定哈希、OAuth、RBAC、Rule of Two、审计日志 |
| Spec compliance matrix | “我们交付了规范要求的哪些内容” | 将交付物映射到 2025-11-25 要求的 checklist |

## 延伸阅读

- [MCP — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 汇总参考
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 协议的发展方向
- [a2a-protocol.org](https://a2a-protocol.org/latest/) — A2A v1.0 参考
- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 标准追踪约定
- [Anthropic — Claude Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview) — 生产级智能体运行时模式
