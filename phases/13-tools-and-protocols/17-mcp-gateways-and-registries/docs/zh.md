# MCP 网关与注册表 — 企业控制平面

> 企业不能允许每个开发者随意安装随机的 MCP 服务器。网关（gateway）会集中处理认证、RBAC、审计、速率限制、缓存和工具投毒检测，然后把合并后的工具表面作为单个 MCP 端点暴露出去。Official MCP Registry（Anthropic + GitHub + PulseMCP + Microsoft，命名空间已验证）是规范的上游来源。本课说明网关适合放在哪里，带你走过一个最小实现，并概览 2026 年的厂商格局。

**类型：** 学习
**语言：** Python（stdlib，最小网关）
**先修要求：** Phase 13 · 15（工具投毒），Phase 13 · 16（OAuth 2.1）
**时间：** 约 45 分钟

## 学习目标

- 解释 MCP 网关位于哪里（在 MCP 客户端和多个后端 MCP 服务器之间）。
- 实现五项网关职责：认证、RBAC、审计、速率限制、策略。
- 在网关层强制执行固定工具哈希清单。
- 区分 Official MCP Registry 与元注册表（Glama、MCPMarket、MCP.so、Smithery、LobeHub）。

## 问题

一家财富 500 强企业有 30 个获批 MCP 服务器、5000 名开发者、合规和审计要求，以及一个希望集中化策略的安全团队。让每个开发者在自己的 IDE 中安装任意服务器是完全不可接受的。

网关模式：

1. 网关作为开发者连接的单个 Streamable HTTP 端点运行。
2. 网关持有每个后端 MCP 服务器的凭据。
3. 每个开发者请求都通过网关自己的 OAuth 进行认证和作用域限定。
4. 网关将调用路由到后端服务器，并应用策略。
5. 所有调用都会记录日志以供审计。

Cloudflare MCP Portals、Kong AI Gateway、IBM ContextForge、MintMCP、TrueFoundry、Envoy AI Gateway 都在 2025-2026 年发布了网关或网关功能。

与此同时，Official MCP Registry 作为规范上游发布：经过策展、命名空间验证、采用反向 DNS 命名的服务器，网关可以从中拉取。元注册表（Glama、MCPMarket、MCP.so、Smithery、LobeHub）会聚合来自多个来源的服务器。

## 概念

### 五项网关职责

1. **认证。** 使用 OAuth 2.1 识别开发者；映射到用户角色。
2. **RBAC。** 按用户设置策略：哪些服务器、哪些工具、哪些作用域。
3. **审计。** 每次调用都记录谁、做了什么、何时发生、结果如何。
4. **速率限制。** 按用户 / 工具 / 服务器设置上限，防止滥用。
5. **策略。** 拒绝被投毒的描述，强制执行 Rule of Two，脱敏 PII。

### 作为单个端点的网关

对开发者来说，网关看起来就像一个 MCP 服务器。内部则路由到 N 个后端。会话 ID（Phase 13 · 09）会在边界处被重写。

### 凭据保管

开发者永远看不到后端令牌。网关持有它们（或代理到确实持有令牌的身份提供方）。在网关上拥有 `notes:read` 权限的开发者，可以通过网关自己的后端凭据传递式访问 notes MCP 服务器——但必须受绑定这种传递访问的策略约束。

### 网关层的工具哈希固定

网关持有一份已批准工具描述的清单（SHA256 哈希）。在发现阶段，它获取每个后端的 `tools/list`，将哈希与清单比较，并移除任何描述已发生变化的工具。这是 Phase 13 · 15 中的 rug-pull 防御被集中应用后的形态。

### 策略即代码

高级网关会用 OPA/Rego、Kyverno 或 Styra 表达策略。像“用户 `alice` 只能在 `acme` 组织的仓库上调用 `github.open_pr`”这样的规则会以声明式方式编码。简单网关使用手写 Python。这两种形态都有效。

### 会话感知路由

当用户的会话包含多个服务器时，网关会进行多路复用：开发者的单个 MCP 会话持有 N 个后端会话，每个服务器一个。来自任何后端的通知都会通过网关路由到开发者的会话。

### 命名空间合并

网关会合并所有后端的工具命名空间，通常在发生冲突时添加前缀。`github.open_pr`、`notes.search`。这让路由变得无歧义。

### 注册表

- **Official MCP Registry (`registry.modelcontextprotocol.io`)。** 在 Anthropic、GitHub、PulseMCP、Microsoft 的监管下发布。命名空间已验证（反向 DNS：`io.github.user/server`）。针对基础质量预先过滤。
- **Glama。** 以搜索为中心的元注册表，聚合许多来源。
- **MCPMarket。** 偏商业化的目录，包含厂商列表。
- **MCP.so。** 社区目录；开放提交。
- **Smithery。** 包管理器风格的安装流程。
- **LobeHub。** 集成在其 LobeChat 应用 UI 中的注册表。

企业网关默认从 Official Registry 拉取，允许管理员从元注册表中策展性添加，并拒绝任何未固定的内容。

### 反向 DNS 命名

Official Registry 要求公共服务器使用反向 DNS 名称：`io.github.alice/notes`。命名空间可以防止抢注，并让信任委托更清晰。

### 厂商概览，2026 年 4 月

| 厂商 | 优势 |
|--------|----------|
| Cloudflare MCP Portals | 边缘托管；集成 OAuth；免费层 |
| Kong AI Gateway | K8s 原生；细粒度策略；日志写入 OpenTelemetry |
| IBM ContextForge | 企业 IAM；合规；审计导出 |
| TrueFoundry | 偏 DevOps；指标优先 |
| MintMCP | 面向开发者平台 |
| Envoy AI Gateway | 开源；可自定义过滤器 |

Phase 17（生产基础设施）会更深入讨论网关运维。

## 使用它

`code/main.py` 提供了一个约 150 行的最小网关：用假的 Bearer token 认证用户，持有按用户划分的 RBAC 策略，将请求路由到两个后端 MCP 服务器，把每次调用写入审计日志，强制执行速率限制，并拒绝任何描述哈希与固定清单不匹配的后端工具。

要关注的内容：

- `RBAC` 字典以 `user_id` 为键，包含允许的 `server_tool` 条目。
- `AUDIT_LOG` 是只追加的事件列表。
- 速率限制对每个用户使用一个令牌桶。
- 固定清单是一个 `server::tool -> hash` 字典。

## 交付它

本课会生成 `outputs/skill-gateway-bootstrap.md`。给定一份企业 MCP 计划（用户、后端、合规），该技能会生成一份网关配置规格。

## 练习

1. 运行 `code/main.py`。以获准用户发起一次调用；然后以未获准用户发起；再发起一次超过速率限制的突发请求。验证这三条流程。

2. 添加一条策略，在结果返回给客户端之前脱敏 PII。用简单的正则处理形如 SSN 的字符串；注明缺口（电子邮件、电话号码）。

3. 扩展审计日志，让它发出 OpenTelemetry GenAI span。Phase 13 · 20 会覆盖确切属性。

4. 为一个包含 50 名开发者、五个后端（notes、github、postgres、jira、slack）的团队设计 RBAC 策略。谁在每个后端上获得只读权限？谁获得写权限？

5. 从头到尾阅读 Cloudflare 的企业 MCP 文章。找出 Cloudflare 已提供、而这个 stdlib 网关没有的一项功能。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| 网关 | “MCP proxy” | 位于客户端和后端之间的集中化服务器 |
| 凭据保管 | “Backend tokens stay server-side” | 开发者永远看不到上游令牌 |
| 会话感知路由 | “Multi-backend session” | 网关为每个开发者会话多路复用 N 个后端会话 |
| 工具哈希固定 | “Approved manifest” | 每个获批工具描述的 SHA256；在中心层阻断 rug-pull |
| RBAC | “Per-user policy” | 针对工具和服务器的基于角色的访问控制 |
| 策略即代码 | “Declarative rules” | 在网关执行的 OPA/Rego、Kyverno、Styra 策略 |
| 审计日志 | “Who, what, when” | 面向合规的只追加事件日志 |
| 速率限制 | “Per-user token bucket” | 防止滥用的每分钟上限 |
| Official MCP Registry | “Canonical upstream” | `registry.modelcontextprotocol.io`，命名空间已验证 |
| 反向 DNS 命名 | “Registry namespace” | `io.github.user/server` 约定 |

## 延伸阅读

- [Official MCP Registry](https://registry.modelcontextprotocol.io/) — 规范上游，命名空间已验证
- [Cloudflare — Enterprise MCP](https://blog.cloudflare.com/enterprise-mcp/) — 带 OAuth 和策略的网关模式
- [agentic-community — MCP gateway registry](https://github.com/agentic-community/mcp-gateway-registry) — 开源参考网关
- [TrueFoundry — What is an MCP gateway?](https://www.truefoundry.com/blog/what-is-mcp-gateway) — 功能对比文章
- [IBM — MCP context forge](https://github.com/IBM/mcp-context-forge) — IBM 的企业网关
