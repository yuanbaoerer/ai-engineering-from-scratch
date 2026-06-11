# MCP 传输 — stdio vs Streamable HTTP vs SSE 迁移

> stdio 只适用于本地，不能用于其他地方。Streamable HTTP（2025-03-26）是远程标准。旧的 HTTP+SSE 传输已弃用，并将在 2026 年中移除。选错传输会带来一次迁移成本；选对传输则能得到一个可远程托管、具备会话连续性和 DNS-rebinding 防护的 MCP 服务器。

**Type:** Learn
**Languages:** Python (stdlib, Streamable HTTP endpoint skeleton)
**Prerequisites:** Phase 13 · 07, 08 (MCP server and client)
**Time:** ~45 minutes

## Learning Objectives

- 根据部署形态（本地 vs 远程、单进程 vs 集群）在 stdio 和 Streamable HTTP 之间做选择。
- 实现 Streamable HTTP 的单端点模式：POST 处理请求，GET 建立会话流。
- 强制执行 `Origin` 校验和 session-id 语义，以抵御 DNS-rebinding。
- 在 2026 年中移除截止日期之前，将遗留 HTTP+SSE 服务器迁移到 Streamable HTTP。

## The Problem

第一个 MCP 远程传输（2024-11）是 HTTP+SSE：两个端点，一个用于客户端的 POST，另一个 Server-Sent-Events 通道用于服务器到客户端的流。它能工作。但它也很笨重：每个会话需要两个端点，在某些 CDN 前会破坏缓存，并且强依赖长连接 SSE，而一些 WAF 会激进地终止这种连接。

2025-03-26 规范用 Streamable HTTP 取代了它：一个端点，POST 用于客户端请求，GET 用于建立会话流，两者共享一个 `Mcp-Session-Id` header。自那以后新建或迁移的每个服务器都使用 Streamable HTTP。旧的 SSE 模式正在被弃用——Atlassian Rovo 已在 2026 年 6 月 30 日移除它；Keboola 是 2026 年 4 月 1 日；大多数剩余企业服务器会在 2026 年底前完成。

而 stdio 对本地服务器仍然重要。Claude Desktop、VS Code，以及所有 IDE 形态的客户端都会通过 stdio 拉起服务器。正确的心智模型是：stdio 用于“这台机器”，Streamable HTTP 用于“通过网络”。不要混用。

## The Concept

### stdio

- 子进程传输。客户端拉起服务器，通过 stdin/stdout 通信。
- 每行一个 JSON 对象。以换行符分隔。
- 没有 session id；进程身份就是会话。
- 不需要认证（子进程继承父进程的信任边界）。
- 永远不要用于远程服务器——否则你需要用 SSH 或 socat 来隧道转发，到了这一步就应该使用 Streamable HTTP。

### Streamable HTTP

单端点 `/mcp`（或任意路径）。支持三种 HTTP 方法：

- **POST /mcp.** 客户端发送 JSON-RPC 消息。服务器返回单个 JSON 响应，或者返回一个包含一个或多个响应的 SSE 流（适用于批量响应以及与该请求相关的通知）。
- **GET /mcp.** 客户端打开一个长连接 SSE 通道。服务器用它发送服务器到客户端的请求（sampling、notifications、elicitation）。
- **DELETE /mcp.** 客户端显式终止会话。

会话由 `Mcp-Session-Id` header 标识：服务器在第一次响应中设置该 header，客户端在之后每个请求中回传。Session id MUST 是密码学安全的随机值（128+ bits）；出于安全考虑，拒绝客户端自选 id。

### Single endpoint vs two

旧规范中的双端点模式在 2026 年仍然可调用——规范称其为“legacy compatible”。但所有新服务器都应该使用单端点。官方 SDK 输出单端点；只有在连接尚未迁移的远程服务器时才使用 legacy mode。

### `Origin` validation and DNS-rebinding

浏览器（目前）不是 MCP 客户端，但攻击者可以构造一个网页，让浏览器向 `localhost:1234/mcp` 发起 POST——而用户的本地 MCP 服务器可能正监听在那里。如果服务器不检查 `Origin`，浏览器的同源策略并不能拯救它，因为 `Origin: http://evil.com` 是合法的跨源请求。

2025-11-25 规范要求服务器拒绝 `Origin` 不在 allowlist 中的请求。Allowlist 通常包含 MCP 客户端主机（`https://claude.ai`、`vscode-webview://*`）以及用于本地 UI 的 localhost 变体。

### Session id lifecycle

1. 客户端发送第一个请求，不带 `Mcp-Session-Id`。
2. 服务器分配一个随机 id，并在响应 header 上设置 `Mcp-Session-Id`。
3. 客户端在所有后续请求以及用于流的 `GET /mcp` 中回传该 header。
4. 服务器可以撤销会话；客户端在后续请求中会看到 404，并且必须重新初始化。
5. 客户端可以显式 DELETE 会话，以便干净关闭。

### Keepalive and reconnect

SSE 连接会断开。客户端通过使用同一个 `Mcp-Session-Id` 重新 GET 来重建连接。服务器 MUST 对中断期间错过的事件排队（在合理窗口内），并通过客户端回传的 `last-event-id` header 进行重放。

Phase 13 · 13 会介绍 Tasks，它可以让长时间运行的工作即使在完整会话重连后也能存活。

### Backwards compatibility probe

希望同时支持新旧服务器的客户端：

1. POST 到 `/mcp`。
2. 如果响应是带 JSON 或 SSE 的 `200 OK`，这就是 Streamable HTTP。
3. 如果响应是 `200 OK`，并且 `Content-Type: text/event-stream` 且带有指向次级端点的 `Location` header，这就是 legacy HTTP+SSE；跟随该 `Location`。

### Cloudflare, ngrok, and hosting

2026 年的生产远程 MCP 服务器运行在 Cloudflare Workers（配合其 MCP Agents SDK）、Vercel Functions，或容器化的 Node/Python 上。关键点：你的托管平台必须支持用于 SSE GET 的长连接 HTTP。Vercel 的免费层限制为 10 秒，不适合。Cloudflare Workers 支持无限期流。

### Gateway composition

当你用网关（Phase 13 · 17）封装多个 MCP 服务器时，该网关是一个单一的 Streamable HTTP 端点，它会重写 session ids 并复用上游。工具在网关层合并；客户端看到的是一个单一逻辑服务器。

### Transport failure modes

- **stdio SIGPIPE.** 子进程在写入中途死亡会引发 SIGPIPE；服务器应该干净退出。客户端应该检测 EOF，并将会话标记为死亡。
- **HTTP 502 / 504.** Cloudflare、nginx 和其他代理会在上游失败时发出这些状态码。Streamable HTTP 客户端应该在短暂 backoff 后重试一次。
- **SSE connection drop.** TCP RST、代理超时或客户端网络变化会关闭流。客户端使用 `Mcp-Session-Id` 和可选的 `last-event-id` 重连以恢复。
- **Session revocation.** 服务器使某个 session id 失效；客户端在下一次请求中看到 404。客户端必须重新握手。
- **Clock skew.** 客户端上的 Resource-TTL 计算与服务器不一致。客户端应该把服务器时间戳视为权威。

### When to bypass Streamable HTTP

一些企业会在自己的网络内部把 MCP 服务器部署在 gRPC 或消息队列传输之后。这是非标准做法——MCP 规范并未正式定义这些传输。网关可以在内部使用 gRPC 的同时，向 MCP 客户端暴露一个 Streamable HTTP 表面。保持外部表面符合规范；翻译工作由网关负责。

## Use It

`code/main.py` 使用 `http.server`（stdlib）实现了一个最小 Streamable HTTP 端点。它在 `/mcp` 上处理 POST、GET 和 DELETE，在第一次响应中设置 `Mcp-Session-Id`，校验 `Origin`，并拒绝来自非 allowlisted origins 的请求。该 handler 复用了 Lesson 07 notes server 的 dispatch 逻辑。

需要关注的点：

- POST handler 读取 JSON-RPC body，进行 dispatch，并写出 JSON 响应（单响应变体；SSE 变体在结构上类似）。
- `Origin` 检查会拒绝默认的 `http://evil.example` 探测，但接受 `http://localhost`。
- Session ids 是随机 128-bit 十六进制字符串；服务器在内存中保存每个会话的状态。

## Ship It

本课会生成 `outputs/skill-mcp-transport-migrator.md`。给定一个 HTTP+SSE（legacy）MCP 服务器，该 skill 会生成迁移到 Streamable HTTP 的计划，包含 session-id 连续性、Origin 检查，以及向后兼容的 probe 支持。

## Exercises

1. 运行 `code/main.py`。从 `curl` POST 一个 `initialize`，并观察 `Mcp-Session-Id` 响应 header。回传该 header 再 POST 第二个请求，并验证会话连续性。

2. 添加一个会打开 SSE 流的 GET handler。每五秒发送一个 `notifications/progress` 事件。使用同一个 session id 重新 GET 来重连，并确认服务器接受它。

3. 实现 `last-event-id` 重放逻辑。在重连时，重放自该 id 之后生成的所有事件。

4. 扩展 `Origin` 校验以支持通配符模式（`https://*.example.com`），并确认它接受 `https://app.example.com`，但拒绝 `https://evil.example.com.attacker.net`。

5. 从官方 registry 中取一个 legacy HTTP+SSE 服务器（有好几个），并草拟迁移方案：端点处理、session id 生成和 header 语义分别需要改什么。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| stdio transport | “本地子进程” | 通过 stdin/stdout 传输 JSON-RPC，以换行符分隔 |
| Streamable HTTP | “远程传输” | 单端点 POST + GET + 可选 SSE，2025-03-26 规范 |
| HTTP+SSE | “Legacy” | 将在 2026 年中移除的双端点模型 |
| `Mcp-Session-Id` | “会话 header” | 服务器分配的随机 id，在每个后续请求中回传 |
| `Origin` allowlist | “DNS-rebinding 防护” | 拒绝 Origin 未获批准的请求 |
| Single endpoint | “一个 URL” | `/mcp` 为所有会话操作处理 POST / GET / DELETE |
| `last-event-id` | “SSE 重放” | 用于恢复断开的流且不丢事件的 header |
| Backwards-compat probe | “新旧检测” | 客户端响应形态检查，用于自动选择传输 |
| Long-lived HTTP | “SSE streaming” | 服务器在一个 TCP 连接上持续数分钟或数小时推送事件 |
| Session revocation | “强制重新初始化” | 服务器使 session id 失效；客户端必须再次握手 |

## Further Reading

- [MCP — Basic transports spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports) — stdio 和 Streamable HTTP 的权威参考
- [MCP — Basic transports spec 2025-03-26](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports) — 引入 Streamable HTTP 的修订版
- [Cloudflare — MCP transport](https://developers.cloudflare.com/agents/model-context-protocol/transport/) — Workers 托管的 Streamable HTTP 模式
- [AWS — MCP transport mechanisms](https://builder.aws.com/content/35A0IphCeLvYzly9Sw40G1dVNzc/mcp-transport-mechanisms-stdio-vs-streamable-http) — 跨部署形态的比较
- [Atlassian — HTTP+SSE deprecation notice](https://community.atlassian.com/forums/Atlassian-Remote-MCP-Server/HTTP-SSE-Deprecation-Notice/ba-p/3205484) — 具体迁移截止日期示例
