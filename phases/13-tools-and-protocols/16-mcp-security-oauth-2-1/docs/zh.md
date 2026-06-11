# MCP 安全 II — OAuth 2.1、资源指示符、增量作用域

> 远程 MCP 服务器需要授权（authorization），而不只是认证（authentication）。2025-11-25 规范与 OAuth 2.1 + PKCE + 资源指示符（resource indicators，RFC 8707）+ 受保护资源元数据（protected-resource metadata，RFC 9728）对齐。SEP-835 增加了增量作用域同意（incremental scope consent），通过 403 WWW-Authenticate 进行升级授权（step-up authorization）。本课把升级流程实现为状态机，让你看到每一次跳转。

**类型：** 构建
**语言：** Python（stdlib，OAuth 状态机模拟器）
**先修要求：** Phase 13 · 09（传输），Phase 13 · 15（安全 I）
**时间：** 约 75 分钟

## 学习目标

- 区分资源服务器（resource server）与授权服务器（authorization server）的职责。
- 走通受 PKCE 保护的 OAuth 2.1 授权码流程。
- 使用 `resource`（RFC 8707）和受保护资源元数据（RFC 9728）来防止混淆代理（confused-deputy）攻击。
- 实现升级授权：服务器返回 403，并用 WWW-Authenticate 请求更高作用域；客户端重新提示用户同意并重试。

## 问题

早期 MCP（2025 年之前）发布远程服务器时使用临时拼凑的 API key，甚至没有认证。2025-11-25 规范用完整的 OAuth 2.1 profile 补上了这个缺口。

三个真实世界需求：

- **普通远程服务器。** 用户安装一个远程 MCP 服务器，用它访问自己的 Notion / GitHub / Gmail。带 PKCE 的 OAuth 2.1 是合适的形态。
- **作用域升级。** 一个已授予 `notes:read` 的笔记服务器，之后可能为了某个具体操作需要 `notes:write`。升级（SEP-835）不是重新做完整流程，而是请求额外作用域。
- **防止混淆代理。** 客户端持有一个受众限定到 Server A 的 token。Server A 是恶意的，并尝试把这个 token 提交给 Server B。资源指示符（RFC 8707）会把 token 固定到它预期的受众。

OAuth 2.1 并不新。新的是 MCP 的 profile：明确要求的流程（仅授权码 + PKCE；默认不支持 implicit、不支持 client credentials）、每个 token 请求都必须带资源指示符，并发布受保护资源元数据，让客户端知道应该去哪里。

## 概念

### 角色

- **客户端。** MCP 客户端（Claude Desktop、Cursor 等）。
- **资源服务器。** MCP 服务器（notes、GitHub、Postgres，或其他任何服务）。
- **授权服务器。** 签发 token。它可以和资源服务器是同一个服务，也可以是独立的 IdP（Auth0、Keycloak、Cognito）。

在 MCP 的 profile 中，资源服务器和授权服务器可以（CAN）是同一个 host，但应该（SHOULD）用 URL 区分。

### 授权码 + PKCE

流程：

1. 客户端生成 `code_verifier`（随机值）和 `code_challenge`（SHA256）。
2. 客户端将用户重定向到 `/authorize?response_type=code&client_id=...&redirect_uri=...&scope=notes:read&code_challenge=...&resource=https://notes.example.com`。
3. 用户同意。授权服务器重定向到 `redirect_uri?code=...`。
4. 客户端向 `/token?grant_type=authorization_code&code=...&code_verifier=...&resource=...` 发起 POST。
5. 授权服务器用保存的 challenge 校验 verifier 的哈希，并签发 access token。
6. 客户端使用 token：在发往资源服务器的每个请求上带 `Authorization: Bearer ...`。

PKCE 防止授权码拦截攻击。资源指示符防止 token 在其他地方有效。

### 受保护资源元数据（RFC 9728）

资源服务器发布一个 `.well-known/oauth-protected-resource` 文档：

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:read", "notes:write", "notes:delete"]
}
```

客户端从资源服务器发现授权服务器。这样可以减少配置——客户端只需要资源 URL。

### 资源指示符（RFC 8707）

token 请求中的 `resource` 参数会固定 token 的预期受众。签发出的 token 包含 `aud: "https://notes.example.com"`。另一个 MCP 服务器收到这个 token 时会检查 `aud` 并拒绝它。

### 作用域模型

作用域是用空格分隔的字符串。常见 MCP 约定：

- `notes:read`, `notes:write`, `notes:delete`
- `admin:*` 用于管理员能力（谨慎使用）
- `profile:read` 用于身份信息

作用域选择应遵循最小权限（least privilege）：现在需要什么就请求什么，需要更多时再升级。

### 升级授权（SEP-835）

用户授予 `notes:read`。之后他们要求 agent 删除一条笔记。服务器响应：

```
HTTP/1.1 403 Forbidden
WWW-Authenticate: Bearer error="insufficient_scope",
    scope="notes:delete", resource="https://notes.example.com"
```

客户端看到 insufficient_scope 错误后，向用户显示同意对话框以请求额外作用域，为它执行一个小型 OAuth 流程，然后用新的 token 重试请求。

### Token 受众校验

每个请求：服务器检查 `token.aud == self.resource_url`。不匹配 = 401。这会阻止跨服务器复用 token。

### 短生命周期 token 与轮换

Access token 应该（SHOULD）是短生命周期的（默认 1 小时）。Refresh token 每次刷新都会轮换。客户端在后台处理静默刷新。

### 禁止 token 透传

采样服务器（Phase 13 · 11）绝不能（MUST NOT）把客户端的 token 透传给其他服务。采样请求就是边界。

### 防止混淆代理

Token 绑定到 `aud`。客户端绑定到 `client_id`。每个请求都要同时校验二者。规范明确禁止旧的“pass-the-token”模式，这种模式在 MCP 之前的远程工具生态中很常见。

### 客户端 ID 发现

每个 MCP 客户端都会在固定 URL 发布自己的元数据。授权服务器可以获取客户端的元数据文档，以发现 redirect URI 和联系信息。这样就不需要手动注册客户端。

### 网关与 OAuth

Phase 13 · 17 展示了企业网关如何处理 OAuth：网关持有上游服务器的凭证，发给客户端的是网关签发的 token，上游 token 永远不会离开网关。这会翻转信任模型——用户只需向网关认证一次；网关处理 N 个服务器的授权。

## 使用它

`code/main.py` 将完整的 OAuth 2.1 升级流程模拟为状态机。它实现了：

- PKCE code-verifier / challenge 生成。
- 带资源指示符的授权码流程。
- 受保护资源元数据端点。
- 带受众检查的 token 校验。
- 在 `insufficient_scope` 上升级。

本课没有 HTTP 服务器；状态机在内存中运行，因此你可以追踪每一次跳转。Phase 13 · 17 的网关课程会把它接到真实传输上。

## 交付它

本课会产出 `outputs/skill-oauth-scope-planner.md`。给定一个带有工具的远程 MCP 服务器，该技能会设计作用域集合、固定规则和升级策略。

## 练习

1. 运行 `code/main.py`。追踪两个作用域的升级流程。注意升级时哪些跳转会重复。

2. 添加 refresh-token 轮换：每次刷新都会签发新的 refresh token，并使旧 token 失效。模拟被盗 refresh token 在轮换后被使用，并确认它会失败。

3. 使用 stdlib http.server 将受保护资源元数据端点实现为真实 HTTP 响应。镜像 Lesson 09 中的 /mcp 端点。

4. 为 GitHub MCP 服务器设计一个作用域层级：读取 repo、写 PR、批准 PR、合并 PR、admin。在每个级别之间使用升级。

5. 阅读 RFC 8707 和 RFC 9728。找出 MCP 对 9728 中哪个字段的使用方式不同于 RFC 示例。（提示：它涉及 `scopes_supported`。）

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| OAuth 2.1 | “现代 OAuth” | 一个整合后的 RFC，要求 PKCE 并禁止 implicit flow |
| PKCE | “持有证明” | code verifier + challenge，用来击败授权码拦截 |
| Resource indicator | “Token 受众” | RFC 8707 的 `resource` 参数，将 token 固定到一个服务器 |
| Protected-resource metadata | “发现文档” | RFC 9728 `.well-known/oauth-protected-resource` |
| Step-up authorization | “增量同意” | SEP-835 按需添加作用域的流程 |
| `insufficient_scope` | “带 WWW-Authenticate 的 403” | 服务器发出的信号，要求为更大作用域重新同意 |
| Confused deputy | “跨服务 token 复用” | 可信持有者不恰当地转发 token 的攻击 |
| Short-lived token | “Access token TTL” | 会快速过期的 bearer；refresh token 用于续期 |
| Scope hierarchy | “最小权限栈” | 分级作用域集合，级别之间使用升级 |
| Client ID metadata | “客户端发现文档” | 客户端发布自身 OAuth 元数据的 URL |

## 延伸阅读

- [MCP — 授权规范](https://modelcontextprotocol.io/specification/draft/basic/authorization) — 权威 MCP OAuth profile
- [den.dev — MCP 十一月授权规范](https://den.dev/blog/mcp-november-authorization-spec/) — 2025-11-25 变更讲解
- [RFC 8707 — OAuth 2.0 的资源指示符](https://datatracker.ietf.org/doc/html/rfc8707) — 受众固定 RFC
- [RFC 9728 — OAuth 2.0 受保护资源元数据](https://datatracker.ietf.org/doc/html/rfc9728) — 发现文档 RFC
- [Aembit — MCP OAuth 2.1、PKCE 与 AI 授权的未来](https://aembit.io/blog/mcp-oauth-2-1-pkce-and-the-future-of-ai-authorization/) — 实用升级流程 walkthrough
