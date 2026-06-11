# iii 原语上的生产级 MCP Auth — DCR、JWKS 轮换、受众绑定 Token

> 第 16 课在内存中搭建了 OAuth 2.1 状态机。到 2026 年，你交付给真实组织的每个 MCP 服务器都会位于生产级认证（auth）之后：动态客户端注册（dynamic client registration，RFC 7591）、授权服务器元数据发现（authorization-server metadata discovery，RFC 8414）、不会在凌晨 3 点令牌验证时失效的 JWKS 轮换，以及拒绝混淆代理复用的受众绑定 token。本课会用 iii 原语把这些全部串起来 — `iii.registerTrigger` 用于 HTTP 和 cron，`iii.registerFunction` 用于认证逻辑，`state::set/get` 用于缓存密钥 — 让认证面像引擎里的其他工作负载一样可观测、可重启、可重放。

**类型：** 构建
**语言：** Python（stdlib，iii 原语在课程环境中被 mock）
**先修：** 第 13 阶段 · 16（OAuth 2.1 状态机），第 13 阶段 · 17（网关）
**时间：** 约 90 分钟

## 学习目标

- 通过 RFC 8414 元数据发现授权服务器，并验证其契约。
- 实现 RFC 7591 动态客户端注册，让 MCP 客户端无需管理员介入即可注册。
- 使用 cron 触发器缓存并轮换 JWKS 密钥，让签名验证能挺过密钥滚动。
- 使用 RFC 8707 resource indicators 将 token 绑定到单个 MCP 资源，并拒绝混淆代理复用。
- 将每个端点和后台任务都接成 iii 原语 — HTTP 触发器、cron 触发器、命名函数以及 `state::*` 读取 — 这样一次重启就能重建认证面。
- 读取 IdP 能力矩阵，并在 IdP 无法满足 MCP 认证配置文件时拒绝部署。

## 问题

第 16 课的模拟器在内存中运行 OAuth 2.1。生产环境有三个仅靠内存模拟器看不到的运维缺口。

第一个缺口是注册。真实组织会运行数百个 MCP 服务器和数千个 MCP 客户端。运维人员不会把每个 Cursor 用户都手工注册成 OAuth 客户端。RFC 7591 动态客户端注册允许客户端向授权服务器 `POST /register`，并当场获得 `client_id`（以及可选的 `client_secret`）。服务器在其 RFC 8414 元数据中发布 `registration_endpoint`；客户端无需带外配置就能发现它。

第二个缺口是密钥轮换。JWT 验证依赖授权服务器的签名密钥，这些密钥以 JSON Web Key Set（JWKS）的形式发布。授权服务器会按计划轮换这些密钥（通常每小时一次，在事件响应期间有时更快）。如果 MCP 服务器只在启动时获取一次 JWKS，那么在轮换窗口之前验证都正常 — 之后每个请求都会失败，直到重启。生产环境会把 JWKS 接成带缓存的值，并用刷新任务在旧密钥过期之前覆盖缓存；另外还会在缓存未命中时做一次回退获取，以处理由比缓存更新的密钥签发的 token 到达的情况。

第三个缺口是受众绑定。第 16 课介绍了 RFC 8707 resource indicators。在生产环境中，这个 indicator 会变成每个请求上的硬性 claim 检查。MCP 服务器会把 `token.aud` 与自己的规范资源 URL 对比，并用 HTTP 401 拒绝不匹配项。这是防止上游 MCP 服务器（或持有发给某个服务器的 token 的恶意客户端）把该 token 重放到同一信任网格中另一台服务器的唯一防线。

本课把上述每个缺口都视为 iii 原语。元数据文档是一个返回某个函数输出的 HTTP 触发器。JWKS 轮换是一个调用 `auth::rotate-jwks` 的 cron 触发器，该函数会写入 `state::set("auth/jwks/<issuer>", ...)`。JWT 验证是其他组件通过 `iii.trigger("auth::validate-jwt", token)` 调用的函数。MCP 服务器本身只是另一个 HTTP 触发器，在分发前调用验证。重启引擎：触发器注册表会重建；state 会保留；认证面无需手工对账即可运行。

## 概念

### RFC 8414 — OAuth 授权服务器元数据

位于 `/.well-known/oauth-authorization-server` 的文档描述了客户端需要的一切：

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

拿到 MCP 资源 URL 的客户端会链式发现：RFC 9728 中的 `oauth-protected-resource`（资源服务器文档）会命名 issuer，然后 `oauth-authorization-server`（本 RFC）会命名每个端点。客户端永远不硬编码授权 URL。

在信任某个 IdP 用于 MCP 之前，你要验证的契约：

- `code_challenge_methods_supported` 包含 `S256`（根据 RFC 7636 的 PKCE）。
- `grant_types_supported` 包含 `authorization_code`，并拒绝 `password` 和 `implicit`。
- 存在 `registration_endpoint`（支持 RFC 7591）。
- 对 OAuth 2.1，`response_types_supported` 恰好是 `["code"]`。

如果缺少其中任意一项，MCP 服务器就会拒绝基于该 IdP 部署。错的是部署清单，不是代码。

### RFC 9728（回顾）— 受保护资源元数据

第 16 课覆盖了 RFC 9728。生产环境中的差异是：这个文档是客户端查找被*此* MCP 服务器信任的授权服务器的唯一位置。单个 MCP 服务器可以接受来自多个 IdP 的 token（一个给员工，一个给合作伙伴）。RFC 9728 声明这个集合；RFC 8414 记录每个 IdP 支持什么。

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### RFC 7591 — 动态客户端注册

没有 DCR 时，每个 MCP 客户端（Cursor、Claude Desktop、自定义 agent）都需要与 IdP 管理员进行带外交换。有了 DCR，客户端会发送：

```json
POST /register
Content-Type: application/json

{
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

服务器返回 `client_id` 和一个用于后续更新的 `registration_access_token`：

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`token_endpoint_auth_method: none` 是运行在用户设备上的 MCP 客户端的正确默认值。它们只获得 `client_id` — 没有可被窃取的 `client_secret`。PKCE 提供了公有客户端需要的持有证明（proof-of-possession）。

三个生产陷阱：

- 注册端点必须按来源 IP 做速率限制。否则，敌对行为者可以编写脚本创建数百万个假注册，并耗尽 `client_id` 命名空间。iii 让这件事很简单：注册 HTTP 触发器在分发给注册器之前调用 `auth::rate-limit` 函数。
- 某些企业 IdP 要求 `software_statement`（一个为客户端背书的签名 JWT）。本课的 mock 会跳过它；生产环境会接入一个验证步骤，拒绝除 localhost 重定向 URI 之外的任何未签名注册。
- `registration_access_token` 必须以哈希形式存储，而不是明文。这个 token 被盗意味着攻击者可以改写客户端的重定向 URI。

### RFC 8707（回顾）— Resource Indicators

第 16 课建立了基本形态。生产规则是：每个 token 请求都包含 `resource=<canonical-mcp-url>`，并且 MCP 服务器在每次调用上验证 `token.aud` 与自己的资源 URL 匹配。如果 MCP 服务器可通过 `https://notes.example.com/mcp` 访问，那么规范 URL 是 `https://notes.example.com` — 排除路径组件，这样单台服务器可以在一个受众下托管多个路径。

### RFC 7636（回顾）— PKCE

PKCE 在 OAuth 2.1 中是强制的。本课的授权码流程始终携带 `code_challenge` 和 `code_verifier`。服务器会拒绝任何没有 verifier 的 token 请求，或 verifier 无法哈希为已存 challenge 的请求。

### MCP Spec 2025-11-25 Auth Profile

MCP 规范（2025-11-25）明确规定了 MCP 服务器的授权层必须做什么：

- 发布 `/.well-known/oauth-protected-resource`（RFC 9728）。
- 只通过 `Authorization: Bearer ...` 接受 token。
- 针对每个请求验证 `aud`、`iss`、`exp` 和所需 scope。
- 对每个 401 和 403 返回带有 `Bearer error=...` 的 `WWW-Authenticate`，并在适用时包含 `scope=` 和 `resource=` 参数。
- 拒绝 `aud` 与规范资源不匹配的 token。
- 拒绝 `iss` 不在受保护资源元数据的 `authorization_servers` 列表中的 token。

OAuth 2.1 草案是基底；RFC 8414/7591/8707/9728 + RFC 7636 是表面；MCP 规范是 profile。

### IdP 能力矩阵

并非每个 IdP 都支持完整的 MCP profile。下表记录截至 2025-11-25 规范的事实性能力声明。它是一个*部署门禁*，不是推荐。

| IdP 类别 | RFC 8414 元数据 | RFC 7591 DCR | RFC 8707 resource | RFC 7636 S256 PKCE | 备注 |
|---|---|---|---|---|---|
| 自托管（Keycloak） | yes | yes | yes（自 24.x 起） | yes | 本课中 MCP profile 的参考 IdP；端到端支持每个 RFC。 |
| 企业 SSO（Microsoft Entra ID） | yes | yes（高级层级） | yes | yes | DCR 可用性因租户层级而异；部署前在目标租户中验证。 |
| 企业 SSO（Okta） | yes | yes（Okta CIC / Auth0） | yes | yes | DCR 可在 Auth0（现在的 Okta CIC）上使用；经典 Okta 组织要求管理员预注册。 |
| 社交登录 IdP（通用） | varies | rarely | rarely | yes | 大多数社交 IdP 把客户端视为静态合作伙伴；不要依赖 DCR。仅作为身份源使用，在其上叠加你自己的 MCP 感知授权服务器。 |
| 自定义 / 自研 | depends | depends | depends | depends | 如果你交付自己的实现，就交付完整 profile。跳过上面四个 RFC 中的任意一个都会破坏 MCP auth 契约。 |

部署清单的拒绝规则：如果选定的 IdP 没有返回 `registration_endpoint`，并且没有在 `code_challenge_methods_supported` 中列出 `S256`，MCP 服务器就拒绝启动。没有降级模式。

### 使用 iii 的 JWKS 轮换模式

生产故障模式是 JWKS 缓存陈旧。用 cron 触发器和 `state::*` 缓存解决它：

```python
iii.registerTrigger(
    "cron",
    {"schedule": "0 */6 * * *", "name": "auth::jwks-refresh"},
    "auth::rotate-jwks",
)
```

每六小时，cron 触发器调用 `auth::rotate-jwks`，该函数获取 `<issuer>/.well-known/jwks.json`，并写入 `state::set("auth/jwks/<issuer>", {keys, fetched_at})`。验证器从 `state::get` 读取。如果某个 token 的 `kid` 在缓存中缺失，就会触发一次同步的 `auth::rotate-jwks` 调用作为回退。这同时处理两种情况：计划轮换（cron）和密钥重叠窗口（同步回退）。

state 形状：

```json
{
  "auth/jwks/https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

同时存在两把密钥是稳态。授权服务器会先引入下一把密钥（`k_2026_04`），再退役上一把密钥（`k_2026_03`），这样用旧密钥签发的 token 在过期之前仍然有效。缓存保存并集；验证器按 `kid` 选择。

### iii 原语接线（本课真正关注的部分）

五个原语组成认证面：

```python
# 1. RFC 8414 metadata document
iii.registerTrigger(
    "http",
    {"path": "/.well-known/oauth-authorization-server", "method": "GET"},
    "auth::serve-asm",
)

# 2. RFC 7591 dynamic client registration
iii.registerTrigger(
    "http",
    {"path": "/register", "method": "POST"},
    "auth::register-client",
)

# 3. JWT validation as a callable function (the resource server triggers it)
iii.registerFunction("auth::validate-jwt", validate_jwt_handler)

# 4. Step-up issuance for incremental scope (SEP-835 from L16)
iii.registerFunction("auth::issue-step-up", issue_step_up_handler)

# 5. Cron-driven JWKS rotation
iii.registerTrigger(
    "cron",
    {"schedule": "0 */6 * * *"},
    "auth::rotate-jwks",
)
iii.registerFunction("auth::rotate-jwks", rotate_jwks_handler)
```

MCP 服务器本身永远不直接调用验证。它会：

```python
result = iii.trigger("auth::validate-jwt", {"token": bearer_token, "resource": self.resource})
if not result["valid"]:
    return {"status": 401, "WWW-Authenticate": result["www_authenticate"]}
```

这种间接层就是 iii 的押注。明天你可以把验证器替换成并行咨询两个 IdP 的 fanout，或者添加 span emitter，或者缓存正向验证结果。MCP 服务器不需要改变。

### 使用受众绑定演示混淆代理

服务器 A（`notes.example.com`）和服务器 B（`tasks.example.com`）都注册到同一个授权服务器。服务器 A 被攻陷。攻击者拿走某个用户的 notes token，并把它重放到服务器 B。

服务器 B 的验证器：

1. 解码 JWT，按 `kid` 获取 JWKS，验证签名。
2. 根据自己的受保护资源元数据的 `authorization_servers` 检查 `iss`。（通过 — 同一个 IdP。）
3. 检查 `aud == "https://tasks.example.com"`。（失败 — token 的 `aud` 是 `https://notes.example.com`。）
4. 返回 401，并带上 `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch"`。

受众 claim 是协议层面对该攻击的唯一防线。为了性能跳过它是最常见的生产错误；验证器必须在每个请求上运行，而不是只在会话开始时运行。

### 失败模式

- **陈旧 JWKS。** 密钥轮换后，验证器会拒绝有效 token。修复方式是上面的 cron+回退模式。永远不要在没有刷新任务的情况下缓存 JWKS。
- **缺少 `aud` claim。** 某些 IdP 默认会省略 `aud`，除非 token 请求中存在 `resource`。验证器必须拒绝缺少 `aud` 的 token，而不是把缺失当作通配符。
- **Scope 升级竞态。** 同一用户的两个并发 step-up 流程可能都成功，并产生两个具有不同 scope 的 access token。验证器必须使用请求上呈现的 token，而不是查找“该用户当前的 scope” — 那会产生 TOCTOU 窗口。
- **注册 token 被盗。** 泄露的 `registration_access_token` 会让攻击者改写重定向 URI。静态存储时对这些 token 做哈希；要求客户端每次更新时呈现明文；怀疑泄露时轮换。
- **未绑定 `iss`。** 接受任意 `iss` 的验证器会让攻击者搭建自己的授权服务器，为目标受众注册客户端，并签发 token。受保护资源元数据的 `authorization_servers` 列表就是 allow-list；必须强制执行。

## 使用它

`code/main.py` 用 stdlib Python 和一个小型 `iii_mock` 注册表演示完整生产流程，该注册表模拟 `iii.registerFunction`、`iii.registerTrigger`、`iii.trigger` 和 `state::set/get`。流程如下：

1. 授权服务器在 `/.well-known/oauth-authorization-server` 发布 RFC 8414 元数据。
2. MCP 客户端调用元数据端点，发现注册端点。
3. MCP 客户端向 `/register`（RFC 7591）发送请求并收到 `client_id`。
4. MCP 客户端使用 `resource` indicator（RFC 8707）运行受 PKCE 保护的授权码流程（RFC 7636）。
5. MCP 客户端带着 `Authorization: Bearer ...` 调用 MCP 服务器上的工具。
6. MCP 服务器触发 `auth::validate-jwt`，该函数从 `state::get` 读取 JWKS。
7. cron 触发器触发 `auth::rotate-jwks`，替换 state 中的 JWKS。
8. 下一次调用无需重启即可针对新密钥验证。
9. 针对另一个 MCP 资源的混淆代理尝试会因受众不匹配得到 401。

这里的 mock JWT 使用带共享密钥的 HS256（因此本课只用 stdlib 就能运行）。生产环境使用 RS256 或 EdDSA，并采用上面的 JWKS 模式；验证逻辑除此之外相同。

## 交付它

本课会产出 `outputs/skill-mcp-auth-iii.md`。给定一个 MCP 服务器配置和一个 IdP 能力集合，该 skill 会生成要注册的 iii 原语、JWKS 轮换计划、scope 映射，以及当 IdP 不支持完整 RFC profile 时要应用的拒绝规则。

## 练习

1. 运行 `code/main.py`。跟踪这 9 步流程。注意在 `auth::rotate-jwks` 覆盖旧数据之前，`state::get` 会在哪里返回陈旧数据，以及下一次请求现在如何针对新密钥验证。

2. 向受保护资源元数据的 `authorization_servers` 列表添加一个新的 IdP。签发一个由新 IdP 签名的 token，并确认验证器接受它。签发一个由未列出 IdP 签名的 token，并确认验证器用 `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"` 拒绝它。

3. 将 `auth::rate-limit` 实现为 iii 函数，并在注册 HTTP 触发器内部、注册器运行之前调用它。使用保存在 `state::set("auth/ratelimit/<ip>", ...)` 中的按来源 IP token-bucket。

4. 阅读 RFC 7591，并找出本课 `/register` handler 没有验证的两个字段。添加验证。（提示：`software_statement` 和 `redirect_uris` URI scheme。）

5. 阅读 MCP spec 2025-11-25 授权章节。找出关于 `WWW-Authenticate` header 的一个规范性要求，而本课的验证器目前没有发出它。添加它。

## 关键术语

| 术语 | 人们常说的叫法 | 实际含义 |
|------|----------------|------------------------|
| ASM | “OAuth 元数据文档” | RFC 8414 `/.well-known/oauth-authorization-server` JSON |
| DCR | “自助式客户端注册” | RFC 7591 `POST /register` 流程 |
| JWKS | “用于 JWT 验证的公钥” | JSON Web Key Set，从 `jwks_uri` 获取，按 `kid` 建索引 |
| Resource indicator | “受众参数” | RFC 8707 `resource` 参数，将 token 绑定到一台服务器 |
| `aud` claim | “Audience” | 验证器与规范资源 URL 对比的 JWT claim |
| Confused deputy | “Token replay” | 为服务器 A 签发的 token 被呈现给服务器 B 的攻击 |
| `iss` allow-list | “可信授权服务器” | 受保护资源元数据的 `authorization_servers` 中命名的集合 |
| Key rotation | “滚动 JWKS” | 带重叠窗口的签名密钥周期性替换 |
| Public client | “原生或浏览器客户端” | 没有 `client_secret` 的 OAuth 客户端；PKCE 用于补偿 |
| `WWW-Authenticate` | “401/403 响应 header” | 携带驱动客户端恢复的 `Bearer error=...` 指令 |

## 延伸阅读

- [MCP — Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/draft/basic/authorization) — 本课实现的 MCP auth profile
- [RFC 8414 — OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414) — 发现契约
- [RFC 7591 — OAuth 2.0 Dynamic Client Registration Protocol](https://datatracker.ietf.org/doc/html/rfc7591) — DCR
- [RFC 7636 — Proof Key for Code Exchange (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) — 公有客户端持有证明
- [RFC 8707 — Resource Indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — 受众绑定
- [RFC 9728 — OAuth 2.0 Protected Resource Metadata](https://datatracker.ietf.org/doc/html/rfc9728) — 资源服务器发现
- [OAuth 2.1 draft](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1) — 合并后的 OAuth 基底
