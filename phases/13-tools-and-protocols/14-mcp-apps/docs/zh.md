# MCP Apps — 通过 `ui://` 提供交互式 UI 资源

> 纯文本工具输出限制了智能体能展示的内容。MCP Apps（SEP-1724，2026 年 1 月 26 日正式发布）让工具可以返回沙箱化的交互式 HTML，并以内联方式渲染在 Claude Desktop、ChatGPT、Cursor、Goose 和 VS Code 中。仪表盘、表单、地图、3D 场景，都可以通过一个扩展实现。本课会讲解 `ui://` 资源方案、`text/html;profile=mcp-app` MIME、iframe 沙箱 postMessage 协议，以及允许服务器渲染 HTML 所带来的安全边界。

**类型：** 构建
**语言：** Python（stdlib，UI 资源发射器）、HTML（示例应用）
**先修：** 第 13 阶段 · 07（MCP server）、第 13 阶段 · 10（resources）
**时间：** 约 75 分钟

## 学习目标

- 从工具调用返回一个 `ui://` 资源，并设置正确的 MIME 和元数据。
- 使用 `_meta.ui.resourceUri`、`_meta.ui.csp` 和 `_meta.ui.permissions` 声明工具关联的 UI。
- 实现用于 UI 到宿主通信的 iframe 沙箱 postMessage JSON-RPC。
- 应用 CSP 和 permissions-policy 默认值，防御源自 UI 的攻击。

## 问题

2025 年时代的 `visualize_timeline` 工具可以返回“这里有 14 条按时间顺序整理的笔记：...”。这只是一个段落。用户真正想要的是交互式时间线。在 MCP Apps 之前，选项是：特定客户端的小组件 API（Claude artifacts、OpenAI Custom GPT HTML），或者根本没有 UI。

MCP Apps（SEP-1724，2026 年 1 月 26 日发布）标准化了这个契约。工具结果包含一个 `resource`，其 URI 是 `ui://...`，MIME 是 `text/html;profile=mcp-app`。宿主会在沙箱化 iframe 中渲染它，使用受限 CSP，并且除非显式授权，否则没有网络访问。iframe 内部的 UI 通过一个很小的 postMessage JSON-RPC 方言向宿主发送消息。

每个兼容客户端（Claude Desktop、ChatGPT、Goose、VS Code）都会以相同方式渲染同一个 `ui://` 资源。一个服务器、一个 HTML 包、通用 UI。

## 概念

### `ui://` 资源方案

工具返回：

```json
{
  "content": [
    {"type": "text", "text": "Here is your notes timeline:"},
    {"type": "ui_resource", "uri": "ui://notes/timeline"}
  ],
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline",
      "csp": {
        "defaultSrc": "'self'",
        "scriptSrc": "'self' 'unsafe-inline'",
        "connectSrc": "'self'"
      },
      "permissions": []
    }
  }
}
```

然后宿主对 `ui://notes/timeline` URI 调用 `resources/read`，并得到：

```json
{
  "contents": [{
    "uri": "ui://notes/timeline",
    "mimeType": "text/html;profile=mcp-app",
    "text": "<!doctype html>..."
  }]
}
```

### Iframe 沙箱

宿主把 HTML 渲染在一个沙箱化的 `<iframe>` 中，并具有：

- `sandbox="allow-scripts allow-same-origin"`（或根据服务器声明使用更严格的配置）
- 通过响应头应用的服务器声明 CSP。
- 没有来自宿主 origin 的 cookie，也没有 localStorage。
- 网络访问限制在 CSP 的 `connectSrc` 中。

### postMessage 协议

iframe 通过 `window.postMessage` 与宿主通信。一个很小的 JSON-RPC 2.0 方言：

始终将 `targetOrigin` 固定为对端的精确 origin，并在接收侧先根据 allowlist 校验 `event.origin`，再处理任何载荷。此通道两侧都绝不要使用 `"*"` —— 消息体会携带工具调用和资源读取。

```js
// iframe to host  (pin to host origin)
window.parent.postMessage({
  jsonrpc: "2.0",
  id: 1,
  method: "host.callTool",
  params: { name: "notes_update", arguments: { id: "note-14", title: "..." } }
}, "https://host.example.com");

// host to iframe  (pin to iframe origin)
iframe.contentWindow.postMessage({
  jsonrpc: "2.0",
  id: 1,
  result: { content: [...] }
}, "https://iframe.example.com");

// receiver on both sides
window.addEventListener("message", (event) => {
  if (event.origin !== "https://expected-peer.example.com") return;
  // safe to process event.data
});
```

UI 可调用的宿主侧方法包括：

- `host.callTool(name, arguments)` — 调用一个服务器工具。
- `host.readResource(uri)` — 读取一个 MCP 资源。
- `host.getPrompt(name, arguments)` — 获取一个 prompt 模板。
- `host.close()` — 关闭 UI。

每次调用仍然都会经过 MCP 协议，并继承服务器的权限。

### 权限

`_meta.ui.permissions` 列表请求额外能力：

- `camera` — 访问用户摄像头（用于扫描文档类 UI）。
- `microphone` — 语音输入。
- `geolocation` — 位置。
- `network:*` — 比仅靠 `connectSrc` 所允许的范围更宽的网络访问。

每项权限都是 UI 渲染前用户会看到的一个提示。

### 安全风险

iframe 中的 HTML 仍然是 HTML。新的攻击面包括：

- **通过 UI 进行提示注入（Prompt-injection via UI）。** 恶意服务器 UI 可以显示看起来像系统消息的文本并诱导用户。宿主渲染应明显区分服务器 UI 和宿主 UI。
- **通过 `connectSrc` 外传数据。** 如果 CSP 允许 `connect-src: *`，UI 可以把数据发送到任何地方。默认值应当严格。
- **点击劫持（Clickjacking）。** UI 覆盖宿主 chrome。宿主必须防止 z-index 操控并强制执行透明度规则。
- **窃取焦点。** UI 获取键盘焦点并捕获下一条消息。宿主必须拦截。

第 13 阶段 · 15 会在 MCP 安全中深入讲解这些内容；本课先做介绍。

### `ui/initialize` 握手

iframe 加载后，会通过 postMessage 发送 `ui/initialize`：

```json
{"jsonrpc": "2.0", "id": 0, "method": "ui/initialize",
 "params": {"theme": "dark", "locale": "en-US", "sessionId": "..."}}
```

宿主用能力列表和会话令牌响应。UI 在后续每次宿主调用中都会使用该会话令牌。

### AppRenderer / AppFrame SDK 原语

ext-apps SDK 暴露两个便捷原语：

- `AppRenderer`（服务器侧）— 包装 React / Vue / Solid 组件，并发射带有正确 MIME 和元数据的 `ui://` 资源。
- `AppFrame`（客户端侧）— 接收资源、挂载 iframe，并调解 postMessage。

你可以使用这些原语，也可以手写 HTML 和 JSON-RPC。

### 生态状态

MCP Apps 于 2026 年 1 月 26 日发布。截至 2026 年 4 月的客户端支持：

- **Claude Desktop。** 自 2026 年 1 月起完整支持。
- **ChatGPT。** 通过 Apps SDK 完整支持（底层使用同一个 MCP Apps 协议）。
- **Cursor。** Beta；通过 settings 启用。
- **VS Code。** 仅限 Insider builds。
- **Goose。** 完整支持。
- **Zed, Windsurf。** 已列入路线图。

生产中的服务器：仪表盘、地图可视化、数据表、图表构建器、沙箱 IDE 预览。

## 使用它

`code/main.py` 使用一个 `visualize_timeline` 工具扩展笔记服务器，该工具返回一个 `ui://notes/timeline` 资源；同时还提供该 URI 上的 `resources/read` 处理器，返回一个小而完整的 HTML 包，其中包含 SVG 时间线。HTML 使用 stdlib 模板化 —— 没有构建系统。由于 stdlib 无法驱动浏览器，postMessage 以 JS 注释的形式勾勒出来。

需要关注：

- 工具响应上的 `_meta.ui` 携带 resourceUri、CSP、permissions。
- HTML 在没有网络访问的情况下渲染；所有数据都已内联。
- JS 通过 `window.parent.postMessage` 调用 `host.callTool`（在此 stdlib 演示中已记录但不会实际执行）。

## 交付它

本课会生成 `outputs/skill-mcp-apps-spec.md`。给定一个会受益于交互式 UI 的工具，该 skill 会生成完整的 MCP Apps 契约：`ui://` URI、CSP、permissions、postMessage 入口点，以及安全检查清单。

## 练习

1. 运行 `code/main.py` 并检查发射出的 HTML。直接在浏览器中打开该 HTML；验证 SVG 是否渲染。然后草拟 UI 会用来调用 `host.callTool("notes_update", ...)` 的 postMessage 契约。

2. 收紧 CSP：移除 `'unsafe-inline'`，并使用基于 nonce 的脚本策略。HTML 生成代码需要做哪些修改？

3. 添加第二个 UI 资源 `ui://notes/editor`，其中包含一个用于就地编辑笔记的表单。当用户提交时，iframe 调用 `host.callTool("notes_update", ...)`。

4. 审计 UI 的攻击面。恶意服务器可能在哪里注入内容？iframe 沙箱能防御什么，不能防御什么？

5. 阅读 SEP-1724 规范，找出 MCP Apps SDK 中这个玩具实现没有使用的一项能力。（提示：组件级状态同步。）

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MCP Apps | “交互式 UI 资源” | 2026-01-26 发布的 SEP-1724 扩展 |
| `ui://` | “App URI scheme” | UI 包的资源方案 |
| `text/html;profile=mcp-app` | “The MIME” | MCP App HTML 的 content-type |
| Iframe sandbox | “渲染容器” | 使用 CSP 和权限对 UI 进行浏览器沙箱化 |
| postMessage JSON-RPC | “UI-to-host wire” | 用于宿主调用的轻量 JSON-RPC-over-postMessage 方言 |
| `_meta.ui` | “Tool-UI binding” | 将工具结果链接到 UI 资源的元数据 |
| CSP | “Content-Security-Policy” | 声明脚本、网络、样式允许来源 |
| AppRenderer | “Server SDK primitive” | 将框架组件转换成 `ui://` 资源 |
| AppFrame | “Client SDK primitive” | 调解 postMessage 的 iframe 挂载助手 |
| `ui/initialize` | “Handshake” | UI 发送给宿主的第一个 postMessage |

## 延伸阅读

- [MCP ext-apps — GitHub](https://github.com/modelcontextprotocol/ext-apps) — 参考实现和 SDK
- [MCP Apps specification 2026-01-26](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx) — 正式规范文档
- [MCP — Apps extension overview](https://modelcontextprotocol.io/extensions/apps/overview) — 高层文档
- [MCP blog — MCP Apps launch](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/) — 2026 年 1 月发布文章
- [MCP Apps API reference](https://apps.extensions.modelcontextprotocol.io/api/) — JSDoc 风格的 SDK 参考
