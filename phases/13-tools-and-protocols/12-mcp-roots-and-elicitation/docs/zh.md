# Roots and Elicitation — 作用域限定与运行中的用户输入

> 硬编码路径会在用户打开另一个项目的瞬间失效。预先填好的工具参数会在用户描述不充分时失效。Roots 将服务器限定在一组由用户控制的 URI 范围内；elicitation 会在工具调用进行到一半时暂停，通过表单或 URL 向用户请求结构化输入。两个客户端原语，分别修复 MCP 常见的两类失败模式。SEP-1036（URL 模式 elicitation，2025-11-25）在 2026 上半年之前仍属于实验性功能——依赖它之前请检查 SDK 版本。

**类型：** 构建
**语言：** Python（标准库，roots + elicitation 演示）
**前置知识：** Phase 13 · 07（MCP server）
**时间：** 约 45 分钟

## Learning Objectives

- 声明 `roots` 并响应 `notifications/roots/list_changed`。
- 将服务器文件操作限制在已声明 root 集合内的 URI 中。
- 使用 `elicitation/create` 在工具调用过程中向用户请求确认或结构化输入。
- 在表单模式（form-mode）和 URL 模式（URL-mode）elicitation 之间做选择（后者是实验性的；已注明漂移风险）。

## The Problem

一个 notes MCP server 在生产环境中会遇到两个具体失败。

**路径假设失效。** 服务器按 `~/notes` 编写。另一台机器上的用户把笔记放在 `~/Documents/Notes`，工具调用要么静默失败（找不到文件），更糟的是写到了错误位置。

**缺少用户才知道的参数。** 用户说“delete the old TPS report note”。模型调用 `notes_delete(title: "TPS report")`，但有三条匹配笔记，分别来自 2023、2024 和 2025 年。工具不能猜。返回“ambiguous”很烦人；对三条都执行则是灾难。

Roots 修复第一个问题：客户端在 `initialize` 时声明服务器可以触碰的一组 URI。Elicitation 修复第二个问题：服务器暂停工具调用并发送 `elicitation/create`，让用户选择具体是哪一个。

## The Concept

### Roots

客户端在 `initialize` 时声明 root 列表：

```json
{
  "capabilities": {"roots": {"listChanged": true}}
}
```

然后服务器可以调用 `roots/list`：

```json
{"roots": [{"uri": "file:///Users/alice/Documents/Notes", "name": "Notes"}]}
```

服务器必须（MUST）把 roots 当作边界：任何 root 集合之外的文件读写都要被拒绝。这并不是由客户端强制执行的（服务器仍然是用户信任的代码），但符合规范的服务器会遵守它。

当用户添加或移除某个 root 时，客户端发送 `notifications/roots/list_changed`。服务器重新调用 `roots/list` 并更新自己的边界。

### Why roots are a client primitive

Roots 由客户端声明，因为它们代表用户的授权模型。用户告诉 Claude Desktop：“允许这个 notes server 访问这两个目录”。服务器不能自行扩大这个作用域。

### Elicitation: the form-mode default

`elicitation/create` 接收一个表单 schema 加一段自然语言提示：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Delete 'TPS report'? Multiple notes match; pick one.",
    "requestedSchema": {
      "type": "object",
      "properties": {
        "note_id": {
          "type": "string",
          "enum": ["note-3", "note-7", "note-14"]
        },
        "confirm": {"type": "boolean"}
      },
      "required": ["note_id", "confirm"]
    }
  }
}
```

客户端渲染一个表单，收集用户答案，然后返回：

```json
{
  "action": "accept",
  "content": {"note_id": "note-14", "confirm": true}
}
```

有三种可能的动作：`accept`（用户填好了）、`decline`（用户关闭了它）、`cancel`（用户中止了整个工具调用）。

表单 schema 是扁平的——v1 不支持嵌套对象。SDK 通常会拒绝任何复杂度超过单层的 schema。

### Elicitation: URL mode (SEP-1036, experimental)

2025-11-25 新增。服务器不发送 schema，而是发送一个 URL：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Sign in to GitHub",
    "url": "https://github.com/login/oauth/authorize?client_id=..."
  }
}
```

客户端在浏览器中打开该 URL，等待完成，并在用户返回后返回结果。这适用于 OAuth 流程、支付授权、文档签署等表单不足以表达的场景。

漂移风险说明：SEP-1036 的响应形状仍在稳定中；有些 SDK 返回回调 URL，另一些返回完成 token。在生产中使用 URL 模式之前，请阅读你所用 SDK 的发布说明。

### When elicitation is the right tool

- 破坏性操作前的用户确认（destructive hint + elicitation）。
- 消歧（从 N 个匹配项中选择一个）。
- 首次运行设置（API keys、目录、偏好设置）。
- OAuth 风格流程（URL 模式）。

### When elicitation is wrong

- 填补模型本可以用自然语言追问的工具必填参数。应使用普通的重新提示，而不是 elicitation 对话框。
- 高频调用。Elicitation 会打断对话；不要在循环内部触发它。
- 任何服务器可以事后验证的内容。验证、返回错误，让模型用文本向用户询问。

### Human-in-the-loop bridge

Elicitation 加上 sampling 共同启用 MCP 的“human-in-the-loop（人在回路中）”模型。服务器的 agent loop 可以暂停以等待用户输入（elicitation）或模型推理（sampling）。Phase 13 · 11 已介绍 sampling；本课介绍 elicitation。把它们组合起来，就能获得完整的循环中控制能力。

## Use It

`code/main.py` 将 notes server 扩展为包含：

- `roots/list` 响应，服务器会在 root-list-changed 通知之后重新查询它。
- 一个 `notes_delete` 工具：当多条笔记匹配时，使用 `elicitation/create` 消歧。
- 一个 `notes_setup` 工具：使用 URL 模式 elicitation 打开首次运行配置页（模拟）。
- 一个边界检查：拒绝对已声明 roots 之外 URI 的操作。

演示运行三个场景：happy path（一个匹配项）、disambiguation（三个匹配项，触发 elicitation）、out-of-root-write（被拒绝）。

## Ship It

本课会产出 `outputs/skill-elicitation-form-designer.md`。给定一个可能需要用户确认或消歧的工具，该 skill 会设计 elicitation 表单 schema 和消息模板。

## Exercises

1. 运行 `code/main.py`。触发消歧路径；确认模拟用户答案会被路由回工具。

2. 添加一个新的 `notes_archive` 工具，要求每次都进行 elicitation 确认（destructive hint）。检查 UX：这与模型用文本重新询问相比如何？

3. 为首次运行 OAuth 流程实现 URL 模式 elicitation。注意漂移风险，并添加 SDK 版本保护。

4. 扩展 `roots/list` 处理：当收到通知时，服务器应原子地重新读取并重新扫描可能已经超出作用域的打开文件句柄。

5. 阅读 GitHub 上的 SEP-1036 issue 讨论线程。找出一个会影响服务器如何处理 URL 模式回调的开放问题。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Root | “授权边界” | 客户端允许服务器触碰的 URI |
| `roots/list` | “服务器请求作用域” | 客户端返回当前 root 集合 |
| `notifications/roots/list_changed` | “用户更改了作用域” | 客户端发出 root 集合已变更的信号 |
| Elicitation | “调用中途询问用户” | 由服务器发起的结构化用户输入请求 |
| `elicitation/create` | “这个方法” | 用于 elicitation 请求的 JSON-RPC 方法 |
| Form mode | “由 schema 驱动的表单” | 在客户端 UI 中渲染为表单的扁平 JSON Schema |
| URL mode | “浏览器重定向” | SEP-1036 实验性功能；打开 URL 并等待 |
| `accept` / `decline` / `cancel` | “用户响应结果” | 服务器要处理的三个分支 |
| Disambiguation | “选一个” | 当工具有 N 个候选项时，常见的 elicitation 用例 |
| Flat form | “仅顶层属性” | Elicitation schema 不能嵌套 |

## Further Reading

- [MCP — Client roots spec](https://modelcontextprotocol.io/specification/draft/client/roots) — roots 的权威参考
- [MCP — Client elicitation spec](https://modelcontextprotocol.io/specification/draft/client/elicitation) — elicitation 的权威参考
- [Cisco — What's new in MCP elicitation, structured content, OAuth enhancements](https://blogs.cisco.com/developer/whats-new-in-mcp-elicitation-structured-content-and-oauth-enhancements) — 2025-11-25 新增内容导览
- [MCP — GitHub SEP-1036](https://github.com/modelcontextprotocol/modelcontextprotocol) — URL 模式 elicitation 提案（实验性，有漂移风险）
- [The New Stack — How elicitation brings human-in-the-loop to AI tools](https://thenewstack.io/how-elicitation-in-mcp-brings-human-in-the-loop-to-ai-tools/) — UX walkthrough
