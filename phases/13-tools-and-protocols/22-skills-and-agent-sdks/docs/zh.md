# Skills and Agent SDKs — Anthropic Skills、AGENTS.md、OpenAI Apps SDK

> MCP 说明“有哪些工具”。Skills 说明“如何完成一项任务”。2026 年的技术栈会把两者分层组合。Anthropic 的 Agent Skills（开放标准，2025 年 12 月）以带有渐进式披露（progressive disclosure）的 SKILL.md 形式发布。OpenAI 的 Apps SDK 是 MCP 加上小组件元数据（widget metadata）。AGENTS.md（现在已出现在 60,000+ 个仓库中）位于仓库根目录，用作项目级 agent 上下文。本课会说明每一层覆盖什么，并构建一个最小的 SKILL.md + AGENTS.md 组合包，使其可以跨 agent 复用。

**类型：** 学习
**语言：** Python（stdlib、SKILL.md 解析器和加载器）
**先修要求：** Phase 13 · 07（MCP server）
**时间：** 约 45 分钟

## 学习目标

- 区分三层：AGENTS.md（项目上下文）、SKILL.md（可复用 know-how）、MCP（工具）。
- 编写带有 YAML frontmatter 和渐进式披露的 SKILL.md。
- 以文件系统风格将 skills 加载到 agent runtime 中。
- 将一个 skill 与 MCP server 和 AGENTS.md 组合起来，让同一个包能在 Claude Code、Cursor 和 Codex 中工作。

## 问题

一名工程师把发布说明编写流程提炼成一个多步骤提示词：“读取最新合并的 PR。按领域分组。分别总结。按照团队风格编写 changelog 条目。发布到 Slack 草稿。”他们把它放在团队的 Notion 文档中。

现在他们想从 Claude Code、Cursor 和 Codex CLI 使用这个工作流。每个 agent 加载指令的方式都不同：Claude Code slash-commands、Cursor rules、Codex `.codex.md`。这名工程师复制了三份工作流，并维护三份副本。

AGENTS.md 和 SKILL.md 合起来可以解决这个问题：

- **AGENTS.md** 位于仓库根目录。每个兼容的 agent 都会在会话开始时读取它。“这个项目如何工作？有哪些约定？运行哪些命令来测试？”
- **SKILL.md** 是一个可移植包：YAML frontmatter（name、description）+ markdown 正文 + 可选资源。支持 skills 的 agent 会按需通过名称加载它们。
- **MCP**（Phase 13 · 06-14）处理 skill 需要调用的工具。

三层，一个可移植产物。

## 概念

### AGENTS.md (agents.md)

于 2025 年末推出，到 2026 年 4 月已被 60,000+ 个仓库采用。仓库根目录下的一个文件。格式：

```markdown
# Project: my-service

## Conventions
- TypeScript with strict mode.
- Use Pydantic for models on the Python side.
- Tests run with `pnpm test`.

## Build and run
- `pnpm dev` for local dev server.
- `pnpm build` for production bundle.
```

Agent 会在会话开始时读取它，并用它来为该项目校准自己的行为。2026 年的每个 coding agent 都支持 AGENTS.md：Claude Code、Cursor、Codex、Copilot Workspace、opencode、Windsurf、Zed。

### SKILL.md 格式

Anthropic 的 Agent Skills（作为开放标准于 2025 年 12 月发布）：

```markdown
---
name: release-notes-writer
description: Write a changelog entry for the latest merged PRs following this project's style.
---

# Release notes writer

When invoked, run these steps:

1. List PRs merged since the last tag. Use `gh pr list --base main --state merged`.
2. Group by label: feature, fix, chore, docs.
3. For each PR in each group, write one line: `- <title> (#<num>)`.
4. Draft the release notes and stage them in CHANGELOG.md.

If the user says "ship", run `git tag vX.Y.Z` and `gh release create`.

## Notes

- Never include commits without a PR.
- Skip "chore" entries from the public changelog.
```

Frontmatter 声明 skill 的身份。正文是在 skill 加载时展示给模型的提示词。

### 渐进式披露

Skills 可以引用子资源，agent 只在需要时才获取这些资源。示例：

```
skills/
  release-notes-writer/
    SKILL.md
    style-guide.md
    template.md
    scripts/
      generate.sh
```

SKILL.md 会说“参见 style-guide.md 了解风格规则”。只有在 skill 正在运行时，agent 才会拉取 style-guide.md。这可以避免把模型可能不需要的细节塞进提示词中。

### 文件系统发现

Agent runtimes 会扫描已知目录来寻找 SKILL.md 文件：

- `~/.anthropic/skills/*/SKILL.md`
- Project `./skills/*/SKILL.md`
- `~/.claude/skills/*/SKILL.md`

加载会按文件夹名称和 frontmatter `name` 进行。Claude Code、Anthropic Claude Agent SDK 和 SkillKit（跨 agent）都遵循这一模式。

### Anthropic Claude Agent SDK

`@anthropic-ai/claude-agent-sdk`（TypeScript）和 `claude-agent-sdk`（Python）会在会话开始时加载 skills，并在 runtime 内部将它们暴露为可调用的“agents”。当用户调用某个 skill 时，agent loop 会分派到该 skill。

### OpenAI Apps SDK

于 2025 年 10 月推出；直接构建在 MCP 之上。把 OpenAI 之前的 Connectors 和 Custom GPT Actions 统一到一个开发者界面下。一个 Apps SDK app 包含：

- 一个 MCP server（tools、resources、prompts）。
- 加上用于 ChatGPT UI 的 widget metadata。
- 加上用于交互界面的可选 MCP Apps `ui://` resource。

同一个协议，更丰富的 UX。

### 通过 SkillKit 实现跨 agent 可移植性

SkillKit 以及类似的跨 agent 分发层等工具，会把单个 SKILL.md 翻译成 32+ 个 AI agents（Claude Code、Cursor、Codex、Gemini CLI、OpenCode 等）的原生格式。一个事实来源；多个消费者。

### 三层技术栈

| 层 | 文件 | 加载时机 | 目的 |
|-------|------|-------------|---------|
| AGENTS.md | repo root | session start | 项目级约定 |
| SKILL.md | skills directory | skill invoked | 可复用工作流 |
| MCP server | external process | tools needed | 可调用动作 |

三者可以组合：agent 在会话开始时读取 AGENTS.md，用户调用某个 skill，skill 的指令包含 MCP 工具调用，agent 通过 MCP client 进行分派。

## 使用它

`code/main.py` 提供一个基于 stdlib 的 SKILL.md 解析器和加载器。它会发现 `./skills/` 下的 skills，解析 YAML frontmatter 和 markdown 正文，并生成一个按 skill name 索引的 dict。然后它模拟一个 agent loop，通过名称调用 `release-notes-writer`。

需要关注的点：

- 使用最小 stdlib 解析器解析 YAML frontmatter（没有 `pyyaml` 依赖）。
- Skill 正文按原样存储；agent 会在调用时把它前置到 system prompt 中。
- 通过 `read_subresource` 函数演示渐进式披露，该函数会按需拉取被引用的文件。

## 交付它

本课会生成 `outputs/skill-agent-bundle.md`。给定一个工作流，skill 会生成组合后的 SKILL.md + AGENTS.md + MCP-server-blueprint 包，可跨 agents 移植。

## 练习

1. 运行 `code/main.py`。在 `skills/` 下添加第二个 skill，并确认 loader 能发现它。

2. 为本课程仓库编写一个 AGENTS.md。包含测试命令、风格约定和 Phase 13 心智模型。

3. 把你团队内部文档中的一个多步骤工作流移植到 SKILL.md。验证它能在 Claude Code 中加载。

4. 手动把这个 skill 翻译成 Cursor 和 Codex 的原生规则格式。统计各格式之间的 diff——这就是 SkillKit 自动化处理的翻译面。

5. 阅读 Anthropic Agent Skills 博客文章。找出 Claude Agent SDK 中本课 loader 未覆盖的一项功能。（提示：agent 子调用。）

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------------|
| SKILL.md | “skill 文件” | YAML frontmatter 加 markdown 正文，由 agent runtime 加载 |
| AGENTS.md | “仓库根目录 agent 上下文” | 会话开始时读取的项目级约定文件 |
| Progressive disclosure | “Lazy-load sub-resources” | Skill 正文引用只在需要时拉取的文件 |
| Frontmatter | “顶部 YAML block” | `---` 分隔符中的元数据（name、description） |
| Claude Agent SDK | “Anthropic 的 skill runtime” | `@anthropic-ai/claude-agent-sdk`，加载 skills 并路由 |
| OpenAI Apps SDK | “MCP + widget meta” | OpenAI 构建在 MCP 加 ChatGPT UI hooks 之上的开发界面 |
| Skill discovery | “文件系统扫描” | 遍历已知目录寻找 SKILL.md，并按名称索引 |
| Cross-agent portability | “一个 skill，多种 agents” | 通过 SkillKit 风格工具把一个 SKILL.md 翻译到 32+ 个 agents |
| Agent Skill | “可移植 know-how” | MCP 工具概念之外的可复用任务模板 |
| Apps SDK | “MCP 加 ChatGPT UI” | 在 MCP 上统一 Connectors 和 Custom GPTs |

## 延伸阅读

- [Anthropic — Agent Skills announcement](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — 2025 年 12 月发布
- [Anthropic — Agent Skills docs](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) — SKILL.md 格式参考
- [OpenAI — Apps SDK](https://developers.openai.com/apps-sdk) — 面向 ChatGPT、基于 MCP 的开发者平台
- [agents.md](https://agents.md/) — AGENTS.md 格式和采用列表
- [Anthropic — anthropics/skills GitHub](https://github.com/anthropics/skills) — 官方 skill 示例
