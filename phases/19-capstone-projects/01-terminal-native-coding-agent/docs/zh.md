# Capstone 01 — 终端原生编码智能体

> 到 2026 年，编码智能体的形态已经定型。一个 TUI 框架、一个有状态的计划、一个沙箱化的工具接口、一个规划-执行-观察-恢复的循环。Claude Code、Cursor 3 和 OpenCode 从远处看都一样。本毕业项目要求你从头到尾构建一个——输入 CLI，输出 Pull Request——并将其与 mini-swe-agent 和 Live-SWE-agent 在 SWE-bench Pro 上进行对比。你将了解到，困难的部分不是模型调用，而是工具循环、沙箱以及 50 轮交互的成本上限。

**类型：** 毕业项目
**语言：** TypeScript / Bun（框架），Python（评估脚本）
**前置要求：** 第 11 阶段（LLM 工程），第 13 阶段（工具与协议），第 14 阶段（智能体），第 15 阶段（自主系统），第 17 阶段（基础设施）
**涵盖阶段：** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18
**时间：** 35 小时

## 问题

编码智能体在 2026 年成为 AI 应用的主流类别。Claude Code（Anthropic）、Cursor 3 配合 Composer 2 和 Agent Tabs（Cursor）、Amp（Sourcegraph）、OpenCode（112k 星标）、Factory Droids 和 Google Jules 都采用了相同的架构变体：终端框架、带权限的工具接口、沙箱，以及围绕前沿模型构建的规划-执行-观察循环。前沿领域很窄——Live-SWE-agent 使用 Opus 4.5 在 SWE-bench Verified 上达到 79.2%——但工程实践很广。大多数故障模式不是模型错误，而是工具循环不稳定、上下文污染、token 成本失控和破坏性文件系统操作。

你无法从外部推理这些智能体。你必须构建一个，看着循环在第 47 轮因 ripgrep 返回 8MB 匹配结果而崩溃，然后重建截断层。这就是本毕业项目的意义所在。

## 概念

框架有四个接口。**Plan（规划）** 维护一个 TodoWrite 风格的状态对象，模型每轮重写。**Act（执行）** 分发工具调用（读取、编辑、运行、搜索、git）。**Observe（观察）** 捕获 stdout/stderr/退出码，截断后将摘要反馈。**Recover（恢复）** 处理工具错误，不会破坏上下文窗口或陷入无限循环。2026 年的形态增加了一个东西：**hooks（钩子）**。`PreToolUse`、`PostToolUse`、`SessionStart`、`SessionEnd`、`UserPromptSubmit`、`Notification`、`Stop` 和 `PreCompact`——可配置的扩展点，操作者在此注入策略、遥测和防护措施。

沙箱是 E2B 或 Daytona。每个任务在全新的 devcontainer 中运行，挂载一个可读写的 git worktree。框架永远不触碰主机文件系统。worktree 在成功或失败后都会被销毁。成本控制在三层执行：每轮 token 上限、每会话美元预算和硬性轮次限制（通常 50 轮）。可观测性层是带有 GenAI 语义约定的 OpenTelemetry span，发送到自托管的 Langfuse。

## 架构

```
  用户 CLI  ->  框架 (Bun + Ink TUI)
                  |
                  v
           规划 / 执行 / 观察循环  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          (通过 OpenRouter，模型无关)
                  v
           工具分发器 (MCP StreamableHTTP 客户端)
                  |
     +------------+------------+----------+
     v            v            v          v
  read/edit    ripgrep     tree-sitter   git/run
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona 沙箱  (worktree 隔离)
                  |
                  v
           hooks: Pre/Post, Session, Prompt, Compact
                  |
                  v
           OpenTelemetry -> Langfuse (spans, tokens, $)
                  |
                  v
           通过 GitHub App 发送 PR
```

## 技术栈

- 框架运行时：Bun 1.2 + Ink 5（React-in-terminal）
- 模型访问：OpenRouter 统一 API，支持 Claude Sonnet 4.7、GPT-5.4-Codex、Gemini 3 Pro、Opus 4.5（用于最难任务）
- 工具传输：Model Context Protocol StreamableHTTP（MCP 2026 修订版）
- 沙箱：E2B 沙箱（JS SDK）或 Daytona devcontainers
- 代码搜索：ripgrep 子进程，tree-sitter 解析器支持 17 种语言（预编译）
- 隔离：每个任务 `git worktree add`，成功/失败后清理
- 评估框架：SWE-bench Pro（验证子集）+ Terminal-Bench 2.0 + 自定义 30 任务保留集
- 可观测性：OpenTelemetry SDK 配合 `gen_ai.*` 语义约定 → 自托管 Langfuse
- PR 发送：GitHub App 使用细粒度 token，范围限制为目标仓库

## 构建它

1. **TUI 和命令循环。** 使用 Ink 脚手架搭建 Bun 项目。接受 `agent run <repo> "<task>"`。打印分屏视图：计划面板（顶部）、工具调用流（中部）、token 预算（底部）。添加 Ctrl-C 取消功能，退出前触发 `SessionEnd` 钩子。

2. **计划状态。** 定义类型化的 TodoWrite schema（pending/in_progress/done 项及备注）。模型每轮以工具调用重写完整状态——不要让它增量修改。将计划持久化到 `.agent/state.json`，以便崩溃后恢复。

3. **工具接口。** 定义六个工具：`read_file`、`edit_file`（带 diff 预览）、`ripgrep`、`tree_sitter_symbols`、`run_shell`（带超时）、`git`（status/diff/commit/push）。通过 MCP StreamableHTTP 暴露，使框架与传输无关。每个工具返回截断输出（每次调用上限 4k token）。

4. **沙箱封装。** 每个任务生成一个 E2B 沙箱。`git worktree add -b agent/$TASK_ID` 创建新分支。所有工具调用在沙箱内执行。主机文件系统不可达。

5. **钩子。** 实现全部八种 2026 钩子类型。接入至少四个用户编写的钩子：(a) `PreToolUse` 破坏性命令防护，阻止 worktree 外的 `rm -rf`；(b) `PostToolUse` token 计费；(c) `SessionStart` 预算初始化；(d) `Stop` 写入最终追踪包。

6. **评估循环。** 克隆 SWE-bench Pro Python 的 30 问题子集。对每个运行你的框架。与 mini-swe-agent（最小基线）比较 pass@1、每任务轮次和每任务美元成本。将结果写入 `eval/results.jsonl`。

7. **成本控制。** 硬性截止：50 轮、200k 上下文、每任务 $5。`PreCompact` 钩子在 150k 标记处将较旧轮次摘要为先前状态块，为新观察腾出空间而不丢失计划。

8. **PR 发送。** 成功后，最后一步是 `git push` + GitHub API 调用，打开一个 PR，正文包含计划和 diff 摘要。

## 使用它

```
$ agent run ./my-repo "Fix the race condition in worker.rs"
[plan]  1 定位 worker.rs 并枚举 mutex 使用
        2 识别竞争下的共享状态
        3 提出修复方案，验证测试
[tool]  ripgrep mutex.*lock -t rust           (44 matches, truncated)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (passed)
[plan]  1 done · 2 done · 3 done
[done]  PR opened: #482   turns=9   tokens=38k   cost=$0.41
```

## 交付它

可交付的 skill 位于 `outputs/skill-terminal-coding-agent.md`。给定仓库路径和任务描述，它在沙箱中运行完整的规划-执行-观察循环，返回 PR URL 和追踪包。本毕业项目的评分标准：

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 vs 基线 | 你的框架 vs mini-swe-agent 在 30 个匹配 Python 任务上的表现 |
| 20 | 架构清晰度 | Plan/Act/Observe 分离、钩子接口、工具 schema——对照 Live-SWE-agent 布局审查 |
| 20 | 安全性 | 沙箱逃逸测试、权限提示、破坏性命令防护通过红队测试 |
| 20 | 可观测性 | 追踪完整性（100% 工具调用有 span）、每轮 token 计费 |
| 15 | 开发者体验 | 冷启动 < 2s、崩溃恢复继续计划、Ctrl-C 干净取消工具执行 |
| **100** | | |

## 练习

1. 将底层模型从 Claude Sonnet 4.7 换成 vLLM 上的 Qwen3-Coder-30B。比较 pass@1 和每任务美元成本。报告开源模型表现不足的地方。

2. 添加一个 `reviewer` 子智能体，在 PR 发送前读取 diff 并可请求修订循环。测量误报审查是否将 SWE-bench 通过率降至单智能体基线以下（提示：通常是的）。

3. 压力测试沙箱：编写一个尝试 `curl` 外部 URL 的任务和一个在 worktree 外写入的任务。确认两者都被 PreToolUse 钩子阻止。记录尝试。

4. 用更小的模型（Haiku 4.5）实现 `PreCompact` 摘要。测量 3x 压缩时计划保真度损失多少。

5. 将 MCP StreamableHTTP 传输换成 stdio。基准测试冷启动和每次调用延迟。为纯本地使用选出赢家。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| Harness（框架） | "智能体循环" | 围绕模型的代码，分发工具、维护计划状态并执行预算 |
| Hook（钩子） | "智能体事件监听器" | 用户编写的脚本，在八个生命周期事件之一由框架运行 |
| Worktree | "Git 沙箱" | 链接到独立路径的 git checkout；可丢弃而不影响主克隆 |
| TodoWrite | "计划状态" | 类型化的 pending/in-progress/done 项列表，模型每轮重写 |
| StreamableHTTP | "MCP 传输" | 2026 MCP 修订版：长期 HTTP 连接，支持双向流；替代 SSE |
| Token ceiling（token 上限） | "上下文预算" | 每轮或每会话的输入+输出 token 上限；触发压缩或终止 |
| pass@1 | "单次尝试通过率" | 首次运行无需重试或窥探测试集即解决的 SWE-bench 任务比例 |

## 延伸阅读

- [Claude Code 文档](https://docs.anthropic.com/en/docs/claude-code) — Anthropic 的参考框架
- [Cursor 3 更新日志](https://cursor.com/changelog) — Agent Tabs 和 Composer 2 产品说明
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — SWE-bench 框架对比的最小基线
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) — 使用 Opus 4.5 达到 79.2% SWE-bench Verified
- [OpenCode](https://opencode.ai) — 开源框架，112k 星标
- [SWE-bench Pro 排行榜](https://www.swebench.com) — 本毕业项目对标评估
- [Model Context Protocol 2026 路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP、能力元数据
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 工具调用和 token 使用的 span schema

---

# English Original

# Capstone 01 — Terminal-Native Coding Agent

> By 2026 the shape of a coding agent is settled. A TUI harness, a stateful plan, a sandboxed tool surface, a loop that plans, acts, observes, recovers. Claude Code, Cursor 3, and OpenCode all look the same from 50 feet. This capstone asks you to build one end to end — CLI in, pull request out — and measure it against mini-swe-agent and Live-SWE-agent on SWE-bench Pro. You will learn why the hard part is not the model call but the tool loop, the sandbox, and the cost ceiling on a 50-turn run.

**Type:** Capstone
**Languages:** TypeScript / Bun (harness), Python (eval scripts)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools and protocols), Phase 14 (agents), Phase 15 (autonomous systems), Phase 17 (infrastructure)
**Phases exercised:** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18
**Time:** 35 hours

## Problem

Coding agents became the dominant AI application category in 2026. Claude Code (Anthropic), Cursor 3 with Composer 2 and Agent Tabs (Cursor), Amp (Sourcegraph), OpenCode (112k stars), Factory Droids, and Google Jules all ship variations of the same architecture: a terminal harness, a permissioned tool surface, a sandbox, and a plan-act-observe loop built around a frontier model. The frontier is narrow — Live-SWE-agent reached 79.2% on SWE-bench Verified with Opus 4.5 — but the engineering craft is wide. Most failure modes are not model mistakes. They are tool-loop instability, context poisoning, runaway token cost, and destructive filesystem operations.

You cannot reason about these agents from the outside. You have to build one, watch the loop crash on turn 47 when ripgrep returns 8MB of matches, and rebuild the truncation layer. That is the point of this capstone.

## Concept

The harness has four surfaces. **Plan** maintains a TodoWrite-style state object that the model rewrites each turn. **Act** dispatches tool calls (read, edit, run, search, git). **Observe** captures stdout / stderr / exit codes, truncates, and feeds the summary back. **Recover** handles tool errors without blowing the context window or looping forever. The 2026 shape adds one more thing: **hooks**. `PreToolUse`, `PostToolUse`, `SessionStart`, `SessionEnd`, `UserPromptSubmit`, `Notification`, `Stop`, and `PreCompact` — configurable extension points where the operator injects policy, telemetry, and guardrails.

The sandbox is E2B or Daytona. Each task runs in a fresh devcontainer with a git worktree mounted read-write. The harness never touches the host filesystem. The worktree gets torn down on success or failure. Cost control is enforced at three layers: a per-turn token ceiling, a per-session dollar budget, and a hard turn limit (typically 50). The observability layer is OpenTelemetry spans with GenAI semantic conventions, shipped to a self-hosted Langfuse.

## Architecture

```
  user CLI  ->  harness (Bun + Ink TUI)
                  |
                  v
           plan / act / observe loop  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          (via OpenRouter, model-agnostic)
                  v
           tool dispatcher (MCP StreamableHTTP client)
                  |
     +------------+------------+----------+
     v            v            v          v
  read/edit    ripgrep     tree-sitter   git/run
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona sandbox  (worktree isolated)
                  |
                  v
           hooks: Pre/Post, Session, Prompt, Compact
                  |
                  v
           OpenTelemetry -> Langfuse (spans, tokens, $)
                  |
                  v
           PR via GitHub app
```

## Stack

- Harness runtime: Bun 1.2 + Ink 5 (React-in-terminal)
- Model access: OpenRouter unified API with Claude Sonnet 4.7, GPT-5.4-Codex, Gemini 3 Pro, Opus 4.5 (for hardest tasks)
- Tool transport: Model Context Protocol StreamableHTTP (MCP 2026 revision)
- Sandbox: E2B sandboxes (JS SDK) or Daytona devcontainers
- Code search: ripgrep subprocess, tree-sitter parsers for 17 languages (pre-compiled)
- Isolation: `git worktree add` per task, cleanup on success / failure
- Eval harness: SWE-bench Pro (verified subset) + Terminal-Bench 2.0 + your own 30-task holdout
- Observability: OpenTelemetry SDK with `gen_ai.*` semconv → self-hosted Langfuse
- PR posting: GitHub App with fine-grained token, scope limited to the target repo

## Build It

1. **TUI and command loop.** Scaffold a Bun project with Ink. Accept `agent run <repo> "<task>"`. Print a split view: plan pane (top), tool-call stream (middle), token budget (bottom). Add cancel on Ctrl-C that fires `SessionEnd` hook before exit.

2. **Plan state.** Define a typed TodoWrite schema (pending / in_progress / done items with notes). Model rewrites the full state each turn as a tool call — do not let it mutate incrementally. Persist plan to `.agent/state.json` so crashes can resume.

3. **Tool surface.** Define six tools: `read_file`, `edit_file` (with diff preview), `ripgrep`, `tree_sitter_symbols`, `run_shell` (with timeout), `git` (status / diff / commit / push). Expose over MCP StreamableHTTP so the harness is transport-agnostic. Every tool returns truncated output (cap at 4k tokens per call).

4. **Sandbox wrapping.** Each task spawns an E2B sandbox. `git worktree add -b agent/$TASK_ID` a fresh branch. All tool calls execute inside the sandbox. Host filesystem is unreachable.

5. **Hooks.** Implement all eight 2026 hook types. Wire at least four user-authored hooks: (a) `PreToolUse` destructive-command guard that blocks `rm -rf` outside the worktree, (b) `PostToolUse` token accounting, (c) `SessionStart` budget initialization, (d) `Stop` writes a final trace bundle.

6. **Eval loop.** Clone a 30-issue subset of SWE-bench Pro Python. Run your harness against each. Compare to mini-swe-agent (the minimal baseline) on pass@1, turns-per-task, and $-per-task. Write the results to `eval/results.jsonl`.

7. **Cost control.** Hard cutoffs: 50 turns, 200k context, $5 per task. `PreCompact` hook summarizes older turns into a prior-state block at the 150k mark, freeing room for new observations without losing the plan.

8. **PR posting.** On success, the final step is `git push` + a GitHub API call that opens a PR with the plan and the diff summary in the body.

## Use It

```
$ agent run ./my-repo "Fix the race condition in worker.rs"
[plan]  1 locate worker.rs and enumerate mutex uses
        2 identify shared state under contention
        3 propose fix, verify tests
[tool]  ripgrep mutex.*lock -t rust           (44 matches, truncated)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (passed)
[plan]  1 done · 2 done · 3 done
[done]  PR opened: #482   turns=9   tokens=38k   cost=$0.41
```

## Ship It

The deliverable skill lives in `outputs/skill-terminal-coding-agent.md`. Given a repo path and a task description, it runs the full plan-act-observe loop in a sandbox and returns a PR URL plus a trace bundle. The rubric for this capstone:

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 vs baseline | Your harness vs mini-swe-agent on 30 matched Python tasks |
| 20 | Architecture clarity | Plan/act/observe separation, hook surface, tool schema — reviewed against Live-SWE-agent layout |
| 20 | Safety | Sandbox escape tests, permission prompts, destructive-command guard passes red-team |
| 20 | Observability | Trace completeness (100% of tool calls spanned), token accounting per turn |
| 15 | Developer UX | Cold-start < 2s, crash recovery resumes plan, Ctrl-C cancels mid-tool cleanly |
| **100** | | |

## Exercises

1. Swap the backing model from Claude Sonnet 4.7 to Qwen3-Coder-30B served on vLLM. Compare pass@1 and $-per-task. Report where the open model underperforms.

2. Add a `reviewer` sub-agent that reads the diff before PR posting and can request a revision loop. Measure whether false-positive reviews drop SWE-bench pass rate below the single-agent baseline (hint: usually yes).

3. Stress-test the sandbox: write a task that tries to `curl` an external URL and a task that writes outside the worktree. Confirm both are blocked by the PreToolUse hook. Log the attempts.

4. Implement `PreCompact` summarization with a smaller model (Haiku 4.5). Measure how much plan fidelity is lost at 3x compaction.

5. Swap MCP StreamableHTTP transport for stdio. Benchmark cold-start and per-call latency. Pick a winner for local-only use.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Harness | "The agent loop" | The code surrounding the model that dispatches tools, maintains plan state, and enforces budgets |
| Hook | "Agent event listener" | A user-authored script run on one of eight lifecycle events by the harness |
| Worktree | "Git sandbox" | A linked git checkout at a separate path; disposable without touching the main clone |
| TodoWrite | "Plan state" | A typed list of pending/in-progress/done items the model rewrites each turn |
| StreamableHTTP | "MCP transport" | 2026 MCP revision: long-lived HTTP connection with bidirectional streaming; replaces SSE |
| Token ceiling | "Context budget" | Per-turn or per-session cap on input+output tokens; triggers compaction or termination |
| pass@1 | "Single-attempt pass rate" | Fraction of SWE-bench tasks solved on the first run without retry or test-set peeking |

## Further Reading

- [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code) — reference harness from Anthropic
- [Cursor 3 changelog](https://cursor.com/changelog) — Agent Tabs and Composer 2 product notes
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — minimal baseline for SWE-bench harness comparison
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) — 79.2% SWE-bench Verified with Opus 4.5
- [OpenCode](https://opencode.ai) — open harness, 112k stars
- [SWE-bench Pro leaderboard](https://www.swebench.com) — the evaluation this capstone targets
- [Model Context Protocol 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP, capability metadata
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — span schema for tool calls and token usage
