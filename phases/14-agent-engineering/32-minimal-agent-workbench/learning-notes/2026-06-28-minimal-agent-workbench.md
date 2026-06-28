# Minimal Agent Workbench 学习笔记

> 日期: 2026-06-28

## 1. 最小工作台的三文件架构

Agent workbench 的最小可行形态是三个文件，每个有明确分工：

| 文件 | 角色 | 类比 |
|------|------|------|
| `AGENTS.md` | 路由器，指向其他文件 | 网站首页导航 |
| `agent_state.json` | 状态快照，每轮读写 | 游戏存档点 |
| `task_board.json` | 任务队列，整个项目周期 | 项目看板 |

核心原则：**路由器越短越好，长手册会被模型忽略**。AGENTS.md 只指向 state、board 和深层规则文档，不承载具体指令。

## 2. agent_state.json vs task_board.json 的分工

```
agent_state.json  = "我现在在哪一步"    （session 级，易失）
task_board.json   = "整个项目有哪些事要做" （项目级，持久）
```

- **state** 是加速器，丢了可以从 board 推断
- **board** 是 source of truth，丢了 = 项目失忆
- 两者不一致时，**信 board 不信 state**

## 3. Claude Code 的原生支持程度

Claude Code 并不完全内置这三个文件：

| 概念 | Claude Code 实现 | 状态 |
|------|-----------------|------|
| 路由器 | AGENTS.md / CLAUDE.md | ✅ 原生 |
| 状态文件 | .claude/ 目录下自建 state.json | ⚠️ 半原生 |
| 任务板 | hooks 机制间接实现 | ⚠️ 间接 |

hooks 是 Claude Code 的事件钩子，在 PreToolUse / PostToolUse 时触发脚本，强制 agent 读写 board 文件。

## 4. hooks 的可靠性边界

hooks 不是 100% 可靠的：

| 场景 | 覆盖情况 |
|------|----------|
| agent 调用 Write/Edit 工具 | ✅ hooks 能拦截 |
| agent 只说话不写文件 | ❌ 不触发 tool use |
| 用户直接关终端 | ❌ 进程死了 |
| session 超时被杀 | ❌ 没机会触发 PostToolUse |

**实际可靠性约 90-95%**，剩余靠 CI 检查 + 降级设计 + 人工兜底。正确心态是"尽可能保证"而非"绝对保证"。

## 5. 任务颗粒度原则

**一个任务 = 一个 agent turn 能做完的事（15-30分钟）**

判断标准：如果不能用一句话描述完成状态，任务就太粗了。

```
❌ "实现用户系统"         → 太粗，断了不知从哪接
✅ "创建 User model"      → 一个 commit 可完成
✅ "实现 /signup 端点"    → 有明确验收
✅ "加输入校验"           → pytest 能验证
```

每个任务必须包含：一个明确目标、一个 owner、一个可执行的验收标准。

## 6. task_board.json vs git log

| | git log | task_board.json |
|---|---|---|
| 方向 | 回顾：发生了什么 | 前瞻：接下来做什么 |
| 状态 | 只有 merged/unmerged | todo / in_progress / done / blocked |
| 验收 | 没有 | 每个任务带 acceptance |
| 适用 | 一个人串行干活 | 多 agent 协作、跨 session 恢复 |

一个人干活时 task board 约等于 git log 的前瞻版；人多了、有 blocked 状态时才不可替代。

## 7. 跨工具通用模式

三个文件的形状跨工具不变，只是换了名字：

- **Claude Code**: AGENTS.md + .claude/state.json + hooks 驱动 board
- **Codex / Cursor**: workspace rules + session memory + chat sidebar
- **自建 Python agent**: 就是 lesson 里写的这三个文件

嵌套 AGENTS.md 的规则：从当前目录往根目录走，沿途拼接所有 AGENTS.md，**nearest wins**。

## 8. 练习实现思路（未完成）

1. **last_run 时间戳**: AgentState 加 `last_run: str`，run_one_turn 前解析时间差，>24h 则 input() 确认
2. **priority 字段**: Task 加 `priority: int`，拉取时用 `max(..., key=lambda t: t.priority)`
3. **JSON Lines 迁移**: 每行一个 JSON 对象，git diff 只显示变化的那一行
4. **lint_workbench.py**: 检查 AGENTS.md 行数、引用文件是否存在、state/board 结构完整性
5. **丢哪个最痛**: task_board.json，因为包含人类判断（优先级、验收标准），不可重建

## 9. 待消化的问题

- 任务粒度拆到多细才够？15-30 分钟是经验值，不同 agent 速度差异大
- hooks 90-95% 的可靠性在生产环境是否足够？需要什么级别的冗余？
- 多 agent 场景下 task board 的并发冲突如何解决？
