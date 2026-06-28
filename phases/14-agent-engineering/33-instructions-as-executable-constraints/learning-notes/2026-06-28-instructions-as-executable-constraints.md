# Agent Instructions as Executable Constraints

> 日期: 2026-06-28

## 1. 核心思想

**写成散文的指令是愿望；写成约束的指令是测试。**

`AGENTS.md` 常见膨胀原因：每次事故加一条规则但从不删除，一年后两千行，agent 只读第一屏。把"要小心"变成"如果 X 发生就 FAIL"，规则就从愿望变成了可执行的约束。

## 2. 五类规则框架

几乎所有 agent 指令都可以归入这五个可检查的类别：

| 类别 | 回答的问题 | 示例 |
|------|-----------|------|
| **Startup** | 开工前必须满足什么？ | "必须先读取 state 文件" |
| **Forbidden** | 绝对不能做什么？ | "不能编辑 release.sh" |
| **Definition of Done** | 什么证明任务完成了？ | "测试退出码为 0" |
| **Uncertainty** | 不确定时怎么办？ | "写问题笔记而不是猜测" |
| **Approval** | 什么需要人工审批？ | "添加新依赖需要批准" |

不符合这五个类别的规则，通常应该拆成两条。

## 3. 规则必须是机器可检查的

每条规则带一个 `check` 字段，指向 `RuleChecker` 类中的一个函数：

```
## startup/state-file-fresh
- category: startup
- check: state_file_fresh          ← 指向一个可执行函数
Agent must read agent_state.json before any tool call.
```

没有 `check` 的规则 = 愿望，直接删掉。

## 4. 渐进式披露：分层架构

| 层级 | 位置 | 何时读取 | 大小限制 |
|------|------|---------|---------|
| **Router** | `AGENTS.md` | 每次会话 | < 50 行 |
| **Rules** | `docs/agent-rules.md` 或 `.claude/rules/` | 每次会话启动时 | 每类别一屏 |
| **Topic docs** | `docs/<topic>.md` | 仅当任务涉及该主题 | 按需深入 |

AGENTS.md 只写路由：这个 repo 是什么、规则文件在哪、几条硬规则。

## 5. Claude Code 原生规则位置

课程用 `docs/agent-rules.md` 是跨 agent 通用方案。Claude Code 原生约定：

```
.claude/
├── rules/          # Claude Code 自动读取
│   ├── startup.md
│   ├── forbidden.md
│   └── done.md
└── CLAUDE.md       # 根配置（路由，< 50 行）
```

## 6. 规则从哪里来

三个来源，优先级递减：

1. **从事故中提取** — 每次出问题，问"哪条规则可以拦住这个 bug"。重复出现的事故才值得加规则。
2. **从现有文档分类** — 翻 CONTRIBUTING.md、PR template、CI 配置，把散文拆成五类，没有 check 的丢掉。
3. **从 agent 行为观察** — 让 agent 跑真实任务，记录它犯的错，再加规则拦它。

**没出过问题就不用写规则。** 强行写的规则没人遵守，还会膨胀到 80+ 条，agent 读不完。

## 7. 测试基线

"改之前先跑一遍现有测试，建立基线"中的**基线 = 改动前测试的结果状态**。

- 本来通过 → 现在失败 = 你引入了 bug
- 本来失败 → 现在通过 = 你修了一个 bug
- 本来通过 → 现在还是通过 = 没破坏现有功能

没有基线，跑完测试发现失败，你不知道是不是原来就失败的。

## 8. 项目中最常见的 Rules

### Startup
- 先读 CLAUDE.md / AGENTS.md
- 确认当前分支，不在 main 上直接改
- 改之前先跑一遍测试建立基线

### Forbidden（最重要的一类）
- 不提交 secrets/keys/env 文件
- 不 force push 到 main
- 不删测试文件来让测试通过
- 不改 API 签名未经审批
- 不改 lock 文件只为了"顺便修一下"

### Definition of Done
- 测试全部通过
- Lint / type check 通过
- 新功能有对应测试

### Uncertainty
- 架构决策先问
- 改依赖版本先问
- 任务模糊时先写问题笔记再动手

### Approval
- 添加新依赖
- 数据库 schema 变更
- API 接口变更

## 9. 生产模式

| 模式 | 做法 | 数据支撑 |
|------|------|---------|
| **Severity tagging** | `block`（阻断）/ `warn`（警告）/ `info`（信息） | 在写规则时标定，避免后期压力下弱化 |
| **Rule expiry** | 默认 90 天过期，60 天无违规触发季度审查 | Cloudflare 数据：有过期机制的规则集 <30 条，没有的膨胀到 80+ |
| **Markdown-as-source, JSON-as-cache** | `agent-rules.md` 是可审阅源文件，`agent-rules.lock.json` 是热路径缓存 | 类似 `package.json` + `package-lock.json` |

## 10. 规则与框架 Guardrails 的关系

| 层级 | 作用 | 示例 |
|------|------|------|
| **规则集** | 人可读、可审阅的合同 | markdown 文件 + check 函数 |
| **框架 guardrails** | 运行时强制执行 | OpenAI SDK guardrails、LangGraph interrupts |

两者都需要：runtime 在每次 turn 拦截违规；规则集证明 runtime 做的是对的。
