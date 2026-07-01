# Workbench 体系学习笔记

> 日期: 2026-07-01

## 1. Multi-Session Handoff：会话交接

Session 结束后，新 agent 花大量时间重新发现进度。解决方法是生成一个**交接包**（handoff packet），包含 7 个字段：

| 字段 | 回答什么 |
|------|----------|
| `summary` | 做了什么 |
| `changed_files` | 改了哪些文件 |
| `commands_run` | 执行了什么命令 |
| `failed_attempts` | 尝试过什么，为什么失败 |
| `open_risks` | 下次可能出问题的地方 |
| `next_action` | **下一个 session 的第一步**（最关键） |
| `verdict_pointer` | 验证/审查报告路径 |

关键原则：**生成而非手写**（手写会在困难时被跳过），**双格式**（md 给人读，json 给 agent 读），**干净状态**（交接前先清理）。

## 2. Handoff vs Compaction

| | Handoff | Compaction |
|---|---------|------------|
| 目的 | 关闭 session，开启新 session | 延长当前 session |
| 时机 | 50-75% 上下文时 | 接近上下文上限时 |
| 产物 | 便携式交接包 | 上下文压缩 |

两者配合使用，但不能混为一谈。拖到 95% 上下文才交接会导致质量下降。

## 3. Workbench 是什么

Workbench（工作台）是**结构化的 agent 工作环境**，由 7 个"表面"组成：

| 课程 | 表面 | 作用 |
|------|------|------|
| 34 | agent_state.json | 状态追踪 |
| 35 | 初始化脚本 | 环境准备 |
| 36 | scope contract | 范围约束 |
| 37 | feedback runner | 接受度测试执行 |
| 38 | verification gate | 验证门控 |
| 39 | reviewer | 审查报告 |
| 40 | handoff packet | 会话交接 |

类比木工工作台：台钳、量具、图纸、工具架都在固定位置，不需要每次重新找工具。

## 4. 为什么需要 Workbench

实验数据：
- 同一个模型，换 harness 可以从 Terminal Bench Top-30 跳到 Top-5
- Vercel 删掉 80% 工具，成功率从 80% 提到 100%
- 88% 企业 AI agent 项目失败原因是**运行时问题**（状态过期、重试脆弱、上下文膨胀），不是推理能力

Prompt-only vs Workbench-guided 的区别：前者没有范围约束、没有验证门控、没有交接，后者每步都有记录。

## 5. 工具（Tools）的广义理解

Vercel "删除 80% 工具"中的工具是广义的，包括：

| 类型 | 例子 |
|------|------|
| 原生工具 | Read, Write, Bash, Grep |
| Skills | tdd, diagnose, brainstorming |
| MCP 服务器 | 飞书、GitHub 等外部集成 |
| 自定义函数 | 用户定义的 function call |

**Skill 算工具**，该删也要删。删工具的好处：选择空间缩小、上下文更干净、减少误用。负空间（不做什么）比正空间（做什么）更重要。

## 6. Workbench Pack：可复用的工作台包

Capstone 课将 7 个表面打包成一个可 `cp -r` 的目录：

```
agent-workbench-pack/
├── AGENTS.md           # agent 规则
├── docs/               # 规则文档
├── schemas/            # JSON schema
├── scripts/            # 运行时脚本
├── bin/install.sh      # 一键安装
└── README.md
```

`install.sh` 做 4 件事：防覆盖、复制 pack、接 CI、打印后续步骤。幂等运行。

## 7. 在 Claude Code 中使用 Workbench

需要按项目配置，不是从零开始：

| 层 | 来源 | 内容 |
|---|------|------|
| 通用层 | workbench pack | schemas、脚本、文档模板 |
| 项目层 | 每个 repo | scope contract、acceptance command、task board |

Claude Code 原生支持 `CLAUDE.md`（相当于 AGENTS.md），其他表面（agent_state.json、scope_contract.json 等）需要手动补充。正是那 20% 的项目定制让 agent 不会越界。

## 8. 待消化的问题

- 如何判断一个工具/ skill 是否该从当前项目中删除？
- Workbench 的 7 个表面中，哪些对小项目是必须的，哪些可以省略？
- 如果团队有多个 agent 产品（Claude Code + Codex），如何维护一份统一的 pack？
