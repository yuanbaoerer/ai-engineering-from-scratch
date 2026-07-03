---
name: loops-md-study
version: 1.0.0
description: 基于 Andrej Karpathy《LOOPS.md》的 9 条规则学习路线。触发词："LOOPS.md"、"Karpathy loops"、"loops 学习"、"agent 九条规则"、"/loops-study"。
---

# LOOPS.md 学习路线

将 Andrej Karpathy《LOOPS.md: Field Notes on Agents That Run for Days》的 9 条规则，映射到本项目课程，提供可执行的 9 日学习路线、每日任务卡、验收清单与进度跟踪。

## Activation

当用户输入以下内容时激活本 skill：
- `/loops-study`
- "LOOPS.md 学习"
- "Karpathy loops 学习"
- "loops 学习路线"
- "agent 九条规则"
- "整理 loops 计划"
- "生成今天的 loops 任务卡"
- "更新 loops 进度 Day N 完成"
- "查看 loops 学习进度"

## Input

可选参数：
- `day=1..9`：指定具体学习天
- `mode=plan|daily|review|export`：计划 / 每日任务 / 最终复盘 / 导出
- `rule=I|II|...|IX`：指定某条规则深入学习

若无参数，默认输出：9 日总览 + 当天任务卡。

## Procedure

### Step 1: 读取课程分布

从项目中定位 LOOPS.md 9 条规则对应的课程：

| 规则 | 主题 | 核心课程 |
|------|------|----------|
| I | Write the loop, not the prompt | `phases/14-agent-engineering/01-the-agent-loop` |
| II | Separate the roles | `phases/14-agent-engineering/39-reviewer-agent` |
| III | Negotiate the contract first | `phases/14-agent-engineering/36-scope-contracts` |
| IV | Write to disk, not to context | `phases/14-agent-engineering/34-repo-memory-and-state` |
| V | Let the loop restart | `phases/14-agent-engineering/40-multi-session-handoff` |
| VI | Score the subjective | `phases/14-agent-engineering/39-reviewer-agent` |
| VII | Read the traces | `phases/14-agent-engineering/37-runtime-feedback-loops` |
| VIII | Delete the harness | `phases/14-agent-engineering/31-agent-workbench-why-models-fail` |
| IX | The bottleneck always moves | `phases/14-agent-engineering/30-eval-driven-agent-development` |

扩展课程：
- `phases/14-agent-engineering/32-minimal-agent-workbench`
- `phases/14-agent-engineering/33-instructions-as-executable-constraints`
- `phases/14-agent-engineering/38-verification-gates`
- `phases/14-agent-engineering/41-workbench-for-real-repos`
- `phases/14-agent-engineering/42-agent-workbench-capstone`
- `phases/19-capstone-projects/20-agent-harness-loop-contract`
- `phases/19-capstone-projects/24-plan-execute-control-flow`
- `phases/19-capstone-projects/27-eval-harness-fixture-tasks`
- `phases/19-capstone-projects/28-observability-otel-traces`

参考学习笔记：
- `phases/14-agent-engineering/31-agent-workbench-why-models-fail/learning-notes/2026-07-04-loops-md-nine-rules.md`

### Step 2: 生成/读取 9 日计划

按 9 日结构输出：
- Day 1：Rule I — 写循环，不是写提示词
- Day 2：Rule II — 分离角色
- Day 3：Rule III — 先协商契约
- Day 4：Rule IV — 写到磁盘，不要写到上下文
- Day 5：Rule V — 让循环重启
- Day 6：Rule VI — 给主观事物评分
- Day 7：Rule VII — 读取轨迹
- Day 8：Rule VIII — 删除 harness
- Day 9：Rule IX — 瓶颈总在移动 + 最终复盘

每日包含 1 个核心规则 + 1–2 个映射课程 + 1 个动手任务。

### Step 3: 生成每日任务卡

每日任务卡固定字段：
- 规则编号与名称
- 课程路径
- 时长（默认 ~75min）
- 学习目标
- 阅读步骤
- 运行命令
- 预期输出
- 验收标准
- 产出物

示例命令模板：
```bash
python3 code/main.py
python3 -m pytest code/tests/ -v
```

### Step 4: 验收与复盘

第 9 日输出最终复盘模板：
- 9 条规则对应的项目课程
- 产出清单（路径）
- 阻塞点
- 下一步行动（进入 Phase 19 harness 毕业项目或回到 Workbench 实践）

### Step 5: 进度跟踪（按天）

读取并更新：`.claude/skills/loops-md-study/loops-study-progress.md`

触发词：
- "更新 loops 进度 Day N 完成"
- "查看 loops 学习进度"

更新规则：
- 将 Day N 状态改为“已完成”
- 填写完成时间与备注
- 将“当前位置”推进到下一天
- 若存在阻塞点，记录到“阻塞点记录”

### Step 6: 导出与复盘

支持导出：
- `loops-study-plan.md`（总计划）
- `loops-study-daily-dayN.md`（每日任务卡）
- `loops-study-review.md`（最终复盘）

## Rules

- 每天聚焦一条规则，避免多规则混学。
- 任务卡优先使用项目内 stdlib-only 课程。
- 每天必须包含“运行命令”和“预期输出”。
- 验收标准必须可检查（测试通过 / demo 输出 / 生成文件 / 完成检查清单）。
- 若某课程未产出 `docs/zh.md` 或 `docs/en.md`，跳过并提示用户选择替代课。
- 输出保持中文，命令保持英文。

## Outputs

默认输出顺序：
1) 9 日计划（简版）
2) 今日任务卡（详细）
3) 验收清单
4) 参考学习笔记路径
5) 当前学习进度（来自 loops-study-progress.md）

## Example Prompts

- "按 loops 学习 skill 给我 Day 3 的任务卡"
- "整理 loops 学习复盘"
- "导出 loops-study-plan.md"
- "给我一份可执行的 9 日 LOOPS.md 计划"
- "更新 loops 进度 Day 1 完成"
- "查看 loops 学习进度"
- "深入讲解 Rule III  Negotiate the contract first"
