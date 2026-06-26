---
name: harness-engineering-study
version: 1.0.0
description: Harness 工程学习路线与每日任务执行器。触发词："harness 学习"、"harness 路线"、"harness 计划"、"搭建 harness"、"agent harness"、"safety harness"、"/harness-study"。
---

# Harness Engineering Study

为本项目的 harness 工程内容，提供可执行的 4 周学习路线、每日任务卡、验收清单与产出模板。

## Activation

当用户输入以下内容时激活本 skill：
- `/harness-study`
- “给我 harness 学习路线”
- “整理 harness 计划”
- “我要搭 harness 工程”
- “按 agent harness / safety harness 路线学习”
- “生成今天的 harness 任务卡”
- “更新 harness 进度 Day N 完成”
- “查看 harness 学习进度”

## Input

可选参数：
- `week=1|2|3|4`：指定学习周次
- `day=1..28`：指定具体天
- `mode=plan|daily|review|export`：计划 / 每日任务 / 周复盘 / 导出

若无参数，默认输出：4 周总览 + 当天任务卡。

## Procedure

### Step 1: 读取课程分布
从项目中定位 harness 相关课程，至少覆盖：
- `phases/14-agent-engineering/31-agent-workbench-why-models-fail`
- `phases/14-agent-engineering/41-workbench-for-real-repos`
- `phases/19-capstone-projects/20-agent-harness-loop-contract`
- `phases/19-capstone-projects/21-tool-registry-schema-validation`
- `phases/19-capstone-projects/22-jsonrpc-stdio-transport`
- `phases/19-capstone-projects/23-function-call-dispatcher`
- `phases/19-capstone-projects/24-plan-execute-control-flow`
- `phases/19-capstone-projects/25-verification-gates-observation-budget`
- `phases/19-capstone-projects/26-sandbox-runner-denylist`
- `phases/19-capstone-projects/27-eval-harness-fixture-tasks`
- `phases/19-capstone-projects/28-observability-otel-traces`
- `phases/19-capstone-projects/29-end-to-end-coding-task-demo`
- `phases/19-capstone-projects/82-jailbreak-taxonomy`
- `phases/19-capstone-projects/83-prompt-injection-detector`
- `phases/19-capstone-projects/84-refusal-evaluation`
- `phases/19-capstone-projects/85-content-classifier-integration`
- `phases/19-capstone-projects/86-constitutional-rules-engine`
- `phases/19-capstone-projects/87-end-to-end-safety-gate`

可选扩展：
- `phases/14-agent-engineering/19-benchmarks-swebench-gaia`
- `phases/14-agent-engineering/20-benchmarks-webarena-osworld`
- `phases/19-capstone-projects/49-lm-eval-harness`

### Step 2: 生成/读取周计划
按 4 周结构输出：
- Week1：概念基线 + Harness 骨架
- Week2：执行控制 + 安全层
- Week3：Safety Harness 全链路
- Week4：扩展评估 + 模板化

每周包含 6 天学习 + 1 天复盘。

### Step 3: 生成每日任务卡
每日任务卡固定字段：
- 课程路径
- 时长（默认 ~90min）
- 目标
- 步骤
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
每周日输出周复盘模板：
- 本周完成课程
- 产出清单（路径）
- 阻塞点
- 下周调整

### Step 5: 进度跟踪（按天）
读取并更新：`.claude/skills/harness-engineering-study/harness-study-progress.md`

触发词：
- “更新 harness 进度 Day N 完成”
- “查看 harness 学习进度”

更新规则：
- 将 Day N 状态改为“已完成”
- 填写完成时间与备注
- 将“当前位置”推进到下一天（若 Day N 为周日则推进到下一周）
- 若存在阻塞点，记录到“阻塞点记录”

### Step 6: 导出与复盘
支持导出：
- `harness-study-plan.md`（总计划）
- `harness-study-daily-dayN.md`（每日任务卡）
- `harness-study-review-weekN.md`（周复盘）

## Rules

- 任务卡优先使用项目内 stdlib-only 课程，避免额外依赖。
- 每天必须包含“运行命令”和“预期输出”。
- 验收标准必须可检查（测试通过 / demo 输出 / 生成文件）。
- 若某课程未产出 `docs/en.md`，跳过并提示用户选择替代课。
- 输出保持中文，命令保持英文。

## Outputs

默认输出顺序：
1) 本周计划（简版）
2) 今日任务卡（详细）
3) 验收清单
4) 产出模板路径
5) 当前学习进度（来自 harness-study-progress.md）

## Example Prompts

- “按 harness 学习 skill 给我 Week2 Day9 的任务卡”
- “整理本周 harness 学习复盘”
- “导出 harness-study-plan.md”
- “给我一份可执行的 4 周 harness 计划”
- “更新 harness 进度 Day 1 完成”
- “查看 harness 学习进度”
