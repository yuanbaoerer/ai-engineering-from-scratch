# Harness 工程学习计划（4 周）

## 总目标
- 搭建并评估 Agent Harness 与 Safety Harness（含状态机、工具体系、验证门链、沙箱、eval、otel、安全门）。
- 输出标准化 trace/report，形成可复用模板。

## Week1｜概念基线 + Harness 骨架
- Day1: `phases/14-agent-engineering/31-agent-workbench-why-models-fail`
- Day2: `phases/14-agent-engineering/41-workbench-for-real-repos`
- Day3: `phases/19-capstone-projects/20-agent-harness-loop-contract`
- Day4: `phases/19-capstone-projects/21-tool-registry-schema-validation`
- Day5: `phases/19-capstone-projects/22-jsonrpc-stdio-transport`
- Day6: `phases/19-capstone-projects/23-function-call-dispatcher`
- Day7: 周复盘（串联 loop/registry/transport/dispatcher）

## Week2｜执行控制 + 安全层
- Day8: `phases/19-capstone-projects/24-plan-execute-control-flow`
- Day9: `phases/19-capstone-projects/25-verification-gates-observation-budget`
- Day10: `phases/19-capstone-projects/26-sandbox-runner-denylist`
- Day11: `phases/19-capstone-projects/27-eval-harness-fixture-tasks`
- Day12: `phases/19-capstone-projects/28-observability-otel-traces`
- Day13: `phases/19-capstone-projects/29-end-to-end-coding-task-demo`
- Day14: 周复盘（最小可交付 agent harness）

## Week3｜Safety Harness 全链路
- Day15: `phases/19-capstone-projects/82-jailbreak-taxonomy`
- Day16: `phases/19-capstone-projects/83-prompt-injection-detector`
- Day17: `phases/19-capstone-projects/84-refusal-evaluation`
- Day18: `phases/19-capstone-projects/85-content-classifier-integration`
- Day19: `phases/19-capstone-projects/86-constitutional-rules-engine`
- Day20: `phases/19-capstone-projects/87-end-to-end-safety-gate`
- Day21: 周复盘（安全门模板化）

## Week4｜扩展评估 + 模板化
- Day22: `phases/14-agent-engineering/19-benchmarks-swebench-gaia`
- Day23: `phases/14-agent-engineering/20-benchmarks-webarena-osworld`
- Day24: `phases/19-capstone-projects/49-lm-eval-harness`
- Day25（可选）: `phases/19-capstone-projects/01-terminal-native-coding-agent`
- Day26（可选）: `phases/13-tools-and-protocols/14-model-context-protocol`
- Day27（可选）: `phases/14-agent-engineering/37-runtime-feedback-loops`
- Day28: 总复盘（模板输出与扩展路径）

## 验收摘要
- Week1: 能跑通最小 harness 骨架（loop + registry + transport + dispatcher）
- Week2: 能跑通 gate/sandbox/eval/otel 端到端链路
- Week3: 能跑通 pre/during/post 三层安全门并输出 trace
- Week4: 能用同一模板适配代码修复与模型评估任务
