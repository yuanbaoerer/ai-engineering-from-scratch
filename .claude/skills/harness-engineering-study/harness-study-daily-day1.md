# Day 1 任务卡：概念基线（prompt-only vs workbench）

## 基本信息
- 课程：`phases/14-agent-engineering/31-agent-workbench-why-models-fail`
- 时长：~45-60 分钟
- 目标：建立 harness/workbench 七层概念，产出对比基线报告

## 任务步骤
1. 阅读 `docs/en.md`，理解七层表面及其缺失影响
2. 运行 demo，生成 prompt-only vs workbench 对比
3. 读取输出文件 `failure_modes.json`
4. 记录“最容易失败的 2 个表面”

## 运行命令
```bash
python3 code/main.py
```

## 预期输出
- 控制台打印两侧运行日志
- 生成 `failure_modes.json`
- 输出包含七层表面的判断结果

## 验收标准
- 能说明“为什么 prompt-only 常在 Verification 与 Handoff 失败”
- 报告内容可作为 Week1 复盘材料

## 产出物
- 对比日志（控制台）
- `failure_modes.json`
- 你的结论（1 段话）
