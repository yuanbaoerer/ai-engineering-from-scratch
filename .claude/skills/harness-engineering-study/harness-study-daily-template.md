# 每日任务卡模板（Harness 工程学习）

## 基本信息
- 日期：Day N
- 课程：`phases/.../<lesson-slug>`
- 时长：~90 分钟
- 目标：一句话说明今天要达成的结果

## 任务步骤
1. 阅读 `docs/en.md`，理解关键概念与接口
2. 运行 demo：`python3 code/main.py`
3. 运行测试：`python3 -m pytest code/tests/ -v`
4. 记录关键输出（截图/文本）与 1 个待改进点
5. 写 3 句“今日 harness 关键收获”

## 预期输出
- 控制台输出包含 XXX
- 生成 `outputs/XXX.json` 或 `traces.jsonl`
- 测试通过（无 FAIL）

## 验收标准
- 能解释 YYY 的作用
- 能说明 ZZZ 的约束/策略
- 产出文件可被后续课程复用

## 产出物清单
- `demo_output.txt`
- `test_results.txt`
- `outputs/XXX.json`
- 复盘笔记（1 段话）

## 常见问题
- 命令无输出：确认在 `code/` 目录下运行
- 测试报错：检查 Python 版本与路径
- 文件缺失：确认 `outputs/` 目录存在
