# Phase 14 Agent 工程复习笔记

> 日期: 2026-06-29
> 复习范围: Lesson 01-04, 31-34（共 8 课）

## 1. 两组课程的关系

| 组别 | 课程 | 核心问题 |
|------|------|----------|
| 第一组 (01-04) | Agent Loop 基础 | Agent 怎么思考 |
| 第二组 (31-34) | Workbench 工程 | Agent 怎么在仓库里可靠工作 |

第一组是算法层（ReAct、ReWOO、Reflexion、ToT/LATS），第二组是工程层（七个表面、三文件架构、可执行约束、持久状态）。

## 2. Agent Loop 基础（Lesson 01-04）

### ReAct 五个必要组件

Message buffer、Tool registry、Stop condition、Turn budget、Observation formatter。缺任何一个就只是 chatbot。

### ReAct vs ReWOO vs ToT

| 模式 | 规划方式 | Token 复杂度 | 适用场景 |
|------|---------|-------------|---------|
| ReAct | 逐步交织（Thought→Action→Observation） | O(n²) | 短任务、未知环境 |
| ReWOO | 先完整规划，再并行执行 | O(n) | 工具已知的结构化任务 |
| ToT | 树搜索 + 自我评估 | 100-1000× CoT | 复杂数学、代码生成 |

### Reflexion 三组件

Actor（执行）→ Evaluator（评估）→ Self-Reflector（反思）。记忆腐烂最实用组合：TTL + 压缩 + 验证过滤。

### LATS = MCTS 统一 ReAct + ToT + Reflexion

关键反直觉点：**噪声评估器让搜索更差**。搜索不创造正确性，只放大评估器判断。评估器质量 = 搜索上限。

## 3. Workbench 工程（Lesson 31-34）

### 七个 Workbench 表面

Instructions → State → Scope → Feedback → Verification → Review → Handoff

核心原则：**循环闭合在 state 文件上，不在聊天记录上**。

### Harness-Compute 分离

- Compute（数据面）= 模型推理、代码执行、文件写入
- Harness（控制面）= 调度、权限、状态管理、监控
- 分离价值：独立演进，换模型不碰控制面

### 三文件架构

| 文件 | 角色 | 丢的后果 |
|------|------|---------|
| AGENTS.md | 路由器（< 50 行） | 可重建 |
| agent_state.json | session 级状态 | 可从 board 推断 |
| task_board.json | 项目级任务队列 | **项目失忆** |

信 board 不信 state。hooks 可靠性 90-95%，剩余靠 CI 兜底。

### 五类可执行规则

Startup / Forbidden / Definition of Done / Uncertainty / Approval

没有 `check` 字段的规则 = 愿望，直接删掉。规则从事故中提取，重复出现才值得加。

### 仓库记忆四个生产加固模式

| 模式 | 要点 |
|------|------|
| 原子写 | temp + fsync + os.replace，同目录同文件系统 |
| 幂等键 | key 在副作用前持久化；两种做法：确定性派生 / checkpoint 预写 |
| 大工件分离 | CSV/生成物存路径不入 state |
| 事件溯源+快照 | 读快照 + 重放事件 = Postgres WAL 同构 |

Schema-first 两层含义：时间上先写 + 强制约束力（load/commit 都过 validate）。

## 4. 复习得分

| 课程 | 得分 | 薄弱点 |
|------|------|--------|
| 01 Agent Loop | 2/3 | Thought 核心价值 |
| 02 ReWOO | 3/3 | — |
| 03 Reflexion | 3/3 | — |
| 04 ToT/LATS | 3/3 | — |
| 31 Why Models Fail | 3/3 | — |
| 32 Minimal Workbench | 3/3 | — |
| 33 Executable Constraints | 3/3 | — |
| 34 Repo Memory & State | 3/3 | — |
| **总计** | **23/24** | |

## 5. 待实操练习

- [ ] Lesson 04: 给 value() 加噪声抖动，观察 MCTS 在不同信噪比下的表现
- [ ] Lesson 34: 实现 StateManager.update（load→改内存→commit）
- [ ] Lesson 34: 实现 IdempotentToolRunner + pending_calls.jsonl
- [ ] Lesson 34: 写 v1→v2 迁移脚本（blockers→risks）+ 未知版本 fail-loud
- [ ] Lesson 34: 保持 StateManager API 不变换 SQLite 后端
- [ ] Lesson 34: 实现"读快照+重放事件"的恢复代码
- [ ] Lesson 34: 两智能体并发写同一状态文件实验
