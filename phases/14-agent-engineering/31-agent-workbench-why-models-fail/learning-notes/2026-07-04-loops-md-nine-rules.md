# LOOPS.md 学习文档：让模型驱动的 Agent 运行数日

> 来源：Andrej Karpathy, *LOOPS.md: Field Notes on Agents That Run for Days*  
> 整理日期：2026-07-04  
> 关联课程：[Phase 14 · 31 Agent Workbench: Why Models Fail](phases/14-agent-engineering/31-agent-workbench-why-models-fail/)

## 核心论点

Agent 系统大多死于弱 harness，而非弱模型。模型能写代码、能审代码、能按评分标准自证；但它无法自行决定：

- 何时停止
- 何时重启
- 结果写到哪里
- 如何跨会话恢复
- 如何给主观质量打分

这些决策属于 **harness**（工作台 / 套索）的职责。LOOPS.md 把长期运行 agent 的可靠性工程提炼为 9 条规则，核心基线是：

| 基线 | 含义 |
|------|------|
| **短循环** | gather → reason → act → verify → repeat |
| **简单状态** | 能用三个文件描述清楚当前状态 |
| **干净契约** | 生成代码前，规划者与评估者就完成标准达成一致 |

---

## 1. Write the loop, not the prompt（写循环，不是写提示词）

### 含义
停止在凌晨三点迭代单条消息。真正产生杠杆的是循环：收集、推理、行动、验证、重复。Prompt 只是循环的输入之一。

### 为什么重要
前沿模型已经足够好，可以无人监督地执行流程。此时比拼的不是 prompt 技巧，而是流程设计：停止条件、轮次预算、错误恢复、工具调度。

### 项目对应课程
- [Phase 14 · 01 Agent Loop](phases/14-agent-engineering/01-the-agent-loop/)：stdlib 实现 ReAct 循环
- [Phase 19 · 20 Agent Harness Loop Contract](phases/19-capstone-projects/20-agent-harness-loop-contract/)：6 状态状态机 + 10 钩子 + 11 事件类型

### 实践要点
- 每个 agent 循环必须有：消息缓冲区、工具注册表、停止条件、轮次预算、观察格式化器
- 用 `max_turns` / `max_tool_calls` / `max_wall_time` 防止无限循环
- 循环终止路径要明确：`finish` 工具、无工具调用、预算耗尽、安全护栏触发

---

## 2. Separate the roles（分离角色）

### 含义
三个角色、三个上下文窗口、三个系统提示：

| 角色 | 职责 | 禁止做的事 |
|------|------|-----------|
| **Planner** | 把模糊需求转成 sprint spec | 碰代码 |
| **Generator** | 写所有实现 | 给自己打分 |
| **Evaluator** | 读 diff、运行测试、证明代码是坏的 | 写代码 |

### 为什么重要
混合角色会让模型变成谄媚者。一旦模型自己评判自己，循环就会悄悄收敛到平庸（slop）。

### 项目对应课程
- [Phase 14 · 39 Reviewer Agent](phases/14-agent-engineering/39-reviewer-agent/)：builder vs reviewer 分离
- [Phase 14 · 38 Verification Gates](phases/14-agent-engineering/38-verification-gates/)：确定性验证 vs 定性审查的分工

### 实践要点
- 同一模型可以担任不同角色，纪律在于输入和姿态
- Reviewer 对 builder 产物只读，不能编辑 diff
- Verification gate 检查事实；reviewer 检查判断

---

## 3. Negotiate the contract first（先协商契约）

### 含义
生成器写第一行代码前，先 propose「完成长什么样」；评估者 push back。双方通过磁盘上的 markdown 文件争论，直到就一份可测试断言清单达成一致。

### 为什么重要
这是从「broken demos」到「working products」的单一最大改变。原始 spec 是边界，contract 才是被评分的东西。

### 项目对应课程
- [Phase 14 · 36 Scope Contracts](phases/14-agent-engineering/36-scope-contracts/)：范围契约 JSON Schema
- [Phase 14 · 32 Minimal Workbench](phases/14-agent-engineering/32-minimal-agent-workbench/)：AGENTS.md 作为路由器
- [Phase 14 · 33 Instructions as Executable Constraints](phases/14-agent-engineering/33-instructions-as-executable-constraints/)

### 实践要点
- 契约必须包含：`allowed_files`、`forbidden_files`、`acceptance_criteria`、`rollback_plan`、`approvals_required`
- 负空间是契约的一半：没有 `forbidden_files` 就不完整
- 用 glob 而非原始路径，让重构不会使契约失效

---

## 4. Write to disk, not to context（写到磁盘，不要写到上下文）

### 含义
上下文窗口会压缩、腐烂、隐藏一小时前你说的话。磁盘上的文件不会撒谎。保持 `feature_list.json`、`progress.md`、`contract.md` 和 append-only 的 `log.md`。

### 为什么重要
如果无法用三个文件描述状态，状态就太复杂。模型应该能崩溃、丢失会话、然后通过读这三个文件恢复。

### 项目对应课程
- [Phase 14 · 34 Repo Memory and State](phases/14-agent-engineering/34-repo-memory-and-state/)：schema 优先的状态管理 + 原子写入
- [Phase 14 · 32 Minimal Workbench](phases/14-agent-engineering/32-minimal-agent-workbench/)：`agent_state.json` + `task_board.json`
- [Phase 14 · 37 Runtime Feedback Loops](phases/14-agent-engineering/37-runtime-feedback-loops/)：`feedback_record.jsonl`

### 实践要点
- 状态文件是权威记录；聊天历史是易失的
- 写入必须是原子的：temp file → fsync → rename
- 大工件不要存进状态文件，状态中只保留路径

---

## 5. Let the loop restart（让循环重启）

### 含义
当前沿模型愿意在运行跑偏时扔掉一切重新开始，那是最好的行为。不要打断它。重启是循环工作正常的表现。只在契约本身错误时插入人类，而不是在构建失败时。

### 为什么重要
老模型会把代码库 patch 成考古现场；新模型在干净评估者和磁盘契约下，会在第 9 次迭代删除项目、第 11 次迭代交付可用版本。

### 项目对应课程
- [Phase 14 · 40 Multi-session Handoff](phases/14-agent-engineering/40-multi-session-handoff/)：干净结束会话、下一个会话恢复
- [Phase 19 · 24 Plan-Execute Control Flow](phases/19-capstone-projects/24-plan-execute-control-flow/)：失败时 replan 而非终止
- [Phase 14 · 37 Runtime Feedback Loops](phases/14-agent-engineering/37-runtime-feedback-loops/)：用真实输出让循环对事实反应

### 实践要点
- 失败不是异常，而是 replan 的输入
- 硬预算 + 硬 replan 上限防止无限重试
- 会话在 50–75% 上下文预算时交接，不要等到 95%

---

## 6. Score the subjective（给主观事物评分）

### 含义
Taste 是可被量化的。写下一个 rubric，包含 design、originality、craft、functionality 四个加权维度。用三个参考点校准：两个好例子、三个坏例子。输出 0–1 分数 + 一段解释差距的文字。

### 为什么重要
模型不会发明 taste，它只会收敛到你描述的 taste。整个游戏就是把 rubric 写得足够仔细，让收敛方向正是你想要的。

### 项目对应课程
- [Phase 14 · 39 Reviewer Agent](phases/14-agent-engineering/39-reviewer-agent/)：5 维度 0–2 分评分标准
- [Phase 14 · 30 Eval-Driven Agent Development](phases/14-agent-engineering/30-eval-driven-agent-development/)：评估器-优化器循环
- [Phase 19 · 27 Eval Harness Fixture Tasks](phases/19-capstone-projects/27-eval-harness-fixture-tasks/)：pass@k、延迟、成本聚合

### 实践要点
- 评分标准必须维度明确、分数可解释
- 用校准集验证评分标准：与历史判定一致性低于 80% 就修订
- 缓解 LLM-as-judge 偏见：顺序对调、匿名作者、奖励简洁

---

## 7. Read the traces（读取轨迹）

### 含义
把 agent 输出写入文件，grep 判断偏离你意图的时刻，精确编辑那个时刻的 prompt，再跑一遍。这和读 stack trace 是同一肌肉；只是 trace 用英语写成，而且大部分是模型在自言自语。

### 为什么重要
跳过这一步，你就是在靠 vibe 调参。每个关于 agent loop 的调试洞察都来自阅读原始 transcript。

### 项目对应课程
- [Phase 14 · 37 Runtime Feedback Loops](phases/14-agent-engineering/37-runtime-feedback-loops/)：结构化反馈记录
- [Phase 19 · 28 Observability with OTel](phases/19-capstone-projects/28-observability-otel-traces/)：OTel GenAI Span + Prometheus metrics
- [Phase 14 · 23 OTel GenAI Conventions](phases/14-agent-engineering/23-otel-genai-conventions/)

### 实践要点
- 反馈日志和遥测日志分开：反馈用于下一轮回读，遥测用于操作员审计
- 每条记录包含：command、stdout_tail、stderr_tail、exit_code、duration_ms、started_at
- 对 stdout/stderr 在写入时脱敏，而不是读取时

---

## 8. Delete the harness（删除 harness）

### 含义
harness 的存在是为了补偿模型。随着模型变强，上一季度写的一半 harness 会变成 overhead。每发布新一代模型，就重读 harness，删除模型现在能免费做的事情。

### 为什么重要
单调增长的 harness 是你已经停止阅读的 harness。上下文重置、sprint 分解、四小时构建约束——这些曾对旧模型是承重结构，对新模型可能是累赘。

### 项目对应课程
- [Phase 14 · 31 Why Models Still Fail](phases/14-agent-engineering/31-agent-workbench-why-models-fail/)：讨论 harness 与模型的关系
- [Phase 14 · 41 Workbench for Real Repos](phases/14-agent-engineering/41-workbench-for-real-repos/)：识别哪些界面仍有杠杆、哪些已成 overhead

### 实践要点
- 把流行 harness 模式转译为分布式系统原语（函数、工作进程、触发器、运行时、队列、持久化、策略）
- 定期审计：这条规则/脚本现在模型能否自觉遵守？
- 不要为还没出现的未来需求写 harness

---

## 9. The bottleneck always moves（瓶颈总在移动）

### 含义
coding 不再是瓶颈时，planning 成为瓶颈；planning 解决后，verification 成为瓶颈；verification 自动化后，taste 成为瓶颈。你不应该「完成」，而应该找到下一个要修复的东西。

### 为什么重要
循环的全部意义就是让下一个瓶颈可见。如果一切顺利，说明你看得不够仔细。

### 项目对应课程
- [Phase 14 · 30 Eval-Driven Agent Development](phases/14-agent-engineering/30-eval-driven-agent-development/)：三层评估体系驱动下一次优化
- [Phase 14 · 41 Workbench for Real Repos](phases/14-agent-engineering/41-workbench-for-real-repos/)：5 个结果指标对比
- [Phase 19 · 27 Eval Harness Fixture Tasks](phases/19-capstone-projects/27-eval-harness-fixture-tasks/)：fixture 任务暴露回归

### 实践要点
- 找到新瓶颈 → 修复它 → 交付更小的 harness → 重复
- 每个护栏和规则都要映射到一个评估用例
- 评估与代码并置，在 CI 中运行，作为合并门控

---

## 项目学习路径

LOOPS.md 的完整工程映射在项目中最清晰的学习线是：

```text
Phase 14 · 31  Why Models Still Fail
      ↓
Phase 14 · 32  Minimal Agent Workbench
      ↓
Phase 14 · 34  Repo Memory and State
      ↓
Phase 14 · 36  Scope Contracts
      ↓
Phase 14 · 37  Runtime Feedback Loops
      ↓
Phase 14 · 38  Verification Gates
      ↓
Phase 14 · 39  Reviewer Agent
      ↓
Phase 14 · 40  Multi-session Handoff
      ↓
Phase 14 · 41  Workbench for Real Repos
```

若希望更偏底层实现，可继续进入 Phase 19 的 Agent Harness 毕业项目：

```text
Phase 19 · 20  Agent Harness Loop Contract
Phase 19 · 21  Tool Registry Schema Validation
Phase 19 · 22  JSON-RPC STDIO Transport
Phase 19 · 23  Function Call Dispatcher
Phase 19 · 24  Plan-Execute Control Flow
Phase 19 · 25  Verification Gates + Observation Budget
Phase 19 · 26  Sandbox Runner Denylist
Phase 19 · 27  Eval Harness Fixture Tasks
Phase 19 · 28  Observability OTel Traces
Phase 19 · 29  End-to-End Coding Task Demo
```

---

## 实践检查清单

- [ ] 能用三个文件描述 agent 当前状态
- [ ] 每个任务开始前已有范围契约，且包含 `forbidden_files`
- [ ] builder 与 reviewer 使用不同系统提示，reviewer 对 diff 只读
- [ ] 验证门禁是确定性的，不依赖 LLM 判断
- [ ] 所有 shell 命令通过反馈运行器捕获 stdout/stderr/exit/duration
- [ ] 状态写入是原子的（temp + fsync + rename）
- [ ] 会话结束时自动生成 `handoff.md` + `handoff.json`
- [ ] 每次模型升级后审计并删除已成 overhead 的 harness
- [ ] 每个护栏和规则都对应一个 CI 中的评估用例

---

## 关键术语表

| 术语 | 含义 |
|------|------|
| **Harness** | 包裹模型的运行时工程：指令、状态、范围、反馈、验证、审查、交接 |
| **Loop** | gather → reason → act → verify → repeat 的循环 |
| **Contract** | 任务开始前就完成标准达成的可测试断言清单 |
| **Trace** | 一次运行的完整思考、行动、观察记录 |
| **Rubric** | 主观质量评分的多维加权标准 |
| **Handoff** | 跨会话传递工作状态的制品包 |
| **Bottleneck** | 当前限制循环产出的环节，会随时间移动 |
