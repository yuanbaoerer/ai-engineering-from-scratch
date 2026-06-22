# ReWOO 与 Plan-and-Execute：解耦规划

> ReAct 在单一流程中交织思考与行动。ReWOO 将二者分离：先制定一个完整计划，再执行。Token 消耗减少 5 倍，在 HotpotQA 上准确率提升 4%，并且你可以将规划器蒸馏为 7B 模型。Plan-and-Execute 将其通用化；Plan-and-Act 将其扩展到 Web 导航场景。

**类型：** 实战
**语言：** Python（标准库）
**前置课程：** 第 14 阶段 · 01（Agent 循环）
**时间：** ~60 分钟

## 学习目标

- 解释 ReWOO 的 Planner / Worker / Solver 分离为何比 ReAct 的交织循环更节省 Token 且更健壮。
- 实现一个计划 DAG、依赖排序执行器以及组合 Worker 输出的求解器——全部使用标准库。
- 判断何时应使用"先规划后执行"模式，何时使用 ReAct 交织模式，参考 2026 年"五种工作流模式"框架（Anthropic）。
- 了解 Plan-and-Act 的合成计划数据何时对长周期 Web 或移动端任务有必要。

## 问题背景

ReAct 的交织式思考-行动-观察循环简单而灵活，但每次工具调用都必须携带完整的先前上下文——包括每一个历史思考步骤。Token 消耗随深度二次方增长。更糟糕的是：当工具在循环中途失败时，模型必须从错误观察中重新推导整个计划。

ReWOO（Xu 等人，arXiv:2305.18323，2023 年 5 月）发现了这一点并提出了一个思路：提前规划整个流程，并行获取证据，最后组合答案。一次 LLM 调用进行规划，N 次工具调用获取证据（可并行），一次 LLM 调用求解。代价是灵活性降低（计划是静态的），但换来了更好的 Token 效率和更清晰的失败模式。

## 核心概念

### 三个角色

```
Planner:  用户问题 -> [计划 DAG]
Workers:  [计划 DAG] -> [证据]          （工具调用，可并行）
Solver:   用户问题, 计划 DAG, 证据 -> 最终答案
```

Planner 生成一个 DAG（有向无环图）。每个节点命名一个工具、其参数以及它依赖的早期节点（如 `#E1`、`#E2` 等引用）。Workers 按拓扑顺序执行节点。Solver 将所有内容整合在一起。

### 为何能减少 5 倍 Token

ReAct 的提示长度随步骤数线性增长。到第 10 步时，提示包含思考 1 + 行动 1 + 观察 1 + 思考 2 + 行动 2 + 观察 2，以此类推。每个中间步骤还冗余地包含原始提示。

ReWOO 只需一次大的 Planner 提示、N 次小的 Worker 提示（每次仅包含工具调用，不含链式上下文）和一次 Solver 提示。在 HotpotQA 上，论文测量到约 5 倍的 Token 减少，同时准确率绝对值提升 4%。

### 为何更健壮

在 ReAct 中，如果 Worker 3 失败，循环必须在中途从错误中推理恢复。在 ReWOO 中，Worker 3 返回一个错误字符串；Solver 在原始计划的上下文中看到它，可以优雅地降级处理。失败定位是按节点的，而非按步骤的。

### 规划器蒸馏

论文的第二个成果：由于 Planner 不会看到观察结果，你可以用 175B 教师模型的 Planner 输出来微调一个 7B 模型。小模型负责规划；推理时不需要大模型。这现在已成为标准做法——许多 2026 年的生产 Agent 使用小 Planner 配大 Executor，或反之。

### Plan-and-Execute（LangChain，2023）

LangChain 团队在 2023 年 8 月的博文将 ReWOO 泛化为一个模式名称：Plan-and-Execute。前端 Planner 发出步骤列表，Executor 执行每个步骤，可选的 Replanner 可以在观察结果后进行修订。这比 ReWOO 更接近 ReAct（Replanner 将观察结果带回规划阶段），但保留了 Token 节省的优势。

### Plan-and-Act（Erdogan 等人，arXiv:2503.09572，ICML 2025）

Plan-and-Act 将该模式扩展到长周期的 Web 和移动端 Agent。其关键贡献是合成计划数据：一个带标签的轨迹生成器生成计划显式的训练数据。用于微调 Planner 模型，使其在 WebArena 类任务上能持续工作超过 30-50 步，而单一 ReAct 轨迹在此时会失去连贯性。

### 何时选择哪种模式

| 模式 | 适用场景 |
|------|---------|
| ReAct | 短任务、未知环境、需要反应式异常处理 |
| ReWOO | 工具已知的结构化任务、Token 敏感、可并行获取证据 |
| Plan-and-Execute | 类似 ReWOO 但需要在部分执行后重新规划 |
| Plan-and-Act | 长周期（>30 步）、Web/移动端/计算机操作 |
| 思维树 | 搜索值得付出代价的场景（第 04 课） |

Anthropic 2024 年 12 月的指导建议：从最简单的开始。如果任务是一次工具调用加一个摘要，不要构建 ReWOO。如果任务是 40 步的研究任务，不要只用 ReAct。

## 动手实现

`code/main.py` 实现了一个玩具版 ReWOO：

- `Planner` — 一个基于脚本的策略，从提示中发出计划 DAG。
- `Worker` — 通过注册表分派每个节点的工具调用。
- `Solver` — 基于脚本的组合，读取证据并生成最终答案。
- 依赖解析 — `#E1` 等引用被替换为先前 Worker 的输出。

该演示通过两步计划回答"法国首都的人口是多少，四舍五入到百万？"：(1) 查找首都，(2) 查找人口，然后求解。

运行方式：

```
python3 code/main.py
```

跟踪信息先显示完整计划，然后是 Worker 结果，最后是 Solver 组合。将 Token 计数（我们打印一个粗略的字符计数）与 ReAct 风格的交织运行进行比较——ReWOO 在这类结构化任务上胜出。

## 实际应用

LangGraph 将 Plan-and-Execute 作为预设方案提供（`create_react_agent` 用于 ReAct，自定义图用于 plan-execute）。CrewAI 的 Flows 直接编码了该模式：你提前定义任务，Flow DAG 执行它们。Plan-and-Act 的合成数据方法目前主要还是研究阶段；运行时模式（显式计划 DAG）已通过 LangGraph 和 CrewAI Flows 在生产环境中部署。

## 交付物

`outputs/skill-rewoo-planner.md` 根据用户请求和工具目录生成 ReWOO 计划 DAG。它在交给执行器之前会验证计划（无环、所有引用已解析、每个工具都存在）。

## 练习

1. 为独立的计划节点并行化 Worker 执行。在一个有 6 个节点和 2 个并行组的 DAG 上，这能带来什么好处？
2. 添加一个 Replanner 节点，当任何 Worker 返回错误时触发。对 ReWOO 的最小改动是什么才能使其变成 Plan-and-Execute？
3. 用小模型（7B 级别）替换 `Planner`，`Solver` 保留前沿模型。比较端到端质量——这种分离在哪里会失败？
4. 阅读 ReWOO 论文第 4 节关于规划器蒸馏的内容。在概念上复现 175B -> 7B 的结果：你需要什么训练数据，如何评估计划质量？
5. 将玩具版移植到 Plan-and-Act 的轨迹形式：计划是序列而非 DAG。权衡取舍有何变化？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| ReWOO | "无观察推理" | 先规划，再并行获取证据，最后求解——规划提示中不包含观察结果 |
| Plan-and-Execute | "LangChain 的 plan-execute 模式" | ReWOO 加上执行后的可选 Replanner 节点 |
| Plan-and-Act | "扩展版 plan-execute" | 显式 Planner/Executor 分离，带有用于长周期任务的合成计划训练数据 |
| 证据引用 | "#E1, #E2, ..." | 计划节点占位符，在分派时被先前 Worker 输出替换 |
| 规划器蒸馏 | "小规划器，大执行器" | 用大教师模型的 Planner 轨迹微调小模型 |
| Token 效率 | "更少的往返" | 论文中相比 ReAct 在 HotpotQA 上减少 5 倍 Token |
| DAG 执行器 | "拓扑分派器" | 按依赖顺序运行计划节点；每层可并行 |

## 延伸阅读

- [Xu 等人，ReWOO：将推理与观察解耦（arXiv:2305.18323）](https://arxiv.org/abs/2305.18323) — 权威论文
- [Erdogan 等人，Plan-and-Act（arXiv:2503.09572）](https://arxiv.org/abs/2503.09572) — 带合成计划的扩展 Planner-Executor
- [LangGraph Plan-and-Execute 教程](https://docs.langchain.com/oss/python/langgraph/overview) — 框架预设方案
- [Anthropic，构建有效的 Agent](https://www.anthropic.com/research/building-effective-agents) — 选择最简单的可行模式
