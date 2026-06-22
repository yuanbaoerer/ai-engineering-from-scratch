# 思维树与 LATS：深思熟虑的搜索

> 单条思维链轨迹没有回溯的余地。ToT（Yao 等人，2023）将推理转化为一棵树，每个节点都进行自我评估。LATS（Zhou 等人，2024）在蒙特卡洛树搜索（MCTS）框架下统一了 ToT、ReAct 和 Reflexion。Game of 24 任务从 4%（CoT）提升到 74%（ToT）；LATS 在 HumanEval 上达到 92.7% pass@1。

**类型：** Build
**语言：** Python（stdlib）
**前置课程：** 第 14 阶段 · 01（Agent Loop）、第 14 阶段 · 03（Reflexion）
**时间：** 约 75 分钟

## 学习目标

- 将推理建模为搜索：节点是"思维"，边是"展开"，价值是"看起来有多有希望"。
- 实现一个基于 stdlib 的 ToT 风格 BFS 树搜索，带自我评估评分。
- 扩展为一个玩具级 LATS MCTS 循环，包含选择/展开/模拟/回溯。
- 判断何时搜索值得付出 token 增量（Game of 24、代码生成），以及何时单条轨迹就够了（简单问答）。

## 问题

思维链是一条线性行走。如果第一步就错了，后续每一步都建立在错误的前提之上。在 Game of 24 任务中（用四个数字通过 + − × ÷ 得到 24），GPT-4 CoT 的准确率仅为 4%。模型很早就选错了子表达式，无法恢复。

推理需要的能力是：提出多个候选方案，评估它们，选择有希望的分支，在遇到死胡同时回溯。这就是搜索。思维树（ToT）和 LATS 是两种经典的表述方式。

## 核心概念

### 思维树（Yao 等人，NeurIPS 2023）

每个节点是一个连贯的中间步骤（"一个思维"）。每个节点可以展开为 K 个子思维。LLM 通过评分提示对每个节点进行自我评估。搜索过程遍历这棵树——BFS、DFS 或 beam search。

```
                     (root: "find 24 from 4 6 4 1")
                    /               |            \
           ("6 - 4 = 2")    ("4 + 1 = 5")    ("4 * 6 = 24")  <- Score: HIGH
              /   \              |                  |
          ...    ...          ...                finish
```

自我评估是核心支撑。论文展示了三种变体：`sure / likely / impossible` 分类、`1..10` 数值评分、以及候选方案投票。三种方法在 Game of 24 上都显著超过了 CoT（从 4% 提升到 74%，使用 GPT-4）。

### LATS（Zhou 等人，ICML 2024）

LATS 在 MCTS 框架下统一了 ToT、ReAct 和 Reflexion。LLM 扮演三个角色：

- **策略（Policy）**：提出候选的下一步行动（ReAct 风格）。
- **价值函数（Value function）**：对部分轨迹评分（ToT 风格自我评估）。
- **自我反思器（Self-reflector）**：在失败时，撰写自然语言反思（Reflexion 风格），并用它来为后续的 rollout 重新设定起点。

环境反馈（观测）融入价值函数，使搜索基于真实工具结果而非仅仅是模型的判断。论文发表时的结果：HumanEval pass@1 92.7%（GPT-4，SOTA），WebShop 平均 75.9（GPT-3.5，接近基于梯度的微调）。

### MCTS 简述

每次迭代包含四个阶段：

1. **选择（Select）** — 使用 UCT（上置信界树搜索）从根节点走到叶子节点。
2. **展开（Expand）** — 通过策略生成 K 个子节点。
3. **模拟（Simulate）** — 从子节点开始使用策略进行 rollout，用价值函数（或环境奖励）对叶子节点评分。
4. **回溯（Backpropagate）** — 沿路径向上更新访问次数和价值估计。

UCT 公式：`Q(s, a) + c * sqrt(ln N(s) / N(s, a))`。第一项是利用，第二项是探索。需按任务调整 `c`。

### 成本现实

搜索会急剧增加 token 消耗。ToT 在 Game of 24 上的 token 使用量是 CoT 的 100–1000 倍。LATS 也类似。这不是免费的；应将搜索保留用于：

- 单条轨迹明显不够的任务（Game of 24、复杂代码）。
- 正确性比运行时间更重要的任务。
- 拥有廉价可靠价值函数的任务（代码的单元测试、数学的显式目标）。

如果任务只有一个正确答案且评估器存在噪声，搜索通常会让结果更差——它会找到一个"高分"的错误答案。

### 2026 年定位

大多数生产级 agent 并不运行 LATS。它们运行的是带有工具验证的 ReAct（CRITIC，第 05 课）。搜索主要出现在特定场景中：

- 以测试作为价值函数的编程 agent（HumanEval 风格）。
- 探索多条查询路径的深度研究 agent。
- LangGraph 子图中的重度规划工作流。

AlphaEvolve（第 11 课）是 2025 年的极致案例：对代码进行进化搜索，使用机器可检查的适应度函数，取得了前沿突破（56 年来首次改进 4×4 矩阵乘法）。

## 动手实现

`code/main.py` 实现了：

- 一个在风格化"选择算术运算符"任务上的小型 ToT BFS。
- 同一任务上的玩具 LATS MCTS 循环（选择/展开/模拟/回溯），使用 UCT 选择。
- 一个组合了符号评分和自我评估评分的价值函数。

运行：

```
python3 code/main.py
```

输出会展示 ToT 使用 BFS 在每个节点展开三个候选方案，以及 LATS 通过 MCTS 收敛到最佳 rollout 的过程。两者都打印了 token 消耗量。

## 在实践中使用

LangGraph 将 ToT 风格的探索作为子图模式提供；LangChain 团队关于 LATS 的博客（2024 年 5 月）是参考教程。LlamaIndex 提供了 `TreeOfThoughts` agent。对于大多数 2026 年的生产级 agent，这种模式通过 `if task_complexity > threshold: use_search()` 门控——参见第 05 课的评估器-优化器模式。

## 交付

`outputs/skill-search-policy.md` 根据任务形态、预算和评估器保真度，在线性 ReAct、ToT、LATS 和进化搜索之间做出选择。

## 练习

1. 运行玩具 LATS，分别使用 UCT c=0.1 和 c=2.0。输出轨迹有什么变化？
2. 将价值函数替换为一个噪声更大的评分器（加入随机抖动）。MCTS 还能找到最佳叶子节点吗？它能容忍的最低信噪比是多少？
3. 实现 beam-search ToT（每层保留 top-k），并与 BFS 比较。在严格的 token 预算下哪个更好？
4. 阅读 LATS 第 5.1 节。复现 HumanEval 的轨迹数量：需要多少次 rollout 才能达到报告的 pass@1？
5. 阅读 LATS 论文中关于"什么时候 LATS 帮助较小"的讨论。写一段决策规则，将任务形态映射到搜索策略。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 思维树（Tree of Thoughts） | "分支 CoT" | Yao 等人——带自我评估的思维节点树 |
| LATS | "LLM 的 MCTS" | Zhou 等人——在 MCTS 下统一 ToT + ReAct + Reflexion |
| UCT | "上置信界" | 平衡利用（Q）与探索（ln N / n）的选择公式 |
| 价值函数（Value function） | "这个状态有多好" | 通过提示获得的 LLM 评分或环境奖励；反馈给回溯 |
| 策略（Policy） | "行动提议者" | ReAct 风格的生成器；产出候选的下一步思维/行动 |
| Rollout | "模拟轨迹" | 从一个节点使用策略走到叶子节点，用价值函数评分 |
| 回溯（Backpropagate） | "更新祖先节点" | 将叶子节点的奖励沿路径向上推送，更新访问次数和 Q |
| 搜索成本（Search cost） | "token 爆炸" | Game of 24 上是 CoT 的 100-1000 倍；采用前先规划预算 |

## 延伸阅读

- [Yao 等人，Tree of Thoughts（arXiv:2305.10601）](https://arxiv.org/abs/2305.10601) — 经典论文
- [Zhou 等人，LATS（arXiv:2310.04406）](https://arxiv.org/abs/2310.04406) — 带 Reflexion 反馈的 MCTS
- [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview) — 搜索的子图模式
- [AlphaEvolve（arXiv:2506.13131）](https://arxiv.org/abs/2506.13131) — 带程序化评估器的进化搜索
