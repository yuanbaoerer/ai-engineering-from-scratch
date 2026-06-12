# 自主编码代理全景（2026）

> SWE-bench Verified 在不到三年内从 4% 提升到 80.9%。同一个 Claude Sonnet 4.5 在 SWE-agent v1 上得 43.2%，在 Cline autonomous 上得 59.8%——模型周围的脚手架现在与模型本身同等重要。OpenHands（前 OpenDevin）是最活跃的 MIT 许可平台，其 CodeAct 循环在沙箱中直接执行 Python 动作，而非 JSON 工具调用。头条数字掩盖了一个方法论问题：500 个 SWE-bench Verified 任务中有 161 个只需要 1-2 行更改，SWE-bench Pro（10+ 行任务）在同一前沿模型上为 23-59%。

**类型：** 学习
**语言：** Python（标准库，CodeAct vs JSON 工具调用对比）
**前置条件：** 第 14 阶段 · 07（工具使用），第 15 阶段 · 01（长时域代理）
**时间：** ~45 分钟

## 问题

"哪个编码代理最好"是错误的问题。正确的问题是：在匹配我工作的任务分布上，使用我将在生产中运行的脚手架，我能获得什么端到端可靠性？

2022 到 2026 年间，该领域认识到脚手架——检索层、规划器、沙箱、编辑验证循环、反馈格式——是承载负载的。Claude Sonnet 4.5 在 SWE-agent v1 上在 SWE-bench Verified 上得 43.2%；同一个模型在 Cline 的自主脚手架内得 59.8%。16.6 个百分点的差异，相同权重。基础模型是组件；循环是产品。

伴随问题是基准饱和掩盖了回归。SWE-bench Verified 接近饱和，简单任务尾部（500 个任务中 161 个需要 ≤2 行）拉高了顶级分数。真实质量更适合在 SWE-bench Pro（10+ 行更改）等分布上测量，同一领先者仍处于 23-59%。

## 概念

### SWE-bench，一段话

SWE-bench（Jimenez 等人）使用带有真实补丁的真实 GitHub issue，要求代理产生使测试套件通过的补丁。SWE-bench Verified（OpenAI，2024）是人工策划的 500 任务子集，移除了模糊和损坏的任务。SWE-bench Pro 是更难的后继——需要 10+ 行更改的任务，当前前沿代理处于 23-59%。

### 2022 → 2026 曲线实际显示了什么

- **2022**：研究模型在原始 SWE-bench 上约 4%。
- **2024**：GPT-4 + Devin 风格脚手架约 14%；SWE-agent 约 12%。
- **2025**：Claude 3.5/3.7 Sonnet 在 Aider 和 SWE-agent 中推入 40-55% 范围。
- **2026**：Claude Sonnet 4.5 和前沿竞争者在 SWE-bench Verified 上达 70-80%+。Epoch AI 的排行榜实时跟踪。

斜率来自三个复合来源：更好的基础模型、更好的脚手架（CodeAct、反思、验证器循环）和更好的基准（Verified 消除噪声）。

### CodeAct vs JSON 工具调用

OpenHands（All-Hands-AI，arXiv:2407.16741，前 OpenDevin）做了一个特定的架构押注：模型不发出宿主解码执行的 JSON 工具调用，而是发出 Python 代码，Jupyter 风格内核在沙箱中运行。代理可以在一个动作中循环文件、链接工具和捕获自己的异常。

权衡：

- **JSON 工具调用**：每个动作是一轮；易于审计；有限的组合性；默认安全因为每次调用都经过显式验证器。
- **CodeAct**：一个动作可以是整个程序；可组合；需要强化的沙箱（OpenHands 使用 Docker 隔离）；故障模式包括沙箱运行时允许的任何内容。

两种架构都在生产中。CodeAct 在开源平台（OpenHands、smolagents）中占主导。JSON 工具调用在托管服务（Anthropic Managed Agents、OpenAI Assistants）中仍占主导，提供商控制执行器。

### 2026 年全景中的脚手架

| 脚手架 | 许可 | 执行模型 | 显著属性 |
|---|---|---|---|
| OpenHands（OpenDevin） | MIT | Docker 中的 CodeAct | 最活跃的开放平台；事件流可重放 |
| SWE-agent | MIT | Agent-Computer Interface (ACI) | 首个端到端 SWE-bench 脚手架 |
| Aider | Apache-2 | 本地仓库中的 diff 编辑 | 最小脚手架，强回归稳定性 |
| Cline | Apache-2 | 带工具策略的 VS Code 代理 | Sonnet 4.5 上最高分开放脚手架 |
| Devin（Cognition） | 专有 | 托管 VM + 规划器 | 首个"AI 软件工程师"产品类别 |
| Claude Code | 专有 | 权限模式 + 例程 | 第 10 课详细涵盖代理循环 |

### 为什么脚手架占主导

编码运行是长时域轨迹（第 1 课）。可靠性在步骤间复合。脚手架在三个地方购买分数：

1. **检索**：找到正确的文件读取是静默瓶颈。SWE-agent 的 ACI、OpenHands 的文件索引和 Aider 的 repo-map 都攻击这一点。
2. **验证器循环**：运行测试、读取堆栈跟踪并重试在 SWE-bench 上是 10+ 分的差异。
3. **故障遏制**：出错时回滚的沙箱防止复合损害。同一个模型有和没有验证器循环看起来像两个不同的产品。

### 基准饱和和真实分布

OpenHands 作者和 Epoch AI 都指出 SWE-bench Verified 有一个简单尾部：500 个任务中 161 个只需要 1-2 行更改。高分部分由此尾部驱动。SWE-bench Pro 限制为 10+ 行更改，即使前沿系统也在 23-59% 范围。你的生产分布几乎肯定更接近 Pro 而非 Verified。

选择代理的含义：运行你自己的 bug 积压的类 Pro 子集。重要的分数是你交付任务的代表性分数。

## 使用它

`code/main.py` 在固定迷你任务分布上比较两个玩具代理脚手架：

1. **JSON 工具调用**脚手架，每轮一个动作。
2. **CodeAct**脚手架，每个动作可以发出一个小的 Python 片段。

两者都使用存根"模型"（确定性规则），因此比较将脚手架与模型质量隔离。输出显示 CodeAct 脚手架以更少轮次解决更多任务，代价是更大的每动作爆炸半径。

## 交付它

`outputs/skill-scaffold-audit.md` 帮助你在采用前审计拟议的编码代理脚手架：检索质量、验证器存在、沙箱隔离和基准到分布的匹配。

## 练习

1. 运行 `code/main.py`。每个脚手架在同一任务集上需要多少轮？每个的每动作爆炸半径是多少？

2. 阅读 OpenHands 论文（arXiv:2407.16741）。论文认为 CodeAct 在复杂任务上优于 JSON 工具调用。识别论文承认的一个失败模式，并写一句话说明该模式在生产中何时会占主导。

3. 从你的 bug 积压中选择一个需要跨两个文件 10+ 行更改的任务。估计前沿模型在（a）JSON 工具调用和（b）CodeAct 下的端到端成功概率。证明差距的合理性。

4. SWE-bench Verified 有 161 个单文件 1-2 行任务。构造一个排除它们的分数。排行榜如何洗牌？

5. 阅读"Introducing SWE-bench Verified"（OpenAI）。解释用于移除模糊任务的具体方法论，并命名策划会遗漏的一个类别。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| SWE-bench | "编码基准" | 带有真实补丁和测试套件的真实 GitHub issue |
| SWE-bench Verified | "清理子集" | 500 个人工策划任务，有简单尾部 |
| SWE-bench Pro | "更难子集" | 10+ 行更改；前沿处于 23-59% |
| CodeAct | "代码即动作" | 代理发出 Python；Jupyter 风格内核在沙箱中执行 |
| JSON 工具调用 | "函数调用" | 每个动作是执行前验证的结构化 JSON 负载 |
| 脚手架 | "代理框架" | 基础模型周围的检索 + 规划器 + 执行器 + 验证器循环 |
| ACI（Agent-Computer Interface） | "SWE-agent 的格式" | 为 LLM 人体工程学设计的命令集，而非人类 shell |
| 验证器循环 | "测试并重试" | 运行测试、读取输出、修改补丁；最大的非模型可靠性收益 |

## 延伸阅读

- [Jimenez 等人 — SWE-bench](https://www.swebench.com/) — 原始基准和方法论。
- [OpenAI — Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — 策划子集如何构建。
- [Wang 等人 — OpenHands: An Open Platform for AI Software Developers](https://arxiv.org/abs/2407.16741) — CodeAct 架构和事件流设计。
- [Epoch AI — SWE-bench 排行榜](https://epoch.ai/benchmarks) — 实时跟踪分数。
- [Anthropic — 测量代理自主性](https://www.anthropic.com/research/measuring-agent-autonomy) — 长时域编码代理可靠性框架。
