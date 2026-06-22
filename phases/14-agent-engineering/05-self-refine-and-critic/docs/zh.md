# Self-Refine 与 CRITIC：迭代输出改进

> Self-Refine（Madaan 等人，2023）让一个 LLM 扮演三个角色——生成、反馈、精炼——循环运行。平均在 7 个任务上提升 +20 个百分点。CRITIC（Gou 等人，2023）通过将验证步骤路由到外部工具来强化反馈环节。到 2026 年，这一模式在所有框架中以"evaluator-optimizer"（Anthropic）或 guardrail loop（OpenAI Agents SDK）的形式广泛部署。

**类型：** 实践构建
**语言：** Python（标准库）
**前置要求：** 第 14 阶段 · 01（Agent 循环）、第 14 阶段 · 03（Reflexion）
**时间：** 约 60 分钟

## 学习目标

- 阐述 Self-Refine 的三个提示词（生成、反馈、精炼），并解释为什么历史记录对精炼提示词至关重要。
- 说明 CRITIC 的关键洞察：LLM 在没有外部依据的情况下，自我验证是不可靠的。
- 用标准库实现一个带历史记录和可选外部验证器的 Self-Refine 循环。
- 将此模式映射到 Anthropic 的"evaluator-optimizer"工作流和 OpenAI Agents SDK 的输出 guardrails。

## 问题背景

一个 agent 产出了一个几乎正确的答案。可能是一行代码有语法错误，可能是一个摘要太长，可能是一个计划遗漏了边界情况。你需要的是：agent 对自己的输出进行批评，然后修复它。

Self-Refine 证明了仅靠一个模型、无需训练数据、无需 RL 就能做到这一点。但有一个陷阱：LLM 在硬事实上的自我验证能力很差。CRITIC 给出了修正方案——将验证步骤路由到外部工具（搜索、代码解释器、计算器、测试运行器）。

这两篇论文共同定义了 2026 年迭代改进的默认范式：生成 → 验证（尽可能使用外部工具）→ 精炼 → 验证通过时停止。

## 核心概念

### Self-Refine（Madaan 等人，NeurIPS 2023）

一个 LLM，三个角色：

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
stop when feedback says "no issues" or budget exhausted.
```

关键细节：`refine` 可以看到完整的历史记录——所有先前的输出和批评——因此不会重复犯错。论文对此进行了消融实验：去掉历史记录后质量急剧下降。

核心成果：在 7 个任务（数学、代码、缩写、对话）上平均提升 +20 个百分点，包括 GPT-4。无需训练，无需外部工具，单模型。

### CRITIC（Gou 等人，arXiv:2305.11738，v4 2024 年 2 月）

Self-Refine 的弱点：反馈环节是 LLM 对自己打分。对于事实性声明，这不可靠（幻觉通常在产生它的模型看来很有说服力）。CRITIC 用 `verify(task, output, tools)` 替代了 `feedback(task, output)`，其中 `tools` 包括：

- 用于事实声明的搜索引擎。
- 用于代码正确性的代码解释器。
- 用于算术的计算器。
- 领域特定的验证器（单元测试、类型检查器、lint 工具）。

验证器生成基于工具结果的结构化批评。精炼器据此进行修正。

核心成果：CRITIC 在事实性任务上优于 Self-Refine，因为批评是有依据的。在没有外部验证器的任务（创意写作、格式化）上，CRITIC 退化为 Self-Refine。

### 停止条件

两种常见形式：

1. **验证器通过。** 外部测试返回成功。在可用时优先使用（单元测试、类型检查器、guardrail 断言）。
2. **未发出反馈。** 模型说"输出没问题"。更便宜但不可靠；需配合最大迭代次数上限。

2026 年的默认做法：将两者结合。"如果验证器通过，或者模型说没问题且迭代次数 >= 2，或者迭代次数 >= max_iterations，则停止。"

### Evaluator-Optimizer（Anthropic，2024）

Anthropic 在 2024 年 12 月的博文中将其列为五种工作流模式之一。两个角色：

- 评估器（Evaluator）：对输出打分并生成批评。
- 优化器（Optimizer）：根据批评修订输出。

循环运行直到评估器通过。这就是 Anthropic 框架下的 Self-Refine/CRITIC。Anthropic 补充的一个关键工程细节是：评估器和优化器的提示词应有显著差异，以防止模型简单地自我盖章。

### OpenAI Agents SDK 输出 guardrails

OpenAI Agents SDK 将此模式以"output guardrails"的形式发布。Guardrail 是一个在 agent 最终输出上运行的验证器。如果 guardrail 触发（抛出 `OutputGuardrailTripwireTriggered`），输出将被拒绝，agent 可以重试。Guardrail 可以调用工具（CRITIC 式），也可以是纯函数（Self-Refine 式）。

### 2026 年的常见陷阱

- **橡皮图章循环。** 同一个模型用相同风格的提示词进行生成和批评，会收敛到"我觉得不错"。使用结构上不同的提示词，或用更小的廉价模型进行批评。
- **过度精炼。** 每次精炼都会增加延迟和 token 消耗。预算 1-3 次；超过后升级到人工审查。
- **在简单任务上使用 CRITIC。** 如果没有外部验证器，CRITIC 会退化为 Self-Refine；不要为一个空壳验证器付出延迟代价。

## 动手实现

`code/main.py` 在一个示例任务上实现了 Self-Refine 和 CRITIC：根据给定主题生成一个简短的要点列表。验证器检查格式（3 个要点，每个不超过 60 个字符）。CRITIC 额外添加了一个外部"事实验证器"，对已知的幻觉进行惩罚。

组件：

- `generate` — 脚本化生产者。
- `feedback` — LLM 风格的自我批评。
- `verify_external` — CRITIC 风格的有依据验证器。
- `refine` — 根据历史记录重写输出。
- 停止条件 — 验证器通过或达到最大 4 次迭代。

运行方式：

```
python3 code/main.py
```

对比 Self-Refine 与 CRITIC 的运行结果。CRITIC 捕获了一个 Self-Refine 遗漏的事实错误，因为外部验证器拥有自我批评所不具备的依据。

## 实际应用

Anthropic 的 evaluator-optimizer 就是这个模式的 Claude 友好表达。OpenAI Agents SDK 的 output guardrails 是 CRITIC 形态的（guardrail 可以调用工具）。LangGraph 提供了一个类似 Self-Refine 的 reflection 节点。Google 的 Gemini 2.5 Computer Use 添加了逐步骤安全评估器，这是 CRITIC 的一个变体：每个动作在执行前都会被验证。

## 部署

`outputs/skill-refine-loop.md` 根据任务形态、验证器可用性和迭代预算配置一个 evaluator-optimizer 循环。输出生成器、评估器/验证器和优化器的提示词，以及停止策略。

## 练习

1. 用 max_iterations=1 运行示例。CRITIC 还有帮助吗？
2. 将外部验证器替换为一个有噪声的版本（随机 30% 误报）。循环会怎样？这就是 2026 年大多数 guardrail 堆栈的现实。
3. 实现一个"不同模型上的生成器-批评者"变体：大模型生成，小模型批评。它能胜过同模型版本吗？
4. 阅读 CRITIC 第 3 节（arXiv:2305.11738 v4）。列出三类验证工具并各举一例。
5. 将 OpenAI Agents SDK 的 `output_guardrails` 映射到 CRITIC 的验证器角色。SDK 哪些地方做对了，哪些地方做错了？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Self-Refine | "能自我修复的 LLM" | 在单个模型中生成 → 反馈 → 精炼的循环，带有历史记录 |
| CRITIC | "基于工具的验证" | 用外部验证器（搜索、代码、计算器、测试）替代反馈 |
| Evaluator-Optimizer | "Anthropic 工作流模式" | 两个角色——评估器打分、优化器修订——循环直到收敛 |
| Output guardrail | "事后检查" | OpenAI Agents SDK 中在 agent 产出输出后运行的验证器 |
| Verify step | "批评阶段" | 关键决策点：有依据验证还是自我评分 |
| Refine history | "模型已经尝试过的内容" | 先前的输出 + 批评被前置到精炼提示词中；去掉后质量崩溃 |
| Rubber-stamp loop | "自我认同失败" | 相同提示词的批评返回"看起来不错"；用结构上不同的提示词修复 |
| Stop condition | "收敛检验" | 验证器通过 或 无反馈且达到迭代上限；绝不使用单一条件 |

## 延伸阅读

- [Madaan 等人，Self-Refine（arXiv:2303.17651）](https://arxiv.org/abs/2303.17651) — 经典论文
- [Gou 等人，CRITIC（arXiv:2305.11738）](https://arxiv.org/abs/2305.11738) — 基于工具的验证
- [Anthropic，Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — evaluator-optimizer 工作流模式
- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/) — CRITIC 形态的 output guardrails 验证器
