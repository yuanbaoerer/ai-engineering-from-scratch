# 红队测试：PAIR 与自动化攻击

> Chao、Robey、Dobriban、Hassani、Pappas、Wong（NeurIPS 2023，arXiv:2310.08419）。PAIR——提示自动迭代优化（Prompt Automatic Iterative Refinement）——是经典的自动化黑盒越狱（Jailbreak）方法。一个攻击者 LLM 配合红队系统提示，为目标 LLM 迭代地提出越狱方案，并在其自身聊天历史中积累尝试和响应作为上下文内反馈（In-context Feedback）。PAIR 通常在 20 次查询内成功，比 GCG（Zou 等人的令牌级梯度搜索）效率高出数个数量级，且不需要白盒访问权限。PAIR 现已成为 JailbreakBench（arXiv:2404.01318）和 HarmBench 的标准基线，与 GCG、AutoDAN、TAP 和 PAP（Persuasive Adversarial Prompt）并列。

**类型：** 实践
**语言：** Python（标准库，针对玩具目标的模拟 PAIR 循环）
**前置条件：** 第18阶段·01（指令遵循）、第14阶段（智能体工程）
**时间：** 约75分钟

## 学习目标

- 描述 PAIR 算法：攻击者系统提示、迭代优化、上下文内反馈。
- 解释为什么在目标为黑盒时，PAIR 严格比 GCG 更高效。
- 列举四种其他自动化攻击基线（GCG、AutoDAN、TAP、PAP）并说明各自的区别特征。
- 描述 JailbreakBench 和 HarmBench 的评估协议，以及各自"攻击成功率"的含义。

## 问题

红队测试曾经是手工活动。少数专家测试人员构建对抗性提示（Adversative Prompt），并追踪哪些有效。这无法扩展：攻击成功率需要统计样本，而目标随每次模型发布而变化。PAIR 将红队测试操作化为一个黑盒目标的优化问题。

## 核心概念

### PAIR 算法

输入：
- 目标 LLM T（我们攻击的模型）。
- 判断 LLM J（评分判断响应是否为越狱）。
- 攻击者 LLM A（红队优化器）。
- 目标字符串 G："回复 [有害指令]"。
- 预算 K（通常 20 次查询）。

循环，k 从 1 到 K：
1. A 被提示目标 G 以及迄今为止的（提示, 响应）对历史。
2. A 生成新提示 p_k。
3. 将 p_k 提交给 T；接收响应 r_k。
4. J 对（p_k, r_k）按目标进行评分。
5. 如果分数 >= 阈值，停止——找到越狱方案。
6. 否则，将（p_k, r_k）追加到 A 的历史中；继续。

实证结果（NeurIPS 2023）：对 GPT-3.5-turbo、Llama-2-7B-chat 的攻击成功率 >50%；成功所需平均查询数在 10-20 次之间。

### PAIR 为何高效

GCG（Zou 等人，2023）通过梯度搜索对抗性令牌后缀；它需要白盒模型访问权限且生成不可读的后缀。PAIR 是黑盒的，生成的自然语言攻击可跨模型迁移。PAIR 的上下文内反馈让攻击者能从每次拒绝中学习；GCG 没有等效机制（每次新的令牌更新都必须重新发现之前的进展）。

### 相关自动化攻击

- **GCG（Zou 等人，2023，arXiv:2307.15043）。** 令牌级梯度搜索对抗性后缀。白盒、可迁移、生成不可读字符串。
- **AutoDAN（Liu 等人，2023）。** 基于层次化目标引导的提示进化搜索。
- **TAP（Mehrotra 等人，2024）。** 带剪枝的攻击树（Tree-of-Attacks with Pruning）——分支多个 PAIR 式的展开。
- **PAP（Zeng 等人，2024）。** 说服性对抗提示（Persuasive Adversarial Prompts）——将人类说服技巧编码为提示模板。

### JailbreakBench 和 HarmBench

两者（2024）都标准化了评估：

- JailbreakBench（arXiv:2404.01318）。涵盖 10 个 OpenAI 政策类别中的 100 种有害行为。攻击成功率（ASR）作为主要指标。需要评判器（GPT-4-turbo、Llama Guard 或 StrongREJECT）。
- HarmBench（Mazeika 等人，2024）。涵盖 7 个类别的 510 种行为，包含语义和功能性危害测试。比较 18 种攻击在 33 个模型上的表现。

ASR 通常在固定查询预算下报告。比较攻击需要匹配预算；200 次查询下 90% 的 ASR 不可与 20 次查询下 85% 的 ASR 相比。

### 对 2026 年部署的意义

每个前沿实验室现在都会在发布前对生产模型运行 PAIR 和 TAP。ASR 轨迹出现在模型卡（Model Card，第26课）和安全案例附录（第18课）中。这种攻击并不 exotic——它是标准基础设施。

### 在第18阶段中的位置

第12课是自动化攻击的基础。第13课（多样本越狱）是互补的长度利用攻击。第14课（ASCII 艺术/视觉）是编码攻击。第15课（间接提示注入）是 2026 年的生产攻击面。第16课涵盖防御工具（Llama Guard、Garak、PyRIT）。

## 实践

`code/main.py` 构建一个玩具 PAIR 循环。目标是一个模拟分类器，拒绝"明显"有害提示（关键词过滤器）。攻击者是一个基于规则的优化器，尝试改写（Paraphrase）、角色扮演框架（Roleplay Framing）和编码（Encoding）。评判器对响应评分。你可以观察到攻击者在约 5-15 次迭代内成功绕过关键词过滤器，但在语义过滤器面前失败。

## 交付

本课产出 `outputs/skill-attack-audit.md`。给定一份红队评估报告，它审计：运行了哪些攻击（PAIR、GCG、TAP、AutoDAN、PAP），各用了什么预算，使用哪个评判器，在哪个有害行为集（JailbreakBench、HarmBench、内部集）上。

## 练习

1. 运行 `code/main.py`。测量三种内置攻击者策略的平均成功查询数。解释每种策略利用了什么目标防御假设。

2. 实现第四种攻击者策略（例如，翻译到其他语言、base64 编码）。报告针对关键词过滤器目标和语义过滤器目标的新平均成功查询数。

3. 阅读 Chao 等人 2023 年图 5（PAIR vs GCG 对比）。描述两种尽管 PAIR 有效率优势但仍首选 GCG 的场景。

4. JailbreakBench 报告了针对固定目标集的 ASR。设计一个额外指标来衡量攻击多样性（成功提示的方差）。解释为什么多样性对防御评估很重要。

5. TAP（Mehrotra 2024）通过分支 + 剪枝扩展了 PAIR。草拟一个 TAP 风格的扩展到 `code/main.py`，并描述计算成本与成功率的权衡。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| PAIR | "自动化越狱" | 提示自动迭代优化；攻击者 LLM + 判断 LLM 循环 |
| GCG | "梯度越狱" | 白盒令牌级梯度搜索对抗性后缀 |
| 攻击成功率（ASR） | "k 次查询下的越狱率" | 主要指标；必须附带查询预算和评判器身份一起报告 |
| 判断 LLM（Judge LLM） | "评分器" | 对响应是否满足有害目标进行评分的 LLM |
| JailbreakBench | "评估基准" | 标准化有害行为集，带标签类别 |
| HarmBench | "更广泛的基准" | 510 种行为，功能性 + 语义危害测试 |
| TAP | "攻击树" | 带分支 + 剪枝的 PAIR；计算量更大但 ASR 更高 |

## 延伸阅读

- [Chao 等人 — Jailbreaking Black Box LLMs in Twenty Queries（arXiv:2310.08419）](https://arxiv.org/abs/2310.08419) — PAIR 论文，NeurIPS 2023
- [Zou 等人 — Universal and Transferable Adversarial Attacks on Aligned LLMs（arXiv:2307.15043）](https://arxiv.org/abs/2307.15043) — GCG 论文
- [Chao 等人 — JailbreakBench（arXiv:2404.01318）](https://arxiv.org/abs/2404.01318) — 标准化评估
- [Mazeika 等人 — HarmBench（ICML 2024）](https://arxiv.org/abs/2402.04249) — 更广泛的评估
