# A/B 测试 LLM 功能 —— GrowthBook、Statsig 与“感觉”问题

> 传统的 A/B 测试并非为非确定性 LLM 而设计。关键区别：评估（Evals）回答“模型能否完成这项工作？”而 A/B 测试回答“用户是否在乎？”两者都不可或缺；仅凭“感觉”上线的时代已经结束。2026 年需要测试的内容：提示词工程（措辞）、模型选择（GPT-4 vs GPT-3.5 vs 开源模型；准确率 vs 成本 vs 延迟）、生成参数（temperature、top-p）。真实案例：一个聊天机器人奖励模型变体使对话时长增加 70%、留存率提高 30%；Nextdoor AI 的邮件主题实验在奖励函数优化后使点击率（CTR）提升 1%；Khan Academy 的 Khanmigo 在延迟与数学准确率之间进行了迭代。平台对比：**Statsig**（于 2025 年 9 月被 OpenAI 以 11 亿美元收购）——提供序贯检验（Sequential Testing）、CUPED、一站式方案。**GrowthBook**——开源、数据仓库原生、支持贝叶斯 + 频率学派 + 序贯检验引擎，具备 CUPED、SRM 检查、Benjamini-Hochberg 与 Bonferroni 校正功能。您可根据对数据仓库 SQL 的偏好以及“被 OpenAI 收购”是否对您的组织构成影响来做出选择。

**类型：** 学习
**语言：** Python（标准库，玩具序贯检验模拟器）
**前置知识：** 第 17 阶段 · 13（可观测性），第 17 阶段 · 20（渐进式部署）
**时间：** 约 60 分钟

## 学习目标

- 区分评估（“模型能否完成工作”）与 A/B 测试（“用户是否在乎”）。
- 列举三个可测试的维度（提示词、模型、参数）并为每个维度选择合适的指标。
- 解释 CUPED、序贯检验（Sequential Testing）和 Benjamini-Hochberg 多重比较校正。
- 基于数据仓库 SQL 倾向和企业收购立场选择 Statsig 或 GrowthBook。

## 问题所在

您手动优化了系统提示词。感觉效果更好了。于是您上线了。转化率的变化只是噪声。您开始指责指标。或者您上线了新模型，但转化率并未改变——是模型性能下降了，还是变化幅度太小而无法被检测到？您不知道，因为您在上线时没有进行 A/B 测试。

评估（Evals）可以回答模型在某个标注数据集上能否完成任务，但无法回答用户是否偏好其输出。只有受控的线上实验才能回答这个问题，且该实验必须具备足够的统计功效、控制非确定性并进行多重比较校正。

## 核心概念

### 评估（Evals） vs A/B 测试

**评估（Evals）** —— 离线、标注数据集、评判器（评分标准、LLM 即评判器或人工）。回答：“在此固定分布下，输出是否正确 / 有帮助 / 安全？”

**A/B 测试** —— 线上、真实用户、随机分配。回答：“新变体是否移动了重要的用户级指标？”

两者缺一不可。评估在曝光前捕获回归问题；A/B 测试在曝光后确认产品影响。

### 测试什么

1. **提示词工程** —— 措辞、系统提示词结构、示例。指标：任务成功率、用户留存率、每次请求成本。
2. **模型选择** —— GPT-4 vs GPT-3.5-Turbo vs Llama-OSS。指标：准确率（任务） + 每次请求成本 + P99 延迟。多目标权衡。
3. **生成参数** —— temperature、top-p、max_tokens。指标：任务特定（输出多样性 vs 确定性）。

### CUPED —— 方差缩减

使用实验前数据的受控实验（Controlled-experiments Using Pre-Experiment Data）。在比较实验后数据前，先回归去除实验前方差。典型的方差缩减幅度为 30-70%。等效样本量相应提升。

实现：Statsig 和 GrowthBook 均已实现。

### 序贯检验（Sequential Testing）

经典 A/B 测试假设固定样本量。序贯检验（“边看边决策”）在重复检验中控制假阳性率。始终有效的序贯程序（如 mSPRT、Howard 的置信序列）允许在明确发现优胜者时提前停止。

### 多重比较校正

以 95% 置信度运行 20 个 A/B 测试，其中有一个会因偶然产生假阳性。Bonferroni 校正收紧每个检验的 α；Benjamini-Hochberg 控制错误发现率（False Discovery Rate）。GrowthBook 同时实现了这两种校正。

### SRM —— 样本比例不匹配（Sample Ratio Mismatch）

分配哈希将用户随机分配到不同变体。如果 50/50 的分配比例变成了 47/53，说明出了问题——SRM 检查会标记这一点。两个平台均已实现。

### Statsig vs GrowthBook

**Statsig**：
- 于 2025 年 9 月被 OpenAI 以 11 亿美元收购。托管型 SaaS。
- 序贯检验、CUPED、保留人群分析。
- 一站式方案：功能开关 + 实验 + 可观测性。
- 最佳适用场景：团队希望使用集成产品，且不介意 OpenAI 的所有权。

**GrowthBook**：
- 开源（MIT 许可）；数据仓库原生（直接读取 Snowflake/BigQuery/Redshift）。
- 多引擎支持：贝叶斯、频率学派、序贯检验。
- CUPED、SRM、Bonferroni、BH 校正。
- 可自托管或使用托管云服务。
- 最佳适用场景：偏好数据仓库 SQL 的团队，由数据团队控制指标层，希望使用开源软件。

### 非确定性使功效计算复杂化

相同的提示词会产生不同的输出。传统的功效计算假设观测值独立同分布（IID）。由于 LLM 的非确定性，等效样本量低于名义样本量。建议将所需样本量乘以约 1.3-1.5 倍作为安全裕度。

### 真实案例结果

- 聊天机器人奖励模型变体：对话时长增加 70%，留存率提高 30%。
- Nextdoor 邮件主题：在奖励函数优化后，点击率（CTR）提升 1%。
- Khan Academy Khanmigo：在延迟与数学准确率之间进行迭代权衡。

### 反模式：凭“感觉”上线

每位资深工程师都能说出一个因“感觉更好”而未经 A/B 测试就上线的功能。其中大多数功能在团队数月内未察觉的情况下导致了产品指标的回归。A/B 测试是强制执行正确流程的关键手段。

### 您应该记住的数字

- Statsig 被 OpenAI 收购：11 亿美元，2025 年 9 月。
- GrowthBook：开源 MIT 许可；贝叶斯 + 频率学派 + 序贯检验。
- CUPED 方差缩减：30-70%。
- LLM 非确定性 → 建议增加 30-50% 的样本量缓冲。

## 使用它

`code/main.py` 模拟了一个具有固定边界和序贯边界的序贯 A/B 测试。展示了序贯检验如何允许您提前停止。

## 交付它

本课程产出 `outputs/skill-ab-plan.md`。根据功能变更、工作负载和基线，选择平台、设置门禁和样本量。

## 练习

1. 运行 `code/main.py`。对于预期 5% 的提升和 3% 的基线转化率，达到 80% 统计功效需要多少样本量？
2. 为一个受医疗法规监管的本地部署客户选择 Statsig 或 GrowthBook。
3. 设计一个 A/B 测试，比较 GPT-4 与 GPT-3.5 在每个已解决工单成本上的表现。主要指标、护栏指标和次要指标分别是什么？
4. 您的金丝雀测试通过了，但 A/B 测试显示转化率下降了 1.2%。您是否上线？请写出升级标准。
5. 对一个实验前方差为实验后 60% 的数据集应用 CUPED。计算等效样本量的提升。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 评估（Eval） | “离线测试” | 对模型能力的标注集评估 |
| A/B 测试 | “实验” | 基于真实用户的线上随机对照比较 |
| CUPED | “方差缩减” | 使用实验前数据进行回归以减少方差 |
| 序贯检验（Sequential test） | “可偷看的检验” | 允许提前停止的始终有效程序 |
| 多重比较 | “族错误” | 运行多个检验会增加假阳性率 |
| Bonferroni | “严格校正” | 将 α 除以检验次数 |
| Benjamini-Hochberg | “BH FDR” | 错误发现率控制，更宽松 |
| SRM | “分配比例错误” | 样本比例不匹配；分配错误 |
| Statsig | “OpenAI 收购的” | 商业化一站式平台，2025 年被收购 |
| GrowthBook | “那个开源的” | MIT 许可的数据仓库原生平台 |
| mSPRT | “序贯概率比检验” | 经典序贯检验程序 |

## 延伸阅读

- [GrowthBook — 如何进行 AI 的 A/B 测试](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/)
- [Statsig — 超越提示词：数据驱动的 LLM 优化](https://www.statsig.com/blog/llm-optimization-online-experimentation)
- [Statsig vs GrowthBook 对比](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools)
- [Deng et al. — CUPED](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Howard — 置信序列](https://arxiv.org/abs/1810.08240)