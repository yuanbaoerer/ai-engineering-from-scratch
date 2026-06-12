# 前沿安全框架 — RSP、PF、FSF

> 三大实验室框架定义了 2026 年前沿能力的行业治理。Anthropic 负责任扩展政策（Responsible Scaling Policy）v3.0（2026 年 2 月）引入了分层的 AI 安全等级（ASL-1 到 ASL-5+），以生物安全等级为模型，ASL-3 于 2025 年 5 月针对 CBRN 相关模型激活。OpenAI 准备度框架（Preparedness Framework）v2（2025 年 4 月）定义了五项跟踪能力标准，并将能力报告与安全防护报告分开。DeepMind 前沿安全框架（Frontier Safety Framework）v3.0（2025 年 9 月）引入了关键能力等级（Critical Capability Levels），包括新的有害操纵 CCL。三者现在都包含竞争者调整条款（Competitor-Adjustment Clause），允许在对等实验室在没有可比安全防护的情况下发布产品时进行延期。跨实验室对齐仍然是结构性的而非术语性的："能力阈值"、"高能力阈值"和"关键能力等级"表示类似的概念。

**类型：** 学习
**语言：** 无
**前置课程：** 第 18 阶段 · 17（WMDP），第 18 阶段 · 07-09（欺骗性失败）
**时间：** ~75 分钟

## 学习目标

- 描述 Anthropic 的 ASL 分层结构以及 ASL-3 的激活条件。
- 列出 OpenAI 准备度框架 v2 的五项跟踪能力标准。
- 描述 DeepMind 的关键能力等级结构和有害操纵 CCL。
- 解释竞争者调整条款及其对竞争动态的重要性。
- 定义安全案例（Safety Case）并描述三支柱结构（监控、不可读性、无能力性）。

## 问题背景

第 7-17 课已经确立了欺骗是可能的、双重用途能力是存在的，评估也有局限性。拥有前沿能力模型的实验室需要一个内部治理结构来：
- 定义何时需要新安全防护的阈值。
- 定义扩展前所需的评估。
- 描述安全案例的形态。
- 处理竞争动态问题（如果竞争者在没有安全防护的情况下发布产品，你怎么办？）。

2025-2026 年的三个框架是当前的最新状态 — 不完美、在演进，且跨实验室的对齐程度足以使治理问题变为框架是否充分，而非框架是否存在。

## 核心概念

### Anthropic 负责任扩展政策 v3.0（2026 年 2 月）

ASL 结构：
- ASL-1：非前沿模型（被低于前沿的基线所涵盖）。
- ASL-2：当前前沿基线；部署时使用常规安全防护。
- ASL-3：灾难性滥用风险显著更高；CBRN 相关能力。2025 年 5 月激活。
- ASL-4：AI 研发-2 交叉阈值；能够自动化入门级 AI 研究的模型。
- ASL-5+：高级 AI 研发；能显著加速有效扩展的模型。

v3.0 新增内容：
- 前沿安全路线图（以删节形式公开）。
- 风险报告（季度，部分经外部审查）。
- AI 研发被细分为 AI 研发-2 和 AI 研发-4。
- 一旦跨越 AI 研发-4，需要提供肯定性安全案例，识别追求非对齐目标的模型所带来的非对齐风险。

### OpenAI 准备度框架 v2（2025 年 4 月 15 日）

五项跟踪能力标准：
- **合理（Plausible）。** 存在合理的威胁模型。
- **可测量（Measurable）。** 可进行实证评估。
- **严重（Severe）。** 危害很大。
- **全新（Net-new）。** 不是已有风险的放大。
- **即时或不可逆（Instantaneous-or-irremediable）。** 危害发生很快或无法挽回。

满足所有五项的能力会被跟踪。其他不会。

PF v2 的其他结构：
- 将能力报告（模型能做什么）与安全防护报告（存在什么控制措施）分开。
- 安全咨询小组审查。
- 领导层批准；董事会安全与安全委员会监督。
- "调整条款"：如果其他实验室在没有可比安全防护的情况下发布产品，OpenAI 可以降低要求。

### DeepMind 前沿安全框架 v3.0（2025 年 9 月）

按领域的关键能力等级（CCL）：
- 生物武器提升（Bioweapon Uplift）
- 网络提升（Cyber Uplift）
- ML 研发加速（ML R&D Acceleration）
- 有害操纵（Harmful Manipulation）（v3.0 新增）：能在高风险场景中实质性改变信念/行为的模型。

v2.0（2025 年 2 月）增加了欺骗性对齐（Deceptive Alignment）章节和 ML 研发 CCL 的更高安全等级。

### 跨实验室对齐

- Anthropic "能力阈值"（Capability Thresholds）。
- DeepMind "关键能力等级"（Critical Capability Levels）。
- OpenAI "高能力阈值"（High Capability Thresholds）。

没有行业标准术语。结构性对齐：三层前沿能力加已发布的评估标准。三者都包含竞争者调整条款（2025 年起）。

### 安全案例

安全案例是在最坏情况假设下部署可接受安全性的书面论证。标准结构针对三个支柱：

- **监控（Monitoring）。** 如果不良行为发生，我们能否检测到？
- **不可读性（Illegibility）。** 模型是否缺乏执行造成伤害的连贯计划的能力？
- **无能力性（Incapability）。** 模型是否缺乏造成相关伤害的能力？

不同的安全案例针对不同的支柱。对于 ASL-3 CBRN 案例，无能力性（通过遗忘）是主要目标。对于欺骗性对齐，监控和不可读性是目标。对于网络提升，三者都相关。

### 竞争动态问题

竞争者调整条款存在争议。批评者认为这会导致逐底竞争：如果三个实验室都会在竞争者违约时降低要求，均衡就会向违约方向移动。辩护者认为替代方案（单方面安全防护）在违约实验室安全意识较弱的情况下会产生更差的结果。

英国 AISI、美国 CAISI 和欧盟 AI 办公室（第 24 课）是外部治理对应方。实验室框架是自愿性的；监管框架正在形成。

### 在第 18 阶段中的位置

第 17-18 课是在欺骗和红队分析之上的测量与治理层。第 19-24 课涵盖福利、偏见、隐私、水印和监管结构。第 28 课描绘了将评估付诸实践的研究生态系统（MATS、Redwood、Apollo、METR）。

## 使用方法

本课无代码。阅读三个主要来源：RSP v3.0、PF v2、FSF v3.0。将每个实验室的分层结构映射到其他框架，并找出每个实验室定义而其他框架未定义的一个阈值。

## 产出

本课产出 `outputs/skill-framework-diff.md`。给定安全框架或发布说明，它将框架的阈值定义、所需评估和安全案例结构与 RSP v3.0、PF v2、FSF v3.0 进行对比，并标记跨实验室差异。

## 练习

1. 阅读 RSP v3.0、PF v2 和 FSF v3.0。编制一个表格，列出每个实验室的 CBRN 阈值、AI 研发阈值和部署前所需评估。

2. 竞争者调整条款存在于三个框架中（2025 年起）。写一段论证支持它；写一段论证反对它。指出每个立场依赖的假设。

3. 为一个跨越 Anthropic AI 研发-4 阈值的模型设计安全案例。列出三个支柱（监控、不可读性、无能力性）各自需要的证据。

4. DeepMind 的 FSF v3.0 引入了有害操纵 CCL。提出三个实证测量指标，用于表明模型已跨越此阈值。

5. 阅读 METR 的 "Common Elements of Frontier AI Safety Policies"（2025）。列出三个最强的跨实验室趋同点和两个最大的分歧点。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| RSP | "Anthropic 的框架" | 负责任扩展政策；ASL 分层；v3.0 2026 年 2 月 |
| PF | "OpenAI 的框架" | 准备度框架；五项标准；v2 2025 年 4 月 |
| FSF | "DeepMind 的框架" | 前沿安全框架；CCL；v3.0 2025 年 9 月 |
| ASL-3 | "生物安全等级 3-类似" | Anthropic 的 CBRN 相关能力等级；2025 年 5 月激活 |
| CCL | "关键能力等级" | DeepMind 的阈值概念；按领域划分 |
| 安全案例 | "正式论证" | 在最坏情况下部署可接受安全性的书面论证 |
| 调整条款 | "竞争者违约许可" | 框架条款，在竞争者没有可比安全防护时降低要求 |

## 扩展阅读

- [Anthropic — Responsible Scaling Policy v3.0（2026 年 2 月）](https://www.anthropic.com/responsible-scaling-policy) — ASL 分层、路线图、AI 研发细分
- [OpenAI — Updating the Preparedness Framework（2025 年 4 月 15 日）](https://openai.com/index/updating-our-preparedness-framework/) — 五项标准、调整条款
- [DeepMind — Strengthening our Frontier Safety Framework（2025 年 9 月）](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — CCL v3.0、有害操纵
- [METR — Common Elements of Frontier AI Safety Policies（2025）](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — 跨实验室比较
