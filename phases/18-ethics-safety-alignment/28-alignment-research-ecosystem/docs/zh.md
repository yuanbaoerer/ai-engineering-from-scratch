# 对齐研究生态系统 — MATS、Redwood、Apollo、METR

> 五个组织定义了2026年非实验室对齐研究层。MATS（ML对齐与理论学者）：自2021年底以来527+研究者，180+论文，10K+引用，h指数47；2024年夏季团队注册为501(c)(3)，约90名学者和40名导师；2025年前校友中80%从事安全/安全工作，200+在Anthropic、DeepMind、OpenAI、UK AISI、RAND、Redwood、METR、Apollo。Redwood Research：Buck Shlegeris创立的应用对齐实验室；引入AI控制（第10课）；与UK AISI合作控制安全案例。Apollo Research：为前沿实验室提供部署前阴谋评估；撰写上下文阴谋（第8课）和AI安全案例。METR（模型评估与威胁研究）：基于任务的能力评估、自主任务时间范围研究；"前沿AI安全政策共同要素"比较实验室框架。Eleos AI Research：模型福祉部署前评估（第19课）；进行Claude Opus 4福祉评估。

**类型：** 知识学习
**语言：** 无
**前置课程：** 第18阶段 · 01-27（先前第18阶段课程）
**时间：** 约45分钟

## 学习目标

- 识别非实验室对齐研究生态系统的五个组织及其核心产出。
- 描述MATS的规模（学者、论文、h指数）及其作为人才管道的角色。
- 描述Redwood的AI控制议程及其与UK AISI的合作。
- 描述METR的基于任务的评估方法论。

## 问题背景

前沿实验室（第18课）在内部进行安全评估并发布选定结果。实验室外部的生态系统是评估被验证的地方，是新型故障模式首次被发现的地方，也是人才被培养的地方。理解生态系统有助于解读哪些研究发现被谁信任。

## 核心概念

### MATS（ML对齐与理论学者）

2021年底启动。研究指导项目；学者与资深研究员一起花10-12周研究特定对齐问题。

规模（2026年）：
- 自成立以来527+研究者。
- 发表180+论文。
- 10K+引用。
- h指数47。
- 2024年夏季：90名学者+40名导师；注册为501(c)(3)。

职业成果：约80%的2025年前校友从事安全/安全工作。200+在Anthropic、DeepMind、OpenAI、UK AISI、RAND、Redwood、METR、Apollo。

### Redwood Research

应用对齐实验室。Buck Shlegeris创立。引入AI控制议程（第10课）。与UK AISI合作控制安全案例。为DeepMind和Anthropic提供评估设计咨询。

经典论文：Greenblatt, Shlegeris et al.，"AI Control"（arXiv:2312.06942, ICML 2024）；对齐伪装（Alignment Faking，Greenblatt, Denison, Wright et al., arXiv:2412.14093，与Anthropic联合）。

风格：特定威胁模型、最坏情况对手、可进行压力测试的具体协议。

### Apollo Research

为前沿实验室提供部署前阴谋评估。撰写上下文阴谋（第8课，arXiv:2412.04984）。2025年OpenAI反阴谋训练协作的合作伙伴。产出AI安全案例（2024年）。

风格：欺骗可能产生的代理设置评估；三支柱分解（失调、目标导向、情境意识）。

### METR（模型评估与威胁研究）

基于任务的能力评估。自主任务完成时间范围研究。"前沿AI安全政策共同要素"（metr.org/common-elements, 2025）比较实验室框架。

与Apollo合作AI阴谋安全案例草案。

风格：长时间范围任务评估、实证能力测量、框架综合。

### Eleos AI Research

模型福祉部署前评估。进行Claude Opus 4福祉评估，记录在系统卡片第5.3节。为第19课的福祉相关声明提供外部方法论检查。

### 流程

MATS培养研究者。毕业生进入Anthropic、DeepMind、OpenAI（实验室安全团队）或Redwood、Apollo、METR、Eleos（外部评估）。外部评估者与实验室以及UK AISI/CAISI合作。出版物将生态系统反馈给MATS，为下一届团队做准备。

### 为什么这一层重要

单一来源评估不可靠：实验室评估自己的模型存在结构性利益冲突。外部评估者可以提出并验证实验室可能低报的故障模式。2024年Sleeper Agents论文（第7课）是Anthropic + Redwood；对齐伪装是Anthropic + Redwood；上下文阴谋是Apollo；反阴谋是Apollo + OpenAI。多组织结构是质量控制。

### 在第18阶段中的位置

第7-11课引用Redwood和Apollo的工作；第18课引用METR的框架比较；第19课引用Eleos。第28课是该阶段其余部分所依赖的生态系统的明确组织映射。

## 使用方法

无代码。阅读METR的"前沿AI安全政策共同要素"作为外部综合如何为实验室内部政策工作增添价值的示例。

## 实战产出

本课程产出`outputs/skill-ecosystem-map.md`。给定一个对齐声明或评估，它识别组织、出版渠道和方法论风格，并与已知对应组织交叉核对。

## 练习

1. 从第7-15课中选择一篇论文并识别涉及的组织。将作者与MATS校友和当前生态系统隶属关系交叉核对。

2. 阅读METR的"前沿AI安全政策共同要素"。识别他们强调的三个跨实验室趋同点和两个最大分歧点。

3. MATS职业成果约80%从事安全/安全工作。论证这种选择压力是适应性的（培养该领域）还是有偏见的（过滤掉异端立场）。

4. Redwood和Apollo都做控制/阴谋工作但风格不同。选择一个故障模式并描述每个组织如何调查它。

5. Eleos AI是唯一纯粹的模型福祉组织。设计一个假设的第二个组织，专注于不同的福祉相关问题（认知自由、机器人具身等）并阐明其方法论。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| MATS | "指导项目" | ML对齐与理论学者；自2021年以来527+研究者 |
| Redwood Research | "控制实验室" | 应用对齐；AI控制论文作者；UK AISI合作伙伴 |
| Apollo Research | "阴谋评估" | 为前沿实验室提供部署前阴谋评估 |
| METR | "任务范围评估" | 基于任务的能力评估；框架综合 |
| Eleos AI | "福祉实验室" | 模型福祉部署前评估 |
| 人才管道 | "MATS -> 实验室" | MATS毕业生流向Anthropic、DM、OpenAI、Redwood、Apollo、METR |
| 外部评估 | "非实验室检查" | 非模型生产者进行的评估；增加可信度 |

## 延伸阅读

- [MATS（ML对齐与理论学者）](https://www.matsprogram.org/) — 指导项目
- [Redwood Research](https://www.redwoodresearch.org/) — AI控制论文
- [Apollo Research](https://www.apolloresearch.ai/) — 阴谋评估
- [METR — 前沿AI安全政策共同要素](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — 框架比较
- [Eleos AI Research](https://www.eleosai.org/research) — 模型福祉方法论
