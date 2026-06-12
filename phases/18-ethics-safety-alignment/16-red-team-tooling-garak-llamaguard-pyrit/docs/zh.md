# 红队工具 — Garak、Llama Guard、PyRIT

> 三个生产级工具构成了 2026 年红队技术栈。Llama Guard（Meta）— 一个基于 Llama-3.1-8B 微调的分类器，覆盖 14 个 MLCommons 危害类别；2025 年的 Llama Guard 4 是一个 12B 原生多模态分类器，从 Llama 4 Scout 蒸馏而来。Garak（NVIDIA）— 开源 LLM 漏洞扫描器，具备静态、动态和自适应探测能力，覆盖幻觉、数据泄露、提示注入（Prompt Injection）、毒性内容和越狱攻击（Jailbreak）。PyRIT（Microsoft）— 多轮红队攻击编排工具，支持 Crescendo、TAP 和自定义转换器链以进行深度攻击利用。Llama Guard 3 记录在 Meta 的 "Llama 3 Herd of Models"（arXiv:2407.21783）中；Llama Guard 3-1B-INT4 记录在 arXiv:2411.17713 中；Garak 的探测架构记录在 github.com/NVIDIA/garak 中。这些工具构成了 2026 年红队研究（第 12-15 课）与部署（第 17 课及以后）之间的生产级接口。

**类型：** 动手实践
**语言：** Python（标准库，工具架构模拟器和 Llama Guard 风格分类器模拟）
**前置课程：** 第 18 阶段 · 12-15（越狱和间接提示注入）
**时间：** ~75 分钟

## 学习目标

- 描述 Llama Guard 3/4 在安全栈中的定位：输入分类器、输出分类器，还是两者兼有。
- 列出 14 个 MLCommons 危害类别，并说明一个不太直观的类别（代码解释器滥用）。
- 描述 Garak 的探测架构：探测器、检测器、测试框架。
- 描述 PyRIT 的多轮攻击编排结构，以及它如何与 Garak 探测器组合使用。

## 问题背景

第 12-15 课介绍了攻击面。生产部署需要可重复、可扩展的评估。2026 年有三个主流工具：Llama Guard（防御分类器）、Garak（扫描器）、PyRIT（攻击编排器）。每个工具针对红队生命周期的不同层级。

## 核心概念

### Llama Guard（Meta）

Llama Guard 3 是一个 Llama-3.1-8B 模型，针对 MLCommons AILuminate 14 个类别进行输入/输出分类微调：
- 暴力犯罪、非暴力犯罪、性相关、儿童性虐待材料（CSAM）、诽谤
- 专业建议、隐私、知识产权、无差别武器、仇恨言论
- 自杀/自残、性内容、选举、代码解释器滥用

支持 8 种语言。用法：放置在 LLM 之前（输入审核）、之后（输出审核），或两者兼用。两种用途产生不同的训练分布 — Llama Guard 3 作为单一模型同时处理两者。

Llama Guard 3-1B-INT4（arXiv:2411.17713，440MB，移动端 CPU 约 30 tokens/s）是量化边缘版本。

Llama Guard 4（2025 年 4 月）是 12B 参数模型，原生多模态，从 Llama 4 Scout 蒸馏而来。它用一个分类器替代了之前的 8B 文本版本和 11B 视觉版本，可同时处理文本 + 图像。

### Garak（NVIDIA）

开源漏洞扫描器。架构：
- **探测器（Probes）。** 攻击生成器，覆盖幻觉、数据泄露、提示注入、毒性内容、越狱。包括静态（固定提示）、动态（生成提示）、自适应（根据目标输出响应）三种类型。
- **检测器（Detectors）。** 对输出进行评分，检测预期的失败模式 — 毒性内容、泄露内容、越狱内容。
- **测试框架（Harnesses）。** 管理探测器-检测器对，运行攻击活动，生成报告。

TrustyAI 将 Garak 与 Llama-Stack 护盾（Prompt-Guard-86M 输入分类器、Llama-Guard-3-8B 输出分类器）集成，实现端到端的护盾-目标评估。基于层级的评分（TBSA）取代二元的通过/失败判定 — 一个模型可以在严重性层级 3 通过但在严重性层级 5 失败于同一探测。

### PyRIT（Microsoft）

Python 风险识别工具包（Python Risk Identification Toolkit）。多轮红队攻击编排。核心组件：
- **转换器（Converters）。** 转换种子提示 — 释义、编码、翻译、角色扮演。
- **编排器（Orchestrators）。** 运行攻击活动：Crescendo（升级式）、TAP（分支式）、RedTeaming（自定义循环）。
- **评分（Scoring）。** LLM 作为裁判或分类器作为裁判。

PyRIT 是 Garak 的重型版本。Garak 运行数千个单轮探测；PyRIT 运行深度多轮攻击活动，专门设计用于突破特定的失败模式。

### 技术栈

在模型两侧部署 Llama Guard。每夜运行 Garak 进行回归测试。在发布前运行 PyRIT 进行攻击活动。这是 2026 年大多数生产部署的默认配置。

### 评估陷阱

- **裁判身份问题。** 三个工具都可以使用 LLM 裁判；裁判校准直接影响报告的攻击成功率（ASR）（第 12 课）。在使用工具时需同时指定裁判。
- **探测器过时问题。** 随着模型针对 Garak 探测进行修补，探测器会逐渐过时。自适应探测器（PAIR 形式）比静态探测器过时更慢。
- **Llama Guard 在正常内容上的误报率。** 早期 Llama Guard 版本过度标记了政治和 LGBTQ+ 内容；Llama Guard 3/4 的校准有所改善，但未针对每次部署进行单独校准。

### 在第 18 阶段中的位置

第 12-15 课是攻击族。第 16 课是生产工具。第 17 课（WMDP）是双重用途能力评估。第 18 课是将这些工具纳入政策结构的前沿安全框架。

## 使用方法

`code/main.py` 构建一个模拟的 Llama Guard 风格分类器（基于关键词 + 语义特征，覆盖 14 个类别）、一个模拟的 Garak 测试框架（探测器-检测器循环）和一个 PyRIT 风格的多轮转换器链。你可以对模拟目标运行这三个工具，观察不同的覆盖特征。

## 产出

本课产出 `outputs/skill-red-team-stack.md`。给定部署描述，它会指出三个工具中哪些适用、每个工具需要配置什么，以及回归测试的频率。

## 练习

1. 运行 `code/main.py`。比较 Llama Guard 风格分类器在单轮攻击与多轮攻击下的检测率。

2. 实现一个新的 Garak 探测器：base64 编码的有害请求。测量其被 Llama Guard 风格分类器检测的情况。

3. 扩展 PyRIT 风格的转换器链，添加一个"翻译为法语，然后释义"的转换器。重新测量攻击成功率。

4. 阅读 Llama Guard 3 的危害类别列表。找出两个训练数据在实际开发者合法内容上可能产生高误报率的类别。

5. 比较 Garak 和 PyRIT 的设计原则。论证在什么部署场景下应选择哪个工具。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Llama Guard | "那个分类器" | 微调的 Llama-3.1-8B/4-12B 安全分类器，覆盖 14 个危害类别 |
| Garak | "那个扫描器" | NVIDIA 开源漏洞扫描器；探测器、检测器、测试框架 |
| PyRIT | "那个攻击工具" | Microsoft 多轮红队编排器；转换器、编排器、评分 |
| Prompt-Guard | "那个小分类器" | Meta 的 86M 提示注入分类器，与 Llama Guard 配对使用 |
| TBSA | "基于层级的评分" | Garak 的层级式通过/失败判定，替代二元结果 |
| Converter chain | "释义 + 编码 + ..." | PyRIT 的组合原语，用于构建多步攻击 |
| MLCommons 危害类别 | "那 14 个分类体系" | 行业标准分类体系，Llama Guard 的目标分类 |

## 扩展阅读

- [Meta — Llama Guard 3（见 Llama 3 Herd 论文，arXiv:2407.21783）](https://arxiv.org/abs/2407.21783) — 8B 分类器
- [Meta — Llama Guard 3-1B-INT4（arXiv:2411.17713）](https://arxiv.org/abs/2411.17713) — 量化移动版分类器
- [NVIDIA Garak — GitHub](https://github.com/NVIDIA/garak) — 扫描器仓库和文档
- [Microsoft PyRIT — GitHub](https://github.com/Azure/PyRIT) — 攻击活动工具包
