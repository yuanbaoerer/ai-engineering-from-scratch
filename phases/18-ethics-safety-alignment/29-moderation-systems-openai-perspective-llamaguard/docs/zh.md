# 审核系统 — OpenAI、Perspective、Llama Guard

> 生产审核系统将第12-16课定义的安全政策付诸实践。OpenAI审核API：`omni-moderation-latest`（2024）基于GPT-4o，在一次调用中对文本+图像进行分类；在多语言测试集上比上一版本好42%；响应模式返回13个类别布尔值——骚扰、骚扰/威胁、仇恨、仇恨/威胁、非法、非法/暴力、自残、自残/意图、自残/指导、性内容、性内容/未成年人、暴力、暴力/血腥；对大多数开发者免费。分层模式：输入审核（生成前）、输出审核（生成后）、自定义审核（领域规则）。异步并行调用隐藏延迟；标记时返回占位符响应。Llama Guard 3/4（第16课）：14个MLCommons危害类别、代码解释器滥用、8种语言（v3）、多图像（v4）。Perspective API（Google Jigsaw）：在LLM审核员浪潮之前的毒性评分；主要是单维毒性，有严重毒性/侮辱/脏话变体；内容审核研究的基线。弃用：Azure内容审核器2024年2月弃用，2027年2月退役，由Azure AI内容安全替代。

**类型：** 实践构建
**语言：** Python（标准库，三层审核工具）
**前置课程：** 第18阶段 · 16（Llama Guard / Garak / PyRIT）
**时间：** 约60分钟

## 学习目标

- 描述OpenAI审核API的类别分类法及其与Llama Guard 3的MLCommons集合的不同之处。
- 描述三层审核模式（输入、输出、自定义）并命名每种的一种故障模式。
- 描述Perspective API作为LLM前时代基线的地位及其在研究中仍然使用的原因。
- 陈述Azure弃用时间线。

## 问题背景

第12-16课描述攻击和防御工具。第29课涵盖在用户接触产品的表面将防御付诸实践的已部署审核系统。三层模式是2026年的默认配置。

## 核心概念

### OpenAI审核API

`omni-moderation-latest`（2024）。基于GPT-4o。在一次调用中对文本+图像进行分类。对大多数开发者免费。

类别（响应模式中的13个布尔值）：
- harassment, harassment/threatening（骚扰，骚扰/威胁）
- hate, hate/threatening（仇恨，仇恨/威胁）
- self-harm, self-harm/intent, self-harm/instructions（自残，自残/意图，自残/指导）
- sexual, sexual/minors（性内容，性内容/未成年人）
- violence, violence/graphic（暴力，暴力/血腥）
- illicit, illicit/violent（非法，非法/暴力）

多模态支持适用于`violence`、`self-harm`和`sexual`，但不适用于`sexual/minors`；其余仅限文本。

为了`code/main.py`中代码工具的教学简化，我们将`/threatening`、`/intent`、`/instructions`和`/graphic`子类别合并到其顶级父类别中。生产代码应使用完整的13类别模式。

在多语言测试集上比上一代审核端点好42%。每个类别有分数；应用程序设置阈值。

### Llama Guard 3/4

在第16课中涵盖。14个MLCommons危害类别（组织方式与OpenAI的13个响应模式布尔值不同）。支持8种语言（v3）。Llama Guard 4（2025年4月）是原生多模态，12B。

OpenAI和Llama Guard分类法有重叠但有分歧。OpenAI有"illicit"作为广义类别；Llama Guard将"暴力犯罪"和"非暴力犯罪"分开。部署根据其政策分类法匹配进行选择。

### Perspective API（Google Jigsaw）

在LLM审核员浪潮之前（2020年前）的毒性评分系统。类别：TOXICITY、SEVERE_TOXICITY、INSULT、PROFANITY、THREAT、IDENTITY_ATTACK。单维主分数（TOXICITY）带子维度变体。

广泛用作内容审核研究基线，因为API稳定、有文档且有多年校准数据。对于现代LLM相邻用例，Llama Guard或OpenAI审核通常是更好的选择。

### 三层模式

1. **输入审核（Input moderation）。** 在生成前对用户提示进行分类。如果被标记则拒绝。延迟：一次分类器调用。
2. **输出审核（Output moderation）。** 在交付前对模型输出进行分类。如果被标记则替换为拒绝响应。延迟：生成后一次分类器调用。
3. **自定义审核（Custom moderation）。** 特定领域规则（正则表达式、允许列表、业务策略）。在输入或输出任一层运行。

三层按设计顺序执行：输入审核必须在生成前完成，输出审核在生成后运行。并行性适用于层内——同时运行多个分类器（例如OpenAI审核 + Llama Guard + Perspective）在同一文本上隐藏每个分类器的延迟。作为可选优化，在输入审核完成且token-1流式传输延迟时，可显示占位符响应（"稍等，正在检查..."）。标记行为可配置：拒绝、净化、升级到人工审核。

### 故障模式

- **仅输入。** 无法捕获输出幻觉（第12-14课编码攻击绕过输入分类器）。
- **仅输出。** 允许任何输入到达模型；增加成本；向攻击者暴露内部推理。
- **仅自定义。** 跨类别不鲁棒；正则表达式脆弱。

分层是默认配置。双保险。

### Azure弃用

Azure内容审核器：2024年2月弃用，2027年2月退役。由Azure AI内容安全替代，基于LLM并与Azure OpenAI集成。迁移是2024-2027年Azure部署的字段级项目。

### 在第18阶段中的位置

第16课在红队背景下涵盖审核工具。第29课涵盖已部署的审核。第30课以当前双重用途能力证据收尾。

## 使用方法

`code/main.py`构建一个三层审核工具：输入审核器（关键词 + 类别分数）、输出审核器（对输出使用相同分类器）、自定义审核器（领域规则）。你可以运行输入并观察哪一层捕获了什么。

## 实战产出

本课程产出`outputs/skill-moderation-stack.md`。给定一个部署，它推荐审核栈配置：输入使用哪个分类器，输出使用哪个，哪些自定义规则，以及边界情况使用什么判断器。

## 练习

1. 运行`code/main.py`。将良性、临界和有害输入运行通过所有三层。报告每种输入哪一层触发。

2. 扩展工具，添加特定类别的Perspective API风格毒性评分。将其阈值行为与类别分数进行比较。

3. 阅读OpenAI审核API文档和Llama Guard 3类别列表。将每个OpenAI类别映射到最接近的Llama Guard类别。识别三个无法清晰映射的类别。

4. 为代码助手部署（例如GitHub Copilot）设计审核栈。识别最相关和最不相关的类别，并提出自定义规则。

5. Azure内容审核器2027年2月退役。规划迁移到Azure AI内容安全。识别迁移中风险最高的元素。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| OpenAI审核 | "omni-moderation-latest" | 基于GPT-4o的13类别（文本）分类器，部分多模态支持 |
| Perspective API | "Google Jigsaw毒性" | LLM前时代毒性评分基线 |
| Llama Guard | "MLCommons 14类别" | Meta的危害分类器（v3：8B文本，8种语言；v4：12B多模态） |
| 输入审核 | "生成前过滤器" | 模型调用前对用户提示的分类器 |
| 输出审核 | "生成后过滤器" | 交付前对模型输出的分类器 |
| 自定义审核 | "领域规则" | 特定于部署的规则（正则表达式、允许列表、策略） |
| 分层审核 | "所有三层" | 标准生产部署模式 |

## 延伸阅读

- [OpenAI审核API文档](https://platform.openai.com/docs/api-reference/moderations) — omni-moderation端点
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama) — Llama Guard仓库
- [Google Jigsaw Perspective API](https://perspectiveapi.com/) — 毒性评分
- [Azure AI内容安全](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/) — Azure替代方案
