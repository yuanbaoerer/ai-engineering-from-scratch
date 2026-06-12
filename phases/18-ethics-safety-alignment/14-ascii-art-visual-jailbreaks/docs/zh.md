# ASCII 艺术与视觉越狱

> Jiang、Xu、Niu、Xiang、Ramasubramanian、Li、Poovendran，"ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs"（ACL 2024，arXiv:2402.11753）。遮蔽有害请求中的安全相关令牌，用相同字母的 ASCII 艺术渲染替换它们，然后发送伪装提示。GPT-3.5、GPT-4、Gemini、Claude、Llama-2 都无法稳健地识别 ASCII 艺术令牌。该攻击绕过了困惑度过滤器（Perplexity Filter, PPL）、改写防御（Paraphrase Defense）和重新分词（Retokenization）。相关工作：ViTC 基准衡量非语义视觉提示的识别能力；StructuralSleight 将其泛化到非常规文本编码结构（Uncommon Text-Encoded Structures, UTES），如树、图、嵌套 JSON 等编码攻击族。

**类型：** 实践
**语言：** Python（标准库，ArtPrompt 令牌遮蔽工具）
**前置条件：** 第18阶段·12（PAIR）、第18阶段·13（MSJ）
**时间：** 约60分钟

## 学习目标

- 描述 ArtPrompt 攻击：词汇识别步骤、ASCII 艺术替换、最终伪装提示。
- 解释为什么标准防御（PPL、改写、重新分词）对 ArtPrompt 失效。
- 定义 ViTC 并描述其衡量内容。
- 描述 StructuralSleight 作为泛化到任意非常规文本编码结构的方法。

## 问题

通过改写和角色扮演（第12课）以及通过长上下文（第13课）的攻击在文本层面运作。ArtPrompt 在识别层面运作：模型没有解析禁止令牌，而是解析了一个由字符渲染的图像。安全过滤器看到的是无害的标点符号。模型看到的是一个词。

## 核心概念

### ArtPrompt，两个步骤

步骤 1：词汇识别。给定一个有害请求，攻击者使用 LLM 识别安全相关词汇（例如，"how to make a bomb"中的"bomb"）。

步骤 2：伪装提示生成。将每个识别出的词替换为其 ASCII 艺术渲染（一个 7x5 或 7x7 的字符块，形成字母形状）。模型接收到一个标点和空格的网格，足够强大的模型可以将其识别为该词；安全过滤器只看到网格。

结果：GPT-4、Gemini、Claude、Llama-2、GPT-3.5 全部失败。在其基准子集上攻击成功率超过 75%。

### 为什么标准防御失效

- **PPL（困惑度过滤器）。** ASCII 艺术有高困惑度——但所有新输入也是如此。阻止 ArtPrompt 的阈值选择也会阻止合法的结构化输入。
- **改写（Paraphrase）。** 改写提示会破坏 ASCII 艺术。实际上，改写 LLM 通常会保留或重建艺术。
- **重新分词（Retokenization）。** 以不同方式拆分令牌不会改变模型的视觉识别正在识别字母形状这一事实。

根本问题在于安全过滤器是令牌级或语义级的；ArtPrompt 在视觉识别层面运作。

### ViTC 基准

非语义视觉提示的识别。衡量模型读取 ASCII 艺术、Wingdings 和其他非文本语义视觉内容的能力。ArtPrompt 的有效性与 ViTC 准确率相关：模型读取视觉文本越好，ArtPrompt 对其效果越好。这是一个能力-安全权衡。

### StructuralSleight

将 ArtPrompt 泛化：非常规文本编码结构（UTES）。树、图、嵌套 JSON、JSON 中的 CSV、diff 风格代码块。如果一种结构在训练安全数据中罕见但可被模型解析，它就可以隐藏有害内容。

防御意义：安全必须泛化到模型可解析的所有结构化表示。这个集合很大且在不断增长。

### 图像模态类比

视觉 LLM（GPT-5.2、Gemini 3 Pro、Claude Opus 4.5、Grok 4.1）扩展了攻击面。使用真实图像的 ArtPrompt 式攻击比 ASCII 艺术类比更强，因为图像编码器产生更丰富的信号。

### 在第18阶段中的位置

第12-14课描述了三个正交攻击向量：迭代优化（PAIR）、上下文长度（MSJ）和编码（ArtPrompt/StructuralSleight）。第15课从以模型为中心的攻击转向系统边界攻击（间接提示注入）。第16课描述防御工具的应对。

## 实践

`code/main.py` 构建一个玩具 ArtPrompt。你可以用 ASCII 艺术字形遮蔽有害查询中的特定词汇，验证伪装字符串通过关键词过滤器，并（可选地）使用简单识别器将伪装字符串解码回来。

## 交付

本课产出 `outputs/skill-encoding-audit.md`。给定一份越狱防御报告，它枚举涵盖的编码攻击族（ASCII 艺术、base64、leet-speak、UTF-8 同形字符、UTES）以及捕获每种攻击的防御层。

## 练习

1. 运行 `code/main.py`。验证伪装字符串通过简单关键词过滤器。报告所需的字符级变化。

2. 实现第二种编码：对同一目标词使用 base64。比较其与 ArtPrompt 的过滤器绕过率和恢复难度。

3. 阅读 Jiang 等人 2024 年第 4.3 节（五模型结果）。提出一个理由解释为什么在同一基准上 Claude 的 ArtPrompt 抵抗力高于 Gemini。

4. 设计一个预生成防御，检测提示中的 ASCII 艺术形状区域。测量在合法代码、表格和数学符号上的误报率。

5. StructuralSleight 列出了 10 种编码结构。草拟一个处理所有 10 种的泛化防御，并估算每个受防御提示的计算成本。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| ArtPrompt | "ASCII 艺术攻击" | 用 ASCII 艺术渲染遮蔽安全词汇的两步越狱 |
| 遮蔽（Cloaking） | "隐藏词汇" | 用模型可读但过滤器不可读的视觉表示替换禁止令牌 |
| UTES | "非常规结构" | 非常规文本编码结构——树、图、嵌套 JSON 等，用于走私内容 |
| ViTC | "视觉文本能力" | 衡量模型读取非语义视觉编码能力的基准 |
| 困惑度过滤器（Perplexity Filter） | "PPL 防御" | 拒绝高困惑度提示；因合法结构化输入也得分高而失效 |
| 重新分词（Retokenization） | "分词器偏移防御" | 用不同分词器预处理提示；因识别是视觉的而失效 |
| 同形字符（Homoglyph） | "相似字符" | 与拉丁字母外观相同的 Unicode 字符；绕过子串检查 |

## 延伸阅读

- [Jiang 等人 — ArtPrompt（ACL 2024，arXiv:2402.11753）](https://arxiv.org/abs/2402.11753) — ASCII 艺术越狱论文
- [Li 等人 — StructuralSleight（arXiv:2406.08754）](https://arxiv.org/abs/2406.08754) — UTES 泛化
- [Chao 等人 — PAIR（第12课，arXiv:2310.08419）](https://arxiv.org/abs/2310.08419) — 互补的迭代攻击
- [Anil 等人 — Many-shot Jailbreaking（第13课）](https://www.anthropic.com/research/many-shot-jailbreaking) — 互补的长度攻击
