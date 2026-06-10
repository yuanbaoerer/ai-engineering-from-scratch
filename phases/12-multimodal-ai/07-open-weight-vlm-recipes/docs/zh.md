# 开源权重 VLM 实战指南：什么才是真正重要的

> 2024-2026 年的开源权重 VLM 文献充斥着消融表格。苹果的 MM1 测试了图像编码器、连接器和数据混合的 13 种组合。Allen AI 的 Molmo 证明了详细的人工标注优于 GPT-4V 蒸馏。Cambrian-1 进行了 20 多种编码器对比。Idefics2 形式化了五轴设计空间。Prismatic VLMs 在受控基准上比较了 27 种训练方案。在所有这些繁杂的结果中，有一小撮结论在各篇论文间保持一致：图像编码器比连接器架构更重要，数据混合比前两者都重要，而详细的人工标注优于蒸馏生成的合成数据。本课替你阅读了这些表格，让你不必再为此头疼。

**类型：** 学习 + 实验
**语言：** Python（标准库，消融表格解析器 + 方案选择器）
**前置条件：** Phase 12 · 05（LLaVA 基线）
**时间：** 约 180 分钟

## 学习目标

- 说出 VLM 五轴设计空间：图像编码器、连接器、LLM、数据混合、分辨率调度。
- 阅读 MM1 / Idefics2 / Cambrian-1 的消融表格，并预测调节哪个旋钮会影响给定基准测试。
- 给定计算预算和任务组合，为新 VLM 选择一套方案（编码器、连接器、数据、分辨率）。
- 解释为什么在相同 token 数量下，详细的人工标注优于 GPT-4V 蒸馏。

## 问题所在

开源权重的 VLM 成百上千。"好"与"最先进"之间的差距大多不在架构，而在数据、分辨率调度和编码器选择。知道模型表现不佳时应该先拧哪个旋钮，能让你避免一个价值五百万 GPU 小时的错误。

2023 年的浪潮（LLaVA-1.5、InstructBLIP、MiniGPT-4）采用图文对预训练 + LLaVA-Instruct-150k。不错的基线。在 MMMU 上大约止步于 35%。

2024 年的浪潮（MM1、Idefics2、Molmo、Cambrian-1、Prismatic VLMs）进行了详尽的消融实验。结果既出人意料又实用。

## 核心概念

### 五轴设计空间

Idefics2（Laurençon 等人，2024）命名了这些轴：

1. **图像编码器。** CLIP ViT-L/14、SigLIP SO400m/14、DINOv2 ViT-g/14、InternViT-6B。编码器在 patch 大小、分辨率和预训练目标上各不相同。
2. **连接器。** MLP（2-4 层）、Q-Former（32 个查询 + 交叉注意力）、Perceiver Resampler（64 个查询）、C-Abstractor（卷积 + 双线性池化）。
3. **语言模型。** Llama-3 8B / 70B、Mistral 7B、Phi-3、Gemma-2、Qwen2.5。LLM 的规模是参数成本的主导因素。
4. **训练数据。** 图文对（CC3M、LAION）、交错式（OBELICS、MMC4）、指令数据（LLaVA-Instruct、ShareGPT4V、PixMo、Cauldron）。
5. **分辨率调度。** 固定 224/336/448、AnyRes、原生动态分辨率。在训练期间逐步提升或保持不变。

每个生产级 VLM 都会在每个轴上做出选择。MMMU 分数的大部分方差由轴 1、4 和 5 解释——而不是你选了哪个连接器。

### 轴 1：编码器 > 连接器

MM1 第 3.2 节表明：将编码器从 CLIP ViT-L/14 换成 SigLIP SO400m/14，MMMU 提升 3 分以上。将连接器从 MLP 换成 Perceiver Resampler，提升不到 1 分。Idefics2 复现了该结论：SigLIP > CLIP，在相同 token 数量下 Q-Former ≈ MLP ≈ Perceiver。

Cambrian-1 的"Cambrian Vision Encoders Match-Up"（Tong 等人，2024）在视觉中心基准（CV-Bench）上测试了 20 多种编码器。排行榜前列是 DINOv2 和 SigLIP 的混合；CLIP 居中；ImageBind 和 ViT-MAE 靠后。从 CLIP ViT-L 到 DINOv2 ViT-g/14 的差距在 CV-Bench 上约为 5-7 分。

2026 年开源 VLM 的默认编码器是 SigLIP 2 SO400m/14，用于语义 + 密集特征；有时与 DINOv2 ViT-g/14 特征拼接（Cambrian 的"Spatial Vision Aggregator"就是这样做的）。

### 轴 2：连接器设计无关紧要

MM1、Idefics2、Prismatic 和 MM-Interleaved 都得出相同结论：在固定视觉 token 数量下，连接器架构几乎不影响结果。一个对 patch token 做均值池化后再过 2 层 MLP 的方案，在相同 token 预算下与 32 查询的 Q-Former 差距在 1 分以内。

真正重要的是 token 数量。更多视觉 token = 更多 LLM 计算量 = 更好的性能，直到某个拐点后收益递减。每张图 64 个 token 对 OCR 来说太少。576-1024 个 token 是大多数开源 VLM 的最佳点。2048+ 只对文档和图表有帮助。

Q-Former 与 MLP 是成本问题，不是质量问题：Q-Former 将 token 上限固定在 32-64，无论图像分辨率如何；MLP 会输出所有 patch token。对于高分辨率输入，Q-Former 节省 LLM 上下文；对于低分辨率，差异可以忽略。

### 轴 3：LLM 规模决定上限

将 LLM 从 7B 翻倍到 13B，在各篇 VLM 论文中都能稳定带来 MMMU 上 2-4 分的提升。到 70B 时大多数基准趋于饱和。VLM 的多模态推理上限就是 LLM 的文本推理上限——视觉编码器只能提供输入，不能替它推理。

这就是 Qwen2.5-VL-72B 和 Claude Opus 4.7 能在 MMMU-Pro 和 ScreenSpot-Pro 上碾压对手的原因：语言大脑足够大。一个 7B 的 VLM 无法通过巧妙的连接器设计来替代 70B VLM。

### 轴 4：数据——详细的人工标注击败蒸馏

Molmo + PixMo（Deitke 等人，2024）是 2024 年每个人都应该读的结果。Allen AI 让人类标注员用 1-3 分钟的密集语音转文本来描述图像，产出了 71.2 万张密集标注图像。训练数据中没有任何 GPT-4V 蒸馏。

Molmo-72B 在 11/11 个基准上击败了 Llama-3.2-90B-Vision。差距不在架构——而在标注质量。详细的人工标注每张图包含的信息量是短网页标注的 5-10 倍，且在事实准确性上保持扎实，而 GPT-4V 蒸馏容易产生幻觉。

ShareGPT4V（Chen 等人，2023）和 Cauldron（Idefics2）用混合人工 + GPT-4V 标注遵循了相同的策略。趋势很明确：对于 2026 年的前沿，标注密度 > 标注数量 > 蒸馏便利性。

### 轴 5：分辨率及其调度

Idefics2 的消融实验：384 -> 448 增加 1-2 分。448 -> 980 配合图像切分（AnyRes）在 OCR 基准上再增加 3-5 分。固定分辨率训练在中等精度处趋于平缓；分辨率逐步提升（从 224 开始，最终到 448 或原生分辨率）训练更快且最终精度更高。

Cambrian-1 进行了分辨率与 token 的权衡实验：在固定计算量下，你可以选择低分辨率多 token 或高分辨率少 token。高分辨率对 OCR 更有利；低分辨率多 token 对一般场景理解更有利。

2026 年生产级方案：第一阶段在 384 固定分辨率下训练，第二阶段采用动态分辨率，最高到 1280，用于 OCR 密集型任务。

### Prismatic 受控对比

Prismatic VLMs（Karamcheti 等人，2024）是控制了所有轴的论文。相同的 13B LLM、相同的指令数据、相同的评估——每次只改变一个轴。结果：

- 每张图的视觉 token 数量解释了约 60% 的方差。
- 编码器选择解释了约 20%。
- 连接器架构解释了约 5%。
- 其余（数据混合、调度器、学习率）解释剩余的约 15%。

这是一个粗略的分解，但它是文献中对"我应该先消融哪个"最清晰的回答。

### 2026 年方案选择器

基于以上证据，2026 年新项目的默认开源 VLM 方案：

- **编码器：** SigLIP 2 SO400m/14，原生分辨率配合 NaFlex；如需分割/定位，则拼接 DINOv2 ViT-g/14 的密集特征。
- **连接器：** 对 patch token 使用 2 层 MLP。除非受 token 限制，否则跳过 Q-Former。
- **LLM：** Qwen2.5 / Llama-3.1 / Gemma 2，7B 用于成本敏感场景，70B 用于质量优先场景，按目标延迟选择。
- **数据：** PixMo + ShareGPT4V + Cauldron，补充任务特定的指令数据。
- **分辨率：** 动态（长边最小 256，最大 1280 像素）。
- **调度：** 第一阶段对齐（仅 projector），第二阶段全量微调，第三阶段任务特定微调。

上述每一项默认值都可以追溯到本课末尾引用论文中的实测消融实验。

## 动手实践

`code/main.py` 是一个消融表格解析器和方案选择器。它编码了 MM1 和 Idefics2 的消融表格（精简版），并允许你查询：

- "给定预算 X 和任务 Y，哪个方案胜出？"
- "如果我在 7B Llama 上将 SigLIP 换成 CLIP，预期的 MMMU 差距是多少？"
- "对于 80% 置信度的答案，我应该先消融哪个轴？"

输出是一个带预期基准差距的排名方案列表，以及一条"先消融这个"的建议。

## 交付成果

本课产出 `outputs/skill-vlm-recipe-picker.md`。给定目标任务组合、计算预算和延迟目标，它会输出一套完整方案（编码器、连接器、LLM、数据混合、分辨率调度），并引用支持每个选择的消融实验。让工程师不必每次启动新 VLM 项目时都重新发明 Idefics2 的消融表格。

## 练习题

1. 阅读 MM1 第 3.2 节。对于固定的 2B LLM，在 5000 万张图像的预算下，哪个编码器胜出？在 13B LLM 下答案会翻转吗？为什么？

2. Cambrian-1 发现，在视觉中心基准上拼接 DINOv2 + SigLIP 优于单独使用任一者，但在 MMMU 上没有增益。预测哪些基准会提升，哪些保持平稳。

3. 你的目标是在 2B LLM 上做一个移动端 UI Agent。选择编码器、连接器、分辨率和数据混合。用具体的消融表格为每个选择辩护。

4. Molmo 发布了 4B 和 72B 模型。4B 与闭源 7B VLM 有竞争力；72B 在 11/11 个基准上击败 Llama-3.2-90B-Vision。这对 LLM 规模瓶颈假说说明了什么？

5. 设计一个消融表格，在 7B VLM 上隔离数据混合质量与编码器质量。最少需要多少次训练运行？提出四个轴的设置。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Ablation（消融） | "拧一个旋钮" | 训练多轮运行，每轮只在设计空间的一个轴上不同，其余保持不变 |
| Connector（连接器） | "桥梁" / "投影器" | 将视觉编码器输出映射到 LLM token 空间的可训练模块（MLP、Q-Former、Perceiver） |
| Detailed human caption（详细人工标注） | "密集标注" | 多句子的人工撰写描述（通常 80-300 个 token），比网页 alt 文本丰富得多 |
| Distillation（蒸馏） | "GPT-4V 标注" | 由更强的专有 VLM 生成的训练数据；方便但容易继承幻觉 |
| AnyRes / dynamic res | "高分辨率路径" | 通过图像切分或 M-RoPE 输入大于编码器原生分辨率的图像的策略 |
| Resolution ramp（分辨率爬坡） | "课程学习" | 从低分辨率开始并逐步提高的训练调度，加速对齐学习 |
| Vision-centric bench（视觉中心基准） | "CV-Bench / BLINK" | 强调细粒度视觉感知而非重语言推理的评估 |
| PixMo | "Molmo 的数据" | Allen AI 的 71.2 万张密集标注图像数据集；人类语音转录为密集标注 |

## 延伸阅读

- [McKinzie 等人 — MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611)
- [Laurençon 等人 — Idefics2 / What matters building VLMs (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Deitke 等人 — Molmo and PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)
- [Tong 等人 — Cambrian-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)
- [Karamcheti 等人 — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)
