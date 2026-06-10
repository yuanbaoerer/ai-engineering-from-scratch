# InternVL3：原生多模态预训练

> 在 InternVL3 之前的所有开源 VLM 都遵循相同的三步配方：取一个在数万亿文本 token 上训练好的文本 LLM，加装一个视觉编码器，然后对连接处进行微调。这种做法可行，但会产生对齐债务（alignment debt）—— 文本 LLM 将其全部预训练预算都花在了纯文本上，无法原生理解视觉 token。当你事后添加视觉能力时，LLM 必须重新学习如何将视觉输入与文本推理关联起来，同时不能遗忘文本知识。InternVL3（Zhu 等人，2025 年 4 月）拒绝了这种事后追加的方法：一次预训练运行，文本和多模态数据从第一步就交错在一起。其结果在 78B 参数的开源模型上，在 MMMU-Pro 上达到了与 Gemini 2.5 Pro 相当的性能。本课将解读原生预训练的理由以及实现它时带来的变化。

**类型：** 学习
**语言：** Python（标准库，训练语料混合器）
**前置条件：** Phase 12 · 05，Phase 12 · 07（recipes）
**时间：** 约 120 分钟

## 学习目标

- 解释为什么事后 VLM 训练会积累对齐债务，并列举三个可测量的症状（catastrophic forgetting、answer drift、visual-text inconsistency）。
- 描述 InternVL3 的原生预训练语料混合策略，以及 text : interleaved : caption 的比例为何重要。
- 比较 V2PE（variable visual position encoding）与 Qwen2-VL 的 M-RoPE。
- 说出 Visual Resolution Router（ViR）和 Decoupled Vision-Language（DvD）部署优化的名称。

## 问题所在

事后 VLM 训练是默认做法。LLaVA、BLIP-2、Qwen-VL、Idefics —— 都是拿一个已经预训练好的 LLM（Llama、Vicuna、Qwen、Mistral），然后添加视觉能力。训练阶段通常如下：

1. 冻结 LLM + 冻结视觉编码器 + 可训练的 projector，在图文对数据上训练以对齐嵌入。
2. 解冻 LLM，在指令数据（LLaVA-Instruct、ShareGPT4V）上训练。
3. 可选的任务特定微调。

三种对齐债务症状会显现出来：

- **灾难性遗忘（Catastrophic forgetting）。** 事后 VLM 会遗忘纯文本技能。GSM8K 分数下降 5-10 分。Hellaswag 分数下降。纯文本智能体出现退化。
- **答案漂移（Answer drift）。** 同一个视觉问题的微小措辞变化会得到不同答案。视觉编码器与 LLM 的连接比 LLM 自身 token 之间的绑定更弱。
- **视觉-文本不一致（Visual-text inconsistency）。** VLM 可以正确描述一张图像，然后回答一个与其自身描述相矛盾的问题。视觉 token 无法像文本那样参与 LLM 的内部一致性检查。

这些症状都有充分的文献记录。MM1.5 第 4 节对其进行了量化。LLaVA-OneVision 的消融实验也暗示了它们的存在。原生预训练就是答案。

## 核心概念

### 原生多模态预训练（Native multimodal pretraining）

InternVL3 从一个从第一步起就是原生多模态的语料库上从头训练。混合比例为：

- 40% 纯文本数据（FineWeb、Proof-Pile-2 等）
- 35% 交错的图像-文本数据（OBELICS、MMC4-style）
- 20% 配对图像-字幕数据
- 5% 视频-文本数据

视觉 token、文本 token 和跨模态交互从第一个梯度步起就参与同一个损失函数。没有对齐预训练，没有 projector 冻结阶段，没有需要恢复的灾难性遗忘。

训练是基础模型的单阶段。指令微调随后进行，但基础模型已经将视觉 token 视为一等公民。

### V2PE（variable visual position encoding，可变视觉位置编码）

Qwen2-VL 使用 M-RoPE 进行固定的轴分配。InternVL3 引入了 V2PE：位置编码根据模态类型（文本、图像、视频）变化，并带有可学习的缩放。在实践中：

- 文本 token 获得 1D 位置（文本索引）。
- 图像 patch 获得 2D 位置（行，列）。
- 视频帧获得 3D 位置（时间，行，列）。

三者共享相同的 RoPE 频率基底，但每个频段的隐藏维度分配是一个可学习的参数，而非固定分割。这赋予了在预训练期间权衡时间分辨率与空间频率分辨率的自由度。

V2PE 的消融实验声称：在相同计算量下，比 M-RoPE 在视频基准上高 1-2 分。不是革命性的，但更简洁。

### Visual Resolution Router（ViR，视觉分辨率路由器）

部署优化。并非所有图像都需要全分辨率编码。一张只有一个低细节物体的照片，以 1280px 原生分辨率编码时会浪费 token。ViR 是一个小型分类器，在编码之前预测回答问题所需的最低分辨率。

路由分为三个层级：低分辨率（256 tokens）、中分辨率（576）、高分辨率（2048+）。在生产流量中，60% 的查询使用低或中分辨率就足够了。净效果：在同等质量下吞吐量提升 2-3 倍。

### Decoupled Vision-Language deployment（DvD，解耦视觉-语言部署）

当你部署一个大型 VLM 时，视觉编码器每张图像只运行一次，但 LLM 为每个输出 token 自回归运行。两个组件有不同的瓶颈（视觉 = GPU 内存带宽用于卷积 + 注意力；LLM = KV cache）。DvD 将它们拆分到不同的 GPU 上，并在两者之间进行流式传输。

对于一个 8B + 400M 编码器模型，DvD 的每节点吞吐量大约是同地部署的两倍。

### 单阶段 vs 多阶段质量

InternVL3 的主要基准声明：在 78B 参数下，匹配 Gemini 2.5 Pro 的 MMMU-Pro。在 38B 下，匹配 GPT-4o。在 8B 下，领先开源 8B 排行榜。全部基于单阶段预训练 + 指令微调的配方。

对齐债务假设是可测量的：InternVL3-8B 在每单位视觉基准收益上，比 Qwen2.5-VL-7B 损失的文本基准分数（MMLU、GSM8K）更少。该模型更像一个通才，因为训练是一个整体，而非两段拼接。

### InternVL3.5 和 InternVL-U

InternVL3.5（2025 年 8 月）扩展了这一配方。相同的原生预训练方法，更多数据，更多参数。MMMU 上的提升是渐进式的。

InternVL-U（2026 年）增加了统一生成能力 —— 通过在同一骨干网络上添加 MMDiT 头部实现图像输出。"U" 代表 "Understanding + generation（理解 + 生成）"，追逐 Transfusion 风格的统一模型（Lesson 12.13）。相同的原生预训练骨干网络同时支持理解和生成头部。

### 原生预训练的权衡

原生预训练并非没有代价：

- **计算。** 从头训练一个新的 VLM 成本与训练一个文本 LLM 相同 —— 数百万 GPU 小时。事后适配复用现有 LLM 权重，节省大部分成本。
- **数据。** 大规模的交错图像-文本语料库很稀缺。OBELICS 有 1.41 亿文档；MMC4 有 5.71 亿。纯文本数据以 15T token 的规模流通。多模态预训练数据稀缺是一个硬性约束。
- **基础 LLM 复用。** 原生预训练放弃了以后更换新 LLM 的选项。事后方法允许你通过仅重新训练 adapter，将 Llama-3.1 替换为 Llama-4。

InternVL3 的赌注是：对齐债务比复用损失更严重。基准测试支持这一主张。但生产成本之高，使得未来实验室难以廉价复制。事后 VLM 将继续存在，因为它们对大多数项目来说仍然更便宜。

## 动手实践

`code/main.py` 是一个训练语料混合器和 ViR 路由器模拟器。它可以：

- 接收目标语料混合比例（%text、%interleaved、%caption、%video），并计算每个模态的预期训练步数。
- 在一批查询上模拟 ViR 路由（分布：50% 低细节、30% 中等、20% 高细节），并报告平均 token 数量。
- 根据编码器与 LLM 的 FLOPs，报告 DvD 吞吐量估算。
- 并排打印事后预训练与原生预训练在参数、计算、数据和预期对齐债务症状方面的对比。

## 交付成果

本节课产出 `outputs/skill-native-vs-posthoc-auditor.md`。给定一个拟议的 VLM 训练计划，它会审计应该选择原生预训练还是事后预训练，标记对齐债务风险，并推荐语料混合比例。在你规划一个新的开源 VLM 项目并需要选择训练策略时使用它。

## 练习题

1. 估算 InternVL3-8B（原生预训练）与 LLaVA-OneVision-7B（事后预训练）之间的计算量差异。GPU 小时的大致比例是多少？什么解释了这个差距？

2. InternVL3 报告的数据比例为 40% 文本 / 35% 交错 / 20% 标题 / 5% 视频。如果你的目标任务以视频为主，请提出一个新的比例，并论证为什么基础模型仍然需要大量的文本和标题数据。

3. 阅读 MM1.5 第 4 节关于遗忘的内容。说出事后训练显示最大回归的确切基准。回归造成了多大损失？

4. ViR 将 60% 的流量路由到低分辨率编码。它会错误路由哪些类型的查询（将需要高分辨率的查询发送到低分辨率）？提出三种路由器失效模式。

5. DvD 将视觉和 LLM 拆分到不同的 GPU 上。在什么流量模式下，DvD 会降低而不是提升吞吐量？

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| Native multimodal pretraining | "一起从头开始" | 文本 + 图像 + 视频 token 从第 1 步起就参与损失计算，而非事后嫁接 |
| Alignment debt | "事后惩罚" | 将视觉能力嫁接在冻结 LLM 上时，产生的文本技能可测量退化和答案一致性下降 |
| V2PE | "可变视觉位置编码" | 按模态可学习的位置编码分配；InternVL3 对 M-RoPE 的改进 |
| ViR | "分辨率路由器" | 小型分类器，在编码前为每个查询选择所需最低分辨率，节省推理 token |
| DvD | "解耦部署" | 视觉编码器在一个 GPU 上，LLM 在另一个 GPU 上，通过流式传输交接；大型 VLM 吞吐量翻倍 |
| InternVL-U | "统一理解 + 生成" | 2026 年的后续工作，在原生预训练骨干网络上添加图像生成头部 |
| Interleaved corpus | "OBELICS / MMC4" | 文本和图像按自然阅读顺序排列的文档；原生预训练的原材料 |

## 延伸阅读

- [Chen 等人 — InternVL 1 (arXiv:2312.14238)](https://arxiv.org/abs/2312.14238)
- [Zhu 等人 — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
- [InternVL3.5 (arXiv:2508.18265)](https://arxiv.org/abs/2508.18265)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Zhang 等人 — MM1.5 (arXiv:2409.20566)](https://arxiv.org/abs/2409.20566)
