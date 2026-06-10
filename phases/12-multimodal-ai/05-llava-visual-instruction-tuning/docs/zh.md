# LLaVA 与视觉指令微调

> LLaVA（2023年4月）是全球被复现最多的多模态架构。它用2层MLP替代了BLIP-2的Q-Former，用朴素的token拼接替代了Flamingo的门控交叉注意力，并在158k条由GPT-4从纯文本caption生成的视觉指令数据上训练。任何在2023至2026年间构建过VLM的从业者，都曾构建过某种LLaVA的变体。LLaVA-1.5加入了AnyRes。LLaVA-NeXT提升了分辨率。LLaVA-OneVision将图像、多图和视频统一到一个recipe中。本课将研读这个recipe，实现projector，并解释为什么"更简单反而赢了"。

**类型：** 构建
**语言：** Python（标准库，projector + instruction-template builder）
**前置知识：** Phase 12 · 02（CLIP），Phase 11（LLM工程——指令微调）
**时间：** 约180分钟

## 学习目标

- 构建一个2层MLP projector，将ViT patch embeddings（维度1024）映射到LLM的embedding维度（维度4096）。
- 走通LLaVA的两阶段recipe：（1）在558k caption pair上做projector对齐，（2）在158k条GPT-4生成的指令数据上做视觉指令微调。
- 构造一个LLaVA格式的prompt，包含图像token占位符、system prompt以及user/assistant轮次。
- 解释为什么社区从Q-Former转向MLP，尽管Q-Former在token预算上占优。

## 问题背景

BLIP-2的Q-Former（第12.03课）将一张图像压缩为32个token。干净、高效、适合刷榜。但它有两个问题。

第一，Q-Former是可训练的，但它的损失函数不是最终任务。阶段1训练ITC+ITM+ITG。阶段2训练LM loss。queries学到某种中间表示，然后LLM再去解码。信息在瓶颈处丢失了。

第二，Q-Former有1.88亿参数，在LLaVA所处的2023年规模下，你必须将它与目标LLM共同设计。换了LLM，就得重训Q-Former。换了vision encoder，也得重训。每种组合都是一个独立的研发项目。

LLaVA的答案简单到令人尴尬：取出ViT的576个patch token，每个都通过一个2层MLP（`1024 → 4096 → 4096`），然后把全部576个token丢进LLM的输入序列。没有瓶颈。没有基于奇怪目标的阶段1预训练。只用LM loss来训练这个MLP。

数据从哪里来？LLaVA的第二个洞见：用GPT-4（纯文本）来生成指令数据。把COCO caption和bounding-box数据喂给GPT-4，让它生成对话、描述和复杂推理问题。免费获得158k条指令-回复轮次。无需人工标注。

结果是：一个VLM，在8张A100上跑了一天，在MMMU上击败了Flamingo，并发布了一个社区可以扩展的开放checkpoint。到2023年底，它已经催生了50多个fork。

## 核心概念

### 架构

LLaVA-1.5 @ 13B：
- Vision encoder：CLIP ViT-L/14 @ 336（阶段1冻结，阶段2可选择性解冻）。
- Projector：2层MLP，GELU激活，`1024 → 4096 → 4096`。
- LLM：Vicuna-13B（后来是Llama-3.1-8B）。

图像 + 文本prompt的前向传播：

```
img -> ViT -> 576个维度为1024的patches
patches -> MLP -> 576个维度为4096的tokens
prompt: system + "<image>"占位符 + user question
将<image> token替换为576个projected tokens
将完整序列喂给LLM
解码回复
```

图像占用了LLM上下文中576个token。在2048上下文下，还剩1472个token给文本。在32k上下文下，这只是一个零头。

### 阶段1：projector对齐

冻结ViT。冻结LLM。只训练2层MLP。数据集：558k图像-文本caption对（LAION-CC-SBU）。损失：在caption上的语言建模，以projected image tokens为条件。

在batch 128下单个epoch，几小时就能完成。projector学会将ViT空间映射到LLM空间。没有任务特定的监督。

### 阶段2：视觉指令微调

解冻projector（保持可训练）。解冻LLM（通常完全解冻，有时用LoRA）。在158k视觉指令轮次上训练。

指令数据是诀窍所在。Liu等人通过以下方式生成：
1. 取一张COCO图像。
2. 提取文本描述（5条人工caption + bounding-box列表）。
3. 发送给GPT-4，使用三种prompt模板：
   - 对话："Generate a back-and-forth dialogue between a user and assistant about this image."
   - 详细描述："Give a rich, detailed description of the image."
   - 复杂推理："Ask a question that requires reasoning about the image, then answer it."
4. 将GPT-4的输出解析为（instruction, response）对。

这一切都不直接接触图像——只使用文本描述。GPT-4会幻觉出合理的图像内容。有些噪声，但有效：158k轮次足以解锁对话能力。

### 为什么社区复现了这个架构

- 没有需要调优的阶段1特定损失。全程使用LM loss。
- Projector在几小时内就能训练完成，而不是几天。
- LLM可以替换（LLaVA-Llama2、LLaVA-Mistral、LLaVA-Llama3），只需重训projector。
- 视觉指令数据管道使用GPT-4，为新领域重新生成的成本很低。

### LLaVA-1.5 与 LLaVA-NeXT

LLaVA-1.5（2023年10月）增加了：
- 学术任务数据（VQA、OKVQA、RefCOCO）混入指令微调。
- 更好的system prompt。
- 2048 → 32k上下文。

LLaVA-NeXT（2024年1月）增加了：
- AnyRes：将高分辨率图像切分为2x2或1x3的336x336网格crop，外加一个全局低分辨率缩略图。每个crop变成576个token；总计每张图像约2880个视觉token。OCR和图表任务大幅提升。
- 更好的指令数据混合，使用ShareGPT4V（高质量GPT-4V caption）。
- 更强的基座LLM（Mistral-7B、Yi-34B）。

### LLaVA-OneVision

第12.08课深入讲解OneVision。简短版本：同样的projector，但使用一个课程化训练方案，覆盖单图、多图和视频，共享视觉token预算。

### 与Q-Former的对比

| | Q-Former（BLIP-2） | MLP（LLaVA） |
|---|---|---|
| 每张图像的视觉token数 | 32 | 576（基础）或2880（AnyRes） |
| 可训练参数 | 188M + LM | 40M + LM |
| 阶段1损失 | ITC+ITM+ITG | 仅LM |
| LLM即插即用 | 需要重训 | 替换后只需少量重训 |
| 多图 | 笨拙 | 自然（拼接） |
| 视频 | 笨拙 | 自然（逐帧拼接） |
| Token预算 | 小 | 大 |

MLP在简单性和token灵活性上获胜。Q-Former在token预算上获胜。到2023年底，token预算已不再是瓶颈约束（LLM上下文增长到32k-128k+），简单性占了上风。

### Prompt格式

```
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...
```

`<image>`是一个占位token。在tokenization之前，它被替换为576个视觉token（AnyRes下为2880个）。Tokenizer看到的序列比它训练时的略长，但LLM能处理这种新输入，因为阶段1已经教会了它。

### 参数经济性

LLaVA-1.5-7B的分解：
- CLIP ViT-L/14 @ 336：303M（阶段1冻结，阶段2常解冻）。
- Projector（2x linear）：约22M可训练。
- Llama-7B：7B。
- 总计：7.3B参数。阶段2可训练：完整的7B + 22M projector。

阶段2的训练成本：约20小时在8xA100上。这是关键数字——一天，一个节点，可复现。这就是LLaVA传播开来的原因。

## 使用它

`code/main.py`实现了：

1. 2层MLP projector（玩具规模下维度为16 → 32 → 32），纯Python实现。
2. Prompt构建管道：system prompt + `<image>`替换为N个projected tokens + user轮次 + assistant生成占位符。
3. 一个可视化工具，展示576-token视觉块在LLM上下文中的样子（占2k / 32k / 128k上下文的百分比）。

## 交付它

本课产出`outputs/skill-llava-vibes-eval.md`。给定一个LLaVA家族的checkpoint，它运行一个10-prompt的vibes-eval套件（3个captioning、3个VQA、2个reasoning、2个refusal），并报告一份人类可读的scorecard。不是benchmark；而是一个smoke test，用于确认projector和LLM连接良好。

## 练习题

1. 计算2层MLP projector在`1024 → 4096 → 4096`时的可训练参数数量。考虑GELU和bias，它占LLaVA-13B的多少比例？

2. 为"refusal"情况构造一个LLaVA prompt——图像中包含一个私人个体。写出预期的assistant回复。为什么LLaVA应该在zero-shot下拒绝？需要什么训练数据来强化这种拒绝？

3. 阅读LLaVA-NeXT博客的AnyRes部分。计算一张1344x672图像在AnyRes下的视觉token数量。与336x336下的基础576个token进行比较。

4. LLaVA的阶段1 projector使用caption上的LM loss训练。如果跳过阶段1直接进入阶段2（视觉指令微调）会发生什么？引用Prismatic VLMs消融实验（arXiv:2402.07865）来回答。

5. LLaVA-Instruct-150k使用GPT-4配合COCO caption生成指令。对于一个新领域（医学X光、卫星图像），描述生成领域指令的四步数据管道。每一步可能出什么问题？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| Projector | "MLP bridge" | 2层MLP，GELU激活，将ViT维度映射到LLM维度 |
| Image token | "<image> placeholder" | Prompt标记，在推理前被替换为N个projected visual tokens |
| Visual instruction tuning | "LLaVA stage 2" | 在GPT-4生成的（图像，指令，回复）三元组上训练 |
| Stage 1 alignment | "Projector pretraining" | 冻结ViT和LLM，用caption上的LM loss训练projector |
| AnyRes | "Multi-crop tiling" | 将高分辨率图像切分为tile网格，拼接每个tile的视觉token |
| LLaVA-Instruct | "GPT-4-generated" | 从COCO caption + GPT-4合成的158k指令-回复对 |
| Vision encoder freeze | "Backbone locked" | CLIP权重在阶段1不更新，有时阶段2也不更新 |
| ShareGPT4V | "Better captions" | 100万条由GPT-4V生成的dense caption，用于更高质量的对齐 |
| VQA | "Visual question answering" | 回答关于图像的自由形式问题的任务 |
| Prismatic VLMs | "Design-space paper" | Karamcheti 2024年的消融实验，系统测试projector和数据选择 |

## 延伸阅读

- [Liu et al. — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) — LLaVA论文。
- [Liu et al. — Improved Baselines with Visual Instruction Tuning (arXiv:2310.03744)](https://arxiv.org/abs/2310.03744) — LLaVA-1.5。
- [Chen et al. — ShareGPT4V (arXiv:2311.12793)](https://arxiv.org/abs/2311.12793) — dense caption数据集。
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865) — 设计空间消融实验。
- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326) — 统一的单图、多图、视频模型。
