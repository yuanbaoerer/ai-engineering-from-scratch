# 从 CLIP 到 BLIP-2 —— Q-Former 作为模态桥梁

> CLIP 可以对齐图像和文本，但无法生成字幕、回答问题或进行对话。BLIP-2（Salesforce，2023）通过一个可训练的小型桥梁解决了这个问题：32 个可学习的查询向量通过交叉注意力机制关注冻结的 ViT 的特征，然后直接插入到冻结的 LLM 的输入流中。这座桥梁仅有 1.88 亿参数，却将一个 110 亿参数的 LLM 与 ViT-g/14 连接起来。到 2026 年为止，每一个基于适配器的 VLM——MiniGPT-4、InstructBLIP、LLaVA 的表亲——都是它的后代。本节课将解读 Q-Former 的架构，解释其两阶段训练，并构建一个将视觉 token 输入冻结文本解码器的简化版本。

**类型：** 构建
**语言：** Python（标准库，交叉注意力 + 可学习查询演示）
**前置知识：** Phase 12 · 02（CLIP），Phase 7（Transformers）
**时间：** 约 180 分钟

## 学习目标

- 解释为什么在成本和稳定性方面，在冻结的视觉编码器和冻结的 LLM 之间设置一个可训练的瓶颈优于端到端微调。
- 实现一个交叉注意力块，其中一组固定的可学习查询关注外部图像特征。
- 梳理 BLIP-2 的两阶段预训练：表征学习（ITC + ITM + ITG），然后是生成学习（使用冻结解码器的 LM 损失）。
- 将 Q-Former 与 LLaVA 中使用的更简单的 MLP 投影器进行比较，并论证每种选择在何时胜出。

## 问题

你有一个冻结的 ViT，每张图像产生 256 个维度为 1408 的 patch token。你有一个冻结的 7B LLM，期望维度为 4096 的 token 嵌入。显而易见的桥梁——一个从 1408 到 4096 的线性层——可以工作，但将所有 256 个 patch token 输入到 LLM 的上下文中，每张图像会消耗 256 个额外 token。在一个包含 32 张图像的批次中，仅视觉模态就消耗了 8192 个 token。

BLIP-2 的问题是：能否将 256 个 token 的图像表示压缩成更少的 token（比如 32 个），同时保留足够的信息让 LLM 生成字幕、回答问题和推理图像？并且能否在不触碰冻结骨干网络的情况下训练这座桥梁，使训练成本仅等于桥梁本身的参数？

答案是：Q-Former。32 个可学习的"查询"向量对 ViT 的 patch token 进行交叉注意力，产生一个 32 个 token 的视觉摘要供 LLM 消费。总共 1.88 亿参数。在接触 LLM 之前，使用对比、匹配和生成目标进行训练。

## 概念

### 可学习查询

Q-Former 的核心技巧：不是让 LLM 的文本 token 关注图像 patch，而是引入一组新的 32 个可学习查询向量 `Q`，让*它们*去关注图像 patch。这些查询是模型的参数——在训练过程中学习，并且每张图像都使用相同的 32 个查询。

经过交叉注意力后，每个查询都持有图像的压缩摘要——"描述主要对象"、"描述背景"、"统计对象数量"等。查询并不会字面意义上专门化为语义标签；它们学习任何能让下游损失下降的编码。

### 架构

Q-Former 是一个小型 transformer（12 层，约 1 亿参数），有两条路径：

1. 查询路径：32 个查询向量先经过自注意力（在它们之间），然后交叉注意力关注冻结的 ViT 的 patch token，最后经过 FFN。
2. 文本路径：一个类似 BERT 的文本编码器与查询路径共享自注意力和 FFN 权重。文本路径禁用交叉注意力。

训练时两条路径同时运行。查询和文本通过共享的自注意力进行交互，这意味着查询可以基于文本进行条件处理，用于需要文本的任务（ITM、ITG）。在 VLM 交接的推理时，只有查询流过，产生 32 个视觉 token。

### 两阶段训练

BLIP-2 分两阶段进行预训练：

第一阶段：表征学习（无 LLM）。三个损失：
- ITC（Image-Text Contrastive，图像-文本对比）：在池化后的查询 token 和文本 CLS token 之间应用 CLIP 风格的对比损失。
- ITM（Image-Text Matching，图像-文本匹配）：二分类器——这个图像-文本对是否匹配？使用困难负样本挖掘。
- ITG（Image-Grounded Text Generation，基于图像的文本生成）：在文本上应用因果 LM 头，以查询为条件。迫使查询编码可生成文本的内容。

仅训练 Q-Former。ViT 被冻结。不涉及 LLM。

第二阶段：生成学习。连接一个冻结的 LLM（OPT-2.7B 或 Flan-T5-XL 等）。通过一个小型线性层将 32 个查询输出投影到 LLM 的嵌入维度。将它们前置到文本提示中。仅在拼接后的提示 + 图像 + 字幕序列上，用 LM 损失训练线性投影和 Q-Former。

第二阶段后，Q-Former + 投影就是完整的视觉适配器。推理时：图像 → ViT → Q-Former → 线性投影 → 前置到文本 → 冻结的 LLM 生成输出。

### 参数经济学

BLIP-2：ViT-g/14（11 亿，冻结）+ OPT-6.7B（67 亿，冻结）+ Q-Former（1.88 亿，训练）= 总计 80 亿，训练 1.88 亿。Q-Former 仅占完整堆栈参数的约 2.4%。训练成本反映了这一点：在少量 A100 上训练数天，而非端到端训练的数周。

质量：BLIP-2 在零样本 VQA 上达到或超过 Flamingo-80B，同时小了 50 倍。这座桥梁有效。

### InstructBLIP 与指令感知的 Q-Former

InstructBLIP（2023）扩展了 Q-Former，增加了一个额外输入：指令文本本身。在交叉注意力时，查询现在可以同时访问图像 patch 和指令。查询可以按指令专门化（"数汽车"、"描述氛围"），而不是学习单一的固定摘要。在 held-out 任务上获得基准提升。

### MiniGPT-4 与仅投影器方法

MiniGPT-4 保留了 Q-Former，但仅训练输出线性投影，同时冻结其他所有部分。成本低，但代价是质量——查询是 BLIP-2 的，不是你自己的。适合快速迭代，但不是最佳架构。

### 为什么 LLaVA 选择了更简单的方式

LLaVA（2023，Lesson 12.05）用普通的 2 层 MLP 替代了 Q-Former，将每个 ViT patch token 投影到 LLM 空间——24x24 网格产生 576 个 token，全部输入 LLM。压缩更差，但让 LLM 关注原始 patch。这在当时是有争议的；到 2023 年底，它成为主流，因为视觉指令数据（LLaVA-Instruct-150k）证明 MLP 可以被训练到保留足够的信号。权衡：LLaVA 的上下文填充更快，但它自然地扩展到多图像和视频。

到 2026 年，领域分化为：Q-Former 在 token 预算重要的地方存活（长视频、多图像）；MLP 投影器在原始质量 per token 是优先事项的地方占主导。

### 门控交叉注意力：Flamingo，先驱

Flamingo（Lesson 12.04）早于 BLIP-2，使用了相同的交叉注意力思想，但在每个冻结的 LLM 层上，而不是作为单一桥梁。BLIP-2 证明你可以仅压缩到输入层，仍然有效。Gemini 和 Idefics 结合了两者：交错的输入 token 加上可选的门控交叉注意力用于上下文少样本学习。

### 2026 年的后代

- Q-Former：BLIP-2、InstructBLIP、MiniGPT-4，以及大多数出于 token 预算原因的视频-语言模型。
- Perceiver resampler：Flamingo 的变体（Lesson 12.04）；Idefics 家族、Eagle、OmniMAE。
- MLP 投影器：LLaVA、LLaVA-NeXT、LLaVA-OneVision、Cambrian-1。
- Attention pool：VILA、PaliGemma。

四种都是有效的。决定因素是你受限于 token 预算还是 quality-per-token。

## 使用它

`code/main.py` 构建了一个标准库 Q-Former 风格的交叉注意力：

1. 模拟 256 个图像 patch token（维度 128）。
2. 实例化 32 个可学习查询（维度 128）。
3. 运行缩放点积交叉注意力（Q 来自查询，K/V 来自 patch）。
4. 通过线性层投影到 LLM 维度（512）。
5. 输出 32 个 LLM 就绪的视觉 token。

所有数学运算使用纯 Python（向量的嵌套循环）。简化但形状正确。注意力权重矩阵被打印出来，以便你可以看到每个查询从哪些 patch 提取信息。

## 交付它

本节课生成 `outputs/skill-modality-bridge-picker.md`。给定一个目标 VLM 配置（视觉编码器 token 数量、LLM 上下文预算、部署约束、质量目标），它会推荐 Q-Former vs MLP vs Perceiver resampler，并附带简短的理由和每种桥梁的参数数量估计。

## 练习

1. 在 PyTorch 中实现交叉注意力块。验证使用 32 个查询和 256 个键/值时，注意力权重矩阵为 32 x 256，且 softmax 后每行和为 1。

2. 在 BLIP-2 第一阶段，Q-Former 同时运行三个损失：ITC、ITM、ITG。为每个损失编写伪代码的前向签名。哪一个需要文本编码器路径处于激活状态？

3. 比较参数数量：Q-Former（12 层，768 隐藏层）vs 2 层 MLP 投影器（1408 → 4096，两层）。在什么 LLM 规模下，1.88 亿参数的 Q-Former 成本在训练效率上得到回报？

4. 阅读 BLIP-2 论文（arXiv:2301.12597）第 3.2 节关于 Q-Former 如何初始化。解释为什么从 BERT-base（而非随机）初始化能加速收敛。

5. 对于一段 10 分钟、1 FPS 采样到 60 帧的视频，计算每帧 token 成本在（Q-Former → 32 token/帧）vs（MLP 投影器 → 576 token/帧）下的对比。哪种能放入 128k token 的 LLM 上下文窗口？

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| Q-Former | "Querying transformer" | 带有 32 个可学习查询向量的小型 transformer，对冻结的 ViT 特征进行交叉注意力 |
| Learnable queries | "Soft prompt for vision" | 一组固定的参数，作为交叉注意力的查询侧；每个模型学习一次，所有输入共享 |
| Cross-attention | "Q from here, K/V from there" | 查询、键和值来自不同来源的注意力机制；查询如何从 ViT patch 提取信息 |
| ITC | "Image-text contrastive" | 应用于 Q-Former 池化查询与文本 CLS 的 CLIP 风格损失 |
| ITM | "Image-text matching" | 对困难负样本挖掘后的对进行二分类；迫使查询区分细粒度不匹配 |
| ITG | "Image-grounded text generation" | 以查询为条件生成文本的因果 LM 损失；迫使查询编码可被文本解码的内容 |
| Two-stage pretraining | "Representation then generative" | 第一阶段单独训练 Q-Former（ITC/ITM/ITG）；第二阶段连接冻结的 LLM，仅训练投影 + Q-Former |
| Frozen backbone | "Do not finetune" | 视觉编码器和 LLM 权重固定；仅训练桥梁 |
| Projection head | "Linear to LLM dim" | 将 Q-Former 输出映射到 LLM 嵌入维度的最终线性层 |
| Perceiver resampler | "Flamingo's version" | 类似的可学习查询交叉注意力，Flamingo 在每个层使用而非作为单一桥梁 |

## 延伸阅读

- [Li et al. — BLIP-2 (arXiv:2301.12597)](https://arxiv.org/abs/2301.12597) — 核心论文。
- [Li et al. — BLIP (arXiv:2201.12086)](https://arxiv.org/abs/2201.12086) — 带有 ITC/ITM/ITG 三重组合的前身。
- [Li et al. — ALBEF (arXiv:2107.07651)](https://arxiv.org/abs/2107.07651) — "align before fuse" — 第一阶段训练的概念先驱。
- [Dai et al. — InstructBLIP (arXiv:2305.06500)](https://arxiv.org/abs/2305.06500) — 指令感知的 Q-Former。
- [Zhu et al. — MiniGPT-4 (arXiv:2304.10592)](https://arxiv.org/abs/2304.10592) — 仅投影器方法。
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — 可学习查询交叉注意力的一般架构。
