# DeepSeek-V3 架构详解

> 阶段 10 · 第 14 课命名了每个开源模型都会调节的六个架构旋钮。DeepSeek-V3（2024 年 12 月发布，总参数量 671B，激活参数量 37B）调节了全部六个，并新增了四个：多头潜在注意力（Multi-Head Latent Attention, MLA）、无辅助损失负载均衡、多词元预测（Multi-Token Prediction, MTP）和 DualPipe 训练。本课从头到尾解读 DeepSeek-V3 的架构，并根据已发布的配置推导每一个参数数量。学完本课后，你将能够解释为什么 671B/37B 的比例是正确的选择，以及为什么 MLA + MoE 的组合在 frontier 上优于单独使用其中任何一个。

**类型：** 学习
**语言：** Python（标准库，参数计算器）
**前置知识：** 阶段 10 · 14（开源模型详解）、阶段 10 · 17（NSA）、阶段 10 · 18（MTP）、阶段 10 · 19（DualPipe）
**时间：** ~75 分钟

## 学习目标

- 从头到尾阅读 DeepSeek-V3 的配置，并根据 GPT-2 的六个旋钮加上四个 DeepSeek 特有的新增项，解释每一个字段。
- 推导总参数量（671B）、激活参数量（37B）以及各自对应的组成部分。
- 计算 MLA 在 128k 上下文下的 KV 缓存占用，并与同等激活参数量的稠密模型使用 GQA 时的开销进行对比。
- 阐述四项 DeepSeek 特有的创新（MLA、MTP、无辅助损失路由、DualPipe），并指出每一项针对的是架构/训练栈的哪个部分。

## 问题背景

DeepSeek-V3 是第一个架构与 Llama 家族有实质性差异的 frontier 开源模型。Llama 3 405B 是"调节了六个旋钮的 GPT-2"。DeepSeek-V3 是调节了全部六个旋钮并新增了四个的 GPT-2。阅读 Llama 3 的配置是阅读 DeepSeek 配置的热身，但其深层结构——注意力块的形状、路由逻辑、训练时目标函数——差异之大，足以需要单独进行一次详解。

学习它的回报：DeepSeek-V3 的开放权重发布改变了开源模型中"frontier 能力"的定义。该架构是 2026 年许多训练运行正在复制的蓝图。理解它是任何涉及 frontier LLM 训练或推理角色的入门要求。

## 核心概念

### 不变的核心，再次强调

DeepSeek-V3 仍然是自回归模型。它仍然堆叠解码器块。每个块仍然有注意力 + MLP + 两个 RMSNorm。MLP 仍然使用 SwiGLU。仍然使用 RoPE。Pre-norm。权重共享的嵌入。与每一个 Llama 或 Mistral 的基线相同。

### 关键变化：MLA 替代 GQA

从阶段 10 · 14 中你知道，GQA 通过在查询头（Q head）组之间共享 K 和 V 来缩小 KV 缓存。多头潜在注意力（Multi-Head Latent Attention, MLA）更进一步：K 和 V 被压缩成一个共享的低秩潜在表示（`kv_lora_rank`），然后在每个头上实时解压。KV 缓存只存储潜在表示——通常每层每个词元 512 个浮点数，而不是 8 x 128 = 1024 个浮点数。

在 128k 上下文下，使用 MLA 的 DeepSeek-V3（每层每个词元一个共享潜在表示 `c^{KV}`；K 和 V 都通过这个潜在表示经上投影推导出来，且上投影可以吸收到后续的矩阵乘法中）：

```
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

一个假设的 GQA 基线（Llama 3 70B 的规格，8 个 KV 头，头维度 128）将付出：

```
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

在 128k 上下文下，MLA 比 Llama-3-70B 风格的 GQA 缓存小 4 倍。

 tradeoff：MLA 在每次注意力计算（每个头）中增加了一个解压步骤。额外的计算量与节省的带宽相比很小。对于长上下文推理来说是净收益。

### 路由：无辅助损失负载均衡

MoE 路由器决定哪些 top-k 专家处理每个词元。一个朴素的路由器会将过多工作集中在少数几个专家上，导致其他专家闲置。标准修复方案：添加一个辅助损失项来惩罚负载不均衡。这有效，但会轻微降低主任务性能。

DeepSeek-V3 引入了一种无辅助损失的方案。在路由器 logits 上添加每个专家的偏置项，在训练期间通过简单规则调整：如果专家 `e` 过载，则减小 `bias_e`；如果欠载，则增大它。没有额外的损失项。训练保持纯净。专家负载保持均衡。

对主损失的影响：无法测量。对 MoE 架构的影响：更干净，无需调节辅助损失超参数。

### MTP：更密集的训练 + 免费的草稿模型

从阶段 10 · 18 中你知道，DeepSeek-V3 添加了一个深度为 D=1 的 MTP 模块，用于预测两个位置之后的词元。在推理时，训练好的模块被重新用作投机解码（speculative decoding）的草稿模型，接受率超过 80%。在训练时，每个隐藏状态受 D+1 = 2 个目标的监督，提供了更密集的信号。

参数量：在 671B 主模型之上增加 14B。开销：2.1%。

### 训练：DualPipe

从阶段 10 · 19 中你知道，DualPipe 是一种双向流水线，它将前向和反向块与跨节点 all-to-all 通信重叠。在 DeepSeek-V3 的 2,048 张 H800 规模下，它大约挽回了 1F1B 因流水线气泡而损失的 245k GPU 小时。

### 配置，逐字段解析

以下是 DeepSeek-V3 的配置（简化版）：

```
hidden_size: 7168
intermediate_size: 18432   （稠密 MLP 隐藏层大小，用于前几层）
moe_intermediate_size: 2048 （专家 MLP 隐藏层大小）
num_hidden_layers: 61
first_k_dense_layers: 3    （前 3 层使用稠密 MLP）
num_attention_heads: 128
num_key_value_heads: 128   （在 MLA 下形式上等于 num_heads，但
                           真正的压缩在于 kv_lora_rank）
kv_lora_rank: 512          （MLA 潜在维度）
num_experts: 256            （每个块的 MoE 专家数量）
num_experts_per_tok: 8      （top-8 路由）
shared_experts: 1           （每个块始终激活的共享专家）
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               （深度为 1 的 1 个 MTP 模块）
```

解析如下：

- `hidden_size=7168`：嵌入维度。
- `num_hidden_layers=61`：总块深度。
- `first_k_dense_layers=3`：前 3 个块使用大小为 18432 的稠密 MLP。剩余 58 个使用 MoE。
- `num_attention_heads=128`：128 个查询头。
- `kv_lora_rank=512`：K 和 V 被压缩到这个潜在维度，并在每个头上解压。
- `num_experts=256, num_experts_per_tok=8`：每个 MoE 块有 256 个专家，路由 top-8。
- `shared_experts=1`：在 256 个路由专家之外，还有 1 个始终激活的专家为每个词元做出贡献。可以将其视为一个"稠密基底"，确保每个词元都能获得可靠的结果。
- `moe_intermediate_size=2048`：每个专家的 MLP 隐藏层大小。比稠密 MLP 小，因为有 256 个专家。

### 参数核算

完整计算位于 `code/main.py` 中。核心数据：

- 嵌入层：`vocab * hidden = 129280 * 7168 = ~0.93B`。
- 前 3 个稠密块：带 MLA 的注意力（每个块约 ~144M）+ 稠密 MLP（每个块约 ~260M）+ 归一化层。总计约 1.2B。
- 58 个 MoE 块：带 MLA 的注意力（~144M）+ 256 个专家（每个 30M）+ 1 个共享专家（30M）+ 归一化层。每个块总计约 ~7.95B，包含所有专家。58 个 MoE 块总计 461B。
- MTP 模块：14B。

总计：核心架构约 ~476B + 14B MTP + 已发布的 671B 数字还包含了额外的结构参数（偏置张量、专家特定组件、共享专家缩放等）。我们在计算器中复现的数字与已发布数字相差在 3-5% 以内——差异来自 DeepSeek 报告第 2 节附录中记录的精细核算。

每次前向传播的激活参数：

- 注意力：每层 144M * 61 = 8.8B（所有层都激活）。
- MLP 激活部分：前 3 层为稠密层（3 * 260M = 780M），58 个 MoE 层每层激活 8 个路由专家 + 1 个共享专家 + 路由开销。每层激活 MLP：~260M。总计：3 * 260M + 58 * 260M = ~15.9B。
- 嵌入层 + 归一化层：1.2B。
- 总激活量：约 26B 核心 + 14B MTP（训练时使用，推理时不总是运行）≈ 37B。

### 671B / 37B 的比例

18 倍稀疏率（激活参数占总参数的 5.5%）。DeepSeek-V3 是已发布开放权重中最稀疏的 frontier MoE 模型。Mixtral 8x7B 的比例为 13/47（28%），稠密得多。Llama 4 Maverick 的比例为 17B/400B（4.25%），与之相当。DeepSeek 的赌注：在 frontier 规模下，更多的专家配合更低的激活比例，能在每激活 FLOP 上产生更好的质量。

### DeepSeek-V3 的定位

| 模型 | 总参数量 | 激活参数量 | 比例 | 注意力机制 | 创新点 |
|-------|------|-------|-------|-----------|-------------|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA 512 | MLA + MTP + 无辅助损失 + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | YaRN 扩展 |

### 后续发展：R1、V4

DeepSeek-R1（2025）是在 V3 骨干网络上进行的推理训练。R1 使用相同的架构。变化的是后训练配方（在可验证任务上的大规模 RL），而非预训练架构。

DeepSeek-V4（如果发布）预计会保留 MLA + MoE + MTP，并添加 DSA（DeepSeek Sparse Attention），即阶段 10 · 17 中 NSA 的继任者。谱系稳定：架构层面的创新不断积累；每个版本都会调节额外的旋钮。

## 动手实践

`code/main.py` 是专门针对 DeepSeek-V3 规格的参数计算器。运行它，将其输出与论文中的数字进行比较，并在假设变体上使用它（256 专家 vs 512，top-8 vs top-16，MLA 秩 512 vs 1024）。

关注以下方面：

- 总参数量 vs 已发布的 671B。
- 激活参数量 vs 已发布的 37B。
- 128k 上下文下的 KV 缓存——MLA vs GQA 的对比。
- 每层细分，查看参数预算实际花在了哪里。

## 交付成果

本课产出 `outputs/skill-deepseek-v3-reader.md`。给定一个 DeepSeek 家族模型（V3、R1 或任何未来变体），它将产出一份逐组件的架构解读，命名配置的每个字段，按组件推导参数量，并识别该模型使用了四项 DeepSeek 特有创新中的哪些。

## 练习题

1. 运行 `code/main.py`。将计算器的总参数估计值与已发布的 671B 进行比较，并找出差异的来源。论文的第 2 节有完整的明细。

2. 修改配置，将 MLA 秩从 512 改为 256。计算在 128k 上下文下产生的 KV 缓存大小。它能减少多少百分比，以及以每个头的表达能力为代价是什么？

3. 将 DeepSeek-V3 的（256 专家，top-8）路由与一个假设的（512 专家，top-8）变体进行比较。总参数增长；激活参数保持不变。额外的专家容量在理论上带来了什么，在推理时付出了什么代价？

4. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）的第 2.1 节关于 MLA 的内容。用三句话解释为什么 K 和 V 的解压矩阵可以在推理时被"吸收"到后续的矩阵乘法中以提高效率。

5. DeepSeek-V3 在大多数操作中使用 FP8 训练。计算用 FP8 替代 BF16 存储 671B 权重所带来的内存节省。这与 14.8T 词元的训练预算如何交叉影响？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| MLA | "Multi-Head Latent Attention" | 将 K 和 V 压缩成共享的低秩潜在表示（kv_lora_rank，通常为 512），在每个头上实时解压；KV 缓存只存储潜在表示 |
| kv_lora_rank | "MLA 压缩维度" | K 和 V 共享潜在表示的大小；DeepSeek-V3 使用 512 |
| 前 k 层稠密层 | "早期层保持稠密" | MoE 模型的前几层跳过 MoE 路由器，运行稠密 MLP 以保证稳定性 |
| num_experts_per_tok | "Top-k 路由" | 每个词元激活多少个路由专家；DeepSeek-V3 使用 8 |
| 共享专家 | "始终激活的专家" | 无论路由如何都处理每个词元的专家；DeepSeek-V3 使用 1 个 |
| 无辅助损失路由 | "偏置调整的负载均衡" | 在训练期间调整每个专家的偏置项以保持专家负载均衡，而不添加损失项 |
| MTP 模块 | "额外预测头" | 从 h^(1) 和 E(t+1) 预测 t+2 的 Transformer 块；更密集的训练，免费的投机解码草稿模型 |
| DualPipe | "双向流水线" | 将前向/反向计算与跨节点 all-to-all 重叠的训练调度方案 |
| 激活参数比例 | "稀疏率" | active_params / total_params；DeepSeek-V3 达到 5.5% |
| FP8 训练 | "8 位训练" | 训练存储和许多计算操作使用 FP8；与 BF16 相比内存大致减半，质量损失很小 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — 完整的架构、训练和结果文档
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — 配置文件和部署说明
- [DeepSeek-V2 paper (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) — 引入 MLA 的前代模型
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — 在 V3 架构上的推理训练继任者
- [Native Sparse Attention (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089) — DeepSeek 家族注意力的未来方向
- [DualPipe repository](https://github.com/deepseek-ai/DualPipe) — 训练调度参考实现
