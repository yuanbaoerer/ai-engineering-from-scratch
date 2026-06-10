# 百万 Token 上下文的长视频理解

> 一段 1 小时的 4K 视频，以 24 FPS 进行 patch 和嵌入，会产生约 6000 万 token。一期 2 小时的播客转录约 30,000 token。一部完整的蓝光电影，即使经过激进的池化压缩，也有数十万 token。Google 的 Gemini 1.5（2024 年 3 月）开启了这个时代，拥有 1000 万 token 的上下文，能够对长达一小时的视频进行可靠的"大海捞针"式召回。LWM（Liu 等人，2024 年 2 月）展示了 ring attention 的扩展路径。LongVILA 和 Video-XL 进一步扩展了视频摄取能力。VideoAgent 用 agentic retrieval 替代了原始上下文。每种方法在计算成本、召回率和工程复杂度之间都有不同的权衡。本课将并置对比这些方法。

**类型：** 构建
**语言：** Python（标准库，大海捞针模拟器 + agentic-retrieval 路由器）
**前置条件：** Phase 12 · 17（视频时序 token）
**时间：** 约 180 分钟

## 学习目标

- 计算不同 FPS 和池化策略下长视频的总视觉 token 数量。
- 解释三种扩展路径：暴力上下文（Gemini 1.5）、ring attention（LWM）、token 压缩（LongVILA / Video-XL）。
- 对比原始上下文视频 VLM 与 agentic-retrieval 视频 VLM（VideoAgent）在准确率和延迟上的差异。
- 为一段 30 分钟的视频设计大海捞针测试，并测量在特定分钟处的召回率。

## 问题所在

一帧 Qwen2.5-VL 大小的 patch，在 384 原生分辨率下约为 729 个 token。经过 3x3 池化后，每帧 81 个 token。一段 30 分钟的片段，以 1 FPS 计算 = 1800 帧 = 145,800 个 token。2025 年的开源 VLM 可以处理，但比较紧张。以 2 FPS 计算，291,600 个 token——只有最大的上下文才能容纳。

一部 2 小时的电影，以 1 FPS 计算是 58.3 万 token。超出了大多数 2026 年开源模型的能力范围；需要 Gemini 2.5 Pro 或更激进的池化。

三种扩展路径应运而生。

## 核心概念

### 路径 1：暴力上下文（Gemini 1.5、Claude Opus）

用硬件解决问题。将上下文扩展到数百万 token，一次性前向传播处理所有内容。

Gemini 1.5 Pro 发布时支持 100 万 token；Gemini 1.5 Ultra 达到 1000 万；2026 年的 Gemini 2.5 Pro 能够可靠处理数小时的视频。论文（arXiv:2403.05530）记录了在约 950 万 token 范围内，大海捞针召回率达到 99.7%。

工程实现：自定义 attention 实现，具有内存层级结构（局部 + 全局 + 稀疏），以及用于长上下文效率的 MoE expert routing。未完全公开细节。非开源。

### 路径 2：Ring Attention（LWM、LongVILA）

Ring attention 将长序列以"环形"方式分布在多个设备上，每个设备持有一个块。跨完整序列的 attention 通过每个设备将其块以环形模式发送给下一个设备，计算部分 attention 并聚合来实现。

LWM（Liu 等人，2024）用这种方式训练了一个支持 100 万 token 上下文的模型。训练计算量随上下文长度线性扩展，而非二次方——attention 上的二次方开销被分摊到环形中的各个设备上。

LongVILA（arXiv:2408.10188）将这一模式适配到 VLM。1400 帧视频，每帧 192 个 token = 26.8 万上下文，通过 8 路并行的 ring attention 进行训练。

### 路径 3：Token 压缩（Video-XL、LongVA）

比暴力上下文更便宜：在 LLM 看到序列之前进行激进的压缩。

Video-XL（arXiv:2409.14485）使用视觉摘要 token：每段包含 N 帧的片段产生一个"摘要"token，该 token 关注这 N 帧。在推理时，LLM 每段只看到一个摘要 token，极大地缩减了上下文。

LongVA 通过"长上下文迁移"技术将 LLM 上下文从 20 万扩展到 200 万。先在长上下文文本上训练，再通过共享表示迁移到长上下文视频。

Token 压缩以牺牲特定时间戳的召回率为代价换取可扩展性。模型大致知道发生了什么，但有时会错过确切的帧。

### 路径 4：Agentic Retrieval（VideoAgent）

不将完整视频输入 LLM。而是将视频视为数据库，用 LLM 来查询它。

VideoAgent（arXiv:2403.10517）：

1. LLM 读取问题。
2. LLM 向检索工具请求相关片段（"给我展示有猫的片段"）。
3. 工具返回匹配片段的时间戳。
4. LLM 通过 VLM 读取这些片段。
5. LLM 组合答案或提出后续查询。

这是将 LLM-as-agent 模式应用于长视频。推理更便宜（只编码相关片段），但工程更复杂（检索质量成为瓶颈）。

### 大海捞针基准测试

标准的长上下文测试：在视频的随机位置插入一个独特的视觉或文本标记，然后提出一个需要召回它的问题。

指标：跨视频长度和标记位置的 Recall@k。

Gemini 2.5 Pro 在长达 90 分钟的视频中召回率超过 99%。开源 72B 模型（Qwen2.5-VL-72B、InternVL3-78B）在 30 分钟时约为 85-90%，超过 60 分钟后下降。

VideoAgent 在 2 小时以上的视频中可以匹敌或超越原始上下文模型，因为如果工具效果好，检索就能命中目标。

### 选择哪种路径

对于 15 分钟的片段，追求前沿准确率：开源 72B + 原生上下文通常可行。选择 Qwen2.5-VL-72B。

对于 30 分钟到 1 小时的内容：开源选 LongVILA 或 Video-XL；闭源选 Gemini 2.5 Pro。质量门槛很重要——前沿方案选择闭源。

对于 2 小时以上的内容：VideoAgent 或类似的检索模式。或者，总结成更小的片段并输入分层摘要。

### 2026 年生产实践模式

实际上，生产环境中的长视频流水线是混合式的：

1. 对整个视频运行动态 FPS 采样 + 激进池化（获得一个 10 万 token 的全局表示）。
2. 将其输入 72B VLM 生成全局摘要。
3. 如果用户提出详细问题，使用摘要作为索引运行 agentic retrieval。

这结合了暴力上下文用于全局理解和检索用于局部细节。

## 动手实践

`code/main.py`：

- 计算从 1 分钟到 3 小时的视频在不同 FPS + 池化策略下的 token 预算。
- 模拟一次大海捞针运行：在随机时间戳注入标记，提出问题，评分召回率。
- 包含一个 agentic-retrieval 路由器模拟器，选择特定片段输入下游 VLM。

运行预算表，感受规模差距。

## 交付成果

本课产出 `outputs/skill-long-video-strategy-planner.md`。给定视频时长和查询复杂度，它在暴力上下文、压缩和 agentic retrieval 之间做出选择，并计算延迟 + 质量预期。

## 练习题

1. 一段 45 分钟的讲座，1 FPS，每帧 81 个 token。总 token 数是多少？能放入哪些模型的上下文？

2. 设计一个大海捞针测试：你在第几分钟注入标记，确切的查询格式是什么？

3. 在 1 小时视频上对比暴力上下文 Qwen2.5-VL-72B（8 万上下文）与 VideoAgent（Claude 3.5 + retrieval）。哪个在召回率上获胜？哪个在延迟上获胜？

4. Ring attention 的内存成本随序列长度线性扩展，随设备数量线性扩展。解释原因，以及如果去掉 ring-rotation 阶段会发生什么。

5. 阅读 Gemini 1.5 论文第 5 节关于大海捞针的内容。论文在 100 万 vs 1000 万 token 边界上发现了什么关于召回率的结果？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|---------|
| Brute context | "更多 token 就行" | 将 LLM 上下文扩展到数百万 token；一次性处理所有内容 |
| Ring attention | "LWM 风格的并行" | 分布式 attention 模式，每个设备持有一个块并轮转 |
| Token compression | "摘要 token" | 在 LLM 之前通过学习的压缩器减少每段 token |
| Needle-in-haystack | "NIH 测试" | 在随机位置插入独特标记，测试时要求模型召回它 |
| Agentic retrieval | "LLM 作为查询规划器" | LLM 向检索工具请求相关片段，通过 VLM 读取，组合答案 |
| VideoAgent | "视频的检索模式" | 经典的 agentic-retrieval 设计：问题 -> 工具 -> 片段 -> 答案 |

## 延伸阅读

- [Gemini Team — Gemini 1.5 (arXiv:2403.05530)](https://arxiv.org/abs/2403.05530)
- [Liu et al. — LWM / RingAttention (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Xue et al. — LongVILA (arXiv:2408.10188)](https://arxiv.org/abs/2408.10188)
- [Shu et al. — Video-XL (arXiv:2409.14485)](https://arxiv.org/abs/2409.14485)
- [Wang et al. — VideoAgent (arXiv:2403.10517)](https://arxiv.org/abs/2403.10517)
