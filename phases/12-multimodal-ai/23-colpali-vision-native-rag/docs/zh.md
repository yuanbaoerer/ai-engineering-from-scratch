# ColPali 与 Vision-Native 文档 RAG

> 传统的 RAG 将 PDF 解析为文本，分割成块，对块进行嵌入，存储向量。每一步都在丢失信号：OCR 丢弃图表数据，分块打断表格行，文本嵌入忽略图形。ColPali（Faysse 等人，2024年7月）提出了一个更简单的问题：为什么要提取文本？通过 PaliGemma 直接嵌入页面图像，使用 ColBERT 风格的 late interaction 进行检索，保留文档携带的所有布局、图形、字体和格式信号。已发布的基准测试显示：在视觉丰富的文档上，端到端准确率比文本 RAG 高 20-40%。ColQwen2、ColSmol 和 VisRAG 扩展了这一范式。本课程将解读 vision-native RAG 的核心思想，并构建一个微型 ColPali 风格的索引器。

**类型：** 构建
**语言：** Python（标准库，multi-vector 索引器 + MaxSim 评分器）
**前置知识：** Phase 11（LLM Engineering — RAG 基础），Phase 12 · 05（LLaVA）
**时间：** ~180 分钟

## 学习目标

- 解释 bi-encoder 检索（每文档一个向量）与 late-interaction 检索（每文档多个向量）之间的区别。
- 描述 ColBERT 的 MaxSim 操作，以及 ColPali 如何将其从文本 token 推广到图像 patch。
- 构建一个微型 ColPali 风格的索引器：页面 → patch 嵌入 → 基于 query-term 嵌入的 MaxSim → top-k 页面。
- 在发票 / 财务报告用例上，对比 ColPali + Qwen2.5-VL 生成器与文本 RAG + GPT-4 的效果。

## 问题所在

在 PDF 上的文本 RAG 丢弃了文档的大部分信息。财务报告的 Q3 收入增长通常在图表中；医学报告的发现结果在带注释的图像中；法律合同的签名块是一个布局事实，而非文本事实。

文本 RAG 流程：

1. PDF → 通过 OCR / pdftotext 提取文本。
2. 文本 → 300-500 token 的块。
3. 块 → bi-encoder 嵌入（一个向量）。
4. 用户查询 → 嵌入 → 余弦相似度 → top-k 块。
5. 块 + 查询 → LLM。

五个有损步骤。图表未被捕获。表格被分块打断。多栏布局被扁平化。图形注释消失。

ColPali 的解决方案：跳过 OCR，直接嵌入页面图像。使用 ColBERT 风格的 late interaction 进行检索，使模型可以在查询时关注到细粒度的 patch。

## 核心概念

### ColBERT（2020）

ColBERT（Khattab & Zaharia, arXiv:2004.12832）是一种文本检索方法。它不为每篇文档生成一个向量，而是为每个 token 生成一个向量。在查询时：

- 查询 token 获得各自的嵌入（N_q 个向量）。
- 文档 token 获得嵌入（N_d 个向量，通常被缓存）。
- 得分 = 对查询 token 求和，每个查询 token 取与文档 token 的最大余弦相似度：Σ_i max_j cos(q_i, d_j)。

这就是 MaxSim 操作。每个查询 token "挑选" 与其最匹配的文档 token。最终得分是总和。

优点：召回率高，能处理词级语义。缺点：每篇文档需要 N_d 个向量，存储开销大。

### ColPali

ColPali（Faysse 等人, arXiv:2407.01449）将 ColBERT 范式应用于图像。

- 每个页面由 PaliGemma（ViT + 语言模型）编码为 patch 嵌入：每页 N_p 个向量。
- 每个用户查询（文本）被编码为查询 token 嵌入：N_q 个向量。
- 得分 = Σ_i max_j cos(q_i, p_j)，即在查询文本 token 和页面图像 patch 之间进行 MaxSim。
- 按总分检索 top-k 页面。

在文档摄取阶段：用 PaliGemma 嵌入每一页，存储所有 patch 嵌入。在查询阶段：嵌入查询 token，对所有已索引页面计算 MaxSim，返回 top-k 页面。

优点：在视觉丰富的文档上，端到端效果比文本 RAG 高 20-40%。每个 patch 向量捕获局部布局和内容。

缺点：每页 N_p 个 patch × 4 字节浮点数 × D 维向量 = 存储快速增长。可通过 PQ / OPQ 量化来缓解。

### ColQwen2 与 ColSmol

ColQwen2（illuin-tech, 2024-2025）将 PaliGemma 替换为 Qwen2-VL。更好的基础编码器，更好的检索效果。

ColSmol 是面向本地 / 边缘设备的小型变体。约 1B 参数的 ColSmol 检索器可在消费级 GPU 上运行。

### VisRAG

VisRAG（Yu 等人, arXiv:2410.10594）是另一种变体：不是对 patch 做 MaxSim，而是将每页池化为单个向量，然后用 VLM 进行 bi-encoder 检索。索引更快、存储更小，但召回率较弱。

质量与成本的权衡：ColPali 追求质量，VisRAG 追求规模。

### M3DocRAG

M3DocRAG（Cho 等人, arXiv:2411.04952）将多模态检索扩展到多页多文档推理。跨文档检索页面，为多页上下文组合提供给 VLM。

### ViDoRe — 基准测试

ColPali 的配套基准测试。Visual Document Retrieval Evaluation（视觉文档检索评估）。任务包括财务报告、科学论文、行政文件、医疗记录、手册。指标：nDCG@5。

ColPali-v1 在 ViDoRe 上得分约 80% nDCG@5；同一份文档上的文本 RAG 得分约 50-60%。

### 端到端 RAG 流程

对于 vision-native RAG：

1. 摄取：PDF → 页面图像 → PaliGemma 编码 → 存储所有 patch 嵌入。
2. 查询：用户文本 → 查询 token 嵌入 → 对所有已索引页面做 MaxSim → top-k 页面。
3. 生成：top-k 页面图像 + 查询 → VLM（Qwen2.5-VL 或 Claude）→ 答案。

全程无需 OCR。图形、图表、字体、布局全部流入答案。

### 存储计算

一份 50 页的财务报告，每页 729 个 patch，128 维嵌入：

- ColPali：50 * 729 * 128 * 4 字节 = ~18 MB 原始数据，PQ 压缩后 ~4 MB。
- 文本 RAG：50 个块 * 768 维 * 4 字节 = ~150 kB。

ColPali 每份文档的存储量约是文本 RAG 的 30 倍。在大规模场景下，OPQ / PQ 可将其降低到约 5-10 倍，通常可以接受。

### 文本 RAG 仍然占优的场景

- 纯文本文档，没有布局信号（维基文章、聊天记录）。文本 RAG 更简单且存储成本更低。
- 数百万页规模的档案库，存储成本占主导。
- 严格的监管要求，需要可提取的 OCR 文本作为检索的补充。

对于 2026 年的其他所有场景 —— 财务报告、科学论文、法律合同、医疗记录、UX 文档 —— vision-native RAG 胜出。

## 动手实践

`code/main.py`：

- 微型 patch 编码器：将"页面"（小型特征向量网格）映射为 patch 嵌入数组。
- MaxSim 评分器：计算查询 token 嵌入集与页面 patch 集之间的 ColBERT 风格得分。
- 索引 5 个微型页面，运行 3 个查询，返回带得分的 top-k 结果。

## 交付成果

本课程产出 `outputs/skill-vision-rag-designer.md`。给定一个文档 RAG 项目，选择 ColPali / ColQwen2 / VisRAG / 文本 RAG 并估算存储规模。

## 练习题

1. 一份 200 页的年度报告，每页 729 个 patch，128 维嵌入，4 字节浮点数。计算原始存储量和 PQ 压缩后（8 倍）的存储量。

2. MaxSim 是 Σ_i max_j cos(q_i, p_j)。这个求和捕获了什么，是简单的平均相似度无法做到的？

3. ColPali 将页面索引为 patch 集合。如果我们改为在词级别索引（像 ColBERT 那样），会有什么变化？权衡是什么？

4. 为一个 100 万页规模的语料库设计端到端流程，查询延迟预算为 500ms。选择 ColQwen2 / VisRAG 并说明理由。

5. 阅读 M3DocRAG（arXiv:2411.04952）。描述其多页注意力模式，以及它与单页 ColPali 检索的区别。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Late interaction | "ColBERT-style" | 使用 per-token 或 per-patch 嵌入 + MaxSim 进行检索，而非单个文档向量 |
| MaxSim | "Max-over-patches" | 对每个查询 token，选取相似度最高的文档 token；对查询求和 |
| Bi-encoder | "Single-vector" | 每篇文档一个向量；更快但丢失细粒度 |
| Multi-vector | "Many-vectors-per-doc" | 每篇文档 / 页面存储 N_p 个向量；存储成本增长但召回率提升 |
| Patch embedding | "Page feature" | 来自 VLM 编码器的每个图像 patch 对应的向量，按页缓存 |
| ViDoRe | "Vision doc bench" | ColPali 的视觉文档检索基准测试套件 |
| PQ quantization | "Product quantization" | 在保持向量相似度的同时将存储压缩约 8 倍的压缩方法 |

## 延伸阅读

- [Faysse et al. — ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [Khattab & Zaharia — ColBERT (arXiv:2004.12832)](https://arxiv.org/abs/2004.12832)
- [Yu et al. — VisRAG (arXiv:2410.10594)](https://arxiv.org/abs/2410.10594)
- [Cho et al. — M3DocRAG (arXiv:2411.04952)](https://arxiv.org/abs/2411.04952)
- [illuin-tech/colpali GitHub](https://github.com/illuin-tech/colpali)
