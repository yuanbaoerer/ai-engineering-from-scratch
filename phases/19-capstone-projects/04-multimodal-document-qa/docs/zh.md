# Capstone 04 — 多模态文档问答（视觉优先 PDF、表格、图表）

> 2026 年，文档问答的前沿从 OCR 后转文本转向了视觉优先的晚期交互。ColPali、ColQwen2.5 和 ColQwen3-omni 将每个 PDF 页面视为图像，使用多向量晚期交互进行嵌入，让查询直接关注图像块。在金融 10-K 报告、科学论文和手写笔记上，这种模式大幅超越了 OCR 优先的方法。在 10k 页上构建端到端管道，并发布与 OCR 优先方法的对比结果。

**类型：** 毕业项目  
**语言：** Python（管道），TypeScript（查看器 UI）  
**前置要求：** 第 4 阶段（计算机视觉），第 5 阶段（NLP），第 7 阶段（transformers），第 11 阶段（LLM 工程），第 12 阶段（多模态），第 17 阶段（基础设施）  
**涉及阶段：** P4 · P5 · P7 · P11 · P12 · P17  
**时间：** 30 小时

## 问题

企业坐拥大量被 OCR 管道损坏的 PDF：扫描的 10-K 报告中表格旋转，科学论文充满公式，图表只有作为图像才有意义，还有手写注释。将这些视为文本优先意味着丢失一半信号。2026 年的解决方案是在原始页面图像上进行晚期交互多向量检索。ColPali（Illuin Tech）引入了这种方法；ColQwen2.5-v0.2 和 ColQwen3-omni 提升了准确性。在 ViDoRe v3 上，视觉优先检索的分数明显高于 OCR 优先方法——而且在图表、表格和手写体上差距更大。

代价是存储和延迟。ColQwen 嵌入每页约 2048 个图像块向量，而不是单个 1024 维向量。原始存储急剧膨胀。DocPruner（2026）实现了 50% 的修剪，且没有可测量的准确性损失。你将索引 10k 页，测量 ViDoRe v3 nDCG@5，提供 2 秒内的答案，并直接与 OCR 优先基线进行比较。

## 概念

晚期交互意味着每个查询 token 与每个图像块 token 进行评分，每个查询 token 的最大得分被求和。你无需单个池化向量就能获得细粒度匹配。多向量索引（Vespa、Qdrant 多向量或 AstraDB）存储每个图像块的嵌入，并在检索时运行 MaxSim。

回答者是一个视觉语言模型，它接收查询以及作为图像的 top-k 检索页面，并输出带有证据区域（边界框或页面引用）的答案。Qwen3-VL-30B、Gemini 2.5 Pro 和 InternVL3 是 2026 年的前沿选择。对于方程和科学符号，OCR 回退（Nougat、dots.ocr）作为可选文本通道被拼接进来。

评估是一个二维矩阵。一个轴：内容类型（纯文字段落、密集表格、条形/折线图、手写笔记、方程）。另一个轴：检索方法（视觉优先晚期交互 vs OCR 优先 vs 混合）。每个单元格获得 nDCG@5 和答案准确性。报告就是交付物。

## 架构

```
PDF -> 页面渲染器 (PyMuPDF, 180 DPI)
         |
         v
  ColQwen2.5-v0.2 嵌入 (每页多向量，约 2048 个图像块)
         |
         +------> DocPruner 50% 压缩
         |
         v
   多向量索引 (Vespa 或 Qdrant 多向量)
         |
查询 ----+----> 检索 top-k 页面 (MaxSim)
         |
         v
  VLM 回答者: Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    输入: 查询 + top-k 页面图像 + 可选 OCR 文本
         |
         v
  带引用页码 + 证据区域的答案
         |
         v
  Streamlit / Next.js 查看器: 在源页面上高亮显示框
```

## 技术栈

- 页面渲染：PyMuPDF (fitz) 180 DPI，纵向标准化
- 晚期交互模型：ColQwen2.5-v0.2 或 ColQwen3-omni（Hugging Face 上的 vidore 团队）
- 索引：Vespa 多向量字段，或 Qdrant 多向量，或 AstraDB 带 MaxSim
- 修剪：DocPruner 2026 策略（保留高方差图像块，50% 压缩，准确性损失 < 0.5%）
- OCR 回退（方程 / 密集表格）：dots.ocr 或 Nougat
- VLM 回答者：Qwen3-VL-30B 自托管或 Gemini 2.5 Pro 托管；InternVL3 作为回退
- 评估：ViDoRe v3 基准，M3DocVQA 用于多页推理
- 查看器 UI：Next.js 15 加画布覆盖层用于证据区域

## 构建它

1. **摄取。** 遍历包含 10-K 报告、科学论文和扫描文档的 10k PDF 页面语料库。将每页渲染为 1536x2048 PNG。保存 `{doc_id, page_num, image_path}`。

2. **嵌入。** 在每页图像上运行 ColQwen2.5-v0.2。输出形状约 2048 个 128 维的图像块嵌入。应用 DocPruner 保留信号最强的一半。写入 Vespa 多向量字段或 Qdrant 多向量。

3. **查询。** 对于每个传入查询，使用查询塔（token 级嵌入）进行嵌入。对索引运行 MaxSim：对于每个查询 token，取页面图像块嵌入的最大点积，求和。返回 top-k 页面。

4. **合成。** 使用查询和 top-5 页面图像调用 Qwen3-VL-30B。提示："仅使用提供的页面回答。通过 (doc_id, page) 引用每个声明，并命名区域（图表、表格、段落）。"

5. **证据区域。** 后处理答案以提取引用的区域。如果 VLM 发出边界框（Qwen3-VL 会），则在查看器中将它们渲染为覆盖层。

6. **OCR 回退。** 对于被识别为公式密集的页面（基于图像方差的启发式），运行 Nougat 或 dots.ocr，并将 OCR 文本作为额外通道与图像一起传递。

7. **评估。** 运行 ViDoRe v3（检索 nDCG@5）和 M3DocVQA（多页 QA 准确性）。在相同语料库上使用相同合成器运行 OCR 优先管道。生成内容类型 × 方法矩阵。

8. **UI。** 先做 Streamlit 原型；然后是 Next.js 15 生产查看器，带有逐页证据区域覆盖层。

## 使用它

```
$ doc-qa ask "EMEA 部门 2024 年营业利润率变化是多少？"
[retrieve]   top-5 页面在 320ms 内检索完成 (ColQwen2.5, MaxSim, Vespa)
[synth]      qwen3-vl-30b, 1.4s, 引用 (form-10k-2024, p. 88) + (..., p. 92)
答案：
  EMEA 营业利润率从 18.2% 下降到 16.8%，下降了 140 个基点。
  引用：10-K-2024.pdf p.88 (表 4，部门营业利润率)
         10-K-2024.pdf p.92 (MD&A，营业表现)
[viewer]     打开并在 p.88 表 4 上叠加高亮边界框
```

## 交付它

`outputs/skill-doc-qa.md` 描述了交付物：一个视觉优先的多模态文档问答系统，针对特定语料库进行调优，并在 ViDoRe v3 上与 OCR 优先基线进行评估。

| 权重 | 标准 | 衡量方式 |
|:---:|------|----------|
| 25 | ViDoRe v3 / M3DocVQA 准确性 | 基准数字 vs OCR 文本基线和已发布排行榜 |
| 20 | 证据区域定位 | 引用区域实际包含答案区间的比例 |
| 20 | 存储和延迟工程 | DocPruner 压缩比，索引 p95，答案 p95 |
| 20 | 多页推理 | 手工标注的 100 题多页集上的准确性 |
| 15 | 源检查 UX | 查看器清晰度，覆盖层保真度，并排比较工具 |
| **100** | | |

## 练习

1. 在相同语料库上测量 ColQwen2.5-v0.2 与 ColQwen3-omni 的对比。一个正确而另一个遗漏的页面是哪些？在索引中添加"内容类别"标签以按类型路由。

2. 激进修剪嵌入（75%、90%）。找到压缩悬崖：ViDoRe nDCG@5 低于 OCR 基线的临界点。

3. 构建混合方案：并行运行 OCR 优先和 ColQwen，使用 RRF 融合，用交叉编码器重排序。混合方案是否优于任一单独方案？在哪里帮助最大？

4. 将 Qwen3-VL-30B 替换为更小的 VLM（Qwen2.5-VL-7B）。测量准确性-成本曲线。

5. 添加手写笔记支持。渲染手写语料库，用 ColQwen 嵌入，测量检索效果。与手写 OCR 管道进行比较。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 晚期交互 | "ColPali 风格检索" | 查询 token 独立地与页面图像块评分；MaxSim 聚合 |
| 多向量 | "每图像块嵌入" | 每个文档有多个向量，而不是一个池化向量 |
| MaxSim | "晚期交互评分" | 对于每个查询 token，取文档向量的最大相似度；求和 |
| DocPruner | "图像块压缩" | 2026 年的修剪方法，保留 50% 的图像块，准确性损失可忽略 |
| ViDoRe v3 | "文档检索基准" | 2026 年衡量视觉文档检索的标准 |
| 证据区域 | "引用的边界框" | 源页面上定位答案区间的边界框 |
| OCR 回退 | "方程通道" | 与视觉一起用于公式或表格密集页面的文本管道 |

## 延伸阅读

- [ColPali (Illuin Tech) 仓库](https://github.com/illuin-tech/colpali) — 参考晚期交互文档检索
- [ColPali 论文 (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449) — 基础方法论文
- [Hugging Face 上的 ColQwen 系列](https://huggingface.co/vidore) — 生产就绪的检查点
- [M3DocRAG (Adobe)](https://arxiv.org/abs/2411.04952) — 多页多模态 RAG 基线
- [Vespa 多向量教程](https://docs.vespa.ai/en/colpali.html) — 参考服务堆栈
- [Qdrant 多向量支持](https://qdrant.tech/documentation/concepts/vectors/#multivectors) — 备选索引
- [AstraDB 多向量](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) — 备选托管索引
- [Nougat OCR](https://github.com/facebookresearch/nougat) — 支持方程的 OCR 回退