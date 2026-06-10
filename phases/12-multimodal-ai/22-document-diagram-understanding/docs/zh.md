# 文档与图表理解

> 文档不是照片。PDF、科学论文、发票或手写表格具有布局、表格、图表、脚注、页眉以及语义结构，这些都无法通过纯图像理解来捕获。VLM 之前的栈是一个流水线：Tesseract OCR + LayoutLMv3 + 表格提取启发式规则。VLM 浪潮用 OCR-free 模型取代了它——Donut（2022）、Nougat（2023）、DocLLM（2023）——直接输出结构化标记。到 2026 年，前沿方案就是"将页面图像以 2576px 原生分辨率喂给 Claude Opus 4.7"，结构化标记输出唾手可得。本课讲述文档 AI 的三个时代。

**类型：** 构建
**语言：** Python（标准库，布局感知文档解析器骨架）
**前置条件：** Phase 12 · 05（LLaVA），Phase 5（NLP）
**时间：** 约 180 分钟

## 学习目标

- 解释文档 AI 的三个时代：OCR pipeline、OCR-free、VLM-native。
- 描述 LayoutLMv3 的三个输入流：text、layout（bbox）、image patches，以及统一的 masking 机制。
- 比较 Donut（OCR-free，image → markup）、Nougat（scientific paper → LaTeX）、DocLLM（layout-aware generative）、PaliGemma 2（VLM-native）。
- 为新任务（发票、科学论文、手写表格、中文收据）选择合适的文档模型。

## 问题所在

"理解这个 PDF"说起来容易做起来难。信息分布在：

- 文本内容（90% 的信号）。
- 布局（页眉、脚注、侧边栏、双栏格式）。
- 表格（行、列、合并单元格）。
- 图形和图表。
- 手写批注。
- 字体和排版（标题 vs 正文）。

原始 OCR 只转储文本，其余全部丢失。一个关注发票的系统需要知道"Total: $1,245"来自右下角，而不是脚注。

## 核心概念

### 时代 1 — OCR pipeline（2021 年之前）

经典栈：

1. PDF → 每页一张图像。
2. Tesseract（或商业 OCR）提取文本并附带每个词的 bounding box。
3. Layout analyzer 识别块（页眉、表格、段落）。
4. Table structure recognizer 解析表格。
5. Domain rules + regex 提取字段。

对干净印刷文本有效。在手写、歪斜扫描、复杂表格、非英文脚本上失效。每种失败模式都需要自定义异常路径。

### TrOCR（2021）

TrOCR（Li et al., arXiv:2109.10282）用基于 transformer encoder-decoder 的模型替代了 Tesseract 的经典 CNN-CTC，该模型在合成 + 真实文本图像上训练。在手写和多语言文本上取得显著进步。仍然是一个流水线（detector → TrOCR → layout），但 OCR 步骤大幅改进。

### 时代 2 — OCR-free（2022-2023）

第一批 OCR-free 模型说：完全跳过检测，直接将图像像素映射到结构化输出。

Donut（Kim et al., arXiv:2111.15664）：
- Encoder-decoder transformer，encoder 为 Swin-B。
- 输出为用于 form understanding 的 JSON、用于 summarization 的 markdown，或任何任务特定的 schema。
- 没有 OCR，没有 layout，没有 detection。

Nougat（Blecher et al., arXiv:2308.13418）：
- 专门针对科学论文训练。
- 输出为 LaTeX / markdown。
- 处理公式、多栏布局、图形。
- 每个 arXiv 解析器都在调用的模型。

这些是专家模型，不是通才。Donut 在科学论文上失效；Nougat 在发票上失效。

### LayoutLMv3（2022）

另一条路线。LayoutLMv3（Huang et al., arXiv:2204.08387）保留 OCR 但增加了 layout 理解：

- 三个输入流：OCR text tokens、per-token 2D bounding boxes、image patches。
- 跨三种模态的 masked training objective（masked text、masked patches、masked layout）。
- 下游任务：classification、entity extraction、table QA。

LayoutLMv3 是 OCR-based document understanding 的巅峰。在表单和发票上表现强劲。需要上游 OCR。在标准化文档基准上达到 VLM 之前的最佳准确率。

### DocLLM（2023）

DocLLM（Wang et al., arXiv:2401.00908）是 LayoutLM 的生成式兄弟。基于 layout tokens 生成自由形式答案。更擅长文档 QA；仍然依赖 OCR 输入。

### 时代 3 — VLM-native（2024+）

2024 年，VLM 变得足够好，可以完全取代流水线。将完整页面图像以高分辨率喂给 VLM，提问，获得答案。

- LLaVA-NeXT 336-tile AnyRes 适用于小型文档。
- Qwen2.5-VL dynamic-resolution 原生支持 2048+ 像素。
- Claude Opus 4.7 支持 2576px 文档。
- PaliGemma 2（2025 年 4 月）专门针对文档 + 手写训练。

VLM-native 与 OCR-pipeline 之间的差距迅速缩小。到 2026 年，VLM-native 在以下方面胜出：

- Scene text（手写 + 印刷，混合脚本）。
- 带有合并单元格的复杂表格。
- 嵌入文本中的数学公式。
- 带有文本注释的图形。

OCR pipeline 仍在以下方面胜出：

- 大规模纯扫描工作负载，其中每页延迟至关重要。
- Pipeline 可靠性（确定性失败 vs VLM hallucinations）。
- 需要可审计 OCR 输出的监管环境。

### Claude 4.7 / GPT-5 前沿

在 2576 像素原生输入下，前沿 VLM 以接近人类的准确率进行文档理解。2026 年初的基准数据：

- DocVQA：Claude 4.7 ~95.1，PaliGemma 2 ~88.4，Nougat ~77.3，pipelined LayoutLMv3 ~83。
- ChartQA：Claude 4.7 ~92.2，GPT-4V ~78。
- VisualMRC：Claude 4.7 ~94。

闭源模型的差距主要在于分辨率和 base-LLM 规模。7B 开源模型落后几个百分点，但正在追赶。

### 数学公式与 LaTeX 输出

科学论文需要公式的精确 LaTeX 输出。Nougat 为此训练。以 LaTeX 为目标的 VLM（Qwen2.5-VL-Math、Nougat 衍生模型）可生成可用的 LaTeX。没有显式 LaTeX 训练的情况下，VLM 生成可读但不精确的转录。

对于 2026 年的科学论文流水线：先用 Nougat 处理 PDF，再用 VLM 处理棘手页面。

### 手写识别

仍然是最难的子任务。混合印刷 + 手写（医生笔记、填写表格）是 OCR pipeline 仍在成本上击败 VLM 的领域。纯手写 VLM 正在改进（Claude 4.7、PaliGemma 2）。

### 2026 年方案选型

对于新的文档 AI 项目：

- 大规模纯印刷发票：LayoutLMv3 + rules，成本高效。
- 混合文档（科学 + 手写 + 表单）：VLM-native（PaliGemma 2 或 Qwen2.5-VL）。
- 完整 arXiv 摄取：Nougat 处理数学，VLM 处理图形。
- 监管场景：OCR pipeline + VLM validator 进行交叉检查。

## 动手实践

`code/main.py`：

- 一个 toy layout-aware tokenizer：给定 (text, bbox) 对，生成 LayoutLMv3 风格的输入。
- 一个 Donut-style task schema generator：用于表单的 JSON 模板。
- 比较 OCR-pipeline、Donut、Nougat 和 VLM-native 每页的 token budgets。

## 交付成果

本课产出 `outputs/skill-document-ai-stack-picker.md`。给定一个文档 AI 项目（domain、scale、quality、regulatory），在 OCR pipeline、OCR-free specialist 和 VLM-native 之间做出选择。

## 练习题

1. 你的项目每天处理 1000 万张发票。哪种栈在不损失准确率的前提下最小化每页成本？

2. 为什么 LayoutLMv3 在 form QA 上优于纯 CLIP-VLM，但在 scene text 上表现不佳？bbox 流放弃了什么？

3. Nougat 生成 LaTeX。提出一个 VLM-native 输出在 LaTeX 保真度上击败 Nougat 的测试用例，以及一个 Nougat 获胜的用例。

4. 阅读 PaliGemma 2 论文（Google, 2024）。与 PaliGemma 1 相比，提升文档准确率的关键训练数据补充是什么？

5. 设计一个监管安全的混合方案：OCR pipeline 为主，VLM 为辅进行交叉检查。如何解决分歧？

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| OCR pipeline | "Tesseract-style" | 分阶段栈：detect → OCR → layout → rules；确定性，脆弱 |
| OCR-free | "Donut-style" | 跳过显式 OCR 的 image-to-output transformer；单一模型 |
| Layout-aware | "LayoutLM" | 输入包含 per-token bbox 坐标；跨模态统一 masking |
| VLM-native | "Frontier VLM" | 将页面图像直接喂给 Claude/GPT/Qwen VLM 高分辨率；无流水线 |
| DocVQA | "Doc benchmark" | Document VQA 标准；最常引用的分数 |
| Markup output | "LaTeX / MD" | 结构化输出格式而非自由文本；支持下游自动化 |

## 延伸阅读

- [Li et al. — TrOCR (arXiv:2109.10282)](https://arxiv.org/abs/2109.10282)
- [Blecher et al. — Nougat (arXiv:2308.13418)](https://arxiv.org/abs/2308.13418)
- [Huang et al. — LayoutLMv3 (arXiv:2204.08387)](https://arxiv.org/abs/2204.08387)
- [Kim et al. — Donut (arXiv:2111.15664)](https://arxiv.org/abs/2111.15664)
- [Wang et al. — DocLLM (arXiv:2401.00908)](https://arxiv.org/abs/2401.00908)
