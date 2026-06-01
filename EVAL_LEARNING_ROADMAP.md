# LLM 评测工程学习路线

> 本路线覆盖从评测基础概念到构建完整评测流水线的全部内容，共 9 课、4 个 Phase。按推荐顺序学习可获得最平滑的递进体验。

## 总览

| # | Phase | 课程 | 主题 | 类型 | 时间 |
|---|-------|------|------|------|------|
| 1 | 05 · NLP 基础 | L27 | RAGAS / DeepEval / G-Eval | 学习 | ~60 min |
| 2 | 05 · NLP 基础 | L28 | 长上下文评测（NIAH、RULER、LongBench） | 学习 | ~60 min |
| 3 | 10 · 从零构建 LLM | L10 | 基准测试、自定义评测、LM Harness | 构建 | ~90 min |
| 4 | 14 · 智能体工程 | L19 | SWE-bench / GAIA / AgentBench | 学习 | ~60 min |
| 5 | 14 · 智能体工程 | L20 | WebArena / OSWorld | 学习 | ~60 min |
| 6 | 14 · 智能体工程 | L30 | 评估驱动的智能体开发 | 学习 | ~60 min |
| 7 | 19 · 综合项目 | C27 | 基于 Fixture 任务的评测框架 | 构建 | ~90 min |
| 8 | 19 · 综合项目 | C41 | 完整评测流水线 | 构建 | ~90 min |
| 9 | 19 · 综合项目 | C49 | 语言模型评测框架（LM Eval Harness） | 构建 | ~90 min |

**总时长：** 约 11 小时

## 学习阶段划分

### 阶段一：评测概念入门（L27、L28）

建立评测领域的全景认知。了解有哪些现成的评测框架（RAGAS、DeepEval、G-Eval），以及当模型声称支持百万 token 上下文时如何验证真实能力。

**L27 — RAGAS / DeepEval / G-Eval**
- 三大框架的设计哲学：RAGAS 专注 RAG 流水线，DeepEval 提供工程友好的单元测试体验，G-Eval 用 GPT-4 自动打分
- 核心指标：faithfulness、relevance、answer correctness
- 适用场景选择

**L28 — 长上下文评测**
- NIAH（Needle in a Haystack）：在长文本中插入关键信息，测试模型能否找到
- RULER：四类合成任务，从"大海捞针"到多跳推理
- LongBench 和 MRCR：真实世界长文本任务
- 关键结论：厂商声称的上下文长度，实际可用范围通常是 60-70%

### 阶段二：评测体系构建（L10）

从概念转向动手。理解为什么基准测试会失效，掌握核心评测指标（perplexity、exact match、F1、BLEU、ELO），并学会构建自定义评测套件。

**L10 — 基准测试、自定义评测、LM Harness**
- 三类评测：基准测试（便宜、标准化、可刷）、自定义评测（高信号、构建成本高）、人类评估（黄金标准、慢且贵）
- 基准测试失效机制：数据污染、应试教育、饱和
- 四种核心指标的实现：exact match、token F1、perplexity、LLM-as-judge
- ELO 评分系统：Chatbot Arena 背后的排名方法
- 自定义评测五步法：定义任务 → 创建测试用例 → 定义评分 → 自动化 → 追踪趋势

### 阶段三：智能体评测专题（L19、L20、L30）

将评测知识扩展到智能体领域。智能体的评测比文本评测复杂得多——需要衡量端到端的任务完成率、轨迹效率和多步骤推理能力。

**L19 — SWE-bench / GAIA / AgentBench**
- SWE-bench：从 GitHub issue 到 PR 的端到端代码修复评测
- SWE-bench Verified：人工验证的高质量子集
- GAIA：通用 AI 助手基准，三个难度级别
- AgentBench：涵盖 8 个环境的综合智能体评测

**L20 — WebArena / OSWorld**
- WebArena：812 个长时域网页任务，跨 4 个自托管应用
- OSWorld：369 个跨 OS 桌面任务（Ubuntu/Windows/macOS）
- 两大失败模式：GUI 定位（像素→元素映射）和操作知识（菜单、快捷键的长尾知识）
- OSWorld-G（定位分离）和 OSWorld-Human（轨迹效率基准）

**L30 — 评估驱动的智能体开发**
- 三层评估架构：静态基准 → 自定义离线评估 → 在线生产评估
- 评估器-优化器反馈循环
- CI 门控和回归追踪
- 将评估嵌入开发工作流，而非事后补救

### 阶段四：从零构建评测系统（C27、C41、C49）

动手构建三个评测系统，从任务定义到结果聚合，掌握评测工程的完整技能栈。

**C27 — 基于 Fixture 任务的评测框架**
- FixtureTask 三元组：goal + setup + verifier
- 三种验证器：file_equals、regex_match、shell_exit_zero
- pass@1 与 pass@k 的计算和意义
- 延迟和成本指标（均值 + P95）
- 结构化 JSON 报告输出

**C41 — 完整评测流水线**
- 四维评测：困惑度、精确匹配、token F1、LLM-as-judge
- 每种评测的正确实现方式（off-by-one、归一化、LCS）
- 本地模拟 judge（正确接口，可无缝替换为真实模型）
- 加权聚合与分层报告（汇总 → 逐评测 → 逐样本）

**C49 — 语言模型评测框架**
- JSONL 任务规范（prompt + targets + metric + extras）
- 五种指标：exact_match、rouge_l、code_exec、multiple_choice、substring_contains
- 可替换模型适配器（ModelAdapter 协议）
- 批量运行器与 leaderboard JSON 输出

## 依赖关系

```
L27 (评测框架概述) ──┐
                     ├──> L10 (评测体系构建) ──> L30 (评估驱动开发)
L28 (长上下文评测) ──┘         │                     │
                               │                     ├──> C27 (Fixture 评测框架)
L19 (SWE-bench/GAIA) ──┐      │                     ├──> C41 (完整评测流水线)
                        ├──> L20 (WebArena/OSWorld) ─┘     └──> C49 (LM Eval Harness)
L10 (评测体系) ─────────┘
```

## 学习建议

1. **按阶段顺序推进。** 阶段一建立概念框架，阶段二掌握核心指标，阶段三扩展到智能体，阶段四动手构建。每个阶段都依赖前一阶段的知识。

2. **学习类和构建类交替进行。** 学习类课程（L27、L28、L19、L20、L30）侧重理解现有框架和方法论；构建类课程（L10、C27、C41、C49）要求动手实现。建议学完一个学习类课程后尽快进入对应的构建类课程。

3. **C27 → C41 → C49 递进关系明确。** C27 构建基础的 pass/fail 评测框架，C41 在此基础上增加多维度评测和聚合，C49 最终形成通用的 JSONL 任务规范和排行榜系统。三门构建课程最好按顺序完成。

4. **关注中文翻译版本。** 每门课程的 `docs/zh.md` 提供了完整的中文翻译，技术术语保留英文原文。中文版本适合快速阅读理解，英文版本适合查阅精确表述。

5. **动手课程要跑测试。** L10、C27、C41、C49 都有配套的 `test_main.py`，通过 `python3 -m pytest code/tests/ -v` 运行。测试通过是理解到位的标志。

## 课程位置

所有课程位于 `phases/` 目录下，文档在对应课程的 `docs/` 子目录中：

```
phases/05-nlp-foundations-to-advanced/
  27-llm-evaluation-frameworks/docs/{en,zh}.md
  28-long-context-evaluation/docs/{en,zh}.md

phases/10-llms-from-scratch/
  10-evaluation/docs/{en,zh}.md

phases/14-agent-engineering/
  19-benchmarks-swebench-gaia/docs/{en,zh}.md
  20-benchmarks-webarena-osworld/docs/{en,zh}.md
  30-eval-driven-agent-development/docs/{en,zh}.md

phases/19-capstone-projects/
  27-eval-harness-fixture-tasks/docs/{en,zh}.md
  41-eval-pipeline/docs/{en,zh}.md
  49-lm-eval-harness/docs/{en,zh}.md
```
