# Agent 可观测性：Langfuse、Phoenix、Opik

> 三大开源 agent 可观测性平台主导 2026 年。Langfuse（MIT）——每月 600 万+ 次 SDK 安装，链路追踪 + 提示词管理 + 评估 + 会话回放。Arize Phoenix（Elastic License 2.0）——深度 agent 专项评估、RAG 相关性、OpenInference 自动埋点。Comet Opik（Apache 2.0）——自动化提示词优化、护栏、LLM-judge 幻觉检测。

**类型：** 学习
**语言：** Python（标准库）
**前置课程：** 第 14 阶段 · 23（OTel GenAI）
**时间：** ~45 分钟

## 学习目标

- 列出三大开源 agent 可观测性平台及其许可证。
- 区分每个平台的优势：Langfuse（提示词管理 + 会话）、Phoenix（RAG + 自动埋点）、Opik（优化 + 护栏）。
- 解释为什么 89% 的组织到 2026 年已经部署了 agent 可观测性。
- 实现一个基于标准库的 trace 到 dashboard 流水线，包含 LLM-judge 评估。

## 问题背景

OTel GenAI（第 23 课）提供了 schema，你仍然需要一个平台来接收 span、运行评估、存储提示词版本并发现回归。三个竞争者各自强调生命周期的不同部分。

## 概念解析

### Langfuse（MIT）

- 每月 600 万+ 次 SDK 安装，19k+ GitHub star。
- 功能：链路追踪、带版本控制 + playground 的提示词管理、评估（LLM-as-judge、用户反馈、自定义）、会话回放。
- 2025 年 6 月：原先的商业模块（LLM-as-a-judge、标注队列、提示词实验、Playground）在 MIT 许可下开源。
- 最强场景：端到端可观测性与紧密的提示词管理闭环。

### Arize Phoenix（Elastic License 2.0）

- 更深入的 agent 专项评估：trace 聚类、异常检测、RAG 检索相关性。
- 原生 OpenInference 自动埋点。
- 与托管版 Arize AX 搭配用于生产环境。
- 无提示词版本控制——定位为与更广泛平台配合使用的漂移/行为回归检测工具。
- 最强场景：RAG 相关性、行为漂移、异常检测。

### Comet Opik（Apache 2.0）

- 通过 A/B 实验实现自动化提示词优化。
- 护栏（PII 脱敏、话题约束）。
- LLM-judge 幻觉检测。
- 来自 Comet 自身测量的基准测试：Opik 日志 + 评估耗时 23.44 秒 vs Langfuse 327.15 秒（约 14 倍差距）——厂商基准仅供方向参考。
- 最强场景：优化闭环、自动化实验、护栏执行。

### 行业数据

根据 Maxim（2026 年实地分析）：89% 的组织已部署 agent 可观测性；质量问题是最主要的生产障碍（32% 的受访者提及）。

### 选型指南

| 需求 | 推荐 |
|------|------|
| 一站式含提示词管理 | Langfuse |
| 深度 RAG 评估 + 漂移检测 | Phoenix |
| 自动化优化 + 护栏 | Opik |
| 开放许可证，无 ELv2 | Langfuse（MIT）或 Opik（Apache 2.0） |
| Datadog / New Relic 集成 | 任意——三者均导出 OTel |

### 常见误区

- **无评估策略。** 没有评估的链路追踪只是昂贵的日志记录。
- **自建 LLM-judge 缺乏校准。** CRITIC 模式（第 05 课）适用——judge 需要外部工具进行事实验证。
- **提示词版本未关联 trace。** 生产回归时，无法二分定位导致问题的提示词版本。

## 动手实践

`code/main.py` 实现了一个基于标准库的 trace 采集器 + LLM-judge 评估器：

- 接收 GenAI 格式的 span。
- 按会话分组，标记失败运行（护栏触发、低置信度评估）。
- 一个脚本化的 LLM-judge 按评分标准对 agent 响应进行打分。
- 一个类 dashboard 摘要：失败率、主要失败原因、评估分数分布。

运行：

```
python3 code/main.py
```

输出：每个会话的评估分数和失败分类，对应 Langfuse/Phoenix/Opik 所展示的内容。

## 使用方式

- **Langfuse** 自托管或云版；通过 OTel 或其 SDK 接入。
- **Arize Phoenix** 自托管；自动埋点 OpenInference。
- **Comet Opik** 自托管或云版；自动化优化闭环。
- **Datadog LLM Observability** 适合已使用 Datadog 的混合运维+ML 团队。

## 交付

`outputs/skill-obs-platform-wiring.md` 选择一个平台，将 trace + 评估 + 提示词版本接入现有 agent。

## 练习

1. 将一周的 OTel trace 导出到 Langfuse 云（免费版）。哪些会话失败了？原因是什么？
2. 为你所在领域编写 LLM-judge 评分标准（事实正确性、语气、范围遵循）。在 50 条 trace 上测试。
3. 对比 Langfuse 的提示词版本控制与 Phoenix 的 trace 聚类。哪个能更快地告诉你哪里出了问题？
4. 阅读 Opik 的护栏文档。将一个 PII 脱敏护栏接入你的某次 agent 运行。
5. 在你的语料库上对三个平台进行基准测试。忽略厂商发布的数字，自己测量。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 链路追踪 | "Span 采集器" | 接收 OTel / SDK span；按会话索引 |
| 提示词管理 | "提示词 CMS" | 与 trace 关联的版本化提示词 |
| LLM-as-judge | "自动化评估" | 独立 LLM 按评分标准对 agent 输出打分 |
| 会话回放 | "Trace 回放" | 逐步回放历史运行以调试 |
| RAG 相关性 | "检索质量" | 检索到的上下文是否匹配查询 |
| Trace 聚类 | "行为分组" | 对相似运行进行聚类以检测漂移 |
| 护栏执行 | "日志时策略" | 对记录内容进行 PII/毒性/范围检查 |

## 延伸阅读

- [Langfuse 文档](https://langfuse.com/) —— 链路追踪、评估、提示词管理
- [Arize Phoenix 文档](https://docs.arize.com/phoenix) —— 自动埋点、漂移检测
- [Comet Opik](https://www.comet.com/site/products/opik/) —— 优化 + 护栏
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) —— 三者共用的 schema
