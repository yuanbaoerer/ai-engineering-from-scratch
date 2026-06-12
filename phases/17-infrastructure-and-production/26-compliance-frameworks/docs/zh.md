# 合规 — SOC 2、HIPAA、GDPR、PCI-DSS、EU AI Act、ISO 42001

> 多框架覆盖是 2026 年企业客户的入场券。**EU AI Act**：自 2024 年 8 月 1 日起生效。大部分高风险要求将于 2026 年 8 月 2 日开始执行。高风险系统义务违规罚款最高 €15M 或全球年营业额 3%（第 99(4) 条）；禁止的 AI 实践违规罚款最高 €35M 或 7%（第 99(3) 条）。如服务欧盟用户则全球适用。**Colorado AI Act**：2026 年 6 月 30 日生效（因 SB25B-004 从 2026 年 2 月推迟）— 高风险系统影响评估、AI 决定申诉权。Virginia 在信贷/就业/住房/教育领域类似。**SOC 2 Type II**：2026 年 B2B AI 事实标准（Type II，非 Type I，金融科技领域）。**GDPR**：最大有记录的 AI 专项罚款是对 Clearview AI 的 €30.5M（荷兰 DPA，2024 年 9 月）；意大利 Garante 对 OpenAI 罚款 €15M（2024 年 12 月，2026 年 3 月上诉后被推翻）。推理时实时 PII 脱敏是可辩护的标准；后处理清理不够。**HIPAA**：医疗领域绑定 — 未经 BAA 不得将 PHI 发送至外部 AI 服务。**PCI-DSS**：AI 交互层覆盖需要配置 + 合同协议，非自动满足。**ISO 42001**：新兴 AI 治理标准，与 ISO 27001 一起成为采购要求。参考实例：OpenAI 维护 SOC 2 Type 2、ISO/IEC 27001:2022、ISO/IEC 27701:2019、GDPR/CCPA/HIPAA (BAA)/FERPA、PCI-DSS（ChatGPT 支付组件）。跨框架映射减少审计疲劳：访问控制跨 ISO 27001 A.5.15-5.18、GDPR 第 32 条、HIPAA §164.312(a)。

**类型：** 学习
**语言：** Python（可选 — 合规是策略 + 流程，不是代码）
**前置课程：** 第 17 阶段 · 25（安全），第 17 阶段 · 13（可观测性）
**时间：** ~60 分钟

## 学习目标

- 枚举 2026 年与 LLM 产品相关的七个框架，并将每个框架对应到客户群体。
- 引用 EU AI Act 执行时间线（2024 年 8 月生效；2026 年 8 月高风险执行）和两级罚款上限（€15M / 3% 高风险义务，€35M / 7% 禁止实践）。
- 解释为什么 GDPR 下后处理 PII 清理不够，并命名实时推理层脱敏为可辩护的标准。
- 描述跨框架控制映射（例如访问控制映射到 ISO 27001 A.5.15-5.18 + GDPR 第 32 条 + HIPAA §164.312(a)）。

## 问题

企业客户的采购部门要求 SOC 2 Type II、GDPR、HIPAA BAA、ISO 27001 和"EU AI Act 合规声明"。你的团队有 SOC 2 Type I。距离 Type II 还有六个月，尚未开始 GDPR 第 30 条记录。

多框架覆盖不是 LLM 问题 — 是企业 SaaS 问题，叠加了 LLM 特定要求。2026 年的采购团队需要的是一个矩阵，每行一个框架、每列一个控制项，而不是一份 PDF。

## 概念

### 七个框架

| 框架 | 范围 | LLM 特定要求 |
|------|------|-------------|
| SOC 2 Type II | B2B SaaS 基线 | 6-12 个月审计的过程控制 |
| HIPAA | 美国医疗 | 需要 BAA；未经签署协议 PHI 不得离开基础设施 |
| GDPR | 欧盟用户 | 实时 PII 脱敏；数据主体权利；第 30 条记录 |
| PCI-DSS | 支付数据 | AI 接触支付需要配置 + 合同 |
| EU AI Act | 服务欧盟用户 | 风险分级；高风险系统：合规评估、文档、日志 |
| Colorado AI Act | 服务 CO 居民 | 影响评估；申诉权 |
| ISO 42001 | AI 治理 | 新兴；与 ISO 27001 配合 |

### EU AI Act 时间线

- 2024 年 8 月 1 日：生效。
- 2025 年 2 月 2 日：禁止的 AI 实践开始执行。
- 2026 年 8 月 2 日：高风险系统开始执行（合规评估、文档、日志）。
- 2027 年 8 月：协调立法下产品中的高风险系统。

风险分级：不可接受（禁止）、高风险（合规 + 日志）、有限风险（透明度）、最小风险（无约束）。大多数 B2B LLM SaaS 属于有限风险；高风险适用于就业、信贷、教育、执法、移民、基本服务。

罚款（第 99 条）：高风险系统义务违规最高 €15M 或全球年营业额 3%（第 99(4) 条）；禁止的 AI 实践违规最高 €35M 或 7%（第 99(3) 条）；取较高者适用。

### GDPR — 实时脱敏是标准

后处理清理（在 LLM 看到数据后脱敏 PII）不是可辩护的姿态 — 模型已经看到了数据。实时推理层脱敏是 2026 年的标准：

- LLM 调用前的实体识别。
- 一致的分词（Mesh 方法）保留语义。
- 仅存储脱敏后的提示 + 同意加入的原始数据。

近期执法：对 Clearview AI 的 €30.5M 罚款（荷兰 DPA，2024 年 9 月）是目前最大的有记录的 AI 专项 GDPR 罚款；对 OpenAI 的 €15M 罚款（意大利 Garante，2024 年 12 月）是最大的 LLM 专项罚款，但在 2026 年 3 月上诉后被推翻，裁决仍在进一步审查中。后处理声明在审计中失败。

### HIPAA — BAA 不是可选的

未经签署的业务合作协议（BAA）不得将 PHI 发送至外部 AI 服务。三大超大规模 LLM 平台（Bedrock、Azure OpenAI、Vertex）均提供 BAA。OpenAI 直接 API 提供 BAA。Anthropic 直接 API 提供 BAA。发送 PHI 前请确认。

### SOC 2 Type II

Type I：控制已设计并文档化。
Type II：控制在 6-12 个月内有效运行。

2026 年 B2B 采购默认要求 Type II。Type I 是入门；Type II 是门槛。

常见审计驱动因素：访问日志（谁看了什么）、变更管理（如何部署的）、风险评估（季度）、事件响应（测试过吗？）。第 17 阶段 · 25 的审计日志可直接复用。

### 跨框架映射

一个访问控制策略满足多个框架控制：

| 控制 | 框架 |
|------|------|
| 访问日志 | ISO 27001 A.5.15-5.18、GDPR 第 32 条、HIPAA §164.312(a) |
| 变更管理 | ISO 27001 A.8.32、PCI DSS Req. 6、HIPAA 泄露通知范围 |
| 传输加密 | ISO 27001 A.8.24、GDPR 第 32 条、HIPAA §164.312(e) |
| 密钥管理 | ISO 27001 A.8.19、PCI DSS Req. 8、SOC 2 CC6.1 |

合规工具（Drata、Vanta、Secureframe）自动化此映射。在规模化时值得投入。

### ISO 42001 — 新兴

2023 年底发布。作为 ISO 27001 的补充，成为日益增长的采购要求。AI 治理框架，包括风险管理、数据质量、透明度、人工监督。

### OpenAI 参考实例

OpenAI 维护 SOC 2 Type 2、ISO/IEC 27001:2022、ISO/IEC 27701:2019、GDPR/CCPA/HIPAA (BAA)/FERPA、PCI-DSS（ChatGPT 支付组件）。这大致是 2026 年的企业入场券。

### 需要记住的数字

- EU AI Act 罚款：最高 €15M / 3%（高风险义务，第 99(4) 条）；最高 €35M / 7%（禁止实践，第 99(3) 条）。
- EU AI Act 高风险执行：2026 年 8 月 2 日。
- 最大有记录的 AI 专项 GDPR 罚款：€30.5M，Clearview AI（荷兰 DPA，2024 年 9 月）。
- 最大 LLM 专项 GDPR 罚款：€15M，OpenAI（意大利 Garante，2024 年 12 月；2026 年 3 月上诉后被推翻）。
- SOC 2 Type II 窗口：6-12 个月的运行控制。
- Colorado AI Act 生效日期：2026 年 6 月 30 日（因 SB25B-004 从 2026 年 2 月推迟）。

## 实践

`code/main.py` 是一个 Python 合规映射电子表格 — 给定一个控制项，列出它满足的框架。

## 产出

本课产出 `outputs/skill-compliance-matrix.md`。给定客户群体和地理位置，指定所需的框架和控制项。

## 练习

1. 你的第一个企业客户要求 SOC 2 Type II、HIPAA BAA、EU AI Act 声明。赢得这笔交易的最小可行合规姿态是什么？
2. 将三个假设的 LLM 产品按 EU AI Act 风险分级分类。高风险时有什么变化？
3. 你不小心将 PHI 发送给了没有 BAA 的提供商。走一遍事件响应流程。
4. 论证 ISO 42001 对于中型 AI 厂商在 2026 年是否"必要"。
5. 将你的 LLM 审计日志字段（第 17 阶段 · 25）映射到至少三个框架控制。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| SOC 2 Type II | "审计过的控制" | 控制运行 6-12 个月，经独立验证 |
| HIPAA BAA | "医疗合同" | 业务合作协议；PHI 必需 |
| GDPR | "欧盟隐私" | 实时 PII 脱敏是 2026 年可辩护的标准 |
| EU AI Act | "欧盟 AI 规则" | 高风险 2026 年 8 月执行；€15M / 3%（高风险义务）— €35M / 7%（禁止实践） |
| Colorado AI Act | "美国 AI 州法律" | 2026 年 6 月 30 日生效（因 SB25B-004 推迟）；影响评估 |
| ISO 42001 | "AI 治理" | AI 风险 + 透明度的新兴框架 |
| ISO 27001 | "安全 ISMS" | 信息安全管理体系基线 |
| 合规评估 | "EU AI 文档包" | 高风险要求：文档、测试、日志 |
| 跨框架映射 | "一个控制，多个框架" | 单一策略满足多个框架控制 |

## 延伸阅读

- [OpenAI 安全与隐私](https://openai.com/security-and-privacy/) — 参考合规实例。
- [GuardionAI — LLM 合规 2026：ISO 42001、EU AI Act、SOC 2、GDPR](https://guardion.ai/blog/llm-compliance-guide-iso-42001-eu-ai-act-soc2-gdpr-2026)
- [Dsalta — SOC 2 Type 2 审计指南 2026：10 个 AI 控制](https://www.dsalta.com/resources/ai-compliance/soc-2-type-2-audit-guide-2026-10-ai-powered-controls-every-saas-team-needs)
- [EU AI Act 官方文本](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) — 一手来源。
- [Colorado AI Act](https://leg.colorado.gov/bills/sb24-205) — 一手来源。
- [ISO/IEC 42001:2023](https://www.iso.org/standard/81230.html) — AI 管理体系标准。
