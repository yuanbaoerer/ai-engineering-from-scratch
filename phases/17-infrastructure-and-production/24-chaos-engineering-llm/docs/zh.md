# LLM 生产环境的混沌工程（Chaos Engineering）

> 2026 年，面向 LLM 的混沌工程已成为一门独立学科。在生产环境运行实验的前提条件：定义的 SLI/SLO、追踪（Trace）+ 指标（Metric）+ 日志（Log）可观测性、自动回滚、运维手册（Runbooks）、值班（On-call）机制。架构包含四个平面：控制平面（实验调度器）、目标平面（服务、基础设施、数据存储）、安全平面（护栏 + 中止 + 流量过滤）、可观测性平面（指标 + 追踪 + 日志）、反馈平面（输入到 SLO 调整）。护栏是强制性的：如果每日错误预算（Error Budget）消耗速率超过预期的 2 倍，消耗速率告警（Burn-rate Alert）会暂停实验；抑制窗口（Suppression Windows）+ Trace-ID 关联可减少告警噪音。节奏：每周一次小型金丝雀测试 + SLO 审查；每月一次实战演练（Game Day）+ 事后复盘；每季度一次跨团队韧性审计 + 依赖关系映射。LLM 专用实验：内存溢出、网络故障、提供商宕机、恶意格式提示词（Malformed Prompt）、KV 缓存驱逐风暴（KV Cache Eviction Storm）。工具：Harness Chaos Engineering（基于 LLM 推荐、爆炸半径缩容、MCP 工具集成）；LitmusChaos（CNCF 毕业项目）；Chaos Mesh（CNCF Kubernetes 原生沙箱项目）。

**类型：** 学习
**语言：** Python（标准库，玩具级混沌实验运行器）
**前置知识：** 第 17 阶段 · 23（面向 AI 的 SRE），第 17 阶段 · 13（可观测性）
**时间：** 约 60 分钟

## 学习目标

- 列举混沌工程的五个前提条件（SLI/SLO、可观测性、回滚、运维手册、值班），并解释为什么跳过任何一个都会使实践失败。
- 绘制四个平面（控制、目标、安全、可观测性）以及到 SLO 的反馈循环图。
- 列举五个 LLM 专用实验（内存溢出、网络故障、提供商宕机、恶意格式提示词、KV 缓存驱逐风暴）。
- 根据技术栈选择合适的工具——Harness、LitmusChaos 或 Chaos Mesh。

## 问题所在

传统技术栈中的混沌测试已经成熟。LLM 技术栈引入了新的故障模式。一个包含 4K Token 的恶意格式提示词会使分词器（Tokenizer）停滞 12 秒。上游提供商返回 429 限流状态码；您的网关进行重试；您的服务因重试放大的并发而发生 OOM。在突发负载下的 KV 缓存驱逐风暴会导致重新预填充（Re-prefill）的级联反应，使计算资源耗尽。

这些故障都不会在单元测试中显现。混沌工程是您在用户发现之前找到它们的方法。

## 核心概念

### 前提条件

在生产环境运行混沌实验前，必须具备：

1. **SLI/SLO** —— 已定义的服务级别指标（Service Level Indicator）和目标（Service Level Objective）。
2. **可观测性（Observability）** —— 追踪、指标、日志，并已接入仪表盘。
3. **自动回滚（Automated Rollback）** —— 第 17 阶段 · 20 中的策略标志回滚。
4. **运维手册（Runbooks）** —— 结构化，参考第 17 阶段 · 23。
5. **值班（On-call）** —— 有人响应。

缺少任何一项，混沌测试都会变成真实的生产事故。

### 四个平面 + 反馈

**控制平面（Control Plane）** —— 实验调度器（Litmus 工作流、Chaos Mesh 调度、Harness UI）。

**目标平面（Target Plane）** —— 服务、Pod、节点、负载均衡器、数据存储。

**安全平面（Safety Plane）** —— 紧急开关、抑制窗口、爆炸半径限制、错误预算门禁。

**可观测性平面（Observability Plane）** —— 常规指标 + Trace-ID 关联，以区分混沌引发的故障和自然发生的故障。

**反馈循环（Feedback Loop）** —— 实验发现反馈到 SLO 调整、运维手册更新和代码修复中。

### 护栏是强制性的

- **消耗速率告警（Burn-rate Alert）**：如果每日错误预算消耗速率超过预期的 2 倍，则暂停实验。
- **抑制窗口（Suppression Windows）**：在实验期间，屏蔽爆炸半径内非实验相关的告警。
- **Trace-ID 关联**：所有实验引发的错误都带有标签，以便值班人员去重。

### 五个 LLM 专用实验

1. **内存溢出（Memory Overload）** —— 通过发送长上下文请求并施加高并发，强制触发 KV 缓存抢占风暴。观察：服务是优雅降级还是崩溃？

2. **网络故障（Network Failure）** —— 切断推理网关与提供商之间的连接。观察：回退机制是否在 SLA 内启动？（第 17 阶段 · 19）

3. **提供商宕机模拟（Provider Outage Simulation）** —— 模拟 OpenAI 返回 100% 的 429 限流响应。观察：路由是否故障转移到 Anthropic？（第 17 阶段 · 16, 19）

4. **恶意格式提示词（Malformed Prompt）** —— 注入会使分词器停滞的载荷（例如，深度嵌套的 Unicode、巨大的 UTF-8 码位）。观察：单个请求是否会锁定整个工作线程？

5. **KV 缓存驱逐风暴（KV Eviction Storm）** —— 通过耗尽 vLLM 块预算（Block Budget）强制触发驱逐。观察：LMCache 能否恢复，还是服务会降级？

### 实验节奏

- **每周** —— 在预发布环境进行小型金丝雀实验，可能在生产环境测试 5% 流量。
- **每月** —— 针对特定场景进行计划中的实战演练；跨团队参加；事后复盘。
- **每季度** —— 跨团队韧性审计；更新依赖关系图。

### 工具

- **Harness Chaos Engineering** —— 商业软件；基于 AI 生成的实验建议；爆炸半径缩容；MCP 工具集成。
- **LitmusChaos** —— CNCF 毕业项目；基于 Kubernetes 工作流。
- **Chaos Mesh** —— CNCF 沙箱项目；Kubernetes 原生 CRD 风格。
- **Gremlin** —— 商业软件；支持广泛。
- **AWS FIS** / **Azure Chaos Studio** —— 托管云服务。

### 从小处着手

第一个实验：在稳态流量下，通过 Pod Kill 终止一个解码副本。观察重路由和恢复情况。如果这个实验运行良好且看起来安全，则升级到网络混沌实验。

第一个 LLM 专用实验：注入一个提供商 429 限流响应，持续 5 分钟。观察回退机制。大多数团队会发现他们的回退机制并未经过充分测试。

### 您应该记住的数字

- 四个平面：控制、目标、安全、可观测性。
- 消耗速率暂停：超过预期每日预算消耗速率的 2 倍。
- 节奏：每周金丝雀测试，每月实战演练，每季度审计。
- 五个 LLM 实验：内存、网络、提供商、恶意格式提示词、KV 风暴。

## 使用它

`code/main.py` 模拟了三个带有安全平面门禁的混沌实验。报告了哪些实验会触发消耗速率中止。

## 交付它

本课程产出 `outputs/skill-chaos-plan.md`。根据技术栈和成熟度，选择前三个实验和工具。

## 练习

1. 运行 `code/main.py`。哪个实验触发了消耗速率门禁？为什么？
2. 为一个基于 vLLM 的 RAG 服务设计前五个混沌实验。包含成功标准。
3. 您的消耗速率告警暂停了一个实验。如何确定根本原因是混沌实验还是自然故障？
4. 论证混沌实验应该在生产环境还是仅在预发布环境运行。什么时候在生产环境运行是正确的选择？
5. 列举三个通用网络混沌无法复现的 LLM 特定故障模式。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| SLI / SLO | “服务目标” | 指标 + 目标；必须的前提条件 |
| 爆炸半径 (Blast radius) | “范围” | 受实验影响的服务/用户集合 |
| 消耗速率告警 (Burn-rate alert) | “预算门禁” | 当错误预算消耗速率 > 预期 2 倍时触发 |
| 实战演练 (Game day) | “月度演练” | 计划中的跨团队混沌演练 |
| LitmusChaos | “CNCF 工作流” | CNCF 毕业的 Kubernetes 混沌工具 |
| Chaos Mesh | “CNCF CRD” | CNCF 沙箱项目，Kubernetes 原生混沌工具 |
| Harness CE | “商业 AI 辅助” | Harness 带有 AI 推荐的混沌工具 |
| 恶意格式提示词 (Malformed prompt) | “分词器炸弹” | 导致分词过程停滞的输入 |
| KV 缓存驱逐风暴 (KV eviction storm) | “抢占级联” | 大规模驱逐触发重新预填充 |

## 延伸阅读

- [DevSecOps School — 混沌工程 2026 指南](https://devsecopsschool.com/blog/chaos-engineering/)
- [Ankush Sharma — LLM 的可观测性（书籍）](https://www.amazon.com/Observability-Large-Language-Models-Engineering-ebook/dp/B0DJSR65TR)
- [LitmusChaos (CNCF)](https://litmuschaos.io/)
- [Chaos Mesh (CNCF)](https://chaos-mesh.org/)
- [Harness Chaos Engineering](https://www.harness.io/products/chaos-engineering)
- [AWS FIS](https://aws.amazon.com/fis/)