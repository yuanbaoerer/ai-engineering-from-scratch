# 无服务器 LLM 的冷启动缓解

> 一个 20 GB 的模型镜像从冷启动到服务需要 5-10 分钟（7B）到 20+ 分钟（70B）。在真正的无服务器世界中，这不是预热——而是宕机。缓解措施在五个层面运作：预置节点镜像（AWS 上的 Bottlerocket，双卷架构）、模型流式加载（NVIDIA Run:ai Model Streamer，vLLM 原生支持）、GPU 内存快照（Modal 检查点，重启速度提升 10 倍）、热池（`min_workers=1`）、分层加载（ServerlessLLM 的 NVMe→DRAM→HBM 管道，延迟降低 10-200 倍），以及迁移输入 token（KB）而非 KV cache（GB）的实时迁移。Modal 发布 2-4s 冷启动作为基准；Baseten 默认 5-10s，预热后可降至亚秒级。本课教你如何度量、预算并叠加这五个层面。

**类型：** 学习
**语言：** Python（stdlib，玩具冷启动路径模拟器）
**前置知识：** Phase 17 · 02（Inference Platform Economics），Phase 17 · 03（GPU Autoscaling）
**时间：** 约 60 分钟

## 学习目标

- 列出冷启动缓解的五个层面，并说出每个层面的一个工具或模式。
- 计算 70B 模型的总冷启动时间，即（节点配置）+（权重下载）+（权重加载到 HBM）+（引擎初始化）之和。
- 解释为什么实时迁移传输的是输入 token（KB）而非 KV cache（GB），以及代价是什么（重新计算）。
- 说出热池的权衡（为闲置 GPU 付费还是接受冷启动尾部延迟），以及 `min_workers > 0` 变为强制要求的 SLA 阈值。

## 问题描述

你的无服务器 LLM 端点在夜间缩放到零。早上 8 点流量激增。第一个请求需要等待：

1. Karpenter 配置 GPU 节点：45-60s。
2. 容器拉取包含权重的 30 GB 镜像：120-300s。
3. 引擎将权重加载到 HBM：45-120s，取决于模型大小和存储速度。
4. vLLM 或 TRT-LLM 初始化 CUDA graphs、KV cache 池、tokenizer：10-30s。

总计：220-510s（大约 3-8 分钟）后才返回第一个 token。你的 SLA 是 2s。你部署了热池（`min_workers=1`），问题似乎消失了——但现在你 24x7 为一块闲置 GPU 付费。如果你的服务有 5 个产品，每个有一个热副本，那就是 5 × 24 × 30 = 3,600 GPU 小时/月，无论是否有用户调用。

冷启动缓解就是如何在保持无服务器经济性的同时，逼近始终在线的延迟。

## 核心概念

### 第一层 — 预置节点镜像（Bottlerocket）

在 AWS 上，Bottlerocket 的双卷架构将操作系统与数据分离。快照包含预拉取容器镜像的数据卷；在 `EC2NodeClass` 中引用快照 ID。新节点启动时权重已在本地 NVMe 上——步骤 2 和部分步骤 3 消失。与 Karpenter 原生协作。大模型每次冷启动典型节省：2-4 分钟。

GCP 上的等效方案：使用预烘焙容器层的自定义 VM 镜像。Azure 上：使用托管磁盘快照的相同模式。

### 第二层 — 模型流式加载（Run:ai Model Streamer）

不是在响应第一个请求之前加载完整文件，而是逐层将权重流式加载到 GPU 内存中，一旦第一个 transformer block 就绪就开始处理。NVIDIA Run:ai Model Streamer 在 vLLM 2026 中原生支持。支持 S3、GCS 和本地 NVMe。通过将 I/O 与计算设置重叠，大模型的权重加载时间大致减半。

### 第三层 — GPU 内存快照（Modal）

Modal 在首次加载后对 GPU 状态（权重、CUDA graphs、KV cache 区域）进行检查点。后续重启直接反序列化到 HBM——比重新初始化快 10 倍。这是最接近"2 秒启动热 GPU"的方案。权衡：快照是按 GPU 拓扑的，所以如果 Karpenter 迁移到不同的 SKU，需要重新检查点。

### 第四层 — 热池（min_workers=1）

最简单的缓解措施：保持一个副本始终就绪。成本是一块 GPU 每小时费率 24x7。对小模型来说算术很残酷（你付 $0.85-$1.50/hr 只为避免 30s 冷启动），对大模型则比较友好（付 $4/hr 避免 5 分钟冷启动）。热池变为强制要求的 SLA 阈值：通常是 70B+ 模型的 TTFT P99 < 60s。

### 第五层 — 分层加载（ServerlessLLM）

ServerlessLLM 将存储视为层级：NVMe（快但大）、DRAM（中等但分层）、HBM（小但即时）。权重预加载到 DRAM；按需加载到 HBM。论文报告冷加载延迟比朴素的磁盘到 HBM 方式降低 10-200 倍。生产采用尚处于早期，但已存在与 vLLM 的集成。

### 第六层 — 实时迁移（附加模式）

当节点不可用（spot 驱逐、节点排空）时，传统模式是冷启动另一个副本并排空请求队列。实时迁移将输入 token（KB）移动到已加载模型的目的地，并在目的地重新计算 KV cache。重新计算比通过网络传输 GB 级 KV cache 更经济。适用于分离式部署。

### 热池的算术

对于 P99 TTFT SLA 为 2s 的服务，问题不是"要不要热池"而是"要多少个热副本，哪些路径需要"。

- 高价值交互路径（实时聊天、语音 agent）：`min_workers=1-2`。
- 后台批处理路径（夜间分类）：可接受缩放到零，5-10 分钟冷启动可容忍。
- 高级层：按租户设置 `min_workers`，配备专用容量。

### 优化前先度量

70B 模型在新节点上的冷启动解剖（说明性）：

| 阶段 | 时间 | 缓解措施 |
|------|------|----------|
| 节点配置 | 50s | Bottlerocket + 预置镜像，热池 |
| 镜像拉取 | 180s | 预置数据卷（消除） |
| 权重到 HBM | 75s | 模型流式加载（减半）；GPU 快照（消除） |
| 引擎初始化 | 20s | 持久化 CUDA graph 缓存 |
| 首次前向 | 3s | 最小固有延迟 |
| **总冷启动** | **328s** | |
| **缓解后总时间** | **约 15s** | 降低 22 倍 |

### 需要记住的数据

- Modal 冷启动：2-4s（使用 GPU 快照）。
- Baseten 默认冷启动：5-10s；预热后亚秒级。
- 原始 70B 冷启动：3-8 分钟。
- Run:ai Model Streamer：约 2 倍权重加载加速。
- ServerlessLLM 分层加载：10-200 倍延迟降低（论文数据）。

## 使用方法

`code/main.py` 模拟有无每种缓解措施的冷启动路径。报告总冷启动时间、热池成本以及热池自行回本的盈亏平衡请求速率。

## 实践产出

本课产出 `outputs/skill-cold-start-planner.md`。给定 SLA、模型大小和流量模式，选择叠加哪些缓解措施。

## 练习

1. 运行 `code/main.py`。计算热副本比通过 SLO 内额外请求丢弃来支付冷启动税更便宜的盈亏平衡请求速率。
2. 你部署了一个 13B 模型，P99 TTFT SLA 为 3s。选择实现目标的最小缓解栈（最少层数）。
3. Bottlerocket 预置消除了镜像拉取，但权重仍需从快照加载到 HBM。如果快照支持的 NVMe 读取速度为 7 GB/s，计算 70B 模型的挂钟时间。
4. 你的无服务器提供商提供 GPU 快照（Modal），但你的团队以"快照会泄露 PII"为由拒绝。论证双方——实际风险是什么，缓解措施是什么（临时快照、加密、命名空间隔离）？
5. 设计分层热池策略：付费用户、试用用户和批处理工作负载各需要多少个热副本？展示计算过程。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 冷启动 | "大暂停" | 从请求到新副本上首个 token 的时间 |
| 热池 | "始终在线最小值" | `min_workers >= 1` 以保持至少一个副本就绪 |
| 预置镜像 | "烘焙 AMI" | 容器权重预驻留的节点镜像 |
| Bottlerocket | "AWS 节点 OS" | AWS 容器优化操作系统，支持双卷快照 |
| 模型流式加载 | "流式加载" | 将权重 I/O 与计算设置重叠 |
| GPU 快照 | "检查点到 HBM" | 序列化加载后的 GPU 状态；重启时反序列化 |
| 分层加载 | "NVMe + DRAM + HBM" | 存储层级结构；按需加载 |
| 实时迁移 | "迁移 token" | 传输输入（KB），在目的地重新计算 KV |
| `min_workers` | "热副本" | 无服务器最小保活数量 |
| 缩放到零 | "完全无服务器" | 空闲时无成本；接受完全冷启动税 |

## 延伸阅读

- [Modal — Cold start performance](https://modal.com/docs/guide/cold-start) — Modal 发布的基准测试和检查点架构。
- [AWS Bottlerocket](https://github.com/bottlerocket-os/bottlerocket) — 预置数据卷快照模式。
- [NVIDIA Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer) — 将权重加载与计算设置重叠。
- [Baseten — Cold-start mitigation](https://www.baseten.co/blog/cold-start-mitigation/) — 预热手册。
- [ServerlessLLM 论文 (USENIX OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/fu) — 分层加载设计。
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — 分离式部署的实时迁移。
