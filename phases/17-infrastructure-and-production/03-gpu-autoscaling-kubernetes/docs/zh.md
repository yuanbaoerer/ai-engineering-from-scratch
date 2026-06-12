# Kubernetes 上的 GPU 自动扩缩 — Karpenter、KAI Scheduler、Gang Scheduling

> 三层而非一层。Karpenter 动态配置节点（不到一分钟，比 Cluster Autoscaler 快 40%）。KAI Scheduler 处理 gang scheduling、拓扑感知和分层队列——它防止 7-of-8 部分分配陷阱，即七个节点等待并因一个缺失的 GPU 而空烧资源。应用级自动扩缩器（NVIDIA Dynamo Planner、llm-d Workload Variant Autoscaler）基于推理特定信号扩缩——队列深度、KV cache 利用率——而非 CPU/DCGM 占空比。经典的 HPA 陷阱是 `DCGM_FI_DEV_GPU_UTIL` 是占空比度量：100% 可能是 10 个请求也可能是 100 个。vLLM 预分配 KV cache 内存，因此内存永远不会触发缩容。本课教你组合三层并避免默认的 Karpenter `WhenEmptyOrUnderutilized` 策略——该策略会在推理进行中终止运行中的 GPU 作业。

**类型：** 学习
**语言：** Python（stdlib，玩具级队列深度自动扩缩模拟器）
**前置课程：** 阶段 17 · 02（推理平台经济学），阶段 17 · 04（vLLM 推理内部机制）
**时间：** 约 75 分钟

## 学习目标

- 画出三层自动扩缩（节点配置、gang scheduling、应用级）图，并说出每层使用的工具。
- 解释为什么 `DCGM_FI_DEV_GPU_UTIL` 对 vLLM 是错误的 HPA 信号，并说出两个替代品（队列深度、KV cache 利用率）。
- 描述 gang scheduling 以及 KAI Scheduler 防止的部分分配故障模式（8 个 GPU 中 7 个空闲）。
- 说出 Karpenter 整合策略（`WhenEmptyOrUnderutilized`）会终止运行中的 GPU 作业，并说明 2026 年的安全替代方案。

## 问题背景

你的团队在 Kubernetes 上部署 LLM 推理服务。你用 `DCGM_FI_DEV_GPU_UTIL` 作为信号设置 HPA。服务在业务时间固定在 100% 利用率。HPA 永远不会扩容——它已经认为你满了。你手动添加一个副本；TTFT 下降。HPA 仍然不扩容。这个信号在对你撒谎。

另外，你使用 Cluster Autoscaler 管理节点。凌晨 2 点到来一个 1M-token 提示；集群花 3 分钟配置一个节点，请求超时。

再另外，你部署一个 70B 模型需要跨 2 个节点使用 8 块 GPU。集群有 7 块 GPU 空闲，分散在 3 个节点上。Cluster Autoscaler 为缺失的 1 块 GPU 配置一个节点。七个节点等待 4 分钟烧钱，同时 Kubernetes 启动最后一块 GPU。

三层，三种不同的故障模式。2026 年的 GPU 感知自动扩缩不是"开启 HPA"，而是组合节点配置、gang scheduling 和应用信号自动扩缩。

## 核心概念

### 第一层 — 节点配置（Karpenter）

Karpenter 监视待处理 Pod，在约 45-60 秒内配置节点（Cluster Autoscaler 通常需要 90-120 秒配置 GPU 节点）。它根据 `NodePool` 约束动态选择实例类型——如果你的 Pod 需要 8 块 H100 但集群没有匹配节点，Karpenter 直接配置一个，而非扩展现有组。

**整合陷阱**：Karpenter 的默认 `consolidationPolicy: WhenEmptyOrUnderutilized` 对 GPU 池很危险。它会终止运行中的 GPU 节点，将 Pod 迁移到更便宜的合适实例。对于推理工作负载，这意味着驱逐运行中的请求并在新节点上重新加载 70B 模型。损失是数分钟的容量加上请求失败。

GPU 池的安全设置：

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

让 Karpenter 在一小时后整合真正空的节点，但不驱逐运行中的作业。

### 第二层 — gang scheduling（KAI Scheduler）

KAI Scheduler（项目原名 "Karp" 后改名）处理默认 kube-scheduler 不做的事：

**Gang scheduling** — 全部或全不调度。一个需要 8 块 GPU 的分布式推理 Pod，要么 8 块全部一起启动，要么一个都不启动。没有这个，你会遇到部分分配陷阱：7/8 个 Pod 启动，无限等待，烧钱。

**拓扑感知** — 知道哪些 GPU 共享 NVLink，哪些在同一个机架上，哪些之间有 InfiniBand。据此放置 Pod。DeepSeek-V3 67B 张量并行工作负载必须留在一个 NVLink 域内；KAI Scheduler 遵守这一点。

**分层队列** — 多个团队以优先级和配额竞争同一 GPU 池。团队 A 的生产紧急任务只在优先级规则允许时被团队 B 的训练任务抢占。

KAI 作为二级调度器与 kube-scheduler 一起部署；你注解工作负载以使用它。Ray 和 vLLM production-stack 都集成了它。

### 第三层 — 应用级信号

**HPA 陷阱**：`DCGM_FI_DEV_GPU_UTIL` 是占空比指标——它度量 GPU 在每个采样区间是否在工作。100% 利用率可能意味着 10 个并发请求或 100 个；GPU 都是忙的。基于占空比扩缩是盲目扩缩。

更糟的是，vLLM 和类似引擎预分配 KV cache 内存（最多 `--gpu-memory-utilization`）。即使只有一个请求，内存使用也保持在约 90%。基于内存的 HPA 永远不会缩容。

**2026 年替代信号**：

- 队列深度（等待 prefill 的请求数量）。
- KV cache 利用率（已分配给活跃序列的块比例）。
- 每副本 P99 TTFT（你的 SLA 信号）。
- Goodput（每秒满足所有 SLO 的请求数）。

NVIDIA Dynamo Planner 和 llm-d Workload Variant Autoscaler 消费这些信号并扩缩副本。它们完全替代 LLM 推理的 HPA。

### 何时使用什么

| 扩缩决策 | 工具 |
|---------|------|
| 添加/移除节点 | Karpenter |
| 调度多 GPU 作业 | KAI Scheduler |
| 添加/移除副本 | Dynamo Planner / llm-d WVA（或基于队列深度的自定义 HPA） |
| 选择 GPU 类型 | Karpenter NodePool |
| 抢占低优先级 | KAI Scheduler 队列 |

### 分离 prefill/decode 使一切复杂化

如果你运行分离 prefill/decode（阶段 17 · 17），你有两类 Pod，扩缩触发器不同：prefill Pod 基于队列深度扩缩，decode Pod 基于 KV cache 压力扩缩。llm-d 将它们暴露为独立的 `Services`，带按角色的 HPA。不要试图在两者前面放一个 HPA。

### 冷启动在这里也重要

冷启动缓解（阶段 17 · 10）是节点配置时间变得用户可感知的地方。Karpenter 的 45-60 秒预热加 20GB 模型加载加引擎初始化意味着从零请求需要 2-5 分钟。为 SLO 关键路径保持热池（`min_workers=1`），或在应用层使用 Modal 风格的检查点。

### 你应该记住的数字

- Karpenter 节点配置：约 45-60 秒 vs Cluster Autoscaler 约 90-120 秒（GPU 节点）。
- KAI Scheduler 防止部分分配浪费——7-of-8 陷阱。
- `DCGM_FI_DEV_GPU_UTIL` 作为 HPA 信号：不可用；用队列深度或 KV 利用率替代。
- Karpenter `WhenEmptyOrUnderutilized`：终止运行中的 GPU 作业。推理场景用 `WhenEmpty + consolidateAfter: 1h`。

```figure
autoscaling
```

## 使用

`code/main.py` 在突发 GPU 工作负载上模拟三层自动扩缩器。比较朴素 HPA（占空比）、队列深度 HPA 和 KAI gang 调度扩缩。报告未满足请求、空闲 GPU 分钟数和综合评分。

## 交付

本课产出 `outputs/skill-gpu-autoscaler-plan.md`。给定集群拓扑、工作负载形态和 SLO，设计三层自动扩缩方案。

## 练习

1. 运行 `code/main.py`。在突发工作负载下，朴素占空比 HPA 丢弃了多少队列深度 HPA 能捕获的请求？差异来自哪里？
2. 为在 H100 SXM5 上运行 Llama 3.3 70B FP8 的集群设计一个 Karpenter NodePool。指定 `capacity-type`、`disruption.consolidationPolicy`、`consolidateAfter` 和一个将非 GPU 工作负载排除在这些节点外的 taint。
3. 你的团队报告部署卡在 Pending 状态，因为"GPU 可用但 Pod 不会调度"。诊断——这是 Karpenter、kube-scheduler 还是 KAI Scheduler？哪些指标可以确认？
4. 选择一个信号用于自动扩缩分离 prefill Pod，另一个信号用于 decode Pod。论证两者。
5. 计算 24x7 生产服务上 `WhenEmptyOrUnderutilized` 整合陷阱的成本，该服务平均每天有 60 次请求丢弃事件，P99 TTFT > 10s。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| Karpenter | "节点配置器" | Kubernetes 节点自动扩缩器；亚分钟配置 |
| Cluster Autoscaler | "老扩缩器" | Kubernetes 节点自动扩缩器前身；更慢，基于组 |
| KAI Scheduler | "GPU 调度器" | 用于 gang + 拓扑 + 队列的二级调度器 |
| Gang scheduling | "全部或全不" | 原子调度 N 个 Pod 或全部推迟 |
| 拓扑感知 | "机架感知" | 基于 NVLink/IB/机架位置放置 Pod |
| `DCGM_FI_DEV_GPU_UTIL` | "GPU 利用率" | 占空比指标；不是 LLM 的扩缩信号 |
| 队列深度 | "等待请求" | prefill 受限扩缩的正确 HPA 信号 |
| KV cache 利用率 | "内存压力" | decode 受限扩缩的正确 HPA 信号 |
| 整合 | "Karpenter 整合" | 终止节点到更便宜的实例类型 |
| `WhenEmpty + 1h` | "安全整合" | 不驱逐运行中 GPU 作业的策略 |

## 延伸阅读

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler) — 设计文档和配置示例。
- [Karpenter Disruption Controls](https://karpenter.sh/docs/concepts/disruption/) — 整合策略语义和 GPU 安全默认值。
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — Dynamo Planner 扩缩信号。
- [Ray docs — KAI Scheduler for RayClusters](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) — Ray 集成模式。
- [AWS EKS Compute and Autoscaling Best Practices](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) — 托管 Kubernetes 特定指南。
- [llm-d GitHub](https://github.com/llm-d/llm-d) — Workload Variant Autoscaler 设计。
