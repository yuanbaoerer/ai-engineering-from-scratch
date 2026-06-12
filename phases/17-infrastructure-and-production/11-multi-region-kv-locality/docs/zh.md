# 多区域 LLM 服务与 KV 缓存局部性

> Round-robin 负载均衡对带缓存的 LLM 推理是有害的。如果请求未命中持有其前缀的节点，则需支付完整的 prefill 开销——在长 prompt 上 P50 约 800 ms，而缓存命中仅需约 80 ms。2026 年的生产模式是 cache-aware 路由器（用 Rust 实现的 vLLM Router、llm-d router），它消费 KV-cache 事件并基于前缀哈希匹配进行路由。最新研究（GORGO）将跨区域网络延迟显式纳入路由目标。商业"跨区域推理"服务（Bedrock cross-region inference、GKE multi-cluster gateways）将推理视为黑盒——它们处理可用性，而非 TTFT。摩根大通和梅奥诊所于 2024 年 11 月在 us-east-1 进行了故障转移演练，耗时约 22 分钟。DR 的现实是：32% 的 LLM DR 故障是因为团队备份了权重但忘记了 tokenizer 文件或量化配置。

**类型：** 学习
**语言：** Python（stdlib，用于模拟前缀缓存感知路由器的玩具模拟器）
**前置要求：** 阶段 17 · 04（vLLM 服务）、阶段 17 · 06（SGLang RadixAttention）
**时间：** 约 60 分钟

## 学习目标

- 解释为什么 round-robin 负载均衡会破坏带缓存的推理，并量化 TTFT 惩罚。
- 绘制 cache-aware 路由器的架构图：输入（KV-cache 事件）、算法（前缀哈希匹配）、决胜因素（GPU 利用率）。
- 列举 LLM DR 故障中 32% 的原因（缺少 tokenizer 文件/量化配置），并列出三文件 DR 清单。
- 区分商业跨区域服务（Bedrock CRI、GKE Multi-Cluster Gateway）与 KV-aware 路由。

## 问题背景

你的服务运行在 us-east-1、us-west-2 和 eu-west-1。你在前面放了一个 ALB 并使用 round-robin。生产环境的前缀缓存命中率降至 8%。TTFT P50 翻了三倍。你的 vLLM 日志显示每个请求都在支付完整的 prefill 开销。

Round-robin 对无状态服务是最优的。LLM 推理在设计上是有状态的——KV 缓存编码了模型看到的所有内容。盲目路由等于路由到错误的缓存。

另外，你的团队有 DR 计划。你将模型权重备份到 S3 跨区域。区域性故障发生；你尝试故障转移；副本拒绝启动。你忘记了 tokenizer.json、量化配置和 RoPE 缩放配置在你未同步的另一个存储桶中。

多区域 LLM 服务是一个缓存问题、一个路由问题和一个 DR 卫生问题——而不是负载均衡器问题。

## 核心概念

### Cache-aware 路由

请求带着 prompt 到达。路由器对前缀进行哈希（比如前 512 个 token）；它询问每个副本"你缓存了这个前缀吗？"。副本在 pub/sub 频道上发布 KV-cache 事件，表示分配和驱逐块。路由器选择匹配的副本，如果没有匹配则退而使用基于 GPU 利用率的决胜因素。

**vLLM Router**（Rust，2026 生产栈）：订阅 `kv.cache.block_added` 事件，维护前缀哈希 → 副本索引的映射，O(1) 查找进行路由。无匹配时退回到最浅队列深度。

**llm-d router**：相同模式，Kubernetes 原生。通过 ControlPlane API 发布事件。

**SGLang RadixAttention**（阶段 17 · 06）是副本内的等价方案。跨副本路由严格在上游。

### 数据

2K token prompt、Llama 3.3 70B FP8、H100 上的 TTFT P50：
- 缓存命中（同一副本，前缀驻留）：约 80 ms。
- 缓存未命中（冷 prefill）：约 800 ms。

10 倍差距。如果你的路由器在副本间达到 60-80% 的前缀缓存命中率，你可以在 N 副本容量下接近单副本性能。如果只有 10%，则接近朴素扩展。

### 跨区域有一个新约束——网络延迟

跨区域 RTT：
- us-east-1 ↔ us-west-2：约 65 ms。
- us-east-1 ↔ eu-west-1：约 75 ms。
- us-east-1 ↔ ap-southeast-1：约 220 ms。

如果路由将请求从 us-east-1 发送到 ap-southeast-1 的热前缀，节省的 prefill（800 → 80 ms）被 440 ms 的往返时间所淹没。GORGO（2026 研究）将此显式化——联合最小化 `prefill_time + network_latency`，而非仅 prefill。通常答案是保持区域路由，除非是 prefill 占主导的超大多 MB 前缀。

### 商业"跨区域推理"对此无帮助

AWS Bedrock cross-region inference 在容量压力时自动将请求路由到其他区域。它优化的是可用性，而非 TTFT，并将推理视为黑盒。GKE Multi-Cluster Gateway 也是如此——服务级故障转移，不感知 KV 缓存。

即使使用这些服务，你仍然需要应用层的 cache-aware 路由器。它们处理"us-east-1 挂了"的情况。Cache-aware 路由处理 TTFT 的情况。

### DR 卫生——32% 缺失文件问题

2026 年广泛引用的数据：32% 的 LLM DR 故障是因为团队备份了权重但忘记了：

- `tokenizer.json` 或 `tokenizer.model`
- 量化配置（`quantize_config.json`、AWQ scales、GPTQ zero-points）
- 模型特定配置（RoPE 缩放、attention masks、chat templates）
- 引擎配置（`vllm_config.yaml`、sampling defaults、LoRA adapter manifests）

修复方案是三文件最低 DR 清单：

1. HF 模型仓库下的所有文件（权重 + 配置 + tokenizer）。
2. 引擎特定的服务配置。
3. 部署清单（K8s YAML、Dockerfile、依赖锁）。

另外：每季度运行一次 DR 演练。摩根大通 us-east-1 演练在 2024 年 11 月达到 22 分钟恢复（30 分钟 SLA），仅因为该流程经过了预演。

### 数据驻留是正交的

欧盟客户的 PHI 不能离开欧盟。如果你的 cache-aware 路由器将巴黎发起的请求发送到 us-east-1 以匹配前缀，无论 TTFT 收益如何，你都违反了 GDPR。在优化缓存之前，先按驻留边界分区路由器。

### 需要记住的数据

- 缓存命中 vs 未命中 TTFT 差距：约 10 倍（2K prompt 上 80 ms vs 800 ms）。
- 跨区域 RTT US-EU：约 75 ms。
- DR 故障：32% 缺失 tokenizer/量化配置。
- 摩根大通 us-east-1 故障转移 2024 年 11 月：22 分钟（30 分钟 SLA）。

## 使用

`code/main.py` 模拟三种路由策略（round-robin、cache-aware 区域路由、cache-aware 全局路由）在多区域工作负载上的表现。报告缓存命中率、TTFT P50/P99 和跨区域费用。

## 产出

本课产出 `outputs/skill-multi-region-router.md`。根据区域、驻留约束和 SLA，设计路由计划。

## 练习

1. 运行 `code/main.py`。在 75 ms RTT 下，跨区域路由在什么 prompt 长度时胜过仅本地路由？
2. 你的缓存命中率从 70% 下降到 12%。诊断三个可能的原因以及确认每个原因的可观测指标。
3. 为一个 70B AWQ 量化模型设计 DR 清单，该模型在 vLLM 上服务并带有 5 个 LoRA adapter。列出所有文件和配置。
4. 论述 Bedrock cross-region inference 对于有严格 TTFT SLO 的金融科技公司是否"足够"。引用具体行为。
5. 一个巴黎发起的请求在 us-east-1 匹配到前缀。你会路由它吗？写出策略。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Cache-aware routing | "智能 LB" | 基于前缀哈希匹配路由到持有 KV-cache 的副本 |
| KV-cache events | "缓存 pub/sub" | 副本发布块添加/驱逐事件；路由器索引 |
| Prefix hash | "缓存键" | 前 N 个 token 的哈希，用作路由器查找 |
| GORGO | "跨区域路由研究" | arXiv 2602.11688；网络延迟作为显式项 |
| Cross-region inference | "Bedrock CRI" | AWS 产品；可用性故障转移，非 TTFT 感知 |
| DR manifest | "备份清单" | 恢复所需的每个文件——不仅仅是权重 |
| Data residency | "GDPR 边界" | 对哪些区域可以看到用户数据的法律约束 |
| RTT | "往返时间" | 网络延迟；US-EU 75 ms，US-APAC 220 ms |
| LLM-aware LB | "缓存命中 LB" | Cache-aware 路由器作为产品类别 |

## 延伸阅读

- [BentoML — Multi-cloud and cross-region inference](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference)
- [arXiv — GORGO (2602.11688)](https://arxiv.org/html/2602.11688v1) — 带网络延迟项的跨区域 KV-cache 复用。
- [TianPan — Multi-Region LLM Serving Cache Locality](https://tianpan.co/blog/2026-04-17-multi-region-llm-serving-data-residency-routing)
- [AWS Bedrock Cross-Region Inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) — 可用性故障转移文档。
- [vLLM Production Stack Router](https://github.com/vllm-project/production-stack) — cache-aware 路由器源码。
