# 解耦预填充/解码 — NVIDIA Dynamo 与 llm-d

> 预填充（prefill）是计算密集型的；解码（decode）是内存密集型的。在同一个 GPU 上运行两者会浪费资源。解耦（disaggregation）将它们拆分到独立的资源池，并通过 NIXL（RDMA/InfiniBand 或 TCP 回退）在它们之间传输 KV 缓存。NVIDIA Dynamo（GTC 2025 发布，1.0 GA）位于 vLLM/SGLang/TRT-LLM 之上——其 Planner Profiler + SLA Planner 自动速率匹配 prefill:decode 比例以满足 SLO。NVIDIA 发布了这个量级的吞吐量提升——developer.nvidia.com（2025-06）显示在 GB200 NVL72 + Dynamo 上 DeepSeek-R1 MoE 在中等延迟下有约 6 倍提升，Dynamo 产品页（developer.nvidia.com，未注明日期）宣称在 GB300 NVL72 + Dynamo 上 MoE 吞吐量比 Hopper 提升高达 50 倍。"30 倍"数字是社区对全栈 Blackwell + Dynamo + DeepSeek-R1 报告的综合；我们未找到单一原始来源明确表示恰好 30 倍，因此将其视为方向性声明。llm-d（Red Hat + AWS）是 Kubernetes 原生的：预填充/解码/路由器作为独立服务，每个角色有独立的 HPA。llm-d 0.5 增加了分层 KV 卸载、缓存感知 LoRA 路由、UCCL 网络、缩容到零。经济性：多个客户披露的内部汇总表明，在恒定 SLA 下，从共享部署切换到使用 Dynamo 的解耦部署，200 万美元级别的推理支出可节省 30-40%（即每年 60-80 万美元）；具体的 200 万→60-80 万数字是内部综合数据，不是单一公开案例研究——将其作为数量级锚点使用，而非引用文献。短提示（<512 token，短输出）不足以证明传输成本的合理性。

**类型：** 学习
**语言：** Python（stdlib，玩具解耦 vs 共享模拟器）
**前置课程：** 第 17 阶段 · 04（vLLM 推理内部），第 17 阶段 · 08（推理指标）
**时间：** 约 75 分钟

## 学习目标

- 解释为什么预填充和解码有不同的最优 GPU 分配，并量化共享模式下的浪费。
- 绘制解耦架构图：预填充池、解码池、通过 NIXL 的 KV 传输、路由器。
- 说出解耦不划算的条件（短提示、短输出）。
- 区分 NVIDIA Dynamo（栈上层）和 llm-d（Kubernetes 原生），并将各自匹配到适用的运维场景。

## 问题

你在 8 个 H100 上运行 Llama 3.3 70B。在混合工作负载（长提示 + 短输出）下，GPU 在解码阶段空闲，因为大部分计算花在了预填充上。在不同工作负载（短提示 + 长输出）下，情况相反。预填充 + 解码共享意味着你对两者都过度配置了资源。

预算影响：20-40% 的 GPU 时间浪费在错误的资源上。你购买 H100 的算力来运行内存密集型的解码，或者购买 H100 的 HBM 带宽来运行计算密集型的预填充。两者都是昂贵的浪费。

解耦将预填充和解码拆分到独立的池中，各自针对瓶颈优化。KV 缓存通过高带宽互连从预填充池传输到解码池。

## 概念

### 为什么瓶颈不同

**预填充**——一次前向传播运行整个输入提示的 Transformer。矩阵乘法占主导；计算密集型。H100 FP8 提供约 2000 TFLOPS 的有效吞吐量。批处理效率好——一次前向传播处理多个 token。

**解码**——每次生成一个 token，每次迭代读取完整权重。内存带宽密集型。HBM3 提供约 3 TB/s。批处理效率仅在高并发时好——权重读取的摊销在批次间分配。

共享部署：你购买同时优化两者的 GPU。H100 两者都擅长但成本相同。在规模上，你希望预填充池在 H100/计算密集型上；解码池在 H200/内存密集型上，或使用激进量化。

### 架构

```
            ┌──────────────┐
  请求 →   │    路由器    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ (仅提示)                      │
            ┌──────────────┐    KV 缓存     ┌───────▼──────┐
            │  预填充池    │ ─── NIXL ────► │   解码池    │
            │  (计算)      │                │  (内存)     │
            └──────────────┘                └──────┬───────┘
                                                   │ tokens
                                                   ▼
                                                 客户端
```

NIXL 是 NVIDIA 的节点间传输层。可用时使用 RDMA/InfiniBand，否则回退到 TCP。传输延迟是真实存在的——在 70B FP8 上 4K token 提示的 KV 缓存传输通常为 20-80 ms。这就是为什么短提示不足以证明解耦的合理性：传输成本超过了节省。

### Dynamo vs llm-d

**NVIDIA Dynamo**（GTC 2025 发布，1.0 GA）：
- 位于 vLLM、SGLang、TRT-LLM 之上作为编排器。
- Planner Profiler 测量工作负载，SLA Planner 自动配置预填充:解码比例。
- Rust 核心，Python 可扩展。
- 吞吐量提升：NVIDIA 报告在 GB200 NVL72 + Dynamo 上 DeepSeek-R1 MoE 在中等延迟下有 6 倍提升（developer.nvidia.com，2025-06）；社区关于全 Blackwell + Dynamo + DeepSeek-R1 栈"高达 30 倍"的报告缺乏单一原始来源，应视为方向性声明。
- GB300 NVL72 + Dynamo：据 Dynamo 产品页（developer.nvidia.com，未注明日期），MoE 吞吐量比 Hopper 提升高达 50 倍。

**llm-d**（Red Hat + AWS，Kubernetes 原生）：
- 预填充/解码/路由器作为独立 Kubernetes 服务。
- 每个角色独立 HPA，使用队列深度（预填充）/ KV 利用率（解码）信号。
- `topologyConstraint packDomain: rack` 将预填充+解码集群打包在同一机架上以实现高带宽 KV 传输。
- llm-d 0.5（2026）：分层 KV 卸载、缓存感知 LoRA 路由、UCCL 网络、缩容到零。

如果你想用托管的栈上层编排器，选择 Dynamo。如果你想用 Kubernetes 原生原语并致力于 CNCF 生态，选择 llm-d。

### 经济性

内部综合数据（不是单一公开案例研究——数量级锚点）：

- 每年 200 万美元共享部署的推理支出。
- 切换到使用 Dynamo 的解耦部署。
- 相同的请求量，相同的 P99 延迟 SLO。
- 报告节省：每年 $600K-$800K（减少 30-40%）。
- 无需新硬件。

我们综合了多个客户披露的数据，而非单一可引用的案例研究；最接近的公开数据点是 Baseten 的使用 Dynamo KV 路由实现 2 倍更快 TTFT / 61% 更高吞吐量（baseten.co，2025-10），以及 VAST + CoreWeave 预测在 40-60% KV 命中率下每美元多 60-130% token（vastdata.com，2025-12）。节省来自对每个池的合理配置；预填充密集型工作负载（8K+ 前缀的 RAG）比平衡型受益更多。

### 何时不该解耦

- 提示 < 512 token 且输出 < 200 token：传输成本主导收益。
- 小型集群（< 4 GPU）：池多样性不足。
- 团队无法操作两个带独立角色扩展的 GPU 池：Dynamo 有帮助但并非轻而易举。
- 无 RDMA 网络：TCP 传输成本更重。

### 路由器与第 17 阶段 · 11 集成

解耦路由器是 KV 缓存感知的（第 17 阶段 · 11）。请求落在持有其前缀的解码池上——如果没有匹配，则走预填充 → 解码流程。命中率和解耦相叠加——缓存感知路由器决定是否需要新的预填充。

### Blackwell 上的 MoE 才是真正的数字所在

GB300 NVL72 + Dynamo 相比 Hopper 基线显示 50 倍 MoE 吞吐量。MoE 专家路由在预填充上是计算密集型的，但在解码上是内存密集型的（专家缓存），因此解耦是双重胜利。2026 年前沿模型推理以 MoE 为主（DeepSeek-V3，未来 GPT-5 变体）。

### 你应该记住的数字

基准数字会变化——NVIDIA 和推理栈每季度发布更新结果。引用前请重新核实。

- DeepSeek-R1 在 GB200 NVL72 + Dynamo 上：在中等延迟下相比基线约 6 倍吞吐量（developer.nvidia.com，2025-06）；社区"高达 30 倍"的说法是全 Blackwell + Dynamo 栈的方向性综合，缺乏单一原始来源。
- GB300 NVL72 + Dynamo：MoE 吞吐量比 Hopper 提升高达 50 倍（developer.nvidia.com，未注明日期）。
- 节省锚点（内部综合，非单一案例研究）：每年 $600-800K，来自 $2M 年支出，恒定 SLA。
- 解耦阈值：提示 >512 token + 输出 >200 token。
- 通过 NIXL 的 KV 传输：70B FP8 上 4K 提示 KV 传输为 20-80 ms。

## 使用

`code/main.py` 模拟共享部署 vs 解耦部署。报告吞吐量、每请求成本和提示长度交叉点。

## 交付

本课产出 `outputs/skill-disaggregation-decider.md`。给定工作负载和集群，决定是否解耦。

## 练习

1. 运行 `code/main.py`。在什么提示长度下解耦优于共享部署？
2. 为 RAG 服务设计预填充池和解码池，P99 前缀长度 8K，输出 300。
3. Dynamo vs llm-d：为没有 Python 运行时偏好的纯 Kubernetes 环境选择一个。
4. 计算 KV 传输成本：70B FP8 上 4K 预填充 = ~500 MB KV。RDMA 100 GB/s 下传输 = 5 ms。TCP 10 GB/s 下 = 50 ms。哪个对你的 SLO 更重要？
5. MoE 专家路由改变 KV 访问模式。对每个 token 激活不同专家的 MoE，解耦表现如何？

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|----------|
| 解耦推理 | "拆分预填充/解码" | 每个阶段独立的 GPU 池 |
| NIXL | "NVIDIA 传输" | Dynamo 的节点间 KV 传输（RDMA/TCP） |
| NVIDIA Dynamo | "编排器" | vLLM/SGLang/TRT-LLM 的栈上层协调器 |
| llm-d | "Kubernetes 原生" | Red Hat + AWS K8s 解耦栈 |
| Planner Profiler | "Dynamo 自动配置" | 测量工作负载，配置池比例 |
| SLA Planner | "Dynamo 策略" | 自动速率匹配预填充:解码以满足 SLO |
| `packDomain: rack` | "llm-d 拓扑" | 将预填充+解码打包在同一机架上以实现快速 KV 传输 |
| UCCL | "统一集合" | llm-d 0.5 网络层，支持缩容到零 |
| MoE 专家路由 | "每个 token 的专家" | DeepSeek-V3 模式；解耦有帮助 |

## 延伸阅读

- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM Disaggregated Serving blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 release notes](https://github.com/llm-d/llm-d/releases)
