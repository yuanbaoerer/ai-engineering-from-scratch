# vLLM 生产栈与 LMCache KV 卸载

> vLLM 的 production-stack 是参考 Kubernetes 部署——路由器、引擎和可观测性连接在一起。LMCache 是将 KV 缓存从 GPU 内存中提取出来并在查询和引擎间复用的 KV 卸载层（CPU DRAM，然后磁盘/Ceph）。vLLM 0.11.0 KV Offloading Connector（2026 年 1 月）通过 Connector API（v0.9.0+）使其异步化和可插拔。卸载延迟对用户不可见。LMCache 即使没有共享前缀也有价值——当 GPU 用完 KV 槽位时，被抢占的请求可以从 CPU 恢复，而无需重新计算预填充。16x H100（80GB HBM）在 4 个 a3-highgpu-4g 上的公开基准测试表明：当 KV 缓存超过 HBM 时，原生 CPU 卸载和 LMCache 都显著提升吞吐量；在低 KV 占用时，所有配置与基线持平，开销很小。

**类型：** 学习
**语言：** Python（stdlib，玩具级 KV 溢出模拟器）
**前置课程：** 阶段 17 · 04（vLLM 推理内部机制），阶段 17 · 06（SGLang/RadixAttention）
**时间：** 约 60 分钟

## 学习目标

- 绘制 vLLM production-stack 层次图：路由器、引擎、KV 卸载、可观测性。
- 解释 KV Offloading Connector API（v0.9.0+）以及 0.11.0 异步路径如何隐藏卸载延迟。
- 量化 LMCache CPU-DRAM 何时有帮助（KV > HBM）vs 增加开销（KV 足够小可以放入 HBM）。
- 在部署约束下，选择原生 vLLM CPU 卸载还是 LMCache connector。

## 问题背景

你的 vLLM 推理在并发增加时 GPU HBM 达到 100%，出现抢占事件。请求被驱逐、重新排队，你在一分钟内对相同的 2K token 提示重新预填充了四次。GPU 算力花在了冗余预填充上；有效吞吐量远低于原始吞吐量。

增加更多 GPU 成本线性增长。增加更多 HBM 不可能。但 CPU DRAM 便宜——一个插槽有 512 GB+，延迟比 HBM 差几个数量级，但对"临时热"KV 缓存来说足够了。

LMCache 将 KV 缓存提取到 CPU DRAM，使被抢占的请求快速恢复，并且跨引擎的重复前缀共享缓存，无需每个引擎重新预填充。

## 核心概念

### vLLM production-stack

`github.com/vllm-project/production-stack` 是参考 Kubernetes 部署：

- **路由器**——缓存感知（阶段 17 · 11）。消费 KV 事件。
- **引擎**——vLLM 工作节点。每个 GPU 或每个 TP/PP 组一个。
- **KV 缓存卸载**——LMCache 部署或原生 connector。
- **可观测性**——Prometheus 抓取、Grafana 仪表板、OTel 追踪。
- **控制平面**——服务发现、配置、滚动更新。

以 Helm chart + operator 形式交付。

### KV Offloading Connector API（v0.9.0+）

vLLM 0.9.0 引入了可插拔 KV 缓存后端的 Connector API。你的引擎将块卸载到 connector；connector 存储它们（RAM、磁盘、对象存储、LMCache）。请求需要块时，connector 将其加载回来。

vLLM 0.11.0（2026 年 1 月）增加了异步卸载路径——卸载可以在后台发生，因此引擎在常见情况下不会被阻塞。端到端延迟和吞吐量仍取决于工作负载形状、KV 缓存命中率和系统压力；vLLM 自己的说明指出自定义内核卸载在低命中率时可能降低吞吐量，且异步调度与投机解码有已知的交互问题。

### 原生 CPU 卸载 vs LMCache

**原生 vLLM CPU 卸载**：引擎本地。在主机 RAM 中存储 KV 块。实现快，零网络跳转。不跨引擎。

**LMCache connector**：集群规模。在共享 LMCache 服务器（CPU DRAM + Ceph/S3 层）中存储块。任何引擎都可访问。已发布 16x H100 基准测试。

当单个引擎有 HBM 压力时选择原生。当多个引擎共享前缀时（具有公共系统提示的 RAG、具有共享模板的多租户）选择 LMCache。

### 基准测试表现

16x H100（80 GB HBM）分布在 4 个 a3-highgpu-4g 上的测试：

- 低 KV 占用（短提示、低并发）：所有配置与基线持平，LMCache 增加约 3-5% 开销。
- 中等占用：LMCache 开始在跨引擎前缀复用上发挥作用。
- KV 超过 HBM：原生 CPU 卸载和 LMCache 都显著提升吞吐量；LMCache 因跨引擎共享收益更大。

### LMCache 发挥决定性作用的场景

- 系统提示跨租户共享的多租户推理。
- 文档块在查询间重复的 RAG。
- 相同基础模型上的微调变体（LoRA），基础模型 KV 复用减少冗余工作。
- 抢占密集型工作负载：从 CPU 恢复比重新预填充更便宜。

### 何时不该启用

- HBM 压力小——你支付开销但没有收益。
- 短上下文（<1K token）——传输时间 > 重新预填充。
- 单租户单提示工作负载——无复用可捕获。

### 与分离式推理的集成

阶段 17 · 17 分离式推理 + LMCache 叠加效应：KV 从预填充池传输到解码池时如果未使用则落入 LMCache；后续查询从 LMCache 拉取。阶段 17 · 11 缓存感知路由器可以路由到其本地或 LMCache 共享缓存匹配的引擎。

### 你应该记住的数字

- vLLM 0.9.0：Connector API 发布。
- vLLM 0.11.0（2026 年 1 月）：异步卸载路径；端到端延迟影响取决于工作负载、KV 命中率和系统压力（非绝对保证）。
- 16x H100 基准测试：当 KV 占用超过 HBM 时 LMCache 有帮助。
- 小 HBM 压力：3-5% 开销，无收益。

```figure
zero-sharding
```

## 使用

`code/main.py` 模拟有和无 LMCache 的抢占密集型工作负载。报告避免的重新预填充次数、吞吐量提升和盈亏平衡 HBM 利用率。

## 交付

本课产出 `outputs/skill-vllm-stack-decider.md`。给定工作负载形状和 vLLM 部署，决定使用原生 vs LMCache vs 都不用。

## 练习

1. 运行 `code/main.py`。在什么 HBM 利用率下 LMCache 开始有回报？
2. 某租户在 200 个查询/小时中共享 6K token 的系统提示。计算每个租户的预期 LMCache 节省。
3. LMCache 服务器是单点故障。设计高可用策略（副本、回退到原生）。
4. LMCache 存储到旋转磁盘上的 Ceph。对于 70B FP8 的 4K token KV（500 MB），读取时间 vs 重新预填充时间？
5. 论证 vLLM 0.11.0 异步路径是否"免费"——开销隐藏在哪里？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| Production-stack | "参考部署" | vLLM 的 Kubernetes Helm chart + operator |
| Connector API | "KV 后端接口" | vLLM 0.9.0+ 可插拔 KV 存储接口 |
| 原生 CPU 卸载 | "引擎本地溢出" | 在同一引擎的主机 RAM 中存储 KV |
| LMCache | "集群 KV 缓存" | 跨引擎 KV 缓存服务器，基于 CPU DRAM + 磁盘 |
| 0.11.0 异步 | "非阻塞卸载" | 卸载隐藏在引擎流之后 |
| 抢占 | "驱逐以腾出空间" | HBM 满时的 KV 缓存重新排列 |
| 前缀复用 | "相同系统提示" | 多个查询共享开头；缓存命中 |
| Ceph 层 | "磁盘层" | 缓存层次中低于 DRAM 的持久存储 |

## 延伸阅读

- [vLLM Blog — KV Offloading Connector (Jan 2026)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack) — Helm chart + operator。
- [LMCache for Enterprise-Scale LLM Inference (arXiv:2510.09665)](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache) — Connector 实现。
- [vLLM 0.11.0 release notes](https://github.com/vllm-project/vllm/releases) — 异步路径详情。
