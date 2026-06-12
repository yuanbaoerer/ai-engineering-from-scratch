# SGLang 与 RadixAttention：前缀密集型工作负载的优化

> SGLang 将 KV cache 视为一等可复用资源，存储在 radix tree 中。当 vLLM 按 FCFS（先到先服务）调度请求时，SGLang 的缓存感知调度器优先处理共享更长前缀的请求——本质上是一次深度优先的 radix 遍历，使热分支保留在 HBM 中。在 Llama 3.1 8B 上使用 ShareGPT 风格的 1K prompt 时，SGLang 达到约 16,200 tok/s，而 vLLM 约 12,500 tok/s，领先约 29%。在前缀密集型 RAG 工作负载上，优势可达 6.4 倍。在语音克隆类工作负载上，缓存命中率超过 86%。2026 年已在 xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS 等平台上部署超过 400,000 块 GPU。需要注意的是，6.4 倍的数字在前缀排序不一致时会消失——排序是工程师的关键杠杆。

**类型：** 学习
**语言：** Python（stdlib，玩具 radix-tree 缓存 + 缓存感知调度器）
**前置课程：** 阶段 17 · 04（vLLM 推理内部机制），阶段 14（Agentic RAG）
**时间：** 约 75 分钟

## 学习目标

- 绘制 RadixAttention 的架构图：前缀如何存储在 radix tree 中，以及 KV blocks 如何在同一分支下的序列间共享。
- 解释缓存感知调度以及为什么 FCFS 不适用于前缀密集型流量。
- 根据前缀缓存命中率和 prompt 长度分布，计算工作负载的预期加速比。
- 说出使 6.4 倍数字成为现实的 prompt 排序规范，以及什么情况下该优势会丧失。

## 问题描述

传统服务将每个请求的 prompt 视为不透明的。即使 5,000 个 RAG 请求都以相同的 2,000 token 系统 prompt 加相同的检索前言开头，vLLM 也会对这 2,000 token 的前缀重复预填充 5,000 次。GPU 反复执行相同的工作。

观察发现：agent 和 RAG 工作负载中的 prompt 几乎总是共享很长的前缀。系统 prompt、工具 schema、few-shot 示例、检索头、对话历史——这些都在请求间重复出现。如果你只存储一次该前缀的 KV cache 并重复使用，就不会再重复预填充。

RadixAttention 正是这样做的。Token 被索引在 radix tree 中；每个节点拥有从根到该节点路径上 token 序列的 KV blocks。新请求遍历树时：任何 token 匹配的节点都可以复用其 KV blocks。预填充成本仅与"新"后缀成比例，而非完整 prompt。

挑战在于调度。如果两个请求共享 2,000 token 的前缀，而第三个请求只共享 200 token，你希望同时服务那两个共享长前缀的请求，使长前缀保留在 HBM 中。FCFS 的行为恰恰相反——它先服务先到达的请求，可能导致热分支在下一个长前缀请求到达前被驱逐。

## 核心概念

### Radix tree 作为 KV 索引

Radix tree（紧凑前缀树）存储 token 序列。每个节点拥有一个 token 范围以及为该范围计算的 KV blocks。子节点将序列扩展一个或多个 token。

```
root
  |- "You are a helpful assistant..."  (2,000 tokens, 124 KV blocks)
       |- "Context: <doc A>..."        (500 tokens, 31 blocks)
            |- "Question: Alice..."    (80 tokens, 5 blocks)
            |- "Question: Bob..."      (95 tokens, 6 blocks)
       |- "Context: <doc B>..."        (520 tokens, 33 blocks)
```

新请求携带系统 prompt + "Context: <doc A>" + "Question: Carol" 到达。调度器遍历：系统前缀匹配（复用 124 个 blocks），doc-A 分支匹配（复用 31 个 blocks），然后仅为 "Question: Carol" 分配新的 blocks（4 个）。预填充成本：4 个新 token 的 blocks。不使用树时：160 个 blocks。预填充节省约 40 倍。

### 缓存感知调度

如果缓存不断被替换，radix-tree 支持的复用就没有意义。两个关键策略：

1. **深度优先调度**。从队列中选择下一个请求时，优先选择与当前运行集合处于同一分支的请求。这使热分支保持固定。
2. **分支级 LRU，而非块级 LRU**。驱逐整个分支（从最久未使用的叶节点开始），而非单个块，使缓存形状匹配 radix 形状。

FCFS 违反了这两条原则。一个共享 2,000 token 的请求排在一个共享 50 token 的请求后面，然后 2,000 token 的分支被驱逐以接纳 50 token 的请求。

### 需要记住的基准数据

- Llama 3.1 8B，H100，ShareGPT 1K prompts：SGLang 约 16,200 tok/s vs vLLM 约 12,500（约 29% 优势）。
- 前缀密集型 RAG（相同系统 + 相同文档，不同问题）：SGLang 最高可达 6.4 倍。
- 语音克隆工作负载：86.4% 的前缀缓存命中率。
- SGLang 客户在生产环境中的命中率：50-99%，取决于 prompt 规范。
- 2026 年已部署超过 400,000 块 GPU。

### 排序陷阱

6.4 倍的数字依赖于一致的 prompt 模板排序。如果客户端在某些请求中将 prompt 构造为 `[system, tools, context, history, question]`，而在另一些请求中构造为 `[system, context, tools, history, question]`，树就无法找到共享前缀。人类看起来是共享前缀的内容，在 radix tree 看来是两个不同的序列。

工程师的杠杆：你的 prompt 模板就是缓存键。固定顺序。将所有不可变内容（system, tools, schemas）放在最前面。然后是检索上下文。最后是用户问题。不要将动态内容交错插入前缀中。

实际案例：将动态内容移出可缓存前缀后，一次部署的缓存命中率从 7% 提升到 74%。

### RadixAttention 的适用场景

适用场景：
- RAG（相同检索前言，不同问题）。
- Agent（相同工具 schema，不同查询）。
- 带长系统 prompt 的对话。
- 带重复前言的语音/视觉工作负载。

不适用场景（退回到 vLLM 级别的吞吐量）：
- 唯一 prompt 的一次性生成（代码补全、无系统 prompt 的开放式对话）。
- 每个请求在前缀中交错唯一内容的动态 prompt。

### 为什么这是一个调度器问题，而不仅仅是内核问题

你可以将 KV 复用作为内核技巧来实现。SGLang 的洞察是：只有当调度器保持热分支常驻时，复用才有价值。朴素的"如果可用就复用"策略会在混合负载下不断驱逐缓存。radix-tree 索引调度器正是将内核技巧转化为 29% 生产优势的关键。

### 与 vLLM 的关系

这两个系统并非严格竞争关系。2026 年 vLLM 添加了前缀缓存（`--enable-prefix-caching`）和缓存感知路由器（Rust 编写的 vLLM Router）。差距缩小但并未完全消失——SGLang 的整个栈以 radix 为核心；vLLM 是后加的。对于前缀复用主导的工作负载，SGLang 仍然是默认选择。对于没有强前缀模式的通用服务，vLLM 仍然持平或更优。

```figure
roofline
```

## 使用

`code/main.py` 实现了一个玩具 radix-tree KV cache 以及两种调度策略：FCFS 和缓存感知。对同一工作负载分别运行两种策略，报告前缀缓存命中率和吞吐量差异。然后运行"乱序"工作负载以展示 6.4 倍优势的消失。

## 交付

本课产出 `outputs/skill-radix-scheduler-advisor.md`。给定工作负载描述（prompt 模板形状、检索模式、并发租户数量），生成 prompt 排序建议以及是否采用 SGLang 的决策。

## 练习

1. 运行 `code/main.py`。在同一工作负载上比较 FCFS 和缓存感知。差异来自哪里——预填充节省、解码节省还是队列延迟？
2. 修改工作负载使 prompt 随机排列 `[system, tools, context]`。重新运行。命中率如何变化？为什么？
3. 计算在 Llama 3.1 8B 上将 2,000 token 系统 prompt 作为一个 radix 分支保持常驻的 HBM 成本。与不使用前缀复用的 16 序列批次的成本进行比较。
4. 阅读 SGLang RadixAttention 论文。用三句话解释为什么树形 LRU 驱逐在前缀密集型负载下优于块形 LRU。
5. 一位客户报告缓存命中率仅为 8%。列出三个可能的原因以及针对每个原因的诊断方法。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| RadixAttention | "SGLang 的那个东西" | KV cache 以 radix tree 索引，使共享前缀复用 blocks |
| Radix tree | "紧凑前缀树" | 每个节点拥有一个 token 范围及其 KV blocks 的树结构 |
| 缓存感知调度器 | "热分支优先" | 优先处理与常驻分支共享的请求的调度器 |
| 前缀缓存命中率 | "你的 prompt 有多少是免费的" | 由复用 KV blocks 服务的 prompt token 比例 |
| FCFS | "先到先服务" | 破坏前缀局部性的默认调度方式 |
| 分支级 LRU | "驱逐叶子节点" | 与 radix 形状匹配的驱逐策略 |
| Prompt 模板排序 | "缓存键" | Prompt 的组件顺序决定了树可以共享什么 |
| 系统 prompt 固定 | "常驻前缀" | 保持不可变的系统部分固定以避免驱逐抖动 |

## 延伸阅读

- [SGLang GitHub](https://github.com/sgl-project/sglang) — 源码和文档。
- [SGLang 文档](https://sgl-project.github.io/) — RadixAttention 和调度细节。
- [SGLang 论文 — Efficiently Programming Large Language Models (arXiv:2312.07104)](https://arxiv.org/abs/2312.07104) — 设计参考。
- [LMSYS 博客 — SGLang with RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/) — 基准数据和调度器原理。
- [vLLM — Prefix Caching](https://docs.vllm.ai/en/latest/features/prefix_caching.html) — vLLM 自己的类似 radix 的实现，供对比参考。
