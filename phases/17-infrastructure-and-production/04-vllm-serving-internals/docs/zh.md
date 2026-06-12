# vLLM 推理内部机制：PagedAttention、Continuous Batching、Chunked Prefill

> vLLM 在 2026 年的主导地位建立在三个相互叠加的默认特性上，而非单一技巧。PagedAttention 始终开启。Continuous batching 在解码迭代之间向活跃批次注入新请求。Chunked prefill 将长提示切片，使解码 token 永远不会饥饿。三者同时开启时，一块 H100 SXM5 上的 Llama 3.3 70B FP8 在 128 并发下可推 2,200-2,400 tok/s——大约比 vLLM 自身默认高 25%，是朴素 PyTorch 循环的 3-4 倍。本课以你能画图的深度阅读调度器和注意力内核，并以 `code/main.py` 中的玩具 continuous batcher 结尾，它以 vLLM 的方式调度 prefill 和 decode。

**类型：** 学习
**语言：** Python（stdlib，玩具级 continuous batching 调度器）
**前置课程：** 阶段 17 · 01（模型推理），阶段 11（LLM 工程）
**时间：** 约 75 分钟

## 学习目标

- 解释 PagedAttention 作为 KV cache 分配器：块、块表，以及为什么在生产负载下碎片率保持在 4% 以下。
- 在迭代级别画出 continuous batching 图：完成的序列如何离开批次，新的如何加入而不排空。
- 用一句话描述 chunked prefill，并说出它保护的延迟指标（提示：是 TTFT 尾部，而非平均吞吐）。
- 说出 2026 年 vLLM v0.18.0 中同时启用所有优化时会踩的坑。

## 问题背景

朴素 PyTorch 推理循环一次运行一个请求：tokenize、prefill、解码直到 EOS、返回。一个用户时可行。一百个用户时，就是一队耐心等待的人。明显的修复——静态批处理——将每个请求填充到窗口中最长提示的长度，将每个解码填充到最长预期输出的长度，并在最慢的序列上阻塞整个批次。你为从未使用的填充付费，快速请求等待慢速请求。

vLLM 同时解决三个问题。PagedAttention 防止 KV cache 碎片吞噬 60-80% 的 GPU 内存——经典连续分配方式就是这样。Continuous batching 让请求在每次解码迭代之间加入和离开批次，因此批次始终充满真实工作。Chunked prefill 将 32k-token 提示切成约 512-token 的切片与解码交错，因此长提示不会冻结 GPU 上的每个解码 token。

2026 年的生产默认是三者全开。你需要理解每个的作用，因为故障模式全在调度器上，不在模型上。

## 核心概念

### PagedAttention 作为虚拟内存系统

KV cache 每个序列为 `num_layers × 2 × num_heads × head_dim × seq_len × bytes_per_element`。Llama 3.3 70B 在 8192 token 时，每个序列在 BF16 下约 1.25 GB。如果你为每个请求预分配 8192 个槽位但平均请求只使用 1500 token，你浪费了约 82% 的预留 HBM。经典批处理付出这个浪费。

PagedAttention 借用操作系统虚拟内存的思想。KV cache 不是每个序列连续的。它以固定大小的块分配（默认 16 token）。每个序列有一个块表，将逻辑 token 位置映射到物理块 ID。当序列增长超过已分配的块时，添加一个块。完成时，块返回池中。

碎片率从 60-80%（经典）降至 4% 以下（PagedAttention）。你不是用标志启用 PagedAttention——它是 vLLM 唯一的分配器。调节旋钮是 `--gpu-memory-utilization`（默认 0.9），它告诉 vLLM 在加载权重和激活后为 KV 块预留多少 HBM。

### 迭代级别的 continuous batching

旧的"动态批处理"等待一个窗口（比如 10 ms）填充批次，然后运行 prefill + decode + decode + decode 直到每个序列完成。快速序列提前离开并空闲，而 GPU 完成慢速序列。

Continuous batching 在每次解码步骤之间操作。将运行中的序列集合称为 `RUNNING` 列表。每次迭代：

1. `RUNNING` 中刚碰到 EOS 或 max_tokens 的序列被移除。
2. 调度器查看等待队列。如果有空闲 KV 块，它接纳新序列（prefill 或恢复）。
3. 前向传播在当前 `RUNNING` 中的内容上运行，每个序列发出一个新 token。

批次大小永远不会填充到固定数字。处于输出不同位置的序列共享一个融合前向传播。2026 年 vLLM 中这叫 `V1 scheduler`。关键不变式：调度器每次解码迭代运行一次，而非每个请求一次。

### Chunked prefill 保护 TTFT 尾部

Prefill 是计算受限的。Llama 3.3 70B 上 32k-token 提示在一块 H100 上需要约 800 ms 纯 prefill 时间。Prefill 运行时，批次中每个其他序列的解码 token 在等待。在推理循环中，一个长提示的首 token 延迟（TTFT）变成其他数十个用户的 token 间延迟（ITL）抖动。

Chunked prefill 将 prefill 切成固定大小的块（默认 512 token），每个块作为一个调度单元。在块之间，调度器可以将解码序列推进一个 token。你用少量绝对 prefill 延迟损失（每块几毫秒）换取更低的解码时抖动。发布基准中混合负载下的 P99 ITL 从约 50 ms 降至约 15 ms。

### 三个默认特性相互作用

三个特性互相假设。PagedAttention 给调度器一个细粒度的 KV 资源来交易。Continuous batching 需要这个细粒度资源，这样接纳新序列不会强制全局重排。Chunked prefill 是调度器在同一个 `RUNNING` 列表上做的决策——它是又一个调度策略，不是独立系统。

你不需要知道每个标志。你需要知道调度器优化什么：在 KV 块预算下的 goodput，受 chunked prefill 切片约束。

### 2026 年 v0.18.0 的坑

在 vLLM v0.18.0 中，你不能将 `--enable-chunked-prefill` 与 draft model 投机解码（`--speculative-model`）组合使用。文档记录的例外是 V1 调度器中的 N-gram GPU 投机解码。不读发布说明就打开每个标志的团队会在启动时遇到运行时错误，而非软回归。如果你的投机收益值得启用 chunked prefill，请重新考虑——2026 年的正确答案通常是 EAGLE-3 不带 chunked prefill，而非 draft model 加上不能编译的 chunked prefill。

### 你应该记住的数字

- Llama 3.3 70B FP8，H100 SXM5，128 并发，三者全开：2,200-2,400 tok/s。
- 同模型，默认 vLLM（无 chunked prefill）：约 1,800 tok/s。
- 同模型，朴素 PyTorch 前向循环：约 600 tok/s。
- 生产负载下 PagedAttention 的 KV 碎片浪费：<4%。
- 混合负载下 P99 ITL：有 chunked prefill 约 15 ms，无则约 50 ms。

### 调度器长什么样

```
while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # schedule prefill chunks + decode in one batch
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # e.g. 512 tokens
        else:
            batch.append(decode_one_token(s))     # 1 token

    run_forward(batch)                            # one fused GPU call
```

`code/main.py` 正是这个循环的 stdlib Python 实现，带假 token 计数和假前向延迟。运行它可以看到 chunked prefill 如何在长 prefill 期间保持解码序列存活。

```figure
tensor-parallel
```

## 使用

`code/main.py` 模拟可切换特性的 vLLM 风格调度器。运行它可以看到：

- `NAIVE` 模式：一次一个请求，无批处理。
- `STATIC` 模式：填充并等待，经典批处理。
- `CONTINUOUS` 模式：迭代级接纳和释放。
- `CONTINUOUS + CHUNKED` 模式：prefill 切片与解码交错。

输出显示总吞吐量（每虚拟秒 token 数）、TTFT 均值和 P99 ITL。`CONTINUOUS + CHUNKED` 行应在混合流量上占优。

## 交付

本课产出 `outputs/skill-vllm-scheduler-reader.md`。给定推理配置（批次大小、KV 内存利用率、chunked prefill 大小、投机配置），产出调度器诊断，指出三个默认特性中哪个是瓶颈以及如何调整。

## 练习

1. 运行 `code/main.py`。在混合短请求和长请求的工作负载上比较 `STATIC` 和 `CONTINUOUS`。吞吐量差距来自哪里——prefill 效率、decode 效率还是尾部延迟？
2. 修改玩具调度器添加 `--max-num-batched-tokens`。对于运行 Llama 3.3 70B FP8 的 H100，正确的值是多少？（提示：它是 KV 块大小和空闲块数量的函数，而非原始 HBM。）
3. 重读 vLLM v0.18.0 发布说明。哪些标志组合互斥？列出它们。
4. 计算 1,000 个请求的 trace 的 KV cache 碎片浪费，平均 1,500 输出 token，标准差 600 token，在 (a) 每请求连续分配最大 8192，(b) 16-token 块的 PagedAttention 下。
5. 用一段话解释为什么 chunked prefill 有助于 P99 ITL 但单独不提升吞吐量。实践中吞吐量提升来自哪里？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| PagedAttention | "KV 技巧" | 固定大小块分配器用于 KV cache；碎片 <4% |
| 块表 | "页表" | 每序列从逻辑 token 位置到物理 KV 块的映射 |
| Continuous batching | "动态批处理，但对了" | 每次解码迭代做接纳/释放决策 |
| Chunked prefill | "prefill 分割" | 将长 prefill 切成 512-token 切片与解码交错 |
| TTFT | "首 token 时间" | prefill + 排队 + 网络；长提示时由 prefill 主导 |
| ITL | "token 间延迟" | 连续解码 token 之间的时间；由批次大小主导 |
| Goodput | "满足 SLO 的吞吐" | 每个请求仍命中 TTFT 和 ITL 目标的 token/sec |
| V1 scheduler | "新调度器" | vLLM 2026 调度器；N-gram spec decode 是 chunked-prefill 兼容路径 |
| `--gpu-memory-utilization` | "内存旋钮" | 权重和激活后为 KV 块预留的 HBM 比例 |

## 延伸阅读

- [vLLM documentation — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/) — 关于 chunked-prefill 和投机解码兼容性的权威来源。
- [vLLM Release Notes (NVIDIA)](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html) — 2026 发布节奏和版本特定行为。
- [vLLM Blog — PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) — 仍然定义如何看待分配器的原始文章。
- [PagedAttention paper (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180) — 碎片分析和调度器设计。
- [Aleksa Gordic — Inside vLLM](https://www.aleksagordic.com/blog/vllm) — 详细的 V1 调度器演练，带火焰图。
