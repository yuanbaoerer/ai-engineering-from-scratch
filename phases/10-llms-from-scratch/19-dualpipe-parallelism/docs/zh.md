# DualPipe 流水线并行

> DeepSeek-V3 在 2,048 块 H800 GPU 上训练，MoE 专家分散在各个节点之间。跨节点专家 all-to-all 通信的代价是每 1 GPU 小时的计算就需要 1 GPU 小时的通信。GPU 有一半时间处于空闲状态。DualPipe（DeepSeek，2024 年 12 月）是一种双向流水线，它将前向和反向计算与它们触发的 all-to-all 通信重叠起来。气泡（bubble）减少，吞吐量提升，而保留两份模型参数副本（即名称中 "dual" 的来源）在专家并行（Expert Parallelism）已经将专家分散到各个 rank 的前提下成本很低。本课以 Learn 类型的方式逐步讲解 DualPipe 实际做了什么，以及 Sea AI Lab 的 DualPipeV 改进如何在牺牲略大一点的气泡的前提下消除 2 倍参数开销。

**类型：** Learn
**语言：** Python（标准库，调度模拟器）
**前置知识：** Phase 10 · 05（分布式训练、FSDP、DeepSpeed），Phase 10 · 14（开源模型架构与 MoE）
**时间：** ~60 分钟

## 学习目标

- 说出 DualPipe 前向-反向块（chunk）的四个组成部分，以及每个部分为何拥有独立的重叠窗口。
- 解释大规模下的流水线气泡问题，以及 "无气泡（bubble-free）" 在技术上与营销上的实际含义。
- 手动追踪一个 DualPipe 调度：8 个 PP rank、16 个微批次（micro-batch），并确认前向流和反向流如何填满彼此的空闲槽位。
- 说明 DualPipeV（Sea AI Lab，2025）所做的权衡：在专家并行未激活时，以略大一点的气泡为代价，消除了 2 倍参数复制。

## 问题背景

在 2k 块 H800 GPU 上训练一个 671B 参数的 MoE 模型，会遭遇三个叠加的瓶颈：

1. **显存压力。** 每块 GPU 持有模型的一部分。在 8k 序列长度、61 层、128 个注意力头的情况下，激活值（activation）显存非常庞大。
2. **流水线气泡。** 传统流水线并行（GPipe、1F1B）会让 GPU 空闲，等待当前阶段的输入或梯度。在 8 个阶段的情况下，即使使用 1F1B 调度，大约 12% 的 GPU 时间也可能是气泡。
3. **跨节点 all-to-all。** 采用专家并行的 MoE 将专家分散到不同节点。每次前向传播都会触发一次 all-to-all 将词元（token）分发给对应的专家，以及另一次 all-to-all 将结果聚合回来。在 2k GPU 的规模下，这很容易达到 1:1 的计算-通信比。

这三个问题各自有独立的解决方案：梯度检查点（gradient checkpointing）解决显存问题，Zero Bubble（Sea AI Lab，2023）解决流水线气泡问题，专家并行通信内核解决 all-to-all 问题。DualPipe 做的是让它们协同工作。该调度在单个前向-反向块内部将计算和通信重叠，同时从流水线的两端注入微批次，并利用由此产生的调度将 all-to-all 隐藏在计算窗口中。

报告结果：流水线气泡近乎消除，DeepSeek-V3 的 14.8T 词元训练过程中 GPU 平均利用率超过 95%。

## 核心概念

### 流水线并行回顾

将一个 N 层模型拆分到 P 个设备上。设备 `i` 持有层 `i * N/P .. (i+1) * N/P - 1`。一个微批次前向流经设备 0 到 P-1，然后反向从 P-1 流回 0。每个设备只有在前一个设备发送输出后才能开始其前向阶段，只有在下游设备发送上游梯度后才能开始反向阶段。

GPipe（Huang et al., 2019）一次只调度一个微批次，浪费了大部分 GPU 时间。1F1B（Narayanan et al., 2021）交错多个微批次的前向和反向传播。Zero Bubble（Qi et al., 2023）将反向传播拆分为两部分 —— 输入梯度（B）和权重梯度（W）—— 并调度它们来填充气泡。在 Zero Bubble 之后，流水线几乎已经排满了。

DualPipe 是下一步。它在之上增加了两个思想：

### 思想 1：块分解

每个前向块被拆分为四个组成部分：

- **注意力（Attention）。** Q/K/V 投影、注意力计算、输出投影。
- **All-to-all 分发（Dispatch）。** 跨节点通信，将词元发送给对应的专家。
- **MLP。** MoE 专家计算。
- **All-to-all 聚合（Combine）。** 跨节点通信，将专家输出带回。

反向块则包含上述每个部分的梯度版本。DualPipe 对它们进行调度，使得 all-to-all 分发与下一个块的注意力计算并行执行，all-to-all 聚合与再下一个块的 MLP 计算并行执行。

### 思想 2：双向调度

大多数流水线调度从阶段 0 注入微批次，流向阶段 P-1。DualPipe 从**两端**同时注入微批次。阶段 0 看到起源于此的前向微批次；阶段 P-1 也看到起源于此的前向微批次。两股流在中间汇合。

为了实现这一点，设备 `i` 必须同时持有流水线前段的层 `i` **和** 流水线后段的层 `P - 1 - i`。这就是 DualPipe 中 "dual" 的部分：每个设备保留两份它所需服务模型层副本（每个方向一份）。在 DeepSeek-V3 的规模下，这是 2 倍的参数复制成本。但这是可承受的，因为专家并行已经将 MoE 专家分散得非常薄，将非专家层复制两份只是小开销。

关键在于，一个方向的前向流和另一个方向的反向流，恰好在一个方向调度会产生气泡的位置重叠。气泡消失了。

### 手动追踪的调度示例

考虑 P = 4 个 rank，8 个微批次，分为 4 个正向 / 4 个反向。时间从左到右流动；行代表设备 rank。

```
           Time →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

解读 "F4/F5R" 标记：rank 1 在同一时间槽内既运行微批次 4 的前向传播（在流水线中从左到右），又运行微批次 5 的前向传播（从右到左）。这就是 "双向" 在操作层面的含义。

在 rank 2 处，两股流更早重叠；在 rank 0 和 P-1 处，它们最晚重叠。在调度的稳定中间阶段，每个 rank 都在运行某个方向的前向传播，同时与另一个方向的反向传播重叠。计算处于忙碌状态。前向传播的 all-to-all 分发隐藏在反向计算中。All-to-all 聚合隐藏在前向计算中。气泡被挤压出去。

### 气泡核算

标准 1F1B 流水线气泡（每个 rank 浪费的时间）：

```
bubble_1F1B = (P - 1) * forward_chunk_time
```

Zero Bubble 改进将其降低，但没有降到零。DualPipe 在稳定阶段，如果微批次数量能被 2 倍流水线深度整除，则气泡为零。在稳定阶段之外（预热和冷却阶段），存在一些气泡，但它不随微批次数量增长 —— 这是论文强调的一个关键性质。

营销术语："无气泡（bubble-free）"。技术术语：气泡不随微批次数量增长。Sea AI Lab 的后续分析（DualPipeV / Cut-in-half）表明，只有当专家并行不是瓶颈时，才能实现完全零气泡；在 EP 驱动的 all-to-all 场景下，总会存在一些调度上的折中。

### DualPipeV —— 改进版本

Sea AI Lab（2025）观察到，当 EP 通信重叠不是重点时，2 倍参数复制是浪费的。他们的 DualPipeV 调度将双向注入折叠成一个在单份参数副本上运行的 "V 形" 调度。气泡略大于 DualPipe，但显存节省非常可观。DeepSeek 在其开源 DualPipe 实现中将 DualPipeV 作为 EP 关闭模式采用。

权衡对比：

| 特性 | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|---------|---------|-----------|------|------------|
| 每设备参数副本数 | 2 | 1 | 1 | 1 |
| 气泡与微批次关系 | 恒定 | 小幅增长 | 增长 | 增长 |
| 计算-通信重叠 | 完全 | 部分 | 最小 | 部分 |
| 适用场景 | 重度 EP 的 MoE | 稠密模型或轻量 EP | 基线 | 任何流水线 |

### 对 14.8T 词元训练运行的意义

DeepSeek-V3 的预训练在 2,048 块 H800 GPU 上消耗了约 14.8T 词元，总计约 280 万 GPU 小时。如果使用朴素的 1F1B，他们会因流水线气泡损失 12-15% 的时间 —— 34 万到 42 万 GPU 小时，足以训练一个完整的 70B 模型。DualPipe 回收了其中的大部分。没有内部日志很难直接量化其贡献，但论文中的声明是训练期间 GPU 平均利用率超过 95%。

对于较小规模的运行（少于 1k GPU），DualPipe 是大材小用 —— 流水线气泡相对于总成本较小，而且稠密模型训练很少遇到 all-to-all 瓶颈。对于多千 GPU 规模的前沿 MoE 训练，它实际上是必需的。

### 在技术栈中的位置

- 与 **FSDP**（Phase 10 · 05）互补。FSDP 将模型参数分片到各个 rank；DualPipe 将计算调度到各个 rank。两者可以结合使用。
- 兼容 **ZeRO-3** 梯度分片。双副本复制的簿记工作需要与 ZeRO 的分片梯度配合。
- 需要针对特定集群拓扑调优的**自定义 all-to-all 内核**。DeepSeek 的开源内核是参考实现。

## 使用它

`code/main.py` 是一个流水线调度模拟器。它接受 `(P, n_micro_batches, schedule)` 参数，并打印 1F1B、Zero Bubble、DualPipe 和 DualPipeV 在稳定阶段的利用率。它是一个教学工具 —— 数字与论文中的定性声明一致，不代表生产环境中实测的加速比。

模拟器的价值：用不同的 P 和微批次数量运行它，观察气泡比例如何对 1F1B 增长，但对 DualPipe 不增长。

真实训练运行的集成注意事项：

- 选择一个能整除微批次数量的流水线并行深度。
- 确保你的专家并行网格支持双向 all-to-all。DeepSeek 的内核是参考实现。
- 首次使用时，预计要花一周时间调试调度本身。簿记工作非常繁琐。
- 监控每个 rank 的 GPU 利用率，而不仅仅是总体利用率。DualPipe 的收益来自于收紧拖后腿的 rank。

## 交付它

本课产出 `outputs/skill-dualpipe-planner.md`。给定一个训练集群规格（GPU 数量、拓扑、互连、模型结构），它会推荐流水线并行策略、使用的调度算法，以及目标规模下的预期气泡比例。

## 练习题

1. 在 `(P=8, micro_batches=16, schedule=dualpipe)` 和 `(P=8, micro_batches=16, schedule=1f1b)` 上运行 `code/main.py`。计算 GPU 利用率差异，并以每百万词元训练所回收的 GPU 小时数表示。

2. 手动绘制 `(P=4, micro_batches=8, schedule=dualpipe)` 的调度表。在每个时间槽标记微批次 ID 和方向。找出第一个没有气泡的时间槽。

3. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）的图 5。找出 DualPipe 前向块中 all-to-all 分发的重叠窗口。解释计算调度如何将其隐藏。

4. 计算 DualPipe 的 2 倍参数开销：一个 70B 稠密模型，P=8 个流水线阶段；以及一个 671B MoE 模型，P=16 个流水线阶段。说明为什么 MoE 情况下的开销比例更小（大部分参数是专家，分片在一个很大的 EP 组中）。

5. 将 DualPipe 与 Chimera（2021 年的一个竞争双向调度器）进行比较。找出 DualPipe 新增而 Chimera 没有的两个具体特性，以论文第 3.4 节为参考。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 流水线气泡（Pipeline bubble） | "每个 rank 的空闲时间" | 流水线阶段等待输入或梯度时浪费的 GPU 周期 |
| 1F1B | "默认流水线调度" | 一个前向 / 一个反向交错调度；DualPipe 击败的基线 |
| Zero Bubble | "Sea AI Lab 2023" | 将反向传播拆分为 B（输入梯度）和 W（权重梯度）；几乎完全收紧流水线 |
| DualPipe | "DeepSeek-V3 调度" | 双向流水线 + 计算-通信重叠；气泡不随微批次数量增长 |
| DualPipeV | "Cut-in-half" | V 形改进，以略大一点的气泡为代价消除 2 倍参数复制 |
| 块（Chunk） | "流水线工作单元" | 一个微批次通过一个流水线阶段的前向或反向传播 |
| All-to-all 分发（All-to-all dispatch） | "将词元发送给专家" | 将词元路由到其分配的 MoE 专家的跨节点通信 |
| All-to-all 聚合（All-to-all combine） | "将专家输出带回" | MLP 之后收集专家输出的跨节点通信 |
| 专家并行（Expert Parallelism, EP） | "专家分散在 GPU 上" | 将 MoE 专家分片到各个 rank，不同 GPU 持有不同专家 |
| 流水线并行（Pipeline Parallelism, PP） | "层分散在 GPU 上" | 将模型层分片到各个 rank；DualPipe 调度的维度 |
| 气泡比例（Bubble fraction） | "浪费的 GPU 时间" | （气泡时间 / 总时间）；DualPipe 将其趋近于零的比例 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437), Section 3.3.2 and Figure 5](https://arxiv.org/abs/2412.19437) —— DualPipe 的主要参考文献
- [DeepSeek — DualPipe GitHub repository](https://github.com/deepseek-ai/DualPipe) —— 开源参考实现，包括 DualPipeV（Cut-in-half）模式
- [Qi et al. — Zero Bubble Pipeline Parallelism (arXiv:2401.10241, Sea AI Lab 2023)](https://arxiv.org/abs/2401.10241) —— Zero Bubble 的前身
- [Sea AI Lab — DualPipe could be better without the Dual](https://sail.sea.com/blog/articles/63) —— 影响 DeepSeek EP 关闭模式的 DualPipeV 分析
- [Narayanan et al. — PipeDream / 1F1B (arXiv:1806.03377, 2018-2021)](https://arxiv.org/abs/1806.03377) —— DualPipe 对比的 1F1B 调度
- [Huang et al. — GPipe (arXiv:1811.06965, 2018)](https://arxiv.org/abs/1811.06965) —— 原始流水线并行论文和气泡问题
