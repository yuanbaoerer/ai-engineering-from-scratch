# 梯度检查点与激活重计算

> 反向传播会保留每一个中间激活值。对于一个 700 亿参数、128K 上下文长度的模型，每个 rank 的激活值高达 3 TB。检查点技术用计算换内存：重计算而非保存。问题在于丢弃哪些段，而答案不是"全部"。

**类型：** 构建
**语言：** Python（使用 numpy，可选 torch）
**前置知识：** 第 10 阶段第 04 课（预训练 Mini-GPT）、第 10 阶段第 05 课（扩展与分布式）
**时间：** 约 70 分钟

## 问题所在

训练 Transformer 时，每一层都需要存储每个可微操作的输入：注意力输入、Q/K/V 投影、softmax 输出、FFN 输入、归一化输出，以及残差流。对于隐藏维度为 `d`、序列长度为 `L`、批次大小为 `B` 的一层，每层的浮点数数量级约为 `12 * B * L * d`。

当 `d=8192, L=8192, B=1` 时，BF16 格式下每层为 800 MB。一个 64 层模型的激活值就达到 51 GB——这还没乘以微批次大小，还没加上注意力 softmax 的中间值（每个头 `L^2`），也没考虑张量并行的部分副本。

这是双面的账单：BF16 权重加优化器状态可能刚好装进 80GB，但激活值会让你超限。梯度检查点（又称激活重计算）是标准解决方案。丢弃大部分激活值；在反向传播时重新执行前向传播来恢复它们。代价：额外的 FLOPs。收益：内存按检查点段数与总层数的比例下降。

如果做得简单粗暴，检查点每步大约增加 33% 的前向传播 FLOPs。如果做得好——按照 Korthikanti 等人的"智能选择"进行选择性检查点——你能以不到 5% 的 FLOP 开销节省 5 倍内存。而在 FP8 矩阵乘法、FSDP 卸载和专家并行 MoE 的背景下，这一点至关重要：你既负担不起内存，也浪费不起计算。

## 核心概念

### 反向传播到底需要什么

`output = layer(input)`。反向传播需要 `grad_input` 和 `grad_params`。为了计算它们，它需要：

- `input`（用于计算线性层的 `grad_params = input.T @ grad_output`）
- 一些激活导数中间值（ReLU/GELU/softmax 的导数依赖于激活值本身）

前向传播会自动在自动求导图中存储这些值。每个 `tensor.retain_grad()` 和每个需要输入的操作都会保留一个引用。

### 朴素全量检查点

将网络分成 `N` 个段。在前向传播期间，只存储每个段的*输入*。当反向传播需要中间值时，重新运行该段的前向传播来实例化它们，然后进行微分。

示例：32 层 Transformer 分成 32 个段，每段 1 层。

- 内存：32 个层输入（小）对比 32 *（每层激活体积）（巨大）。
- 额外计算：每个段多一次前向传播，即总共约多 33% 的前向 FLOPs（因为反向传播是前向的 2 倍，完整步骤变成 1 + 1 + 2 = 4 个单位，而不是 1 + 2 = 3）。

这是 Chen 等人 2016 年的原始方案：每 `sqrt(L)` 层一个检查点，以平衡内存和计算。对于 L=64，就是 8 个检查点。

### 选择性检查点（Korthikanti 2022）

并非所有激活值的存储成本都相同。注意力 softmax 输出是 `B*L*L*heads`，随序列长度*二次方*增长。FFN 隐藏激活是 `B*L*4d`，线性增长。对于长序列，softmax 占主导。

选择性检查点保留存储成本低的激活值（线性投影、残差），只重计算昂贵的那些（注意力）。你付出极小的重计算 FLOPs，却节省了 O(L^2) 的内存。

Megatron-Core 将其实现为"选择性"激活重计算。用于大多数 2024 年以后的前沿训练任务中。

### 卸载

重计算的替代方案：在前向和反向传播之间将激活值传输到 CPU 内存。需要 PCIe 带宽；当空闲带宽超过实例化成本时更有利。混合策略很常见：检查点一些层，卸载另一些。

FSDP2 将卸载作为一等选项提供。当 GPU 受内存瓶颈限制但 CPU-GPU 传输有剩余容量时，卸载效果显著。

### 重计算成本模型

每步 FLOPs，朴素检查点每 `k` 层（共 `L` 层）：

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # 段内每层多一次前向传播
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

使用选择性检查点时，你只重计算注意力核，而非整层：

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### 内存节省模型

每层激活体积：`A`。对于 `L` 层，总激活内存：`L * A`。

全量检查点（段大小为 1）：只存储 `L * input_volume`（标准 Transformer 下约为 `L * 1/10 A`）。节省约 `9 * L * A * 1/10`。

每 `k` 层检查点：存储 `L/k * A` 加上活动段内 `k-1` 层的值。

当 `k = sqrt(L)` 时，内存和重计算成本都按 `sqrt(L)` 缩放——对于均匀成本层，这是最优权衡。

### 何时不该检查点

- 流水线阶段中最内层已经在运行的层。它们反正要完成。
- 如果首层和末层占阶段计算主导（在 Transformer 中很少见）。
- 已经使用 FlashAttention 的注意力核——Flash 已经快速重计算了 softmax，所以额外的层级检查点收益甚微。

### 实现模式

1. **函数包装器：** 用 `torch.utils.checkpoint.checkpoint(fn, input)` 包装一段。PyTorch 只存储 `input`，反向传播时重计算其他一切。

2. **基于装饰器：** 将层标记为可检查点；训练器在配置时决定哪些段被包装。

3. **手动显式重计算：** 自己写反向传播，调用自定义的 `recompute_forward`，用存储的输入复制前向传播。

三种方式功能结果相同。包装器是标准写法。

### 与 TP / PP / FP8 的交互

- **张量并行：** 检查点输入必须在重计算时聚合或重新分散；要考虑通信成本。
- **流水线并行：** 典型模式是检查点每个流水线阶段的前向传播，以便逆序微批次可以复用激活内存。
- **FP8 重计算：** 重计算期间更新的 amax 历史必须与原始前向传播匹配，否则 FP8 尺度会漂移。大多数框架会快照尺度。

## 动手构建

### 步骤 1：带段的分段玩具模型

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### 步骤 2：需要所有激活值的朴素反向传播

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### 步骤 3：每 k 层检查点的内存

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### 步骤 4：成本模型

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### 步骤 5：内存估算器

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### 步骤 6：最优段大小

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### 步骤 7：选择性检查点决策

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## 实际应用

- **torch.utils.checkpoint**：`from torch.utils.checkpoint import checkpoint` —— PyTorch 中的标准包装器。包装一个函数；只存储输入，反向传播时重计算。
- **Megatron-Core 激活重计算**：支持 `selective`（选择性）、`full`（全量）和 `block`（块）模式。2024 年以后前沿训练的标准配置。
- **FSDP2 卸载**：`module.to_empty(device="cpu")` 配合 FSDP2 中的 `offload_policy` 将激活值分片到 CPU 而非重计算。
- **DeepSpeed ZeRO-Offload**：用于优化器状态和激活值的 CPU 卸载，与检查点互为补充。

## 交付成果

本课产出 `outputs/prompt-activation-recompute-policy.md` —— 一个提示词，接收你的模型配置（层数、隐藏维度、序列长度、批次大小）和可用 GPU 内存，输出每层重计算策略（无 / 选择性 / 全量 / 卸载）。

## 练习题

1. 验证正确性。运行 `model_forward` + `model_backward`（全量激活）对比 `model_forward_checkpointed` + `model_backward_checkpointed`（分段）。参数梯度必须在机器精度下完全相同。

2. 将段大小 `k` 从 1 扫到 `L`。绘制 FLOP 开销和内存曲线。找到曲线的拐点。

3. 实现选择性检查点：存储注意力模块的输入但不存储其中间值。测量 32 层模型在 seq=8192 时相对于全层检查点的 FLOP 开销。

4. 添加卸载。将段输入保存到模拟的"CPU 缓冲区"（一个单独的列表）。将"PCIe 带宽"测量为字节/时间，找到卸载与重计算之间的盈亏平衡点。

5. 用和不用 `torch.utils.checkpoint` 对一个真实 PyTorch Transformer 进行基准测试。测量内存（通过 `torch.cuda.max_memory_allocated`）和步进时间。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| 梯度检查点 (Gradient checkpointing) | "通过重做前向传播来节省内存" | 只存储段输入；反向传播期间重计算中间值以获取梯度支持张量 |
| 激活重计算 (Activation recomputation) | "和检查点一样" | 同一技术的高性能计算风格名称 |
| 段大小 (k) | "每个检查点多少层" | 中间值被丢弃并一起实例化的层数 |
| 选择性检查点 (Selective checkpointing) | "Korthikanti 的技巧" | 只重计算存储昂贵的激活值（注意力 softmax）；保留便宜的那些 |
| 全量检查点 (Full checkpointing) | "朴素版本" | 在每个段中重计算每层的中间值 |
| 块检查点 (Block checkpointing) | "粗粒度" | 检查点整个 Transformer 块；最大粒度 |
| FLOP 开销 (FLOP overhead) | "计算税" | 每步额外 FLOPs =（重计算 FLOPs）/（前向 + 反向 FLOPs）；朴素 33%，选择性 5% |
| 激活卸载 (Activation offload) | "运到 CPU" | 在前向->反向传播之间将激活值移到 CPU 内存；重计算的替代方案 |
| sqrt-L 规则 (sqrt-L rule) | "经典最优解" | 对于均匀成本层，最优检查点间隔为 sqrt(L) 层 |
| 注意力-softmax 体积 (Attention-softmax volume) | "O(L^2) 问题" | L^2 * heads * batch 个浮点数；在长上下文中主导激活内存 |

## 延伸阅读

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- 形式化梯度检查点的原始论文
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) -- 选择性激活重计算及形式化成本分析
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) -- 通过反向模式实例化实现恒定内存的替代方法
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) -- 大规模激活卸载
- [PyTorch torch.utils.checkpoint 文档](https://pytorch.org/docs/stable/checkpoint.html) -- 标准 API
- [Megatron-Core 激活重计算文档](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) -- 选择性、全量和块模式
