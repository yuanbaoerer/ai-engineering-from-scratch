# KV 缓存、Flash Attention 和推理优化

> 训练是并行且 FLOP 受限的。推理是串行且内存受限的。不同的瓶颈，不同的技巧。

**类型:** 构建
**语言:** Python
**前置知识:** 第 7 阶段 · 02（自注意力）、第 7 阶段 · 05（完整 Transformer）、第 7 阶段 · 07（GPT）
**时间:** 约 75 分钟

## 问题所在

朴素自回归解码器生成 `N` 个 token 需要 `O(N²)` 的工作量：每一步对整个前缀重新计算注意力。对于 4K token 的响应，那是 1600 万次注意力操作，其中大部分是冗余的。前缀 token 的每个隐藏状态一旦计算就是确定的——你只需要对缓存的键和值运行新 token 的查询。

除此之外，注意力本身移动大量数据。标准注意力物化 N×N 分数矩阵、N×d softmax 输出、N×d 最终输出——对 HBM 的读写太多。对于 N≥2K，注意力在成为 FLOP 受限之前就变成内存受限。经典注意力核对现代 GPU 的利用率低 4-10 倍。

两个优化，都来自 Dao et al.，将前沿推理从"慢"推到"快"：

1. **KV 缓存。** 存储每个前缀 token 的 K 和 V 向量。每个新 token 的注意力是对缓存键的一次查询。推理从 `O(N²)` 减少到每生成步 `O(N)`。
2. **Flash Attention。** 将注意力计算分块，使完整的 N×N 矩阵永远不触碰 HBM。所有 softmax + 矩阵乘法在 SRAM 中完成。在 A100 上 2-4 倍墙钟加速；在 H100 FP8 上 5-10 倍。

到 2026 年两者都是通用的。每个生产推理栈（vLLM、TensorRT-LLM、SGLang、llama.cpp）都假设它们。每个前沿模型都启用 Flash Attention。

## 核心概念

![KV 缓存增长和 Flash Attention 分块](../assets/kv-cache-flash-attn.svg)

### KV 缓存数学

每解码器层，每 token，每头：

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K 和 V
```

对于 7B 模型，32 层、32 头、d_head=128、fp16：

```
每 token 每层 = 2 * 128 * 2 = 512 字节
每 token (32 层) = 16 KB
每 32K 上下文 = 512 MB
```

对于 Llama 3 70B（80 层、d_head=128、GQA 8 个 KV 头）：

```
每 token 每层 = 2 * 8 * 128 * 2 = 4096 字节 (4 KB)
每 32K 上下文 = 10.4 GB
```

这 10 GB 就是为什么 Llama 3 70B 在 128K 上下文下，批大小为 1 时，KV 缓存就占了 40 GB A100 的大部分。

**GQA 是 KV 缓存的胜利。** 64 头的 MHA 将是 32 GB。MLA 压缩更多。

拖动维度，观察缓存大小变化。推高序列长度或批次大小，看看它多快突破单 GPU：

```figure
kv-cache-sizer
```

### Flash Attention——分块技巧

标准注意力：

```
S = Q @ K^T          (HBM 读, N×N, HBM 写)
P = softmax(S)       (HBM 读, HBM 写)
O = P @ V            (HBM 读, HBM 写)
```

三次 HBM 往返。在 H100 上，HBM 带宽 3 TB/s；SRAM 30 TB/s。每次 HBM 往返比保持所有数据在片上慢 10 倍。

Flash Attention：

```
对每个 Q 块 (块大小 ~128 × 128):
    将 Q_tile 加载到 SRAM
    对每个 K, V 块:
        将 K_tile, V_tile 加载到 SRAM
        计算 S_tile = Q_tile @ K_tile^T     (SRAM)
        运行 softmax 聚合                   (SRAM)
        累积到 O_tile                        (SRAM)
    将 O_tile 写入 HBM
```

每块一次 HBM 往返。总内存占用从 `O(N²)` 降到 `O(N)`。反向传播重新计算前向传播的一些值而非存储它们——又一次内存节省。

**数值技巧。** 运行 softmax 在块间维护 `(max, sum)`，使最终归一化是精确的。不是近似——Flash Attention 计算与标准注意力逐位相同的输出（fp16 非结合性除外）。

**版本演进：**

| 版本 | 年份 | 关键变化 | 参考硬件加速 |
|------|------|----------|-------------|
| Flash 1 | 2022 | 分块 SRAM 核 | A100 上 2 倍 |
| Flash 2 | 2023 | 更好并行，因果优先排序 | A100 上 3 倍 |
| Flash 3 | 2024 | Hopper 异步，FP8 | H100 上 1.5-2 倍（约 740 TFLOPs FP16） |
| Flash 4 | 2026 | Blackwell 5 级流水线，软件 exp2 | 推理优先（初始仅前向） |

Flash 4 发布时仅支持前向传播。训练仍使用 Flash 3。Flash 4 的 GQA 和 varlen 支持待定（2026 年中）。

### 推测解码——另一个延迟胜利

廉价模型提出 N 个 token。大模型并行验证所有 N 个。如果验证接受 k 个 token，你为 k 次生成付了 1 次大模型前向传播。典型 k=3-5 在代码和散文上。

2026 年默认：
- **EAGLE 2 / Medusa。** 集成草稿头共享验证器的隐藏状态。无质量损失 2-3 倍加速。
- **草稿模型推测解码。** 消费级硬件上 2-4 倍加速。
- **前向解码。** Jacobi 迭代；无需草稿模型。小众但免费。

### 连续批处理

经典批处理推理：等待最慢的序列完成，然后开始新批次。短响应提前完成时浪费 GPU。

连续批处理（首先在 Orca 中发布，现用于 vLLM、TensorRT-LLM、SGLang）：旧请求完成后立即将新请求换入批次。典型聊天工作负载 5-10 倍吞吐量提升。

### PagedAttention——KV 缓存作为虚拟内存

vLLM 的头条特性。KV 缓存在 16 token 块中分配；页表将逻辑位置映射到物理块。让你在并行样本（束搜索、并行采样）间共享 KV，热交换前缀用于提示缓存，以及碎片整理内存。比朴素连续分配 4 倍吞吐量提升。

```figure
flash-attention-memory
```

## 动手实现

参见 `code/main.py`。我们实现：

1. 朴素 `O(N²)` 增量解码器。
2. `O(N)` KV 缓存解码器。
3. 模拟 Flash Attention 运行最大值算法的分块 softmax。

### 第一步：KV 缓存

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

简单：在每层、每头的列表中持续增长每 token 的 K、V 向量。

### 第二步：分块 softmax

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention 风格的 softmax(qK^T)V，带运行最大/和。"""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

与一次性 `softmax(qK) V` 逐位相同的输出，但任意时刻工作集是 `tile × d_head` 块，不是完整的 `N × d_head`。

### 第三步：比较朴素 vs 缓存解码在 100 token 生成上

计算注意力操作数。朴素：`O(N²)` = 5050。缓存：`O(N)` = 100。代码打印两者。

## 使用场景

```python
# HuggingFace transformers 在仅解码器 generate() 上自动启用 KV 缓存。
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # Hopper 用 FA3
    torch_dtype="bfloat16",
)
# generate() 自动使用 KV 缓存
```

vLLM 生产：

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

跨请求的提示前缀缓存是 2026 年的大胜利——相同的系统提示、少样本示例或长上下文文档在调用间复用 KV。对于重复工具提示的代理工作负载，前缀缓存通常带来 5 倍吞吐量提升。

## 交付使用

参见 `outputs/skill-inference-optimizer.md`。该技能为新推理部署选择注意力实现、KV 缓存策略、量化和推测解码。

## 练习

1. **简单。** 运行 `code/main.py`。确认朴素和缓存解码器产生相同输出；注意操作数差异。
2. **中等。** 实现提示前缀缓存：给定提示 P 和多个补全，对 P 运行一次前向传播填充 KV 缓存，然后按补全分支。测量 vs 对每个重新编码 P 的加速。
3. **困难。** 实现玩具 PagedAttention：固定 16 token 块的 KV 缓存带空闲列表。序列完成后，将其块返回池中。模拟 1,000 次不同长度的聊天补全。比较碎片化 vs 连续分配。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| KV 缓存 | "让解码变快的技巧" | 存储每个前缀 token 的 K 和 V；新查询关注它们而非重新计算。 |
| HBM | "GPU 主存" | 高带宽内存；H100 上 80 GB，B200 上 192 GB。约 3 TB/s 带宽。 |
| SRAM | "片上内存" | 每 SM 快速内存，H100 上每 SM 约 256 KB。约 30 TB/s 带宽。 |
| Flash Attention | "分块注意力核" | 在不物化 N×N 到 HBM 的情况下计算注意力。 |
| 连续批处理 | "无等待批处理" | 将完成的序列换出，新的换入，不排空批次。 |
| PagedAttention | "vLLM 的头条" | KV 缓存在固定块中分配带页表；消除碎片化。 |
| 提示前缀缓存 | "复用长提示" | 跨请求缓存共享前缀的 KV；代理的主要成本削减。 |
| 推测解码 | "草稿 + 验证" | 廉价草稿模型提出 token；大模型一次验证 k 个。 |

## 延伸阅读

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 1。
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Flash 2。
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Flash 3。
- [FlashAttention-4 release notes (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention) — Blackwell 5 级流水线和软件 exp2 技巧；阅读 repo README 了解本课提到的仅前向发布注意事项。
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM 论文。
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 推测解码。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1/2 论文用于本课引用的集成草稿方法。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — 与 EAGLE 一起引用的 Medusa 方法。
- [vLLM docs — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) — 16 token 块和页表设计的权威深入讲解。
