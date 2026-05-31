# 混合专家 (MoE)

> 密集 70B Transformer 对每个 token 激活每个参数。671B MoE 对每个 token 只激活 37B，并在每个基准上击败它。稀疏性是这十年最重要的扩展思想。

**类型:** 构建
**语言:** Python
**前置知识:** 第 7 阶段 · 05（完整 Transformer）、第 7 阶段 · 07（GPT）
**时间:** 约 45 分钟

## 问题所在

密集 Transformer 的推理 FLOPs 等于其参数量（乘以 2 用于前向传播）。扩大密集模型，每个 token 付全部账单。到 2024 年，前沿触碰计算墙：要有意义地更智能，你需要每 token 指数级更多 FLOPs。

混合专家打破了这种联系。将每个 FFN 替换为 `E` 个独立专家 + 一个路由器，为每个 token 选 `k` 个专家。总参数 = `E × FFN_size`。每 token 活跃参数 = `k × FFN_size`。典型 2026 年配置：`E=256`，`k=8`。存储随 `E` 扩展，计算随 `k` 扩展。

2026 年前沿几乎全是 MoE：DeepSeek-V3（671B 总 / 37B 活跃）、Mixtral 8×22B、Qwen2.5-MoE、Llama 4、Kimi K2、gpt-oss。在 Artificial Analysis 的独立排行榜上，前 10 开源模型全是 MoE。

## 核心概念

![MoE 层：路由器为每个 token 从 E 个专家中选 k 个](../assets/moe.svg)

### FFN 替换

密集 Transformer 块：

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE 块：

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # 每个 token 从 E 中选 k
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

每个专家是独立的 FFN（通常是 SwiGLU）。路由器是单个线性层。每个 token 选择自己的 `k` 个专家，获得其输出的门控混合。

### 负载均衡问题

如果路由器将 90% 的 token 送到专家 3，其他专家就饿死了。三种修复方案已被尝试：

1. **辅助负载均衡损失** (Switch Transformer, Mixtral)。添加与专家使用方差成比例的惩罚。有效，但增加超参数和第二个梯度信号。
2. **专家容量 + token 丢弃** (早期 Switch)。每个专家最多处理 `C × N/E` 个 token；溢出 token 跳过该层。损害质量。
3. **无辅助损失均衡** (DeepSeek-V3)。添加可学习的每专家偏置，偏移路由器的 top-k 选择。偏置在训练损失之外更新。对主目标无惩罚。2024 年的大解锁。

DeepSeek-V3 的方法：每个训练步骤后，对每个专家，检查其使用率是否高于或低于目标。将偏置调整 `±γ`。选择使用 `scores + bias`。用于门控的专家概率是原始 `scores` 不变。解耦路由与表达。

### 共享专家

DeepSeek-V2/V3 还将专家分为*共享*和*路由*。每个 token 通过所有共享专家。路由专家通过 top-k 选择。共享专家捕获通用知识；路由专家专精化。V3 运行 1 个共享专家加 256 个路由专家中的 top-8。

### 细粒度专家

经典 MoE (GShard, Switch)：每个专家与完整 FFN 一样宽。`E` 小 (8-64)，`k` 小 (1-2)。

现代细粒度 MoE (DeepSeek-V3, Qwen-MoE)：每个专家更窄（1/8 FFN 大小）。`E` 大 (256+)，`k` 更大 (8+)。相同总参数，但组合扩展快得多。`C(256, 8) = 400 万亿` 种每 token 可能的"专家"。质量上升，延迟持平。

### 成本概况

每 token，每层：

| 配置 | 活跃参数 / token | 总参数 |
|------|------------------|--------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B (密集) | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2 (MoE) | ~32B | 1T |

DeepSeek-V3 在几乎每个基准上击败 Llama 3 70B（密集），同时每 token **活跃 FLOPs 更少**。更多参数 = 更多知识。更多活跃 FLOPs = 每 token 更多计算。MoE 将它们解耦。

### 陷阱：内存

所有专家都在 GPU 上，无论哪个被激活。671B 模型在 fp16 权重下需要约 1.3 TB VRAM。前沿 MoE 部署需要专家并行——跨 GPU 分片专家，跨网络路由 token。延迟由全对全通信主导，而非矩阵乘法。

## 动手实现

参见 `code/main.py`。纯标准库中的紧凑 MoE 层，包含：

- `n_experts=8` 个类 SwiGLU 专家（每个一个线性层，用于说明）
- top-k=2 路由
- softmax 归一化门控权重
- 通过每专家偏置的无辅助损失均衡

### 第一步：路由器

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # 对所选专家的原始分数做 softmax
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

偏置影响选择，不影响门控权重。这就是 DeepSeek-V3 的技巧——偏置纠正负载不平衡，不引导模型的预测。

### 第二步：对 100 个 token 运行路由器

跟踪哪些专家被激活多少次。没有偏置时，使用率偏斜。有偏置更新循环（过度使用的专家 `-γ`，不足的 `+γ`），使用率在几次迭代后收敛到均匀分布。

### 第三步：参数量比较

打印 MoE 配置的"密集等价"。DeepSeek-V3 形状：256 路由 + 1 共享，8 活跃，d_model=7168。总参数量令人瞠目。活跃量是密集 Llama 3 70B 的七分之一。

## 使用场景

HuggingFace 加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026 年生产推理：vLLM 原生支持 MoE 路由。SGLang 有最快的专家并行路径。两者都自动处理 top-k 选择和专家并行。

**何时选择 MoE：**
- 你想要前沿质量但更低的每 token 推理成本。
- 你有 VRAM / 专家并行基础设施。
- 你的工作负载是 token 密集型（聊天、代码）而非上下文密集型（长文档）。

**何时不选 MoE：**
- 边缘部署——你为任何活跃 FLOP 付全部存储。
- 延迟关键的单用户服务——专家路由增加开销。
- 小型模型 (<7B)——MoE 的质量优势只在计算阈值（约 6B 活跃参数）以上出现。

## 交付使用

参见 `outputs/skill-moe-configurator.md`。该技能根据参数预算、训练 token 和部署目标为新 MoE 选择 E、k 和共享专家布局。

## 练习

1. **简单。** 运行 `code/main.py`。观察无辅助损失偏置更新如何在 50 次迭代内均匀化专家使用。
2. **中等。** 用基于哈希的路由器（确定性，无学习）替换可学习路由器。比较质量和平衡性。为什么可学习路由器更好？
3. **困难。** 实现 GRPO 风格的"rollout 匹配路由"（DeepSeek-V3.2 技巧）：记录推理时哪些专家被激活，在梯度计算期间强制相同路由。在玩具策略梯度设置上测量效果。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 专家 | "众多 FFN 中的一个" | 独立前馈网络；参数专用于 FFN 计算的稀疏切片。 |
| 路由器 | "门控" | 小型线性层，对每个 token 对每个专家评分；top-k 选择。 |
| Top-k 路由 | "每 token k 个活跃专家" | 每个 token 的 FFN 计算恰好经过 k 个专家，按门控加权。 |
| 辅助损失 | "负载均衡惩罚" | 惩罚偏斜专家使用的额外损失项。 |
| 无辅助损失 | "DeepSeek-V3 的技巧" | 仅通过路由器选择上的每专家偏置均衡；无额外梯度。 |
| 共享专家 | "始终开启" | 每个 token 都通过的额外专家；捕获通用知识。 |
| 专家并行 | "按专家分片" | 将不同专家分配到不同 GPU；跨网络路由 token。 |
| 稀疏性 | "活跃参数 < 总参数" | 比率 `k × expert_size / (E × expert_size)`；DeepSeek-V3 约 37/671 ≈ 5.5%。 |

## 延伸阅读

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — 这个想法。
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — Switch，经典 MoE。
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mixtral 8×7B。
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MLA + 无辅助损失 MoE + MTP。
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) — 基于偏置的均衡论文。
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) — 本课路由器使用的细粒度 + 共享专家分割。
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) — 原始共享专家论文。
