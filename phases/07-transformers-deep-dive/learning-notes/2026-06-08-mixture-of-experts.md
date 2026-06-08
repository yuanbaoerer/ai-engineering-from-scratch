# 混合专家 (MoE) 学习笔记

> 日期: 2026-06-08

## 1. FFN 是什么

FFN = Feed-Forward Network（前馈网络），Transformer 每个 block 的两个核心部分之一：

- **Attention 层**：负责 token 之间的关系
- **FFN 层**：对每个 token 独立做非线性变换，存储"知识"

模型的大部分参数都在 FFN 里，这就是为什么 MoE 替换 FFN 层如此有效——它稀疏化了模型中最"重"的部分。

FFN 结构（现代模型用 SwiGLU）：

```
FFN(x) = W2 · activation(W1 · x + b1) + b2
```

## 2. MoE 的核心机制

将每个 FFN 替换为 E 个独立专家 + 一个路由器：

- 总参数 = E × FFN_size
- 每 token 活跃参数 = k × FFN_size
- 典型 2026 配置：E=256，k=8

```
输入 token
    │
    ├── 共享专家 × 1 ──→ 输出（必过）
    │
    └── 路由专家 × 256
         └── 选中 top-8 ──→ 输出
    │
    └── 加权求和 ──→ 最终输出
```

## 3. C(256,8) 的含义

8 = top-k，即每个 token 激活的专家数。从 256 个专家中选 8 个：

```
C(256, 8) = 256! / (8! × 248!) ≈ 4.3 × 10^13
```

约 400 万亿种不同组合，每个 token 可以被不同的 8 个专家组合处理，表达灵活性极高。

## 4. 门控是什么

门控 = 控制信号通过的"门"。路由器给每个专家打分，选中 top-k 后做 softmax 变成权重（gate weights），决定每个专家的输出占总输出的百分比。

```python
# 选中 8 个专家后
output = 0
for expert_idx, gate_weight in zip(top_k_experts, gate_weights):
    output += gate_weight * expert[expert_idx](x)
```

来自神经网络术语：LSTM 的 forget gate、input gate 都是"用一个值控制另一个值的通过比例"。

## 5. 共享专家 vs 路由专家

| 类型 | 行为 | 擅长 |
|------|------|------|
| 共享专家 | 每个 token 都经过 | 通用知识、基础语法、常见模式 |
| 路由专家 | 只有被选中的 token 才经过 | 特定领域的深度知识 |

共享专家作为"兜底"，确保所有 token 都有一层基础计算，即使路由器分配错了也不会完全丢失信息。

## 6. MoE 的效率权衡

### 计算效率

| 指标 | 效果 |
|------|------|
| 每 token FLOPs | **降低** — 只激活 k 个专家 |
| 推理延迟 | **略增** — 路由器 + 专家并行通信开销 |

### 存储效率

| 指格 | 效果 |
|------|------|
| 参数总量 | **增加** — 671B vs 70B |
| VRAM 需求 | **大幅增加** — 所有专家常驻 GPU |
| 通信开销 | **主导延迟** — 专家分散在多 GPU，token 需跨网络路由 |

核心 tradeoff：**用存储换计算**。GPU 显存相对便宜，但 FLOPs 越来越贵。

## 7. 专家没有自己的 context

Context 长度由 **Attention 层**决定，不是 FFN 层。MoE 只替换 FFN 部分，Attention 层保持不变。

每个专家只是 FFN 子模块（`output = W2 @ activation(W1 @ x)`），没有自己的 Attention、embedding、层归一化。专家像大脑的某个功能区，不是独立的小脑。

## 8. DeepSeek-V3 的无辅助损失均衡

传统方案用辅助损失惩罚不均衡，但增加超参数和第二个梯度信号。

DeepSeek-V3 的突破：每个训练步骤后，对每个专家检查使用率，调整可学习偏置 ±γ。选择用 `scores + bias`，门控权重用原始 `scores`。偏置只影响选谁，不影响怎么加权，解耦了路由与表达。

## 9. 成本对比

| 模型 | 活跃参数/Token | 总参数 |
|------|---------------|--------|
| Llama 3 70B (密集) | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |

DeepSeek-V3 用不到一半的活跃计算，性能超越密集 70B，但需要约 1.3TB VRAM。

## 10. FLOPS 与 FLOPs

- **FLOPs**（小写 s）：总浮点运算次数，表示工作量
- **FLOPS**（大写 S）：每秒浮点运算次数，表示算力

GPU 算力参考：

| GPU | FP16 算力 |
|-----|----------|
| A100 | 312 TFLOPS |
| H100 | 989 TFLOPS |
| B200 | 2250 TFLOPS |
