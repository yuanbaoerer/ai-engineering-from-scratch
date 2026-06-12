# 扩展定律

> 2020 年 Kaplan 论文说：更大模型，更低损失。2022 年 Hoffmann 论文说：你训练不足了。计算分两个桶——参数和 token——而分配并不明显。

**类型:** 学习
**语言:** Python
**前置知识:** 第 7 阶段 · 05（完整 Transformer）、第 7 阶段 · 07（GPT）
**时间:** 约 45 分钟

## 问题所在

当你有 C FLOPs 的训练计算并想要最佳模型时，你面对两个旋钮：

1. **多少参数 (N)？** 更大模型，更高容量。
2. **多少训练 token (D)？** 更多数据，更好利用容量。

FLOPs 大约按 `6 × N × D` 缩放。你可以推高 N 降低 D，或推高 D 降低 N。哪个更好？

2022 年之前，答案是"大力推 N"。GPT-3 (2020) 是 175B 参数训练约 300B token。比率约 1.7 token 每参数。Kaplan 扩展定律支持了这一点。

Hoffmann et al. (2022)，训练了一个叫 Chinchilla 的小型模型家族，发现了不同的结果：最优比率接近 **20 token 每参数**。GPT-3 训练不足 10 倍。Chinchilla (70B 参数，1.4T token) 在每个基准上击败 GPT-3 (175B，300B token)，推理成本低 2.5 倍。

2026 年是 Chinchilla 的世界——但有一个重要转折。Llama 3 8B 在 15 万亿 token 上训练，比率 1,875 token 每参数。是 Chinchilla 最优的 94 倍。对于大规模使用的模型，推理成本比训练成本更重要，因此过度训练（超过 Chinchilla）以获得更小的可部署足迹是 2026 年的默认。

## 核心概念

![Chinchilla 曲线：不同 N/D 比率下的损失 vs 计算](../assets/scaling-laws.svg)

### Hoffmann 定律

来自 Chinchilla 论文，损失遵循：

```
L(N, D) = A / N^α + B / D^β + E
```

- `N` = 参数（非嵌入）。
- `D` = 训练 token。
- `α ≈ 0.34`，`β ≈ 0.28`（大致对称）。
- `E ≈ 1.69`，不可约损失上限。
- `A ≈ 406`，`B ≈ 411`。

两个项随扩展相互权衡。在固定计算 (C = 6ND) 下对 `N` 求导并求解：

```
N_opt ≈ 0.6 × (C/6)^0.5
D_opt ≈ 0.6 × (C/6)^0.5
D_opt / N_opt ≈ 20
```

计算最优：20 token 每参数。

### 为什么还要过度训练

Chinchilla 最优最小化每训练 FLOP 的训练损失。但你只付一次训练成本；推理成本永远付。

对于每月服务万亿 token 的聊天机器人，推理主导总成本。Llama 的方法：训练更小、更久。8B 在 15T token 上是深度推理优化的：

- 适合消费级 GPU。
- 延迟是 70B Chinchilla 最优的一小部分。
- 质量对大多数任务足够接近。

DeepMind 2024 年论文（"过度训练是最优"）将此形式化。对于推理主导的工作负载，正确比率接近 100-500 token 每参数，取决于服务量。

### 涌现 vs 平滑性

声称：某些能力（算术、多步推理、思维链跟随）在某个规模"突然涌现"。

Schaeffer et al. (2023) 认为这是测量伪影：涌现指标使用不连续评分（精确匹配、阈值准确率），隐藏了底层 logits 的平滑改进。连续指标（交叉熵）显示平滑曲线。

2026 年共识是：通过连续损失的预测是可靠的。基准跳跃通常是评分器伪影。按连续指标规划预算。

### 2026 年图景

扩展定律仍然有效，但：

| 因素 | 如何改变 |
|------|----------|
| 数据质量 | 策划"好" token（Phi 风格）将曲线移动 >2 倍有效计算 |
| MoE | 总参数与活跃 FLOPs 解耦；按活跃 FLOP 的扩展定律 |
| 后训练 | 某些能力（指令跟随、代码）随 SFT+RLHF 移动多于预训练 |
| 多模态 | 图像 + 文本 token 一起扩展；每模态独立曲线 |
| 合成数据 | 模型生成训练数据；有效计算可以复合 |

Muon 优化器（Kimi Moonlight，2024）在匹配数据下显示约 2 倍有效计算增益超过 AdamW。一些 2026 年训练运行默认使用 Muon。改变扩展定律中的绝对常数，不改变其形状。

```figure
scaling-laws
```

## 动手实现

参见 `code/main.py`。我们实现 Chinchilla 损失方程，并在多个计算预算下求解计算最优 `(N, D)`。

### 第一步：Chinchilla 损失

```python
def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    return A / N ** alpha + B / D ** beta + E
```

在固定 `C = 6ND` 下绘制 `L` 作为 `(N, D)` 上的等高线。找到最小值。

### 第二步：计算最优前沿

对 `1e17` 到 `1e25` FLOPs 的计算预算，找到在 `6ND = C` 约束下最小化损失的 `(N, D)`。验证比率 `D/N ≈ 20`。

### 第三步：过度训练成本

计算训练 10 倍更小模型（最优 N 的 1/10，最优 D 的 10 倍）所付出的额外损失。报告推理 FLOP 节省（与 N 成正比）作为交换。

### 第四步：与真实模型比较

代入已知的 GPT-3、Chinchilla、Llama 3 8B、DeepSeek-V3（活跃参数）的 `(N, D)` 对，比较预测 vs 报告损失。

## 使用场景

你不太可能自己训练前沿模型。但扩展定律告诉你：

1. **你的微调是否有足够数据。** 如果你的任务特定数据低于基础模型每参数 20 token，预计在某个损失地板饱和。
2. **是否选择更大的基础模型。** 如果你将所有预算花在推理上，选择更小、训练更久的模型。
3. **回报在哪里递减。** 超过 Chinchilla 最优的 1000 倍，log-loss 变化变成噪声。

**2026 年的研究轨迹：**

- **数据受限体制。** 网络有有限数量的高质量 token（过滤后约 5-10 万亿英语）。前沿预训练正在接近这个上限。合成数据、多语言、多模态和 RLHF 规模微调是下一个杠杆。
- **计算乘数技巧。** Muon 优化器、MoE、更好的数据策划——每个都移动绝对常数，不改变渐近线。
- **RL 的扩展定律。** 开放问题。早期证据表明 RL 样本中的幂律，但指数与预训练非常不同。

## 交付使用

参见 `outputs/skill-training-budget-estimator.md`。该技能根据计算预算、部署约束和目标损失为新训练运行选择 `(N, D, hours, GPU)`。

## 练习

1. **简单。** 运行 `code/main.py`。打印计算预算 `1e20`、`1e22`、`1e24` 下的 Chinchilla 最优 `(N, D)`。与真实模型表比较。
2. **中等。** 实现 Hoffmann 损失-作为-计算-函数曲线。绘制计算最优前沿的损失 vs `log10(C)`。识别定律预测我们需要 `>10^28` FLOPs 才能再降低 0.1 交叉熵的时间点。
3. **困难。** 在 5 个微型模型（100K 到 10M 参数）上在同一数据集上拟合你自己的扩展定律。估计 `α` 和 `E`。你的指数与已发布的匹配得如何？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 参数 (N) | "模型大小" | 非嵌入权重计数；决定容量。 |
| Token (D) | "训练数据" | 看到的训练 token 数量；决定参数被利用得多好。 |
| 计算 (C) | "花费的 FLOPs" | 标准 Transformer 约 `6 × N × D`。 |
| Chinchilla 最优 | "D/N ≈ 20" | 最小化预训练每 FLOP 损失的比率。 |
| 过度训练 | "超过 Chinchilla" | 花额外训练 FLOPs 节省推理 FLOPs；D/N >> 20。 |
| 不可约损失 | "地板" | 扩展定律中的 `E` 项；数据本身的熵。 |
| 涌现能力 | "规模下的突然跳跃" | 通常是评分器伪影；连续损失是平滑的。 |
| 有效计算 | "训练效率乘数" | 更好的数据/优化器/架构乘以一个 FLOP 走多远。 |

## 延伸阅读

- [Kaplan et al. (2020). Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — 第一篇扩展定律论文；训练不足。
- [Hoffmann et al. (2022). Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — Chinchilla。
- [Schaeffer et al. (2023). Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) — 涌现作为测量伪影。
- [Sardana, Frankle (2024). Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws](https://arxiv.org/abs/2401.00448) — Llama 过度训练对其工作负载正确的原因。
- [Jordan et al. (2024). Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) — 2 倍计算乘数。
