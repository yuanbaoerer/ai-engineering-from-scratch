# 流匹配与校正流

> 扩散模型需要 20–50 个采样步骤，因为它们沿着一条从噪声到数据的弯曲路径行进。流匹配（Flow Matching，Lipman et al., 2023）和校正流（Rectified Flow，Liu et al., 2022）训练的是直线路径。路径越直，所需步骤越少，推理越快。Stable Diffusion 3、Flux.1 和 AudioCraft 2 都在 2024 年切换到了流匹配。

**类型：** 构建
**语言：** Python
**先修要求：** 第 8 阶段 · 06（DDPM），第 1 阶段 · 微积分
**时间：** 约 45 分钟

## 问题

DDPM 的反向过程是一个从 `N(0, I)` 回到数据分布的 1000 步随机游走。DDIM 将其压缩到 20–50 个确定性步骤。你想要更少的步骤——理想情况下只需一步。阻碍在于，求解反向过程的 ODE 是刚性的；路径是弯曲的。

如果你能训练模型，让从噪声到数据的路径是一条*直线*，那么从 `t=1` 到 `t=0` 的单个欧拉步就能奏效。流匹配直接构建这一点：定义从 `x_1 ∼ N(0, I)` 到 `x_0 ∼ data` 的直线插值，训练一个向量场 `v_θ(x, t)` 去匹配它的时间导数，并在推理时积分。

校正流（Liu 2022）更进一步：通过 reflow 过程迭代地把路径拉直，产生一个逐渐更接近线性的 ODE。经过两次 reflow 迭代后，2 步采样器就能匹配 50 步 DDPM 的质量。

## 概念

![流匹配：噪声与数据之间的直线插值](../assets/flow-matching.svg)

### 直线流

定义：

```
x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]
```

其中 `x_0 ~ data`，`x_1 ~ N(0, I)`。沿这条直线的时间导数是常数：

```
dx_t / dt = x_1 - x_0
```

定义一个神经向量场 `v_θ(x_t, t)`，并训练它去匹配这个导数：

```
L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

这就是**条件流匹配**（conditional flow matching）损失（Lipman 2023）。训练无需仿真：你永远不需要展开 ODE。只需采样 `(x_0, x_1, t)` 并做回归。

### 采样

推理时，沿时间*反向*积分学习到的向量场：

```
x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
```

从 `x_1 ~ N(0, I)` 开始，用欧拉步下降到 `t=0`。

### 校正流（Liu 2022）

直线流可行，但学习到的路径*实际上并不直*——它们会弯曲，因为多个 `x_0` 可能映射到同一个 `x_1`。校正流的 reflow 步骤如下：

1. 用随机配对训练流模型 v_1。
2. 通过将 v_1 从 `x_1` 积分到其落点 `x_0`，采样 N 对 `(x_1, x_0)`。
3. 在这些配对样本上训练 v_2。由于这些配对现在是“ODE 匹配”的，它们之间的直线插值会真正更平坦。
4. 重复。

实践中，2 次 reflow 迭代就能接近线性，从而实现 2–4 步推理。SDXL-Turbo、SD3-Turbo、LCM 都是从流匹配模型蒸馏而来的模型。

### 为什么它在 2024 年赢得了图像生成

三个原因：

1. **无需仿真的训练**——训练期间不需要展开 ODE，实现非常简单。
2. **更好的损失几何**——直线路径具有一致的信噪比，而 DDPM 的 ε-loss 在调度边缘处 SNR 很差。
3. **更快的推理**——4–8 步即可达到 SDXL-Turbo 质量；结合一致性蒸馏可做到 1 步。

## 流匹配 vs DDPM——精确联系

带高斯条件路径的流匹配，本质上是带有*特定噪声调度*的扩散。选取 `x_t = α(t) x_0 + σ(t) x_1` 调度，流匹配会恢复 Stratonovich 形式重写的扩散，其中 `v = α'·x_0 - σ'·x_1`。对于高斯路径，两者在代数上等价。

流匹配新增的是：目标的*清晰性*（一个普通速度）、更干净的损失，以及尝试非高斯插值器的自由度。

## 构建它

`code/main.py` 在一个双峰高斯混合上实现 1-D 流匹配。向量场 `v_θ(x, t)` 是一个小型 MLP，使用直线目标进行训练。推理时，分别积分 1、2、4 和 20 个欧拉步，并比较样本质量。

### 第 1 步：训练损失

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # backprop + update
```

### 第 2 步：多步推理

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### 第 3 步：比较步数

预期 4 步采样器已经能匹配 20 步质量——这对延迟来说非常重要。

## 陷阱

- **时间参数化。** 流匹配使用 `t ∈ [0, 1]`，其中 `t=0` 是数据，`t=1` 是噪声。DDPM 使用 `t ∈ [0, T]`，其中 `t=0` 是数据，`t=T` 是噪声。方向相同，尺度不同。论文经常把这一点写错。
- **调度选择。** 校正流的直线是“那个”流匹配调度，但你可以使用余弦或 logit-normal 的 t 采样（SD3 就这样做）来获得更好的尺度覆盖。
- **Reflow 成本。** 为 reflow 生成配对数据集，等价于每个样本做一次完整推理。只有在你真的需要 1–2 步推理时才做 reflow。
- **无分类器引导仍然适用。** 只需在线性组合中把 ε 换成 v：`v_cfg = (1+w) v_cond - w v_uncond`。

## 使用它

| 使用场景 | 2026 技术栈 |
|----------|-----------|
| 文生图，最佳质量 | 流匹配：SD3、Flux.1-dev |
| 文生图，1–4 步 | 蒸馏后的流匹配：Flux.1-schnell、SD3-Turbo、SDXL-Turbo |
| 实时推理 | 从流匹配基座做一致性蒸馏（LCM、PCM） |
| 音频生成 | 流匹配：Stable Audio 2.5、AudioCraft 2 |
| 视频生成 | 流匹配与扩散混合（Sora、Veo、Stable Video） |
| 科学 / 物理（粒子轨迹、分子） | 流匹配 + 等变向量场 |

只要一篇 2025–2026 年的论文说“比扩散更快”，它几乎总是流匹配 + 蒸馏。

## 交付它

保存 `outputs/skill-fm-tuner.md`。该技能接收一个扩散风格的模型规格，并将其转换为流匹配训练配置：调度选择、时间采样分布（uniform / logit-normal）、优化器、reflow 计划、目标步数、评估协议。

## 练习

1. **简单。** 运行 `code/main.py`，比较 1 步与 20 步相对于真实数据分布的 MSE。
2. **中等。** 将均匀 `t` 采样切换为 logit-normal（把采样集中在中间 t）。模型质量是否提升？
3. **困难。** 实现一次 reflow 迭代：通过积分第一个模型生成配对的 (x_0, x_1)，在这些配对上训练第二个模型，并比较 1 步样本质量。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| 流匹配 | “直线扩散” | 训练 `v_θ(x, t)`，使其沿插值器匹配 `x_1 - x_0`。 |
| 校正流 | “Reflow” | 将学习到的流逐步拉直的迭代过程。 |
| 速度场 | “v_θ” | 模型输出——`x_t` 应该移动的方向。 |
| 直线插值器 | “路径” | `x_t = (1-t)·x_0 + t·x_1`；目标导数很简单。 |
| 欧拉采样器 | “一阶 ODE 求解器” | 最简单的积分器；当路径较直时效果很好。 |
| Logit-normal t | “SD3 采样” | 将 `t` 采样集中到梯度最强的中间值附近。 |
| 一致性蒸馏 | “1 步采样器” | 训练学生模型，把任意 `x_t` 直接映射到 `x_0`。 |
| 带速度的 CFG | “v-CFG” | `v_cfg = (1+w) v_cond - w v_uncond`；同一个技巧，换了变量。 |

## 生产说明：Flux.1-schnell 是最快形态的流匹配

流匹配在生产中的胜利案例是 Flux.1-schnell——一个经过流匹配的 DiT，被蒸馏到 1–4 个推理步骤，同时保持 Flux-dev 级别的质量。Niels 的“在 8GB 机器上运行 Flux” notebook 是参考部署配方：T5 + CLIP 编码，量化 MMDiT 去噪（schnell 用 4 步，而 dev 用 50 步），VAE 解码。成本核算如下：

| 变体 | 步数 | L4 上 1024² 延迟 | 总 FLOPs（相对值） |
|---------|-------|------------------------|------------------------|
| Flux.1-dev（原始） | 50 | ~15 s | 1.0× |
| Flux.1-schnell | 4 | ~1.2 s | 0.08×（快 12×） |
| SDXL-base | 30 | ~4 s | 0.25× |
| SDXL-Lightning 2-step | 2 | ~0.3 s | 0.03× |

生产规则：**流匹配基座 + 蒸馏 = 2026 年快速文生图的默认方案。** 每个主要厂商都在交付这种组合：SD3-Turbo（SD3 + flow + distillation）、Flux-schnell（Flux-dev + rectified-flow straightening）、CogView-4-Flash。纯扩散基座只存在于遗留 checkpoint 中。

## 延伸阅读

- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — 校正流。
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — 流匹配。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3，大规模校正流。
- [Albergo, Vanden-Eijnden (2023). Stochastic Interpolants](https://arxiv.org/abs/2303.08797) — 覆盖 FM + 扩散的通用框架。
- [Song et al. (2023). Consistency Models](https://arxiv.org/abs/2303.01469) — 扩散 / 流的 1 步蒸馏。
- [Sauer et al. (2023). Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042) — turbo 变体。
- [Black Forest Labs (2024). Flux.1 models](https://blackforestlabs.ai/announcing-black-forest-labs/) — 生产中的流匹配。
