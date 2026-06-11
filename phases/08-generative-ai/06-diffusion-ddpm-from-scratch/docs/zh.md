# 扩散模型——从零实现 DDPM

> Ho、Jain、Abbeel（2020）给了这个领域一个再也戒不掉的配方。用一千个小步骤把数据加噪摧毁。训练一个神经网络来预测噪声。推理时反转这个过程。今天，所有主流图像、视频、3D 和音乐模型都运行在这个循环之上，可能再叠加流匹配（flow matching）或一致性技巧。

**类型：** 构建
**语言：** Python
**先修：** 阶段 3 · 02（Backprop）、阶段 8 · 02（VAE）
**时间：** 约 75 分钟

## 问题

你想要一个 `p_data(x)` 的采样器。GAN 会玩一个经常发散的极小极大游戏。VAE 会从高斯解码器产生模糊样本。你真正想要的是一个训练目标，它 (a) 是单一稳定损失（没有鞍点、没有极小极大），(b) 是 `log p(x)` 的下界（因此你有似然），并且 (c) 样本质量达到 SOTA。

Sohl-Dickstein 等（2015）给出了理论答案：定义一个马尔可夫链 `q(x_t | x_{t-1})`，逐步加入高斯噪声，并训练一个反向链 `p_θ(x_{t-1} | x_t)` 来去噪。Ho、Jain、Abbeel（2020）展示了损失可以简化成一行——预测噪声——并整理了数学。2020 年它还只是一个新奇想法。2021 年它生成了最先进样本。2022 年它变成了 Stable Diffusion。到 2026 年，它已经成为底层基座。

## 概念

![DDPM: forward noise, reverse denoise](../assets/ddpm.svg)

**前向过程 `q`。** 用 `T` 个小步骤加入高斯噪声。闭式形式——也就是数学可处理的原因——是累计步骤同样是高斯分布：

```
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

其中 `α̅_t = ∏_{s=1..t} (1 - β_s)`，`β_t` 按某个调度变化。把 `β_t` 在 T=1000 步内从 1e-4 线性取到 0.02，`x_T` 就近似为 `N(0, I)`。

**反向过程 `p_θ`。** 学习一个神经网络 `ε_θ(x_t, t)`，预测被加入的噪声。给定 `x_t`，按如下方式去噪：

```
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

其中 `σ_t` 要么是 `sqrt(β_t)`，要么是可学习方差。这个表达式很丑，但它只是代数——给定后验 `q(x_{t-1} | x_t, x_0)` 解出 `x_{t-1}`，再用噪声预测得到的估计替换 `x_0`。

**训练损失。**

```
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

从数据中采样 `x_0`，随机选择一个 `t`，采样 `ε ~ N(0, I)`，用闭式形式一次性计算带噪的 `x_t`，然后回归噪声。一个损失，没有极小极大，没有 KL，没有重参数化技巧。

**采样。** 从 `x_T ~ N(0, I)` 开始。从 `t = T` 到 `1` 迭代反向步骤。完成。

## 为什么它有效

三个直觉：

1. **去噪容易；生成困难。** 在 `t=T` 时，数据是纯噪声——网络要解决的是一个平凡问题。在 `t=0` 时，网络只需要清理几个像素。在中间的 `t`，问题很难，但网络会从每个噪声等级经由同一组权重获得许多梯度。

2. **伪装成去噪的分数匹配。** Vincent（2011）证明，预测噪声等价于估计 `∇_x log q(x_t | x_0)`，即*分数*（score）。反向 SDE 使用这个分数沿密度梯度向上移动——一次被引导的随机游走，走向高概率区域。

3. **ELBO 化简为简单 MSE。** 完整变分下界在每个时间步都有一个 KL 项。在 DDPM 的参数化下，这些 KL 项会化简为带特定系数的噪声预测 MSE；Ho 去掉了这些系数（称之为“simple” loss），质量反而*提升*了。

## 构建它

`code/main.py` 实现了一个 1-D DDPM。数据是一个双峰混合分布。“网络”是一个小型 MLP，接收 `(x_t, t)` 并输出预测噪声。训练就是这一行损失。采样会迭代反向链。

### 步骤 1：前向调度（闭式形式）

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### 步骤 2：一次性采样 `x_t`

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### 步骤 3：一次训练步骤

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### 步骤 4：反向采样

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

对于一个 40 个时间步、24 单元 MLP 的 1-D 问题，它大约 200 个 epoch 就能学会双峰混合分布。

## 时间条件化

网络需要知道自己正在为哪个时间步去噪。有两个标准选项：

- **正弦嵌入。** 类似 Transformer 位置编码。`embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`。经过一个 MLP，再广播进网络。
- **FiLM / group-norm 条件化。** 把嵌入投影成每个通道的 scale/bias（FiLM），用于每个 block。

我们的玩具代码使用正弦 → 拼接。生产级 U-Net 使用 FiLM。

## 常见陷阱

- **调度非常重要。** 线性 `β` 是 DDPM 默认值，但余弦调度（Nichol & Dhariwal，2021）在相同计算量下给出更好的 FID。如果质量停滞，切换调度。
- **时间步嵌入很脆弱。** 把原始 `t` 当作 float 传入对玩具 1-D 有效，但对图像会失败；始终使用合适的嵌入。
- **V-prediction vs ε-prediction。** 对于狭窄区间（非常小或非常大的 t），`ε` 的信噪比很差。V-prediction（`v = α·ε - σ·x`）更稳定；SDXL、SD3 和 Flux 都使用它。
- **无分类器引导。** 推理时，同时计算有条件和无条件的 `ε`，然后 `ε_cfg = (1 + w) · ε_cond - w · ε_uncond`，其中 `w ≈ 3-7`。第 08 课会覆盖。
- **1000 步太多了。** 生产使用 DDIM（20-50 步）、DPM-Solver（10-20 步）或蒸馏（1-4 步）。见第 12 课。

## 使用它

| 角色 | 2026 年典型技术栈 |
|------|-----------------------|
| 图像像素空间扩散（小型、玩具） | DDPM + U-Net |
| 图像潜空间扩散 | VAE 编码器 + U-Net 或 DiT（第 07 课） |
| 视频潜空间扩散 | 时空 DiT（Sora、Veo、WAN） |
| 音频潜空间扩散 | Encodec + diffusion transformer |
| 科学（分子、蛋白质、物理） | 等变扩散（EDM、RFdiffusion、AlphaFold3） |

扩散是通用生成骨干。流匹配（第 13 课）是 2024-2026 年的竞争者，通常在相同质量下赢在推理速度。

## 交付它

保存 `outputs/skill-diffusion-trainer.md`。该技能接收数据集 + 计算预算并输出：调度（linear/cosine/sigmoid）、预测目标（ε/v/x）、步数、引导强度、采样器家族和评估协议。

## 练习

1. **简单。** 在 `code/main.py` 中把 T 从 40 改成 10。样本质量（输出的可视化直方图）如何下降？在什么 T 下双峰结构崩塌？
2. **中等。** 从 ε-prediction 切换到 v-prediction。重新推导反向步骤。比较最终样本质量。
3. **困难。** 添加无分类器引导。以类别标签 `c ∈ {0, 1}` 为条件，在训练时 10% 的时间丢弃它，并在采样时使用 `ε = (1+w)·ε_cond - w·ε_uncond`。测量 `w = 0, 1, 3, 7` 时的条件模式命中率。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Forward process | “加噪” | 固定马尔可夫链 `q(x_t \| x_{t-1})`，用于摧毁数据。 |
| Reverse process | “去噪” | 学到的链 `p_θ(x_{t-1} \| x_t)`，用于重建数据。 |
| β schedule | “噪声阶梯” | 每步方差；线性、余弦或 sigmoid。 |
| α̅ | “Alpha bar” | 累积乘积 `∏(1 - β)`；给出从 `x_0` 到 `x_t` 的闭式形式。 |
| Simple loss | “噪声上的 MSE” | `\|\|ε - ε_θ(x_t, t)\|\|²`；所有变分推导都塌缩为它。 |
| ε-prediction | “预测噪声” | 输出是被加入的噪声；标准 DDPM。 |
| V-prediction | “预测速度” | 输出是 `α·ε - σ·x`；在不同 t 上条件更好。 |
| DDPM | “那篇论文” | Ho 等，2020；线性 β、1000 步、U-Net。 |
| DDIM | “确定性采样器” | 非马尔可夫采样器，20-50 步，相同训练目标。 |
| Classifier-free guidance | “CFG” | 混合有条件和无条件噪声预测，以放大条件。 |

## 生产说明：扩散推理是一个步数问题

DDPM 论文运行 T=1000 个反向步骤。没有人把这个部署到生产。每个真实推理栈都会选择以下三种策略之一——而每种策略都清楚对应到“延迟来自哪里”的生产表述：

1. **更快的采样器，同一个模型。** DDIM（20-50 步）、DPM-Solver++（10-20）、UniPC（8-16）。作为反向循环的直接替代；训练好的 `ε_θ` 权重不变。延迟降低 20-50×。
2. **蒸馏。** 训练一个学生模型，用更少步骤匹配教师：Progressive Distillation（2 → 1）、Consistency Models（任意 → 1-4）、LCM、SDXL-Turbo、SD3-Turbo。再把延迟降低 5-10×，但需要重新训练。
3. **缓存和编译。** `torch.compile(unet, mode="reduce-overhead")`、TensorRT-LLM 的扩散后端、`xformers`/SDPA 注意力、bf16 权重。每步延迟约降低 2×。可与 (1) 和 (2) 叠加。

对于生产扩散服务器，预算对话与生产 LLM 文献描述的一样：延迟是 `num_steps × step_cost + VAE_decode`，吞吐量是 `batch_size × (num_steps × step_cost)^-1`。TTFT 很小（一步）；TPOT 等价物是完整响应时间，因为从用户角度看，图像生成是“一次性”给出结果。

## 延伸阅读

- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — 超前时代的扩散论文。
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM。
- [Song, Meng, Ermon (2021). Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — DDIM，更少步骤。
- [Nichol & Dhariwal (2021). Improved DDPM](https://arxiv.org/abs/2102.09672) — 余弦调度、可学习方差。
- [Dhariwal & Nichol (2021). Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) — 分类器引导。
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG。
- [Karras et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364) — 统一记法，最干净的配方。
