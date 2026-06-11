# 自编码器与变分自编码器（VAE）

> 普通自编码器先压缩再重建。它会记忆，但不会生成。加入一个技巧——强迫编码看起来像高斯分布——你就得到一个采样器。这个单一技巧，也就是 `z = μ + σ·ε` 的重参数化，是为什么你在 2026 年使用的每一个潜空间扩散和流匹配图像模型，输入端都有一个 VAE。

**类型：** 构建
**语言：** Python
**先修要求：** 第 3 阶段 · 02（反向传播）、第 3 阶段 · 07（CNN）、第 8 阶段 · 01（分类法）
**时间：** 约 75 分钟

## 问题

把一个 784 像素的 MNIST 数字压缩成 16 个数字的编码，然后重建。普通自编码器会在重建 MSE 上表现很好，但编码空间是一团疙疙瘩瘩的混乱结构。从编码空间中随机选一个点再解码，你得到的是噪声。它没有采样器。它只是披着生成模型外衣的压缩模型。

你真正想要的是：（a）编码空间是一个干净、平滑、可采样的分布——比如各向同性高斯 `N(0, I)`，（b）解码任意样本都会产生一个合理的数字，（c）编码器和解码器仍然能很好地压缩。三个目标，一个架构，一个损失。

Kingma 的 2013 年 VAE 通过以下方式解决这个问题：训练编码器输出一个*分布* `q(z|x) = N(μ(x), σ(x)²)`，通过 KL 惩罚把这个分布拉向先验 `N(0, I)`，然后在解码前从 `q(z|x)` 中采样 `z`。推理时，丢掉编码器，采样 `z ~ N(0, I)`，再解码。KL 惩罚正是强迫编码空间具有结构的东西。

到 2026 年，VAE 很少作为独立产品交付——在原始图像质量上它们已经被扩散超越——但它们是每一个潜空间扩散模型（SD 1/2/XL/3、Flux、AudioCraft）首选的编码器。学会 VAE，就学会了你所用每条图像流水线中那层看不见的第一层。

## 概念

![Autoencoder vs VAE: the reparameterization trick](../assets/vae.svg)

**自编码器。** `z = encoder(x)`，`x̂ = decoder(z)`，损失 = `||x - x̂||²`。编码空间没有结构。

**VAE 编码器。** 输出两个向量：`μ(x)` 和 `log σ²(x)`。它们定义了 `q(z|x) = N(μ, diag(σ²))`。

**重参数化技巧。** 从 `q(z|x)` 采样不可微。把样本改写为 `z = μ + σ·ε`，其中 `ε ~ N(0, I)`。现在 `z` 是 `(μ, σ)` 的确定性函数，再加上一个非参数噪声——梯度可以流经 `μ` 和 `σ`。

**损失。** 证据下界（Evidence Lower BOund，ELBO），包含两项：

```
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

重建项把 `x̂` 推向 `x`。KL 项把 `q(z|x)` 推向先验。二者相互权衡。小 β（<1）= 样本更清晰，编码空间不那么高斯。大 β（>1）= 编码空间更干净，样本更模糊。β-VAE（Higgins 2017）让这个旋钮出名，并开启了解耦表示研究。

**采样。** 推理时：抽取 `z ~ N(0, I)`，送入解码器。一次前向传播——不像扩散那样迭代采样。

## 构建它

`code/main.py` 实现了一个不使用 numpy 或 torch 的小型 VAE。输入是从 8 维、2 分量高斯混合中抽取的 8 维合成数据。编码器和解码器都是单隐藏层 MLP。我们实现 tanh 激活、前向传播、损失，以及手写反向传播。这不是生产代码——而是教学代码。

### 第 1 步：编码器前向传播

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

使用 `log σ²` 而不是 `σ`，这样网络输出不受约束（对 σ 使用 softplus 是个陷阱——当 σ ≈ 0 时梯度会死亡）。

### 第 2 步：重参数化并解码

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### 第 3 步：ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

因为两个分布都是高斯，所以 KL 有精确闭式解。不要做数值积分。到 2026 年仍有人在生产代码中使用蒙特卡洛 KL 估计——它没有理由地慢 3 倍。

### 第 4 步：生成

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

这就是生成模型。五行。

## 陷阱

- **后验坍塌（Posterior collapse）。** KL 项过于激进地把 `q(z|x) → N(0, I)`，导致 `z` 不携带任何关于 `x` 的信息。修复：β 退火（从 β=0 开始，逐渐升到 1）、free bits，或跳过非活跃维度上的 KL。
- **样本模糊。** 高斯解码器似然意味着 MSE 重建，而 MSE 对 L2 来说贝叶斯最优解是均值——一组合理数字的均值就是一个模糊数字。修复：离散解码器（VQ-VAE、NVAE），或只把 VAE 当作编码器，并在潜变量上叠加扩散（Stable Diffusion 就是这么做的）。
- **β 太大、太早。** 见后验坍塌。从 β≈0.01 开始，然后逐渐升高。
- **潜变量维度太小。** 16 维适合 MNIST，256 维适合 ImageNet 256²，2048 维适合 ImageNet 1024²。Stable Diffusion 的 VAE 把 512×512×3 压缩成 64×64×4（空间面积下采样 32 倍，通道上也等效为 32 倍压缩）。

## 使用它

2026 年的 VAE 技术栈：

| 场景 | 选择 |
|-----------|------|
| 扩散的图像潜变量编码器 | Stable Diffusion VAE (`sd-vae-ft-ema`) 或 Flux VAE |
| 音频潜变量编码器 | Encodec（Meta）、SoundStream 或 DAC（Descript） |
| 视频潜变量 | Sora 的时空 patch、Latte VAE、WAN VAE |
| 解耦表示学习 | β-VAE、FactorVAE、TCVAE |
| 离散潜变量（用于 transformer 建模） | VQ-VAE、RVQ（ResidualVQ） |
| 用于生成的连续潜变量 | 普通 VAE，然后在该潜空间中条件化一个 flow/diffusion 模型 |

潜空间扩散模型就是一个 VAE，并且在编码器和解码器之间住着一个扩散模型。VAE 做粗压缩，扩散模型负责重活。视频（VAE + video-diffusion DiT）和音频（Encodec + MusicGen transformer）也是同样模式。

## 交付它

保存 `outputs/skill-vae-trainer.md`。

技能接受：数据集画像 + 潜变量维度目标 + 下游用途（重建、采样或潜空间扩散输入），并输出：架构选择（plain/β/VQ/RVQ）、β schedule、潜变量维度、解码器似然（Gaussian vs categorical），以及评估计划（重建 MSE、每维 KL、`q(z|x)` 与 `N(0, I)` 之间的 Fréchet 距离）。

## 练习

1. **简单。** 把 `code/main.py` 中的 `β` 改成 `0.01`、`0.1`、`1.0`、`5.0`。记录最终重建 MSE 和 KL。哪个 β 对你的合成数据来说是 Pareto 最优？
2. **中等。** 把高斯解码器似然替换为伯努利似然（交叉熵损失）。在同一组合成数据的二值化版本上比较样本质量。
3. **困难。** 把 `code/main.py` 扩展成一个迷你 VQ-VAE：用 K=32 个条目的 codebook 中的最近邻查找替换连续 `z`。比较重建 MSE，并报告有多少 codebook 条目被使用（codebook collapse 是真实存在的）。

## 关键术语

| 术语 | 人们通常怎么说 | 它实际意味着什么 |
|------|-----------------|-----------------------|
| 自编码器 | 编码-解码网络 | `x → z → x̂`，学习 MSE。不是生成式的。 |
| VAE | 带采样器的 AE | 编码器输出一个分布，KL 惩罚塑造编码空间。 |
| ELBO | 证据下界 | `log p(x) ≥ recon - KL[q(z\|x) \|\| p(z)]`；当 `q = p(z\|x)` 时紧。 |
| 重参数化 | `z = μ + σ·ε` | 把随机节点改写成确定性部分 + 纯噪声。允许通过采样反传。 |
| 先验 | `p(z)` | 潜变量的目标分布，通常是 `N(0, I)`。 |
| 后验坍塌 | “KL 项赢了” | 编码器忽略 `x`，输出先验；解码器只能幻觉生成。 |
| β-VAE | 可调 KL 权重 | `loss = recon + β·KL`。β 越高 = 更解耦但更模糊。 |
| VQ-VAE | 离散潜变量 | 用最近的 codebook 向量替换连续 `z`；支持 transformer 建模。 |

## 生产说明：VAE 是扩散服务器里最热的路径

在 Stable Diffusion / Flux / SD3 流水线中，每个请求会调用 VAE 两次——如果做 img2img / inpainting，调用一次编码；再调用一次解码。在 1024² 分辨率下，解码器通路通常是整条流水线中最大的激活内存峰值，因为它会把 `128×128×16` 潜变量上采样回 `1024×1024×3`。这带来两个实际后果：

- **切片或平铺解码。** `diffusers` 暴露了 `pipe.vae.enable_slicing()` 和 `pipe.vae.enable_tiling()`。平铺用少量接缝伪影换取 `O(tile²)` 内存，而不是 `O(H·W)`。对消费级 GPU 上的 1024²+ 至关重要。
- **bf16 解码器，最终 resize 使用 fp32 数值。** SD 1.x VAE 是以 fp32 发布的，当在 1024²+ 下被转换为 fp16 时会*静默地产生 NaN*。SDXL 发布了 `madebyollin/sdxl-vae-fp16-fix`——始终优先使用 fp16-fix 变体，或使用 bf16。

## 延伸阅读

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — VAE 论文。
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) — 解耦 β-VAE。
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — VQ-VAE。
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) — 最先进的图像 VAE。
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion；VAE 作为编码器。
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — Encodec，音频 VAE 标准。
