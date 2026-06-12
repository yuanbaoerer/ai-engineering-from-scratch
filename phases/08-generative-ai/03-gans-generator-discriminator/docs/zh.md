# GAN —— 生成器 vs 判别器

> Goodfellow 在 2014 年的技巧是完全跳过密度。两个网络。一个制造伪造样本。一个抓住它们。它们相互对抗，直到伪造样本与真实样本不可区分。它本不该有效。它也经常无效。但一旦有效，在窄领域中它的样本至今仍是文献里最清晰的。

**类型：** 构建
**语言：** Python
**先修要求：** 第 3 阶段 · 02（反向传播）、第 3 阶段 · 08（优化器）、第 8 阶段 · 02（VAE）
**时间：** 约 75 分钟

## 问题

VAE 会产生模糊样本，因为它们的 MSE 解码器损失对*均值*图像是贝叶斯最优的——而许多合理数字的均值就是一个模糊数字。你想要的是一种奖励*合理性*的损失，而不是奖励与某一个目标在像素级的接近。合理性没有闭式表达。你必须学习它。

Goodfellow 的想法：训练一个分类器 `D(x)` 来区分真实图像和伪造图像。训练一个生成器 `G(z)` 来欺骗 `D`。`G` 的损失信号就是 `D` 当前认为某个东西看起来真实的原因。随着 `G` 改进，这个信号也会更新，追逐一个移动目标。如果两个网络都收敛，`G` 就在从未写下 `log p(x)` 的情况下学到了数据分布。

这就是对抗训练。数学上它是一个 minimax 博弈：

```
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

到 2026 年，GAN 不再是 SOTA 生成器（扩散和流匹配夺走了王冠）。但 StyleGAN 2/3 仍然是交付过的最清晰人脸模型，GAN 判别器被用作扩散训练中的*感知损失*，而对抗训练也驱动了快速的一步蒸馏（SDXL-Turbo、SD3-Turbo、LCM），让你能够交付实时扩散。

## 概念

![GAN training: generator and discriminator in minimax](../assets/gan.svg)

**生成器 `G(z)`。** 把噪声向量 `z ~ N(0, I)` 映射到样本 `x̂`。它是一个形似解码器的网络（全连接或转置卷积）。

**判别器 `D(x)`。** 把样本映射到一个标量概率（或分数）。真实 → 1，伪造 → 0。

**损失。** 两个交替更新：

- **训练 `D`：** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`。对真实=1、伪造=0 做二元交叉熵。
- **训练 `G`：** `loss_G = -log D(G(z))`。这是 Goodfellow 使用的*非饱和*形式（原始的 `log(1 - D(G(z)))` 会在 `D` 很自信时饱和并杀死梯度）。

**训练循环。** `D` 一步，`G` 一步。重复。

**为什么有效。** 如果 `G` 完美匹配 `p_data`，那么 `D` 最多只能随机猜测，并在所有位置输出 0.5；`G` 不再获得梯度。达到均衡。

**为什么会坏。** 模式坍塌（`G` 找到一个 `D` 无法分类的模式并永远复制它）、梯度消失（`D` 学得太快，`log D` 饱和）、训练不稳定（学习率、批大小，任何东西都可能导致）。

## 让 GAN 真正可用的变体

| 年份 | 创新 | 修复点 |
|------|------------|-----|
| 2015 | DCGAN | Conv/deconv、batch norm、LeakyReLU——第一个稳定架构。 |
| 2017 | WGAN, WGAN-GP | 用 Wasserstein 距离 + 梯度惩罚替代 BCE。修复梯度消失。 |
| 2017 | Spectral normalization | 对判别器施加 Lipschitz 约束。2026 年的判别器仍在使用。 |
| 2018 | Progressive GAN | 先训练低分辨率，再添加层。首次得到百万像素级结果。 |
| 2019 | StyleGAN / StyleGAN2 | 映射网络 + 自适应实例归一化。固定领域照片级真实感的最先进水平。 |
| 2021 | StyleGAN3 | 无 alias、平移等变——到 2026 年仍是人脸黄金标准。 |
| 2022 | StyleGAN-XL | 条件化、类别感知、更大规模。 |
| 2024 | R3GAN | 以更强正则化重新包装；无需技巧即可在 1024² 上工作。 |

```figure
gan-minimax
```

## 构建它

`code/main.py` 在一维数据上训练一个小型 GAN：两个高斯的混合。生成器和判别器都是单隐藏层 MLP。我们手写实现前向、反向和 minimax 循环。目标是在它们发生时看见两个关键失效模式（模式坍塌 + 梯度消失）。

### 第 1 步：非饱和损失

原版 Goodfellow 损失 `log(1 - D(G(z)))` 会在 D 高置信度地把 G 的伪造样本判为伪造时趋近于 0。此时 G 的梯度基本为零——G 无法改进。非饱和形式 `-log D(G(z))` 的渐近行为相反：当 D 很自信时它会爆大，给 G 一个强信号。

```python
def g_loss(d_fake):
    # maximize log D(G(z))  <=>  minimize -log D(G(z))
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### 第 2 步：每一步生成器对应一步判别器

```python
for step in range(steps):
    # train D
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # train G
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # fresh fakes
    update_G(fake_batch)
```

给 G 使用新鲜伪造样本，否则梯度是过期的。

### 第 3 步：观察模式坍塌

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] mode collapse: one mode is starved")
```

典型症状：两个真实模式中的一个不再被生成。判别器停止纠正它，因为它从未作为伪造样本出现过。

## 陷阱

- **判别器太强。** 将 D 的学习率降低 2-5 倍，或加入 instance/layer 噪声。如果 D 达到 >95% 准确率，G 就死了。
- **生成器记住了一个模式。** 给 D 输入加噪声，使用 minibatch-discriminator 层，或切换到 WGAN-GP。
- **Batch norm 泄露统计量。** 真实 batch + 伪造 batch 经过同一个 BN 层会混合它们的统计量。改用 instance norm 或 spectral norm。
- **Inception-score gaming。** FID 和 IS 在低样本数下很噪。评估时使用 ≥10k 样本。
- **条件任务里一次采样是谎言。** 你仍然需要 CFG scale、truncation trick 和重新采样，才能得到可用输出。

## 使用它

2026 年的 GAN 技术栈：

| 场景 | 选择 |
|-----------|------|
| 照片级真人脸、固定姿态 | StyleGAN3（最清晰、最小） |
| 动漫 / 风格化人脸 | StyleGAN-XL 或 Stable Diffusion LoRA |
| 图像到图像翻译 | Pix2Pix / CycleGAN（第 8 阶段 · 04）或 ControlNet（第 8 阶段 · 08） |
| 快速一步文生图 | 扩散的对抗蒸馏（SDXL-Turbo、SD3-Turbo） |
| 扩散训练器内部的感知损失 | 在图像裁剪上训练的小型 GAN 判别器 |
| 任何多模态、开放式任务 | 不要用——用扩散或流匹配 |

GAN 清晰但狭窄。一旦你的领域打开——照片、任意文本提示、视频——就切换到扩散。对抗技巧会作为组件继续存在（感知损失、蒸馏），而不是作为独立生成器。

## 交付它

保存 `outputs/skill-gan-debugger.md`。技能接受一次失败的 GAN 运行（损失曲线、样本网格、数据集大小），并输出可能原因的排序列表、一行修复建议，以及重新运行协议。

## 练习

1. **简单。** 使用默认设置运行 `code/main.py`。然后设置 `D_LR = 5 * G_LR` 并重新运行。G 的损失多快坍塌到一个常数？
2. **中等。** 用 WGAN 损失替换 Goodfellow BCE 损失：`loss_D = E[D(fake)] - E[D(real)]`，`loss_G = -E[D(fake)]`，并把 D 的权重裁剪到 `[-0.01, 0.01]`。训练是否更稳定？比较墙钟收敛时间。
3. **困难。** 把一维示例扩展到二维数据（环上的 8 个高斯混合）。追踪生成器在 1k、5k、10k 步时捕捉了 8 个模式中的多少个。实现 minibatch discrimination 并重新测量。

## 关键术语

| 术语 | 人们通常怎么说 | 它实际意味着什么 |
|------|-----------------|-----------------------|
| 生成器 | "G" | 噪声到样本的网络，`G: z → x̂`。 |
| 判别器 | "D" | 分类器 `D: x → [0, 1]`，判断真实 vs 伪造。 |
| Minimax | "这个博弈" | 联合目标的 `min_G max_D`。 |
| 非饱和损失 | "那个修复" | 对 G 使用 `-log D(G(z))`，而不是 `log(1 - D(G(z)))`。 |
| 模式坍塌 | "G 记住了一个东西" | 尽管数据多样，生成器只产生少数不同输出。 |
| WGAN | "Wasserstein" | 用 Earth-Mover 距离 + 梯度惩罚替代 BCE；梯度更平滑。 |
| 谱归一化 | "Lipschitz 技巧" | 约束 D 的权重范数以限制其斜率；稳定训练。 |
| StyleGAN | "那个能用的" | 映射网络 + AdaIN；人脸领域同类最佳，2026 年仍如此。 |

## 生产说明：一次前向推理是 GAN 持久的优势

GAN 在开放领域生成的样本质量上不再胜出，但它们仍然赢在推理成本上。用生产推理文献的词汇来说，GAN 具有：

- **没有 prefill，也没有 decode 阶段。** 一次 `G(z)` 前向传播。TTFT ≈ 总延迟。
- **没有 KV-cache 压力。** 唯一状态是权重。批大小受激活内存限制，而不是缓存限制。
- **连续批处理很简单。** 因为每个请求都消耗相同的固定 FLOPs，所以服务器目标占用率下的静态 batch 通常就是最优。不需要 in-flight 调度器。

这就是为什么 GAN 蒸馏（SDXL-Turbo、SD3-Turbo、ADD、LCM）是 2026 年快速文生图的主导技术：它把 20-50 步的扩散流水线压缩成 1-4 次 GAN 风格的前向传播，同时保留扩散基座的分布。对抗损失作为训练时旋钮继续存在，用来把慢生成器变成快生成器。

## 延伸阅读

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — 原始 GAN 论文。
- [Radford et al. (2015). Unsupervised Representation Learning with DCGAN](https://arxiv.org/abs/1511.06434) — 第一个稳定架构。
- [Arjovsky, Chintala, Bottou (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875) — WGAN。
- [Miyato et al. (2018). Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957) — SN。
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2。
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3。
- [Sauer et al. (2023). Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) — SDXL-Turbo。
