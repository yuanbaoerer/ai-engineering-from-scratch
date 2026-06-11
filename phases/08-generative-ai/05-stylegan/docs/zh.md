# StyleGAN

> 大多数生成器会把 `z` 同时搅入每一层。StyleGAN 把这件事拆开：先把 `z` 映射到中间变量 `w`，再通过 AdaIN 在每个分辨率层级*注入* `w`。这一个改变解开了潜空间的纠缠，并让照片级真实人脸在连续七年里成为已解决的问题。

**类型：** 构建
**语言：** Python
**先修：** 阶段 8 · 03（GANs）、阶段 4 · 08（Normalization）、阶段 3 · 07（CNNs）
**时间：** 约 45 分钟

## 问题

DCGAN 通过一堆转置卷积把 `z` 映射成图像。问题是：`z` 控制一切——姿态、光照、身份、背景——全都纠缠在一起。沿着 `z` 的一个轴移动，四者都会改变。你无法要求模型“同一个人，不同姿态”，因为表示并不是这样分解的。

Karras 等（2019，NVIDIA）提出：停止把 `z` 直接喂给卷积层。把一个常量 `4×4×512` 张量作为网络输入。学习一个 8 层 MLP，把 `z ∈ Z → w ∈ W`。通过*自适应实例归一化*（adaptive instance normalization，AdaIN）在每个分辨率注入 `w`：归一化每个卷积特征图，然后用 `w` 的仿射投影进行缩放和平移。为随机细节（皮肤毛孔、发丝）加入逐层噪声。

结果是：`W` 大致拥有正交的轴，可区分“高层风格”（姿态、身份）与“细节风格”（光照、颜色）。你可以在两张图之间交换风格：低分辨率层使用图像 A 的 `w`，高分辨率层使用图像 B 的 `w`。这解锁了编辑、跨域风格化，以及整个“StyleGAN 反演”（StyleGAN inversion）研究方向。

## 概念

![StyleGAN: mapping network + AdaIN + per-layer noise](../assets/stylegan.svg)

**映射网络。** `f: Z → W`，一个 8 层 MLP。`Z = N(0, I)^512`。`W` 不被强制为高斯分布——它会学习适配数据的形状。

**合成网络。** 从一个可学习常量 `4×4×512` 开始。每个分辨率块：`upsample → conv → AdaIN(w_i) → noise → conv → AdaIN(w_i) → noise`。分辨率翻倍：4、8、16、32、64、128、256、512、1024。

**AdaIN。**

```
AdaIN(x, y) = y_scale · (x - mean(x)) / std(x) + y_bias
```

其中 `y_scale` 和 `y_bias` 来自 `w` 的仿射投影。先按特征图归一化，再重新赋予风格。这里的“风格”是特征图的一阶和二阶统计量。

**逐层噪声。** 向每个特征图加入单通道高斯噪声，并由可学习的逐通道因子缩放。它控制随机细节，而不影响全局结构。

**截断技巧。** 推理时，采样 `z`，计算 `w = mapping(z)`，然后 `w' = ŵ + ψ·(w - ŵ)`，其中 `ŵ` 是许多样本上的平均 `w`。`ψ < 1` 用多样性换质量。几乎每个 StyleGAN 演示都使用 `ψ ≈ 0.7`。

## StyleGAN 1 → 2 → 3

| 版本 | 年份 | 创新 |
|---------|------|------------|
| StyleGAN | 2019 | 映射网络 + AdaIN + 噪声 + 渐进式增长。 |
| StyleGAN2 | 2020 | 权重解调（weight demodulation）取代 AdaIN（修复水滴伪影）；跳连/残差架构；路径长度正则化。 |
| StyleGAN3 | 2021 | 无混叠卷积（alias-free convolution）+ 等变核；消除纹理粘在像素网格上的问题。 |
| StyleGAN-XL | 2022 | 类条件、1024²、ImageNet。 |
| R3GAN | 2024 | 用更强正则重新包装；以少 20 倍参数在 FFHQ-1024 上缩小与扩散模型的差距。 |

在 2026 年，StyleGAN3 仍然是以下场景的默认选择：(a) 狭窄领域的高 FPS 照片级真实生成，(b) 小样本领域自适应（用 100 张图训练新数据集，冻结映射网络），(c) 基于反演的编辑（找到能重建真实照片的 `w`，再编辑这个 `w`）。对于开放域文本到图像，它不是合适工具——扩散模型才是。

## 构建它

`code/main.py` 实现了一个 1-D 玩具版“style-GAN lite”：一个映射 MLP，一个合成函数（接收可学习常量向量，并用从 `w` 派生的 scale/bias 进行调制），以及逐层噪声。它展示了通过仿射调制注入 `w`，能够匹配或优于把 `z` 拼接到生成器输入中。

### 步骤 1：映射网络

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### 步骤 2：自适应实例归一化

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

每个特征图的 scale 和 bias 都通过线性投影来自 `w`。

### 步骤 3：逐层噪声

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

每个通道的 Sigma 是可学习的。

## 常见陷阱

- **水滴伪影。** StyleGAN 1 会在特征图中产生团块状水滴，因为 AdaIN 把均值归零了。StyleGAN 2 的权重解调通过缩放卷积权重而不是激活来修复这个问题。
- **纹理粘连。** StyleGAN 1 和 2 的纹理会跟随像素坐标，而不是物体坐标（插值时可见）。StyleGAN 3 的无混叠卷积用加窗 sinc 滤波器修复了这一点。
- **模式覆盖。** 截断 `ψ < 0.7` 看起来干净，但只从一个狭窄锥体中采样；如果需要多样性，请使用 `ψ = 1.0`。
- **反演是有损的。** 把真实照片反演到 `W` 通常通过优化或编码器完成（e4e、ReStyle、HyperStyle）。多次迭代后结果会漂移。

## 使用它

| 使用场景 | 方法 |
|----------|----------|
| 照片级真实人脸（动漫、产品、狭窄领域） | StyleGAN3 FFHQ / 自定义微调 |
| 从照片做人脸编辑 | e4e 反演 + StyleSpace / InterFaceGAN 方向 |
| 换脸 / 重演 | StyleGAN + 编码器 + 融合 |
| 头像流水线 | 使用 ADA 的 StyleGAN3 低数据微调 |
| 从少量图像做领域自适应 | 冻结映射网络，微调合成网络 |
| 多模态或文本条件生成 | 不要用——使用扩散模型 |

对于答案是“某个人脸照片”的产品级演示，StyleGAN 在推理成本（单次前向，在 4090 上 <10ms）和同等质量门槛下的锐利度方面胜过扩散模型。

## 交付它

保存 `outputs/skill-stylegan-inversion.md`。该技能接收一张真实照片并输出：反演方法（e4e / ReStyle / HyperStyle）、预期潜变量损失、编辑预算（在出现伪影之前你可以在 `W` 中移动多远），以及一组已知好用的编辑方向（年龄、表情、姿态）。

## 练习

1. **简单。** 使用 `adain_on=True` 和 `adain_on=False` 运行 `code/main.py`。比较固定潜变量和扰动潜变量下输出的分布宽度。
2. **中等。** 实现混合正则化（mixing regularization）：对于一个训练批次，计算 `w_a`、`w_b`，并在合成前半段使用 `w_a`、后半段使用 `w_b`。解码器是否学到了可解耦风格？
3. **困难。** 取一个预训练 StyleGAN3 FFHQ 模型（ffhq-1024.pkl）。通过在带标签样本上训练 SVM，找到控制“微笑”的 `w` 方向；报告在身份漂移之前可以推动多远。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Mapping network | “那个 MLP” | `f: Z → W`，8 层，把潜空间几何与数据统计解耦。 |
| W space | “风格空间” | 映射网络的输出；大致解耦。 |
| AdaIN | “自适应实例归一化” | 归一化特征图，然后用 `w` 投影进行缩放 + 平移。 |
| Truncation trick | “Psi” | `w = mean + ψ·(w - mean)`，ψ<1 用多样性换质量。 |
| Path-length regularization | “PL reg” | 惩罚 `w` 单位变化导致的图像大变化；让 `W` 更平滑。 |
| Weight demodulation | “StyleGAN2 的修复” | 归一化卷积权重而不是激活；消除水滴伪影。 |
| Alias-free | “StyleGAN3 的技巧” | 加窗 sinc 滤波器；消除纹理粘在像素网格上的问题。 |
| Inversion | “为真实图像找 w” | 优化或编码 `x → w`，使 `G(w) ≈ x`。 |

## 生产说明：为什么 StyleGAN 在 2026 年仍然能上线

4090 上的 StyleGAN3 可以在 10 ms 内生成一张 1024² FFHQ 人脸——`num_steps = 1`，没有 VAE 解码，没有交叉注意力过程。按生产术语来说，这是任何图像生成器的延迟下限。同分辨率下，50 步 SDXL + VAE 解码流水线约为 3 秒。这是 **300× 差距**，对于狭窄领域产品（头像服务、证件照流水线、库存人脸生成），它在总拥有成本（TCO）上胜出。

两个运营后果：

- **没有调度器，没有批处理器。** 在目标占用率下使用静态批次就是最优。连续批处理（对 LLM 和扩散模型至关重要）没有任何收益，因为每个请求消耗相同 FLOPs。
- **截断 `ψ` 是安全旋钮。** `ψ < 0.7` 从映射网络范围中的狭窄锥体采样。这是服务层控制样本方差的唯一杠杆。高峰负载时降低 `ψ`，为高级用户提高它。

## 延伸阅读

- [Karras et al. (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) — StyleGAN。
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2。
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3。
- [Tov et al. (2021). Designing an Encoder for StyleGAN Image Manipulation](https://arxiv.org/abs/2102.02766) — e4e 反演。
- [Sauer et al. (2022). StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets](https://arxiv.org/abs/2202.00273) — StyleGAN-XL。
- [Huang et al. (2024). R3GAN: The GAN is dead; long live the GAN!](https://arxiv.org/abs/2501.05441) — 现代极简 GAN 配方。
