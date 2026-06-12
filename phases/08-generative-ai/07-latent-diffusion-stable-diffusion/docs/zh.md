# 潜在扩散与 Stable Diffusion

> 在 512×512 图像的像素空间里做扩散，计算量简直是"计算犯罪"。Rombach 等人（2022）注意到：生成一张图并不需要全部 78.6 万个维度——你只需要足够捕捉语义结构的表示，其余细节交给单独的解码器即可。把扩散过程放进 VAE 的潜在空间里运行。仅仅这个想法，就是 Stable Diffusion。

**类型：** 构建
**语言：** Python
**先修要求：** 第 8 阶段 · 02（VAE）、第 8 阶段 · 06（DDPM）、第 7 阶段 · 09（ViT）
**时间：** 约 75 分钟

## 问题

在 512² 的像素空间中做扩散，意味着 U-Net 要运行在形状为 `[B, 3, 512, 512]` 的张量上。对于一个 5 亿参数的 U-Net，每个采样步骤大约需要 100 GFLOPS。50 步就是每张图 5 TFLOPS。如果用十亿张图训练，计算账单会荒唐到不可接受。

这些 FLOPs 中的大部分都花在把感知上不重要的细节推过网络——例如那些有损 VAE 本可以压缩掉的高频纹理。Rombach 的想法是：先训练一次 VAE（*第一阶段*），冻结它，然后完全在 4 通道 64×64 的潜在空间（*第二阶段*）中运行扩散。同样的 U-Net。像素数只有 1/16。FLOPs 大约减少 64 倍，却能得到相近质量。

这就是 Stable Diffusion 的配方。SD 1.x / 2.x 使用在 `64×64×4` 潜变量上运行的 8.6 亿参数 U-Net，SDXL 使用在 `128×128×4` 潜变量上运行的 26 亿参数 U-Net，SD3 则把 U-Net 换成了带流匹配（flow matching）的扩散 Transformer（Diffusion Transformer, DiT）。Flux.1-dev（Black Forest Labs，2024）发布的是 120 亿参数的 DiT-MMDiT。它们都运行在同一个两阶段基底上。

## 概念

![潜在扩散：VAE 压缩 + 潜在空间中的扩散](../assets/latent-diffusion.svg)

**两个阶段，分开训练。**

1. **阶段 1 — VAE。** 编码器 `E(x) → z`，解码器 `D(z) → x`。目标压缩率：每个空间轴下采样 8×，并调整通道数，使总潜在大小约为像素数的 1/16。损失 = 重建损失（L1 + LPIPS 感知损失）+ KL（权重较小，这样 `z` 不会被强行压成过于高斯，因为我们并不需要从 `z` 做精确采样）。通常还会配合对抗损失训练，让解码出的图像更锐利。

2. **阶段 2 — 在 `z` 上做扩散。** 把 `z = E(x_real)` 当作数据。训练一个 U-Net（或 DiT）来去噪 `z_t`。推理时：通过扩散采样出 `z_0`，然后令 `x = D(z_0)`。

**文本条件。** 还需要两个额外组件。一个冻结的文本编码器（SD 1.x 使用 CLIP-L，SD 2/XL 使用 CLIP-L+OpenCLIP-G，SD3 和 Flux 使用 T5-XXL）。以及一个交叉注意力注入机制：每个 U-Net 块接收 `[Q = 图像特征, K = V = 文本 token]` 并将其混合进去。这些 token 是文本影响图像的唯一通道。

**损失函数与第 06 课完全相同。** 仍然是 DDPM / flow matching 对噪声的 MSE。你只是替换了数据所在的域。

## 架构变体

| 模型 | 年份 | 主干 | 潜变量形状 | 文本编码器 | 参数量 |
|-------|------|----------|--------------|--------------|--------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L（77 个 token） | 860M |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 865M |
| SDXL | 2023 | U-Net + refiner | 128×128×4 | CLIP-L + OpenCLIP-G | 2.6B + 6.6B |
| SDXL-Turbo | 2023 | 蒸馏模型 | 128×128×4 | same | 1-4 步采样 |
| SD3 | 2024 | MMDiT（多模态 DiT） | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 2B / 8B |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B |
| Flux.1-schnell | 2024 | MMDiT 蒸馏版 | 128×128×16 | T5-XXL + CLIP-L | 12B，1-4 步 |

趋势是：用 DiT 替换 U-Net（在潜在 patch 上运行的 Transformer），扩大文本编码器（T5 在提示词遵循度上优于 CLIP），增加潜变量通道数（4 → 16 带来更多细节余量）。

```figure
noise-schedule
```

## 构建它

`code/main.py` 把一个玩具版 1-D "VAE"（恒等编码器 + 解码器，仅用于演示；真实 VAE 会是卷积网络）叠在第 06 课的 DDPM 之上，并加入带无分类器引导（classifier-free guidance）的类别条件。它展示了同一个扩散损失无论作用在原始 1-D 值上，还是作用在编码值上，都同样有效——这是关键洞见。

### 步骤 1：编码器/解码器

```python
def encode(x):    return x * 0.5          # 玩具"压缩"到更小尺度
def decode(z):    return z * 2.0
```

真实 VAE 有训练好的权重。为了教学，这个线性映射已经足够展示：扩散在 `z` 上运行，并不关心原始数据空间是什么。

### 步骤 2：在 `z` 空间中做扩散

与第 06 课相同的 DDPM。网络看到的数据是 `z = E(x)`。采样得到 `z_0` 后，用 `D(z_0)` 解码。

### 步骤 3：无分类器引导

训练期间，10% 的时间丢弃类别标签（替换为空 token）。推理时，同时计算 `ε_cond` 和 `ε_uncond`，然后：

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0` = 无引导（多样性最高），`w = 3` = 默认值，`w = 7+` = 饱和 / 过度锐化。

### 步骤 4：文本条件（概念，不是代码）

用冻结文本编码器的输出替换类别标签。通过交叉注意力把文本嵌入送入 U-Net：

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

这是类别条件扩散模型与 Stable Diffusion 之间唯一实质性的差异。

## 陷阱

- **VAE 尺度不匹配。** SD 1.x 的 VAE 在编码后会应用一个缩放常数（`scaling_factor ≈ 0.18215`）。忘记它会让 U-Net 在方差严重错误的潜变量上训练。每个 checkpoint 都会附带这个值。
- **文本编码器悄悄出错。** SD3 需要 T5-XXL 且 token 数 >=128；退回到仅 CLIP 会有明显损失。始终检查 `use_t5=True`，否则提示词保真度会崩。
- **混用潜在空间。** SDXL、SD3、Flux 都使用不同的 VAE。在 SDXL 潜变量上训练的 LoRA 不能用于 SD3。Hugging Face diffusers 0.30+ 会拒绝加载不匹配的 checkpoint。
- **CFG 太高。** `w > 10` 会生成饱和、油腻的图像，并以牺牲多样性为代价过拟合提示词。甜点区间是 `w = 3-7`。
- **负向提示词泄漏。** 空的负向提示词会变成 null token；填写了内容的负向提示词会变成 `ε_uncond`。它们不是一回事；有些 pipeline 会悄悄默认使用 null。

## 使用它

2026 年的生产栈：

| 目标 | 推荐主干 |
|--------|----------------------|
| 窄领域、成对数据、从零训练模型 | SDXL 微调（LoRA / 全量）——最快上线 |
| 开放域文生图、开放权重 | Flux.1-dev（12B，Apache / 非商用）或 SD3.5-Large |
| 最快推理、开放权重 | Flux.1-schnell（1-4 步，Apache）或 SDXL-Lightning |
| 最佳提示词遵循度、托管服务 | GPT-Image / DALL-E 3（仍然如此）、Midjourney v7、Imagen 4 |
| 编辑工作流 | Flux.1-Kontext（2024 年 12 月）——原生接受图像 + 文本 |
| 研究、基线 | SD 1.5——古老但研究充分 |

## 交付它

保存 `outputs/skill-sd-prompter.md`。该技能接收文本提示词 + 目标风格，并输出：模型 + checkpoint、CFG scale、采样器、负向提示词、分辨率、可选 ControlNet/IP-Adapter 组合，以及逐步 QA 检查清单。

## 练习

1. **简单。** 用引导 `w ∈ {0, 1, 3, 7, 15}` 运行 `code/main.py`。记录每个类别的样本均值。在哪个 `w` 下，类别均值会偏离真实数据均值之外？
2. **中等。** 把玩具线性编码器替换为 tanh-MLP 编码器/解码器对，并加入重建损失。重新在新潜变量上训练扩散。样本质量是否变化？
3. **困难。** 用 diffusers 搭建一个真实 Stable Diffusion 推理：加载 `sdxl-base`，用 CFG=7 运行 30 步 Euler，计时。然后切换到 `sdxl-turbo`，用 4 步和 CFG=0。相同主体，不同质量——描述发生了什么变化以及原因。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| First stage | "VAE" | 训练好的编码器/解码器对；把 512² 压缩到 64²。 |
| Second stage | "U-Net" | 作用在潜在空间上的扩散模型。 |
| CFG | "引导尺度" | `(1+w)·ε_cond - w·ε_uncond`；调节条件强度。 |
| Null token | "空提示词嵌入" | 用于 `ε_uncond` 的无条件嵌入。 |
| Cross-attention | "文本进入模型的方式" | 每个 U-Net 块都把文本 token 当作 K 和 V 来注意。 |
| DiT | "Diffusion Transformer" | 用潜在 patch 上的 Transformer 替换 U-Net；扩展性更好。 |
| MMDiT | "Multi-modal DiT" | SD3 的架构：文本流和图像流使用联合注意力。 |
| VAE scaling factor | "魔法数字" | 将潜变量除以约 5.4，使扩散在单位方差空间中运行。 |

## 生产注记：在 8GB 消费级 GPU 上运行 Flux-12B

参考 Flux 集成是典型的"我有一张消费级 GPU，能上线吗？"配方。诀窍与生产推理文献中列出的三旋钮配方相同，只是应用到扩散 DiT 上：

1. **交错加载。** Flux 有三个网络永远不需要同时共存于显存中：T5-XXL 文本编码器（fp32 下约 10 GB）、CLIP-L（较小）、12B MMDiT，以及 VAE。先编码提示词，*删除* 编码器，加载 DiT，去噪，*删除* DiT，加载 VAE，解码。8GB 消费级 GPU 一次只能装下一个阶段。
2. **通过 bitsandbytes 做 4-bit 量化。** 在 T5 编码器和 DiT 上都使用 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)`。内存减少 8×，按照 Aritra 的基准（notebook 中有链接），文生图质量下降几乎不可感知。
3. **CPU offload。** `pipe.enable_model_cpu_offload()` 会随着每次前向传播推进，在 CPU 和 GPU 之间自动交换模块。延迟增加 10-20%，但能让 pipeline 跑起来。

内存账是：量化后的 `10 GB T5 / 8 = 1.25 GB`，量化 DiT 为 `12 B params × 0.5 bytes = ~6 GB`，再加上激活值。用 stas00 的说法，这是 TP=1 推理的极端端点——没有模型并行，最大化量化。生产中你会在 H100 上运行 TP=2 或 TP=4；对于单台开发笔记本，这就是配方。

## 延伸阅读

- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion。
- [Podell et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) — SDXL。
- [Peebles & Xie (2023). Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) — DiT。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3、MMDiT。
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG。
- [Labs (2024). Flux.1 — Black Forest Labs announcement](https://blackforestlabs.ai/announcing-black-forest-labs/) — Flux.1 家族。
- [Hugging Face Diffusers docs](https://huggingface.co/docs/diffusers/index) — 上述所有 checkpoint 的参考实现。
