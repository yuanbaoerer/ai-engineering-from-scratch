# ControlNet、LoRA 与条件控制

> 仅靠文本是一种笨拙的控制信号。ControlNet 让你克隆一个预训练扩散模型，并用深度图、姿态骨架、涂鸦或边缘图来引导它。LoRA 让你只训练 1000 万个参数，就能微调一个 20 亿参数模型。二者合在一起，把 Stable Diffusion 从玩具变成了 2026 年每家机构都在交付的图像 pipeline。

**类型：** 构建
**语言：** Python
**先修要求：** 第 8 阶段 · 07（潜在扩散）、第 10 阶段（从零构建 LLM——LoRA 基础）
**时间：** 约 75 分钟

## 问题

像“一位穿红裙子的女人在繁忙街道上遛狗”这样的提示词，并没有告诉模型狗在*哪里*、女人是什么*姿态*，也没有说明街道的*透视*。文本大约只能钉住你指定一张图所需信息的 10%。其余信息是视觉性的，无法用文字高效描述。

为每一种信号（姿态、深度、Canny、分割）从零训练一个新的条件模型，成本过高。你希望冻结 26 亿参数的 SDXL 主干，接上一个读取条件的小型侧网络，让它轻推主干的中间特征。这就是 ControlNet。

你还希望教会模型新的概念（你的脸、你的产品、你的风格），但不重新训练整个模型。你想要的是小 100 倍的增量。这就是 LoRA——插入到现有注意力权重中的低秩适配器（low-rank adapters）。

ControlNet + LoRA + 文本 = 2026 年实践者的工具箱。多数生产图像 pipeline 会在 SDXL / SD3 / Flux 基座之上叠加 2-5 个 LoRA、1-3 个 ControlNet，以及一个 IP-Adapter。

## 概念

![ControlNet 克隆编码器；LoRA 添加低秩增量](../assets/controlnet-lora.svg)

### ControlNet（Zhang 等人，2023）

取一个预训练 SD。*克隆* U-Net 的编码器半边。冻结原模型。训练这个克隆，让它接受额外条件输入（边缘、深度、姿态）。用*零卷积*跳连（初始化为零的 1×1 卷积——一开始是 no-op，然后学习一个增量）把克隆接回原模型的解码器半边。

```
SD U-Net decoder:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

零卷积初始化意味着 ControlNet 一开始等价于恒等操作——即使训练前也不会造成伤害。用 100 万组三元组（提示词、条件、图像）和标准扩散损失训练。

各模态的 ControlNet 会作为小型侧模型发布（SDXL 约 360M，SD 1.5 约 70M）。推理时可以组合它们：

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA（Hu 等人，2021）

对模型中的任意线性层 `W ∈ R^{d×d}`，冻结 `W` 并添加一个低秩增量：

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

其中 `r << d`。注意力层常用秩 4-16，重度微调常用秩 64-128。新参数数量是 `2 · d · r`，而不是 `d²`。对于 `d=640` 的 SDXL 注意力层，`r=16` 时：每个适配器 2 万参数，而不是 41 万参数——减少 20 倍。放到整个模型上：一个 LoRA 通常是 20-200MB，而基座模型是 5GB。

推理时可以缩放 LoRA：`W' = W + α · B @ A`。`α = 0.5-1.5` 是常见范围。多个 LoRA 会加性叠加（但要记住通常的注意事项：它们会以非线性方式相互作用）。

### IP-Adapter（Ye 等人，2023）

一个很小的适配器，接受一张*图像*作为条件（与文本并列）。它使用 CLIP 图像编码器生成图像 token，并将其与文本 token 一起注入交叉注意力。每个基座模型约 20MB。让你无需训练 LoRA，也能做“生成一张具有这个参考图风格的图像”。

## 可组合性矩阵

| 工具 | 控制什么 | 大小 | 何时使用 |
|------|------------------|------|-------------|
| ControlNet | 空间结构（姿态、深度、边缘） | 70-360MB | 精确布局、构图 |
| LoRA | 风格、主体、概念 | 20-200MB | 个性化、风格 |
| IP-Adapter | 来自参考图的风格或主体 | 20MB | 无法用文本描述外观时 |
| Textual Inversion | 把单一概念作为新 token | 10KB | 旧方案，大多已被 LoRA 替代 |
| DreamBooth | 对主体做全量微调 | 2-5GB | 强身份一致性、高计算量 |
| T2I-Adapter | 更轻量的 ControlNet 替代品 | 70MB | 边缘设备、推理预算有限 |

ControlNet ≈ 空间。LoRA ≈ 语义。二者都要用。

## 构建它

`code/main.py` 在 1-D 上模拟这两种机制：

1. **LoRA。** 一个预训练线性层 `W`。冻结它。训练一个低秩 `B @ A`，使 `W + BA` 匹配目标线性层。展示 `r = 1` 足以完美学习一个 rank-1 修正。

2. **ControlNet-lite。** 一个“冻结基座”预测器，以及一个读取额外信号的“侧网络”。侧网络输出由一个初始化为零的可学习标量门控（我们的零卷积版本）控制。训练并观察这个 gate 逐渐升高。

### 步骤 1：LoRA 数学

```python
def lora(W, A, B, x, alpha=1.0):
    # W 冻结；A、B 是可训练的低秩因子。
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### 步骤 2：零初始化侧网络

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate 初始化为 0
h = base(x) + gated
```

在第 0 步，输出与基座完全相同。训练早期 `gate` 会缓慢更新——不会发生灾难性漂移。

## 陷阱

- **过度放大 LoRA。** `α = 2` 或 `α = 3` 是常见的“让它更强”黑客做法，但会产生过度风格化 / 破损的输出。保持 `α ≤ 1.5`。
- **ControlNet 权重冲突。** 同时使用权重 1.0 的 Pose ControlNet 和权重 1.0 的 Depth ControlNet 通常会过冲。权重总和 ≈ 1.0 是安全默认值。
- **LoRA 用在错误基座上。** SDXL LoRA 在 SD 1.5 上会悄悄无效，因为注意力维度不匹配。Diffusers 0.30+ 会发出警告。
- **Textual Inversion 漂移。** 在一个 checkpoint 上训练的 token，到另一个 checkpoint 上会严重漂移。LoRA 更可移植。
- **LoRA 权重合并与存储。** 你可以把 LoRA 烘焙进基座模型权重以加快推理（运行时无需相加），但会失去运行时缩放 `α` 的能力。两种版本都保留。

## 使用它

| 目标 | 2026 pipeline |
|------|---------------|
| 复现某品牌的艺术风格 | 用约 30 张精选图以 rank 32 训练 LoRA |
| 把我的脸放进生成图 | DreamBooth 或 LoRA + IP-Adapter-FaceID |
| 指定姿势 + 提示词 | ControlNet-Openpose + SDXL + 文本 |
| 深度感知构图 | ControlNet-Depth + SD3 |
| 参考图 + 提示词 | IP-Adapter + 文本 |
| 精确布局 | ControlNet-Scribble 或 ControlNet-Canny |
| 替换背景 | ControlNet-Seg + Inpainting（第 09 课） |
| 快速 1 步风格化 | SDXL-Turbo 上的 LCM-LoRA |

## 交付它

保存 `outputs/skill-sd-toolkit-composer.md`。该技能接收一个任务（输入资产：提示词、可选参考图、可选姿态、可选深度、可选涂鸦），并输出工具栈、权重，以及可复现的随机种子协议。

## 练习

1. **简单。** 在 `code/main.py` 中，把 LoRA rank `r` 从 1 改到 4。LoRA 在哪个 rank 下能精确匹配一个 rank-2 目标增量？
2. **中等。** 在两个目标变换上分别训练两个 LoRA。把它们一起加载，并展示它们的加性相互作用。什么时候这种相互作用会打破线性？
3. **困难。** 使用 diffusers 叠加：SDXL-base + Canny-ControlNet（权重 0.8）+ 风格 LoRA（α 0.8）+ IP-Adapter（权重 0.6）。随着堆栈权重变化，测量 FID 与提示词遵循度之间的权衡。

## 关键术语

| 术语 | 大家怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| ControlNet | “空间控制” | 克隆编码器 + 零卷积跳连；读取条件图像。 |
| Zero convolution | “一开始是恒等” | 初始化为零的 1×1 卷积；ControlNet 一开始是 no-op。 |
| LoRA | “低秩适配器” | `W + B @ A`，`r << d`；参数量比全量微调少 100 倍。 |
| rank r | “那个旋钮” | LoRA 压缩率；典型值 4-16，重度个性化用 64+。 |
| α | “LoRA 强度” | LoRA 增量的运行时缩放。 |
| IP-Adapter | “参考图” | 通过 CLIP-image token 实现的小型图像条件适配器。 |
| DreamBooth | “主体全量微调” | 在约 30 张某主体图像上训练完整模型。 |
| Textual Inversion | “新 token” | 只学习一个新的词嵌入；旧方案，大多已被替代。 |

## 生产注记：LoRA 热切换、ControlNet 通道、多租户服务

真实的文生图 SaaS 会在同一个基座 checkpoint 上服务数百个 LoRA 和十几个 ControlNet。这个服务问题看起来很像 LLM 多租户（生产文献通常在 continuous batching 和 LoRAX / S-LoRA 下讨论 LLM 场景）：

- **热切换 LoRA，不要合并。** 把 `W' = W + α·B·A` 合并进基座，会让每步推理快约 3-5%，但会冻结 `α` 和基座。把 LoRA 作为 rank-r 增量常驻显存；diffusers 暴露了 `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])`，用于按请求激活。切换成本就是 `2 · d · r · num_layers` 的权重——MB 级，亚秒级。
- **ControlNet 是第二条注意力通道。** 克隆编码器与基座并行运行。两个权重 1.0 的 ControlNet = 每步多两次额外前向传播，而不是一次合并前向。批大小余量会平方级下降。按每个活跃 ControlNet 约 1.5× 的 step 成本做预算。
- **LoRA 也可以量化。** 如果你量化了基座（见第 07 课，8GB 上跑 Flux），LoRA 增量也能干净地量化到 8-bit 或 4-bit。QLoRA 风格加载让你可以在 4-bit Flux 基座上叠加 5-10 个 LoRA，而不会爆内存。

Flux 专用：Niels 的 Flux-on-8GB notebook 将基座量化到 4-bit；在这个量化基座上叠加风格 LoRA（`pipe.load_lora_weights("user/style-lora")`），并设置 `weight_name="pytorch_lora_weights.safetensors"`，仍然可用。这是 2026 年大多数 SaaS 机构交付的配方。

## 延伸阅读

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — ControlNet。
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA（最初用于 LLM；后来移植到扩散模型）。
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — IP-Adapter。
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — ControlNet 的更轻量替代方案。
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — DreamBooth。
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter docs](https://huggingface.co/docs/diffusers/training/controlnet) — 参考 pipeline。
