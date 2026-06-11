# 评估——FID、CLIP Score、人类偏好

> 每个生成模型排行榜都会引用 FID、CLIP score，以及来自人类偏好竞技场的胜率。每个数字都有一个执着的研究者可以钻空子的失效模式。如果你不了解这些失效模式，就无法区分真正的改进和刷榜式运行。

**类型：** 构建
**语言：** Python
**先修要求：** 第 8 阶段 · 01（分类体系），第 2 阶段 · 04（评估指标）
**时间：** 约 45 分钟

## 问题

生成模型通常根据*样本质量*和*条件遵循度*来评判。两者都没有闭式度量。你的模型必须渲染 10,000 张图像；必须有某种东西给它们打分；你还必须相信这些分数在不同模型家族、不同分辨率、不同架构之间具有可比性。三个指标经受住了 2014–2026 年的考验：

- **FID（Fréchet Inception Distance）。** 在 Inception 网络特征空间中，真实分布与生成分布之间的距离。越低越好。
- **CLIP score。** 生成图像的 CLIP 图像嵌入与提示词的 CLIP 文本嵌入之间的余弦相似度。越高越好。衡量提示词遵循度。
- **人类偏好。** 在同一提示词上让两个模型正面对战，由人类（或 GPT-4 级模型）选择更好的一个，并聚合成 Elo 分数。

你还会看到：IS（inception score，基本已退役）、KID、CMMD、ImageReward、PickScore、HPSv2、MJHQ-30k。每个指标都在修正前一个指标的某种失效。

## 概念

![FID、CLIP 和偏好：三个轴，不同失效模式](../assets/evaluation.svg)

### FID——样本质量

Heusel et al. (2017)。步骤：

1. 为 N 张真实图像和 N 张生成图像提取 Inception-v3 特征（2048 维）。
2. 分别为每个样本池拟合一个高斯分布：计算均值 `μ_r, μ_g` 和协方差 `Σ_r, Σ_g`。
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`。

解释：特征空间中两个多元高斯之间的 Fréchet 距离。越低 = 分布越相似。

失效模式：
- **小 N 有偏。** FID 是在特征分布上做均方统计——小 N 会低估协方差，给出虚假偏低的 FID。始终使用 N ≥ 10,000。
- **依赖 Inception。** Inception-v3 训练于 ImageNet。远离 ImageNet 的领域（人脸、艺术、含文字图像）会产生没有意义的 FID。应使用领域特定的特征提取器。
- **刷指标。** 过拟合 Inception 先验可以在不提升视觉质量的情况下得到低 FID。用 CMMD（见下文）来对抗。

### CLIP score——提示词遵循度

Radford et al. (2021)。对于一张生成图像 + 一个提示词：

```
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

在 30k 张生成图像上取平均 → 得到一个可在模型之间比较的标量。

失效模式：
- **CLIP 自身盲点。** CLIP 的组合推理较弱（“a red cube on a blue sphere” 经常失败）。模型可以在 CLIP score 上排名很高，却并没有真正遵循复杂提示。
- **短提示偏置。** 短提示在真实世界中有更多 CLIP 图像匹配。较长提示会机械地得到更低的 CLIP score。
- **提示词刷分。** 在提示词中加入 “high quality, 4k, masterpiece” 会抬高 CLIP score，却不改善图文绑定。

CMMD（Jayasumana et al., 2024）修复了其中一些问题：它使用 CLIP 特征而不是 Inception，并使用 maximum-mean discrepancy 而不是 Fréchet。它更擅长检测细微的质量差异。

### 人类偏好——真实标准

选取一组提示词。分别用模型 A 和模型 B 生成。把成对结果展示给人类（或强 LLM 裁判）。将胜负聚合为 Elo 或 Bradley-Terry 分数。基准包括：

- **PartiPrompts（Google）**：1,600 个多样化提示，12 个类别。
- **HPSv2**：107k 条人类标注，广泛用作自动代理指标。
- **ImageReward**：137k 个提示-图像偏好对，MIT 许可证。
- **PickScore**：在 Pick-a-Pic 的 260 万条偏好上训练。
- **Chatbot-Arena 风格的图像竞技场**：https://imagearena.ai/ 以及其他平台。

失效模式：
- **评审方差。** 非专家与专家的偏好不同。两者都要使用。
- **提示词分布。** 精挑细选的提示词会偏向某个模型家族。必须记录说明。
- **LLM 裁判奖励黑客。** GPT-4 裁判会被“好看但错误”的输出欺骗。要与人类评审交叉验证。

## 组合使用

生产级评估报告应包括：

1. 在 10–30k 个样本上，针对保留的真实分布计算 FID（样本质量）。
2. 在同一批样本上，针对其提示词计算 CLIP score / CMMD（遵循度）。
3. 在盲测竞技场中相对上一代模型的胜率（整体偏好）。
4. 失效模式分析：随机抽取 50 个输出，标记已知问题（手部解剖、文字渲染、一致的物体数量）。

任何单一指标都是谎言。三个相互印证的指标 + 定性审查，才是一个主张。

## 构建它

`code/main.py` 在合成“特征向量”上实现 FID、类 CLIP-score 和 Elo 聚合（我们使用 4 维向量作为 Inception 特征的替身）。你会看到：

- 小 N 和大 N 上的 FID 计算——也就是偏差。
- 用特征池之间的余弦相似度表示的“CLIP score”。
- 来自合成偏好流的 Elo 更新规则。

### 第 1 步：四行实现 FID

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### 第 2 步：CLIP 风格余弦相似度

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### 第 3 步：Elo 聚合

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## 陷阱

- **N=1000 的 FID。** 在 N<10k 时，该启发式指标不可靠。报告低 N FID 的论文是在刷指标。
- **跨分辨率比较 FID。** Inception 的 299×299 resize 会改变特征分布。只在匹配分辨率下比较。
- **只报告一个随机种子。** 至少运行 3 个 seed。报告标准差。
- **通过负向提示词抬高 CLIP score。** 某些流水线会通过过拟合提示来提升 CLIP。检查是否出现视觉饱和。
- **提示词重叠导致 Elo 偏差。** 如果两个模型在训练时都见过某个基准提示，Elo 就没有意义。使用保留提示集。
- **付费众包人类评估偏斜。** Prolific、MTurk 标注者偏年轻 / 技术友好。要混入招募来的艺术 / 设计专家。

## 使用它

2026 年生产评估协议：

| 支柱 | 最低要求 | 推荐做法 |
|--------|---------|-------------|
| 样本质量 | 在 10k 样本上对保留真实集计算 FID | + 5k 样本 CMMD + 按类别子集计算 FID |
| 提示词遵循度 | 在 30k 样本上计算 CLIP score | + HPSv2 + ImageReward + VQA 风格问答 |
| 偏好 | 相对基线的 200 对盲测 | + 2000 对人类评审 + LLM 裁判 + Chatbot Arena |
| 失效分析 | 50 个手动标记 | 500 个手动标记 + 自动安全分类器 |

四个支柱都在一份报告里 = 主张。单独任何一个 = 营销。

## 交付它

保存 `outputs/skill-eval-report.md`。该技能接收一个新模型 checkpoint + 基线，并输出完整评估计划：样本量、指标、失效模式探针、签核标准。

## 练习

1. **简单。** 运行 `code/main.py`。在同一组合成分布上比较 N=100 与 N=1000 的 FID。报告偏差幅度。
2. **中等。** 从合成 CLIP 风格特征实现 CMMD（公式见 Jayasumana et al., 2024）。比较它相对 FID 对质量差异的敏感性。
3. **困难。** 复现 HPSv2 设置：从 Pick-a-Pic 的一个子集中取 1000 个图像-提示对，在偏好数据上微调一个小型 CLIP-based 评分器，并测量它与保留集的一致性。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| FID | “Fréchet Inception Distance” | 对真实与生成的 Inception 特征分别做高斯拟合后的 Fréchet 距离。 |
| CLIP score | “文本-图像相似度” | CLIP 图像嵌入与文本嵌入之间的余弦相似度。 |
| CMMD | “FID 的替代品” | 基于 CLIP 特征的 MMD；偏差更小，不依赖高斯假设。 |
| IS | “Inception score” | Exp KL(p(y|x) || p(y))；在现代模型上相关性很差，已退役。 |
| HPSv2 / ImageReward / PickScore | “学习到的偏好代理” | 在人类偏好上训练的小模型；用作自动裁判。 |
| Elo | “国际象棋评分” | 对成对胜负进行 Bradley-Terry 聚合。 |
| PartiPrompts | “基准提示集” | Google 策划的 1,600 个提示，覆盖 12 个类别。 |
| FD-DINO | “自监督替代品” | 使用 DINOv2 特征的 FD；对 ImageNet 之外的领域更好。 |

## 生产说明：评估也是推理工作负载

在 10k 样本上运行 FID 意味着生成 10k 张图像。对于一个 50 步 SDXL base，在单张 L4 上以 1024² 分辨率运行，这大约是 11 小时的单请求推理。评估预算是真实存在的，其框架正是离线推理场景（最大化吞吐量，忽略 TTFT）：

- **尽可能批处理，忘掉延迟。** 离线评估 = 在显存可容纳的最大尺寸上做静态批处理。在 80GB H100 上用 `num_images_per_prompt=8` 运行 `pipe(...).images`，墙钟时间比单请求快 4–6×。
- **缓存真实特征。** 对真实参考集进行的 Inception（FID）或 CLIP（CLIP-score、CMMD）特征提取只运行*一次*，保存为 `.npz`。不要每次评估都重新计算。

对于 CI / 回归门禁：每个 PR 在 500 样本子集上运行 FID + CLIP score（约 30 分钟）；每晚运行完整 10k FID + HPSv2 + Elo。

## 延伸阅读

- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — FID 论文。
- [Jayasumana et al. (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603) — CMMD。
- [Radford et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — CLIP。
- [Wu et al. (2023). HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) — HPSv2。
- [Xu et al. (2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) — ImageReward。
- [Yu et al. (2023). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) — PartiPrompts。
- [Stein et al. (2023). Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675) — 失效模式综述。
