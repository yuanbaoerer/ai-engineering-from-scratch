# 3D 生成

> 3D 是二维到三维借力最强的模态。2023 年的突破是 3D Gaussian Splatting。2024-2026 年的生成式推进，是在其上叠加多视角扩散 + 3D 重建，从单个提示词或照片生成对象和场景。

**类型：** 学习
**语言：** Python
**先修要求：** 第 4 阶段（视觉）、第 8 阶段 · 07（潜空间扩散，Latent Diffusion）
**时间：** 约 45 分钟

## 问题

3D 内容很棘手：

- **表示。** 网格（mesh）、点云、体素网格、有符号距离场（SDF）、神经辐射场（NeRF）、3D 高斯。每种都有取舍。
- **数据稀缺。** ImageNet 有 1400 万张图像。最大的干净 3D 数据集（Objaverse-XL，2023）约有 1000 万个对象，其中多数质量较低。
- **内存。** 一个 512³ 体素网格有 128M 个体素；一个可用的场景 NeRF 需要每条射线 1M 次采样。生成比重建更难。
- **监督。** 对 2D 图像，你拥有像素。对 3D，你通常只有少量 2D 视图，并且必须把它们提升到 3D。

2026 年的栈把两个问题分开。首先，用扩散模型生成 *2D 多视角图像*。其次，把一个 *3D 表示*（通常是 Gaussian splatting）拟合到这些图像上。

## 核心概念

![3D 生成：多视角扩散 + 3D 重建](../assets/3d-generation.svg)

### 表示：3D Gaussian Splatting（Kerbl et al., 2023）

把场景表示为约 100 万个 3D 高斯组成的云。每个高斯有 59 个参数：位置（3）、协方差（6，或四元数 4 + 尺度 3）、不透明度（1）、球谐颜色（3 阶时 48 个，0 阶时 3 个）。

渲染 = 投影 + alpha 合成。在 4090 上 1080p 约 100 fps。可微。通过对真实照片做梯度下降来拟合。一个场景可在消费级 GPU 上 5-30 分钟内拟合完成。

其上的两个 2023-2024 年创新：
- **生成式 Gaussian splats。** LGM、LRM、InstantMesh 等模型直接从一张或几张图像预测高斯云。
- **4D Gaussian Splatting。** 带逐帧偏移的高斯，用于动态场景。

### 多视角扩散

微调一个预训练图像扩散模型，使其从文本提示或单张图像生成同一对象的多个一致视图。Zero123（Liu et al., 2023）、MVDream（Shi et al., 2023）、SV3D（Stability，2024）、CAT3D（Google，2024）。通常输出物体周围 4-16 个视图，再通过 Gaussian splatting 或 NeRF 提升到 3D。

### 文本到 3D 流水线

| 模型 | 输入 | 输出 | 时间 |
|-------|-------|--------|------|
| DreamFusion (2022) | text | 经 SDS 得到 NeRF | 每个资产约 1 小时 |
| Magic3D | text | 网格 + 纹理 | 约 40 分钟 |
| Shap-E (OpenAI, 2023) | text | 隐式 3D | 约 1 分钟 |
| SJC / ProlificDreamer | text | NeRF / mesh | 约 30 分钟 |
| LRM (Meta, 2023) | image | triplane | 约 5 s |
| InstantMesh (2024) | image | mesh | 约 10 s |
| SV3D (Stability, 2024) | image | 新视角 | 约 2 分钟 |
| CAT3D (Google, 2024) | 1-64 images | 3D NeRF | 约 1 分钟 |
| TripoSR (2024) | image | mesh | 约 1 s |
| Meshy 4 (2025) | text + image | PBR mesh | 约 30 s |
| Rodin Gen-1.5 (2025) | text + image | PBR mesh | 约 60 s |
| Tencent Hunyuan3D 2.0 (2025) | image | mesh | 约 30 s |

2025-2026 年方向：直接生成适合游戏引擎的、带 PBR 材质的文本到网格模型。对通用对象来说，多视角扩散中间步骤仍是表现最好的配方。

### NeRF（背景知识）

神经辐射场（Neural Radiance Field，Mildenhall et al., 2020）。一个小型 MLP 接收 `(x, y, z, view direction)` 并输出 `(color, density)`。通过沿射线积分进行渲染。其质量优于基于网格的新视角合成，但渲染慢 100-1000 倍。在大多数实时用途中已被 Gaussian splatting 取代，但仍在研究中占主导地位。

## 动手构建

`code/main.py` 实现一个玩具版 2D “Gaussian splatting” 拟合：把合成目标图像（平滑渐变）表示为一组 2D Gaussian splats 的和。通过梯度下降优化位置、颜色和协方差来匹配目标。你会看到两个核心操作：前向渲染（splat + alpha-composite）和通过梯度下降拟合。

### 第 1 步：2D Gaussian splat

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### 第 2 步：通过求和 splat 来渲染

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

真实的 3D Gaussian splatting 会按深度排序高斯，并按顺序做 alpha 合成。我们的 2D 玩具例子只是求和。

### 第 3 步：通过梯度下降拟合

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## 常见陷阱

- **视角不一致。** 如果独立生成 4 个视图，而它们对对象结构的描述不一致，3D 拟合会变得模糊。修复方式：使用带共享注意力的多视角扩散。
- **背面幻觉。** 单图 → 3D 必须发明不可见的一侧。质量差异会非常大。
- **Gaussian splat 爆炸。** 无约束训练会增长到 1000 万个 splat 并过拟合。稠密化 + 剪枝启发式（来自 3D-GS 原论文）是必需的。
- **拓扑问题。** 从隐式场（SDF）得到的网格经常有孔洞或自交。交付前先运行重网格工具（例如 Blender 的 voxel remesh）。
- **训练数据许可证。** Objaverse 许可证混杂；商业使用因模型而异。

## 使用它

| 任务 | 2026 年选择 |
|------|-----------|
| 从照片重建场景 | Gaussian splatting（3DGS、Gsplat、Scaniverse） |
| 游戏用文本到 3D 对象 | Meshy 4 或 Rodin Gen-1.5（PBR 输出） |
| 图像到 3D | Hunyuan3D 2.0、TripoSR、InstantMesh |
| 少量图像的新视角合成 | CAT3D、SV3D |
| 动态场景重建 | 4D Gaussian Splatting |
| 头像 / 穿衣人体 | Gaussian Avatar、HUGS |
| 研究 / SOTA | 上周刚发布的那个 |

如果要在游戏或电商流水线中交付生产级 3D：Meshy 4 或 Rodin Gen-1.5 会输出可以直接进入 Unity / Unreal 的 PBR 网格。

## 交付它

保存 `outputs/skill-3d-pipeline.md`。该技能接收一份 3D 简报（输入：文本 / 单图 / 少量图像；输出：网格 / splat / NeRF；用途：渲染 / 游戏 / VR），并输出：流水线（多视角扩散 + 拟合，或直接网格模型）、基础模型、迭代预算、拓扑后处理、所需材质通道。

## 练习

1. **简单。** 用 4、16、64 个高斯运行 `code/main.py`。报告最终 MSE 与目标的对比。
2. **中等。** 扩展到彩色高斯（RGB）。确认重建结果匹配目标颜色模式。
3. **困难。** 使用 gsplat 或 Nerfstudio，从 50 张照片采集重建一个真实对象。报告拟合时间和在留出视图上的最终 SSIM。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| 3D Gaussian Splatting | “3DGS” | 把场景表示为一团 3D 高斯；可微 alpha 合成渲染。 |
| NeRF | “Neural radiance field” | 在 3D 点输出颜色 + 密度的 MLP；通过射线积分渲染。 |
| Triplane | “三个 2-D 平面” | 把 3D 分解为三个轴对齐的 2D 特征网格；比体积表示更便宜。 |
| SDS | “Score distillation sampling” | 使用 2D 扩散分数作为伪梯度来训练 3D 模型。 |
| Multi-view diffusion | “一次生成多个视角” | 输出一批一致相机视角的扩散模型。 |
| PBR | “Physically-based rendering” | 包含反照率、粗糙度、金属度、法线通道的材质。 |
| Densification | “增长 splat” | 3DGS 训练启发式：在高梯度区域拆分 / 克隆 splat。 |

## 生产备注：3D 还没有共享底座

不同于图像（潜空间扩散 + DiT）和视频（时空 DiT），2026 年的 3D 还没有单一主导运行时。生产决策树会按表示方式分叉：

- **NeRF / triplane。** 推理是 ray-marching + 每个采样点一次 MLP 前向。一次 512² 渲染需要数百万次 MLP 前向。要积极批处理射线采样；SDPA/xformers 适用。
- **多视角扩散 + LRM 重建。** 两阶段流水线。阶段 1（多视角 DiT）就是类似第 07 课的扩散服务器。阶段 2（LRM Transformer）是对视图的一次性前向。整体延迟画像是“扩散 + 一次性前向”——按阶段选择相应服务原语。
- **SDS / DreamFusion。** 这是逐资产优化，不是推理。应构建作业系统，而不是请求处理器。

对大多数 2026 年产品，正确答案是“按请求运行多视角扩散模型，异步重建为 3DGS，再为实时查看提供 3DGS”。这把工作负载干净地拆成 GPU 推理服务器（快）和离线优化器（慢）。

## 延伸阅读

- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — NeRF。
- [Kerbl et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 3DGS。
- [Poole et al. (2022). DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988) — SDS。
- [Liu et al. (2023). Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328) — Zero123。
- [Shi et al. (2023). MVDream](https://arxiv.org/abs/2308.16512) — 多视角扩散。
- [Hong et al. (2023). LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) — LRM。
- [Gao et al. (2024). CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314) — CAT3D。
- [Stability AI (2024). Stable Video 3D (SV3D)](https://stability.ai/research/sv3d) — SV3D。
