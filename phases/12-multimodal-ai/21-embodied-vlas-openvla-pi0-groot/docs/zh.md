# 具身 VLA：RT-2、OpenVLA、π0、GR00T

> 首次让模型从网站读取食谱并在厨房机器人上执行的是 RT-2（Google DeepMind，2023 年 7 月）。RT-2 将动作离散化为文本 token，在网页数据与机器人动作数据上共同微调 VLM，并证明了网页规模的视觉-语言知识可以迁移到机器人控制中。OpenVLA（2024 年 6 月）发布了开源的 7B 参考实现。Physical Intelligence 的 π0 系列（2024-2025）引入了 flow-matching 动作专家。NVIDIA 的 GR00T N1（2025 年 3 月）为大规模人形机器人提供了双系统（System 1 / System 2）控制。VLA 原语——vision-language-action，即一个能看、能读、能动的单一模型——是连接本阶段理解模型与第 15 阶段自主系统的桥梁。

**类型：** 学习
**语言：** Python（标准库，动作 tokenizer + VLA 推理骨架）
**前置知识：** 第 12 阶段 · 05（LLaVA），第 15 阶段（自主系统，作为参考）
**时间：** 约 180 分钟

## 学习目标

- 描述动作 tokenization：离散分箱编码（RT-2）、FAST 高效动作 token、连续 flow-matching 动作（π0）。
- 解释为何在网页数据 + 机器人数据上共同微调能够保留对新任务的一般知识迁移。
- 在同一机器人任务上比较 OpenVLA（开源 7B Llama+VLM）、π0（flow-matching）和 GR00T N1（双系统）。
- 说出 Open X-Embodiment 数据集的名称及其作为 RT-X 训练语料库的作用。

## 问题背景

自 1970 年代以来，能够根据自然语言指令做家务的机器人一直是研究目标。2020 年代的答案是：vision-language-action（VLA）模型。与用于 VQA 的相同 VLM 架构，但输出的是动作（关节扭矩、末端执行器位姿、离散命令）而非文本。

VLA 特有的挑战：

1. 动作空间是连续的（关节角度、力）且高维的（7-DOF 机械臂 + 3-DOF 夹爪 = 10 维，30 Hz）。
2. 机器人专用训练数据稀缺。Open X-Embodiment 约有 ~100 万条轨迹；网页文本-图像数据有 50 亿+。
3. 控制频率很重要。30 Hz 控制环路意味着每个动作只有 33 毫秒的预算。
4. 安全性。错误的动作会损坏硬件、伤害人类或破坏财产。

## 核心概念

### 动作 Tokenization（RT-2）

RT-2 的诀窍：将每个关节目标表示为量化的文本 token。将归一化的 [-1, 1] 范围离散化为 256 个分箱，每个分箱映射到一个词表 ID。一个 10-DOF 动作在每个控制步变为 10 个 token。

在以下混合数据上共同微调 PaLM-X VLM：

- 网页图像-文本对（图像描述、VQA）。
- 机器人演示，动作表示为 token。

模型看到 "pick up the red cube"（语言）→ 图像（视觉）→ 10-token 动作序列（离散化的关节目标）。网页预训练保留了一般知识迁移：RT-2 能够执行 "move towards the fast-moving object"，即使 "fast-moving" 不在训练数据中。

RT-2 论文中的推理速度为 3-5 Hz，受限于 VLM 的自回归解码。

### OpenVLA —— 开源 7B 参考实现

OpenVLA（Kim 等人，2024 年 6 月）是开源权重的 RT-2 等价实现。7B Llama 主干网络，DINOv2 + SigLIP 双视觉编码器，256 分箱上的动作 tokenization。

在 Open X-Embodiment（22 个机器人共 97 万条轨迹）上训练。提供 LoRA 微调支持以适应新机器人。

推理：在 A100 上使用量化可达到 4-5 Hz。对于慢速操作足够快，但对于高频控制不够。

### FAST Tokenizer —— 更快的动作解码

Pertsch 等人（2024）表明离散分箱 tokenization 效率低下——大多数动作聚集在分箱空间的一小块区域。FAST（Frequency-domain Action Sequence Tokenizer）通过 DCT 压缩动作序列并量化系数。

一条 30 步的动作轨迹变为约 10 个 FAST token，而非 300 个离散分箱 token。推理速度提升 3-5 倍，且质量无损失。

### π0 与 Flow-Matching 动作

Physical Intelligence 的 π0（Black 等人，2024 年 10 月）用 flow-matching 动作专家替代了离散动作 token：

- 一个小型动作 transformer 读取 VLM 的隐藏状态，通过 rectified flow 输出连续的 50 步动作序列。
- 动作头使用 flow-matching loss 训练；VLM 预训练保持不变。
- 推理：完整动作序列在约 5 步去噪中生成，等效于 50 Hz 控制。

π0 声称：在广泛的操纵任务套件上击败 OpenVLA 和 Octo。连续动作形式保留了离散化会破坏的平滑性。

π0.5 和 π0-FAST 是增量升级。π0-FAST 将 FAST tokenization 与 flow matching 结合。

### GR00T N1 —— 面向人形机器人的双系统

NVIDIA 的 GR00T N1（2025 年 3 月）专为高自由度人形机器人（>30 DOF，全身）构建：

- System 2：一个大型 VLM 读取场景 + 指令，以约 1 Hz 的频率生成高级子目标。
- System 1：一个小的动作头 transformer，根据子目标生成低级的 50-100 Hz 关节命令。

这种拆分映射到 Kahneman 的快思考与慢思考：System 2 规划，System 1 执行。优势在于：慢速的 VLM 级规划不会阻塞快速控制；System 1 保持小巧以保证低延迟。

GR00T N1.7（2025 年末）改进了数据规模。GR00T 使用来自 Omniverse 的 sim-to-real 数据进行微调。

### Open X-Embodiment

训练数据。RT-X（2023 年 10 月）整合了 22 个数据集，涵盖 22 个机器人的 100 万条轨迹。Open X-Embodiment 是所有人都在使用的语料库：

- ALOHA / Bridge V2 / Droid / RT-2 Kitchen / Language Table。
- 每个样本：（机器人状态、相机视角、指令、动作序列）。
- 训练规范：统一动作空间、归一化关节范围、调整相机尺寸。

OpenVLA 和 π0 在 Open X-Embodiment 上训练。通过在新机器人上进行 100-1000 个任务特定演示的 LoRA 微调，弥合与任何特定机器人的领域差距。

### 共同微调 vs 仅机器人数据

共同微调将网页 VQA 数据与机器人轨迹混合。比例很重要：VQA 太多模型会忘记动作；机器人数据太多模型会丢失一般知识。

RT-2 的比例：约 1:1。OpenVLA：约 0.5:1 网页对机器人。π0：类似。精确比例是每个数据集大小需要调整的超参数。

仅机器人数据训练会产生任务特定模型，在分布外指令上失败。共同微调是 "pick up the red cube（在演示中）" 和 "pick up the third largest object from the left（新颖表述）" 之间的区别。

### 安全性与动作限制

每个生产级 VLA 都配备：

- 硬关节限制（不能超过规格扭矩）。
- 速度限制（软裁剪）。
- 工作空间边界（末端执行器不能离开桌面）。
- 新任务的人工在环审批。

这些位于 VLA 之外作为控制层检查。VLA 的输出是建议，而非命令。

## 使用它

`code/main.py`：

- 实现 256 分箱动作 tokenization 和反 tokenization。
- 勾勒基于 DCT + 量化的 FAST tokenizer。
- 比较每种动作步的 token 数量（离散分箱、FAST、连续 flow）。
- 打印 RT-2 → OpenVLA → π0 → GR00T 的演进谱系摘要。

## 交付它

本节课产出 `outputs/skill-vla-action-format-picker.md`。给定一个机器人任务（操纵、导航、人形全身），在离散分箱 + RT-2、FAST + OpenVLA、flow-matching + π0 或双系统 + GR00T 之间做出选择。

## 练习题

1. 一个 10-DOF 机械臂，控制频率 30 Hz。256 分箱的离散分箱 tokenization 每秒输出多少 token？7B VLM 能跟得上吗？

2. FAST tokenization 将 30 步轨迹压缩到约 10 个 token。如果轨迹包含高频运动（例如击鼓），用户会损失什么？

3. π0 的 flow-matching 头在约 5 步内去噪。与 OpenVLA 4-5 Hz 的自回归解码相比，吞吐量如何？

4. GR00T 的 System 1 / System 2 拆分映射到 Kahneman。提出一个不同的拆分（System 3？）可能有助于双足行走。

5. 阅读 Open X-Embodiment 第 4 节关于数据集整理的内容。说出防止领域泄漏的三条整理规则。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|------------|----------|
| VLA | "Vision-language-action" | 接收图像 + 指令并输出动作命令的模型 |
| Action tokenization | "Discrete bins" | 将连续关节目标量化为每维 256 个分箱，每个分箱是一个词表 ID |
| FAST tokenizer | "Frequency action tokens" | DCT + 量化，将 30 步轨迹压缩到约 10 个 token |
| Co-fine-tune | "Mix web + robot" | 在网页 VQA 数据与机器人演示上共同训练，以保留一般知识 |
| Flow-matching action head | "π0 continuous output" | 小型 transformer，通过 rectified flow 输出 50 步动作序列 |
| System 1 / System 2 | "Dual-system control" | 大型 VLM 慢速规划，小型动作头快速执行；GR00T 模式 |
| Open X-Embodiment | "RT-X dataset" | 100 万条轨迹的跨机器人数据集；训练语料库 |

## 延伸阅读

- [Brohan et al. — RT-2 (arXiv:2307.15818)](https://arxiv.org/abs/2307.15818)
- [Kim et al. — OpenVLA (arXiv:2406.09246)](https://arxiv.org/abs/2406.09246)
- [Black et al. — π0 (arXiv:2410.24164)](https://arxiv.org/abs/2410.24164)
- [NVIDIA — GR00T N1 (arXiv:2503.14734)](https://arxiv.org/abs/2503.14734)
- [Open X-Embodiment Collab — RT-X (arXiv:2310.08864)](https://arxiv.org/abs/2310.08864)
