# 仿真到现实迁移

> 在仿真器中训练出的策略，如果一到硬件上就失败，那它学到的只是对仿真器的记忆。域随机化、域适应和系统辨识，是让学习型控制器跨越现实差距的三件工具。

**类型：** 学习
**语言：** Python
**先修要求：** 第 9 阶段 · 08（PPO），第 2 阶段 · 10（偏差/方差）
**时间：** 约 45 分钟

## 问题

训练真实机器人既慢、又危险、还昂贵。一个双足机器人需要数百万个训练 episode 才能学会行走；真实双足机器人哪怕只摔倒一次，也可能损坏硬件。仿真给了你无限重置、确定性可复现性、并行环境，以及不会造成物理损坏的训练场。

但仿真器是错的。轴承的摩擦比 MuJoCo 模型更大。相机有镜头畸变，而仿真器里并没有包含。电机有延迟、回程间隙（backlash）和饱和，而 99% 的仿真模型都会忽略这些。风、灰尘和可变光照会破坏在洁净渲染环境中训练出的策略。**现实差距（reality gap）**——仿真分布与真实分布之间的系统性差异——是机器人强化学习部署的核心问题。

你需要一个对*仿真到现实分布偏移（sim-to-real distribution shift）*具有鲁棒性的策略。历史上有三类方法：随机化仿真器（域随机化，domain randomization），用少量真实数据适应策略（域适应/微调，domain adaptation / fine-tuning），或者辨识真实系统参数并让仿真与之匹配（系统辨识，system identification）。到 2026 年，主流配方会把三者与大规模并行仿真（Isaac Sim、Isaac Lab、GPU 上的 Mujoco MJX）结合起来。

## 概念

![三种仿真到现实机制：域随机化、适应、系统辨识](../assets/sim-to-real.svg)

**域随机化（Domain Randomization, DR）。** Tobin 等人 2017，Peng 等人 2018。在训练期间，随机化所有可能与真实机器人不同的仿真参数：质量、摩擦系数、电机 PD 增益、传感器噪声、相机位置、光照、纹理、接触模型。策略会学习“今天处在哪个仿真器中”的条件分布，并在整个范围内泛化。如果真实机器人落在训练包络之内，策略就能工作。

- **优点：** 不需要真实数据。一套配方，可以用于许多机器人。
- **缺点：** 过度随机化的训练会产生一个“通用”但过于保守的策略。噪声太多 ≈ 正则化太强。

**系统辨识（System Identification, SI）。** 在训练前，用真实世界数据拟合仿真器参数。如果你能测量真实机器人手臂关节的摩擦，就把这个值填入仿真器。然后训练一个期望这些参数值的策略。它需要访问真实系统，但能直接缩小现实差距。

- **优点：** 精确、低噪声的训练目标。
- **缺点：** 残余模型误差对策略是不可见的；很小的未辨识效应（例如电机死区）仍然会破坏部署。

**域适应（Domain Adaptation）。** 先在仿真中训练，再用少量真实数据微调。两种形式：

- **Real2Sim2Real：** 使用真实 rollout 学习残差仿真器 `f(s, a, z) - f_sim(s, a)`，再在修正后的仿真中训练。这样不用太多真实数据就能缩小差距。
- **观测适应：** 训练一个策略，通过学习到的特征提取器（例如 GAN 像素到像素映射）把真实观测 → 类仿真观测。控制器仍然保持在仿真域中。

**特权学习 / 教师-学生（Privileged learning / teacher-student）。** Miki 等人 2022（ANYmal 四足机器人）。在仿真中训练一个可以访问特权信息（真实摩擦、地形高度、IMU 漂移）的*教师*。再蒸馏出一个只看到真实传感器观测的*学生*。学生会学习从历史中推断特权特征，从而在不同物理参数下保持鲁棒。

**大规模并行仿真。** 2024–2026。Isaac Lab、Mujoco MJX、Brax 都可以在单块 GPU 上运行数千个并行机器人。PPO 配合 4,096 个并行人形机器人，可以在数小时内收集相当于数年的经验。随着训练分布变宽，“现实差距”会缩小；当这 4,096 个环境中的每一个都有不同的随机化参数时，DR 几乎是免费的。

**2026 年真实世界配方（四足行走示例）：**

1. 使用大规模并行仿真，并随机化重力、摩擦、电机增益、载荷。
2. 使用特权信息（地形图、机体速度真值）训练教师策略。
3. 只使用本体感知（腿部关节编码器），从教师蒸馏学生策略。
4. 可选：通过真实 IMU 上的自编码器进行观测适应。
5. 部署。在 10+ 个环境中零样本运行。如果失败，用带安全约束的 PPO 做几分钟真实世界微调。

## 构建它

本课代码是在带有*噪声*转移的 GridWorld 上演示域随机化的微型示例。我们训练一个策略，让它在“仿真”中经历随机化的滑移概率，并在“真实”环境中用训练期间从未见过的滑移水平评估。这个结构可以直接映射到从 MuJoCo 到硬件的迁移。

### 步骤 1：参数化仿真

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip` 是仿真器暴露的一个参数。在真实机器人中，它可以是摩擦、质量、电机增益——任何会在仿真与现实之间发生偏移的东西。

### 步骤 2：使用 DR 训练

在每个 episode 开始时，采样 `slip ~ Uniform[0.0, 0.4]`。训练 PPO / Q-learning / 任何算法。重复很多个 episode。

### 步骤 3：在“真实”滑移上零样本评估

在 `slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}` 上评估。前四个在训练支持集内；`0.5` 和 `0.7` 在支持集外。DR 训练出的策略应该在支持集内接近最优，并在支持集外优雅退化。固定滑移训练的策略在训练滑移之外会很脆弱。

### 步骤 4：与窄分布训练比较

训练第二个策略，只使用 `slip = 0.0`。在同样的 `slip` 扫描上评估。你应该会看到：只要真实滑移 > 0，性能就会灾难性下降。

## 常见陷阱

- **随机化过多。** 在 `slip ∈ [0, 0.9]` 上训练，你的策略会变得过于规避风险，以至于永远不尝试最优路径。要匹配*预期的*真实世界分布，而不是“任何事情都可能发生”。
- **随机化过少。** 在很窄的切片上训练，策略完全无法泛化。使用自适应课程（自动域随机化，Automatic Domain Randomization），随着策略改进逐渐拓宽分布。
- **参数空间辨识错误。** 随机化了错误的东西（真实差距是电机延迟，你却随机化相机色相），DR 不会有帮助。先对真实机器人做剖析（profile）。
- **特权信息泄漏。** 如果教师使用全局状态来选择动作，而不仅仅是观测，就可能产生一个学生永远追不上的目标。要确保教师策略在给定观测历史下对学生来说是可实现的。
- **仿真到仿真迁移失败。** 如果你的策略对更难的仿真变体都不鲁棒，它也不会对真实世界鲁棒。部署前一定要在保留的仿真变体上测试。
- **没有真实世界安全包络。** 一个在仿真中有效、并且“在现实中也有效”的策略，如果没有低层安全护盾，仍然可能损坏硬件。要在非学习控制器中加入速率限制、扭矩限制、关节限制。

## 使用它

2026 年的仿真到现实技术栈：

| 领域 | 技术栈 |
|--------|-------|
| 腿式运动（ANYmal、Spot、人形机器人） | Isaac Lab + DR + 特权教师 / 学生 |
| 操作（灵巧手、抓取与放置） | Isaac Lab + DR + 用于视觉的 DR-GAN |
| 自动驾驶 | CARLA / NVIDIA DRIVE Sim + DR + 真实微调 |
| 无人机竞速 | RotorS / Flightmare + DR + 在线适应 |
| 手指/手内操作 | OpenAI Dactyl（空前规模的 DR） |
| 工业机械臂 | MuJoCo-Warp + SI + 少量真实微调 |

对于所有尺度的控制，工作流都是一致的：尽可能拟合仿真；对无法拟合的部分做随机化；训练巨大策略；蒸馏；带安全护盾部署。

## 交付它

保存为 `outputs/skill-sim2real-planner.md`：

```markdown
---
name: sim2real-planner
description: Plan a sim-to-real transfer pipeline for a given robot + task, covering DR, SI, and safety.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

Given a robot platform, a task, and access to real hardware time, output:

1. Reality gap inventory. Suspected sources ranked by expected impact (contact, sensing, actuation delay, vision).
2. DR parameters. Exact list, ranges, distribution. Justify each range against real measurements.
3. SI steps. Which parameters to measure; measurement method.
4. Teacher/student split. What privileged info the teacher uses; what obs the student uses.
5. Safety envelope. Low-level limits, emergency stops, backup controller.

Refuse to deploy without (a) a zero-shot sim-variant test, (b) a safety shield, (c) a rollback plan. Flag any DR range wider than 3× measured real variability as likely over-randomized.
```

## 练习

1. **简单。** 在固定滑移 GridWorld（slip=0.0）上训练一个 Q-learning 智能体。在 slip ∈ {0.0, 0.1, 0.3, 0.5} 上评估。绘制 return vs slip。
2. **中等。** 训练一个 DR Q-learning 智能体，采样 `slip ~ Uniform[0, 0.3]`。评估同样的扫描。在 slip=0.5（分布外）时，DR 带来了多大收益？
3. **困难。** 实现一个课程：从 slip=0.0 开始，每当策略达到最优值的 90% 时，就拓宽 DR 范围。测量达到 slip=0.3 零样本性能所需的总环境步数，并与固定 DR 基线比较。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| 现实差距（Reality gap） | “仿真到现实的差异” | 训练与部署之间在物理/感知上的分布偏移。 |
| 域随机化（Domain randomization, DR） | “跨随机仿真训练” | 在训练期间随机化仿真参数，使策略能够泛化。 |
| 系统辨识（System identification, SI） | “测量真实系统并拟合仿真” | 估计真实物理参数；设置仿真使其匹配。 |
| 域适应（Domain adaptation） | “在真实数据上微调” | 仿真训练后的少量真实世界微调；可以适应观测或动力学。 |
| 特权信息（Privileged info） | “给教师的真值” | 只有仿真才拥有的信息；学生必须从观测历史中推断它。 |
| 教师/学生（Teacher/student） | “将特权信息蒸馏为可观测能力” | 教师用捷径训练；学生学习在没有捷径的情况下模仿。 |
| ADR | “自动域随机化” | 随着策略改进而拓宽 DR 范围的课程。 |
| Real2Sim | “用真实数据缩小差距” | 学习一个残差，使仿真模仿真实 rollout。 |

## 延伸阅读

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — 最初的 DR 论文（机器人视觉）。
- [Peng et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) — 面向动力学的 DR，四足运动。
- [OpenAI et al. (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) — Dactyl，大规模 ADR。
- [Miki et al. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822) — ANYmal 的教师-学生方法。
- [Makoviychuk et al. (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470) — 驱动 2025–2026 部署的大规模并行仿真。
- [Akkaya et al. (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113) — ADR 课程方法。
- [Sutton & Barto (2018). Ch. 8 — Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 支撑现代仿真到现实流水线的 Dyna 框架（使用模型进行规划 + rollout）。
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303) — 仿真到现实方法分类与基准结果。
