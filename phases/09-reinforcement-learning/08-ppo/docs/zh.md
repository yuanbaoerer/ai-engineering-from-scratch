# 近端策略优化（PPO）

> A2C 在一次更新后就丢弃每次 rollout。PPO 用截断的重要性比率（clipped importance ratio）包裹策略梯度，让你可以在同一批数据上训练 10+ 个 epoch，而不会让策略爆炸。Schulman 等人（2017）。到 2026 年，它仍然是默认的策略梯度算法。

**类型：** 构建
**语言：** Python
**先修：** Phase 9 · 06（REINFORCE），Phase 9 · 07（Actor-Critic）
**时间：** 约 75 分钟

## 问题

A2C（第 07 课）是同策略（on-policy）的：梯度 `E_{π_θ}[A · ∇ log π_θ]` 要求数据采样自*当前*的 `π_θ`。执行一次更新后，`π_θ` 就变了；你刚用过的数据现在成了异策略（off-policy）数据。再次使用它会让梯度有偏。

Rollout 很昂贵。在 Atari 上，跨 8 个环境 × 128 步的一次 rollout = 1024 个转移，还要十几秒的环境时间。只做一次梯度步骤就把它丢掉很浪费。

信赖域策略优化（Trust Region Policy Optimization，TRPO，Schulman 2015）是第一个修复方案：约束每次更新，使旧策略与新策略之间的 KL 散度保持在 `δ` 以下。理论上很干净，但每次更新都需要一次共轭梯度求解。到 2026 年，已经没人运行 TRPO 了。

PPO（Schulman 等人，2017）用一个简单的截断目标替代了硬信赖域约束。只多一行代码。每次 rollout 做十个 epoch。没有共轭梯度。理论保证足够好。九年后，它仍然是从 MuJoCo 到 RLHF 的默认策略梯度算法。

## 概念

![PPO clipped surrogate objective: ratio clipping at 1 ± ε](../assets/ppo.svg)

**重要性比率。**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

这是新策略相对于收集数据的策略的似然比。`r_t = 1` 表示没有变化。`r_t = 2` 表示新策略选择 `a_t` 的可能性是旧策略的两倍。

**截断代理目标。**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

两个项：

- 如果优势 `A_t > 0` 且比率试图增长到超过 `1 + ε`，截断会让梯度变平——不要把一个好动作推到比旧概率高出 `+ε` 以上。
- 如果优势 `A_t < 0` 且比率试图增长到超过 `1 - ε`（意思是相对于截断后的降低，我们会让一个坏动作更可能发生），截断会给梯度加上上限——不要把一个坏动作推到低于 `-ε`。

`min` 处理另一个方向：如果比率已经朝*有利*方向移动，你仍然会得到梯度（不会在会伤害你的那一侧截断）。

典型的 `ε = 0.2`。把目标画成 `r_t` 的函数：它是一个分段线性函数，在“好的一侧”有平坦的屋顶，在“坏的一侧”有平坦的地板。

**完整 PPO 损失。**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

与 A2C 相同的 actor-critic 结构。三个系数，通常是 `c_v = 0.5`、`c_e = 0.01`、`ε = 0.2`。

**训练循环。**

1. 跨 `N` 个并行环境、每个环境 `T` 步，收集 `N × T` 个转移。
2. 计算优势（GAE），并把它们冻结为常量。
3. 将 `π_{θ_old}` 冻结为当前 `π_θ` 的一个快照。
4. 对 `K` 个 epoch，对每个 `(s, a, A, V_target, log π_old(a|s))` 的小批量：
   - 计算 `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`。
   - 应用 `L^{CLIP}` + 价值损失 + 熵。
   - 梯度步骤。
5. 丢弃这次 rollout。返回步骤 1。

`K = 10` 且小批量大小为 64 是一组标准超参数。PPO 很鲁棒：具体数字在 ±50% 范围内通常影响不大。

**KL 惩罚变体。** 原论文提出了一个替代版本，使用自适应 KL 惩罚：`L = L^{PG} - β · KL(π_θ || π_old)`，其中 `β` 会根据观测到的 KL 调整。截断版本后来成为主流；KL 变体仍存在于 RLHF 中（在那里，相对于参考策略的 KL 本来就是你始终想要的一个单独约束）。

## 构建它

### 第 1 步：在 rollout 时捕获 `log π_old(a | s)`

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

快照只在 rollout 时获取一次。它在更新 epoch 期间不会改变。

### 第 2 步：计算 GAE 优势（第 07 课）

与 A2C 相同。在整个批量上归一化。

### 第 3 步：截断代理更新

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

“截断 → 零梯度”模式是 PPO 的核心。如果新策略已经在有利方向漂移得太远，更新就会停止。

### 第 4 步：价值与熵

给 critic 目标添加标准 MSE，并给 actor 添加熵奖励，与 A2C 相同。

### 第 5 步：诊断

每次更新都要观察三件事：

- **平均 KL** `E[log π_old - log π_θ]`。应保持在 `[0, 0.02]`。如果它冲过 `0.1`，降低 `K_EPOCHS` 或 `LR`。
- **截断比例（clip fraction）** ——比率落在 `[1-ε, 1+ε]` 之外的样本比例。应为 `~0.1-0.3`。如果是 `~0`，说明截断从未触发 → 提高 `LR` 或 `K_EPOCHS`。如果是 `~0.5+`，说明你正在过拟合这次 rollout → 降低它们。
- **解释方差（explained variance）** `1 - Var(V_target - V_pred) / Var(V_target)`。Critic 质量指标。随着 critic 学习，它应朝 1 上升。

## 陷阱

- **截断系数调错。** `ε = 0.2` 是事实标准。调到 `0.1` 会让更新过于胆小；`0.3+` 会引入不稳定。
- **epoch 太多。** `K > 20` 经常会使训练不稳定，因为策略会漂移到离 `π_old` 很远。限制 epoch，尤其是对大网络。
- **没有奖励归一化。** 大的奖励尺度会侵蚀截断范围。在计算优势前对奖励归一化（运行标准差）。
- **忘记优势归一化。** 每批量零均值/单位标准差归一化是标准做法。跳过它会让 PPO 在多数 benchmark 上崩掉。
- **学习率没有衰减。** PPO 受益于把 LR 线性衰减到零。恒定 LR 往往更差。
- **重要性比率数学错误。** 为了数值稳定，始终使用 `exp(log_new - log_old)`，不要用 `new / old`。
- **梯度符号错误。** 最大化代理目标 = *最小化* `-L^{CLIP}`。符号翻转是最常见的 PPO bug。

## 使用它

PPO 是 2026 年默认的 RL 算法，覆盖的领域多得出人意料：

| 用例 | PPO 变体 |
|----------|-------------|
| MuJoCo / 机器人控制 | 带高斯策略的 PPO，GAE(0.95) |
| Atari / 离散游戏 | 带分类策略的 PPO，滚动 128 步 rollout |
| LLM 的 RLHF | PPO，带相对于参考模型的 KL 惩罚，奖励来自响应末尾的 RM |
| 大规模游戏智能体 | IMPALA + PPO（AlphaStar、OpenAI Five） |
| 推理 LLM | GRPO（第 12 课）——没有 critic 的 PPO 变体 |
| 仅偏好数据 | DPO——PPO+KL 的闭式折叠，无在线采样 |

PPO 的*损失形状*——截断代理 + 价值 + 熵——是 DPO、GRPO 以及几乎所有 RLHF pipeline 的脚手架。

## 交付它

保存为 `outputs/skill-ppo-trainer.md`：

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## 练习

1. **简单。** 在 4×4 GridWorld 上运行 PPO，设置 `ε=0.2, K=4`。在匹配环境步数的情况下，将样本效率与 A2C（每次 rollout 一个 epoch）比较。
2. **中等。** 扫描 `K ∈ {1, 4, 10, 30}`。绘制回报 vs 环境步数，并跟踪每次更新的平均 KL。在这个任务上，KL 从哪个 `K` 开始爆炸？
3. **困难。** 用自适应 KL 惩罚替换截断代理目标（如果 `KL > 2·target` 则 `β` 翻倍，如果 `KL < target/2` 则 `β` 减半）。比较最终回报、稳定性以及无截断程度。

## 关键术语

| 术语 | 人们怎么说 | 它实际表示什么 |
|------|-----------------|-----------------------|
| 重要性比率 | "r_t(θ)" | `π_θ(a\|s) / π_old(a\|s)`；与收集数据的策略之间的偏离。 |
| 截断代理目标 | "PPO's main trick" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`；在有利侧超过截断后梯度变平。 |
| 信赖域 | "TRPO / PPO intent" | 限制每次更新的 KL，以保证单调改进。 |
| KL 惩罚 | "Soft trust region" | 替代版 PPO：`L - β · KL(π_θ \|\| π_old)`。自适应 `β`。 |
| 截断比例 | "How often clipping triggers" | 诊断指标——应为 0.1-0.3；超出说明调参不当。 |
| 多 epoch 训练 | "Data reuse" | 对每次 rollout 做 K 个 epoch；用方差成本换取样本效率。 |
| 近似同策略 | "Mostly on-policy" | PPO 名义上是同策略，但 K>1 个 epoch 会安全地使用轻微异策略的数据。 |
| PPO-KL | "The other PPO" | KL 惩罚变体；用于 RLHF，其中相对于参考模型的 KL 本来就是一个约束。 |

## 延伸阅读

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) —— 原论文。
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) —— TRPO，PPO 的前身。
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) —— 对每个 PPO 超参数做消融。
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) —— InstructGPT；RLHF 中使用 PPO 的配方。
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) —— 带 PyTorch 的清晰现代表述。
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) —— 许多论文使用的单文件 PPO 参考实现。
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) —— 在语言模型上运行 PPO 的生产配方；请与第 09 课（RLHF）一起阅读。
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) —— “37 个代码级优化”论文；哪些 PPO 技巧是关键承重结构，哪些只是经验传说。
