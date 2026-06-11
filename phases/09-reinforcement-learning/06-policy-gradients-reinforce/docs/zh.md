# 策略梯度 — 从零实现 REINFORCE

> 停止估计价值。直接参数化策略，计算期望回报的梯度，然后沿上坡方向更新。Williams（1992）用一个定理写出了它。这正是 PPO、GRPO 以及每一个 LLM 强化学习循环存在的原因。

**类型：** 构建
**语言：** Python
**先修：** Phase 3 · 03（反向传播），Phase 9 · 03（蒙特卡洛），Phase 9 · 04（时序差分学习）
**时间：** 约 75 分钟

## 问题

Q-learning 和 DQN 参数化的是*价值*函数。你通过 `argmax Q` 来选择动作。这对于离散动作和离散状态来说没问题。但当动作是连续的（对一个 10 维力矩向量做哪种 `argmax`？），或者当你想要一个随机策略（`argmax` 按构造就是确定性的）时，它就会失效。

策略梯度（policy gradients）改为参数化*策略*本身。`π_θ(a | s)` 是一个神经网络，输出动作上的分布。通过从该分布采样来行动。计算期望回报关于 `θ` 的梯度。沿上坡方向更新。没有 `argmax`。没有贝尔曼递归。只有对 `J(θ) = E_{π_θ}[G]` 的梯度上升。

REINFORCE 定理（Williams 1992）告诉你这个梯度是可计算的：`∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`。运行一个 episode。计算回报。把它与每一步的 `∇ log π_θ(a | s)` 相乘。取平均。做梯度上升。完成。

2026 年的每一种 LLM-RL 算法——PPO、DPO、GRPO——都是 REINFORCE 的改进。把它练到手上有感觉，是学习本阶段后续内容，以及 Phase 10 · 07（RLHF 实现）和 Phase 10 · 08（DPO）的前提。

## 概念

![Policy gradient: softmax policy, log-π gradient, return-weighted update](../assets/policy-gradient.svg)

**策略梯度定理（policy gradient theorem）。** 对于任何由 `θ` 参数化的策略 `π_θ`：

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

其中 `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}` 是从步骤 `t` 开始的折扣回报。期望是在从 `π_θ` 采样得到的完整轨迹 `τ` 上计算的。

**证明很短。** 在期望下对 `J(θ) = Σ_τ P(τ; θ) G(τ)` 求导。使用 `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)`（对数导数技巧，log-derivative trick）。分解 `log P(τ; θ) = Σ log π_θ(a_t | s_t) + 不依赖 θ 的环境项`。环境项会消失。两行代数就得到定理。

**方差降低技巧。** 原始 REINFORCE 的方差大得要命——回报是噪声，`∇ log π` 是噪声，它们的乘积非常噪声。两个标准修复方法：

1. **基线相减（baseline subtraction）。** 对任何不依赖 `a_t` 的基线 `b(s_t)`，用 `G_t - b(s_t)` 替换 `G_t`。这是无偏的，因为 `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`。典型选择：由评论家（critic）学习的 `b(s_t) = V̂(s_t)` → actor-critic（Lesson 07）。
2. **从当前步开始的回报（reward-to-go）。** 用 `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)` 替换 `Σ_t G_t · ∇ log π_θ(a_t | s_t)`。对于给定动作，只有未来回报重要——过去奖励只会贡献零均值噪声。

合并之后得到：

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

这就是带基线的 REINFORCE——A2C（Lesson 07）和 PPO（Lesson 08）的直接祖先。

**Softmax 策略参数化。** 对于离散动作，标准选择是：

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

其中 `f_θ` 是任意神经网络，为每个动作输出一个分数。该梯度有一个简洁形式：

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

也就是被采取动作的分数减去它在策略下的期望值。

**连续动作的高斯策略（Gaussian policy）。** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`。`∇ log N(a; μ, σ)` 有闭式形式。这就是 Phase 9 · 07 的 SAC 所需要的全部。

## 构建它

### 步骤 1：softmax 策略网络

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

对于表格型环境，使用线性策略（每个动作一个权重向量）。对于 Atari，换成 CNN，并保留 softmax 头。

### 步骤 2：采样和对数概率

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### 步骤 3：rollout，并捕获 log-probs

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### 步骤 4：REINFORCE 更新

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

梯度 `∇ log π(a|s) = e_a - π(·|s)`（`a` 的 one-hot 减去概率）是 softmax 策略梯度的核心。把它刻进肌肉记忆。

### 步骤 5：基线

最近若干 episode 中 `G` 的运行均值，就足以让 4×4 GridWorld 跑起来；它大约需要 500 个 episode 收敛。把基线升级为学习得到的 `V̂(s)`，你就得到 actor-critic。

## 常见陷阱

- **梯度爆炸。** 回报可能非常大。在与 `∇ log π` 相乘之前，总是要在 batch 内把 `G` 归一化到 `~N(0, 1)`。
- **熵坍缩（entropy collapse）。** 策略过早收敛到近乎确定性的动作，停止探索，然后卡住。修复方法：向目标函数加入熵奖励 `β · H(π(·|s))`。
- **高方差。** 原始 REINFORCE 需要数千个 episode。评论家基线（Lesson 07）或 TRPO/PPO 的信赖域（trust region，Lesson 08）是标准修复方法。
- **样本效率低。** On-policy 意味着每条转移在一次更新后就要丢弃。通过重要性采样（importance sampling）进行 off-policy 修正可以把数据带回来，但代价是增加方差（PPO 的 ratio 就是一个被裁剪的 IS 权重）。
- **非平稳梯度。** 100 个 episode 之前的同一个梯度使用的是旧的 `π`。因此 on-policy 方法每隔少量 rollout 就会更新一次。
- **信用分配（credit assignment）。** 如果没有 reward-to-go，过去奖励会贡献噪声。始终使用 reward-to-go。

## 使用它

在 2026 年，REINFORCE 很少被直接运行，但它的梯度公式无处不在：

| 使用场景 | 派生方法 |
|----------|---------------|
| 连续控制 | 使用高斯策略的 PPO / SAC |
| LLM RLHF | 带 KL 惩罚、运行在 token 级策略上的 PPO |
| LLM 推理（DeepSeek） | GRPO——带组相对基线、没有 critic 的 REINFORCE |
| 多智能体 | 集中式 critic 的 REINFORCE（MADDPG、COMA） |
| 离散动作机器人 | A2C、A3C、PPO |
| 仅偏好设置 | DPO——把 REINFORCE 重写为偏好似然损失，无需采样 |

当你在 2026 年的训练脚本中读到 `loss = -advantage * log_prob` 时，那就是带基线的 REINFORCE。整篇论文（DPO、GRPO、RLOO）都可以看作建立在这一行之上的方差降低技巧。

## 交付它

保存为 `outputs/skill-policy-gradient-trainer.md`：

```markdown
---
name: policy-gradient-trainer
description: Produce a REINFORCE / actor-critic / PPO training config for a given task and diagnose variance issues.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

Given an environment (discrete / continuous actions, horizon, reward stats), output:

1. Policy head. Softmax (discrete) or Gaussian (continuous) with parameter counts.
2. Baseline. None (vanilla), running mean, learned `V̂(s)`, or A2C critic.
3. Variance controls. Reward-to-go on by default, return normalization, gradient clip value.
4. Entropy bonus. Coefficient β and decay schedule.
5. Batch size. Episodes per update; on-policy data freshness contract.

Refuse REINFORCE-no-baseline on horizons > 500 steps. Refuse continuous-action control with a softmax head. Flag any run with `β = 0` and observed policy entropy < 0.1 as entropy-collapsed.
```

## 练习

1. **简单。** 在 4×4 GridWorld 上实现 REINFORCE，使用线性 softmax 策略。不使用基线，训练 1,000 个 episode。绘制学习曲线；测量方差（回报的标准差）。
2. **中等。** 加入运行均值基线。重新训练。将样本效率和方差与原始运行进行比较。基线把达到收敛所需的步数减少了多少？
3. **困难。** 加入熵奖励 `β · H(π)`。扫描 `β ∈ {0, 0.01, 0.1, 1.0}`。绘制最终回报和策略熵。在这个任务上甜点区在哪里？

## 关键术语

| 术语 | 人们通常怎么说 | 它实际是什么意思 |
|------|-----------------|-----------------------|
| 策略梯度（Policy gradient） | “直接训练策略” | `∇J(θ) = E[G · ∇ log π_θ(a\|s)]`；由对数导数技巧推导而来。 |
| REINFORCE | “最早的 PG 算法” | Williams（1992）；蒙特卡洛回报乘以对数策略梯度。 |
| 对数导数技巧（Log-derivative trick） | “得分函数估计器” | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`；让期望的梯度变得可处理。 |
| 基线（Baseline） | “方差降低” | 从 `G` 中减去的任意 `b(s)`；无偏，因为 `E[b · ∇ log π] = 0`。 |
| 从当前步开始的回报（Reward-to-go） | “只有未来回报算数” | 使用 `G_t^{from t}` 而不是完整的 `G_0`；正确且方差更低。 |
| 熵奖励（Entropy bonus） | “鼓励探索” | `+β · H(π(·\|s))` 项让策略避免坍缩。 |
| On-policy | “用你刚刚看到的数据训练” | 梯度期望是相对于当前策略的——不能直接复用旧数据。 |
| 优势（Advantage） | “比平均好多少” | `A(s, a) = G(s, a) - V(s)`；带基线 REINFORCE 所乘的有符号量。 |

## 延伸阅读

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) —— 原始 REINFORCE 论文。
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) —— 带函数逼近的现代策略梯度定理。
- [Sutton & Barto (2018). Ch. 13 — Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf) —— 教科书式讲解。
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) —— 带 PyTorch 代码的清晰教学说明。
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) —— 方差降低和自然梯度视角，将 REINFORCE 连接到信赖域家族（TRPO、PPO）。
