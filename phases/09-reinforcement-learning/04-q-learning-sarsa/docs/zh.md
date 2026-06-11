# 时序差分 — Q-Learning 与 SARSA

> 蒙特卡洛方法会等到回合结束。TD 通过自举（bootstrapping）下一个价值估计，在每一步之后就更新。Q-learning 是离策略（off-policy）且乐观的；SARSA 是同策略（on-policy）且谨慎的。两者都只是一行代码。两者也支撑着本阶段的每一种深度强化学习方法。

**类型：** 构建
**语言：** Python
**先修要求：** 第 9 阶段 · 01（MDP），第 9 阶段 · 02（动态规划），第 9 阶段 · 03（蒙特卡洛）
**时间：** 约 75 分钟

## 问题

蒙特卡洛方法可行，但它有两个昂贵的要求。它需要会终止的回合，并且只有在最终回报到来之后才更新。如果你的回合有 1,000 步，MC 就会等待 1,000 步才更新任何东西。它高方差、低偏差，在实践中速度很慢。

动态规划则呈现相反的特征——零方差的自举备份——但需要已知模型。

时序差分（temporal difference, TD）学习在两者之间折中。根据单个转移 `(s, a, r, s')`，构造一步目标 `r + γ V(s')`，并将 `V(s)` 朝它轻推。不需要模型。不需要完整回合。由于在等式右侧使用近似的 `V` 会引入偏差，但相比 MC 方差显著更低，而且从第一步起就能在线更新。

这是所有现代 RL——DQN、A2C、PPO、SAC——所围绕的支点。第 9 阶段剩下的内容，都是在本课你将编写的一步 TD 更新之上叠加函数近似和技巧。

## 概念

![Q-learning vs SARSA: off-policy max vs on-policy Q(s', a')](../assets/td.svg)

**用于 V 的 TD(0) 更新：**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

括号中的量是 TD 误差 `δ = r + γ V(s') - V(s)`。它是 MC 中 `G_t - V(s_t)` 的在线对应物。收敛要求 `α` 满足 Robbins-Monro 条件（`Σ α = ∞`，`Σ α² < ∞`），并且所有状态被无限次访问。

**Q-learning。** 一种用于控制的离策略 TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

`max` 假设从 `s'` 开始会遵循*贪婪*策略，无论智能体实际采取什么动作。这种解耦使 Q-learning 能在智能体通过 ε-greedy 探索时学习 `Q*`。Mnih 等人（2015）将其转化为 Atari 上的深度 Q-learning（第 05 课）。

**SARSA。** 一种同策略 TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

这个名字来自五元组 `(s, a, r, s', a')`。SARSA 使用智能体接下来*实际*采取的动作 `a'`，而不是贪婪的 `argmax`。它会收敛到当前正在运行的任意 ε-greedy `π` 的 `Q^π`，当极限中 `ε → 0` 时则成为 `Q*`。

**悬崖行走的差异。** 在经典的悬崖行走任务中（掉下悬崖 = 奖励 -100），Q-learning 会学到沿悬崖边缘前进的最优路径，但在探索期间偶尔会受到惩罚。SARSA 会学到一条离悬崖一步远的更安全路径，因为它把探索噪声也计入了自己的 Q 值。随着训练进行，当 `ε → 0` 时两者都会达到最优。在实践中这很重要：当部署时确实仍在发生探索，SARSA 的行为会更加保守。

**Expected SARSA。** 将 `Q(s', a')` 替换为其在 `π` 下的期望值：

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

比 SARSA 方差更低（不采样 `a'`），但同样是同策略目标。它通常是现代教材中的默认选择。

**n 步 TD 与 TD(λ)。** 通过在自举前等待 `n` 步，在 TD(0) 和 MC 之间插值。`n=1` 是 TD，`n=∞` 是 MC。TD(λ) 用几何权重 `(1-λ)λ^{n-1}` 对所有 `n` 进行平均。大多数深度 RL 使用 3 到 20 之间的 `n`。

## 构建它

### 步骤 1：基于 ε-greedy 策略的 SARSA

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

八行。与 Q-learning 的*唯一*区别就是目标那一行。

### 步骤 2：Q-learning

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max` 将目标与行为解耦。这个单个符号就是同策略和离策略之间的差异。

### 步骤 3：学习曲线

跟踪每 100 个回合的平均回报。Q-learning 在简单的确定性 GridWorld 上收敛更快；SARSA 在悬崖行走上更保守。在 `code/main.py` 中的 4×4 GridWorld 上，使用 `α=0.1, ε=0.1` 时，两者在约 2,000 个回合后都接近最优。

### 步骤 4：与 DP 真值比较

运行价值迭代（第 02 课）得到 `Q*`。检查 `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`。一个健康的表格型 TD 智能体在 4×4 GridWorld 上训练 10,000 个回合后，应落在 `~0.5` 以内。

## 常见陷阱

- **初始 Q 值很重要。** 乐观初始化（在负奖励任务中令 `Q = 0`）会鼓励探索。悲观初始化可能把贪婪策略永远困住。
- **α 调度。** 常数 `α` 对非平稳问题没问题。衰减的 `α_n = 1/n` 在理论上能收敛，但实践中太慢——把 `α` 固定在 `[0.05, 0.3]`，并监控学习曲线。
- **ε 调度。** 从较高值开始（`ε=1.0`），衰减到 `ε=0.05`。"GLIE"（greedy in the limit with infinite exploration，极限贪婪且无限探索）是收敛条件。
- **Q-learning 中的最大化偏差。** 当 `Q` 有噪声时，`max` 算子会向上偏置。它会导致过估计——Hasselt 的 Double Q-learning（第 05 课中的 DDQN 使用了它）用两张 Q 表修复这个问题。
- **不终止的回合。** TD 可以在没有终止状态的情况下学习，但你需要限制步数，或在限制处正确处理自举。标准做法：把步数上限视为非终止，继续自举。
- **状态哈希。** 如果状态是元组/张量，请使用可哈希键（使用元组而不是列表；浮点数元组要四舍五入，而不是使用原始值）。

## 使用它

2026 年的 TD 版图：

| 任务 | 方法 | 原因 |
|------|--------|--------|
| 小型表格环境 | Q-learning | 直接学习最优策略。 |
| 同策略安全关键任务 | SARSA / Expected SARSA | 探索期间更保守。 |
| 高维状态 | DQN（第 9 阶段 · 05） | 带经验回放和目标网络的神经网络 Q 函数。 |
| 连续动作 | SAC / TD3（第 9 阶段 · 07） | 在 Q 网络上进行 TD 更新；策略网络输出动作。 |
| LLM RL（基于奖励模型） | PPO / GRPO（第 9 阶段 · 08、12） | Actor-critic，通过 GAE 获得 TD 风格优势。 |
| 离线 RL | CQL / IQL（第 9 阶段 · 08） | 带保守正则化的 Q-learning。 |

你在 2026 年论文中读到的 90% 的“RL”，都是 Q-learning 或 SARSA 的某种扩展。在深入阅读之前，先让自己的手指真正理解表格型更新。

## 交付它

保存为 `outputs/skill-td-agent.md`：

```markdown
---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Given a tabular or small-feature environment, output:

1. Algorithm. Q-learning / SARSA / Expected SARSA / n-step variant. One-sentence reason tied to on-policy vs off-policy and variance.
2. Hyperparameters. α, γ, ε, decay schedule.
3. Initialization. Q_0 value (optimistic vs zero) and justification.
4. Convergence diagnostic. Target learning curve, `|Q - Q*|` check if DP is possible.
5. Deployment caveat. How will exploration behave at inference? Is SARSA's conservatism needed?

Refuse to apply tabular TD to state spaces > 10⁶. Refuse to ship a Q-learning agent without a max-bias caveat. Flag any agent trained with ε held at 1.0 throughout (no exploitation phase).
```

## 练习

1. **简单。** 在 4×4 GridWorld 上实现 Q-learning 和 SARSA。为 2,000 个回合绘制学习曲线（每 100 个回合的平均回报）。谁收敛得更快？
2. **中等。** 构建一个悬崖行走环境（4×12，最后一行是悬崖，奖励 -100，并重置到起点）。比较 Q-learning 和 SARSA 的最终策略。截取各自采取路径的截图。哪一个更靠近悬崖？
3. **困难。** 实现 Double Q-learning。在带噪声奖励的 GridWorld 中（每步奖励加入高斯噪声 σ=5），展示 Q-learning 会明显高估 `V*(0,0)`，而 Double Q-learning 不会。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| TD error | “更新信号” | `δ = r + γ V(s') - V(s)`，自举残差。 |
| TD(0) | “一步 TD” | 每次转移后只使用下一状态的估计进行更新。 |
| Q-learning | “离策略 RL 入门” | 对下一状态动作取 `max` 的 TD 更新；无论行为策略如何，都学习 `Q*`。 |
| SARSA | “同策略 Q-learning” | 使用实际下一动作的 TD 更新；为当前 ε-greedy π 学习 `Q^π`。 |
| Expected SARSA | “低方差 SARSA” | 将采样到的 `a'` 替换为其在 π 下的期望。 |
| GLIE | “正确的探索调度” | Greedy in the Limit with Infinite Exploration；Q-learning 收敛所需条件。 |
| Bootstrapping | “在目标中使用当前估计” | 区分 TD 与 MC 的关键。它是偏差来源，但能大幅降低方差。 |
| Maximization bias | “Q-learning 会过估计” | 对有噪声估计取 `max` 会向上偏置；由 Double Q-learning 修复。 |

## 延伸阅读

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — 原始论文与收敛证明。
- [Sutton & Barto (2018). Ch. 6 — Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0)、SARSA、Q-learning、Expected SARSA。
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — 最大化偏差的修复方法。
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) — Expected SARSA 的动机。
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — 创造 SARSA 这一名称的论文（当时称为“modified connectionist Q-learning”）。
- [Sutton & Barto (2018). Ch. 7 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) — 将 TD(0) 泛化为 TD(n)，这是从 Q-learning 通往资格迹，以及后来 PPO 中 GAE 的路径。
