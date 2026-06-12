# MDP、状态、动作与奖励

> 马尔可夫决策过程（Markov Decision Process, MDP）由五件事组成：状态、动作、转移、奖励和折扣。强化学习（RL）中的一切——Q-learning、PPO、DPO、GRPO——都在这个结构上进行优化。学会一次，后面的强化学习就能顺畅许多。

**类型：** 学习
**语言：** Python
**先修要求：** Phase 1 · 06（概率与分布），Phase 2 · 01（机器学习分类）
**时间：** 约 45 分钟

## 问题

你正在写一个国际象棋机器人。或者一个库存规划器。或者一个交易智能体。或者训练推理模型的 PPO 循环。四个完全不同的领域，却有一个令人意外的事实：它们都可以归约为同一个数学对象。

监督学习给你 `(x, y)` 对，并要求你拟合一个函数。强化学习不给你标签——只有一串状态、你采取的动作，以及一个标量奖励。那一步棋赢了吗？补货决策省钱了吗？交易盈利了吗？LLM 刚刚生成的 token 是否从评判器那里带来了更高的奖励？

在形式化之前，你无法从这条流中学习。"我看到了什么"、"我做了什么"、"接下来发生了什么"、"这有多好"——每一项都必须变成一个可以推理的对象。这种形式化就是马尔可夫决策过程。本阶段中的每个 RL 算法，包括最后的 RLHF 和 GRPO 循环，都在这个结构上进行优化。

## 概念

![马尔可夫决策过程：状态、动作、转移、奖励、折扣](../assets/mdp.svg)

**五个对象。**

- **状态（States）** `S`。智能体做决策所需的一切。在 GridWorld 中，是格子位置。在国际象棋中，是棋盘。在 LLM 中，是上下文窗口加上任何记忆。
- **动作（Actions）** `A`。可选项。向上/下/左/右移动。走一步棋。发出一个 token。
- **转移（Transitions）** `P(s' | s, a)`。给定状态 `s` 和动作 `a`，下一个状态的分布。国际象棋中是确定性的，库存问题中是随机的，LLM 解码中近似确定。
- **奖励（Rewards）** `R(s, a, s')`。标量信号。胜利 = +1，失败 = -1。收入减成本。GRPO 中的对数似然比项。
- **折扣（Discount）** `γ ∈ [0, 1)`。未来奖励相对于当前奖励有多重要。`γ = 0.99` 对应约 100 步的视野；`γ = 0.9` 对应约 10 步。

**马尔可夫性质（Markov property）** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`。未来只依赖当前状态。如果并非如此，说明状态表示不完整——这不是方法失败，而是状态设计失败。

**策略与回报。** 策略（policy）`π(a | s)` 将状态映射为动作分布。回报（return）`G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …` 是未来奖励的折扣和。价值（value）`V^π(s) = E[G_t | s_t = s]` 是在策略 `π` 下从 `s` 出发的期望回报。Q 值（Q-value）`Q^π(s, a) = E[G_t | s_t = s, a_t = a]` 是以特定动作开始时的期望回报。每个 RL 算法都会估计这两者之一，然后相应地改进 `π`。

**贝尔曼方程（Bellman equations）。** 本阶段所有内容都会用到的不动点方程：

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

它们把期望回报拆成"这一步的奖励"加上"你落到的位置的折扣价值"。这是递归的。Phase 9 中的每个算法，要么迭代这个方程直到收敛（动态规划），要么从中采样（蒙特卡洛），要么用它进行一步自举（时序差分）。

```figure
discount-horizon
```

## 构建它

### 第 1 步：一个极小的确定性 MDP

一个 4×4 的 GridWorld。智能体从左上角开始，终止状态在右下角，每走一步奖励 -1，动作 `{up, down, left, right}`。见 `code/main.py`。

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

五行。这就是整个环境。确定性转移，恒定的单步惩罚，吸收型终止状态。

### 第 2 步：展开一个策略

策略是从状态到动作分布的函数。最简单的策略：均匀随机。

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

运行随机策略 1000 次。对于这个 4×4 棋盘，平均回报大约在 -60 到 -80 之间。最优回报是 -6（沿直线路径向下向右）。缩小这个差距，就是 Phase 9 的全部内容。

### 第 3 步：通过贝尔曼方程精确计算 `V^π`

对于小型 MDP，贝尔曼方程是一个线性系统。枚举状态，应用期望，迭代直到价值不再变化。

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

这就是迭代策略评估（iterative policy evaluation）。它是 Sutton & Barto 中的第一个算法，也是后续每个 RL 方法的理论基础。

### 第 4 步：`γ` 是一个具有物理意义的超参数

有效视野大约是 `1 / (1 - γ)`。`γ = 0.9` → 10 步。`γ = 0.99` → 100 步。`γ = 0.999` → 1000 步。

太低，智能体会短视。太高，信用分配会变得嘈杂，因为许多早期步骤都要共同为遥远未来的奖励负责。LLM RLHF 通常使用 `γ = 1`，因为 episode 很短且有边界。控制任务使用 `0.95–0.99`。长视野策略游戏使用 `0.999`。

## 常见陷阱

- **非马尔可夫状态。** 如果你需要最近三次观测才能做决定，那么"状态"就不只是当前观测。修复方法：堆叠帧（Atari 上的 DQN 堆叠 4 帧）或使用循环状态（在观测上使用 LSTM/GRU）。
- **稀疏奖励。** 只有胜负的奖励会让大状态空间中的学习几乎不可能。塑造奖励（中间信号），或用模仿学习进行自举（Phase 9 · 09）。
- **奖励黑客（reward hacking）。** 优化代理奖励经常会产生病态行为。OpenAI 的赛艇智能体没有完成比赛，而是原地绕圈，不断收集道具。始终根据目标结果定义奖励，而不是根据代理指标定义奖励。
- **折扣设定错误。** 在无限视野任务上使用 `γ = 1` 会让每个价值都变成无穷大。必须用有限视野或 `γ < 1` 来截断。
- **奖励尺度。** {+100, -100} 和 {+1, -1} 的奖励会给出相同的最优策略，但梯度幅度可能截然不同。在输入 PPO/DQN 之前，将其归一化到接近 `[-1, 1]` 的范围。

## 使用它

2026 年的技术栈会在接触代码之前，把每个 RL 流水线都归约为一个 MDP：

| 场景 | 状态 | 动作 | 奖励 | γ |
|-----------|-------|--------|--------|---|
| 控制（运动、操作） | 关节角度 + 速度 | 连续力矩 | 任务特定的塑形奖励 | 0.99 |
| 游戏（国际象棋、围棋、扑克） | 棋盘 + 历史 | 合法走法 | 胜利=+1 / 失败=-1 | 1.0（有限） |
| 库存 / 定价 | 库存 + 需求 | 订购数量 | 收入 - 成本 | 0.95 |
| 面向 LLM 的 RLHF | 上下文 token | 下一个 token | 结束时的奖励模型分数 | 1.0（episode 约 200 个 token） |
| 面向推理的 GRPO | 提示 + 部分回答 | 下一个 token | 结束时的验证器 0/1 | 1.0 |

在写任何训练循环之前，先写出这五元组。大多数"RL 不工作"的 bug 报告，最终都可以追溯到纸面上的 MDP 形式化已经坏了。

## 交付它

保存为 `outputs/skill-mdp-modeler.md`：

```markdown
---
name: mdp-modeler
description: 给定一个任务描述，在训练前生成马尔可夫决策过程规范，并标记形式化风险。
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

给定一个任务（控制 / 游戏 / 推荐 / LLM 微调），输出：

1. 状态。精确的特征向量或张量规范。说明马尔可夫性质为何成立。
2. 动作。离散集合或连续范围。维度。
3. 转移。确定性、具有已知模型的随机性，或只能采样。
4. 奖励。函数和来源。稀疏还是塑形。终止奖励还是逐步奖励。
5. 折扣。数值和视野依据。

如果状态是非马尔可夫的，且没有明确提到帧堆叠或循环状态，则拒绝交付该 MDP。如果奖励不是根据目标结果定义的，则拒绝交付。标记无限视野任务上的任何 `γ ≥ 1.0`。将任何大于典型单步奖励 100 倍的奖励范围标记为可能的梯度爆炸来源。
```

## 练习

1. **简单。** 在 `code/main.py` 中实现 4×4 GridWorld 和随机策略 rollout。运行 10,000 个 episode。报告回报的均值和标准差。与最优回报（-6）比较。
2. **中等。** 对均匀随机策略，使用 `γ ∈ {0.5, 0.9, 0.99}` 运行 `policy_evaluation`。将每个 `V` 打印为 4×4 网格。解释为什么终止状态附近的状态价值会随着更大的 `γ` 增长得更快。
3. **困难。** 将 GridWorld 变为随机环境：每个动作以概率 `p = 0.1` 滑向相邻方向。重新评估均匀策略。`V[start]` 会变好还是变差？为什么？

## 关键术语

| 术语 | 人们通常怎么说 | 它真正的含义 |
|------|-----------------|-----------------------|
| MDP | "强化学习设置" | 满足马尔可夫性质的五元组 `(S, A, P, R, γ)`。 |
| 状态 | "智能体看到的东西" | 在所选策略类下，对未来动态足够充分的统计量。 |
| 策略 | "智能体的行为" | 条件分布 `π(a \| s)` 或确定性映射 `s → a`。 |
| 回报 | "总奖励" | 从当前步骤开始的折扣和 `Σ γ^t r_t`。 |
| 价值 | "一个状态有多好" | 在 `π` 下从 `s` 出发的期望回报。 |
| Q 值 | "一个动作有多好" | 在 `π` 下从 `s` 出发并以动作 `a` 开始的期望回报。 |
| 贝尔曼方程 | "动态规划递归" | 将价值 / Q 分解为一步奖励加折扣后继价值的不动点分解。 |
| 折扣 `γ` | "未来 vs 当前" | 远未来奖励上的几何权重；有效视野 `~1/(1-γ)`。 |

## 延伸阅读

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — 经典教材。第 3 章介绍 MDP 和贝尔曼方程；第 1 章解释奖励假说，它是后续每节课的基础。
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) — 贝尔曼方程的源头。
- [OpenAI Spinning Up — Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — 从深度 RL 角度出发的简洁 MDP 入门。
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — 关于 MDP 和精确求解方法的运筹学参考书。
- [Littman (1996). Algorithms for Sequential Decision Making (PhD thesis)](https://www.cs.rutgers.edu/~mlittman/papers/thesis-main.pdf) — 将 MDP 作为动态规划特例来推导的最清晰版本。
