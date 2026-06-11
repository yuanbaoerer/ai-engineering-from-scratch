# 蒙特卡洛方法 — 从完整回合中学习

> 动态规划需要模型。蒙特卡洛只需要回合。运行策略，观察回报，对它们求平均。这是强化学习中最简单的想法，也是开启后续所有内容的关键。

**类型：** 构建
**语言：** Python
**先修要求：** Phase 9 · 01（MDP），Phase 9 · 02（动态规划）
**时间：** 约 75 分钟

## 问题

动态规划很优雅，但它假设你可以对每个状态和动作查询 `P(s' | s, a)`。现实世界中几乎没有什么系统是这样工作的。机器人无法解析地计算施加某个关节力矩后相机像素的分布。定价算法无法对每一种可能的客户反应做积分。LLM 无法枚举一个 token 之后所有可能的续写。

你需要一种只要求能够从环境中*采样*的方法。运行策略。得到一条轨迹 `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`。用它来估计价值。这就是蒙特卡洛（Monte Carlo）。

从 DP 到 MC 的转变在思想上很重要：我们从*已知模型 + 精确备份*转向*采样式 rollout + 平均回报*。方差会跃升，但适用范围会爆炸式扩大。本课之后的每一个 RL 算法——TD、Q-learning、REINFORCE、PPO、GRPO——本质上都是蒙特卡洛估计器，只是有时在其上叠加了自举（bootstrapping）。

## 核心概念

![蒙特卡洛：rollout、计算回报、求平均；首次访问 vs 每次访问](../assets/monte-carlo.svg)

**核心思想一句话：** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)`，其中 `G^{(i)}(s)` 是在策略 `π` 下访问 `s` 之后观察到的回报。

**首次访问 MC vs 每次访问 MC。** 给定一个多次访问状态 `s` 的回合，首次访问 MC（first-visit MC）只统计第一次访问后的回报；每次访问 MC（every-visit MC）统计所有访问。二者在极限下都是无偏的。首次访问更容易分析（iid 样本）。每次访问每个回合使用更多数据，实践中通常收敛更快。

**增量均值。** 不存储所有回报，而是更新运行平均值：

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

重新整理：`V_new = V_old + α · (target - V_old)`，其中 `α = 1/n`。把 `1/n` 换成常数步长 `α ∈ (0, 1)`，你就得到一个非平稳 MC 估计器，它可以跟踪 `π` 的变化。这个动作就是从 MC 跳到 TD，再跳到每一种现代 RL 算法的完整核心。

**探索现在成了问题。** DP 通过枚举触达每个状态。MC 只能看到策略访问到的状态。如果 `π` 是确定性的，状态空间中的整片区域永远不会被采样，它们的价值估计会永远停留在零。按历史顺序，有三种修复方式：

1. **探索起始（exploring starts）。** 从随机的 (s, a) 对开始每个回合。能保证覆盖；但实践中不现实（你不能把机器人“重置”到任意状态）。
2. **ε-贪心（ε-greedy）。** 相对于当前 Q 采取贪心动作，但以概率 `ε` 选择一个随机动作。所有状态-动作对都会渐近地被采样到。
3. **离策略 MC（off-policy MC）。** 在行为策略 `μ` 下收集数据，通过重要性采样（importance sampling）学习目标策略 `π`。方差很高，但它是通向 DQN 这类回放缓冲区方法的桥梁。

**蒙特卡洛控制。** 评估 → 改进 → 评估，就像策略迭代一样，但评估是基于采样的：

1. 运行 `π`，得到一个回合。
2. 根据观察到的回报更新 `Q(s, a)`。
3. 使 `π` 相对于 `Q` 变为 ε-贪心。
4. 重复。

在温和条件下（每个对被无限次访问，`α` 满足 Robbins-Monro 条件），以概率 1 收敛到 `Q*` 和 `π*`。

## 构建它

### 第 1 步：rollout → (s, a, r) 列表

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

没有模型，只有 `env.reset()` 和 `env.step(s, a)`。接口和 gym 环境一样，但被精简了。

### 第 2 步：计算回报（反向扫描）

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

一次遍历，`O(T)`。反向递推 `G_t = r_{t+1} + γ G_{t+1}` 避免了重复求和。

### 第 3 步：首次访问 MC 评估

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

真正工作的只有三行：在首次访问时把状态标记为 seen，增加计数，更新运行均值。

### 第 4 步：ε-贪心 MC 控制（同策略）

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### 第 5 步：与 DP 金标准比较

你对 `V^π` 的 MC 估计应该在回合数 → ∞ 时与第 02 课中的 DP 结果一致。实践中：在 4×4 GridWorld 上运行 50,000 个回合，可以让你距离 DP 答案在 `~0.1` 以内。

## 常见陷阱

- **无限回合。** MC 要求回合必须*终止*。如果你的策略可能永远循环，就设置 `max_steps` 上限，并把达到上限视为隐式失败。带随机策略的 GridWorld 经常超时——这是正常的，只要确保你正确计数即可。
- **方差。** MC 使用完整回报。在长回合中，方差很大——结尾处一次倒霉的奖励会以同样幅度移动 `V(s_0)`。TD 方法（第 04 课）通过自举来削减这一点。
- **状态覆盖。** 在一个新鲜 Q 上做贪心 MC，若存在并列，只会永远尝试其中一个动作。你*必须*探索（ε-贪心、探索起始、UCB）。
- **非平稳策略。** 如果 `π` 发生变化（如 MC 控制中），旧回报来自不同策略。常数-α MC 可以处理这一点；样本平均 MC 不行。
- **离策略重要性采样。** 权重 `π(a|s)/μ(a|s)` 会沿着轨迹相乘。方差会随 horizon 爆炸。用逐决策加权 IS 截断，或切换到 TD。

## 使用它

蒙特卡洛方法在 2026 年的角色：

| 使用场景 | 为什么用 MC |
|----------|--------|
| 短视野游戏（21 点、扑克） | 回合自然终止；回报干净。 |
| 对已记录策略做离线评估 | 对存储轨迹上的折扣回报求平均。 |
| 蒙特卡洛树搜索（AlphaZero） | 从树叶节点开始的 MC rollout 指导选择。 |
| LLM RL 评估 | 对给定策略的采样补全计算平均奖励。 |
| PPO 中的基线估计 | 优势目标 `A_t = G_t - V(s_t)` 使用 MC 的 `G_t`。 |
| 教学 RL | 最简单但真正能工作的算法——剥离自举以看见核心。 |

现代深度 RL 算法（PPO、SAC）通过 `n` 步回报或 GAE，在纯 MC（完整回报）和纯 TD（一步自举）之间插值。两个端点都是同一个估计器的实例。

## 交付它

保存为 `outputs/skill-mc-evaluator.md`：

```markdown
---
name: mc-evaluator
description: Evaluate a policy via Monte Carlo rollouts and produce a convergence report with DP-comparison if available.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

Given an environment (episodic, with reset+step API) and a policy, output:

1. Method. First-visit vs every-visit MC. Reason.
2. Episode budget. Target number, variance diagnostic, expected standard error.
3. Exploration plan. ε schedule (if needed) or exploring starts.
4. Gold-standard comparison. DP-optimal V* if tabular; otherwise a bound from a Q-learning / PPO baseline.
5. Termination check. Max-step cap, timeouts, handling of non-terminating trajectories.

Refuse to run MC on non-episodic tasks without a finite horizon cap. Refuse to report V^π estimates from fewer than 100 episodes per state for tabular tasks. Flag any policy with zero-variance actions as an exploration risk.
```

## 练习

1. **简单。** 在 4×4 GridWorld 上实现均匀随机策略的首次访问 MC 评估。运行 10,000 个回合。绘制 `V(0,0)` 随回合数变化的曲线，并与 DP 答案对比。
2. **中等。** 用 `ε ∈ {0.01, 0.1, 0.3}` 实现 ε-贪心 MC 控制。比较 20,000 个回合后的平均回报。曲线是什么样的？偏差-方差权衡在哪里？
3. **困难。** 用重要性采样实现*离策略* MC：在均匀随机策略 `μ` 下收集数据，估计确定性最优策略 `π` 的 `V^π`。比较普通 IS、逐决策 IS、加权 IS。哪一个方差最低？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| 蒙特卡洛 | “随机采样” | 通过对来自分布的 iid 样本求平均来估计期望。 |
| 回报 `G_t` | “未来奖励” | 从步骤 `t` 到回合结束的折扣奖励和：`Σ_{k≥0} γ^k r_{t+k+1}`。 |
| 首次访问 MC | “每个状态只计一次” | 一个回合中只有第一次访问会贡献到价值估计。 |
| 每次访问 MC | “使用所有访问” | 每次访问都会贡献；略有偏，但样本效率更高。 |
| ε-贪心 | “探索噪声” | 以概率 `1-ε` 选择贪心动作；以概率 `ε` 选择随机动作。 |
| 重要性采样 | “纠正从错误分布采样的问题” | 用 `π(a\|s)/μ(a\|s)` 乘积对回报重新加权，以便从 `μ` 数据估计 `V^π`。 |
| 同策略 | “从我自己的数据学习” | 目标策略 = 行为策略。原版 MC、PPO、SARSA。 |
| 离策略 | “从别人的数据学习” | 目标策略 ≠ 行为策略。重要性采样 MC、Q-learning、DQN。 |

## 延伸阅读

- [Sutton & Barto (2018). Ch. 5 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 经典处理方式。
- [Singh & Sutton (1996). Reinforcement Learning with Replacing Eligibility Traces](https://link.springer.com/article/10.1007/BF00114726) — 首次访问 vs 每次访问分析。
- [Precup, Sutton, Singh (2000). Eligibility Traces for Off-Policy Policy Evaluation](http://incompleteideas.net/papers/PSS-00.pdf) — 离策略 MC 与方差控制。
- [Mahmood et al. (2014). Weighted Importance Sampling for Off-Policy Learning](https://arxiv.org/abs/1404.6362) — 现代低方差 IS 估计器。
- [Tesauro (1995). TD-Gammon, A Self-Teaching Backgammon Program](https://dl.acm.org/doi/10.1145/203330.203343) — MC/TD 自我博弈收敛到超人类水平的第一个大规模经验证明；也是本阶段后半部分每一课的概念先驱。
