# 多智能体强化学习

> 单智能体强化学习（single-agent RL）假设环境是平稳的。把两个正在学习的智能体放进同一个世界，这个假设就会失效：每个智能体都是另一个智能体环境的一部分，而且双方都在变化。多智能体强化学习（Multi-Agent RL, MARL）就是一组技巧，用来在马尔可夫假设不再成立时仍然让学习收敛。

**类型：** 构建
**语言：** Python
**先修要求：** 第 9 阶段 · 04（Q-learning）、第 9 阶段 · 06（REINFORCE）、第 9 阶段 · 07（Actor-Critic）
**时间：** 约 45 分钟

## 问题

一个机器人学习在房间中导航，这是单智能体强化学习问题。一个足球队不是。AlphaStar 对战 StarCraft 对手不是。由竞价智能体组成的市场不是。两辆车在四向停车路口协商通行不是。许多现实世界中的多对多问题都不是。

在每个多智能体设定中，从任意一个智能体的视角看，其他智能体*就是*环境的一部分。随着它们学习并改变自身行为，环境会变得非平稳（non-stationary）。马尔可夫性质（Markov property）——“下一个状态只依赖当前状态和我的动作”——会被破坏，因为下一个状态还取决于*其他*智能体选择了什么，而它们的策略又是移动目标。

这会破坏表格型方法的收敛性证明（Q-learning 的保证假设环境是平稳的）。它也会破坏朴素的深度强化学习：智能体会在循环中互相追逐，永远无法收敛到稳定策略。你需要多智能体专用技术：集中式训练 / 分散式执行（centralized training / decentralized execution）、反事实基线（counterfactual baselines）、联赛训练（league play）、自博弈（self-play）。

2026 年的应用包括：机器人群体、交通路由、自动驾驶车队、市场模拟器、多智能体 LLM 系统（第 16 阶段），以及任何拥有多个智能玩家的游戏。

## 概念

![四种 MARL 模式：独立、集中式 critic、自博弈、联赛](../assets/marl.svg)

**形式化：马尔可夫博弈（Markov Game）。** 它是 MDP 的泛化：状态 `S`、联合动作 `a = (a_1, …, a_n)`、转移 `P(s' | s, a)`，以及每个智能体的奖励 `R_i(s, a, s')`。每个智能体 `i` 都在自己的策略 `π_i` 下最大化自己的回报。如果奖励完全相同，它就是**完全合作型**。如果是零和的，它就是**对抗型**。如果二者混合，它就是**一般和（general-sum）**。

**核心挑战：**

- **非平稳性。** 从智能体 `i` 的视角看，`P(s' | s, a_i)` 依赖于正在变化的 `π_{-i}`。
- **信用分配（Credit assignment）。** 在共享奖励下，到底是哪个智能体造成了这个结果？
- **探索协调。** 智能体必须探索互补策略，而不是冗余地探索同一个状态。
- **可扩展性。** 联合动作空间会随 `n` 指数增长。
- **部分可观测性。** 每个智能体只能看到自己的观测；全局状态是隐藏的。

**四种主流模式：**

**1. 独立 Q-learning / 独立 PPO（IQL, IPPO）。** 每个智能体学习自己的 Q 或策略，把其他智能体当作环境的一部分。简单，有时有效（尤其当经验回放起到一种平滑的智能体建模技巧作用时）。理论收敛性：没有。实践中：对弱耦合任务还可以，对强耦合任务很差。

**2. 集中式训练，分散式执行（CTDE）。** 最常见的现代范式。每个智能体都有自己的*策略* `π_i`，该策略以局部观测 `o_i` 为条件——部署时采用标准的分散式执行。在*训练*期间，集中式 critic `Q(s, a_1, …, a_n)` 以完整全局状态和联合动作为条件。示例：
- **MADDPG**（Lowe et al. 2017）：为每个智能体配备集中式 critic 的 DDPG。
- **COMA**（Foerster et al. 2017）：反事实基线——问“如果我当时采取动作 `a'`，我的奖励会是多少？”——隔离出我的贡献。
- **MAPPO** / **IPPO** 搭配共享 critic（Yu et al. 2022）：使用集中式价值函数的 PPO。到 2026 年，它在合作型 MARL 中占主导地位。
- **QMIX**（Rashid et al. 2018）：价值分解——`Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))`，并使用单调混合。

**3. 自博弈。** 同一个智能体的两个副本互相对战。对手的策略*就是*我过去某个快照中的策略。AlphaGo / AlphaZero / MuZero。OpenAI Five。最适合零和游戏；训练信号是对称的。

**4. 联赛训练。** 自博弈向一般和 / 对抗环境的扩展：保留一组过去和当前策略，从联赛中采样一个对手，并与其对战训练。加入 exploiters（专门击败当前最强策略）和 main exploiters（专门击败 exploiters）。AlphaStar（StarCraft II）。当游戏存在“石头-剪刀-布”式策略循环时，这是必要的。

**通信。** 允许智能体彼此发送学习到的消息 `m_i`。这适用于合作型设定。Foerster et al. (2016) 表明，可微的智能体间通信可以端到端训练。今天基于 LLM 的多智能体系统（第 16 阶段）本质上是在用自然语言通信。

## 构建它

本课使用一个 6×6 GridWorld，其中有两个合作智能体。它们从相对的角落出发，必须到达一个共享目标。共享奖励：只要任一智能体仍在移动，每步 `-1`；当二者都到达时 `+10`。见 `code/main.py`。

### 第 1 步：多智能体环境

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # two agents

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

*联合*动作空间是 `|A|² = 16`。全局状态是两个位置。

### 第 2 步：独立 Q-learning

每个智能体运行自己的 Q-table，以联合状态为键。在每一步：二者都选择 ε-greedy 动作，收集联合转移，然后各自用共享奖励更新自己的 Q。

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

它在这个任务上有效，因为奖励是密集且对齐的。它会在强耦合任务上失败（例如，一个智能体必须*等待*另一个智能体的任务）。

### 第 3 步：使用分解价值更新的集中式 Q

使用一个定义在联合动作上的 Q：`Q(s, a_1, a_2)`。从共享奖励进行更新。执行时通过边缘化实现分散式执行：`π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`。这是用指数级联合动作空间换取一个*正确的*全局视角。

### 第 4 步：简单自博弈（对抗型 2 智能体）

同一个智能体，两个角色。训练智能体 A 对抗智能体 B；每经过 `K` 个 episode，就把 A 的权重复制到 B。对称训练，持续进步。迷你版 AlphaZero 配方。

## 陷阱

- **非平稳回放。** 对独立智能体使用经验回放比单智能体更糟，因为旧转移是由现在已经过时的对手生成的。修复：重标注，或按新近程度加权。
- **信用分配模糊。** 长 episode 之后才得到共享奖励；没有清晰方法说明哪个智能体做出了贡献。修复：反事实基线（COMA），或为每个智能体进行奖励塑形。
- **策略漂移 / 追逐。** 每个智能体的最佳响应都会随着其他智能体的更新而改变。修复：集中式 critic、较慢的学习率，或一次只冻结一个智能体。
- **通过协调进行奖励黑客。** 智能体找到了设计者没有预料到的协调式漏洞。拍卖智能体会收敛到出价为零。修复：谨慎的奖励设计、行为约束。
- **探索冗余。** 两个智能体探索相同的状态-动作对。修复：每个智能体的熵奖励，或角色条件化。
- **联赛循环。** 纯自博弈可能陷入支配循环。修复：使用拥有多样化对手的联赛训练。
- **样本爆炸。** `n` 个智能体 × 状态空间 × 联合动作。用函数近似来近似；因子化动作空间（每个智能体一个策略输出头）。

## 使用它

2026 年 MARL 应用地图：

| 领域 | 方法 | 备注 |
|--------|--------|-------|
| 合作导航 / 操作 | MAPPO / QMIX | CTDE；共享 critic + 分散式 actor。 |
| 双人游戏（国际象棋、围棋、扑克） | 结合 MCTS 的自博弈（AlphaZero） | 零和；对称训练。 |
| 复杂多人游戏（Dota、StarCraft） | 联赛训练 + 模仿预训练 | OpenAI Five、AlphaStar。 |
| 自动驾驶车队 | CTDE MAPPO / 带 attention 的 PPO | 部分观测；可变团队规模。 |
| 拍卖市场 | 博弈论均衡 + RL | 当 `n` → ∞ 时使用平均场 RL。 |
| LLM 多智能体系统（第 16 阶段） | 自然语言通信 + 角色条件化 | RL 循环位于智能体规划层。 |

到 2026 年，MARL 增长最快的领域是基于 LLM 的系统：由语言模型智能体组成的群体进行协商、辩论、构建软件。这里的 RL 体现为对*轨迹级*输出的偏好优化，而不是 token 级输出（第 16 阶段 · 03）。

## 交付它

保存为 `outputs/skill-marl-architect.md`：

```markdown
---
name: marl-architect
description: Pick the right multi-agent RL regime (IPPO, CTDE, self-play, league) for a given task.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with `n` agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).
```

## 练习

1. **简单。** 在 2 智能体合作型 GridWorld 上训练独立 Q-learning。平均回报 > 0 需要多少个 episode？绘制联合学习曲线。
2. **中等。** 添加一个“协调”任务：只有当两个智能体在同一回合踏上目标时，目标才算达成。独立 Q 还会收敛吗？哪里出问题了？
3. **困难。** 为 MAPPO 风格训练实现一个集中式 critic，并在协调任务上把收敛速度与独立 PPO 进行比较。

## 关键术语

| 术语 | 人们常说的说法 | 它的真实含义 |
|------|-----------------|-----------------------|
| 马尔可夫博弈 | “多智能体 MDP” | `(S, A_1, …, A_n, P, R_1, …, R_n)`；每个智能体都有自己的奖励。 |
| CTDE | “集中式训练，分散式执行” | 训练时使用联合 critic；每个智能体的策略只使用局部观测。 |
| IPPO | “独立 PPO” | 每个智能体分别运行 PPO。简单基线；常常被低估。 |
| MAPPO | “多智能体 PPO” | 使用以全局状态为条件的集中式价值函数的 PPO。 |
| QMIX | “单调价值分解” | `Q_tot = f_monotone(Q_1, …, Q_n)` 允许分散式 argmax。 |
| COMA | “反事实多智能体” | Advantage = 我的 Q 减去在我的动作上边缘化后的期望 Q。 |
| 自博弈 | “智能体对战过去的自己” | 单个智能体，两个角色；零和游戏的标准做法。 |
| 联赛训练 | “种群训练” | 缓存过去策略，从池中采样对手；处理策略循环。 |

## 延伸阅读

- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — 使用集中式 critic 的 CTDE。
- [Foerster et al. (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) — 用于信用分配的反事实基线。
- [Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485) — 带单调性的价值分解。
- [Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955) — PPO 在 MARL 中出人意料地强大。
- [Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) — 大规模联赛训练。
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — 零和游戏中的纯自博弈。
- [Sutton & Barto (2018). Ch. 15 — Neuroscience & Ch. 17 — Frontiers](http://incompleteideas.net/book/RLbook2020.pdf) — 包含教材对多智能体设定的简短处理，以及 CTDE 旨在解决的非平稳性问题。
- [Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635) — 覆盖合作型、竞争型和混合型 MARL 及其收敛结果的综述。
