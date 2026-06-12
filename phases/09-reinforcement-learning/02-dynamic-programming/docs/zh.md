# 动态规划（Dynamic Programming）— 策略迭代与价值迭代

> 动态规划就是"作弊版"的强化学习。你已经知道转移函数和奖励函数；你只需要反复迭代贝尔曼方程，直到 `V` 或 `π` 不再变化。它是所有基于采样的方法都试图逼近的基准。

**类型：** 构建
**语言：** Python
**先修要求：** 第 9 阶段 · 01（MDP）
**时间：** 约 75 分钟

## 问题

你有一个模型已知的 MDP：对任意状态-动作对，都可以查询 `P(s' | s, a)` 和 `R(s, a, s')`。库存经理知道需求分布。棋盘游戏有确定性转移。一个网格世界（GridWorld）只需要四行 Python。你拥有一个*模型*。

无模型强化学习（model-free RL，如 Q-learning、PPO、REINFORCE）是为没有模型的情况发明的——你只能从环境中采样。但当你确实拥有模型时，就有更快、更好的方法：动态规划。Bellman 在 1957 年设计了这些方法。它们至今仍定义着正确性：当人们说"这个 MDP 的最优策略"时，他们指的就是 DP 会返回的策略。

在 2026 年，你需要掌握它们有三个原因。第一，RL 研究中的每个表格型环境（GridWorld、FrozenLake、CliffWalking）都会用 DP 求解，以生成金标准策略。第二，精确值可以让你*调试*采样方法：如果 Q-learning 对 `V*(s_0)` 的估计与 DP 答案相差 30%，那你的 Q-learning 就有 bug。第三，现代离线 RL 和规划方法（MCTS、AlphaZero 的搜索、第 9 阶段 · 10 中的基于模型的 RL）都会在学习到的或给定的模型上迭代贝尔曼备份（Bellman backup）。

## 概念

![策略迭代和价值迭代并排展示](../assets/dp.svg)

**两个算法，本质上都是对贝尔曼方程做不动点迭代（fixed-point iteration）。**

**策略迭代（Policy iteration）。** 交替执行两个步骤，直到策略不再变化。

1. *评估（Evaluation）：* 给定策略 `π`，通过反复应用 `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]` 来计算 `V^π`，直到收敛。
2. *改进（Improvement）：* 给定 `V^π`，让 `π` 相对于 `V^π` 变为贪心策略：`π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`。

收敛是有保证的，因为 (a) 每次改进步骤要么保持 `π` 不变，要么严格提高某些状态的 `V^π`，(b) 确定性策略的空间是有限的。即使状态空间很大，通常也会在约 5–20 次外层迭代内收敛。

**价值迭代（Value iteration）。** 将评估和改进压缩为一次扫描。应用贝尔曼*最优性*方程：

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

重复直到 `max_s |V_{new}(s) - V(s)| < ε`。最后通过选择贪心动作来提取策略。每次迭代严格来说更快——没有内层评估循环——但通常需要更多迭代才能收敛。

**广义策略迭代（Generalized policy iteration, GPI）。** 这是统一的表述方式。价值函数和策略被锁定在一个双向改进循环中；任何推动二者走向相互一致的方法（异步价值迭代、修正策略迭代、Q-learning、actor-critic、PPO）都是 GPI 的一个实例。

**为什么 `γ < 1` 很重要。** 贝尔曼算子在上确界范数（sup-norm）下是一个 `γ`-压缩映射：`||T V - T V'||_∞ ≤ γ ||V - V'||_∞`。压缩性意味着唯一不动点和几何收敛。去掉 `γ < 1` 后，你就失去了这个保证——你需要有限时域，或一个吸收性的终止状态。

```figure
value-iteration-gamma
```

## 构建它

### 步骤 1：构建 GridWorld MDP 模型

使用第 01 课中的同一个 4×4 GridWorld。我们添加一个随机版本：智能体有 `0.1` 的概率滑向一个随机的垂直方向。

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)` 返回一个 `(s', r, p)` 列表。这就是完整的模型。

### 步骤 2：策略评估

给定策略 `π(s) = {action: prob}`，迭代贝尔曼方程，直到 `V` 不再变化：

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### 步骤 3：策略改进

用相对于 `V` 的贪心策略替换 `π`。如果 `π` 没有变化，就返回——我们已经到达最优。

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### 步骤 4：把它们串起来

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # arbitrary start
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

4×4 上的典型收敛情况：4–6 次外层迭代。输出 `V*(0,0) ≈ -6`，以及一个会严格减少步数的策略。

### 步骤 5：价值迭代（单循环版本）

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

相同的不动点，更少的代码行。

## 常见陷阱

- **忘记处理终止状态。** 如果你对吸收状态应用贝尔曼方程，它仍然会选出一个"最佳动作"，但这个动作什么都不会改变。用 `if s == terminal: V[s] = 0` 做保护。
- **上确界范数 vs L2 收敛。** 使用 `max |V_new - V|`，而不是平均值。理论保证针对的是上确界范数。
- **原地更新 vs 同步更新。** 原地更新 `V[s]`（Gauss-Seidel）比单独使用 `V_new` 字典（Jacobi）收敛更快。生产代码使用原地更新。
- **策略平局。** 如果两个动作有相同的 Q 值，`argmax` 可能在每次迭代中用不同方式打破平局，导致"策略稳定"检查发生振荡。使用稳定的平局处理方式（固定顺序中的第一个动作）。
- **状态空间爆炸。** DP 每次扫描的复杂度是 `O(|S| · |A|)`。它可以工作到约 10⁷ 个状态。再往上，你需要函数近似（第 9 阶段 · 05 及以后）。

## 使用它

在 2026 年，DP 是正确性基线，也是规划器的内层循环：

| 使用场景 | 方法 |
|----------|--------|
| 精确求解一个小型表格型 MDP | 价值迭代（更简单）或策略迭代（外层步骤更少） |
| 验证 Q-learning / PPO 实现 | 在玩具环境上与 DP 最优的 V* 比较 |
| 基于模型的 RL（第 9 阶段 · 10） | 在学习到的转移模型上做贝尔曼备份 |
| AlphaZero / MuZero 中的规划 | 蒙特卡洛树搜索（Monte Carlo Tree Search）= 异步贝尔曼备份 |
| 离线 RL（CQL、IQL） | 保守 Q 迭代（Conservative Q-iteration）——带有 OOD 动作惩罚的 DP |

每当有人说"最优价值函数"时，他们的意思就是"DP 不动点"。当你在论文中看到 `V*` 或 `Q*` 时，请想象这个循环。

## 交付它

保存为 `outputs/skill-dp-solver.md`：

```markdown
---
name: dp-solver
description: 通过策略迭代或价值迭代精确求解一个小型表格型 MDP。报告收敛行为。
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

给定一个模型已知的 MDP，输出：

1. 选择。策略迭代 vs 价值迭代。理由要关联到 |S|、|A|、γ。
2. 初始化。V_0、起始策略。收敛敏感性。
3. 停止。上确界范数容差 ε。预期扫描次数。
4. 验证。精确计算出的 V*(s_0)。提取出的贪心策略。
5. 使用。这个基线将如何用于调试/评估基于采样的方法。

拒绝在 > 10⁷ 的状态空间上运行 DP。没有上确界范数检查时，拒绝声称已收敛。将无限时域任务中的任何 γ ≥ 1 标记为违反保证。
```

## 练习

1. **简单。** 在 4×4 GridWorld 上用 `γ ∈ {0.9, 0.99}` 运行价值迭代。直到 `max |ΔV| < 1e-6` 需要多少次扫描？将 `V*` 打印成 4×4 网格。
2. **中等。** 在*随机* GridWorld（滑移概率 `0.1`）上比较策略迭代与价值迭代。统计：扫描次数、墙钟时间、最终的 `V*(0,0)`。哪个在迭代次数上收敛更快？哪个在墙钟时间上更快？
3. **困难。** 构建修正策略迭代：在评估步骤中，只运行 `k` 次扫描，而不是运行到收敛。绘制 `k ∈ {1, 2, 5, 10, 50}` 时 `V*(0,0)` 误差随 `k` 的变化。曲线告诉你评估/改进之间的权衡是什么？

## 关键术语

| 术语 | 人们常说 | 它实际的意思 |
|------|-----------------|-----------------------|
| 策略迭代 | "DP 算法" | 交替进行评估（`V^π`）和改进（相对于 `V^π` 的贪心 `π`），直到策略不再变化。 |
| 价值迭代 | "更快的 DP" | 在一次扫描中应用贝尔曼最优性备份；以几何速度收敛到 `V*`。 |
| 贝尔曼算子 | "那个递归" | `(T V)(s) = max_a Σ P (r + γ V(s'))`；在上确界范数下是一个 `γ`-压缩映射。 |
| 压缩映射 | "DP 为什么会收敛" | 任何满足 `\|\|T x - T y\|\| ≤ γ \|\|x - y\|\|` 的算子 `T` 都有唯一不动点。 |
| GPI | "一切都是 DP" | 广义策略迭代：任何推动 `V` 和 `π` 达到相互一致的方法。 |
| 同步更新 | "Jacobi 风格" | 在一次扫描中始终使用旧的 `V`；便于清晰分析，但更慢。 |
| 原地更新 | "Gauss-Seidel 风格" | 在 `V` 被更新的同时使用它；实践中收敛更快。 |

## 延伸阅读

- [Sutton & Barto (2018). Ch. 4 — Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf) — 关于策略迭代和价值迭代的经典表述。
- [Bertsekas (2019). Reinforcement Learning and Optimal Control](http://www.athenasc.com/rlbook.html) — 对压缩映射论证的严谨处理。
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — 修正策略迭代及其收敛分析。
- [Howard (1960). Dynamic Programming and Markov Processes](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) — 最初的策略迭代论文。
- [Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming](http://www.athenasc.com/ndpbook.html) — 从 DP 到近似 DP / 深度 RL 的桥梁，后续每一课都会用到。
