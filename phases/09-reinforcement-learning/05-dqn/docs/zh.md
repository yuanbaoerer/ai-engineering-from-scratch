# 深度 Q 网络（DQN）

> 2013 年：Mnih 在原始像素上训练了一个 Q-learning 网络，在七个 Atari 游戏中击败了所有经典 RL 智能体。2015 年：扩展到 49 个游戏，发表在 Nature 上，开启了深度强化学习（deep RL）时代。DQN 就是 Q-learning 加上三个让函数近似稳定下来的技巧。

**类型：** 构建
**语言：** Python
**先修：** Phase 3 · 03（反向传播），Phase 9 · 04（Q-learning，SARSA）
**时间：** ~75 分钟

## 问题

表格型 Q-learning 需要为每一个（状态，动作）对保存一个单独的 Q 值。一个国际象棋棋盘大约有 10⁴³ 个状态。一帧 Atari 图像是 210×160×3 = 100,800 个特征。表格型 RL 在几千个状态时就已经难以为继，更不用说数十亿个状态了。

事后看来，修复方法很明显：用神经网络 `Q(s, a; θ)` 替代 Q 表。但这个“事后很明显”的想法花了几十年才真正可用。朴素地把函数近似用于 Q-learning，会在“致命三元组”（deadly triad）下发散——函数近似 + 自举（bootstrapping）+ 离策略学习（off-policy learning）。Mnih 等人（2013，2015）识别出了三个能稳定学习过程的工程技巧：

1. **经验回放（Experience replay）** 去相关化转移样本。
2. **目标网络（Target network）** 冻结自举目标。
3. **奖励裁剪（Reward clipping）** 归一化梯度幅度。

Atari 上的 DQN 是第一次用同一个架构和同一套超参数，从原始像素出发解决了数十个控制问题。此后构建的所有“深度 RL”方法——DDQN、Rainbow、Dueling、Distributional、R2D2、Agent57——都叠加在这个三技巧基础之上。

## 概念

![DQN training loop: env, replay buffer, online net, target net, Bellman TD loss](../assets/dqn.svg)

**目标函数。** DQN 在神经 Q 函数上最小化一步 TD 损失：

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = 在线网络（online network），每一步通过梯度下降更新。`θ^-` = 目标网络（target network），周期性地从 `θ` 复制而来（大约每 10,000 步一次）。`D` = 过去转移样本的回放缓冲区（replay buffer）。

**三个技巧，按重要性排序：**

**经验回放。** 一个包含 `~10⁶` 条转移的环形缓冲区。每个训练步骤都从中均匀随机采样一个小批量（minibatch）。这会打破时间相关性（连续帧几乎相同），让网络能多次从罕见的有奖励转移中学习，并去相关化连续的梯度更新。没有它，带神经网络的 on-policy TD 在 Atari 上会发散。

**目标网络。** 在 Bellman 方程两边都使用同一个网络 `Q(·; θ)`，会导致目标在每次更新时都移动——相当于“追着自己的尾巴跑”。修复方法：保留第二个网络 `Q(·; θ^-)`，其权重被冻结。每隔 `C` 步，复制 `θ → θ^-`。这会让回归目标一次稳定数千个梯度步骤。软更新 `θ^- ← τ θ + (1-τ) θ^-`（用于 DDPG、SAC）是更平滑的变体。

**奖励裁剪。** Atari 奖励幅度从 1 到 1000+ 不等。裁剪到 `{-1, 0, +1}` 可以阻止任何单个游戏主导梯度。当奖励幅度本身很重要时，这样做是错误的；但对 Atari 来说没问题，因为重要的是符号。

**Double DQN。** Hasselt（2016）修复了最大化偏差：使用在线网络来*选择*动作，使用目标网络来*评估*它。

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

这是一个即插即用的替换，效果始终更好。默认就用它。

**其他改进（Rainbow，2017）：** 优先经验回放（prioritized replay，更频繁采样高 TD 误差转移）、dueling 架构（分离 `V(s)` 和优势函数头）、noisy networks（学习式探索）、n 步回报、分布式 Q（distributional Q，C51/QR-DQN）、多步自举。每一项都会带来几个百分点的提升；收益大致可以叠加。

## 构建它

这里的代码只使用标准库、不用 numpy——我们在一个很小的连续 GridWorld 上手写单隐藏层 MLP，因此每个训练步骤只需微秒级时间。算法与大规模 Atari DQN 完全相同。

### Step 1: replay buffer

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

Atari 使用约 50,000 的容量；对我们的玩具环境来说，5,000 就够了。

### Step 2: a tiny Q-network (manual MLP)

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

前向传播：线性 → ReLU → 线性。这就是整个网络。

### Step 3: the DQN update

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

它的形状就是第 04 课中的 Q-learning，只有两个区别：(a) 我们通过可微的 `Q(·; θ)` 反向传播，而不是索引一张表；(b) 目标使用 `Q(·; θ^-)`。

### Step 4: the outer loop

每个 episode 中，在 `Q(·; θ)` 上按 ε-greedy 采取动作，把转移推入缓冲区，采样一个小批量，执行一次梯度步骤，并周期性同步 `θ^- ← θ`。模式如下：

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

在我们这个 16 维 one-hot 状态的小型 GridWorld 上，智能体会在约 500 个 episode 内学到接近最优的策略。在 Atari 上，把它扩展到 2 亿帧，并加入 CNN 特征提取器。

## 陷阱

- **致命三元组。** 函数近似 + 离策略 + 自举可能会发散。DQN 用目标网络 + 回放来缓解；不要移除任何一个。
- **探索。** ε 必须衰减，通常在训练最初约 10% 的时间里从 1.0 衰减到 0.01。如果早期探索不足，Q 网络会收敛到局部盆地。
- **过估计。** 对有噪声的 Q 取 `max` 会产生向上偏差。生产环境中始终使用 Double DQN。
- **奖励尺度。** 裁剪或归一化奖励；梯度幅度与奖励幅度成正比。
- **回放缓冲区冷启动。** 不要在缓冲区拥有几千条转移之前开始训练。只在 ~20 个样本上的早期梯度会过拟合。
- **目标同步频率。** 太频繁 ≈ 没有目标网络；太不频繁 ≈ 目标过旧。Atari DQN 使用 10,000 个环境步。经验法则：每隔训练跨度的约 1/100 同步一次。
- **观测预处理。** Atari DQN 堆叠 4 帧，使状态满足马尔可夫性。任何包含速度信息的环境都需要帧堆叠或循环状态。

## 使用它

到 2026 年，DQN 很少是最先进方法，但仍然是参考级离策略算法：

| 任务 | 首选方法 | 为什么不用 DQN？ |
|------|------------------|--------------|
| 类 Atari 的离散动作任务 | Rainbow DQN 或 Muesli | 同一框架，更多技巧。 |
| 连续控制 | SAC / TD3（Phase 9 · 07） | DQN 没有策略网络。 |
| On-policy / 高吞吐 | PPO（Phase 9 · 08） | 没有回放缓冲区；更容易扩展。 |
| 离线 RL | CQL / IQL / Decision Transformer | 保守 Q 目标，没有自举爆炸。 |
| 大离散动作空间（推荐系统） | 带动作嵌入的 DQN，或 IMPALA | 可以；细节装饰很重要。 |
| LLM RL | PPO / GRPO | 序列级，而不是步骤级；损失不同。 |

这些经验仍然适用。回放和目标网络出现在 SAC、TD3、DDPG、SAC-X、AlphaZero 的自博弈缓冲区，以及每一种离线 RL 方法中。奖励裁剪则以 PPO 中优势归一化的形式继续存在。这个架构就是蓝图。

## 发布它

保存为 `outputs/skill-dqn-trainer.md`：

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
```

## 练习

1. **简单。** 运行 `code/main.py`。绘制每个 episode 的 return 曲线。运行均值超过 -10 需要多少个 episode？
2. **中等。** 禁用目标网络（在 Bellman 目标两边都使用在线网络）。测量训练不稳定性——return 会振荡还是发散？
3. **困难。** 添加 Double DQN：用在线网络选择 `argmax a'`，用目标网络评估。比较在带噪声奖励的 GridWorld 上训练 1,000 个 episode 后，有无 Double DQN 时 `Q(s_0, best_a)` 相对于真实 `V*(s_0)` 的偏差。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| DQN | “Deep Q-learning” | 使用神经 Q 函数、回放缓冲区和目标网络的 Q-learning。 |
| Experience replay | “打乱的转移样本” | 每个梯度步骤从环形缓冲区均匀采样；去相关化数据。 |
| Target network | “冻结的自举” | 用于 Bellman 目标的 Q 网络周期性副本；稳定训练。 |
| Deadly triad | “RL 为什么会发散” | 函数近似 + 自举 + 离策略 = 没有收敛保证。 |
| Double DQN | “最大化偏差的修复” | 在线网络选择动作，目标网络评估它。 |
| Dueling DQN | “V 和 A 头” | 分解 Q = V + A - mean(A)；输出相同，梯度流更好。 |
| Rainbow | “所有技巧” | DDQN + PER + dueling + n-step + noisy + distributional 合在一起。 |
| PER | “Prioritized Replay” | 按 TD 误差幅度成比例采样转移。 |

## 延伸阅读

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — 2013 年 NeurIPS workshop 论文，开启了深度 RL。
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Nature 论文，49 个游戏的 DQN。
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — DDQN。
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) — dueling DQN。
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) — 叠加技巧论文。
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) — 清晰的现代表述。
- [Sutton & Barto (2018). Ch. 9 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) — 教科书中对“致命三元组”（函数近似 + 自举 + 离策略）的处理，DQN 的目标网络和回放缓冲区正是为驯服它而设计的。
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/) — 用于消融研究的参考单文件 DQN；适合与本课从零实现版本一起阅读。
