# 游戏强化学习 —— AlphaZero、MuZero 与 LLM 推理时代

> 1992 年：TD-Gammon 用纯 TD 在双陆棋中击败人类冠军。2016 年：AlphaGo 击败李世石。2017 年：AlphaZero 从零开始统治国际象棋、日本将棋和围棋。2024 年：DeepSeek-R1 证明了同一套配方在推理任务上同样有效，只是用 GRPO 替代 PPO。游戏是推动本阶段每一次突破的基准。

**类型：** 构建
**语言：** Python
**先修要求：** 第 9 阶段 · 05（DQN）、第 9 阶段 · 08（PPO）、第 9 阶段 · 09（RLHF）、第 9 阶段 · 10（MARL）
**时间：** 约 120 分钟

## 问题

游戏拥有强化学习想要的一切。清晰的奖励（胜/负）。无限的回合（自博弈重置）。完美的仿真（游戏本身就是模拟器）。离散或小规模连续动作空间。迫使智能体具备对抗鲁棒性的多智能体结构。

而且，游戏正是每一次重大强化学习突破的试验场。TD-Gammon（双陆棋，1992）。Atari-DQN（2013）。AlphaGo（2016）。AlphaZero（2017）。OpenAI Five（Dota 2，2019）。AlphaStar（星际争霸 II，2019）。MuZero（学习到的模型，2019）。AlphaTensor（矩阵乘法，2022）。AlphaDev（排序算法，2023）。DeepSeek-R1（数学推理，2025）—— 最新一次证明游戏强化学习技术可以用于文本的案例。

这个收官课程通过一个统一视角梳理三个里程碑架构——AlphaZero、MuZero 和 GRPO：**自博弈 + 搜索 + 策略改进**。每一个都泛化了前一个；尤其是 GRPO，它把 AlphaZero 的配方应用到 LLM 推理上，其中 token 是动作，数学验证是胜利信号。

## 概念

![AlphaZero ↔ MuZero ↔ GRPO: same loop, different environments](../assets/rl-games.svg)

**统一循环。**

```
while True:
    trajectory = self_play(current_policy, search)     # play game against self
    policy_target = search.improved_policy(trajectory) # search improves raw policy
    policy_net.update(policy_target, value_target)     # supervised on search output
```

**AlphaZero（2017）。** Silver 等人。给定一个规则已知的游戏（国际象棋、日本将棋、围棋）：

- 策略-价值网络（policy-value network）：一个塔 `f_θ(s) → (p, v)`。`p` 是合法走法上的先验分布。`v` 是期望的游戏结果。
- 蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS）：在每一步，展开一棵可能后续走法的树。使用 `(p, v)` 作为先验 + 自举估计。用 UCB（PUCT）选择节点：`a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`。
- 自博弈（self-play）：让智能体对战自身。在第 `t` 步，MCTS 的访问分布 `π_t` 成为策略训练目标。
- 损失：`L = (v - z)² - π · log p + c · ||θ||²`。`z` 是游戏结果（+1 / 0 / -1）。

零人类知识。零手工启发式。一套配方，在各自数千万局自博弈之后掌握了国际象棋、日本将棋和围棋。

**MuZero（2019）。** Schrittwieser 等人。移除了“规则必须已知”的要求。

- 不使用固定环境，而是学习一个*潜在动力学模型*（latent dynamics model）`(h, g, f)`：
  - `h(s)`：把观测编码为潜在状态。
  - `g(s_latent, a)`：预测下一个潜在状态 + 奖励。
  - `f(s_latent)`：预测策略先验 + 价值。
- MCTS 在*学习到的潜在空间*中运行。同样的搜索，同样的训练循环。
- 可用于围棋、国际象棋、日本将棋*以及* Atari —— 一个算法，不需要规则知识。

**随机 MuZero（Stochastic MuZero，2022）。** 加入随机动力学和机会节点（chance nodes）；扩展到双陆棋这类游戏。

**Muesli、Gumbel MuZero（2022-2024）。** 在样本效率和确定性搜索方面的改进。

**GRPO（2024-2025）。** DeepSeek-R1 配方。同样是 AlphaZero 形状的循环，但应用于语言模型推理：

- “游戏”：回答一道数学 / 编程 / 推理问题。“胜利”= 验证器（测试用例通过、数值答案匹配）返回 1。
- 策略：LLM。动作：token。状态：提示词 + 到目前为止的回答。
- 没有 critic（PPO 风格的 V_φ）。相反，对每个提示词，从策略中采样 `G` 个补全。计算每个补全的奖励。使用**组相对优势**（group-relative advantage）`A_i = (r_i - mean_r) / std_r` 作为 REINFORCE 风格更新的信号。
- 对参考策略加入 KL 惩罚以防止漂移（类似 RLHF）。
- 完整损失：

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

没有奖励模型，没有 critic，没有 MCTS。组相对基线取代了这三者。在推理基准上，它以一小部分计算量达到或超过 PPO-RLHF 的质量。

**完整的 R1 配方。** DeepSeek-R1（DeepSeek 2025）是一篇论文中的两个模型：

- **R1-Zero。** 从 DeepSeek-V3 基座模型开始。不做 SFT。直接应用 GRPO，并使用两个奖励组成部分：*准确性奖励*（基于规则——最终答案是否能解析成正确数字 / 代码是否通过单元测试）和*格式奖励*（补全是否把思维链包在 `<think>…</think>` 标签中）。经过数千步，平均响应长度从约 100 个 token 增长到约 10,000 个 token，数学基准分数上升到接近 o1-preview 的水平。模型从零开始学会推理。缺点是：它的思维链通常难以阅读、混用多种语言，并且缺乏风格润色。
- **R1。** 用四阶段流水线修复 R1-Zero 的可读性问题：
  1. **冷启动 SFT（Cold-start SFT）。** 收集几千条格式清晰的长 CoT 演示。用它们对基座模型做监督微调。这提供了一个可读的起点。
  2. **面向推理的 GRPO。** 使用准确性+格式奖励应用 GRPO，再加入一个*语言一致性*奖励来防止代码切换（code-switching）。
  3. **拒绝采样 + 第二轮 SFT。** 从 RL 检查点采样约 600K 条推理轨迹，只保留最终答案正确且 CoT 可读的样本，并与约 200K 条非推理 SFT 样本（写作、问答、自我认知）合并。再次微调基座模型。
  4. **全谱系 GRPO。** 再进行一轮 RL，同时覆盖推理（基于规则的奖励）和通用对齐（基于偏好的有帮助性/无害性奖励）。

结果是在开放权重下，在 AIME 和 MATH-500 上匹配 o1，并且足够小，可以进行蒸馏。同一篇论文还发布了六个蒸馏后的稠密模型（从 Qwen-1.5B 到 Llama-70B），方法是在 R1 的推理轨迹上进行 SFT —— 学生模型本身不做 RL。强 RL 教师的蒸馏，在学生模型规模上持续优于从零开始的 RL。

**为什么推理要用 GRPO 而不是 PPO。** DeepSeekMath 论文（2024 年 2 月）给出三个原因：（1）无需训练价值网络，内存减半；（2）组基线天然适合推理任务产生的稀疏终局奖励；（3）按提示词归一化使不同难度问题上的优势值可比，而 PPO 的单一 critic 做不到这一点。

**无搜索 vs 基于搜索。** 游戏已经分化：

- *长时域完全信息游戏*（围棋、国际象棋）：仍然基于搜索。AlphaZero / MuZero 占主导。
- *LLM 推理*：生产环境中还没有 MCTS；对完整 rollout 使用 GRPO，推理计算则使用 best-of-N。过程奖励模型（Process Reward Models, PRMs）暗示逐步搜索可能会被重新加入。

## 构建它

`code/main.py` 中的代码实现了**微型 GRPO**——一个带有多组样本的老虎机（bandit）。算法和 LLM 上的算法相同；只是策略和环境更简单。它教授的是*损失*和*组相对优势*，也就是 2025 年的创新点。

### 第 1 步：一个极小的验证器环境

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

在真实 GRPO 中，验证器运行单元测试或检查数学等价性。

### 第 2 步：策略：每个提示词上 K 个答案 token 的 softmax

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

等价于 LLM 在给定提示词条件下的最后一层输出。

### 第 3 步：组采样与组相对优势

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL penalty: pull theta toward reference
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

组相对优势是 2024 年 DeepSeek 的技巧。不需要 critic。“基线”是组均值，归一化使用组标准差。

### 第 4 步：与 REINFORCE 基线比较（无价值函数）

同样的设置、同样的计算量、普通 REINFORCE。GRPO 收敛更快，也更稳定。

### 第 5 步：观察熵和 KL

与 RLHF 相同的诊断指标：到参考策略的平均 KL、策略熵、随时间变化的奖励。一旦这些指标稳定，训练就完成了。

## 常见陷阱

- **通过利用验证器进行奖励黑客（reward hacking）。** GRPO 继承了 RLHF 的风险：如果验证器错误或可被利用，LLM 就会找到这个漏洞。鲁棒的验证器（多个测试用例、形式化证明）很重要。
- **组大小过小。** 组基线的方差按 `1/√G` 变化。低于 `G = 4` 时，优势信号噪声很大；标准选择是 `G = 8` 到 `64`。
- **长度偏差。** 不同长度的 LLM 补全具有不同的对数概率。按 token 数归一化，或使用序列级对数概率，或截断到最大长度。
- **纯自博弈循环。** AlphaZero 风格训练在一般和博弈中可能陷入支配循环。可通过多样化对手池（联赛对战，第 10 课）缓解。
- **搜索-策略不匹配。** AlphaZero 训练策略去模仿搜索输出。如果策略网络太小，无法表示搜索分布，训练就会停滞。
- **计算门槛。** MuZero / AlphaZero 需要海量计算。单个消融实验往往需要数百 GPU 小时。用于学习的微型演示存在（例如在四子棋上训练 AlphaZero）。
- **验证器覆盖率。** 对有 bug 的解法也能通过的单元测试会强化这个 bug。要设计能捕捉边界情况的验证器。

## 使用它

按领域划分的 2026 年游戏强化学习格局：

| 领域 | 主导方法 |
|--------|-----------------|
| 双人零和棋盘游戏（围棋、国际象棋、日本将棋） | AlphaZero / MuZero / KataGo |
| 非完美信息纸牌游戏（扑克） | CFR + 深度学习（DeepStack、Libratus、Pluribus） |
| Atari / 像素游戏 | Muesli / MuZero / IMPALA-PPO |
| 大型多人策略游戏（Dota、星际争霸） | PPO + 自博弈 + 联赛（OpenAI Five、AlphaStar） |
| LLM 数学/代码推理 | GRPO（DeepSeek-R1、Qwen-RL、开放复现） |
| LLM 对齐 | DPO / RLHF-PPO（不是 GRPO；验证器是偏好而非可验证信号） |
| 机器人 | PPO + DR（不是游戏 RL，但使用相同的策略梯度工具） |
| 组合优化问题 | AlphaZero 变体（AlphaTensor、AlphaDev） |

这套*配方*——自博弈、搜索增强的改进、策略蒸馏——横跨文本、像素和物理控制。GRPO 是最年轻的实例；还会有更多实例出现。

## 交付它

保存为 `outputs/skill-game-rl-designer.md`：

```markdown
---
name: game-rl-designer
description: Design a game-RL or reasoning-RL training pipeline (AlphaZero / MuZero / GRPO) for a given domain.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Given a target (perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial), output:

1. Environment fit. Known rules? Markov? Stochastic? Multi-agent? Informs AlphaZero vs MuZero vs GRPO.
2. Search strategy. MCTS (PUCT with learned prior), Gumbel-sampled, best-of-N, or none.
3. Self-play plan. Symmetric self-play / league / offline data / verifier-generated.
4. Target signal. Game outcome / verifier reward / preference / learned model. Include robustness plan.
5. Diagnostics. Win rate vs baseline, ELO curve, verifier pass rate, KL to reference.

Refuse AlphaZero on imperfect-info games (route to CFR). Refuse GRPO without a trusted verifier. Refuse any game-RL pipeline without a fixed baseline opponent set (self-play ELO is uncalibrated otherwise).
```

## 练习

1. **简单。** 在 `code/main.py` 中实现 GRPO bandit。训练 2 个提示词 × 每个 4 个答案 token。使用 `G=8` 在 < 1,000 次更新内收敛。
2. **中等。** 接入 PPO（裁剪版）和 vanilla REINFORCE。在同一个 bandit 上比较它们与 GRPO 的样本效率和奖励方差。
3. **困难。** 扩展到长度为 2 的“推理链”：智能体发出两个 token，验证器对这个 token 对给出奖励。衡量 GRPO 如何处理两步序列中的信用分配。（提示：按*完整序列*计算组优势，并传播到两个 token 位置。）

## 关键术语

| 术语 | 人们常说 | 它真正的含义 |
|------|-----------------|-----------------------|
| MCTS | “带学习网络的树搜索” | 蒙特卡洛树搜索（Monte Carlo Tree Search）；使用学习到的 `(p, v)` 先验进行 UCB1/PUCT 选择。 |
| AlphaZero | “自博弈 + MCTS” | 训练策略-价值网络去匹配 MCTS 访问次数和游戏结果。 |
| MuZero | “学习模型版 AlphaZero” | 相同循环，但通过学习到的动力学在潜在空间中进行。 |
| GRPO | “无 critic 的 PPO” | 组相对策略优化（Group Relative Policy Optimization）；带组均值基线 + KL 的 REINFORCE。 |
| PUCT | “AlphaZero 的 UCB” | `Q + c · p · √N / (1 + N_a)` —— 在价值估计和先验之间取得平衡。 |
| Self-play | “智能体对战过去的自己” | 零和游戏的标准方法；对称训练信号。 |
| League play | “基于种群的自博弈” | 从过去智能体、当前智能体和 exploiters 中采样对手。 |
| Verifier reward | “可验证 RL” | 奖励来自确定性检查器（测试通过、答案匹配）。 |
| Process reward | “PRM” | 对每个推理步骤打分，而不仅是最终答案。 |

## 延伸阅读

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270).
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404).
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4).
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z).
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) —— 引入 GRPO 和组相对基线的论文。
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) —— 完整的四阶段 R1 配方，以及 R1-Zero 消融实验。
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) —— 大规模 CFR + 深度学习。
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) —— 开启这一切的论文。
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) —— 使用自定义奖励函数应用 GRPO 的生产级参考。
- [Qwen Team (2024). Qwen2.5-Math — GRPO replication](https://github.com/QwenLM/Qwen2.5-Math) —— 在多个规模上开放复现 R1 配方。
- [Sutton & Barto (2018). Ch. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) —— 关于自博弈、搜索和“设计奖励”的教材框架；R1 在 LLM 规模上实例化了这些思想。
