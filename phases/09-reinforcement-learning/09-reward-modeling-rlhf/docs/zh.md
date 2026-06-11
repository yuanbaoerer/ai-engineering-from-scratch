# 奖励建模与 RLHF

> 人类无法为“好的助手回复”手写一个奖励函数，但他们可以比较两个回复并选出更好的那个。把奖励模型拟合到这些比较上，然后让语言模型围绕它进行强化学习。Christiano 2017。InstructGPT 2022。这就是把 GPT-3 变成 ChatGPT 的配方。到 2026 年，它大多正在被 DPO 取代——但心智模型仍然适用。

**类型：** 构建
**语言：** Python
**先修要求：** 第 5 阶段 · 05（情感分析），第 9 阶段 · 08（PPO）
**时间：** 约 45 分钟

## 问题

你用下一词元预测（next-token prediction）目标训练了一个语言模型。它能写出语法正确的英语。它也会撒谎、啰嗦，并且该拒绝时不拒绝。你无法通过更多预训练来修复这一点——网络文本正是问题所在，而不是解药。

你想要一个*标量奖励*，用来表示“对于指令 X，回复 A 比回复 B 更好”。手写这样的奖励函数是不可能的。“有帮助性”（helpfulness）不是词元上的闭式表达式。但人类可以比较两个输出并标记偏好。这种数据可以低成本地大规模收集。

RLHF（Christiano et al. 2017；Ouyang et al. 2022）把偏好转换成奖励模型，然后通过 PPO 针对该奖励优化语言模型。三步：SFT → RM → PPO。这是在 2023–2025 年交付 ChatGPT、Claude、Gemini 以及所有其他对齐 LLM 的配方。

到 2026 年，PPO 步骤大多被 DPO（第 10 阶段 · 08）取代，因为它更便宜，并且在对齐微调上效果几乎一样好。但*奖励模型*这一部分仍然支撑着每一个 Best-of-N 采样器、每一条从可验证奖励进行 RL 的流水线，以及每一个使用过程奖励模型的推理模型。理解 RLHF，就理解了整个对齐技术栈。

## 概念

![三阶段 RLHF：SFT、在成对偏好上训练 RM、带 KL 惩罚的 PPO](../assets/rlhf.svg)

**阶段 1：监督微调（Supervised Fine-Tuning，SFT）。** 从一个预训练基础模型开始。在人类编写的目标行为示范上进行微调（遵循指令的回复、有帮助的回答等）。结果：一个 `π_SFT` 模型，它*偏向良好行为*，但仍然拥有无界动作空间。

**阶段 2：奖励模型训练。**

- 收集针对提示词 `x` 的回复对 `(y_+, y_-)`，由人类标注为“y_+ 优于 y_-”。
- 训练奖励模型 `R_φ(x, y)`，让它给 `y_+` 分配更高分数。
- 损失：**Bradley-Terry 成对 logistic**：

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ 是 sigmoid。奖励差异意味着偏好的对数赔率（log-odds）。BT 自 1952 年（Bradley-Terry）以来就是标准方法，也是现代 RLHF 中的主流选择。

- `R_φ` 通常从 SFT 模型初始化，并在顶层加一个标量头。相同的 Transformer 主干；一个线性层输出奖励。

**阶段 3：带 KL 惩罚、针对 RM 的 PPO。**

- 从 `π_SFT` 初始化可训练策略 `π_θ`。保留一个冻结的*参考*模型 `π_ref = π_SFT`。
- 回复 `y` 结束时的奖励：

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  KL 惩罚防止 `π_θ` 任意偏离 `π_SFT`——它是一个*正则化器*，而不是硬性的信任域。`β` 通常为 `0.01`-`0.05`。
- 使用该奖励运行 PPO（第 08 课）。优势是在词元级轨迹上计算的，但 RM 只对完整回复打分。

**为什么需要 KL？** 没有它，PPO 会乐于找到奖励黑客策略——RM 只在分布内补全上训练过。一个分布外回复可能比任何人类编写的回复得分都高。KL 让 `π_θ` 留在 RM 训练过的流形附近。它是 RLHF 中最重要的单个旋钮。

**2026 年状态：**

- **DPO**（Rafailov 2023）：闭式代数把阶段 2+3 折叠成一个基于偏好数据的监督损失。没有 RM，没有 PPO。只需一小部分计算量，就能在对齐基准上达到同等质量。第 10 阶段 · 08 会讲到。
- **GRPO**（DeepSeek 2024–2025）：PPO 的变体，用组相对基线代替评论家（critic），奖励来自*验证器*（代码能运行 / 数学答案匹配），而不是人类训练的 RM。它是推理模型的主流方法。第 9 阶段 · 12 会讲到。
- **过程奖励模型（Process Reward Models，PRMs）：** 给部分解答（每个推理步骤）打分，用于 RLHF 和 GRPO 的推理变体。
- **Constitutional AI / RLAIF：** 使用对齐的 LLM 生成偏好，而不是使用人类。扩展偏好预算。

## 构建它

本课使用微型合成“提示词”和“回复”，它们表示为字符串。RM 是一个基于词袋表示的线性打分器。没有真正的 LLM——重要的是流水线的*形状*，不是规模。见 `code/main.py`。

### 步骤 1：合成偏好数据

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

在真实 RLHF 中，这会被人类标注员替代。形状——`(prompt, preferred_response, rejected_response)`——完全相同。

### 步骤 2：Bradley-Terry 奖励模型

线性分数：`R(x, y) = w · bag(y)`。训练时最小化 BT 成对 log 损失：

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

几百次更新后，`w` 会给好词词元分配正权重，给坏词分配负权重。

### 步骤 3：RM 之上的类 PPO 策略

我们的玩具策略从词表中生成单个词元。我们在 RM 下给该词元打分，计算 `log π_θ(token | prompt)`，加入到参考策略的 KL 惩罚，并应用裁剪 PPO 代理目标。

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo-style update on theta, treating reward as the return
    ...
```

### 步骤 4：监控 KL

每次更新都跟踪平均 `KL(π_θ || π_ref)`。如果它爬升到 `~5-10` 以上，说明策略已经远离 `π_SFT`——`β` 过低，或者奖励黑客正在开始。这是真实 RLHF 中最重要的诊断指标。

### 步骤 5：使用 TRL 的生产配方

一旦你理解了玩具流水线，下面就是真实库用户会写出的同一个循环。Hugging Face 的 [TRL](https://huggingface.co/docs/trl) 是参考实现——阶段 2 使用 `RewardTrainer`，阶段 3 使用 `PPOTrainer`（内置到参考模型的 KL）。

```python
# Stage 2: reward model from pairwise preferences
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# dataset rows: {"prompt", "chosen", "rejected"} — Bradley-Terry format
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# Stage 3: PPO against the RM with KL penalty to the SFT reference
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # frozen

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats includes: mean_kl, clip_frac, value_loss — the three PPO diagnostics
```

库会替你完成三件事。`adap_kl_ctrl=True` 实现自适应 β 调度：如果观测到的 KL 超过 `target_kl`，β 翻倍；如果低于一半，β 减半。参考模型按惯例被冻结——你绝不能意外让它与 `policy` 共享参数。价值头与策略位于同一个主干上（`AutoModelForCausalLMWithValueHead` 会附加一个标量 MLP 头），这就是为什么 TRL 会分别报告 `policy/kl` 和 `value/loss`。

## 陷阱

- **过度优化 / 奖励黑客。** RM 并不完美；`π_θ` 会找到得分很高但质量很差的对抗性补全。症状：奖励无限上升，而人类评估分数停滞或下降。修复：提前停止，提高 `β`，拓宽 RM 训练数据。
- **长度黑客。** 在有帮助回复上训练的 RM 往往会隐式奖励长度。策略学会给回复填充内容。补救：长度归一化奖励，或使用带长度感知 RM 的 RLAIF。
- **RM 太小。** RM 至少需要和策略一样大。微型 RM 无法忠实地给策略输出打分。
- **KL 调参。** β 太低 → 漂移和奖励黑客。β 太高 → 策略几乎不变。标准技巧是使用一个以每步固定 KL 为目标的*自适应* β。
- **偏好数据噪声。** 约 30% 的人类标签是有噪声或模棱两可的。可通过在一致性过滤后的数据上训练 RM，或在 BT 中使用温度来校准。
- **离策略问题。** 第一轮之后，PPO 数据会略微离策略。像第 08 课那样监控裁剪比例。

## 使用它

2026 年的 RLHF 是分层的：

| 层级 | 目标 | 方法 |
|-------|--------|--------|
| 指令遵循、有帮助性、无害性 | 对齐 | DPO（第 10 阶段 · 08）优先于 RLHF-PPO。 |
| 推理正确性（数学、代码） | 能力 | 使用验证器奖励的 GRPO（第 9 阶段 · 12）。 |
| 长时程多步骤任务 | 智能体式 | PPO / GRPO，配合步骤上的过程奖励模型。 |
| 安全 / 拒绝行为 | 安全 | RLHF-PPO，配合单独的安全 RM，或 Constitutional AI。 |
| 推理时 Best-of-N | 快速对齐 | 解码时使用 RM；无需训练策略。 |
| 奖励蒸馏 | 推理计算 | 在冻结 LM 之上训练一个小型“奖励头”。 |

RLHF 是 2022–2024 年的*核心*方法。到 2026 年，生产对齐流水线以 DPO 优先，只有在 RM 密集型或安全关键步骤中才使用 PPO。

## 交付它

保存为 `outputs/skill-rlhf-architect.md`：

```markdown
---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

给定一个基础 LM、一个目标行为（对齐 / 推理 / 拒绝 / 智能体）、以及一份偏好或验证器预算，输出：

1. 阶段。SFT？RM？DPO？GRPO？并给出理由。
2. 偏好或验证器来源。人类、AI 反馈、基于规则、单元测试通过，或奖励蒸馏。
3. KL 策略。固定 β、自适应 β，或 DPO（隐式 KL）。
4. 诊断指标。平均 KL、奖励稳定性、过度优化防护（保留人类评估）。
5. 安全门。红队集合、拒绝率、与有帮助性 RM 分离的安全 RM。

如果没有 KL 监控，就拒绝交付 RLHF-PPO。如果 RM 小于目标策略，就拒绝使用它。拒绝只基于长度的奖励。标记任何没有留出盲测人类评估集的流水线，说明其缺乏过度优化保护。
```

## 练习

1. **简单。** 在 `code/main.py` 中用 500 个合成偏好对训练 Bradley-Terry 奖励模型。在保留的 100 个偏好对上测量成对准确率。应超过 90%。
2. **中等。** 用 `β ∈ {0.0, 0.1, 1.0}` 运行玩具 PPO-RLHF 循环。对每个值，绘制更新过程中 RM 分数与到参考策略 KL 的关系。哪些运行发生了奖励黑客？
3. **困难。** 在同一偏好数据上实现 DPO（闭式偏好似然损失），并与 RLHF-PPO 流水线在所用计算量和最终达到的 RM 分数上进行比较。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| RLHF | “对齐 RL” | 三阶段 SFT + RM + PPO 流水线（Christiano 2017，Ouyang 2022）。 |
| 奖励模型（Reward Model，RM） | “打分网络” | 通过 Bradley-Terry 拟合成对偏好得到的已学习标量函数。 |
| Bradley-Terry | “成对 logistic 损失” | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`；标准 RM 目标。 |
| KL 惩罚 | “留在参考模型附近” | 奖励中的 `β · KL(π_θ \|\| π_ref)`；反奖励黑客的正则化器。 |
| 奖励黑客 | “古德哈特定律” | 策略利用 RM 缺陷；症状：奖励上升，人类评估持平。 |
| RLAIF | “AI 标注偏好” | 标签来自另一个 LM 而非人类的 RLHF。 |
| PRM | “过程奖励模型” | 给部分推理步骤打分；用于推理流水线。 |
| Constitutional AI | “Anthropic 的方法” | 由显式规则引导的 AI 生成偏好。 |

## 延伸阅读

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — 开创 RLHF 的论文。
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — ChatGPT 背后的配方。
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — 更早用于摘要的 RLHF。
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO；2026 年后 RLHF 时代的默认方法。
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — RLAIF 与自我批评循环。
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) — HH 论文。
- [Hugging Face TRL library](https://huggingface.co/docs/trl) — 生产级 `RewardTrainer` 和 `PPOTrainer`。阅读 trainer 源码，了解自适应 KL 和价值头细节。
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) by Lambert, Castricato, von Werra, Havrilla — 三阶段流水线的经典图解教程。
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — 该库；`examples/` 中有面向 Llama、Mistral 和 Qwen 的端到端 RLHF 脚本。
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) — 奖励假说视角；思考奖励黑客的必备先修内容。
