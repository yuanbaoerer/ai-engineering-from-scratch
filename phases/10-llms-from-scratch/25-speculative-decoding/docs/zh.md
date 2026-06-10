# 投机解码 (Speculative Decoding) 与 EAGLE

> 一个前沿的大语言模型 (LLM) 生成一个词元 (token) 需要对数十亿参数进行一次完整的前向传播。而这一次前向传播的资源利用率其实极低：大多数时候，一个小得多的模型就能正确猜出接下来的 3-5 个词元，大模型只需要*验证*这个猜测是否正确。如果猜对了，你就用一次前向传播的代价换回了 5 个词元。投机解码 (Leviathan et al. 2023) 精确地实现了这一点，而 EAGLE-3 (2025) 将接受率推到了每次验证约 4.5 个词元——在输出分布完全匹配的前提下实现了 4-5 倍的加速。

**类型:** 实战构建
**语言:** Python (使用 numpy)
**前置知识:** 阶段 10 第 12 课 (推理优化), 阶段 10 第 04 课 (预训练 Mini-GPT)
**时长:** ~75 分钟

## 问题所在

一个 70B 级别模型在 H100 上的解码吞吐量通常为每秒 40-80 个词元。每个词元都需要一次完整的前向传播，从 HBM 中读取所有模型权重。你无法在不改变输出的前提下让模型变小。你无法在内存限制之外增加批次大小。你陷入了困境——除非你能让模型每次前向传播输出不止一个词元。

自回归生成看起来本质上是串行的：`x_{t+1} = sample(p(· | x_{1:t}))`。但这里存在一个并发机会。如果你有一个廉价的预测器说"接下来的 4 个词元很可能是 [a, b, c, d]"，你就可以在**大模型的一次前向传播**中验证这 5 个位置，并接受最长的匹配前缀。

Leviathan, Kalai, Matias (2023, "Fast Inference from Transformers via Speculative Decoding") 通过一个巧妙的接受/拒绝规则精确地实现了这一点，该规则保留了目标模型的采样分布。相同的输出分布，2-4 倍的速度提升。

## 核心概念

### 双模型架构

- **目标模型 (Target model)** `M_p`：你想要从中采样的那个大、慢、高质量的模型。分布：`p(x)`。
- **草稿模型 (Draft model)** `M_q`：一个小、快、低质量的模型。分布：`q(x)`。通常小 5-30 倍。

每一步：

1. 草稿模型自回归地提出 `K` 个词元：`x_1, x_2, ..., x_K ~ q`。
2. 目标模型并行地对所有 `K+1` 个位置运行**一次**前向传播，为每个被提议的词元生成 `p(x_k)`。
3. 通过下面修改后的拒绝采样规则从左到右逐个接受/拒绝词元。接受最长的匹配前缀。
4. 如果有任何词元被拒绝，从修正后的分布中采样替换词元并停止。否则从 `p(· | x_1...x_K)` 中采样一个额外词元。

如果草稿与目标完全匹配，你每次目标前向传播就能得到 K+1 个词元。如果草稿在位置 1 就错了，你只能得到 1 个词元。

### 精确性规则

投机解码在**分布上与从 p 采样可证明等价**。拒绝规则如下：

```
For each drafted token x_t:
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        accept x_t
    else:
        sample replacement from residual: (p - q)+ / ||(p - q)+||_1
        stop
```

其中 `(p - q)+` 表示逐点差值的正部。当草稿和目标一致时 (`p ≈ q`)，接受率接近 1。当它们不一致时，残差分布的构造保证了整体样本仍然精确地服从 `p`。

**贪心情况 (Greedy case)。** 对于 temperature=0 的采样，只需检查 `argmax(p) == x_t`。如果成立则接受；否则输出 `argmax(p)` 并停止。

### 预期加速比

如果草稿模型的词元级接受率为 `α`，每次目标前向传播产生的预期词元数为：

```
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = 草稿长度, α ∈ [0, 1]
```

当 `α = 0.8, K = 4` 时：`(1 - 0.8^5)/(1 - 0.8) = 3.36` 个词元每次前向传播。一次目标前向传播的成本大致为 `cost_q * K + cost_p`（K 次草稿步加一次目标验证）。如果 `cost_p >> cost_q * K`，吞吐量上的加速比就是 `3.36× / 1 = 3.36×`。

唯一真正重要的参数是 `α`，它完全取决于草稿模型与目标模型的对齐程度。一个好的草稿模型就是一切。

### 训练草稿模型：蒸馏 (Distillation)

一个随机的小模型作为草稿模型效果很差。标准做法是从目标模型蒸馏：

1. 选择一个小型架构（70B 目标对应约 1B，7B 目标对应约 500M）。
2. 在大型文本语料库上运行目标模型；存储其下一个词元的分布。
3. 用 KL 散度训练草稿模型去拟合目标模型的分布（而不是拟合真实词元）。

结果：`α` 在代码上通常为 0.6-0.8，在自然语言对话上为 0.7-0.85。生产中可实现 2-3 倍加速。

### EAGLE：树状草稿 + 特征复用

Li, Wei, Zhang, Zhang (2024, "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty") 观察到标准投机解码中的两个低效之处：

1. 草稿模型执行 K 个串行步骤，每一步都是全栈计算。但草稿模型可以复用目标模型的特征（隐藏状态）——目标模型在最近的验证中已经计算了丰富的表示，而草稿模型却在从头重新推导。
2. 草稿模型输出一条线性链。如果草稿模型能输出一棵*树*状候选（每个节点多个猜测），目标模型的一次前向传播就可以通过树注意力掩码并行验证多条候选路径，并挑选最长的被接受分支。

EAGLE-1 的改动：
- 草稿输入 = 目标模型在位置 t 的最终隐藏状态，而非原始词元。
- 草稿架构 = 1 个 Transformer 解码器层（而非独立的小模型）。
- 输出 = 深度 4-6、每层 K = 4-8 个候选的树。

EAGLE-2 (2024) 增加了动态树拓扑：树在草稿模型不确定的地方变宽，在自信的地方保持窄。在不增加验证成本的前提下提高了有效接受率 `α_effective`。

EAGLE-3 (Li et al. 2025, "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test") 移除了固定的顶层特征依赖，并用一种新的"测试时模拟"损失训练草稿模型——草稿模型在训练时匹配目标模型在测试时的分布，而非强制教学 (teacher-forced) 训练分布。接受率从 0.75 (EAGLE-2) 提升到 0.82 (EAGLE-3)，每次验证的平均词元数从 3.0 提升到 4.5。

### 树注意力验证

当草稿模型输出一棵树时，目标模型使用**树注意力掩码 (tree attention mask)** 在一次前向传播中验证它——这是一种编码了树拓扑结构的因果掩码，而非纯线性链。每个词元只关注其在树中的祖先节点。验证过程仍然是一次前向传播、一次矩阵乘法；拓扑掩码只增加少量额外的 KV 缓存条目。

```
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

如果 `a, b` 是竞争性的第一个词元候选，`c, d, e, f` 是第二个词元候选，全部六个位置都在一次前向传播中完成验证。输出是任意被接受路径上的最长前缀。

### 何时有效，何时无效

**有效的情况：**
- 对话 / 补全任务中可预测的文本（代码、常见英语、结构化输出）。`α` 很高。
- 解码阶段 GPU 计算未被充分利用的场景（内存受限阶段）。树状草稿利用了可用的 FLOPs。

**无效 / 无收益的情况：**
- 高度随机的输出（高温下的创意写作）。`α` 下降到接近 `1/|vocab|`。
- 并发度极高的批次服务——批处理本身已经填满了 FLOPs，留给树验证的空间很小。
- 非常小的目标模型，草稿模型无法比它小多少。

生产环境中通常报告：对话任务 2-3 倍墙钟时间加速，代码生成 3-5 倍，创意写作接近零。

## 动手实现

`code/main.py`：

- 一个参考实现 `speculative_decode(target, draft, prompt, K, temperature)`，实现精确的拒绝规则，并验证它保留了目标模型的分布（与纯目标采样相比，经验 KL < 0.01）。
- 一个 EAGLE 风格的树状草稿器，构建深度为 K、采用 top-p 分支的树。
- 一个树注意力掩码构建器，为验证器生成正确的因果模式。
- 一个接受率测试框架，在一个微型语言模型上运行两者（从一个 GPT-2-medium 目标模型蒸馏一个 GPT-2-small）。

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """One round of speculative decoding. Returns list of accepted tokens."""
    # 1. Draft K tokens
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. Target computes p at every drafted position + 1 extra
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. Accept/reject left-to-right
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. All K accepted → sample bonus token from target
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## 如何使用

- **vLLM** 和 **SGLang** 原生支持投机解码。参数：`--speculative_model`, `--num_speculative_tokens`。通过 `--spec_decoding_algorithm eagle` 参数支持 EAGLE-2/3。
- **NVIDIA TensorRT-LLM** 原生支持 Medusa 和 EAGLE 树。
- **参考草稿模型**：`Qwen/Qwen3-0.6B-spec`（为 Qwen3-32B 提供草稿）, `meta-llama/Llama-3.2-1B-Instruct-spec`（为 70B 提供草稿）。
- **Medusa 头 (Medusa heads)** (Cai et al. 2024, "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads")：不单独使用草稿模型，而是在目标模型本身上添加 K 个并行预测头。部署更简单，接受率略低于 EAGLE。

## 交付成果

本课产出 `outputs/skill-speculative-tuning.md` —— 一项技能，用于分析目标模型的工作负载并选择：草稿模型、K（草稿长度）、树宽度、温度，以及何时回退到普通解码。

## 练习题

1. 实现精确的拒绝规则并进行经验验证。通过 `speculative_decode` 和普通目标采样各运行 10K 个样本；计算两种输出分布之间的总变差距离 (TV distance)。应 < 0.01。

2. 计算加速公式。给定固定的 `α` 和 `K`，绘制每次目标前向传播的预期词元数。为 α ∈ {0.5, 0.7, 0.9} 找到最优的 K。

3. 训练一个微型草稿模型。以 124M GPT-2 为目标模型，在 100M 词元上用 KL 损失蒸馏一个 30M GPT-2 草稿模型。在留出文本上测量 `α`。预期：0.6-0.7。

4. 实现 EAGLE 风格的树状草稿。不输出链式结构，而是让草稿模型在每一层输出 top-3 分支。构建树注意力掩码。验证目标模型接受最长的正确分支。

5. 测量失效模式。在 temperature=1.5（高随机性）下运行投机解码。展示 `α` 如何崩溃，以及由于草稿开销导致算法比普通解码更慢。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|----------|----------|
| 目标模型 (Target model) | "大模型" | 你想要从中采样的那个慢、高质量的模型（p 分布） |
| 草稿模型 (Draft model) | "投机者" | 小、快的预测器（q 分布）；通常小 5-30 倍 |
| K / 草稿长度 | "前瞻" | 每次验证步推测的词元数量 |
| α / 接受率 | "命中率" | 草稿提议被接受的逐词元概率 |
| 精确拒绝规则 | "接受测试" | `r < p/q` 的比较，保留目标模型的分布 |
| 残差分布 | "修正后的 p-q" | `(p - q)+ / ||(p - q)+||_1`，拒绝时从中采样的分布 |
| 树状草稿 | "分支推测" | 草稿输出一棵候选树，通过树结构注意力掩码一次验证 |
| 树注意力掩码 | "拓扑掩码" | 编码树拓扑的因果掩码，每个节点只关注其祖先 |
| Medusa 头 | "并行头" | 在目标模型本身上添加 K 个额外预测头；无需单独草稿模型 |
| EAGLE 特征复用 | "隐藏状态草稿" | 草稿输入是目标模型的最后一个隐藏状态，而非原始词元，从而缩小草稿模型 |
| 测试时模拟损失 | "EAGLE-3 训练" | 在匹配目标模型测试时分布的输出上训练草稿，而非强制教学 |

## 延伸阅读

- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — 精确拒绝规则与理论加速分析
- [Chen, Borgeaud, Irving et al., 2023 — "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) — DeepMind 的同期投机采样论文
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) — 草稿模型的并行头替代方案
- [Li, Wei, Zhang, Zhang, 2024 — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077) — 特征复用与树状草稿
- [Li et al., 2024 — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858) — 动态树拓扑
- [Li et al., 2025 — "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840) — 训练时与测试时分布匹配
- [Fu, Haotian, Peng et al., 2024 — "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057) — Jacobi/前瞻解码，一种无需投机器的替代方案
