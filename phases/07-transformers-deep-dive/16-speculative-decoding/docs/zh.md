# 推测解码 — 草稿、验证、重复

> 自回归解码是串行的。每个 token 等待前一个。推测解码打破链条：廉价模型草拟 N 个 token，昂贵模型一次前向传播验证所有 N 个。当草稿正确时，你为 N 次生成付一次大前向传播。

**类型:** 构建
**语言:** Python
**前置知识:** 第 7 阶段 · 07（GPT 因果 LM）、第 7 阶段 · 12（KV 缓存和 Flash Attention）
**时间:** 约 60 分钟

## 问题所在

70B LLM 采样一个 token 在 H100 上约 30 ms。3B 草稿模型约 3 ms。如果我们让 3B 草拟 5 个 token，然后运行 70B *一次*验证全部 5 个，总计 `5×3 + 30 = 45 ms` 获得最多 5 个接受 token——vs `5×30 = 150 ms` 的直线生成。这就是完整的推测解码卖点：用少量额外 GPU 内存（草稿模型）换取 2-4 倍更低的解码延迟。

技巧必须保持分布。Leviathan et al. (2023) 和 Chen et al. 同时引入的推测采样保证输出序列与大模型自己产生的**同分布**。无质量权衡。只是更快。

四族草稿-验证器对主导 2026 年推理：

1. **普通推测 (Leviathan 2023)。** 独立草稿模型（如 Llama 3 1B）+ 验证器（如 Llama 3 70B）。
2. **Medusa (Cai 2024)。** 验证器上的多个解码头并行预测位置 `t+1..t+k`。无独立草稿模型。
3. **EAGLE 家族 (Li 2024, 2025)。** 重用验证器隐藏状态的轻量草稿；比普通更高的接受率；典型 3-4 倍。
4. **前向解码 (Fu 2024)。** Jacobi 迭代；完全不需要草稿模型。自推测。小众但无依赖。

2026 年每个生产推理栈默认发布推测解码。vLLM、TensorRT-LLM、SGLang 和 llama.cpp 都至少支持普通 + EAGLE-2。

## 核心概念

### 核心算法

给定验证器 `M_q` 和更便宜的草稿 `M_p`：

1. 设 `x_1..x_k` 为已解码的前缀。
2. **草拟**：用 `M_p` 自回归提议 `d_{k+1}, d_{k+2}, ..., d_{k+N}`，草稿概率 `p_1..p_N`。
3. **并行验证**：对 `x_1..x_k, d_{k+1}, ..., d_{k+N}` 运行 `M_q` 一次，获得位置 `k+1..k+N+1` 的验证器概率 `q_1..q_{N+1}`。
4. **从左到右接受/拒绝每个草稿 token**：对每个 `i`，以概率 `min(1, q_i(d_i) / p_i(d_i))` 接受。
5. 在位置 `j` 首次拒绝时：从归一化的"残差"分布 `(q_j - p_j)_+` 中采样 `t_j`。`j` 之后的所有草稿被丢弃。
6. 接受全部 `N` 个时：从 `q_{N+1}` 多采样一个额外 token `t_{N+1}`（免费奖励 token）。

残差分布技巧是保持输出精确分布为 `M_q` 自己从头采样时的数学洞见。

### 什么决定加速

设 `α` = 每草稿 token 的期望接受率。设 `c` = 草稿-验证器成本比。每步：

- 朴素生成每 token 做 1 次大模型调用。
- 推测每 `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` 个 token 做 1 次大模型调用，当 `α` 高时。

`α = 0.75` 和 `N = 5` 的典型经验法则：大模型调用减少 3 倍。草稿成本便宜 5 倍。总墙钟下降约 2.5 倍。

**α 取决于：**

- 草稿近似验证器的程度。同家族/同训练数据显著提升 α。
- 解码策略。贪心草稿对贪心验证器：高 α。温度采样：更难匹配；接受率下降。
- 任务类型。代码和结构化输出接受更多（可预测）；自由形式创意写作接受更少。

### Medusa——无草稿模型的草拟

Medusa 用验证器上的额外输出头替换草稿模型。在位置 `t`：

```
共享主干 → 隐藏 h_t
    ├── head_0: 预测 t+1 的 token  (标准 LM 头)
    ├── head_1: 预测 t+2 的 token
    ├── head_2: 预测 t+3 的 token
    ├── head_3: 预测 t+4 的 token
```

每个头输出自己的 logits。推理时从每个头采样得到候选序列，然后用树注意力方案一次前向传播验证，同时考虑所有候选延续。

优点：无第二个模型。缺点：增加可训练参数；需要监督微调阶段（约 1B token）；接受率比好的草稿的普通推测略低。

### EAGLE——通过重用隐藏状态获得更好草稿

EAGLE-1/2/3 (Li et al., 2024-2025) 将草稿模型做成微型 Transformer（通常 1 层），接收验证器的最后层隐藏状态。因为草稿看到验证器的特征表示，其预测与验证器的输出分布强相关。接受率从约 0.6（普通）上升到 0.85+。

EAGLE-3 (2025) 添加了候选延续上的树搜索。vLLM 和 SGLang 将 EAGLE-2/3 作为 Llama 3/4 和 Qwen 3 的默认推测路径发布。

### KV 缓存之舞

验证一次前向传播将 `N` 个草稿 token 送入验证器。这将验证器的 KV 缓存扩展 `N` 个条目。如果一些草稿被拒绝，你必须将缓存回滚到接受的前缀长度。

生产实现（vLLM 的 `--speculative-model`、TensorRT-LLM 的 LookaheadDecoder）用临时 KV 缓冲区处理。先写入，接受后提交。概念不难，但细节繁琐。

## 动手实现

参见 `code/main.py`。我们实现核心推测采样算法（拒绝步骤 + 残差分布），包含：

- 一个"大模型"，对手工编码分布的确定性 softmax（这样我们可以解析地验证接受数学）。
- 一个"草稿模型"，对大模型的扰动。
- 一个接受/拒绝循环，产生与直接采样相同的边际分布。

### 第一步：拒绝步骤

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` 是均匀随机数。`q_prob` 是验证器对草拟 token 的概率。`p_prob` 是草稿模型的概率。Leviathan 定理表明这个伯努利决策，加上拒绝时从残差采样，精确保持验证器的分布。

### 第二步：残差分布

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

从 `q` 逐元素减去 `p`，将负值钳位到零，重新归一化。在任何拒绝时从中采样。

### 第三步：一个推测步骤

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

五个接受 → 一个奖励 → 一次验证器传递产生六个 token。

### 第四步：测量接受率

在不同草稿质量水平下运行 10,000 次推测步骤。绘制接受率 vs 草稿和验证器分布间的 KL 散度。你应该看到干净的单调关系。

### 第五步：验证分布等价性

经验上：推测循环产生的 token 直方图应该匹配直接从验证器采样产生的直方图。这是实践中的 Leviathan 定理。卡方检验在采样误差内确认。

## 使用场景

生产：

```bash
# vLLM 带 EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# vLLM 带普通草稿模型
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

截至 2026 年中，TensorRT-LLM 有最快的 Medusa 路径。`faster-whisper` 用小草稿为 Whisper-large 包装推测解码。

**选择草稿：**

| 策略 | 何时选择 | 加速 |
|------|----------|------|
| 普通草稿 (1B/3B Llama 家族) | 快速原型，无训练 | 1.8-2.3 倍 |
| Medusa 头 | 你可以微调验证器 | 2-3 倍 |
| EAGLE-2 / 3 | 生产，最高速 | 3-4 倍 |
| 前向 | 无草稿，无训练，无额外参数 | 1.3-1.6 倍 |

**何时不推测解码：**

- 1-5 个 token 的单序列生成。开销主导。
- 高度创意/高温度采样（α 下降）。
- 内存受限部署（草稿模型增加 VRAM）。

## 交付使用

参见 `outputs/skill-spec-decode-picker.md`。该技能为新推理工作负载选择推测解码策略（普通 / Medusa / EAGLE / 前向）和调优参数（N、草稿温度）。

## 练习

1. **简单。** 运行 `code/main.py`。确认推测 token 分布在 50,000 个 token 上匹配验证器的直接采样分布，卡方 p > 0.05。
2. **中等。** 绘制加速（每大模型前向的 token 数）作为 `N` 的函数，`α = 0.5, 0.7, 0.85`。识别每个 α 的最优 `N`。（提示：每验证调用的期望 token = `(1 - α^{N+1}) / (1 - α)`。）
3. **困难。** 实现微型 Medusa：取第 14 课的毕业项目 GPT，添加 3 个额外 LM 头预测位置 t+2、t+3、t+4。在 tinyshakespeare 上用联合多头损失训练。比较接受率 vs 通过截断同一模型制作的普通草稿。
4. **困难。** 实现回滚：从 10 token 前缀 KV 缓存开始，送入 5 个草稿 token，模拟位置 3 的拒绝。验证你的缓存在下一次迭代中正确读取"前缀 + 前 2 个接受草稿"。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 草稿模型 | "便宜的那个" | 提议候选 token 的较小模型；通常比验证器便宜 10-50 倍。 |
| 验证器 | "大的那个" | 我们保持其分布的目标模型；每推测步运行一次。 |
| 接受率 (α) | "草稿多常正确" | 验证器接受草稿的每 token 概率。典型 0.7-0.9。 |
| 残差分布 | "拒绝后备" | 归一化的 `(q - p)_+`；拒绝时从中采样保持验证器分布。 |
| 奖励 token | "免费的那个" | 所有 N 个草稿接受时，从验证器的下一步分布多采样一个。 |
| Medusa | "无草稿的推测" | 验证器上的多个 LM 头并行预测位置 t+1..t+k。 |
| EAGLE | "隐藏状态草稿" | 以验证器最后层隐藏状态为条件的微型 Transformer 草稿。 |
| 前向解码 | "Jacobi 迭代" | 使用定点迭代的自推测；无草稿模型。 |
| 树注意力 | "一次验证多个候选" | 同时考虑多个草稿延续的分支验证。 |
| KV 回滚 | "撤销被拒绝的草稿" | 临时 KV 缓冲区；接受后提交，拒绝后丢弃。 |

## 延伸阅读

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 核心算法和等价定理。
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — 并行引入；干净的伯努利拒绝证明。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — Medusa 论文；树注意力验证。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1；隐藏状态条件草稿。
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — EAGLE-2；动态树深度。
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) — EAGLE-3。
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — 前向，无草稿方法。
- [vLLM docs — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) — 权威生产参考，四种策略全部接入。
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE) — EAGLE-1/2/3 的参考代码。
