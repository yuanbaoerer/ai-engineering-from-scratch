# 为什么需要 Transformer — RNN 的问题

> RNN 逐个处理 token，Transformer 则一次性处理所有 token。这一个架构层面的押注，改变了 2017 年之后深度学习的每一条扩展曲线。

**类型:** 学习
**语言:** Python
**前置知识:** 第 3 阶段（深度学习核心）、第 5 阶段 · 09（序列到序列）、第 5 阶段 · 10（注意力机制）
**时间:** 约 45 分钟

## 问题所在

2017 年之前，地球上最先进的序列模型——语言、翻译、语音——全都是循环神经网络（Recurrent Neural Network）。LSTM 和 GRU 横扫了相当于 ImageNet 级别的翻译基准测试长达五年之久。它们是当时唯一可用的工具。

但它们有三个致命弱点。**顺序计算**意味着无法沿时间轴并行化：token `t+1` 需要依赖 token `t` 的隐藏状态。一个 1,024 个 token 的序列意味着在每周期可执行 1,000,000 次浮点运算的 GPU 上需要 1,024 步串行操作。在为并行计算而设计的硬件上，训练的墙钟时间随序列长度线性增长。

**梯度消失**（Vanishing Gradients）意味着 50 个 token 之前的信息已经经过 50 次非线性变换的压缩。门控循环单元（LSTM、GRU）缓解了这一问题，但从未真正消除。长距离依赖——"我去年夏天在京都的飞机上读的那本书是……"——频繁地无法被捕获。

**固定宽度的隐藏状态**意味着编码器在解码器看到任何东西之前，就把整个源序列压缩成了一个向量。无论源序列是 5 个 token 还是 500 个 token，瓶颈的形状都一样。

2017 年的论文《Attention Is All You Need》提出了一个激进的方案：彻底抛弃循环结构。让每个位置并行地关注其他所有位置。用一次大规模矩阵乘法代替 1,024 次串行乘法来完成训练。

到 2026 年，这一结果已主导了所有模态。语言（GPT-5、Claude 4、Llama 4）、视觉（ViT、DINOv2、SAM 3）、音频（Whisper）、生物学（AlphaFold 3）、机器人（RT-2）。相同的模块，不同的输入。

## 核心概念

![RNN 顺序计算 vs Transformer 并行注意力](../assets/rnn-vs-transformer.svg)

**循环即瓶颈。** RNN 的计算方式为 `h_t = f(h_{t-1}, x_t)`。每一步都依赖上一步。你无法在 `h_4` 计算完之前计算 `h_5`。在拥有 10,000+ 个并行核心的现代 GPU 上，这在长序列上浪费了 99% 的硅片算力。

**注意力即广播。** 自注意力（Self-Attention）同时为所有 `(i, j)` 对计算 `output_i = sum_j(a_ij * v_j)`。整个 N×N 的注意力矩阵在一次批量矩阵乘法中完成。没有任何步骤依赖其他步骤。GPU 非常喜欢这种模式。

**加速并非常数。** 这是 `O(N)` 串行深度与 `O(1)` 串行深度之间的差异。在实践中，在 N=512 的相同硬件上，Transformer 每个 epoch 的训练速度快 5-10 倍，而随着序列长度增长，差距会继续扩大，直到触碰注意力的 `O(N²)` 内存墙（Flash Attention 后来解决了这个问题——见第 12 课）。

**Transformer 的代价。** 注意力的内存开销为 `O(N²)`。对于 2K 上下文，完全没问题。对于 128K 上下文，则需要滑动窗口、RoPE 外推、Flash Attention 分块或线性注意力变体。循环结构的时间和内存都是 `O(N)`；Transformer 用内存换取了时间，然后又通过并行化赢回了时间。

**归纳偏置的转移。** RNN 假设局部性和近期性。Transformer 不做任何假设——每一对 token 都是注意力的候选对象。这就是为什么 Transformer 需要更多数据才能训练好，但一旦有了足够数据就能扩展得更远。Chinchilla（2022）对此进行了形式化：给定足够的 token，Transformer 总是能在同等参数量下击败 RNN。

## 动手实现

这里没有神经网络——我们通过数值模拟核心瓶颈，让你在笔记本电脑上感受到两者之间的差距。

### 第一步：测量串行深度

参见 `code/main.py`。我们构建两个函数。一个将序列编码为链式加法（串行，类似 RNN）。另一个将其编码为并行归约（广播，类似注意力）。相同的数学运算，不同的依赖图。

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # can't parallelize: h depends on previous h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # every x is independent
```

我们对长达 100,000 个元素的序列进行计时。RNN 版本是 O(N) 的，且只使用单条 CPU 流水线。即使在纯 Python 中，注意力风格的归约在长度 >= 1,000 时就已经更快，因为 Python 的 `sum()` 是用 C 实现的，迭代时没有每步的解释器开销。

### 第二步：计算理论操作数

两种算法都执行 N 次加法。区别在于**依赖深度**（Dependency Depth）：在下一步可以开始之前，必须顺序完成的操作数量。RNN 深度 = N。注意力深度 = 使用树归约为 log(N)，使用并行扫描则为 1。决定 GPU 耗时的不是操作数量，而是深度。

### 第三步：长序列上的经验扩展

我们打印一张计时表，使 O(N) 的差距一目了然。在 2026 年的 Mac 笔记本上，1,000 个元素以下的序列太快而无法测量。100,000 个元素的序列呈现出清晰的线性扫描。将其扩展到 16,384 个 token 的 Transformer 配合 12 层 LSTM 等价模型，你就能理解为什么 2016 年训练的墙钟时间是一个拦路虎。

## 使用场景

2026 年仍然选择 RNN 的情况：

| 场景 | 选择 |
|------|------|
| 流式推理，逐 token 生成，常量内存 | RNN 或状态空间模型（Mamba、RWKV） |
| 超长序列（>1M token），注意力内存开销爆炸 | 线性注意力、Mamba 2、Hyena |
| 无矩阵乘法加速器的边缘设备 | 深度可分离 RNN 在 FLOPs/瓦特上仍有优势 |
| 其他所有场景（训练、批量推理、上下文最长 128K） | Transformer |

状态空间模型（State-Space Model, SSM）如 Mamba 本质上是具有结构化参数化的 RNN，兼具两者优势：`O(N)` 扫描内存，通过选择性扫描实现并行训练。它们以更好的长上下文扩展能力恢复了 90% 的 Transformer 质量。2026 年，大多数前沿实验室训练的是 SSM+Transformer 混合模型（如 Jamba、Samba）——循环结构并没有消亡，它只是变成了一个组件。

## 交付使用

参见 `outputs/skill-architecture-picker.md`。该技能根据序列长度、吞吐量和训练预算约束为新的序列问题选择架构。它应该始终拒绝在未说明权衡的情况下为超过 1B token 的训练推荐纯 RNN。

## 练习

1. **简单。** 取 `code/main.py` 中的 `rnn_style`，将标量隐藏状态替换为长度为 64 的隐藏状态向量。重新测量。串行开销随隐藏状态维度增长了多少？
2. **中等。** 用纯 Python 实现并行前缀和（Hillis-Steele scan）。验证它在长度 1024 上产生的数值输出与串行扫描相同。计算深度。
3. **困难。** 将注意力风格的归约移植到 GPU 上的 PyTorch。在序列长度从 64 扫描到 65,536 的过程中对两者计时。绘制并解释曲线形状。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 循环（Recurrence） | "RNN 是顺序的" | 步骤 `t` 依赖步骤 `t-1` 的计算，强制沿时间轴串行执行。 |
| 串行深度（Serial Depth） | "计算图有多深" | 依赖操作的最长链；即使在无限硬件上也限制墙钟时间。 |
| 注意力（Attention） | "让 token 互相查看" | 加权求和 `sum_j a_ij v_j`，其中 `a_ij` 来自位置 i 和 j 之间的相似度分数。 |
| 上下文窗口（Context Window） | "模型能看到多少" | 注意力层可作为输入的位置数；二次内存开销在此体现。 |
| 归纳偏置（Inductive Bias） | "架构中内置的假设" | 关于数据形态的先验知识；CNN 假设平移不变性，RNN 假设近期性。 |
| 状态空间模型（State-Space Model） | "有代数支撑的 RNN" | 通过结构化状态空间矩阵实现并行训练的循环参数化。 |
| 二次瓶颈（Quadratic Bottleneck） | "为什么上下文代价这么高" | 注意力内存 = `O(N²)` 随序列长度增长；Flash Attention 隐藏了常数，但无法改变量级。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 这篇论文在主流 NLP 中终结了循环结构。
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 注意力机制的诞生地，最初附加在 RNN 上。
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — 原始的 LSTM 论文，供记录。
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — 针对 Transformer 的现代循环回应。
