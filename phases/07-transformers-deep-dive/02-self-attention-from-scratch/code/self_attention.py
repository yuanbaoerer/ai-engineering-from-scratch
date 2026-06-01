"""
从零实现自注意力机制 (Self-Attention from Scratch)

本模块实现了 Transformer 的核心组件——自注意力机制，包括：
- 数值稳定的 softmax
- 缩放点积注意力 (Scaled Dot-Product Attention)
- 单头自注意力 (SelfAttention)
- 多头自注意力 (MultiHeadSelfAttention)

核心公式：Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

参考文献：
- Vaswani et al., "Attention Is All You Need" (2017)
  https://arxiv.org/abs/1706.03762
"""

import numpy as np


def softmax(x):
    """
    数值稳定的 softmax 函数。

    通过减去每行最大值防止 exp() 溢出：
        softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

    参数:
        x: 输入矩阵，沿最后一个轴计算 softmax
    返回:
        概率分布矩阵，每行之和为 1
    """
    # 减去行最大值，防止 exp() 数值溢出
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def scaled_dot_product_attention(Q, K, V):
    """
    缩放点积注意力 (Scaled Dot-Product Attention)。

    这是 Transformer 注意力机制的核心运算：
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    缩放因子 1/sqrt(d_k) 的作用：防止点积值过大导致 softmax 梯度消失。
    当 d_k 较大时，点积的方差约为 d_k，除以 sqrt(d_k) 使方差回到 1。

    参数:
        Q: Query 矩阵，形状 (seq_len, dk)
        K: Key 矩阵，形状 (seq_len, dk)
        V: Value 矩阵，形状 (seq_len, dv)
    返回:
        output: 注意力加权后的输出，形状 (seq_len, dv)
        weights: 注意力权重矩阵，形状 (seq_len, seq_len)
    """
    dk = Q.shape[-1]
    # 计算注意力分数：Q @ K^T，然后除以 sqrt(d_k) 进行缩放
    scores = Q @ K.T / np.sqrt(dk)
    # 将分数转换为概率分布（注意力权重）
    weights = softmax(scores)
    # 用注意力权重对 Value 进行加权求和
    output = weights @ V
    return output, weights


class SelfAttention:
    """
    单头自注意力层 (Single-Head Self-Attention)。

    将输入 X 分别通过三个线性变换得到 Q、K、V：
        Q = X @ W_q
        K = X @ W_k
        V = X @ W_v
    然后计算缩放点积注意力。

    参数:
        d_model: 输入维度（嵌入维度）
        dk: Query/Key 的投影维度
        dv: Value 的投影维度
        seed: 随机种子，确保可复现
    """

    def __init__(self, d_model, dk, dv, seed=42):
        rng = np.random.default_rng(seed)
        # Xavier/Glorot 初始化：scale = sqrt(2 / (fan_in + fan_out))
        # 保持前向传播中方差稳定，避免梯度消失/爆炸
        scale_qk = np.sqrt(2.0 / (d_model + dk))
        self.Wq = rng.normal(0, scale_qk, (d_model, dk))  # Query 权重矩阵
        self.Wk = rng.normal(0, scale_qk, (d_model, dk))  # Key 权重矩阵
        scale_v = np.sqrt(2.0 / (d_model + dv))
        self.Wv = rng.normal(0, scale_v, (d_model, dv))    # Value 权重矩阵
        self.dk = dk

    def forward(self, X):
        """
        前向传播：计算自注意力。

        参数:
            X: 输入矩阵，形状 (seq_len, d_model)
        返回:
            output: 注意力输出，形状 (seq_len, dv)
            weights: 注意力权重，形状 (seq_len, seq_len)
        """
        # 线性投影：将输入映射到 Q、K、V 空间
        Q = X @ self.Wq  # Query：表示"我在找什么"
        K = X @ self.Wk  # Key：表示"我有什么信息"
        V = X @ self.Wv  # Value：表示"我能提供什么"
        return scaled_dot_product_attention(Q, K, V)


class MultiHeadSelfAttention:
    """
    多头自注意力层 (Multi-Head Self-Attention)。

    多头注意力的核心思想：将注意力计算拆分到多个"头"并行进行，
    每个头可以学习不同的注意力模式（如语法关系、语义相似性等）。

    流程：
        1. 将输入投影到 n_heads 个子空间，每个头独立计算注意力
        2. 拼接所有头的输出
        3. 通过输出投影矩阵 W_o 融合多头信息

    参数:
        d_model: 输入/输出维度
        n_heads: 注意力头的数量（d_model 必须能被 n_heads 整除）
        seed: 随机种子
    """

    def __init__(self, d_model, n_heads, seed=42):
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.n_heads = n_heads
        # 每个头的维度：d_k = d_v = d_model / n_heads
        self.dk = d_model // n_heads
        self.dv = d_model // n_heads
        # 创建 n_heads 个独立的单头注意力层
        self.heads = [
            SelfAttention(d_model, self.dk, self.dv, seed=seed + i)
            for i in range(n_heads)
        ]
        # 输出投影矩阵：将拼接后的多头输出映射回 d_model 维
        rng = np.random.default_rng(seed + n_heads)
        scale = np.sqrt(2.0 / (d_model + d_model))
        self.Wo = rng.normal(0, scale, (n_heads * self.dv, d_model))

    def forward(self, X):
        """
        前向传播：计算多头自注意力。

        参数:
            X: 输入矩阵，形状 (seq_len, d_model)
        返回:
            output: 多头注意力输出，形状 (seq_len, d_model)
            all_weights: 每个头的注意力权重列表
        """
        head_outputs = []
        all_weights = []
        # 并行计算每个头的注意力
        for head in self.heads:
            out, w = head.forward(X)
            head_outputs.append(out)
            all_weights.append(w)
        # 拼接所有头的输出：(seq_len, n_heads * dv)
        concatenated = np.concatenate(head_outputs, axis=-1)
        # 输出投影：(seq_len, n_heads * dv) @ (n_heads * dv, d_model) -> (seq_len, d_model)
        output = concatenated @ self.Wo
        return output, all_weights


def print_attention_matrix(weights, tokens):
    """
    打印注意力权重矩阵。

    参数:
        weights: 注意力权重矩阵，形状 (n, n)
        tokens: token 列表，用于行/列标签
    """
    # 打印列标题
    print(f"\n{'':>6}", end="")
    for token in tokens:
        print(f"{token:>6}", end="")
    print()
    # 打印每行：行标签 + 权重值
    for i, token in enumerate(tokens):
        print(f"{token:>6}", end="")
        for j in range(len(tokens)):
            print(f"{weights[i][j]:6.3f}", end="")
        print()


def ascii_heatmap(weights, tokens, chars=" ░▒▓█"):
    """
    用 ASCII 字符绘制注意力权重热力图。

    权重值越大，使用的字符越"密"（越暗），直观展示注意力分布。

    参数:
        weights: 注意力权重矩阵
        tokens: token 列表
        chars: 用于表示不同密度的字符（从稀到密）
    """
    # 打印列标题
    print(f"\n{'':>6}", end="")
    for t in tokens:
        print(f"{t:>6}", end="")
    print()
    w_max = weights.max()
    for i in range(len(tokens)):
        print(f"{tokens[i]:>6}", end="")
        for j in range(len(tokens)):
            # 将权重值映射到字符索引
            level = int(weights[i][j] * (len(chars) - 1) / w_max)
            level = min(level, len(chars) - 1)
            print(f"{'  ' + chars[level] + '   '}", end="")
        print()


if __name__ == "__main__":
    # === 示例句子 ===
    sentence = ["The", "cat", "sat", "on", "the", "mat"]
    n_tokens = len(sentence)
    d_model = 16  # 嵌入维度
    dk = 8        # Query/Key 投影维度
    dv = 8        # Value 投影维度

    # 用随机向量模拟词嵌入（实际中由 Embedding 层生成）
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (n_tokens, d_model))

    # === 演示 1：单头自注意力 ===
    print("=" * 60)
    print("SELF-ATTENTION FROM SCRATCH")
    print("=" * 60)

    print(f"\nSentence: {' '.join(sentence)}")
    print(f"Tokens: {n_tokens}, d_model: {d_model}, dk: {dk}, dv: {dv}")
    print(f"Input shape: {X.shape}")

    attn = SelfAttention(d_model, dk, dv, seed=42)
    output, weights = attn.forward(X)

    print(f"\nOutput shape: {output.shape}")
    print("\nAttention weights:")
    print_attention_matrix(weights, sentence)

    print("\nASCII heatmap (darker = higher attention):")
    ascii_heatmap(weights, sentence)

    # === 演示 2：多头自注意力 ===
    print("\n" + "=" * 60)
    print("MULTI-HEAD SELF-ATTENTION")
    print("=" * 60)

    n_heads = 2
    mha = MultiHeadSelfAttention(d_model, n_heads, seed=42)
    mha_output, head_weights = mha.forward(X)

    print(f"\nHeads: {n_heads}")
    print(f"Output shape: {mha_output.shape}")

    # 打印每个头的注意力权重，展示不同头学到不同模式
    for h, hw in enumerate(head_weights):
        print(f"\nHead {h + 1} attention weights:")
        print_attention_matrix(hw, sentence)

    # === 演示 3：Softmax 数值稳定性 ===
    print("\n" + "=" * 60)
    print("SOFTMAX DEMO")
    print("=" * 60)

    # 普通 logits
    logits = np.array([2.0, 1.0, 0.1])
    probs = softmax(logits)
    print(f"\nLogits:  {logits}")
    print(f"Softmax: {probs.round(4)}")
    print(f"Sum:     {probs.sum():.4f}")

    # 大数值 logits：展示数值稳定性（减去 max 后不会溢出）
    large_logits = np.array([100.0, 200.0, 300.0])
    probs_large = softmax(large_logits)
    print(f"\nLarge logits:  {large_logits}")
    print(f"Softmax:       {probs_large.round(4)}")
    print(f"Sum:           {probs_large.sum():.4f}")
    print("(Numerically stable - no overflow)")
