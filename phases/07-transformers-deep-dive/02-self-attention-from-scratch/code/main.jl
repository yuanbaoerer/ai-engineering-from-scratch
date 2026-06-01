"""
从零实现自注意力机制 (Self-Attention from Scratch) —— Julia 版本

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

using Random
using LinearAlgebra
using Printf


"""
    softmax_rows(M) -> Matrix

数值稳定的逐行 softmax。

对每行先减去最大值（防止 exp 溢出），再计算 exp 和归一化：
    softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
"""
function softmax_rows(M::Matrix{Float64})::Matrix{Float64}
    out = similar(M)
    for i in 1:size(M, 1)
        row = M[i, :]
        m = maximum(row)       # 找到行最大值，用于数值稳定
        e = exp.(row .- m)     # 减去最大值后取 exp
        s = sum(e)
        out[i, :] = e ./ s    # 归一化为概率分布
    end
    return out
end


"""
    scaled_dot_product_attention(Q, K, V) -> (output, weights)

缩放点积注意力 (Scaled Dot-Product Attention)。

核心公式：Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

缩放因子 1/sqrt(d_k) 的作用：防止点积值过大导致 softmax 梯度消失。

参数:
- Q: Query 矩阵，形状 (seq_len, dk)
- K: Key 矩阵，形状 (seq_len, dk)
- V: Value 矩阵，形状 (seq_len, dv)

返回:
- output: 注意力加权后的输出，形状 (seq_len, dv)
- weights: 注意力权重矩阵，形状 (seq_len, seq_len)
"""
function scaled_dot_product_attention(Q::Matrix{Float64}, K::Matrix{Float64},
                                      V::Matrix{Float64})
    dk = size(Q, 2)
    scores = (Q * transpose(K)) ./ sqrt(dk)  # Q @ K^T / sqrt(d_k)
    weights = softmax_rows(scores)             # softmax 归一化
    output = weights * V                       # 注意力加权求和
    return output, weights
end


"""
单头自注意力层 (Single-Head Self-Attention)。

将输入 X 分别通过三个线性变换得到 Q、K、V，然后计算缩放点积注意力。

字段:
- Wq: Query 权重矩阵，形状 (d_model, dk)
- Wk: Key 权重矩阵，形状 (d_model, dk)
- Wv: Value 权重矩阵，形状 (d_model, dv)
- dk: Query/Key 投影维度
"""
struct SelfAttention
    Wq::Matrix{Float64}
    Wk::Matrix{Float64}
    Wv::Matrix{Float64}
    dk::Int
end


"""
    SelfAttention(d_model, dk, dv; seed=42)

构造函数，使用 Xavier 初始化权重矩阵。
Xavier 初始化：scale = sqrt(2 / (fan_in + fan_out))，保持前向传播方差稳定。
"""
function SelfAttention(d_model::Int, dk::Int, dv::Int; seed::Int=42)
    rng = MersenneTwister(seed)
    # Xavier 初始化：保持前向传播方差稳定
    scale_qk = sqrt(2.0 / (d_model + dk))
    scale_v = sqrt(2.0 / (d_model + dv))
    Wq = scale_qk .* randn(rng, d_model, dk)  # Query 权重
    Wk = scale_qk .* randn(rng, d_model, dk)  # Key 权重
    Wv = scale_v .* randn(rng, d_model, dv)    # Value 权重
    return SelfAttention(Wq, Wk, Wv, dk)
end


"""
    forward(attn, X) -> (output, weights)

前向传播：Q = X @ Wq, K = X @ Wk, V = X @ Wv，然后计算注意力。

参数:
- attn: SelfAttention 层
- X: 输入矩阵，形状 (seq_len, d_model)
"""
function forward(attn::SelfAttention, X::Matrix{Float64})
    Q = X * attn.Wq  # Query：表示"我在找什么"
    K = X * attn.Wk  # Key：表示"我有什么信息"
    V = X * attn.Wv  # Value：表示"我能提供什么"
    return scaled_dot_product_attention(Q, K, V)
end


"""
多头自注意力层 (Multi-Head Self-Attention)。

将注意力计算拆分到多个"头"并行进行，每个头可以学习不同的注意力模式。
流程：
1. 将输入投影到 n_heads 个子空间，每个头独立计算注意力
2. 拼接所有头的输出
3. 通过输出投影矩阵 Wo 融合多头信息

字段:
- heads: SelfAttention 层的向量
- Wo: 输出投影矩阵，形状 (n_heads * dv, d_model)
- n_heads: 注意力头的数量
"""
struct MultiHeadSelfAttention
    heads::Vector{SelfAttention}
    Wo::Matrix{Float64}
    n_heads::Int
end


"""
    MultiHeadSelfAttention(d_model, n_heads; seed=42)

构造函数，创建 n_heads 个独立的单头注意力层和输出投影矩阵。
"""
function MultiHeadSelfAttention(d_model::Int, n_heads::Int; seed::Int=42)
    @assert n_heads > 0 "n_heads must be > 0"
    @assert d_model > 0 "d_model must be > 0"
    @assert d_model % n_heads == 0 "d_model must be divisible by n_heads"
    dk = d_model ÷ n_heads  # 每个头的维度
    dv = d_model ÷ n_heads
    # 创建 n_heads 个独立的单头注意力层
    heads = [SelfAttention(d_model, dk, dv; seed=seed + i) for i in 1:n_heads]
    rng = MersenneTwister(seed + n_heads + 1)
    # 输出投影矩阵：将拼接后的多头输出映射回 d_model 维
    scale = sqrt(2.0 / (d_model + d_model))
    Wo = scale .* randn(rng, n_heads * dv, d_model)
    return MultiHeadSelfAttention(heads, Wo, n_heads)
end


"""
    forward(mha, X) -> (output, weights_per_head)

前向传播：计算多头自注意力。

1. 并行计算每个头的注意力
2. 拼接所有头的输出
3. 通过输出投影矩阵 Wo 融合
"""
function forward(mha::MultiHeadSelfAttention, X::Matrix{Float64})
    head_outputs = Matrix{Float64}[]
    weights_per_head = Matrix{Float64}[]
    # 并行计算每个头的注意力
    for head in mha.heads
        out, w = forward(head, X)
        push!(head_outputs, out)
        push!(weights_per_head, w)
    end
    # 拼接所有头的输出并投影
    concat = hcat(head_outputs...)
    return concat * mha.Wo, weights_per_head
end


"""
打印注意力权重矩阵。
"""
function print_attention_matrix(weights::Matrix{Float64}, tokens::Vector{String})
    # 列标题
    print("\n      ")
    for token in tokens
        @printf("%6s", token)
    end
    println()
    # 每行：行标签 + 权重值
    for i in 1:length(tokens)
        @printf("%6s", tokens[i])
        for j in 1:length(tokens)
            @printf("%6.3f", weights[i, j])
        end
        println()
    end
end


"""
用 ASCII 字符绘制注意力权重热力图。
权重值越大，使用的字符越"密"。
"""
function ascii_heatmap(weights::Matrix{Float64}, tokens::Vector{String};
                       chars::String=" .:-=+*#%@")
    # 列标题
    print("\n      ")
    for t in tokens
        @printf("%6s", t)
    end
    println()
    w_max = maximum(weights)
    # 热力图主体
    for i in 1:length(tokens)
        @printf("%6s", tokens[i])
        for j in 1:length(tokens)
            # 将权重值映射到字符索引
            level = Int(floor(weights[i, j] * (length(chars) - 1) / w_max))
            level = min(level, length(chars) - 1)
            ch = chars[level + 1]
            @printf("    %s ", ch)
        end
        println()
    end
end


"""
Softmax 数值稳定性演示。
展示减去最大值后，即使输入值很大也不会溢出。
"""
function demo_softmax_stability()
    println("\n" * "=" ^ 60)
    println("SOFTMAX NUMERIC STABILITY")
    println("=" ^ 60)
    # 普通 logits
    logits = reshape([2.0, 1.0, 0.1], 1, 3)
    probs = softmax_rows(logits)
    @printf("\nLogits:  [%s]\n", join([@sprintf("%.4f", v) for v in logits], ", "))
    @printf("Softmax: [%s]\n", join([@sprintf("%.4f", v) for v in probs], ", "))
    @printf("Sum:     %.4f\n", sum(probs))

    # 大数值 logits：展示数值稳定性
    big_logits = reshape([100.0, 200.0, 300.0], 1, 3)
    big_probs = softmax_rows(big_logits)
    @printf("\nLarge logits:  [%s]\n",
            join([@sprintf("%.1f", v) for v in big_logits], ", "))
    @printf("Softmax:       [%s]\n",
            join([@sprintf("%.4f", v) for v in big_probs], ", "))
    @printf("Sum:           %.4f\n", sum(big_probs))
    println("(no overflow because we subtract the row maximum before exp)")
end


"""
演示单头自注意力：用随机向量模拟词嵌入，计算注意力权重并可视化。
"""
function demo_self_attention()
    println("=" ^ 60)
    println("SELF-ATTENTION FROM SCRATCH")
    println("=" ^ 60)

    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    n_tokens = length(tokens)
    d_model = 16  # 嵌入维度
    dk = 8        # Query/Key 投影维度
    dv = 8        # Value 投影维度

    # 用随机向量模拟词嵌入（实际中由 Embedding 层生成）
    rng = MersenneTwister(42)
    X = randn(rng, n_tokens, d_model)

    @printf("\nSentence: %s\n", join(tokens, " "))
    @printf("Tokens: %d  d_model: %d  dk: %d  dv: %d\n", n_tokens, d_model, dk, dv)
    @printf("Input shape: (%d, %d)\n", size(X, 1), size(X, 2))

    # 创建自注意力层并计算前向传播
    attn = SelfAttention(d_model, dk, dv; seed=42)
    output, weights = forward(attn, X)
    @printf("\nOutput shape: (%d, %d)\n", size(output, 1), size(output, 2))
    println("\nAttention weights:")
    print_attention_matrix(weights, tokens)
    println("\nASCII heatmap (denser char = higher attention):")
    ascii_heatmap(weights, tokens)
    return tokens, X, d_model
end


"""
演示多头自注意力：展示不同头学到不同的注意力模式。
"""
function demo_multi_head(tokens::Vector{String}, X::Matrix{Float64}, d_model::Int)
    println("\n" * "=" ^ 60)
    println("MULTI-HEAD SELF-ATTENTION")
    println("=" ^ 60)
    n_heads = 2
    mha = MultiHeadSelfAttention(d_model, n_heads; seed=42)
    out, head_weights = forward(mha, X)
    @printf("\nHeads: %d  Output shape: (%d, %d)\n",
            n_heads, size(out, 1), size(out, 2))
    # 打印每个头的注意力权重，展示不同头学到不同模式
    for (h, w) in enumerate(head_weights)
        @printf("\nHead %d attention weights:\n", h)
        print_attention_matrix(w, tokens)
    end
end


function main()
    tokens, X, d_model = demo_self_attention()
    demo_multi_head(tokens, X, d_model)
    demo_softmax_stability()
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
