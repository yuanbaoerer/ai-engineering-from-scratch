//! 从零实现自注意力机制 (Self-Attention from Scratch) —— Rust 版本
//!
//! 仅使用标准库，手动实现行优先矩阵运算和缩放点积注意力。
//! 适合理解注意力机制的底层计算细节。
//!
//! 核心公式：Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
//!
//! 参考文献：
//!   - Vaswani 2017, "Attention Is All You Need": https://arxiv.org/abs/1706.03762
//!   - candle reference attention kernel: https://github.com/huggingface/candle/blob/main/candle-nn/src/ops.rs
//!   - Karpathy llm.c attention forward pass: https://github.com/karpathy/llm.c/blob/master/train_gpt2.c
//!
//! 编译运行: rustc --edition 2021 main.rs -o /tmp/sa && /tmp/sa

use std::f32::consts::E;

/// 行优先 (row-major) 矩阵，底层存储为一维 Vec<f32>。
///
/// 索引方式：data[i * cols + j] 访问第 i 行第 j 列元素。
/// 这是大多数深度学习框架（如 PyTorch 默认）采用的内存布局。
struct Mat {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Mat {
    /// 创建全零矩阵
    fn zeros(rows: usize, cols: usize) -> Self {
        Mat { rows, cols, data: vec![0.0; rows * cols] }
    }

    /// 按 (行, 列) 读取元素
    #[inline] fn at(&self, i: usize, j: usize) -> f32 { self.data[i * self.cols + j] }
    /// 按 (行, 列) 写入元素
    #[inline] fn set(&mut self, i: usize, j: usize, v: f32) { self.data[i * self.cols + j] = v; }

    /// 矩阵乘法：self @ b
    ///
    /// 使用 i-k-j 循环顺序，对稀疏 A 矩阵跳过零元素以优化性能。
    /// 要求 self.cols == b.rows。
    fn matmul(&self, b: &Mat) -> Mat {
        assert_eq!(self.cols, b.rows, "shape mismatch: {}x{} @ {}x{}", self.rows, self.cols, b.rows, b.cols);
        let mut out = Mat::zeros(self.rows, b.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let aik = self.at(i, k);
                if aik == 0.0 { continue; }  // 跳过零元素，优化稀疏矩阵
                let row_base = i * out.cols;
                let bk_base = k * b.cols;
                for j in 0..b.cols {
                    out.data[row_base + j] += aik * b.data[bk_base + j];
                }
            }
        }
        out
    }

    /// 矩阵转置：返回 self^T
    fn transpose(&self) -> Mat {
        let mut t = Mat::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                t.set(j, i, self.at(i, j));
            }
        }
        t
    }

    /// 标量乘法：原地将所有元素乘以 s
    fn scale(&mut self, s: f32) {
        for v in self.data.iter_mut() { *v *= s; }
    }
}

/// 数值稳定的逐行 softmax。
///
/// 算法：对每行先减去最大值（防止 exp 溢出），再计算 exp 和归一化。
/// softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
fn softmax_rows(m: &Mat) -> Mat {
    let mut out = Mat::zeros(m.rows, m.cols);
    for i in 0..m.rows {
        // 找到当前行的最大值
        let mut row_max = f32::NEG_INFINITY;
        for j in 0..m.cols { if m.at(i, j) > row_max { row_max = m.at(i, j); } }
        // 计算 exp(x - max) 并累加求和
        let mut sum = 0.0f32;
        for j in 0..m.cols {
            let e = E.powf(m.at(i, j) - row_max);
            out.set(i, j, e);
            sum += e;
        }
        // 归一化：每个元素除以总和
        let inv = 1.0 / sum;
        for j in 0..m.cols {
            let v = out.at(i, j) * inv;
            out.set(i, j, v);
        }
    }
    out
}

/// 缩放点积注意力 (Scaled Dot-Product Attention)。
///
/// 核心公式：Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
///
/// 步骤：
/// 1. 计算注意力分数 scores = Q @ K^T
/// 2. 缩放 scores = scores / sqrt(d_k)（防止点积过大导致 softmax 梯度消失）
/// 3. 归一化 weights = softmax(scores)
/// 4. 加权求和 output = weights @ V
fn scaled_dot_product_attention(q: &Mat, k: &Mat, v: &Mat) -> (Mat, Mat) {
    let dk = q.cols as f32;
    let k_t = k.transpose();
    let mut scores = q.matmul(&k_t);       // Q @ K^T
    scores.scale(1.0 / dk.sqrt());          // 除以 sqrt(d_k) 缩放
    let weights = softmax_rows(&scores);    // softmax 归一化
    let out = weights.matmul(v);            // 注意力加权求和
    (out, weights)
}

/// 确定性伪随机数生成器（无外部依赖）。
///
/// 使用 Lehmer LCG (线性同余生成器) + Box-Muller 变换生成高斯随机数。
/// - LCG: state = state * a + c，取高 31 位作为均匀随机数
/// - Box-Muller: 将两个均匀随机数转换为标准正态分布
struct Rng { state: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { state: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1 } }
    /// 生成下一个 u32 随机数（取 state 的高 31 位）
    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.state >> 33) as u32
    }
    /// 生成 (0, 1) 区间的均匀分布随机数
    fn uniform(&mut self) -> f32 {
        (self.next_u32() as f32 + 1.0) / (u32::MAX as f32 + 2.0)
    }
    /// Box-Muller 变换：将两个均匀随机数转换为标准正态分布 N(0,1)
    fn gauss(&mut self) -> f32 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

/// 生成指定形状的高斯随机矩阵，元素 ~ N(0, scale²)
fn randn(rows: usize, cols: usize, scale: f32, rng: &mut Rng) -> Mat {
    let mut m = Mat::zeros(rows, cols);
    for v in m.data.iter_mut() { *v = rng.gauss() * scale; }
    m
}

/// 单头自注意力层 (Single-Head Self-Attention)。
///
/// 将输入 X 通过三个线性变换得到 Q、K、V，然后计算缩放点积注意力。
/// 权重矩阵使用 Xavier 初始化：scale = sqrt(2 / (fan_in + fan_out))
struct SelfAttention {
    wq: Mat,  // Query 权重矩阵：(d_model, dk)
    wk: Mat,  // Key 权重矩阵：(d_model, dk)
    wv: Mat,  // Value 权重矩阵：(d_model, dv)
}

impl SelfAttention {
    fn new(d_model: usize, dk: usize, dv: usize, rng: &mut Rng) -> Self {
        // Xavier 初始化：保持前向传播方差稳定
        let s_qk = (2.0 / (d_model + dk) as f32).sqrt();
        let s_v = (2.0 / (d_model + dv) as f32).sqrt();
        SelfAttention {
            wq: randn(d_model, dk, s_qk, rng),
            wk: randn(d_model, dk, s_qk, rng),
            wv: randn(d_model, dv, s_v, rng),
        }
    }

    /// 前向传播：Q = X @ Wq, K = X @ Wk, V = X @ Wv，然后计算注意力
    fn forward(&self, x: &Mat) -> (Mat, Mat) {
        let q = x.matmul(&self.wq);  // Query：表示"我在找什么"
        let k = x.matmul(&self.wk);  // Key：表示"我有什么信息"
        let v = x.matmul(&self.wv);  // Value：表示"我能提供什么"
        scaled_dot_product_attention(&q, &k, &v)
    }
}

/// 打印注意力权重矩阵
fn print_attention(weights: &Mat, tokens: &[&str]) {
    // 列标题
    print!("      ");
    for t in tokens { print!("{:>7}", t); }
    println!();
    // 每行：行标签 + 权重值
    for i in 0..weights.rows {
        print!("{:>6}", tokens[i]);
        for j in 0..weights.cols { print!("{:>7.3}", weights.at(i, j)); }
        println!();
    }
}

/// 用 ASCII 字符绘制注意力权重热力图
///
/// 字符从稀到密：' ' < ░ < ▒ < ▓ < █，权重越大字符越密
fn ascii_heatmap(weights: &Mat, tokens: &[&str]) {
    let chars = [' ', '\u{2591}', '\u{2592}', '\u{2593}', '\u{2588}'];
    // 找到最大权重值用于归一化
    let mut w_max = 0.0f32;
    for v in &weights.data { if *v > w_max { w_max = *v; } }
    // 列标题
    print!("      ");
    for t in tokens { print!("{:>7}", t); }
    println!();
    // 热力图主体
    for i in 0..weights.rows {
        print!("{:>6}", tokens[i]);
        for j in 0..weights.cols {
            // 将权重值映射到字符索引
            let level = ((weights.at(i, j) * (chars.len() - 1) as f32) / w_max) as usize;
            let level = level.min(chars.len() - 1);
            print!("     {} ", chars[level]);
        }
        println!();
    }
}

/// 数值稳定的 softmax（一维向量版本，用于演示）
fn softmax_vec(logits: &[f32]) -> Vec<f32> {
    let mut m = f32::NEG_INFINITY;
    for &x in logits { if x > m { m = x; } }
    let exps: Vec<f32> = logits.iter().map(|x| (x - m).exp()).collect();
    let s: f32 = exps.iter().sum();
    exps.into_iter().map(|x| x / s).collect()
}

fn main() {
    // === 示例句子 ===
    let sentence = ["The", "cat", "sat", "on", "the", "mat"];
    let n_tokens = sentence.len();
    let d_model: usize = 16;  // 嵌入维度
    let dk: usize = 8;        // Query/Key 投影维度
    let dv: usize = 8;        // Value 投影维度

    println!("{}", "=".repeat(60));
    println!("SELF-ATTENTION FROM SCRATCH (Rust port)");
    println!("{}", "=".repeat(60));

    // 用随机向量模拟词嵌入（实际中由 Embedding 层生成）
    let mut rng = Rng::new(42);
    let x = randn(n_tokens, d_model, 1.0, &mut rng);
    println!("\nSentence: {}", sentence.join(" "));
    println!("Tokens: {}, d_model: {}, dk: {}, dv: {}", n_tokens, d_model, dk, dv);
    println!("Input shape: ({}, {})", x.rows, x.cols);

    // 创建自注意力层并计算前向传播
    let mut rng_w = Rng::new(42);
    let attn = SelfAttention::new(d_model, dk, dv, &mut rng_w);
    let (out, weights) = attn.forward(&x);

    println!("\nOutput shape: ({}, {})", out.rows, out.cols);
    println!("\nAttention weights:");
    print_attention(&weights, &sentence);

    println!("\nASCII heatmap (darker = higher attention):");
    ascii_heatmap(&weights, &sentence);

    // === Softmax 数值稳定性演示 ===
    println!("\n{}", "=".repeat(60));
    println!("SOFTMAX DEMO");
    println!("{}", "=".repeat(60));

    // 普通 logits
    let logits = [2.0f32, 1.0, 0.1];
    let probs = softmax_vec(&logits);
    println!("\nLogits:  {:?}", logits);
    println!("Softmax: {:?}", probs.iter().map(|p| (p * 10000.0).round() / 10000.0).collect::<Vec<_>>());
    println!("Sum:     {:.4}", probs.iter().sum::<f32>());

    // 大数值 logits：展示数值稳定性（减去 max 后不会溢出）
    let large = [100.0f32, 200.0, 300.0];
    let probs_l = softmax_vec(&large);
    println!("\nLarge logits:  {:?}", large);
    println!("Softmax:       {:?}", probs_l.iter().map(|p| (p * 10000.0).round() / 10000.0).collect::<Vec<_>>());
    println!("Sum:           {:.4}", probs_l.iter().sum::<f32>());
    println!("(numerically stable, no overflow)");

    // === 性能基准测试 ===
    println!("\n{}", "=".repeat(60));
    println!("MICROBENCH: 10K attention forwards");
    println!("{}", "=".repeat(60));
    let start = std::time::Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..10_000 {
        let (o, _) = attn.forward(&x);
        sink += o.at(0, 0);  // 防止编译器优化掉整个循环
    }
    let elapsed = start.elapsed();
    println!("10K forwards in {:.2}ms ({:.0}/sec)  sink={:.4}",
        elapsed.as_secs_f64() * 1000.0,
        10_000.0 / elapsed.as_secs_f64(),
        sink,
    );
}
