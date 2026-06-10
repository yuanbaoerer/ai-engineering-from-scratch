# Omni 模型：Qwen2.5-Omni 与 Thinker-Talker 拆分

> GPT-4o 在 2024 年 5 月的产品演示之所以具有颠覆性，并非因为底层模型，而是因为产品形态——一个语音交互界面：你说话，模型看到摄像头所看到的画面，并在 250 毫秒内回话。2024 年和 2025 年，开源生态系统的其余时间都在竞相达到这种产品体验。Qwen2.5-Omni（2025 年 3 月）是参考性的开源设计：一个 Thinker（大型文本生成 transformer）加上一个 Talker（并行语音生成 transformer），通过流式语音 token 连接。Mini-Omni 简化了它，Moshi 匹配了它的延迟，GLM-4-Voice 将其扩展到中文。本课程解读 Thinker-Talker 架构以及使流式实时对话得以实现的延迟预算。

**类型：** 构建
**语言：** Python（标准库，流式管道延迟模拟器 + VAD 循环）
**前置知识：** Phase 12 · 19（音频 LLM），Phase 12 · 16（any-to-any）
**时间：** 约 180 分钟

## 学习目标

- 将推理管道拆分为 Thinker（文本推理）和 Talker（语音合成），并解释为什么并行流式传输有效。
- 逐组件计算对话交互的首音频字节时间（TTFAB）预算。
- 描述 TMRoPE 在 Thinker 内部跨视觉、音频和文本的时间对齐位置编码。
- 说出三种实时对话模式：半双工、轮流对话、全双工。

## 问题

实时语音助手必须快速完成很多事情：

1. 听到用户。实时语音 token 化，语音活动检测（VAD）以判断用户何时说完。
2. 可选地看。摄像头输入以 2-4 FPS 传入，与音频一起流式输入 Thinker。
3. 思考。基于对话历史组织回复。
4. 说话。合成音频 token，解码为波形，流式传输到用户扬声器。

每一步都增加延迟。对话感要求总往返时间 < 500 毫秒——低于此值，用户便不再注意到延迟。GPT-4o 声称约 250 毫秒。Moshi 约 160 毫秒。Qwen2.5-Omni 约 350-500 毫秒。

每个组件都需要流式传输。不能"全部批处理后再解码"。

## 概念

### Thinker 和 Talker

Qwen2.5-Omni 的分解：

- Thinker：一个 7B-80B 的文本生成 transformer。消费交错的文本 + 图像 + 音频 token。输出代表要说内容的文本 token。
- Talker：一个较小的语音生成 transformer（200M-1B）。消费 Thinker 的文本输出 token 加上最近的语音上下文 token。输出离散语音 token（residual-VQ 索引）。
- 语音解码器：一个流式波形解码器（SNAC、MoVQGAN 系列），将语音 token 实时转换为音频采样。

这种分离很重要。Thinker 必须足够大才能进行良好的推理。Talker 可以很小，因为它的工作是局部的——将文本转换为语音 token。更大的 Talker 并不会更具表现力；它只会更慢。

两者并行运行：

1. Thinker 发出文本 token t_i。
2. Talker 消费 t_i（通过流式传输）并发出语音 token s_i, s_{i+1}, ..., s_{i+k}。
3. 语音解码器随着语音 token 的到来而消费它们，并发出音频采样。
4. 当 Thinker 处理到文本 token t_{i+3} 时，Talker 已经为 t_0..t_{i+2} 流式传输了音频。

### TMRoPE —— 时间对齐的多模态位置

Thinker 需要整合图像帧（例如以 4 FPS 到达）、音频帧（以 50 帧/秒到达）以及对话历史中的文本。朴素的序列顺序（所有图像，然后所有音频，然后文本）会丢失时间对齐。

TMRoPE 为每个 token 分配绝对时间戳。视觉 token 在 t=2.3s。音频 token 在 t=2.32s。用户说"stop"的文本 token 在 t=2.35s。RoPE 按时间戳旋转注意力；模型将它们视为时间并发的。

这是"他一边挥手一边说你好"能够工作的基础设施——模型在同一概念时刻看到视频帧和音频。

### 流式语音合成

语音 token 必须流式传输。Mini-Omni（Xie & Wu, 2024）引入了"语言模型可以在流式思考的同时听和说"：Thinker 输出 token 和 Talker 输出 token 在同一个序列中交错。Thinker 提交下一个文本 token 后，Talker 立即触发。没有批处理边界。

Moshi（Défossez et al., 2024 年 10 月）是最快的开源实现。在单张 A100 上 TTFAB 为 160 毫秒。架构：一个单一的 7B transformer，在交替位置发出文本和语音 token，并带有"内心独白"，将思考流与说话流分开。这实际上是将 Thinker + Talker 融合到一个模型中，通过精心训练实现。

### VAD 和轮流对话

语音活动检测在输入端运行。两种模式：

- 半双工：用户说话，模型倾听。模型说话，用户倾听。通过 VAD 静音检测（约 200 毫秒）进行清晰的交接。
- 全双工：双方可以同时说话。模型可以发出反馈声（"uh-huh"）或打断。困难得多。Moshi 支持此功能。

Qwen2.5-Omni 默认支持半双工，通过静音阈值进行轮流对话。全双工需要应用层处理。

### Qwen3-Omni（2025 年 11 月）

继任者。Qwen3-80B Thinker，更大的 Talker，改进的 TMRoPE-v2。延迟接近 GPT-4o 的 250 毫秒。开放权重。OmniBench 基准测试与 Gemini 2.0 Live 具有竞争力。

### 生产延迟预算

对于典型的流式交互：

- 麦克风 -> 音频 token：40-80 毫秒。
- 预填充（提示 + 历史）：7B 模型 100-200 毫秒，70B 模型则长得多。
- 首个 Thinker 文本 token：40 毫秒。
- Talker 处理首个文本 token：20 毫秒。
- 首个语音 token 提交：40 毫秒。
- Residual-VQ 解码：30 毫秒。
- 语音波形解码：50-80 毫秒。

7B 模型总 TTFAB：320-510 毫秒，70B 模型 600-900 毫秒。前沿质量通常意味着 70B+；因此存在前沿延迟差距。

### Token 速率计算

对于 16kHz 语音，基础语音 token 为 50 Hz，每秒输出需要 50 个语音 token。Talker 必须以 ≥50 tok/s 的速度发出 token 才能跟上。在 H100 上典型的 LLM 吞吐量为 30-80 tok/s，小型（200-300M）Talker 足够快；7B Talker 则会落后。

这就是为什么存在小型专用 Talker 模型，而不是"直接用主模型"。

## 使用它

`code/main.py`：

- 使用模拟的 token 发射速率模拟 Thinker-Talker 管道。
- 为可配置的模型大小和麦克风采样率计算 TTFAB。
- 演示带有 VAD 静音阈值的半双工轮流对话。

## 交付它

本课程产出 `outputs/skill-omni-streaming-budget.md`。给定实时语音产品的目标 TTFAB 和功能集（视觉输入、双语、全双工），选择 Qwen2.5-Omni、Qwen3-Omni、Moshi 或 Mini-Omni，并确定 Thinker/Talker 的大小。

## 练习

1. 你的目标 TTFAB 是 300 毫秒。在 7B Thinker 和 300M Talker 上，写出每个组件的延迟。

2. Qwen2.5-Omni 使用 TMRoPE。描述当用户在 t=1s 开始说话，摄像头在 t=1.2s 捕捉到一个手势时，模型看到了什么。

3. 全双工支持要求模型在倾听的同时发出音频。提出一种训练数据格式来教授这一点。

4. 阅读 Moshi 论文的第 4 节。描述"内心独白"的分离方式，以及它为什么避免了 Thinker-Talker 拆分。

5. 计算吞吐量预算：Talker 必须以多快的速度发出 token，才能跟上 16kHz 语音在 50 个基础层 token/秒的情况？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Thinker | "推理大脑" | 生成要说内容的大型文本生成 transformer |
| Talker | "语音生成嘴巴" | 从 Thinker 的文本生成离散语音 token 的小型 transformer |
| TTFAB | "延迟预算" | 首音频字节时间：从用户语音结束到首个音频采样输出 |
| TMRoPE | "时间对齐 RoPE" | 使用绝对时间戳跨视觉、音频、文本的位置编码 |
| 半双工 | "轮流对话" | 用户和模型交替；VAD 静音检测用户说完 |
| 全双工 | "同时" | 模型可以同时说话和倾听；能够发出反馈声 |
| 内心独白 | "Moshi 分离方式" | 思考流和说话流交错的单模型设计 |

## 延伸阅读

- [Xu et al. — Qwen2.5-Omni (arXiv:2503.20215)](https://arxiv.org/abs/2503.20215)
- [Qwen Team — Qwen3-Omni (arXiv:2509.17765)](https://arxiv.org/html/2509.17765v1)
- [Xie & Wu — Mini-Omni (arXiv:2408.16725)](https://arxiv.org/abs/2408.16725)
- [Défossez et al. — Moshi (arXiv:2410.00037)](https://arxiv.org/abs/2410.00037)
- [Zeng et al. — GLM-4-Voice (arXiv:2412.02612)](https://arxiv.org/abs/2412.02612)
