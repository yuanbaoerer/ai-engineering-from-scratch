# 音频-语言模型：从 Whisper 到 Audio Flamingo 3 的演进

> Whisper（Radford 等人，2022年12月）奠定了语音识别的基础——68万小时的弱监督多语言语音数据、一个简单的编码器-解码器 Transformer、一个让后续所有 ASR 发布都引用它的基准测试。但识别不等于推理。要回答"这段录音里有哪些乐器"、"说话者表达了什么情绪"或"第三分钟发生了什么"，需要的是音频理解，而非转录。Qwen-Audio、SALMONN、LTU 以及 NVIDIA 的 Audio Flamingo 3（AF3，2025年7月）逐步构建了这一技术栈：保留 Whisper 级别的编码器，接入 Q-former，在音频-文本指令数据上训练，并引入思维链推理。本课将梳理这一演进脉络。

**类型：** 构建
**语言：** Python（标准库，log-Mel 频谱图 + 音频 Q-former 骨架）
**前置知识：** Phase 6（语音与音频），Phase 12 · 03（Q-Former）
**时间：** 约180分钟

## 学习目标

- 从波形计算 log-Mel 频谱图：分窗、FFT、滤波器组、对数变换。
- 比较编码器选项：Whisper 编码器、BEATs、AF-Whisper 混合编码器。各自在什么场景下胜出。
- 构建一个音频 Q-former：N 个可学习的查询向量对频谱图 patch 进行交叉注意力。
- 解释级联式（Whisper 后接 LLM）与端到端音频-LLM 训练：为什么端到端在推理任务上扩展性更好。

## 问题背景

语音识别已被 Whisper 解决。音频的 OCR 已成为一种商品能力。但"商品化"止步于转录。如果模型无法对听到的内容进行推理——时间信息、说话人、情绪、音乐结构、环境声音——仅靠转录无法驱动产品功能。

三条显而易见的路线：

1. **级联（Cascade）：** Whisper 转录，LLM 对转录文本进行推理。在纯语音场景下有效。对音乐、环境音频、多说话人重叠、情绪分析则失效。

2. **端到端音频-LLM：** 音频编码器将音频 token 直接输入 LLM，跳过转录环节。保留声学信息（情绪、说话人、环境）。需要新的训练数据。

3. **混合式：** 音频编码器 + 既能转录又能推理的文本解码器。Qwen-Audio 和 Audio Flamingo 选择了这条路线。

## 核心概念

### Log-Mel 频谱图：输入特征

每个音频编码器都从同一个特征开始：log-Mel 频谱图。

1. 重采样至 16 kHz。
2. 使用 25ms 窗口、10ms 步长的短时傅里叶变换（STFT）。
3. 取 FFT 结果的幅度值。
4. 应用 Mel 滤波器组（通常为 80 个，在 0-8000 Hz 范围内对数间隔）以映射到感知频率。
5. 对数压缩（log(1 + x)）以处理动态范围。

结果：一个形状为 (T, 80) 的二维数组，其中 T 为时间帧数。对于 30 秒、帧率为 100 Hz 的片段：(3000, 80)。

### Whisper 的编码器

Whisper 的编码器是一个 12 层的 ViT 风格 Transformer，将 log-Mel 频谱图作为时间帧序列处理。输出：每个时间帧对应一个隐藏状态向量。

对于 ASR，Whisper 的解码器是一个交叉注意力 Transformer，以编码器输出为条件生成文本 token。标准的编码器-解码器架构。

对于 ALM（音频-语言模型），你需要将编码器输出作为另一个 LLM 的输入。典型模式：Whisper 编码器冻结，Q-former 可训练，LLM 冻结或微调。

### BEATs 与音频专用编码器

Whisper 在语音主导的数据上训练。它在音乐和环境音频上表现较弱。

BEATs（Chen 等人，2022）是一个在 AudioSet 上自监督训练的 Transformer。在相同参数量下，比 Whisper 更好地捕捉音乐和环境声音。

AF-Whisper（Audio Flamingo 3 的混合编码器）：将 Whisper + BEATs 的特征拼接作为音频输入。Whisper 承载语言信号，BEATs 承载声学信号。

### 音频 Q-former

与 BLIP-2 的视觉 Q-former 模式相同。固定数量的可学习查询向量（通常为 32 或 64 个）对音频编码器的输出帧进行交叉注意力。这些查询向量成为 LLM 消费的音频 token。

训练对齐阶段：仅训练 Q-former，在音频-文本对（AudioCaps、Clotho）上使用对比学习 + 描述生成损失。指令阶段：端到端，解冻 LLM，在指令数据上训练。

### 演进脉络 —— SALMONN、Qwen-Audio、AF3

**SALMONN**（Tang 等人，2023）：Whisper + BEATs + Q-former + LLaMA。首个具备真正推理能力的开源音频-LLM。MMAU 基准测试综合得分约 0.55。

**Qwen-Audio**（Chu 等人，2023）：类似架构，在更丰富的数据集上训练，针对多轮对话优化。MMAU 约 0.60。

**LTU —— Listen, Think, Understand**（Gong 等人，2023）：显式推理数据，专注于对音频片段的思维链推理。规模较小但更聚焦。

**Audio Flamingo 3**（Goel 等人，2025年7月）：当前开源 SOTA。8B LLM 骨干（Qwen2 7B），Whisper-large 编码器拼接 BEATs，64 查询 Q-former，在 100 万+ 音频-文本指令对上训练。MMAU 0.72，在某些子任务上达到闭源前沿水平。

AF3 还引入了音频的按需思维链（on-demand chain-of-thought）：模型可以选择性地输出思考 token（"让我先识别乐器：..."），然后再给出最终答案。在复杂推理任务上，启用思考后准确率提升 3-5 个百分点。

### 级联式 vs 端到端

**级联式流水线：**

1. Whisper 将音频转录为文本。
2. LLM 对文本进行推理。

对于"总结这个播客"这类任务表现完美。但在以下场景失效：
- "这首歌的情绪是什么？"——情绪在声音中，不在文字里。
- "谁在说话，Alice 还是 Bob？"——需要说话人识别。
- "爆炸发生在第几秒？"——文本中丢失了时间定位信息。
- "这是真实音频还是生成的？"——深度伪造检测需要声学特征。

端到端保留了声学信号。Qwen-Audio 和 AF3 原生处理音乐、环境和情绪。

### 2026 年生产实践建议

对于一个新的音频理解产品：

- **级联式，如果：** 目标是转录，没有音乐，没有情绪推理。
- **AF3 / Qwen-Audio 系列，如果：** 涉及音乐、情绪、多说话人或复杂音频推理。

级联式更便宜、更简单。端到端能力更强。

### MMAU —— 音频推理基准测试

MMAU（Massive Multimodal Audio Understanding）是 2024-2025 年的音频推理基准测试：

- 10,000 个音频-文本 QA 对，涵盖语音、音乐、环境声音。
- 覆盖分类、时间推理、因果推理、开放式问答。
- 测试级联流水线系统性遗漏的能力。

开源 SOTA（AF3）为 0.72；闭源前沿约 0.78（Gemini 2.5 Pro、Claude Opus 4.7）。差距小于 VideoMME 的开源-闭源差距，表明音频-LLM 正在成熟。

## 动手实践

`code/main.py`：

- 在标准库中实现 log-Mel 频谱图计算：分窗、朴素 DFT、Mel 滤波器组。
- 音频 Q-former 骨架：给定编码器输出帧，计算 Q、K、V、注意力，并输出 N 个 token。
- 在玩具任务上比较级联式与端到端。

## 产出交付

本课产出 `outputs/skill-audio-llm-pipeline-picker.md`。给定一个音频任务（转录、音乐标签、情绪推理、多说话人 diarization、环境分类），选择级联式、端到端 AF3 或混合方案。

## 练习题

1. 计算一段 30 秒、16kHz、25ms 窗口、10ms 步长、80 个 Mel 频段的 log-Mel 频谱图维度。在 48kHz 下会如何变化？

2. 为什么 Whisper 在音乐上表现不佳？BEATs 捕捉了哪些 Whisper 无法捕捉的音频特征？

3. 64 查询 vs 32 查询的音频 Q-former：在什么任务复杂度下 64 查询更有优势？32 查询在什么场景下节省计算？

4. 阅读 AF3 第 4 节关于按需思考（on-demand thinking）的内容。提出三个思维链帮助最大的音频任务。

5. 使用 AF3 的输出实现一个最小的 diarization 流水线。如何标记说话人变化？

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|---------|---------|
| Log-Mel spectrogram | "Mel 特征" | 经过 Mel 滤波器组后的对数幅度值的二维（时间，频率）数组 |
| Audio Q-former | "音频 Perceiver" | 从音频编码器输出到固定长度查询向量的交叉注意力瓶颈层，查询向量输入 LLM |
| Cascaded | "ASR 后接 LLM" | Whisper 转录、文本 LLM 推理的流水线；丢失声学信息 |
| End-to-end | "音频-LLM" | 音频特征通过 Q-former 直接进入 LLM；保留声学信号 |
| BEATs | "AudioSet 编码器" | 在 AudioSet 上自监督训练的 Transformer；在音乐和环境声音上表现强 |
| MMAU | "音频推理基准" | 1 万个涵盖语音、音乐、环境的 QA 对；2024 年评估标准 |
| On-demand thinking | "音频 CoT" | 模型可选择性地在最终答案前输出推理 token，准确率提升 3-5 个百分点 |

## 延伸阅读

- [Radford 等人 —— Whisper (arXiv:2212.04356)](https://arxiv.org/abs/2212.04356)
- [Chu 等人 —— Qwen-Audio (arXiv:2311.07919)](https://arxiv.org/abs/2311.07919)
- [Goel 等人 —— Audio Flamingo 3 (arXiv:2507.08128)](https://arxiv.org/abs/2507.08128)
- [Tang 等人 —— SALMONN (arXiv:2310.13289)](https://arxiv.org/abs/2310.13289)
- [Gong 等人 —— LTU (arXiv:2305.10790)](https://arxiv.org/abs/2305.10790)
