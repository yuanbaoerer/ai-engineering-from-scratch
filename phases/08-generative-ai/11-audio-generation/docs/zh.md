# 音频生成

> 音频是 16-48 kHz 的一维信号。一个五秒片段有 80-240k 个采样点。没有 Transformer 会直接对这么长的序列做注意力。2026 年每个生产级音频模型的解决方案都相同：神经编解码器（Encodec、SoundStream、DAC）把音频压缩成 50-75 Hz 的离散 token，然后由 Transformer 或扩散模型生成 token。

**类型：** 构建
**语言：** Python
**先修要求：** 第 6 阶段 · 02（音频特征）、第 6 阶段 · 04（ASR）、第 8 阶段 · 06（DDPM）
**时间：** 约 45 分钟

## 问题

三类音频生成任务：

1. **文本转语音。** 给定文本，生成语音。干净语音是窄带的，并且有很强的音素结构——基于 token 的 Transformer 已经很好地解决了这个问题。VALL-E（Microsoft）、NaturalSpeech 3、ElevenLabs、OpenAI TTS。
2. **音乐生成。** 给定提示词（文本、旋律、和弦进行、流派），生成音乐。分布宽得多。MusicGen（Meta）、Stable Audio 2.5、Suno v4、Udio、Riffusion。
3. **音效 / 声音设计。** 给定提示词，生成环境声或拟音（Foley）。AudioGen、AudioLDM 2、Stable Audio Open。

三者都运行在同一个底座上：神经音频编解码器 + token 自回归或扩散生成器。

## 核心概念

![音频生成：编解码器 token + Transformer 或扩散](../assets/audio-generation.svg)

### 神经音频编解码器

Encodec（Meta，2022）、SoundStream（Google，2021）、Descript Audio Codec（DAC，2023）。卷积编码器把波形压缩为逐时间步向量；残差向量量化（residual vector quantization，RVQ）把每个向量转换为 K 个码本索引的级联。解码器执行反向过程。24 kHz 音频以 2 kbps 编码，使用 8 个 RVQ 码本、75 Hz = 600 tokens/sec。

```
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8
```

### 上层的两种生成范式

**Token 自回归。** 把 RVQ token 展平成一个序列，运行仅解码器 Transformer。MusicGen 使用“延迟并行”（delayed parallel）方式，以逐流偏移并行发出 K 个码本流。VALL-E 根据文本提示 + 3 秒语音样本生成语音 token。

**潜空间扩散。** 把编解码器 token 打包为连续潜变量，或用类别扩散对它们建模。Stable Audio 2.5 在连续音频潜变量上使用流匹配（flow matching）。AudioLDM 2 使用文本到梅尔谱再到音频的扩散流程。

2024-2026 年趋势：流匹配正在音乐领域胜出（推理更快、样本更干净），而 token 自回归仍主导语音，因为它天然因果并且很适合流式传输。

## 生产格局

| 系统 | 任务 | 主干 | 延迟 |
|--------|------|----------|---------|
| ElevenLabs V3 | TTS | Token-AR + 神经声码器 | 首 token 约 300ms |
| OpenAI GPT-4o audio | 全双工语音 | 端到端多模态 AR | 约 200ms |
| NaturalSpeech 3 | TTS | 潜空间流匹配 | 非流式 |
| Stable Audio 2.5 | 音乐 / SFX | DiT + 音频潜变量上的流匹配 | 1 分钟片段约 10s |
| Suno v4 | 完整歌曲 | 未披露；疑似 token-AR | 每首约 30s |
| Udio v1.5 | 完整歌曲 | 未披露 | 每首约 30s |
| MusicGen 3.3B | 音乐 | Encodec 32kHz 上的 Token-AR | 实时 |
| AudioCraft 2 | 音乐 + SFX | 流匹配 | 5 秒片段约 5s |
| Riffusion v2 | 音乐 | 频谱图扩散 | 约 10s |

## 动手构建

`code/main.py` 模拟核心思路：在合成“音频 token”序列上训练一个微型 next-token Transformer，这些序列来自两种不同“风格”（风格 A 是低高 token 交替，风格 B 是单调斜坡）。以风格为条件进行采样。

### 第 1 步：合成音频 token

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "speech-like": alternating
        return [i % vocab_size for i in range(length)]
    # "music-like": ramp
    return [(i * 3) % vocab_size for i in range(length)]
```

### 第 2 步：训练一个微型 token 预测器

一个以风格为条件的二元语法风格预测器。重点是这个模式：编解码器 token → 交叉熵训练 → 自回归采样。

### 第 3 步：条件采样

给定风格 token 和起始 token，从预测分布中采样下一个 token。持续生成 20-40 个 token。

## 常见陷阱

- **编解码器质量限制输出质量。** 如果编解码器无法忠实表示某种声音，生成器质量再高也没用。DAC 是当前最好的开源选择。
- **RVQ 误差累积。** 每一层 RVQ 都对上一层残差建模。第 1 层的错误会传播。对更高层使用 temperature 0 采样会有帮助。
- **音乐结构。** 30 秒 token 在 75 Hz 下超过 20k token。Transformer 很难处理。MusicGen 使用滑动窗口 + 提示延续；Stable Audio 使用较短片段 + 交叉淡化。
- **边界伪影。** 在生成片段之间交叉淡化需要谨慎的 overlap-add。
- **对干净数据的饥渴。** 音乐生成器需要数万小时授权音乐。Suno / Udio 的 RIAA 诉讼（2024）把这个问题推到了台前。
- **声音克隆伦理。** 3 秒样本加一段文本提示就足以让 VALL-E / XTTS / ElevenLabs 克隆声音。每个生产模型都需要滥用检测 + 退出名单。

## 使用它

| 任务 | 2026 年栈 |
|------|------------|
| 商业 TTS | ElevenLabs、OpenAI TTS 或 Azure Neural |
| 声音克隆（已验证同意） | XTTS v2（开源）或 ElevenLabs Pro |
| 背景音乐，快速 | Stable Audio 2.5 API、Suno 或 Udio |
| 带歌词的音乐 | Suno v4 或 Udio v1.5 |
| 音效 / 拟音 | AudioCraft 2、ElevenLabs SFX 或 Stable Audio Open |
| 实时语音智能体 | GPT-4o realtime 或 Gemini Live |
| 开源权重音乐研究 | MusicGen 3.3B、Stable Audio Open 1.0、AudioLDM 2 |
| 配音 / 翻译 | HeyGen、ElevenLabs Dubbing |

## 交付它

保存 `outputs/skill-audio-brief.md`。该技能接收一份音频简报（任务、时长、风格、声音、许可证），并输出：模型 + 托管方案、提示词格式（流派标签、风格描述符、结构标记）、编解码器 + 生成器 + 声码器链路、种子协议，以及评估计划（MOS / CLAP 分数 / TTS 的 CER / 用户 A/B）。

## 练习

1. **简单。** 运行 `code/main.py` 并显式设置风格。验证生成序列符合该风格的模式。
2. **中等。** 添加延迟并行解码：模拟 2 条 token 流，它们必须保持 1 步偏移。训练一个联合预测器。
3. **困难。** 使用 HuggingFace transformers 在本地运行 MusicGen-small。用三个不同提示词生成 10 秒片段；做 A/B 测试评估风格遵循度。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|-----------------|-----------------------|
| Codec | “神经压缩” | 音频编码器 / 解码器；典型输出是 50-75 Hz token。 |
| RVQ | “Residual VQ” | K 个量化器的级联；每个都对上一层的残差建模。 |
| Token | “一个编解码器符号” | 码本中的离散索引；典型大小为 1024 或 2048。 |
| Delayed parallel | “偏移码本” | 以交错偏移发出 K 条 token 流以缩短序列长度。 |
| Flow matching | “2024 年音频赢家” | 扩散的更直路径替代方案；采样更快。 |
| Voice prompt | “3 秒样本” | 说话人嵌入或 token 前缀，用于引导克隆声音。 |
| Mel spectrogram | “那个可视图” | 对数幅度感知频谱图；许多 TTS 系统会使用。 |
| Vocoder | “Mel 转波形” | 把梅尔频谱图转换回音频的神经组件。 |

## 生产备注：音频是流式传输问题

音频是用户期望“边生成边到达”的唯一输出模态，而不是一次性全部返回。用生产术语说，这意味着 TPOT（Time Per Output Token，每输出 token 时间）很重要，因为目标吞吐量是用户的聆听速度，而不是阅读速度。对以约 75 tokens/second（Encodec）tokenize 的 16kHz 音频，服务器必须为每个用户生成 ≥75 tokens/sec 才能保持平滑播放。

两个架构后果：

- **流匹配音频模型无法轻易流式化。** Stable Audio 2.5 和 AudioCraft 2 会一次性渲染固定长度片段。要流式传输，你需要把片段分块并重叠边界——类似滑动窗口扩散——相较编解码器 AR 模型会增加 100-300ms 延迟开销。

如果产品是“实时语音聊天”或“实时音乐续写”，选择编解码器 AR 路线。如果是“提交后渲染 30 秒片段”，流匹配在质量和总延迟上胜出。

## 延伸阅读

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — 编解码器标准。
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — 第一个被广泛使用的神经音频编解码器。
- [Kumar et al. (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) — DAC。
- [Wang et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) — VALL-E。
- [Copet et al. (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) — MusicGen。
- [Liu et al. (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) — AudioLDM 2。
- [Stability AI (2024). Stable Audio 2.5](https://stability.ai/news/introducing-stable-audio-2-5) — 2025 年使用流匹配的文本生成音乐模型。
