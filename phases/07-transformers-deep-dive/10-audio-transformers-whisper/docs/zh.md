# 音频 Transformer — Whisper 架构

> 音频是频率随时间的图像。Whisper 是一个吃梅尔频谱图并说话的 ViT。

**类型:** 学习
**语言:** Python
**前置知识:** 第 7 阶段 · 05（完整 Transformer）、第 7 阶段 · 08（编码器-解码器）、第 7 阶段 · 09（ViT）
**时间:** 约 45 分钟

## 问题所在

Whisper (OpenAI, Radford et al. 2022) 之前，最先进的自动语音识别 (ASR) 意味着 wav2vec 2.0 和 HuBERT——自监督特征提取器加微调头。高质量，昂贵的数据管道，领域脆弱。多语言语音识别需要每个语系单独的模型。

Whisper 下了三个赌注：

1. **在所有数据上训练。** 680,000 小时从互联网抓取的弱标注音频，覆盖 97 种语言。没有干净的学术语料。没有音素标签。
2. **多任务单一模型。** 一个解码器联合训练转录、翻译、语音活动检测、语言识别和时间戳，通过任务 token。
3. **标准编码器-解码器 Transformer。** 编码器消费 log 梅尔频谱图。解码器自回归生成文本 token。没有声码器、没有 CTC、没有 HMM。

结果：Whisper large-v3 在口音、噪声和零干净标注数据的语言上都是稳健的。它是 2026 年每个开源语音助手和大多数商业语音助手的默认语音前端。

## 核心概念

![Whisper 流程: 音频 → 梅尔 → 编码器 → 解码器 → 文本](../assets/whisper.svg)

### 第一步——重采样 + 加窗

16 kHz 音频。裁剪/填充到 30 秒。计算 log 梅尔频谱图：80 个梅尔 bin，10 ms 步长 → ~3,000 帧 × 80 特征。这是 Whisper 看到的"输入图像"。

### 第二步——卷积 stem

两个核为 3、步长为 2 的 Conv1D 层将 3,000 帧减少到 1,500。在不增加大量参数的情况下将序列长度减半。

### 第三步——编码器

24 层（large 版）Transformer 编码器处理 1,500 个时间步。正弦位置编码、自注意力、GELU FFN。产生 1,500 × 1,280 隐藏状态。

### 第四步——解码器

24 层 Transformer 解码器。它从 BPE 词表自回归生成 token，该词表是 GPT-2 的超集，加上几个音频特定的特殊 token。

### 第五步——任务 token

解码器提示以控制 token 开头，告诉模型做什么：

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

或

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

模型在这种约定上训练。你通过前缀控制任务。2026 年指令微调的等价物，但应用于语音。

### 第六步——输出

束搜索（宽度 5）加 log 概率阈值。当 `<|notimestamps|>` token 不存在时，每 0.02 秒音频预测时间戳。

### Whisper 大小

| 模型 | 参数 | 层数 | d_model | 头数 | VRAM (fp16) |
|------|------|------|---------|------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB（4 层解码器） |

Large-v3-turbo (2024) 将解码器从 32 层减到 4。解码速度快 8 倍，WER 回退 <1 点。这种解码速度解锁是 Whisper-turbo 成为 2026 年实时语音助手默认的原因。

### Whisper 做不到的事

- 无说话人分离（谁在说话）。搭配 pyannote 使用。
- 无原生实时流式——30 秒窗口是固定的。现代包装器（`faster-whisper`、`WhisperX`）通过 VAD + 重叠添加流式。
- 无超过 30 秒的长形式上下文，除非外部分块。实际中效果良好，因为人类语音很少需要长上下文进行转录。

### 2026 年格局

| 任务 | 模型 | 备注 |
|------|------|------|
| 英语 ASR | Whisper-turbo、Moonshine | Moonshine 在边缘设备上快 4 倍 |
| 多语言 ASR | Whisper-large-v3 | 97 种语言 |
| 流式 ASR | faster-whisper + VAD | 150 ms 延迟目标可实现 |
| TTS | Piper、XTTS-v2、Kokoro | 编码器-解码器模式，但 Whisper 形状 |
| 音频 + 语言 | AudioLM、SeamlessM4T | 文本 token + 音频 token 在一个 Transformer 中 |

## 动手实现

参见 `code/main.py`。我们不训练 Whisper——我们构建 log 梅尔频谱图管道 + 任务 token 提示格式化器。这些是你在生产中实际接触的部分。

### 第一步：合成音频

生成 440 Hz、16 kHz 采样的 1 秒正弦波。16,000 个样本。

### 第二步：log 梅尔频谱图（简化版）

完整梅尔频谱图需要 FFT。我们做简化的分帧 + 每帧能量版本，展示管道而不需要 `librosa`：

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

帧 = 25 ms，hop = 10 ms。匹配 Whisper 的窗口化。每帧能量代表梅尔 bin 用于教学。

### 第三步：填充到 30 秒

Whisper 总是处理 30 秒的块。将频谱图填充（或裁剪）到 3,000 帧。

### 第四步：构建提示 token

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

这就是整个任务控制面。4 个 token 的前缀。

## 使用场景

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

更快，OpenAI 兼容：

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**2026 年何时选择 Whisper：**

- 用一个模型做多语言 ASR。
- 噪声多样音频的稳健转录。
- 研究/原型 ASR——最快起点。

**何时选择其他：**

- 边缘设备超低延迟流式——Moonshine 在匹配质量下击败 Whisper。
- 需要 <200 ms 的实时对话 AI——专用流式 ASR。
- 说话人分离——Whisper 不做这个；搭配 pyannote。

## 交付使用

参见 `outputs/skill-asr-configurator.md`。该技能为新语音应用选择 ASR 模型、解码参数和预处理管道。

## 练习

1. **简单。** 运行 `code/main.py`。确认 16 kHz 信号在 10 ms hop 下的帧数约 100。30 秒：约 3,000 帧。
2. **中等。** 使用 `numpy.fft` 构建完整 log 梅尔频谱图。验证 80 个梅尔 bin 与 `librosa.feature.melspectrogram(n_mels=80)` 在数值误差内匹配。
3. **困难。** 实现流式推理：将音频分块为 10 秒窗口、2 秒重叠，对每个块运行 Whisper，合并转录。在 5 分钟播客样本上测量词错误率 vs 单次传递。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 梅尔频谱图 | "音频图像" | 2D 表示：一个轴是频率 bin，另一个是时间帧；每单元 log 缩放能量。 |
| Log-mel | "Whisper 看到的" | 通过 log 的梅尔频谱图；近似人类对响度的感知。 |
| 帧 | "一个时间切片" | 25 ms 窗口的样本；10 ms 步长重叠。 |
| 任务 token | "语音的提示前缀" | 解码器提示中的特殊 token，如 `<|transcribe|>` / `<|translate|>`。 |
| 语音活动检测 (VAD) | "找到语音" | 在 ASR 之前去除静音的门控；大幅削减成本。 |
| CTC | "连接主义时间分类" | 经典 ASR 损失用于无对齐训练；Whisper 不使用它。 |
| Whisper-turbo | "小解码器，完整编码器" | large-v3 编码器 + 4 层解码器；解码快 8 倍。 |
| Faster-whisper | "生产包装器" | CTranslate2 重实现；int8 量化；比 OpenAI 参考快 4 倍。 |

## 延伸阅读

- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper 论文。
- [OpenAI Whisper repo](https://github.com/openai/whisper) — 参考代码 + 模型权重。阅读 `whisper/model.py` 从上到下查看 Conv1D stem + 编码器 + 解码器，约 400 行。
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — 步骤 5-6 描述的束搜索 + 任务 token 逻辑在这里；500 行，完全可读。
- [Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) — 先驱；在某些设置中仍是 SOTA 特征。
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 生产包装器，比参考快 4 倍。
- [Jia et al. (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) — 2024 年边缘友好 ASR，Whisper 形状但更小。
- [HuggingFace blog — "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper) — 标准微调配方，包括梅尔频谱图预处理器和 token-时间戳处理。
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — 完整实现（编码器、解码器、交叉注意力、生成），镜像本课的架构图。
