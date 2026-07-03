# Capstone 03 — 实时语音助手（ASR 到 LLM 到 TTS）

> 一个让人感觉自然的语音助手，其端到端延迟低于 800 毫秒，能准确判断你何时停止说话，处理打断，并且能在不中断的情况下调用工具。Retell、Vapi、LiveKit Agents 和 Pipecat 在 2026 年都达到了这个标准。它们采用相同的架构：流式 ASR、轮次检测器、流式 LLM 和流式 TTS，所有组件通过 WebRTC 连接，并在每个环节设置了严格的延迟预算。构建一个这样的系统，测量 WER、MOS 和误切断率，并在丢包环境下测试。

**类型：** 毕业项目  
**语言：** Python（智能体 + 管道），TypeScript（Web 客户端）  
**前置要求：** 第 6 阶段（语音与音频），第 7 阶段（transformers），第 11 阶段（LLM 工程），第 13 阶段（工具），第 14 阶段（智能体），第 17 阶段（基础设施）  
**涉及阶段：** P6 · P7 · P11 · P13 · P14 · P17  
**时间：** 30 小时

## 问题

语音是 2025-2026 年发展最快的 AI 用户体验类别。技术门槛每个季度都在降低。OpenAI Realtime API、Gemini 2.5 Live、Cartesia Sonic-2、ElevenLabs Flash v3、LiveKit Agents 1.0 和 Pipecat 0.0.70 都将首次音频输出低于 800 毫秒作为首要目标。门槛不仅仅是延迟，而是交互体验：不打断用户，不被用户打断，从句子中间的中断中恢复，在对话中调用工具而不中断音频流，以及在抖动的移动网络中保持稳定。

你无法通过串联三个 REST 调用来达到这个目标。架构必须是端到端的流水线式流处理。构建它，故障模式就会显现：为电话音频调优的 VAD 在背景电视声音下误触发，轮次检测器等待永远不来的标点符号，TTS 在输出前缓冲 400 毫秒。本毕业项目的目标是在负载下逐个修复这些问题，并发布一份延迟与质量报告。

## 概念

管道有五个流式阶段：**音频输入**（来自浏览器或 PSTN 的 WebRTC）、**ASR**（来自 Deepgram Nova-3 或 faster-whisper 的流式部分转录）、**轮次检测**（VAD 加上一个小型轮次检测器模型，读取部分转录以获取完成线索）、**LLM**（一旦轮次被判定完成就开始流式输出 token）、**TTS**（在第一个 LLM token 后约 200 毫秒内开始流式输出音频）。

三个横切关注点。**打断**：当用户在助手说话时开始说话，TTS 会立即取消，ASR 立即接管。**工具使用**：对话中的函数调用（天气、日历）必须在侧通道上运行，不阻塞音频；如果延迟超过 300 毫秒，助手会预填充一个确认 token（"稍等..."）。**背压**：在丢包情况下，部分转录会被保留，VAD 提高语音门限阈值，助手避免在未确认的消息上说话。

测量标准是量化的。在 Hamming VAD 基准测试中，15 dB 信噪比下 WER 低于 8%。在 100 次测量通话中，首次音频输出 p50 低于 800 毫秒。误切断率低于 3%。TTS 的 MOS 高于 4.2。在单个 g5.xlarge 实例上支持 50 个并发通话。这些数字就是交付物。

## 架构

```
浏览器 / Twilio PSTN
        |
        v
   WebRTC / SIP 边缘
        |
        v
  LiveKit Agents 1.0  (或 Pipecat 0.0.70)
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5         轮次检测器         侧通道
(Deepgram         (Silero)          (LiveKit)        工具
 Nova-3 /         语音门限        完成度评分        (天气、
 Whisper-v3)      每 20ms          基于部分转录     日历)
   |                   |              |
   +--------+----------+--------------+
            v
        LLM (流式)
     GPT-4o-realtime / Gemini 2.5 Flash /
     级联 Claude Haiku 4.5
            |
            v
        TTS 流式
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     音频返回给调用者
            |
            v
   OpenTelemetry 语音追踪 -> Langfuse
```

## 技术栈

- 传输层：LiveKit Agents 1.0 (WebRTC) 加 Twilio PSTN 网关；Pipecat 0.0.70 作为备选框架
- ASR：Deepgram Nova-3（流式，首次部分转录低于 300 毫秒）或 faster-whisper Whisper-v3-turbo 自托管
- VAD：Silero VAD v5 加 LiveKit 轮次检测器（读取部分转录的小型 transformer）
- LLM：OpenAI GPT-4o-realtime（紧密集成），Gemini 2.5 Flash Live，或级联 Claude Haiku 4.5（流式补全，独立音频路径）
- TTS：Cartesia Sonic-2（最低首字节延迟），ElevenLabs Flash v3，或开源 Orpheus 自托管
- 工具：FastMCP 侧通道用于天气/日历/预订；如果工具调用超过 300 毫秒，助手会预发出填充词
- 可观测性：OpenTelemetry 语音 span，Langfuse 语音追踪带音频回放
- 部署：单个 g5.xlarge（24GB 显存）用于自托管 Whisper + Orpheus；托管 API 用于最低延迟

## 构建它

1. **WebRTC 会话。** 搭建 LiveKit 房间和流式传输麦克风音频的 Web 客户端。在服务器端，附加一个加入房间的智能体工作器。

2. **ASR 流式处理。** 将 20ms PCM 帧馈送到 Deepgram Nova-3（或 GPU 上的 faster-whisper）。订阅部分和最终转录。记录每个部分转录的延迟。

3. **VAD 和轮次检测器。** 在帧流上运行 Silero VAD v5。在语音结束事件时，对最新的部分转录触发 LiveKit 轮次检测器。只有当 VAD 检测到 500 毫秒静音且轮次检测器完成度评分 > 0.6 时，才判定为"轮次完成"。

4. **LLM 流式处理。** 轮次完成后，使用当前对话和最终转录开始 LLM 调用。流式输出 token。在第一个 token 时，交给 TTS。

5. **TTS 流式处理。** Cartesia Sonic-2 流式返回音频块。第一个块必须在第一个 LLM token 后 200 毫秒内离开服务器。将音频块发送到 LiveKit 房间；客户端通过 WebRTC 抖动缓冲区播放。

6. **打断。** 当 VAD 在 TTS 播放期间检测到新的用户语音时，立即取消 TTS 流，丢弃剩余的 LLM 输出，并重新启动 ASR。发布一个 `tts_canceled` span。

7. **工具侧通道。** 将天气和日历注册为函数调用工具。调用时，并发执行调用；如果 300 毫秒内未解决，让 LLM 发出"稍等，我查一下"作为填充词；工具返回后继续。

8. **评估框架。** 录制 100 次通话。计算 WER（对比保留的转录）、误切断率（用户在句子中间时 TTS 被取消）、首次音频输出 p50、TTS MOS（人工或 NISQA）以及抖动丢失测试（丢弃 3% 的数据包）。

9. **负载测试。** 在单个 g5.xlarge 上使用合成调用者驱动 50 个并发通话。测量持续的首次音频输出 p95。

## 使用它

```
调用者："明天东京天气怎么样"
[asr  ] 部分转录 @280ms: "明天东京"
[asr  ] 部分转录 @540ms: "明天东京天气怎么样"
[turn ] 完成度评分 0.82 @820ms；判定完成
[llm  ] 第一个 token @960ms
[tool ] weather.tokyo tomorrow -> 68/52 多云 @1140ms
[tts  ] 首次音频输出 @1040ms: "明天东京天气多云..."
轮次延迟：1040ms 用户停止说话 -> 音频输出
```

## 交付它

`outputs/skill-voice-agent.md` 是交付物。给定一个领域（客户支持、排程或自助终端），它搭建一个 LiveKit 智能体，包含针对测量标准调优的 ASR/VAD/LLM/TTS 管道。评分标准：

| 权重 | 标准 | 衡量方式 |
|:---:|------|----------|
| 25 | 端到端延迟 | 100 次录制通话中 p50 首次音频输出低于 800 毫秒 |
| 20 | 轮次质量 | Hamming VAD 基准测试中误切断率低于 3% |
| 20 | 工具使用正确性 | 对话中的工具调用返回正确数据而不中断音频 |
| 20 | 丢包下的可靠性 | 注入 3% 丢包时的 WER 和轮次稳定性 |
| 15 | 评估框架完整性 | 使用公共配置的可复现测量 |
| **100** | | |

## 练习

1. 将 Deepgram Nova-3 替换为 g5.xlarge 上的 faster-whisper v3 turbo。测量延迟和 WER 差距。找出 CPU 与 GPU 决策重要的地方。

2. 添加打断仲裁策略：当用户在工具调用期间打断时，智能体应该怎么做？比较三种策略（硬取消、完成工具然后停止、将下一轮排队）。

3. 运行对抗性轮次检测器测试：给用户在句子中间添加长停顿。调优 VAD 静音阈值和轮次检测器评分阈值，以在不超过 900 毫秒的情况下获得最低误切断率。

4. 通过 Twilio 在 PSTN 上部署相同的智能体。比较 PSTN 和 WebRTC 的首次音频输出。解释抖动缓冲区和编解码器差异。

5. 添加非英语语言（日语、西班牙语）的语音活动检测。测量 Silero VAD v5 的误触发率与语言特定微调的对比。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 轮次检测 | "话语结束" | 给定 VAD 静音和部分转录，判定用户已说完话的分类器 |
| 打断 | "中断处理" | 当 VAD 检测到新用户语音时，在 TTS 播放中取消 |
| 首次音频输出 | "延迟" | 从用户停止说话到第一个音频包离开服务器的时间 |
| VAD | "语音门限" | 将音频帧分类为语音或静音的模型；Silero VAD v5 是 2026 年默认选项 |
| 抖动缓冲区 | "音频平滑" | 客户端缓冲区，短暂保存数据包以吸收网络波动 |
| 填充词 | "确认 token" | 当工具响应慢时，智能体发出的短语以避免沉默 |
| MOS | "平均意见分" | 感知语音质量评分；NISQA 是自动化替代方案 |

## 延伸阅读

- [LiveKit Agents 1.0](https://github.com/livekit/agents) — 参考 WebRTC 智能体框架
- [Pipecat](https://github.com/pipecat-ai/pipecat) — 备选的 Python 优先流式智能体框架
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 集成语音模型参考
- [Deepgram Nova-3 文档](https://developers.deepgram.com/docs) — 流式 ASR 参考
- [Silero VAD v5](https://github.com/snakers4/silero-vad) — VAD 参考模型
- [Cartesia Sonic-2](https://docs.cartesia.ai) — 低延迟 TTS 参考
- [Retell AI 架构](https://docs.retellai.com) — 生产级语音助手架构
- [Vapi.ai 生产堆栈](https://docs.vapi.ai) — 备选生产参考