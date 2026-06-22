# 语音智能体：Pipecat 与 LiveKit

> 语音智能体在 2026 年已成为正式的生产类别。Pipecat 提供基于帧的 Python 管道（VAD → STT → LLM → TTS → 传输层）。LiveKit Agents 通过 WebRTC 将 AI 模型桥接到用户。生产环境的端到端延迟目标在高端架构中为 450–600ms。

**类型：** 学习
**语言：** Python（标准库）
**前置条件：** 第 14 阶段 · 01（智能体循环）、第 14 阶段 · 12（工作流模式）
**时间：** 约 60 分钟

## 学习目标

- 描述 Pipecat 基于帧的管道：DOWNSTREAM（源→汇）和 UPSTREAM（控制流）。
- 列出典型语音管道阶段以及 Pipecat 支持的传输层。
- 解释 LiveKit Agents 的两种语音智能体类（MultimodalAgent、VoicePipelineAgent）及各自的适用场景。
- 概述 2026 年的生产延迟预期以及它们如何影响架构选择。

## 问题背景

语音智能体不是简单地在文本循环上外挂 TTS。延迟预算非常紧张（约 600ms），部分音频是默认行为，轮次检测本身就是一个模型，而传输层从电话 SIP 到 WebRTC 各不相同。你要么构建基于帧的管道（Pipecat），要么依赖平台（LiveKit）。

## 核心概念

### Pipecat（pipecat-ai/pipecat）

- 基于帧的 Python 管道框架。
- `Frame` → `FrameProcessor` 链。
- 两种流方向：
  - **DOWNSTREAM** — 源 → 汇（音频输入，TTS 输出）。
  - **UPSTREAM** — 反馈与控制（取消、指标、打断）。
- `PipelineTask` 通过事件（`on_pipeline_started`、`on_pipeline_finished`、`on_idle_timeout`）和观察者管理生命周期，用于指标/追踪/RTVI。

典型管道：

```
VAD (Silero) → STT → LLM（上下文在用户/助手之间交替） → TTS → 传输层
```

传输层：Daily、LiveKit、SmallWebRTCTransport、FastAPI WebSocket、WhatsApp。

Pipecat Flows 添加了结构化对话（状态机）。Pipecat Cloud 是托管运行时。

### LiveKit Agents（livekit/agents）

- 通过 WebRTC 将 AI 模型桥接到用户。
- 核心概念：`Agent`、`AgentSession`、`entrypoint`、`AgentServer`。
- 两种语音智能体类：
  - **MultimodalAgent** — 通过 OpenAI Realtime 或同等服务直接处理音频。
  - **VoicePipelineAgent** — STT → LLM → TTS 级联；提供文本级控制。
- 基于 Transformer 模型的语义轮次检测。
- 原生 MCP 集成。
- 通过 SIP 实现电话接入。
- 通过 LiveKit Inference 支持 50+ 模型，无需 API 密钥；另有 200+ 模型通过插件支持。

### 商业平台

Vapi（在优化的高端架构上约 450–600ms）和 Retell（在 180 次测试通话中端到端约 600ms）构建于这些框架之上。当你想要一个无需 WebRTC 团队的托管语音方案时，可以选择这些平台。

### 常见错误模式

- **没有打断处理。** 用户中断时智能体仍在说话。在 Pipecat 中需要 UPSTREAM 取消帧，LiveKit 中需要等效机制。
- **忽略 STT 置信度。** 低置信度的转录结果被当作事实喂给 LLM。应设置置信度阈值或请求用户确认。
- **TTS 句中断截断。** 当管道在语音中途取消时，TTS 需要知道这一情况或切断音频。
- **忽略延迟预算。** 每个组件增加 50–200ms。上线前要汇总整条链路的延迟。

### 2026 年典型延迟

- VAD：20–60ms
- STT 部分结果：100–250ms
- LLM 首 token：150–400ms
- TTS 首音频：100–200ms
- 传输层 RTT：30–80ms

端到端 450–600ms 为高端水平。800–1200ms 为常见水平。超过 1500ms 会感觉卡顿。

## 动手实践

`code/main.py` 是一个基于帧的示例管道，包含：

- `Frame` 类型（音频、转录文本、文本、TTS 音频、控制）。
- `Processor` 接口，包含 `process(frame)` 方法。
- 五阶段管道（VAD → STT → LLM → TTS → 传输层）作为脚本化的处理器。
- 一个 UPSTREAM 取消帧用于演示打断。

运行方式：

```
python3 code/main.py
```

追踪输出展示了正常流程以及一个在语音中途停止 TTS 的打断取消。

## 实际应用

- **Pipecat** — 完全控制，自定义处理器，Python 优先，可插拔提供商。
- **LiveKit Agents** — WebRTC 优先的部署和电话接入。
- **Vapi / Retell** — 无需 WebRTC 团队的托管语音智能体。
- **OpenAI Realtime / Gemini Live** — 直接音频输入/输出（MultimodalAgent）。

## 交付成果

`outputs/skill-voice-pipeline.md` 搭建了一个 Pipecat 风格的语音管道骨架，包含 VAD + STT + LLM + TTS + 传输层及打断处理。

## 练习

1. 为示例管道添加指标观察者：统计每秒每个阶段的帧数。延迟在哪个阶段积累最多？
2. 实现置信度门控 STT：低于阈值时请求用户"请再说一遍"。
3. 添加语义轮次检测：简单规则——如果转录文本以"?"结尾，则认为轮次结束。
4. 阅读 Pipecat 的传输层文档。将 stdlib 传输层替换为 SmallWebRTCTransport 配置（存根）。
5. 对比 OpenAI Realtime 与 STT+LLM+TTS 级联在相同查询上的表现。文本级控制带来了多大的延迟开销？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Frame（帧） | "事件" | 管道中有类型的数据单元（音频、转录文本、文本、控制） |
| Processor（处理器） | "管道阶段" | 包含 process(frame) 的处理器 |
| DOWNSTREAM | "正向流" | 从源到汇：音频输入，语音输出 |
| UPSTREAM | "反馈流" | 控制流：取消、指标、打断 |
| VAD | "语音活动检测" | 检测用户何时在说话 |
| 语义轮次检测 | "智能轮次结束" | 基于模型判断用户是否说完 |
| MultimodalAgent | "直接音频智能体" | 音频进，音频出；中间无文本 |
| VoicePipelineAgent | "级联智能体" | STT + LLM + TTS；文本级控制 |

## 扩展阅读

- [Pipecat 文档](https://docs.pipecat.ai/getting-started/introduction) — 基于帧的管道、处理器、传输层
- [LiveKit Agents 文档](https://docs.livekit.io/agents/) — WebRTC + 语音原语
- [Vapi](https://vapi.ai/) — 托管语音平台
- [Retell AI](https://www.retellai.com/) — 托管语音，延迟基准测试
