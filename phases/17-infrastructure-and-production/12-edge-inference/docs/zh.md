# 边缘推理 — Apple Neural Engine、Qualcomm Hexagon、WebGPU/WebLLM、Jetson

> 核心边缘约束是内存带宽，而非算力。移动 DRAM 带宽为 50-90 GB/s；数据中心 HBM3 超过 2-3 TB/s——差距达 30-50 倍。Decode 是内存瓶颈，因此差距是决定性的。2026 年的格局分为四类。Apple M4/A18 Neural Engine 峰值 38 TOPS，采用统一内存（无 CPU↔NPU 拷贝）。Qualcomm Snapdragon X Elite / 8 Gen 4 Hexagon 达到 45 TOPS。WebGPU + WebLLM 在 M3 Max 上运行 Llama 3.1 8B（Q4）约 41 tok/s（约为原生的 70-80%）；17.6k GitHub stars，OpenAI 兼容 API，移动端覆盖率约 70-75%。NVIDIA Jetson Orin Nano Super（8GB）可运行 Llama 3.2 3B / Phi-3；AGX Orin 通过 vLLM 运行 gpt-oss-20b 约 40 tok/s；Jetson T4000（JetPack 7.1）是 AGX Orin 的 2 倍性能。TensorRT Edge-LLM 支持 EAGLE-3、NVFP4、chunked prefill——在 CES 2026 上由 Bosch、ThunderSoft、MediaTek 展示。

**类型：** 学习
**语言：** Python（stdlib，用于模拟带宽瓶颈 decode 的玩具模拟器）
**前置要求：** 阶段 17 · 04（vLLM 服务内部原理）、阶段 17 · 09（生产量化）
**时间：** 约 60 分钟

## 学习目标

- 解释为什么移动 LLM 推理受内存带宽限制，而算力是次要的。
- 列举四个边缘目标（Apple ANE、Qualcomm Hexagon、WebGPU/WebLLM、NVIDIA Jetson）并为每个匹配一个用例。
- 列出 2026 年 WebGPU 覆盖差距（Firefox Android 追赶中）以及 Safari iOS 26 上线时间。
- 为每个目标选择量化格式（Core ML INT4 + FP16 用于 ANE、QNN INT8/INT4 用于 Hexagon、WebGPU Q4 用于浏览器、NVFP4 用于 Jetson Thor）。

## 问题背景

一个客户想要设备端聊天机器人：语音优先、默认私有、离线工作。在 MacBook Pro M3 Max 上，Llama 3.1 8B Q4 运行约 55 tok/s——没问题。在 iPhone 16 Pro 上，同一模型运行 3 tok/s——不行。在 Snapdragon 8 Gen 3 的中端 Android 上，7 tok/s。通过 Chrome Android v121+ 上的 WebGPU 在浏览器中，4-8 tok/s，取决于设备。

吞吐量差异不是移植问题。它是带宽差距乘以量化格式再乘以 NPU 是否可从用户空间访问。2026 年的边缘推理是四个不同的问题，需要四个不同的解决方案。

## 核心概念

### 带宽是真正的天花板

Decode 为每个 token 读取完整的权重集。一个 7B 模型的 Q4 量化版本为 3.5 GB。以 50 GB/s 的速度读取 3.5 GB 需要 70 ms——理论天花板约 14 tok/s。在 90 GB/s（高端移动 DRAM）下，天花板升至约 25 tok/s。在这个数字之下，再多算力也无济于事。

数据中心 HBM3 以 3 TB/s 的速度读取同样的 3.5 GB 仅需 1.2 ms——天花板为 830 tok/s。相同的模型，相同的权重。不同的内存子系统。

### Apple Neural Engine（M4 / A18）

- 最高 38 TOPS。统一内存（CPU 和 ANE 共享同一内存池）——无拷贝开销。
- 通过 Core ML + `.mlmodel` 编译模型访问，或通过 PyTorch 使用 Metal Performance Shaders（MPS）。
- Llama.cpp Metal 后端使用 MPS，而非直接使用 ANE；原生 ANE 需要 Core ML 转换。
- 2026 年 iOS 应用的最佳实践路径：Core ML + INT4 权重 + FP16 激活值。

### Qualcomm Hexagon（Snapdragon X Elite / 8 Gen 4）

- 最高 45 TOPS。与 SoC 中的 CPU 和 GPU 集成，但内存域独立。
- QNN（Qualcomm Neural Network）SDK 和 AI Hub 提供从 PyTorch/ONNX 的转换。
- Chat templates、Llama 3.2、Phi-3 均作为 AI Hub 上的一等工件发布。

### Intel / AMD NPU（Lunar Lake、Ryzen AI 300）

- 40-50 TOPS。软件落后于 Apple/Qualcomm；OpenVINO 在改进但仍属小众。
- 最适合 Windows ARM copilot 应用；在 AMD/Intel 桌面端原生用于本地优先场景。

### WebGPU + WebLLM

- 通过 WebGPU compute shaders 在浏览器中运行模型；无需安装。
- Llama 3.1 8B Q4 在 M3 Max 上约 41 tok/s——通过相同后端约为原生的 70-80%。
- WebLLM 在 GitHub 上有 17.6k stars；OpenAI 兼容 JS API；Apache 2.0 协议。
- 2026 年覆盖率：Chrome Android v121+、Safari iOS 26 GA、Firefox Android 仍在追赶。总体移动端覆盖率约 70-75%。

### NVIDIA Jetson 系列

- Orin Nano Super（8GB）：可运行 Llama 3.2 3B、Phi-3，达到不错的 tok/s。
- AGX Orin：通过 vLLM 运行 gpt-oss-20b 约 40 tok/s。
- Thor / T4000（JetPack 7.1）：AGX Orin 的 2 倍性能，支持 EAGLE-3 和 NVFP4。
- TensorRT Edge-LLM（2026）支持 EAGLE-3 投机解码、NVFP4 权重、chunked prefill——数据中心优化移植到边缘。

### 每个目标的量化选择

| 目标 | 格式 | 说明 |
|------|------|------|
| Apple ANE | INT4 权重 + FP16 激活值 | Core ML 转换路径 |
| Qualcomm Hexagon | QNN INT8 / INT4 | AI Hub 转换器 |
| WebGPU / WebLLM | Q4 MLC（q4f16_1） | 使用 `mlc_llm convert_weight` + 编译的 `.wasm`；不支持 GGUF |
| Jetson Orin Nano | Q4 GGUF 或 TRT-LLM INT4 | 内存瓶颈 |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Edge-LLM 路径 |

### 边缘上的长上下文陷阱

Llama 3.1 的 128K 上下文是数据中心特性。在 8 GB RAM 的手机上，4 GB 模型 + 2 GB KV cache（32K tokens）+ OS 开销 = OOM。边缘部署将上下文保持在 4K-8K，除非接受激进的 KV 量化（Q4 KV）。

### 语音是杀手级应用

语音代理对延迟敏感（首 token < 500 ms）。本地推理完全消除网络延迟。结合语音转文本（Whisper Turbo 变体可在边缘运行），边缘推理成为生产质量的语音循环。

### 需要记住的数据

- Apple M4 / A18 ANE：38 TOPS。
- Qualcomm Hexagon SD X Elite：45 TOPS。
- WebLLM M3 Max：Llama 3.1 8B Q4 约 41 tok/s。
- AGX Orin：通过 vLLM 运行 gpt-oss-20b 约 40 tok/s。
- 数据中心-边缘带宽差距：30-50 倍。
- WebGPU 移动端覆盖率：约 70-75%（Firefox Android 落后）。

## 使用

`code/main.py` 通过带宽瓶颈数学计算各边缘目标的理论 decode 吞吐量天花板。与观测基准对比，指出带宽（而非算力）是瓶颈的地方。

## 产出

本课产出 `outputs/skill-edge-target-picker.md`。根据平台（iOS/Android/浏览器/Jetson）、模型和延迟/内存预算，选择量化格式和转换管道。

## 练习

1. 运行 `code/main.py`。对于 Snapdragon 8 Gen 3（约 77 GB/s 带宽）上的 7B Q4 模型，计算 decode 天花板。与观测的 6-8 tok/s 对比——运行时是否高效？
2. Android 上的 WebGPU 需要 Chrome v121+。为旧浏览器设计回退方案——通过相同的 OpenAI 兼容 API 在服务端处理。
3. 你的 iOS 应用需要 4K 上下文流式处理。哪种模型/格式组合让你在 iPhone 16 上保持 4 GB 以下活动内存？
4. Jetson AGX Orin 以 40 tok/s 运行 gpt-oss-20b。Jetson Nano 只能运行 3B。如果你的产品面向两者，如何统一推理栈？
5. 论述"WebLLM 在 2026 年是否已可用于生产"。引用覆盖率、性能以及 Firefox Android 的差距。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| ANE | "Apple 神经引擎" | M 系列和 A 系列中的设备端 NPU；统一内存 |
| Hexagon | "Qualcomm NPU" | Snapdragon NPU；通过 QNN SDK 访问 |
| WebGPU | "浏览器 GPU" | W3C 标准化的浏览器 GPU API；Chrome/Safari 2026 |
| WebLLM | "浏览器 LLM 运行时" | MLC-LLM 项目；Apache 2.0；OpenAI 兼容 JS |
| Jetson | "NVIDIA 边缘" | Orin Nano / AGX / Thor / T4000 系列 |
| TRT Edge-LLM | "边缘 TensorRT" | 2026 年 TensorRT-LLM 的边缘移植；EAGLE-3 + NVFP4 |
| Unified memory | "共享内存池" | CPU 和 NPU 看到同一 RAM；无拷贝开销 |
| Bandwidth-bound | "内存受限" | Decode 受限于读取权重的字节/秒 |
| Core ML | "Apple 转换" | Apple 用于 ANE 原生模型的框架 |
| QNN | "Qualcomm 栈" | Qualcomm Neural Network SDK |

## 延伸阅读

- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/) — 格局和基准测试。
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/) — Orin / AGX / Thor。
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) — 2026 年边缘移植公告。
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2) — 设计和基准测试。
- [Apple Core ML](https://developer.apple.com/documentation/coreml) — ANE 原生转换。
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — 为 Hexagon 预转换的模型。
