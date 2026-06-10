# 多模态智能体与计算机使用（Capstone）

> 2026 年的前沿产品是一个多模态智能体，它能够读取屏幕截图、点击按钮、浏览网页界面、填写表单，并完成端到端的工作流。SeeClick 和 CogAgent（2024）证明了 GUI grounding 的基础能力。Ferret-UI 扩展到了移动端。ChartAgent 引入了针对图表的视觉工具使用。VisualWebArena 和 AgentVista（2026）是前沿模型追逐的基准测试——即使是 Gemini 3 Pro 和 Claude Opus 4.7，在 AgentVista 的困难任务上也只能获得约 30% 的分数。本 Capstone 整合了第 12 阶段的所有线索：感知（高分辨率 VLM）、推理（带工具使用的 LLM）、grounding（坐标输出）、长程记忆和评估。

**类型：** Capstone
**语言：** Python（标准库，action schema + agent loop 骨架）
**前置条件：** Phase 12 · 05（LLaVA），Phase 12 · 09（Qwen-VL JSON），Phase 14（Agent Engineering）
**时间：** 约 240 分钟

## 学习目标

- 设计多模态智能体循环：感知 → 推理 → 行动 → 观察 → 重复。
- 构建 GUI grounding 输出 schema（点击坐标、输入文本、滚动、拖拽），使 VLM 能够以 JSON 格式输出。
- 对比仅截图智能体、可访问性树智能体和混合智能体。
- 在一个小型的 VisualWebArena 切片上设置多模态智能体基准评估。

## 问题描述

一个预订网站工作流："帮我找一趟 4 月 15 日飞往东京的航班，靠过道座位，价格低于 800 美元，预订它。"

多模态智能体需要：

1. 截取浏览器屏幕截图。
2. 将截图 + URL + 目标解析为计划。
3. 输出结构化动作：点击（在 x,y 处）、输入 "Tokyo"（在元素 E 处）、向下滚动、选择（单选按钮）。
4. 将动作应用到浏览器。
5. 观察新状态（下一张截图）。
6. 重复直到任务完成。

每一步都是一次多模态 VLM 调用。VLM 输出必须是可解析的 JSON。错误会在步骤间累积，因此恢复机制很重要。

## 核心概念

### GUI grounding —— 基础能力

GUI grounding 是指：给定一张屏幕截图和自然语言指令，输出要点击的 (x, y) 坐标（或其他动作）。

SeeClick（arXiv:2401.10935）是首个大规模开源成果：在合成 + 真实 GUI 数据上微调 VLM，将坐标作为纯文本 token 输出。可行。

CogAgent（arXiv:2312.08914）增加了 1120×1120 高分辨率编码，用于密集界面。得分：网页导航约 84%。

Ferret-UI（arXiv:2404.05719）专注于移动端界面，集成 iOS 可访问性数据。

输出格式通常为 JSON：

```json
{"action": "click", "x": 384, "y": 220, "element_desc": "Search button"}
```

`element_desc` 有助于恢复：如果坐标在不同截图间发生偏移，语义提示可以让系统重新进行 grounding。

### Action schemas

典型的 action schema 包含 6–10 种动作类型：

- `click`: (x, y)
- `type`: (text, x?, y?)
- `scroll`: (direction, amount)
- `drag`: (x0, y0, x1, y1)
- `select`: (option_index)
- `hover`: (x, y)
- `navigate`: (url)
- `wait`: (ms)
- `done`: (success, explanation)

智能体每步输出一个动作。浏览器包装器执行该动作并返回新状态。

### 仅截图 vs 可访问性树

两种输入模式：

- 仅截图：完整图像，无结构信息。最通用；适用于任何应用。
- 可访问性树：结构化 DOM / iOS 可访问性信息。Grounding 更可靠；在树可用的情况下工作良好。
- 混合：两者结合，树作为可靠的原子动作 grounder，截图用于语义上下文。

生产环境智能体在可能的情况下使用混合模式。浏览器自动化（Selenium + 可访问性）始终有树；桌面应用有时也有。

### 长程记忆

一个 20 步的工作流会生成 20 张截图。VLM 的上下文很快就会被填满。三种压缩策略：

- Summary-chain：每 5 步后总结已发生的事情，丢弃旧截图。
- Skip-frame：保留第一张、最后一张和每第 3 张截图。
- 工具记录日志：执行动作，保留文本日志记录已完成的操作；不再回看旧截图。

Claude 的 computer-use API 使用日志模式。更简单，更可靠。

### 视觉工具使用

ChartAgent（arXiv:2510.04514）引入了针对图表理解的视觉工具使用：裁剪、缩放、OCR、调用外部检测。智能体可以输出 "裁剪到区域 (100, 200, 300, 400) 然后调用 OCR" 作为工具调用。工具返回文本；VLM 继续推理。

这种模式可以泛化：set-of-mark prompting、区域标注和外部检测工具都适用于相同的"输出工具调用，接收结构化响应"的 schema。

### 2026 年基准测试

- ScreenSpot-Pro。约 1k 张网页截图的 GUI grounding。开源 SOTA Qwen2.5-VL-72B 约 85%。前沿模型约 90%。
- VisualWebArena。端到端网页任务（购物、论坛、分类信息）。开源 SOTA 约 20%。Gemini 3 Pro 约 27%。
- AgentVista（arXiv:2602.23166）。2026 年最难的基准测试。跨 12 个领域的真实工作流。前沿模型得分 27–40%；开源模型 10–20%。
- WebArena / WebShop。较旧的基准测试；已被前沿模型饱和。

### 为什么它仍然困难

智能体性能瓶颈：

1. 精细尺度的视觉 grounding。"点击那个小 X" 在移动分辨率下经常失败。
2. 长程规划。执行 10 个动作后，智能体会偏离目标。
3. 错误恢复。当点击失败（按错按钮）时，检测 + 恢复很少在训练数据中出现。
4. 跨页面上下文。在标签页之间跳转或处理长表单时会丢失状态。

研究方向：记忆架构、显式重新规划、多模态验证（动作成功的截图匹配）。

### Capstone 构建任务

Capstone 任务：构建一个 computer-use 智能体，能够：

1. 读取预订网站模拟页面的 HTML + 截图。
2. 规划多步序列：搜索 → 选择 → 填写表单 → 提交。
3. 输出与 action schema 匹配的 JSON 动作。
4. 在固定的 10 任务切片上进行评估。

本课程提供易于扩展为真实浏览器的脚手架代码。

## 使用它

`code/main.py` 是 Capstone 脚手架：

- Action schema JSON 定义（10 个动作）。
- 模拟浏览器状态作为字典。
- 智能体循环骨架：接收状态、输出动作、应用、循环。
- 10 任务迷你基准测试（合成页面）用于衡量端到端成功率。
- 动作失败时的错误恢复钩子。

## 交付它

本课程产出 `outputs/skill-multimodal-agent-designer.md`。给定一个 computer-use 产品（领域、动作集、评估目标），设计完整的智能体循环、记忆策略、grounding 模式和预期基准分数。

## 练习

1. 扩展 action schema，添加一个 `screenshot_region` 工具（裁剪 + 缩放）。哪些任务会受益？

2. 阅读 AgentVista（arXiv:2602.23166）。描述最困难的任务类别，以及为什么前沿模型仍然失败。

3. 长程记忆压缩：设计一个 summary-chain，保持活跃的截图不超过 4 张，日志数量不限。

4. 构建一个错误恢复钩子：当动作失败（未找到按钮）时，智能体接下来应该做什么？

5. 对比仅截图的 Claude 4.7 和混合截图 + 可访问性树的 Qwen2.5-VL 在 10 个网页任务上的表现。哪些任务上谁获胜？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| GUI grounding | "点击坐标" | 模型在截图上为目标指令输出 (x,y) 坐标 |
| Action schema | "工具定义" | 有效动作的 JSON 描述（click、type、scroll、drag） |
| Accessibility tree | "结构化 DOM" | 来自浏览器/iOS API 的机器可读 UI 层级结构 |
| Hybrid agent | "截图 + 树" | 同时使用图像和结构化信息；比单独使用任一更可靠 |
| Visual tool use | "缩放/裁剪/检测" | 智能体在计划中途调用外部视觉工具（OCR、检测） |
| Summary-chain | "记忆压缩" | 定期文本摘要替代长截图历史 |
| VisualWebArena | "端到端网页基准" | 2024 年端到端网页任务基准测试 |
| AgentVista | "2026 困难基准" | 12 个领域的真实工作流；即使是 Gemini 3 Pro 也只获得约 30% |

## 延伸阅读

- [Cheng et al. — SeeClick (arXiv:2401.10935)](https://arxiv.org/abs/2401.10935)
- [Hong et al. — CogAgent (arXiv:2312.08914)](https://arxiv.org/abs/2312.08914)
- [You et al. — Ferret-UI (arXiv:2404.05719)](https://arxiv.org/abs/2404.05719)
- [ChartAgent (arXiv:2510.04514)](https://arxiv.org/abs/2510.04514)
- [Koh et al. — VisualWebArena (arXiv:2401.13649)](https://arxiv.org/abs/2401.13649)
- [AgentVista (arXiv:2602.23166)](https://arxiv.org/abs/2602.23166)
