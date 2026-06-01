# 基准测试：WebArena 和 OSWorld

> WebArena 通过四个自托管应用测试网页智能体能力。OSWorld 通过 Ubuntu、Windows、macOS 测试桌面智能体能力。在发布时（2023-2024），两者都显示了顶级智能体与人类之间的巨大差距。差距正在缩小；失败模式没有改变。

**类型：** 学习
**语言：** Python（标准库）
**前置要求：** Phase 14 · 19（SWE-bench、GAIA）
**时间：** 约 60 分钟

## 学习目标

- 描述 WebArena 的四个自托管应用以及为什么基于执行的评估很重要。
- 解释 OSWorld 为什么使用真实 OS 截图而不是无障碍 API。
- 说出 OSWorld 的两个主要失败模式：GUI 定位和操作知识。
- 总结 OSWorld-G 和 OSWorld-Human 在基准测试基础上增加了什么。

## 问题

通用智能体可以调用工具。它们能驱动浏览器完成 20 次点击来完成购物结算吗？它们能仅使用键盘和鼠标配置 Linux 机器吗？这些就是 WebArena 和 OSWorld 回答的问题。

## 核心概念

### WebArena（Zhou 等人，ICLR 2024）

- 812 个长时域任务，分布在四个自托管网页应用：一个购物网站、一个论坛、一个类 GitLab 开发工具、一个商业 CMS。
- 附加工具：地图、计算器、草稿本。
- 评估基于 gym API 的执行方式——订单是否已提交、issue 是否已关闭、CMS 页面是否已更新？
- 发布时：最好的 GPT-4 智能体达到 14.41% 成功率 vs 人类 78.24%。

自托管的方式很关键——因为目标应用被固定且可复现，所以基准测试不会出现不稳定的情况。

### 扩展

- **VisualWebArena** — 视觉定位任务，成功取决于解读图像（截图作为一等公民的观察方式）。
- **TheAgentCompany**（2024 年 12 月）— 增加了终端 + 编码；更像是真实的远程工作环境。

### OSWorld（Xie 等人，NeurIPS 2024）

- 369 个真实计算机任务，跨 Ubuntu、Windows、macOS。
- 对真实应用的自由键盘和鼠标控制。
- 1920×1080 截图作为观察方式。
- 发布时：最好的模型 12.24% vs 人类 72.36%。

### 主要失败模式

1. **GUI 定位。** 像素 → 元素映射。模型难以在 1920×1080 中可靠地定位 UI 元素。
2. **操作知识。** 哪个菜单有设置项、哪个键盘快捷键、哪个偏好面板。人类花数年积累的知识长尾。

### 后续工作

- **OSWorld-G** — 564 个样本的定位套件 + Jedi 训练集。将定位与规划分离，以便独立衡量。
- **OSWorld-Human** — 人工策划的黄金动作序列。显示顶级智能体使用了 1.4-2.7 倍于必要步骤的步数（轨迹效率差距）。

### 为什么这很重要

Claude computer use、OpenAI CUA、Gemini 2.5 Computer Use（第 21 课）都在以 WebArena 和 OSWorld 形成的工作负载上训练。基准测试是目标；生产模型是交付的答案。

### 基准测试常见误区

- **仅截图评估。** OSWorld 是截图驱动的；在 OSWorld 上评估使用 DOM 或无障碍 API 的智能体会遗漏定位挑战。
- **忽略轨迹长度。** 仅打分成功率忽略了 OSWorld-Human 揭示的 1.4-2.7 倍步骤低效问题。
- **过时的自托管应用。** WebArena 的应用固定了特定版本；更新而不重新整理会破坏可比性。

## 动手构建

`code/main.py` 实现了一个简易的网页智能体框架：

- 一个最小的"购物应用"状态机：list_items、add_to_cart、checkout。
- 3 个任务的黄金轨迹。
- 一个尝试每个任务的脚本化智能体。
- 基于执行的评估器（状态检查）和轨迹效率指标（步骤 vs 黄金标准）。

运行它：

```
python3 code/main.py
```

输出：每个任务的成功率和轨迹效率，反映 OSWorld-Human 的方法论。

## 使用场景

- **WebArena Verified** 自托管在内部集群上用于持续评估。
- **OSWorld** 在 VM 集群上用于桌面智能体。
- **计算机使用智能体**（第 21 课）— Claude、OpenAI CUA、Gemini — 都在此类工作负载上训练。
- **你自己的产品流程** — 为你最重要的 20 个任务捕获黄金轨迹；每周对智能体运行。

## 交付产出

`outputs/skill-web-desktop-harness.md` 构建一个网页/桌面智能体框架，带基于执行的评估和轨迹效率指标。

## 练习

1. 用第二个应用（论坛）扩展简易框架。编写 3 个任务加黄金轨迹。
2. 添加每个任务的轨迹效率报告。在你的简易版本上，智能体是黄金标准的 1x、2x 还是 3x？
3. 实现一个"干扰"工具——一个黄金轨迹从不使用的工具。脚本化智能体是否会被诱惑？
4. 阅读 OSWorld-G。你如何在自己的评估中分离定位失败和规划失败？
5. 阅读 WebArena 的应用 README。当你升级其中一个固定版本的应用时，什么会出问题？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| WebArena | "网页智能体基准" | 812 个任务，跨 4 个自托管应用；gym 风格评估 |
| VisualWebArena | "视觉 WebArena" | 视觉定位的 WebArena；截图作为观察方式 |
| OSWorld | "桌面智能体基准" | 369 个任务，在真实 Ubuntu/Windows/macOS 上 |
| GUI grounding | "像素到元素映射" | 模型在 1920x1080 中定位 UI 元素 |
| 操作知识 | "OS 操作经验" | 哪个菜单、哪个快捷键、哪个偏好面板 |
| OSWorld-G | "定位套件" | 564 个纯定位样本 + 训练集 |
| OSWorld-Human | "黄金轨迹" | 用于衡量效率的人工专家动作序列 |
| 轨迹效率 | "步骤数相对黄金标准" | 智能体步骤数除以人类最少步骤数 |

## 延伸阅读

- [Zhou 等人，WebArena (arXiv:2307.13854)](https://arxiv.org/abs/2307.13854) — 四应用网页基准测试
- [Xie 等人，OSWorld (arXiv:2404.07972)](https://arxiv.org/abs/2404.07972) — 跨 OS 桌面基准测试
- [Anthropic，Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — Claude 的基准测试塑造的能力
- [OpenAI，Computer-Using Agent](https://openai.com/index/computer-using-agent/) — OSWorld 和 WebArena 数据
