---
name: save-learning-notes
version: 1.0.0
description: >
  Organize multi-round Q&A learning sessions into structured Chinese learning
  notes, saved to the current lesson's learning-notes directory.
  Trigger phrases: "整理笔记", "保存笔记", "生成学习笔记", "save notes",
  "organize notes", or `/save-learning-notes`.
tags: [learning, notes, ai-engineering, study]
---

# Save Learning Notes

将当前 Claude 会话中针对某个 lesson 的多轮 Q&A 整理成结构化的中文学习笔记，保存到该 lesson 的 `learning-notes/` 目录下。

## Activation

This skill activates when the user says things like:
- `/save-learning-notes`
- "整理笔记"
- "保存笔记"
- "生成学习笔记"
- "save notes"
- "organize notes"

## 用户工作流

用户会针对某个具体 lesson（如 `phases/11-llm-engineering/03-structured-outputs/`）做以下事情：
1. 让 Claude 解释 `docs/zh.md` 的内容
2. 基于内容多轮提问
3. 当感觉学得差不多时，输入"整理笔记"触发本 skill

## Procedure

### Step 1: 检测当前 lesson

回顾本次会话历史，搜索被 Read 工具读取过的文件路径，匹配 `phases/<phase>/<lesson>/docs/(zh|en).md` 这种模式。

- 找到匹配路径 → 提取出 lesson 目录
- 找到多个不同 lesson 的文档 → 取最近被读过的那个
- 找不到任何 lesson 文档 → 进入 Step 1a 手动指定流程

**Step 1a: 无法自动检测**

如果对话中没有任何 `docs/zh.md` 或 `docs/en.md` 的读取记录，询问用户：
> "未检测到你正在学习的 lesson，请手动指定（如 11-llm-engineering/03-structured-outputs）："

用户输入后，进入 Step 2。

### Step 2: 用户确认

向用户展示检测到的 lesson 路径，请求确认：

```
检测到你正在学习以下 lesson：
phases/11-llm-engineering/03-structured-outputs/

笔记将保存到：
phases/11-llm-engineering/03-structured-outputs/learning-notes/2026-06-11-structured-outputs.md

确认保存吗？
```

使用 AskUserQuestion 让用户确认。用户确认后继续；如果用户想改路径，先让用户重新指定。

### Step 3: 回溯对话内容

仔细阅读整个对话历史（特别是用户的提问和 Claude 的回答），识别以下类型的知识内容：
- 概念定义（如"什么是 X"）
- 机制解释（如"Y 是怎么工作的"）
- 对比分析（如"X 和 Y 的区别"）
- 关键洞察（如用户的"原来如此"对应的解释）
- 代码示例（如果有）
- 待消化的开放问题

### Step 4: 提取并组织笔记

将识别出的内容按主题分组，形成 5-10 个层级清晰的小节。每节聚焦一个独立的知识点。

**整理原则**：
- 保留用户的核心疑问和关键洞察
- 删除冗余的解释、重复的描述、不必要的元评论
- 提炼概念，使其比对话更精炼
- 不堆砌原始段落，只保留"用户真正理解到的内容"
- 使用表格、列表、代码块等增强可读性

### Step 5: 生成文件内容

按以下格式生成 Markdown 内容：

```markdown
# <主题标题>

> 日期: YYYY-MM-DD

## 1. <小节标题>

<内容>

## 2. <小节标题>

...
```

要点：
- 标题：概括本次对话核心主题（2-8 个字）
- 日期：使用当前日期
- 小节：5-10 节，每节一个独立知识点
- 元素：可使用表格、代码块、列表
- 语言：全中文

### Step 6: 确定文件路径

文件路径格式：`phases/<phase>/<lesson>/learning-notes/YYYY-MM-DD-<topic-slug>.md`

- 日期：当前日期（格式 `YYYY-MM-DD`）
- topic-slug：从笔记主题提取的英文 kebab-case，2-4 个词
- 重名处理：如果文件已存在，追加 `-2`、`-3` 等后缀

使用 Bash 工具检查目标目录是否存在，不存在则创建：
```bash
mkdir -p phases/<phase>/<lesson>/learning-notes
```

### Step 7: 写入文件

使用 Write 工具将笔记内容写入文件。

### Step 8: 反馈

告知用户：
- 笔记文件的完整路径
- 笔记包含多少个小节
- 简单概括笔记涵盖的核心主题（一两句话）

## 笔记格式参考

参照 `phases/07-transformers-deep-dive/learning-notes/2026-06-01-attention-and-training.md` 的格式：

```markdown
# Transformer 核心概念学习笔记

> 日期: 2026-06-01

## 1. 为什么需要 Transformer

2016 年主流模型是 RNN/LSTM，核心问题是**必须逐步计算**...

## 2. Softmax

将任意实数向量转换为**概率分布**的函数：

```
softmax(z_i) = e^(z_i) / Σ e^(z_j)
```

## 3. Q、K、V 的含义

每个 token 通过三个投影矩阵生成三个向量：

| 向量 | 含义 | 类比 |
|------|------|------|
| **Q (Query)** | 我在找什么 | 搜索关键词 |
| **K (Key)** | 我包含什么 | 书脊上的标签 |
| **V (Value)** | 被选中后能提供什么 | 书的实际内容 |
```

## Rules

- 笔记内容必须是中文
- 不添加"本节要点""总结回顾"等元评论段落
- 不复述用户问过的原问题，除非该问题本身有教育价值
- 不超过 12 个小节；如果内容更多，优先合并相似主题
- 表格优于长列表，列表优于段落
- 关键概念用 `**粗体**` 标注
- 如果对话中包含待用户消化的开放问题，可以单独列出一节"待消化的问题"
