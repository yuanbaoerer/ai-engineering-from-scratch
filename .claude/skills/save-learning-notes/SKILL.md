---
name: save-learning-notes
version: 1.0.0
description: >
  Organize multi-round Q&A learning sessions into structured Chinese learning
  notes, saved to the current lesson's learning-notes directory, then update
  the root LEARNING_PATH.md with the latest learning position and study log.
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

### Step 8: 更新学习路径

笔记文件成功写入后，维护项目根目录的 `LEARNING_PATH.md`，用于记录当前学习位置、已学习 lesson、待复习内容和学习日志。

**重要**：只有 Step 7 成功写入笔记文件后才执行本步骤，避免学习路径指向不存在的笔记。如果本步骤失败，不能回滚或删除已保存的学习笔记；在最终反馈中如实说明学习路径未更新。

#### Step 8a: 读取或创建学习路径文件

目标文件固定为项目根目录：`LEARNING_PATH.md`。

- 如果文件不存在，创建标准模板（见 Step 8b）。
- 如果文件存在且包含标准标记区块，按 Step 8c 更新。
- 如果文件存在但缺少标准标记区块，保留原有内容，并在文件底部追加标准学习进度区块；不要覆盖用户已有内容。

标准标记区块包括：

```markdown
<!-- learning-path:current:start -->
<!-- learning-path:current:end -->
<!-- learning-path:completed:start -->
<!-- learning-path:completed:end -->
<!-- learning-path:review:start -->
<!-- learning-path:review:end -->
<!-- learning-path:log:start -->
<!-- learning-path:log:end -->
```

#### Step 8b: 标准学习路径模板

新建文件或追加标准区块时，使用以下结构。将示例值替换为当前 lesson、当前日期和实际笔记路径。

```markdown
# AI Engineering 学习路径

## 当前位置

<!-- learning-path:current:start -->
- 当前 Phase: <phase-slug>
- 当前 Lesson: <lesson-slug>
- 最近笔记: [<note-file-name>](<note-path>)
- 更新时间: <YYYY-MM-DD>
<!-- learning-path:current:end -->

## 已学习 Lessons

<!-- learning-path:completed:start -->
| 日期 | Phase | Lesson | 笔记 |
|------|-------|--------|------|
| <YYYY-MM-DD> | <phase-slug> | <lesson-slug> | [笔记](<note-path>) |
<!-- learning-path:completed:end -->

## 待复习 / 待消化

<!-- learning-path:review:start -->
<!-- learning-path:review:end -->

## 学习日志

<!-- learning-path:log:start -->
- <YYYY-MM-DD> 保存 `phases/<phase-slug>/<lesson-slug>` 学习笔记：[笔记](<note-path>)
<!-- learning-path:log:end -->
```

#### Step 8c: 更新规则

根据当前 lesson 路径 `phases/<phase-slug>/<lesson-slug>/` 和本次笔记路径更新 `LEARNING_PATH.md`：

1. **当前位置**：更新 `learning-path:current` 标记区块中的当前 phase、当前 lesson、最近笔记链接和当前日期。
2. **已学习 Lessons**：
   - 如果表格中已有相同 `phase + lesson` 的行，替换该行的日期和笔记链接。
   - 如果不存在相同 lesson，新增一行。
3. **待复习 / 待消化**：
   - 从本次生成的笔记内容中查找 `待消化的问题`、`开放问题`、`待复习` 等小节。
   - 如果能提取到简短摘要，追加一条如下格式的记录：

```markdown
- <YYYY-MM-DD> `phases/<phase-slug>/<lesson-slug>`: <摘要>
```

   - 如果没有相关内容，跳过本项，不要制造空泛复习项。
4. **学习日志**：每次保存都追加一条日志，即使同一个 lesson 已在“已学习 Lessons”中存在。

#### Step 8d: 已有非标准 `LEARNING_PATH.md` 的处理

如果 `LEARNING_PATH.md` 已存在但没有上述标准标记，说明它可能是用户手写或由其他 skill 生成的学习路径。此时：

1. 保留原文件全部内容不变。
2. 在文件底部追加分隔线和标准区块：

```markdown
---

## 当前学习进度

<使用 Step 8b 中从“## 当前位置”开始的标准结构>
```

3. 后续执行再按标准标记区块更新。

### Step 9: 反馈

告知用户：
- 笔记文件的完整路径
- 笔记包含多少个小节
- 简单概括笔记涵盖的核心主题（一两句话）
- `LEARNING_PATH.md` 是否已更新
- 当前记录的学习位置

反馈示例：

```text
已保存学习笔记：phases/13-tools-and-protocols/03-example/learning-notes/2026-06-12-example.md
共整理 7 个小节，涵盖本次对话中的核心概念、机制和实践注意事项。
已更新学习路径：LEARNING_PATH.md
当前位置：phases/13-tools-and-protocols/03-example
```

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
