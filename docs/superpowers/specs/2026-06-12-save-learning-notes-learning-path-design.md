# `/save-learning-notes` 学习路径维护设计

> 日期: 2026-06-12
> 状态: 已确认设计，等待实现计划

## 背景

当前 `.claude/skills/save-learning-notes/SKILL.md` 会把一次 lesson 学习会话整理成中文 Markdown 笔记，并保存到该 lesson 的 `learning-notes/` 目录。用户希望每次执行该 skill 后，同时维护一份学习路径，记录当前学到哪里、哪些 lesson 已学过、后续需要复习什么，方便下次继续学习。

## 目标

- 在每次成功保存学习笔记后，维护根目录 `LEARNING_PATH.md`。
- 让用户打开仓库即可看到上次学习位置。
- 记录已学习 lesson 列表，并在同一 lesson 重复保存时更新列表中的最新笔记。
- 保留追加式学习日志，形成历史轨迹。
- 记录待复习或待消化内容；无法可靠提取时跳过，不阻塞笔记保存。

## 非目标

- 不引入 JSON 状态文件。
- 不实现独立进度管理 CLI 或自动跳转功能。
- 不改变学习笔记本身的生成格式，除非为了提取待复习信息需要读取已生成内容。
- 不强制计算完整课程完成百分比；可在后续需求中扩展。

## 推荐方案

使用单个根目录 Markdown 文件：`LEARNING_PATH.md`。

该文件对人类可读，并通过 HTML 注释标记可维护区块，方便 skill 稳定更新。推荐结构如下：

```markdown
# AI Engineering 学习路径

## 当前位置

- 当前 Phase: 13-tools-and-protocols
- 当前 Lesson: 03-example-lesson
- 最近笔记: [2026-06-12-example.md](phases/13-tools-and-protocols/03-example-lesson/learning-notes/2026-06-12-example.md)
- 更新时间: 2026-06-12

## 已学习 Lessons

<!-- learning-path:completed:start -->
| 日期 | Phase | Lesson | 笔记 |
|------|-------|--------|------|
| 2026-06-12 | 13-tools-and-protocols | 03-example-lesson | [笔记](phases/13-tools-and-protocols/03-example-lesson/learning-notes/2026-06-12-example.md) |
<!-- learning-path:completed:end -->

## 待复习 / 待消化

<!-- learning-path:review:start -->
- 2026-06-12 `phases/13-tools-and-protocols/03-example-lesson`: 待消化的问题摘要
<!-- learning-path:review:end -->

## 学习日志

<!-- learning-path:log:start -->
- 2026-06-12 保存 `phases/13-tools-and-protocols/03-example-lesson` 学习笔记：[笔记](phases/13-tools-and-protocols/03-example-lesson/learning-notes/2026-06-12-example.md)
<!-- learning-path:log:end -->
```

## 行为设计

### 1. 触发时机

在现有流程中，学习笔记成功写入文件之后，新增“更新学习路径”步骤。只有笔记保存成功后才更新 `LEARNING_PATH.md`，避免进度文件指向不存在的笔记。

### 2. 文件不存在时

如果根目录没有 `LEARNING_PATH.md`：

1. 创建标准模板。
2. 写入当前 lesson 为“当前位置”。
3. 在“已学习 Lessons”添加当前 lesson。
4. 在“学习日志”追加本次记录。
5. 如果本次笔记包含待复习内容，添加到“待复习 / 待消化”。

### 3. 文件存在且包含标准区块时

按区块更新：

- `当前位置`：替换为当前 phase、lesson、最近笔记和更新时间。
- `learning-path:completed`：
  - 如果当前 lesson 已存在，更新该行的日期和笔记链接。
  - 如果当前 lesson 不存在，新增一行。
- `learning-path:review`：从本次笔记中提取“待消化的问题”或开放问题摘要；没有则不添加。
- `learning-path:log`：追加一条本次保存记录。同一 lesson 多次保存也保留多条日志。

### 4. 文件存在但缺少标准区块时

保留原有内容，在文件底部追加一个完整的标准学习路径区块，并用当前 lesson 初始化。这样避免覆盖用户手写内容。

### 5. 重复 lesson 处理

用户选择“列表+日志”：

- “已学习 Lessons”中每个 lesson 只保留一行，代表该 lesson 的最新学习笔记。
- “学习日志”每次保存都追加一条，保留多次学习或复习历史。

### 6. 待复习内容提取

优先从生成的笔记内容中寻找以下小节或表达：

- `待消化的问题`
- `开放问题`
- `待复习`
- 明确表示还需要回顾的条目

提取结果应简短，适合一行展示。若没有相关内容，跳过“待复习 / 待消化”更新。

## Skill 文档变更范围

更新 `.claude/skills/save-learning-notes/SKILL.md`：

1. 在描述中说明执行后会维护根目录 `LEARNING_PATH.md`。
2. 在 Procedure 中新增“更新学习路径”步骤，位于写入笔记之后、反馈之前。
3. 增加 `LEARNING_PATH.md` 模板和区块标记规则。
4. 更新最终反馈要求，增加学习路径文件路径和当前学习位置。
5. 增加规则：不得因无法提取复习项而让笔记保存失败。

## 反馈格式

执行完成后，skill 应告知用户：

- 学习笔记完整路径
- 笔记包含的小节数量
- 笔记核心主题概括
- `LEARNING_PATH.md` 已更新
- 当前记录的学习位置

示例：

```text
已保存学习笔记：phases/.../learning-notes/2026-06-12-example.md
共整理 7 个小节，涵盖 structured outputs 的 schema、约束与实践注意事项。
已更新学习路径：LEARNING_PATH.md
当前位置：phases/13-tools-and-protocols/03-example-lesson
```

## 错误处理

- 如果 `LEARNING_PATH.md` 不存在：创建它。
- 如果存在但格式不标准：不覆盖原内容，在底部追加标准区块。
- 如果标准区块中已有当前 lesson：更新完成列表行，仍追加日志。
- 如果无法解析待复习问题：跳过待复习更新。
- 如果写入 `LEARNING_PATH.md` 失败：如实告知用户笔记已保存但学习路径未更新。

## 验证方式

这是 skill 文档行为变更，验证重点是文档是否足够明确让 skill 执行一致：

- 检查新增步骤在“写入文件”之后、“反馈”之前。
- 检查 `LEARNING_PATH.md` 模板路径是根目录文件。
- 检查重复 lesson 行为同时满足“完成列表更新”和“日志追加”。
- 检查异常场景不会阻塞已经成功保存的学习笔记。
- 检查最终反馈包含学习路径更新结果。
