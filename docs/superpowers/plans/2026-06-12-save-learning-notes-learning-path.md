# Save Learning Notes Learning Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update `/save-learning-notes` so each successful notes save also maintains root `LEARNING_PATH.md` with current position, completed lessons, review items, and an append-only study log.

**Architecture:** This is a skill-documentation change, not an application-code change. The implementation edits `.claude/skills/save-learning-notes/SKILL.md` to add explicit operational steps and templates that future skill executions can follow; the existing root `LEARNING_PATH.md` must be preserved and only extended during actual skill runs.

**Tech Stack:** Markdown skill instructions, Claude Code skill frontmatter, repository documentation, shell-based verification.

---

## File Structure

- Modify: `.claude/skills/save-learning-notes/SKILL.md`
  - Responsibility: Defines the behavior of `/save-learning-notes`; add learning path maintenance instructions after note writing and before user feedback.
- Read-only reference: `docs/superpowers/specs/2026-06-12-save-learning-notes-learning-path-design.md`
  - Responsibility: Approved design spec for the learning path feature.
- Read-only existing file: `LEARNING_PATH.md`
  - Responsibility: Existing personalized learning path file. Do not overwrite during implementation. The skill instructions must say future executions should preserve existing content and append standard managed sections if markers are missing.
- No automated test files are currently present for skills in this repository. Verification uses deterministic grep/status checks against the Markdown skill document.

## Current Context

`LEARNING_PATH.md` already exists and contains a personalized course plan created on 2026-05-28. Because it does not contain the new managed markers, the new skill behavior must preserve that content and append a standard managed progress section at the bottom during future `/save-learning-notes` runs.

The existing skill has these relevant sections:

- `Step 7: 写入文件`
- `Step 8: 反馈`
- `## Rules`

The new learning path step belongs between the current Step 7 and current Step 8. After insertion, renumber the feedback step to Step 9.

---

### Task 1: Add Learning Path Behavior to Skill Metadata and Workflow

**Files:**
- Modify: `.claude/skills/save-learning-notes/SKILL.md:4-8`
- Modify: `.claude/skills/save-learning-notes/SKILL.md:125-135`

- [ ] **Step 1: Inspect the current skill text**

Run:

```bash
grep -nE 'description:|Step 7|Step 8|## Rules' .claude/skills/save-learning-notes/SKILL.md
```

Expected: output includes `description:` near the top, `### Step 7: 写入文件`, `### Step 8: 反馈`, and `## Rules`.

- [ ] **Step 2: Update the frontmatter description**

Edit `.claude/skills/save-learning-notes/SKILL.md` lines 4-8 so the description becomes:

```yaml
description: >
  Organize multi-round Q&A learning sessions into structured Chinese learning
  notes, saved to the current lesson's learning-notes directory, then update
  the root LEARNING_PATH.md with the latest learning position and study log.
  Trigger phrases: "整理笔记", "保存笔记", "生成学习笔记", "save notes",
  "organize notes", or `/save-learning-notes`.
```

- [ ] **Step 3: Insert the new learning path step after Step 7**

In `.claude/skills/save-learning-notes/SKILL.md`, replace the current section:

```markdown
### Step 8: 反馈

告知用户：
- 笔记文件的完整路径
- 笔记包含多少个小节
- 简单概括笔记涵盖的核心主题（一两句话）
```

with this complete section:

```markdown
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

- 当前 Phase: <phase-slug>
- 当前 Lesson: <lesson-slug>
- 最近笔记: [<note-file-name>](<note-path>)
- 更新时间: <YYYY-MM-DD>

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

1. **当前位置**：更新为当前 phase、当前 lesson、最近笔记链接和当前日期。
2. **已学习 Lessons**：
   - 如果表格中已有相同 `phase + lesson` 的行，替换该行的日期和笔记链接。
   - 如果不存在相同 lesson，新增一行。
3. **待复习 / 待消化**：
   - 从本次生成的笔记内容中查找 `待消化的问题`、`开放问题`、`待复习` 等小节。
   - 如果能提取到简短摘要，追加一条：`- <YYYY-MM-DD> `phases/<phase-slug>/<lesson-slug>`: <摘要>`。
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
```

- [ ] **Step 4: Run metadata and step-order verification**

Run:

```bash
grep -nE 'LEARNING_PATH.md|Step 7|Step 8|Step 9|learning-path:completed:start|learning-path:log:start' .claude/skills/save-learning-notes/SKILL.md
```

Expected: output includes the updated description reference to `LEARNING_PATH.md`, `Step 8: 更新学习路径`, `Step 9: 反馈`, and the managed marker names.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add .claude/skills/save-learning-notes/SKILL.md
git commit -m "feat: update notes skill learning path workflow" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

Expected: commit succeeds with one modified file.

---

### Task 2: Add Explicit Learning Path Rules and Failure Semantics

**Files:**
- Modify: `.claude/skills/save-learning-notes/SKILL.md:168-177`

- [ ] **Step 1: Inspect the Rules section**

Run:

```bash
grep -nA20 '^## Rules' .claude/skills/save-learning-notes/SKILL.md
```

Expected: output shows the existing rules about Chinese content, section count, tables, and open questions.

- [ ] **Step 2: Append learning path rules**

Add these bullets to the end of the `## Rules` list in `.claude/skills/save-learning-notes/SKILL.md`:

```markdown
- 每次成功保存学习笔记后，都要尝试更新根目录 `LEARNING_PATH.md`
- 更新 `LEARNING_PATH.md` 时必须保留用户已有内容；如果缺少标准标记，只能在底部追加标准区块
- `已学习 Lessons` 中同一个 lesson 只保留一行，并更新为最新笔记链接
- `学习日志` 必须追加记录；同一个 lesson 多次保存也保留多条日志
- 无法提取待复习内容时跳过该项，不得因此阻塞笔记保存或学习路径更新
- 如果 `LEARNING_PATH.md` 更新失败，最终反馈必须说明“笔记已保存，但学习路径未更新”并给出失败原因
```

- [ ] **Step 3: Verify rules were added exactly once**

Run:

```bash
grep -nE '每次成功保存学习笔记|保留用户已有内容|同一个 lesson 只保留一行|学习日志.*追加|无法提取待复习|笔记已保存，但学习路径未更新' .claude/skills/save-learning-notes/SKILL.md
```

Expected: six lines are printed, one for each new rule.

- [ ] **Step 4: Commit Task 2**

Run:

```bash
git add .claude/skills/save-learning-notes/SKILL.md
git commit -m "docs: add learning path rules to notes skill" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

Expected: commit succeeds with one modified file.

---

### Task 3: Verify the Skill Against the Approved Spec

**Files:**
- Read-only reference: `docs/superpowers/specs/2026-06-12-save-learning-notes-learning-path-design.md`
- Verify: `.claude/skills/save-learning-notes/SKILL.md`
- Verify: `LEARNING_PATH.md`

- [ ] **Step 1: Confirm the existing root learning path was not modified**

Run:

```bash
git diff -- LEARNING_PATH.md
```

Expected: no output. The implementation changes the skill instructions only; it does not edit the existing learning path file.

- [ ] **Step 2: Check all required behavior phrases exist in the skill**

Run:

```bash
grep -nE '维护项目根目录的 `LEARNING_PATH.md`|只有 Step 7 成功写入笔记文件后才执行|保留原有内容|更新为当前 phase|已有相同 `phase \+ lesson`|每次保存都追加一条日志|如果 `LEARNING_PATH.md` 更新失败' .claude/skills/save-learning-notes/SKILL.md
```

Expected: output includes lines covering trigger timing, non-overwrite behavior, current position update, duplicate lesson update, append-only log, and failure feedback.

- [ ] **Step 3: Check there are no unresolved placeholders in the skill**

Run:

```bash
grep -nE 'TBD|TODO|待定|不确定|PLACEHOLDER' .claude/skills/save-learning-notes/SKILL.md || true
```

Expected: no output.

- [ ] **Step 4: Check final git status**

Run:

```bash
git status --short
```

Expected: no output after both task commits, or only intentional untracked files if the executor created temporary files and then needs to remove them.

- [ ] **Step 5: Report completion**

Report these facts to the user:

```text
Updated .claude/skills/save-learning-notes/SKILL.md to maintain LEARNING_PATH.md after notes are saved.
Preserved the existing root LEARNING_PATH.md; no implementation-time edits were made to it.
Verified required behavior phrases and placeholder scan.
```

No commit is required for Task 3 unless verification discovers and fixes a missing instruction.

---

## Self-Review

- Spec coverage: Covered root `LEARNING_PATH.md`, current position, completed lessons, duplicate lesson update, append-only log, review extraction, non-standard existing file preservation, final feedback, and failure semantics.
- Placeholder scan: The plan intentionally uses `<phase-slug>`, `<lesson-slug>`, `<note-path>`, and `<YYYY-MM-DD>` inside skill templates because they are runtime placeholders for future skill executions, not missing implementation details. No `TBD` or unspecified implementation steps remain.
- Type/name consistency: Marker names are consistent across the plan: `learning-path:completed`, `learning-path:review`, and `learning-path:log`. The file name is consistently `LEARNING_PATH.md`.
