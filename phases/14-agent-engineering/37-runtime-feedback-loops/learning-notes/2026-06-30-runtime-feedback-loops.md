# Runtime Feedback Loops 学习笔记

> 日期: 2026-06-30

## 1. 核心问题

Agent 说"测试跑完了，全部通过"——但实际上什么都没跑。它在幻觉输出，或者跑了命令但没读结果。**没有结构化反馈的 Agent 循环等于开盲盒。**

## 2. Feedback Runner 是什么

一个 `run_with_feedback()` 包装器，强制每条 shell 命令都经过它，捕获结构化结果并写入 JSONL 文件：

```
Agent 决定执行命令
    → run_with_feedback(command, agent_note)
        → subprocess.run() 执行
        → 捕获 stdout / stderr / exit_code / duration
        → 写入 feedback_record.jsonl
    → Agent 下一轮读这条记录
```

## 3. Feedback Record 的字段

| 字段 | 作用 |
|------|------|
| `command` | 精确 argv，无 shell 展开 |
| `stdout_tail` / `stderr_tail` | 确定性截断（头+尾），保证不爆 token 预算 |
| `exit_code` | 明确的成功/失败信号 |
| `duration_ms` | 发现慢探针和失控进程 |
| `agent_note` | Agent 在读结果前写的一行预期 |

## 4. 三个关键设计原则

- **Refuse-on-null** — `exit_code` 为 null 时，循环必须拒绝推进。没有退出码 = 没有进展。这是防止 Agent 幻觉的硬闸门。
- **确定性截断** — 同样的输出永远产生同样的记录，避免随机采样。50MB 日志截断后必须可控。
- **Feedback ≠ Telemetry** — Feedback 给下一 turn 用（Agent 自己读），Telemetry 给运维人员跨时间审查用（OTel、Langfuse 等）。

## 5. 三个生产模式

1. **写入时脱敏，不是读取时** — stdout/stderr 可能泄露密钥（`Bearer`、`password=`、AWS AKIA、Slack token），runner 在追加 JSONL 之前就做 redaction。攻击者拿到的是磁盘文件，读时脱敏来不及。
2. **文件轮转，不是单文件** — `feedback_record.jsonl` 硬限 1MB，溢出后轮转到 `.1`、`.2`、丢弃 `.5`。Agent 每轮只读当前文件，保证 loader 开销有界。
3. **parent_command_id 追踪重试链** — 每条记录带 `command_id`，重试记录带 `parent_command_id` 指向上一次尝试。审查时能看到"这个命令是上一个失败命令的重试"，而不是把重试误判为独立成功。

## 6. 实际使用最需要注意的坑

| 坑 | 为什么危险 |
|----|-----------|
| 截断预算没算对 | LLM 上下文窗口有限，一条 50MB stdout 截断后仍然可能过大，需要根据模型 token 限制反推 head_lines 和 tail_lines 上限 |
| exit_code null 没硬拒绝 | 最常见的 bug：subprocess 超时或被 kill，exit_code 是 null，Agent 却照样说"成功了" |
| 没在写入时脱敏 | 调试阶段 stdout 里常有 `Authorization: Bearer sk-xxx`，先写文件再脱敏迟早泄露 |

## 7. 基础概念补充

- **subprocess** — Python 标准库执行外部命令的方式，runner 在它外面包了一层确保留下结构化记录
- **stdout / stderr / stdin** — std = standard，Linux 每个进程自带三个默认流：标准输入（stdin）、标准输出（stdout）、标准错误（stderr）。分开设计使 stdout 可管道重定向，stderr 默认打到屏幕上保证错误不被淹没
