# Agent Loop 核心概念

> 日期: 2026-06-22

## 1. ReAct 的三个标签

ReAct（Yao et al. 2022）每一步输出三个部分：

| 标签 | 作用 |
|------|------|
| **Thought** | 推理下一步该做什么 |
| **Action** | 选择并执行一个工具 |
| **Observation** | 接收工具返回的结果 |

核心价值：Thought 步骤让模型能**制定计划、追踪进度、处理异常**，这是纯 action-only prompting 做不到的。

## 2. 五个必要组件

一个 agent loop 缺少任何一个就只是 chatbot：

1. **Message buffer** — 持续增长的消息历史
2. **Tool registry** — 名称 → 可调用函数的映射
3. **Stop condition** — finish / no tool calls / max turns / max tokens / guardrail
4. **Turn budget** — 防止无限循环的硬上限
5. **Observation formatter** — 把工具输出转成模型可读的字符串

## 3. Provider Schema 隔离

Anthropic 和 OpenAI 的消息格式**不能混用**：

| Provider | 助手消息 | 工具结果 |
|----------|---------|---------|
| Anthropic | `tool_use` block 在 content 中 | `tool_result` block 在 user 消息中 |
| OpenAI | `tool_calls` 数组在 assistant 消息上 | `tool` role 消息 |

实现时必须在构造 `AgentLoop` 时选定 provider，序列化时走对应的转换函数。

## 4. 模型如何判断停止

`finish` 是**控制信号**，不是注册的 tool：

```python
if reply["kind"] == "finish":
    return reply["content"]  # 直接退出，不走 tool dispatch
```

模型通过 prompt 中的隐含指令判断何时 finish — 认为任务完成就输出 finish 而非 action。

其他停止条件（max turns、guardrail 等）是**安全网**，不是正常退出路径。

## 5. 2026 年的三个陷阱

| 陷阱 | 问题 | 解决方案 |
|------|------|---------|
| **Trust boundary collapse** | 工具输出可能含恶意指令 | Lesson 27 (prompt injection) |
| **Cascading failure** | agent 无法区分"我失败了"和"任务不可能" | Lesson 26 |
| **Loop length explosion** | 40-400 步后难以定位问题 | Lesson 23 (observability) + Lesson 30 (eval) |

重点：这些不是 ReAct 模式本身的问题，而是**大规模部署 agent 时的系统性风险**。

## 6. ReAct 与框架的关系

所有主流框架（Claude Agent SDK、OpenAI Agents SDK、LangGraph、AutoGen、CrewAI）都在 ReAct 基础上**叠加功能**，而非替换循环本身：

- LangGraph → 有状态图 + checkpoint
- AutoGen → actor 模型消息传递
- CrewAI → 角色模板
- OpenAI Agents SDK → Handoff + Guardrail

但也有非 ReAct 路径：Plan-and-Execute（ReWOO）、Reflection、Multi-agent debate。

## 7. 为什么是 observability 问题

当 agent 跑了 40 步后出错：

- 没有 observability → 不知道哪一步决策失误
- 没有 eval → 不知道该改 prompt、换模型、还是修工具定义

trace 日志必须记录每一步的 Thought、Action、Observation，才能事后分析。这就是为什么 Lesson 23 和 Lesson 30 是后续必修内容。
