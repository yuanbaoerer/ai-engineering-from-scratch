# Agent Workbench 工程：为什么强大模型仍然失败

> 日期: 2026-06-28

## 1. 核心问题：模型没错，错在对"工作"的理解

前沿模型在真实仓库任务上失败，不是因为 Python 写得差，而是缺少围绕模型的**运行环境**。它不知道什么算完成、能改哪些文件、哪些测试是权威的、下次怎么接手。这是 **workbench bug**，不是模型 bug。

## 2. 七个 Workbench 表面

| 表面 | 作用 | 缺失后果 |
|------|------|---------|
| **Instructions** | 启动规则、禁止操作、完成定义 | Agent 猜测什么算"交付" |
| **State** | 当前任务、已改文件、阻塞项、下一步 | 每次会话从零开始 |
| **Scope** | 允许/禁止文件、验收标准 | 编辑泄漏到无关代码 |
| **Feedback** | 捕获真实命令输出回传循环 | Agent 在 400 错误时宣布成功 |
| **Verification** | 测试、lint、冒烟测试、范围检查 | "看起来不错"直接进 main |
| **Review** | 第二角色独立审查 | 建造者给自己的作业打分 |
| **Handoff** | 改了什么、为什么、还剩什么 | 下次会话重新发现一切 |

关键原则：**循环闭合在 state 文件上，不在聊天记录上。聊天是易失的，仓库才是系统记录。**

## 3. Workbench ≠ Prompt Engineering ≠ Framework

| 概念 | 作用范围 | 核心区别 |
|------|---------|---------|
| Prompt Engineering | 单轮对话 | 告诉模型"这一轮你想要什么" |
| Workbench | 跨轮次、跨会话 | 告诉模型"怎么在多轮中做工程" |
| Framework | 运行时 | 提供 runtime（LangGraph、Agents SDK），workbench 在 runtime 内部工作 |

大多数 agent 失败故事本质是 workbench 失败穿了 prompt engineering 的外衣。

## 4. 与分布式系统原语的映射

每个 workbench 表面对应一个已有的分布式系统原语：

| 原语 | Agent 对应 |
|------|-----------|
| Function | 工具调用、规则检查、验证步骤 |
| Worker | Builder、Reviewer、Verifier、MCP server |
| Trigger | Agent loop tick、HTTP 请求、cron、文件变更 |
| Runtime | Claude Code 进程、LangGraph 运行时 |
| Queue | 任务板、反馈日志、审查收件箱 |
| Session persistence | `agent_state.json`、检查点、仓库本身 |
| Authorization policy | 允许/禁止文件、审批边界、MCP 能力列表 |

**所有 "harness engineering" 新词汇都能翻译回这些原语。** 术语会变，工程不变。

## 5. UX 描述 vs 构建系统

业界文章（LangChain、Addy Osmani、Anthropic 等）本质上是在描述 agent 系统的**用户界面层**——怎么配置、怎么用、组件叫什么名字。

这门课的立场是从分布式系统原语出发**构建**底层系统。如果底层原语搭对了，七个表面自然就出来了。**再多的 AGENTS.md 润色也修复不了缺失的队列。**

## 6. Harness-Compute 分离（控制面/数据面分离）

- **Compute（数据面）**：真正干活的——模型推理、代码执行、文件写入
- **Harness（控制面）**：决定谁干什么、在哪里干、什么时候停——调度、权限、状态管理、监控

类比：车辆在跑 = 数据面；交通信号灯、收费站 = 控制面。

分离后可以独立扩展和替换：换模型不影响控制面，升级 harness 不需要改模型本身。这是分布式系统几十年前就有的老概念，agent 社区重新起了个名字。

## 7. 数据支撑：Harness 的实际影响

- **Terminal Bench 2.0**：同一个模型，只改 harness，从 30 名开外跳到第 5
- **Vercel**：删掉 80% 的工具，成功率从 80% 提升到 100%
- **Harvey**：法律 agent 仅通过 harness 优化，准确率翻倍
- **88%** 的企业 AI agent 项目未达生产——失败集中在 runtime 而非推理

## 8. Claude Code 中的 Harness 工程化实践

按优先级排列：

1. **Scope（最先加）**：AGENTS.md 里写清楚允许和禁止的文件，防止 agent 乱改
2. **Verification（其次）**：用 hooks 强制跑测试/lint，不让 agent 自己宣布成功
3. **Handoff（然后）**：每个 session 结束写摘要，让下次接手不用从头猜
4. **Instructions**：细化完成标准、代码风格、禁止操作
5. **State**：关键信息写进文件，不依赖聊天记录
6. **Feedback**：要求 agent 跑命令后必须看输出再继续
7. **Review**：让 agent 换角色审查自己，或自己再过一遍

## 9. Verification Hook 与新功能测试的配合

**验证 hook 管的是"已有测试必须通过"，不管"必须有测试"。** 所以需要跟 Scope + Acceptance Criteria 配合：

- **Scope + 验收条件**保证"新代码必须带测试"：在任务描述里写清楚验收条件，测试不存在则 hook 报错
- **Verification hook**保证"已有的测试全部通过"：每次写/改文件后自动跑 pytest

```json
// .claude/settings.json - Hook 配置示例
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "command": "python3 -m pytest tests/ -x --tb=short 2>&1 || echo 'VERIFICATION FAILED'"
      }
    ]
  }
}
```

两种结果：
- 测试不存在 → pytest 报 "no tests collected"，hook 返回非零，agent 被迫补测试
- 测试存在但失败 → pytest 返回非零，agent 被迫修复

边界情况：新功能测试需要 mock 外部服务时，用 marker 区分：`pytest -m "not integration"`，跳过需要外部依赖的测试。
