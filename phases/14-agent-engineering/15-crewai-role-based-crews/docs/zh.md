# CrewAI：基于角色的团队与流程

> CrewAI 是 2026 年基于角色的多智能体框架。四个原语：Agent、Task、Crew、Process。两种顶层形态：Crew（自主、基于角色的协作）和 Flow（事件驱动、确定性）。文档直言："对于任何生产就绪的应用，从 Flow 开始。"

**类型：** 学习 + 构建
**语言：** Python（stdlib）
**前置条件：** 第 14 阶段 · 12（工作流模式），第 14 阶段 · 14（Actor 模型）
**时间：** 约 75 分钟

## 学习目标

- 说出 CrewAI 的四个原语（Agent、Task、Crew、Process）及其各自负责的内容。
- 区分顺序（Sequential）、层级（Hierarchical）和计划中的共识（Consensus）流程；根据工作负载选择其一。
- 区分 Crew（自主、基于角色）与 Flow（事件驱动、确定性），并解释文档中的生产建议。
- 使用 `@tool` 装饰器和 `BaseTool` 子类连接工具；理解结构化输出与自由文本的区别。
- 说出 CrewAI 的四种记忆类型及其各自适用场景。
- 实现一个 stdlib 的三智能体团队（研究员、撰稿人、编辑），生成一份简报。
- 识别 CrewAI 的三种失败模式：提示膨胀、管理器 LLM 开销、脆弱的交接。

## 问题

采用多智能体框架的团队会遇到同样的障碍。"自主协作"在演示中听起来很棒。然后客户提交了一个 bug，你需要确定性的重放。或者财务部门询问每次运行 LLM 路由团队的成本。或者值班人员需要在凌晨三点知道哪个智能体卡住了。

自由形式的 LLM 路由团队无法干净地回答这些问题。纯 DAG 能回答所有这些问题，但失去了头脑风暴智能体所需的探索性形态。

CrewAI 的拆分对这种权衡是诚实的。Crew 用于协作、基于角色、探索性的工作。Flow 用于事件驱动、代码主导、可审计的生产环境。同一个框架，两种形态，按需选择。

## 概念

### 四个原语

CrewAI 的接口很小。记住这个，其余的都是配置。

- **Agent。** `role + goal + backstory + tools + (optional) llm`。backstory 很重要，它塑造了语气、判断力以及智能体何时停止。工具是智能体可以调用的函数（详见下文）。
- **Task。** `description + expected_output + agent + (optional) context + (optional) output_pydantic`。可复用的工作单元。`expected_output` 是契约。`context` 列出上游任务，其输出会传递给当前任务。`output_pydantic` 强制结构化形态。
- **Crew。** 容器。拥有 `agents` 列表、`tasks` 列表、`process` 以及可选的 `memory` + `verbose` + `manager_llm` 设置。
- **Process。** 执行策略。顺序（Sequential）、层级（Hierarchical）、共识（Consensus）（计划中）。决定运行的形态。

智能体之间不直接看到彼此。任务引用智能体。Crew 对任务排序。Process 决定谁选择下一个任务。这就是完整的思维模型。

> **已验证** CrewAI 0.86（2026-05）。较新版本可能重命名或合并流程类型；在依赖特定形态之前，请检查 [CrewAI Processes 文档](https://docs.crewai.com/concepts/processes)。

### 顺序 vs 层级 vs 共识

- **顺序（Sequential）。** 任务按声明顺序运行。任务 N 的输出作为任务 N+1 的 `context` 可用。成本最低。最可预测。当顺序固定时使用。
- **层级（Hierarchical）。** 一个管理器 Agent（单独的 LLM 调用）在专家之间路由。CrewAI 从你的 `manager_llm` 配置或默认值生成管理器。管理器每轮选择下一个任务，可以拒绝或重新路由。当你有四个或更多专家且顺序确实取决于先前输出时使用。
- **共识（Consensus）。** 计划中，尚未在公共 API 中实现。文档为未来基于投票的流程保留了此名称。目前不要依赖它。

层级在每次专家调用之上增加了每轮 LLM 调用（管理器）。在五步运行中，令牌成本可能增加三倍。仅在需要路由时才为此付费。

### Crew vs Flow

这是文档在 2026 年提出的框架。

- **Crew。** LLM 驱动的自主性。框架在运行时选择形态。适用于：研究、头脑风暴、初稿、路径本身就是答案的地方。难以重放。难以测试。原型成本低。
- **Flow。** 你拥有的事件驱动图。`@start` 标记入口。`@listen(topic)` 标记在另一个步骤发出该主题时触发的步骤。每个步骤都是普通 Python（可以在内部调用 Crew）。适用于：生产环境。可观测。可测试。确定性。

文档 2026 年的生产建议：从 Flow 开始。当自主性值得其成本时，在 Flow 步骤内部通过 `Crew.kickoff()` 调用引入 Crew。Flow 提供审计跟踪，Crew 提供探索能力。组合使用，而非二选一。

### 工具集成

有三种方式给 Agent 提供工具。选择最简单且适合的方式。

1. **`@tool` 装饰器。** 纯函数变成工具。签名是 schema；docstring 是 LLM 看到的描述。最适合一次性辅助函数。

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool` 子类。** 基于类的工具，具有显式参数 schema、异步支持、重试。当工具有状态（客户端、缓存）或需要结构化参数时使用。

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **内置工具包。** CrewAI 提供第一方适配器：`SerperDevTool`、`FileReadTool`、`DirectoryReadTool`、`CodeInterpreterTool`、`RagTool`、`WebsiteSearchTool`。一次导入即可连接。

结构化输出使用 Pydantic。在 Task 上传递 `output_pydantic=MyModel`。CrewAI 验证 LLM 响应是否符合模型，并强制转换或重试。将其与紧凑的 `expected_output` 字符串配对。自由文本输出适用于草稿；结构化输出是下游 Flow 可以消费的。

### 记忆钩子

CrewAI 开箱即提供四种记忆类型。它们可以组合：一个 Crew 可以同时启用所有四种。

> **已验证** CrewAI 0.86（2026-05）。最近的版本通过统一的 `Memory` 系统路由所有内容，该系统封装了这四个存储。下面的概念模型仍然成立，但公共类接口可能在较新版本中合并为单个 `Memory` 入口点；请检查 [CrewAI memory 文档](https://docs.crewai.com/concepts/memory) 了解当前 API。

- **短期（Short-term）。** 单次运行内的对话缓冲区。运行结束时清除。
- **长期（Long-term）。** 跨运行持久化。存储在向量数据库中（默认为 Chroma，可替换）。通过与当前任务的相似性检索。
- **实体（Entity）。** 按实体的事实。"客户 X 使用企业版计划。"按实体键控，而非按相似性。跨运行保留。
- **上下文（Contextual）。** 组装时检索。在 Agent 需要时拉取相关记忆，而非预加载。

在 Crew 上通过 `memory=True` 或按类型配置启用。由你配置的嵌入提供商支持（默认为 OpenAI，可替换为本地）。记忆是 CrewAI 相较于更轻量框架的优势之一；纯 LangGraph 需要你自己连接所有这些。

### 何时适合使用 CrewAI

- 三到六个具有命名角色和协作工作流的 Agent。起草、审查、规划、头脑风暴。
- 路由中 LLM 对下一步的判断是价值一部分的场景（层级模式）。
- 团队更喜欢阅读 `role + goal + backstory` 而非图定义的任何地方。

### 何时不适合使用 CrewAI

- 具有严格排序的确定性 DAG。使用 LangGraph（第 13 课）。图形态是正确的抽象；CrewAI 的角色框架是阻力。
- 亚秒级延迟预算。层级增加往返次数。即使是顺序模式也会序列化包含 backstory 和先前输出的提示。
- 单智能体循环。跳过框架；智能体循环（第 1 课）加工具注册表更短。

第 17 课（智能体框架权衡）以矩阵形式展示了这些。简而言之：CrewAI 位于"协作、基于角色"的角落。

### 依赖形态

独立于 LangChain。Python 3.10 到 3.13。使用 `uv`。星标数：见 [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)（截至 2026-05 的快照）。AWS Bedrock 集成有文档记录；供应商基准测试报告在 QA 工作负载上比 LangGraph 有显著加速，但方法论（数据集、硬件、评估指标）未发布，因此仅将框架供应商数据视为方向性参考。

### 此模式的常见问题

- **Backstory 导致的提示膨胀。** 每个 Agent 2000 字的 backstory 和五人团队会在第一次工具调用之前耗尽上下文预算。将 backstory 控制在 200 字以内。在 Agent 之间复用短语；不要重复五次相同的风格。
- **管理器 LLM 令牌税。** 层级流程在每次专家调用之前增加一次管理器 LLM 调用。在五任务团队中，这是六次 LLM 调用而非五次，且管理器调用携带完整任务列表和先前输出。除非路由取决于输出，否则切换到顺序模式。
- **脆弱的交接。** 任务 N 的 `expected_output` 是"一个大纲"。任务 N+1 将其作为 `context` 读取并尝试解析三个部分。LLM 生成了四个部分。下游 Agent 即兴发挥。通过在任务 N 上使用 `output_pydantic` 修复，这样任务 N+1 读取的是类型化对象，而非自由文本。
- **Crew 当生产用。** 自由形式的 Crew 在没有 Flow 包装的情况下部署到生产环境。输出变异性高；无法重放；值班人员无法将错误运行与正常运行进行对比。用 Flow 包装。

## 构建它

`code/main.py` 实现了两种形态的 stdlib 版本以及一个三智能体团队。

形态：

- `Agent`、`Task` 数据类，匹配 CrewAI 的接口。
- `SequentialCrew.kickoff(inputs)` 按声明顺序运行任务，将输出作为 `context` 传递。
- `HierarchicalCrew.kickoff(topic)` 添加管理器 Agent，每轮选择下一个专家，在"完成"时停止。
- `Flow` 具有 `@start` 和 `@listen(topic)` 装饰器、一个小型事件循环和跟踪。
- `tool(name)` 装饰器，镜像 CrewAI 的 `@tool` 形态。
- `Memory` 具有 `short_term`、`long_term`、`entity` 存储；模拟的相似性使用 numpy。
- 模拟 LLM 响应是基于角色和输入前缀的硬编码字符串。无网络。确定性。

具体演示：研究员、撰稿人、编辑团队生成关于"agent engineering 2026"的简报。研究员拉取（模拟的）来源。撰稿人起草。编辑精简。同一个团队通过 Flow 运行以展示确定性形态。

运行：

```bash
python3 code/main.py
```

跟踪覆盖：顺序团队通过 `context` 传递输出，层级团队具有管理器选择（研究员、撰稿人、编辑，然后"完成"），Flow 使用显式主题（`researched`、`drafted`、`edited`）运行相同的三个步骤，通过 `@tool` 路由的工具调用，以及跨两次启动持久化的长期记忆。

Crew 跟踪是流动的；管理器原则上可以重新排序。Flow 跟踪是固定的。这个选择就是课程。

## 使用它

- **CrewAI Flow** 用于生产环境。即使 Flow 只是一个调用 `Crew.kickoff()` 的步骤。Flow 提供审计边界。
- **CrewAI Crew（顺序）** 用于有明确顺序的协作工作，特别是初稿和审查循环。
- **CrewAI Crew（层级）** 当路由取决于输出且你有四个或更多专家时。
- **LangGraph**（第 13 课）用于显式状态机、持久恢复、严格排序。
- **AutoGen v0.4**（第 14 课）用于 Actor 模型并发和故障隔离。
- **OpenAI Agents SDK**（第 16 课）用于具有交接和防护的 OpenAI 优先产品。
- **Claude Agent SDK**（第 17 课）用于具有子智能体会话存储的 Claude 优先产品。

## 交付它

`outputs/skill-crew-or-flow.md` 为任务选择 Crew vs Flow 并构建最小实现。对没有 backstory 的 Crew、没有显式主题的 Flow、专家少于三人的层级进行硬拒绝。

## 常见陷阱

- **Backstory 作为装饰。** 它塑造输出。每个 Agent 测试三个变体；变异性是真实的。选择一个，冻结它。
- **跳过 `expected_output`。** 没有每个任务的契约，下游任务会接收 LLM 生成的任何内容。Crew 运行；审计失败。
- **记忆始终开启。** 长期记忆每次运行都写入。向量数据库增长。检索变得嘈杂。将写入范围限定在事实持久的任务上。
- **管理器提示漂移。** 层级的管理器提示是隐式的。如果路由变得奇怪，在详细模式下转储并阅读。
- **Crew 中的工具副作用。** Crew 可能比预期更多次调用工具。POST、DELETE、支付属于 Flow 步骤，绝不是 Crew 工具。

## 练习

1. 将顺序团队转换为 Flow。计算可变性降低的接触点。注意可读性下降的地方。
2. 向团队添加实体记忆：关于客户的事实跨启动持久化。验证检索拉取了正确的实体。
3. 实现一个层级流程，管理器在撰稿人输出少于三个段落之前拒绝路由到编辑。跟踪重试。
4. 为（模拟的）Web 搜索连接一个 `BaseTool` 子类。比较跟踪形态与 `@tool` 装饰器版本。
5. 向编辑任务添加 `output_pydantic=Brief`，其中 `Brief` 具有 `title`、`summary`、`sections`。使撰稿人任务输出一次格式错误的 JSON；在跟踪中验证 CrewAI 的重试行为。
6. 阅读 CrewAI 的文档介绍。将玩具移植到真实的 `crewai` API。stdlib 版本跳过了哪些保证？
7. 将 AgentOps 或 Langfuse（第 24 课）连接到真实运行。你在 stdlib 版本中错过了哪些跟踪？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| Agent | "角色" | Role + goal + backstory + tools |
| Task | "工作单元" | Description + expected output + assignee + 可选结构化输出 |
| Crew | "智能体团队" | Agents + Tasks + Process 的容器 |
| Process | "执行策略" | Sequential / Hierarchical / Consensus（计划中） |
| Flow | "确定性工作流" | 事件驱动、代码主导、可测试 |
| Backstory | "角色提示" | Agent 的语气和判断力塑造者 |
| `@tool` | "函数工具" | 将函数转换为 Agent 可调用工具的装饰器 |
| `BaseTool` | "类工具" | 基于类的工具，具有参数 schema、重试、异步支持 |
| 实体记忆 | "按实体的事实" | 作用域为客户/账户/问题的记忆 |
| 长期记忆 | "跨运行记忆" | 基于向量的记忆，在启动之间保留 |
| 上下文记忆 | "即时检索" | 在 Agent 需要时拉取的记忆 |
| 管理器 LLM | "路由 Agent" | 层级流程中选择下一个任务的额外 LLM |
| `expected_output` | "任务契约" | 告诉 Agent（和审计）返回什么形态的字符串 |

## 延伸阅读

- [CrewAI 文档介绍](https://docs.crewai.com/en/introduction)：概念和推荐的生产路径
- [CrewAI Flows 指南](https://docs.crewai.com/en/concepts/flows)：事件驱动形态、`@start`、`@listen`
- [CrewAI 工具参考](https://docs.crewai.com/en/concepts/tools)：`@tool`、`BaseTool`、内置工具包
- [CrewAI 记忆](https://docs.crewai.com/en/concepts/memory)：短期、长期、实体、上下文
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)：多智能体何时有用，何时无用
- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview)：状态机替代方案
