# LangGraph — 代理的状态机

> 手写的 ReAct 循环是一个 `while True`。用 LangGraph 编写的 ReAct 循环是一个可以保存检查点、中断、分支和时间回溯的图。代理本身没有改变，改变的是围绕它的运行框架。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 11 · 09（函数调用）、阶段 11 · 14（模型上下文协议）
**时间：** 约 75 分钟

## 问题

你部署了一个函数调用代理。它运行三个回合没问题，然后出了问题：模型尝试了一个返回 500 的工具，用户在任务中途改变了主意，或者代理决定在没有人签字的情况下退款。`while True:` 循环没有钩子。你无法暂停它，无法回滚它，也无法分支出"如果模型选择了另一个工具会怎样"。一旦这东西超过演示阶段，代理就变成了一个黑盒——要么成了，要么没成。

下一步其实很明显。代理本身就是一个状态机——系统提示词加上消息历史加上待处理的工具调用再加上下一步动作。把状态机显式化：为"模型思考"、"工具运行"、"人工审批"设置节点，为它们之间的条件转换设置边。一旦图变得显式，框架就能免费获得四个能力：检查点（在步骤之间保存状态）、中断（暂停等待人工干预）、流式传输（流式输出 token 和中间事件）和时间回溯（回滚到之前的状态并尝试不同的分支）。

LangGraph 就是提供这种抽象的库。它不是 LangChain 意义上的代理框架（"这是一个 AgentExecutor，祝你好运"）。它是一个具有一流状态、一流持久化和一流中断的图运行时。代理循环是你画出来的，而不是手写出来的。

## 概念

![LangGraph StateGraph：节点、边和检查点](../assets/langgraph-stategraph.svg)

`StateGraph` 有三个要素。

1. **状态。** 一个类型化字典（TypedDict 或 Pydantic 模型），在图中流动。每个节点接收完整状态并返回部分更新，LangGraph 使用每个字段的*归约器*合并它们——对于应该累积的列表使用 `operator.add`，默认是覆盖。
2. **节点。** Python 函数 `state -> partial_state`。每个都是离散步骤："调用模型"、"运行工具"、"总结"。
3. **边。** 节点之间的转换。静态边通向一个地方。条件边使用路由函数 `state -> next_node_name`，这样图可以根据模型输出进行分支。

你编译图。编译绑定拓扑结构，附加一个检查点器（可选，但对生产环境必不可少），并返回一个可运行对象。你用初始状态和 `thread_id` 调用它。执行的每一步都会持久化一个以 `(thread_id, checkpoint_id)` 为键的检查点。

### 四大超能力

**检查点。** 每次节点转换都将新状态写入存储（测试用内存，生产用 Postgres/Redis/SQLite）。用相同的 `thread_id` 再次调用图来恢复。图从暂停的地方继续。

**中断。** 用 `interrupt_before=["human_review"]` 标记节点，执行在该节点运行前停止。状态被保留。你的 API 向用户返回"等待审批"。后续对相同 `thread_id` 的请求带 `Command(resume=...)` 恢复执行。

**流式传输。** `graph.stream(state, mode="updates")` 在事件发生时产生状态增量。`mode="messages"` 在模型节点内部流式传输 LLM token。`mode="values"` 产生完整快照。你选择什么在 UI 中展示。

**时间回溯。** `graph.get_state_history(thread_id)` 返回完整的检查点日志。将任何之前的 `checkpoint_id` 传递给 `graph.invoke`，你就能从那个点分叉。对于调试（"如果模型选择了工具 B 而不是 A 会怎样？"）和重放生产轨迹的回归测试非常有用。

### 归约器才是关键

每个状态字段都有一个归约器。大多数默认值都没问题——新值覆盖旧值。但消息列表需要 `operator.add`，这样新消息会追加而不是替换。并行边通过归约器合并更新。如果两个节点都更新了 `messages` 而你忘记了 `Annotated[list, add_messages]`，第二个会静默胜出，你丢失半个回合。归约器是这个库唯一微妙的东西；搞对了它，其余的就顺理成章了。

### 四个节点的 ReAct 图

一个生产级别的 ReAct 代理有四个节点和两条边：

1. `agent` — 用当前消息历史调用 LLM。返回助手消息（可能包含 tool_calls）。
2. `tools` — 执行最后一条助手消息中的任何 tool_calls，将工具结果追加为工具消息。
3. 一条从 `agent` 出发的条件边，如果最后一条消息有 tool_calls 则路由到 `tools`，否则到 `END`。
4. 一条从 `tools` 回到 `agent` 的静态边。

就这样。你获得了完整的 ReAct 循环（思考 → 行动 → 观察 → 思考 → …），包括检查点、中断和流式传输，只用了大约 40 行代码。

### StateGraph vs Send（扇出）

`Send(node_name, state)` 让一个节点可以分发并行子图。示例：代理决定同时查询三个检索器。每个 `Send` 生成目标节点的并行执行；它们的输出通过状态归约器合并。这就是 LangGraph 表达协调器-工作者模式的方式，无需线程原语。

### 子图

编译后的图可以充当另一个图中的节点。外图看到的是单个节点；内图有自己的状态和自己的检查点。这就是团队构建监督者-工作者代理的方式：监督者图将用户意图路由到按域划分的工作者子图。

## 构建

### 步骤 1：状态和节点

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages` 是让消息列表累积而不是覆盖的归约器。忘记它是最常见的 LangGraph bug。

### 步骤 2：用线程运行

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

每次更新都是一个字典 `{node_name: state_delta}`。你的前端可以把这些流式传输到 UI，这样用户就能看到"代理正在思考… 调用 search_web… 获得结果… 回答中。"

### 步骤 3：添加人工在环中断

标记一个节点，使其在运行前暂停执行。

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # pause before every tool call
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] is set. Inspect proposed tool calls.
# If approved:
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# If denied: write a rejection message and resume
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

状态、检查点和线程都在中断中保持不变。除了执行期间，没有什么保存在内存中。

### 步骤 4：用于调试的时间回溯

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Fork from a prior checkpoint
target = history[3].config  # three steps back
for event in app.stream(None, target, stream_mode="values"):
    pass  # replay from that point forward
```

传入 `None` 作为输入会从给定检查点重放；传入一个值会在恢复前将它作为更新追加到该检查点的状态。这就是如何在不重跑整个对话的情况下重现一次糟糕的代理运行。

### 步骤 5：为生产环境更换检查点器

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

提供了 SQLite、Redis 和 Postgres。`MemorySaver` 是给测试用的。任何需要跨重启持久化的东西都需要真正的存储。

## 技能

> 你把代理构建为图，而不是 `while True` 循环。

在使用 LangGraph 之前，花 60 秒做一下设计：

1. **命名节点。** 每个离散决策或副作用动作都是一个节点。"代理思考"、"工具运行"、"审核员批准"、"响应流式传输。"如果你列不出来，说明任务还不适合用代理的形态。
2. **声明状态。** 最小的 TypedDict，每个列表字段都要有归约器。不要把所有东西都塞进 `messages`；把任务特定的字段（一个工作中的 `plan`、一个 `budget` 计数器、一个 `retrieved_docs` 列表）提升到顶层。
3. **画边。** 除非下一步取决于模型输出，否则都是静态边。每条条件边需要一个带命名分支的路由函数。
4. **一开始就选择检查点器。** 测试用 `MemorySaver`，其他用 Postgres/Redis/SQLite。不要不带检查点器就部署——没有检查点器就没有恢复、没有中断、没有时间回溯。
5. **在工具运行之前决定中断，而不是之后。** 审批放在进入副作用节点的边上，这样你可以在危害发生前取消；验证放在模型输出的边上，这样你可以廉价地拒绝坏调用。
6. **默认使用流式传输。** UI 用 `mode="updates"`，模型节点内部 token 级流式传输用 `mode="messages"`，评估时完整快照用 `mode="values"`。

拒绝部署一个没有检查点器的 LangGraph 代理。拒绝部署一个在副作用*之后*才中断的代理。拒绝部署一个没有 `add_messages` 作为归约器的 `messages` 字段。

## 练习

1. **简单。** 用一个计算器工具和一个网络搜索工具实现上面的四节点 ReAct 图。验证 `list(app.get_state_history(config))` 对于两个回合的对话返回至少四个检查点。
2. **中等。** 添加一个在 `agent` 之前运行的 `planner` 节点，将结构化的 `plan: list[str]` 写入状态。让 `agent` 标记计划步骤为已完成。如果 `plan` 在检查点恢复后丢失（归约器错误），测试失败。
3. **困难。** 构建一个使用 `Send` 在三个子图（`researcher`、`writer`、`reviewer`）之间路由的监督者图。每个子图有自己的状态和检查点器。在外图上添加 `interrupt_before=["writer"]`，以便人工可以审批研究简报。确认从之前的检查点时间回溯只重跑分叉的分支。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------------|-----------------------|
| StateGraph | "LangGraph 图" | 在编译前添加节点和边的构建器对象。 |
| Reducer | "字段如何合并" | 当节点返回对该字段的更新时应用的函数 `(old, new) -> merged`；默认是覆盖，`add_messages` 是追加。 |
| Thread | "对话 ID" | 一个 `thread_id` 字符串，限定一个会话的所有检查点。 |
| Checkpoint | "暂停的状态" | 节点转换后完整图状态的持久化快照，以 `(thread_id, checkpoint_id)` 为键。 |
| Interrupt | "暂停等待人工" | `interrupt_before` / `interrupt_after` 在节点边界停止执行；用 `Command(resume=...)` 恢复。 |
| Time-travel | "从之前的步骤分叉" | `graph.invoke(None, config_with_old_checkpoint_id)` 从该检查点向前重放。 |
| Send | "并行子图分发" | 一个节点可以返回的构造器，用来生成目标节点的 N 个并行执行。 |
| Subgraph | "作为节点的编译图" | 用作另一个图节点的编译好的 StateGraph；保持自己的状态作用域。 |

## 延伸阅读

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) — StateGraph、归约器、检查点器和中断的规范参考。
- [LangGraph 概念：状态、归约器、检查点器](https://langchain-ai.github.io/langgraph/concepts/low_level/) — 本课使用的心理模型，来自源头。
- [LangGraph 持久化和检查点](https://langchain-ai.github.io/langgraph/concepts/persistence/) — Postgres/SQLite/Redis 存储、检查点命名空间和线程 ID 的详细信息。
- [LangGraph 人工在环](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) — `interrupt_before`、`interrupt_after`、`Command(resume=...)` 和编辑状态模式。
- [Yao et al., "ReAct：在语言模型中协同推理和行动"（ICLR 2023）](https://arxiv.org/abs/2210.03629) — 每个 LangGraph 代理实现的模式；阅读它了解推理链的理由。
- [Anthropic — 构建有效的代理（2024 年 12 月）](https://www.anthropic.com/research/building-effective-agents) — 哪些图形状（链、路由器、协调器-工作者、评估器-优化器）更值得选择以及何时选择。
- 阶段 11 · 09（函数调用） — 每个 LangGraph 代理节点重用的工具调用原语。
- 阶段 11 · 14（模型上下文协议） — 通过 MCP 适配器插入 LangGraph `ToolNode` 的外部工具发现。
- 阶段 11 · 17（代理框架权衡） — 何时选择 LangGraph 而不是 CrewAI、AutoGen 或 Agno。