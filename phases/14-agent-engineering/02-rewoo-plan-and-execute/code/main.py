"""Toy ReWOO — Planner, Workers, Solver. Stdlib only.

Demonstrates the decoupled pattern from Xu et al. (arXiv:2305.18323):
  1. Planner emits a DAG of (tool, args) steps with references (#E1, #E2, ...).
  2. Workers run each step in topological order.
  3. Solver composes the final answer from question + plan + evidence.

Compare run_rewoo() vs run_react() at the bottom for token-use intuition.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable


# =============================================================================
# 数据结构：计划步骤与计划
# =============================================================================

@dataclass
class PlanStep:
    """计划 DAG 中的一个节点。

    每个步骤包含：
    - id: 唯一标识符，用作证据引用（如 "E1", "E2"）
    - tool: 要调用的工具名称（如 "search", "round_million"）
    - args: 工具参数字典，值可以是字面量或对前序步骤的引用（如 "#E1"）

    示例：
        PlanStep("E1", "search", {"query": "capital of France"})
        PlanStep("E2", "search", {"query": "population of #E1"})  # #E1 引用 E1 的输出
    """
    id: str
    tool: str
    args: dict[str, Any]


@dataclass
class Plan:
    """完整的计划 DAG，包含有序的步骤列表。

    步骤之间通过 #E1, #E2 等引用建立依赖关系。
    执行前需要进行拓扑排序以确定正确的执行顺序。
    """
    steps: list[PlanStep]


# =============================================================================
# 工具注册表：管理可用工具的分发
# =============================================================================

class ToolRegistry:
    """工具注册表，负责工具的注册和调用分发。

    在 ReWOO 架构中，Worker 通过此注册表执行具体的工具调用。
    注册表将工具名称映射到实际的函数实现。
    """

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., str]] = {}

    def register(self, name: str, fn: Callable[..., str]) -> None:
        """注册一个工具函数。

        Args:
            name: 工具名称，与计划中的 tool 字段对应
            fn: 工具函数，接受关键字参数并返回字符串结果
        """
        self._tools[name] = fn

    def dispatch(self, name: str, args: dict[str, Any]) -> str:
        """分发工具调用，返回结果或错误信息。

        错误处理策略：
        - 未知工具名返回错误字符串（不抛异常）
        - 工具执行异常被捕获并转为错误字符串

        这种设计确保 Worker 执行不会中断整个流程，
        Solver 可以在最终答案中优雅地处理部分失败。
        """
        fn = self._tools.get(name)
        if fn is None:
            return f"error: unknown tool {name!r}"
        try:
            return fn(**args)
        except Exception as e:
            return f"error: {type(e).__name__}: {e}"


# =============================================================================
# 引用解析：将 #E1 等占位符替换为实际证据
# =============================================================================

# 正则表达式：匹配 #E 后跟数字的引用模式
REFERENCE_RE = re.compile(r"#E(\d+)")


def resolve_references(value: Any, evidence: dict[str, str]) -> Any:
    """递归解析参数中的证据引用。

    将字符串中的 "#E1", "#E2" 等占位符替换为对应 Worker 的输出。
    非字符串类型的值（如数字、列表）原样返回。

    示例：
        resolve_references("population of #E1", {"E1": "Paris"})
        # => "population of Paris"

        resolve_references("#E2", {"E1": "Paris", "E2": "11.2 million"})
        # => "11.2 million"
    """
    if not isinstance(value, str):
        return value
    return REFERENCE_RE.sub(lambda m: evidence.get(f"E{m.group(1)}", m.group(0)),
                            value)


# =============================================================================
# 拓扑排序：确定计划步骤的执行顺序
# =============================================================================

def topological(plan: Plan) -> list[PlanStep]:
    """对计划步骤进行拓扑排序，确保依赖关系被正确处理。

    算法：贪心拓扑排序
    1. 维护已解析步骤集合 known
    2. 每轮遍历未处理步骤，找出所有依赖都已解析的步骤
    3. 将这些步骤加入结果，并标记为已解析
    4. 重复直到所有步骤处理完毕或检测到循环

    Args:
        plan: 包含步骤的计划

    Returns:
        拓扑排序后的步骤列表

    Raises:
        RuntimeError: 如果计划存在循环或无法解析的引用

    示例：
        对于计划 [E1(search), E2(search #E1), E3(round #E2)]
        执行顺序为：E1 -> E2 -> E3（因为 E2 依赖 E1，E3 依赖 E2）
    """
    resolved: list[PlanStep] = []  # 已确定顺序的步骤
    known: set[str] = set()        # 已解析步骤的 ID 集合
    pending = list(plan.steps)     # 待处理步骤

    while pending:
        progress = False
        rest: list[PlanStep] = []  # 本轮无法处理的步骤
        for step in pending:
            # 提取当前步骤所有参数中的引用（如 ["1", "2"] 表示引用 #E1, #E2）
            refs = REFERENCE_RE.findall(str(step.args))
            # 检查所有引用的依赖是否都已解析
            if all(f"E{r}" in known for r in refs):
                resolved.append(step)
                known.add(step.id)
                progress = True
            else:
                rest.append(step)
        # 如果没有进展，说明存在循环或无法解析的引用
        if not progress:
            raise RuntimeError("cyclic plan or unresolved reference")
        pending = rest
    return resolved


# =============================================================================
# Worker 执行：按拓扑顺序运行计划节点
# =============================================================================

def run_workers(plan: Plan, tools: ToolRegistry) -> dict[str, str]:
    """执行计划中的所有 Worker，收集证据。

    流程：
    1. 对计划进行拓扑排序
    2. 按顺序执行每个步骤：
       - 解析参数中的引用（替换为已执行步骤的输出）
       - 通过工具注册表调用工具
       - 将结果存入证据字典
    3. 返回所有步骤的输出（证据）

    Args:
        plan: 要执行的计划
        tools: 工具注册表

    Returns:
        证据字典，键为步骤 ID（如 "E1"），值为工具输出

    示例：
        对于计划：
            E1: search("capital of France") -> "Paris"
            E2: search("population of #E1") -> "11.2 million metro"
            E3: round_million("#E2") -> "11 million"

        返回：{"E1": "Paris", "E2": "11.2 million metro", "E3": "11 million"}
    """
    evidence: dict[str, str] = {}
    for step in topological(plan):
        # 解析参数中的引用，将 #E1 替换为 evidence["E1"] 的实际值
        bound_args = {k: resolve_references(v, evidence) for k, v in step.args.items()}
        # 调用工具并存储结果
        evidence[step.id] = tools.dispatch(step.tool, bound_args)
    return evidence


# =============================================================================
# 脚本化组件：用于演示的固定实现
# =============================================================================

class ScriptedPlanner:
    """脚本化的 Planner 实现。

    在真实系统中，Planner 会调用 LLM 来生成计划。
    这里使用预定义的固定计划进行演示。
    """

    def __init__(self, plan: Plan) -> None:
        self.plan = plan

    def plan_for(self, question: str) -> Plan:
        """为给定问题返回预定义的计划。"""
        return self.plan


class ScriptedSolver:
    """脚本化的 Solver 实现。

    在真实系统中，Solver 会调用 LLM 来整合证据生成答案。
    这里使用简单的模板替换。
    """

    def __init__(self, answer_template: str) -> None:
        self.template = answer_template

    def solve(self, question: str, plan: Plan, evidence: dict[str, str]) -> str:
        """使用模板和证据生成最终答案。"""
        return self.template.format(**evidence)


# =============================================================================
# 模拟工具：用于演示的假搜索和数字处理
# =============================================================================

def fake_search(query: str) -> str:
    """模拟搜索工具。

    根据查询关键词返回预设结果，模拟真实的搜索引擎调用。
    在真实系统中，这会调用 Google API 或其他搜索服务。
    """
    if "capital of france" in query.lower():
        return "Paris"
    if "population of paris" in query.lower():
        return "11.2 million metro"
    if "capital of germany" in query.lower():
        return "Berlin"
    return f"no result for {query!r}"


def rounded_million(text: str) -> str:
    """模拟数字处理工具。

    从文本中提取数字并四舍五入到百万位。
    示例：输入 "11.2 million metro"，输出 "11 million"
    """
    m = re.search(r"([0-9]+\.?[0-9]*)", text)
    if not m:
        return "unknown"
    return f"{round(float(m.group(1)))} million"


# =============================================================================
# ReWOO 运行记录：跟踪执行过程和 Token 消耗
# =============================================================================

@dataclass
class ReWOORun:
    """记录一次完整 ReWOO 执行的运行数据。

    用于对比 ReWOO 和 ReAct 的 Token 消耗差异。
    """
    question: str                    # 用户问题
    plan: Plan                       # 执行的计划
    evidence: dict[str, str] = field(default_factory=dict)  # 收集的证据
    answer: str = ""                 # 最终答案
    planner_chars: int = 0           # Planner 阶段消耗的字符数
    worker_chars: int = 0            # Worker 阶段消耗的字符数
    solver_chars: int = 0            # Solver 阶段消耗的字符数


def run_rewoo(question: str, planner: ScriptedPlanner,
              tools: ToolRegistry, solver: ScriptedSolver) -> ReWOORun:
    """执行完整的 ReWOO 流程。

    三个阶段：
    1. Planner 阶段：生成计划 DAG
       - 消耗 = 问题长度 + 所有步骤的工具名和参数长度
    2. Worker 阶段：按拓扑顺序执行所有步骤
       - 消耗 = 每个步骤的参数长度 + 输出长度
    3. Solver 阶段：整合证据生成答案
       - 消耗 = 问题长度 + 所有证据长度 + 答案长度

    Args:
        question: 用户问题
        planner: 计划器
        tools: 工具注册表
        solver: 求解器

    Returns:
        ReWOORun 对象，包含完整执行记录和 Token 消耗统计
    """
    # 阶段 1: Planner 生成计划
    plan = planner.plan_for(question)
    planner_chars = len(question) + sum(len(s.tool) + len(str(s.args))
                                        for s in plan.steps)

    # 阶段 2: Workers 执行计划，收集证据
    evidence = run_workers(plan, tools)
    worker_chars = sum(len(str(s.args)) + len(v) for s, v in zip(plan.steps,
                                                                 evidence.values()))

    # 阶段 3: Solver 整合证据，生成答案
    answer = solver.solve(question, plan, evidence)
    solver_chars = len(question) + worker_chars + len(answer)

    return ReWOORun(question=question, plan=plan, evidence=evidence,
                    answer=answer,
                    planner_chars=planner_chars, worker_chars=worker_chars,
                    solver_chars=solver_chars)


# =============================================================================
# ReAct 模拟：用于对比 Token 消耗
# =============================================================================

def run_react_mock(question: str, tools: ToolRegistry,
                   trajectory: list[tuple[str, dict[str, Any]]]) -> int:
    """模拟 ReAct 风格的交织循环，计算总字符消耗。

    ReAct 的关键问题：每一步都要携带完整的先前上下文。

    消耗模型：
    - 每步消耗 = 原始问题 + 历史上下文 + 当前工具调用
    - 历史上下文 = 所有先前步骤的（工具名 + 参数 + 观察结果 + 固定开销）
    - 最后一步还要加上最终答案的消耗

    这导致 Token 消耗随步骤数二次方增长。

    Args:
        question: 用户问题
        tools: 工具注册表
        trajectory: 模拟的执行轨迹，每项为 (工具名, 参数)

    Returns:
        总字符消耗（近似 Token 数）
    """
    prompt_chars = len(question)
    total = 0
    history_chars = 0  # 累积的历史上下文长度

    for name, args in trajectory:
        # 每步都包含：原始问题 + 历史上下文 + 当前工具调用
        total += prompt_chars + history_chars + len(name) + len(str(args))
        # 执行工具获取观察结果
        obs = tools.dispatch(name, args)
        # 历史上下文累积：工具调用 + 观察结果 + 固定开销（思考/格式等）
        history_chars += len(name) + len(str(args)) + len(obs) + 40

    # 最后一步还要加上最终答案的消耗
    total += prompt_chars + history_chars
    return total


# =============================================================================
# 主函数：演示 ReWOO 与 ReAct 的对比
# =============================================================================

def main() -> None:
    """演示 ReWOO 的完整执行流程，并与 ReAct 进行 Token 消耗对比。"""
    print("=" * 70)
    print("REWOO — Planner, Workers, Solver (Phase 14, Lesson 02)")
    print("=" * 70)

    # 1. 注册可用工具
    tools = ToolRegistry()
    tools.register("search", fake_search)           # 搜索工具
    tools.register("round_million", rounded_million)  # 数字处理工具

    # 2. 定义计划 DAG
    #    这是一个三步计划，用于回答"法国首都的人口是多少，四舍五入到百万？"
    #
    #    依赖关系：E2 依赖 E1（需要先知道首都名称），E3 依赖 E2（需要先获取原始人口）
    #
    #    DAG 结构：
    #    E1 (search "capital of France")
    #     ↓
    #    E2 (search "population of #E1")  # #E1 会被替换为 E1 的输出
    #     ↓
    #    E3 (round_million "#E2")         # #E2 会被替换为 E2 的输出
    plan = Plan(steps=[
        PlanStep("E1", "search", {"query": "capital of France"}),
        PlanStep("E2", "search", {"query": "population of #E1"}),
        PlanStep("E3", "round_million", {"text": "#E2"}),
    ])

    # 3. 创建脚本化的 Planner 和 Solver
    planner = ScriptedPlanner(plan)
    solver = ScriptedSolver(
        "The capital of France is {E1}; rounded population is {E3}."
    )

    # 4. 执行 ReWOO 流程
    run = run_rewoo("What is the population of the capital of France, rounded?",
                    planner, tools, solver)

    # 5. 打印执行结果
    print("\nPLAN")
    for step in run.plan.steps:
        print(f"  {step.id}: {step.tool}({step.args})")

    print("\nEVIDENCE")
    for k, v in run.evidence.items():
        print(f"  {k} -> {v}")

    print(f"\nFINAL: {run.answer}")

    # 6. 与 ReAct 进行 Token 消耗对比
    #    模拟 ReAct 执行相同的三步任务
    react_chars = run_react_mock(
        run.question, tools,
        [("search", {"query": "capital of France"}),
         ("search", {"query": "population of Paris"}),
         ("round_million", {"text": "11.2 million metro"})])

    # 计算 ReWOO 总消耗
    rewoo_chars = run.planner_chars + run.worker_chars + run.solver_chars

    print("\nTOKEN INTUITION (chars, approximate)")
    print(f"  react total  : {react_chars}")
    print(f"  rewoo total  : {rewoo_chars}")
    print(f"  ratio        : {react_chars / max(rewoo_chars, 1):.2f}x")
    print("\npaper claim: ~5x fewer tokens on HotpotQA. toy approximates the shape.")


if __name__ == "__main__":
    main()
