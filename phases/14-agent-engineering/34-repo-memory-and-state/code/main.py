"""Schema-first agent state with atomic writes.

Writes JSON Schema files for `agent_state.json` and `task_board.json`,
implements a tiny stdlib validator that handles the subset we need
(required, type, enum, pattern, items), and a StateManager with
temp-and-rename writes so a partial failure cannot corrupt the file.

Run: python3 code/main.py
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------
# HERE  指向本文件所在的目录（code/）
# WORK  指向工作目录，所有运行时产物（schema、状态文件）都写在这里
HERE = Path(__file__).parent
WORK = HERE / "workdir"

# ---------------------------------------------------------------------------
# 智能体状态 Schema（agent_state.schema.json）
# ---------------------------------------------------------------------------
# 仓库记忆的"合约"：定义了哪些字段必须存在、每个字段的类型和约束。
# 有了这个合约，任何不符合规范的写入都会在提交时被拒绝，
# 避免脏数据污染工作台的真实来源（source of truth）。
#
# 课程文档中的核心分类：
#   属于仓库记忆：当前任务 ID、访问过的文件、假设、阻塞问题、下一步行动
#   不属于仓库记忆：原始聊天记录、推理轨迹、模型 ID 等易失信息
STATE_SCHEMA: dict[str, Any] = {
    "$id": "agent_state.schema.json",
    "type": "object",
    # 必需字段——缺少任何一个都会在验证时抛出 SchemaError
    "required": ["schema_version", "active_task_id", "touched_files", "next_action"],
    "properties": {
        # schema_version: 用于迁移的版本号，当前固定为 1。
        # 当 schema 变更时，管理器会拒绝加载它无法迁移的版本。
        "schema_version": {"type": "integer", "enum": [1]},
        # active_task_id: 当前正在处理的任务 ID。
        # 可以是 null（无任务）、空字符串（待分配）或 T-xxx 格式（如 T-001）。
        # pattern 确保 ID 符合 T-至少三位数字 的格式。
        "active_task_id": {"type": ["string", "null"], "pattern": r"^(T-\d{3,}|)$"},
        # touched_files: 本次会话中智能体读写过的文件路径列表。
        # 用于追踪变更范围，避免重复工作。
        "touched_files": {"type": "array", "items": {"type": "string"}},
        # assumptions: 智能体在推理过程中做出的假设记录。
        # 暴露给下一个会话，减少重复推理。
        "assumptions": {"type": "array", "items": {"type": "string"}},
        # blockers: 阻碍任务完成的问题列表（如缺少 API key、依赖未安装等）。
        # 下一个会话可以立即看到这些阻塞点。
        "blockers": {"type": "array", "items": {"type": "string"}},
        # next_action: 智能体建议的下一步操作，供下一个会话直接执行。
        "next_action": {"type": "string"},
    },
}

# ---------------------------------------------------------------------------
# 任务看板 Schema（task_board.schema.json）
# ---------------------------------------------------------------------------
# 看板是一个任务数组，每个任务都有固定的字段结构。
# status 的枚举值限定了任务只能在四种状态之间流转：
#   todo -> in_progress -> done/blocked
# owner 的枚举值限定了任务只能分配给三类角色：
#   builder（构建者）、reviewer（审查者）、human（人类）
BOARD_SCHEMA: dict[str, Any] = {
    "$id": "task_board.schema.json",
    "type": "array",
    "items": {
        "type": "object",
        "required": ["id", "goal", "owner", "acceptance", "status"],
        "properties": {
            # id: 任务唯一标识，格式 T-至少三位数字（如 T-001、T-0123）
            "id": {"type": "string", "pattern": r"^T-\d{3,}$"},
            # goal: 任务目标的自然语言描述
            "goal": {"type": "string"},
            # owner: 任务归属，只能是 builder / reviewer / human 之一
            "owner": {"type": "string", "enum": ["builder", "reviewer", "human"]},
            # acceptance: 验收标准列表，每个元素是一条可执行的测试命令或断言
            "acceptance": {"type": "array", "items": {"type": "string"}},
            # status: 任务状态，只能是以下四种之一
            "status": {"type": "string", "enum": ["todo", "in_progress", "done", "blocked"]},
        },
    },
}


# ---------------------------------------------------------------------------
# 自定义异常：SchemaError
# ---------------------------------------------------------------------------
# 当验证失败时抛出，包含 JSONPath 格式的错误位置和具体原因。
# 调用方（如 StateManager.commit）可以捕获此异常来阻止脏数据写入。
class SchemaError(Exception):
    pass


# ---------------------------------------------------------------------------
# 内部辅助：类型检查
# ---------------------------------------------------------------------------
# JSON Schema 的 type 字段可以是单个字符串（"string"）或字符串数组（["string", "null"]）。
# 这个函数处理两种情况，逐一检查值是否匹配任一允许的类型。
# 注意：Python 的 bool 是 int 的子类，所以 integer 类型必须排除 bool。
def _check_type(value: Any, types: str | list[str]) -> bool:
    type_list = [types] if isinstance(types, str) else types
    for t in type_list:
        if t == "object" and isinstance(value, dict):
            return True
        if t == "array" and isinstance(value, list):
            return True
        if t == "string" and isinstance(value, str):
            return True
        if t == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if t == "null" and value is None:
            return True
    return False


# ---------------------------------------------------------------------------
# 核心：递归 Schema 验证器
# ---------------------------------------------------------------------------
# 实现了 JSON Schema 规范的一个子集：
#   - type:     类型检查（支持联合类型如 ["string", "null"]）
#   - enum:     值必须在允许的枚举列表中
#   - pattern:  字符串必须匹配正则表达式
#   - required: 对象必须包含所有必需字段
#   - properties: 递归验证嵌套对象的每个属性
#   - items:    递归验证数组中的每个元素
#
# path 参数使用 JSONPath 语法（如 $.touched_files[0]），
# 当验证失败时，错误信息会精确指出问题所在的字段位置。
def validate(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    # 类型检查：值的类型必须匹配 schema 中声明的类型
    if "type" in schema and not _check_type(value, schema["type"]):
        raise SchemaError(f"{path}: expected {schema['type']}, got {type(value).__name__}")
    # 枚举检查：值必须在允许的枚举列表中
    if "enum" in schema and value not in schema["enum"]:
        raise SchemaError(f"{path}: {value!r} not in {schema['enum']}")
    # 正则检查：字符串必须匹配 pattern
    if "pattern" in schema and isinstance(value, str) and not re.match(schema["pattern"], value):
        raise SchemaError(f"{path}: {value!r} does not match /{schema['pattern']}/")
    # 对象验证：检查必需字段、未知字段，然后递归验证每个属性
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                raise SchemaError(f"{path}: missing required field {key!r}")
        properties = schema.get("properties", {})
        unexpected = sorted(set(value.keys()) - set(properties.keys()))
        if unexpected:
            raise SchemaError(f"{path}: unexpected fields {unexpected}")
        for key, sub in properties.items():
            if key in value:
                validate(value[key], sub, f"{path}.{key}")
    # 数组验证：递归验证数组中的每个元素
    if isinstance(value, list) and "items" in schema:
        for idx, item in enumerate(value):
            validate(item, schema["items"], f"{path}[{idx}]")


# ---------------------------------------------------------------------------
# 原子写入：tempfile + fsync + os.replace
# ---------------------------------------------------------------------------
# 为什么原子写入不是可选的？
#   课程文档引用了 Hive 项目的 bug（Issue #6263）：
#   直接用 write_text() 写入 state.json，如果中途异常，文件会处于半写状态。
#   下一个会话从损坏的状态恢复，没有任何信号表明数据不完整。
#
# 原子写入的三步协议：
#   1. 在目标目录创建临时文件（mkstemp），避免跨文件系统重命名失败
#   2. 写入内容 -> flush -> fsync（确保数据落盘，不只是在 OS 缓存中）
#   3. os.replace：POSIX 和 Windows 上都是原子操作，用临时文件覆盖目标文件
#
# 如果任何步骤失败，临时文件会被清理，目标文件保持不变。
def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# StateManager：加载、验证、提交状态
# ---------------------------------------------------------------------------
# 核心设计原则：
#   - load()  读取 -> 验证 -> 返回，确保只返回符合 schema 的数据
#   - commit() 验证 -> 原子写入，确保脏数据永远无法持久化
#
# 这是"Schema 优先"理念的直接体现：
#   定义合约（schema） -> 验证合约 -> 拒绝违约
#   而不是：写入数据 -> 事后检查 -> 发现问题已经太晚
#
# 生产环境扩展方向（课程文档提到的模式）：
#   - 事件溯源：每次变更追加到 state.events.jsonl，定期快照到 state.json
#   - 幂等键：非幂等工具调用前记录 ID 到 pending_calls.jsonl，重试时跳过已执行的调用
#   - 大型工件分离：不把 CSV/转录文本存在状态中，只存路径
class StateManager:
    def __init__(self, state_path: Path, schema: dict[str, Any]):
        self.state_path = state_path  # 状态文件的持久化路径
        self.schema = schema          # 用于验证的 JSON Schema

    def load(self) -> Any:
        """从磁盘加载状态，验证后返回。"""
        raw = json.loads(self.state_path.read_text())
        validate(raw, self.schema)  # 加载时验证，拒绝损坏的文件
        return raw

    def commit(self, state: Any) -> None:
        """验证状态并原子性地写入磁盘。"""
        validate(state, self.schema)  # 提交前验证，阻止脏数据
        atomic_write(self.state_path, json.dumps(state, indent=2) + "\n")


def main() -> None:
    # -----------------------------------------------------------------------
    # 第一步：初始化工作目录和 schema 文件
    # -----------------------------------------------------------------------
    # 将 JSON Schema 写入 workdir/schemas/，方便人类和工具审查。
    # 这些文件是"合约的合约"——它们定义了状态文件的结构。
    WORK.mkdir(exist_ok=True)
    schema_dir = WORK / "schemas"
    schema_dir.mkdir(exist_ok=True)
    (schema_dir / "agent_state.schema.json").write_text(json.dumps(STATE_SCHEMA, indent=2) + "\n")
    (schema_dir / "task_board.schema.json").write_text(json.dumps(BOARD_SCHEMA, indent=2) + "\n")

    state_path = WORK / "agent_state.json"
    board_path = WORK / "task_board.json"

    # -----------------------------------------------------------------------
    # 第二步：创建 StateManager 实例
    # -----------------------------------------------------------------------
    # 每个 StateManager 绑定一个状态文件和一个 schema。
    # load/commit 操作会自动进行 schema 验证。
    mgr = StateManager(state_path, STATE_SCHEMA)
    board_mgr = StateManager(board_path, BOARD_SCHEMA)

    # -----------------------------------------------------------------------
    # 第三步：写入初始状态
    # -----------------------------------------------------------------------
    # agent_state.json：智能体的运行时状态
    #   - schema_version: 1（用于未来的迁移）
    #   - active_task_id: None（尚未开始任何任务）
    #   - touched_files: []（未读写任何文件）
    #   - assumptions: []（无假设）
    #   - blockers: []（无阻塞）
    #   - next_action: "pick next task"（下一步：选择任务）
    initial_state = {
        "schema_version": 1,
        "active_task_id": None,
        "touched_files": [],
        "assumptions": [],
        "blockers": [],
        "next_action": "pick next task",
    }
    # task_board.json：任务看板，初始时有一个待办任务
    #   - T-001: 验证 /signup 的请求负载
    #   - owner: builder（由构建者执行）
    #   - acceptance: 一条 pytest 命令作为验收标准
    #   - status: todo（待办）
    initial_board = [
        {
            "id": "T-001",
            "goal": "validate /signup payloads",
            "owner": "builder",
            "acceptance": ["pytest -x test_app.py::test_signup_rejects_short_password"],
            "status": "todo",
        }
    ]
    mgr.commit(initial_state)        # 原子写入 agent_state.json
    board_mgr.commit(initial_board)  # 原子写入 task_board.json

    # -----------------------------------------------------------------------
    # 第四步：模拟智能体工作——更新状态
    # -----------------------------------------------------------------------
    # 智能体读取状态，选择 T-001 任务，更新 active_task_id 和 next_action。
    # 然后提交到磁盘。下一个会话加载时会看到这个更新后的状态。
    state = mgr.load()
    board = board_mgr.load()
    state["active_task_id"] = board[0]["id"]   # 标记当前任务为 T-001
    state["next_action"] = "read existing /signup handler"  # 下一步：阅读现有的 /signup 处理器
    mgr.commit(state)

    # -----------------------------------------------------------------------
    # 第五步：验证往返——加载并打印最终状态
    # -----------------------------------------------------------------------
    # 这一步证明：写入 -> 加载 -> 数据完整无损。
    # 在实际工作台中，这就是下一个会话启动时发生的事情。
    print("state:", json.dumps(mgr.load(), indent=2))
    print("board:", json.dumps(board_mgr.load(), indent=2))

    # -----------------------------------------------------------------------
    # 第六步：演示 Schema 验证拒绝脏数据
    # -----------------------------------------------------------------------
    # 故意构造一个违反 schema 的状态：active_task_id = "T-bogus"
    # 模式 ^T-\d{3,}$ 要求 T- 后跟至少三位数字，"bogus" 不满足。
    # 验证器会抛出 SchemaError，commit 不会执行，文件保持不变。
    # 这就是"Schema 优先"的价值：
    #   错误的写入在写入之前就被拒绝，而不是事后才发现文件损坏。
    bad = dict(state)
    bad["active_task_id"] = "T-bogus"
    try:
        mgr.commit(bad)
    except SchemaError as exc:
        print("rejected bad write:", exc)


if __name__ == "__main__":
    main()
