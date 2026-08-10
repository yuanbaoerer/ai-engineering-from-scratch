"""Deterministic verification gate with coverage floor, --strict mode, and signed overrides.

Combines a task's scope_report, rule_report, feedback log, and an optional
coverage_report into a single verification_report.json. No LLM judges; LLM
judgment lives on the reviewer side (Phase 14 · 39). Overrides require a signed
entry in overrides.jsonl with reason, user, and HEAD commit.

Run: python3 code/main.py
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

HERE = Path(__file__).parent
OVERRIDES_PATH = HERE / "overrides.jsonl"
COVERAGE_FLOOR_DEFAULT = 0.80
COVERAGE_REGRESSION_DELTA = 0.01

# 审计密钥用于签名覆盖条目。
# 在生产环境中，应从密钥管理器读取。
# 失败关闭：仅当明确设置VERIFY_DEMO_MODE=1时才回退到演示密钥，
# 并大声警告，以免意外进入CI。
_OVERRIDE_SECRET_ENV = "VERIFY_OVERRIDE_SECRET"
_DEMO_MODE_ENV = "VERIFY_DEMO_MODE"


def _load_override_secret() -> str:
    """加载覆盖签名密钥。

    根据文档，覆盖需要签名以确保审计线索。
    这个函数从环境变量加载密钥，如果没有设置则使用演示密钥。

    Returns:
        覆盖签名密钥

    Raises:
        RuntimeError: 如果没有设置必要的环境变量
    """
    secret = os.environ.get(_OVERRIDE_SECRET_ENV)
    if secret:
        return secret
    if os.environ.get(_DEMO_MODE_ENV) == "1":
        print(
            f"WARNING: {_OVERRIDE_SECRET_ENV} unset and {_DEMO_MODE_ENV}=1; "
            "using insecure demo secret. Do not record real overrides in this mode.",
            file=sys.stderr,
        )
        return "demo-override-secret-do-not-ship"
    raise RuntimeError(
        f"refused to start: {_OVERRIDE_SECRET_ENV} is unset. "
        f"Set the env var, or pass {_DEMO_MODE_ENV}=1 to run the lesson demo only."
    )


@dataclass
class Finding:
    """验证关卡发现的数据结构。

    每个Finding代表验证过程中发现的一个问题或检查结果。
    根据文档，severity为"block"的发现会阻止passed: true，
    而"warn"级别的发现只会被标注到判定中。

    Attributes:
        code: 发现代码，用于唯一标识发现类型，如"acceptance.missing"
        severity: 严重级别，可以是"block"（阻止）或"warn"（警告）
        detail: 详细描述，解释发现的具体内容
    """
    code: str
    severity: str
    detail: str


@dataclass
class Artifacts:
    """验证关卡的输入产物集合。

    根据文档，验证关卡读取代理已产生的产物并做出判断。
    Artifacts包含了所有必要的输入数据，对应文档中提到的：
    - scope_report.json: 范围契约报告
    - rule_report.json: 规则报告
    - feedback_record.jsonl: 反馈日志
    - coverage_report.json: 覆盖率报告（可选）

    Attributes:
        task_id: 任务唯一标识符
        acceptance_commands: 验收命令列表，这些命令必须运行且退出码为零
        feedback: 反馈记录列表，包含命令执行结果
        scope_report: 范围报告，包含forbidden_writes和off_scope_writes
        rule_report: 规则报告列表，包含规则是否通过
        coverage_report: 覆盖率报告，包含当前和之前的覆盖率
        head_commit: 当前HEAD提交哈希，用于覆盖审计
    """
    task_id: str
    acceptance_commands: list[str]
    feedback: list[dict[str, object]]
    scope_report: dict[str, object]
    rule_report: list[dict[str, object]]
    coverage_report: dict[str, float] | None = None  # {"current": 0.84, "previous": 0.85}
    head_commit: str = ""


@dataclass
class VerdictReport:
    """验证关卡的判定报告。

    根据文档，验证关卡每次任务关闭时生成一份verification_report.json，
    写入outputs/verification/<task_id>.json。CI使用相同的路径。

    Attributes:
        task_id: 任务唯一标识符
        passed: 是否通过验证，为true当且仅当没有block级别的发现
        strict: 是否启用严格模式，将warn提升为block
        findings: 发现列表，包含所有检查结果
        coverage: 覆盖率信息
        head_commit: 当前HEAD提交哈希
    """
    task_id: str
    passed: bool
    strict: bool
    findings: list[Finding] = field(default_factory=list)
    coverage: dict[str, float] | None = None
    head_commit: str = ""


def _acceptance_findings(art: Artifacts) -> list[Finding]:
    """检查验收命令的执行情况。

    根据文档，验收命令是"退出码为零即为'完成'的shell命令"。
    这个函数检查两个关键点：
    1. 所有验收命令是否都已运行
    2. 所有验收命令的退出码是否为零

    这两个检查都是block级别的，因为文档指出：
    "所有验收命令都已运行"和"所有验收命令退出码为零"都是block级别的检查。

    Args:
        art: 包含任务产物的Artifacts对象

    Returns:
        Finding列表，包含所有验收相关的发现
    """
    findings: list[Finding] = []
    commands_run = [str(rec.get("command")) for rec in art.feedback]
    accept_set = set(art.acceptance_commands)
    for cmd in art.acceptance_commands:
        if cmd not in commands_run:
            findings.append(Finding("acceptance.missing", "block", f"never ran: {cmd}"))
    for rec in art.feedback:
        cmd_str = str(rec.get("command"))
        if rec.get("exit_code") is None:
            findings.append(Finding("feedback.null_exit", "block", f"missing exit for {cmd_str}"))
        elif rec.get("exit_code") != 0 and cmd_str in accept_set:
            findings.append(
                Finding("acceptance.failed", "block", f"acceptance exit {rec.get('exit_code')} on {cmd_str}")
            )
    return findings


def _scope_findings(art: Artifacts) -> list[Finding]:
    """检查范围报告中的写入操作。

    根据文档，范围检查有两个关键点：
    1. 范围检查没有禁止的写入 - 这是block级别
    2. 范围检查没有越界的写入 - 这是block或warn级别

    forbidden_writes是绝对禁止的写入，如修改关键配置文件。
    off_scope_writes是越界写入，可能被允许但需要警告。

    Args:
        art: 包含任务产物的Artifacts对象

    Returns:
        Finding列表，包含所有范围相关的发现
    """
    findings: list[Finding] = []
    if art.scope_report.get("forbidden_writes"):
        findings.append(Finding("scope.forbidden", "block",
                                f"forbidden writes: {art.scope_report['forbidden_writes']}"))
    if art.scope_report.get("off_scope_writes"):
        findings.append(Finding("scope.off_scope", "warn",
                                f"off-scope writes: {art.scope_report['off_scope_writes']}"))
    return findings


def _rule_findings(art: Artifacts) -> list[Finding]:
    """检查规则报告中的失败规则。

    根据文档，所有block级别的规则都必须通过。
    规则报告中的每个规则都有一个"passed"字段，指示规则是否通过。
    如果规则未通过，则产生一个block级别的发现。

    Args:
        art: 包含任务产物的Artifacts对象

    Returns:
        Finding列表，包含所有规则失败相关的发现
    """
    return [Finding("rule.failed", "block", f"rule failed: {row.get('slug')}")
            for row in art.rule_report if not row.get("passed")]


def _coverage_findings(art: Artifacts, floor: float) -> list[Finding]:
    """检查覆盖率报告。

    根据文档，覆盖率下限是一等检查：
    1. 覆盖率必须达到最低要求（默认80%）
    2. 覆盖率不能比上次合并的覆盖率下降超过1个百分点

    这体现了Anthropic Hybrid Norm：将可验证的奖励（测试+覆盖率）与标准评分配对。
    覆盖率检查是确定性的，而LLM判断属于审阅端。

    Args:
        art: 包含任务产物的Artifacts对象
        floor: 覆盖率下限，默认0.80

    Returns:
        Finding列表，包含所有覆盖率相关的发现
    """
    findings: list[Finding] = []
    if not art.coverage_report:
        findings.append(Finding("coverage.missing", "warn",
                                "no coverage_report.json; cannot enforce floor"))
        return findings
    current = float(art.coverage_report.get("current", 0.0))
    previous = float(art.coverage_report.get("previous", current))
    if current < floor:
        findings.append(Finding("coverage.below_floor", "block",
                                f"coverage {current:.2%} below floor {floor:.0%}"))
    delta = previous - current
    if delta > COVERAGE_REGRESSION_DELTA and not math.isclose(
        delta, COVERAGE_REGRESSION_DELTA, rel_tol=1e-9
    ):
        findings.append(Finding("coverage.regression", "block",
                                f"coverage dropped {delta:.2%} (prev {previous:.2%} -> {current:.2%})"))
    elif delta > 0 and not math.isclose(delta, 0.0, abs_tol=1e-12):
        findings.append(Finding("coverage.minor_regression", "warn",
                                f"coverage dropped {delta:.2%}"))
    return findings


def verify(
    art: Artifacts,
    strict: bool = False,
    coverage_floor: float = COVERAGE_FLOOR_DEFAULT,
) -> VerdictReport:
    """验证关卡的主函数。

    这是验证关卡的核心函数，实现了文档中描述的确定性验证过程。
    它将所有检查结果合并为一个判定结果。

    根据文档，验证关卡是"工作台产物上的确定性函数"，
    不能使用LLM判断。LLM判断属于审阅端。

    Args:
        art: 包含任务产物的Artifacts对象
        strict: 是否启用严格模式，将warn提升为block
        coverage_floor: 覆盖率下限，默认0.80

    Returns:
        VerdictReport对象，包含验证结果
    """
    findings = (
        _acceptance_findings(art)
        + _scope_findings(art)
        + _rule_findings(art)
        + _coverage_findings(art, coverage_floor)
    )
    if strict:
        # --strict promotes every warning to a block. Opt-in by release branch only.
        findings = [Finding(f.code, "block" if f.severity == "warn" else f.severity, f.detail)
                    for f in findings]
    blocking = [f for f in findings if f.severity == "block"]
    return VerdictReport(
        task_id=art.task_id,
        passed=not blocking,
        strict=strict,
        findings=findings,
        coverage=art.coverage_report,
        head_commit=art.head_commit,
    )


def _sign(payload: dict[str, object]) -> str:
    """为覆盖条目生成HMAC签名。

    根据文档，覆盖是签名变更，而不是代理决定。
    每个覆盖都需要签名，以确保审计线索。

    Args:
        payload: 要签名的载荷字典

    Returns:
        签名字符串（前32个十六进制字符）
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hmac.new(_load_override_secret().encode(), canonical, hashlib.sha256).hexdigest()[:32]


def record_override(
    task_id: str, finding_code: str, reason: str, user_id: str, head_commit: str
) -> dict[str, object]:
    """记录一个签名的覆盖条目。

    根据文档，覆盖需要：
    1. 所有五个字段都必须填写：task_id, finding_code, reason, user_id, head_commit
    2. 覆盖条目会被签名，以确保审计线索
    3. 覆盖记录在overrides.jsonl文件中

    这体现了文档中"签名覆盖日志，而非Slack线程"的原则。

    Args:
        task_id: 任务ID
        finding_code: 发现代码
        reason: 覆盖原因
        user_id: 覆盖用户ID
        head_commit: 当前HEAD提交哈希

    Returns:
        包含签名的覆盖条目字典

    Raises:
        ValueError: 如果缺少任何必需字段
    """
    if not all([task_id, finding_code, reason, user_id, head_commit]):
        raise ValueError("override requires task_id, finding_code, reason, user_id, head_commit")
    payload = {
        "task_id": task_id,
        "finding_code": finding_code,
        "reason": reason,
        "user_id": user_id,
        "head_commit": head_commit,
        "ts": time.time(),
    }
    payload["signature"] = _sign({k: v for k, v in payload.items() if k != "signature"})
    with OVERRIDES_PATH.open("a") as fh:
        fh.write(json.dumps(payload) + "\n")
    return payload


def verify_signature(entry: dict[str, object]) -> bool:
    """验证覆盖条目的签名。

    根据文档，运行时拒绝任何缺少签名的覆盖；
    审计线索由git追踪。

    Args:
        entry: 覆盖条目字典

    Returns:
        签名是否有效
    """
    expected = entry.get("signature")
    payload = {k: v for k, v in entry.items() if k != "signature"}
    return hmac.compare_digest(_sign(payload), str(expected))


def main() -> None:
    """演示验证关卡的工作流程。

    这个函数演示了三个场景：
    1. T-001: 干净通过 - 所有检查都通过
    2. T-002: 范围蔓延 - 有范围违规和规则失败
    3. T-003: 缺少验收 - 没有运行验收命令

    这些场景对应文档中描述的典型失败模式：
    - "看起来不错"但没有实际测试
    - "测试通过了"但没有运行记录
    - "验收标准已满足"但被宽泛解读

    还演示了签名覆盖的过程，展示如何通过签名覆盖来绕过警告级别的发现。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="promote every warn to block")
    ap.add_argument("--floor", type=float, default=COVERAGE_FLOOR_DEFAULT)
    args = ap.parse_args()

    accept = ["pytest -x test_app.py::test_signup_rejects_short_password"]
    cases = [
        Artifacts(
            task_id="T-001",
            acceptance_commands=accept,
            feedback=[{"command": accept[0], "exit_code": 0}],
            scope_report={"forbidden_writes": [], "off_scope_writes": []},
            rule_report=[{"slug": "done/tests-pass", "passed": True}],
            coverage_report={"current": 0.84, "previous": 0.85},
            head_commit="a1b2c3d",
        ),
        Artifacts(
            task_id="T-002",
            acceptance_commands=accept,
            feedback=[{"command": accept[0], "exit_code": 0}],
            scope_report={"forbidden_writes": ["scripts/release.sh"], "off_scope_writes": ["README.md"]},
            rule_report=[{"slug": "forbidden/no-release-script-edits", "passed": False}],
            coverage_report={"current": 0.62, "previous": 0.80},
            head_commit="b2c3d4e",
        ),
        Artifacts(
            task_id="T-003",
            acceptance_commands=accept,
            feedback=[],
            scope_report={"forbidden_writes": [], "off_scope_writes": []},
            rule_report=[{"slug": "done/tests-pass", "passed": False}],
            head_commit="c3d4e5f",
        ),
    ]

    for art in cases:
        report = verify(art, strict=args.strict, coverage_floor=args.floor)
        path = HERE / f"verification_report_{art.task_id}.json"
        path.write_text(json.dumps(
            {"task_id": report.task_id, "passed": report.passed, "strict": report.strict,
             "head_commit": report.head_commit, "coverage": report.coverage,
             "findings": [asdict(f) for f in report.findings]},
            indent=2) + "\n")
        flag = " (strict)" if report.strict else ""
        print(f"task {report.task_id}{flag}: passed={report.passed} findings={len(report.findings)}")
        for f in report.findings:
            print(f"  [{f.severity}] {f.code}: {f.detail}")
        print()

    # Demo a signed override on the off-scope warning that T-002 actually emits.
    try:
        entry = record_override(
            task_id="T-002",
            finding_code="scope.off_scope",
            reason="reviewer approved README update for the new signup contract",
            user_id="rohitg00",
            head_commit="b2c3d4e",
        )
        print(f"override recorded: signature={entry['signature']} verified={verify_signature(entry)}")
    except RuntimeError as exc:
        print(f"override demo skipped: {exc}")


if __name__ == "__main__":
    main()
