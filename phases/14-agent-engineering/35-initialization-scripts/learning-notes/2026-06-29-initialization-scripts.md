# Agent 初始化脚本学习笔记

> 日期: 2026-06-29

## 1. 核心问题：Setup Tax

Agent 每次启动都重复做相同的"热身"工作：猜测 Python 版本、反复列出 repo 文件、尝试 import 不存在的包。这些操作浪费大量 token。

**解决方案**：写一个 init 脚本，在 agent 启动前运行，执行所有探测并将结果写入 `init_report.json`。Agent 启动时直接读取报告，不再重复检查。

## 2. 代码结构

`code/main.py` 包含：

| 组件 | 作用 |
|------|------|
| `Probe` 数据类 | 每个探测返回 `(name, status, detail, duration_ms)` |
| `@_timed` 装饰器 | 自动计时，超过 3s 标记为 warn |
| 6 个探测函数 | runtime / dependencies / test_command / env / state_freshness / lkg_diff |
| `lock_is_fresh()` | 检查 prereqs.lock 是否在 TTL 内且指纹匹配 |
| `main()` | 检查 lock → 跑探测 → 写 report → 有 fail 就 exit 1 |

## 3. 六个探测

| 探测 | 检查内容 | 失败含义 |
|------|----------|----------|
| `probe_runtime` | Python 版本 ≥ 要求 | 版本不兼容 |
| `probe_dependencies` | 依赖包可 import | 缺包 |
| `probe_test_command` | 测试命令在 PATH 上 | 无法验证工作 |
| `probe_env` | 必需环境变量存在 | 运行时会报错 |
| `probe_state_freshness` | state 文件是否过期 | 陈旧状态有风险 |
| `probe_lkg_diff` | 相对 last-known-good 的 diff ≤ 50 文件 | 漂移过大需人工确认 |

## 4. 三大设计原则

| 原则 | 含义 |
|------|------|
| **Fail loud** | 探测失败 = 拒绝启动，不静默跳过 |
| **幂等性** | 跑两次结果一致（除时间戳），可安全用于 CI/hooks |
| **无网络/无 LLM** | 所有探测 < 3s，纯本地确定性操作 |

## 5. Lock 文件机制

```
第 1 次运行 → 跑全部探测 → 全部通过 → 写 prereqs.lock（指纹 + 时间戳）
第 2 次运行（24h 内）→ 读 lock → 指纹没变 → 跳过探测
```

**类比**：Docker 层缓存 — 依赖没变就复用上一层结果，不重新安装。

**失效条件**：超过 24h TTL / 依赖变了（指纹不匹配）/ lock 文件损坏。

## 6. 可移植性

Init 脚本**不依赖任何特定框架**。它只做本地确定性检查：读文件、查环境变量、跑 git diff、用 importlib 检查包。所以 Bash、Make、Claude Code hooks、GitHub Actions、Docker entrypoint 都能调用它。

## 7. CI 工作流 vs 个人使用

| 场景 | 是否需要 CI | 是否需要 init 脚本 |
|------|------------|-------------------|
| 个人 Claude Code 写代码 | 不需要 | 有用（可选） |
| 团队多 agent 协作 | 需要 | 必须 |
| 自动化 CI/CD 管道 | 需要 | 必须 |

个人偶尔用 Claude Code 改几个文件，了解概念即可，不必强求。

## 8. 与 requirements.txt 同步

`REQUIRED_DEPS` 应该从 `requirements.txt` 动态读取，而非硬编码：

```python
def _load_deps_from_requirements():
    req_path = Path(__file__).parent.parent / "requirements.txt"
    return [line.split("==")[0].split(">=")[0].strip()
            for line in req_path.read_text().splitlines()
            if line.strip() and not line.startswith("#")]
```

一处维护，避免手动同步。

## 9. 给已有项目构造 Init

步骤：
1. 盘点项目需要检查什么（版本、依赖、测试命令、环境变量）
2. 参考 `main.py` 结构，改 `REQUIRED_*` 常量适配项目
3. 集成到工作流（Claude Code hook / CI / 手动）

## 10. Skill 自动化

`outputs/skill-init-script.md` 是一个自动化模板，让 Claude 帮你项目一键生成完整 init 体系（init_agent.py + CI workflow + pre_task hook + 文档）。使用方式：在 Claude Code 里说"用 init-script skill"。

## 待消化的问题

- 本课程没有 tests 目录，按 AGENTS.md 规范应该有 5+ 单元测试，可以自己补写
- init 脚本与 Phase 33（规则）、Phase 34（Repo Memory）、Phase 38（验证门）、Phase 40（Handoff）的联动关系值得后续深入
