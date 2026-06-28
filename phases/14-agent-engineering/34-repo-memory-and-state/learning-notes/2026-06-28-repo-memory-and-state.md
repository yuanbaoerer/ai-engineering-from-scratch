# 仓库记忆与持久状态

> 日期: 2026-06-28
> Lesson: phases/14-agent-engineering/34-repo-memory-and-state

## 1. 核心命题：聊天易失，仓库持久

智能体的**聊天记录是易失的**（volatile），会话关闭即丢失；**仓库是持久的**（durable），文件落盘可 diff、可重读。

典型失败：会话 1 改完一个文件，会话 2 重新打开却读不到上次上下文，于是重复已做的工作，甚至覆盖已完成文件。

解法是**仓库记忆**：把智能体状态以 JSON 文件形式存到受版本控制的仓库里，让下一个会话、下一个智能体、下一个审查者都读同一份真相来源（system of record）。

## 2. 什么属于仓库记忆

判断标准：**三个月后 CI 重跑时这条信息还有用吗？** 有用→放仓库；没用→放遥测。

| 属于 | 不属于 |
|------|--------|
| 当前任务 ID | 原始聊天记录 |
| 本次会话访问过的文件 | token 级推理轨迹 |
| 智能体做出的假设 | "用户看起来很沮丧" |
| 未解决的阻塞 | 采样结果 |
| 下一步行动 | 供应商特定的模型 ID |

## 3. Schema-first：先定契约，且契约不可绕过

**Schema-first = 在写任何读写状态的代码之前，先用 JSON Schema 把字段/类型/取值范围定义成强制契约，所有读写都被这份契约卡住，漂移当场被拒。**

两层含义缺一不可：

1. **时间上先写**：字段在写第一行业务代码前就钉死。
2. **强制约束力**：`StateManager.load` 和 `commit` 都过 validate，写者绕不开、读者读到的必然合规。只画文档、代码不强制 → 不叫 schema-first，叫"有份文档"。
3. **项目特定**：字段由项目真实工作流反推，不是套通用模板。skill 的 refusal rule：项目没有 acceptance 命令就不准加 `status: done`；没有版本控制就不准交付状态文件。

反义词对照：

| | Schema-first | 无 schema |
|---|---|---|
| 字段来源 | 契约先于写者 | 智能体边用边发明 |
| 漂移 | 塞未声明字段 → 拒绝 | 字段越积越多，无人知哪些是活字段 |
| 审查 | diff 可读 | 每次结构都不同 |
| 迁移 | `schema_version` 未知即 fail-loud | 静默兼容，旧数据是脏的 |

## 4. 手写验证器（stdlib 子集）的细节

课程不用 `jsonschema` 库，自己实现 required/type/enum/pattern/items 子集，验证逻辑带 `$` 路径定位。

两个易错点：

- **`integer` 要排除 `bool`**：Python 里 `True` 是 `int` 子类，`isinstance(True, int)` 为真，所以显式 `not isinstance(value, bool)`。
- **拒绝未声明字段**（`unexpected`）：防智能体偷偷塞 schema 没约定的字段，这是契约的硬执行，杜绝结构漂移。

读端也校验的意义：**任何不合规文件都无法进入工作台**，不论它是谁写的。

## 5. 原子写入：半截文件比没文件更糟

状态文件是真相来源，写一半损坏会让会话在脏状态上恢复且无信号（lesson 引用 Hive Issue #6263：`write_text()` 写入 + 异常被静默吞）。

模式：

```python
def atomic_write(path, content):
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)  # 关键：同目录
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(content); fh.flush(); os.fsync(fh.fileno())  # 落盘
        os.replace(tmp, path)                                       # 原子替换
    except Exception:
        Path(tmp).unlink(missing_ok=True)                           # 清理半成品
        raise
```

**为什么临时文件必须在同目录**：`os.replace` 只在同一文件系统内原子，跨目录/跨分区可能变成"复制+删除"，中途崩溃会损坏。`os.replace` 在 POSIX 和 Windows 上都是原子的。

## 6. 幂等：重试安全

**幂等 = 同一操作执行 1 次和 N 次结果完全一样，重试安全。**

- 读操作天然幂等（查 1 次 = 查 10 次）。
- 写操作分两类：
  - 幂等写（覆盖式/带条件）：`x=5`、`set flag=true`、`UPDATE WHERE id=1`。
  - 非幂等写（累加式/创建式）：`x=x+1`、`INSERT`、发邮件、扣款、上传——每执行一次多一份效果，重试放大伤害。

判别法：**重放后世界比原来多出什么？** 多出"无"→幂等；多出"一份额外东西"→非幂等。

把非幂等改造成幂等的三招：

1. **唯一键去重**（本课用）：执行前查 ID 是否已存在，存在就跳过复用缓存。
2. **带条件覆盖式写**：upsert（`ON CONFLICT DO UPDATE`），让重复收敛到同一终态。
3. **状态机守卫**：扣款前查订单状态是否 `pending`，扣完改 `paid`，重复请求被状态挡住。

工业标准：Stripe 请求头 `Idempotency-Key`、数据库主键 upsert、消息队列"至少一次"+业务去重抬到"恰好一次"、REST 中 `PUT` 幂等而 `POST` 非幂等。

## 7. 幂等键：解决崩溃恢复的重复副作用

崩溃窗口——智能体调了 `send_email`、邮件真发出去了，但还没把"调用成功"写进 checkpoint 就崩溃。恢复时管理器从上一个 checkpoint 重放，认为这次调用没完成，**重新发起 → 又发一封**。

副作用与 checkpoint 持久化之间存在永恒时间差，任何"先做事后记账"系统都有此窗口。

**幂等键 = 给每次工具调用分配的唯一标识，做"这件事做过没有"的去重判断。**

致命细节：**key 必须在副作用发生前就持久化**。错误做法是执行时才 `random_uuid()`——重试时生成新 key，去重形同虚设。正确两种：

- 做法A：key 由调用参数**确定性派生**（同参数→同 key），重试参数没变 key 就没变。
- 做法B（本课）：key 在规划阶段生成并写入 checkpoint，执行时复用已落盘的 key。`pending_calls.jsonl` 就是载体。

机制：执行前登记 pending（副作用前持久化）→ 真正执行 → 记 done。重试时查 key 状态：done→复用结果跳过执行；pending→不重复发起；没有→真执行。

**幂等键与原子写是同一道防线（崩溃后状态可信）的两个构件**：登记表自己写坏则去重不可信，整个幂等保证坍塌，所以登记表也走原子写。

## 8. 四个生产加固模式

| 模式 | 要点 |
|------|------|
| 原子写 | temp + fsync + os.replace，非可选；Hive Issue #6263 是反面教材 |
| 幂等键 | 非幂等工具调用必配；key 在副作用前持久化；登记表本身原子写 |
| 大工件分离 | CSV/长转录/生成物存路径不入 state，checkpoint 保持小而快 |
| 事件溯源+快照 | 每次变更追加 `state.events.jsonl`，定期快照 `state.json`；恢复=读快照+重放其后事件（同构 Postgres WAL） |

## 9. 迁移：未知版本 fail-loud

`schema_version` 整数是契约。管理器加载到未知版本时**拒绝读取**，绝不猜。schema 升级时旁边附一个幂等迁移脚本（`tools/migrate_state.py`），启动时跑。

练习方向（项目演进时契约怎么跟升级）：

- 加 `schema_version` 并写 v1→v2 迁移（`blockers` 改名 `risks`）。
- 后端从本地文件换 SQLite，保持 `StateManager` API 不变（体会抽象边界）。
- 两智能体 50ms 间隔并发写同一状态文件，观察问题并验证原子重命名救命。

## 10. 与前后课的关系

- 前置：第 32 课（最小工作台三文件）→ 本课给它加 schema 与持久化。
- 后续：第 35 课启动时调 manager 的初始化脚本；第 38 课读 state 打分的验证门；第 40 课消费同一 schema 的交接包。
- 横向：LangGraph checkpointer / Letta memory blocks / OpenAI Agents SDK session store = 同一理念不同存储后端；本课 schema 就是这些后端失效时手动读状态所需的工具。

> **一句话收束**：schema-first 钉死契约，原子写保证落盘不损坏，幂等键保证崩溃重试不重复副作用——三者共同回答"崩溃之后，状态还能信吗"。

## 11. 掌握不熟练 / 待消化

- **StateManager.update 未实现**：mission 列了 `load/update/commit`，代码只有 `load/commit`，update 语义（load→改内存→commit）只在 demo 里隐式体现，没动手写过。
- **幂等键落地代码没写过**：能讲清机制，但 `IdempotentToolRunner` + `pending_calls.jsonl` 的实现、以及"模拟崩溃中途重试"的对比 demo 还停留在伪代码，未真正验证。
- **练习 3 迁移脚本**未做：v1→v2（`blockers`→`risks`）的迁移 + 未知版本 fail-loud 没有实现经验。
- **练习 4 SQLite 后端**未做：保持 `StateManager` API 不变换后端的抽象边界，只理解概念未实操。
- **事件溯源 vs 快照**只在概念层，没有写过"读快照+重放事件"的恢复代码，对 WAL 同构的理解停留在类比。
- **并发写竞争**（练习 5）：原子重命名如何救命只停留在推理，未用实验观察"无锁竞争下两智能体同时写"的实际表现。
