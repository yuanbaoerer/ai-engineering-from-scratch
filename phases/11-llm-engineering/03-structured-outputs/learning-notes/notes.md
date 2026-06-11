# 结构化输出 Q&A 学习笔记

> 来源：阅读 `docs/zh.md` 后与 Claude 的对话整理
> 日期：2026-06-11
> 主题：结构化输出（JSON / Schema / 约束解码）

---

## 一、文档整体结构回顾

课程分四层递进地讲清楚"如何让 LLM 返回可靠的结构化数据"：

1. **问题层**：LLM 返回字符串，应用需要 JSON——这是生产系统崩溃的主要来源之一
2. **方案层**：四层谱系（Prompt → JSON Mode → Schema Mode → Constrained Decoding），可靠性递增
3. **实现层**：从零写 schema 验证器、类到 schema 转换、token 过滤模拟、提取管道
4. **生产层**：OpenAI / Anthropic / Instructor 的真实 SDK 用法

**核心论点**：这不是 prompt 工程问题，而是**解码问题**。模型从左到右生成 token，绝大多数候选会产生无效 JSON——没有约束时模型会选到语法灾难但语义合理的英文 token。

---

## 二、Schema 模式深度解析

### 在谱系中的位置
四层谱系的第三层，向上承接 JSON Mode（只保证语法），向下接通 Constrained Decoding（token 级强制）。

### 它保证什么 vs 不保证什么

| 保证 | 不保证 |
|------|--------|
| 键名正确（不会多/少字段） | 值的正确性（值幻觉） |
| 类型正确（number 就是 number） | 语义合理性 |
| 必需字段存在 | 字段的业务含义 |
| 数值约束（min/max/enum） | 事实准确性 |

**关键洞察**：Schema 模式解决的是**"格式问题"**，不是"事实问题"。文本说 $348，模型可能输出 `{"price": 299.99}`——类型对、数值在范围内，但数字本身错了。

### 各家提供商的接口形式

| 提供商 | 接入方式 | Schema 藏在哪里 |
|--------|----------|----------------|
| OpenAI | `response_format=Product` (Pydantic) | Pydantic 类自动生成 JSON Schema |
| OpenAI | `response_format={"type":"json_schema",...}` | 直接传 schema |
| Anthropic | `tools=[{name, input_schema}]` | 工具的参数 schema |
| Gemini | `response_schema=Product` | 独立字段 |

**要点**：四种接口，底层目标一致——给一个 JSON Schema，强制输出匹配它。

---

## 三、工程化 vs 模型训练端

### 核心结论
**Schema 模式是纯工程化方案，模型权重不需要改变。**

### 四层方案各自靠什么

| 层次 | 主要依赖 | 模型权重是否改变 |
|------|----------|------------------|
| Prompt-based | 模型训练学到的指令遵循能力 | 否，但极度依赖训练质量 |
| JSON Mode | 解码时检测/修正 JSON 语法 | 否 |
| Schema Mode | 解码时 FSM 屏蔽非法 token | **否** |
| Constrained Decoding | 解码时 FSM/PDA 屏蔽非法 token | **否** |

### 那模型训练扮演什么角色？
- 提供基础语言理解和指令遵循能力
- 让屏蔽后的合法 token 中，模型仍能挑出**语义上合适**的那个
- 模型**不知道**自己被约束了，也**不关心**约束是什么

### 为什么工程化方案更优雅
- 模型完全通用，schema 外置、可插拔
- 同一个 GPT-5-mini 可以服务上万个不同 schema 场景
- 不需要为每个 schema 重新标注数据 + 微调 + 部署

**对比**：训练端方案（schema-aware 微调、JSON-native 架构）有学术研究，但生产系统都不用——边际收益小，成本高。

---

## 四、非法 Token 屏蔽的实际机制

### 系统追踪的两个状态
屏蔽器在每一步**同时**追踪：

**状态 A：JSON 解析器状态**
- 当前深度、刚关闭的结构、value/key 位置

**状态 B：Schema 期望状态**
- 当前 schema 字段期望什么类型
- 已知约束（min/max/enum/required）

**只有同时满足两个状态**的 token 才放行。

### 完整运行流程

```
Schema 定义
    ↓ 编译（一次性）
FSM 状态转移表 + Token 分类表
    ↓ 每步生成
增量解析 → 状态转移 → 查合法 token bitmask → 应用 mask 到 logits → 采样
```

### Token 是怎么被分类的
启动时一次预分类（O(词表大小)），每个 token 标上句法标签：
- 结构性（`{`、`[`、`,`、`:`）
- 字符串（`"hello"`、`"foo"`）
- 数字（`123`、`45`）
- 关键字（`true`、`false`、`null`）
- 闭合（`}`、`]`）

**注意多字符 token**：分类看**首字符**为粗筛，最终过滤要看完整状态机。

### 具体例子：`{"price":` 之后合法什么

假设 schema 要求 `price: number`：
- 解析器允许：`0-9`, `"`, `{`, `[`, `t`, `f`, `n`, `-`
- Schema 允许：`0-9`, `-`
- **最终放行**：仅 `0-9` 和 `-`
- `"hello"`、`true`、`{`、`}` 全部屏蔽

### 不同实现方案的复杂度

| 方案 | 方法 | 性能 |
|------|------|------|
| 简单 FSM | 手写规则 | 脆弱，难处理嵌套 |
| Outlines (2023) | Schema → 正则 → FSM | ~微秒/token |
| XGrammar (2025) | PDA + 持久化数据结构 + mask 缓存 | **~100ns/token** |
| OpenAI (闭源) | 类似 XGrammar + 针对性优化 | 高性能 |

**XGrammar 关键创新**：用 PDA（带栈的自动机）替代纯 FSM，**栈负责追踪深度**，状态数只和 schema 复杂度相关，和嵌套深度无关。避免深度嵌套时状态爆炸。

---

## 五、Schema 与函数的关系

### 关键澄清
"结构化输出基于函数参数限制"这个说法**部分对**，但容易误解。

**事实**：
- Schema 是约束的真相之源（✓ 这部分对）
- 在 LLM 生成时系统依据 schema 实时约束 token（✓ 这部分对）
- 但不一定以"函数"形式存在（✗ 这部分需修正）

### Schema 的四种形式（都指向同一个东西）

```python
# 1. Pydantic 模型（最简洁）
class Product(BaseModel):
    price: float
    in_stock: bool

# 2. 原生 JSON Schema
{"type":"object", "properties":{"price":{"type":"number"}}}

# 3. Anthropic 工具定义
{"name": "extract", "input_schema": {...}}

# 4. OpenAI 函数定义
{"type":"function", "function":{"parameters": {...}}}
```

系统**不在乎它怎么来**，只在乎解析出来的 JSON Schema 长什么样。

### 系统的"知道"不是魔法，是编译产物

**请求开始时（一次性的离线工作）**：
```python
schema = {...}
fsm = compile_schema_to_fsm(schema)  # 状态转移表
```

**每步生成时（O(1) 查表）**：
```python
current_state = fsm.transition(prev_state, last_token)
valid_mask = fsm.get_valid_mask(current_state)
logits = apply_mask(model_logits, valid_mask)
```

**关键**：LLM 在 Layer 3 之外，**完全不知道 schema 存在**。约束力完全在 Layer 2/3（解码阶段工程产物）。

---

## 六、关键概念速查

| 概念 | 一句话定义 |
|------|------------|
| JSON Mode | API 标志，保证语法有效 JSON，但不保证 schema 匹配 |
| Schema Mode | JSON + 匹配特定 schema（正确的键、类型、约束） |
| Constrained Decoding | Token 级强制，屏蔽会产生无效输出的 token |
| Token Masking | 把特定 token 的 logit 设为负无穷，使模型无法选中 |
| FSM（有限状态机）| 状态 + 转移函数 + 输入字母表，编译 schema 的基础工具 |
| PDA（下推自动机）| 带栈的 FSM，能处理嵌套结构，避免深度爆炸 |
| Pydantic | Python 类型验证库，可自动生成 JSON Schema |
| Instructor | 跨提供商的 wrapper，验证失败自动重试 |
| 值幻觉 | 输出符合 schema 但值是错的（schema 抓不到） |
| 状态合并 | 状态机优化：等价状态共享 mask，节省内存 |

---

## 七、待消化的开放问题

1. **嵌套深度的实际极限**：XGrammar 的 PDA 方案在多深嵌套时性能会下降？是否有过工业级 benchmark？
2. **跨 schema 的状态机缓存**：相同字段名（如 `price`）跨多个 schema 能否共享编译产物？
3. **约束解码对模型推理速度的实际影响**：100ns/token 是额外开销，但前向传播本身多久？比例是多少？
4. **数组长度的强制**：文档提到 `minItems`/`maxItems` 未必在解码层强制，实际生产里支持到什么程度？
5. **值幻觉的工程化缓解**：除了事后验证，有没有办法在解码阶段就"鼓励"模型选高置信度的 token？

---

## 八、一句话总结

**结构化输出 = JSON Schema 作为契约 + 解码阶段用 FSM/PDA 屏蔽非法 token + 完全对模型透明**。同一个通用 LLM 可以服务任意 schema，不需要为每个 schema 重新训练。Schema 模式解决了"格式可靠性"问题，但"值的事实正确性"需要其他手段（RAG、人工审核、置信度过滤）来补充。
