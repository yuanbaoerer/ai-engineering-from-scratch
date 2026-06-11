# 结构化输出 — JSON Schema、Pydantic、Zod、约束解码

> “礼貌地要求模型返回 JSON” 即便在前沿模型上，也会有 5% 到 15% 的失败率。结构化输出（structured outputs）用约束解码（constrained decoding）弥合这个差距：模型会被字面意义上阻止输出任何违反 schema 的 token。OpenAI 的 strict mode、Anthropic 的 schema 类型化 tool use、Gemini 的 `responseSchema`、Pydantic AI 的 `output_type`，以及 Zod 的 `.parse`，都是同一个思想的五种表面形态。本课会构建 schema 校验器和 strict-mode 合约，学习者将在每条生产级抽取流水线中使用它们。

**类型：** 构建
**语言：** Python（stdlib，JSON Schema 2020-12 子集）
**前置知识：** Phase 13 · 02（function calling 深入讲解）
**时间：** 约 75 分钟

## 学习目标

- 使用合适的约束（enum、min/max、required、pattern）为抽取目标编写 JSON Schema 2020-12。
- 解释为什么 strict mode 和约束解码提供的保证不同于“生成后再校验”。
- 区分三种失败模式：解析错误、schema 违规、模型拒绝。
- 交付一条带类型化修复和类型化拒绝处理的抽取流水线。

## 问题

一个读取采购订单邮件的 agent 需要把自由文本转换成 `{customer, line_items, total_usd}`。有三种方法。

**方法一：提示模型输出 JSON。** “请用 JSON 回复，字段包括 customer、line_items、total_usd。” 在前沿模型上有 85% 到 95% 的时间有效。它会以六种方式失败：缺少大括号、尾随逗号、类型错误、幻觉字段、在 token 限制处被截断、泄露类似 “Here is your JSON:” 的说明性文字。

**方法二：生成后校验。** 自由生成、解析、按 schema 校验，失败时重试。可靠但昂贵——每次重试都要付费，而且截断类 bug 每发生一次就会多花一个回合。

**方法三：约束解码。** provider 在解码时强制执行 schema。无效 token 会从采样分布中被屏蔽。输出保证可解析，也保证能通过校验。失败被收敛为一种模式：拒绝（模型判断输入不适合该 schema）。

到 2026 年，每个前沿 provider 都提供了某种形式的方法三。

- **OpenAI。** `response_format: {type: "json_schema", strict: true}`，如果模型拒绝，则响应中包含 `refusal`。
- **Anthropic。** 对 `tool_use` 输入进行 schema 强制执行；`stop_reason: "refusal"` 不是信号，`end_turn` 且没有 tool call 才是信号。
- **Gemini。** 在请求级别使用 `responseSchema`；2026 年 Gemini 为选定类型提供 token 级 grammar constraints。
- **Pydantic AI。** `output_type=InvoiceModel` 会产生类型为 `InvoiceModel` 的结构化 `RunResult`。
- **Zod (TypeScript)。** 运行时解析器，用 Zod schema 校验 provider 输出；可与 OpenAI 的 `beta.chat.completions.parse` 配合使用。

共同点是：只声明一次 schema，并端到端强制执行。

## 概念

### JSON Schema 2020-12 — 通用语言

每个 provider 都接受 JSON Schema 2020-12。最常用的结构包括：

- `type`：取值为 `object`、`array`、`string`、`number`、`integer`、`boolean`、`null` 之一。
- `properties`：字段名到子 schema 的映射。
- `required`：必须出现的字段名列表。
- `enum`：允许值的封闭集合。
- `minimum` / `maximum`（数字），`minLength` / `maxLength` / `pattern`（字符串）。
- `items`：应用于每个数组元素的子 schema。
- `additionalProperties`：`false` 禁止额外字段（默认值因模式而异）。

OpenAI strict mode 额外增加三项要求：每个属性都必须列在 `required` 中、所有位置都必须设置 `additionalProperties: false`，并且不能有未解析的 `$ref`。如果违反这些要求，API 会在请求阶段返回 400。

### Pydantic，Python 绑定

Pydantic v2 通过 `model_json_schema()` 从 dataclass 形状的模型生成 JSON Schema。Pydantic AI 对其进行封装，因此你可以这样写：

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

agent 框架会在边界处把该 schema 翻译成 OpenAI strict mode、Anthropic `input_schema` 或 Gemini `responseSchema`。模型输出会以类型化的 `Invoice` 实例返回。校验错误会抛出带类型化错误路径的 `ValidationError`。

### Zod，TypeScript 绑定

Zod（`z.object({customer: z.string(), ...})`）是 TS 中的等价物。OpenAI 的 Node SDK 暴露了 `zodResponseFormat(Invoice)`，它会翻译成 API 的 JSON Schema payload。

### 拒绝

Strict mode 不能强迫模型回答。如果输入无法适配 schema（“这封邮件是一首诗，不是发票”），模型会输出一个包含原因的 `refusal` 字段。你的代码必须把它作为一等结果处理，而不是当成失败。拒绝也可作为安全信号：当要求模型从受保护内容邮件中抽取信用卡号时，模型会返回拒绝，并附带安全原因。

### 开放实现中的约束解码

开放权重实现使用三种技术。

1. **基于语法的解码**（`outlines`、`guidance`、`lm-format-enforcer`）：从 schema 构建确定性有限自动机；在每一步屏蔽会违反 FSM 的 token logits。
2. **带 JSON 解析器的 logit masking**：让流式 JSON 解析器与模型同步运行；每一步都计算合法的 next-token 集合。
3. **带校验器的 speculative decoding**：廉价 draft model 提议 token，verifier 强制执行 schema。

商业 provider 会在幕后选择其中一种。2026 年的最佳水平是：对于短结构化输出，比普通生成更快；对于长输出，速度大致相同。

### 三种失败模式

1. **解析错误。** 输出不是合法 JSON。在 strict mode 下不会发生。在非 strict provider 上仍可能发生。
2. **Schema 违规。** 输出能解析，但违反 schema。在 strict mode 下不会发生。在 strict mode 之外很常见。
3. **拒绝。** 模型拒绝。必须作为类型化结果处理。

### 重试策略

当你不在 strict mode 中（Anthropic tool use、非 strict OpenAI、较旧的 Gemini）时，恢复模式是：

```
generate -> parse -> validate -> if fail, inject error and retry, max 3x
```

一次重试通常足够。三次重试能兜住弱模型的偶发抖动。超过三次通常说明 schema 有问题：模型无法为某些输入满足它，需要修复 prompt 或 schema。

### 小模型支持

约束解码适用于小模型。在结构化任务上，一个带 grammar enforcement 的 3B 参数开放模型，会胜过一个只靠原始 prompting 的 70B 参数模型。这就是结构化输出对生产环境重要的主要原因：它把可靠性与模型规模解耦。

## 使用它

`code/main.py` 提供了一个用 stdlib 编写的最小 JSON Schema 2020-12 校验器（types、required、enum、min/max、pattern、items、additionalProperties）。它包装了一个 `Invoice` schema，并让一个假的 LLM 输出通过校验器，演示解析错误、schema 违规和拒绝路径。在生产中，把这个假的输出替换成任意 provider 的真实响应即可。

需要关注：

- 校验器返回类型化的 `[ValidationError]` 列表，其中包含 path 和 message。这正是你希望暴露给重试 prompt 的形状。
- 拒绝分支不会重试。它会记录日志并返回一个类型化拒绝。Phase 14 · 09 会把拒绝用作安全信号。
- `additionalProperties: false` 检查会在对抗性测试输入上触发，展示 strict mode 为什么能关上幻觉字段的大门。

## 交付它

本课会产出 `outputs/skill-structured-output-designer.md`。给定一个自由文本抽取目标（发票、支持工单、简历等），该 skill 会生成一个兼容 strict-mode 的 JSON Schema 2020-12，以及一个与之镜像的 Pydantic 模型，并预置类型化拒绝和重试处理 stub。

## 练习

1. 运行 `code/main.py`。添加第四个测试用例，让它的 `total_usd` 是负数。确认校验器会通过 `minimum` 约束路径拒绝它。

2. 扩展校验器以支持带 discriminator 的 `oneOf`。常见场景是：`line_item` 可以是 product，也可以是 service，并由 `kind` 标记。Strict mode 在这里有微妙规则；请查看 OpenAI 的 structured outputs guide。

3. 把同一个 Invoice schema 写成 Pydantic BaseModel，并将 `model_json_schema()` 输出与你手写的 schema 进行比较。找出 Pydantic 默认设置、而手写版本遗漏的那个字段。

4. 测量拒绝率。构造十个不应被抽取的输入（一段歌词、一个数学证明、一封空邮件），并通过启用 strict mode 的真实 provider 运行它们。统计拒绝与幻觉输出的数量。这就是你设计拒绝感知重试的 ground truth。

5. 从头到尾阅读 OpenAI 的 structured outputs guide。找出它在 strict mode 中明确禁止、但普通 JSON Schema 允许的一个结构。然后设计一个非必要地使用该禁用结构的 schema，并将其重构为 strict-compatible。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| JSON Schema 2020-12 | “Schema 规范” | 每个现代 provider 都支持的 IETF-draft schema dialect |
| Strict mode | “保证 schema” | OpenAI 通过约束解码强制执行 schema 的标志 |
| Constrained decoding | “Logit masking” | 解码时强制执行，屏蔽无效 next-tokens |
| Refusal | “模型拒绝” | 输入无法适配 schema 时的类型化结果 |
| Parse error | “无效 JSON” | 输出无法解析为 JSON；在 strict 下不可能发生 |
| Schema violation | “形状错误” | 已解析但违反 types / required / enum / range |
| `additionalProperties: false` | “不允许额外字段” | 禁止未知字段；OpenAI strict 要求设置 |
| Pydantic BaseModel | “类型化输出” | 会生成并校验 JSON Schema 的 Python 类 |
| Zod schema | “TypeScript 输出类型” | 用于 provider 输出校验的 TS 运行时 schema |
| Grammar enforcement | “开放权重约束解码” | 基于 FSM 的 logit masking，例如 outlines / guidance |

## 延伸阅读

- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — strict mode、refusals 与 schema 要求
- [OpenAI — Introducing structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) — 2024 年 8 月发布文章，解释解码保证
- [Pydantic AI — Output](https://ai.pydantic.dev/output/) — 会序列化到各 provider 的类型化 output_type 绑定
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — canonical spec
- [Microsoft — Structured outputs in Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) — 企业部署说明与 strict-mode 注意事项
