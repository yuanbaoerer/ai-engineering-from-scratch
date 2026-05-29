# 结构化输出：JSON、Schema 验证与约束解码

> 你的 LLM 返回的是字符串。你的应用需要的是 JSON。这个差距导致的生产系统崩溃比任何模型幻觉都多。结构化输出是自然语言与类型化数据之间的桥梁。做好了，你的 LLM 就变成了一个可靠的 API。做砸了，你凌晨 3 点还在用正则表达式解析自由文本。

**类型：** Build
**语言：** Python
**前置要求：** Phase 10，课程 01-05（从零构建 LLM）
**时间：** 约 90 分钟
**相关内容：** Phase 5 · 20（结构化输出与约束解码）涵盖了解码器层面的理论（FSM/CFG logit 处理器、Outlines、XGrammar）。本课程专注于生产 SDK 表面（OpenAI `response_format`、Anthropic 工具使用、Instructor）——如果想了解 API 底层发生了什么，请先阅读 Phase 5 · 20。

## 学习目标

- 使用 OpenAI 和 Anthropic API 参数实现 JSON 模式和 schema 约束输出
- 构建一个 Pydantic 验证层，拒绝格式错误的 LLM 输出并通过错误反馈进行重试
- 解释约束解码如何在 token 层面强制生成有效 JSON，无需后处理
- 设计可靠的提取 prompt，将非结构化文本转换为类型化数据结构

## 问题

你问 LLM："从这段文本中提取产品名称、价格和库存状态。"它回答：

```
The product is the Sony WH-1000XM5 headphones, which cost $348.00 and are currently in stock.
```

这是一个完全正确的答案。但对你的应用来说完全没用。你的库存系统需要 `{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true}`。你需要的是一个具有特定键、特定类型和特定值约束的 JSON 对象。你不需要一个句子。

 naive 的解决方案：在 prompt 中加上"用 JSON 响应"。这在 90% 的情况下有效。另外 10% 的情况是模型用 markdown 代码块包裹 JSON，或者加上"这是 JSON："这样的前言，或者产生语法无效的 JSON（因为提前关闭了括号）。你的 JSON 解析器崩溃了。你的管道坏了。你加了 try/except 和重试循环。重试有时会产生不同的数据。现在你在解析问题之上又有了 consistency 问题。

这不是 prompt 工程问题。这是解码问题。模型从左到右生成 token。在每个位置，它从 100K+ 选项的词表中挑选最可能的下一个 token。在任何给定位置，这些选项中的大多数都会产生无效的 JSON。如果模型刚刚输出了 `{"price":`，下一个 token 必须是数字、引号（字符串）、`null`、`true`、`false` 或负号。其他任何内容都会产生无效的 JSON。如果没有约束，模型可能会选择一个完全合理的英语单词，但在句法上是灾难性的错误。

## 概念

### 结构化输出谱系

结构化输出控制有四个层次，每个层次都比前一个更可靠。

```mermaid
graph LR
    subgraph Spectrum["结构化输出谱系"]
        direction LR
        A["Prompt-based\n'Return JSON'\n~90% valid"] --> B["JSON Mode\nGuaranteed valid JSON\nNo schema guarantee"]
        B --> C["Schema Mode\nJSON + matches schema\nGuaranteed compliance"]
        C --> D["Constrained Decoding\nToken-level enforcement\n100% compliance"]
    end

    style A fill:#1a1a2e,stroke:#ff6b6b,color:#fff
    style B fill:#1a1a2e,stroke:#ffa500,color:#fff
    style C fill:#1a1a2e,stroke:#51cf66,color:#fff
    style D fill:#1a1a2e,stroke:#0f3460,color:#fff
```

**Prompt-based**（"用有效 JSON 响应"）：无强制执行。模型通常遵守但有时不遵守。可靠性：约 90%。失败模式：markdown 代码块、前言文本、截断输出、错误结构。

**JSON 模式**：API 保证输出是有效的 JSON。OpenAI 的 `response_format: { type: "json_object" }` 启用此功能。输出将可以解析而不会出错。但它可能不符合你预期的 schema——额外的键、错误的类型、缺失的字段。

**Schema 模式**：API 接受 JSON Schema 并保证输出匹配它。2026 年每个主要提供商都原生支持：OpenAI 的 `response_format: { type: "json_schema", json_schema: {...} }`（也作为 `tool_choice="required"`）、Anthropic 带 `input_schema` 的工具使用，以及 Gemini 的 `response_schema` + `response_mime_type: "application/json"`。输出具有你指定的确切键、类型和约束。

**约束解码**：在生成过程中的每个 token 位置，解码器屏蔽所有会产生无效输出的 token。如果 schema 要求数字而模型即将输出字母，该 token 被设置为概率零。模型只能产生导致有效输出的 token。这就是 OpenAI 的结构化输出模式以及 Outlines 和 Guidance 等库在底层实现的方式。

### JSON Schema：契约语言

JSON Schema 是你告诉模型（或验证层）输出必须是什么形状的方式。每个主要的结构化输出系统都使用它。

```json
{
  "type": "object",
  "properties": {
    "product": { "type": "string" },
    "price": { "type": "number", "minimum": 0 },
    "in_stock": { "type": "boolean" },
    "categories": {
      "type": "array",
      "items": { "type": "string" }
    }
  },
  "required": ["product", "price", "in_stock"]
}
```

这个 schema 表示：输出必须是一个对象，包含字符串 `product`、非负数 `price`、布尔 `in_stock`，以及可选的字符串数组 `categories`。任何不匹配的输出都会被拒绝。

Schema 处理困难情况：嵌套对象、带类型项的数组、枚举（将字符串约束为特定值）、模式匹配（字符串上的正则表达式）和组合器（oneOf、anyOf、allOf 用于多态输出）。

### Pydantic 模式

在 Python 中，你不需要手写 JSON Schema。你定义一个 Pydantic 模型，它会为你生成 schema。

```python
from pydantic import BaseModel

class Product(BaseModel):
    product: str
    price: float
    in_stock: bool
    categories: list[str] = []
```

这会生成与上面相同的 JSON Schema。Instructor 库（和 OpenAI 的 SDK）直接接受 Pydantic 模型：传入模型类，返回验证后的实例。如果 LLM 输出不匹配，Instructor 会自动重试。

### 函数调用 / 工具使用

同一个问题的另一种接口。不是让模型直接生成 JSON，而是定义具有类型化参数的"工具"（函数）。模型输出一个带有结构化参数的函数调用。OpenAI 称之为"函数调用"。Anthropic 称之为"工具使用"。结果相同：结构化数据。

```mermaid
graph TD
    subgraph ToolUse["工具使用流程"]
        U["用户：从这段评论文本\n提取产品信息"] --> M["模型处理输入"]
        M --> TC["工具调用：\nextract_product(\n  product='Sony WH-1000XM5',\n  price=348.00,\n  in_stock=true\n)"]
        TC --> V["根据函数 schema 验证"]
        V --> R["结构化结果：\n{product, price, in_stock}"]
    end

    style U fill:#1a1a2e,stroke:#0f3460,color:#fff
    style TC fill:#1a1a2e,stroke:#e94560,color:#fff
    style V fill:#1a1a2e,stroke:#ffa500,color:#fff
    style R fill:#1a1a2e,stroke:#51cf66,color:#fff
```

当模型需要选择调用哪个函数而不仅仅是填充参数时，工具使用是首选。如果你有 10 个不同的提取 schema，模型必须根据输入选择正确的 schema，工具使用可以同时提供 schema 选择和结构化输出。

### 常见失败模式

即使有 schema 强制执行，结构化输出也可能以微妙的方式失败。

**幻觉值**：输出匹配 schema 但包含虚构数据。当文本说的是 $348 时，模型生成 `{"price": 299.99}`。Schema 验证无法捕捉这个——类型是正确的，但值是错误的。

**枚举混淆**：你将字段约束为 `["in_stock", "out_of_stock", "preorder"]`。模型输出 `"available"`——语义正确，但不在允许的集合中。良好的约束解码可以防止这种情况。基于 prompt 的方法不行。

**嵌套对象深度**：深度嵌套的 schema（4+ 层）会产生更多错误。每层嵌套都是模型可能丢失结构追踪的另一个地方。

**数组长度**：模型可能产生过多或过少的数组项。Schema 支持 `minItems` 和 `maxItems`，但并非所有提供商都在解码层面强制执行。

**可选字段省略**：模型省略了在技术上是可选的但对你的用例语义上重要的字段。在 schema 中将它们设置为必需，即使数据有时缺失——强制模型明确产生 `null`。

## 构建它

### 步骤 1：JSON Schema 验证器

从头开始构建一个验证器，检查 Python 对象是否匹配 JSON Schema。这是在输出端验证合规性的运行组件。

```python
import json

def validate_schema(data, schema):
    errors = []
    _validate(data, schema, "", errors)
    return errors

def _validate(data, schema, path, errors):
    schema_type = schema.get("type")

    if schema_type == "object":
        if not isinstance(data, dict):
            errors.append(f"{path}: expected object, got {type(data).__name__}")
            return
        for key in schema.get("required", []):
            if key not in data:
                errors.append(f"{path}.{key}: required field missing")
        properties = schema.get("properties", {})
        for key, value in data.items():
            if key in properties:
                _validate(value, properties[key], f"{path}.{key}", errors)

    elif schema_type == "array":
        if not isinstance(data, list):
            errors.append(f"{path}: expected array, got {type(data).__name__}")
            return
        min_items = schema.get("minItems", 0)
        max_items = schema.get("maxItems", float("inf"))
        if len(data) < min_items:
            errors.append(f"{path}: array has {len(data)} items, minimum is {min_items}")
        if len(data) > max_items:
            errors.append(f"{path}: array has {len(data)} items, maximum is {max_items}")
        items_schema = schema.get("items", {})
        for i, item in enumerate(data):
            _validate(item, items_schema, f"{path}[{i}]", errors)

    elif schema_type == "string":
        if not isinstance(data, str):
            errors.append(f"{path}: expected string, got {type(data).__name__}")
            return
        enum_values = schema.get("enum")
        if enum_values and data not in enum_values:
            errors.append(f"{path}: '{data}' not in allowed values {enum_values}")

    elif schema_type == "number":
        if not isinstance(data, (int, float)):
            errors.append(f"{path}: expected number, got {type(data).__name__}")
            return
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and data < minimum:
            errors.append(f"{path}: {data} is less than minimum {minimum}")
        if maximum is not None and data > maximum:
            errors.append(f"{path}: {data} is greater than maximum {maximum}")

    elif schema_type == "boolean":
        if not isinstance(data, bool):
            errors.append(f"{path}: expected boolean, got {type(data).__name__}")

    elif schema_type == "integer":
        if not isinstance(data, int) or isinstance(data, bool):
            errors.append(f"{path}: expected integer, got {type(data).__name__}")
```

### 步骤 2：Pydantic 风格模型到 Schema

构建一个最小的类到 schema 转换器。定义一个 Python 类并自动生成其 JSON Schema。

```python
class SchemaField:
    def __init__(self, field_type, required=True, default=None, enum=None, minimum=None, maximum=None):
        self.field_type = field_type
        self.required = required
        self.default = default
        self.enum = enum
        self.minimum = minimum
        self.maximum = maximum

def python_type_to_schema(field):
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }

    schema = {}

    if field.field_type in type_map:
        schema["type"] = type_map[field.field_type]
    elif field.field_type == list:
        schema["type"] = "array"
        schema["items"] = {"type": "string"}
    elif isinstance(field.field_type, dict):
        schema = field.field_type

    if field.enum:
        schema["enum"] = field.enum
    if field.minimum is not None:
        schema["minimum"] = field.minimum
    if field.maximum is not None:
        schema["maximum"] = field.maximum

    return schema

def model_to_schema(name, fields):
    properties = {}
    required = []

    for field_name, field in fields.items():
        properties[field_name] = python_type_to_schema(field)
        if field.required:
            required.append(field_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }
```

### 步骤 3：约束 Token 过滤器

模拟约束解码。给定一个部分 JSON 字符串和一个 schema，确定在当前位置哪些 token 类别是有效的。

```python
def next_valid_tokens(partial_json, schema):
    stripped = partial_json.strip()

    if not stripped:
        return ["{"]

    try:
        json.loads(stripped)
        return ["<EOS>"]
    except json.JSONDecodeError:
        pass

    last_char = stripped[-1] if stripped else ""

    if last_char == "{":
        return ['"', "}"]
    elif last_char == '"':
        if stripped.endswith('":'):
            return ['"', "0-9", "true", "false", "null", "[", "{"]
        return ["a-z", '"']
    elif last_char == ":":
        return [" ", '"', "0-9", "true", "false", "null", "[", "{"]
    elif last_char == ",":
        return [" ", '"', "{", "["]
    elif last_char in "0123456789":
        return ["0-9", ".", ",", "}", "]"]
    elif last_char == "}":
        return [",", "}", "]", "<EOS>"]
    elif last_char == "]":
        return [",", "}", "<EOS>"]
    elif last_char == "[":
        return ['"', "0-9", "true", "false", "null", "{", "[", "]"]
    else:
        return ["any"]

def demonstrate_constrained_decoding():
    partial_states = [
        '',
        '{',
        '{"product"',
        '{"product":',
        '{"product": "Sony"',
        '{"product": "Sony",',
        '{"product": "Sony", "price":',
        '{"product": "Sony", "price": 348',
        '{"product": "Sony", "price": 348}',
    ]

    print(f"{'Partial JSON':<45} {'Valid Next Tokens'}")
    print("-" * 80)
    for state in partial_states:
        valid = next_valid_tokens(state, {})
        display = state if state else "(empty)"
        print(f"{display:<45} {valid}")
```

### 步骤 4：提取管道

将所有内容组合成提取管道：定义 schema、模拟 LLM 生成结构化输出、验证输出并处理重试。

```python
def simulate_llm_extraction(text, schema, attempt=0):
    if "headphones" in text.lower() or "sony" in text.lower():
        if attempt == 0:
            return '{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true, "categories": ["audio", "headphones"]}'
        return '{"product": "Sony WH-1000XM5", "price": 348.00, "in_stock": true}'

    if "laptop" in text.lower():
        return '{"product": "MacBook Pro 16", "price": 2499.00, "in_stock": false, "categories": ["computers"]}'

    return '{"product": "Unknown", "price": 0, "in_stock": false}'

def extract_with_retry(text, schema, max_retries=3):
    for attempt in range(max_retries):
        raw = simulate_llm_extraction(text, schema, attempt)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  Attempt {attempt + 1}: JSON parse error -- {e}")
            continue

        errors = validate_schema(data, schema)
        if not errors:
            return data

        print(f"  Attempt {attempt + 1}: Schema validation errors -- {errors}")

    return None

product_schema = {
    "type": "object",
    "properties": {
        "product": {"type": "string"},
        "price": {"type": "number", "minimum": 0},
        "in_stock": {"type": "boolean"},
        "categories": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["product", "price", "in_stock"],
}
```

### 步骤 5：运行完整管道

```python
def run_demo():
    print("=" * 60)
    print("  Structured Output Pipeline Demo")
    print("=" * 60)

    print("\n--- Schema Definition ---")
    product_fields = {
        "product": SchemaField(str),
        "price": SchemaField(float, minimum=0),
        "in_stock": SchemaField(bool),
        "categories": SchemaField(list, required=False),
    }
    generated_schema = model_to_schema("Product", product_fields)
    print(json.dumps(generated_schema, indent=2))

    print("\n--- Schema Validation ---")
    test_cases = [
        ({"product": "Test", "price": 10.0, "in_stock": True}, "Valid object"),
        ({"product": "Test", "price": -5.0, "in_stock": True}, "Negative price"),
        ({"product": "Test", "in_stock": True}, "Missing price"),
        ({"product": "Test", "price": "ten", "in_stock": True}, "String as price"),
        ("not an object", "String instead of object"),
    ]

    for data, label in test_cases:
        errors = validate_schema(data, product_schema)
        status = "PASS" if not errors else f"FAIL: {errors}"
        print(f"  {label}: {status}")

    print("\n--- Constrained Decoding Simulation ---")
    demonstrate_constrained_decoding()

    print("\n--- Extraction Pipeline ---")
    texts = [
        "The Sony WH-1000XM5 headphones are priced at $348 and currently available.",
        "The new MacBook Pro 16-inch laptop costs $2499 but is sold out.",
        "This is a random sentence with no product info.",
    ]

    for text in texts:
        print(f"\n  Input: {text[:60]}...")
        result = extract_with_retry(text, product_schema)
        if result:
            print(f"  Output: {json.dumps(result)}")
        else:
            print(f"  Output: FAILED after retries")
```

## 使用它

### OpenAI 结构化输出

```python
# from openai import OpenAI
# from pydantic import BaseModel
#
# client = OpenAI()
#
# class Product(BaseModel):
#     product: str
#     price: float
#     in_stock: bool
#
# response = client.beta.chat.completions.parse(
#     model="gpt-5-mini",
#     messages=[
#         {"role": "system", "content": "Extract product information."},
#         {"role": "user", "content": "Sony WH-1000XM5, $348, in stock"},
#     ],
#     response_format=Product,
# )
#
# product = response.choices[0].message.parsed
# print(product.product, product.price, product.in_stock)
```

OpenAI 的结构化输出模式在内部使用约束解码。模型生成的每个 token 都保证产生匹配 Pydantic schema 的输出。无需重试。无需验证。约束被 baked 到解码过程中。

### Anthropic 工具使用

```python
# import anthropic
#
# client = anthropic.Anthropic()
#
# response = client.messages.create(
#     model="claude-opus-4-7",
#     max_tokens=1024,
#     tools=[{
#         "name": "extract_product",
#         "description": "Extract product information from text",
#         "input_schema": {
#             "type": "object",
#             "properties": {
#                 "product": {"type": "string"},
#                 "price": {"type": "number"},
#                 "in_stock": {"type": "boolean"},
#             },
#             "required": ["product", "price", "in_stock"],
#         },
#     }],
#     messages=[{"role": "user", "content": "Extract: Sony WH-1000XM5, $348, in stock"}],
# )
```

Anthropic 通过工具使用实现结构化输出。模型发出带有匹配 input_schema 的结构化参数的工具调用。相同的结果，不同的 API 表面。

### Instructor 库

```python
# pip install instructor
# import instructor
# from openai import OpenAI
# from pydantic import BaseModel
#
# client = instructor.from_openai(OpenAI())
#
# class Product(BaseModel):
#     product: str
#     price: float
#     in_stock: bool
#
# product = client.chat.completions.create(
#     model="gpt-5-mini",
#     response_model=Product,
#     messages=[{"role": "user", "content": "Sony WH-1000XM5, $348, in stock"}],
# )
```

Instructor 包装任何 LLM 客户端并添加自动重试和验证。如果第一次尝试验证失败，它会将错误作为上下文发送回模型并要求它修复输出。这适用于任何提供商，而不仅仅是 OpenAI。

## 发货

本课程生成 `outputs/prompt-structured-extractor.md`——一个可重用的 prompt 模板，根据 schema 定义从任何文本中提取结构化数据。给它一个 JSON Schema 和非结构化文本，它返回验证过的 JSON。

它还生成 `outputs/skill-structured-outputs.md`——一个决策框架，用于根据你的提供商、可靠性要求和 schema 复杂性选择正确的结构化输出策略。

## 练习

1. 扩展 schema 验证器以支持 `oneOf`（数据必须恰好匹配多个 schema 之一）。这处理多态输出——例如，一个字段可以是具有不同形状的 `Product` 或 `Service` 对象。

2. 构建一个"schema diff"工具，比较两个 schema 并识别破坏性更改（删除必需字段、更改类型）与非破坏性更改（添加可选字段、放松约束）。这对于在生产环境中对你的提取 schema 进行版本控制至关重要。

3. 实现一个更现实的约束解码模拟器。给定一个 JSON Schema 和 100 个 token 的词表（字母、数字、标点符号、关键字），逐步走完生成过程，在每个位置屏蔽无效 token。测量每个步骤中有多少百分比的词表是有效的。

4. 构建一个提取评估套件。创建 50 个带有人工标记 JSON 输出的产品描述。在所有 50 个上运行你的提取管道并测量精确匹配、字段级准确性和类型合规性。识别哪些字段最难正确提取。

5. 为你的提取管道添加"置信度分数"。对于每个提取的字段，估计模型的置信度（基于 token 概率，或通过运行 3 次提取并测量一致性）。将低置信度字段标记为需要人工审核。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| JSON mode | "Returns JSON" | API 标志，保证语法上有效的 JSON 输出，但不强制执行任何特定 schema |
| Structured output | "Typed JSON" | 匹配特定 JSON Schema 的输出，具有正确的键、类型和约束 |
| Constrained decoding | "Guided generation" | 在每个 token 位置，屏蔽会产生无效输出的 token——保证 100% schema 合规 |
| JSON Schema | "A JSON template" | 用于描述 JSON 数据结构、类型和约束的声明式语言（用于 OpenAPI、JSON Forms 等） |
| Pydantic | "Python dataclasses+" | 定义带类型验证的数据模型的 Python 库，用于 FastAPI 和 Instructor 生成 JSON Schema |
| Function calling | "Tool use" | LLM 输出结构化函数调用（名称 + 类型化参数）而不是自由文本——OpenAI 和 Anthropic 都支持 |
| Instructor | "Pydantic for LLMs" | 包装 LLM 客户端以返回验证的 Pydantic 实例的 Python 库，验证失败时自动重试 |
| Token masking | "Filtering the vocabulary" | 在生成过程中将特定 token 概率设置为零，使模型无法产生它们 |
| Schema compliance | "Matches the shape" | 输出具有每个必需字段、正确的类型、值在约束内，且没有额外的禁止字段 |
| Retry loop | "Try again until it works" | 将验证错误发送回模型并要求它修复输出——Instructor 自动执行此操作，最高可达可配置的最大次数 |

## 进一步阅读

- [OpenAI Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs) -- OpenAI API 中基于 JSON Schema 的约束解码官方文档
- [Willard & Louf, 2023 -- "Efficient Guided Generation for Large Language Models"](https://arxiv.org/abs/2307.09702) -- Outlines 论文，描述如何将 JSON Schema 编译为有限状态机以进行 token 级约束
- [Instructor documentation](https://python.useinstructor.com/) -- 从任何 LLM 获取结构化输出的标准库，带有 Pydantic 验证和重试
- [Anthropic Tool Use Guide](https://docs.anthropic.com/en/docs/tool-use) -- Claude 如何通过带 JSON Schema input_schema 的工具使用实现结构化输出
- [JSON Schema specification](https://json-schema.org/) -- 每个主要结构化输出系统使用的 schema 语言的完整规范
- [Outlines library](https://github.com/outlines-dev/outlines) -- 使用正则表达式和 JSON Schema 编译为有限状态机的开源约束生成
- [Dong et al., "XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models" (MLSys 2025)](https://arxiv.org/abs/2411.15100) -- 当前最先进的语法引擎；下推自动机编译，以约 100 ns/token 屏蔽 token
- [Beurer-Kellner et al., "Prompting Is Programming: A Query Language for Large Language Models" (LMQL)](https://arxiv.org/abs/2212.06094) -- LMQL 论文，将约束解码框架化为带有类型和值约束的查询语言
- [Microsoft Guidance (framework docs)](https://github.com/guidance-ai/guidance) -- 模板驱动的约束生成；vendor-agnostic 的 Outlines 和 XGrammar 补充