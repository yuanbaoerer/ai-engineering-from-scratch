"""
Function Calling & Tool Use — 从零构建的 LLM 工具调用演示。

展示 OpenAI / Anthropic / Google 通用的四步模式：
  1. Define  — 用 JSON Schema 描述工具签名
  2. Detect  — 模型根据用户意图决定调用哪个工具
  3. Execute — 在沙箱中安全执行，返回结果
  4. Return  — 把结果喂回模型，决定是否继续调用

本文件不依赖任何 LLM API，用关键词匹配模拟模型决策，
重点演示工具注册、参数校验、沙箱隔离和多轮循环。
"""

import json
import math
import re
import time


# ── 工具注册表 ──────────────────────────────────────────────
# 全局字典，key = 工具名，value = { definition, function }
# definition 遵循 OpenAI function calling 的 JSON Schema 格式
TOOL_REGISTRY = {}


def register_tool(name, description, parameters, function):
    """注册一个工具到全局注册表。

    Args:
        name: 工具名，如 "calculator"
        description: 工具用途描述，模型靠这段文字决定何时调用
        parameters: JSON Schema 格式的参数定义
        function: 实际执行的 Python 函数
    """
    TOOL_REGISTRY[name] = {
        "definition": {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        },
        "function": function,
    }


def calculator(expression, precision=2):
    """安全的数学表达式计算器。

    白名单只允许数字和 + - * / . ( ) 空格，防止注入。
    eval() 传入空 __builtins__，仅暴露 math 模块。

    Args:
        expression: 数学表达式，如 "(10 + 5) * 3"
        precision: 结果保留的小数位数，默认 2
    Returns:
        {"result": float, "expression": str} 或 {"error": True, "message": str}
    """
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return {"error": True, "message": f"Invalid characters in expression: {expression}"}
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return {"result": round(float(result), precision), "expression": expression}
    except Exception as e:
        return {"error": True, "message": str(e)}


# ── 模拟数据源 ──────────────────────────────────────────────
# 真实场景中这些数据来自外部 API，这里用字典模拟。
WEATHER_DB = {
    "tokyo": {"temp_c": 18, "condition": "cloudy", "humidity": 72, "wind_kph": 14},
    "new york": {"temp_c": 22, "condition": "sunny", "humidity": 45, "wind_kph": 8},
    "london": {"temp_c": 12, "condition": "rainy", "humidity": 88, "wind_kph": 22},
    "san francisco": {"temp_c": 16, "condition": "foggy", "humidity": 80, "wind_kph": 18},
    "sydney": {"temp_c": 25, "condition": "sunny", "humidity": 55, "wind_kph": 10},
}


def get_weather(city, units="celsius"):
    """查询指定城市的当前天气。

    Args:
        city: 城市名，不区分大小写
        units: "celsius" 或 "fahrenheit"
    Returns:
        天气数据字典，或错误信息 + 城市建议列表
    """
    key = city.lower().strip()
    if key not in WEATHER_DB:
        # 城市不存在时，按前缀模糊匹配提供建议
        suggestions = [c for c in WEATHER_DB if c.startswith(key[:3])]
        return {
            "error": True,
            "message": f"City '{city}' not found.",
            "suggestions": suggestions,
            "code": "CITY_NOT_FOUND",
        }
    data = WEATHER_DB[key].copy()
    if units == "fahrenheit":
        data["temp_f"] = round(data["temp_c"] * 9 / 5 + 32, 1)
        del data["temp_c"]
    data["city"] = city
    return data


# 模拟搜索引擎数据库
SEARCH_DB = {
    "python function calling": [
        {"title": "OpenAI Function Calling Guide", "url": "https://platform.openai.com/docs/guides/function-calling", "snippet": "Learn how to connect LLMs to external tools."},
        {"title": "Anthropic Tool Use", "url": "https://docs.anthropic.com/en/docs/tool-use", "snippet": "Claude can interact with external tools and APIs."},
    ],
    "MCP protocol": [
        {"title": "Model Context Protocol", "url": "https://modelcontextprotocol.io", "snippet": "An open standard for connecting AI models to data sources."},
    ],
    "weather API": [
        {"title": "OpenWeatherMap API", "url": "https://openweathermap.org/api", "snippet": "Free weather API with current, forecast, and historical data."},
    ],
}


def web_search(query, max_results=3):
    """模拟网络搜索，按关键词在本地数据库中匹配。

    Args:
        query: 搜索关键词
        max_results: 最多返回结果数
    Returns:
        {"query": str, "results": list, "total": int}
    """
    key = query.lower().strip()
    for db_key, results in SEARCH_DB.items():
        if db_key in key or key in db_key:
            return {"query": query, "results": results[:max_results], "total": len(results)}
    return {"query": query, "results": [], "total": 0}


# 模拟文件系统
FILE_SYSTEM = {
    "data/config.json": '{"model": "gpt-4o", "temperature": 0.7, "max_tokens": 4096}',
    "data/users.csv": "name,email,role\nAlice,alice@example.com,admin\nBob,bob@example.com,user",
    "README.md": "# My Project\nA tool-use agent built from scratch.",
}


def read_file(path):
    """读取模拟文件系统中的文件。

    安全检查：拦截路径遍历（.. 和绝对路径），防止读取敏感文件。

    Args:
        path: 相对路径，如 "data/config.json"
    Returns:
        {"path": str, "content": str, "size_bytes": int, "lines": int} 或错误信息
    """
    if ".." in path or path.startswith("/"):
        return {"error": True, "message": "Path traversal not allowed.", "code": "FORBIDDEN"}
    if path not in FILE_SYSTEM:
        available = list(FILE_SYSTEM.keys())
        return {"error": True, "message": f"File '{path}' not found.", "available_files": available, "code": "NOT_FOUND"}
    content = FILE_SYSTEM[path]
    return {"path": path, "content": content, "size_bytes": len(content), "lines": content.count("\n") + 1}


def run_code(code, language="python"):
    """在受限沙箱中执行 Python 代码。

    安全机制：
    - 黑名单拦截危险操作（import os/subprocess、exec、eval、open 等）
    - __builtins__ 只暴露安全的内建函数，不暴露 __import__ / eval / exec
    - 用户代码中需要把结果赋给 result 变量，函数会将其提取出来

    Args:
        code: 要执行的 Python 代码
        language: 目前只支持 "python"
    Returns:
        {"success": True, "result": Any, "variables": dict} 或错误信息
    """
    if language != "python":
        return {"error": True, "message": f"Language '{language}' not supported. Only 'python' is available."}
    forbidden = ["import os", "import sys", "import subprocess", "exec(", "eval(", "__import__", "open("]
    for pattern in forbidden:
        if pattern in code:
            return {"error": True, "message": f"Forbidden operation: {pattern}", "code": "SECURITY_VIOLATION"}
    try:
        local_vars = {}
        # 只暴露安全的内建函数，防止逃逸沙箱
        exec(
            code,
            {
                "__builtins__": {
                    "print": print, "range": range, "len": len, "str": str,
                    "int": int, "float": float, "list": list, "dict": dict,
                    "sum": sum, "min": min, "max": max, "abs": abs, "round": round,
                    "sorted": sorted, "enumerate": enumerate, "zip": zip,
                    "map": map, "filter": filter, "math": math,
                }
            },
            local_vars,
        )
        # 提取用户设置的 result 变量作为返回值
        result = local_vars.get("result", None)
        return {
            "success": True,
            "result": result,
            "variables": {k: str(v) for k, v in local_vars.items() if not k.startswith("_")},
        }
    except Exception as e:
        return {"error": True, "message": f"{type(e).__name__}: {e}"}


def register_all_tools():
    """批量注册所有工具到全局注册表。

    每个工具包含：
    - name: 工具名（模型调用时用这个名字）
    - description: 工具用途（模型靠这段文字决定何时调用）
    - parameters: JSON Schema 格式的参数定义（类型、描述、必填项、枚举值）
    - function: 实际执行的 Python 函数
    """
    register_tool(
        "calculator",
        "Evaluate a mathematical expression. Supports +, -, *, /, parentheses, and decimals. Returns the numeric result.",
        {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression, e.g. '(10 + 5) * 3'"},
                "precision": {"type": "integer", "description": "Decimal places in result", "default": 2},
            },
            "required": ["expression"],
        },
        calculator,
    )
    register_tool(
        "get_weather",
        "Get current weather for a city. Returns temperature, condition, humidity, and wind speed.",
        {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'Tokyo' or 'San Francisco'"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature units, defaults to celsius"},
            },
            "required": ["city"],
        },
        get_weather,
    )
    register_tool(
        "web_search",
        "Search the web for information. Returns a list of results with title, URL, and snippet.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Maximum results to return", "default": 3},
            },
            "required": ["query"],
        },
        web_search,
    )
    register_tool(
        "read_file",
        "Read the contents of a file. Returns the file content, size, and line count.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path, e.g. 'data/config.json'"},
            },
            "required": ["path"],
        },
        read_file,
    )
    register_tool(
        "run_code",
        "Execute Python code in a sandboxed environment. Set a 'result' variable to return output.",
        {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
                "language": {"type": "string", "enum": ["python"], "description": "Programming language"},
            },
            "required": ["code"],
        },
        run_code,
    )


def simulate_model_decision(user_message, tools, conversation_history):
    """模拟 LLM 的工具调用决策。

    真实场景中，模型会分析用户消息 + 工具定义，输出 tool_calls。
    这里用关键词匹配来模拟，重点演示后续的执行流程。

    决策逻辑：
    - 含 "weather/temperature/forecast" → get_weather（支持并行多个城市）
    - 含 "calculate/compute/math" → calculator
    - 含 "search/find/look up" → web_search
    - 含 "read/file/open/show" → read_file
    - 含 "run/execute/code" → run_code
    - 都不匹配 → 返回空列表（模型直接回答，不调用工具）

    Args:
        user_message: 用户输入的文本
        tools: 工具定义列表（模型靠这个知道有哪些工具可用）
        conversation_history: 对话历史（本 demo 未使用，真实场景中模型会参考）
    Returns:
        tool_calls 列表，每项 {"name": str, "arguments": dict}
    """
    msg = user_message.lower()

    # 匹配天气查询 — 支持同时查询多个城市（并行调用）
    if any(word in msg for word in ["weather", "temperature", "forecast"]):
        cities = []
        for city in WEATHER_DB:
            if city in msg:
                cities.append(city)
        # 尝试匹配首字母大写的城市名（如 "Tokyo"）
        if not cities:
            for word in msg.split():
                if word.capitalize() in [c.title() for c in WEATHER_DB]:
                    cities.append(word)
        # 默认查询 Tokyo
        if not cities:
            cities = ["tokyo"]
        calls = []
        for city in cities:
            calls.append({"name": "get_weather", "arguments": {"city": city.title()}})
        return calls

    # 匹配数学计算 — 提取表达式中的运算符部分
    if any(word in msg for word in ["calculate", "compute", "math", "what is", "how much"]):
        for run in re.findall(r"[0-9.+\-*/()\s]{3,}", msg):
            expr = run.strip()
            if not any(c.isdigit() for c in expr):
                continue
            if not any(c in expr for c in "+-*/"):
                continue
            if "error" not in calculator(expr):
                return [{"name": "calculator", "arguments": {"expression": expr}}]
        for token in msg.split():
            if any(c in token for c in "+-*/"):
                return [{"name": "calculator", "arguments": {"expression": token}}]
        # 尝试从整条消息中提取数学字符
        if "+" in msg or "-" in msg or "*" in msg or "/" in msg:
            expr = "".join(c for c in msg if c in "0123456789+-*/.() ")
            if expr.strip():
                return [{"name": "calculator", "arguments": {"expression": expr.strip()}}]
        return [{"name": "calculator", "arguments": {"expression": "0"}}]

    # 匹配搜索 — 去掉常见前缀词
    if any(word in msg for word in ["search", "find", "look up", "google"]):
        query = msg.replace("search for", "").replace("look up", "").replace("find", "").strip()
        return [{"name": "web_search", "arguments": {"query": query}}]

    # 匹配文件读取 — 按文件名匹配，找不到就默认读 README.md
    if any(word in msg for word in ["read", "file", "open", "cat", "show"]):
        for path in FILE_SYSTEM:
            if path.split("/")[-1].split(".")[0] in msg:
                return [{"name": "read_file", "arguments": {"path": path}}]
        return [{"name": "read_file", "arguments": {"path": "README.md"}}]

    # 匹配代码执行
    if any(word in msg for word in ["run", "execute", "code", "python"]):
        return [{"name": "run_code", "arguments": {"code": "result = 'Hello from the sandbox!'", "language": "python"}}]

    # 没有匹配到任何工具，模型应该直接回答
    return []


def execute_tool_call(tool_call):
    """执行单个工具调用，返回结果和耗时。

    对应四步模式中的 "Execute" 环节。
    从全局注册表中查找工具函数，用 **kwargs 解包参数调用。

    Args:
        tool_call: {"name": str, "arguments": dict}
    Returns:
        {"tool": str, "result": Any, "execution_time_ms": float}
    """
    name = tool_call["name"]
    args = tool_call["arguments"]

    if name not in TOOL_REGISTRY:
        return {"tool": name, "result": {"error": True, "message": f"Unknown tool: {name}", "code": "UNKNOWN_TOOL"}, "execution_time_ms": 0}

    tool = TOOL_REGISTRY[name]
    func = tool["function"]
    start = time.time()

    try:
        result = func(**args)
    except TypeError as e:
        # 参数不匹配（如传了多余的参数）会触发 TypeError
        result = {"error": True, "message": f"Invalid arguments: {e}"}

    elapsed_ms = round((time.time() - start) * 1000, 2)
    return {"tool": name, "result": result, "execution_time_ms": elapsed_ms}


def validate_tool_arguments(tool_name, arguments):
    """根据 JSON Schema 校验工具参数是否合法。

    校验内容：
    1. 工具是否存在
    2. 参数是否为 dict
    3. 必填字段是否缺失
    4. 参数类型是否匹配（string/integer/number/boolean/array/object）
    5. enum 值是否在允许范围内

    Args:
        tool_name: 工具名
        arguments: 参数字典
    Returns:
        错误信息列表，空列表表示校验通过
    """
    if tool_name not in TOOL_REGISTRY:
        return [f"Unknown tool: {tool_name}"]

    schema = TOOL_REGISTRY[tool_name]["definition"]["function"]["parameters"]
    errors = []

    if not isinstance(arguments, dict):
        return [f"Arguments must be an object, got {type(arguments).__name__}"]

    # 检查必填字段
    for required_field in schema.get("required", []):
        if required_field not in arguments:
            errors.append(f"Missing required argument: {required_field}")

    properties = schema.get("properties", {})
    for arg_name, arg_value in arguments.items():
        if arg_name not in properties:
            errors.append(f"Unknown argument: {arg_name}")
            continue

        prop_schema = properties[arg_name]
        expected_type = prop_schema.get("type")

        # JSON Schema 类型 → Python 类型映射
        type_checks = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        if expected_type in type_checks:
            if not isinstance(arg_value, type_checks[expected_type]):
                errors.append(f"Argument '{arg_name}': expected {expected_type}, got {type(arg_value).__name__}")

        # 枚举值校验
        if "enum" in prop_schema and arg_value not in prop_schema["enum"]:
            errors.append(f"Argument '{arg_name}': '{arg_value}' not in {prop_schema['enum']}")

    return errors


def run_function_calling_loop(user_message, max_iterations=5):
    """完整的 function calling 循环。

    这是本 demo 的核心，模拟真实 agent 的执行流程：

    1. 把用户消息加入对话历史
    2. 调用 simulate_model_decision 模拟模型决策
    3. 如果模型决定调用工具 → 执行所有 tool calls
    4. 把执行结果以 "tool" 角色加入对话历史
    5. 循环继续（最多 max_iterations 轮），直到模型不再调用工具

    在真实场景中，步骤 2 是调用 LLM API，步骤 4 的结果会
    作为上下文送回给模型，让模型决定下一步做什么。

    Args:
        user_message: 用户输入
        max_iterations: 最大循环次数，防止无限循环
    Returns:
        {"conversation": list, "tool_results": list, "iterations": int}
    """
    conversation = [{"role": "user", "content": user_message}]
    tool_definitions = [t["definition"] for t in TOOL_REGISTRY.values()]
    all_tool_results = []

    for iteration in range(max_iterations):
        # 模拟模型分析对话历史 + 工具定义，决定是否调用工具
        tool_calls = simulate_model_decision(user_message, tool_definitions, conversation)

        if not tool_calls:
            # 模型没有选择调用工具，循环结束
            break

        # 执行所有 tool calls（支持并行）
        results = []
        for call in tool_calls:
            result = execute_tool_call(call)
            results.append(result)

        # 把 assistant 的 tool_calls 加入对话历史
        conversation.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

        # 把每个工具的执行结果以 "tool" 角色加入对话历史
        # 真实场景中，这些结果会作为上下文送回给模型
        for result in results:
            conversation.append({
                "role": "tool",
                "content": json.dumps(result["result"]),
                "tool_name": result["tool"],
            })

        all_tool_results.extend(results)
        break  # 本 demo 单轮即止，真实场景会继续循环

    return {
        "conversation": conversation,
        "tool_results": all_tool_results,
        "iterations": iteration + 1 if tool_calls else 0,
    }


def run_demo():
    """运行完整的 function calling 演示。

    包含以下演示环节：
    1. 工具注册 — 展示所有已注册工具的名称和参数
    2. 参数校验 — 演示合法/非法参数的校验结果
    3. 直接执行 — 绕过模型决策，直接调用工具
    4. 完整循环 — 模拟完整的 "用户提问 → 模型决策 → 工具执行 → 返回结果" 流程
    5. 并行调用 — 一次请求同时查询多个城市的天气
    6. 安全检查 — 演示路径遍历、危险代码等被拦截
    """
    register_all_tools()

    print("=" * 60)
    print("  Function Calling & Tool Use Demo")
    print("=" * 60)

    # ── 环节 1：展示已注册的工具 ──
    print("\n--- Registered Tools ---")
    for name, tool in TOOL_REGISTRY.items():
        desc = tool["definition"]["function"]["description"][:60]
        params = list(tool["definition"]["function"]["parameters"].get("properties", {}).keys())
        print(f"  {name}: {desc}...")
        print(f"    params: {params}")

    # ── 环节 2：参数校验演示 ──
    print(f"\n--- Argument Validation ---")
    validation_tests = [
        ("get_weather", {"city": "Tokyo"}, "Valid call"),
        ("get_weather", {}, "Missing required arg"),
        ("get_weather", {"city": "Tokyo", "units": "kelvin"}, "Invalid enum value"),
        ("calculator", {"expression": 123}, "Wrong type (int for string)"),
        ("unknown_tool", {"x": 1}, "Unknown tool"),
    ]
    for tool_name, args, label in validation_tests:
        errors = validate_tool_arguments(tool_name, args)
        status = "VALID" if not errors else f"ERRORS: {errors}"
        print(f"  {label}: {status}")

    # ── 环节 3：直接执行工具 ──
    print(f"\n--- Tool Execution ---")
    direct_tests = [
        {"name": "calculator", "arguments": {"expression": "(10 + 5) * 3 / 2"}},
        {"name": "get_weather", "arguments": {"city": "Tokyo"}},
        {"name": "get_weather", "arguments": {"city": "Mars"}},
        {"name": "web_search", "arguments": {"query": "python function calling"}},
        {"name": "read_file", "arguments": {"path": "data/config.json"}},
        {"name": "read_file", "arguments": {"path": "../etc/passwd"}},
        {"name": "run_code", "arguments": {"code": "result = sum(range(1, 101))"}},
        {"name": "run_code", "arguments": {"code": "import os; os.system('rm -rf /')"}},
    ]
    for call in direct_tests:
        result = execute_tool_call(call)
        print(f"\n  {call['name']}({json.dumps(call['arguments'])})")
        print(f"    -> {json.dumps(result['result'], indent=None)[:100]}")
        print(f"    time: {result['execution_time_ms']}ms")

    # ── 环节 4：完整循环演示 ──
    print(f"\n--- Full Function Calling Loop ---")
    test_queries = [
        "What's the weather in Tokyo?",
        "Calculate (100 + 250) * 0.15",
        "Search for MCP protocol",
        "Read the config file",
        "Run some Python code",
        "Tell me a joke",  # 不匹配任何工具，模型直接回答
    ]
    for query in test_queries:
        print(f"\n  User: {query}")
        result = run_function_calling_loop(query)
        if result["tool_results"]:
            for tr in result["tool_results"]:
                print(f"    Tool: {tr['tool']} ({tr['execution_time_ms']}ms)")
                print(f"    Result: {json.dumps(tr['result'], indent=None)[:90]}")
        else:
            print(f"    [No tool called -- direct response]")
        print(f"    Iterations: {result['iterations']}")

    # ── 环节 5：并行调用演示 ──
    print(f"\n--- Parallel Tool Calls ---")
    multi_city_query = "What's the weather in tokyo and london?"
    print(f"  User: {multi_city_query}")
    result = run_function_calling_loop(multi_city_query)
    print(f"  Tool calls made: {len(result['tool_results'])}")
    for tr in result["tool_results"]:
        city = tr["result"].get("city", "unknown")
        temp = tr["result"].get("temp_c", "N/A")
        print(f"    {city}: {temp}C, {tr['result'].get('condition', 'N/A')}")

    # ── 环节 6：安全检查演示 ──
    print(f"\n--- Security Checks ---")
    security_tests = [
        ("read_file", {"path": "../../etc/passwd"}),
        ("run_code", {"code": "import subprocess; subprocess.run(['ls'])"}),
        ("calculator", {"expression": "__import__('os').system('ls')"}),
    ]
    for tool_name, args in security_tests:
        result = execute_tool_call({"name": tool_name, "arguments": args})
        blocked = result["result"].get("error", False)
        print(f"  {tool_name}({list(args.values())[0][:40]}): {'BLOCKED' if blocked else 'ALLOWED'}")


if __name__ == "__main__":
    run_demo()
