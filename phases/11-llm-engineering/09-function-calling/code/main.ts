// ──────────────────────────────────────────────────────────────
// Function Calling & Tool Use — TypeScript 版本
// ──────────────────────────────────────────────────────────────
// 与 function_calling.py 功能对等，展示 OpenAI / Anthropic / Google
// 通用的四步模式：
//   1. Define  — 用 JSON Schema 描述工具签名
//   2. Detect  — 模型根据用户意图决定调用哪个工具
//   3. Execute — 在沙箱中安全执行，返回结果
//   4. Return  — 把结果喂回模型，决定是否继续调用
//
// 本文件不依赖任何 LLM API，用关键词匹配模拟模型决策。
// Sources:
//   https://platform.openai.com/docs/guides/function-calling
//   https://docs.anthropic.com/en/docs/build-with-claude/tool-use
//   https://ai.google.dev/gemini-api/docs/function-calling

// ── 类型定义 ────────────────────────────────────────────────
// JSON 值的联合类型，对应 JSON Schema 中的所有合法值
type JsonValue = string | number | boolean | null | JsonValue[] | { [k: string]: JsonValue };

// 参数类型枚举，对应 JSON Schema 的 type 字段
type ParamType = "string" | "integer" | "number" | "boolean" | "array" | "object";

// 单个参数的 Schema 定义
type ParamSchema = {
  type: ParamType;          // 参数类型
  description?: string;     // 参数描述（模型靠这个理解参数含义）
  enum?: readonly JsonValue[]; // 可选值列表（如 "celsius" | "fahrenheit"）
  default?: JsonValue;      // 默认值
};

// 工具参数的整体 Schema（JSON Schema 的 object 类型）
type ToolParameters = {
  type: "object";
  properties: Readonly<Record<string, ParamSchema>>; // 参数名 → 参数定义
  required?: readonly string[];                       // 必填参数列表
};

// 工具定义，遵循 OpenAI function calling 的格式
type ToolDefinition = {
  type: "function";
  function: {
    name: string;          // 工具名（模型调用时用这个名字）
    description: string;   // 工具用途（模型靠这段文字决定何时调用）
    parameters: ToolParameters; // 参数 Schema
  };
};

// 工具函数签名：接收参数字典，返回 JSON 结果
type ToolFunction = (args: Readonly<Record<string, JsonValue>>) => JsonValue;

// 注册表中的条目：定义 + 实现函数
type RegisteredTool = {
  definition: ToolDefinition;
  fn: ToolFunction;
};

// ── 全局工具注册表 ──────────────────────────────────────────
// 用 Map 存储，key = 工具名，value = { definition, fn }
const TOOL_REGISTRY: Map<string, RegisteredTool> = new Map();

/**
 * 注册一个工具到全局注册表。
 * @param name - 工具名，如 "calculator"
 * @param description - 工具用途描述，模型靠这段文字决定何时调用
 * @param parameters - JSON Schema 格式的参数定义
 * @param fn - 实际执行的 TypeScript 函数
 */
function registerTool(name: string, description: string, parameters: ToolParameters, fn: ToolFunction): void {
  TOOL_REGISTRY.set(name, {
    definition: { type: "function", function: { name, description, parameters } },
    fn,
  });
}

// ── 工具实现 ────────────────────────────────────────────────

// 计算器：正则白名单，只允许数字和运算符
const ARITH_RE = /^[\d+\-*/().\s]+$/;

/**
 * 安全的数学表达式计算器。
 * 用正则白名单过滤非法字符，然后用 new Function() 求值。
 * @param args.expression - 数学表达式，如 "(10 + 5) * 3"
 * @param args.precision - 结果保留的小数位数，默认 2
 */
function calculator(args: Readonly<Record<string, JsonValue>>): JsonValue {
  const expression = String(args.expression ?? "");
  const precision = typeof args.precision === "number" ? args.precision : 2;
  if (!ARITH_RE.test(expression)) {
    return { error: true, message: "Invalid characters in expression: " + expression };
  }
  try {
    // new Function() 相对安全，因为外层无法访问局部变量
    // eslint-disable-next-line no-new-func
    const value = new Function("return (" + expression + ")")() as unknown;
    const num = Number(value);
    if (!Number.isFinite(num)) return { error: true, message: "non-finite result" };
    return { result: Number(num.toFixed(precision)), expression };
  } catch (err) {
    return { error: true, message: String(err) };
  }
}

// 模拟天气数据库（真实场景中来自外部 API）
const WEATHER_DB: Readonly<Record<string, { temp_c: number; condition: string; humidity: number; wind_kph: number }>> = {
  tokyo: { temp_c: 18, condition: "cloudy", humidity: 72, wind_kph: 14 },
  "new york": { temp_c: 22, condition: "sunny", humidity: 45, wind_kph: 8 },
  london: { temp_c: 12, condition: "rainy", humidity: 88, wind_kph: 22 },
  "san francisco": { temp_c: 16, condition: "foggy", humidity: 80, wind_kph: 18 },
  sydney: { temp_c: 25, condition: "sunny", humidity: 55, wind_kph: 10 },
};

/**
 * 查询指定城市的当前天气。
 * @param args.city - 城市名，不区分大小写
 * @param args.units - "celsius" 或 "fahrenheit"，默认摄氏度
 */
function getWeather(args: Readonly<Record<string, JsonValue>>): JsonValue {
  const city = String(args.city ?? "");
  const units = String(args.units ?? "celsius");
  const key = city.toLowerCase().trim();
  const row = WEATHER_DB[key];
  if (!row) {
    // 城市不存在时，按前缀模糊匹配提供建议
    const suggestions = Object.keys(WEATHER_DB).filter((c) => c.startsWith(key.slice(0, 3)));
    return { error: true, message: "City '" + city + "' not found.", suggestions, code: "CITY_NOT_FOUND" };
  }
  if (units === "fahrenheit") {
    return { city, condition: row.condition, humidity: row.humidity, wind_kph: row.wind_kph, temp_f: Number((row.temp_c * 9 / 5 + 32).toFixed(1)) };
  }
  return { city, ...row };
}

// 模拟搜索引擎数据库
const SEARCH_DB: Readonly<Record<string, ReadonlyArray<{ title: string; url: string; snippet: string }>>> = {
  "python function calling": [
    { title: "OpenAI Function Calling Guide", url: "https://platform.openai.com/docs/guides/function-calling", snippet: "Connect LLMs to external tools." },
    { title: "Anthropic Tool Use", url: "https://docs.anthropic.com/en/docs/build-with-claude/tool-use", snippet: "Claude can interact with tools and APIs." },
  ],
  "mcp protocol": [
    { title: "Model Context Protocol", url: "https://modelcontextprotocol.io", snippet: "Open standard connecting models to data sources." },
  ],
  "weather api": [
    { title: "OpenWeatherMap API", url: "https://openweathermap.org/api", snippet: "Free weather API." },
  ],
};

/**
 * 模拟网络搜索，按关键词在本地数据库中匹配。
 * @param args.query - 搜索关键词
 * @param args.max_results - 最多返回结果数，默认 3
 */
function webSearch(args: Readonly<Record<string, JsonValue>>): JsonValue {
  const query = String(args.query ?? "");
  const maxResults = typeof args.max_results === "number" ? args.max_results : 3;
  const key = query.toLowerCase().trim();
  for (const dbKey of Object.keys(SEARCH_DB)) {
    if (dbKey.includes(key) || key.includes(dbKey)) {
      const all = SEARCH_DB[dbKey];
      return { query, results: all.slice(0, maxResults), total: all.length };
    }
  }
  return { query, results: [], total: 0 };
}

// 模拟文件系统
const FILE_SYSTEM: Readonly<Record<string, string>> = {
  "data/config.json": '{"model": "gpt-4o", "temperature": 0.7, "max_tokens": 4096}',
  "data/users.csv": "name,email,role\nAlice,alice@example.com,admin\nBob,bob@example.com,user",
  "README.md": "# My Project\nA tool-use agent built from scratch.",
};

/**
 * 读取模拟文件系统中的文件。
 * 安全检查：拦截路径遍历（.. 和绝对路径），防止读取敏感文件。
 * @param args.path - 相对路径，如 "data/config.json"
 */
function readFile(args: Readonly<Record<string, JsonValue>>): JsonValue {
  const path = String(args.path ?? "");
  if (path.includes("..") || path.startsWith("/")) {
    return { error: true, message: "Path traversal not allowed.", code: "FORBIDDEN" };
  }
  if (!(path in FILE_SYSTEM)) {
    return { error: true, message: "File '" + path + "' not found.", available_files: Object.keys(FILE_SYSTEM), code: "NOT_FOUND" };
  }
  const content = FILE_SYSTEM[path];
  return { path, content, size_bytes: content.length, lines: content.split("\n").length };
}

/**
 * 在受限沙箱中执行 JavaScript 代码。
 * 安全机制：
 * - 黑名单拦截危险操作（require/process/fs/child_process/eval/Function）
 * - 只传入 Math 对象，无法访问全局的 process、require 等
 * - 用户代码中需要把结果赋给 result 变量，函数会将其提取出来
 * @param args.code - 要执行的 JavaScript 代码
 * @param args.language - 目前只支持 "javascript"
 */
function runCode(args: Readonly<Record<string, JsonValue>>): JsonValue {
  const code = String(args.code ?? "");
  const language = String(args.language ?? "javascript");
  if (language !== "javascript") {
    return { error: true, message: "Language '" + language + "' not supported." };
  }
  // 黑名单：拦截 Node.js 和浏览器的危险 API
  const FORBIDDEN = ["require(", "process.", "fs.", "child_process", "import ", "eval(", "Function("];
  for (const p of FORBIDDEN) {
    if (code.includes(p)) {
      return { error: true, message: "Forbidden operation: " + p, code: "SECURITY_VIOLATION" };
    }
  }
  try {
    // new Function() 创建一个隔离作用域，只传入 Math
    // eslint-disable-next-line no-new-func
    const fn = new Function("Math", "let result; " + code + "; return result;");
    const result = fn(Math) as unknown;
    return { success: true, result: result as JsonValue };
  } catch (err) {
    return { error: true, message: (err as Error).name + ": " + (err as Error).message };
  }
}

/**
 * 批量注册所有工具到全局注册表。
 * 每个工具包含：
 * - name: 工具名（模型调用时用这个名字）
 * - description: 工具用途（模型靠这段文字决定何时调用）
 * - parameters: JSON Schema 格式的参数定义
 * - fn: 实际执行的 TypeScript 函数
 */
function registerAllTools(): void {
  registerTool(
    "calculator",
    "Evaluate a math expression. Supports +, -, *, /, parentheses, decimals.",
    {
      type: "object",
      properties: {
        expression: { type: "string", description: "Math expression, e.g. '(10 + 5) * 3'" },
        precision: { type: "integer", description: "Decimal places", default: 2 },
      },
      required: ["expression"],
    },
    calculator,
  );
  registerTool(
    "get_weather",
    "Get current weather for a city.",
    {
      type: "object",
      properties: {
        city: { type: "string", description: "City name" },
        units: { type: "string", description: "celsius or fahrenheit", enum: ["celsius", "fahrenheit"] },
      },
      required: ["city"],
    },
    getWeather,
  );
  registerTool(
    "web_search",
    "Search the web.",
    {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query" },
        max_results: { type: "integer", description: "Max results", default: 3 },
      },
      required: ["query"],
    },
    webSearch,
  );
  registerTool(
    "read_file",
    "Read file contents.",
    {
      type: "object",
      properties: { path: { type: "string", description: "Relative path" } },
      required: ["path"],
    },
    readFile,
  );
  registerTool(
    "run_code",
    "Execute JavaScript in a sandbox. Assign to 'result' to return output.",
    {
      type: "object",
      properties: {
        code: { type: "string", description: "JavaScript code to run" },
        language: { type: "string", description: "javascript only", enum: ["javascript"] },
      },
      required: ["code"],
    },
    runCode,
  );
}

type ToolCall = { name: string; arguments: Readonly<Record<string, JsonValue>> };

/**
 * 模拟 LLM 的工具调用决策。
 *
 * 真实场景中，模型会分析用户消息 + 工具定义，输出 tool_calls。
 * 这里用正则关键词匹配来模拟，重点演示后续的执行流程。
 *
 * 决策逻辑：
 * - 含 "weather/temperature/forecast" → get_weather（支持并行多个城市）
 * - 含 "calculate/compute/math" → calculator
 * - 含 "search/find/look up" → web_search
 * - 含 "read/file/open/show" → read_file
 * - 含 "run/execute/code" → run_code
 * - 都不匹配 → 返回空数组（模型直接回答，不调用工具）
 *
 * @param userMessage - 用户输入的文本
 * @returns tool_calls 数组，每项 { name, arguments }
 */
function simulateModelDecision(userMessage: string): ToolCall[] {
  const msg = userMessage.toLowerCase();
  // 匹配天气查询 — 支持同时查询多个城市（并行调用）
  if (/weather|temperature|forecast/.test(msg)) {
    const cities = Object.keys(WEATHER_DB).filter((c) => msg.includes(c));
    const targets = cities.length > 0 ? cities : ["tokyo"];
    return targets.map((city) => ({
      name: "get_weather",
      arguments: { city: city.replace(/\b\w/g, (c) => c.toUpperCase()) },
    }));
  }
  // 匹配数学计算 — 提取连续的数学字符
  if (/calculate|compute|math|what is|how much/.test(msg)) {
    const m = msg.match(/[\d+\-*/().\s]{3,}/);
    if (m) return [{ name: "calculator", arguments: { expression: m[0].trim() } }];
    return [{ name: "calculator", arguments: { expression: "0" } }];
  }
  // 匹配搜索 — 去掉常见前缀词
  if (/search|find|look up/.test(msg)) {
    const query = msg.replace(/search for|look up|find|search/g, "").trim();
    return [{ name: "web_search", arguments: { query } }];
  }
  // 匹配文件读取 — 按文件名匹配，找不到就默认读 README.md
  if (/read|file|open|show/.test(msg)) {
    for (const path of Object.keys(FILE_SYSTEM)) {
      const stem = path.split("/").pop()?.split(".")[0] ?? "";
      if (stem.length > 0 && msg.includes(stem)) {
        return [{ name: "read_file", arguments: { path } }];
      }
    }
    return [{ name: "read_file", arguments: { path: "README.md" } }];
  }
  // 匹配代码执行
  if (/run|execute|code|javascript/.test(msg)) {
    return [{ name: "run_code", arguments: { code: "result = 'Hello from the sandbox!'", language: "javascript" } }];
  }
  // 没有匹配到任何工具，模型应该直接回答
  return [];
}

type ToolResult = { tool: string; result: JsonValue; executionTimeMs: number };

/**
 * 执行单个工具调用，返回结果和耗时。
 * 对应四步模式中的 "Execute" 环节。
 * 从全局注册表中查找工具函数，调用并返回结果。
 *
 * @param call - { name, arguments } 格式的工具调用
 */
function executeToolCall(call: ToolCall): ToolResult {
  const tool = TOOL_REGISTRY.get(call.name);
  if (!tool) {
    return { tool: call.name, result: { error: true, message: "Unknown tool: " + call.name, code: "UNKNOWN_TOOL" }, executionTimeMs: 0 };
  }
  const start = Date.now();
  let result: JsonValue;
  try {
    result = tool.fn(call.arguments);
  } catch (err) {
    // 参数不匹配（如传了多余的参数）会触发异常
    result = { error: true, message: "Invalid arguments: " + (err as Error).message };
  }
  return { tool: call.name, result, executionTimeMs: Date.now() - start };
}

/**
 * 根据 JSON Schema 校验工具参数是否合法。
 *
 * 校验内容：
 * 1. 工具是否存在
 * 2. 参数是否为对象（非 null、非数组）
 * 3. 必填字段是否缺失
 * 4. 参数类型是否匹配
 * 5. enum 值是否在允许范围内
 *
 * @param toolName - 工具名
 * @param args - 参数对象
 * @returns 错误信息数组，空数组表示校验通过
 */
function validateToolArguments(toolName: string, args: unknown): string[] {
  const tool = TOOL_REGISTRY.get(toolName);
  if (!tool) return ["Unknown tool: " + toolName];
  if (args === null || typeof args !== "object" || Array.isArray(args)) {
    return ["Arguments must be an object, got " + typeof args];
  }
  const schema = tool.definition.function.parameters;
  const errors: string[] = [];
  // 检查必填字段
  for (const required of schema.required ?? []) {
    if (!(required in (args as Record<string, unknown>))) {
      errors.push("Missing required argument: " + required);
    }
  }
  // JSON Schema 类型 → TypeScript 类型检查函数
  const typeChecks: Readonly<Record<ParamType, (v: unknown) => boolean>> = {
    string: (v) => typeof v === "string",
    integer: (v) => Number.isInteger(v),
    number: (v) => typeof v === "number",
    boolean: (v) => typeof v === "boolean",
    array: (v) => Array.isArray(v),
    object: (v) => v !== null && typeof v === "object" && !Array.isArray(v),
  };
  for (const [argName, argValue] of Object.entries(args as Record<string, unknown>)) {
    const prop = schema.properties[argName];
    if (!prop) {
      errors.push("Unknown argument: " + argName);
      continue;
    }
    if (!typeChecks[prop.type](argValue)) {
      errors.push("Argument '" + argName + "': expected " + prop.type + ", got " + typeof argValue);
    }
    // 枚举值校验
    if (prop.enum && !prop.enum.includes(argValue as JsonValue)) {
      errors.push("Argument '" + argName + "': '" + String(argValue) + "' not in " + JSON.stringify(prop.enum));
    }
  }
  return errors;
}

/**
 * 简化的 function calling 循环（单轮）。
 *
 * 与 Python 版的区别：TypeScript 版只做单轮迭代，不维护对话历史。
 * Python 版支持 max_iterations=5 的多轮循环。
 *
 * 流程：
 * 1. 调用 simulateModelDecision 模拟模型决策
 * 2. 如果模型决定调用工具 → 执行所有 tool calls（支持并行）
 * 3. 返回结果
 *
 * @param userMessage - 用户输入
 */
function runFunctionCallingLoop(userMessage: string): { toolResults: ToolResult[]; iterations: number } {
  const calls = simulateModelDecision(userMessage);
  if (calls.length === 0) return { toolResults: [], iterations: 0 };
  // 并行执行所有 tool calls
  const results = calls.map((c) => executeToolCall(c));
  return { toolResults: results, iterations: 1 };
}

/**
 * 运行完整的 function calling 演示。
 *
 * 包含以下演示环节：
 * 1. 工具注册 — 展示所有已注册工具的名称和参数
 * 2. 参数校验 — 演示合法/非法参数的校验结果
 * 3. 直接执行 — 绕过模型决策，直接调用工具
 * 4. 完整循环 — 模拟 "用户提问 → 模型决策 → 工具执行 → 返回结果"
 * 5. 并行调用 — 一次请求同时查询多个城市的天气
 * 6. 安全检查 — 演示路径遍历、危险代码等被拦截
 */
function main(): void {
  registerAllTools();
  console.log("=".repeat(60));
  console.log("  Function Calling and Tool Use");
  console.log("=".repeat(60));

  // ── 环节 1：展示已注册的工具 ──
  console.log("\n--- Registered Tools ---");
  for (const [name, tool] of TOOL_REGISTRY) {
    const params = Object.keys(tool.definition.function.parameters.properties);
    console.log("  " + name + ": " + tool.definition.function.description.slice(0, 60) + " | params: " + params.join(","));
  }

  // ── 环节 2：参数校验演示 ──
  console.log("\n--- Argument Validation ---");
  const validationTests: ReadonlyArray<{ tool: string; args: unknown; label: string }> = [
    { tool: "get_weather", args: { city: "Tokyo" }, label: "Valid call" },
    { tool: "get_weather", args: {}, label: "Missing required arg" },
    { tool: "get_weather", args: { city: "Tokyo", units: "kelvin" }, label: "Invalid enum value" },
    { tool: "calculator", args: { expression: 123 }, label: "Wrong type (number for string)" },
    { tool: "unknown_tool", args: { x: 1 }, label: "Unknown tool" },
  ];
  for (const { tool, args, label } of validationTests) {
    const errors = validateToolArguments(tool, args);
    console.log("  " + label + ": " + (errors.length === 0 ? "VALID" : "ERRORS: " + errors.join(" / ")));
  }

  // ── 环节 3：直接执行工具 ──
  console.log("\n--- Direct Tool Execution ---");
  const directTests: readonly ToolCall[] = [
    { name: "calculator", arguments: { expression: "(10 + 5) * 3 / 2" } },
    { name: "get_weather", arguments: { city: "Tokyo" } },
    { name: "get_weather", arguments: { city: "Mars" } },
    { name: "web_search", arguments: { query: "python function calling" } },
    { name: "read_file", arguments: { path: "data/config.json" } },
    { name: "read_file", arguments: { path: "../etc/passwd" } },
    { name: "run_code", arguments: { code: "let s=0; for(let i=1;i<=100;i++) s+=i; result=s;" } },
    { name: "run_code", arguments: { code: "require('child_process').exec('ls')" } },
  ];
  for (const call of directTests) {
    const r = executeToolCall(call);
    const argsStr = JSON.stringify(call.arguments);
    const resStr = JSON.stringify(r.result).slice(0, 90);
    console.log("\n  " + call.name + "(" + argsStr.slice(0, 60) + ")");
    console.log("    -> " + resStr);
    console.log("    time: " + r.executionTimeMs + "ms");
  }

  // ── 环节 4：完整循环演示 ──
  console.log("\n--- Function Calling Loop ---");
  const queries = [
    "What's the weather in Tokyo?",
    "Calculate (100 + 250) * 0.15",
    "Search for MCP protocol",
    "Read the config file",
    "Run some JavaScript code",
    "Tell me a joke",  // 不匹配任何工具，模型直接回答
  ];
  for (const q of queries) {
    const { toolResults, iterations } = runFunctionCallingLoop(q);
    console.log("\n  User: " + q);
    for (const tr of toolResults) {
      console.log("    Tool: " + tr.tool + " (" + tr.executionTimeMs + "ms)");
    }
    if (toolResults.length === 0) console.log("    [No tool called]");
    console.log("    Iterations: " + iterations);
  }

  // ── 环节 5：并行调用演示 ──
  console.log("\n--- Parallel Tool Calls ---");
  const { toolResults: multi } = runFunctionCallingLoop("What's the weather in tokyo and london?");
  console.log("  Tool calls made: " + multi.length);
  for (const tr of multi) {
    const r = tr.result as Record<string, JsonValue>;
    console.log("    " + String(r.city) + ": " + String(r.temp_c ?? r.temp_f) + ", " + String(r.condition));
  }

  // ── 环节 6：安全检查演示 ──
  console.log("\n--- Security Checks ---");
  const securityTests: ReadonlyArray<{ tool: string; args: Record<string, JsonValue> }> = [
    { tool: "read_file", args: { path: "../../etc/passwd" } },
    { tool: "run_code", args: { code: "process.exit(0)" } },
    { tool: "calculator", args: { expression: "Function('return 1')()" } },
  ];
  for (const { tool, args } of securityTests) {
    const r = executeToolCall({ name: tool, arguments: args });
    const blocked = typeof r.result === "object" && r.result !== null && (r.result as Record<string, JsonValue>).error === true;
    const firstArg = Object.values(args)[0];
    const argDisplay = String(firstArg).slice(0, 40);
    console.log("  " + tool + "(" + argDisplay + "): " + (blocked ? "BLOCKED" : "ALLOWED"));
  }
}

main();
