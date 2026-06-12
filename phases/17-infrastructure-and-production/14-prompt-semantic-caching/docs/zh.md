# Prompt 缓存与语义缓存经济学

> **定价快照日期为 2026-04。** 以下数字反映了本课发布时抓取的厂商费率卡；在下游引用前请通过链接文档核实。

> 缓存发生在两层。L2（提供商级）prompt/prefix 缓存复用重复前缀的 attention KV——Anthropic 的 prompt caching 文档声称在长 prompt 上最多可降低 90% 成本和 85% 延迟；对于 Claude 3.5 Sonnet，缓存读取价格为 $0.30/M，而新写入为 $3.00/M，5 分钟 TTL，1 小时 TTL 选项有 2 倍写入溢价（docs.anthropic.com, 2026-04）。OpenAI prompt caching 自动应用于 ≥1024 token 的 prompt，缓存输入价格约为新输入的 10% 折扣（platform.openai.com, 2026-04）；每个模型的确切缓存费率取决于当前费率卡。L1（应用级）语义缓存在 embedding 相似度命中时完全跳过 LLM。厂商的"95% 准确率"指的是匹配正确性，而非命中率——报告的生产命中率从 10%（开放式聊天）到 70%（结构化 FAQ）不等；两者均未发布官方基线，因此将其视为社区遥测数据而非保证。生产中的陷阱：并行化会杀死缓存（在首次缓存写入完成前发出 N 个并行请求可使开销膨胀数倍），前缀内的动态内容会完全阻止缓存命中。ProjectDiscovery 报告通过将动态文本移出可缓存前缀，将命中率从 7% 提升到 74%（2025-11）。

**类型：** 学习
**语言：** Python（stdlib，用于模拟两层缓存的玩具模拟器）
**前置要求：** 阶段 17 · 04（vLLM 服务内部原理）、阶段 17 · 06（SGLang RadixAttention）
**时间：** 约 60 分钟

## 学习目标

- 区分 L2 prompt/prefix 缓存（提供商处的 KV 复用）与 L1 语义缓存（在相似 prompt 上跳过 LLM）。
- 解释 Anthropic 的 `cache_control` 显式标记和两种 TTL 选项（5 分钟 vs 1 小时）及其价格倍数。
- 根据命中率、prompt/响应比例和 token 价格计算预期月度节省。
- 列举使账单膨胀 5-10 倍的并行化反模式和使命中率崩溃的动态内容反模式。

## 问题背景

你为 RAG 服务添加了 prompt caching。账单保持不变。你测量命中率；只有 7%。你的 prompt 看起来是静态的但其实不是——system prompt 包含格式化到分钟的当前日期、一个请求 ID，以及为了多样性而随机重排的示例。每个请求都写入新的缓存条目，读取为零。

另外，你的 agent 对每个用户问题运行十个并行工具调用。全部十个在第一个缓存写入完成前到达提供商。十次写入，零次读取。你的账单是"使用缓存"预期成本的 5-10 倍。

缓存是一个协议，不是一个开关。两层，两种不同的失败模式。

## 核心概念

### L2 — 提供商 prompt/prefix 缓存

提供商存储可缓存前缀的 attention KV，并在下一个匹配该前缀的请求时复用。你支付一次写入成本，读取几乎免费。

**Anthropic（Claude 3.5 / 3.7 / 4 系列）**：在请求中使用显式 `cache_control` 标记。你标记哪些块是可缓存的。TTL：5 分钟（写入成本为基准的 1.25 倍）或 1 小时（写入成本为基准的 2 倍）。缓存读取：Claude 3.5 Sonnet 上 $0.30/M vs 新输入 $3.00/M——便宜 10 倍（docs.anthropic.com, 2026-04）。费率因模型而异（Opus/Haiku 单独发布）；始终交叉检查当前定价页面。

**OpenAI**：自动缓存 ≥1024 token 的 prompt（platform.openai.com, 2026-04）。无显式标志。在当前 gpt-4o/gpt-5 费率卡上，缓存输入约为新输入的 10% 价格。文档和发布说明均未发布官方命中率基线；社区报告集中在 30-60%（配合仔细的 prompt 设计）。监控 `usage.cached_tokens` 来测量你自己的命中率。

**Google（Gemini）**：通过显式 API 进行上下文缓存；1M token 上下文意味着缓存收益更大。

**自托管（vLLM、SGLang）**：阶段 17 · 06 涵盖 RadixAttention——在你自己的计算资源上使用相同模式。

### L1 — 应用级语义缓存

在调用 LLM 之前，对 prompt 进行哈希、embedding，并查找相似的缓存请求（余弦相似度高于阈值，通常 0.95+）。命中时返回缓存响应。未命中时调用 LLM 并缓存结果。

开源方案：Redis Vector Similarity、GPTCache、Qdrant。商业方案：Portkey Cache、Helicone Cache。

厂商的准确率声明指的是返回的缓存响应在语义上合适的频率——而非你的命中频率。生产命中率：

- 开放式聊天：10-15%。
- 结构化 FAQ / 客服：40-70%。
- 代码问题：20-30%（小变体会杀死命中）。
- 语音代理重复 prompt：50-80%（语音归一化固定集）。

### 并行化反模式

你的 agent 并行发起 10 个工具调用。全部 10 个有相同的 4K token system prompt。Anthropic 的缓存写入是按请求的；第一次缓存写入在提供商看到 prompt 后约 300 ms 完成。请求 2-10 在同一毫秒窗口到达，每个都看到缓存未命中。你支付 10 次写入溢价，0 次读取折扣。

修复：先批量串行——先发请求 1，等 1 的缓存填充后再发 2-10。第一次工具调用增加 300 ms；节省 5-10 倍的账单。

### 动态内容反模式

你的 system prompt 看起来像：

```
You are a helpful assistant. The current time is 14:32:17.
User ID: abc123. Today is Tuesday...
```

每个请求都是唯一的。每个请求都写入。零命中。

修复：将真正静态的内容移到可缓存前缀；在缓存边界之后追加动态内容：

```
[cacheable]
You are a helpful assistant. [rules, examples, instructions]
[/cacheable]
[dynamic, not cached]
Current time: 14:32:17. User: abc123.
```

ProjectDiscovery 通过这种方式将缓存命中率从 7% 提升到 74%，并公开了具体做法。

### 批量 + 缓存组合用于夜间工作负载

Batch API（阶段 17 · 15）在 24 小时周转时间下提供 50% 折扣。在此基础上叠加缓存输入可再获得约 10 倍的折扣。夜间分类、标注和报告生成工作负载通过组合可降至同步无缓存成本的约 10%。

### 需要记住的数据

定价数据于 2026-04 从链接的厂商文档抓取，每隔几个月会变动——在依赖前请重新核实。

- Anthropic 缓存读取：Claude 3.5 Sonnet 上 $0.30/M，约为新输入价格的 10%（docs.anthropic.com）。
- Anthropic 缓存写入溢价：1.25 倍（5 分钟 TTL）或 2 倍（1 小时 TTL）。
- OpenAI 自动缓存：适用于 ≥1024 token 的 prompt；在当前费率卡上缓存输入价格约为新输入的 10%（platform.openai.com）。
- 语义缓存命中率（社区报告）：开放式聊天约 10%；结构化 FAQ 最高约 70%。非厂商记录的基线。
- ProjectDiscovery：通过将动态内容移出前缀，命中率从 7% 提升到 74%（项目博客, 2025-11）。
- 并行化反模式：当 N 个并行请求错过首次缓存写入时，典型报告为账单膨胀 5-10 倍。

## 使用

`code/main.py` 在混合工作负载上模拟 L1 + L2 缓存。报告命中率、账单，并展示并行化惩罚。

## 产出

本课产出 `outputs/skill-cache-auditor.md`。根据 prompt 模板和流量，审计可缓存性并建议重构。

## 练习

1. 运行 `code/main.py`。切换并行化标志。账单变化多少？
2. 你的 system prompt 包含日期。将其移出。展示前后的命中率数学计算。
3. 计算在你的请求到达率下，1 小时 TTL（2 倍写入）vs 5 分钟 TTL（1.25 倍写入）的盈亏平衡点。
4. 语义缓存在 0.95 阈值下命中 20%。在 0.85 下命中 50% 但你会看到不正确的缓存响应。选择正确的阈值并说明理由。
5. 你对每个用户问题批量处理 10 个并行子查询。在不增加端到端延迟的情况下重写以提高缓存友好性。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| L2 prompt cache | "前缀缓存" | 提供商为重复前缀存储 KV |
| `cache_control` | "Anthropic 缓存标记" | 显式属性标记可缓存块 |
| Cache write premium | "写入税" | 首次未命中到缓存的额外成本（1.25 倍或 2 倍） |
| L1 semantic cache | "embedding 缓存" | 应用级的哈希 + embedding，在调用 LLM 之前 |
| GPTCache | "LLM 缓存库" | 流行的开源 L1 缓存库 |
| Cache hit rate | "命中数 / 总数" | 由缓存服务的请求比例 |
| Parallelization anti-pattern | "N 次写入陷阱" | N 个并行请求导致 N 次缓存未命中 |
| Dynamic content trap | "prompt 中的时间陷阱" | 前缀中的动态字节杀死命中率 |
| RadixAttention | "副本内缓存" | SGLang 的前缀缓存实现 |

## 延伸阅读

- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 官方 `cache_control` 语义和 TTL。
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching) — 自动缓存行为和资格。
- [TianPan — Semantic Caching for LLMs Production](https://tianpan.co/blog/2026-04-10-semantic-caching-llm-production)
- [ProjectDiscovery — Cut LLM Costs 59% With Prompt Caching](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)
- [DigitalOcean / Anthropic — Prompt Caching](https://www.digitalocean.com/blog/prompt-caching-with-digital-ocean)
