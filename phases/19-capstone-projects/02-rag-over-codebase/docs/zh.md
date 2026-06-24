# Capstone 02 — 基于代码库的 RAG（跨仓库语义搜索）

> 2026 年，每一个严肃的工程团队都在运行内部代码搜索系统，它理解的是含义而非字符串。Sourcegraph Amp、Cursor 的代码库问答、Augment 的企业图谱、Aider 的 repomap、Pinterest 的内部 MCP——都是同一套范式：摄取多个仓库，用 tree-sitter 解析，嵌入函数级和类级 chunk，混合搜索，重排序，带引用地回答。本 Capstone 要求你构建一个能处理 10 个仓库、200 万行代码，并在每次 git push 时完成增量重索引的系统。

**类型：** Capstone
**语言：** Python（摄取）、TypeScript（API + UI）
**前置要求：** Phase 5（NLP 基础）、Phase 7（transformers）、Phase 11（LLM 工程）、Phase 13（工具）、Phase 17（基础设施）
**涉及阶段：** P5 · P7 · P11 · P13 · P17
**时间：** 30 小时

## 问题

到 2026 年，每个前沿编程智能体都配备了代码库检索层，因为仅靠上下文窗口无法解决跨仓库问题。Claude 的 100 万 token 上下文有所帮助，但并不能消除对排序检索的需求。对原始 chunk 做朴素余弦搜索，会在生成代码、monorepo 重复以及很少被导入的符号的长尾分布上产生劣质结果。生产级方案是：在 AST 感知的 chunk 上做混合（稠密 + BM25）搜索，配合重排序器，并以符号引用图谱为后盾。

你需要通过索引一个真实的代码仓库集群——而非一个教程仓库——来学习，衡量 MRR@10、引用忠实度和增量新鲜度。失败模式是基础设施层面的：10 万文件的 monorepo、一次修改半个文件的 push、需要跨四个仓库才能正确回答的查询。

## 概念

AST 感知的摄取流水线用 tree-sitter 解析每个文件，提取函数和类节点，在节点边界而非固定 token 窗口处切分 chunk。每个 chunk 获得三种表示：稠密嵌入（Voyage-code-3 或 nomic-embed-code）、稀疏 BM25 词项，以及一段简短的自然语言摘要。摘要增加了第三种可检索模态——用户问"X 是如何授权的"，即使代码中只有 `check_permission`，摘要也会提到 "authz"。

检索是混合的。一个查询同时触发稠密搜索和 BM25 搜索，合并 top-k 后交给交叉编码器重排序器（Cohere rerank-3 或 bge-reranker-v2-gemma-2b）。重排序后的列表送入长上下文合成器（Claude Sonnet 4.7 配合提示缓存，或自托管的 Llama 3.3 70B），要求按文件和行范围引用每一个声明。没有引用的答案会被后过滤器拒绝。

增量新鲜度是基础设施问题。Git push 触发 diff：哪些文件变了、哪些符号变了。只有受影响的 chunk 重新嵌入。受影响的跨文件符号边（导入、方法调用）被重新计算。索引保持一致，无需每次提交都重新处理 200 万行代码。

## 架构

```
git push --> webhook --> 摄取工作流 (LlamaIndex Workflow)
                           |
                           v
             tree-sitter 解析 + AST chunk 切分
                           |
            +--------------+----------------+
            v              v                v
          稠密嵌入       BM25 索引        摘要 (LLM)
       (Voyage / bge)  (Tantivy)       (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      符号图谱 (Neo4j / kuzu)
                            |
  query --> LangGraph agent (retrieve -> rerank -> synth)
                            |
                            v
                 Claude Sonnet 4.7 1M context
                            |
                            v
                 答案 + file:line 引用
```

## 技术栈

- 解析：tree-sitter，支持 17 种语言语法（Python、TS、Rust、Go、Java、C++ 等）
- 稠密嵌入：Voyage-code-3（托管）或 nomic-embed-code-v1.5（自托管），bge-code-v1 兜底
- 稀疏索引：Tantivy（Rust）+ BM25F，按符号名与函数体加权
- 向量数据库：Qdrant 1.12（混合搜索），或 pgvector + pgvectorscale（向量少于 5000 万的团队）
- Chunk 摘要模型：Claude Haiku 4.5 或 Gemini 2.5 Flash，提示缓存
- 重排序器：Cohere rerank-3 或 bge-reranker-v2-gemma-2b 自托管
- 编排：LlamaIndex Workflows（摄取）、LangGraph（查询智能体）
- 合成器：Claude Sonnet 4.7（1M 上下文），提示缓存
- 符号图谱：Neo4j（托管）或 kuzu（嵌入式），存储导入和调用边
- 可观测性：Langfuse，为每次检索 + 合成步骤记录 span

## Build It

1. **摄取遍历器。** 在每次 push hook 时遍历 git 历史，收集变更文件。对每个文件用 tree-sitter 解析，提取函数和类节点及其完整源码范围。输出 chunk 记录 `{repo, path, start_line, end_line, symbol, body}`。

2. **Chunk 摘要器。** 批量将 chunk 送入 Haiku 4.5 调用，系统提示使用提示缓存。提示词："Summarize this function in one sentence, naming its public contract and side effects." 将摘要与 chunk 一起存储。

3. **嵌入池。** 两个并行队列：稠密嵌入（Voyage-code-3，batch 128）和摘要嵌入（同一模型，但作用于摘要字符串）。将向量写入 Qdrant，payload 为 `{repo, path, start_line, end_line, symbol, kind}`。

4. **BM25 索引。** 字段加权的 Tantivy 索引：符号名权重 4，函数体权重 1，摘要权重 2。支持"查找名为 X 的函数"和"查找实现 X 功能的函数"两种查询。

5. **符号图谱。** 对每个 chunk 记录边：导入（本文件使用仓库 Z 的符号 Y）、调用（本函数调用类 C 的方法 M）、继承。存储在 kuzu 中。查询时用于跨仓库扩展检索。

6. **查询智能体。** LangGraph，三个节点。`retrieve` 并行触发稠密 + BM25 搜索，按 (repo, path, symbol) 去重。`rerank` 在 top-50 上运行交叉编码器，保留 top-10。`synth` 将重排序后的 chunk 放入上下文调用 Claude Sonnet 4.7，缓存系统提示，要求 file:line 引用。

7. **引用强制执行。** 解析模型输出；任何没有 `(repo/path:start-end)` 锚点的声明被标记要求重新提问或丢弃。只返回带引用的答案。

8. **增量重索引。** 每次 webhook 时计算符号级 diff。只重新嵌入文本变更的 chunk。对导入变更的 chunk 重新计算符号边。衡量指标：50 文件的 push 在 60 秒内完成 200 万行代码库的重索引。

9. **评估。** 标注 100 个跨仓库问题，附带黄金 file:line 答案。衡量 MRR@10、nDCG@10、引用忠实度（可验证锚点的声明占比）、p50/p99 延迟。

## Use It

```
$ code-rag ask "how is S3 multipart abort wired into our retry budget?"
[retrieve]  12 chunks dense + 7 chunks bm25, 16 unique after dedup
[rerank]    top-5 kept (cohere rerank-3)
[synth]     claude-sonnet-4.7, cache hit rate 68%, 2.1s
answer:
  Multipart aborts are triggered by `AbortMultipartOnFail` in
  services/uploader/retry.go:122-148, which decrements the per-bucket
  retry budget defined in config/budgets.yaml:34-51 ...
  citations: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

## Ship It

交付物：skill `outputs/skill-codebase-rag.md`。给定一个仓库集合，它能搭建摄取流水线、混合索引和查询智能体，对任何跨仓库问题返回带引用的答案。评分标准：

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 检索质量 | 100 题留出集上的 MRR@10 和 nDCG@10 |
| 20 | 引用忠实度 | 答案中可验证 file:line 锚点的声明占比 |
| 20 | 延迟与规模 | 索引语料库规模下 10k QPS 时的 p95 查询延迟 |
| 20 | 增量索引正确性 | 50 文件提交从 git push 到可搜索的时间 |
| 15 | UX 与答案格式 | 引用可点击性、代码片段预览、追问能力 |
| **100** | | |

## 练习

1. 将 Voyage-code-3 替换为自托管的 nomic-embed-code，衡量 MRR@10 差异。报告启用重排序后差距是否缩小。

2. 向语料库注入 20% 生成代码（LLM 产生的样板代码）并重新评估。观察检索污染。在 payload 中添加 "generated" 标志并降低这些结果的权重。

3. 在你的语料库规模下基准测试 Qdrant 混合搜索 vs pgvector + pgvectorscale。报告 batch size 1 时的 p99。

4. 添加基于采样的漂移检查：每周重新运行 100 题评估。MRR@10 下降超过 5% 时告警。

5. 扩展到跨语言符号解析：一个 Python 函数通过 gRPC 调用 Go 服务。使用符号图谱将它们关联起来。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| AST 感知 chunking | "函数级切分" | 在 tree-sitter 节点边界而非固定 token 窗口处切分代码 |
| 混合搜索 | "稠密 + 稀疏" | 并行运行 BM25 和向量搜索，合并 top-k，重排序 |
| 交叉编码器重排序 | "第二阶段排序" | 对每个 (query, candidate) 对一起打分的模型，比余弦相似度更准确 |
| 提示缓存 | "缓存的系统提示" | 2026 年 Claude / OpenAI 的功能，对重复前缀 token 折扣最高达 90% |
| 符号图谱 | "代码图谱" | 跨文件和仓库的导入、调用、继承边 |
| 引用忠实度 | "有依据的回答率" | 用户可通过点击锚点并阅读引用来验证的声明占比 |
| 增量重索引 | "push 到可搜索时间" | 从 git push 到变更符号可查询的挂钟时间 |

## 延伸阅读

- [Sourcegraph Amp](https://ampcode.com) — 生产级跨仓库代码智能
- [Sourcegraph Cody RAG 架构](https://sourcegraph.com/blog/how-cody-understands-your-codebase) — 本 Capstone 的参考深度解析
- [Aider repo-map](https://aider.chat/docs/repomap.html) — tree-sitter 排序的仓库视图
- [Augment Code 企业图谱](https://www.augmentcode.com) — 商业符号图谱 RAG
- [Qdrant 混合搜索文档](https://qdrant.tech/documentation/concepts/hybrid-queries/) — 参考实现
- [Voyage AI 代码嵌入](https://docs.voyageai.com/docs/embeddings) — Voyage-code-3 详情
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank) — 交叉编码器参考
- [Pinterest MCP 内部搜索](https://medium.com/pinterest-engineering) — 内部平台参考
