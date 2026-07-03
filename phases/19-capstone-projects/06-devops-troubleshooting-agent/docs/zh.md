# Capstone 06 — DevOps 故障排查智能体（Kubernetes）

> AWS 的 DevOps 智能体已正式发布，Resolve AI 发布了它的 K8s 剧本，NeuBird 演示了语义监控，Metoro 将 AI SRE 与每个服务的 SLO 关联起来。生产形态已经确定：告警 webhook 触发，智能体读取遥测数据，遍历 K8s 对象图，对根本原因假设进行排序，并发布带有审批按钮的 Slack 简报。默认只读。每次修复都由人工审批。本毕业项目就是这个智能体，在 20 个合成事件上进行评估，并在三个共享案例上与 AWS 的智能体进行比较。

**类型：** 毕业项目  
**语言：** Python（智能体），TypeScript（Slack 集成）  
**前置要求：** 第 11 阶段（LLM 工程），第 13 阶段（工具和 MCP），第 14 阶段（智能体），第 15 阶段（自主系统），第 17 阶段（基础设施），第 18 阶段（安全）  
**涉及阶段：** P11 · P13 · P14 · P15 · P17 · P18  
**时间：** 30 小时

## 问题

2025-2026 年的 SRE 叙事变成了："AI 智能体分类事件，人类审批修复方案。" AWS DevOps Agent、Resolve AI、NeuBird、Metoro、PagerDuty AIOps 都在生产中发布了这种形态。智能体读取 Prometheus 指标、Loki 日志、Tempo 追踪、kube-state-metrics 以及 K8s 对象的知识图谱。它在五分钟内生成带有遥测引用的排序根本原因假设。它永远不会在没有通过 Slack 获得明确人工批准的情况下执行破坏性命令。

大部分艰苦工作是界定范围和安全，而不是推理。智能体需要一个默认只读的 RBAC 表面、一个加固的 MCP 工具服务器，以及每个命令的考虑与执行审计日志。它需要知道何时超出能力范围并升级。而且它必须运行得足够便宜，这样 OOM-kill 级联不会产生 5000 美元的智能体账单。

## 概念

智能体在知识图谱上操作。节点是 K8s 对象（Pod、Deployment、Service、Node、HPA、PVC）加上遥测源（Prometheus 序列、Loki 流、Tempo 追踪）。边编码所有权（Pod -> ReplicaSet -> Deployment）、调度（Pod -> Node）和观测（Pod -> Prometheus 序列）。图通过 kube-state-metrics 同步保持新鲜，并在每次告警时重新采样。

当告警触发时，智能体从受影响的对象开始进行根本原因分析。它遍历边，拉取相关的遥测切片（最近 15 分钟），并起草一个假设。假设根据证据进行排名：有多少遥测引用支持它，有多新，有多具体。前三个假设发送到 Slack，带有图路径可视化和修复操作的审批按钮。

修复是门控的。允许的默认操作是只读的。破坏性操作（缩容、回滚、删除 Pod）需要 Slack 审批；ArgoCD 回滚钩子需要智能体从未持有的认证令牌。审计日志记录智能体*考虑*的每个命令——不仅仅是执行的——这样审查过程可以捕捉到未遂事件。

## 架构

```
PagerDuty / Alertmanager webhook
           |
           v
     FastAPI 接收器
           |
           v
   LangGraph 根本原因智能体
           |
           +---- 只读 MCP 工具 ----+
           |                       |
           v                       v
   K8s 知识图谱              遥测切片
     (Neo4j / kuzu)          Prometheus, Loki, Tempo
   所有权 + 调度             最近 15m, 限定范围
           |
           v
   假设排序 (证据权重)
           |
           v
   Slack 简报 + 审批按钮
           |
           v (已批准)
   ArgoCD 回滚钩子 / PagerDuty 升级
           |
           v
   审计日志: 考虑 vs 执行, 每个命令
```

## 技术栈

- 可观测性源：Prometheus、Loki、Tempo、kube-state-metrics
- 知识图谱：Neo4j（托管）或 kuzu（嵌入式），包含 K8s 对象 + 遥测边
- 智能体：LangGraph，带每工具允许列表，默认只读
- 工具传输：FastMCP over StreamableHTTP；破坏性工具在审批门控后的单独服务器
- 模型：Claude Sonnet 4.7 用于根本原因推理，Gemini 2.5 Flash 用于日志摘要
- 修复：ArgoCD 回滚 webhook，PagerDuty 升级，Slack 审批卡片
- 审计：仅追加的结构化日志（考虑、执行、审批、结果）
- 部署：K8s 部署，带有自己的窄 RBAC 角色；独立命名空间

## 构建它

1. **图摄取。** 每 30 秒将 kube-state-metrics 同步到 Neo4j/kuzu。节点：Pod、Deployment、Node、Service、PVC、HPA。边：OWNED_BY、SCHEDULED_ON、EXPOSES、MOUNTS、SCALES。遥测覆盖边：OBSERVED_BY（Pod 被 Prometheus 序列观测）。

2. **告警接收器。** FastAPI 端点，接受 PagerDuty 或 Alertmanager webhook。提取受影响的对象和 SLO 违规。

3. **只读工具表面。** 通过 FastMCP 封装 kubectl、Prometheus 查询、Loki logql、Tempo traceql。每个工具有窄 RBAC 动词（"get"、"list"、"describe"）。默认服务器中没有 "delete"、"exec"、"scale"。

4. **根本原因智能体。** LangGraph 有三个节点：`sample` 拉取最近 15 分钟的遥测切片，`walk` 查询图中相邻对象，`hypothesize` 起草带有遥测引用的排序根本原因候选。

5. **证据评分。** 每个假设的分数 = 新鲜度 * 具体性 * 图路径长度倒数 * 引用数量。返回前三个。

6. **Slack 简报。** 发布附件，包含假设、图路径可视化（服务器端渲染的子图图像），以及最多一个修复操作的审批按钮。

7. **修复门控。** 破坏性工具（缩容、回滚、删除）位于第二个 MCP 服务器上，需要审批令牌。只有在 Slack 卡片被人工批准后，智能体才能调用它们。

8. **审计日志。** 仅追加的 JSONL：对于每个候选命令，记录是否被考虑、是否执行、谁批准的。每天发送到 S3。

9. **合成事件套件。** 构建 20 个场景：OOMKill 级联、DNS 抖动、HPA 抖动、PVC 满、嘈杂邻居、故障 sidecar、错误 ConfigMap 发布、证书轮换、镜像拉取退避等。根据根本原因准确性和假设时间对智能体进行评分。

## 使用它

```
webhook: alert.pagerduty.com -> checkout-api SLO 违规, 错误率 14%
[graph]   受影响: Deployment checkout-api (3 Pods, Node ip-10-2-3-4)
[walk]    相邻对象: ReplicaSet checkout-api-abc, Service checkout-api,
           最近发布 14 分钟前
[sample]  prometheus 错误率 14%, 上升趋势; loki 500s on /api/v2/pay
[hypo]    #1 错误发布: 最新镜像 checkout-api:v2.41 /healthz 失败
          引用: deploy.yaml (rev 42), prometheus errorRate, loki 500 stack
[slack]   [回滚到 v2.40]  [升级]  [忽略]
          (需要审批；智能体不会单方面回滚)
```

## 交付它

`outputs/skill-devops-agent.md` 是交付物。给定 K8s 集群和告警源，智能体生成排序的根本原因假设和 Slack 门控的修复流程。

| 权重 | 标准 | 衡量方式 |
|:---:|------|----------|
| 25 | 场景套件上的 RCA 准确性 | 20 个合成事件中 ≥80% 正确的根本原因 |
| 20 | 安全性 | 破坏性操作保护在审计日志中从未在没有 Slack 审批的情况下触发 |
| 20 | 假设时间 | 从告警到 Slack 简报 p50 低于 5 分钟 |
| 20 | 可解释性 | 每个假设有图路径和遥测引用 |
| 15 | 集成完整性 | PagerDuty、Slack、ArgoCD、Prometheus 端到端工作 |
| **100** | | |

## 练习

1. 在 AWS DevOps 智能体演示的相同三个事件上运行你的智能体。发布并排比较。报告智能体在哪里出现分歧。

2. 添加"未遂"审计，标记智能体*考虑*过的任何如果没有批准就会是破坏性的命令。测量一周内的未遂率。

3. 将假设模型从 Claude Sonnet 4.7 替换为自托管的 Llama 3.3 70B。测量 RCA 准确性差异和每事件成本。

4. 构建因果过滤器：区分相关的遥测峰值和真正的根本原因。在 20 个场景标签上训练一个小分类器。

5. 添加回滚演练：针对暂存集群使用相同清单运行 ArgoCD 回滚。在 Slack 审批按钮之前，在实时集群中验证回滚计划。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| K8s 知识图谱 | "集群图" | 节点 = K8s 对象 + 遥测序列；边 = 所有权、调度、观测 |
| 默认只读 | "限定 RBAC" | 智能体的服务帐户只有 get/list/describe 动词；破坏性动词在审批后的单独服务器中 |
| 审计日志 | "考虑 vs 执行" | 每个候选命令的仅追加记录，是否运行，谁批准 |
| 假设排序 | "证据评分" | 新鲜度 × 具体性 × 图路径长度倒数 × 引用数量 |
| Slack 审批卡片 | "HITL 门控" | 带修复按钮的交互式 Slack 消息；智能体在人工点击前无法继续 |
| 遥测引用 | "证据指针" | 支持声明的 Prometheus 查询、Loki 选择器或 Tempo 追踪 URL |
| MTTR | "解决时间" | 从告警触发到 SLO 恢复的墙钟时间 |

## 延伸阅读

- [AWS DevOps Agent 正式发布](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — 2026 年权威参考
- [Resolve AI K8s 故障排查](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai) — 竞争对手参考
- [NeuBird 语义监控](https://www.neubird.ai) — 语义图方法
- [Metoro AI SRE](https://metoro.io) — SLO 优先的生产框架
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) — 集群状态源
- [LangGraph](https://langchain-ai.github.io/langgraph/) — 参考智能体编排器
- [FastMCP](https://github.com/jlowin/fastmcp) — Python MCP 服务器框架
- [ArgoCD 回滚](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/) — 门控修复目标