# 安全 —— 密钥管理、API 密钥轮换、审计日志与护栏

> 通过集中式保险库（HashiCorp Vault、AWS Secrets Manager、Azure Key Vault）消除密钥扩散。切勿在配置文件、版本控制系统（VCS）中的环境文件或电子表格中存储凭证。使用 IAM 角色而非静态密钥；CI/CD 使用 OIDC。AI 网关模式是 2026 年的解决方案：应用 → 网关 → 模型提供商，网关在运行时从保险库中拉取凭证。在保险库中轮换凭证，所有应用在几分钟内即可生效——无需重新部署，无需在 Slack 上询问“谁有新密钥”。轮换策略：≤90 天；使用 TruffleHog / GitGuardian / Gitleaks 在每次提交时扫描。零信任（Zero-trust）：MFA、SSO、RBAC/ABAC、短期令牌、设备态势。PII 清洗（PII Scrubbing）使用实体识别在转发前对 PHI/PII 进行脱敏；一致的 Token 化（Mesh 方法）将敏感值映射到稳定的占位符，以便 LLM 保留代码/关系语义。网络出站：LLM 服务位于专用 VPC/VNet 子网，仅允许访问 `api.openai.com`、`api.anthropic.com` 等；阻止所有其他出站流量。2026 年的事件驱动因素：Vercel 供应链攻击，通过泄露的 CI/CD 凭证窃取了数千个客户部署的环境变量。

**类型：** 学习
**语言：** Python（标准库，玩具级 PII 清洗器 + 审计日志写入器）
**前置知识：** 第 17 阶段 · 19（AI 网关），第 17 阶段 · 13（可观测性）
**时间：** 约 60 分钟

## 学习目标

- 列举四种密钥管理反模式（VCS 中的配置文件、硬编码的环境变量、电子表格、静态密钥）并说出其替代方案。
- 解释“AI 网关从保险库拉取”模式作为 2026 年生产环境标准。
- 实现一个具有一致 Token 化（相同值 → 相同占位符）的 PII 清洗器，以保留语义。
- 说出 2026 年 Vercel 供应链事件及其关于 CI/CD 凭证卫生的教训。

## 问题所在

一名实习生提交了包含 API 密钥的 `.env` 文件。他们很快删除了该文件。但密钥已经存在于 Git 历史中——GitGuardian 扫描发现了它，而您的轮换流程是“在 Slack 上通知团队，更新 40 个配置文件，重新部署所有服务。”8 小时后，一半的服务上线了，另一半还在等待部署窗口。

另外，用户提示词中包含“我的社保号是 123-45-6789。”提示词被发送给 OpenAI。您有商业关联协议（BAA），但内部政策要求在转发前对 PII 进行脱敏。您没有这样做。

此外，您的 EKS 集群中的 LLM Pod 可以访问任何互联网主机。有人通过 DNS 查询向攻击者控制的域名窃取了数据。没有任何东西阻止它。

LLM 服务的安全必须处理所有这三个层面：基于保险库的凭证、PII 清洗和网络出站过滤，以及审计日志。

## 核心概念

### 集中式保险库 + IAM 角色拉取

**保险库（Vault）**：HashiCorp Vault、AWS Secrets Manager、Azure Key Vault、GCP Secret Manager。单一事实来源。

**IAM 角色**：应用/网关通过其 IAM 身份进行身份验证，而非静态密钥。保险库在令牌有效期内返回密钥。

**AI 网关模式**：网关在请求时从保险库中拉取 `OPENAI_API_KEY`。在保险库中轮换密钥；下一个请求将获得新密钥。无需重新部署。

### 轮换策略 ≤ 90 天

所有 API 密钥、保险库根令牌、CI/CD 凭证。尽可能实现自动轮换。手动轮换需记录并跟踪。

### 密钥扫描

- **TruffleHog** —— 对提交进行正则表达式和熵检测。
- **GitGuardian** —— 商业软件，准确率高。
- **Gitleaks** —— 开源软件，在 CI 中运行。

在每次提交时运行。如果检测到新密钥，则阻止 PR。

### 零信任态势

- 所有账户强制要求 MFA。
- 通过 SAML/OIDC 实现 SSO。
- RBAC（基于角色）或 ABAC（基于属性）实现细粒度访问控制。
- 短期令牌（以小时为单位，而非天）。
- 设备态势：仅允许启用了磁盘加密的公司设备。

### PII / PHI 清洗

在提示词离开您的基础设施之前：

1. 实体识别（Entity Recognition）（spaCy NER、Presidio、商业软件）。
2. 脱敏匹配的实体：`"My SSN is 123-45-6789"` → `"My SSN is [SSN_TOKEN_A3F]"`。
3. 一致的 Token 化（Mesh 方法）：相同的值映射到相同的占位符，以便 LLM 保留关系。
4. 可选的 LLM 响应逆向映射。

静态正则表达式过滤器可捕获基本模式；NER 可捕获更多。建议两者结合使用。

### 输入 + 输出护栏

输入：阻止已知的越狱攻击、禁止的话题；按用户限制速率。

输出：正则表达式清洗泄露的密钥（API 密钥模式、拒绝语境中的电子邮件模式）；分类器检测策略违规。

### 网络出站白名单

LLM 服务位于专用子网：
- 白名单：`api.openai.com`、`api.anthropic.com`、向量数据库端点、保险库端点。
- 其他所有流量：丢弃。
- DNS 通过仅允许白名单的解析器进行（避免 DNS 隧道窃取数据）。

### 审计日志

不可变的每次 LLM 调用日志，包含：
- 时间戳。
- 用户/租户。
- 提示词哈希（出于隐私考虑，不记录原始提示词）。
- 模型 + 版本。
- Token 计数。
- 成本。
- 响应哈希。
- 任何护栏触发事件。

根据监管要求保留（SOC 2 保留 1 年，HIPAA 保留 6 年）。

### 2026 年 Vercel 事件

供应链攻击：泄露的 CI/CD 凭证窃取了数千个客户部署的环境变量。教训：CI/CD 凭证等同于生产凭证。存储在保险库中。严格控制范围。积极轮换。

### 您应该记住的数字

- 轮换策略：≤ 90 天。
- 每次提交扫描：TruffleHog / GitGuardian / Gitleaks。
- Vercel 2026：CI/CD 凭证泄露 → 数千个客户环境变量泄露。
- 审计日志保留：SOC 2 = 1 年，HIPAA = 6 年。

## 使用它

`code/main.py` 实现了一个具有一致 Token 化功能的玩具级 PII 清洗器和一个仅追加的审计日志。

## 交付它

本课程产出 `outputs/skill-llm-security-plan.md`。根据监管范围和当前状态，规划保险库迁移、清洗器、出站和审计日志。

## 练习

1. 运行 `code/main.py`。发送两个引用相同社保号（SSN）的提示词。确认两者获得相同的占位符。
2. 为一个在 EKS 上运行 vLLM 并调用 OpenAI + Anthropic + Weaviate 的部署设计网络出站策略。
3. 您在 Git 历史中发现了一个密钥（2 年前）。正确的响应是什么——轮换密钥、清理历史记录，还是两者都做？请说明理由。
4. 您的审计日志每天增长 10 GB。设计保留层级（热数据 30 天，温数据 12 个月，冷数据 6 年）。
5. 论证逆向 Token 化（将真实值替换回 LLM 响应中）的复杂性是否值得，而不是让占位符保持可见。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 保险库 (Vault) | “密钥存储” | 集中式凭证管理服务 |
| IAM 角色 | “基于身份的身份验证” | 应用承担的角色；返回短期凭证 |
| OIDC for CI/CD | “云颁发的令牌” | CI 中没有静态密钥——通过 OIDC 进行身份验证 |
| TruffleHog / GitGuardian / Gitleaks | “密钥扫描器” | 提交时的密钥检测 |
| RBAC / ABAC | “访问控制” | 基于角色 vs 基于属性 |
| PII 清洗 | “数据脱敏” | 移除或 Token 化敏感实体 |
| 一致的 Token 化 (Consistent tokenization) | “稳定的占位符” | 相同的值 → 每次都产生相同的令牌 |
| Mesh 方法 (Mesh approach) | “Mesh Token 化” | 保留语义的 Token 化模式 |
| 出站白名单 (Egress whitelist) | “出站允许列表” | 仅允许访问特定域名 |
| 审计日志 (Audit log) | “不可变历史记录” | 用于合规的仅追加记录 |

## 延伸阅读

- [Doppler — 高级 LLM 安全](https://www.doppler.com/blog/advanced-llm-security)
- [Portkey — 使用密钥引用管理 LLM API 密钥](https://portkey.ai/blog/secret-references-ai-api-key-management/)
- [Datadog — LLM 护栏最佳实践](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [JumpServer — 密钥管理最佳实践 2026](https://www.jumpserver.com/blog/secret-management-best-practices-2026)
- [Microsoft Presidio](https://github.com/microsoft/presidio) —— PII 检测和匿名化。
- [HashiCorp Vault 文档](https://developer.hashicorp.com/vault/docs)