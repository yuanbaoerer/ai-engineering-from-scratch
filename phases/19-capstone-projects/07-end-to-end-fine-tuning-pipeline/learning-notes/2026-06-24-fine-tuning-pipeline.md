# 端到端微调流水线架构

> 日期: 2026-06-24

## 1. 流水线整体架构

8 个阶段按 DAG 顺序执行，从原始数据到模型发布：

| 序号 | 阶段 | 函数 | 产物类型 | 生产工具 |
|------|------|------|----------|----------|
| 1 | 数据清洗 | `stage_data` | dataset | Datatrove + Nemotron-CC + Presidio |
| 2 | 污染检测 | `stage_contamination` | report | MinHash LSH |
| 3 | SFT 监督微调 | `stage_sft` | checkpoint | Axolotl + ZeRO-3 |
| 4 | DPO 偏好优化 | `stage_dpo` | checkpoint | TRL DPO |
| 5 | 量化 | `stage_quantize` | quant | GPTQ + AWQ + GGUF |
| 6 | 部署 | `stage_serve` | endpoint | vLLM + EAGLE-3 |
| 7 | 评估 | `stage_eval` | report | lm-eval-harness |
| 8 | 模型卡 | `stage_model_card` | report | MOF 2026 模板 |

## 2. Artifact + Manifest 模型

**核心设计**：基于内容哈希的资产管理，而非基于文件路径。

- **Artifact**：每个阶段的输出产物，包含 name、kind、payload、produced_by
- **Manifest**：中央注册表，按名称索引所有 Artifact
- **content_hash()**：对 payload 做 SHA256 取前 12 位，作为产物的唯一指纹

关键机制：后续阶段通过 `前序Artifact.content_hash()` 引用前序产物，而非直接传递数据对象。这建立了**不可变的依赖链**——修改任何中间产物的 payload，其哈希变化会自动传播到所有下游。

## 3. 阶段间依赖链

```
dataset.content_hash()  ──→  contamination_check
dataset.content_hash()  ──→  sft_checkpoint
sft_checkpoint.content_hash()  ──→  dpo_checkpoint
dpo_checkpoint.content_hash()  ──→  quants
dpo_checkpoint.content_hash()  ──→  endpoint
dpo_checkpoint.content_hash()  ──→  eval_report
sft_checkpoint.content_hash()  ──→  model_card
```

注意：`quants`、`endpoint`、`eval_report` 三个阶段都依赖 `dpo_checkpoint`，形成分叉。`model_card` 则回溯引用 `sft_checkpoint` 的哈希，用于可复现性声明。

## 4. cfg 配置系统

`cfg` 是声明式配置字典，等价于 YAML 配置文件的 Python 内存表示：

```python
cfg = {
    "base_model": "llama-3.3-8b",   # 基础模型
    "raw_examples": 300_000,        # 原始样本数
    "seed": 7,                      # 随机种子（可复现性）
    "dpo_beta": 0.08,              # DPO 温度参数
}
```

各阶段通过 `cfg.get("key")` 读取所需配置项。实际生产中从 `config/llama3.3-8b-domainX.yaml` 加载。

## 5. 数据准备阶段细节

三个串行过滤步骤，每步有独立保留率：

| 步骤 | 保留率 | 工具 | 作用 |
|------|--------|------|------|
| 去重 | 94% | Datatrove | 移除重复样本 |
| 质量过滤 | 91% | Nemotron-CC 分类器 | 移除低质量样本 |
| PII 脱敏 | 99.5% | Presidio | 移除个人身份信息 |

最终保留数 = 原始数 × 0.94 × 0.91 × 0.995。300k 原始数据约保留 252k。

## 6. 污染检测原理

使用 MinHash LSH（局部敏感哈希）检测训练数据与评测基准的重叠：

- 检测基准：MMLU-Pro、MT-Bench-v2、RewardBench-2
- 状态判断：所有基准 overlap_examples == 0 → "clean"，否则 "dirty"
- 目的：防止数据泄露——训练时见过评测数据会导致评测结果失真

## 7. DPO vs GRPO 偏好对齐

| 方法 | 原理 | 适用场景 |
|------|------|----------|
| **DPO** | 直接用偏好对（chosen/rejected）优化策略，beta 控制强度 | 通用偏好对齐 |
| **GRPO** | 群组相对策略优化，用可验证奖励（如数学正确性） | 数学/代码等有明确正确答案的任务 |

DPO 无需训练奖励模型，比 RLHF 更简单稳定。GRPO 是 DeepSeek R1 的方案，适合可验证奖励的场景。

## 8. 量化与部署技术栈

**三种量化格式**对比：

| 格式 | 大小 | 特点 |
|------|------|------|
| GPTQ-INT4 | 4.6 GB | 训练后量化，GPU 优化（Marlin） |
| AWQ-INT4 | 4.8 GB | 激活感知，保护重要通道 |
| GGUF Q4_K_M | 5.1 GB | llama.cpp 格式，支持 CPU/GPU 混合推理 |

**部署架构**：vLLM 0.7 + EAGLE-3 推测解码。EAGLE-3 提前预测 N 个 token 的草稿头，目标模型验证接受，接受率 0.74 表示 74% 的推测被采纳。

## 9. 可复现性设计

整个流水线的可复现性体现在三个层面：

1. **种子控制**：`cfg["seed"]` 贯穿所有随机操作
2. **内容哈希**：每个产物的 SHA256 指纹确保可追溯——修改任何中间步骤都会导致下游哈希链变化
3. **一键重跑**：`./pipeline.sh config/llama3.3-8b-domainX.yaml` 即可端到端复现

模型卡（MOF 2026）要求包含训练配置哈希、数据许可证声明、评估结果和复现命令。
