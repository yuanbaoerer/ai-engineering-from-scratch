# 微调流水线运行环境与 uv 工具链

> 日期: 2026-06-25

## 1. 脚手架的本质:纯标准库的编排模拟

`07-end-to-end-fine-tuning-pipeline/code/main.py` 不是真实训练代码,而是一个**流水线编排骨架(scaffold)**:

- 只用 Python 标准库:`hashlib` / `json` / `time` / `dataclasses` / `typing`
- Axolotl、TRL、vLLM、GPTQ 等工具**仅出现在 docstring 的"实际生产使用"说明里**,代码并不 `import` 它们
- 每个阶段的输出(payload)是**预设的常量值**,不发生真实训练/量化/推理

它的价值在于演示真实生产流水线的**编排骨架**:声明式配置 + 基于内容哈希的产物追踪 + DAG 顺序编排。真正实战时,需把 `stage_sft` 等模拟函数替换成真正调用训练框架的实现(并补齐 GPU 与数据)。

## 2. 运行环境与依赖要求

| 项 | 要求 | 说明 |
|----|------|------|
| 第三方库 | **无** | 纯标准库,零额外依赖 |
| Python 版本 | ≥ 3.9 | 用了 `dict[str, Artifact]`、`list[tuple[str, Stage]]` 等内置泛型注解(Python 3.9+ 语法) |
| 项目脚本门槛 | 3.11+ | `env_setup.sh` 中 `PYTHON_MIN_MINOR=11` |
| 系统现状 | Python 3.12.3 | 满足所有要求 |

结论:这个脚手架用系统 `python3` 即可直接跑,不需要 venv、不需要装包。

## 3. 项目的 uv 现状

uv 已装(0.10.11),但**项目本身不是 uv 项目化结构**:

| 检查项 | 结果 |
|--------|------|
| `pyproject.toml` | ❌ 不存在(根目录和所有 phase 都没有) |
| `uv.lock` | ❌ 不存在 |
| `.python-version` | ❌ 不存在 |
| `requirements.txt` | ✅ 仅根目录有一份(课程通用依赖) |
| `.venv` | 运行 `env_setup.sh` 后生成 |

仓库是带 `requirements.txt` 的普通目录。`env_setup.sh` 虽然用 uv 执行,但走的是 `uv venv` + `uv pip install` 的**传统 pip 流程**,不是 `uv sync`。`requirements.txt` 里的 torch/transformers/datasets/anthropic 等尚未安装,脚本只装了 `numpy matplotlib jupyter scikit-learn pandas` 五个核心包。

## 4. uv 缓存复用机制:全局缓存 + 硬链接

uv 的核心卖点是**跨 venv 复用同一份库,零拷贝**。机制分两层:

```
            uv 全局缓存 (~/.cache/uv)
            ┌────────────────────────────┐
            │ wheels-v6/  下载的 wheel   │
            │ archive-v0/ 解包后的文件  ◄── 真正复用的源头
            └────────────────────────────┘
                        │ 硬链接(hardlink)
           ┌────────────┼────────────┐
           ▼            ▼            ▼
      .venv-A       .venv-B       .venv-C
```

**实测验证**(在机器上建两个独立 venv 各装 `numpy==2.4.6`):

```
venv-A:  inode=4695353   .../_multiarray_umath.cpython-311-x86_64-linux-gnu.so
venv-B:  inode=4695353   .../_multiarray_umath.cpython-311-x86_64-linux-gnu.so
✅ 相同 inode → 同一份物理文件被两个 venv 硬链接复用
```

- 相同 inode = 两个 venv 里的文件指向**磁盘上同一个物理文件**,不是拷贝
- 第二个 venv 装同一个库只用了 **30ms**(不联网、不解压,直接硬链接缓存)
- `du` 会把硬链接文件也计入,所以每个 venv 显示 31M,实际物理只占一份

**复用前提**:包版本 + Python 版本 + 平台三者完全一致才命中缓存。不同 Python 版本(3.11 vs 3.12)的同一库**不能复用**——wheel 是按 Python 版本编译的,缓存按 (包,版本,py版本,平台) 分桶。

## 5. link 模式与降级策略

uv 默认用硬链接(hardlink);文件系统不支持时自动降级:

```
hardlink(默认,最优) → reflink(CoW) → symlink → copy
```

- WSL2 的 ext4 默认支持 hardlink,无需配置
- 可用环境变量 `UV_LINK_MODE` 强制指定:`copy` / `symlink` / `hardlink` / `clonedlink`
- 一般不用动,默认即最优

## 6. 激活与使用 .venv

`env_setup.sh` 在仓库根目录生成 `.venv`(Python 3.11.15),含 numpy/matplotlib/sklearn/pandas/jupyter。

| 场景 | 命令 |
|------|------|
| 激活(交互终端) | `source /home/yuanbaoer/PyProjects/ai-engineering-from-scratch/.venv/bin/activate` |
| 退出 | `deactivate` |
| 不激活直接跑 | `.venv/bin/python main.py` |
| uv 跑(自动找 .venv) | `uv run python <path>/main.py` |
| fish/csh/PowerShell | 用对应的 `activate.fish` / `activate.csh` / `Activate.ps1` |

VSCode/Cursor 里:`Ctrl+Shift+P` → `Python: Select Interpreter` → 选 `.venv/bin/python`,之后运行/调试/Jupyter 全部走该环境。

验证激活:`which python` 应输出含 `.venv` 的路径。激活后提示符显示 `(.venv)`。

## 7. 运行输出四段解读

`python main.py` 的输出分四段:

| 段落 | 内容 | 关键读法 |
|------|------|----------|
| 执行日志 | 8 个阶段逐个 `running...` → 产物名 + 哈希 | 看阶段顺序与每个产物的指纹 |
| manifest 清单 | 全部产物:名称/类型/哈希/生产者 | 看依赖关系如何串起来 |
| eval report | 4 个维度的提升量(delta) | 相对基座的提升 |
| served endpoint | 部署后的性能与成本指标 | 服务化质量 |

**第一段每行格式**:`[阶段名(14字符)] -> artifact '名称' hash=指纹`。哈希是 SHA256 的前 12 位,由 payload 序列化为 JSON(sort_keys=True)后计算。

**清单里的 5 种 kind**:`dataset` / `checkpoint` / `quant` / `endpoint` / `report`。

## 8. 哈希依赖链在输出中的体现

哈希不是装饰,是**可追溯性的载体**——后续产物的 payload 里存着前序产物的哈希:

```
dataset ──ca747e11e566──► sft_checkpoint、contamination_check
sft_checkpoint ──fdd2cd36──► dpo_checkpoint、model_card
dpo_checkpoint ──ef90ec04──► quants、endpoint、eval_report
```

可直接验证:eval report 的 `from` 字段是 `ef90ec046fe3`,正好等于 `dpo_checkpoint` 的哈希——说明这次评测针对的是 DPO 之后的模型。修改任何中间产物的 payload,其哈希变化会传播到所有下游,从而被检测到。

## 9. eval 与 serve 指标解读

**eval report(delta = 相对基座的提升)**:

| 指标 | 值 | 含义 |
|------|----|------|
| `mmlu_pro_delta` | 3.2 | 知识推理 +3.2 个百分点 |
| `mt_bench_v2_delta` | 0.41 | 多轮对话 +0.41 分(满分 10) |
| `rewardbench2_delta` | 0.08 | 偏好对齐 +0.08 分 |
| `llama_guard_4_pass` | 0.987 | 安全通过率 98.7%(绝对值,非 delta) |

**served endpoint**:

| 指标 | 值 | 含义 |
|------|----|------|
| `eagle_acceptance` | 0.74 | 推测解码接受率 74%——草稿模型 token 被采纳比例 |
| `p99_bs8_ms` | 126 | batch=8 时 99 分位延迟 126ms(尾部 SLA) |
| `tokens_per_sec_bs32` | 6400 | batch=32 吞吐 6400 tokens/s |
| `dollars_per_mtokens` | 0.28 | 成本 $0.28/百万 token |

`eagle_acceptance: 0.74` 是快又省的关键:74% 的 token 由小草稿模型预测、被大模型批量验证通过,省掉大模型逐 token 自回归的大部分算力。

## 10. Datatrove:大规模 LLM 数据处理框架

Hugging Face 出品的**大规模数据处理框架**,解决"用有限内存处理 TB 级文本,做去重/质量过滤/清洗"。

在流水线里的分工(`stage_data` 生产者 = Datatrove+Nemotron-CC+Presidio):

| 工具 | 职责 |
|------|------|
| **Datatrove** | 去重(dedup) |
| Nemotron-CC | 质量过滤 |
| Presidio | PII 脱敏 |

`stage_contamination` 也复用 Datatrove 的 MinHashLSH 做污染检测——同一套算法既能去重也能检测训练数据与评测基准的重叠。

**架构:可组合的管道阶段**,每个阶段是可插拔的 step:

```
reader ──► extractor ──► tokenizer ──► filters ──► dedup ──► writer
读原始数据   从HTML提正文   统计token数    质量过滤   精确+     写盘
(parquet/   去标签         做长度过滤   语言过滤   MinHash   (parquet)
 jsonl)                                          去重
```

特点:**流式处理**(不全部载入内存,能扛 TB 级)、**分布式**(默认 local,可配 Slurm 集群)、**可复现**(配置可序列化,中间产物落盘)。

## 11. MinHash + LSH 原理

精确去重抓不住"几乎相同但有细微差异"的文档(转载加水印、改标点)。MinHash + LSH 解决模糊重复检测:

1. **MinHash**:对文档的 n-gram 集合做哈希取最小值作为"指纹"。相似文档 → 指纹大概率相同。把一个文档压缩成几十~几百个哈希值组成的签名。
2. **LSH(局部敏感哈希)**:把签名分桶,让"可能相似"的文档落进同一桶,只比较桶内文档 → 把 O(n²) 两两比对降到近线性。

**一鱼两吃**:MinHashLSH 既能去重(找训练集内部重复),也能做污染检测(把评测基准题目建索引,看训练数据是否命中)。微调场景数据量小,但数据卫生同样关键——数据不干净,后续 SFT/DPO 都是空中楼阁。
