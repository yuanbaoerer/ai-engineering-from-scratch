# LoRA 与 QLoRA 微调

> 全量微调一个 7B 模型需要 56GB VRAM。你没有。大多数公司也没有。LoRA 让你只需训练不到 1% 的参数就能在 6GB 内完成同模型的微调。这并非妥协——它在大多数任务上能达到全量微调的质量。整个开源微调生态都建立在这一技巧之上。

**类型：** 构建
**语言：** Python
**前置要求：** 第十阶段，第 6 课（指令微调 / SFT）
**时间：** 约 75 分钟
**相关说明：** 第十阶段从头讲解 SFT/DPO 循环。本课将这些技术集成到 2026 年的 PEFT 工具链中（PEFT、TRL、Unsloth、Axolotl、LLaMA-Factory）。

## 学习目标

- 通过向预训练模型的注意力层注入低秩适配矩阵（A 和 B）来实现 LoRA
- 计算 LoRA 与全量微调的参数量节省：rank r 与 d_model 维度训练 2*r*d 个参数，而非 d²
- 使用 QLoRA（4 位量化基础模型 + LoRA 适配器）在消费级 GPU 显存中微调模型
- 将 LoRA 权重合并回基础模型用于部署，并比较带适配器与不带适配器的推理速度

## 问题背景

你有一个基础模型。Llama 3 8B。你想让用它以公司风格回答客户支持工单。SFT 是解决方案。但 SFT 存在成本问题。

全量微调会更新模型中的每一个参数。Llama 3 8B 有 80 亿参数。在 fp16 中，每个参数占用 2 字节。仅加载权重就需要 16GB。训练期间，还需要梯度（16GB）、Adam 优化器状态（动量 + 方差共 32GB）以及激活值。总计：一个 8B 模型大约需要 56GB VRAM。

A100 80GB 勉强能装下。两个 A100 在云服务商处每小时费用为 3-4 美元。在 50,000 个样本上训练 3 个 epoch 需要 6-10 小时，费用为 30-40 美元一次实验。调试超参数需要跑 10 次实验，在部署任何东西之前就已经花费了 400 美元。

放大到 Llama 3 70B，数字变得荒谬。仅权重就需要 140GB。你需要一个集群。每次实验 100+ 美元。

还有一个更深层的问题。全量微调会修改模型中的每一个权重。如果你在客户支持数据上微调，可能会损害模型的通用能力。这叫做灾难性遗忘。模型在你的任务上变好了，但在其他所有方面变差了。

你需要一种方法，能用更少的参数、更少的显存，且不破坏模型现有知识。

## 核心概念

### LoRA：低秩适配

Edward Hu 和微软的同事于 2021 年 6 月发表了 LoRA。论文的洞察：微调期间的权重更新具有低内在秩。你不需要更新 4096×4096 权重矩阵中的全部 1670 万个参数。更新中的有用信息可以被秩为 16 或 32 的矩阵捕获。

数学原理如下。标准线性层计算：

```
y = Wx
```

其中 W 是 d_out × d_in 的矩阵。对于 4096×4096 的注意力投影，这包含 16,777,216 个参数。

LoRA 冻结 W 并添加低秩分解：

```
y = Wx + BAx
```

其中 B 是 (d_out × r)，A 是 (r × d_in)。秩 r 远小于 d——通常为 8、16 或 32。

对于 r=16 的 4096×4096 层：
- 原始参数：4096 × 4096 = 16,777,216
- LoRA 参数：(4096 × 16) + (16 × 4096) = 65,536 + 65,536 = 131,072
- 缩减比例：131,072 / 16,777,216 = 0.78%

你只需训练 0.78% 的参数，就能获得 95-100% 的质量。

```mermaid
graph LR
    X["输入 x"] --> W["冻结 W (d x d)"]
    X --> A["A (r x d)"]
    A --> B["B (d x r)"]
    W --> Plus["+ (合并)"]
    B --> Plus
    Plus --> Y["输出 y"]

    style W fill:#1a1a2e,stroke:#e94560,color:#fff
    style A fill:#0f3460,stroke:#16213e,color:#fff
    style B fill:#0f3460,stroke:#16213e,color:#fff
```

A 以随机高斯分布初始化。B 初始化为零。这意味着 LoRA 的贡献从零开始——模型从原始行为开始训练，逐步学习适应。

### 缩放因子：Alpha

LoRA 引入了一个缩放因子 alpha，用于控制低秩更新对输出影响的程度：

```
y = Wx + (alpha / r) * BAx
```

当 alpha = r 时，缩放为 1x。当 alpha = 2r（常见的默认值）时，缩放为 2x。这个超参数独立于基础学习率，控制 LoRA 路径的学习率。

实践建议：
- alpha = 2 * rank 是社区常见约定（原始论文大多数实验中使用 alpha = rank）
- alpha = rank 给出 1x 缩放，保守但稳定
- 更高的 alpha 意味着每步更大的更新，可以加速收敛或导致不稳定

### LoRA 应用位置

Transformer 有许多线性层。不需要向所有层添加 LoRA。原始论文测试了不同的组合：

| 目标层 | 可训练参数 (7B) | 质量 |
|--------|----------------|------|
| 仅 q_proj | 4.7M | 良好 |
| q_proj + v_proj | 9.4M | 更好 |
| q_proj + k_proj + v_proj + o_proj | 18.9M | 注意力任务最佳 |
| 所有线性层（注意力 + MLP） | 37.7M | 边际增益，参数翻倍 |

大多数任务的最佳位置：q_proj + v_proj。这针对的是自注意力中的查询和值投影，它们控制模型关注什么以及提取什么信息。对于代码生成等复杂任务，添加 MLP 层有帮助，但会使参数数量翻倍，对于简单任务回报递减。

### 秩的选择

秩 r 控制适应的表达能力：

| 秩 | 可训练参数（每层） | 最佳用途 |
|----|-------------------|----------|
| 4 | 32,768 | 简单分类、情感分析 |
| 8 | 65,536 | 单领域问答、摘要 |
| 16 | 131,072 | 多领域任务、指令遵循 |
| 32 | 262,144 | 复杂推理、代码生成 |
| 64 | 524,288 | 大多数任务回报递减 |
| 128 | 1,048,576 | 很少有充分理由 |

Hu 等人表明，对于简单任务，r=4 已经能捕获大部分适应。r=8 和 r=16 是实际应用中最常见的选择。超过 r=64 很少能提高质量，反而开始失去 LoRA 的显存优势。

### QLoRA：4 位量化 + LoRA

Tim Dettmers 和华盛顿大学的同事于 2023 年 5 月发表了 QLoRA。核心思想：将冻结的基础模型量化为 4 位精度，然后在上面以 fp16 附加 LoRA 适配器。

这显著改变了显存方程：

| 方法 | 权重显存 (7B) | 训练显存 (7B) | 所需 GPU |
|------|---------------|---------------|----------|
| 全量微调 (fp16) | 14GB | 约 56GB | 1x A100 80GB |
| LoRA (fp16 基础模型) | 14GB | 约 18GB | 1x A100 40GB |
| QLoRA (4 位基础模型) | 3.5GB | 约 6GB | 1x RTX 3090 24GB |

QLoRA 做出三个技术贡献：

**NF4（正态浮点 4 位）**：专为神经网络权重设计的新数据类型。神经网络权重大致服从正态分布。NF4 将其 16 个量化级别放置在标准正态分布的分位数上。对于正态分布数据，这在信息论上是最优的。相比均匀 4 位量化（INT4）或标准 Float4，它损失的信息更少。

**双重量化**：量化常数本身也占用显存。每 64 个权重块需要一个 fp32 缩放因子（4 字节）。对于 7B 模型，那是额外的 0.4GB。双重量化将这些常数量化为 fp8，将开销减少到 0.1GB。虽小但会累积。

**分页优化器**：训练期间，优化器状态（Adam 的动量和方差）可能在长序列上超过 GPU 显存。分页优化器使用 NVIDIA 的统一内存，当 GPU 显存耗尽时自动将优化器状态分页到 CPU RAM，并在需要时分页回来。这以一些吞吐量为代价防止 OOM 崩溃。

### 质量之争

减少参数或量化基础模型是否会影响质量？来自多篇论文的结果：

| 方法 | MMLU (5-shot) | MT-Bench | HumanEval |
|------|--------------|----------|-----------|
| 全量微调 (Llama 2 7B) | 48.3 | 6.72 | 14.6 |
| LoRA r=16 | 47.9 | 6.68 | 14.0 |
| QLoRA r=16 (NF4) | 47.5 | 6.61 | 13.4 |
| QLoRA r=64 (NF4) | 48.1 | 6.70 | 14.2 |

LoRA r=16 在大多数基准测试中与全量微调的差距在 1% 以内。QLoRA r=16 再损失一小部分。QLoRA r=64 实际上与全量微调持平，同时使用少 90% 的显存。

### 实际成本

在 50,000 个样本上微调 Llama 3 8B（3 个 epoch）：

| 方法 | GPU | 时间 | 成本 |
|------|-----|------|------|
| 全量微调 | 2x A100 80GB | 8 小时 | 约 $32 |
| LoRA r=16 | 1x A100 40GB | 4 小时 | 约 $8 |
| QLoRA r=16 | 1x RTX 4090 24GB | 6 小时 | 约 $5 |
| QLoRA r=16 (Unsloth) | 1x RTX 4090 24GB | 2.5 小时 | 约 $2 |
| QLoRA r=16 | 1x T4 16GB | 12 小时 | 约 $4 |

在单张消费级 GPU 上使用 QLoRA，成本低于一顿午餐。这就是为什么开源权重微调社区在 2023 年爆发，也是为什么下面的每个训练框架在 2026 年默认使用 QLoRA。

### 2026 PEFT 技术栈

| 框架 | 定位 | 选择场景 |
|------|------|----------|
| **Hugging Face PEFT** | 标准的 LoRA/QLoRA/DoRA/IA3 库 | 你需要完全控制，且你的训练循环已经在 `transformers.Trainer` 上 |
| **TRL** | HF 的强化学习训练器（SFT、DPO、GRPO、PPO、ORPO） | 你在 SFT 后需要 DPO/GRPO；构建在 PEFT 之上 |
| **Unsloth** | 前向/反向传播的 Triton 内核重写 | 你想要 2-5 倍加速 + 减半显存且无精度损失；Llama/Mistral/Qwen 系列 |
| **Axolotl** | PEFT + TRL + DeepSpeed + Unsloth 的 YAML 配置封装 | 你想要可复现的版本控制训练运行 |
| **LLaMA-Factory** | PEFT + TRL 之上的 GUI/CLI/API | 你想要零代码微调；支持 100+ 模型家族 |
| **torchtune** | 原生 PyTorch 方案，无 `transformers` 依赖 | 你想要最小依赖，且你的组织已标准化在 PyTorch 上 |

经验法则：研究用途或一次性实验 → PEFT。可复现的生产流水线 → 启用 Unsloth 内核的 Axolotl。一次性原型 → LLaMA-Factory。

### 合并适配器

训练后，你有两样东西：冻结的基础模型和一个小型 LoRA 适配器（通常 10-100MB）。你可以：

1. **保持分离**：加载基础模型，加载适配器在上面。为不同任务切换适配器。这就是如何从一个基础模型提供多个微调变体。

2. **永久合并**：计算 W' = W + (alpha/r) * BA 并将结果保存为新的完整模型。合并后的模型与原始模型大小相同。无推理开销。无需管理适配器。

对于提供多个任务（客户支持适配器、代码适配器、翻译适配器），保持分离。对于部署单个专用模型，合并。

高级合并技术用于组合多个适配器：

- **TIES-Merging**（Yadav 等，2023）：裁剪小幅度参数，解决符号冲突，然后合并。减少适配器之间的干扰。
- **DARE**（Yu 等，2023）：在合并前随机丢弃适配器参数，然后重新缩放剩余参数。令人惊讶地有效结合多种能力。
- **任务算术**：简单地对适配器权重进行加减。添加"代码"适配器和"数学"适配器通常会产生一个在两者都擅长的模型。

### 何时不微调

微调是第三选择，不是第一选择。

**第一：提示工程。** 写一个更好的系统提示。添加少样本示例。使用思维链。这不花钱，几分钟就能完成。如果提示能达到 80% 的效果，你可能不需要微调。

**第二：RAG。** 如果模型需要了解你的特定数据（文档、知识库、产品目录），检索比将其融入权重更便宜、更易维护。参见第 6 课。

**第三：微调。** 当你需要模型采用无法通过提示实现的特定风格、格式或推理模式时使用。当你需要一致的结构化输出时。当你需要将大模型蒸馏到小模型时。当延迟重要且你无法承受少样本提示的额外 token 时。

```mermaid
graph TD
    Start["需要更好的模型行为？"] --> PE["尝试提示工程"]
    PE -->|"有效"| Done["上线"]
    PE -->|"不够"| RAG["需要外部知识？"]
    RAG -->|"是"| RAGBuild["构建 RAG 流水线"]
    RAG -->|"否，需要风格/格式改变"| FT["使用 LoRA/QLoRA 微调"]
    RAGBuild -->|"有效"| Done
    RAGBuild -->|"也需要风格改变"| FT
    FT --> Done

    style Start fill:#1a1a2e,stroke:#e94560,color:#fff
    style Done fill:#0f3460,stroke:#16213e,color:#fff
```

## 构建它

我们从零开始用纯 PyTorch 实现 LoRA。不需要库。不需要魔法。你将构建 LoRA 层，将其注入模型，训练它，然后将权重合并回来。

### 步骤 1：LoRA 层

```python
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.A = nn.Parameter(torch.randn(in_features, rank) * (1 / math.sqrt(rank)))
        self.B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x):
        return (x @ self.A @ self.B) * self.scaling
```

A 以缩放的随机值初始化。B 初始化为零。乘积 BA 从零开始，所以模型从其原始行为开始。

### 步骤 2：带 LoRA 的线性层封装

```python
class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank=8, alpha=16):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.linear(x) + self.lora(x)
```

原始线性层被冻结。只有 LoRA 参数（A 和 B）可训练。

### 步骤 3：将 LoRA 注入模型

```python
def inject_lora(model, target_modules, rank=8, alpha=16):
    for param in model.parameters():
        param.requires_grad = False

    lora_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(t in name for t in target_modules):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = dict(model.named_modules())[parent_name]
                lora_linear = LinearWithLoRA(module, rank, alpha)
                setattr(parent, child_name, lora_linear)
                lora_layers[name] = lora_linear
    return lora_layers
```

首先，冻结模型中的每个参数。然后遍历模型树，找到与目标名称匹配的线性层，并用 LoRA 封装版本替换它们。LoRA A 和 B 矩阵是整个模型中唯一的可训练参数。

### 步骤 4：计数参数

```python
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": 100 * trainable / total if total > 0 else 0
    }
```

### 步骤 5：合并权重回来

```python
def merge_lora_weights(model):
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            with torch.no_grad():
                merged = (
                    module.lora.A @ module.lora.B
                ) * module.lora.scaling
                module.linear.weight.data += merged.T
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
            setattr(parent, child_name, module.linear)
```

合并后，LoRA 层消失。模型与原始大小相同，适应已融入权重。无推理开销。

### 步骤 6：模拟 QLoRA 量化

```python
def quantize_to_nf4(tensor, block_size=64):
    blocks = tensor.reshape(-1, block_size)
    scales = blocks.abs().max(dim=1, keepdim=True).values / 7.0
    scales = torch.clamp(scales, min=1e-8)
    quantized = torch.round(blocks / scales).clamp(-8, 7).to(torch.int8)
    return quantized, scales

def dequantize_from_nf4(quantized, scales, original_shape):
    dequantized = quantized.float() * scales
    return dequantized.reshape(original_shape)
```

这通过将权重映射到 64 个块的 16 个离散级别来模拟 4 位量化。生产级 QLoRA 使用 bitsandbytes 库在 GPU 上进行真正的 NF4 量化。

### 步骤 7：训练循环

```python
def train_lora(model, data, epochs=5, lr=1e-3, batch_size=4):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        indices = torch.randperm(len(data["inputs"]))

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            x = data["inputs"][batch_idx]
            y = data["targets"][batch_idx]

            output = model(x)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

    return losses
```

### 步骤 8：完整演示

```python
def demo():
    torch.manual_seed(42)
    d_model = 256
    n_classes = 10

    model = nn.Sequential(
        nn.Linear(d_model, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, n_classes),
    )

    n_samples = 500
    x = torch.randn(n_samples, d_model)
    y = torch.randint(0, n_classes, (n_samples,))
    y_onehot = torch.zeros(n_samples, n_classes).scatter_(1, y.unsqueeze(1), 1.0)

    data = {"inputs": x, "targets": y_onehot}

    params_before = count_parameters(model)

    lora_layers = inject_lora(
        model, target_modules=["0", "2"], rank=8, alpha=16
    )

    params_after = count_parameters(model)

    losses = train_lora(model, data, epochs=20, lr=1e-3)

    merge_lora_weights(model)
    params_merged = count_parameters(model)

    return {
        "params_before": params_before,
        "params_after": params_after,
        "params_merged": params_merged,
        "losses": losses,
    }
```

演示创建一个小模型，将 LoRA 注入两个层，训练它，然后将权重合并回来。参数量从完全可训练降至 LoRA 训练期间约 1% 可训练，然后在合并后恢复到原始架构。

## 使用它

使用 Hugging Face 生态系统，在真实模型上使用 LoRA 只需约 20 行代码：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

对于 QLoRA，添加 bitsandbytes 量化：

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=bnb_config,
    device_map="auto",
)

model = get_peft_model(model, lora_config)
```

就这样。相同的训练循环。相同的数据流水线。基础模型现在以 4 位存储，LoRA 适配器以 fp16 训练，整个系统可以装进 6GB。

使用 Hugging Face Trainer 进行训练：

```python
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

dataset = load_dataset("tatsu-lab/alpaca", split="train[:5000]")

training_args = TrainingArguments(
    output_dir="./lora-llama",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained("./lora-adapter")
```

保存的适配器为 10-100MB。基础模型保持不变。你可以在 Hugging Face Hub 上分享适配器，而无需重新分发完整模型。

## 上线它

本课产出：
- `outputs/prompt-lora-advisor.md` —— 帮助你决定 LoRA 秩、目标模块和超参数的提示
- `outputs/skill-fine-tuning-guide.md` —— 教智能体何时及如何微调的技能

## 练习

1. **秩消融研究。** 使用秩 2、4、8、16、32 和 64 运行演示。绘制最终损失 vs. 秩的图。找到回报递减的点——在该点 doubled rank 不再使损失减半。对于 256 维特征的简单分类任务，这应该在 r=8-16 左右。

2. **目标模块比较。** 修改 inject_lora 以仅针对层"0"、仅层"2"、仅层"4"以及全部三层。为每个变体训练 20 个 epoch。比较收敛速度和最终损失。这反映了真实决策：针对 q_proj vs v_proj vs 所有线性层。

3. **量化误差分析。** 取训练后模型在 quantize_to_nf4 / dequantize_from_nf4 前后的权重矩阵。计算均方误差、最大绝对误差以及原始权重与重构权重之间的相关性。使用 block_size 值 32、64、128 和 256 进行实验。

4. **多适配器服务。** 在数据的不同子集上训练两个 LoRA 适配器（偶数索引 vs 奇数索引）。保存两个适配器。加载基础模型一次，然后切换适配器并验证每个适配器对相同输入产生不同输出。这就是生产系统如何从一个基础模型提供多个微调模型。

5. **合并 vs. 未合并推理。** 比较同一 100 个输入上 merge_lora_weights 前后 LoRA 模型的输出。验证输出相同（在 1e-5 的浮点容差内）。然后对两者进行推理速度基准测试——合并后应该稍快，因为它是单一矩阵乘法而非两个。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|----------|
| LoRA | "高效微调" | 低秩适配：冻结基础权重，训练两个小矩阵 A 和 B，它们的乘积逼近完整权重更新 |
| QLoRA | "在笔记本上微调" | 量化 LoRA：以 4 位 NF4 加载基础模型，在其上以 fp16 训练 LoRA 适配器，使 7B 微调可以在 6GB VRAM 内完成 |
| 秩 (r) | "模型能学多少" | A 和 B 矩阵的内部维度；控制表达能力 vs. 参数量 |
| Alpha | "LoRA 学习率" | 应用于 LoRA 输出的缩放因子；alpha/r 缩放适应对最终输出的贡献 |
| NF4 | "4 位量化" | 正态浮点 4：一种 4 位数据类型，量化级别位于正态分布分位数上，对神经网络权重最优 |
| 适配器 | "小的训练部分" | LoRA A 和 B 矩阵保存为单独文件（10-100MB），可加载在任何基础模型副本之上 |
| 目标模块 | "哪些层用 LoRA" | 注入 LoRA 适配器的特定线性层（q_proj、v_proj 等） |
| 合并 | "融入权重" | 计算 W + (alpha/r) * BA 并替换原始权重，消除推理时的适配器开销 |
| 分页优化器 | "训练时不 OOM" | 当 GPU 显存耗尽时将优化器状态（Adam 动量、方差）卸载到 CPU |
| 灾难性遗忘 | "微调破坏了其他能力" | 更新所有权重导致模型失去先前学习的能力 |

## 扩展阅读

- Hu 等人，《LoRA：大型语言模型的低秩适配》（2021）—— 引入低秩分解方法的原始论文，在 GPT-3 175B 上以低至 4 的秩测试
- Dettmers 等人，《QLoRA：量化语言模型的高效微调》（2023）—— 引入 NF4、双重量化和分页优化器，使 65B 能在单张 48GB GPU 上微调
- PEFT 库文档（huggingface.co/docs/peft）—— Hugging Face 生态系统中 LoRA、QLoRA 和其他参数高效方法的标准库
- Yadav 等人，《TIES-Merging：合并模型时解决干扰》（2023）—— 在不损失质量的情况下组合多个 LoRA 适配器的技术
- [Rafailov 等人，《直接偏好优化：你的语言模型实际上是奖励模型》（NeurIPS 2023）](https://arxiv.org/abs/2305.18290) —— DPO 推导；SFT 之后的偏好调优阶段，无需奖励模型。
- [TRL 文档](https://huggingface.co/docs/trl/) —— `SFTTrainer`、`DPOTrainer`、`KTOTrainer` 以及与 PEFT/bitsandbytes/Unsloth 集成的官方参考。
- [Unsloth 文档](https://docs.unsloth.ai/) —— 融合内核使微调吞吐量翻倍、显存减半；TRL 下的性能层。
- [Axolotl 文档](https://axolotl-ai-cloud.github.io/axolotl/) —— YAML 配置的多 GPU SFT/DPO/QLoRA 训练器；配置即代码的替代方案。