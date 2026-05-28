# 开发环境

> 工具体系塑造思维方式。一次配好，配到位。

**类型：** 构建
**语言：** Python、Node.js、Rust
**前置条件：** 无
**时长：** 约 45 分钟

## 学习目标

- 从零配置 Python 3.11+、Node.js 20+ 和 Rust 工具链
- 配置虚拟环境和包管理器，实现可复现构建
- 通过 CUDA/MPS 验证 GPU 访问权限，运行测试张量操作
- 理解四层技术栈：系统层、包管理、运行时、AI 库

## 问题背景

你即将通过 200+ 课时学习 AI 工程，使用 Python、TypeScript、Rust 和 Julia。如果环境配置有问题，每一课都会变成和工具斗争，而非学习本身。

大多数人会跳过环境配置这一步。然后花数小时调试 import 错误、版本冲突和缺失的 CUDA 驱动。我们要把这件事做一次、做正确。

## 核心概念

AI 工程环境有四层结构：

```mermaid
graph TD
    A["4. AI/ML 库\nPyTorch、JAX、transformers 等"] --> B["3. 语言运行时\nPython 3.11+、Node 20+、Rust、Julia"]
    B --> C["2. 包管理器\nuv、pnpm、cargo、juliaup"]
    C --> D["1. 系统基础\n操作系统、shell、git、编辑器、GPU 驱动"]
```

我们自底向上安装。每一层依赖它下面的一层。

## 从零构建

### 步骤 1：系统基础

检查系统并安装基础工具。

```bash
# macOS
xcode-select --install
brew install git curl wget

# Ubuntu/Debian
sudo apt update && sudo apt install -y build-essential git curl wget

# Windows（使用 WSL2）
wsl --install -d Ubuntu-24.04
```

### 步骤 2：Python（使用 uv）

我们使用 `uv`——比 pip 快 10-100 倍，且自动处理虚拟环境。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv python install 3.12

uv venv
source .venv/bin/activate  # Windows 上使用 .venv\Scripts\activate

uv pip install numpy matplotlib jupyter
```

验证：

```python
import sys
print(f"Python {sys.version}")

import numpy as np
print(f"NumPy {np.__version__}")
a = np.array([1, 2, 3])
print(f"向量: {a}, 与自身点积: {np.dot(a, a)}")
```

### 步骤 3：Node.js（使用 pnpm）

用于 TypeScript 相关课程（Agent、MCP 服务器、Web 应用）。

```bash
curl -fsSL https://fnm.vercel.app/install | bash
fnm install 22
fnm use 22

npm install -g pnpm

node -e "console.log('Node', process.version)"
```

### 步骤 4：Rust

用于性能敏感的课程（推理、系统级代码）。

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

rustc --version
cargo --version
```

### 步骤 5：Julia（可选）

用于数学密集型课程，Julia 在这些场景表现出色。

```bash
curl -fsSL https://install.julialang.org | sh

julia -e 'println("Julia ", VERSION)'
```

### 步骤 6：GPU 配置（如有）

```bash
# NVIDIA
nvidia-smi

# 安装支持 CUDA 的 PyTorch
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

```python
import torch
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

没有 GPU？没问题。大部分课程可以在 CPU 上运行。训练密集型课程可使用 Google Colab 或云端 GPU。

### 步骤 7：验证所有配置

运行验证脚本：

```bash
python phases/00-setup-and-tooling/01-dev-environment/code/verify.py
```

## 使用说明

你的环境现已准备好应对课程中的每一课。以下是各语言的使用场景：

| 语言 | 应用阶段 | 包管理器 |
| --- | --- | --- |
| Python | Phase 1-12（机器学习、深度学习、NLP、视觉、音频、LLM） | uv |
| TypeScript | Phase 13-17（工具协议、Agent、群体智能、基础设施） | pnpm |
| Rust | Phase 12、15-17（性能敏感系统） | cargo |
| Julia | Phase 1（数学基础） | Pkg |

## 产出

本课程产出一个验证脚本，任何人都可以运行它来检查环境配置。

参见 `outputs/prompt-env-check.md`，其中包含帮助 AI 助手诊断环境问题的提示词模板。

## 练习

1. 运行验证脚本并修复任何失败项
2. 为本课程创建一个 Python 虚拟环境并安装 PyTorch
3. 用四种语言各写一个 "hello world" 并分别运行