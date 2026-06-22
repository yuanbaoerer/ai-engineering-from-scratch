# 毕业项目第15课 — 宪法安全Harness + 红队靶场

> Anthropic的Constitutional Classifiers、Meta的Llama Guard 4、Google的ShieldGemma-2、NVIDIA的Nemotron 3 Content Safety和X-Guard的多语言覆盖定义了2026年的安全分类器堆栈。garak、PyRIT、NVIDIA Aegis和promptfoo成为了标准对抗评估工具。NeMo Guardrails v0.12将它们整合到生产管道中。本毕业项目将所有内容连接在一起：围绕目标应用的分层安全harness、运行6+攻击家族的自主红队agent，以及产生可衡量无害性差异的宪法自我批评运行。

**类型：** 毕业项目
**语言：** Python（安全管道、红队）、YAML（策略配置）
**前置要求：** 阶段10（从头构建LLM）、阶段11（LLM工程）、阶段13（工具）、阶段14（agent）、阶段18（伦理、安全、对齐）
**练习阶段：** P10 · P11 · P13 · P14 · P18
**时间：** 25小时

## 问题

2026年LLM安全的前沿不是分类器是否有效（它们大体有效），而是如何围绕生产应用正确组合它们，而不会过度拒绝或留下明显漏洞。Llama Guard 4处理英语策略违规。X-Guard（132种语言）处理多语言越狱。ShieldGemma-2捕获基于图像的提示注入。NVIDIA Nemotron 3 Content Safety覆盖企业类别。Anthropic的Constitutional Classifiers是训练期间使用的独立方法，而不是服务期间。

攻击演变也很重要。PAIR和TAP自动化越狱发现。GCG运行基于梯度的后缀攻击。多轮和代码切换攻击利用agent内存。任何部署的LLM都需要红队靶场——garak和PyRIT是规范驱动器——加上文档化的缓解措施和CVSS评分的发现。

你将加固目标应用（8B指令调优模型或其他毕业项目中的RAG聊天机器人之一），对其运行6+攻击家族，并产生前/后无害性测量。

## 概念

安全管道是五层。**输入清理**：去除零宽字符、解码base64/rot13、规范化Unicode。**策略层**：NeMo Guardrails v0.12导轨（域外、毒性、PII提取）。**分类器门**：输入上的Llama Guard 4、非英语上的X-Guard、图像输入上的ShieldGemma-2。**模型**：目标LLM。**输出过滤器**：输出上的Llama Guard 4、Presidio PII擦除、适用时的引用强制。**HITL层**：标记为高风险的输出进入Slack队列。

红队靶场在调度器上运行。PAIR和TAP自主发现越狱。GCG运行基于梯度的后缀攻击。ASCII / base64 / rot13编码攻击。多轮攻击（角色扮演采用、内存利用）。代码切换攻击（将英语与斯瓦希里语或泰语混合）。每次运行生成带有CVSS评分和披露时间线的结构化发现文件。

宪法自我批评运行是训练时的干预。获取1k有害尝试提示，让模型起草回复，根据书面宪法（不伤害规则）批评它，并在批评循环上重新训练。在保留的评估上测量前/后无害性差异。

## 架构

```
request (text / image / multilingual)
      |
      v
input sanitize (strip zero-width, decode, normalize)
      |
      v
NeMo Guardrails v0.12 rails (off-domain, policy)
      |
      v
classifier gate:
  Llama Guard 4 (English)
  X-Guard (multilingual, 132 langs)
  ShieldGemma-2 (image prompts)
  Nemotron 3 Content Safety (enterprise)
      |
      v (allowed)
target LLM
      |
      v
output filter: Llama Guard 4 + Presidio PII + citation check
      |
      v
HITL tier for flagged outputs

parallel:
  red-team scheduler
    -> garak (classic attacks)
    -> PyRIT (orchestrated red team)
    -> autonomous jailbreak agent (PAIR + TAP)
    -> GCG suffix attacks
    -> multilingual / code-switch
    -> multi-turn persona adoption

output: CVSS-scored findings + disclosure timeline + before/after harmlessness delta
```

## 技术栈

- 安全分类器：Llama Guard 4、ShieldGemma-2、NVIDIA Nemotron 3 Content Safety、X-Guard
- 导轨框架：NeMo Guardrails v0.12 + OPA
- 红队驱动器：garak（NVIDIA）、PyRIT（Microsoft Azure）、NVIDIA Aegis、promptfoo
- 越狱agent：PAIR（Chao et al., 2023）、Tree-of-Attacks（TAP）、GCG后缀
- 宪法训练：Anthropic风格自我批评循环 + 批评上的SFT
- PII擦除：Presidio
- 目标：8B指令调优模型或其他毕业项目的RAG聊天机器人之一

## 构建它

1. **目标设置。**在vLLM上部署8B指令调优模型（或重用其他毕业项目的RAG聊天机器人）。这是被测应用。

2. **安全管道包装。**将五层管道连接到目标周围。验证每层都是可单独观察的（Langfuse中每层一个span）。

3. **分类器覆盖。**加载Llama Guard 4、X-Guard（多语言）、ShieldGemma-2（图像）。在小型标记集上运行每个以建立基线。

4. **红队调度器。**调度garak、PyRIT、PAIR agent、TAP agent、GCG运行器、多轮攻击者和代码切换攻击者。每个在单独的队列上运行。

5. **攻击套件。**六个攻击家族：(1) PAIR自动化越狱，(2) TAP树搜索攻击，(3) GCG梯度后缀，(4) ASCII / base64 / rot13编码，(5) 多轮角色扮演，(6) 多语言代码切换。报告每个家族的成功率。

6. **宪法自我批评。**策划1k有害尝试提示。对于每个提示，目标起草回复。批评LLM根据书面宪法评分（"不伤害"、"引用证据"、"拒绝非法请求"）。批评者反对的提示被重写；目标在批评改进的对上进行微调。在保留的评估上测量前/后无害性。

7. **过度拒绝测量。**跟踪良性提示集（例如XSTest）上的假阳性率。目标必须在良性问题上保持有用。

8. **CVSS评分。**对于每次成功的越狱，使用CVSS 4.0评分（攻击向量、复杂性、影响）。生成披露时间线和缓解计划。

9. **靶场自动化。**以上所有内容在cron上运行；发现写入队列；过度拒绝回归警报触发到Slack。

## 使用它

```
$ safety probe --model=target --family=PAIR --budget=50
[attacker]   PAIR agent running on target
[attack]     attempt 1/50: disguise query as academic research ... blocked
[attack]     attempt 2/50: appeal to roleplay ... blocked
[attack]     attempt 3/50: chain-of-thought coax ... SUCCEEDED
[finding]    CVSS 4.8 medium: roleplay bypass on target
[range]      7 successes out of 50 (14% success rate)
```

## 发布它

`outputs/skill-safety-harness.md`是交付物。生产级分层安全管道加上可重现的红队靶场，带有前/后无害性差异。

| 权重 | 标准 | 如何测量 |
|:----:|------|----------|
| 25 | 攻击面覆盖 | 6+攻击家族练习，2+语言 |
| 20 | 真阳性/假阳性权衡 | 攻击阻止率 vs XSTest良性通过率 |
| 20 | 自我批评差异 | 保留评估上的前/后无害性 |
| 20 | 文档和披露 | CVSS评分的发现和时间线 |
| 15 | 自动化和可重复性 | 一切在cron上运行并带有警报 |
| **100** | | |

## 练习

1. 运行garak的提示注入插件针对RAG聊天机器人，比较有和没有输出过滤器层的攻击成功率。

2. 添加第七个攻击家族：通过检索文档的间接提示注入。测量所需的额外防御。

3. 实现"拒绝并帮助"模式：当导轨阻止时，目标提供更安全的相关答案而不是简单拒绝。测量XSTest差异。

4. 多语言覆盖差距：找到X-Guard表现不佳的语言。提出针对该语言的微调数据集。

5. 在30B模型上运行宪法自我批评，测量差异是否按比例缩放。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 分层安全 | "纵深防御" | 输入、门、输出、HITL的多层导轨 |
| Llama Guard 4 | "Meta的安全分类器" | 2026年参考输入/输出内容分类器 |
| PAIR | "越狱agent" | 论文（Chao et al.）关于LLM驱动的越狱发现 |
| TAP | "树搜索攻击" | PAIR的树搜索变体 |
| GCG | "贪婪坐标梯度" | 基于梯度的对抗后缀攻击 |
| 宪法自我批评 | "Anthropic风格训练" | 目标起草 -> 批评者评分 -> 重写 -> 重新训练 |
| XSTest | "良性探测集" | 过度拒绝回归基准 |
| CVSS 4.0 | "严重性分数" | 安全发现的标准漏洞评分 |

## 进一步阅读

- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — 训练时参考
- [Meta Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026年输入/输出分类器
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b) — 图像 + 多模态安全
- [NVIDIA Nemotron 3 Content Safety](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/) — 企业参考
- [X-Guard (arXiv:2504.08848)](https://arxiv.org/abs/2504.08848) — 132语言多语言安全
- [garak](https://github.com/NVIDIA/garak) — NVIDIA红队工具包
- [PyRIT](https://github.com/Azure/PyRIT) — Microsoft红队框架
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 导轨框架
- [PAIR (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — 越狱agent论文