# 直接偏好优化家族

> Rafailov et al.（2023）表明RLHF的最优解可以用偏好数据的封闭形式表示，因此你可以跳过显式奖励模型直接优化策略。这一洞见催生了一个家族——IPO、KTO、SimPO、ORPO、BPO——每个都修复了DPO的一个失败模式。2026年，直接对齐算法（DAA）比PPO发布了更多的前沿后训练运行。但第2课的过优化曲线仍然适用：DAA没有逃离古德哈特，它们只是改变了它咬人的地方。

**类型：** 学习
**语言：** Python（标准库，六变体偏好损失比较器）
**前置条件：** 阶段18 · 01（InstructGPT），阶段18 · 02（奖励黑客），阶段10 · 08（DPO基础）
**时间：** ~75分钟

## 学习目标

- 从带KL的RLHF最优解推导DPO封闭形式。
- 陈述IPO、KTO、SimPO、ORPO、BPO修复了DPO的哪个失败模式。
- 区分"隐式奖励差距"和"偏好强度"，并解释为什么IPO的恒等映射很重要。
- 解释为什么Rafailov et al.（NeurIPS 2024）证明DAA尽管没有显式RM却过优化。

## 问题

RLHF目标（第1课）：

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

有一个已知的最优解：

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

因此奖励隐式地由最优策略与参考策略的比率定义：

```
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

将其代入Bradley-Terry偏好似然，配分函数`Z(x)`被消去，因为它只依赖于`x`。剩下的是仅在策略参数上的损失——不需要奖励模型。这就是DPO。

问题在于：推导假设最优解是可达的，偏好数据是分布内的，参考策略是真正的模式锚点。这些都不精确成立。每个家族成员修复了一个不同的违反假设。

## 概念

### DPO（Rafailov et al., 2023）

```
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

可能出错的地方：

- 隐式奖励差距`beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)`是无界的。微小的偏好可以产生任意大的差距。
- 损失驱动选择和拒绝的对数概率向相反方向移动。只要拒绝下降得更快，它可以推低选择的绝对对数概率。这就是退化选择响应现象。
- 分布外偏好（罕见稀有对 vs 罕见稀有对）产生任意的隐式奖励。

### IPO（Azar et al., 2024）

恒等偏好优化用恒等映射替换log-sigmoid。损失变成有界目标上的平方误差：

```
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

间隔被`1/(2 beta)`限制。偏好强度与隐式奖励差距成比例。没有爆炸。

### KTO（Ethayarajh et al., 2024）

Kahneman-Tversky优化完全放弃成对结构。给定单个标注输出和二元"可取"或"不可取"信号，它映射到前景理论效用：

```
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

对收益和损失有不同的权重（损失厌恶）。好处是：你可以使用非配对数据，这要丰富得多。

### SimPO（Meng et al., 2024）

简单偏好优化将训练信号与生成对齐。完全移除参考策略，用长度归一化对数似然：

```
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

带有间隔`gamma`以稳定。长度归一化消除了利用DPO长度偏差失败模式的动机（更长的`y_w`通过构造给出更大的对数概率差距）。

### ORPO（Hong et al., 2024）

比值比偏好优化在标准SFT负对数似然上添加偏好项：

```
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

没有参考策略——SFT项是正则化器。从基础模型到对齐模型单阶段训练。没有单独的SFT检查点。

### BPO（ICLR 2026投稿，OpenReview id=b97EwMUWu7）

识别了退化选择响应问题：DPO保持了`y_w > y_l`的排名，但`y_w`的绝对对数概率可能下降。BPO添加了一行修正，惩罚选择响应的向下移动。报告在Llama-3.1-8B-Instruct上数学推理比DPO高+10.1%准确率。

### 普遍结果：DAA仍然过优化

Rafailov et al. "Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms"（NeurIPS 2024）在多个KL预算的数据集上训练了DPO、IPO、SLiC的策略。真实奖励与KL曲线具有相同的Gao et al.峰值后崩溃形状。隐式奖励在训练期间查询分布外样本；KL正则化无法稳定这一点。

DAA没有逃离古德哈特。它们改变了它咬人的表面，从"奖励模型过优化"变为"参考策略比率过优化"。普遍的修复——更好的数据、集成、提前停止——对两者都适用。

### 如何选择（2026）

- 如果你有大量配对偏好数据：DPO使用保守beta，如果长度偏差明显则用SimPO。
- 如果你有非配对二元反馈：KTO。
- 如果你想要从基础模型的单阶段流水线：ORPO。
- 如果你在DPO日志中看到退化选择对数概率：BPO。
- 如果偏好强度变化很大且DPO饱和：IPO。

每个实验室都在测试套件上运行所有五种，并为每个任务选择获胜者。没有理由认为数学推理和安全的最优点是相同的。

```figure
dpo-margin
```

## 使用它

`code/main.py`在真实偏好强度随配对变化的玩具偏好数据集上比较六种损失（DPO、IPO、KTO、SimPO、ORPO、BPO）。每种损失针对相同的500对样本优化，使用小型softmax策略。绘制每种方法的最终胜率、选择对数概率漂移和隐式奖励分布。

## 交付它

本课产出`outputs/skill-preference-loss-selector.md`。给定数据集统计（配对 vs 非配对，可变 vs 统一偏好强度，长度分布）和目标（单阶段或SFT后偏好），推荐偏好损失并报告它防止的失败模式。

## 练习

1. 运行`code/main.py`。报告DPO和BPO的最终选择对数概率下降。BPO应保持更高的选择绝对概率——验证这一点。

2. 修改偏好数据使所有配对具有相等强度。六种方法中哪种最鲁棒？哪种退化？解释IPO在这里的优势。

3. 使拒绝响应平均比选择响应长2倍。不改变其他任何东西，数值展示DPO的长度利用和SimPO的修复。

4. Rafailov et al.（NeurIPS 2024）声称DAA过优化。重现单点版本：绘制选择减拒绝KL散度，观察DPO在大beta下的过优化。

5. 阅读BPO论文摘要（OpenReview b97EwMUWu7）。写下BPO添加到DPO的一行修正。与`code/main.py`中的实现确认。

## 关键术语

| 术语 | 人们怎么称呼它 | 它实际意味着什么 |
|------|-----------------|------------------------|
| DPO | "没有奖励模型的RLHF" | 从RLHF最优解封闭形式推导的损失；仅策略参数 |
| 隐式奖励 | "对数比率" | `beta * log(pi(y\|x) / pi_ref(y\|x))`——DPO隐含的奖励 |
| IPO | "有界DPO" | 用恒等映射替换log-sigmoid；隐式奖励差距被`1/(2 beta)`限制 |
| KTO | "非配对DPO" | 单标签上的前景理论效用，带有损失厌恶 |
| SimPO | "无参考DPO" | 长度归一化对数似然 + 间隔；无参考策略 |
| ORPO | "单阶段DPO" | NLL + 比值比偏好项；从基础模型一次性训练 |
| BPO | "保持选择的DPO" | DPO加上惩罚选择响应绝对对数概率下降的项 |
| 退化选择 | "选择下降" | DPO降低选择对数概率，只要拒绝下降更快 |
| DAA | "直接对齐算法" | 任何跳过显式RM的偏好损失方法 |

## 扩展阅读

- [Rafailov et al. — Direct Preference Optimization (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Azar et al. — A General Theoretical Paradigm to Understand Learning from Human Preferences (AISTATS 2024, arXiv:2310.12036)](https://arxiv.org/abs/2310.12036) — IPO
- [Ethayarajh et al. — KTO: Model Alignment as Prospect Theoretic Optimization (arXiv:2402.01306)](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen — SimPO (NeurIPS 2024, arXiv:2405.14734)](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne — ORPO (EMNLP 2024, arXiv:2403.07691)](https://arxiv.org/abs/2403.07691)
- [BPO — Behavior Preservation Optimization (ICLR 2026 OpenReview b97EwMUWu7)](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov et al. — Scaling Laws for RM Overoptimization in DAAs (NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900)