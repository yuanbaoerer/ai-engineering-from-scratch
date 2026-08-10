"""
Advanced Prompting Pipeline / 高级提示词管道
=============================================
Few-Shot + Chain-of-Thought + Self-Consistency + Tree-of-Thought + ReAct

同一个模型、同一任务、同一数据，从78%到91%的准确率差距，
不是更好的模型，而是更好的推理策略。

本模块实现了六种提示词工程技术：
1. Zero-shot             - 零样本提示词
2. Zero-shot CoT        - 零样本链式思维
3. Few-shot CoT         - 少量样本链式思维
4. Self-consistency     - 自洽性投票
5. Tree-of-Thought (ToT) - 树状思维
6. ReAct                - 推理+行动

参考文献：
- Wei et al. (2022) - Chain-of-Thought Prompting
- Wang et al. (2023) - Self-Consistency
- Yao et al. (2023) - Tree of Thoughts
- Yao et al. (2022) - ReAct
"""

import json
import re
import os
from collections import Counter
from openai import OpenAI


# =============================================================================
# 示例库 / Example Bank
# =============================================================================
# GSM8K_EXAMPLES: 少量样本CoT的示例库
# 每个示例包含：question（问题）、reasoning（推理链）、answer（答案）
# 推理链是将普通少量样本转换为CoT少量样本的关键
#
# GSM8K_EXAMPLES: example bank for few-shot CoT
# Each example contains: question, reasoning chain, and final answer
# The reasoning chain is what transforms a plain few-shot example into a CoT example

GSM8K_EXAMPLES = [
    {
        "question": (
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast "
            "every morning and bakes muffins for her friends every day with four. "
            "She sells every remaining egg at the farmers' market for $2. "
            "How much does she make every day at the farmers' market?"
        ),
        "reasoning": (
            "Janet's ducks lay 16 eggs per day. She eats 3 and bakes with 4, "
            "using 3 + 4 = 7 eggs. So she has 16 - 7 = 9 eggs left. "
            "She sells each for $2, so she makes 9 * 2 = $18 per day."
        ),
        "answer": "18",
    },
    {
        "question": (
            "A robe takes 2 bolts of blue fiber and half that much white fiber. "
            "How many bolts in total does it take?"
        ),
        "reasoning": (
            "It takes 2 bolts of blue fiber. "
            "Half of 2 is 1, so it takes 1 bolt of white fiber. "
            "In total, 2 + 1 = 3 bolts."
        ),
        "answer": "3",
    },
    {
        "question": (
            "Josh decides to try flipping a house. He buys a house for $80,000 "
            "and puts $50,000 in repairs. This increased the value of the house "
            "by 150%. How much profit did he make?"
        ),
        "reasoning": (
            "The house cost $80,000. Repairs cost $50,000. "
            "Total investment: 80,000 + 50,000 = $130,000. "
            "The value increased by 150% of $80,000: 80,000 * 1.5 = $120,000. "
            "New value: 80,000 + 120,000 = $200,000. "
            "Profit: 200,000 - 130,000 = $70,000."
        ),
        "answer": "70000",
    },
    {
        "question": (
            "James writes a 3-page letter to 2 different friends twice a week. "
            "How many pages does he write a year?"
        ),
        "reasoning": (
            "He writes to 2 friends, so 2 letters each time. "
            "Each letter is 3 pages, so 2 * 3 = 6 pages per session. "
            "He does this twice a week: 6 * 2 = 12 pages per week. "
            "In a year (52 weeks): 12 * 52 = 624 pages."
        ),
        "answer": "624",
    },
    {
        "question": (
            "Every day, Wendi feeds each of her chickens three cups of mixed "
            "chicken feed, containing seeds, mealworms, and vegetables. She gives "
            "the chickens their feed in three separate meals. In the morning, she "
            "gives her flock of chickens 15 cups of feed. In the afternoon, she "
            "gives her chickens another 25 cups of feed. How many cups of feed "
            "does she need to give her chickens in the final meal of the day if "
            "the carry-over from prior feedings was 35 cups?"
        ),
        "reasoning": (
            "Morning feed: 15 cups. Afternoon feed: 25 cups. "
            "Total so far: 15 + 25 = 40 cups. "
            "Each chicken gets 3 cups/day. Morning she gives 15 cups = 1/3 of total daily. "
            "Total daily = 15 * 3 = 45 cups. "
            "She gave 15 + 25 = 40 cups in first two meals. "
            "Last meal needs: 45 - 40 = 5 cups."
        ),
        "answer": "5",
    },
]


# =============================================================================
# 答案提取 / Answer Extraction
# =============================================================================
# extract_answer: 从模型输出中提取最终数值答案
#
# 模型输出格式不固定，可能包含：
# - "The answer is 18" / "答案是18"
# - "#### 18" (GSM8K标准格式)
# - "= 18" / "= $18"
#
# 本函数尝试多种正则模式，按优先级匹配
#
# extract_answer: Extract final numerical answer from model output
#
# Model output formats may vary. This function tries multiple regex patterns
# in order of priority to find the answer.

def extract_answer(text):
    """从LLM输出文本中提取数值答案 / Extract numerical answer from LLM output text."""
    if not text:
        return None

    # 优先级1: "The answer is $18" 或 "The answer is 18"
    # Priority 1: "The answer is $18" or "The answer is 18"
    patterns = [
        r"[Tt]he answer is[:\s]*\$?([\d,]+\.?\d*)",
        # 优先级2: "The answer is 18" (无美元符号)
        # Priority 2: "The answer is 18" (without dollar sign)
        r"[Tt]he answer is[:\s]*([\d,]+\.?\d*)",
        # 优先级3: "#### 18" (GSM8K数据集标准格式)
        # Priority 3: "#### 18" (GSM8K dataset standard format)
        r"#### ([\d,]+\.?\d*)",
        # 优先级4: "= 18" 或 "= $18" 在行尾
        # Priority 4: "= 18" or "= $18" at end of line
        r"= \$?([\d,]+\.?\d*)\s*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            # 移除数字中的逗号 / Remove commas from numbers
            return match.group(1).replace(",", "")

    # 备选方案：返回文本中最后一个数字
    # Fallback: return the last number found in text
    numbers = re.findall(r"[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


# =============================================================================
# 提示词构建器 / Prompt Builders
# =============================================================================
# 提示词构建器将系统消息、示例和目标问题组合成单个提示词
# Prompt builders combine system message, examples, and target question into a single prompt


def build_cot_prompt(question, examples, num_examples=3):
    """
    构建少量样本链式思维提示词 / Build few-shot Chain-of-Thought prompt

    Args:
        question: 待求解的问题 / The question to solve
        examples: 示例列表 / List of examples
        num_examples: 使用示例数量，默认3个 / Number of examples to use, default 3

    Returns:
        (system_message, user_message) 元组 / tuple
    """
    # 系统消息：角色定义 + 输出格式要求
    # System message: role definition + output format requirements
    system = (
        "You are a precise math problem solver. "
        "For each problem, show your step-by-step reasoning clearly. "
        "After your reasoning, state your final answer on the last line "
        "in exactly this format: 'The answer is [number]'."
    )

    # 构建示例文本：将每个示例格式化为 "Q: 问题\nA: 推理 答案。"
    # Build example text: format each as "Q: question\nA: reasoning The answer is X."
    example_text = ""
    for ex in examples[:num_examples]:
        example_text += f"Q: {ex['question']}\n"
        example_text += f"A: {ex['reasoning']} The answer is {ex['answer']}.\n\n"

    # 用户消息：示例 + 当前问题
    # User message: examples + current question
    user = f"{example_text}Q: {question}\nA:"
    return system, user


def build_zero_shot_cot_prompt(question):
    """
    构建零样本链式思维提示词 / Build zero-shot Chain-of-Thought prompt

    通过附加"让我们逐步思考"触发短语，无需示例即可启用推理
    Enables reasoning without examples by appending "let's think step by step" trigger

    Args:
        question: 待求解的问题 / The question to solve

    Returns:
        (system_message, user_message) 元组 / tuple
    """
    system = (
        "You are a precise math problem solver. "
        "Show your step-by-step reasoning. "
        "End with: 'The answer is [number]'."
    )
    user = f"Q: {question}\nA: Let's think step by step."
    return system, user


def build_zero_shot_prompt(question):
    """
    构建零样本提示词（无推理过程）/ Build zero-shot prompt (no reasoning)

    仅返回最终数值答案，不展示推理步骤
    Returns only the final numerical answer without showing reasoning steps

    Args:
        question: 待求解的问题 / The question to solve

    Returns:
        (system_message, user_message) 元组 / tuple
    """
    system = (
        "You are a precise math problem solver. "
        "Give only the final numerical answer. "
        "End with: 'The answer is [number]'."
    )
    user = f"Q: {question}\nA:"
    return system, user


# =============================================================================
# LLM调用封装 / LLM Call Wrapper
# =============================================================================


def call_llm(client, model, system, user, temperature=0.0):
    """
    通用LLM调用函数 / Generic LLM call function

    Args:
        client: OpenAI客户端实例 / OpenAI client instance
        model: 模型名称（如"gpt-4o"）/ Model name (e.g., "gpt-4o")
        system: 系统消息 / System message
        user: 用户消息 / User message
        temperature: 采样温度，0.0=确定性输出 / Sampling temperature, 0.0=deterministic

    Returns:
        LLM输出的文本 / Text of LLM output
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# =============================================================================
# 各技术求解器 / Technique Solvers
# =============================================================================


def zero_shot_solve(question, client, model):
    """
    零样本求解 / Zero-shot solve

    直接给出问题，不提供任何示例
    Directly pose the question without any examples

    Returns:
        (answer, full_response) 元组 / tuple
    """
    system, user = build_zero_shot_prompt(question)
    text = call_llm(client, model, system, user, temperature=0.0)
    return extract_answer(text), text


def zero_shot_cot_solve(question, client, model):
    """
    零样本链式思维求解 / Zero-shot Chain-of-Thought solve

    通过"让我们逐步思考"触发模型的推理能力
    Trigger model's reasoning ability via "let's think step by step"

    Returns:
        (answer, full_response) 元组 / tuple
    """
    system, user = build_zero_shot_cot_prompt(question)
    text = call_llm(client, model, system, user, temperature=0.0)
    return extract_answer(text), text


def few_shot_cot_solve(question, examples, client, model, num_examples=3):
    """
    少量样本链式思维求解 / Few-shot Chain-of-Thought solve

    提供3-5个带有推理链的示例，比零样本CoT更有效
    Provides 3-5 examples with reasoning chains, more effective than zero-shot CoT

    Args:
        question: 待求解问题 / Question to solve
        examples: 示例库 / Example bank
        client: OpenAI客户端 / OpenAI client
        model: 模型名称 / Model name
        num_examples: 示例数量，默认3 / Number of examples, default 3

    Returns:
        (answer, full_response) 元组 / tuple
    """
    system, user = build_cot_prompt(question, examples, num_examples)
    text = call_llm(client, model, system, user, temperature=0.0)
    return extract_answer(text), text


# =============================================================================
# 自洽性投票 / Self-Consistency Voting
# =============================================================================
# Wang等人（2023）提出：在temperature > 0时采样N条推理路径，通过多数票选择答案
# Wang et al. (2023): Sample N reasoning paths at temperature > 0, vote via majority


def self_consistency_solve(question, examples, client, model, n_samples=5):
    """
    自洽性求解 / Self-consistency solve

    采样N条独立推理路径，对最终答案取多数票
    Sample N independent reasoning paths and take majority vote on final answer

    关键 insight：
    - 单条CoT路径可能包含推理错误
    - 采样多条路径时，错误会相互抵消
    - Temperature必须 > 0 以获得多样性路径

    Key insight:
    - Single CoT path may contain reasoning errors
    - Errors cancel out when sampling multiple paths
    - Temperature must be > 0 for diverse paths

    Args:
        question: 待求解问题 / Question to solve
        examples: 示例库 / Example bank
        client: OpenAI客户端 / OpenAI client
        model: 模型名称 / Model name
        n_samples: 采样数量，默认5 / Number of samples, default 5

    Returns:
        (best_answer, confidence, reasonings, vote_counts) 元组 / tuple
    """
    system, user = build_cot_prompt(question, examples)

    answers = []
    reasonings = []

    # Temperature 0.7：在随机性和确定性之间取得平衡
    # Temperature 0.7: balance between randomness and determinism
    for _ in range(n_samples):
        text = call_llm(client, model, system, user, temperature=0.7)
        reasonings.append(text)
        answer = extract_answer(text)
        if answer is not None:
            answers.append(answer)

    if not answers:
        return None, 0.0, reasonings, Counter()

    # 统计投票：计算每个答案出现的次数
    # Vote counting: count occurrences of each answer
    vote_counts = Counter(answers)
    best_answer = vote_counts.most_common(1)[0][0]
    confidence = vote_counts[best_answer] / len(answers)

    return best_answer, confidence, reasonings, vote_counts


# =============================================================================
# 树状思维 / Tree-of-Thought (ToT)
# =============================================================================
# Yao等人（2023）提出：探索多个推理分支，评估并剪枝
# Yao et al. (2023): Explore multiple reasoning branches, evaluate and prune


def generate_initial_thoughts(question, client, model, breadth=3):
    """
    生成初始思考列表 / Generate initial list of thoughts

    ToT第一步：从问题出发，生成多个不同的解决思路
    ToT Step 1: From the problem, generate multiple different solution approaches

    Args:
        question: 待求解问题 / Question to solve
        client: OpenAI客户端 / OpenAI client
        model: 模型名称 / Model name
        breadth: 生成思路数量 / Number of thoughts to generate

    Returns:
        思路列表 / List of thoughts
    """
    system = (
        "You are a math problem solver exploring different solution approaches. "
        "Generate one distinct approach to solving this problem. "
        "Show your partial reasoning. Do not give the final answer yet."
    )
    thoughts = []

    # Temperature 0.9：高随机性，鼓励不同思路
    # Temperature 0.9: high randomness, encourages diverse approaches
    for i in range(breadth):
        user = (
            f"Problem: {question}\n\n"
            f"Generate approach #{i + 1} (use a different strategy than previous approaches). "
            f"Think about: arithmetic breakdown, working backwards, estimation, "
            f"or algebraic formulation."
        )
        text = call_llm(client, model, system, user, temperature=0.9)
        thoughts.append(text)

    return thoughts


def evaluate_thought(thought, question, client, model):
    """
    评估思考质量 / Evaluate thought quality

    ToT第二步：用LLM作为评估器，对每个思路打分（0.0-1.0）
    ToT Step 2: Use LLM as evaluator to score each thought (0.0-1.0)

    评估维度：
    - 算术正确性 / Arithmetic correctness
    - 逻辑连贯性 / Logical coherence
    - 离答案的进度 / Progress toward answer

    Evaluation dimensions:
    - Arithmetic correctness
    - Logical coherence
    - Progress toward answer

    Args:
        thought: 待评估的思考文本 / Thought text to evaluate
        question: 原始问题 / Original question
        client: OpenAI客户端 / OpenAI client
        model: 模型名称 / Model name

    Returns:
        0.0-1.0之间的分数 / Score between 0.0 and 1.0
    """
    system = (
        "You are a math reasoning evaluator. "
        "Score the following partial reasoning on a scale from 0.0 to 1.0. "
        "Consider: correctness of arithmetic, logical coherence, "
        "progress toward the answer. "
        "Respond with ONLY a number between 0.0 and 1.0."
    )
    user = f"Problem: {question}\n\nReasoning so far:\n{thought}\n\nScore:"
    text = call_llm(client, model, system, user, temperature=0.0)

    try:
        score = float(re.search(r"([\d.]+)", text).group(1))
        return min(max(score, 0.0), 1.0)  # 限制在[0, 1]范围内 / Clamp to [0, 1]
    except (AttributeError, ValueError):
        return 0.5  # 解析失败时返回中性分数 / Return neutral score on parse failure


def extend_thought(thought, question, client, model, breadth=2):
    """
    扩展思考 / Extend thought

    ToT第三步：沿同一思路继续推理，生成多个可能的延续
    ToT Step 3: Continue reasoning along the same thought, generate multiple possible continuations

    Args:
        thought: 当前思考 / Current thought
        question: 原始问题 / Original question
        client: OpenAI客户端 / OpenAI client
        model: 模型名称 / Model name
        breadth: 每个思路扩展数量 / Number of extensions per thought

    Returns:
        扩展后的思考列表 / List of extended thoughts
    """
    system = (
        "You are a math problem solver continuing a line of reasoning. "
        "Take the partial reasoning below and extend it further toward a solution. "
        "Show your continued reasoning. If you reach the final answer, "
        "state it as: 'The answer is [number]'."
    )
    extensions = []

    # Temperature 0.8：中等随机性，平衡创造性和连贯性
    # Temperature 0.8: medium randomness, balance creativity and coherence
    for i in range(breadth):
        user = (
            f"Problem: {question}\n\n"
            f"Reasoning so far:\n{thought}\n\n"
            f"Continue this reasoning (approach #{i + 1}):"
        )
        text = call_llm(client, model, system, user, temperature=0.8)
        extensions.append(f"{thought}\n\n{text}")

    return extensions


def tree_of_thought_solve(question, client, model, breadth=3, depth=3):
    """
    树状思维求解 / Tree-of-Thought solve

    完整ToT流程：
    1. 生成breadth个初始思路 / Generate breadth initial thoughts
    2. 评估每个思路 / Evaluate each thought
    3. 选择top-k个思路继续 / Select top-k thoughts to continue
    4. 重复直到depth层 / Repeat until depth layers
    5. 返回最佳思路的答案 / Return answer from best thought

    Complete ToT flow:
    1. Generate breadth initial thoughts
    2. Evaluate each thought
    3. Select top-k thoughts to continue
    4. Repeat until depth layers
    5. Return answer from best thought

    Args:
        question: 待求解问题 / Question to solve
        client: OpenAI客户端 / OpenAI client
        model: 模型名称 / Model name
        breadth: 每层分支数 / Branching factor per layer
        depth: 树深度 / Tree depth

    Returns:
        (answer, best_reasoning) 元组 / tuple
    """
    # 第一层：生成初始思路并评估 / Layer 1: generate and evaluate initial thoughts
    thoughts = generate_initial_thoughts(question, client, model, breadth)
    scored = [(t, evaluate_thought(t, question, client, model)) for t in thoughts]
    scored.sort(key=lambda x: x[1], reverse=True)  # 按分数降序排序 / Sort by score descending

    # 后续层：扩展高分思路 / Subsequent layers: extend high-scoring thoughts
    for current_depth in range(1, depth):
        next_thoughts = []

        # 只扩展前top_k个思路 / Only extend top_k thoughts
        top_k = min(2, len(scored))
        for thought, score in scored[:top_k]:
            extensions = extend_thought(thought, question, client, model, breadth)
            for ext in extensions:
                ext_score = evaluate_thought(ext, question, client, model)
                next_thoughts.append((ext, ext_score))

        if next_thoughts:
            scored = sorted(next_thoughts, key=lambda x: x[1], reverse=True)

    best_thought = scored[0][0] if scored else ""
    return extract_answer(best_thought), best_thought


# =============================================================================
# ReAct：推理+行动 / ReAct: Reasoning + Acting
# =============================================================================
# Yao等人（2022）提出：在思考-行动-观察循环中交错推理与工具使用
# Yao et al. (2022): Interleave reasoning with tool use in Think-Act-Observe loop


def react_solve(question, client, model, max_steps=5):
    """
    ReAct求解 / ReAct solve

    模型在"思考"、"行动（计算）"、"观察"之间交替，直到得出答案
    Model alternates between "Thought", "Action (calculate)", and "Observation"
    until reaching an answer

    循环：
    1. Thought: 生成推理步骤 / Generate reasoning step
    2. Action: 如需计算，输出"Action: calculate [表达式]" / If calculation needed, output "Action: calculate [expression]"
    3. Observation: 返回计算结果 / Return calculation result
    4. 重复直到Answer: / Repeat until Answer:

    Loop:
    1. Thought: generate reasoning step
    2. Action: if calculation needed, output "Action: calculate [expression]"
    3. Observation: return calculation result
    4. Repeat until Answer:

    Args:
        question: 待求解问题 / Question to solve
        client: OpenAI客户端 / OpenAI client
        model: 模型名称 / Model name
        max_steps: 最大循环步数 / Maximum loop steps

    Returns:
        (answer, full_trace) 元组 / tuple
    """
    system = (
        "You are a math problem solver that can use a calculator. "
        "For each step, output exactly one of:\n"
        "Thought: [your reasoning]\n"
        "Action: calculate [expression]\n"
        "Answer: [final number]\n\n"
        "When you need to compute something, use Action: calculate. "
        "You will receive the result as an Observation. "
        "When you have the final answer, use Answer:."
    )

    conversation = f"Q: {question}\n"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": conversation},
    ]

    for step in range(max_steps):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=512,
        )
        text = response.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": text})

        # 检查是否已有答案 / Check if answer is already provided
        answer_match = re.search(r"Answer:\s*\$?([\d,]+\.?\d*)", text)
        if answer_match:
            return answer_match.group(1).replace(",", ""), text

        # 检查是否需要计算 / Check if calculation is needed
        calc_match = re.search(r"Action:\s*calculate\s+(.+)", text)
        if calc_match:
            expression = calc_match.group(1).strip()
            try:
                # 在沙箱中执行数学表达式 / Execute math expression in sandbox
                # 只允许基本数学运算 / Only allow basic math operations
                result = eval(expression, {"__builtins__": {}}, {})
                observation = f"Observation: {result}"
            except Exception as e:
                observation = f"Observation: Error - {e}"
            messages.append({"role": "user", "content": observation})

    # 超过最大步数，返回提取的答案 / Max steps exceeded, return extracted answer
    full_text = "\n".join(
        m["content"] for m in messages if m["role"] == "assistant"
    )
    return extract_answer(full_text), full_text


# =============================================================================
# 升级策略管道 / Escalation Pipeline
# =============================================================================


def solve_with_escalation(question, examples, client, model):
    """
    升级策略求解 / Escalation solve

    组合多种技术，自动选择最优方法：
    1. 先用Few-shot CoT单次求解（便宜）
    2. 如果置信度低于0.8，用Self-consistency
    3. 如果置信度仍然低，用Tree-of-Thought（最贵）

    Combine multiple techniques, automatically select best:
    1. First try Few-shot CoT single call (cheap)
    2. If confidence < 0.8, try Self-consistency
    3. If still low confidence, try Tree-of-Thought (most expensive)

    Args:
        question: 待求解问题 / Question to solve
        examples: 示例库 / Example bank
        client: OpenAI客户端 / OpenAI client
        model: 模型名称 / Model name

    Returns:
        包含答案、方法、置信度的字典 / Dict with answer, method, confidence
    """
    # 步骤1：单次Few-shot CoT / Step 1: Single Few-shot CoT
    single_answer, single_text = few_shot_cot_solve(
        question, examples, client, model
    )

    # 步骤2：Self-consistency投票 / Step 2: Self-consistency voting
    sc_answer, confidence, reasonings, votes = self_consistency_solve(
        question, examples, client, model, n_samples=5
    )

    # 如果置信度 >= 0.8，返回自洽性结果 / If confidence >= 0.8, return self-consistency result
    if confidence >= 0.8:
        return {
            "answer": sc_answer,
            "method": "self_consistency",
            "confidence": confidence,
            "votes": dict(votes),
            "reasoning": reasonings[0],
        }

    # 步骤3：升级到Tree-of-Thought / Step 3: Escalate to Tree-of-Thought
    tot_answer, tot_reasoning = tree_of_thought_solve(
        question, client, model, breadth=3, depth=2
    )

    return {
        "answer": tot_answer,
        "method": "tree_of_thought",
        "confidence": None,
        "votes": dict(votes),
        "reasoning": tot_reasoning,
    }


# =============================================================================
# 技术对比 / Technique Comparison
# =============================================================================


def run_comparison(questions, expected_answers, examples, client, model):
    """
    运行技术对比实验 / Run technique comparison experiment

    在同一组问题上比较所有技术的准确率
    Compare accuracy of all techniques on the same set of problems

    Args:
        questions: 问题列表 / List of questions
        expected_answers: 预期答案列表 / List of expected answers
        examples: 示例库 / Example bank
        client: OpenAI客户端 / OpenAI client
        model: 模型名称 / Model name

    Returns:
        各技术的准确率结果 / Accuracy results for each technique
    """
    methods = {
        "zero_shot": lambda q: zero_shot_solve(q, client, model),
        "zero_shot_cot": lambda q: zero_shot_cot_solve(q, client, model),
        "few_shot_cot": lambda q: few_shot_cot_solve(q, examples, client, model),
        "self_consistency": lambda q: (
            self_consistency_solve(q, examples, client, model, n_samples=5)[:2]
        ),
    }

    results = {name: {"correct": 0, "total": 0} for name in methods}

    for i, (question, expected) in enumerate(zip(questions, expected_answers)):
        print(f"\nProblem {i + 1}: {question[:60]}...")
        for name, solver in methods.items():
            answer, *_ = solver(question)
            is_correct = str(answer) == str(expected)
            results[name]["total"] += 1
            if is_correct:
                results[name]["correct"] += 1
            status = "CORRECT" if is_correct else f"WRONG (got {answer}, expected {expected})"
            print(f"  {name:20s}: {status}")

    print("\n" + "=" * 50)
    print("ACCURACY SUMMARY")
    print("=" * 50)
    for name, counts in results.items():
        acc = counts["correct"] / counts["total"] * 100 if counts["total"] > 0 else 0
        print(f"  {name:20s}: {acc:.1f}% ({counts['correct']}/{counts['total']})")

    return results


# =============================================================================
# 结构化提示词 / Structured Prompts
# =============================================================================


def build_structured_prompt(question, context=None):
    """
    构建结构化提示词（使用XML标签）/ Build structured prompt (using XML tags)

    使用XML标签组织提示词结构，防止模型混淆各部分
    使用XML tags to organize prompt structure, preventing model confusion

    Args:
        question: 问题 / Question
        context: 可选上下文信息 / Optional context information

    Returns:
        (system, user) 元组 / tuple
    """
    system = """<role>
You are a precise mathematical problem solver with expertise in word problems.
</role>

<rules>
- Show all arithmetic steps explicitly
- Use one line per calculation
- State units where applicable
- End with exactly: 'The answer is [number]'
- If the problem is ambiguous, state your interpretation before solving
</rules>

<output_format>
## Interpretation
[One sentence restating the problem]

## Solution
[Step-by-step calculations]

## Answer
The answer is [number].
</output_format>"""

    user_parts = []
    if context:
        user_parts.append(f"<context>\n{context}\n</context>")
    user_parts.append(f"<problem>\n{question}\n</problem>")

    return system, "\n\n".join(user_parts)


# =============================================================================
# 提示链 / Prompt Chaining
# =============================================================================


def prompt_chain_solve(question, client, model):
    """
    提示链求解 / Prompt chain solve

    将复杂任务分解为多个步骤，每个提示的输出成为下一个的输入：
    1. 提取关键数值和关系 / Extract key values and relationships
    2. 设置并求解方程 / Set up and solve equations
    3. 验证答案 / Verify answer

    Break complex tasks into multiple steps, each prompt's output becomes next's input:
    1. Extract key values and relationships
    2. Set up and solve equations
    3. Verify answer

    Args:
        question: 问题 / Question
        client: OpenAI客户端 / OpenAI client
        model: 模型名称 / Model name

    Returns:
        (answer, chain_info) 元组 / tuple
    """
    # 步骤1：提取关键事实 / Step 1: Extract key facts
    extract_system = (
        "Extract the key numerical values and relationships from this math problem. "
        "List each as: [variable]: [value] [unit]. "
        "Then list each relationship as: [description]."
    )
    facts = call_llm(client, model, extract_system, question, temperature=0.0)

    # 步骤2：基于提取的事实求解 / Step 2: Solve based on extracted facts
    solve_system = (
        "You are a math solver. Given the extracted facts below, "
        "set up and solve the equations step by step. "
        "End with: 'The answer is [number]'."
    )
    solve_user = f"Facts:\n{facts}\n\nOriginal problem: {question}"
    solution = call_llm(client, model, solve_system, solve_user, temperature=0.0)

    # 步骤3：验证答案 / Step 3: Verify answer
    verify_system = (
        "Verify this math solution by plugging the answer back into "
        "the original problem. Does it check out? "
        "If yes, restate: 'The answer is [number]'. "
        "If no, solve it correctly and state: 'The answer is [number]'."
    )
    verify_user = f"Problem: {question}\n\nProposed solution:\n{solution}"
    verified = call_llm(client, model, verify_system, verify_user, temperature=0.0)

    return extract_answer(verified), {
        "facts": facts,
        "solution": solution,
        "verification": verified,
    }


# =============================================================================
# 测试问题 / Test Questions
# =============================================================================
# 用于演示和测试的数学问题集 / Math problem set for demo and testing

TEST_QUESTIONS = [
    {
        "question": (
            "Natalia sold clips to 48 of her friends in April, "
            "and then she sold half as many clips in May. "
            "How many clips did Natalia sell altogether in April and May?"
        ),
        "answer": "72",
    },
    {
        "question": (
            "Weng earns $12 an hour for babysitting. Yesterday, she just "
            "did 50 minutes of babysitting. How much did she earn?"
        ),
        "answer": "10",
    },
    {
        "question": (
            "Betty is saving money for a new wallet which costs $100. "
            "Betty has only half of the money she needs. Her parents decided "
            "to give her $15 for that purpose, and her grandparents twice as "
            "much as her parents. How much more money does Betty need to buy "
            "the wallet?"
        ),
        "answer": "5",
    },
    {
        "question": (
            "Julie is reading a 120-page book. Yesterday, she was able to "
            "read 12 pages and today, she read twice as many pages as yesterday. "
            "If she wants to read half of the remaining pages tomorrow, "
            "how many pages should she read?"
        ),
        "answer": "42",
    },
    {
        "question": (
            "James writes a 3-page letter to 2 different friends twice a week. "
            "How many pages does he write a year?"
        ),
        "answer": "624",
    },
]


# =============================================================================
# 主函数 / Main Function
# =============================================================================


if __name__ == "__main__":
    # 初始化OpenAI客户端 / Initialize OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "your-api-key"))
    model = os.environ.get("LLM_MODEL", "gpt-4o")

    print("=" * 60)
    print("ADVANCED PROMPTING PIPELINE")
    print("Few-Shot + CoT + Self-Consistency + Tree-of-Thought")
    print("=" * 60)

    questions = [t["question"] for t in TEST_QUESTIONS]
    expected = [t["answer"] for t in TEST_QUESTIONS]

    # 技术对比 / Technique Comparison
    print("\n--- Technique Comparison ---")
    run_comparison(questions, expected, GSM8K_EXAMPLES, client, model)

    # 升级管道 / Escalation Pipeline
    print("\n\n--- Escalation Pipeline ---")
    for test in TEST_QUESTIONS[:2]:
        print(f"\nQ: {test['question'][:80]}...")
        result = solve_with_escalation(
            test["question"], GSM8K_EXAMPLES, client, model
        )
        print(f"  Method: {result['method']}")
        print(f"  Answer: {result['answer']} (expected: {test['answer']})")
        print(f"  Confidence: {result['confidence']}")

    # 提示链 / Prompt Chaining
    print("\n\n--- Prompt Chaining ---")
    for test in TEST_QUESTIONS[:2]:
        print(f"\nQ: {test['question'][:80]}...")
        answer, chain = prompt_chain_solve(test["question"], client, model)
        print(f"  Answer: {answer} (expected: {test['answer']})")
        print(f"  Steps: extract -> solve -> verify")

    # ReAct / ReAct
    print("\n\n--- ReAct ---")
    for test in TEST_QUESTIONS[:2]:
        print(f"\nQ: {test['question'][:80]}...")
        answer, trace = react_solve(test["question"], client, model)
        print(f"  Answer: {answer} (expected: {test['answer']})")

    print("\n\nDone.")