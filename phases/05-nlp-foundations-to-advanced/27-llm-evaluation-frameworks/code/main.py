"""
LLM Evaluation Frameworks - Toy Implementation

本模块实现了一个简化的 RAG (Retrieval-Augmented Generation) 评估框架，
用于演示核心评估指标的计算逻辑。

基于 RAGAS (Retrieval Augmented Generation Assessment) 框架和 G-Eval 方法。
生产环境中应使用 NLI (Natural Language Inference) 模型和 LLM-as-Judge 替代词法重叠。

评估维度:
  1. Faithfulness (忠实度) - 回答是否基于检索到的上下文
  2. Answer Relevance (答案相关性) - 回答是否与问题相关
  3. Context Precision (上下文精确率) - 检索到的上下文中有多少是相关的
  4. Context Recall (上下文召回率) - 相关上下文是否都被检索到了
  5. G-Eval Correctness (G-Eval 正确性) - 基于声明级别的正确性评估
"""

import re
from collections import Counter


# 停用词集合 - 过滤常见功能词以提高词法匹配质量
STOP = {"a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "at",
        "to", "for", "with", "and", "or", "but", "this", "that", "by", "as"}


def tokenize(text):
    """
    将文本分词并移除停用词。

    使用正则表达式提取字母数字序列，转换为小写，过滤停用词。
    这是词法级别的处理；生产环境应使用语义嵌入 (Semantic Embeddings)。

    Args:
        text: 输入文本字符串

    Returns:
        list: 分词后的 token 列表 (已移除停用词)
    """
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


def split_sentences(text):
    """
    将文本按句子边界分割。

    使用句号、感叹号、问号作为分隔符。
    用于将回答拆分为独立的声明 (claims) 进行逐句评估。

    Args:
        text: 输入文本字符串

    Returns:
        list: 句子列表
    """
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def faithfulness(answer, context):
    """
    RAGAS Faithfulness 指标 - 评估回答的忠实度。

    核心问题: "回答中有多少信息可以由检索到的上下文推导出来?"

    计算逻辑:
      1. 将回答拆分为独立的声明 (claims)
      2. 对每个声明，检查其 token 是否出现在上下文中
      3. 如果重叠率 >= 50%，认为该声明被上下文支持
      4. 返回被支持的声明比例

    与 LLM-as-Judge 的区别:
      - 本实现使用词法重叠 (lexical overlap) 作为近似
      - 生产环境应使用 NLI 模型判断蕴涵关系 (entailment)
      - NLI 能理解语义等价，例如 "released" vs "launched"

    Args:
        answer: 模型生成的回答
        context: 检索到的上下文文本

    Returns:
        float: 忠实度分数 [0.0, 1.0]
              1.0 = 完全忠实 (所有声明都有上下文支持)
              0.0 = 完全不忠实
    """
    # 构建上下文词集，用于快速查找
    context_set = set(tokenize(context))

    # 将回答拆分为独立声明
    claims = split_sentences(answer)
    if not claims:
        return 0.0

    # 统计被上下文支持的声明数量
    supported = 0
    for claim in claims:
        claim_tokens = tokenize(claim)
        if not claim_tokens:
            continue

        # 计算声明 token 与上下文的重叠数量
        overlap = sum(1 for t in claim_tokens if t in context_set)

        # 如果重叠率 >= 50%，认为该声明被支持
        if overlap / len(claim_tokens) >= 0.5:
            supported += 1

    # 返回被支持的声明比例
    return supported / len(claims)


def answer_relevance(question, answer):
    """
    RAGAS Answer Relevance 指标 - 评估答案与问题的相关性。

    核心问题: "回答是否与问题相关? 是否回答了问题?"

    计算逻辑 (Jaccard 相似度):
      1. 分别提取问题和回答的 token
      2. 计算交集 / 并集 比率

    局限性:
      - 词法级别无法理解语义相关性
      - 例如 "iPhone" 和 "Apple's smartphone" 不会匹配
      - 生产环境应使用语义相似度 (如 cosine similarity of embeddings)

    Args:
        question: 用户提出的问题
        answer: 模型生成的回答

    Returns:
        float: 相关性分数 [0.0, 1.0]
              1.0 = 完全相关 (token 完全重叠)
              0.0 = 完全不相关 (无重叠)
    """
    q_tokens = set(tokenize(question))
    a_tokens = set(tokenize(answer))
    if not q_tokens or not a_tokens:
        return 0.0

    # Jaccard 相似度 = 交集大小 / 并集大小
    return len(q_tokens & a_tokens) / len(q_tokens | a_tokens)


def context_precision(retrieved_chunks, relevant_chunks):
    """
    RAGAS Context Precision 指标 - 评估检索的精确率。

    核心问题: "检索到的上下文中有多少是真正相关的?"

    与传统 IR 指标的区别:
      - 不按排名加权 (RAGAS 原始版本有 @K 变体)
      - 仅计算精确率，不考虑召回

    为什么重要:
      - 检索到无关文档会稀释上下文，增加幻觉风险
      - 高精确率 = 模型有更干净的上下文

    Args:
        retrieved_chunks: 检索到的文档块列表
        relevant_chunks: 真正相关的文档块列表 (ground truth)

    Returns:
        float: 精确率分数 [0.0, 1.0]
              1.0 = 检索到的全部相关
              0.0 = 检索到的全部不相关
    """
    if not retrieved_chunks:
        return 0.0

    # 计算检索到的块中有多少在相关块列表中
    hits = sum(1 for c in retrieved_chunks if c in relevant_chunks)
    return hits / len(retrieved_chunks)


def context_recall(retrieved_chunks, gold_answer_tokens):
    """
    RAGAS Context Recall 指标 - 评估检索的召回率。

    核心问题: "回答问题所需的信息是否都被检索到了?"

    计算逻辑:
      1. 将所有检索到的块合并为一个文本
      2. 检查标准答案的每个 token 是否出现在检索文本中
      3. 返回覆盖比例

    与 Faithfulness 的区别:
      - Faithfulness: 回答 -> 上下文 (回答是否基于上下文)
      - Context Recall: 上下文 -> 答案 (上下文是否包含答案信息)

    Args:
        retrieved_chunks: 检索到的文档块列表
        gold_answer_tokens: 标准答案的 token 列表

    Returns:
        float: 召回率分数 [0.0, 1.0]
              1.0 = 检索到了所有需要的信息
              0.0 = 检索到的信息完全不覆盖答案
    """
    # 合并所有检索到的文本
    retrieved_text = " ".join(retrieved_chunks)
    retrieved_set = set(tokenize(retrieved_text))

    if not gold_answer_tokens:
        return 0.0

    # 计算标准答案 token 被检索文本覆盖的比例
    covered = sum(1 for t in gold_answer_tokens if t in retrieved_set)
    return covered / len(gold_answer_tokens)


def g_eval_correctness(actual, expected, threshold=0.5):
    """
    G-Eval Correctness 指标 - 基于声明级别的正确性评估。

    G-Eval 框架 (Liu et al., 2023) 使用 LLM 生成评估步骤并打分。
    本实现简化为声明级别的词法匹配。

    核心问题: "回答中的声明是否与预期答案一致?"

    计算逻辑:
      1. 将实际回答拆分为独立声明
      2. 对每个声明，检查其与预期答案的 token 重叠率
      3. 如果重叠率 >= 阈值 (默认 50%)，认为该声明正确
      4. 返回正确声明的比例

    与 Faithfulness 的区别:
      - Faithfulness: 回答 vs 上下文 (是否基于检索结果)
      - G-Eval: 回答 vs 标准答案 (是否正确)

    生产环境中的 G-Eval:
      - 使用 LLM 生成详细的评估步骤 (Chain-of-Thought)
      - 让 LLM 按步骤打分 (1-5 分)
      - 结合多个评估维度 (流畅性、连贯性、正确性等)

    Args:
        actual: 模型生成的实际回答
        expected: 预期的标准答案
        threshold: token 重叠率阈值 (默认 0.5)

    Returns:
        float: 正确性分数 [0.0, 1.0]
              1.0 = 所有声明都正确
              0.0 = 所有声明都不正确
    """
    # 将实际回答拆分为声明
    a_claims = split_sentences(actual)

    # 构建预期答案的 token 集合
    e_set = set(tokenize(expected))

    if not a_claims:
        return 0.0

    # 统计正确的声明数量
    supported = 0
    for c in a_claims:
        c_tokens = tokenize(c)
        if not c_tokens:
            continue

        # 计算声明与预期答案的 token 重叠率
        overlap = sum(1 for t in c_tokens if t in e_set) / len(c_tokens)

        # 如果重叠率 >= 阈值，认为声明正确
        if overlap >= threshold:
            supported += 1

    # 返回正确声明的比例
    return supported / len(a_claims)


def main():
    """
    主函数 - 演示 RAG 评估指标的计算。

    设计了三个测试用例，覆盖不同质量的回答:
      - Case 0: 忠实且正确的回答 (所有指标应较高)
      - Case 1: 包含幻觉的回答 (日期错误，G-Eval 和 Faithfulness 应下降)
      - Case 2: 离题的回答 (与问题无关，相关性和 G-Eval 应崩溃)
    """
    # 测试用例集合
    cases = [
        # Case 0: 忠实且正确的回答
        # 预期: 所有指标都较高
        {
            "question": "When was the first iPhone released?",
            "context": [
                "Apple released the first iPhone on June 29, 2007.",
                "Steve Jobs announced the iPhone at Macworld in January 2007.",
            ],
            "answer": "The first iPhone was released on June 29, 2007.",
            "expected": "June 29, 2007",
            "gold_relevant": ["Apple released the first iPhone on June 29, 2007."],
        },
        # Case 1: 包含幻觉的回答
        # 预期: G-Eval 下降 (日期错误)，Faithfulness 部分下降 (年份错误)
        {
            "question": "When was the first iPhone released?",
            "context": [
                "Apple released the first iPhone on June 29, 2007.",
                "The moon landing was in 1969.",
            ],
            "answer": "The first iPhone launched on June 29, 2006, shortly after the moon landing.",
            "expected": "June 29, 2007",
            "gold_relevant": ["Apple released the first iPhone on June 29, 2007."],
        },
        # Case 2: 离题的回答
        # 预期: 相关性和 G-Eval 崩溃 (回答与问题无关)
        {
            "question": "When was the first iPhone released?",
            "context": [
                "Apple released the first iPhone on June 29, 2007.",
                "Android launched in 2008.",
            ],
            "answer": "Apple is a technology company based in Cupertino.",
            "expected": "June 29, 2007",
            "gold_relevant": ["Apple released the first iPhone on June 29, 2007."],
        },
    ]

    # 评估输出
    print("=== toy RAG eval: faithfulness / relevance / context precision & recall / G-Eval ===")
    print()
    for i, case in enumerate(cases):
        # 合并上下文为单个文本
        ctx_joined = " ".join(case["context"])

        # 计算所有评估指标
        f = faithfulness(case["answer"], ctx_joined)           # RAGAS Faithfulness
        r = answer_relevance(case["question"], case["answer"])  # RAGAS Answer Relevance
        cp = context_precision(case["context"], case["gold_relevant"])  # RAGAS Context Precision
        cr = context_recall(case["context"], tokenize(case["expected"]))  # RAGAS Context Recall
        ge = g_eval_correctness(case["answer"], case["expected"])  # G-Eval Correctness

        # 输出结果
        print(f"case {i}: {case['question']}")
        print(f"  answer:   {case['answer']}")
        print(f"  expected: {case['expected']}")
        print(f"  faithfulness        = {f:.2f}")
        print(f"  answer-relevance    = {r:.2f}")
        print(f"  context-precision   = {cp:.2f}")
        print(f"  context-recall      = {cr:.2f}")
        print(f"  g-eval correctness  = {ge:.2f}")
        print()

    # 结果解读
    print("interpretation:")
    print("  case 0 = faithful + correct      -> all metrics high")
    print("  case 1 = hallucinated date        -> g-eval drops, faithfulness partial")
    print("  case 2 = off-topic answer         -> relevance + g-eval collapse")
    print()
    print("note: toy uses lexical overlap. production uses NLI + LLM-as-judge.")
    print("shape of the eval loop is identical.")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
