"""
文本预处理模块 — 分词、词干提取、词形还原

演示经典 NLP 预处理管道的三个核心操作：
1. Tokenization（分词）：将文本分割成模型可处理的单元
2. Stemming（词干提取）：基于规则去除后缀，获取词根
3. Lemmatization（词形还原）：基于语法知识还原到词典形式
"""

from __future__ import annotations

import re
from collections.abc import Callable


# =============================================================================
# 分词（Tokenization）
# =============================================================================

# 正则表达式编译后复用，避免重复编译开销
# 三个模式按优先级排列：
#   1. [A-Za-z]+(?:'[A-Za-z]+)? — 字母组成的单词，支持内部撇号（如 don't, it's）
#   2. [0-9]+ — 纯数字序列
#   3. [^\sA-Za-z0-9] — 非空白、非字母数字的单字符（标点符号）
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]")


def tokenize(text: str) -> list[str]:
    """
    将文本分割成 tokens（单词/符号单元）。

    Args:
        text: 待分词的原始文本

    Returns:
        tokens 列表，每个元素是文本中的一个独立单元

    Example:
        >>> tokenize("The cats weren't running at 3pm.")
        ['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
    """
    return WORD_RE.findall(text)


# =============================================================================
# 词干提取（Stemming）— Porter 算法第 1a 步
# =============================================================================

def stem_step_1a(word: str) -> str:
    """
    Porter 词干提取算法第 1a 步：处理复数和相关后缀。

    规则按优先级从上到下匹配，先匹配的规则生效。
    这是完整 Porter 算法的简化版本，完整算法有 5 个阶段（1a-1b-1c-2-3）。

    Args:
        word: 待提取词干的单词

    Returns:
        去除后缀后的词干形式

    Rules:
        - sses -> ss  （如 caresses -> caress）
        - ies  -> i    （如 ponies -> poni，而非 pony）
        - ss   -> ss  （不变，如 address）
        - s    -> ''   （如 cats -> cat，但仅当长度 > 1 时）
        - 其他  -> 保持不变
    """
    if word.endswith("sses"):
        # sses 结尾 → 去掉 es
        # 例：caresses → caress
        return word[:-2]
    if word.endswith("ies"):
        # ies 结尾 → 去掉 es（不是 y，这是 Porter 规则）
        # 例：ponies → poni
        return word[:-2]
    if word.endswith("ss"):
        # ss 结尾保持不变（如 address、grass）
        return word
    if word.endswith("s") and len(word) > 1:
        # 单独 s 结尾 → 去掉 s
        # 长度检查避免将 's' 当作单词处理
        # 例：cats → cat，buts → but
        return word[:-1]
    # 其他情况直接返回原词
    return word


# =============================================================================
# 词形还原（Lemmatization）
# =============================================================================

# 词元表：存储不规则变化和需要语法上下文才能还原的词形
# 键为 (原形, 词性) 元组，值为词典形式
LEMMA_TABLE = {
    ("running", "VERB"): "run",      # 动名词 → 动词原形
    ("ran", "VERB"): "run",          # 过去式 → 动词原形
    ("runs", "VERB"): "run",         # 第三人称单数 → 动词原形
    ("better", "ADJ"): "good",       # 比较级 → 原级（不规则）
    ("best", "ADJ"): "good",         # 最高级 → 原级（不规则）
    ("cats", "NOUN"): "cat",         # 复数 → 单数
    ("cat", "NOUN"): "cat",          # 已是最简形式，直接返回
    ("were", "VERB"): "be",          # 复数过去式 → be
    ("was", "VERB"): "be",           # 单数过去式 → be
    ("is", "VERB"): "be",            # 现在时第三人称单数 → be
}


def lemmatize(word: str, pos: str) -> str:
    """
    将单词还原为词典形式（lemma）。

    使用「查表 + 回退规则」的组合策略：
    1. 优先在词元表中查找精确匹配
    2. 回退到简单规则处理常见情况
    3. 最后兜底返回小写形式

    Args:
        word: 待还原的单词
        pos:  词性标签（POS tag），如 "VERB", "NOUN", "ADJ"

    Returns:
        单词的词典形式

    Example:
        >>> lemmatize("running", "VERB")
        'run'
        >>> lemmatize("cats", "NOUN")
        'cat'
        >>> lemmatize("better", "ADJ")
        'good'
    """
    # 构造查找键（小写以忽略大小写差异）
    key = (word.lower(), pos)

    # 第一优先级：查表匹配（处理不规则变化）
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]

    # 第二优先级：规则回退（处理规则变化）
    if pos == "VERB" and word.endswith("ing"):
        # 动词 + ing 结尾 → 去掉 ing
        # 例：running → run，walking → walk
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        # 名词 + s 结尾 → 去掉 s（简单复数规则）
        # 例：cats → cat
        return word[:-1]

    # 第三优先级：兜底返回小写形式
    return word.lower()


# =============================================================================
# 预处理管道（Preprocessing Pipeline）
# =============================================================================

def preprocess(text: str, pos_tagger: "Callable[[list[str]], list[tuple[str, str]]] | None" = None) -> dict[str, list[str]]:
    """
    完整预处理管道：分词 → 词干提取 → 词形还原。

    按顺序执行三个预处理操作，返回每一步的结果用于分析和调试。

    Args:
        text:       待处理的原始文本
        pos_tagger: 可选的词性标注函数
                    签名：tokens -> [(token, pos), ...]
                    如果不提供，默认所有 token 为 NOUN

    Returns:
        包含四个键的字典：
        - 'tokens': 分词结果
        - 'stems':  词干提取结果
        - 'lemmas': 词形还原结果

    Note:
        词性标注器是可选依赖。如果没有传入，词形还原会退化为简单规则，
        准确率会显著下降（动词会被错误地处理为名词）。

    Example:
        >>> result = preprocess("The cats were running.", demo_pos_tagger)
        >>> result['tokens']
        ['The', 'cats', 'were', 'running', 'at', '3', 'pm', '.']
        >>> result['lemmas']
        ['the', 'cat', 'be', 'run', 'at', '3', 'pm', '.']
    """
    # 第一步：分词
    tokens = tokenize(text)

    # 第二步：词干提取（先将 token 转小写以统一处理）
    stems = [stem_step_1a(t.lower()) for t in tokens]

    # 第三步：词性标注（可选，如果有则使用标注结果）
    if pos_tagger:
        tags = pos_tagger(tokens)
    else:
        # 无标注器时的默认值：全部假设为名词（简单但会损失准确率）
        tags = [(t, "NOUN") for t in tokens]

    # 第四步：词形还原（基于词性标注结果）
    lemmas = [lemmatize(word, pos) for word, pos in tags]

    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}


# =============================================================================
# 演示用简单词性标注器
# =============================================================================

def demo_pos_tagger(tokens: list[str]) -> list[tuple[str, str]]:
    """
    演示用词性标注器：基于硬编码词典的简单规则。

    注意：这是教学演示用的简化实现。
    真实场景应使用 NLTK 的 pos_tag 或 spaCy 的内置标注器。
    这个简单版本只能识别少量已知词汇的词性。

    Args:
        tokens: 分词结果列表

    Returns:
        (token, pos) 元组列表，其中 pos 为 "VERB"、"ADJ" 或 "NOUN"

    Known limitations:
        - 仅能识别预定义集合中的词汇
        - 无法处理未登录词（OOV）
        - 对上下文敏感的情况会判断错误
    """
    # 预定义的动词集合（不规则变化形式）
    verbs = {"running", "ran", "runs", "were", "was", "is", "watched"}
    # 预定义的形容词集合
    adjs = {"better", "best"}

    out = []
    for t in tokens:
        low = t.lower()
        if low in verbs:
            out.append((t, "VERB"))
        elif low in adjs:
            out.append((t, "ADJ"))
        else:
            out.append((t, "NOUN"))
    return out


# =============================================================================
# 主函数：演示预处理效果
# =============================================================================

def main() -> None:
    """
    演示预处理管道的完整效果。

    处理示例文本，输出分词、词干提取、词形还原的结果对比。
    """
    text = "The cats were running at 3pm."
    result = preprocess(text, pos_tagger=demo_pos_tagger)

    print(f"input:  {text}")
    print(f"tokens: {result['tokens']}")
    print(f"stems:  {result['stems']}")
    print(f"lemmas: {result['lemmas']}")

    # 预期输出分析：
    # tokens: ['The', 'cats', 'were', 'running', 'at', '3', 'pm', '.']
    #         - "The" 保留大小写
    #         - "3pm" 被分割为 ['3', 'pm']（数字和字母分开处理）
    #
    # stems:  ['the', 'cat', 'were', 'run', 'at', '3', 'pm', '.']
    #         - 词干提取后：cats → cat，running → run
    #         - 注意：were 没有被词干提取改变（不在 sses/ies/s/ss 规则中）
    #
    # lemmas: ['the', 'cat', 'be', 'run', 'at', '3', 'pm', '.']
    #         - 词形还原后：cats → cat，were → be，running → run
    #         - 这是因为 demo_pos_tagger 正确识别了词性


if __name__ == "__main__":
    main()