
# 1. 使用 NLI 的 faithfulness (RAGAS风格)
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification", model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", top_k=None)

LLM = Callable[[str], str]  # 定义 LLM 类型为接受字符串输入并返回字符串输出的可调用对象

def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
    Answer: {answer}"""

    return llm(prompt).splitlines()

def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)

# 2. answer relevance
import numpy as np
from sentence_transformers import SentenceTransformer

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\nAnswer: {answer}\nQuestions:"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)

