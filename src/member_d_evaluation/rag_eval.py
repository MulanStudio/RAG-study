#!/usr/bin/env python3
"""
RAG 评测脚本（英文语料优先 + LLM-as-a-Judge）

支持：
- 多维度 LLM 评分
- 语义相似度评估（替代硬编码 exact match）
- 数值精确匹配
"""

import json
import os
import re
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from src.member_e_system.app import OilfieldRAG

logger = logging.getLogger(__name__)


class SemanticEvaluator:
    """
    基于 Embedding 的语义相似度评估器
    
    替代硬编码的字符串匹配，使用向量相似度判断答案是否正确。
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = None
        self.model_name = model_name
        self._initialized = False
    
    def _lazy_init(self):
        """懒加载模型，避免不必要的初始化开销"""
        if self._initialized:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self._initialized = True
            logger.info(f"SemanticEvaluator initialized with {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed, semantic similarity disabled")
            self._initialized = True  # 标记为已尝试初始化
    
    def similarity_score(self, standard: str, submitted: str) -> float:
        """
        计算两个文本的语义相似度 (0-1)
        
        Args:
            standard: 标准答案
            submitted: 提交的答案
        
        Returns:
            相似度分数 0.0-1.0
        """
        self._lazy_init()
        if not self.model:
            return 0.0
        
        try:
            embeddings = self.model.encode([standard, submitted])
            # 计算余弦相似度
            from numpy import dot
            from numpy.linalg import norm
            similarity = dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
            return float(max(0.0, min(1.0, similarity)))
        except Exception as e:
            logger.warning(f"Similarity calculation failed: {e}")
            return 0.0
    
    def is_match(self, standard: str, submitted: str, threshold: float = 0.85) -> bool:
        """
        判断答案是否匹配（语义等价）
        
        策略：
        1. 数值答案：精确匹配数字部分
        2. 选择题：匹配选项字母
        3. 其他：语义相似度 >= threshold
        """
        # 1. 数值精确匹配
        std_nums = re.findall(r"-?\d+\.?\d*", standard)
        sub_nums = re.findall(r"-?\d+\.?\d*", submitted)
        if std_nums and sub_nums:
            # 主数值匹配
            return std_nums[0] == sub_nums[0]
        
        # 2. 选择题匹配（只比较字母）
        std_letter = re.match(r"^([A-Ha-h])", standard.strip())
        sub_letter = re.match(r"^([A-Ha-h])", submitted.strip())
        if std_letter and sub_letter:
            return std_letter.group(1).upper() == sub_letter.group(1).upper()
        
        # 3. 语义相似度
        return self.similarity_score(standard, submitted) >= threshold
    
    def evaluate(self, standard: str, submitted: str) -> Dict:
        """
        完整评估，返回详细结果
        """
        similarity = self.similarity_score(standard, submitted)
        is_match = self.is_match(standard, submitted)
        
        return {
            "similarity_score": round(similarity, 3),
            "is_match": is_match,
            "standard": standard,
            "submitted": submitted
        }


# 全局评估器实例（懒加载）
_semantic_evaluator: Optional[SemanticEvaluator] = None

def get_semantic_evaluator() -> SemanticEvaluator:
    """获取全局语义评估器实例"""
    global _semantic_evaluator
    if _semantic_evaluator is None:
        _semantic_evaluator = SemanticEvaluator()
    return _semantic_evaluator


EVAL_CASES = [
    {
        "id": "math_basic",
        "tag": "commonsense/math",
        "question": "1+1",
        "standard_answer": "2"
    },
    {
        "id": "geo_perm",
        "tag": "csv/analysis",
        "question": "Which geological zone has the highest permeability, and what is its lithology?",
        "standard_answer": "Zone_A, Sandstone"
    },
    {
        "id": "latam_q3_revenue",
        "tag": "csv/finance",
        "question": "What is the Q3 2024 revenue (M USD) for Latin America?",
        "standard_answer": "340.0"
    },
    {
        "id": "me_q3_revenue",
        "tag": "csv/finance",
        "question": "What is the Q3 2024 revenue (M USD) for the Middle East?",
        "standard_answer": "850.5"
    },
    {
        "id": "digital_churn",
        "tag": "csv/finance",
        "question": "What is the Q3 2024 churn rate for Global Digital Solutions?",
        "standard_answer": "1.8%"
    },
    {
        "id": "fcf_q3",
        "tag": "csv/finance",
        "question": "What is the Q3 2024 free cash flow (M USD)?",
        "standard_answer": "210.0"
    }
]


REFUSAL_PATTERNS = [
    "i cannot find this information",
    "cannot find this information",
    "cannot find",
    "unable to find",
    "无法找到",
]


def _parse_score(text: str) -> int:
    match = re.search(r"Score:\s*([0-5])", text)
    if match:
        return int(match.group(1))
    return 0


def _parse_yes_no(text: str) -> str:
    match = re.search(r"Verdict:\s*(yes|no)", text, re.IGNORECASE)
    return match.group(1).lower() if match else "no"


def _is_refusal(answer: str) -> bool:
    lower = answer.lower()
    return any(pat in lower for pat in REFUSAL_PATTERNS)


def _judge_multidim(llm, question: str, standard_answer: str, answer: str) -> Tuple[Dict[str, int], str]:
    prompt = f"""Score the submitted answer on five aspects (1-5), using ONLY the question, standard answer, and submitted answer.

Aspects:
1. Groundedness: whether the submitted answer matches the standard answer.
2. Relevance: whether the answer addresses the question.
3. Coherence: logical structure and consistency.
4. Fluency: grammatical correctness and readability.
5. Similarity: semantic similarity to the standard answer.

Question: {question}
Standard Answer: {standard_answer}
Submitted Answer: {answer}

Output format:
Groundedness: <1-5>
Relevance: <1-5>
Coherence: <1-5>
Fluency: <1-5>
Similarity: <1-5>
Reason: <short reason>"""
    response = llm.invoke(prompt).content
    scores = {}
    for key in ["Groundedness", "Relevance", "Coherence", "Fluency", "Similarity"]:
        match = re.search(rf"{key}:\s*([0-5])", response, re.IGNORECASE)
        scores[key.lower()] = int(match.group(1)) if match else 0
    return scores, response


def _extract_numbers(text: str) -> List[str]:
    nums = re.findall(r"-?\d+(?:\.\d+)?%?", text)
    return nums


def _adjust_scores(question: str, standard_answer: str, answer: str, scores: Dict[str, int]) -> Dict[str, int]:
    """对打分做规则微调：数字一致性与冗长惩罚"""
    std_nums = _extract_numbers(standard_answer)
    ans_nums = _extract_numbers(answer)

    # 数字一致性
    if std_nums:
        if not ans_nums:
            scores["groundedness"] = max(1, scores["groundedness"] - 1)
            scores["similarity"] = max(1, scores["similarity"] - 1)
        elif std_nums[0] != ans_nums[0]:
            scores["groundedness"] = max(1, scores["groundedness"] - 1)
            scores["similarity"] = max(1, scores["similarity"] - 1)
        else:
            scores["groundedness"] = min(5, scores["groundedness"] + 1)
            scores["similarity"] = min(5, scores["similarity"] + 1)

    # 冗长惩罚
    if len(answer) > max(200, len(standard_answer) * 3):
        scores["coherence"] = max(1, scores["coherence"] - 1)
        scores["fluency"] = max(1, scores["fluency"] - 1)

    return scores


def _format_context(docs: List, max_chars: int = 3000) -> str:
    joined = "\n\n".join(doc.page_content for doc in docs)
    return joined[:max_chars]


def run_eval() -> Dict:
    rag = OilfieldRAG()
    rag.initialize()

    if not rag.llm:
        raise RuntimeError("LLM not available; cannot run LLM-as-a-Judge evaluation.")

    results = []
    error_stats = {
        "missing_standard": 0,
        "low_groundedness": 0,
        "scored_cases": 0
    }

    for case in EVAL_CASES:
        question = case["question"]
        docs, retrieval_debug = rag.retriever.retrieve(question, top_k=6)
        answer, gen_debug = rag.generator.generate(question, docs)

        standard_answer = case.get("standard_answer")
        if not standard_answer:
            error_stats["missing_standard"] += 1
            scores, judge_raw = {
                "groundedness": 0,
                "relevance": 0,
                "coherence": 0,
                "fluency": 0,
                "similarity": 0
            }, "Missing standard_answer for this case."
        else:
            scores, judge_raw = _judge_multidim(
                rag.llm, question, standard_answer, answer
            )
            scores = _adjust_scores(question, standard_answer, answer, scores)
            if scores.get("groundedness", 0) <= 2:
                error_stats["low_groundedness"] += 1
            error_stats["scored_cases"] += 1

        overall = round(sum(scores.values()) / 5, 2)

        results.append({
            "id": case["id"],
            "tag": case["tag"],
            "question": question,
            "standard_answer": standard_answer,
            "answer": answer,
            "scores": {
                "groundedness": scores["groundedness"],
                "relevance": scores["relevance"],
                "coherence": scores["coherence"],
                "fluency": scores["fluency"],
                "similarity": scores["similarity"],
                "overall": overall
            },
            "judge_raw": {
                "multidim": judge_raw
            },
            "retrieval_debug": retrieval_debug,
            "generation_debug": gen_debug
        })

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": rag.config["models"]["llm"]["model_name"],
        "results": results,
        "error_stats": error_stats
    }


def main():
    report = run_eval()

    out_dir = os.path.join("data", "eval_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "rag_eval_en.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 简要汇总输出
    scores = [r["scores"]["overall"] for r in report["results"]]
    scored_scores = [r["scores"]["overall"] for r in report["results"] if r["standard_answer"]]
    avg_score = round(sum(scored_scores) / len(scored_scores), 2) if scored_scores else 0
    print("=" * 60)
    print("RAG Evaluation Complete")
    print(f"Cases: {len(scores)} | Scored: {len(scored_scores)} | Avg Overall: {avg_score}/5")
    if "error_stats" in report:
        print(f"Error Stats: {report['error_stats']}")
    print(f"Report: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
