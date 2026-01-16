#!/usr/bin/env python3
"""
RAG 评测脚本（英文语料优先 + LLM-as-a-Judge）
"""

import json
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple

from src.member_e_system.app import OilfieldRAG


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


def _judge_with_standard(llm, question: str, standard_answer: str, answer: str) -> Tuple[int, str]:
    prompt = f"""You are evaluating answer correctness (1-5).
Scoring must use ONLY the question, the standard answer, and the submitted answer.

Question: {question}
Standard Answer: {standard_answer}
Submitted Answer: {answer}

Score 1 (incorrect) to 5 (fully correct).
Output:
Score: <1-5>
Reason: <short reason>"""
    response = llm.invoke(prompt).content
    return _parse_score(response), response


def _is_commonsense_math(question: str) -> bool:
    try:
        from src.member_b_retrieval.text_processing import is_commonsense_math
        return is_commonsense_math(question)
    except Exception:
        return False


def _solve_commonsense_math(question: str) -> str:
    try:
        from src.member_b_retrieval.text_processing import solve_commonsense_math
        return solve_commonsense_math(question)
    except Exception:
        return ""


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
        "low_correctness": 0,
        "scored_cases": 0
    }

    for case in EVAL_CASES:
        question = case["question"]
        docs, retrieval_debug = rag.retriever.retrieve(question, top_k=6)
        answer, gen_debug = rag.generator.generate(question, docs)

        standard_answer = case.get("standard_answer")
        if _is_commonsense_math(question):
            expected = _solve_commonsense_math(question)
            if expected and answer.strip() == expected:
                correctness_score, judge_raw = 5, "Math commonsense: exact match."
            else:
                correctness_score, judge_raw = 1, "Math commonsense: mismatch."
        elif not standard_answer:
            error_stats["missing_standard"] += 1
            correctness_score, judge_raw = 0, "Missing standard_answer for this case."
        else:
            correctness_score, judge_raw = _judge_with_standard(
                rag.llm, question, standard_answer, answer
            )
            if correctness_score <= 2:
                error_stats["low_correctness"] += 1
            error_stats["scored_cases"] += 1

        overall = correctness_score

        results.append({
            "id": case["id"],
            "tag": case["tag"],
            "question": question,
            "standard_answer": standard_answer,
            "answer": answer,
            "scores": {
                "answer_correctness": overall
            },
            "judge_raw": {
                "answer_correctness": judge_raw
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
    scores = [r["scores"]["answer_correctness"] for r in report["results"]]
    scored_scores = [r["scores"]["answer_correctness"] for r in report["results"] if r["standard_answer"]]
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
