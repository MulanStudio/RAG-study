#!/usr/bin/env python3
"""
查询鲁棒性测试套件

测试目标：无论用户如何表述问题（加寒暄、背景、废话等），
只要核心问题相同，答案应该一致。

运行方法：
    python tests/test_query_robustness.py
"""

import os
import sys

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.member_e_system.app import OilfieldRAG


# 测试用例：每组包含多种表述方式，但核心问题相同
ROBUSTNESS_TEST_CASES = [
    {
        "id": "slb_revenue",
        "core_question": "SLB的2023年营收是多少？",
        "variations": [
            "SLB的2023年营收是多少？",
            "你好，SLB的2023年营收是多少？",
            "您好，请问SLB的2023年营收是多少？",
            "你好，我是小明，想问一下SLB的2023年营收是多少？谢谢！",
            "Hi, what is SLB's 2023 revenue?",
            "Hello! I'm new here, could you tell me SLB's revenue in 2023?",
            "嗯，就是那个SLB，2023年营收多少来着？",
            "麻烦问一下，关于SLB的2023年营收数据",
        ],
        "expected_keywords": ["SLB", "33.1", "Billion", "USD"],  # 答案应包含的关键词
    },
    {
        "id": "q3_revenue",
        "core_question": "North America Onshore Drilling的Q3 2024收入是多少？",
        "variations": [
            "North America Onshore Drilling的Q3 2024收入是多少？",
            "请问，North America Onshore Drilling的Q3 2024 Revenue是？",
            "你好你好，想了解一下North America Onshore Drilling在Q3 2024的收入情况",
            "What is the Q3 2024 revenue for North America Onshore Drilling?",
        ],
        "expected_keywords": ["480"],
    },
    {
        "id": "commonsense_reject",
        "core_question": "1+1=?",
        "variations": [
            "1+1=?",
            "你好，1+1等于多少？",
            "请问一加一等于几？",
            "Hi, what is 1+1?",
            "Hello there! Could you tell me what 1+1 equals?",
        ],
        "expected_keywords": ["know", "不知道"],  # 应该拒绝回答
    },
]


def extract_answer_keywords(answer: str) -> set:
    """提取答案中的关键词（用于比较）"""
    import re
    # 提取数字
    nums = re.findall(r'\d+\.?\d*', answer)
    # 提取英文单词
    words = re.findall(r'[A-Za-z]+', answer)
    return set(nums + words)


def check_answer_consistency(answers: list, expected_keywords: list) -> dict:
    """检查多个答案是否一致"""
    results = {
        "total": len(answers),
        "consistent": 0,
        "has_expected": 0,
        "details": []
    }
    
    # 提取每个答案的关键词
    keyword_sets = [extract_answer_keywords(a) for a in answers]
    
    # 检查是否包含期望的关键词
    for i, (answer, kw_set) in enumerate(zip(answers, keyword_sets)):
        has_expected = any(
            kw.lower() in answer.lower() for kw in expected_keywords
        )
        if has_expected:
            results["has_expected"] += 1
        results["details"].append({
            "answer": answer[:100],
            "has_expected": has_expected
        })
    
    # 计算一致性（关键词重叠度）
    if len(keyword_sets) >= 2:
        first_set = keyword_sets[0]
        for kw_set in keyword_sets[1:]:
            overlap = len(first_set & kw_set)
            total = len(first_set | kw_set)
            if total > 0 and overlap / total > 0.5:
                results["consistent"] += 1
    
    return results


def run_robustness_test(rag: OilfieldRAG, verbose: bool = True):
    """运行鲁棒性测试"""
    print("=" * 60)
    print("🧪 查询鲁棒性测试")
    print("=" * 60)
    
    all_results = []
    
    for test_case in ROBUSTNESS_TEST_CASES:
        case_id = test_case["id"]
        core_q = test_case["core_question"]
        variations = test_case["variations"]
        expected_kw = test_case["expected_keywords"]
        
        print(f"\n📋 测试: {case_id}")
        print(f"   核心问题: {core_q}")
        print(f"   变体数量: {len(variations)}")
        
        answers = []
        for i, variation in enumerate(variations):
            answer = rag.ask(variation, verbose=False)
            answers.append(answer)
            
            if verbose:
                short_v = variation[:40] + "..." if len(variation) > 40 else variation
                short_a = answer[:60] + "..." if len(answer) > 60 else answer
                print(f"   {i+1}. Q: {short_v}")
                print(f"      A: {short_a}")
        
        # 检查一致性
        consistency = check_answer_consistency(answers, expected_kw)
        
        passed = consistency["has_expected"] >= len(variations) * 0.7  # 70% 以上包含期望关键词
        status = "✅ PASS" if passed else "❌ FAIL"
        
        print(f"\n   {status}")
        print(f"   - 包含期望关键词: {consistency['has_expected']}/{consistency['total']}")
        
        all_results.append({
            "case_id": case_id,
            "passed": passed,
            "details": consistency
        })
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 测试总结")
    print("=" * 60)
    
    passed_count = sum(1 for r in all_results if r["passed"])
    total_count = len(all_results)
    
    print(f"   通过: {passed_count}/{total_count}")
    
    if passed_count < total_count:
        print("\n   ⚠️ 失败的测试:")
        for r in all_results:
            if not r["passed"]:
                print(f"      - {r['case_id']}")
    
    return all_results


def main():
    print("正在初始化 RAG 系统...")
    rag = OilfieldRAG(data_dir="data/", config_path="config/config.yaml")
    rag.initialize()
    
    results = run_robustness_test(rag, verbose=True)
    
    # 返回退出码
    all_passed = all(r["passed"] for r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
