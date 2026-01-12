import os
import sys
import time
from streamlit_app import initialize_rag_system, process_query

# 设置环境变量
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

def run_benchmark():
    print("🚀 正在初始化 RAG 系统进行评测...")
    # 模拟 Streamlit 的缓存机制，这里直接调用
    # 注意：streamlit_app.py 里的 @st.cache_resource 在纯脚本里无效，但不影响逻辑
    # 我们需要稍微修改一下导入，或者 mock st
    
    # 为了简单，我们直接实例化系统，不依赖 streamlit 上下文
    # 但由于 initialize_rag_system 内部用了 st.info 等，直接调用会报错
    # 策略：我们手动复制初始化逻辑，或者 mock st
    pass

# Mock streamlit to avoid errors when importing/running logic
import unittest.mock as mock
sys.modules["streamlit"] = mock.MagicMock()
import streamlit as st

# 现在重新导入逻辑
from streamlit_app import initialize_rag_system, process_query

def evaluate_answer(question, answer, sources, expected_keywords):
    print(f"\n📝 [Test Case]: {question}")
    print("-" * 50)
    print(f"🤖 AI Answer: {answer[:300]}...") # 只打印前300字
    
    # 1. 检查召回源
    retrieved_sources = [os.path.basename(doc.metadata.get('source', '')) for doc, _ in sources]
    print(f"📚 Retrieved Sources: {retrieved_sources}")
    
    # 2. 关键词匹配 (简单评分)
    score = 0
    missing = []
    for kw in expected_keywords:
        if kw.lower() in answer.lower():
            score += 1
        else:
            missing.append(kw)
    
    max_score = len(expected_keywords)
    print(f"✅ Keyword Score: {score}/{max_score}")
    if missing:
        print(f"❌ Missing Keywords: {missing}")
        
    return score == max_score, retrieved_sources

def main():
    print("🔄 初始化系统...")
    rag = initialize_rag_system()
    
    test_cases = [
        {
            "q": "Explain how Mud Pulse Telemetry works and what affects its signal strength?",
            "expected": ["pressure", "valve", "attenuation", "viscosity", "frequency"],
            "type": "Technical"
        },
        {
            "q": "Compare the revenue growth of Latin America vs Middle East in Q3 2024.",
            "expected": ["12.0%", "3.5%", "Latin America", "Middle East"],
            "type": "Financial"
        },
        {
            "q": "What are the advantages of RSS over Slide Drilling?",
            "expected": ["continuous rotation", "hole quality", "spiraling", "slide"],
            "type": "Comparison"
        }
    ]
    
    results = []
    
    for case in test_cases:
        response, sources, _ = process_query(case["q"], rag)
        success, retrieved = evaluate_answer(case["q"], response, sources, case["expected"])
        results.append({
            "question": case["q"],
            "success": success,
            "retrieved": retrieved
        })
        
    print("\n" + "="*50)
    print("📊 评测总结 (Benchmark Summary)")
    print("="*50)
    all_passed = True
    for res in results:
        status = "✅ PASS" if res["success"] else "❌ FAIL"
        if not res["success"]: all_passed = False
        print(f"{status} | {res['question'][:30]}... | Sources: {res['retrieved']}")

    if not all_passed:
        print("\n⚠️  发现问题，准备优化...")
        sys.exit(1) # 返回非0状态码表示需要优化
    else:
        print("\n✨ 所有测试通过！系统表现良好。")

if __name__ == "__main__":
    main()

