"""
Query Decomposition Module
问题分解模块 - 将复杂问题拆分为多个子问题，提高检索覆盖率

适用场景：
- 比较类问题："Compare A vs B"
- 多实体问题："What are X, Y, and Z?"
- 因果推理问题："Why did X happen and what was the impact?"
"""

import json
import re
from typing import List, Dict, Optional

# 可选导入 - 允许在没有 langchain 的情况下运行基础功能
try:
    from langchain_core.documents import Document
except ImportError:
    # 定义一个简单的 Document 类用于测试
    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}


class QueryDecomposer:
    """问题分解器"""
    
    DECOMPOSE_PROMPT = """You are a query analyzer. Your task is to break down complex questions into simpler sub-questions that can be answered independently.

RULES:
1. If the question is already simple (single entity, single fact), return it as-is in a list.
2. For comparison questions, create one sub-question for each item being compared.
3. For multi-part questions, create one sub-question for each part.
4. For causal/reasoning questions, create sub-questions for: what happened, why it happened, what was the result.
5. Keep sub-questions concise and focused.
6. Maximum 4 sub-questions.

INPUT QUESTION: {query}

OUTPUT FORMAT (JSON only, no other text):
{{
    "is_complex": true/false,
    "sub_queries": ["sub_question_1", "sub_question_2", ...],
    "aggregation_type": "compare" | "list" | "summarize" | "single",
    "reasoning": "brief explanation of why you decomposed this way"
}}

Examples:

Input: "What is hydraulic fracturing?"
Output: {{"is_complex": false, "sub_queries": ["What is hydraulic fracturing?"], "aggregation_type": "single", "reasoning": "Simple definition question"}}

Input: "Compare the revenue growth of Latin America vs Middle East in Q3 2024"
Output: {{"is_complex": true, "sub_queries": ["What is the revenue growth of Latin America in Q3 2024?", "What is the revenue growth of Middle East in Q3 2024?"], "aggregation_type": "compare", "reasoning": "Comparison requires data for both regions"}}

Input: "Why did drilling stop on November 12 and what actions were taken?"
Output: {{"is_complex": true, "sub_queries": ["What caused drilling to stop on November 12?", "What actions or responses were taken after drilling stopped on November 12?"], "aggregation_type": "summarize", "reasoning": "Causal question with follow-up about response"}}

Now analyze the input question:
"""

    AGGREGATION_PROMPTS = {
        "compare": """Based on the retrieved information for each sub-question, provide a comparative analysis.

Sub-questions and their retrieved contexts:
{sub_results}

Original Question: {original_query}

Instructions:
1. Clearly state the values/facts for each item being compared
2. Highlight key differences and similarities
3. Use specific numbers and data from the context
4. Structure your answer for easy comparison

Answer:""",

        "list": """Based on the retrieved information, compile a comprehensive list answering the original question.

Sub-questions and their retrieved contexts:
{sub_results}

Original Question: {original_query}

Instructions:
1. Combine information from all sub-queries
2. Remove duplicates
3. Present in a clear, organized format

Answer:""",

        "summarize": """Based on the retrieved information, provide a comprehensive summary answering the original question.

Sub-questions and their retrieved contexts:
{sub_results}

Original Question: {original_query}

Instructions:
1. Synthesize information from all sub-queries
2. Present a coherent narrative
3. Address all aspects of the original question
4. Include specific details and data

Answer:""",

        "single": """Based on the retrieved context, answer the question directly.

Context:
{sub_results}

Question: {original_query}

Answer:"""
    }

    def __init__(self, llm):
        self.llm = llm
    
    def decompose(self, query: str) -> Dict:
        """
        分解问题
        Returns: {
            "is_complex": bool,
            "sub_queries": List[str],
            "aggregation_type": str,
            "reasoning": str
        }
        """
        if not self.llm:
            # 无LLM时的fallback：简单规则判断
            return self._rule_based_decompose(query)
        
        prompt = self.DECOMPOSE_PROMPT.format(query=query)
        
        try:
            response = self.llm.invoke(prompt).content.strip()
            
            # 提取JSON部分
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = json.loads(json_match.group())
                # 验证必需字段
                if "sub_queries" not in result or not result["sub_queries"]:
                    result["sub_queries"] = [query]
                if "aggregation_type" not in result:
                    result["aggregation_type"] = "single"
                if "is_complex" not in result:
                    result["is_complex"] = len(result["sub_queries"]) > 1
                return result
            else:
                return self._fallback_result(query)
                
        except (json.JSONDecodeError, Exception) as e:
            print(f"⚠️ Query decomposition failed: {e}")
            return self._fallback_result(query)
    
    def _rule_based_decompose(self, query: str) -> Dict:
        """基于规则的简单分解（无LLM时使用）"""
        query_lower = query.lower()
        
        # 比较类问题
        if any(kw in query_lower for kw in ["compare", "vs", "versus", "difference between"]):
            # 尝试提取比较对象
            return {
                "is_complex": True,
                "sub_queries": [query],  # 简化处理
                "aggregation_type": "compare",
                "reasoning": "Detected comparison keywords"
            }
        
        # 多部分问题（包含 and/or）
        if " and " in query_lower and "?" in query:
            parts = query.split(" and ")
            if len(parts) == 2:
                return {
                    "is_complex": True,
                    "sub_queries": [p.strip() + "?" if not p.strip().endswith("?") else p.strip() for p in parts],
                    "aggregation_type": "summarize",
                    "reasoning": "Split by 'and'"
                }
        
        return self._fallback_result(query)
    
    def _fallback_result(self, query: str) -> Dict:
        """默认返回结果"""
        return {
            "is_complex": False,
            "sub_queries": [query],
            "aggregation_type": "single",
            "reasoning": "Fallback: treating as simple query"
        }
    
    def get_aggregation_prompt(self, aggregation_type: str) -> str:
        """获取对应的聚合Prompt"""
        return self.AGGREGATION_PROMPTS.get(aggregation_type, self.AGGREGATION_PROMPTS["single"])


def retrieve_with_decomposition(
    query: str,
    decomposer: QueryDecomposer,
    retrieve_func,  # 原有的检索函数
    llm,
    verbose: bool = True
) -> tuple:
    """
    带问题分解的检索流程
    
    Args:
        query: 原始问题
        decomposer: QueryDecomposer 实例
        retrieve_func: 原有检索函数，签名为 func(query) -> List[Document]
        llm: LLM实例
        verbose: 是否打印调试信息
    
    Returns:
        (final_answer, all_docs, debug_info)
    """
    debug_info = []
    
    # Step 1: 分解问题
    decomposition = decomposer.decompose(query)
    
    if verbose:
        print(f"📊 Query Decomposition:")
        print(f"   - Is Complex: {decomposition['is_complex']}")
        print(f"   - Sub-queries: {decomposition['sub_queries']}")
        print(f"   - Aggregation: {decomposition['aggregation_type']}")
    
    debug_info.append(f"Decomposition: {decomposition}")
    
    # Step 2: 对每个子问题检索
    sub_results = []
    all_docs = []
    seen_contents = set()
    
    for i, sub_query in enumerate(decomposition["sub_queries"]):
        if verbose:
            print(f"\n🔍 Retrieving for sub-query {i+1}: {sub_query}")
        
        docs = retrieve_func(sub_query)
        
        # 去重
        unique_docs = []
        for doc in docs:
            if doc.page_content not in seen_contents:
                unique_docs.append(doc)
                seen_contents.add(doc.page_content)
                all_docs.append(doc)
        
        # 记录子问题的检索结果
        context = "\n".join([d.page_content[:500] for d in unique_docs[:3]])
        sub_results.append({
            "sub_query": sub_query,
            "context": context,
            "doc_count": len(unique_docs)
        })
        
        if verbose:
            print(f"   -> Found {len(unique_docs)} unique docs")
    
    # Step 3: 聚合生成最终答案
    if llm and decomposition["is_complex"]:
        # 构建聚合上下文
        sub_results_text = ""
        for i, sr in enumerate(sub_results):
            sub_results_text += f"\n--- Sub-question {i+1}: {sr['sub_query']} ---\n"
            sub_results_text += f"Retrieved Context:\n{sr['context']}\n"
        
        # 获取聚合Prompt
        agg_prompt_template = decomposer.get_aggregation_prompt(decomposition["aggregation_type"])
        agg_prompt = agg_prompt_template.format(
            sub_results=sub_results_text,
            original_query=query
        )
        
        if verbose:
            print(f"\n🤖 Generating aggregated answer...")
        
        try:
            final_answer = llm.invoke(agg_prompt).content
        except Exception as e:
            final_answer = f"Error generating answer: {e}"
            debug_info.append(f"Generation error: {e}")
    else:
        # 简单问题或无LLM：直接返回检索到的内容
        if sub_results:
            final_answer = sub_results[0]["context"]
        else:
            final_answer = "No relevant information found."
    
    debug_info.append(f"Total unique docs: {len(all_docs)}")
    
    return final_answer, all_docs, debug_info


# ============ 测试函数 ============

def test_decomposer():
    """测试问题分解功能"""
    print("=" * 60)
    print("🧪 Testing Query Decomposer (Rule-based, no LLM)")
    print("=" * 60)
    
    # 无LLM测试（规则模式）
    decomposer = QueryDecomposer(llm=None)
    
    test_queries = [
        "What is hydraulic fracturing?",
        "Compare the revenue of SLB vs Halliburton",
        "Why did drilling stop and what actions were taken?",
        "What is the BOP pressure rating for well ZT-09?",
    ]
    
    for q in test_queries:
        print(f"\n❓ Query: {q}")
        result = decomposer.decompose(q)
        print(f"   Complex: {result['is_complex']}")
        print(f"   Sub-queries: {result['sub_queries']}")
        print(f"   Aggregation: {result['aggregation_type']}")


if __name__ == "__main__":
    test_decomposer()
