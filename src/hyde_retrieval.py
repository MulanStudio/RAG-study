"""
HyDE (Hypothetical Document Embeddings) Module
假设文档嵌入模块 - 通过生成假设答案来改善检索效果

核心原理：
1. 用户提问 -> LLM 生成一个"假设的理想答案"
2. 用这个假设答案进行向量检索（而非原问题）
3. 假设答案的语义更接近真实文档，检索效果更好

适用场景：
- 问题表述与文档内容措辞差异大
- 专业术语查询（用户可能不知道专业术语）
- 需要推理的问题

参考论文：
Precise Zero-Shot Dense Retrieval without Relevance Labels (2022)
https://arxiv.org/abs/2212.10496
"""

from typing import List, Optional, Tuple
from langchain_core.documents import Document


class HyDERetriever:
    """HyDE 检索器"""
    
    # 针对不同问题类型的假设生成 Prompt
    HYDE_PROMPTS = {
        "default": """You are an expert technical writer. Given a question, write a hypothetical document passage that would perfectly answer this question.

IMPORTANT:
- Write as if you are quoting from an actual technical document
- Include specific details, numbers, and technical terms
- Do NOT say "I don't know" - generate a plausible answer
- Keep it concise (2-4 sentences)

Question: {query}

Hypothetical Document Passage:""",

        "technical": """You are a senior Oil & Gas engineer writing technical documentation.
Given this question, write a passage that would appear in a technical manual or report answering this question.

Include:
- Technical terminology
- Specific measurements/values where applicable
- Standard industry practices

Question: {query}

Technical Documentation Passage:""",

        "financial": """You are a financial analyst writing a quarterly report.
Given this question, write a passage that would appear in a financial report answering this question.

Include:
- Specific percentages and numbers
- Comparison data if relevant
- Time periods (Q1, 2024, etc.)

Question: {query}

Financial Report Passage:""",

        "comparison": """You are writing a technical comparison document.
Given this question, write a passage comparing the items mentioned.

Include:
- Key differences and similarities
- Specific metrics for comparison
- Pros and cons if applicable

Question: {query}

Comparison Document Passage:""",

        "causal": """You are writing an incident report or root cause analysis.
Given this question about why something happened, write a passage explaining the cause.

Include:
- Timeline if relevant
- Root cause identification
- Contributing factors

Question: {query}

Incident Analysis Passage:"""
    }

    def __init__(self, llm, vectorstore, reranker=None):
        """
        初始化 HyDE 检索器
        
        Args:
            llm: LLM 实例（用于生成假设文档）
            vectorstore: 向量数据库
            reranker: 可选的重排序器
        """
        self.llm = llm
        self.vectorstore = vectorstore
        self.reranker = reranker
    
    def _detect_query_type(self, query: str) -> str:
        """
        检测问题类型，选择合适的 HyDE Prompt
        """
        query_lower = query.lower()
        
        # 因果类问题
        if any(kw in query_lower for kw in ["why", "reason", "cause", "because", "为什么", "原因"]):
            return "causal"
        
        # 比较类问题
        if any(kw in query_lower for kw in ["compare", "vs", "versus", "difference", "比较"]):
            return "comparison"
        
        # 财务类问题
        if any(kw in query_lower for kw in ["revenue", "growth", "profit", "cost", "%", "billion", "营收", "增长"]):
            return "financial"
        
        # 技术类问题
        if any(kw in query_lower for kw in ["how", "what is", "explain", "mechanism", "process", "如何", "什么是"]):
            return "technical"
        
        return "default"
    
    def generate_hypothetical_document(self, query: str, query_type: str = None) -> str:
        """
        生成假设文档
        
        Args:
            query: 用户问题
            query_type: 问题类型（可选，自动检测）
        
        Returns:
            假设文档内容
        """
        if not self.llm:
            return query  # 无 LLM 时返回原问题
        
        if query_type is None:
            query_type = self._detect_query_type(query)
        
        prompt_template = self.HYDE_PROMPTS.get(query_type, self.HYDE_PROMPTS["default"])
        prompt = prompt_template.format(query=query)
        
        try:
            response = self.llm.invoke(prompt)
            hypothetical_doc = response.content.strip()
            return hypothetical_doc
        except Exception as e:
            print(f"⚠️ HyDE generation failed: {e}")
            return query  # 失败时返回原问题
    
    def retrieve(
        self, 
        query: str, 
        k: int = 10,
        use_hyde: bool = True,
        combine_with_original: bool = True
    ) -> Tuple[List[Document], dict]:
        """
        使用 HyDE 进行检索
        
        Args:
            query: 用户问题
            k: 返回文档数量
            use_hyde: 是否使用 HyDE
            combine_with_original: 是否同时用原问题检索并合并结果
        
        Returns:
            (documents, debug_info)
        """
        debug_info = {
            "original_query": query,
            "hypothetical_doc": None,
            "query_type": None,
            "hyde_used": use_hyde and self.llm is not None
        }
        
        all_docs = []
        seen_contents = set()
        
        if use_hyde and self.llm:
            # Step 1: 检测问题类型
            query_type = self._detect_query_type(query)
            debug_info["query_type"] = query_type
            
            # Step 2: 生成假设文档
            hypothetical_doc = self.generate_hypothetical_document(query, query_type)
            debug_info["hypothetical_doc"] = hypothetical_doc[:200] + "..." if len(hypothetical_doc) > 200 else hypothetical_doc
            
            # Step 3: 用假设文档检索
            hyde_docs = self.vectorstore.similarity_search(hypothetical_doc, k=k)
            
            for doc in hyde_docs:
                if doc.page_content not in seen_contents:
                    doc.metadata["retrieval_method"] = "hyde"
                    all_docs.append(doc)
                    seen_contents.add(doc.page_content)
        
        if combine_with_original or not use_hyde or not self.llm:
            # 用原问题也检索一遍
            original_docs = self.vectorstore.similarity_search(query, k=k)
            
            for doc in original_docs:
                if doc.page_content not in seen_contents:
                    doc.metadata["retrieval_method"] = "original"
                    all_docs.append(doc)
                    seen_contents.add(doc.page_content)
        
        # Step 4: 可选的重排序
        if self.reranker and all_docs:
            pairs = [[query, doc.page_content] for doc in all_docs]
            scores = self.reranker.predict(pairs)
            doc_scores = list(zip(all_docs, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            all_docs = [doc for doc, _ in doc_scores[:k]]
        else:
            all_docs = all_docs[:k]
        
        return all_docs, debug_info


def create_hyde_enhanced_retriever(
    llm,
    vectorstore,
    bm25,
    reranker,
    splits,
    docstore
):
    """
    创建 HyDE 增强版检索函数
    
    这个函数返回一个可以直接替换原有 _retrieve_documents 的检索函数
    """
    hyde_retriever = HyDERetriever(llm, vectorstore, reranker)
    
    def enhanced_retrieve(query: str, k: int = 5) -> Tuple[List[Document], dict]:
        """HyDE + BM25 混合检索"""
        
        # HyDE 检索
        hyde_docs, debug_info = hyde_retriever.retrieve(
            query, 
            k=k*2,  # 多检索一些，后面会过滤
            use_hyde=True,
            combine_with_original=True
        )
        
        # BM25 补充
        from src.text_processing import tokenize_text
        tokenized_query = tokenize_text(query)
        from rank_bm25 import BM25Okapi
        bm25_docs = bm25.get_top_n(tokenized_query, splits, n=k)
        
        # 合并结果
        seen = set(doc.page_content for doc in hyde_docs)
        for doc in bm25_docs:
            if doc.page_content not in seen:
                doc.metadata["retrieval_method"] = "bm25"
                hyde_docs.append(doc)
                seen.add(doc.page_content)
        
        # 最终重排序
        if reranker and hyde_docs:
            pairs = [[query, doc.page_content] for doc in hyde_docs]
            scores = reranker.predict(pairs)
            doc_scores = list(zip(hyde_docs, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            final_docs = [doc for doc, _ in doc_scores[:k]]
        else:
            final_docs = hyde_docs[:k]
        
        # Parent Document 转换
        result_docs = []
        seen_parent_ids = set()
        for doc in final_docs:
            parent_id = doc.metadata.get("parent_id")
            if parent_id and parent_id in docstore:
                if parent_id not in seen_parent_ids:
                    result_docs.append(docstore[parent_id])
                    seen_parent_ids.add(parent_id)
            else:
                result_docs.append(doc)
        
        return result_docs, debug_info
    
    return enhanced_retrieve


# ============ 测试函数 ============

def test_hyde_prompts():
    """测试 HyDE Prompt 类型检测"""
    print("=" * 60)
    print("🧪 Testing HyDE Query Type Detection")
    print("=" * 60)
    
    retriever = HyDERetriever(llm=None, vectorstore=None)
    
    test_cases = [
        ("What is hydraulic fracturing?", "technical"),
        ("Why did drilling stop?", "causal"),
        ("Compare SLB vs Halliburton revenue", "comparison"),
        ("What is the revenue growth in Q3 2024?", "financial"),
        ("Where is the nearest hospital?", "default"),
    ]
    
    for query, expected_type in test_cases:
        detected = retriever._detect_query_type(query)
        status = "✅" if detected == expected_type else "❌"
        print(f"{status} Query: {query}")
        print(f"   Expected: {expected_type}, Got: {detected}")
    
    print("\n✅ Query type detection test complete")


if __name__ == "__main__":
    test_hyde_prompts()
