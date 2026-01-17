"""
检索模块 - 混合检索 + RRF 融合

负责人：成员B（检索工程师）

核心策略：
1. 分组向量检索（按文档类型）
2. BM25 关键词检索
3. HyDE 假设文档检索
4. RRF 多路融合
5. Reranker 精排
"""

import os
import sys
from typing import List, Dict, Tuple, Optional
from langchain_core.documents import Document

# 复用已有模块
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, project_root)
from src.member_b_retrieval.query_decomposition import QueryDecomposer
from src.member_b_retrieval.hyde_retrieval import HyDERetriever
from src.member_b_retrieval.rrf_fusion import rrf_fuse


class RAGRetriever:
    """
    RAG 检索器 - 封装所有检索逻辑
    
    Example:
        retriever = RAGRetriever(vectorstore, bm25, reranker, config)
        docs = retriever.retrieve("What is the revenue of SLB?")
    """
    
    def __init__(
        self,
        vectorstore,
        bm25,
        splits: List[Document],
        reranker,
        docstore: Dict,
        llm=None,
        config: Dict = None
    ):
        self.vectorstore = vectorstore
        self.bm25 = bm25
        self.splits = splits
        self.reranker = reranker
        self.docstore = docstore
        self.llm = llm
        self.config = config or {}
        
        # 初始化子模块
        self.query_decomposer = QueryDecomposer(llm)
        self.hyde_retriever = HyDERetriever(llm, vectorstore, reranker=None) if llm else None
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_decomposition: bool = True,
        use_hyde: bool = True
    ) -> Tuple[List[Document], Dict]:
        """
        执行检索
        
        Args:
            query: 用户问题
            top_k: 返回文档数量
            use_decomposition: 是否使用问题分解
            use_hyde: 是否使用 HyDE
        
        Returns:
            (documents, debug_info)
        """
        from src.member_b_retrieval.text_processing import normalize_query
        debug_info = {"original_query": query}
        normalized_query = normalize_query(query)
        debug_info["normalized_query"] = normalized_query
        
        # Step 1: 问题分解（可选）
        if use_decomposition and self.llm:
            decomposition = self.query_decomposer.decompose(query)
            sub_queries = decomposition["sub_queries"]
            debug_info["decomposition"] = decomposition
        else:
            sub_queries = [normalized_query or query]
        
        # Step 2: 对每个子问题检索
        all_rankings = []
        
        all_scores = []
        for sub_q in sub_queries:
            sub_q = normalize_query(sub_q) or sub_q
            intent = self._detect_query_intent(sub_q)
            # 2.1 分组向量检索
            vector_docs, vector_scores = self._grouped_vector_search(sub_q, intent=intent)
            if vector_docs:
                all_rankings.append((vector_docs, 1.0, "vector"))
                all_scores.extend(vector_scores)
            
            # 2.2 BM25 检索
            bm25_docs = self._bm25_search(sub_q)
            if bm25_docs:
                bm25_weight = 1.1 if intent == "finance" else 0.8
                all_rankings.append((bm25_docs, bm25_weight, "bm25"))
            
            # 2.3 HyDE 检索（可选）
            if use_hyde and self.hyde_retriever:
                hyde_docs, hyde_debug = self.hyde_retriever.retrieve(
                    sub_q, k=10, use_hyde=True, combine_with_original=False
                )
                if hyde_docs:
                    hyde_weight = 1.0 if intent in ["finance", "calc"] else 1.2
                    all_rankings.append((hyde_docs, hyde_weight, "hyde"))
                debug_info["hyde"] = hyde_debug
        
        if not all_rankings:
            return [], debug_info
        
        # Step 3: RRF 融合
        rankings = [r[0] for r in all_rankings]
        weights = [r[1] for r in all_rankings]
        
        fused = rrf_fuse(rankings, weights, k=60, top_n=top_k * 3)
        candidate_docs = [doc for doc, score in fused]
        
        debug_info["rrf_fusion"] = f"Merged {len(all_rankings)} rankings"
        
        # Step 4: 类型覆盖保底
        candidate_docs = self._ensure_type_coverage(query, candidate_docs)
        
        # Step 5: Parent Document 还原
        parent_docs = self._restore_parents(candidate_docs)
        
        # Step 6: Reranker 精排
        if self.reranker and parent_docs:
            final_docs = self._rerank(query, parent_docs, top_k)
        else:
            final_docs = parent_docs[:top_k]
        
        # Step 7: 实体类问题保底（如公司营收）
        final_docs = self._ensure_entity_record_in_top(query, final_docs, top_k)
        
        # 计算平均相似度分数
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        debug_info["avg_similarity_score"] = avg_score
        debug_info["max_similarity_score"] = max(all_scores) if all_scores else 0.0
        
        return final_docs, debug_info

    def _ensure_entity_record_in_top(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        q_lower = query.lower()
        if not ("revenue" in q_lower or "营收" in q_lower or "employee" in q_lower or "员工" in q_lower):
            return docs
        entities = self._extract_entities(query)
        if not entities:
            return docs
        seen = set(doc.page_content for doc in docs)
        for doc in self.splits:
            if doc.metadata.get("type") == "excel_record":
                content_lower = doc.page_content.lower()
                if any(ent.lower() in content_lower for ent in entities):
                    if doc.page_content not in seen:
                        docs = [doc] + docs
                        seen.add(doc.page_content)
                    break
        return docs[:top_k]
    
    def _grouped_vector_search(self, query: str, intent: str = "general") -> Tuple[List[Document], List[float]]:
        """分组向量检索，返回文档和相似度分数"""
        docs = []
        scores = []
        seen = set()
        cfg = self.config.get("retrieval", {})
        boost = 2 if intent in ["finance", "calc"] else 1
        filters = [
            {"type": "excel_record", "k": cfg.get("k_excel", 10) * boost},
            {"type": "contract_clause", "k": cfg.get("k_word", 5)},
            {"type": "image_caption", "k": cfg.get("k_image", 3)},
            {"type": "markdown_section", "k": cfg.get("k_tech", 5)},
            {"type": "pdf_table_record", "k": cfg.get("k_pdf_table", 5) * boost},
            {"type": "pdf_text", "k": cfg.get("k_pdf_text", 5)},
            {"type": "ppt_slide", "k": cfg.get("k_ppt", 5)},
            {"type": None, "k": cfg.get("k_general", 5)},  # General
        ]
        
        for f in filters:
            try:
                kwargs = {"k": f["k"]}
                if f["type"]:
                    kwargs["filter"] = {"type": f["type"]}
                
                # 使用 similarity_search_with_score 获取分数
                results = self.vectorstore.similarity_search_with_score(query, **kwargs)
                
                for doc, score in results:
                    if doc.page_content not in seen:
                        docs.append(doc)
                        # Chroma 返回的是距离，越小越好，转换为相似度（1 - distance）
                        similarity = max(0, 1 - score) if score <= 1 else 1 / (1 + score)
                        scores.append(similarity)
                        seen.add(doc.page_content)
            except:
                pass
        
        return docs, scores

    def _detect_query_intent(self, query: str) -> str:
        q = query.lower()
        finance_keywords = ["revenue", "ebitda", "margin", "cash flow", "growth", "q1", "q2", "q3", "q4", "usd", "arr"]
        calc_keywords = ["sum", "total", "difference", "increase", "decrease", "average", "avg", "compare", "higher", "lower"]
        if any(k in q for k in finance_keywords):
            return "finance"
        if any(k in q for k in calc_keywords):
            return "calc"
        return "general"

    def _ensure_type_coverage(self, query: str, docs: List[Document]) -> List[Document]:
        """为指定类型提供检索保底，提升多模态覆盖率"""
        cfg = self.config.get("retrieval", {})
        must_types = cfg.get(
            "must_include_types",
            ["pdf_table_record", "excel_record", "ppt_slide", "image_caption"]
        )
        q_lower = query.lower()
        if "revenue" in q_lower or "营收" in q_lower or "usd" in q_lower:
            if "excel_record" not in must_types:
                must_types.append("excel_record")
            if "pdf_table_record" not in must_types:
                must_types.append("pdf_table_record")
        if not must_types:
            return docs

        seen = set(doc.page_content for doc in docs)
        present_types = set(doc.metadata.get("type") for doc in docs if doc.metadata)
        
        type_k_map = {
            "pdf_table_record": cfg.get("k_pdf_table", 6),
            "pdf_text": cfg.get("k_pdf_text", 5),
            "ppt_slide": cfg.get("k_ppt", 5),
            "image_caption": cfg.get("k_image", 3),
            "excel_record": cfg.get("k_excel", 10),
        }

        for doc_type in must_types:
            if doc_type in present_types:
                continue
            try:
                k = type_k_map.get(doc_type, 3)
                results = self.vectorstore.similarity_search(
                    query, k=k, filter={"type": doc_type}
                )
                for doc in results:
                    if doc.page_content not in seen:
                        docs.append(doc)
                        seen.add(doc.page_content)
                        break
            except Exception:
                continue

        # 关键词实体兜底：直接扫 excel_record
        entities = self._extract_entities(query)
        if entities:
            for doc in self.splits:
                if doc.metadata.get("type") == "excel_record":
                    content_lower = doc.page_content.lower()
                    if any(ent.lower() in content_lower for ent in entities):
                        if doc.page_content not in seen:
                            docs.append(doc)
                            seen.add(doc.page_content)
                            if len(entities) >= 1:
                                break
        
        return docs

    def _extract_entities(self, query: str) -> List[str]:
        import re
        entities = re.findall(r"\b[A-Z]{2,}\b", query)
        return [e.strip() for e in entities if e.strip()]
    
    def _bm25_search(self, query: str, n: int = 15) -> List[Document]:
        """BM25 关键词检索"""
        from src.member_b_retrieval.text_processing import tokenize_text
        tokenized = tokenize_text(query)
        return self.bm25.get_top_n(tokenized, self.splits, n=n)
    
    def _restore_parents(self, child_docs: List[Document]) -> List[Document]:
        """还原 Parent Document"""
        parents = []
        seen_parent_ids = set()
        
        for child in child_docs:
            parent_id = child.metadata.get("parent_id")
            
            if parent_id and parent_id in self.docstore:
                if parent_id not in seen_parent_ids:
                    parents.append(self.docstore[parent_id])
                    seen_parent_ids.add(parent_id)
            else:
                parents.append(child)
        
        return parents
    
    def _rerank(
        self,
        query: str,
        docs: List[Document],
        top_k: int
    ) -> List[Document]:
        """Reranker 精排"""
        from src.member_b_retrieval.text_processing import extract_key_terms, tokenize_text, normalize_query
        entities = self._extract_entities(query)
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)
        
        doc_scores = list(zip(docs, scores))
        
        # 额外加分规则
        final_pairs = []
        key_terms = extract_key_terms(query)
        for doc, score in doc_scores:
            final_score = score
            if key_terms:
                doc_tokens = set(tokenize_text(normalize_query(doc.page_content)))
                overlap = sum(1 for t in key_terms if t in doc_tokens)
                final_score += min(1.0, overlap * 0.2)
            
            # 事件类问题加分
            if "[EVENT:" in doc.page_content:
                if any(kw in query.lower() for kw in ["why", "reason", "stop", "fail", "为什么"]):
                    final_score += 2.0

            # 财务问题：实体匹配的 excel 记录强加分
            if doc.metadata.get("type") == "excel_record" and entities:
                content_lower = doc.page_content.lower()
                if any(ent.lower() in content_lower for ent in entities):
                    final_score += 4.0
            
            final_pairs.append((doc, final_score))
        
        final_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in final_pairs[:top_k]]


# 便捷函数
def create_retriever(vectorstore, bm25, splits, reranker, docstore, llm=None, config=None):
    """创建检索器实例"""
    return RAGRetriever(
        vectorstore=vectorstore,
        bm25=bm25,
        splits=splits,
        reranker=reranker,
        docstore=docstore,
        llm=llm,
        config=config
    )
