"""
RRF (Reciprocal Rank Fusion) Module
倒数排名融合模块 - 更科学地融合多路检索结果

核心原理：
RRF 比简单的加权平均更鲁棒，因为它只关注排名顺序，不关注原始分数的大小。
这解决了不同检索器分数量级不同的问题。

RRF 公式：
RRF_score(d) = Σ 1 / (k + rank_i(d))

其中：
- d 是文档
- k 是常数（通常为 60）
- rank_i(d) 是文档 d 在第 i 个检索器中的排名（从 1 开始）

参考论文：
Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods (2009)
https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
"""

from typing import List, Dict, Tuple, Any, Callable
from collections import defaultdict


class RRFRetriever:
    """RRF 多路检索融合器"""
    
    def __init__(self, k: int = 60):
        """
        初始化 RRF 融合器
        
        Args:
            k: RRF 常数，控制排名靠后的文档权重衰减速度
               - 较小的 k (如 20)：更倾向于排名靠前的文档
               - 较大的 k (如 60)：更平滑的权重分布
        """
        self.k = k
        self.retrievers: List[Dict[str, Any]] = []
    
    def add_retriever(
        self, 
        name: str, 
        retrieve_func: Callable[[str], List[Any]],
        weight: float = 1.0,
        doc_id_func: Callable[[Any], str] = None
    ):
        """
        添加一个检索器
        
        Args:
            name: 检索器名称（用于调试）
            retrieve_func: 检索函数，输入 query，输出文档列表
            weight: 该检索器的权重（RRF 分数会乘以这个权重）
            doc_id_func: 从文档中提取唯一标识的函数
                         默认使用 doc.page_content 的前 200 字符
        """
        if doc_id_func is None:
            doc_id_func = lambda doc: doc.page_content[:200] if hasattr(doc, 'page_content') else str(doc)[:200]
        
        self.retrievers.append({
            "name": name,
            "func": retrieve_func,
            "weight": weight,
            "doc_id_func": doc_id_func
        })
    
    def retrieve(self, query: str, top_k: int = 10) -> Tuple[List[Any], Dict]:
        """
        执行 RRF 融合检索
        
        Args:
            query: 查询字符串
            top_k: 返回的文档数量
        
        Returns:
            (融合后的文档列表, 调试信息)
        """
        # 存储每个文档的 RRF 分数
        rrf_scores: Dict[str, float] = defaultdict(float)
        # 存储文档对象（用 doc_id 索引）
        doc_map: Dict[str, Any] = {}
        # 调试信息
        debug_info = {
            "retrievers": [],
            "fusion_method": "RRF",
            "k": self.k
        }
        
        for retriever in self.retrievers:
            name = retriever["name"]
            func = retriever["func"]
            weight = retriever["weight"]
            doc_id_func = retriever["doc_id_func"]
            
            try:
                # 执行检索
                docs = func(query)
                
                retriever_debug = {
                    "name": name,
                    "doc_count": len(docs),
                    "weight": weight
                }
                
                # 计算 RRF 分数
                for rank, doc in enumerate(docs, start=1):
                    doc_id = doc_id_func(doc)
                    
                    # RRF 公式: 1 / (k + rank) * weight
                    rrf_score = weight / (self.k + rank)
                    rrf_scores[doc_id] += rrf_score
                    
                    # 保存文档对象
                    if doc_id not in doc_map:
                        doc_map[doc_id] = doc
                        # 添加检索来源信息
                        if hasattr(doc, 'metadata'):
                            doc.metadata["rrf_sources"] = [name]
                    else:
                        # 文档被多个检索器检索到
                        if hasattr(doc_map[doc_id], 'metadata'):
                            if "rrf_sources" not in doc_map[doc_id].metadata:
                                doc_map[doc_id].metadata["rrf_sources"] = []
                            if name not in doc_map[doc_id].metadata["rrf_sources"]:
                                doc_map[doc_id].metadata["rrf_sources"].append(name)
                
                debug_info["retrievers"].append(retriever_debug)
                
            except Exception as e:
                debug_info["retrievers"].append({
                    "name": name,
                    "error": str(e)
                })
        
        # 按 RRF 分数排序
        sorted_docs = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # 构建结果
        result_docs = []
        for doc_id, score in sorted_docs:
            doc = doc_map[doc_id]
            if hasattr(doc, 'metadata'):
                doc.metadata["rrf_score"] = score
            result_docs.append(doc)
        
        debug_info["total_unique_docs"] = len(doc_map)
        debug_info["top_scores"] = [(doc_id[:50], f"{score:.4f}") for doc_id, score in sorted_docs[:5]]
        
        return result_docs, debug_info


def create_rrf_enhanced_retrieve(
    vectorstore,
    bm25,
    splits,
    reranker=None,
    llm=None,
    k: int = 60
):
    """
    创建 RRF 增强版检索函数
    
    融合以下检索源：
    1. 向量检索（原始问题）
    2. BM25 关键词检索
    3. HyDE 检索（如果有 LLM）
    
    Returns:
        一个可以替换原有检索函数的 RRF 融合检索函数
    """
    from hyde_retrieval import HyDERetriever
    
    rrf = RRFRetriever(k=k)
    
    # 检索器 1: 向量检索
    def vector_retrieve(query: str) -> List:
        return vectorstore.similarity_search(query, k=15)
    
    rrf.add_retriever(
        name="vector",
        retrieve_func=vector_retrieve,
        weight=1.0
    )
    
    # 检索器 2: BM25 关键词检索
    def bm25_retrieve(query: str) -> List:
        from src.member_b_retrieval.text_processing import tokenize_text
        tokenized = tokenize_text(query)
        return bm25.get_top_n(tokenized, splits, n=15)
    
    rrf.add_retriever(
        name="bm25",
        retrieve_func=bm25_retrieve,
        weight=0.8  # BM25 权重略低
    )
    
    # 检索器 3: HyDE 检索（如果有 LLM）
    if llm:
        hyde_retriever = HyDERetriever(llm, vectorstore, reranker=None)
        
        def hyde_retrieve(query: str) -> List:
            docs, _ = hyde_retriever.retrieve(
                query, 
                k=15,
                use_hyde=True,
                combine_with_original=False
            )
            return docs
        
        rrf.add_retriever(
            name="hyde",
            retrieve_func=hyde_retrieve,
            weight=1.2  # HyDE 权重略高
        )
    
    def rrf_retrieve(query: str, top_k: int = 10) -> Tuple[List, Dict]:
        """执行 RRF 融合检索"""
        docs, debug = rrf.retrieve(query, top_k=top_k)
        
        # 可选：使用 reranker 进行最终精排
        if reranker and docs:
            pairs = [[query, doc.page_content] for doc in docs]
            scores = reranker.predict(pairs)
            doc_scores = list(zip(docs, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            docs = [doc for doc, _ in doc_scores]
            debug["reranked"] = True
        
        return docs, debug
    
    return rrf_retrieve


# ============ 便捷函数：计算单次 RRF ============

def rrf_fuse(
    rankings: List[List[Any]], 
    weights: List[float] = None,
    k: int = 60,
    top_n: int = 10,
    doc_id_func: Callable[[Any], str] = None
) -> List[Tuple[Any, float]]:
    """
    快速 RRF 融合函数
    
    Args:
        rankings: 多个排序列表 [[doc1, doc2, ...], [doc3, doc1, ...], ...]
        weights: 每个列表的权重，默认全为 1.0
        k: RRF 常数
        top_n: 返回前 n 个结果
        doc_id_func: 文档 ID 提取函数
    
    Returns:
        [(doc, rrf_score), ...]
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    
    if doc_id_func is None:
        doc_id_func = lambda doc: doc.page_content[:200] if hasattr(doc, 'page_content') else str(doc)[:200]
    
    rrf_scores: Dict[str, float] = defaultdict(float)
    doc_map: Dict[str, Any] = {}
    
    for ranking, weight in zip(rankings, weights):
        for rank, doc in enumerate(ranking, start=1):
            doc_id = doc_id_func(doc)
            rrf_scores[doc_id] += weight / (k + rank)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
    
    sorted_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    
    return [(doc_map[doc_id], score) for doc_id, score in sorted_results]


# ============ 测试函数 ============

def test_rrf_basic():
    """测试 RRF 基础功能"""
    print("=" * 60)
    print("🧪 Testing RRF Fusion (Basic)")
    print("=" * 60)
    
    # 模拟文档类
    class MockDoc:
        def __init__(self, content, metadata=None):
            self.page_content = content
            self.metadata = metadata or {}
    
    # 模拟三个检索器的结果
    vector_results = [
        MockDoc("Document A - Vector top 1"),
        MockDoc("Document B - Vector top 2"),
        MockDoc("Document C - Vector top 3"),
        MockDoc("Document D - Vector top 4"),
    ]
    
    bm25_results = [
        MockDoc("Document C - Vector top 3"),  # 与 vector 重叠
        MockDoc("Document E - BM25 only"),
        MockDoc("Document A - Vector top 1"),  # 与 vector 重叠
        MockDoc("Document F - BM25 only 2"),
    ]
    
    hyde_results = [
        MockDoc("Document A - Vector top 1"),  # 三路都有
        MockDoc("Document G - HyDE only"),
        MockDoc("Document C - Vector top 3"),  # 与 vector/bm25 重叠
    ]
    
    # 使用 rrf_fuse
    fused = rrf_fuse(
        rankings=[vector_results, bm25_results, hyde_results],
        weights=[1.0, 0.8, 1.2],
        k=60,
        top_n=5
    )
    
    print("\n📊 Fusion Results:")
    for i, (doc, score) in enumerate(fused, 1):
        print(f"  {i}. Score: {score:.4f} | {doc.page_content[:50]}")
    
    # 验证 Document A 应该排名最高（三路都有）
    assert "Document A" in fused[0][0].page_content, "Document A should be top 1 (in all 3 retrievers)"
    
    # 验证 Document C 应该排名第二（两路都有）
    assert "Document C" in fused[1][0].page_content, "Document C should be top 2 (in 2 retrievers)"
    
    print("\n✅ RRF Basic Test Passed!")
    return True


def test_rrf_weights():
    """测试 RRF 权重影响"""
    print("\n" + "=" * 60)
    print("🧪 Testing RRF Weight Impact")
    print("=" * 60)
    
    class MockDoc:
        def __init__(self, content):
            self.page_content = content
            self.metadata = {}
    
    retriever1 = [MockDoc("Doc X"), MockDoc("Doc Y")]
    retriever2 = [MockDoc("Doc Y"), MockDoc("Doc X")]
    
    # 权重相等时
    equal_fused = rrf_fuse([retriever1, retriever2], weights=[1.0, 1.0], top_n=2)
    print("\nEqual weights [1.0, 1.0]:")
    for doc, score in equal_fused:
        print(f"  {doc.page_content}: {score:.4f}")
    
    # 权重不等时
    weighted_fused = rrf_fuse([retriever1, retriever2], weights=[2.0, 0.5], top_n=2)
    print("\nUnequal weights [2.0, 0.5]:")
    for doc, score in weighted_fused:
        print(f"  {doc.page_content}: {score:.4f}")
    
    # 验证权重生效
    assert weighted_fused[0][0].page_content == "Doc X", "With higher weight on retriever1, Doc X should be top"
    
    print("\n✅ RRF Weight Test Passed!")
    return True


if __name__ == "__main__":
    test_rrf_basic()
    test_rrf_weights()
    print("\n🎉 All RRF tests passed!")
