"""
生成模块 - Prompt 管理 + CRAG 自我修正

负责人：成员C（Prompt 工程师）

核心功能：
1. Prompt 模板管理
2. CRAG 文档质量评估
3. 查询重写
4. 答案生成
"""

import os
import yaml
from typing import List, Dict, Tuple, Optional
from langchain_core.documents import Document


class PromptManager:
    """Prompt 模板管理器"""
    
    DEFAULT_PROMPTS = {
        "generation": """You are a Senior Technical Expert in the Oil & Gas industry.
Answer the user's question based on the Context provided.

INSTRUCTIONS:
1. Answer in the SAME LANGUAGE as the question.
2. Cite specific numbers and data from the context.
3. If the answer is not in the context, say "I cannot find this information."

Context:
{context}

Question: {query}

Answer:""",

        "grade": """You are a grader assessing relevance of a document to a question.
Question: {query}
Document: {doc_content}
Is this document relevant? Answer only 'yes' or 'no'.""",

        "rewrite": """Rewrite this question to be better for search:
Question: {original_query}
Rewritten:""",

        "aggregation_compare": """Compare the items based on the information below:

{sub_results}

Original Question: {original_query}

Provide a clear comparison with specific data points."""
    }
    
    def __init__(self, config_path: str = None):
        self.prompts = self.DEFAULT_PROMPTS.copy()
        
        if config_path and os.path.exists(config_path):
            self._load_from_config(config_path)
    
    def _load_from_config(self, config_path: str):
        """从配置文件加载 Prompt"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if 'prompts' in config:
                self.prompts.update(config['prompts'])
        except Exception as e:
            print(f"⚠️ 加载 Prompt 配置失败: {e}")
    
    def get(self, name: str) -> str:
        """获取 Prompt 模板"""
        return self.prompts.get(name, "")
    
    def format(self, name: str, **kwargs) -> str:
        """格式化 Prompt"""
        template = self.get(name)
        return template.format(**kwargs)


class CRAGModule:
    """
    CRAG (Corrective RAG) 自我修正模块
    
    流程:
    1. 评估检索文档质量
    2. 如果质量差，重写查询
    3. 重新检索
    """
    
    def __init__(self, llm, prompt_manager: PromptManager):
        self.llm = llm
        self.prompts = prompt_manager
    
    def grade_documents(
        self,
        query: str,
        documents: List[Document]
    ) -> Tuple[List[Document], bool]:
        """
        评估文档质量
        
        Returns:
            (relevant_docs, needs_rewrite)
        """
        if not self.llm or not documents:
            return documents, False
        
        relevant = []
        
        for doc in documents:
            prompt = self.prompts.format(
                "grade",
                query=query,
                doc_content=doc.page_content[:800]
            )
            
            try:
                response = self.llm.invoke(prompt).content.strip().lower()
                if "yes" in response:
                    relevant.append(doc)
            except:
                relevant.append(doc)  # 出错时保留
        
        # 如果少于一半相关，需要重写
        needs_rewrite = len(relevant) < len(documents) / 2
        
        return relevant, needs_rewrite
    
    def rewrite_query(self, query: str) -> str:
        """重写查询"""
        if not self.llm:
            return query
        
        prompt = self.prompts.format("rewrite", original_query=query)
        
        try:
            response = self.llm.invoke(prompt).content.strip()
            return response
        except:
            return query


class AnswerGenerator:
    """
    答案生成器
    
    Example:
        generator = AnswerGenerator(llm, prompt_manager)
        answer = generator.generate(query, docs)
    """
    
    def __init__(self, llm, prompt_manager: PromptManager):
        self.llm = llm
        self.prompts = prompt_manager
        self.crag = CRAGModule(llm, prompt_manager)
    
    def generate(
        self,
        query: str,
        documents: List[Document],
        use_crag: bool = True
    ) -> Tuple[str, Dict]:
        """
        生成答案
        
        Args:
            query: 用户问题
            documents: 检索到的文档
            use_crag: 是否使用 CRAG 自我修正
        
        Returns:
            (answer, debug_info)
        """
        debug_info = {}
        
        if not self.llm:
            return "LLM 不可用", debug_info
        
        if not documents:
            return "未找到相关文档", debug_info
        
        final_docs = documents
        
        # CRAG 质量检查
        if use_crag:
            filtered_docs, needs_rewrite = self.crag.grade_documents(query, documents)
            
            if needs_rewrite:
                debug_info["crag_status"] = "rewrite_triggered"
                new_query = self.crag.rewrite_query(query)
                debug_info["rewritten_query"] = new_query
                # 注意：这里需要外部重新检索，本模块只负责生成
            else:
                debug_info["crag_status"] = "passed"
                final_docs = filtered_docs if filtered_docs else documents
        
        # 构建上下文
        context = "\n\n".join([doc.page_content for doc in final_docs])
        
        # 生成答案
        prompt = self.prompts.format(
            "generation",
            context=context,
            query=query
        )
        
        try:
            response = self.llm.invoke(prompt)
            answer = response.content
        except Exception as e:
            answer = f"生成失败: {e}"
        
        return answer, debug_info
    
    def generate_comparison(
        self,
        query: str,
        sub_results: List[Dict]
    ) -> str:
        """
        生成比较类答案
        
        Args:
            query: 原始问题
            sub_results: [{"sub_query": "...", "context": "..."}, ...]
        """
        if not self.llm:
            return "LLM 不可用"
        
        # 构建子结果文本
        sub_text = ""
        for i, sr in enumerate(sub_results, 1):
            sub_text += f"\n--- Sub-question {i}: {sr['sub_query']} ---\n"
            sub_text += f"Context: {sr['context']}\n"
        
        prompt = self.prompts.format(
            "aggregation_compare",
            sub_results=sub_text,
            original_query=query
        )
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"生成失败: {e}"


# 便捷函数
def create_generator(llm, config_path: str = None):
    """创建生成器实例"""
    prompt_manager = PromptManager(config_path)
    return AnswerGenerator(llm, prompt_manager)
