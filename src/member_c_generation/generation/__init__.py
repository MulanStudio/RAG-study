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
2. Do NOT include citations or source-referencing language.
3. If the answer is not in the context, say "I don't know."

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

Provide a clear comparison with specific data points.""",

        "faithfulness_check": """You are a strict fact-checker.
Question: {query}
Answer: {answer}
Context: {context}

Is the answer fully supported by the context? Answer only 'yes' or 'no'."""
        ,
        "choice_answer": """You are answering a multiple-choice question.
Answer format must be: "<OptionLetter>. <OptionText>".
No explanation.

Context:
{context}

Question:
{query}

Answer:"""
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

    def _detect_language(self, text: str) -> str:
        for ch in text:
            if "\u4e00" <= ch <= "\u9fff":
                return "zh"
        return "en"

    def _refusal_message(self, query: str) -> str:
        if self._detect_language(query) == "zh":
            return "不知道。"
        return "I don't know."

    def _verify_answer(self, query: str, answer: str, context: str) -> bool:
        if not self.llm:
            return True

    def _covers_key_items(self, query: str, answer: str) -> bool:
        """检查答案是否覆盖问题中的关键项（轻量规则）"""
        from src.member_b_retrieval.text_processing import tokenize_text, normalize_query, extract_key_terms
        a_norm = normalize_query(answer)
        a_tokens = set(tokenize_text(a_norm))
        key_terms = extract_key_terms(query)
        if not key_terms:
            return True
        
        if any(ch.isdigit() for ch in query) and not any(ch.isdigit() for ch in answer):
            return False
        return any(t in a_tokens for t in key_terms)

    def _extractive_fallback(self, query: str, documents: List[Document]) -> str:
        """从上下文中抽取与问题最相关的片段作为答案"""
        from src.member_b_retrieval.text_processing import extract_key_terms
        import re

        if not documents:
            return self._refusal_message(query)

        key_terms = extract_key_terms(query)
        candidates = []

        for doc in documents[:5]:
            text = doc.page_content.replace("\n", " ")
            sentences = re.split(r"(?<=[.!?])\s+", text)
            for sent in sentences:
                if not sent.strip():
                    continue
                score = sum(1 for t in key_terms if t in sent.lower())
                if any(ch.isdigit() for ch in sent):
                    score += 1
                if score > 0:
                    candidates.append((score, sent.strip()))

        if not candidates:
            return self._refusal_message(query)

        candidates.sort(key=lambda x: x[0], reverse=True)
        top_sents = [s for _, s in candidates[:2]]
        return " ".join(top_sents)

    def _extract_choice_options(self, query: str) -> List[Tuple[str, str]]:
        """从题目中提取选项列表"""
        import re
        options = []
        pattern = re.compile(r"^\s*([A-H])[\).:\-]\s*(.+)$")
        for line in query.splitlines():
            match = pattern.match(line.strip())
            if match:
                options.append((match.group(1), match.group(2).strip()))
        return options

    def _format_choice_answer(self, answer: str, options: List[Tuple[str, str]]) -> str:
        """确保选择题输出格式为 'A. Option text'"""
        if not options:
            return answer
        mapping = {k.upper(): v for k, v in options}
        answer_clean = answer.strip()
        # 仅选项字母
        if len(answer_clean) == 1 and answer_clean.upper() in mapping:
            return f"{answer_clean.upper()}. {mapping[answer_clean.upper()]}"
        # 形如 "A" or "A."
        if answer_clean[:1].upper() in mapping:
            letter = answer_clean[:1].upper()
            return f"{letter}. {mapping[letter]}"
        # 选项文本匹配
        for letter, text in options:
            if text.lower() in answer_clean.lower():
                return f"{letter}. {text}"
        return answer
    
    def _extract_numbers(self, text: str) -> List[Tuple[float, str]]:
        """抽取数值及其单位（简化版）"""
        import re
        numbers = []
        pattern = re.compile(r"(-?\d+(?:\.\d+)?)\s*(%|bps|m\s*usd|usd|bbl|m3)?", re.IGNORECASE)
        for match in pattern.finditer(text):
            value = float(match.group(1))
            unit = (match.group(2) or "").strip().lower()
            numbers.append((value, unit))
        return numbers

    def _extract_compare_entities(self, query: str) -> List[str]:
        import re
        q = query.strip()
        q_lower = q.lower()
        if " vs " in q_lower:
            parts = re.split(r"\bvs\b", q, flags=re.IGNORECASE)
            return [p.strip(" .,:;") for p in parts if p.strip()]
        if " versus " in q_lower:
            parts = re.split(r"\bversus\b", q, flags=re.IGNORECASE)
            return [p.strip(" .,:;") for p in parts if p.strip()]
        m = re.search(r"compare\s+(.+?)\s+(?:and|vs|versus)\s+(.+)", q, flags=re.IGNORECASE)
        if m:
            return [m.group(1).strip(" .,:;"), m.group(2).strip(" .,:;")]
        return []

    def _try_context_calculation(self, query: str, documents: List[Document]) -> Optional[str]:
        """基于上下文进行计算/比较"""
        import re
        q_lower = query.lower()
        compare_keywords = ["compare", "higher", "lower", "greater", "less", "largest", "smallest", "vs", "versus"]
        calc_keywords = ["sum", "total", "difference", "increase", "decrease", "average", "avg"]

        # 比较类
        if any(k in q_lower for k in compare_keywords):
            entities = self._extract_compare_entities(query)
            if len(entities) >= 2:
                ent_a, ent_b = entities[0], entities[1]
                vals = {}
                for doc in documents[:6]:
                    text = doc.page_content.replace("\n", " ")
                    sentences = re.split(r"(?<=[.!?])\s+", text)
                    for sent in sentences:
                        sent_lower = sent.lower()
                        if ent_a.lower() in sent_lower or ent_b.lower() in sent_lower:
                            nums = self._extract_numbers(sent)
                            if nums:
                                if ent_a.lower() in sent_lower and ent_a not in vals:
                                    vals[ent_a] = nums[0]
                                if ent_b.lower() in sent_lower and ent_b not in vals:
                                    vals[ent_b] = nums[0]
                        if len(vals) >= 2:
                            break
                    if len(vals) >= 2:
                        break
                if len(vals) >= 2:
                    a_val, a_unit = vals[ent_a]
                    b_val, b_unit = vals[ent_b]
                    if a_val == b_val:
                        return f"{ent_a} and {ent_b} are equal ({a_val}{a_unit})."
                    higher = ent_a if a_val > b_val else ent_b
                    return f"{ent_a}: {a_val}{a_unit}, {ent_b}: {b_val}{b_unit}. {higher} is higher."

        # 计算类
        if any(k in q_lower for k in calc_keywords):
            numbers = []
            for doc in documents[:6]:
                text = doc.page_content.replace("\n", " ")
                numbers.extend(self._extract_numbers(text))
            if len(numbers) >= 2:
                a, unit_a = numbers[0]
                b, unit_b = numbers[1]
                unit = unit_a if unit_a == unit_b else unit_a or unit_b
                if "sum" in q_lower or "total" in q_lower:
                    return f"{a + b}{unit}"
                if "difference" in q_lower or "decrease" in q_lower:
                    return f"{abs(a - b)}{unit}"
                if "increase" in q_lower:
                    return f"{a - b}{unit}"
                if "average" in q_lower or "avg" in q_lower:
                    return f"{(a + b) / 2}{unit}"

        return None
    
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
            return self._refusal_message(query), debug_info
        
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
        
        # 尝试材料内计算/比较
        calc_answer = self._try_context_calculation(query, final_docs)
        if calc_answer:
            debug_info["context_calc"] = "answered"
            return calc_answer, debug_info

        # 构建上下文
        context = "\n\n".join([doc.page_content for doc in final_docs])

        # 选择题优先
        choice_options = self._extract_choice_options(query)
        prompt_name = "choice_answer" if choice_options else "generation"
        prompt = self.prompts.format(
            prompt_name,
            context=context,
            query=query
        )
        
        try:
            response = self.llm.invoke(prompt)
            answer = response.content
        except Exception as e:
            answer = f"生成失败: {e}"
            return answer, debug_info

        if choice_options:
            answer = self._format_choice_answer(answer, choice_options)

        if not self._covers_key_items(query, answer):
            debug_info["coverage_guard"] = "extractive_fallback"
            return self._extractive_fallback(query, final_docs), debug_info

        if not self._verify_answer(query, answer, context):
            debug_info["faithfulness_guard"] = "extractive_fallback"
            return self._extractive_fallback(query, final_docs), debug_info
        
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
