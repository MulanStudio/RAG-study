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
import logging
from typing import List, Dict, Tuple, Optional
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


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
            except Exception as e:
                logger.debug(f"Document grading failed: {e}")
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
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}")
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
        prompt = self.prompts.format(
            "faithfulness_check",
            query=query,
            answer=answer,
            context=context[:3000]
        )
        try:
            verdict = self.llm.invoke(prompt).content.strip().lower()
            return verdict.startswith("yes")
        except Exception:
            return True

    def _llm_confidence_check(self, query: str, context: str) -> float:
        """
        使用 LLM 判断上下文能否回答问题，返回置信度分数 (0.0-1.0)。
        """
        if not self.llm:
            return 1.0  # 无 LLM 时放行
        
        prompt = self.prompts.format(
            "confidence_check",
            context=context[:3000],
            query=query
        )
        try:
            response = self.llm.invoke(prompt).content.strip()
            # 提取数字
            import re
            match = re.search(r"(\d+\.?\d*)", response)
            if match:
                score = float(match.group(1))
                return min(1.0, max(0.0, score))
            return 0.5  # 无法解析时默认中等置信度
        except Exception:
            return 1.0  # 出错时放行

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
        # 实体/缩写匹配
        import re
        ans_tokens = set(re.findall(r"[A-Za-z0-9]+", answer_clean))
        for letter, text in options:
            opt_tokens = set(re.findall(r"[A-Za-z0-9]+", text))
            if ans_tokens & opt_tokens:
                return f"{letter}. {text}"
        # 数值匹配
        import re
        ans_nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", answer_clean)]
        if ans_nums:
            for letter, text in options:
                opt_nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]
                if any(abs(a - o) < 1e-6 for a in ans_nums for o in opt_nums):
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

    def _parse_kv_record(self, text: str) -> Dict[str, str]:
        """解析 Table Record / Excel Record 为键值对"""
        record = {}
        if ":" not in text:
            return record
        # 兼容 "Table Record ...: key: val | key2: val2"
        if "Table Record" in text:
            parts = text.split(":", 1)[1]
        else:
            parts = text
        for seg in parts.split("|"):
            seg = seg.strip()
            if ":" in seg:
                key, val = seg.split(":", 1)
                record[key.strip().lower()] = val.strip()
        return record

    def _extract_entities(self, query: str) -> List[str]:
        """提取问题中的实体（如公司缩写）"""
        import re
        entities = re.findall(r"\b[A-Z]{2,}\b", query)
        return [e.strip() for e in entities if e.strip()]

    def _extract_layer_names(self, query: str) -> List[str]:
        """提取地层/层位名称（如 Zone_A）"""
        import re
        names = re.findall(r"\bZone_[A-Za-z0-9]+\b", query)
        if names:
            return names
        matches = re.findall(r"\bLayer\s+([A-Za-z0-9_]+)\b", query, flags=re.IGNORECASE)
        return [m.strip() for m in matches if m.strip()]

    def _match_record(self, query: str, record: Dict[str, str]) -> int:
        """根据查询匹配记录，返回匹配分数"""
        from src.member_b_retrieval.text_processing import extract_key_terms
        terms = extract_key_terms(query)
        score = 0
        for term in terms:
            for k, v in record.items():
                if term in k or term in v.lower():
                    score += 1
        entities = self._extract_entities(query)
        if entities:
            record_text = " ".join([f"{k} {v}" for k, v in record.items()]).lower()
            for ent in entities:
                if ent.lower() in record_text:
                    score += 3
        return score

    def _pick_value_by_query(self, query: str, record: Dict[str, str]) -> Optional[Tuple[str, str]]:
        """根据问题挑选最可能的字段值"""
        from src.member_b_retrieval.text_processing import extract_key_terms
        q_lower = query.lower()
        if "厚度" in q_lower or "thickness" in q_lower:
            return None
        if ("porosity" in q_lower or "孔隙度" in q_lower) and "porosity_avg" in record:
            return record.get("porosity_avg"), "porosity_avg"
        if ("permeability" in q_lower or "渗透率" in q_lower) and "permeability_md" in record:
            return record.get("permeability_md"), "permeability_md"
        # 直接匹配季度字段
        if ("q1" in q_lower and "2024" in q_lower) and "q1_2024" in record:
            return record.get("q1_2024"), "q1_2024"
        if ("q2" in q_lower and "2024" in q_lower) and "q2_2024" in record:
            return record.get("q2_2024"), "q2_2024"
        if ("q3" in q_lower and "2024" in q_lower) and "q3_2024" in record:
            return record.get("q3_2024"), "q3_2024"
        if ("q4" in q_lower and "2024" in q_lower) and "q4_2024" in record:
            return record.get("q4_2024"), "q4_2024"
        # 员工数
        if ("employee" in q_lower or "员工" in q_lower) and "employees" in record:
            return record.get("employees"), "employees"
        # 营收
        if "revenue" in q_lower or "营收" in q_lower:
            for k in record.keys():
                if "revenue" in k or "营收" in k:
                    return record.get(k), k
        terms = extract_key_terms(query)
        if not terms:
            return None
        best_key = None
        best_score = 0
        for k in record.keys():
            score = sum(1 for t in terms if t in k)
            if score > best_score:
                best_score = score
                best_key = k
        if best_key:
            return record.get(best_key), best_key
        # fallback: 返回包含数字的字段
        for k, v in record.items():
            if any(ch.isdigit() for ch in v):
                return v, k
        return None

    def _standardize_number(self, value: str, decimals: int = 2) -> str:
        """标准化数值格式（统一小数位数）"""
        import re
        # 提取数值部分
        match = re.search(r"(-?\d+(?:\.\d+)?)", value)
        if not match:
            return value
        
        num_str = match.group(1)
        try:
            num = float(num_str)
            # 整数不加小数点
            if num == int(num) and decimals == 0:
                formatted = str(int(num))
            else:
                formatted = f"{num:.{decimals}f}"
            # 替换原始数值
            return value.replace(num_str, formatted, 1)
        except ValueError:
            return value
    
    def _format_value_with_unit(self, value: str, key: str) -> str:
        """根据字段名补充单位，并标准化格式"""
        import re
        if not value:
            return value
        
        # 如果已有单位，只做格式化
        has_unit = bool(re.search(r"[a-zA-Z%]", value))
        
        key_lower = (key or "").lower()
        
        # 确定单位和小数位数
        unit = ""
        decimals = 2  # 默认2位小数
        
        if "billion" in key_lower or ("revenue" in key_lower and "m" not in key_lower):
            unit = " Billion USD"
            decimals = 1
        elif "million" in key_lower or "m_usd" in key_lower or "m usd" in key_lower:
            unit = " Million USD"
            decimals = 1
        elif "arr" in key_lower or "cash flow" in key_lower or "free_cash_flow" in key_lower:
            unit = " Million USD"
            decimals = 1
        elif "profit" in key_lower or "income" in key_lower or "ebitda" in key_lower:
            unit = " Billion USD"
            decimals = 1
        elif "usd" in key_lower:
            unit = " USD"
            decimals = 2
        elif "margin" in key_lower or "rate" in key_lower or "churn" in key_lower:
            unit = "%"
            decimals = 1
        elif "porosity" in key_lower or "saturation" in key_lower:
            unit = "%"
            decimals = 1
        elif "depth" in key_lower or key_lower.endswith("_m") or "thickness" in key_lower:
            unit = " m"
            decimals = 1
        elif "permeability" in key_lower:
            unit = " mD"
            decimals = 1
        elif "employees" in key_lower or "headcount" in key_lower:
            unit = " employees"
            decimals = 0
        elif "growth" in key_lower:
            unit = "%"
            decimals = 1
        elif "price" in key_lower:
            unit = " USD"
            decimals = 2
        elif "volume" in key_lower or "production" in key_lower:
            unit = " bbl"
            decimals = 0
        
        # 标准化数值
        result = self._standardize_number(value, decimals)
        
        # 添加单位（如果还没有）
        if unit and not has_unit:
            result = result + unit
        
        return result

    def _get_best_record(
        self,
        query: str,
        documents: List[Document],
        required_keys: Optional[List[str]] = None,
        layer: Optional[str] = None,
    ) -> Optional[Dict[str, str]]:
        """返回最匹配的记录（可指定必要字段/层位）"""
        best_record = None
        best_score = 0
        entities = self._extract_entities(query)
        layers = [layer] if layer else self._extract_layer_names(query)
        for doc in documents:
            if "Table Record" in doc.page_content or "Source File" in doc.page_content:
                record = self._parse_kv_record(doc.page_content)
                if record:
                    if required_keys and any(k not in record for k in required_keys):
                        continue
                    if entities:
                        record_text = " ".join([f"{k} {v}" for k, v in record.items()]).lower()
                        if not any(ent.lower() in record_text for ent in entities):
                            continue
                    if layers:
                        layer_val = str(record.get("layer", "")).lower()
                        if not any(l.lower() == layer_val for l in layers):
                            continue
                    score = self._match_record(query, record)
                    if score > best_score:
                        best_score = score
                        best_record = record
        return best_record

    def _get_best_record_value(self, query: str, documents: List[Document]) -> Optional[Tuple[str, str, Dict[str, str]]]:
        """从记录类文档中挑选最匹配的字段值"""
        q_lower = query.lower()
        required_keys = None
        if "permeability" in q_lower or "渗透率" in q_lower:
            required_keys = ["permeability_md"]
        if "porosity" in q_lower or "孔隙度" in q_lower:
            required_keys = ["porosity_avg"]
        best_record = self._get_best_record(query, documents, required_keys=required_keys)
        if not best_record:
            return None
        picked = self._pick_value_by_query(query, best_record)
        if not picked:
            return None
        value, key = picked
        value = self._format_value_with_unit(value, key)
        return value, key, best_record

    def _match_choice_by_value(self, options: List[Tuple[str, str]], value_text: str) -> Optional[str]:
        """根据数值或文本匹配选项"""
        if not value_text:
            return None
        # 直接命中文本
        for letter, text in options:
            if text.lower() in value_text.lower():
                return f"{letter}. {text}"
        # 实体/缩写匹配
        import re
        tokens = set(re.findall(r"[A-Za-z0-9]+", value_text))
        for letter, text in options:
            opt_tokens = set(re.findall(r"[A-Za-z0-9]+", text))
            if tokens & opt_tokens:
                return f"{letter}. {text}"
        # 数值匹配
        def nums(text: str) -> List[float]:
            import re
            return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]
        val_nums = nums(value_text)
        if not val_nums:
            return None
        for letter, text in options:
            opt_nums = nums(text)
            for vn in val_nums:
                if any(abs(vn - on) < 1e-6 for on in opt_nums):
                    return f"{letter}. {text}"
        return None

    def _resolve_choice_from_records(
        self, query: str, options: List[Tuple[str, str]], documents: List[Document]
    ) -> Optional[str]:
        """优先用结构化记录解析选择题"""
        # 1) 基于最佳记录的字段值
        best = self._get_best_record_value(query, documents)
        if best:
            value_text, _, _ = best
            matched = self._match_choice_by_value(options, value_text)
            if matched:
                return matched
            # 选项为公司/实体名时：匹配记录字段
            _, _, record = best
            record_text = " ".join(str(v) for v in record.values()).lower()
            for letter, opt_text in options:
                if opt_text.lower() in record_text:
                    return f"{letter}. {opt_text}"
        # 2) 问题中带数值时：从记录中找对应选项
        import re
        q_nums = re.findall(r"-?\d+(?:\.\d+)?", query)
        if q_nums:
            for doc in documents:
                text = doc.page_content
                if any(n in text for n in q_nums):
                    for letter, opt_text in options:
                        if opt_text.lower() in text.lower():
                            return f"{letter}. {opt_text}"
        return None

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

        # 差值计算（同一记录内 Qx/Qy）
        if ("difference" in q_lower or "差值" in q_lower) and ("q3" in q_lower and "q2" in q_lower):
            best_record = self._get_best_record(query, documents, required_keys=["q2_2024", "q3_2024"])
            if best_record:
                try:
                    q2 = float(str(best_record["q2_2024"]).replace("%", "").strip())
                    q3 = float(str(best_record["q3_2024"]).replace("%", "").strip())
                    diff = round(abs(q3 - q2), 2)
                    metric = str(best_record.get("metric", "")).lower()
                    if "m_usd" in metric:
                        return f"{diff} Million USD"
                    if "usd" in metric:
                        return f"{diff} USD"
                    if "margin" in metric or "rate" in metric:
                        return f"{diff}%"
                    return str(diff)
                except Exception:
                    pass
        if ("difference" in q_lower or "差值" in q_lower) and ("q3" in q_lower and "q1" in q_lower):
            best_record = self._get_best_record(query, documents, required_keys=["q1_2024", "q3_2024"])
            if best_record:
                try:
                    q1 = float(str(best_record["q1_2024"]).replace("%", "").strip())
                    q3 = float(str(best_record["q3_2024"]).replace("%", "").strip())
                    diff = round(abs(q3 - q1), 2)
                    metric = str(best_record.get("metric", "")).lower()
                    if "m_usd" in metric:
                        return f"{diff} Million USD"
                    if "usd" in metric:
                        return f"{diff} USD"
                    if "margin" in metric or "rate" in metric:
                        return f"{diff}%"
                    return str(diff)
                except Exception:
                    pass

        # 平均值计算（同一记录内 Q1/Q2 等）
        if ("average" in q_lower or "平均" in q_lower) and "q1" in q_lower and "q2" in q_lower:
            best_record = self._get_best_record(query, documents, required_keys=["q1_2024", "q2_2024"])
            if best_record:
                try:
                    q1 = float(str(best_record["q1_2024"]).replace("%", "").strip())
                    q2 = float(str(best_record["q2_2024"]).replace("%", "").strip())
                    avg = round((q1 + q2) / 2, 2)
                    metric = str(best_record.get("metric", "")).lower()
                    if "m_usd" in metric:
                        return f"{avg} Million USD"
                    if "usd" in metric:
                        return f"{avg} USD"
                    if "margin" in metric or "rate" in metric:
                        return f"{avg}%"
                    return str(avg)
                except Exception:
                    pass

        # Porosity 平均值（指定多个层位）
        if ("porosity" in q_lower or "孔隙度" in q_lower) and ("平均" in q_lower or "average" in q_lower):
            layers = self._extract_layer_names(query)
            if len(layers) >= 2:
                values = []
                for layer in layers[:2]:
                    rec = self._get_best_record(
                        query, documents, required_keys=["porosity_avg"], layer=layer
                    )
                    if rec and "porosity_avg" in rec:
                        try:
                            values.append(float(str(rec["porosity_avg"]).replace("%", "").strip()))
                        except Exception:
                            pass
                if len(values) == 2:
                    avg = (values[0] + values[1]) / 2
                    return f"{avg}%"

        # 渗透率
        if "permeability" in q_lower or "渗透率" in q_lower:
            layers = self._extract_layer_names(query)
            layer = layers[0] if layers else None
            rec = self._get_best_record(query, documents, required_keys=["permeability_md"], layer=layer)
            if rec and "permeability_md" in rec:
                return self._format_value_with_unit(rec["permeability_md"], "permeability_md")

        # 厚度计算
        if "厚度" in q_lower or "thickness" in q_lower:
            best_record = self._get_best_record(
                query, documents, required_keys=["depth_top_m", "depth_bottom_m"]
            )
            if best_record:
                try:
                    top = float(best_record["depth_top_m"])
                    bottom = float(best_record["depth_bottom_m"])
                    thickness = bottom - top
                    return f"{thickness} m"
                except Exception:
                    pass

        # 直接字段查询（表格/记录类）
        best = self._get_best_record_value(query, documents)
        if best:
            value, _, _ = best
            entities = self._extract_entities(query)
            if entities:
                return f"{entities[0]}: {value}"
            return value

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

    def _finalize_answer(self, answer: str) -> str:
        """裁剪为简短直接答案（单行）"""
        if not answer:
            return answer
        ans = answer.strip().replace("\n", " ").strip()
        # 取第一句
        import re
        parts = re.split(r"(?<=[.!?])\s+", ans)
        if parts:
            ans = parts[0]
        return ans
    
    def generate(
        self,
        query: str,
        documents: List[Document],
        use_crag: bool = True,
        retrieval_score: float = 1.0
    ) -> Tuple[str, Dict]:
        """
        生成答案
        
        Args:
            query: 用户问题
            documents: 检索到的文档
            use_crag: 是否使用 CRAG 自我修正
            retrieval_score: 检索相似度分数 (0.0-1.0)
        
        Returns:
            (answer, debug_info)
        """
        debug_info = {"retrieval_score": retrieval_score}
        
        if not self.llm:
            return "LLM 不可用", debug_info
        
        if not documents:
            return self._refusal_message(query), debug_info
        
        # ============ 二级置信度保护 ============
        # 所有问题都经过 LLM 置信度判断，根据检索分数调整阈值
        context_preview = "\n\n".join([doc.page_content for doc in documents[:5]])
        llm_confidence = self._llm_confidence_check(query, context_preview)
        debug_info["llm_confidence"] = llm_confidence
        debug_info["retrieval_score"] = retrieval_score
        
        # 动态阈值：检索分数高时用更宽松的置信度阈值
        if retrieval_score >= 0.3:
            confidence_threshold = 0.2  # 高检索分数，宽松阈值
        elif retrieval_score >= 0.15:
            confidence_threshold = 0.4  # 中等检索分数
        else:
            confidence_threshold = 0.6  # 低检索分数，严格阈值
        
        debug_info["confidence_threshold"] = confidence_threshold
        if llm_confidence < confidence_threshold:
            debug_info["confidence_guard"] = f"llm_confidence_too_low ({llm_confidence:.2f} < {confidence_threshold})"
            return self._refusal_message(query), debug_info
        # ============ 二级保护结束 ============
        
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

        # 选择题优先
        choice_options = self._extract_choice_options(query)
        prompt_name = "choice_answer" if choice_options else "generation"
        if choice_options:
            resolved = self._resolve_choice_from_records(query, choice_options, final_docs)
            if resolved:
                return resolved, debug_info

        # 尝试材料内计算/比较
        calc_answer = self._try_context_calculation(query, final_docs)
        if calc_answer:
            debug_info["context_calc"] = "answered"
            if choice_options:
                mapped = self._match_choice_by_value(choice_options, calc_answer)
                if mapped:
                    return mapped, debug_info
                return self._format_choice_answer(calc_answer, choice_options), debug_info
            return calc_answer, debug_info
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
        else:
            answer = self._finalize_answer(answer)

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
