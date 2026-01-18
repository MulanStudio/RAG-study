"""
元数据清洗模块 - 标准化文档元数据

负责人：成员A（数据工程师）

功能：
1. 元数据字段标准化
2. 从文件名/内容自动提取关键信息（日期、公司名、季度等）
3. 生成内容预览
"""

import os
import re
import logging
from typing import List, Dict, Optional
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class MetadataCleaner:
    """
    元数据清洗与标准化
    
    Example:
        cleaner = MetadataCleaner()
        cleaned_docs = cleaner.clean(raw_docs)
    """
    
    # 标准元数据字段及类型
    STANDARD_FIELDS = {
        "source": str,           # 文件路径
        "type": str,             # 文档类型
        "title": str,            # 文档标题
        "date": str,             # 日期 (YYYY 或 YYYY-MM-DD)
        "quarter": str,          # 季度 (Q1/Q2/Q3/Q4)
        "company": str,          # 公司名
        "author": str,           # 作者
        "section": str,          # 章节
        "page": int,             # 页码
        "row_index": int,        # 行索引
        "slide_index": int,      # 幻灯片索引
        "keywords": str,         # 关键词
        "content_preview": str,  # 内容预览
    }
    
    # 已知公司名模式
    COMPANY_PATTERNS = [
        ("SLB", ["slb", "schlumberger"]),
        ("Halliburton", ["halliburton", "hal"]),
        ("Baker Hughes", ["baker hughes", "bakerhughes", "bkr"]),
        ("COSL", ["cosl", "中海油服", "中海油田服务"]),
        ("CNOOC", ["cnooc", "中海油", "中国海洋石油"]),
        ("Weatherford", ["weatherford"]),
        ("NOV", ["nov", "national oilwell varco"]),
        ("TechnipFMC", ["technipfmc", "technip", "fmc"]),
    ]
    
    # 财务指标关键词
    FINANCIAL_KEYWORDS = [
        "revenue", "profit", "ebitda", "margin", "growth", "cash flow",
        "营收", "利润", "毛利", "净利", "增长", "现金流",
        "q1", "q2", "q3", "q4", "quarterly", "annual", "fiscal",
    ]
    
    # 技术关键词
    TECHNICAL_KEYWORDS = [
        "drilling", "completion", "production", "reservoir",
        "rig", "well", "casing", "cement", "mud", "bit",
        "钻井", "完井", "生产", "油藏", "钻机", "套管",
    ]
    
    def clean(self, docs: List[Document], verbose: bool = True) -> List[Document]:
        """
        清洗所有文档的元数据
        
        Args:
            docs: 原始文档列表
            verbose: 是否打印处理信息
            
        Returns:
            清洗后的文档列表
        """
        cleaned = []
        stats = {"total": 0, "with_company": 0, "with_date": 0, "with_quarter": 0}
        
        for doc in docs:
            cleaned_meta = self._clean_metadata(doc.metadata, doc.page_content)
            doc.metadata = cleaned_meta
            cleaned.append(doc)
            
            # 统计
            stats["total"] += 1
            if cleaned_meta.get("company"):
                stats["with_company"] += 1
            if cleaned_meta.get("date"):
                stats["with_date"] += 1
            if cleaned_meta.get("quarter"):
                stats["with_quarter"] += 1
        
        if verbose:
            logger.info(f"Metadata cleaned: {stats['total']} docs")
            logger.info(f"  - With company: {stats['with_company']}")
            logger.info(f"  - With date: {stats['with_date']}")
            logger.info(f"  - With quarter: {stats['with_quarter']}")
        
        return cleaned
    
    def _clean_metadata(self, meta: Dict, content: str) -> Dict:
        """
        清洗单个文档的元数据
        
        Args:
            meta: 原始元数据
            content: 文档内容
            
        Returns:
            清洗后的元数据
        """
        cleaned = {}
        
        # 1. 保留并标准化已有字段
        for field, dtype in self.STANDARD_FIELDS.items():
            if field in meta:
                try:
                    value = meta[field]
                    if value is not None and str(value).strip():
                        cleaned[field] = dtype(value)
                except (ValueError, TypeError):
                    pass
        
        # 2. 从文件名提取信息
        source = meta.get("source", "")
        filename = os.path.basename(source).lower() if source else ""
        
        # 提取年份
        if "date" not in cleaned:
            date_match = re.search(r'(20\d{2})', filename)
            if date_match:
                cleaned["date"] = date_match.group(1)
            else:
                # 从内容中提取
                date_match = re.search(r'(20\d{2})', content[:500])
                if date_match:
                    cleaned["date"] = date_match.group(1)
        
        # 提取季度
        if "quarter" not in cleaned:
            quarter_match = re.search(r'Q([1-4])', filename, re.IGNORECASE)
            if quarter_match:
                cleaned["quarter"] = f"Q{quarter_match.group(1)}"
            else:
                # 从内容中提取
                quarter_match = re.search(r'Q([1-4])\s*20\d{2}|20\d{2}\s*Q([1-4])', content[:500], re.IGNORECASE)
                if quarter_match:
                    q = quarter_match.group(1) or quarter_match.group(2)
                    cleaned["quarter"] = f"Q{q}"
        
        # 3. 提取公司名
        if "company" not in cleaned:
            company = self._extract_company(filename, content)
            if company:
                cleaned["company"] = company
        
        # 4. 生成内容预览
        cleaned["content_preview"] = self._generate_preview(content)
        
        # 5. 提取/生成关键词
        if "keywords" not in cleaned or not cleaned.get("keywords"):
            keywords = self._extract_keywords(content, cleaned)
            if keywords:
                cleaned["keywords"] = ", ".join(keywords)
        
        # 6. 生成标题（如果没有）
        if "title" not in cleaned:
            cleaned["title"] = self._generate_title(content, cleaned)
        
        # 7. 标记文档类别
        cleaned["category"] = self._categorize_document(content, cleaned)
        
        return cleaned
    
    def _extract_company(self, filename: str, content: str) -> Optional[str]:
        """从文件名或内容提取公司名"""
        search_text = f"{filename} {content[:1000]}".lower()
        
        for company_name, patterns in self.COMPANY_PATTERNS:
            for pattern in patterns:
                if pattern in search_text:
                    return company_name
        
        return None
    
    def _generate_preview(self, content: str, max_length: int = 200) -> str:
        """生成内容预览"""
        # 去除多余空白
        preview = re.sub(r'\s+', ' ', content).strip()
        
        # 截断
        if len(preview) > max_length:
            preview = preview[:max_length] + "..."
        
        return preview
    
    def _extract_keywords(self, content: str, meta: Dict) -> List[str]:
        """提取关键词"""
        keywords = []
        content_lower = content.lower()
        
        # 添加公司名
        if meta.get("company"):
            keywords.append(meta["company"])
        
        # 添加时间信息
        if meta.get("date"):
            keywords.append(meta["date"])
        if meta.get("quarter"):
            keywords.append(meta["quarter"])
        
        # 检测财务关键词
        for kw in self.FINANCIAL_KEYWORDS:
            if kw in content_lower:
                keywords.append(kw)
                break
        
        # 检测技术关键词
        for kw in self.TECHNICAL_KEYWORDS:
            if kw in content_lower:
                keywords.append(kw)
                break
        
        # 提取大写缩写（如 API, BOP, MWD）
        acronyms = re.findall(r'\b[A-Z]{2,5}\b', content[:500])
        keywords.extend(list(set(acronyms))[:3])
        
        return list(set(keywords))[:8]  # 最多 8 个关键词
    
    def _generate_title(self, content: str, meta: Dict) -> str:
        """生成文档标题"""
        doc_type = meta.get("type", "document")
        
        # 如果有公司名和日期，生成描述性标题
        parts = []
        if meta.get("company"):
            parts.append(meta["company"])
        if meta.get("date"):
            parts.append(meta["date"])
        if meta.get("quarter"):
            parts.append(meta["quarter"])
        
        if parts:
            return f"{' '.join(parts)} - {doc_type}"
        
        # 否则用首行作为标题
        first_line = content.split('\n')[0].strip()
        if first_line and len(first_line) < 100:
            return first_line
        
        return f"Untitled {doc_type}"
    
    def _categorize_document(self, content: str, meta: Dict) -> str:
        """文档分类"""
        content_lower = content.lower()
        doc_type = meta.get("type", "")
        
        # 财务类
        financial_indicators = ["revenue", "profit", "ebitda", "margin", "usd", "billion", "million", "营收", "利润"]
        if any(ind in content_lower for ind in financial_indicators):
            return "financial"
        
        # 技术类
        if doc_type in ["markdown_section", "pdf_text"]:
            tech_indicators = ["drilling", "well", "rig", "completion", "钻井", "完井"]
            if any(ind in content_lower for ind in tech_indicators):
                return "technical"
        
        # 合同类
        if doc_type == "contract_clause":
            return "contract"
        
        # 数据表类
        if doc_type in ["excel_record", "pdf_table_record"]:
            return "data_table"
        
        # 图片类
        if doc_type == "image_caption":
            return "image"
        
        # 演示文稿类
        if doc_type == "ppt_slide":
            return "presentation"
        
        return "general"


def clean_metadata(docs: List[Document], verbose: bool = True) -> List[Document]:
    """便捷函数：清洗文档元数据"""
    cleaner = MetadataCleaner()
    return cleaner.clean(docs, verbose=verbose)
