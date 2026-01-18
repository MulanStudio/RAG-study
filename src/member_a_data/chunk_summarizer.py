"""
文本块摘要模块 - 为每个 chunk 生成精炼摘要

负责人：成员A（数据工程师）+ 成员C（Prompt 工程师）

功能：
1. 为文本块生成精炼摘要（LLM）
2. 根据文档类型选择不同的摘要策略
3. 摘要可用于增强检索（在原文前添加）
"""

import os
import sys
import logging
from typing import List, Dict, Optional
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# 默认摘要 Prompt 模板
DEFAULT_PROMPTS = {
    "excel_record": """Summarize this data record in ONE sentence.
Focus on: entity name, key metrics (with numbers), time period.
Be specific and include actual values.

Record: {content}

Summary (one sentence):""",

    "pdf_table_record": """Summarize this table data in ONE sentence.
Focus on: what data it contains, key values, entity names.
Include specific numbers if present.

Table data: {content}

Summary (one sentence):""",

    "ppt_slide": """Summarize this presentation slide in ONE sentence.
Focus on: main message, key data points.

Slide: {content}

Summary (one sentence):""",

    "contract_clause": """Summarize this contract clause in ONE sentence.
Focus on: what it defines, key terms, obligations.

Clause: {content}

Summary (one sentence):""",

    "image_caption": """Summarize what this image shows in ONE sentence.
Focus on: type of content, key information visible.

Image description: {content}

Summary (one sentence):""",

    "default": """Summarize this text in 2-3 sentences.
Focus on: main topic, key facts, important numbers or metrics.
Be concise and specific.

Text: {content}

Summary (2-3 sentences):"""
}


class ChunkSummarizer:
    """
    文本块摘要生成器
    
    策略：
    1. 短文本（<300字）：跳过摘要，直接用原文
    2. 中等文本（300-2000字）：生成 1-2 句摘要
    3. 长文本（>2000字）：生成 2-3 句结构化摘要
    
    Example:
        summarizer = ChunkSummarizer(llm=my_llm)
        summarized_docs = summarizer.summarize(docs)
    """
    
    def __init__(
        self,
        llm=None,
        prompts: Dict[str, str] = None,
        min_length: int = 300,
        max_input_length: int = 3000,
        prepend_summary: bool = True,
        batch_size: int = 10
    ):
        """
        Args:
            llm: LangChain LLM 实例
            prompts: 自定义摘要 Prompt（按文档类型）
            min_length: 低于此长度的文档跳过摘要
            max_input_length: 输入到 LLM 的最大长度
            prepend_summary: 是否将摘要添加到原文开头
            batch_size: 批处理大小（预留，用于并行）
        """
        self.llm = llm
        self.prompts = prompts or DEFAULT_PROMPTS
        self.min_length = min_length
        self.max_input_length = max_input_length
        self.prepend_summary = prepend_summary
        self.batch_size = batch_size
    
    def summarize(
        self,
        docs: List[Document],
        verbose: bool = True
    ) -> List[Document]:
        """
        为每个文档生成摘要
        
        Args:
            docs: 文档列表
            verbose: 是否打印进度
            
        Returns:
            添加了摘要的文档列表
        """
        if not self.llm:
            logger.warning("No LLM provided, skipping summarization")
            return self._fallback_summarize(docs)
        
        summarized = []
        stats = {"total": 0, "skipped": 0, "llm_generated": 0, "fallback": 0}
        
        for i, doc in enumerate(docs):
            content_len = len(doc.page_content)
            stats["total"] += 1
            
            # 短文本：跳过
            if content_len < self.min_length:
                doc.metadata["summary"] = doc.page_content[:200]
                doc.metadata["summary_type"] = "skipped_short"
                summarized.append(doc)
                stats["skipped"] += 1
                continue
            
            # 中长文本：生成摘要
            try:
                summary = self._generate_summary(doc)
                doc.metadata["summary"] = summary
                doc.metadata["summary_type"] = "llm_generated"
                
                # 可选：将摘要添加到原文开头（增强检索）
                if self.prepend_summary and summary:
                    doc.page_content = f"[Summary: {summary}]\n\n{doc.page_content}"
                
                stats["llm_generated"] += 1
                
            except Exception as e:
                logger.warning(f"Summarization failed for doc {i}: {e}")
                doc.metadata["summary"] = doc.page_content[:200]
                doc.metadata["summary_type"] = "fallback"
                stats["fallback"] += 1
            
            summarized.append(doc)
            
            # 进度打印
            if verbose and (i + 1) % 50 == 0:
                print(f"📝 Summarized {i + 1}/{len(docs)} chunks")
        
        if verbose:
            logger.info(f"Summarization complete: {stats['total']} docs")
            logger.info(f"  - Skipped (short): {stats['skipped']}")
            logger.info(f"  - LLM generated: {stats['llm_generated']}")
            logger.info(f"  - Fallback: {stats['fallback']}")
        
        return summarized
    
    def _generate_summary(self, doc: Document) -> str:
        """生成单个文档的摘要"""
        doc_type = doc.metadata.get("type", "default")
        content = doc.page_content[:self.max_input_length]
        
        # 获取对应类型的 Prompt
        prompt_template = self.prompts.get(doc_type, self.prompts["default"])
        prompt = prompt_template.format(content=content)
        
        # 调用 LLM
        response = self.llm.invoke(prompt)
        summary = response.content.strip()
        
        # 清理摘要
        summary = self._clean_summary(summary)
        
        return summary[:500]  # 限制摘要长度
    
    def _clean_summary(self, summary: str) -> str:
        """清理摘要文本"""
        # 去除常见的 LLM 前缀
        prefixes_to_remove = [
            "Summary:", "Here is the summary:", "The summary is:",
            "This text summarizes:", "In summary,",
            "摘要：", "总结：", "本文摘要：",
        ]
        
        for prefix in prefixes_to_remove:
            if summary.lower().startswith(prefix.lower()):
                summary = summary[len(prefix):].strip()
                break
        
        # 去除引号
        if summary.startswith('"') and summary.endswith('"'):
            summary = summary[1:-1]
        
        return summary.strip()
    
    def _fallback_summarize(self, docs: List[Document]) -> List[Document]:
        """无 LLM 时的回退方案：使用首句/首行作为摘要"""
        for doc in docs:
            content = doc.page_content
            
            # 尝试取首句
            import re
            sentences = re.split(r'(?<=[.!?。！？])\s+', content[:500])
            if sentences:
                summary = sentences[0][:200]
            else:
                summary = content[:200]
            
            doc.metadata["summary"] = summary
            doc.metadata["summary_type"] = "no_llm_fallback"
        
        return docs


class CachedChunkSummarizer(ChunkSummarizer):
    """
    带缓存的摘要生成器
    
    使用文档内容的 hash 作为 key，避免重复生成摘要
    """
    
    def __init__(self, cache_dir: str = ".summary_cache", **kwargs):
        super().__init__(**kwargs)
        self.cache_dir = cache_dir
        self._cache = {}
        self._load_cache()
    
    def _load_cache(self):
        """加载缓存"""
        import json
        cache_file = os.path.join(self.cache_dir, "summaries.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self._cache = json.load(f)
                logger.info(f"Loaded {len(self._cache)} cached summaries")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self._cache = {}
    
    def _save_cache(self):
        """保存缓存"""
        import json
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = os.path.join(self.cache_dir, "summaries.json")
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(self._cache)} summaries to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _get_content_hash(self, content: str) -> str:
        """计算内容 hash"""
        import hashlib
        return hashlib.md5(content[:1000].encode()).hexdigest()[:16]
    
    def summarize(self, docs: List[Document], verbose: bool = True) -> List[Document]:
        """带缓存的摘要生成"""
        if not self.llm:
            return self._fallback_summarize(docs)
        
        summarized = []
        stats = {"total": 0, "cached": 0, "new": 0}
        
        for i, doc in enumerate(docs):
            content_hash = self._get_content_hash(doc.page_content)
            stats["total"] += 1
            
            # 检查缓存
            if content_hash in self._cache:
                summary = self._cache[content_hash]
                doc.metadata["summary"] = summary
                doc.metadata["summary_type"] = "cached"
                
                if self.prepend_summary and summary:
                    doc.page_content = f"[Summary: {summary}]\n\n{doc.page_content}"
                
                stats["cached"] += 1
            else:
                # 生成新摘要
                if len(doc.page_content) < self.min_length:
                    summary = doc.page_content[:200]
                    doc.metadata["summary_type"] = "skipped_short"
                else:
                    try:
                        summary = self._generate_summary(doc)
                        doc.metadata["summary_type"] = "llm_generated"
                        
                        if self.prepend_summary and summary:
                            doc.page_content = f"[Summary: {summary}]\n\n{doc.page_content}"
                    except Exception as e:
                        logger.warning(f"Summarization failed: {e}")
                        summary = doc.page_content[:200]
                        doc.metadata["summary_type"] = "fallback"
                
                doc.metadata["summary"] = summary
                self._cache[content_hash] = summary
                stats["new"] += 1
            
            summarized.append(doc)
            
            if verbose and (i + 1) % 50 == 0:
                print(f"📝 Summarized {i + 1}/{len(docs)} chunks")
        
        # 保存缓存
        if stats["new"] > 0:
            self._save_cache()
        
        if verbose:
            logger.info(f"Summarization complete: {stats['total']} docs")
            logger.info(f"  - Cached: {stats['cached']}")
            logger.info(f"  - New: {stats['new']}")
        
        return summarized


def summarize_chunks(
    docs: List[Document],
    llm=None,
    use_cache: bool = True,
    verbose: bool = True
) -> List[Document]:
    """
    便捷函数：为文档生成摘要
    
    Args:
        docs: 文档列表
        llm: LangChain LLM 实例
        use_cache: 是否使用缓存
        verbose: 是否打印进度
        
    Returns:
        添加了摘要的文档列表
    """
    if use_cache:
        summarizer = CachedChunkSummarizer(llm=llm)
    else:
        summarizer = ChunkSummarizer(llm=llm)
    
    return summarizer.summarize(docs, verbose=verbose)
