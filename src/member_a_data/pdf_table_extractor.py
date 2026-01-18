"""
PDF Table Extractor Module
PDF 表格提取模块 - 专门优化 PDF 中表格数据的提取

问题背景：
PyPDFLoader 只能提取纯文本，会完全丢失表格结构：
  原始 PDF 表格:
  | Company | Revenue | Growth |
  |---------|---------|--------|
  | SLB     | 33.1B   | 12%    |
  
  PyPDFLoader 提取结果:
  "Company Revenue Growth SLB 33.1B 12%"  ← 结构完全丢失！

解决方案：
1. 使用 pdfplumber 提取表格结构
2. 将表格转换为结构化文本（Markdown 或自然语言）
3. 保留行列关系，便于后续检索

依赖安装：
pip install pdfplumber
"""

import os
from typing import List, Dict, Optional, Tuple
from langchain_core.documents import Document


class PDFTableExtractor:
    """PDF 表格提取器"""
    
    def __init__(self, use_natural_language: bool = True):
        """
        初始化 PDF 表格提取器
        
        Args:
            use_natural_language: 是否将表格转换为自然语言描述
                                  True: "SLB's revenue in 2023 was 33.1 Billion USD"
                                  False: "| Company | Revenue | ... |"
        """
        self.use_natural_language = use_natural_language
        self._check_dependencies()
    
    def _check_dependencies(self):
        """检查依赖是否安装"""
        try:
            import pdfplumber
            self.pdfplumber = pdfplumber
            self.has_pdfplumber = True
        except ImportError:
            self.has_pdfplumber = False
            print("⚠️ pdfplumber 未安装，将使用 fallback 方法")
            print("   安装命令: pip install pdfplumber")
    
    def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        从 PDF 中提取所有表格
        
        Args:
            pdf_path: PDF 文件路径
        
        Returns:
            [{"page": 1, "table_index": 0, "headers": [...], "rows": [[...], ...], "raw": ...}, ...]
        """
        if not self.has_pdfplumber:
            return []
        
        tables = []
        try:
            with self.pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_tables = page.extract_tables()
                    
                    for table_idx, table in enumerate(page_tables):
                        if not table or len(table) < 2:
                            continue  # 跳过空表格或只有一行的表格
                        
                        # 第一行通常是表头
                        headers = [str(cell).strip() if cell else "" for cell in table[0]]
                        rows = []
                        
                        for row in table[1:]:
                            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                            if any(cleaned_row):  # 跳过全空行
                                rows.append(cleaned_row)
                        
                        if headers and rows:
                            tables.append({
                                "page": page_num,
                                "table_index": table_idx,
                                "headers": headers,
                                "rows": rows,
                                "raw": table
                            })
        
        except Exception as e:
            print(f"⚠️ PDF 表格提取失败 {pdf_path}: {e}")
        
        return tables
    
    def table_to_natural_language(self, table_info: Dict, source_file: str = "") -> List[str]:
        """
        将表格转换为自然语言描述（每行一条记录）
        
        这种格式对 RAG 检索更友好，因为：
        1. 语义完整：每条记录包含完整的列名和值
        2. 易于匹配：用户问 "SLB revenue" 能匹配到 "SLB's revenue was..."
        """
        results = []
        headers = table_info["headers"]
        rows = table_info["rows"]
        
        for row in rows:
            # 构建自然语言描述
            parts = []
            for header, value in zip(headers, row):
                if header and value and value.lower() not in ["", "n/a", "unknown", "-"]:
                    # 清理 header
                    header_clean = header.replace("_", " ").strip()
                    parts.append(f"{header_clean}: {value}")
            
            if parts:
                # 添加来源信息
                source_info = f"[Source: {os.path.basename(source_file)}, Page {table_info['page']}]" if source_file else ""
                record = f"Table Record {source_info}: " + " | ".join(parts)
                results.append(record)
        
        return results
    
    def table_to_markdown(self, table_info: Dict) -> str:
        """
        将表格转换为 Markdown 格式
        
        适用于需要保留完整表格结构的场景
        """
        headers = table_info["headers"]
        rows = table_info["rows"]
        
        # 构建 Markdown 表格
        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        for row in rows:
            # 确保 row 长度与 headers 一致
            padded_row = row + [""] * (len(headers) - len(row))
            lines.append("| " + " | ".join(padded_row[:len(headers)]) + " |")
        
        return "\n".join(lines)
    
    def load_pdf_with_tables(self, pdf_path: str) -> List[Document]:
        """
        加载 PDF 并提取表格为 Document 列表
        
        这是主要的接口函数，可以直接替换 PyPDFLoader
        
        Returns:
            Document 列表，每个表格行是一个 Document
        """
        documents = []
        
        # 提取表格
        tables = self.extract_tables_from_pdf(pdf_path)
        
        if not tables:
            # 如果没有提取到表格，使用 fallback
            return self._fallback_load(pdf_path)
        
        print(f"✅ 从 {os.path.basename(pdf_path)} 提取到 {len(tables)} 个表格")
        
        for table_info in tables:
            if self.use_natural_language:
                # 转换为自然语言记录
                records = self.table_to_natural_language(table_info, pdf_path)
                for record in records:
                    doc = Document(
                        page_content=record,
                        metadata={
                            "source": pdf_path,
                            "type": "pdf_table_record",
                            "page": table_info["page"],
                            "table_index": table_info["table_index"]
                        }
                    )
                    documents.append(doc)
            else:
                # 转换为 Markdown 表格
                md_table = self.table_to_markdown(table_info)
                doc = Document(
                    page_content=md_table,
                    metadata={
                        "source": pdf_path,
                        "type": "pdf_table_markdown",
                        "page": table_info["page"],
                        "table_index": table_info["table_index"]
                    }
                )
                documents.append(doc)
        
        # 同时也加载非表格文本（使用 PyPDFLoader）
        text_docs = self._load_text_content(pdf_path)
        documents.extend(text_docs)
        
        return documents
    
    def _load_text_content(self, pdf_path: str) -> List[Document]:
        """使用 PyPDFLoader 加载文本内容"""
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            # 标记为文本类型
            for doc in docs:
                doc.metadata["type"] = "pdf_text"
            return docs
        except Exception as e:
            print(f"⚠️ PDF 文本加载失败 {pdf_path}: {e}")
            return []
    
    def _fallback_load(self, pdf_path: str) -> List[Document]:
        """Fallback: 当 pdfplumber 不可用时使用"""
        print(f"   使用 fallback 方法加载 {os.path.basename(pdf_path)}")
        return self._load_text_content(pdf_path)


def load_pdfs_with_table_extraction(directory: str) -> List[Document]:
    """
    便捷函数：加载目录下所有 PDF，自动提取表格
    
    可以直接替换原有的 PDF 加载逻辑：
    
    旧代码:
        pdf_loader = DirectoryLoader("data", glob="**/*.pdf", loader_cls=PyPDFLoader)
        docs.extend(pdf_loader.load())
    
    新代码:
        docs.extend(load_pdfs_with_table_extraction("data"))
    """
    import glob
    
    extractor = PDFTableExtractor(use_natural_language=True)
    all_docs = []
    
    pdf_files = glob.glob(os.path.join(directory, "**/*.pdf"), recursive=True)
    
    for pdf_path in pdf_files:
        try:
            docs = extractor.load_pdf_with_tables(pdf_path)
            all_docs.extend(docs)
            print(f"   📄 {os.path.basename(pdf_path)}: {len(docs)} documents")
        except Exception as e:
            print(f"   ⚠️ 加载失败 {pdf_path}: {e}")
    
    return all_docs


# ============ 测试函数 ============

def test_table_conversion():
    """测试表格转换功能"""
    print("=" * 60)
    print("🧪 Testing PDF Table Extraction (Mock Data)")
    print("=" * 60)
    
    # 模拟提取到的表格数据
    mock_table = {
        "page": 1,
        "table_index": 0,
        "headers": ["Company", "Revenue_2023", "Growth_Rate", "Region"],
        "rows": [
            ["SLB", "33.1B USD", "12%", "Global"],
            ["Halliburton", "23.0B USD", "8%", "Americas"],
            ["Baker Hughes", "25.5B USD", "10%", "Global"],
        ],
        "raw": None
    }
    
    extractor = PDFTableExtractor(use_natural_language=True)
    
    # 测试自然语言转换
    print("\n📝 Natural Language Format:")
    nl_records = extractor.table_to_natural_language(mock_table, "test.pdf")
    for record in nl_records:
        print(f"   {record}")
    
    # 测试 Markdown 转换
    print("\n📝 Markdown Format:")
    md_table = extractor.table_to_markdown(mock_table)
    print(md_table)
    
    # 验证
    assert len(nl_records) == 3, "Should have 3 records"
    assert "SLB" in nl_records[0], "First record should contain SLB"
    assert "33.1B USD" in nl_records[0], "First record should contain revenue"
    
    print("\n✅ Table Conversion Test Passed!")
    return True


def test_pdf_extraction():
    """测试实际 PDF 提取（需要 pdfplumber）"""
    print("\n" + "=" * 60)
    print("🧪 Testing PDF Extraction (Real PDF)")
    print("=" * 60)
    
    extractor = PDFTableExtractor(use_natural_language=True)
    
    if not extractor.has_pdfplumber:
        print("⚠️ pdfplumber 未安装，跳过实际 PDF 测试")
        return True
    
    # 尝试加载测试 PDF
    test_pdf = "data/China_Oilfield_Services_Annual_Report.pdf"
    if os.path.exists(test_pdf):
        docs = extractor.load_pdf_with_tables(test_pdf)
        print(f"\n📄 Loaded {len(docs)} documents from {test_pdf}")
        
        # 展示前几个文档
        for i, doc in enumerate(docs[:3]):
            print(f"\n[Doc {i+1}] Type: {doc.metadata.get('type', 'unknown')}")
            print(f"   Content: {doc.page_content[:150]}...")
    else:
        print(f"⚠️ 测试文件不存在: {test_pdf}")
    
    print("\n✅ PDF Extraction Test Complete!")
    return True


if __name__ == "__main__":
    test_table_conversion()
    test_pdf_extraction()
    print("\n🎉 All PDF extraction tests complete!")
