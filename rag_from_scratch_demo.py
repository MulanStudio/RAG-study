import os
import sys

# 检查 OpenAI API Key
if "OPENAI_API_KEY" not in os.environ:
    print("⚠️  警告: 未检测到 OPENAI_API_KEY 环境变量。")
    print("请先设置您的 API Key，例如: export OPENAI_API_KEY='sk-...'")
    print("为了演示代码逻辑，程序将继续运行，但在调用 LLM 时可能会失败。\n")

try:
    from langchain_community.document_loaders import (
        TextLoader, 
        DirectoryLoader, 
        PyPDFLoader, 
        UnstructuredExcelLoader,
        Docx2txtLoader
    )
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    # from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    # 直接引入 sentence-transformers 的原生 CrossEncoder
    from sentence_transformers import CrossEncoder
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
except ImportError as e:
    print(f"缺少必要的库: {e}")
    print("请运行以下命令安装：")
    print("pip install langchain langchain-community langchain-huggingface langchain-chroma chromadb sentence-transformers pypdf openpyxl docx2txt unstructured rank_bm25")
    sys.exit(1)

def run_rag_demo():
    print("--- 1. Indexing (索引阶段) ---")
    
    # 1.1 Load (加载文档) - 支持多格式
    print("正在加载 knowledge_base/ 和 downloads/ 目录下的文档...")
    
    loaders = [
        # 加载 Markdown
        DirectoryLoader("knowledge_base", glob="**/*.md", loader_cls=TextLoader),
        # 加载下载的 PDF
        DirectoryLoader("downloads", glob="**/*.pdf", loader_cls=PyPDFLoader),
        # 加载 Excel
        DirectoryLoader("downloads", glob="**/*.xlsx", loader_cls=UnstructuredExcelLoader),
        # 加载 Word
        DirectoryLoader("downloads", glob="**/*.docx", loader_cls=Docx2txtLoader),
    ]
    
    docs = []
    for loader in loaders:
        try:
            loaded_docs = loader.load()
            docs.extend(loaded_docs)
            print(f"✅ 成功加载: {loader.glob} (数量: {len(loaded_docs)})")
        except Exception as e:
            print(f"⚠️  加载失败 {loader.glob}: {e}")

    if not docs:
        print("❌ 未找到任何文档，请检查目录。")
        return

    print(f"总计加载文档数: {len(docs)}")
    print(f"文档总字符数: {sum(len(d.page_content) for d in docs)}")

    # 1.2 Split (文档切分)
    # 使用 RecursiveCharacterTextSplitter 智能切分，保持语义完整
    print("正在切分文档...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(docs)
    print(f"文档已切分为 {len(splits)} 个片段 (Chunks)。")

    # 1.3 Embed & Store (向量化与存储)
    # 将文本片段转换为向量，并存入 Chroma 向量数据库
    print("正在进行向量化并存入 ChromaDB (使用免费的 HuggingFace 模型)...")
    try:
        # 使用本地 CPU 运行的免费模型，不需要 API Key
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings
        )
        
        # 1.4 加载 Reranker 模型
        print("正在加载 Reranker 模型 (BAAI/bge-reranker-base)...")
        # 直接使用 CrossEncoder，不依赖 LangChain 封装
        reranker = CrossEncoder("BAAI/bge-reranker-base")
        print("向量数据库与重排序器 (Reranker) 构建完成。")
            
    except Exception as e:
        print(f"❌ 向量化失败: {e}")
        return

    print("\n--- 2. Retrieval Only (仅演示检索阶段) ---")
    print("由于没有本地大模型 (LLM)，我们将直接展示 RAG 检索到的知识片段。")
    print("这一步是 RAG 成功的关键：如果找到了正确片段，LLM 只需要把它们润色一下。")

    # 2.4 Invoke (执行查询)
    questions = [
        "全球最大的油服公司是谁？",
        "什么是水力压裂（Fracking）？它有什么作用？",
        "Schlumberger 的 2023 年营收是多少？(请查找 Excel 数据)",
        "合同中规定的钻井日费率(day rate)是多少？(请查找 Word 合同)",
        "中海油服(COSL)的业务概况是什么？(请查找 PDF 年报)",
    ]

    for q in questions:
        print(f"\n" + "="*40)
        print(f"用户提问: {q}")
        print("-" * 40)
        
        # 1. 第一步：粗排 (召回 Top 20)
        print("🔍 1. 初步检索 (Recall Top 20)...")
        initial_docs = vectorstore.similarity_search(q, k=20)
        
        # 2. 第二步：精排 (Rerank)
        print("🔍 2. 重排序 (Reranking)...")
        # 构造 input pairs: [[query, doc_text1], [query, doc_text2], ...]
        pairs = [[q, doc.page_content] for doc in initial_docs]
        
        # 计算分数
        scores = reranker.predict(pairs)
        
        # 将文档和分数打包，并按分数降序排列
        doc_score_pairs = list(zip(initial_docs, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # 取 Top 3
        top_k_docs = doc_score_pairs[:3]
        
        for i, (doc, score) in enumerate(top_k_docs):
            print(f"\n[排名 #{i+1} | 相关性得分: {score:.4f}]:")
            # 打印源文件名 (如果有元数据)
            source = doc.metadata.get('source', 'unknown')
            print(f"来源: {source}")
            print(f"内容: {doc.page_content[:300].replace(chr(10), ' ')}...") # 替换换行符以便展示
            
    print("\n" + "="*40)
    print("✅ 演示结束。")
    print("原理解释: 我们成功找到了问题的答案所在位置。")
    print("如果此时接入一个 LLM (如 GPT-4 或本地 Ollama)，它就会阅读上述片段并输出通顺的自然语言回答。")

if __name__ == "__main__":
    run_rag_demo()

