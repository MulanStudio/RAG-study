#!/usr/bin/env python3
"""
🛢️ 油田服务 RAG 系统 - 主应用入口

使用方法:
    1. 把组委会数据放到 data/ 文件夹
    2. 运行: python app.py

或者指定数据目录:
    python app.py --data_dir /path/to/data

启动 Web UI:
    python app.py --mode web
"""

import os
import sys
import argparse
import uuid
import yaml
from typing import List, Dict

# 设置环境
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def load_config(config_path: str = "config/config.yaml") -> Dict:
    """加载配置文件"""
    default_config = {
        "data": {"root_dir": "data/"},
        "models": {
            "llm": {"model_name": "qwen2.5:3b", "base_url": "http://127.0.0.1:11434"},
            "embedding": {"model_name": "sentence-transformers/all-MiniLM-L6-v2"},
            "reranker": {"model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2"}
        },
        "indexing": {
            "chunk_size_parent": 2000,
            "chunk_overlap_parent": 200,
            "chunk_size_child": 400,
            "chunk_overlap_child": 50
        }
    }
    
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            # 合并默认配置
            for key in default_config:
                if key not in config:
                    config[key] = default_config[key]
            return config
    
    return default_config


class OilfieldRAG:
    """
    油田服务 RAG 系统主类
    
    Example:
        rag = OilfieldRAG(data_dir="data/")
        answer = rag.ask("What is the revenue of SLB?")
    """
    
    def __init__(self, data_dir: str = "data/", config_path: str = "config/config.yaml"):
        self.data_dir = data_dir
        self.config = load_config(config_path)
        self.config_path = config_path
        
        self.vectorstore = None
        self.bm25 = None
        self.reranker = None
        self.llm = None
        self.retriever = None
        self.generator = None
        self.docstore = {}
        self.splits = []
        
        self._initialized = False
    
    def initialize(self, verbose: bool = True):
        """初始化系统（加载数据、构建索引）"""
        if self._initialized:
            print("⚠️ 系统已初始化")
            return
        
        print("=" * 60)
        print("🛢️ 油田服务 RAG 系统 初始化")
        print("=" * 60)
        
        # 1. 加载数据
        print("\n📂 Step 1: 加载数据...")
        from src.loaders import load_all_documents
        
        # 检查 VLM
        vlm = self._init_vlm()
        
        docs = load_all_documents(self.data_dir, vlm=vlm, verbose=verbose)
        
        if not docs:
            print("❌ 未找到任何文档，请检查数据目录")
            return
        
        # 2. 文档切分 (Parent-Child)
        print("\n📑 Step 2: 文档切分...")
        self.splits, self.docstore = self._split_documents(docs)
        print(f"   生成 {len(self.splits)} 个子块")
        
        # 3. 构建向量索引
        print("\n🔍 Step 3: 构建向量索引...")
        self.vectorstore = self._build_vectorstore(self.splits)
        
        # 4. 构建 BM25 索引
        print("\n📝 Step 4: 构建 BM25 索引...")
        self.bm25 = self._build_bm25(self.splits)
        
        # 5. 加载 Reranker
        print("\n🎯 Step 5: 加载 Reranker...")
        self.reranker = self._load_reranker()
        
        # 6. 加载 LLM
        print("\n🤖 Step 6: 连接 LLM...")
        self.llm = self._init_llm()
        
        # 7. 初始化检索器和生成器
        print("\n⚙️ Step 7: 初始化检索器和生成器...")
        from src.retrieval import RAGRetriever
        from src.generation import create_generator
        
        self.retriever = RAGRetriever(
            vectorstore=self.vectorstore,
            bm25=self.bm25,
            splits=self.splits,
            reranker=self.reranker,
            docstore=self.docstore,
            llm=self.llm,
            config=self.config
        )
        
        self.generator = create_generator(self.llm, self.config_path)
        
        self._initialized = True
        
        print("\n" + "=" * 60)
        print("✅ 系统初始化完成!")
        print(f"   文档数: {len(docs)}")
        print(f"   索引块: {len(self.splits)}")
        print(f"   LLM: {'在线' if self.llm else '离线'}")
        print("=" * 60)
    
    def ask(self, question: str, verbose: bool = False) -> str:
        """
        问答接口
        
        Args:
            question: 用户问题
            verbose: 是否打印调试信息
        
        Returns:
            答案字符串
        """
        if not self._initialized:
            self.initialize()
        
        if verbose:
            print(f"\n❓ 问题: {question}")
        
        # 检索
        docs, retrieval_debug = self.retriever.retrieve(question, top_k=5)
        
        if verbose:
            print(f"📚 检索到 {len(docs)} 个文档")
            for i, doc in enumerate(docs[:3], 1):
                print(f"   {i}. {doc.page_content[:80]}...")
        
        # 生成
        answer, gen_debug = self.generator.generate(question, docs)
        
        if verbose:
            print(f"\n💬 答案: {answer[:200]}...")
        
        return answer
    
    def _split_documents(self, docs: List) -> tuple:
        """Parent-Child 切分"""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        cfg = self.config["indexing"]
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg["chunk_size_parent"],
            chunk_overlap=cfg["chunk_overlap_parent"]
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg["chunk_size_child"],
            chunk_overlap=cfg["chunk_overlap_child"]
        )
        
        docstore = {}
        child_docs = []
        
        for raw_doc in docs:
            # 切分为 Parent
            if len(raw_doc.page_content) > cfg["chunk_size_parent"]:
                parents = parent_splitter.split_documents([raw_doc])
            else:
                parents = [raw_doc]
            
            for parent in parents:
                parent_id = str(uuid.uuid4())
                parent.metadata["doc_id"] = parent_id
                docstore[parent_id] = parent
                
                # 切分为 Child
                children = child_splitter.split_documents([parent])
                for child in children:
                    child.metadata["parent_id"] = parent_id
                    child_docs.append(child)
        
        return child_docs, docstore
    
    def _build_vectorstore(self, docs: List):
        """构建向量数据库"""
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        
        model_name = self.config["models"]["embedding"]["model_name"]
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        return Chroma.from_documents(documents=docs, embedding=embeddings)
    
    def _build_bm25(self, docs: List):
        """构建 BM25 索引"""
        from rank_bm25 import BM25Okapi
        from src.text_processing import tokenize_text
        
        tokenized = [tokenize_text(doc.page_content) for doc in docs]
        return BM25Okapi(tokenized)
    
    def _load_reranker(self):
        """加载 Reranker"""
        from sentence_transformers import CrossEncoder
        
        model_name = self.config["models"]["reranker"]["model_name"]
        return CrossEncoder(model_name)
    
    def _init_llm(self):
        """初始化 LLM"""
        try:
            from langchain_ollama import ChatOllama
            
            cfg = self.config["models"]["llm"]
            llm = ChatOllama(
                model=cfg["model_name"],
                base_url=cfg["base_url"]
            )
            llm.invoke("hi")  # 测试连接
            print(f"   ✅ LLM 连接成功: {cfg['model_name']}")
            return llm
        except Exception as e:
            print(f"   ⚠️ LLM 连接失败: {e}")
            return None
    
    def _init_vlm(self):
        """初始化 VLM (图片理解)"""
        if not self.config.get("models", {}).get("vlm", {}).get("enabled", False):
            return None
        
        try:
            from langchain_ollama import ChatOllama
            
            cfg = self.config["models"]["vlm"]
            vlm = ChatOllama(
                model=cfg["model_name"],
                base_url=cfg["base_url"]
            )
            vlm.invoke("hi")
            print(f"   ✅ VLM 可用: {cfg['model_name']}")
            return vlm
        except:
            print("   ⚠️ VLM 不可用，图片将使用文件名作为描述")
            return None


def run_cli(rag: OilfieldRAG):
    """命令行交互模式"""
    print("\n🎮 进入交互模式 (输入 'quit' 退出)")
    print("-" * 40)
    
    while True:
        try:
            question = input("\n❓ 请输入问题: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 再见!")
                break
            
            if not question:
                continue
            
            answer = rag.ask(question, verbose=True)
            print(f"\n💬 答案:\n{answer}")
            
        except KeyboardInterrupt:
            print("\n👋 再见!")
            break


def run_web(rag: OilfieldRAG):
    """启动 Web UI"""
    print("\n🌐 启动 Web UI...")
    print("   请访问: http://localhost:8501")
    
    # 使用原有的 streamlit_app
    os.system("streamlit run streamlit_app.py")


def main():
    parser = argparse.ArgumentParser(description="油田服务 RAG 系统")
    parser.add_argument("--data_dir", type=str, default="data/", help="数据目录")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="配置文件")
    parser.add_argument("--mode", type=str, choices=["cli", "web"], default="cli", help="运行模式")
    parser.add_argument("--question", type=str, help="直接提问（非交互模式）")
    
    args = parser.parse_args()
    
    # 创建 RAG 实例
    rag = OilfieldRAG(data_dir=args.data_dir, config_path=args.config)
    rag.initialize()
    
    if args.question:
        # 直接回答问题
        answer = rag.ask(args.question, verbose=True)
        print(f"\n💬 答案:\n{answer}")
    elif args.mode == "web":
        run_web(rag)
    else:
        run_cli(rag)


if __name__ == "__main__":
    main()
