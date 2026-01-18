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
import logging
from typing import List, Dict

# 设置环境
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))


def load_config(config_path: str = "config/config.yaml") -> Dict:
    """加载配置文件"""
    default_config = {
        "data": {"root_dir": "data/"},
        "models": {
            "llm": {"provider": "azure_openai", "model_name": "gpt-5-chat", "base_url": ""},
            "embedding": {"provider": "azure_openai", "model_name": "text-embedding-3-large"},
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
        from src.member_a_data.loaders import load_all_documents
        
        # 检查 VLM
        vlm = self._init_vlm()
        
        docs = load_all_documents(self.data_dir, vlm=vlm, verbose=verbose)
        
        if not docs:
            print("❌ 未找到任何文档，请检查数据目录")
            return
        
        # 1.5. 预处理：元数据清洗 + 摘要生成
        docs = self._preprocess_documents(docs, verbose=verbose)
        
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
        from src.member_b_retrieval.retrieval import RAGRetriever
        from src.member_c_generation.generation import create_generator
        
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
        retrieval_score = retrieval_debug.get("max_similarity_score", 1.0)
        core_query = retrieval_debug.get("core_query", question)
        
        if verbose:
            print(f"📚 检索到 {len(docs)} 个文档 (相似度: {retrieval_score:.2f})")
            if core_query != question:
                print(f"   💡 核心问题: {core_query}")
            for i, doc in enumerate(docs[:3], 1):
                print(f"   {i}. {doc.page_content[:80]}...")
        
        # 生成（传入检索分数和核心问题用于置信度判断和对齐检查）
        answer, gen_debug = self.generator.generate(
            question, docs, 
            retrieval_score=retrieval_score,
            core_query=core_query
        )
        
        if verbose:
            print(f"\n💬 答案: {answer[:200]}...")
        
        return answer
    
    def _preprocess_documents(self, docs: List, verbose: bool = True) -> List:
        """
        预处理文档：元数据清洗 + 摘要生成
        
        Args:
            docs: 原始文档列表
            verbose: 是否打印进度
            
        Returns:
            预处理后的文档列表
        """
        preprocess_cfg = self.config.get("preprocessing", {})
        
        # 1. 元数据清洗
        if preprocess_cfg.get("enable_metadata_cleaning", True):
            print("\n🧹 Step 1.5a: 元数据清洗...")
            from src.member_a_data.metadata_cleaner import clean_metadata
            docs = clean_metadata(docs, verbose=verbose)
            print(f"   清洗完成：{len(docs)} 个文档")
        
        # 2. 文本块摘要（可选，依赖 LLM）
        if preprocess_cfg.get("enable_summarization", False):
            print("\n📝 Step 1.5b: 生成文本块摘要...")
            
            # 初始化 LLM（如果还没有）
            if not self.llm:
                self.llm = self._init_llm()
            
            # 获取摘要配置
            sum_cfg = preprocess_cfg.get("summarization", {})
            
            from src.member_a_data.chunk_summarizer import CachedChunkSummarizer
            
            # 从配置读取 prompts
            prompts = sum_cfg.get("prompts", None)
            
            summarizer = CachedChunkSummarizer(
                llm=self.llm,
                prompts=prompts,
                min_length=sum_cfg.get("min_length", 300),
                max_input_length=sum_cfg.get("max_input_length", 3000),
                prepend_summary=sum_cfg.get("prepend_summary", True),
                cache_dir=sum_cfg.get("cache_dir", ".summary_cache")
            )
            
            docs = summarizer.summarize(docs, verbose=verbose)
            print(f"   摘要生成完成：{len(docs)} 个文档")
        
        return docs
    
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
    
    def _compute_data_fingerprint(self, docs: List) -> str:
        """计算数据指纹，用于判断是否需要重建索引"""
        import hashlib
        # 使用文档数量 + 前100个文档的前100字符作为指纹
        fingerprint_data = f"{len(docs)}:"
        for doc in docs[:100]:
            fingerprint_data += doc.page_content[:100]
        return hashlib.md5(fingerprint_data.encode()).hexdigest()[:16]
    
    def _get_embeddings(self):
        """获取 embedding 模型实例"""
        emb_cfg = self.config["models"]["embedding"]
        provider = emb_cfg.get("provider", "huggingface")
        
        if provider == "azure_openai":
            from src.member_e_system.azure_openai_client import (
                create_azure_openai_client,
                AzureOpenAIEmbeddings,
                load_azure_settings,
            )
            azure_cfg = load_azure_settings(self.config)
            client = create_azure_openai_client(
                azure_cfg["team_domain"],
                azure_cfg["api_key"]
            )
            return AzureOpenAIEmbeddings(client, azure_cfg["embedding_model"])
        elif provider == "huggingface_local":
            # 本地 GPU 加速模型
            from langchain_huggingface import HuggingFaceEmbeddings
            model_name = emb_cfg.get("model_name", "BAAI/bge-large-en-v1.5")
            device = emb_cfg.get("device", "cuda")
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': device},
                encode_kwargs={'batch_size': 256, 'normalize_embeddings': True}
            )
        else:
            from langchain_huggingface import HuggingFaceEmbeddings
            model_name = emb_cfg["model_name"]
            return HuggingFaceEmbeddings(model_name=model_name)
    
    def _build_vectorstore(self, docs: List):
        """构建向量数据库（支持持久化缓存）"""
        from langchain_chroma import Chroma
        import time
        
        # 过滤空内容，避免 embedding 报错
        docs = [doc for doc in docs if doc.page_content and doc.page_content.strip()]
        print(f"   📦 向量构建输入文档数: {len(docs)}")
        
        # 获取持久化配置
        sys_cfg = self.config.get("system", {}).get("vector_db", {})
        persist_dir = sys_cfg.get("persist_dir", ".cache/chroma_db")
        force_rebuild = sys_cfg.get("force_rebuild", False)
        
        # 计算数据指纹
        data_fingerprint = self._compute_data_fingerprint(docs)
        hash_file = os.path.join(persist_dir, "data_fingerprint.txt")
        
        # 获取 embedding 模型
        embeddings = self._get_embeddings()
        
        # 检查是否可以从缓存加载
        if os.path.exists(persist_dir) and not force_rebuild:
            if os.path.exists(hash_file):
                with open(hash_file, 'r') as f:
                    cached_fingerprint = f.read().strip()
                if cached_fingerprint == data_fingerprint:
                    print("   💾 从缓存加载向量索引...")
                    start = time.time()
                    vectorstore = Chroma(
                        persist_directory=persist_dir,
                        embedding_function=embeddings
                    )
                    elapsed = time.time() - start
                    print(f"   ✅ 缓存加载完成，用时 {elapsed:.1f}s")
                    return vectorstore
                else:
                    print("   🔄 数据已变化，重建索引...")
            else:
                print("   🔄 缓存无指纹文件，重建索引...")
        
        # 重新构建索引
        start = time.time()
        print("   ⏱️ 开始构建 Chroma 向量索引...")
        
        # 确保目录存在
        os.makedirs(persist_dir, exist_ok=True)
        
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        
        # 保存数据指纹
        with open(hash_file, 'w') as f:
            f.write(data_fingerprint)
        
        elapsed = time.time() - start
        print(f"   ✅ 向量索引完成并持久化，用时 {elapsed:.1f}s")
        return vectorstore
    
    def _build_bm25(self, docs: List):
        """构建 BM25 索引"""
        from rank_bm25 import BM25Okapi
        from src.member_b_retrieval.text_processing import tokenize_text
        
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
            cfg = self.config["models"]["llm"]
            provider = cfg.get("provider", "ollama")
            if provider == "azure_openai":
                from src.member_e_system.azure_openai_client import create_azure_openai_client, AzureOpenAIChat, load_azure_settings
                azure_cfg = load_azure_settings(self.config)
                client = create_azure_openai_client(
                    azure_cfg["team_domain"],
                    azure_cfg["api_key"]
                )
                model = azure_cfg["completion_model"] or cfg["model_name"]
                llm = AzureOpenAIChat(client, model, temperature=cfg.get("temperature", 0.1))
                llm.invoke("hi")
                print(f"   ✅ LLM 连接成功 (Azure): {model}")
                return llm
            else:
                from langchain_ollama import ChatOllama
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
        except Exception as e:
            print(f"   ⚠️ VLM 不可用，图片将使用文件名作为描述 ({type(e).__name__})")
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
    """启动 Web UI（已弃用）"""
    print("\n⚠️ Web UI 模式已弃用")
    print("   请使用命令行模式: python src/member_e_system/app.py --question '你的问题'")
    print("   或交互模式: python src/member_e_system/app.py")
    sys.exit(1)


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
