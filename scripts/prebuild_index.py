#!/usr/bin/env python3
"""
🛢️ 预构建向量索引脚本

使用场景：比赛前一晚运行，提前构建好索引，比赛当天秒开。

使用方法:
    python scripts/prebuild_index.py --data_dir competition_data/
    
或使用默认 data/ 目录:
    python scripts/prebuild_index.py
"""

import os
import sys
import argparse
import time

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.member_e_system.app import OilfieldRAG


def prebuild(data_dir: str, config_path: str = "config/config.yaml"):
    """预构建向量索引"""
    print("=" * 60)
    print("🛢️ 预构建向量索引")
    print("=" * 60)
    print(f"   数据目录: {data_dir}")
    print(f"   配置文件: {config_path}")
    print()
    
    start_time = time.time()
    
    # 创建 RAG 实例并初始化
    rag = OilfieldRAG(data_dir=data_dir, config_path=config_path)
    rag.initialize()
    
    elapsed = time.time() - start_time
    
    print()
    print("=" * 60)
    print("✅ 预构建完成!")
    print(f"   总耗时: {elapsed:.1f} 秒")
    print(f"   文档数: {len(rag.splits)}")
    print()
    print("📌 比赛当天启动时将自动加载缓存，秒开！")
    print("=" * 60)
    
    # 健康检查
    print()
    print("🔍 执行健康检查...")
    health_check(rag)


def health_check(rag: OilfieldRAG):
    """系统健康检查"""
    checks = []
    
    # 1. 向量索引
    if rag.vectorstore:
        checks.append(("向量索引", "✅", "已加载"))
    else:
        checks.append(("向量索引", "❌", "未加载"))
    
    # 2. BM25 索引
    if rag.bm25:
        checks.append(("BM25 索引", "✅", "已加载"))
    else:
        checks.append(("BM25 索引", "❌", "未加载"))
    
    # 3. Reranker
    if rag.reranker:
        checks.append(("Reranker", "✅", "已加载"))
    else:
        checks.append(("Reranker", "⚠️", "未加载"))
    
    # 4. LLM
    if rag.llm:
        checks.append(("LLM", "✅", "在线"))
    else:
        checks.append(("LLM", "❌", "离线"))
    
    # 5. 测试问答
    try:
        test_q = "What is the revenue?"
        answer = rag.ask(test_q, verbose=False)
        if answer and "don't know" not in answer.lower():
            checks.append(("问答测试", "✅", "正常"))
        else:
            checks.append(("问答测试", "⚠️", f"回答: {answer[:50]}..."))
    except Exception as e:
        checks.append(("问答测试", "❌", str(e)[:50]))
    
    print()
    print("健康检查结果:")
    print("-" * 40)
    for name, status, msg in checks:
        print(f"   {status} {name}: {msg}")
    print("-" * 40)
    
    # 总结
    failed = sum(1 for _, status, _ in checks if status == "❌")
    if failed == 0:
        print("   🎉 系统状态良好，准备比赛！")
    else:
        print(f"   ⚠️ 有 {failed} 项检查未通过，请排查")


def main():
    parser = argparse.ArgumentParser(description="预构建向量索引")
    parser.add_argument("--data_dir", type=str, default="data/", 
                        help="数据目录（放组委会数据）")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="配置文件路径")
    
    args = parser.parse_args()
    
    # 检查数据目录
    if not os.path.exists(args.data_dir):
        print(f"❌ 数据目录不存在: {args.data_dir}")
        print("   请先将组委会数据放入该目录")
        sys.exit(1)
    
    prebuild(args.data_dir, args.config)


if __name__ == "__main__":
    main()
