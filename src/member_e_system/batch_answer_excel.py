#!/usr/bin/env python3
"""
批量回答 Excel 问题并回写答案列

期望表结构：
 - 第一列: no.
 - 第二列: question
 - 第三列: answer
"""

import argparse
import os
import pandas as pd
import yaml

from src.member_e_system.app import OilfieldRAG


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """尝试规范列名到 no./question/answer"""
    col_map = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in {"no", "no.", "index", "id"}:
            col_map[col] = "no."
        elif key in {"question", "questions", "query", "q"}:
            col_map[col] = "question"
        elif key in {"answer", "answers", "response"}:
            col_map[col] = "answer"
    return df.rename(columns=col_map)


def main() -> None:
    parser = argparse.ArgumentParser(description="Excel 批量问答")
    parser.add_argument("--input", help="输入 Excel 路径")
    parser.add_argument("--output", help="输出 Excel 路径（默认覆盖输入文件）")
    parser.add_argument("--data_dir", default="data/", help="数据目录")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件")
    parser.add_argument("--question_col", default="question", help="问题列名")
    parser.add_argument("--answer_col", default="answer", help="答案列名")
    parser.add_argument("--start_row", type=int, default=0, help="从第几行开始处理（0-based）")
    parser.add_argument("--limit", type=int, default=None, help="最多处理多少行")
    parser.add_argument("--resume", action="store_true", help="跳过已有答案的行")
    parser.add_argument("--save_every", type=int, default=0, help="每处理 N 行保存一次，0 表示不定期保存")
    args = parser.parse_args()

    # 读取配置文件中的默认值
    cfg = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    batch_cfg = cfg.get("batch_answer", {})

    input_path = args.input or batch_cfg.get("input_path")
    if not input_path:
        raise ValueError("Input file not provided. Use --input or set batch_answer.input_path in config.")

    output_path = args.output or batch_cfg.get("output_path") or input_path
    question_col = args.question_col or batch_cfg.get("question_col", "question")
    answer_col = args.answer_col or batch_cfg.get("answer_col", "answer")
    start_row = args.start_row if args.start_row is not None else batch_cfg.get("start_row", 0)
    limit = args.limit if args.limit is not None else batch_cfg.get("limit", None)
    resume = args.resume or batch_cfg.get("resume", False)
    save_every = args.save_every if args.save_every is not None else batch_cfg.get("save_every", 0)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_excel(input_path)
    df = normalize_columns(df)

    if question_col not in df.columns:
        raise ValueError(f"Question column not found: {question_col}")

    if answer_col not in df.columns:
        df[answer_col] = ""

    rag = OilfieldRAG(data_dir=args.data_dir, config_path=args.config)
    rag.initialize()

    start = max(0, int(start_row))
    end = len(df) if limit is None else min(len(df), start + int(limit))

    for idx in range(start, end):
        question = str(df.at[idx, question_col]).strip()
        if not question or question.lower() in {"nan", "none"}:
            continue
        if resume:
            existing = str(df.at[idx, answer_col]).strip()
            if existing and existing.lower() not in {"nan", "none"}:
                continue
        answer = rag.ask(question, verbose=False)
        df.at[idx, answer_col] = answer
        if save_every and save_every > 0 and (idx - start + 1) % save_every == 0:
            df.to_excel(output_path, index=False)

    df.to_excel(output_path, index=False)
    print(f"✅ 已写回答案: {output_path} (rows {start}-{end-1})")


if __name__ == "__main__":
    main()
