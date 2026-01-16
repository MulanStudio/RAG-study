"""
文本处理工具

偏向英文数据源的轻量分词，同时兼容中文连续词块。
"""

import re
from typing import List


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?|[\u4e00-\u9fff]+")


def tokenize_text(text: str) -> List[str]:
    """将文本切分为可用于 BM25 的 token 列表"""
    if not text:
        return []
    lowered = text.lower()
    return _TOKEN_PATTERN.findall(lowered)

