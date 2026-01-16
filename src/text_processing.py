"""
文本处理工具

偏向英文数据源的轻量分词，同时兼容中文连续词块。
"""

import re
from typing import List


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?|[\u4e00-\u9fff]+")
_SPACE_PATTERN = re.compile(r"\s+")
_COMMON_TYPO_MAP = {
    "permiablity": "permeability",
    "permability": "permeability",
    "permeablity": "permeability",
    "litology": "lithology",
}
_ABBREV_MAP = {
    "bop": "blowout preventer",
    "mwd": "measurement while drilling",
    "npt": "non productive time",
}
_STOPWORDS = {
    "the", "is", "are", "what", "which", "and", "of", "for", "to",
    "in", "on", "a", "an", "with", "by", "as", "was", "were", "be",
    "does", "do", "did", "compare", "between", "vs", "versus",
    "的", "是", "什么", "哪些", "比较", "与"
}

_MATH_PATTERN = re.compile(r"^\s*([\d\.\s]+)([+\-*/])([\d\.\s]+)\s*$")


def tokenize_text(text: str) -> List[str]:
    """将文本切分为可用于 BM25 的 token 列表"""
    if not text:
        return []
    lowered = text.lower()
    return _TOKEN_PATTERN.findall(lowered)


def normalize_query(text: str) -> str:
    """轻量归一化：拼写修正 + 常见缩写展开 + 空白规范化"""
    if not text:
        return ""
    lowered = text.lower()
    for typo, fix in _COMMON_TYPO_MAP.items():
        lowered = re.sub(rf"\b{re.escape(typo)}\b", fix, lowered)
    for abbr, full in _ABBREV_MAP.items():
        lowered = re.sub(rf"\b{re.escape(abbr)}\b", full, lowered)
    return _SPACE_PATTERN.sub(" ", lowered).strip()


def extract_key_terms(text: str) -> List[str]:
    """抽取问题中的关键词（用于覆盖与匹配）"""
    normalized = normalize_query(text)
    tokens = tokenize_text(normalized)
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 2]


def is_commonsense_math(text: str) -> bool:
    """检测是否为基础算数表达式（如 1+1）"""
    if not text:
        return False
    return _MATH_PATTERN.match(text.strip()) is not None


def solve_commonsense_math(text: str) -> str:
    """计算基础算数表达式结果（仅支持 + - * /）"""
    match = _MATH_PATTERN.match(text.strip())
    if not match:
        return ""
    left, op, right = match.group(1).strip(), match.group(2), match.group(3).strip()
    try:
        a = float(left)
        b = float(right)
    except ValueError:
        return ""

    if op == "+":
        result = a + b
    elif op == "-":
        result = a - b
    elif op == "*":
        result = a * b
    else:
        if b == 0:
            return ""
        result = a / b

    if result.is_integer():
        return str(int(result))
    return str(result)

