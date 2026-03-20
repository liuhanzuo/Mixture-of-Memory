"""
文本处理工具模块。

提供常用的文本操作:
- 截断
- 词频统计
- 关键词提取 (简易版)
- 文本相似度 (基于词袋 Jaccard)
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Sequence


def truncate_text(text: str, max_chars: int = 500, suffix: str = "...") -> str:
    """截断文本到指定最大字符数。

    Args:
        text: 原始文本.
        max_chars: 最大字符数.
        suffix: 截断后的后缀标记.

    Returns:
        截断后的文本.
    """
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(suffix)] + suffix


def word_count(text: str) -> int:
    """统计文本中的词数 (按空格分割)。

    对中文文本, 返回字符数作为近似词数。

    Args:
        text: 输入文本.

    Returns:
        词数.
    """
    # 检查是否主要是 CJK 字符
    cjk_chars = len(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', text))
    if cjk_chars > len(text) * 0.3:
        # 中文: 返回非空白字符数
        return len(re.sub(r'\s+', '', text))
    else:
        # 英文: 按空格分词
        return len(text.split())


def tokenize_simple(text: str) -> list[str]:
    """简易分词: 转小写, 按非字母数字字符分割。

    Args:
        text: 输入文本.

    Returns:
        token 列表.
    """
    text = text.lower()
    tokens = re.findall(r'[a-z0-9\u4e00-\u9fff\u3400-\u4dbf]+', text)
    return tokens


def extract_keywords(
    text: str,
    top_k: int = 10,
    stop_words: set[str] | None = None,
) -> list[str]:
    """从文本中提取高频关键词 (简易版, 基于词频)。

    Args:
        text: 输入文本.
        top_k: 返回的关键词数量.
        stop_words: 停用词集合.

    Returns:
        关键词列表, 按频率降序.
    """
    if stop_words is None:
        stop_words = _DEFAULT_STOP_WORDS

    tokens = tokenize_simple(text)
    # 过滤停用词和过短的 token
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]

    counter = Counter(tokens)
    return [word for word, _ in counter.most_common(top_k)]


def compute_text_similarity(text_a: str, text_b: str) -> float:
    """计算两段文本的相似度 (Jaccard 系数, 基于词袋)。

    这是一个轻量级的相似度度量, 不依赖外部模型。
    适合用于 L2 合并判断等简单场景。

    Args:
        text_a: 文本 A.
        text_b: 文本 B.

    Returns:
        Jaccard 相似度, 范围 [0, 1].
    """
    tokens_a = set(tokenize_simple(text_a))
    tokens_b = set(tokenize_simple(text_b))

    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b

    return len(intersection) / len(union)


def normalize_whitespace(text: str) -> str:
    """规范化空白字符: 多个连续空白压缩为单个空格, 去除首尾空白。"""
    return re.sub(r'\s+', ' ', text).strip()


def sentence_split(text: str) -> list[str]:
    """简易句子分割 (支持中英文)。

    Args:
        text: 输入文本.

    Returns:
        句子列表.
    """
    # 按中英文句号、问号、感叹号分割
    sentences = re.split(r'(?<=[.!?。！？])\s*', text)
    return [s.strip() for s in sentences if s.strip()]


def contains_any(text: str, keywords: Sequence[str], case_sensitive: bool = False) -> bool:
    """检查文本是否包含任一关键词。

    Args:
        text: 待检查文本.
        keywords: 关键词列表.
        case_sensitive: 是否区分大小写.

    Returns:
        True 如果包含至少一个关键词.
    """
    if not case_sensitive:
        text = text.lower()
        keywords = [k.lower() for k in keywords]
    return any(k in text for k in keywords)


# 英文常用停用词
_DEFAULT_STOP_WORDS: set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "because", "but", "and",
    "or", "if", "while", "about", "up", "it", "its", "this", "that",
    "these", "those", "i", "me", "my", "we", "our", "you", "your", "he",
    "him", "his", "she", "her", "they", "them", "their", "what", "which",
    "who", "whom",
}
