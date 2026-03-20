"""
utils — 通用工具函数集合。

提供日志、IO、文本处理、随机种子、时间工具等基础设施。
"""

from src.utils.logging import setup_logging, get_logger
from src.utils.io import save_json, load_json, ensure_dir, safe_write
from src.utils.text import truncate_text, word_count, extract_keywords, compute_text_similarity
from src.utils.seeds import set_seed, get_random_id
from src.utils.time import now_iso, elapsed_ms, Timer

__all__ = [
    # logging
    "setup_logging",
    "get_logger",
    # io
    "save_json",
    "load_json",
    "ensure_dir",
    "safe_write",
    # text
    "truncate_text",
    "word_count",
    "extract_keywords",
    "compute_text_similarity",
    # seeds
    "set_seed",
    "get_random_id",
    # time
    "now_iso",
    "elapsed_ms",
    "Timer",
]
