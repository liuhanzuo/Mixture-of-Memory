"""
日志工具模块。

提供统一的日志配置，支持:
- 控制台彩色输出
- 文件日志
- 按模块名获取 logger
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


_LOGGING_INITIALIZED = False

# 日志格式
_CONSOLE_FORMAT = "%(asctime)s │ %(levelname)-7s │ %(name)-30s │ %(message)s"
_FILE_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str | int = "INFO",
    log_file: str | Path | None = None,
    console: bool = True,
    file_level: str | int | None = None,
) -> None:
    """配置全局日志。

    在项目入口处调用一次即可。重复调用会跳过。

    Args:
        level: 控制台日志级别, 如 "DEBUG", "INFO", "WARNING".
        log_file: 日志文件路径 (可选). 若指定则同时写入文件.
        console: 是否输出到控制台.
        file_level: 文件日志级别 (默认与 level 相同).
    """
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return
    _LOGGING_INITIALIZED = True

    # 解析级别
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    if file_level is None:
        file_level = level
    elif isinstance(file_level, str):
        file_level = getattr(logging, file_level.upper(), logging.DEBUG)

    root_logger = logging.getLogger()
    root_logger.setLevel(min(level, file_level))

    # 清除已有 handlers (避免重复)
    root_logger.handlers.clear()

    # ---- 控制台 handler ---- #
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(
            logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT)
        )
        root_logger.addHandler(console_handler)

    # ---- 文件 handler ---- #
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            str(log_path), mode="a", encoding="utf-8",
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(
            logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT)
        )
        root_logger.addHandler(file_handler)

    # 降低第三方库的日志噪音
    for noisy_lib in ("transformers", "tokenizers", "urllib3", "httpx", "httpcore"):
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """获取指定名称的 logger。

    如果全局日志尚未初始化，会自动调用 setup_logging() 进行默认配置。

    Args:
        name: logger 名称, 通常为 __name__.

    Returns:
        logging.Logger 实例.
    """
    if not _LOGGING_INITIALIZED:
        setup_logging()
    return logging.getLogger(name)
