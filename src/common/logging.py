"""
日志模块。

提供统一的日志格式和配置。
支持：
  - 彩色终端输出
  - 文件日志
  - 按模块名获取 logger
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_INITIALIZED = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> None:
    """初始化全局日志配置。

    Args:
        level: 日志级别 (DEBUG / INFO / WARNING / ERROR)。
        log_file: 可选的日志文件路径。
    """
    global _INITIALIZED
    if _INITIALIZED:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # --- 控制台 handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root_logger.addHandler(console_handler)

    # --- 文件 handler ---
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        root_logger.addHandler(file_handler)

    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """获取命名 logger。

    Args:
        name: logger 名称，通常传入 ``__name__``。

    Returns:
        配置好的 Logger 实例。
    """
    # 确保至少有基本配置
    if not _INITIALIZED:
        setup_logging()
    return logging.getLogger(name)
