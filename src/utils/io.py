"""
IO 工具模块。

提供常用的文件读写操作:
- JSON 序列化/反序列化
- 目录安全创建
- 原子写入 (先写临时文件再 rename)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在, 不存在则递归创建。

    Args:
        path: 目录路径.

    Returns:
        创建后的 Path 对象.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(
    data: Any,
    path: str | Path,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> Path:
    """将数据保存为 JSON 文件。

    自动创建父目录。

    Args:
        data: 要序列化的数据.
        path: 目标文件路径.
        indent: 缩进空格数.
        ensure_ascii: 是否转义非 ASCII 字符.

    Returns:
        写入的文件路径.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, default=str)

    logger.debug(f"[IO] Saved JSON: {p}")
    return p


def load_json(path: str | Path) -> Any:
    """从 JSON 文件加载数据。

    Args:
        path: JSON 文件路径.

    Returns:
        反序列化后的 Python 对象.

    Raises:
        FileNotFoundError: 文件不存在.
        json.JSONDecodeError: JSON 格式错误.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.debug(f"[IO] Loaded JSON: {p}")
    return data


def safe_write(
    content: str,
    path: str | Path,
    encoding: str = "utf-8",
) -> Path:
    """安全写入文本文件 (先写临时文件, 再原子 rename)。

    在写入大型文件或关键数据时使用, 避免写入中断导致文件损坏。

    Args:
        content: 文本内容.
        path: 目标文件路径.
        encoding: 编码.

    Returns:
        写入的文件路径.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # 在同目录下创建临时文件, 保证 rename 是原子操作 (同一文件系统)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(p.parent),
        prefix=f".{p.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
        os.replace(tmp_path, str(p))
        logger.debug(f"[IO] Safe-wrote: {p}")
    except Exception:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    return p


def read_text(path: str | Path, encoding: str = "utf-8") -> str:
    """读取文本文件内容。

    Args:
        path: 文件路径.
        encoding: 编码.

    Returns:
        文件文本内容.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return p.read_text(encoding=encoding)


def file_size_mb(path: str | Path) -> float:
    """获取文件大小 (MB)。"""
    return Path(path).stat().st_size / (1024 * 1024)
