"""
随机种子与可复现性工具。

提供:
- 统一的随机种子设置 (Python / NumPy / PyTorch / CUDA)
- 随机 ID 生成
"""

from __future__ import annotations

import logging
import os
import random
import uuid
from typing import Optional

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """设置全局随机种子, 确保实验可复现。

    设置以下随机源:
    - Python random
    - os 环境变量 PYTHONHASHSEED
    - NumPy (如果已安装)
    - PyTorch CPU & CUDA (如果已安装)

    Args:
        seed: 随机种子值.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # 确定性算法 (可能降低性能)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    logger.info(f"[Seeds] Global seed set to {seed}")


def get_random_id(prefix: str = "", length: int = 8) -> str:
    """生成一个随机 ID。

    Args:
        prefix: ID 前缀 (如 "obj_", "entry_").
        length: UUID 截取长度.

    Returns:
        格式为 "{prefix}{uuid_hex[:length]}" 的随机 ID.
    """
    uid = uuid.uuid4().hex[:length]
    return f"{prefix}{uid}"


def get_deterministic_id(content: str, prefix: str = "") -> str:
    """基于内容生成确定性 ID (使用 hash)。

    同样的内容总是生成同样的 ID, 适合用于去重。

    Args:
        content: 用于生成 ID 的内容字符串.
        prefix: ID 前缀.

    Returns:
        确定性 ID.
    """
    import hashlib
    h = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}{h}"
