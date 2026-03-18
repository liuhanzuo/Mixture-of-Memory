"""
可复现性种子管理模块。

统一设置 Python / NumPy / PyTorch / CUDA 的随机种子。
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch

from src.common.logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """设置全局随机种子以保证实验可复现。

    Args:
        seed: 随机种子值。
        deterministic: 是否启用 CUDA 确定性模式
            （可能略微降低性能，但保证可复现）。
    """
    logger.info(f"设置全局随机种子: {seed} (deterministic={deterministic})")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch >= 1.8
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # 某些算子不支持确定性模式，降级处理
            pass
