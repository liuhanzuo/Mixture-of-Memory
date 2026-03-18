"""
数据模块。

提供合成数据集、RULER 和 LongMemEval 数据加载器。
"""

from src.data.synthetic_dataset import (
    SyntheticMemoryDataset,
    SyntheticSample,
    build_synthetic_splits,
)

__all__ = [
    "SyntheticMemoryDataset",
    "SyntheticSample",
    "build_synthetic_splits",
]
