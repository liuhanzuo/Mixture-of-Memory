"""
Sparse Memory Retrieval module — Sliding Window + Sparse Memory.

Components:
- SparseMemoryBank: EMA-based memory with FIFO write + top-k read
- ConcatFusionAttention: Local (window) + memory concat fusion with bypass gate
- SparseMemoryModel: HF model wrapper that patches attention layers
"""

from .attention import ConcatFusionAttention
from .memory_bank import SparseMemoryBank
from .model import SparseMemoryModel

__all__ = [
    "SparseMemoryBank",
    "ConcatFusionAttention",
    "SparseMemoryModel",
]
