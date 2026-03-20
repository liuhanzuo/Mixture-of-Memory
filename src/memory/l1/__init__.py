"""
L1: 在线连续关联矩阵记忆 (Online Continuous Associative Matrix Memory)

L1 是一个衰减关联矩阵记忆，不是条目存储。
核心更新公式: M_t = λ M_{t-1} + Σ_i ρ_i k_i v_i^T
用于在线补偿 SWA 窗口之外的历史信息。
"""

from .assoc_memory import AssociativeMemoryL1
from .writer import L1Writer
from .reader import L1Reader
from .gating import L1Gate

__all__ = [
    "AssociativeMemoryL1",
    "L1Writer",
    "L1Reader",
    "L1Gate",
]
