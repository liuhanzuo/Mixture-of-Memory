"""
MoM (Mixture-of-Memory) 记忆系统顶层包。

当前活跃方向是 `src.memory.mem_space`（per-layer Memory-Space adapter，
live 训练入口 `scripts/train_mem_space_dolmino_cpt.py`）。

旧的三级层次化记忆栈 (L1/L2/L3 agent-memory + MemoryScheduler + MoMState)
已于 2026-06-12 归档至 `legacy/src_dead_subsystems/`，不再从本包导出。
如需复活，请从 legacy 目录 `git mv` 回原位并恢复下面的 lazy import 表。
"""

__all__ = []
