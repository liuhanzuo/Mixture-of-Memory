"""
MAG / MAC 记忆注入模块。

MAG (Memory-Augmented Generation) — 侵入式中间层注入:
- MemoryEncoder: 将 L2/L3 文本记忆编码为向量 (共享 backbone embedding)
- ContextSelector: Learned Scorer 选择有用的记忆 context
- MAGGate: 在 Transformer 中间层通过 CrossAttn + 门控注入记忆

MAC (Memory-Augmented Context) — 非侵入式输入端注入:
- MemoryEncoder: 同上
- ContextSelector: 同上
- PrefixProjector: 将记忆向量映射为 soft prefix tokens, 拼接到 input embedding

MAC 优势: 零侵入 backbone, 语言能力完全保持, 训练更简单。
"""

from src.memory.mag.memory_encoder import MemoryEncoder
from src.memory.mag.context_selector import ContextSelector
from src.memory.mag.mag_gate import MAGGate
from src.memory.mag.prefix_projector import PrefixProjector

__all__ = [
    "MemoryEncoder",
    "ContextSelector",
    "MAGGate",
    "PrefixProjector",
]
