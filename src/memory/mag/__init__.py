"""
MAG (Memory-Augmented Generation) 模块。

实现 Titans 风格的记忆门控注入机制:
- MemoryEncoder: 将 L2/L3 文本记忆编码为向量 (共享 backbone embedding)
- ContextSelector: Learned Scorer 选择有用的记忆 context
- MAGGate: 在 Transformer 中间层通过 CrossAttn + 门控注入记忆

核心公式:
    m_agg = CrossAttn(h, {m_1, ..., m_K})   # query=hidden, key/value=memory
    g = σ(W_g [h; m_agg])                    # 门控信号
    h' = h + g ⊙ W_o m_agg                   # 残差融合
"""

from src.memory.mag.memory_encoder import MemoryEncoder
from src.memory.mag.context_selector import ContextSelector
from src.memory.mag.mag_gate import MAGGate

__all__ = [
    "MemoryEncoder",
    "ContextSelector",
    "MAGGate",
]
