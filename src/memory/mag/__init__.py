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

__all__ = [
    "MemoryEncoder",
    "ContextSelector",
    "MAGGate",
    "PrefixProjector",
    "CompressedMemoryCache",
]


def __getattr__(name: str):
    """延迟导入: 避免包初始化时触发循环导入链。

    循环链: mag.__init__ → memory_encoder → src.memory.l2 → src.memory.__init__
            → scheduler → mag.context_selector (但 mag.__init__ 还没完成)
    """
    if name == "MemoryEncoder":
        from src.memory.mag.memory_encoder import MemoryEncoder
        return MemoryEncoder
    if name == "ContextSelector":
        from src.memory.mag.context_selector import ContextSelector
        return ContextSelector
    if name == "MAGGate":
        from src.memory.mag.mag_gate import MAGGate
        return MAGGate
    if name == "PrefixProjector":
        from src.memory.mag.prefix_projector import PrefixProjector
        return PrefixProjector
    if name == "CompressedMemoryCache":
        from src.memory.mag.compressed_memory import CompressedMemoryCache
        return CompressedMemoryCache
    raise AttributeError(f"module 'src.memory.mag' has no attribute {name!r}")
