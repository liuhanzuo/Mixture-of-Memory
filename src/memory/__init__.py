"""
MoM (Mixture-of-Memory) 记忆系统顶层包。

三级层次化记忆:
- L1: 在线连续关联矩阵记忆 (同步更新)
- L2: 事件/状态级记忆对象 (异步, chunk/turn 结束时更新)
- L3: 语义/画像级长期记忆 (异步, session 结束时更新)

MemoryScheduler 负责协调三级记忆的读写时机。
MoMState 封装整个记忆系统的运行时状态。
"""

__all__ = [
    # L1
    "AssociativeMemoryL1",
    "L1Writer",
    "L1Reader",
    "L1Gate",
    # L2
    "L2MemoryObject",
    "ChatMessage",
    "L2Aggregator",
    "L2ObjectStore",
    "L2Merger",
    "L2Retriever",
    # L3
    "L3ProfileEntry",
    "L3Summarizer",
    "L3ProfileStore",
    "L3Reviser",
    "L3Formatter",
    # 顶层
    "MemoryScheduler",
    "MoMState",
]

# ---- 延迟导入: 避免 scheduler → mag.context_selector 循环导入链 ---- #
_LAZY_IMPORTS = {
    # L1
    "AssociativeMemoryL1": ("src.memory.l1", "AssociativeMemoryL1"),
    "L1Writer": ("src.memory.l1", "L1Writer"),
    "L1Reader": ("src.memory.l1", "L1Reader"),
    "L1Gate": ("src.memory.l1", "L1Gate"),
    # L2
    "L2MemoryObject": ("src.memory.l2", "L2MemoryObject"),
    "ChatMessage": ("src.memory.l2", "ChatMessage"),
    "L2Aggregator": ("src.memory.l2", "L2Aggregator"),
    "L2ObjectStore": ("src.memory.l2", "L2ObjectStore"),
    "L2Merger": ("src.memory.l2", "L2Merger"),
    "L2Retriever": ("src.memory.l2", "L2Retriever"),
    # L3
    "L3ProfileEntry": ("src.memory.l3", "L3ProfileEntry"),
    "L3Summarizer": ("src.memory.l3", "L3Summarizer"),
    "L3ProfileStore": ("src.memory.l3", "L3ProfileStore"),
    "L3Reviser": ("src.memory.l3", "L3Reviser"),
    "L3Formatter": ("src.memory.l3", "L3Formatter"),
    # 顶层
    "MemoryScheduler": ("src.memory.scheduler", "MemoryScheduler"),
    "MoMState": ("src.memory.state", "MoMState"),
}


def __getattr__(name: str):
    """延迟导入: 避免包初始化时 scheduler → mag 的循环导入。"""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"module 'src.memory' has no attribute {name!r}")
