"""L3: 语义/画像级长期记忆层。

L3 通过总结和抽象 L2 的片段级记忆对象, 生成长期语义记忆和用户画像,
如兴趣偏好、长期项目、行为风格等。L3 是低频异步更新的, 默认在会话结束时触发。
"""

from src.memory.l3.summarizer import L3ProfileEntry, L3Summarizer
from src.memory.l3.profile_store import L3ProfileStore
from src.memory.l3.reviser import L3Reviser
from src.memory.l3.formatter import L3Formatter

__all__ = [
    "L3ProfileEntry",
    "L3Summarizer",
    "L3ProfileStore",
    "L3Reviser",
    "L3Formatter",
]
