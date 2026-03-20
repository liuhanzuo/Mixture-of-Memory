"""
tasks — 合成基准任务与评测数据生成器。

提供三类合成任务：
- SyntheticUpdateTask: 测试记忆更新（覆写、过期、矛盾）
- ProfileTask: 测试用户画像建模
- LongHorizonChatTask: 测试长程对话记忆
"""

from src.tasks.synthetic_update_task import SyntheticUpdateTask
from src.tasks.profile_task import ProfileTask
from src.tasks.longhorizon_chat_task import LongHorizonChatTask

__all__ = [
    "SyntheticUpdateTask",
    "ProfileTask",
    "LongHorizonChatTask",
]
