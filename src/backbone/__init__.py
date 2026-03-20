"""
Backbone 模块 —— 提供统一的模型前向接口。

支持:
- Full Attention 基线 (上界参照)
- SWA / Local Attention 骨干 (主实验)
- Memory-Readable 扩展接口 (用于 L1 记忆读出集成)
"""

from src.backbone.hidden_state_types import BackboneOutput
from src.backbone.interfaces import BackboneModel, MemoryReadableBackbone
from src.backbone.swa_model import SWABackbone
from src.backbone.full_attention_model import FullAttentionBackbone

__all__ = [
    "BackboneOutput",
    "BackboneModel",
    "MemoryReadableBackbone",
    "SWABackbone",
    "FullAttentionBackbone",
]
