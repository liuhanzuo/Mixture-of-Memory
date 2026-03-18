"""Backbone 模块：冻结的 HuggingFace 因果语言模型封装。"""

from src.backbone.lm_wrapper import FrozenLMWrapper
from src.backbone.hidden_extractor import HiddenStateExtractor
from src.backbone.generation import MemoryAugmentedGenerator

__all__ = [
    "FrozenLMWrapper",
    "HiddenStateExtractor",
    "MemoryAugmentedGenerator",
]
