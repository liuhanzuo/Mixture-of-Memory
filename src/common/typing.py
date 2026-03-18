"""
类型别名模块。

集中定义项目中常用的类型标注，提升代码可读性。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

# ---------- 通用张量相关 ----------
TensorDict = Dict[str, Tensor]
"""字符串到张量的映射，常用于模型输出。"""

BatchDict = Dict[str, Any]
"""一个 batch 的数据字典，值可以是 Tensor 或其他 Python 对象。"""

# ---------- 形状标注 ----------
# 以下为语义化的形状别名，便于文档和注释中引用
Shape = Tuple[int, ...]
"""通用张量形状。"""

BatchSize = int
SeqLen = int
HiddenDim = int
NumMemories = int
MemSlots = int
MemDim = int
TopK = int
BlockLen = int

# ---------- 配置相关 ----------
ConfigDict = Dict[str, Any]
"""普通 dict 形式的配置。"""

# ---------- 设备相关 ----------
DeviceLike = Union[str, torch.device]
"""可以被 torch 接受的设备描述符。"""
