"""
Full Attention 骨干模型实现。

作为实验中的上界基线（upper-bound baseline），使用完整的全局注意力，
不受滑动窗口限制。用于对比 SWA + MoM 方案能否在更低 context cost 下
逼近全注意力的长程记忆能力。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.backbone.hidden_state_types import BackboneOutput
from src.backbone.interfaces import BackboneModel

logger = logging.getLogger(__name__)


class FullAttentionBackbone(BackboneModel):
    """基于全局注意力的骨干模型（上界基线）。

    支持:
    - 从 HuggingFace 加载真实预训练模型（关闭 sliding_window）
    - Debug tiny 模式（与 SWA 共享 debug transformer 结构）
    """

    def __init__(
        self,
        model: nn.Module,
        hidden_dim: int,
        num_layers: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self._model = model
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._device = torch.device(device) if isinstance(device, str) else device
        self._dtype = dtype

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "FullAttentionBackbone":
        """从配置构造 Full Attention 骨干模型。

        配置字段:
            model_name_or_path: HuggingFace 模型路径
            hidden_dim: 隐藏维度（debug 模式使用）
            num_layers: 层数（debug 模式使用）
            debug: 是否使用 debug tiny 模型
            device: 设备
            dtype: 数据类型
        """
        device = cfg.get("device", "cpu")
        dtype_str = cfg.get("dtype", "float32")
        dtype = getattr(torch, dtype_str, torch.float32)

        if cfg.get("debug", False):
            from src.backbone.swa_model import _build_debug_transformer

            hidden_dim = cfg.get("hidden_dim", 128)
            num_layers = cfg.get("num_layers", 2)
            model = _build_debug_transformer(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=cfg.get("num_heads", 4),
                vocab_size=cfg.get("vocab_size", 1000),
                max_seq_len=cfg.get("max_seq_len", 512),
            )
            logger.info(
                "FullAttentionBackbone: 使用 debug tiny 模型 "
                f"(d={hidden_dim}, L={num_layers})"
            )
        else:
            model, hidden_dim, num_layers = _load_hf_full_attention(cfg)
            logger.info(
                f"FullAttentionBackbone: 加载模型 {cfg.model_name_or_path} "
                f"(d={hidden_dim}, L={num_layers}, full attention)"
            )

        model = model.to(device=device, dtype=dtype)
        return cls(
            model=model,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            device=device,
            dtype=dtype,
        )

    # ------------------------------------------------------------------
    # 前向传播
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> BackboneOutput:
        """全注意力前向传播。"""
        input_ids = input_ids.to(self._device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)
        if labels is not None:
            labels = labels.to(self._device)

        # 检查是否是 debug transformer
        from src.backbone.swa_model import _DebugTransformer

        if isinstance(self._model, _DebugTransformer):
            return self._model(input_ids, attention_mask=attention_mask, labels=labels)

        # HuggingFace 模型
        hf_out = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs,
        )
        return BackboneOutput(
            last_hidden_state=(
                hf_out.hidden_states[-1]
                if hf_out.hidden_states
                else hf_out.last_hidden_state
            ),
            logits=getattr(hf_out, "logits", None),
            all_hidden_states=(
                list(hf_out.hidden_states) if hf_out.hidden_states else None
            ),
            attention_mask=attention_mask,
            loss=getattr(hf_out, "loss", None),
        )

    # ------------------------------------------------------------------
    # 接口实现
    # ------------------------------------------------------------------

    def get_hidden_dim(self) -> int:
        return self._hidden_dim

    def get_num_layers(self) -> int:
        return self._num_layers

    def get_device(self) -> torch.device:
        return self._device


# ======================================================================
# HuggingFace 加载（全注意力模式）
# ======================================================================


def _load_hf_full_attention(cfg: DictConfig) -> tuple[nn.Module, int, int]:
    """从 HuggingFace 加载预训练模型，并强制关闭 sliding window。

    Returns:
        (model, hidden_dim, num_layers)
    """
    try:
        from transformers import AutoModelForCausalLM, AutoConfig
    except ImportError as e:
        raise ImportError("需要安装 transformers: pip install transformers") from e

    model_path = cfg.model_name_or_path
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # 强制关闭 sliding window，使用全局注意力
    if hasattr(hf_config, "sliding_window"):
        hf_config.sliding_window = None
        logger.info("FullAttentionBackbone: 已关闭 sliding_window")
    if hasattr(hf_config, "max_window_layers"):
        hf_config.max_window_layers = 0
        logger.info("FullAttentionBackbone: 已设置 max_window_layers=0")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=hf_config,
        trust_remote_code=True,
        torch_dtype=getattr(torch, cfg.get("dtype", "float32")),
    )

    hidden_dim = hf_config.hidden_size
    num_layers = hf_config.num_hidden_layers
    return model, hidden_dim, num_layers
