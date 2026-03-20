"""
SWA (Sliding Window Attention) 骨干模型实现。

这是 MoM 系统的主实验骨干 —— 仅使用局部注意力窗口，
需要依赖外部记忆层（L1/L2/L3）来补偿长程信息缺失。

当前实现提供两种模式:
1. 基于 HuggingFace Transformers 加载真实 SWA 模型（如 Qwen2/Mistral）
2. Debug 模式: 使用轻量随机 Transformer 用于单元测试
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.backbone.hidden_state_types import BackboneOutput
from src.backbone.interfaces import MemoryReadableBackbone

logger = logging.getLogger(__name__)


class SWABackbone(MemoryReadableBackbone):
    """基于滑动窗口注意力的骨干模型。

    支持:
    - 从 HuggingFace 加载真实预训练模型
    - Debug tiny 模式（随机权重小 Transformer）
    - 外部 L1 记忆读出注入
    """

    def __init__(
        self,
        model: nn.Module,
        hidden_dim: int,
        num_layers: int,
        window_size: int = 4096,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self._model = model
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._window_size = window_size
        self._device = torch.device(device) if isinstance(device, str) else device
        self._dtype = dtype

        # 记忆注入的线性门控（延迟初始化）
        self._memory_gate: Optional[nn.Linear] = None

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "SWABackbone":
        """从配置构造 SWA 骨干模型。

        配置字段:
            model_name_or_path: HuggingFace 模型路径
            window_size: 滑动窗口大小
            hidden_dim: 隐藏维度（debug 模式使用）
            num_layers: 层数（debug 模式使用）
            debug: 是否使用 debug tiny 模型
            device: 设备
            dtype: 数据类型
        """
        device = cfg.get("device", "cpu")
        dtype_str = cfg.get("dtype", "float32")
        dtype = getattr(torch, dtype_str, torch.float32)
        window_size = cfg.get("window_size", 4096)

        if cfg.get("debug", False):
            # Debug tiny 模型 —— 随机权重小 Transformer
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
                "SWABackbone: 使用 debug tiny 模型 "
                f"(d={hidden_dim}, L={num_layers}, W={window_size})"
            )
        else:
            # 从 HuggingFace 加载真实模型
            model, hidden_dim, num_layers = _load_hf_model(cfg)
            logger.info(
                f"SWABackbone: 加载模型 {cfg.model_name_or_path} "
                f"(d={hidden_dim}, L={num_layers}, W={window_size})"
            )

        model = model.to(device=device, dtype=dtype)
        return cls(
            model=model,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            window_size=window_size,
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
        """标准前向传播（无记忆注入）。"""
        outputs = self._run_model(input_ids, attention_mask, labels, **kwargs)
        return outputs

    def forward_with_memory(
        self,
        input_ids: torch.Tensor,
        memory_readout: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> BackboneOutput:
        """带 L1 记忆读出的前向传播。

        将记忆读出通过门控机制融合到最后一层隐藏状态中:
            h' = h + gate(h) * memory_readout
        """
        outputs = self._run_model(input_ids, attention_mask, labels, **kwargs)

        # 融合记忆读出
        h = outputs.last_hidden_state  # (B, T, D)

        # 处理 memory_readout 维度
        if memory_readout.dim() == 2:
            # (B, D) -> (B, 1, D) -> broadcast
            memory_readout = memory_readout.unsqueeze(1).expand_as(h)

        # 延迟初始化门控层
        if self._memory_gate is None:
            self._memory_gate = nn.Linear(self._hidden_dim, self._hidden_dim).to(
                device=h.device, dtype=h.dtype
            )
            nn.init.zeros_(self._memory_gate.bias)

        gate = torch.sigmoid(self._memory_gate(h))  # (B, T, D)
        h_fused = h + gate * memory_readout

        return BackboneOutput(
            last_hidden_state=h_fused,
            logits=outputs.logits,
            all_hidden_states=outputs.all_hidden_states,
            attention_mask=outputs.attention_mask,
            loss=outputs.loss,
            extra={**outputs.extra, "memory_gate_values": gate},
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

    @property
    def window_size(self) -> int:
        return self._window_size

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _run_model(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        **kwargs: Any,
    ) -> BackboneOutput:
        """运行底层模型并统一输出格式。"""
        input_ids = input_ids.to(self._device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)
        if labels is not None:
            labels = labels.to(self._device)

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
            last_hidden_state=hf_out.hidden_states[-1] if hf_out.hidden_states else hf_out.last_hidden_state,
            logits=getattr(hf_out, "logits", None),
            all_hidden_states=list(hf_out.hidden_states) if hf_out.hidden_states else None,
            attention_mask=attention_mask,
            loss=getattr(hf_out, "loss", None),
        )


# ======================================================================
# Debug Tiny Transformer
# ======================================================================


class _DebugTransformer(nn.Module):
    """极简 Transformer，仅用于单元测试和 debug 模式。"""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        vocab_size: int = 1000,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> BackboneOutput:
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        h = self.embed(input_ids) + self.pos_embed(positions)

        # 简单的 causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=input_ids.device)

        all_hidden = [h.clone()]
        # TransformerEncoder 不原生支持逐层 hook，这里简单用循环
        for layer in self.encoder.layers:
            h = layer(h, src_mask=causal_mask)
            all_hidden.append(h.clone())

        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        return BackboneOutput(
            last_hidden_state=h,
            logits=logits,
            all_hidden_states=all_hidden,
            attention_mask=attention_mask,
            loss=loss,
        )


def _build_debug_transformer(
    hidden_dim: int = 128,
    num_layers: int = 2,
    num_heads: int = 4,
    vocab_size: int = 1000,
    max_seq_len: int = 512,
) -> _DebugTransformer:
    """构建 debug 用的小型 Transformer。"""
    return _DebugTransformer(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
    )


def _load_hf_model(cfg: DictConfig) -> tuple[nn.Module, int, int]:
    """从 HuggingFace 加载预训练模型。

    Returns:
        (model, hidden_dim, num_layers)
    """
    try:
        from transformers import AutoModelForCausalLM, AutoConfig
    except ImportError as e:
        raise ImportError("需要安装 transformers: pip install transformers") from e

    model_path = cfg.model_name_or_path
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # 设置滑动窗口（如果模型支持）
    window_size = cfg.get("window_size", 4096)
    if hasattr(hf_config, "sliding_window"):
        hf_config.sliding_window = window_size
    if hasattr(hf_config, "max_window_layers"):
        # Qwen2 风格: 仅前 N 层使用 SWA
        hf_config.max_window_layers = cfg.get(
            "max_window_layers", hf_config.num_hidden_layers
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=hf_config,
        trust_remote_code=True,
        torch_dtype=getattr(torch, cfg.get("dtype", "float32")),
    )

    hidden_dim = hf_config.hidden_size
    num_layers = hf_config.num_hidden_layers
    return model, hidden_dim, num_layers
