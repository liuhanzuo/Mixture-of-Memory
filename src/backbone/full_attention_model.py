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
    - 真实的自回归生成 (通过 HF model.generate)
    """

    def __init__(
        self,
        model: nn.Module,
        hidden_dim: int,
        num_layers: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        tokenizer: Any = None,
        is_debug_mode: bool = False,
        generation_config: dict[str, Any] | None = None,
    ):
        self._model = model
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._device = torch.device(device) if isinstance(device, str) else device
        self._dtype = dtype
        self._tokenizer = tokenizer
        self._is_debug = is_debug_mode
        self._generation_config = generation_config or {}

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
            generation: 生成参数字典
        """
        device = cfg.get("device", "cpu")
        dtype_str = cfg.get("dtype", "float32")
        dtype = getattr(torch, dtype_str, torch.float32)

        # 提取生成配置
        gen_cfg = cfg.get("generation", {})
        generation_config = {}
        if gen_cfg:
            from omegaconf import OmegaConf
            generation_config = OmegaConf.to_container(gen_cfg, resolve=True) if not isinstance(gen_cfg, dict) else dict(gen_cfg)

        tokenizer = None
        is_debug_mode = False

        if cfg.get("debug", False):
            is_debug_mode = True
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
            model, hidden_dim, num_layers, tokenizer = _load_hf_full_attention(cfg)
            logger.info(
                f"FullAttentionBackbone: 加载模型 {cfg.model_name_or_path} "
                f"(d={hidden_dim}, L={num_layers}, full attention, "
                f"tokenizer={'✓' if tokenizer else '✗'})"
            )

        model = model.to(device=device, dtype=dtype)
        return cls(
            model=model,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            device=device,
            dtype=dtype,
            tokenizer=tokenizer,
            is_debug_mode=is_debug_mode,
            generation_config=generation_config,
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
    # 生成
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        """自回归生成。

        对于真实 HF 模型，调用 model.generate()；
        对于 debug 模型，使用简单的 greedy 循环。

        Returns:
            生成的完整 token 序列 (包含 prompt), shape ``(B, T + new_tokens)``。
        """
        input_ids = input_ids.to(self._device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)

        if self._is_debug:
            return self._debug_generate(
                input_ids, attention_mask, max_new_tokens=max_new_tokens
            )

        # 合并配置中的默认生成参数
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
        }
        for k, v in self._generation_config.items():
            if k not in gen_kwargs:
                gen_kwargs[k] = v
        gen_kwargs.update(kwargs)

        # 设置 pad_token_id 避免警告
        if self._tokenizer is not None and self._tokenizer.pad_token_id is not None:
            gen_kwargs.setdefault("pad_token_id", self._tokenizer.pad_token_id)
        elif self._tokenizer is not None and self._tokenizer.eos_token_id is not None:
            gen_kwargs.setdefault("pad_token_id", self._tokenizer.eos_token_id)

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
        return output_ids

    def _debug_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        max_new_tokens: int = 32,
    ) -> torch.Tensor:
        """Debug 模型的简单 greedy 生成。"""
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            with torch.no_grad():
                output = self._model(generated, attention_mask=attention_mask)
            next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_token)], dim=-1
                )
        return generated

    # ------------------------------------------------------------------
    # 接口实现
    # ------------------------------------------------------------------

    def get_hidden_dim(self) -> int:
        return self._hidden_dim

    def get_num_layers(self) -> int:
        return self._num_layers

    def get_device(self) -> torch.device:
        return self._device

    def get_tokenizer(self) -> Any:
        """返回关联的 tokenizer。"""
        return self._tokenizer

    def get_hf_model(self) -> Any:
        """返回底层 HuggingFace 模型。"""
        return self._model

    def is_debug(self) -> bool:
        """是否为 debug 模式。"""
        return self._is_debug

    @property
    def tokenizer(self) -> Any:
        """便捷属性: 访问 tokenizer。"""
        return self._tokenizer


# ======================================================================
# HuggingFace 加载（全注意力模式）
# ======================================================================


def _load_hf_full_attention(cfg: DictConfig) -> tuple[nn.Module, int, int, Any]:
    """从 HuggingFace 加载预训练模型，并强制关闭 sliding window。

    Returns:
        (model, hidden_dim, num_layers, tokenizer)
    """
    try:
        from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    except ImportError as e:
        raise ImportError("需要安装 transformers: pip install transformers") from e

    model_path = cfg.model_name_or_path
    logger.info(f"正在加载模型配置: {model_path}")
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # 强制关闭 sliding window，使用全局注意力
    if hasattr(hf_config, "sliding_window"):
        hf_config.sliding_window = None
        logger.info("FullAttentionBackbone: 已关闭 sliding_window")
    if hasattr(hf_config, "max_window_layers"):
        hf_config.max_window_layers = 0
        logger.info("FullAttentionBackbone: 已设置 max_window_layers=0")

    logger.info(f"正在加载模型权重: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=hf_config,
        trust_remote_code=True,
        torch_dtype=getattr(torch, cfg.get("dtype", "float32")),
    )

    # 加载 tokenizer
    tokenizer = None
    try:
        logger.info(f"正在加载 tokenizer: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Tokenizer: pad_token 未设置，已设为 eos_token")
        logger.info(
            f"Tokenizer 加载完成: vocab_size={tokenizer.vocab_size}, "
            f"pad_token='{tokenizer.pad_token}'"
        )
    except Exception as e:
        logger.warning(f"Tokenizer 加载失败: {e}. 将在无 tokenizer 模式下运行。")

    hidden_dim = hf_config.hidden_size
    num_layers = hf_config.num_hidden_layers
    return model, hidden_dim, num_layers, tokenizer
