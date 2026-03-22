"""
SWA (Sliding Window Attention) 骨干模型实现。

这是 MoM 系统的主实验骨干 —— 仅使用局部注意力窗口，
需要依赖外部记忆层（L1/L2/L3）来补偿长程信息缺失。

当前实现提供两种模式:
1. 基于 HuggingFace Transformers 加载真实 SWA 模型（如 Qwen3-1.7B）
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

try:
    from src.memory.mag.mag_gate import MAGGate
except ImportError:
    MAGGate = None  # MAG 模块未安装时回退

logger = logging.getLogger(__name__)


class SWABackbone(MemoryReadableBackbone):
    """基于滑动窗口注意力的骨干模型。

    支持:
    - 从 HuggingFace 加载真实预训练模型 (Qwen3-1.7B 等)
    - Debug tiny 模式（随机权重小 Transformer）
    - 外部 L1 记忆读出注入
    - 真实的自回归生成 (通过 HF model.generate)
    """

    def __init__(
        self,
        model: nn.Module,
        hidden_dim: int,
        num_layers: int,
        window_size: int = 4096,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        tokenizer: Any = None,
        is_debug_mode: bool = False,
        generation_config: dict[str, Any] | None = None,
    ):
        self._model = model
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._window_size = window_size
        self._device = torch.device(device) if isinstance(device, str) else device
        self._dtype = dtype
        self._tokenizer = tokenizer
        self._is_debug = is_debug_mode
        self._generation_config = generation_config or {}

        # 记忆注入的线性门控（延迟初始化）
        self._memory_gate: Optional[nn.Linear] = None

        # MAG 门控模块 (外部设置)
        self._mag_gate: Optional[nn.Module] = None

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
            generation: 生成参数字典
        """
        device = cfg.get("device", "cpu")
        dtype_str = cfg.get("dtype", "float32")
        dtype = getattr(torch, dtype_str, torch.float32)
        window_size = cfg.get("window_size", 4096)

        # 提取生成配置
        gen_cfg = cfg.get("generation", {})
        generation_config = {}
        if gen_cfg:
            from omegaconf import OmegaConf
            generation_config = OmegaConf.to_container(gen_cfg, resolve=True) if not isinstance(gen_cfg, dict) else dict(gen_cfg)

        tokenizer = None
        is_debug_mode = False

        if cfg.get("debug", False):
            # Debug tiny 模型 —— 随机权重小 Transformer
            is_debug_mode = True
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
            # 从 HuggingFace 加载真实模型 + tokenizer
            model, hidden_dim, num_layers, tokenizer = _load_hf_model(cfg)
            logger.info(
                f"SWABackbone: 加载模型 {cfg.model_name_or_path} "
                f"(d={hidden_dim}, L={num_layers}, W={window_size}, "
                f"tokenizer={'✓' if tokenizer else '✗'})"
            )

        model = model.to(device=device, dtype=dtype)
        return cls(
            model=model,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            window_size=window_size,
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

    def forward_with_mag(
        self,
        input_ids: torch.Tensor,
        mag_memory_vectors: torch.Tensor,
        mag_memory_mask: Optional[torch.Tensor] = None,
        mag_selection_weights: Optional[torch.Tensor] = None,
        memory_readout: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> BackboneOutput:
        """带 MAG 记忆注入的前向传播。

        在 Transformer 中间若干层通过 CrossAttn + Gate 注入 L2/L3 记忆向量。
        可同时接收 L1 memory_readout 在最后一层做门控融合。

        流程:
        1. 运行 backbone 获取所有层的 hidden states
        2. 在配置的中间层通过 MAGGate 注入记忆
        3. (可选) 在最后一层融合 L1 readout
        4. 用修改后的 last_hidden_state 重新计算 logits

        Args:
            input_ids: (B, T) 输入 token ids.
            mag_memory_vectors: (B, K, D) 编码后的 L2/L3 记忆向量.
            mag_memory_mask: (B, K) 记忆有效性 mask.
            mag_selection_weights: (B, K) Context Selector 选择权重.
            memory_readout: (B, T, D) 或 (B, D), 可选的 L1 readout.
            attention_mask: (B, T) 注意力 mask.
            labels: (B, T) 目标 token ids.

        Returns:
            BackboneOutput, 其中 last_hidden_state 已经过 MAG 注入.
        """
        if self._mag_gate is None:
            logger.warning("[SWABackbone] MAG gate 未设置, 回退到标准 forward")
            return self.forward(input_ids, attention_mask, labels, **kwargs)

        # Step 1: 获取所有层的 hidden states
        outputs = self._run_model(input_ids, attention_mask, labels, **kwargs)

        extra = dict(outputs.extra)

        # Step 2: MAG 注入 — 在中间层注入记忆
        if outputs.all_hidden_states is not None and mag_memory_vectors.shape[1] > 0:
            modified_hidden = self._mag_gate.inject_into_all_hidden_states(
                all_hidden_states=outputs.all_hidden_states,
                memory_vectors=mag_memory_vectors,
                memory_mask=mag_memory_mask,
                selection_weights=mag_selection_weights,
            )
            # 使用 MAG 注入后的最后一层 hidden state
            h = modified_hidden[-1]
            extra["mag_injected"] = True
            extra["mag_stats"] = self._mag_gate.get_stats()
        else:
            h = outputs.last_hidden_state
            extra["mag_injected"] = False

        # Step 3: (可选) L1 readout 融合
        if memory_readout is not None:
            if memory_readout.dim() == 2:
                memory_readout = memory_readout.unsqueeze(1).expand_as(h)
            if self._memory_gate is None:
                self._memory_gate = nn.Linear(self._hidden_dim, self._hidden_dim).to(
                    device=h.device, dtype=h.dtype
                )
                nn.init.zeros_(self._memory_gate.bias)
            l1_gate = torch.sigmoid(self._memory_gate(h))
            h = h + l1_gate * memory_readout
            extra["l1_gate_values"] = l1_gate

        # Step 4: 重新计算 logits (使用修改后的 h)
        logits = self._compute_logits(h)

        # 重新计算 loss (如果需要)
        loss = outputs.loss
        if labels is not None and logits is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                labels.to(logits.device).view(-1),
                ignore_index=-100,
            )

        return BackboneOutput(
            last_hidden_state=h,
            logits=logits,
            all_hidden_states=outputs.all_hidden_states,
            attention_mask=outputs.attention_mask,
            loss=loss,
            extra=extra,
        )

    def set_mag_gate(self, mag_gate: nn.Module) -> None:
        """设置 MAG 门控模块。

        Args:
            mag_gate: MAGGate 实例.
        """
        self._mag_gate = mag_gate
        logger.info(
            f"[SWABackbone] 已设置 MAG gate: "
            f"injection_layers={getattr(mag_gate, 'injection_layers', 'N/A')}"
        )

    def _compute_logits(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        """从 hidden states 计算 logits (使用 lm_head)。"""
        if isinstance(self._model, _DebugTransformer):
            return self._model.lm_head(hidden_states)

        # HuggingFace 模型: model.lm_head
        if hasattr(self._model, "lm_head"):
            return self._model.lm_head(hidden_states)

        logger.warning("[SWABackbone] 未找到 lm_head, 无法重新计算 logits")
        return None

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
        对于 debug 模型，使用简单的 greedy/sampling 循环。

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
        # 配置文件中的参数作为默认值，函数参数优先
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
            # 更新 attention_mask
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
        """返回关联的 tokenizer (真实模型模式下可用)。"""
        return self._tokenizer

    def get_hf_model(self) -> Any:
        """返回底层 HuggingFace 模型 (debug 模式下返回 _DebugTransformer)。"""
        return self._model

    def is_debug(self) -> bool:
        """是否为 debug 模式。"""
        return self._is_debug

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def tokenizer(self) -> Any:
        """便捷属性: 访问 tokenizer。"""
        return self._tokenizer

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


def _load_hf_model(cfg: DictConfig) -> tuple[nn.Module, int, int, Any]:
    """从 HuggingFace 加载预训练模型和 tokenizer。

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

    # 设置滑动窗口（如果模型支持）
    window_size = cfg.get("window_size", 4096)
    if hasattr(hf_config, "sliding_window"):
        hf_config.sliding_window = window_size
        logger.info(f"SWA: 设置 sliding_window={window_size}")
    if hasattr(hf_config, "max_window_layers"):
        # Qwen2/Qwen3 风格: 仅前 N 层使用 SWA
        hf_config.max_window_layers = cfg.get(
            "max_window_layers", hf_config.num_hidden_layers
        )
        logger.info(f"SWA: 设置 max_window_layers={hf_config.max_window_layers}")

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
        # 确保 pad_token 存在
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
