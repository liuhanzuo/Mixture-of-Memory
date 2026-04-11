"""
KVMemoryInjector: 将记忆信息注入到 backbone self-attention 的 KV cache 中。

支持三种注入模式:

1. svd_only (KVMemoryInjector):
   - 原始方案: SVD 压缩 → 虚拟 KV pairs → per-layer α 标量缩放
   - 可训练参数: 4 个标量 (per-layer α)
   - 问题: SVD 虚拟 KV 与正常 KV 语义不匹配, softmax 竞争不公平

2. svd_adapter (KVAdapterInjector):
   - 改进方案: SVD 压缩 → LoRA 适配器变换 → 虚拟 KV pairs
   - 可训练参数: per-layer LoRA_k + LoRA_v (~262K, 共享参数时)
   - 优势: LoRA 让虚拟 KV 学会"伪装"成正常 KV, 解决尺度/语义不匹配

3. raw_kv (RawKVInjector):
   - Trivial baseline: 直接用记忆 token 的原始 KV (不做 SVD)
   - 可训练参数: per-layer α 标量 + 可选 mean-pooling 压缩
   - 优势: 每个虚拟 KV 对应真实 token, 与正常 KV 在同一空间
   - 代价: 虚拟 token 数 = 记忆 token 数 (可能几百个)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ======================================================================
# 配置
# ======================================================================

@dataclass
class KVMemoryInjectorConfig:
    """KVMemoryInjector 通用配置。

    Attributes:
        hidden_dim: Transformer 隐藏维度.
        num_layers: backbone 总层数.
        injection_layers: 注入的 Transformer 层索引列表.
        num_attention_heads: backbone 的 attention head 数.
        num_key_value_heads: backbone 的 KV head 数 (GQA).
        head_dim: 每个 head 的维度.
        init_alpha: α 的初始值 (建议 0.1~0.3, 保守注入).
        max_alpha: α 的上限 (sigmoid 后的最大值, 防止过度注入).
        injection_mode: 注入模式 ("svd_only" | "svd_adapter" | "raw_kv").
        lora_rank: LoRA 适配器的秩 (仅 svd_adapter 模式).
        lora_share_params: 是否跨注入层共享 LoRA 参数.
        max_raw_kv_tokens: raw_kv 模式下最大虚拟 token 数 (超过则 mean-pool 压缩).
    """
    hidden_dim: int = 4096
    num_layers: int = 36
    injection_layers: list[int] = field(default_factory=lambda: [])
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    init_alpha: float = 0.1
    max_alpha: float = 0.5
    # 注入模式
    injection_mode: str = "svd_only"  # "svd_only" | "svd_adapter" | "raw_kv"
    # LoRA 适配器参数 (svd_adapter 模式)
    lora_rank: int = 16
    lora_share_params: bool = True
    # RawKV 参数
    max_raw_kv_tokens: int = 128


# ======================================================================
# 共用的 Attention Forward 逻辑
# ======================================================================

def _custom_attention_forward(
    attn_module: nn.Module,
    hidden_states: torch.Tensor,
    key_states_extra: torch.Tensor,
    value_states_extra: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """自定义 attention forward: 在 KV 前面拼接额外的虚拟 KV tokens。

    Args:
        attn_module: backbone 的 Qwen3Attention 模块.
        hidden_states: (B, T, D) 经过 input_layernorm 后的 hidden states.
        key_states_extra: (B, num_kv_heads, r, head_dim) 额外的 key states.
        value_states_extra: (B, num_kv_heads, r, head_dim) 额外的 value states.
        attention_mask: (B, 1, T, T) causal mask.
        position_embeddings: (cos, sin) rotary embeddings.

    Returns:
        attn_output: (B, T, D) attention 输出.
    """
    B, T, D = hidden_states.shape
    head_dim = attn_module.head_dim
    hidden_shape = (B, T, -1, head_dim)

    # 标准 Q/K/V 投影
    query_states = attn_module.q_norm(attn_module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = attn_module.k_norm(attn_module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = attn_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # 应用 RoPE (只对原始 Q/K, 虚拟 KV 不加 RoPE — 位置无关)
    if position_embeddings is not None:
        cos, sin = position_embeddings
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # ★ 拼接虚拟 KV pairs: [virtual_KV; original_KV]
    r = key_states_extra.shape[2]
    key_states = torch.cat([key_states_extra.to(key_states.dtype), key_states], dim=2)
    value_states = torch.cat([value_states_extra.to(value_states.dtype), value_states], dim=2)

    # 扩展 attention_mask
    if attention_mask is not None:
        virtual_mask = torch.zeros(B, 1, T, r, device=attention_mask.device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([virtual_mask, attention_mask], dim=-1)

    # GQA: repeat KV heads
    from transformers.models.qwen3.modeling_qwen3 import repeat_kv
    num_key_value_groups = attn_module.num_key_value_groups
    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)

    # 标准 scaled dot-product attention
    scaling = attn_module.scaling
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    # Reshape + output projection
    attn_output = attn_output.transpose(1, 2).contiguous()
    input_shape = hidden_states.shape[:-1]
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn_module.o_proj(attn_output)

    return attn_output


def _forward_decoder_layer_with_kv(
    layer_idx: int,
    decoder_layer: nn.Module,
    hidden_states: torch.Tensor,
    key_states_extra: torch.Tensor,
    value_states_extra: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """替代 Qwen3DecoderLayer.forward, 在 self-attention 中注入额外 KV。

    完整复现 DecoderLayer 逻辑:
        residual = h → layernorm → self_attn (带虚拟 KV) → + residual
        residual = h → layernorm → mlp → + residual
    """
    # Pre-attention LayerNorm
    residual = hidden_states
    hidden_states = decoder_layer.input_layernorm(hidden_states)

    # Self-Attention (带虚拟 KV 注入)
    attn_output = _custom_attention_forward(
        attn_module=decoder_layer.self_attn,
        hidden_states=hidden_states,
        key_states_extra=key_states_extra,
        value_states_extra=value_states_extra,
        attention_mask=attention_mask,
        position_embeddings=position_embeddings,
    )
    hidden_states = residual + attn_output

    # MLP
    residual = hidden_states
    hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
    hidden_states = decoder_layer.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states


# ======================================================================
# 方案 1: SVD-Only (原始方案, 仅 per-layer α 标量)
# ======================================================================

class KVMemoryInjector(nn.Module):
    """SVD-Only 注入: 虚拟 KV pairs + per-layer 可学习标量 α。

    可训练参数: 仅 4 个标量 (每个注入层一个 α).
    """

    def __init__(self, config: KVMemoryInjectorConfig | dict[str, Any] | None = None):
        super().__init__()
        if config is None:
            config = KVMemoryInjectorConfig()
        elif isinstance(config, dict):
            config = KVMemoryInjectorConfig(**{
                k: v for k, v in config.items()
                if k in KVMemoryInjectorConfig.__dataclass_fields__
            })
        self.config = config
        self._injection_layers = set(config.injection_layers)

        # Per-layer 可学习标量 α (通过 sigmoid 映射到 [0, max_alpha])
        if config.max_alpha > 0 and config.init_alpha > 0:
            init_logit = math.log(config.init_alpha / (config.max_alpha - config.init_alpha + 1e-8))
        else:
            init_logit = -2.0

        self.alpha_logits = nn.ParameterDict({
            str(layer_idx): nn.Parameter(torch.tensor(init_logit))
            for layer_idx in config.injection_layers
        })

        num_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"[KVMemoryInjector] 初始化: injection_layers={config.injection_layers}, "
            f"init_alpha={config.init_alpha:.3f}, max_alpha={config.max_alpha:.3f}, "
            f"total_params={num_params}"
        )

    @property
    def injection_layers(self) -> set[int]:
        return self._injection_layers

    def get_alpha(self, layer_idx: int) -> torch.Tensor:
        """获取第 layer_idx 层的注入强度 α ∈ [0, max_alpha]。"""
        key = str(layer_idx)
        if key not in self.alpha_logits:
            return torch.tensor(0.0)
        return self.config.max_alpha * torch.sigmoid(self.alpha_logits[key])

    def should_inject(self, layer_idx: int) -> bool:
        return layer_idx in self._injection_layers

    def forward_layer_attn(
        self,
        layer_idx: int,
        attn_module: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        virtual_kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
        inference_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """自定义 attention forward, 在 KV 上拼接虚拟 tokens。"""
        B, T, D = hidden_states.shape
        head_dim = attn_module.head_dim
        hidden_shape = (B, T, -1, head_dim)

        query_states = attn_module.q_norm(attn_module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = attn_module.k_norm(attn_module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = attn_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        alpha_value = None
        if (virtual_kv_cache is not None
                and layer_idx in virtual_kv_cache
                and self.should_inject(layer_idx)):

            virtual_keys, virtual_values = virtual_kv_cache[layer_idx]
            alpha = self.get_alpha(layer_idx) * inference_scale
            alpha_value = alpha.detach()

            scaled_vk = virtual_keys * alpha
            scaled_vv = virtual_values * alpha
            r = virtual_keys.shape[2]

            key_states = torch.cat([scaled_vk.to(key_states.dtype), key_states], dim=2)
            value_states = torch.cat([scaled_vv.to(value_states.dtype), value_states], dim=2)

            if attention_mask is not None:
                virtual_mask = torch.zeros(B, 1, T, r, device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([virtual_mask, attention_mask], dim=-1)

        from transformers.models.qwen3.modeling_qwen3 import repeat_kv
        num_key_value_groups = attn_module.num_key_value_groups
        key_states = repeat_kv(key_states, num_key_value_groups)
        value_states = repeat_kv(value_states, num_key_value_groups)

        scaling = attn_module.scaling
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        input_shape = hidden_states.shape[:-1]
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_module.o_proj(attn_output)

        return attn_output, alpha_value

    def forward_decoder_layer(
        self,
        layer_idx: int,
        decoder_layer: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        virtual_kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
        inference_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """替代 Qwen3DecoderLayer.forward, 在 self-attention 中注入虚拟 KV。"""
        residual = hidden_states
        hidden_states = decoder_layer.input_layernorm(hidden_states)

        alpha_value = None
        if self.should_inject(layer_idx) and virtual_kv_cache is not None:
            attn_output, alpha_value = self.forward_layer_attn(
                layer_idx=layer_idx,
                attn_module=decoder_layer.self_attn,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                virtual_kv_cache=virtual_kv_cache,
                inference_scale=inference_scale,
            )
        else:
            attn_output, _ = decoder_layer.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )

        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
        hidden_states = decoder_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, alpha_value

    def get_stats(self) -> dict[str, Any]:
        stats = {}
        for layer_idx in sorted(self._injection_layers):
            alpha = self.get_alpha(layer_idx)
            stats[f"alpha_layer_{layer_idx}"] = alpha.item()
        return stats

    def __repr__(self) -> str:
        alphas = {int(k): f"{self.get_alpha(int(k)).item():.4f}" for k in self.alpha_logits}
        return f"KVMemoryInjector(injection_layers={sorted(self._injection_layers)}, alphas={alphas})"


# ======================================================================
# 方案 2: SVD + LoRA 适配器 (推荐方案)
# ======================================================================

class KVLoRAAdapter(nn.Module):
    """Per-layer LoRA 适配器: 将 SVD 虚拟 KV 映射到 backbone 能理解的 KV 空间。

    数学:
        K' = K + K @ A_k @ B_k   (LoRA on keys)
        V' = V + V @ A_v @ B_v   (LoRA on values)

    其中 A ∈ ℝ^(head_dim × lora_rank), B ∈ ℝ^(lora_rank × head_dim)。
    初始化: A ~ N(0, 1/sqrt(d)), B = 0 → 初始时 LoRA 输出为 0, 不改变原始 KV。
    """

    def __init__(self, head_dim: int, lora_rank: int = 16):
        super().__init__()
        self.head_dim = head_dim
        self.lora_rank = lora_rank

        # Key LoRA: A_k, B_k
        self.lora_k_A = nn.Parameter(torch.randn(head_dim, lora_rank) / math.sqrt(head_dim))
        self.lora_k_B = nn.Parameter(torch.zeros(lora_rank, head_dim))

        # Value LoRA: A_v, B_v
        self.lora_v_A = nn.Parameter(torch.randn(head_dim, lora_rank) / math.sqrt(head_dim))
        self.lora_v_B = nn.Parameter(torch.zeros(lora_rank, head_dim))

        # Per-layer 可学习缩放 (替代简单的 α 标量, 更细粒度)
        # 初始化为较小值, 让注入从弱开始
        self.scale_k = nn.Parameter(torch.tensor(0.1))
        self.scale_v = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        virtual_keys: torch.Tensor,
        virtual_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """对虚拟 KV 应用 LoRA 适配。

        Args:
            virtual_keys: (B, num_kv_heads, r, head_dim)
            virtual_values: (B, num_kv_heads, r, head_dim)

        Returns:
            adapted_keys: (B, num_kv_heads, r, head_dim)
            adapted_values: (B, num_kv_heads, r, head_dim)
        """
        # LoRA: K' = K + scale_k * K @ A_k @ B_k
        # (B, H, r, d) @ (d, lora_r) @ (lora_r, d) → (B, H, r, d)
        k_delta = torch.matmul(
            torch.matmul(virtual_keys, self.lora_k_A.to(virtual_keys.dtype)),
            self.lora_k_B.to(virtual_keys.dtype),
        )
        adapted_keys = virtual_keys + self.scale_k * k_delta

        v_delta = torch.matmul(
            torch.matmul(virtual_values, self.lora_v_A.to(virtual_values.dtype)),
            self.lora_v_B.to(virtual_values.dtype),
        )
        adapted_values = virtual_values + self.scale_v * v_delta

        return adapted_keys, adapted_values


class KVAdapterInjector(nn.Module):
    """SVD + LoRA 适配器注入: 在 SVD 压缩后用可训练的 LoRA 适配虚拟 KV。

    核心思想:
        SVD 虚拟 KV 的主成分方向与正常 token KV 语义不匹配,
        通过 LoRA 适配器学习一个轻量级变换, 让虚拟 KV "伪装"成
        backbone 能理解的正常 KV, 使 softmax 竞争公平。

    可训练参数 (共享参数时):
        - 1 个 KVLoRAAdapter: 2 × 2 × head_dim × lora_rank
        - 例: head_dim=128, lora_rank=16 → 2 × 2 × 128 × 16 = 8192 + 2 scale = 8194
        - 不共享时: 4 × 8194 = 32776 (4 个注入层)

    Usage::
        injector = KVAdapterInjector(config)
        # virtual_kv_cache 来自 compress_memory_for_kv_injection (SVD 压缩)
        for layer_idx in range(num_layers):
            h, alpha = injector.forward_decoder_layer(
                layer_idx, decoder_layer, h,
                virtual_kv_cache=virtual_kv_cache, ...
            )
    """

    def __init__(self, config: KVMemoryInjectorConfig | dict[str, Any] | None = None):
        super().__init__()
        if config is None:
            config = KVMemoryInjectorConfig()
        elif isinstance(config, dict):
            config = KVMemoryInjectorConfig(**{
                k: v for k, v in config.items()
                if k in KVMemoryInjectorConfig.__dataclass_fields__
            })
        self.config = config
        self._injection_layers = set(config.injection_layers)

        head_dim = config.head_dim
        lora_rank = config.lora_rank

        # 创建 LoRA 适配器
        if config.lora_share_params:
            # 共享参数: 所有注入层用同一个 LoRA (注册为子模块)
            self.shared_adapter = KVLoRAAdapter(head_dim, lora_rank)
            self.adapters = nn.ModuleDict()  # 空的, 不用
        else:
            # 独立参数: 每层一个 LoRA
            self.shared_adapter = None
            self.adapters = nn.ModuleDict({
                str(l): KVLoRAAdapter(head_dim, lora_rank)
                for l in config.injection_layers
            })

        num_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"[KVAdapterInjector] 初始化: injection_layers={config.injection_layers}, "
            f"lora_rank={lora_rank}, share_params={config.lora_share_params}, "
            f"total_params={num_params:,}"
        )

    @property
    def injection_layers(self) -> set[int]:
        return self._injection_layers

    def should_inject(self, layer_idx: int) -> bool:
        return layer_idx in self._injection_layers

    def get_adapter(self, layer_idx: int) -> KVLoRAAdapter | None:
        """获取指定层的 LoRA 适配器。"""
        if self.shared_adapter is not None:
            return self.shared_adapter
        key = str(layer_idx)
        return self.adapters.get(key, None)

    def get_alpha(self, layer_idx: int) -> torch.Tensor:
        """获取第 layer_idx 层的等效注入强度 (LoRA scale_k 和 scale_v 的均值)。

        KVAdapterInjector 没有显式的 alpha 参数，但 LoRA adapter 的
        scale_k/scale_v 起到了类似的作用。返回它们的均值作为等效 alpha，
        以兼容 eval_mag.py 中的 eval_gate_activation 评估。
        """
        adapter = self.get_adapter(layer_idx)
        if adapter is None:
            return torch.tensor(0.0)
        return (adapter.scale_k.abs() + adapter.scale_v.abs()) / 2

    def forward_decoder_layer(
        self,
        layer_idx: int,
        decoder_layer: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        virtual_kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
        inference_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """替代 Qwen3DecoderLayer.forward, 在 self-attention 中注入经 LoRA 适配的虚拟 KV。"""
        alpha_value = None

        if (self.should_inject(layer_idx)
                and virtual_kv_cache is not None
                and layer_idx in virtual_kv_cache):

            virtual_keys, virtual_values = virtual_kv_cache[layer_idx]
            adapter = self.get_adapter(layer_idx)

            if adapter is not None:
                # ★ LoRA 适配: 让虚拟 KV 学会匹配正常 KV 的分布
                adapted_keys, adapted_values = adapter(virtual_keys, virtual_values)
            else:
                adapted_keys, adapted_values = virtual_keys, virtual_values

            # inference_scale 缩放
            if inference_scale != 1.0:
                adapted_keys = adapted_keys * inference_scale
                adapted_values = adapted_values * inference_scale

            # 记录 scale 值用于日志
            if adapter is not None:
                alpha_value = torch.tensor(
                    (adapter.scale_k.detach().item() + adapter.scale_v.detach().item()) / 2
                )

            # 使用共用的 decoder layer forward
            hidden_states = _forward_decoder_layer_with_kv(
                layer_idx=layer_idx,
                decoder_layer=decoder_layer,
                hidden_states=hidden_states,
                key_states_extra=adapted_keys,
                value_states_extra=adapted_values,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )
        else:
            # 标准 forward (无注入)
            residual = hidden_states
            hidden_states = decoder_layer.input_layernorm(hidden_states)
            attn_output, _ = decoder_layer.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )
            hidden_states = residual + attn_output
            residual = hidden_states
            hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
            hidden_states = decoder_layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

        return hidden_states, alpha_value

    def get_stats(self) -> dict[str, Any]:
        stats = {}
        for layer_idx in sorted(self._injection_layers):
            adapter = self.get_adapter(layer_idx)
            if adapter is not None:
                stats[f"scale_k_layer_{layer_idx}"] = adapter.scale_k.item()
                stats[f"scale_v_layer_{layer_idx}"] = adapter.scale_v.item()
                # LoRA 权重范数 (诊断用)
                stats[f"lora_k_norm_layer_{layer_idx}"] = (
                    adapter.lora_k_A.norm().item() * adapter.lora_k_B.norm().item()
                )
                stats[f"lora_v_norm_layer_{layer_idx}"] = (
                    adapter.lora_v_A.norm().item() * adapter.lora_v_B.norm().item()
                )
        return stats

    def __repr__(self) -> str:
        info_parts = []
        for layer_idx in sorted(self._injection_layers):
            adapter = self.get_adapter(layer_idx)
            if adapter is not None:
                info_parts.append(
                    f"L{layer_idx}(sk={adapter.scale_k.item():.4f},sv={adapter.scale_v.item():.4f})"
                )
        return f"KVAdapterInjector(layers={sorted(self._injection_layers)}, {', '.join(info_parts)})"


# ======================================================================
# 方案 3: Raw KV 直接注入 (Trivial Baseline)
# ======================================================================

class RawKVInjector(nn.Module):
    """MemoryLLM 风格的原始 KV 直接注入，支持可学习的 per-layer alpha 缩放。

    核心思想 (借鉴 MemoryLLM: https://arxiv.org/abs/2402.04624):
        直接将记忆文本过 backbone 各层后得到的 token-level KV concat 到
        正常 token 的 KV 前面。

    改进 (v2): 添加可学习的 per-layer alpha 缩放
        - 初始化为极小值 (sigmoid(-5) ≈ 0.007)，从极弱注入开始
        - 训练时 alpha 可以逐步增大，但受 max_alpha 限制
        - 解决原始版本全量注入 (alpha=1.0) 导致 PPL 爆炸的问题
        - 设置 learnable_alpha=False 可退回原始 MemoryLLM 行为 (alpha=1.0)

    可训练参数:
        - learnable_alpha=True: per-layer alpha logit (每层 1 个标量)
        - learnable_alpha=False: 无 (纯 concat)
    """

    def __init__(self, config: KVMemoryInjectorConfig | dict[str, Any] | None = None):
        super().__init__()
        if config is None:
            config = KVMemoryInjectorConfig()
        elif isinstance(config, dict):
            config = KVMemoryInjectorConfig(**{
                k: v for k, v in config.items()
                if k in KVMemoryInjectorConfig.__dataclass_fields__
            })
        self.config = config
        self._injection_layers = set(config.injection_layers)

        # ★ 可学习 alpha: 从极小值开始，逐步增强
        # init_alpha 和 max_alpha 控制注入强度范围
        # 如果 init_alpha > 0 且 max_alpha > 0，启用可学习 alpha
        self._learnable_alpha = (config.init_alpha > 0 and config.max_alpha > 0)

        if self._learnable_alpha:
            # 计算初始 logit: sigmoid(logit) * max_alpha = init_alpha
            # → logit = log(init_alpha / (max_alpha - init_alpha))
            init_alpha_clamped = min(config.init_alpha, config.max_alpha - 1e-6)
            init_logit = math.log(init_alpha_clamped / (config.max_alpha - init_alpha_clamped + 1e-8))

            self.alpha_logits = nn.ParameterDict({
                str(layer_idx): nn.Parameter(torch.tensor(init_logit))
                for layer_idx in config.injection_layers
            })
            logger.info(
                f"[RawKVInjector] 可学习 alpha 已启用: "
                f"init_alpha={config.init_alpha:.4f}, max_alpha={config.max_alpha:.3f}, "
                f"init_logit={init_logit:.3f}, "
                f"injection_layers={config.injection_layers}"
            )
        else:
            self.alpha_logits = nn.ParameterDict()  # 空的，兼容 state_dict
            logger.info(
                f"[RawKVInjector] MemoryLLM 风格初始化 (无 alpha 缩放): "
                f"injection_layers={config.injection_layers}, "
                f"max_raw_kv_tokens={config.max_raw_kv_tokens}"
            )

    @property
    def injection_layers(self) -> set[int]:
        return self._injection_layers

    def should_inject(self, layer_idx: int) -> bool:
        return layer_idx in self._injection_layers

    def get_alpha(self, layer_idx: int) -> torch.Tensor:
        """获取第 layer_idx 层的注入强度 alpha。

        - learnable_alpha=True: alpha = sigmoid(logit) * max_alpha ∈ [0, max_alpha]
        - learnable_alpha=False: alpha = 1.0 (原始 MemoryLLM 行为)
        """
        if not self._learnable_alpha:
            return torch.tensor(1.0)
        key = str(layer_idx)
        if key not in self.alpha_logits:
            return torch.tensor(1.0)
        return self.config.max_alpha * torch.sigmoid(self.alpha_logits[key])

    def forward_decoder_layer(
        self,
        layer_idx: int,
        decoder_layer: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        virtual_kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
        inference_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """替代 Qwen3DecoderLayer.forward, 直接 concat 原始 token-level KV。

        改进: 支持可学习的 alpha 缩放，从极弱注入开始。
        """
        if (self.should_inject(layer_idx)
                and virtual_kv_cache is not None
                and layer_idx in virtual_kv_cache):

            raw_keys, raw_values = virtual_kv_cache[layer_idx]
            # raw_keys: (B, num_kv_heads, N_tokens, head_dim)

            # ★ 应用 alpha 缩放 (可学习 or 固定 1.0)
            alpha = self.get_alpha(layer_idx)
            effective_scale = alpha * inference_scale
            alpha_value = alpha.detach()

            if effective_scale != 1.0:
                raw_keys = raw_keys * effective_scale
                raw_values = raw_values * effective_scale

            hidden_states = _forward_decoder_layer_with_kv(
                layer_idx=layer_idx,
                decoder_layer=decoder_layer,
                hidden_states=hidden_states,
                key_states_extra=raw_keys,
                value_states_extra=raw_values,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )
            return hidden_states, alpha_value
        else:
            residual = hidden_states
            hidden_states = decoder_layer.input_layernorm(hidden_states)
            attn_output, _ = decoder_layer.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )
            hidden_states = residual + attn_output
            residual = hidden_states
            hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
            hidden_states = decoder_layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

        return hidden_states, None

    def get_stats(self) -> dict[str, Any]:
        """返回每层的 alpha 值统计。"""
        stats = {}
        for layer_idx in sorted(self._injection_layers):
            alpha = self.get_alpha(layer_idx)
            stats[f"alpha_layer_{layer_idx}"] = alpha.item()
        return stats

    def __repr__(self) -> str:
        if self._learnable_alpha:
            alphas = {int(k): f"{self.get_alpha(int(k)).item():.4f}" for k in self.alpha_logits}
            return f"RawKVInjector(learnable_alpha, injection_layers={sorted(self._injection_layers)}, alphas={alphas})"
        return f"RawKVInjector(MemoryLLM-style, injection_layers={sorted(self._injection_layers)}, no_alpha)"


# ======================================================================
# 工厂函数
# ======================================================================

def create_kv_injector(
    config: KVMemoryInjectorConfig | dict[str, Any],
) -> KVMemoryInjector | KVAdapterInjector | RawKVInjector:
    """根据 injection_mode 创建对应的注入器。

    Args:
        config: 注入器配置.

    Returns:
        对应模式的注入器实例.
    """
    if isinstance(config, dict):
        config = KVMemoryInjectorConfig(**{
            k: v for k, v in config.items()
            if k in KVMemoryInjectorConfig.__dataclass_fields__
        })

    mode = config.injection_mode
    if mode == "svd_only":
        return KVMemoryInjector(config)
    elif mode == "svd_adapter":
        return KVAdapterInjector(config)
    elif mode == "raw_kv":
        return RawKVInjector(config)
    else:
        raise ValueError(f"未知的 injection_mode: {mode}, 可选: svd_only, svd_adapter, raw_kv")


# ======================================================================
# 记忆压缩函数
# ======================================================================

def compress_memory_for_kv_injection(
    memory_hidden_states: dict[int, torch.Tensor],
    backbone_layers: nn.ModuleList,
    attention_mask: torch.Tensor | None = None,
    svd_rank: int = 8,
    normalize_keys: bool = True,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """将 per-layer 记忆 hidden states 压缩为虚拟 KV pairs (SVD 方式)。

    用于 svd_only 和 svd_adapter 模式。

    核心流程 (对每层 l):
        1. 用第 l 层的 k_proj/v_proj/k_norm 投影 hidden states → K, V
        2. 分头: K → (B, num_kv_heads, N, head_dim), V 同理
        3. 计算 M_h = K_h^T V_h ∈ ℝ^(head_dim × head_dim)
        4. SVD: M_h ≈ U_r Σ_r V_r^T
        5. 虚拟 KV: K̃_h = U_r √Σ_r, Ṽ_h = V_r √Σ_r

    注意: 不加 RoPE! 虚拟 KV 是位置无关的。
    """
    virtual_kv_cache = {}

    for layer_idx, hidden_states in memory_hidden_states.items():
        B, N, D = hidden_states.shape

        attn = backbone_layers[layer_idx].self_attn
        head_dim = attn.head_dim
        num_kv_heads = attn.k_proj.out_features // head_dim

        with torch.no_grad():
            k_all = attn.k_proj(hidden_states)
            v_all = attn.v_proj(hidden_states)

            if hasattr(attn, 'k_norm'):
                k_all = attn.k_norm(k_all.view(B, N, num_kv_heads, head_dim)).view(B, N, -1)

            if attention_mask is not None:
                mask_3d = attention_mask.unsqueeze(-1).to(k_all.dtype)
                k_all = k_all * mask_3d
                v_all = v_all * mask_3d

            k_heads = k_all.view(B, N, num_kv_heads, head_dim).permute(0, 2, 1, 3)
            v_heads = v_all.view(B, N, num_kv_heads, head_dim).permute(0, 2, 1, 3)

            M = torch.matmul(k_heads.transpose(-2, -1), v_heads)

            r = min(svd_rank, head_dim)
            M_f32 = M.float()

            try:
                U, S, Vh = torch.linalg.svd(M_f32, full_matrices=False)
            except RuntimeError as e:
                logger.warning(f"[compress_memory] Layer {layer_idx} SVD 失败 ({e}), 使用随机 fallback")
                vk = torch.randn(B, num_kv_heads, r, head_dim, device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
                vv = torch.randn(B, num_kv_heads, r, head_dim, device=hidden_states.device, dtype=hidden_states.dtype) * 0.01
                virtual_kv_cache[layer_idx] = (vk, vv)
                continue

            U_r = U[:, :, :, :r]
            S_r = S[:, :, :r]
            Vh_r = Vh[:, :, :r, :]

            sqrt_S = torch.sqrt(S_r.clamp(min=1e-12))

            virtual_keys = (U_r * sqrt_S.unsqueeze(-2)).transpose(-2, -1)
            virtual_values = Vh_r * sqrt_S.unsqueeze(-1)

            virtual_keys = virtual_keys.to(dtype=hidden_states.dtype)
            virtual_values = virtual_values.to(dtype=hidden_states.dtype)

            if normalize_keys:
                virtual_keys = F.normalize(virtual_keys, dim=-1)

        virtual_kv_cache[layer_idx] = (virtual_keys, virtual_values)

        logger.debug(
            f"[compress_memory] Layer {layer_idx}: "
            f"{N} tokens → {r} virtual KV pairs "
            f"(num_kv_heads={num_kv_heads}, head_dim={head_dim})"
        )

    return virtual_kv_cache


def extract_raw_kv_for_injection(
    memory_hidden_states: dict[int, torch.Tensor],
    backbone_layers: nn.ModuleList,
    attention_mask: torch.Tensor | None = None,
    max_tokens: int = 128,
    with_grad: bool = False,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """将 per-layer 记忆 hidden states 直接投影为 token-level KV pairs (不做 SVD)。

    用于 raw_kv 模式 (MemoryLLM 风格)。

    核心流程 (对每层 l):
        1. 用第 l 层的 k_proj/v_proj/k_norm 投影 hidden states → K, V
        2. 分头: K → (B, num_kv_heads, N, head_dim), V 同理
        3. 如果 N > max_tokens, 做 chunk mean-pooling 压缩到 max_tokens
        4. 直接返回 token-level KV (不做 SVD)

    注意: 不加 RoPE! 虚拟 KV 是位置无关的 (cross-attention 语义)。

    Args:
        memory_hidden_states: {layer_idx: (B, N, D)} 每层的记忆 hidden states.
        backbone_layers: backbone 的 nn.ModuleList.
        attention_mask: (B, N) token 级别 mask (1=有效, 0=padding).
        max_tokens: 最大虚拟 token 数, 超过则 mean-pool 压缩.
        with_grad: 是否保留梯度 (端到端训练时设为 True, 让梯度流过 KV 投影).

    Returns:
        raw_kv_cache: {layer_idx: (raw_keys, raw_values)}
            raw_keys: (B, num_kv_heads, N', head_dim)  N' = min(N_valid, max_tokens)
            raw_values: (B, num_kv_heads, N', head_dim)
    """
    raw_kv_cache = {}

    for layer_idx, hidden_states in memory_hidden_states.items():
        B, N, D = hidden_states.shape

        attn = backbone_layers[layer_idx].self_attn
        head_dim = attn.head_dim
        num_kv_heads = attn.k_proj.out_features // head_dim

        # ★ MemoryLLM 风格: with_grad=True 时保留梯度, 让端到端训练的梯度
        #   流过 k_proj/v_proj → backbone LoRA 参数
        ctx = torch.no_grad() if not with_grad else torch.enable_grad()
        with ctx:
            k_all = attn.k_proj(hidden_states)  # (B, N, num_kv_heads * head_dim)
            v_all = attn.v_proj(hidden_states)  # (B, N, num_kv_heads * head_dim)

            if hasattr(attn, 'k_norm'):
                k_all = attn.k_norm(k_all.view(B, N, num_kv_heads, head_dim)).view(B, N, -1)

            # 应用 attention_mask: 去掉 padding tokens
            if attention_mask is not None:
                mask_3d = attention_mask.unsqueeze(-1).to(k_all.dtype)  # (B, N, 1)
                k_all = k_all * mask_3d
                v_all = v_all * mask_3d
                # 计算有效 token 数
                valid_count = attention_mask.sum(dim=1).long()  # (B,)
                max_valid = valid_count.max().item()
            else:
                max_valid = N

            # 分头: (B, N, num_kv_heads, head_dim) → (B, num_kv_heads, N, head_dim)
            k_heads = k_all.view(B, N, num_kv_heads, head_dim).permute(0, 2, 1, 3)
            v_heads = v_all.view(B, N, num_kv_heads, head_dim).permute(0, 2, 1, 3)

            # 截断到有效 token 数
            if max_valid < N:
                k_heads = k_heads[:, :, :max_valid, :]
                v_heads = v_heads[:, :, :max_valid, :]

            actual_tokens = k_heads.shape[2]

            # 如果超过 max_tokens, 做 chunk mean-pooling 压缩
            if actual_tokens > max_tokens and max_tokens > 0:
                chunk_size = math.ceil(actual_tokens / max_tokens)
                # Pad to multiple of chunk_size
                pad_len = chunk_size * max_tokens - actual_tokens
                if pad_len > 0:
                    k_pad = torch.zeros(B, num_kv_heads, pad_len, head_dim,
                                        device=k_heads.device, dtype=k_heads.dtype)
                    v_pad = torch.zeros(B, num_kv_heads, pad_len, head_dim,
                                        device=v_heads.device, dtype=v_heads.dtype)
                    k_heads = torch.cat([k_heads, k_pad], dim=2)
                    v_heads = torch.cat([v_heads, v_pad], dim=2)

                # Reshape and mean-pool
                total_len = k_heads.shape[2]
                k_heads = k_heads.view(B, num_kv_heads, max_tokens, chunk_size, head_dim).mean(dim=3)
                v_heads = v_heads.view(B, num_kv_heads, max_tokens, chunk_size, head_dim).mean(dim=3)

                logger.debug(
                    f"[extract_raw_kv] Layer {layer_idx}: "
                    f"{actual_tokens} tokens → {max_tokens} (chunk_size={chunk_size})"
                )

        raw_kv_cache[layer_idx] = (k_heads, v_heads)

        logger.debug(
            f"[extract_raw_kv] Layer {layer_idx}: "
            f"{N} tokens → {k_heads.shape[2]} raw KV pairs "
            f"(num_kv_heads={num_kv_heads}, head_dim={head_dim})"
            f"{' [with_grad]' if with_grad else ''}"
        )

    return raw_kv_cache
