"""Q-Filters attention layer & KV cache wrappers.

Design
------
Q-Filters compresses the post-RoPE KV cache by projecting cached keys onto the
per-head top-`rank` right-singular vectors of the calibration Q-matrix and
keeping the highest-scoring `kv_budget - recent_window` keys plus the most
recent `recent_window` keys.

To remain compatible with FlashAttention / SDPA / eager kernels during prefill,
`QFiltersAttention` does NOT reimplement the attention math. It wraps the
original `LlamaAttention.forward`, lets prefill proceed with the full KV, and
then asks the cache to prune itself to `kv_budget` tokens per head. By doing
this as a post-forward callback on a custom `DynamicCache` subclass, any
subsequent forward (next chunk / next decoding step) sees the compressed cache.

The cache subclass, `QFiltersCache`, stores the filter tensors and the config;
the wrapped attention forward triggers `cache.compress_layer(layer_idx)` after
its call chain completes for that layer.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import rotate_half

from .compression import compress_kv

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class QFiltersConfig:
    """Hyperparameters for Q-Filters KV compression.

    Matches the CLI defaults used by `scripts/eval_qfilters.py`. Do NOT mutate
    these defaults inside the code; go through the CLI.

    Attributes:
        kv_budget: total number of tokens kept per head after compression.
        filter_rank: number of right-singular vectors kept per head during
            calibration (= the dimensionality of the filter subspace).
        recent_window: N most-recent tokens that are always kept unpruned.
        calibration_chunks: number of calibration sequences consumed by
            `compute_filters`.
    """

    kv_budget: int = 512
    filter_rank: int = 2
    recent_window: int = 64
    calibration_chunks: int = 8

    def __post_init__(self) -> None:
        if self.kv_budget <= 0:
            raise ValueError(f"kv_budget must be > 0, got {self.kv_budget}")
        if self.filter_rank <= 0:
            raise ValueError(f"filter_rank must be > 0, got {self.filter_rank}")
        if self.recent_window < 0:
            raise ValueError(f"recent_window must be >= 0, got {self.recent_window}")
        if self.recent_window >= self.kv_budget:
            # 2026-04-26: equality silently degenerates compress_kv to pure
            # sliding-window (keep_old<=0 branch in compression.py skips the
            # filter-scoring path → attention sinks at positions 0-3 are
            # evicted → Llama-2 pg19 PPL 1685.88 cliff). Fail loud instead.
            # Ref: ops/research_notes/20260426_qfilters_recent_eq_kv_edge_case.md
            raise ValueError(
                f"recent_window ({self.recent_window}) must be < "
                f"kv_budget ({self.kv_budget}); equality disables filter scoring."
            )
        if self.calibration_chunks <= 0:
            raise ValueError(
                f"calibration_chunks must be > 0, got {self.calibration_chunks}"
            )


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


class QFiltersCache(DynamicCache):
    """DynamicCache that prunes each layer to `config.kv_budget` tokens on demand.

    The attention-level post-forward hook calls `compress_layer(layer_idx)` after
    the vanilla kernel has produced its output; this leaves the current forward
    unaffected and only shrinks state for subsequent forwards.

    Filters are expected as ``{layer_idx: Tensor[H, D, R]}`` (H = num_kv_heads,
    D = head_dim, R = filter_rank). They are moved onto the cache tensor's
    device lazily on first compress call, and cached there.
    """

    def __init__(
        self,
        filters: Dict[int, torch.Tensor],
        config: QFiltersConfig,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(config, QFiltersConfig):
            raise TypeError(f"config must be QFiltersConfig, got {type(config)}")
        self._qf_config = config
        # Don't move filters yet (they may be fp32 CPU tensors); do it lazily.
        self._qf_filters_cpu: Dict[int, torch.Tensor] = dict(filters)
        self._qf_filters_cache: Dict[str, Dict[int, torch.Tensor]] = {}
        # Logical position counter: total tokens EVER inserted into this cache
        # across sub-window forwards. This is distinct from the physical cache
        # length (returned by `get_seq_length()`), which shrinks to `kv_budget`
        # after compression. RoPE needs the logical position so preserved K's
        # pre-compression absolute positions stay aligned with new Q positions.
        # See harness-bug diagnosis 2026-04-25 (S4 repro=172.68 vs S1=7.28).
        self._qf_seen_tokens: int = 0
        # Rotary module for Patch A re-rotation. Set by `attach_rotary` after
        # construction (via make_qfilters_cache).
        self._qf_rotary: Optional[nn.Module] = None

    def attach_rotary(self, rotary: nn.Module) -> None:
        """Attach a LlamaRotaryEmbedding module for post-compression re-rotation."""
        self._qf_rotary = rotary

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        # Accumulate logical seen_tokens on layer 0 only (every forward hits
        # all layers; counting once keeps the counter == new-token-count).
        if layer_idx == 0:
            self._qf_seen_tokens += int(key_states.shape[-2])
        return super().update(key_states, value_states, layer_idx, *args, **kwargs)

    def get_qf_seen_tokens(self) -> int:
        """Logical position cursor (total tokens inserted). Use this for
        `cache_position = arange(seen - T, seen)` so Q's RoPE aligns with
        preserved K's pre-compression absolute positions."""
        return self._qf_seen_tokens

    # ---- helpers ---------------------------------------------------------- #

    def _filters_on(self, layer_idx: int, device: torch.device) -> Optional[torch.Tensor]:
        f = self._qf_filters_cpu.get(layer_idx)
        if f is None:
            return None
        key = f"{device.type}:{device.index if device.index is not None else -1}"
        per_device = self._qf_filters_cache.setdefault(key, {})
        t = per_device.get(layer_idx)
        if t is None:
            t = f.to(device=device, dtype=torch.float32)
            per_device[layer_idx] = t
        return t

    # ---- pruning ---------------------------------------------------------- #

    def compress_layer(self, layer_idx: int) -> None:
        """Prune layer `layer_idx` in place to at most `kv_budget` tokens.

        Patch A (2026-04-25): after selecting kept keys at original positions
        `gather_idx`, apply a delta RoPE rotation so preserved keys' RoPE
        encodes NEW positions `[0, budget)`. This makes physical cache length
        == logical position, so HF's default `cache_position` path (which
        numbers new Q tokens starting from physical seq_len) aligns correctly.
        Without this, preserved K rotated at e.g. pos 960-1023 would be read
        as if at pos 0-63 by HF's default mask/position scheme, creating a
        ~900-token Q-K RoPE offset → PPL blowup (172.68 vs 7.28 dense).
        """
        if layer_idx < 0 or layer_idx >= len(self.layers):
            return
        layer = self.layers[layer_idx]
        if not getattr(layer, "is_initialized", False):
            return
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or values is None or keys.numel() == 0:
            return
        t = keys.shape[-2]
        budget = self._qf_config.kv_budget
        if t <= budget:
            return
        filters = self._filters_on(layer_idx, keys.device)
        if filters is None:
            # No calibrated filters: keep most recent `budget` tokens.
            kept_k = keys[..., -budget:, :].contiguous()
            kept_v = values[..., -budget:, :].contiguous()
            # Build gather_idx = [t-budget, ..., t-1] for the re-rotation step.
            B, H, _, D = keys.shape
            gather_idx = torch.arange(t - budget, t, device=keys.device).view(1, 1, budget).expand(B, H, budget)
        else:
            kept_k, kept_v, gather_idx = compress_kv(
                queries_proj=None,
                filters=filters,
                keys=keys,
                values=values,
                budget=budget,
                recent_window=self._qf_config.recent_window,
            )

        # Patch A: re-rotate kept_k from old_pos (gather_idx) to new_pos [0, budget_eff).
        rotary = getattr(self, "_qf_rotary", None)
        if rotary is not None:
            kept_k = self._rerotate_keys(kept_k, gather_idx, rotary)

        layer.keys = kept_k.contiguous()
        layer.values = kept_v.contiguous()

    @staticmethod
    def _rerotate_keys(
        keys: torch.Tensor,
        old_pos: torch.Tensor,
        rotary: nn.Module,
    ) -> torch.Tensor:
        """Apply delta-RoPE to move keys from `old_pos` to new positions [0, N).

        RoPE(k, p) = k·cos(p) + rot(k)·sin(p). Going from old_pos → new_pos is
        equivalent to applying RoPE with angle (new_pos - old_pos), because
        RoPE rotations compose additively per-frequency.

        Args:
            keys:    [B, H, N, D] at RoPE positions `old_pos`
            old_pos: [B, H, N]    or [1, 1, N] (broadcastable) — original absolute positions
            rotary:  LlamaRotaryEmbedding module (cached on cache by patch_model)

        Returns:
            keys re-rotated to positions [0, N).
        """
        B, H, N, D = keys.shape
        device = keys.device
        # old_pos may be [B, H, N] (from filter path) or [B, H, N] expanded from
        # a smaller source; flatten heads for rotary.forward which wants [B, L].
        # Per-head positions differ in the filter path, so we compute cos/sin
        # per (B, H) and reshape.
        new_pos = torch.arange(N, device=device, dtype=old_pos.dtype).view(1, 1, N).expand(B, H, N)
        delta = new_pos.to(torch.long) - old_pos.to(torch.long)  # [B, H, N]

        # rotary.forward expects position_ids [B, L]; we flatten (B*H, N).
        flat_delta = delta.reshape(B * H, N)
        # Dummy tensor for dtype routing inside rotary.forward.
        dummy = keys.new_empty(B * H, 1, 1, D)
        cos, sin = rotary(dummy, flat_delta)  # [B*H, N, D]
        cos = cos.view(B, H, N, D)
        sin = sin.view(B, H, N, D)

        # Apply: k_new = k * cos + rotate_half(k) * sin (same shape as keys).
        keys_rot = keys * cos + rotate_half(keys) * sin
        return keys_rot


# --------------------------------------------------------------------------- #
# Attention patching
# --------------------------------------------------------------------------- #


class QFiltersAttention:
    """Factory for a forward wrapper around LlamaAttention.forward.

    We intentionally do NOT subclass LlamaAttention; rather, we bind a new
    `forward` function onto each attention instance that (a) delegates to the
    original forward (preserving SDPA / FlashAttention / eager compatibility)
    and then (b) asks the attached `QFiltersCache` to prune its layer.
    """

    @staticmethod
    def make_forward(orig_forward, layer_idx: int):
        """Return a function to install as ``attn.forward``.

        ``orig_forward`` is a bound method (``attn.forward``); we call it
        directly so ``self`` binding is preserved without us having to pass it.
        """

        def forward(
            hidden_states,
            position_embeddings=None,
            attention_mask=None,
            past_key_values=None,
            **kwargs,
        ):
            out = orig_forward(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
            # Post-forward compression. Only QFiltersCache knows how to do this.
            if past_key_values is not None and hasattr(past_key_values, "compress_layer"):
                try:
                    past_key_values.compress_layer(layer_idx)
                except Exception as e:  # pragma: no cover - defensive
                    logger.warning(
                        "QFiltersCache.compress_layer(%d) raised %s; continuing",
                        layer_idx, e,
                    )
            return out

        forward._qfilters_layer_idx = layer_idx
        return forward


def _iter_llama_attention(model: nn.Module):
    """Yield (layer_idx, self_attn) for each Llama-style transformer block."""
    root = getattr(model, "model", model)
    layers = getattr(root, "layers", None)
    if layers is None:
        raise RuntimeError(
            "patch_model: could not find model.model.layers; "
            "only Llama-family architectures are supported."
        )
    for i, block in enumerate(layers):
        attn = getattr(block, "self_attn", None)
        if attn is None:
            continue
        yield i, attn


def patch_model(
    model: nn.Module,
    filters: Dict[int, torch.Tensor],
    config: QFiltersConfig,
) -> nn.Module:
    """Install the Q-Filters post-forward compression hook on every attention layer.

    Side effects:
        * Each attention module's `forward` is replaced with the wrapper that
          triggers `QFiltersCache.compress_layer` after vanilla forward.
        * `model._qfilters_filters` and `model._qfilters_config` are stashed
          so downstream code can construct caches via `make_qfilters_cache`.

    Idempotent: patching an already-patched model is a no-op on that module.
    """
    if not isinstance(config, QFiltersConfig):
        raise TypeError(f"config must be QFiltersConfig, got {type(config)}")
    for layer_idx, attn in _iter_llama_attention(model):
        if getattr(attn, "_qfilters_patched", False):
            continue
        orig_forward = attn.forward
        attn._qfilters_orig_forward = orig_forward
        attn.forward = QFiltersAttention.make_forward(orig_forward, layer_idx)
        attn._qfilters_patched = True
    model._qfilters_filters = filters
    model._qfilters_config = config
    # Stash rotary emb for Patch A re-rotation. Llama models have it at
    # model.model.rotary_emb in HF transformers >= 4.45.
    root = getattr(model, "model", model)
    rotary = getattr(root, "rotary_emb", None)
    if rotary is None:
        # Fallback: look inside first layer's self_attn (older HF).
        for _, attn in _iter_llama_attention(model):
            rotary = getattr(attn, "rotary_emb", None)
            if rotary is not None:
                break
    if rotary is None:
        logger.warning(
            "patch_model: could not locate rotary_emb on model; Patch A "
            "re-rotation disabled (preserved K will use stale RoPE positions)."
        )
    model._qfilters_rotary = rotary
    return model


def unpatch_model(model: nn.Module) -> nn.Module:
    """Undo `patch_model` (mostly for tests / interactive use)."""
    for _, attn in _iter_llama_attention(model):
        if getattr(attn, "_qfilters_patched", False):
            attn.forward = attn._qfilters_orig_forward
            del attn._qfilters_orig_forward
            attn._qfilters_patched = False
    if hasattr(model, "_qfilters_filters"):
        del model._qfilters_filters
    if hasattr(model, "_qfilters_config"):
        del model._qfilters_config
    return model


def make_qfilters_cache(model: nn.Module) -> QFiltersCache:
    """Construct a fresh `QFiltersCache` using the filters/config stashed by `patch_model`."""
    filters = getattr(model, "_qfilters_filters", None)
    config = getattr(model, "_qfilters_config", None)
    if filters is None or config is None:
        raise RuntimeError(
            "make_qfilters_cache: model has not been patched; "
            "call patch_model(model, filters, config) first."
        )
    cache = QFiltersCache(filters, config)
    rotary = getattr(model, "_qfilters_rotary", None)
    if rotary is not None:
        cache.attach_rotary(rotary)
    return cache
