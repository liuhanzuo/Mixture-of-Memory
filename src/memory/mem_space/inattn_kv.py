"""TRUE in-attention K/V concat channel (2026-06-18).

The architecturally-correct readout test for the memory project. Two prior
probes (raw-KV +1.0, evidence-oracle +2.5) both injected retrieved precise
tokens as a PREFIX block (position-0 / landmark prefix) and both failed → the
bottleneck is the frozen reader cannot consume injected KV through a prefix
block. The one mechanism never tested: concatenate the retrieved raw K,V
DIRECTLY into the wrapped self-attention's native key/value tensors so the
H-query tokens attend over ``[native_KV ; retrieved_raw_KV]`` in ONE softmax,
with the retrieved KV carrying their REAL source RoPE positions (landmark §4b).

This module installs a thin wrapper around a single ``LlamaAttention.forward``.
When no injection is stashed (``self._inattn_kv is None``) it calls the ORIGINAL
bound forward unchanged → byte-identical. When the wrapping MemorySpaceLayer has
stashed retrieved (already-RoPE'd) K and (raw) V, the wrapper appends them to the
native key/value on the sequence axis and extends the attention mask by an
all-allowed block of width R so every query may attend to them (keys-only: the
retrieved KV are never queries, so they produce no output rows).

Implementation mirrors transformers 5.5.4 ``LlamaAttention.forward``
(models/llama/modeling_llama.py:251-289). It is intentionally version-coupled —
this is an eval-time probe, not a production training path.
"""
from __future__ import annotations

import types
from typing import Callable, Optional, Tuple

import torch

from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    rotate_half,
)


def rope_keys_only(
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """Apply RoPE to a key tensor only (q-free variant of apply_rotary_pos_emb).

    ``k``: [B, n_kv, R, head_dim].  ``cos``/``sin``: [B, R, head_dim].
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (k * cos) + (rotate_half(k) * sin)


def build_retrieved_kv(
    self_attn: torch.nn.Module,
    retrieved_hidden: torch.Tensor,
    retrieved_pos: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    pre_norm: Optional[torch.nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project retrieved raw token hidden states through the layer's NATIVE
    k_proj / v_proj and RoPE the keys at their REAL source positions.

    Args:
        self_attn: the wrapped LlamaAttention (provides k_proj/v_proj/head_dim).
        retrieved_hidden: [B, R, d_model] retrieved raw token hidden states
            (the LAYER-INPUT, pre-LayerNorm hidden — same representation the
            store / oracle capture hold).
        retrieved_pos: [B, R] long source in-chunk positions (RoPE phase).
        position_embeddings: (cos, sin) of the CURRENT chunk, each [B|1, T, hd].
        pre_norm: the wrapped decoder layer's ``input_layernorm``. Native K/V are
            computed as ``k_proj(input_layernorm(h))``; the retrieved hidden are
            pre-LN layer inputs too, so we MUST apply the same norm before
            projecting for the injected K/V to live in the native distribution.
            When None (smoke / debug) the norm is skipped.

    Returns:
        (K_raw, V_raw): each [B, n_kv_heads, R, head_dim]; K_raw is RoPE'd at the
        retrieved source positions, V_raw is unrotated (Llama V carries no RoPE).
    """
    B, R, _ = retrieved_hidden.shape
    hd = self_attn.head_dim
    _h = retrieved_hidden
    if pre_norm is not None:
        _h = pre_norm(_h)
    K_raw = self_attn.k_proj(_h).view(B, R, -1, hd).transpose(1, 2)
    V_raw = self_attn.v_proj(_h).view(B, R, -1, hd).transpose(1, 2)

    cos, sin = position_embeddings                       # [B|1, T, hd]
    T = cos.shape[1]
    _pos = retrieved_pos.to(device=cos.device, dtype=torch.long).clamp_(0, T - 1)
    if cos.shape[0] == 1 and B > 1:
        cos = cos.expand(B, T, cos.shape[-1])
        sin = sin.expand(B, T, sin.shape[-1])
    _gi = _pos.unsqueeze(-1).expand(B, R, cos.shape[-1])  # [B, R, hd]
    cos_r = cos.gather(1, _gi)                            # [B, R, hd]
    sin_r = sin.gather(1, _gi)
    K_raw = rope_keys_only(K_raw, cos_r.to(K_raw.dtype), sin_r.to(K_raw.dtype))
    return K_raw, V_raw


def make_inattn_attention_forward(self_attn: torch.nn.Module) -> Callable:
    """Return a forward that appends ``self._inattn_kv`` to the native K/V.

    Installed on the chosen layer's ``self_attn`` (binds the original forward as
    ``self_attn._orig_forward``). When ``self._inattn_kv`` is None the original
    forward runs unchanged (byte-identical).
    """
    orig_forward = self_attn.forward

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings=None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        **kwargs,
    ):
        injected = getattr(self, "_inattn_kv", None)
        if injected is None:
            # No injection this call → exact native path (byte-identical).
            return orig_forward(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )

        # Stash is either (K_raw, V_raw) — the original eval-time probe — or
        # (K_raw, V_raw, col_bias) — Method A's raw-KV readout, where col_bias is
        # a grad-bearing additive LOG-weight per (query-token, retrieved-token)
        # injected into the retrieved KV columns so the trainable gist scorer's
        # selection weight participates in the ONE softmax (Landmark §4b:
        # cross-block weight = token-score × landmark-score, in additive space).
        col_bias = None
        if len(injected) == 3:
            K_raw, V_raw, col_bias = injected            # [B,n_kv,R,hd], [B,Tq,R]
        else:
            K_raw, V_raw = injected                      # [B, n_kv, R, hd] each
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        Lq = q.shape[2]
        Lk_native = k.shape[2]
        R = K_raw.shape[2]

        # Concat retrieved K/V onto the native key/value sequence axis. The
        # retrieved KV are KEYS-ONLY (not appended to q) so they never produce
        # output rows; every query may attend to them (extra native context).
        k = torch.cat([k, K_raw.to(k.dtype)], dim=2)     # [B, n_kv, Lk+R, hd]
        v = torch.cat([v, V_raw.to(v.dtype)], dim=2)

        # Build / extend an explicit additive mask of shape [B, 1, Lq, Lk+R].
        neg_inf = torch.finfo(q.dtype).min
        if attention_mask is None:
            # Native path was implicit-causal (SDPA is_causal). Reconstruct the
            # causal [Lq, Lk_native] block, then append R all-allowed columns.
            base = torch.zeros(Lq, Lk_native, dtype=q.dtype, device=q.device)
            base = base.masked_fill(
                torch.triu(
                    torch.ones(Lq, Lk_native, dtype=torch.bool, device=q.device),
                    diagonal=1,
                ),
                neg_inf,
            )
            base = base.view(1, 1, Lq, Lk_native).expand(
                q.shape[0], 1, Lq, Lk_native
            )
        else:
            # An explicit 4-D additive mask was supplied (e.g. the extended seq).
            base = attention_mask
            if base.dim() == 4 and base.shape[-1] != Lk_native:
                # Defensive: slice/trust the native-key columns.
                base = base[..., :Lk_native]
        allowed = torch.zeros(
            base.shape[0], 1, Lq, R, dtype=base.dtype, device=base.device
        )
        if col_bias is not None:
            # col_bias: [B, Tq, R] additive log-weight on the retrieved columns.
            # Broadcast onto the [B, 1, Lq, R] allowed block (one head axis). Only
            # applied when the query length matches (the bypass / real LM forward,
            # Tq == Lq); the wrapped layer may also be invoked on a DIFFERENT
            # extended sequence (Lq != Tq) where the per-token bias does not
            # align — there the retrieved columns keep a uniform (zero) bias.
            _cb = col_bias.to(dtype=base.dtype, device=base.device)
            if _cb.dim() == 3 and _cb.shape[-1] == R and _cb.shape[1] == Lq:
                allowed = allowed + _cb.unsqueeze(1)
        full_mask = torch.cat([base, allowed], dim=-1)   # [B, 1, Lq, Lk+R]

        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.llama.modeling_llama import (
            eager_attention_forward,
        )

        # (B) TWO-STAGE grouped-softmax readout (2026-06-20 within-block dilution
        # fix). When enabled, the retrieved R columns are NOT flattened into one
        # softmax with the native keys; instead each `group_size`-token sub-block
        # competes as a single unit at the top level (stage 1) with an internal
        # softmax among its own tokens (stage 2) → a 64-token block keeps the
        # needle at ~40% mass instead of drowning it in a flat softmax. Pure
        # inference (gated behind the flag); off → byte-identical native path.
        _grouped = getattr(self, "_rawkv_grouped_readout", False)
        _gs = int(getattr(self, "_rawkv_subblock_size", 64))
        if _grouped and R > 0 and (R % _gs == 0):
            from ._grouped_two_stage_attention_ref import (
                grouped_two_stage_attention,
            )
            n_rep = q.shape[1] // k.shape[1]
            _kh = k.repeat_interleave(n_rep, dim=1) if n_rep > 1 else k
            _vh = v.repeat_interleave(n_rep, dim=1) if n_rep > 1 else v
            attn_output, attn_weights = grouped_two_stage_attention(
                q, _kh, _vh, base_mask=base, Lk_native=Lk_native, R=R,
                group_size=_gs, scaling=self.scaling,
                block_logbias=None,   # variant A (equal-weight grouping)
            )
            attn_output = attn_output.transpose(1, 2).reshape(
                *input_shape, -1
            ).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            q,
            k,
            v,
            full_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return types.MethodType(forward, self_attn)


def install_inattn_wrapper(self_attn: torch.nn.Module) -> None:
    """Replace ``self_attn.forward`` with the injection-aware wrapper (idempotent).

    The original bound forward is captured inside the closure; the stash slot
    ``_inattn_kv`` defaults to None so the very first call is byte-identical.
    """
    if getattr(self_attn, "_inattn_installed", False):
        return
    self_attn._inattn_kv = None
    self_attn.forward = make_inattn_attention_forward(self_attn)
    self_attn._inattn_installed = True
