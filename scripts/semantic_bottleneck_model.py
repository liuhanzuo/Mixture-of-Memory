#!/usr/bin/env python3
"""Semantic-bottleneck 1B Llama — shared builder for train + probe.

Feasibility experiment (2026-07-07): does explicitly forcing the semantic
information to funnel through a low-rank BOTTLENECK at a mid layer ``j`` make the
"early/mid = understand, top = generate" division of labour cleaner?

Design (option (a), the simplest workable one): a HARD low-rank projection
inserted on the OUTPUT of decoder layer ``j``::

    h_j  ->  down(d -> d_bottle)  ->  GELU  ->  up(d_bottle -> d)  ->  h_j'

with NO residual bypass, so every bit of information that layers ``j+1..L`` (the
"generation" stack) receive must pass through a rank-<=d_bottle channel. This is
exactly the point: it forces the semantic state to be *compressible* at depth j,
which is also the quantity QCMem caches (``h_j``). The baseline arm uses
``bottleneck_dim=0`` -> the wrapper is a no-op -> standard from-scratch Llama.

Both arms share this builder so layer-wise probing can rebuild the exact arch
and load the raw ``state_dict`` checkpoint.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM


# Per-size architecture shapes. Keys are the only fields that differ across the
# 1b/3b/7b scale-up ladder; everything else (rope, norm eps, no bias/dropout) is
# shared and set in make_config below.
#   1b: Llama-3.2-1B         (hidden 2048 / 16L / 32h / 8kv / hd 64  / ffn 8192)
#   3b: Llama-3.2-3B         (hidden 3072 / 28L / 24h / 8kv / hd 128 / ffn 8192)
#   7b: Llama-2-7B-ish       (hidden 4096 / 32L / 32h /32kv / hd 128 / ffn 11008)
_SIZE_SHAPES = {
    "1b": dict(
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=64,
        tie_word_embeddings=True,
    ),
    "3b": dict(
        hidden_size=3072,
        intermediate_size=8192,
        num_hidden_layers=28,
        num_attention_heads=24,
        num_key_value_heads=8,
        head_dim=128,
        tie_word_embeddings=True,
    ),
    "7b": dict(
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        tie_word_embeddings=False,
    ),
}


def make_config(size: str = "1b", vocab_size: int = 128256, seq_len: int = 4096) -> LlamaConfig:
    """Llama-shaped config (random init, trained from scratch) for a given size.

    size in {"1b", "3b", "7b"}. Shared fields (rope_theta=500000, silu,
    rms_norm_eps=1e-5, no attention bias/dropout) match the original 1B recipe.
    """
    size = size.lower()
    if size not in _SIZE_SHAPES:
        raise ValueError(f"unknown size {size!r}; expected one of {sorted(_SIZE_SHAPES)}")
    shape = _SIZE_SHAPES[size]
    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=shape["hidden_size"],
        intermediate_size=shape["intermediate_size"],
        num_hidden_layers=shape["num_hidden_layers"],
        num_attention_heads=shape["num_attention_heads"],
        num_key_value_heads=shape["num_key_value_heads"],
        head_dim=shape["head_dim"],
        hidden_act="silu",
        max_position_embeddings=max(seq_len, 4096),
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
        tie_word_embeddings=shape["tie_word_embeddings"],
        attention_bias=False,
        attention_dropout=0.0,
    )


def make_1b_config(vocab_size: int = 128256, seq_len: int = 4096) -> LlamaConfig:
    """Backward-compatible alias for make_config("1b", ...).

    Kept so the currently-running 1B training keeps its exact old behaviour.
    """
    return make_config("1b", vocab_size=vocab_size, seq_len=seq_len)


class BottleneckLayer(nn.Module):
    """Wraps one LlamaDecoderLayer; funnels its output through d->d_bottle->d.

    transformers>=5 LlamaDecoderLayer.forward returns a plain tensor. We keep a
    tuple-safe path just in case.

    Split point for QCMem's d_bottle-width persist path (2026-08-17)
    ---------------------------------------------------------------
    ``forward`` computes ``up(act(down(h)))``, so its OUTPUT is back at
    ``hidden_size``. When this layer is the last layer of a QCMem WRITE band
    (``resume_j == bottleneck_layer + 1``),
    :meth:`src.memory.qcmem.qcmem_model.QCMemModel._write_band` deliberately stops
    ONE OP SHORT and persists ``act(down(h))`` (width ``d_bottle``), re-applying
    ``self.up`` on the READ side. That is an exact algebraic rearrangement, not an
    approximation: ``f(up(s))`` with ``s = act(down(h))`` is the same composition as
    ``f(up(act(down(h))))``.

    Three properties of THIS module are what make the split legal, so a redesign
    must re-check them:
      1. ``up`` is a bias-free ``nn.Linear`` — a pure linear map applied last.
      2. ``act`` is ELEMENTWISE (GELU) and sits BEFORE ``up``, so the stored
         post-activation value is token-local; nothing after the store mixes
         tokens except the read band's own attention, which is what it is for.
      3. There is NO residual bypass and NOTHING between ``up`` and the layer
         return. If a norm over the d_bottle axis, a second nonlinearity after
         ``up``, or a residual add were introduced, the split would either change
         the function or stop saving bytes, and it MUST be revisited.
    """

    def __init__(self, inner: nn.Module, hidden_size: int, bottleneck_dim: int):
        super().__init__()
        self.inner = inner
        self.bottleneck_dim = bottleneck_dim
        self.down = nn.Linear(hidden_size, bottleneck_dim, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, hidden_size, bias=False)
        # Llama-scale init so the projection behaves like a normal sublayer.
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.normal_(self.up.weight, std=0.02)

    def forward(self, *args, **kwargs):
        out = self.inner(*args, **kwargs)
        if isinstance(out, tuple):
            h = out[0]
            h = self.up(self.act(self.down(h)))
            return (h,) + tuple(out[1:])
        h = out
        h = self.up(self.act(self.down(h)))
        return h


def build_bottleneck_model(
    bottleneck_layer: int,
    bottleneck_dim: int,
    vocab_size: int = 128256,
    seq_len: int = 4096,
    dtype: torch.dtype = torch.bfloat16,
    size: str = "1b",
) -> LlamaForCausalLM:
    """Fresh (random-init) Llama of the given size, optionally with a bottleneck
    after layer j.

    size in {"1b", "3b", "7b"}; default "1b" preserves the original behaviour.
    bottleneck_dim <= 0  ->  no bottleneck (baseline arm).
    """
    cfg = make_config(size, vocab_size=vocab_size, seq_len=seq_len)
    model = LlamaForCausalLM(cfg)
    model = model.to(dtype)
    if bottleneck_dim and bottleneck_dim > 0:
        assert 0 <= bottleneck_layer < cfg.num_hidden_layers, bottleneck_layer
        inner = model.model.layers[bottleneck_layer]
        wrapped = BottleneckLayer(inner, cfg.hidden_size, bottleneck_dim).to(dtype)
        model.model.layers[bottleneck_layer] = wrapped
    return model


if __name__ == "__main__":
    for size in ("1b", "3b", "7b"):
        for bd in (0, 512):
            m = build_bottleneck_model(bottleneck_layer=6, bottleneck_dim=bd, size=size)
            n = sum(p.numel() for p in m.parameters())
            n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
            print(f"size={size} bottleneck_dim={bd}: total={n/1e9:.4f}B trainable={n_tr/1e9:.4f}B")
            del m
