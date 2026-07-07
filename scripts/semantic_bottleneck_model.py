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


def make_1b_config(vocab_size: int = 128256, seq_len: int = 4096) -> LlamaConfig:
    """Llama-3.2-1B-shaped config (random init, trained from scratch)."""
    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=64,
        hidden_act="silu",
        max_position_embeddings=max(seq_len, 4096),
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
        attention_bias=False,
        attention_dropout=0.0,
    )


class BottleneckLayer(nn.Module):
    """Wraps one LlamaDecoderLayer; funnels its output through d->d_bottle->d.

    transformers>=5 LlamaDecoderLayer.forward returns a plain tensor. We keep a
    tuple-safe path just in case.
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
) -> LlamaForCausalLM:
    """Fresh (random-init) 1B Llama, optionally with a bottleneck after layer j.

    bottleneck_dim <= 0  ->  no bottleneck (baseline arm).
    """
    cfg = make_1b_config(vocab_size=vocab_size, seq_len=seq_len)
    model = LlamaForCausalLM(cfg)
    model = model.to(dtype)
    if bottleneck_dim and bottleneck_dim > 0:
        assert 0 <= bottleneck_layer < cfg.num_hidden_layers, bottleneck_layer
        inner = model.model.layers[bottleneck_layer]
        wrapped = BottleneckLayer(inner, cfg.hidden_size, bottleneck_dim).to(dtype)
        model.model.layers[bottleneck_layer] = wrapped
    return model


if __name__ == "__main__":
    for bd in (0, 512):
        m = build_bottleneck_model(bottleneck_layer=6, bottleneck_dim=bd)
        n = sum(p.numel() for p in m.parameters())
        n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"bottleneck_dim={bd}: total={n/1e9:.4f}B trainable={n_tr/1e9:.4f}B")
