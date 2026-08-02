"""Faithful Qwen3 / transformers-5.14 attention hijack for prefill-then-compress
KV-cache baselines (Paper A P1.6).

The *compression algorithm* is the vendored, unmodified GQA-aware
``SnapKVCluster`` / ``PyramidKVCluster`` from Zefan-Cai (see
``src/baselines/pyramidkv/pyramidkv_utils.py`` and ``PROVENANCE.md``). Only the
attention forward wrapper is re-implemented here, because the upstream
monkeypatches target transformers 4.37 + Llama/Mistral and cannot run on this
env (transformers 5.14 + Qwen3, rewritten attention API).

Design (exactly the upstream SnapKV/PyramidKV contract):

* **Full prefill is EXACT.** On the prefill call (``q_len > 1``) the attention
  output is computed over the FULL, uncompressed key/value tensors (same as
  stock Qwen3 SDPA) — compression NEVER perturbs the prefill logits. Only the
  KV that is *stored* for the decode phase is compressed: we run
  ``kv_cluster.update_kv(...)`` (observation-window scoring -> top-k over the
  past + keep the recent window; PyramidKV adds the per-layer pyramidal budget)
  and write the compressed K/V straight into the layer's ``DynamicLayer``. The
  clusters return kv-head-granular tensors (8 heads on Qwen3), matching the
  cache layout, so the write-back is direct.
* **Decode reads the compressed cache.** On the decode call (``q_len == 1``) the
  new token's K/V is appended to the compressed cache and attention runs over
  it. Because ``q_len == 1`` the causal mask is ``None`` (full visibility over
  the retained slots), so per-layer differing compressed lengths (PyramidKV)
  are fine.
* **True RoPE positions on decode.** The retained keys keep the RoPE that was
  applied at their TRUE prefill positions; the decode query must be RoPE'd at
  the TRUE next logical position, not at the (shorter) compressed cache length.
  ``generate_kvcompress`` below supplies explicit ``position_ids`` on every
  decode step — the faithful equivalent of upstream's
  ``prepare_inputs_for_generation`` override (which tracked ``kv_seq_len``
  separately from the compressed cache length).

When the prompt is shorter than the budget (``q_len < max_capacity_prompt``) the
clusters return the input unchanged, so the wrapper is a bit-for-bit no-op vs
stock Qwen3 SDPA — this is the ``--self_test`` faithfulness gate.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers.models.qwen3 import modeling_qwen3 as _q3  # noqa: E402
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402

# Vendored GQA-aware clusters + factories (Zefan-Cai, verbatim).
from src.baselines.pyramidkv.pyramidkv_utils import (  # noqa: E402
    SnapKVCluster,
    PyramidKVCluster,
)

_VALID_METHODS = ("snapkv", "pyramidkv")

# Saved stock forward so uninstall / stock comparison stays exact.
_ORIG_QWEN3_ATTENTION_FORWARD = _q3.Qwen3Attention.forward


# --------------------------------------------------------------------------- #
# Hijacked Qwen3Attention.forward (reproduces modeling_qwen3.py:252-291 verbatim
# except for the KV-compression insert around the cache update).
# --------------------------------------------------------------------------- #
def _qwen3_attention_forward_kvcompress(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values=None,
    **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # --- identical to stock Qwen3Attention.forward (q_norm/k_norm + RoPE) ---
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = _q3.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    q_len = query_states.shape[-2]

    # --- KV-compression insert (the ONLY deviation from stock) --------------
    # Prefill (q_len > 1): attention output is computed over the FULL key/value
    # (exact); the cache stores the COMPRESSED K/V for the decode phase.
    # Decode (q_len == 1): append to the compressed cache and attend over it.
    if past_key_values is not None:
        if q_len > 1:
            attn_key, attn_value = key_states, value_states  # EXACT full prefill
            kv_cluster = getattr(self, "kv_cluster", None)
            if kv_cluster is not None:
                k_comp, v_comp = kv_cluster.update_kv(
                    key_states, query_states, value_states,
                    attention_mask, self.num_key_value_groups,
                )
            else:
                k_comp, v_comp = key_states, value_states
            # Store compressed (layer starts empty on prefill -> cat == compressed).
            past_key_values.update(k_comp, v_comp, self.layer_idx)
            # Record the actual retained length for the equal-retained-token audit.
            self._kvc_retained_len = int(k_comp.shape[-2])
        else:
            attn_key, attn_value = past_key_values.update(
                key_states, value_states, self.layer_idx)
    else:
        attn_key, attn_value = key_states, value_states

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, _q3.eager_attention_forward
    )

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        attn_key,
        attn_value,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


# --------------------------------------------------------------------------- #
# install / uninstall
# --------------------------------------------------------------------------- #
def install_kv_compression(model, method: str, *, max_capacity_prompt: int = 6657,
                           window_size: int = 32, kernel_size: int = 5,
                           pooling: str = "avgpool", gqa_score_agg: str = "mean",
                           beta: int = 20):
    """Attach a vendored KV-compression cluster to every Qwen3 attention layer
    and monkeypatch ``Qwen3Attention.forward`` to the faithful wrapper.

    ``method`` in {"snapkv", "pyramidkv"}. ``max_capacity_prompt`` is the total
    retained tokens per layer INCLUDING the ``window_size`` recent tokens
    (SnapKV: uniform on all layers; PyramidKV: the per-layer average under the
    pyramidal schedule). Returns the resolved config dict.
    """
    method = method.lower()
    if method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}")
    if max_capacity_prompt - window_size <= 0:
        raise ValueError("max_capacity_prompt must exceed window_size")

    inner = model.model if hasattr(model, "model") else model
    num_hidden_layers = int(inner.config.num_hidden_layers)

    for layer in inner.layers:
        attn = layer.self_attn
        if method == "snapkv":
            attn.kv_cluster = SnapKVCluster(
                window_size=window_size,
                max_capacity_prompt=max_capacity_prompt,
                kernel_size=kernel_size,
                pooling=pooling,
                merge=None,
                gqa_score_agg=gqa_score_agg,
            )
        else:  # pyramidkv
            attn.kv_cluster = PyramidKVCluster(
                num_hidden_layers=num_hidden_layers,
                layer_idx=attn.layer_idx,
                window_size=window_size,
                max_capacity_prompt=max_capacity_prompt,
                kernel_size=kernel_size,
                pooling=pooling,
                beta=beta,
                merge=None,
                gqa_score_agg=gqa_score_agg,
            )
        attn._kvc_retained_len = None

    _q3.Qwen3Attention.forward = _qwen3_attention_forward_kvcompress

    return {
        "method": method,
        "max_capacity_prompt": max_capacity_prompt,
        "window_size": window_size,
        "kernel_size": kernel_size,
        "pooling": pooling,
        "gqa_score_agg": gqa_score_agg,
        "beta": beta,
        "num_hidden_layers": num_hidden_layers,
    }


def uninstall_kv_compression(model):
    """Restore stock ``Qwen3Attention.forward`` and drop attached clusters."""
    _q3.Qwen3Attention.forward = _ORIG_QWEN3_ATTENTION_FORWARD
    inner = model.model if hasattr(model, "model") else model
    for layer in inner.layers:
        attn = layer.self_attn
        if hasattr(attn, "kv_cluster"):
            del attn.kv_cluster
        if hasattr(attn, "_kvc_retained_len"):
            del attn._kvc_retained_len


def retained_kv_stats(model, dtype_bytes: Optional[int] = None) -> dict:
    """Per-layer retained KV length + total compressed KV bytes after a prefill.

    Reads each layer's compressed ``DynamicLayer`` (keys/values). Returns min /
    max / mean retained length across layers and the summed K+V byte count — the
    audit that backs the equal-retained-token claim vs CoMem's read budget.
    """
    inner = model.model if hasattr(model, "model") else model
    lens = []
    total_bytes = 0
    per_layer = []
    for li, layer in enumerate(inner.layers):
        attn = layer.self_attn
        rl = getattr(attn, "_kvc_retained_len", None)
        if rl is not None:
            lens.append(int(rl))
        per_layer.append(rl)
    return {
        "per_layer_retained_len": per_layer,
        "min_retained_len": min(lens) if lens else None,
        "max_retained_len": max(lens) if lens else None,
        "mean_retained_len": (sum(lens) / len(lens)) if lens else None,
        "n_layers_recorded": len(lens),
    }


def compressed_kv_bytes(cache) -> tuple[int, list]:
    """Sum K+V bytes actually held in a (compressed) ``DynamicCache`` plus a
    per-layer retained-length list. This is the on-device storage footprint that
    the decode phase pays for (contrast with CoMem's persistent bounded store)."""
    total = 0
    per_layer = []
    for layer in cache.layers:
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or (hasattr(keys, "numel") and keys.numel() == 0):
            per_layer.append(0)
            continue
        total += keys.numel() * keys.element_size()
        total += values.numel() * values.element_size()
        per_layer.append(int(keys.shape[-2]))
    return total, per_layer


# --------------------------------------------------------------------------- #
# Explicit greedy generation over the compressed cache, with instrumentation.
#
# We do NOT use model.generate(): its prepare_inputs_for_generation derives the
# decode position from the CACHE length, which is WRONG once the cache is
# compressed (the retained keys keep their TRUE prefill RoPE, and the new query
# must be RoPE'd at the true next logical position). Instead we run an explicit
# loop passing explicit position_ids — the faithful equivalent of the upstream
# prepare_inputs_for_generation override that tracked kv_seq_len separately.
# --------------------------------------------------------------------------- #
def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def generate_kvcompress(model, input_ids, *, max_new_tokens: int,
                        eos_token_ids=None, extra_end_token_ids=None,
                        stats: Optional[dict] = None):
    """Greedy decode with the KV-compression hijack installed.

    * Full-prefill attention is EXACT (over uncompressed K/V); only the STORED
      cache is compressed. ``stats`` (if given) is filled with the measured
      full-prefill latency, decode latency/tok, peak GPU memory, compressed
      retained KV bytes + per-layer retained lengths, prompt token count, and
      ``full_prompt_seen=True`` (this family MUST full-prefill the whole prompt).

    Returns ``generated_ids`` (1-D LongTensor of NEW tokens only).
    """
    import time
    from transformers.cache_utils import DynamicCache

    device = input_ids.device
    prompt_len = int(input_ids.shape[1])
    eos = set(int(e) for e in (eos_token_ids or []))
    eos |= set(int(e) for e in (extra_end_token_ids or []))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    cache = DynamicCache(config=model.config)

    # --- full prefill (exact attention; cache stored compressed by hijack) ---
    _sync(); t0 = time.perf_counter()
    out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    _sync(); prefill_s = time.perf_counter() - t0
    prefill_peak_gb = (torch.cuda.max_memory_allocated() / (1024 ** 3)
                       if torch.cuda.is_available() else None)

    logits = out.logits[:, -1, :]
    next_id = int(torch.argmax(logits, dim=-1).item())
    generated = [next_id]

    # --- greedy decode over the compressed cache w/ TRUE logical positions ---
    _sync(); td0 = time.perf_counter()
    n_decoded = 0
    if next_id not in eos:
        for step in range(1, max_new_tokens):
            true_pos = prompt_len + step - 1  # position of the token being fed
            tok = torch.tensor([[next_id]], dtype=torch.long, device=device)
            pos = torch.tensor([[true_pos]], dtype=torch.long, device=device)
            out = model(input_ids=tok, past_key_values=cache, use_cache=True,
                        position_ids=pos)
            n_decoded += 1
            next_id = int(torch.argmax(out.logits[:, -1, :], dim=-1).item())
            generated.append(next_id)
            if next_id in eos:
                break
    _sync(); decode_s = time.perf_counter() - td0
    n_decoded += 1  # the first token from prefill logits counts as one decode

    if stats is not None:
        kv_bytes, per_layer_len = compressed_kv_bytes(cache)
        stats.update({
            "prompt_tokens": prompt_len,
            "full_prompt_seen": True,
            "prefill_latency_s": round(prefill_s, 4),
            "prefill_peak_gb": (round(prefill_peak_gb, 3)
                                if prefill_peak_gb is not None else None),
            "decode_latency_s": round(decode_s, 4),
            "decode_tokens": n_decoded,
            "decode_latency_per_tok_ms": (round(decode_s / n_decoded * 1000, 3)
                                          if n_decoded else None),
            "compressed_kv_bytes": kv_bytes,
            "compressed_kv_MB": round(kv_bytes / (1024 ** 2), 3),
            "per_layer_retained_len": per_layer_len,
            "min_retained_len": min(per_layer_len) if per_layer_len else None,
            "max_retained_len": max(per_layer_len) if per_layer_len else None,
            "mean_retained_len": (round(sum(per_layer_len) / len(per_layer_len), 1)
                                  if per_layer_len else None),
        })

    return torch.tensor(generated, dtype=torch.long, device=device)
