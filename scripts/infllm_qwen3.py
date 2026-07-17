#!/usr/bin/env python
"""InfLLM (thunlp/InfLLM, arXiv:2402.04617) baseline adapter for Qwen3-8B.

InfLLM is the training-free, retrieval-memory long-context peer of CoMem/QCMem:
distant context is chunked into fixed-size *memory units*, a few relevant units
are looked up per decode block, and attention runs over
``[attention-sinks; retrieved units; local sliding window]``. This module wraps
Qwen3 so its RULER / BABILong outputs can be scored by the SAME scorers as the
QCMem drivers (``scripts/eval_ruler_qcmem.py`` / ``scripts/eval_qcmem_babilong.py``).

WHY A CUSTOM PATCH (feasibility note)
-------------------------------------
InfLLM ships ``inf_llm/utils/patch.py::patch_hf`` written against transformers
~4.37: it re-implements ``LlamaModel/Qwen2Model.forward`` mirroring that era's
internals (per-attention ``self.rotary_emb``, ``self.num_heads`` attributes,
``decoder_layer(..., position_ids=rope_module)`` calls) and only dispatches
``Qwen2ForCausalLM``. Qwen3 needs transformers >= 4.51 (both nodes run 5.5.4 /
5.13.1), where the model/attention/cache APIs changed and Qwen3 adds per-head
q/k RMSNorm. There is NO single transformers version where InfLLM's stock patch
AND Qwen3 both work, so we re-target InfLLM's design to transformers-5.x Qwen3
here:
  * REUSE verbatim: ``inf_llm.attention.inf_llm_forward`` (the memory attention
    closure + ``ContextManager``) and ``RotaryEmbeddingESM`` (InfLLM applies RoPE
    inside the ContextManager, so the attention hands it RAW pre-RoPE q/k).
  * NEW (Qwen3-specific): apply Qwen3 ``q_norm``/``k_norm`` on the head_dim by
    composing them into the ``project_q``/``project_k`` callables passed to the
    reused closure (q_norm acts on the last dim; reshape->norm->flatten is
    preserved when the closure re-views to (b,len,heads,head_dim)).
  * NEW: minimal transformers-5.x ``Qwen3Model.forward`` /
    ``Qwen3DecoderLayer.forward`` that thread the per-layer ContextManager via
    ``past_key_values`` (a plain tuple) and skip the stock causal-mask / Cache
    machinery entirely (InfLLM never materialises the O(L^2) mask).

The stock ``Qwen3ForCausalLM.forward`` is left untouched — it calls
``self.model(...)`` and returns ``outputs.past_key_values``, which our patched
model_forward populates with the tuple of ContextManagers, so InfLLM's own
``GreedySearch`` (which feeds ``past_key_values`` back per chunk) works unchanged.

Requires CUDA (ContextManager uses ``torch.cuda.stream``).
"""
from __future__ import annotations

import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_INFLLM_DIR = os.path.join(PROJECT_ROOT, "external", "InfLLM")
for _p in (PROJECT_ROOT, _INFLLM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.modeling_outputs import BaseModelOutputWithPast  # noqa: E402

from inf_llm.attention import RotaryEmbeddingESM  # noqa: E402
from inf_llm.attention import ATTN_FORWRAD  # noqa: E402
from inf_llm.utils import GreedySearch  # noqa: E402


# --------------------------------------------------------------------------- #
# Canonical InfLLM memory config for an 8B GQA model.
# Copied from InfLLM's own config/llama-3-inf-llm.yaml (the closest published
# 8B recipe), with rope base taken from the model's own config (Qwen3 = 1e6).
# --------------------------------------------------------------------------- #
DEFAULT_MEM_CONFIG = {
    "type": "inf-llm",
    "block_size": 128,        # tokens per memory unit
    "n_init": 128,            # attention-sink initial tokens
    "n_local": 4096,          # local sliding-window size
    "topk": 16,               # memory units retrieved per exec block
    "repr_topk": 4,           # representative tokens per unit
    "max_cached_block": 32,   # max memory units kept on GPU
    "exc_block_size": 512,    # tokens per execution block
    "fattn": False,           # triton multi-stage flash-attn (kept off: simplest)
    "cache_strategy": "lru",
    "async_global_stream": False,  # GLOBAL_STREAM stays None -> stream(None) no-op
    "pin_memory": False,
    "faiss": False,
    "perhead": False,
    "distance_scale": 1.0,
    # prefill chunk size (execution granularity for the prompt), not a mem-attn kwarg
    "chunk_size": 8192,
}


def _make_qwen3_hf_forward(inner_forward, position_bias):
    """Wrap InfLLM's reused ``inf_llm_forward`` closure for a transformers-5.x
    Qwen3Attention module, inserting Qwen3 q_norm/k_norm."""

    def hf_forward(self, hidden_states, position_embeddings=None,
                   attention_mask=None, past_key_values=None,
                   position_ids=None, use_cache=True, **kwargs):
        head_dim = self.head_dim

        def project_q(x):
            b, l, _ = x.shape
            h = self.q_proj(x).view(b, l, -1, head_dim)
            h = self.q_norm(h)                      # Qwen3: RMSNorm on head_dim
            return h.reshape(b, l, -1)

        def project_k(x):
            b, l, _ = x.shape
            h = self.k_proj(x).view(b, l, -1, head_dim)
            h = self.k_norm(h)                      # Qwen3: RMSNorm on head_dim
            return h.reshape(b, l, -1)

        num_heads = self.config.num_attention_heads
        num_kv = self.config.num_key_value_heads
        # InfLLM applies RoPE inside the ContextManager -> pass RAW q/k (post
        # qk-norm, pre-RoPE), exactly mirroring InfLLM's stock huggingface_forward.
        o, pkv = inner_forward(
            self, hidden_states, hidden_states, position_bias,
            True, past_key_values,
            project_q, project_k, self.v_proj, self.o_proj,
            head_dim, num_heads, num_kv,
        )
        return o, pkv

    return hf_forward


def _qwen3_decoder_forward(self, hidden_states, attention_mask=None,
                           position_ids=None, past_key_values=None,
                           use_cache=True, position_embeddings=None, **kwargs):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, pkv = self.self_attn(
        hidden_states, past_key_values=past_key_values, use_cache=True)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states, pkv


def _qwen3_model_forward(self, input_ids=None, attention_mask=None,
                         position_ids=None, past_key_values=None,
                         inputs_embeds=None, use_cache=None, **kwargs):
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    hidden_states = inputs_embeds

    new_pkv = []
    for i, layer in enumerate(self.layers):
        past = past_key_values[i] if past_key_values is not None else None
        hidden_states, cache = layer(hidden_states, past_key_values=past,
                                     use_cache=True)
        new_pkv.append(cache)

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=tuple(new_pkv),
    )


def patch_qwen3_inf_llm(model, mem_config: dict, rope_base: float):
    """In-place patch a transformers-5.x Qwen3ForCausalLM to InfLLM memory
    attention. Returns the same model."""
    inner_model = model.model
    attn_type = mem_config.get("type", "inf-llm")

    # Attention-kwargs consumed by inf_llm_forward (drop loader-only keys).
    attn_kwargs = {
        k: v for k, v in mem_config.items()
        if k not in ("type", "chunk_size", "path", "base", "distance_scale")
    }
    inner_forward = ATTN_FORWRAD[attn_type](**attn_kwargs)

    head_dim = int(getattr(model.config, "head_dim",
                           model.config.hidden_size // model.config.num_attention_heads))
    position_bias = RotaryEmbeddingESM(
        head_dim, rope_base, mem_config.get("distance_scale", 1.0)
    )
    inner_model.position_bias = position_bias

    hf_forward = _make_qwen3_hf_forward(inner_forward, position_bias)

    Attention = inner_model.layers[0].self_attn.__class__
    DecoderLayer = inner_model.layers[0].__class__
    Model = inner_model.__class__

    for layer in inner_model.layers:
        attn = layer.self_attn
        attn._old_forward = attn.forward
        attn.forward = hf_forward.__get__(attn, Attention)
        layer._old_forward = layer.forward
        layer.forward = _qwen3_decoder_forward.__get__(layer, DecoderLayer)

    inner_model._old_forward = inner_model.forward
    inner_model.forward = _qwen3_model_forward.__get__(inner_model, Model)
    return model


def load_infllm_qwen3(model_path, device="cuda:0", dtype=torch.bfloat16,
                      mem_config: dict | None = None):
    """Load Qwen3-8B, apply InfLLM patching, return (model, tokenizer, searcher,
    resolved_mem_config)."""
    cfg = dict(DEFAULT_MEM_CONFIG)
    if mem_config:
        cfg.update(mem_config)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, attn_implementation="eager",
        trust_remote_code=True, local_files_only=True,
    ).to(device).eval()

    rope_base = float(getattr(model.config, "rope_theta", 1000000.0))
    cfg.setdefault("base", rope_base)
    patch_qwen3_inf_llm(model, cfg, rope_base=cfg["base"])

    searcher = GreedySearch(model, tokenizer)
    return model, tokenizer, searcher, cfg


@torch.inference_mode()
def infllm_generate(searcher, input_ids, max_new_tokens, chunk_size,
                    extra_end_token_ids=None):
    """Run InfLLM greedy decode on a single pre-tokenised prompt (batch 1).

    ``searcher.clear()`` is called first so every sample starts from an empty
    memory (fresh per-layer ContextManagers are created lazily inside the patched
    model_forward when ``past_key_values is None``)."""
    searcher.clear()
    if input_ids.dim() == 2:
        input_ids = input_ids[0]
    out = searcher.generate(
        input_ids=input_ids,
        max_length=max_new_tokens,
        chunk_size=chunk_size,
        extra_end_token_ids=extra_end_token_ids or [],
    )
    searcher.clear()
    text = out[0] if isinstance(out, list) else out
    return text.strip()
