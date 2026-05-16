"""Memory-Space v0 — in-place patching of a HuggingFace Llama model.

Design reference:
    ops/research_notes/20260426_memory_space_design_direction.md §2.1

We walk ``model.model.layers`` and swap each targeted ``LlamaDecoderLayer`` for
a ``MemorySpaceLayer`` that wraps the original.  This is purely additive — the
original decoder-layer module is retained as ``wrapped_layer`` on the wrapper,
so the patch is reversible (bookkeeping below).

Idempotency and re-patching are not supported in v0: we raise if a model has
already been patched, to avoid double-wrapping on mistake.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .config import MemorySpaceConfig
from .l2_compressor import L2Compressor
from .l3_summary import L3SummaryPool
from .layer import MemorySpaceLayer
from .memory_bank import MemoryBank


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def apply_mem_space_to_model(
    model: nn.Module,
    config: MemorySpaceConfig,
    layer_indices: Optional[Sequence[int]] = None,
) -> Tuple[nn.Module, List[MemorySpaceLayer]]:
    """Replace selected LlamaDecoderLayer modules with MemorySpaceLayer.

    Args:
        model: a HuggingFace Llama-family model (expects
            ``model.model.layers`` to be a ``nn.ModuleList`` of
            ``LlamaDecoderLayer`` — also works on bare ``LlamaModel``).
        config: shared ``MemorySpaceConfig`` applied to every patched layer.
        layer_indices: optional iterable of layer indices to patch.  ``None``
            → patch every layer.  Invalid indices raise ``IndexError``.

    Returns:
        (patched_model, mem_layers) where ``mem_layers`` is the list of newly
        inserted ``MemorySpaceLayer`` instances, in the order they appear in
        the decoder stack (useful for outer-loop aux-loss collection).

    Side effects:
        * Each replaced ``LlamaDecoderLayer`` becomes the ``wrapped_layer``
          attribute of its MemorySpaceLayer, so no parameters are lost.
        * ``model._mem_space_config`` and ``model._mem_space_layers`` are
          stashed for downstream code (training loop / aux-loss collector).

    Raises:
        RuntimeError: if the model has no ``.layers``, or has already been
            patched.
    """
    if not isinstance(config, MemorySpaceConfig):
        raise TypeError(f"config must be MemorySpaceConfig, got {type(config)}")

    root = getattr(model, "model", model)
    layers = getattr(root, "layers", None)
    if layers is None:
        raise RuntimeError(
            "apply_mem_space_to_model: could not locate `model.model.layers`; "
            "only Llama-family (or bare LlamaModel) architectures are supported."
        )
    if getattr(model, "_mem_space_layers", None) is not None:
        raise RuntimeError(
            "Model is already patched with MemorySpaceLayer. Double patching "
            "is not supported in v0."
        )

    n_layers = len(layers)
    if layer_indices is None:
        target_indices = list(range(n_layers))
    else:
        target_indices = list(layer_indices)
        for i in target_indices:
            if i < 0 or i >= n_layers:
                raise IndexError(
                    f"layer index {i} out of range for {n_layers} layers"
                )

    # Infer d_model.  Prefer config.hidden_size, fall back to layer.hidden_size.
    d_model = getattr(model.config, "hidden_size", None) if hasattr(model, "config") else None
    if d_model is None:
        d_model = getattr(layers[0], "hidden_size", None)
    if d_model is None:
        raise RuntimeError(
            "apply_mem_space_to_model: could not determine model hidden_size."
        )

    mem_layers: List[MemorySpaceLayer] = []

    # Branch-3 (2026-04-26): if config.shared_memory_bank is True, allocate ONE
    # MemoryBank up-front and pass it to every MemorySpaceLayer. This threads
    # intra-chunk BPTT through depth (layer i writes → layer i+1 reads), and
    # avoids state_dict duplication because MemorySpaceLayer uses
    # ``object.__setattr__`` to register the shared bank as a plain attribute.
    # Reference: ops/research_notes/20260426_mem_space_v0_branch3_writeback_bptt.md §3.
    slot_dim = d_model if config.slot_dim is None else config.slot_dim
    shared_bank: Optional[MemoryBank] = None
    if config.shared_memory_bank:
        shared_bank = MemoryBank(
            num_slots=config.num_slots,
            slot_dim=slot_dim,
            init_noise=config.slot_init_noise,
            slot_init=config.slot_init,
            slot_value_norm_cap=config.slot_value_norm_cap,
        )

    # L3 Summary Pool (2026-05-15): if enabled, create a single shared
    # L3SummaryPool and pass it to every MemorySpaceLayer. The pool computes
    # summary tokens from the top-layer H at the end of each chunk; the
    # summary is consumed by the NEXT chunk's joint attention at every layer.
    l3_pool: Optional[L3SummaryPool] = None
    if config.use_l3_summary:
        l3_pool = L3SummaryPool(
            d_model=d_model,
            num_summary=config.l3_n_summary,
            num_heads=config.l3_n_heads,
            n_layers=config.l3_n_layers,
        )

    # L2 Token Compressor (2026-05-16, Phase 11): if enabled, create a single
    # shared L2Compressor (peer to L3SummaryPool). The compressor produces
    # token-compressed KV latents at the END of each chunk via a post-forward
    # hook on the LAST patched layer; the next chunk's layers read these
    # latents to prepend pseudo-tokens to the joint-attn extended sequence.
    l2_compressor: Optional[L2Compressor] = None
    if config.use_l2:
        n_h = model.config.num_attention_heads
        # NOTE on n_kv_heads: in v0 we use L2 as PSEUDO-TOKENS (double-projection
        # through the wrapped layer's K/V projections), so kv_b's output
        # dimension must equal d_model (= n_h * d_head). For GQA models
        # (Llama-3-8B-Instruct has num_key_value_heads=8 ≠ num_attention_heads=32)
        # we force n_kv_heads = n_h here so K_recon/V_recon are [B, n_l2, d_model].
        # Stage-2 (direct K/V injection into the attention's KV cache) would
        # use the actual num_key_value_heads instead.
        n_kv = n_h
        d_head = d_model // n_h
        # Place on the same device as the model parameters.
        try:
            device_ref = next(model.parameters()).device
        except StopIteration:
            device_ref = torch.device("cpu")
        l2_compressor = L2Compressor(
            d_model=d_model,
            n_heads=n_h,
            n_kv_heads=n_kv,
            d_head=d_head,
            compress_ratio=config.l2_compress_ratio,
            d_c=config.l2_d_c,
            d_h_rope=config.l2_d_h_rope,
            init_scale=config.l2_init_scale,
        ).to(device_ref)

    for i in target_indices:
        orig = layers[i]
        wrapper = MemorySpaceLayer(
            orig, config, d_model=d_model, shared_bank=shared_bank,
            l3_pool=l3_pool, l2_compressor=l2_compressor,
        )
        layers[i] = wrapper
        mem_layers.append(wrapper)

    model._mem_space_config = config
    model._mem_space_layers = mem_layers
    # Expose the shared bank (or None) so the training loop's _reset_banks /
    # detach_ can touch a single bank instead of walking every layer.
    model._mem_space_shared_bank = shared_bank
    # Expose L3 pool as a registered submodule on the model so its parameters
    # appear in model.parameters() and state_dict. Also stash current L3
    # summary state for the chunked forward path.
    model._l3_pool = l3_pool
    if l3_pool is not None:
        # Register as a named module on root so it shows up in state_dict
        root.add_module("l3_pool", l3_pool)
        # Initialize L3 state: None (cold start on first chunk — layers will
        # get l3_summaries=None and skip the L3 prepend).
        l3_pool._current_summary = None
        model._l3_summary_for_next_chunk = None

        # Register a forward hook on the LAST patched layer to compute L3
        # from the current chunk's final hidden states and stash it on the
        # pool for the next chunk's layers to read.
        _last_mem_layer = mem_layers[-1]

        def _l3_post_forward_hook(module, args, output):
            """After the last MemorySpaceLayer, stash this chunk's final hidden
            states (DETACHED) so the NEXT chunk's layers can compute fresh L3
            summary tokens.

            We stash detached H, not pre-computed summary tokens, for two
            reasons:
            (1) chunk-local BPTT — backward through pool() with chunk-i's graph
                would re-enter graphs already freed by chunk-i's loss.backward(),
                causing ``RuntimeError: Trying to backward through the graph a
                second time``.
            (2) The pool's parameters need a fresh gradient path inside chunk
                i+1's own forward. By calling pool() inside chunk i+1 (with
                chunk-i's detached H), the pool weights see chunk i+1's loss.
            """
            # Extract hidden states from output
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            # h is [B, T, d] — the output of the last patched layer.
            pool = model._l3_pool
            if pool is not None:
                # Stash DETACHED H. Chunk i+1's MemorySpaceLayer.forward will
                # call pool(detached_H) to produce summary tokens with a fresh
                # gradient path inside chunk i+1's graph.
                pool._prev_chunk_h = h.detach()
                # Clear per-chunk cache: chunk i+1's first MemorySpaceLayer
                # forward will recompute, then cache for layers 1..31.
                if hasattr(pool, "_chunk_summary_cache"):
                    object.__setattr__(pool, "_chunk_summary_cache", None)

        _last_mem_layer.register_forward_hook(_l3_post_forward_hook)

    # Expose L2 compressor as a registered submodule on the model root so its
    # parameters appear in model.parameters() / state_dict (exactly once;
    # MemorySpaceLayer attaches it via object.__setattr__ to avoid duplication).
    model._l2_compressor = l2_compressor
    if l2_compressor is not None:
        root.add_module("l2_compressor", l2_compressor)

        # Register a post-forward hook on the LAST patched layer that
        # recomputes prev_latents from the current chunk's final hidden states
        # (DETACHED — same chunk-locality reasoning as the L3 hook).
        _last_mem_layer_l2 = mem_layers[-1]

        def _l2_post_forward_hook(module, args, output):
            """After the last MemorySpaceLayer, compress this chunk's final
            hidden states into latent KV tokens for the next chunk to read.

            We use detached H so backprop through `compress` only happens
            inside the *current* chunk (where this hook fires) — there is no
            cross-chunk BPTT through L2 (matches the design in
            docs/L2_IMPLEMENTATION_PLAN_20260516.md §4.2).
            """
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            comp = model._l2_compressor
            if comp is not None:
                with torch.no_grad():
                    comp.prev_latents = comp(h.detach())

        _last_mem_layer_l2.register_forward_hook(_l2_post_forward_hook)

    return model, mem_layers


def _reset_l2(model: nn.Module) -> None:
    """Zero the L2 compressor's cross-chunk state (prev_latents).

    Called by the training/eval loop at document boundaries (parallel to
    ``_reset_banks``). No-op if the model was patched without ``use_l2``.
    """
    comp = getattr(model, "_l2_compressor", None)
    if comp is not None:
        comp.reset()
