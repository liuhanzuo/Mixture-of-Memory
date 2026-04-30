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

import torch.nn as nn

from .config import MemorySpaceConfig
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

    for i in target_indices:
        orig = layers[i]
        wrapper = MemorySpaceLayer(
            orig, config, d_model=d_model, shared_bank=shared_bank,
        )
        layers[i] = wrapper
        mem_layers.append(wrapper)

    model._mem_space_config = config
    model._mem_space_layers = mem_layers
    # Expose the shared bank (or None) so the training loop's _reset_banks /
    # detach_ can touch a single bank instead of walking every layer.
    model._mem_space_shared_bank = shared_bank
    return model, mem_layers
