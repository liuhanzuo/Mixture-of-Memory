"""Per-layer Memory-Space architecture (v0 prototype).

Implements the "memory space" design spec'd out in
`ops/research_notes/20260426_memory_space_design_direction.md`:

    Every selected LlamaDecoderLayer maintains its own bank of N slot vectors.
    At each forward the layer:
        1. Projects a pooled summary of the current hidden states and scores
           the slots via softmax(Q_sel · K_sel^T / sqrt(selector_dim)).
        2. Straight-through top-k selects `k` slots (hard indices, soft grads).
        3. Prepends those slots as "memory tokens" to the decoder layer's
           sequence and runs a single joint softmax through the underlying
           LlamaDecoderLayer.  Current-token outputs feed the next layer;
           memory-position outputs are used to update the corresponding slots.
        4. In-place EMA writeback on the selected slot positions, gated by a
           learnable β with linear warmup from 0 → writeback_gate_max.

Public API
----------
    MemorySpaceConfig         — dataclass with all hyperparameters.
    MemoryBank                — per-layer, per-sample slot state.
    MemorySpaceLayer          — wraps a single HF LlamaDecoderLayer.
    apply_mem_space_to_model  — walks `model.model.layers` and swaps them in.

This is a V0 research prototype — correctness and clean hook points first,
performance later.
"""
from .config import MemorySpaceConfig
from .memory_bank import MemoryBank
from .selector import TopKSelector
from .layer import MemorySpaceLayer
from .patch import apply_mem_space_to_model

__all__ = [
    "MemorySpaceConfig",
    "MemoryBank",
    "TopKSelector",
    "MemorySpaceLayer",
    "apply_mem_space_to_model",
]
