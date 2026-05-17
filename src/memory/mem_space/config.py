"""Memory-Space v0 — configuration dataclass.

Defaults follow §2.1 of
`ops/research_notes/20260426_memory_space_design_direction.md`:

    * N = 128 slots per layer, per sample
    * top_k = 16 slots selected at each step
    * writeback gate β initialised at 0 with linear warmup to 0.3
    * Load-balance aux loss weight 0.01 (Switch Transformer flavour)

All fields are plain Python scalars so the dataclass is trivially
serialisable (useful for logging / CLI overrides in later stages).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


_VALID_SLOT_INIT = {"zero", "random", "hidden_pool", "strided_token"}


@dataclass
class MemorySpaceConfig:
    """Hyperparameters for a per-layer memory space.

    See design doc §2.1 for the semantics of each field.  Anything that can be
    swept at Stage-1 warmup time lives here; genuinely layer-local choices
    (rope on/off, etc.) are also exposed so per-layer overrides can be built
    later without changing the layer code.

    Attributes:
        num_slots: N — number of slots per sample, per layer.
        top_k: k — number of slots prepended to the layer sequence each step.
        slot_dim: slot vector dimensionality.  `None` → use backbone hidden_size.
        selector_dim: projection dim for the Q_sel / K_sel scoring head.
        writeback_gate_init: initial value of the gate *logit* mapped through
            tanh — we default to 0.0 so β ≈ 0 at step 0 (Flamingo-style).
        writeback_gate_warmup_steps: #steps to linearly ramp β_max.
        writeback_gate_max: β_max — the gate is clamped to this ceiling.
        slot_init: {"zero", "random", "hidden_pool", "strided_token"} — see memory_bank.py.
            "strided_token" assigns evenly-spaced distinct tokens to each slot (Fix K, 2026-04-29).
        slot_init_noise: std of the N(0, σ) noise added on hidden_pool init.
        load_balance_weight: weight on the Switch-Transformer style aux loss.
        slot_dropout: fraction of slots randomly masked at train time.
        use_rope_for_slots: if False, memory tokens get position_id = 0 and
            inherit "position-less" RoPE.  The design doc leaves this off
            by default.
        return_aux_losses: expose load-balance / writeback-magnitude losses
            through the layer's `last_aux_losses` attribute.
        enable_writeback: kill-switch for debugging — selector + joint attn
            still run, slots stay frozen at their initial value.
    """

    num_slots: int = 128
    top_k: int = 16
    slot_dim: Optional[int] = None
    selector_dim: int = 128

    writeback_gate_init: float = 0.0
    writeback_gate_warmup_steps: int = 2000
    writeback_gate_max: float = 0.3

    slot_init: str = "hidden_pool"
    slot_init_noise: float = 1.0

    load_balance_weight: float = 0.01
    entropy_aux_weight: float = 0.0
    selector_temperature: float = 1.0
    key_repulsion_weight: float = 0.01
    key_repulsion_threshold: float = 0.3
    peak_routing_weight: float = 0.1
    slot_dropout: float = 0.0

    use_rope_for_slots: bool = False
    return_aux_losses: bool = True
    enable_writeback: bool = True

    # Stage-2a (2026-04-26): Tier-3 cure froze `hidden_to_slot` because it sits
    # on a detached / discarded pathway (O_mem_slot.detach() → memory_bank.write
    # → _reset_banks every chunk). Held-out PPL=2.1278 on Llama-3-8B confirmed
    # the cure generalizes (Branch 1). To test whether making the writeback
    # pathway gradient-bearing improves PPL further, set this to False to allow
    # `hidden_to_slot` to train. Default remains True for backward compat.
    hidden_to_slot_frozen: bool = True

    # Branch-3 (2026-04-26): writeback-BPTT. When True, a single MemoryBank is
    # shared across every patched decoder layer. Combined with gradient-bearing
    # writeback (O_mem_slot NOT detached, beta passed as tensor) this threads
    # intra-chunk BPTT through the full 32-layer depth: each layer's write
    # produces a new `slots` tensor that the NEXT layer reads → gate_param and
    # hidden_to_slot pick up end-to-end gradient from "writing a good slot
    # helps the next layer's LM loss". `_reset_banks` at chunk boundary +
    # init-time .detach() in MemoryBank preserve the inter-chunk break.
    # Set False to fall back to per-layer banks (no cross-layer BPTT) — useful
    # as an ablation; layer-local writeback grad still flows via the selector
    # path but cannot compound across depth.
    # Reference: ops/research_notes/20260426_mem_space_v0_branch3_writeback_bptt.md §3 (Option A.2).
    shared_memory_bank: bool = True

    # FIX X.1 (2026-04-30): slot value norm cap.
    # When > 0, slot values are clamped to this norm ceiling after each writeback.
    # Prevents slot norm explosion during the uniform-routing phase (when all slots
    # receive equal write weight and compound 32× per chunk). 0 = disabled (default).
    slot_value_norm_cap: float = 5.0  # Fix Y: default 5.0 (Arm C cap=10.0 showed stable; 5.0 tighter)

    # Stage-1 SWA (2026-04-27): sliding-window attention window size.
    # When > 0, each content token in the extended sequence attends locally
    # to only the last `swa_window` content tokens (plus ALL slot tokens which
    # always attend globally).  This forces the model to rely on memory slots
    # for information beyond the window, creating a gradient signal for
    # content-addressed retrieval training.
    # Default 0 = full causal attention within the content sequence
    # (backward-compatible with all pre-SWA checkpoints).
    # Recommended: swa_window = chunk_size // 8  (e.g. 512 for chunk_size=4096)
    # so the local window covers 12.5% of the chunk.
    # Reference: ops/research_notes/20260427_swa_memory_design.md §Stage1
    # H6 (2026-05-09, LM2-inspired): LSTM-style dual-gate writeback.
    # When True, replace the single global β with content-conditioned per-feature
    # input + forget gates:
    #     g_in, g_forget = sigmoid_split(W_m·M_prev + W_i·new_repr + bias)
    #     slots[idx] = g_in * tanh(new_repr) + g_forget * slots[idx]
    # forget_bias_init=1.0 makes g_forget≈sigmoid(1)≈0.73 at init (LSTM heuristic
    # "remember by default"). Per-feature gates are slot_dim-dim each → 2 × slot_dim
    # gate dims per slot (≈ 8M params for slot_dim=4096 with shared projection).
    # Reference: LM2 paper (arXiv:2502.06049), src/memory.py:259-263 + create_gates.
    use_dual_gate: bool = False
    forget_bias_init: float = 1.0
    input_bias_init: float = 0.0
    dual_gate_tanh_new: bool = True   # apply tanh to new content (LM2 default)

    # H6b (2026-05-09): hard top-k masking on cross-attn read.
    # When > 0, each query only attends to the top read_topk slots (others get -inf
    # before softmax). Forces the model to commit to specific slots rather than
    # smear soft attention across all 64-128. Useful as ablation arm for H6.
    # 0 = soft attention over all slots (default).
    read_topk: int = 0

    swa_window: int = 0

    # L3 Summary-Token module (2026-05-15): Q-Former-style cross-attn pool that
    # produces K dense summary tokens per chunk from the backbone's top-layer H.
    # These are prepended to the joint-attn extended sequence alongside L1 slots.
    # Reference: docs/L3_SUMMARY_RESEARCH.md §4-5.
    use_l3_summary: bool = False
    l3_n_summary: int = 64
    l3_n_layers: int = 2
    l3_n_heads: int = 8

    # Pure-L3 ablation (2026-05-15): If True, skip L1 slot prepending + dual-gate
    # writeback. Used for pure-L3 ablation where only L3 summary tokens are active.
    disable_l1_inject: bool = False

    # L2 token-compressed KV memory (2026-05-16, Phase 11): NSA / DeepSeek-V4-CSA
    # learned-gated attention pool over groups of g=16 tokens, producing
    # token-compressed KV latents that the next chunk reads. Cold-start near-zero
    # init (kv_b std=l2_init_scale) so initial L2 contribution ≈ 0.
    # Reference: docs/L2_DEEPSEEK_MLA_RESEARCH.md, docs/L2_IMPLEMENTATION_PLAN_20260516.md
    use_l2: bool = False
    l2_compress_ratio: int = 16  # g
    l2_d_c: int = 512
    l2_d_h_rope: int = 64
    l2_init_scale: float = 0.001

    # Gradient checkpointing on the wrapped LlamaDecoderLayer forward (Phase 11,
    # 2026-05-16). Trades ~2x compute for ~50% activation memory reduction —
    # required to fit L1+L2+L3 stack on H20 (97GB VRAM) at chunk_size=1024 +
    # 4k context = 4 chunks/sample BPTT. P8 (L1+L3 only) didn't need this; L2's
    # per-chunk kv_b activations push peak past 97GB.
    gradient_checkpointing: bool = False

    # v5 cold-start alpha gating (2026-05-17): on the first chunk of each
    # sample (cold start — memory bank not yet initialized), set the output-
    # side fusion coefficient alpha to 0 so that next_hidden = bypass_h
    # (no noisy uninitialised slot content injected into hidden states).
    # Selector / extended forward / writeback still execute normally, allowing
    # the first chunk's content to be written into the memory bank.
    zero_alpha_on_cold_start: bool = False

    def __post_init__(self) -> None:
        if self.num_slots <= 0:
            raise ValueError(f"num_slots must be > 0, got {self.num_slots}")
        if self.top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {self.top_k}")
        if self.top_k > self.num_slots:
            raise ValueError(
                f"top_k ({self.top_k}) must not exceed num_slots ({self.num_slots})"
            )
        if self.selector_dim <= 0:
            raise ValueError(f"selector_dim must be > 0, got {self.selector_dim}")
        if self.slot_init not in _VALID_SLOT_INIT:
            raise ValueError(
                f"slot_init must be one of {_VALID_SLOT_INIT}, got {self.slot_init!r}"
            )
        if not 0.0 <= self.slot_dropout < 1.0:
            raise ValueError(
                f"slot_dropout must be in [0, 1), got {self.slot_dropout}"
            )
        if self.writeback_gate_max < 0.0:
            raise ValueError(
                f"writeback_gate_max must be >= 0, got {self.writeback_gate_max}"
            )
        if self.writeback_gate_warmup_steps < 0:
            raise ValueError(
                "writeback_gate_warmup_steps must be >= 0, got "
                f"{self.writeback_gate_warmup_steps}"
            )
        if self.swa_window < 0:
            raise ValueError(
                f"swa_window must be >= 0 (0 = full causal), got {self.swa_window}"
            )
