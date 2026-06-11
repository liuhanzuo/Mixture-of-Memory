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
_VALID_WRITEBACK_MODE = {"dual_gate", "lowrank_gate", "diag_gate", "scalar_beta"}


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

    # v6 (2026-05-18): Direct-replacement writeback — skip EMA, write s_new
    # directly into the selected slots.  Equivalent to β=1 but does not require
    # the gate warmup ramp.  Default False = current EMA behaviour.
    use_replace_writeback: bool = False

    # v7 (2026-05-18): Always-on "global" slots that receive a replacement write
    # (slot ← s_new) on every forward call regardless of top-k routing score.
    # These are the LAST num_global_slots indices in the bank (e.g. slots[504..511]
    # for num_global_slots=8 with num_slots=512). When 0 (default) this feature
    # is disabled. Intended to provide an EMA-free accumulation register for
    # tasks that require exact running state across many chunks (e.g. counting).
    num_global_slots: int = 0

    slot_init: str = "hidden_pool"
    slot_init_noise: float = 1.0

    load_balance_weight: float = 0.01
    # P7 / loss-free balancing (2026-06-05): online per-slot routing-logit bias
    # that equalises slot usage WITHOUT producing an interfering aux-loss
    # gradient. Borrowed from "Auxiliary-Loss-Free Load Balancing for MoE"
    # (DeepSeek, arXiv:2408.15664). Unlike the Switch-Transformer
    # `load_balance_weight` aux (which pushes routing toward uniform and can
    # *worsen* the slot-collapse failure mode by injecting a task-interfering
    # gradient), this maintains a per-slot bias that is added ONLY to the
    # selection logits (top-k index choice). The returned `scores`/`ste_weights`
    # — i.e. the gradient path — stay based on the *unbiased* logits, so the
    # bias never perturbs the LM/task gradient. The bias is updated online in a
    # no_grad block from the observed per-slot load (under-used → bias raised).
    #   use_loss_free_balance: master switch (default False = back-compat; when
    #     False the selector is byte-identical to the pre-P7 behaviour).
    #   loss_free_update_rate: sign-update step size for the bias each batch.
    # NOTE: when use_loss_free_balance=True you should set load_balance_weight=0
    # to neutralise the uniform-pushing Switch aux — the two mechanisms target
    # the same goal and should NOT both be non-zero simultaneously.
    use_loss_free_balance: bool = False
    loss_free_update_rate: float = 0.001
    # P10 (2026-06-06): Straight-Through Gumbel top-k selection. When enabled
    # (and only in training mode), i.i.d. Gumbel(0,1) noise is added to the
    # SELECTION logits before torch.topk so the set of slots that win the top-k
    # is stochastic — improving exploration and reducing key over-smearing.
    # Mirrors the loss-free-balance convention: the noise touches ONLY which
    # slots get selected (the gradient-free index path), NEVER the returned
    # `scores`/`ste_weights` (which stay computed from the original noise-free
    # logits), so the LM/task gradient is unperturbed. At eval (not self.training)
    # no noise is added regardless of the flag. When False (default) the selector
    # is byte-identical to pre-P10. See status/MEMORY_PROTOCOL_PLAN.md [P10].
    #   use_st_gumbel_topk: master switch (default False = back-compat).
    #   st_gumbel_temperature: scales the Gumbel noise (g * temperature) before
    #     it is added to the selection logits. 1.0 = standard ST-Gumbel.
    use_st_gumbel_topk: bool = False
    st_gumbel_temperature: float = 1.0
    # P11 (2026-06-06): delta-rule writeback + normalized readout. Two
    # independent sub-features, BOTH default off → byte-identical to pre-P11.
    #
    #   use_delta_rule_writeback: when True, the GATED writeback paths
    #     (dual_gate / lowrank_gate / diag_gate — i.e. any write() call that
    #     supplies a forget_gate) switch from the LM2 two-independent-gate form
    #       slot_new = g_in · new_content + g_forget · slot_old
    #     to the delta-rule (residual) form using ONLY the input gate:
    #       slot_new = slot_old + g_in · (new_content − slot_old)
    #     i.e. the gate scales the RESIDUAL between the new value and what is
    #     already stored, tying forget = (1 − g_in). The independent forget gate
    #     is ignored on this path. The legacy single-gate EMA path is ALREADY
    #     this residual form, so it is unchanged. Applies in BOTH train and eval
    #     (it changes stored state, so must be consistent). Default False.
    #
    #   normalize_readout: when True, the retrieved/readout memory vector
    #     `M_sel_hidden` is L2-normalized per token and rescaled to the local
    #     hidden-state reference norm (× readout_norm_scale) BEFORE it is gated /
    #     injected, so the gate sees a memory signal whose magnitude is
    #     comparable to the local-attention output. When False (default) the
    #     existing SHRINK-ONLY clamp (only attenuates M_sel_hidden when it
    #     exceeds the hidden-state scale, never amplifies) is preserved
    #     byte-identically. Applies in BOTH train and eval (forward-path readout
    #     scaling must be consistent). Default False.
    #
    #   readout_norm_scale: multiplier on the hidden-state reference norm used
    #     as the target magnitude when normalize_readout is True. 1.0 = match the
    #     local-attention scale exactly. Ignored when normalize_readout is False.
    use_delta_rule_writeback: bool = False
    normalize_readout: bool = False
    readout_norm_scale: float = 1.0
    entropy_aux_weight: float = 0.0
    selector_temperature: float = 1.0
    key_repulsion_weight: float = 0.01
    key_repulsion_threshold: float = 0.3
    # v9 (2026-06-01): L3 summary-token diversity regularizer. Penalises the
    # L3 Q-Former output tokens for collapsing to a single direction (which
    # would degenerate multi_query routing back to single_query). Applied only
    # on layer_idx==0 (l3_pool is a shared singleton across all 32 layers).
    l3_diversity_weight: float = 0.1
    l3_diversity_threshold: float = 0.5
    peak_routing_weight: float = 0.1
    # P1 / v12 (2026-06-01): summary-reconstruction auxiliary loss weight.
    # When > 0, a small MemoryReconDecoder reconstructs the current chunk's L3
    # summary tokens from the slot VALUES written this chunk; the loss
    # MSE(S_hat, stopgrad(S_L3)) gives the write path a near-distance objective
    # ("store content that is decodable"). 0 = disabled (default; back-compat).
    # Requires use_l3_summary=True (L3 summary tokens are the recon target).
    # The decoder is a shared singleton across all 32 layers; the loss is only
    # computed on layer_idx==0. See versions/v12_summary_reconstruction.md and
    # status/MEMORY_PROTOCOL_PLAN.md [P1]. Motivated by the toy passcode
    # diagnostic (commit e5bb181): addressing worked but exact_acc stayed 0.
    l_recon_weight: float = 0.0
    # ICAE-style token-level reconstruction aux loss (2026-06-07).
    # When > 0, an L3TokenReconHead reconstructs the CURRENT chunk's DISCRETE
    # input token ids from a fresh (grad-bearing) L3 summary of that chunk:
    #   hidden_C --l3_pool--> summary_C [B,K,d] --head--> dec_hidden [B,T,d]
    #   --frozen lm_head--> logits [B,T,V];  loss = CE(logits, chunk_input_ids).
    # Unlike l_recon_weight (which MSE-reconstructs the CONTINUOUS L3 hidden and
    # detaches the target → trivial collapse, no semantic pressure), this loss
    # (a) reconstructs DISCRETE tokens via cross-entropy, (b) does NOT detach,
    # so gradient flows back INTO the L3 pool, forcing it to learn a genuine
    # semantic compression that is decodable to text (ICAE, arXiv:2307.06945).
    # The K-token summary is a real bottleneck (chunk_size/l3_n_summary× e.g.
    # 512/64 = 8×). Requires use_l3_summary=True. 0 = disabled (back-compat).
    # The decode head is computed in the TRAINING STEP (not in layer.forward),
    # because decoder layers never receive the discrete token ids.
    l3_recon_token_weight: float = 0.0
    # Max sequence length the L3TokenReconHead's positional query bank supports.
    # Set = chunk_size by build_model. Must be >= the longest chunk decoded.
    l3_recon_max_positions: int = 1024
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

    # Writeback-gate mode selector (2026-06-04): cost-controlled alternatives to
    # the LM2-style dual_gate, motivated by the large-slot_dim experiment
    # (slot_dim=16384 makes dual_gate's two Linear(slot_dim, 2*slot_dim) blow up
    # to ~34B params/layer → OOM). All modes produce per-feature input/forget
    # gates g_in/g_forget and reuse memory_bank.write's dual-gate path; only the
    # logit computation differs.
    #   "dual_gate"   — current full-connected LM2 gate (4*slot_dim^2/layer).
    #                   Selected automatically when use_dual_gate=True for back-compat.
    #   "lowrank_gate"— two-stage low-rank projection: compress (s_new, M_prev)
    #                   to rank r then expand to 2*slot_dim. ~4*slot_dim*r/layer.
    #   "diag_gate"   — per-feature diagonal params (no full matrix). ~6*slot_dim/layer.
    #   "scalar_beta" — no gate projection; fall back to the legacy single-scalar
    #                   EMA writeback (gate_param β). Cheapest baseline.
    writeback_mode: str = "dual_gate"
    lowrank_gate_rank: int = 256

    # v8-A (2026-05-18): Per-group forget bias override for global slots.
    # When != forget_bias_init, the global slots' forget gate logit is shifted by
    # (global_slot_forget_bias - forget_bias_init) at inference time, effectively
    # giving global slots a different initial forget tendency.
    # Default = forget_bias_init (no override, same as regular slots).
    global_slot_forget_bias: float = 1.0  # matches forget_bias_init default

    # v8-C (2026-05-18): Input-gate-only writeback for global slots.
    # slot_global ← g_in · tanh(s_new)  (no forget term, pure write register)
    # Requires use_dual_gate=True (reuses gate_proj_new/gate_proj_mem projections).
    global_slot_input_gate_only: bool = False

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

    # P1-v2 (2026-05-31): Break gate freeze deadlock by removing protective
    # guards that prevent gradient flow to K_sel and inject_gate.
    # no_detach_slots_in_selector: if True, do NOT detach slots before K_sel
    #   → K_sel gets gradient from routing path (risk: slot norm instability)
    # no_slot_delta_clip: if True, skip the norm-clipping of slot_delta
    #   → inject_gate gets stronger gradient signal
    # inject_gate_bias_init: override the inject_gate bias initial value
    #   Default -0.1523 (g≈0.462). Set -2.0 for g≈0.12 (gate must learn to open).
    no_detach_slots_in_selector: bool = False
    no_slot_delta_clip: bool = False
    inject_gate_bias_init: float = -0.1523

    # P1-v3 routing mode (2026-05-31): controls how per-token logits are
    # aggregated into per-slot scores.
    # "max_pool" (default): max over T tokens per slot — structurally collapses
    #   to uniform when T >> N (root cause of routing failure).
    # "chunk_query": mean-pool hidden states first → single query per chunk →
    #   one logit per slot. Eliminates the max-pool uniformity problem.
    # "multi_query" (v8, 2026-06-01): use the M L3 summary tokens as M
    #   independent sub-queries; per-slot relevance aggregated across queries
    #   via logsumexp, then global top-k. Preserves intra-chunk semantic
    #   heterogeneity (avoids single-query collapse). Falls back to max_pool
    #   when L3 summaries are unavailable (cold start / L3 disabled).
    # "slot_query" (v11, 2026-06-01): slot-as-query cross-attention routing.
    #   Inverts query/key roles — each slot attends over the chunk's T tokens
    #   and computes its own relevance via softmax-weighted similarity (soft-max
    #   pooling). Sidesteps the single-query collapse entirely because the N
    #   routing queries ARE the slots (diversity guaranteed by slot content +
    #   key_repulsion_loss), not a pooled chunk query. Does not use L3.
    routing_pool_mode: str = "max_pool"
    # v8 multi-query routing (2026-06-01): logsumexp temperature over the M
    # sub-query (L3 summary) dimension. tau→0 ≈ max (one strong query wins),
    # tau→∞ ≈ mean (all queries must agree). 1.0 is the balanced default.
    multi_query_tau: float = 1.0

    # P2 (2026-06-03): decoupled cross-attention READ path.
    # When True, the memory contribution to hidden states is produced by a
    # dedicated CrossAttentionMemoryV2.read module (slots get their OWN softmax,
    # out_proj zero-initialised, LoRA-B style) instead of the legacy
    # KV-prepend joint-attention path. This bypasses the "injection dilution"
    # root cause (researcher report 2026-06-03): in the prepend path the k=16
    # slot KV tokens share a single softmax with up to 1024 live tokens, so
    # memory receives only ~1.5% attention mass at long context, then gets
    # further attenuated by slot_delta clip + inject_gate to ~0.2%.
    #
    # With use_decoupled_read=True:
    #   * The slot KV-prepend block (M_sel_hidden) is NOT added to the extended
    #     sequence — the wrapped layer runs on [L3 | L2 | H] (or pure bypass),
    #     so slots no longer compete in the live-token softmax.
    #   * A standalone cross-attn read (Q=hidden, K/V=slots) computes the memory
    #     contribution over its OWN softmax and is added to next_hidden via the
    #     same content-conditioned inject_gate g. out_proj=0 → step-0 output = 0
    #     so behaviour at init is identical to "no memory injection".
    #   * Top-k routing + writeback are UNCHANGED (still drive which slots get
    #     written); only the READ-to-hidden path is decoupled.
    #
    # Default False = legacy prepend path, fully backward-compatible.
    # See versions/v15_decoupled_read.md + status/MEMORY_PROTOCOL_PLAN.md [P2].
    use_decoupled_read: bool = False

    # P8 (2026-06-05): dedicated memory cross-attention READ path with its OWN
    # softmax + a per-head content-dependent gate that is ACTIVE at init.
    # Motivation (researcher P2 root cause + the two hard data points
    # 2026-06-05): the KV-prepend joint-attention path puts the ~16 slot KV
    # tokens in the SAME softmax as up to ~1024 live tokens, so the slots get
    # only ~0.2% of the attention mass — diluted to irrelevance. Even with the
    # P7 routing fix (stable, learnable routing) BABILong stayed flat because
    # there is no gradient/signal flowing through memory. LongBench QA F1 fell
    # to 2.94 vs base 13.95 — the prepend read-back is destroying context.
    #
    # P8 vs P2 (use_decoupled_read):
    #   * BOTH give slots a standalone cross-attention with an independent
    #     softmax (Q=hidden, K/V=selected/all slots) and mask off the H->L1
    #     prepend so live tokens no longer share their softmax with slot KV.
    #   * P2 uses CrossAttentionMemoryV2 with out_proj ZERO-init and blends via
    #     the shared scalar-ish inject_gate (g≈0.12 with inject_gate_bias_init
    #     -2.0). The zero-init out_proj + tiny gate means the read path is
    #     ~dead at init, so it inherits the same "no gradient through memory"
    #     problem the dilution caused — just from the other side.
    #   * P8 uses a DEDICATED MemoryCrossAttentionRead module whose output IS
    #     active at init: out_proj is small-random (NOT zero) and the blend is a
    #     per-head, content-dependent gate (sigmoid) initialised so the
    #     effective contribution is ~memory_xattn_gate_init (default 0.4, in the
    #     0.3-0.5 band). This guarantees real gradient flows through the memory
    #     read from step 0. Reference designs: YOCO (2405.05254), Memorizing
    #     Transformers (2203.08913), Infini-attention (2404.07143).
    #
    # When True: the slot KV-prepend (M_sel_hidden) is removed from the extended
    # sequence (H->L1 masked, same mask_h_to_l1 plumbing as P2) and the memory
    # READ contribution is produced by the dedicated cross-attn + per-head gate.
    # Writeback + routing are UNCHANGED. Default False = legacy prepend path.
    # use_memory_xattn takes precedence over use_decoupled_read if both are set.
    # See status/MEMORY_PROTOCOL_PLAN.md [P8] + versions/v17_memory_xattn.md.
    use_memory_xattn: bool = False
    # Effective per-head gate contribution at init (sigmoid space). 0.4 sits in
    # the 0.3-0.5 band the plan asks for: large enough that the read path is
    # active from step 0 (real gradient), small enough not to swamp the residual
    # stream of the frozen backbone before the gate learns to modulate.
    memory_xattn_gate_init: float = 0.4
    # D6 (2026-06-09): single-variable ablation flag to DISABLE the learnable
    # null/sink slot inside MemoryCrossAttentionRead. Default False = sink ON
    # (backward-compatible; matches P8-nullsink / P11 baseline). When True the
    # read softmax has NO "attend to nothing" escape column — every query must
    # distribute all of its mass across the real slots. Isolates the null-sink's
    # contribution while holding the dedicated xattn read mechanism fixed.
    memory_xattn_disable_null_sink: bool = False

    # FastMem (Gated Delta Rule continuous memory, 2026-05-21):
    # Per-layer fast-weight memory that captures a continuous running summary
    # of ALL tokens (complementing discrete top-k slot routing which only
    # stores ~12.5%).  Uses the Gated Delta Rule update for high associative
    # capacity with chunk-wise parallelism.
    # Reference: ops/research_notes/20260521_fast_weight_memory_research.md §3.
    use_fast_mem: bool = False
    fast_mem_num_heads: int = 4         # H: number of fast-weight heads
    fast_mem_d_state: int = 128         # d_k = d_v per head
    fast_mem_chunk_size: int = 16       # BPTT window for sequential fallback (ignored when fla available)
    fast_mem_fusion_init: float = -2.0  # sigmoid(-2)≈0.12 initial contribution

    # EXP-R1 (2026-06-11): two-stage dead-slot recycling.
    # Breaks the L1 cold-start "rich-get-richer" deadlock (gp-59 root-cause
    # 20260611_dead_slot_recycling.md): with strided_token init + selected-only
    # delta-write, ~91/128 slots stay frozen at their chunk-0 token snapshot
    # forever while the same ~32 live slots monopolise all content. This is a
    # WRITE-SIDE content-staleness loop — the selector is greedy and noise-free
    # and must STAY that way (the read-side ROUTE-A arm4 noise experiment
    # collapsed top1_sim 0.99→0.11; do NOT perturb selection).
    #
    # Mechanism (VQ-VAE "random restart" precedent, Dhariwal Jukebox 2020):
    #   ① RESET: every `dead_slot_reset_interval` chunks (per-sample), detect
    #      slots that were NEVER selected by top-k in the last interval window
    #      (per-sample zero-usage counter), and OVERWRITE their CONTENT with
    #      strided tokens from the CURRENT chunk's hidden states, picking the
    #      tokens whose content is MOST cosine-distant from the existing LIVE
    #      slots (maximise diversity, occupy the content region live slots do not
    #      cover). NEVER pooled-mean broadcast (that homogenises → key collapse).
    #      Only dead-slot rows are touched; live slots' content/keys/selection
    #      are byte-identical. Because K_sel(slot_content) is recomputed every
    #      forward (selector.py:263-266), a reset slot's key changes immediately
    #      → it "re-enters view" and can be selected by natural content match.
    #   ② GRACE-WINDOW FORCED WRITE: for the next `dead_slot_grace_chunks`
    #      chunks, FORCE the just-reset slots into the top-k WRITE set (the
    #      memory-bank write idx), so delta-rule gives them real content and
    #      refines their key from "raw token snapshot" onto the manifold. This
    #      is WRITE-ONLY: under use_memory_xattn the read attends ALL slots via a
    #      separate softmax and the H→L1 prepend is masked, so forcing the write
    #      idx never forces the query to READ these slots (the key discriminator
    #      vs arm4: arm4 forced "which slots are read" and destroyed precise
    #      hits; here the selector still decides reads by content similarity).
    #
    # Default OFF (interval=0): every code path below is a no-op when
    # dead_slot_reset_interval <= 0, so the layer is byte-identical to P11.
    dead_slot_reset_interval: int = 0
    dead_slot_reset_mode: str = "strided_current"   # {"strided_current", "zero"}
    dead_slot_grace_chunks: int = 1

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
        if self.writeback_mode not in _VALID_WRITEBACK_MODE:
            raise ValueError(
                f"writeback_mode must be one of {_VALID_WRITEBACK_MODE}, "
                f"got {self.writeback_mode!r}"
            )
        if self.writeback_mode == "lowrank_gate" and self.lowrank_gate_rank <= 0:
            raise ValueError(
                f"lowrank_gate_rank must be > 0, got {self.lowrank_gate_rank}"
            )
        if not 0.0 < self.memory_xattn_gate_init < 1.0:
            raise ValueError(
                "memory_xattn_gate_init must be in (0, 1) (sigmoid space), got "
                f"{self.memory_xattn_gate_init}"
            )
        # EXP-R1 dead-slot recycling.
        if self.dead_slot_reset_interval < 0:
            raise ValueError(
                "dead_slot_reset_interval must be >= 0 (0 = disabled), got "
                f"{self.dead_slot_reset_interval}"
            )
        if self.dead_slot_grace_chunks < 0:
            raise ValueError(
                "dead_slot_grace_chunks must be >= 0, got "
                f"{self.dead_slot_grace_chunks}"
            )
        if self.dead_slot_reset_mode not in {"strided_current", "zero"}:
            raise ValueError(
                "dead_slot_reset_mode must be one of {'strided_current', "
                f"'zero'}}, got {self.dead_slot_reset_mode!r}"
            )
