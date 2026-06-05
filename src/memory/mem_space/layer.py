"""Memory-Space v0 — MemorySpaceLayer wrapping a LlamaDecoderLayer.

Design reference:
    ops/research_notes/20260426_memory_space_design_direction.md §2.1, §2.2, §2.3

Forward (high level)
--------------------
Given current hidden states ``H_l ∈ [B, T, d]`` entering a decoder layer:

    1.  Lazy-init slots from pooled H_l (+ gaussian noise) on first forward.
    2.  Top-k select slots via selector on mean-pooled H_l.
    3.  Gather ``M_sel ∈ [B, k, slot_dim]``, project to hidden-dim if needed.
    4.  Joint attention by KV-prepend: run the wrapped LlamaDecoderLayer on the
        extended sequence ``[M_sel, H_l]``.  We build (a) extended
        ``position_embeddings`` where slot positions inherit RoPE at
        position 0 (position-less memory tokens) and (b) an explicit additive
        attention mask that is causal among H but gives both slot-queries and
        H-queries full visibility into the slot block.
    5.  Split outputs: current-token positions [k:] become the next-layer
        hidden states; memory-token positions [0:k] are O_mem.
    6.  In-place EMA writeback on the selected slot positions, gated by β =
        σ(gate_param) · warmup_frac(step) · gate_max.

Key decisions (documented in the spec)
--------------------------------------
* RoPE for slots: we default to position-0 for every slot.  Turning on
  ``use_rope_for_slots`` is *not* implemented in v0 — the flag is reserved
  for Stage 2.  Raise if the user sets it (fail-loud on unimplemented knobs).
* Gate parameterisation: the design doc says "tanh-gated, init = 0 → output ≈
  0"; we implement this as ``β = σ(gate_param) · warmup · gate_max``.  With
  ``writeback_gate_init = 0``  →  σ(0) = 0.5, so after the warmup completes
  the *effective* β = 0.5 · gate_max = 0.15 by default (spec'd range 0.1-0.3).
  The sign is always non-negative, matching "EMA writeback".
* Aux-loss exposure: we stash ``last_aux_losses`` on the module (side
  channel) because LlamaDecoderLayer.forward returns a bare Tensor in
  transformers ≥ 5.0, not a tuple.  A collector in the outer training loop
  can walk ``model.model.layers`` and sum them.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _ckpt

from .config import MemorySpaceConfig
from .fast_mem import FastMemModule
from .memory_bank import MemoryBank
from .selector import TopKSelector


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _build_causal_attn_mask(
    T: int,
    dtype: torch.dtype,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Return [B, 1, T, T] additive causal mask matching HF LlamaModel's
    SDPA-path prep (`_prepare_4d_causal_attention_mask_for_sdpa`).

    Why this exists: the outer ``LlamaModel.forward`` always passes a
    prepared 4-D additive causal mask to every decoder layer under SDPA.
    Our bypass call at :class:`MemorySpaceLayer.forward` used to pass
    ``attention_mask=None``, which HF resolves by installing its own
    ``F.scaled_dot_product_attention(..., is_causal=True)`` path. For bf16,
    the *explicit-mask* and *is_causal flag* SDPA kernels can produce
    numerically distinct outputs (~0.2 % relative) — confirmed by the H2
    unit test `tests/test_bypass_call_dispatch.py` (2026-04-26:

        max|A-B| = 1.562e-02 across 5 seeds, bf16, layer 0

    which matched the §5.4 bypass-parity probe's ``err(L0) = 1.562e-02``
    exactly).

    Fix: pre-compute the same 4-D causal mask HF would have built, and
    pass it to the bypass ``wrapped_layer(...)`` call, so both the bypass
    and extended forwards dispatch SDPA through the same explicit-mask
    code path.

    Convention: 0 = allowed, ``torch.finfo(dtype).min`` = masked out.
    """
    mask = torch.zeros(T, T, dtype=dtype, device=device)
    neg_inf = torch.finfo(dtype).min
    mask = mask.masked_fill(
        torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1
        ),
        neg_inf,
    )
    return mask.view(1, 1, T, T).expand(batch_size, 1, T, T).contiguous()


def _build_extended_attn_mask(
    k: int,
    T: int,
    dtype: torch.dtype,
    device: torch.device,
    batch_size: int,
    swa_window: int = 0,
    k_l3: int = 0,
    mask_h_to_l1: bool = False,
) -> torch.Tensor:
    """Return [B, 1, L, L] additive mask for the joint-attn extended seq.

    The extended sequence layout is: [L3(k_l3) | L1(k) | H(T)].
    Total length L = k_l3 + k + T.

    Convention: 0 means "allowed", ``-inf`` means "masked out".

    Attention pattern:
        * L3 rows (0..k_l3-1): attend to everything (full row of zeros).
        * L1 rows (k_l3..k_l3+k-1): attend to everything (full row of zeros).
        * H rows (k_l3+k..L-1):
          - cols 0..k_l3-1 (L3 keys): always allowed.
          - cols k_l3..k_l3+k-1 (L1 keys): always allowed.
          - cols k_l3+k..L-1 (H keys): causal (or SWA-causal if swa_window>0).

    When k_l3 == 0: behaviour is IDENTICAL to the pre-L3 implementation.
    """
    prefix = k_l3 + k
    L = prefix + T
    # Default to "allowed everywhere".
    mask = torch.zeros(L, L, dtype=dtype, device=device)
    neg_inf = torch.finfo(dtype).min

    if T > 0:
        if swa_window <= 0:
            # Full-causal behaviour within H×H block.
            causal = torch.triu(
                torch.full((T, T), neg_inf, dtype=dtype, device=device),
                diagonal=1,
            )
            mask[prefix:, prefix:] = causal
        else:
            # SWA: for each pair (i, j) in H-space (0-indexed within H),
            # allow if j <= i AND (i - j) < swa_window; mask otherwise.
            hh = torch.full((T, T), neg_inf, dtype=dtype, device=device)
            rows = torch.arange(T, device=device).unsqueeze(1)  # [T, 1]
            cols = torch.arange(T, device=device).unsqueeze(0)  # [1, T]
            allowed = (cols <= rows) & ((rows - cols) < swa_window)
            hh[allowed] = 0.0
            mask[prefix:, prefix:] = hh
        # L3/L1-queries and H-queries→L3/L1 keys: already 0 (allowed).

    # P2 decoupled-read (2026-06-03): when mask_h_to_l1=True, BLOCK H-queries
    # from attending to the L1 slot block (cols k_l3..k_l3+k-1). This removes
    # the "injection dilution" — live tokens no longer share their softmax with
    # the prepended slot KV. L1-queries→H attention is KEPT (so the writeback's
    # O_mem_hidden is computed exactly as before); only the H→L1 read direction
    # is severed. The memory READ contribution is instead produced by the
    # standalone CrossAttentionMemoryV2.read path in forward().
    if mask_h_to_l1 and k > 0 and T > 0:
        mask[prefix:, k_l3:k_l3 + k] = neg_inf

    # Broadcast to [B, 1, L, L].
    return mask.view(1, 1, L, L).expand(batch_size, 1, L, L).contiguous()


def _build_extended_attn_mask_l2(
    k_l3: int,
    k_l2: int,
    k_l1: int,
    T: int,
    dtype: torch.dtype,
    device: torch.device,
    batch_size: int,
    swa_window: int = 0,
    mask_h_to_l1: bool = False,
) -> torch.Tensor:
    """Return [B, 1, L, L] additive mask for the L2-extended joint-attn seq.

    Extended layout: [L3(k_l3) | L2(k_l2) | L1(k_l1) | H(T)].
    Total length L = k_l3 + k_l2 + k_l1 + T.

    Attention pattern:
        * L3 rows: attend to everything (full row of zeros).
        * L2 rows: attend to everything.
        * L1 rows: attend to everything.
        * H rows:
          - cols 0..prefix-1 (L3, L2, L1 keys): always allowed.
          - cols prefix..L-1 (H keys): causal (or SWA-causal if swa_window>0).

    When k_l2 == 0: collapses to the [L3 | L1 | H] layout (same as
    ``_build_extended_attn_mask`` with the same k_l3/k_l1).
    """
    prefix = k_l3 + k_l2 + k_l1
    L = prefix + T
    mask = torch.zeros(L, L, dtype=dtype, device=device)
    neg_inf = torch.finfo(dtype).min

    if T > 0:
        if swa_window <= 0:
            causal = torch.triu(
                torch.full((T, T), neg_inf, dtype=dtype, device=device),
                diagonal=1,
            )
            mask[prefix:, prefix:] = causal
        else:
            hh = torch.full((T, T), neg_inf, dtype=dtype, device=device)
            rows = torch.arange(T, device=device).unsqueeze(1)
            cols = torch.arange(T, device=device).unsqueeze(0)
            allowed = (cols <= rows) & ((rows - cols) < swa_window)
            hh[allowed] = 0.0
            mask[prefix:, prefix:] = hh

    # P2 decoupled-read: sever H→L1 attention (cols k_l3+k_l2 .. prefix-1).
    if mask_h_to_l1 and k_l1 > 0 and T > 0:
        mask[prefix:, k_l3 + k_l2:prefix] = neg_inf

    return mask.view(1, 1, L, L).expand(batch_size, 1, L, L).contiguous()


def _extend_position_embeddings(
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepend k "position-0" entries to the (cos, sin) rotary tables.

    Each of cos, sin has shape ``[1 or B, T, head_dim]``.  We slice the
    position-0 entry and tile it k times.  Result has shape
    ``[*, k+T, head_dim]``.
    """
    cos, sin = position_embeddings
    # Handle both [1, T, D] and [B, T, D] layouts.
    if cos.dim() != 3 or sin.dim() != 3:
        raise ValueError(
            f"position_embeddings must be 3-D tensors; got cos={tuple(cos.shape)}, "
            f"sin={tuple(sin.shape)}"
        )
    cos0 = cos[:, :1, :]                                       # [*, 1, D]
    sin0 = sin[:, :1, :]
    cos_ext = torch.cat([cos0.expand(cos.shape[0], k, cos.shape[-1]), cos], dim=1)
    sin_ext = torch.cat([sin0.expand(sin.shape[0], k, sin.shape[-1]), sin], dim=1)
    return cos_ext, sin_ext


# --------------------------------------------------------------------------- #
# MemorySpaceLayer
# --------------------------------------------------------------------------- #


class MemorySpaceLayer(nn.Module):
    """Decoder layer wrapper that adds a per-layer memory-space.

    The wrapped ``LlamaDecoderLayer`` is kept unchanged — we only feed it an
    *extended* sequence at forward time and split its output.  This keeps all
    of HF's attention-implementation branching (SDPA / eager / FlashAttention)
    intact, matching the pattern used in ``src/memory/qfilters/layer.py``.

    Side-channel outputs (always populated after a forward):
        self.last_aux_losses: dict with keys
            * "load_balance": scalar (the Switch-style MoE aux loss)
            * "beta":         scalar (the gate value actually used)
            * "slot_usage":   [N] fraction of the batch that picked each slot
        self.last_idx, self.last_scores — for downstream probing.
    """

    # Class-level counter so each instance gets a stable ID across __init__
    # calls.  Used to restrict diagnostic logging to layer-0 only (avoids
    # 32× log spam from all patched decoder layers).
    _instance_counter: int = 0

    # Per-model flag: when True, forward() bypasses all memory logic and
    # calls the wrapped decoder layer directly (teacher mode for KD).
    # Set via model-level helper; default False preserves P1 behaviour.
    _memory_disabled: bool = False

    def __init__(
        self,
        wrapped_layer: nn.Module,
        config: MemorySpaceConfig,
        *,
        d_model: int,
        shared_bank: Optional[MemoryBank] = None,
        l3_pool: Optional[nn.Module] = None,
        l2_compressor: Optional[nn.Module] = None,
        recon_decoder: Optional[nn.Module] = None,
        n_heads: Optional[int] = None,
        n_kv_heads: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not isinstance(config, MemorySpaceConfig):
            raise TypeError(f"config must be MemorySpaceConfig, got {type(config)}")
        if config.use_rope_for_slots:
            raise NotImplementedError(
                "use_rope_for_slots=True is reserved for Stage 2 (not implemented "
                "in v0 — slot positions are hard-coded to RoPE position 0)."
            )

        self.wrapped_layer = wrapped_layer
        self.config = config
        self.d_model = d_model

        slot_dim = d_model if config.slot_dim is None else config.slot_dim
        self.slot_dim = slot_dim

        # Branch-3 (2026-04-26): optional shared bank across layers. When a
        # shared_bank is passed in (and config.shared_memory_bank is True), the
        # bank is registered as a non-submodule attribute so the same state is
        # visible to every MemorySpaceLayer and intra-chunk writes thread BPTT
        # through depth. We mark it with object.__setattr__ to bypass
        # nn.Module.__setattr__'s auto-registration logic that would otherwise
        # create N copies in the state_dict.
        if shared_bank is not None:
            if shared_bank.num_slots != config.num_slots:
                raise ValueError(
                    f"shared_bank.num_slots {shared_bank.num_slots} != "
                    f"config.num_slots {config.num_slots}"
                )
            if shared_bank.slot_dim != slot_dim:
                raise ValueError(
                    f"shared_bank.slot_dim {shared_bank.slot_dim} != slot_dim {slot_dim}"
                )
            # Register as a plain attribute (NOT a submodule) so shared state
            # is not duplicated across all 32 wrappers' state_dicts.
            object.__setattr__(self, "memory_bank", shared_bank)
            self._owns_bank = False
        else:
            self.memory_bank = MemoryBank(
                num_slots=config.num_slots,
                slot_dim=slot_dim,
                init_noise=config.slot_init_noise,
                slot_init=config.slot_init,
                slot_value_norm_cap=config.slot_value_norm_cap,
            )
            self._owns_bank = True

        # L3 summary pool — shared single instance across all layers (like
        # shared_bank). Registered via object.__setattr__ to avoid duplicating
        # in state_dict across 32 layers.
        if l3_pool is not None:
            object.__setattr__(self, "l3_pool", l3_pool)
        else:
            object.__setattr__(self, "l3_pool", None)

        # L2 token compressor — shared single instance across all layers
        # (peer to l3_pool). Registered via object.__setattr__ so the L2
        # parameters are NOT duplicated in every layer's state_dict.
        # patch.py registers l2_compressor as a named submodule on the model
        # root so its parameters do appear in model.parameters() / state_dict
        # exactly once.
        if l2_compressor is not None:
            object.__setattr__(self, "l2", l2_compressor)
        else:
            object.__setattr__(self, "l2", None)

        # MemoryReconDecoder (P1 / v12, 2026-06-01) — shared single instance
        # across all layers (peer to l3_pool). Registered via object.__setattr__
        # so the decoder params are NOT duplicated in every layer's state_dict;
        # patch.py registers it once as a named submodule on the model root.
        if recon_decoder is not None:
            object.__setattr__(self, "recon_decoder", recon_decoder)
        else:
            object.__setattr__(self, "recon_decoder", None)
        self.selector = TopKSelector(
            d_model=d_model,
            slot_dim=slot_dim,
            selector_dim=config.selector_dim,
            top_k=config.top_k,
            num_slots=config.num_slots,
            temperature=config.selector_temperature,
        )
        # P1-v2: wire config flag to selector
        self.selector._no_detach_slots = config.no_detach_slots_in_selector
        self.selector._routing_pool_mode = config.routing_pool_mode
        # v8 multi-query routing (2026-06-01): logsumexp aggregation temperature.
        self.selector._multi_query_tau = getattr(config, "multi_query_tau", 1.0)
        # v10 (2026-06-01): threshold for the post-projection q_multi diversity
        # loss. Reuse the L3 diversity threshold so both losses share semantics.
        self.selector._q_multi_diversity_threshold = getattr(
            config, "l3_diversity_threshold", 0.5
        )
        # P7 loss-free balancing (2026-06-05; arXiv:2408.15664): wire config →
        # selector. When enabled, an online per-slot bias steers top-k selection
        # toward balanced slot usage WITHOUT an interfering aux gradient. Should
        # be paired with load_balance_weight=0.
        self.selector.use_loss_free_balance = getattr(
            config, "use_loss_free_balance", False
        )
        self.selector.loss_free_update_rate = getattr(
            config, "loss_free_update_rate", 0.001
        )
        # P10 (2026-06-06): straight-through Gumbel top-k. When enabled, Gumbel
        # noise is added to the selection logits (training mode only) so which
        # slots win top-k is stochastic. Default off → byte-identical to pre-P10.
        self.selector.use_st_gumbel_topk = getattr(
            config, "use_st_gumbel_topk", False
        )
        self.selector.st_gumbel_temperature = getattr(
            config, "st_gumbel_temperature", 1.0
        )

        # Learnable slot↔hidden projections. We do NOT take the slot_dim==d_model
        # shortcut (Identity) because that path has zero trainable capacity and was
        # empirically responsible for the residual-gap pathology after fix1+fix2
        # (see ops/research_notes/20260426_mem_space_v0_tier2_residual_gap.md).
        # Fix3 (2026-04-26): always-on Linear so there is a parameterised path
        # between the slot bank and the decoder's K/V projections.
        self.slot_to_hidden = nn.Linear(slot_dim, d_model, bias=False)
        self.hidden_to_slot = nn.Linear(d_model, slot_dim, bias=False)
        # Tier-3 (2026-04-26): input-side zero on slot_to_hidden does NOT give
        # bypass parity because slot K/V are concatenated into the same softmax
        # as the bypass path, and k=64 phantom exp(0)=1 logits in the denominator
        # attenuate every H-query's attention output (compounded 32× → 60-90%
        # signal loss). See ops/research_notes/20260426_mem_space_v0_tier3_fix3_fail.md §2.
        # Replacement: keep slot_to_hidden small-random; bypass parity is now
        # guaranteed structurally by the OUTPUT-side tanh(alpha) gate in forward().
        nn.init.normal_(self.slot_to_hidden.weight, std=0.02)
        nn.init.normal_(self.hidden_to_slot.weight, std=0.02)

        # Flamingo-style output gate: tanh(alpha) multiplier on (ext_h - bypass_h).
        # Fix D.1 (2026-04-28): init = 0.5 → tanh(0.5) ≈ 0.462 → LM gradient flows
        # to selector immediately from step 1. Prior init = 0 caused alpha = 0 at
        # init which algebraically zeroed the gradient to all selector parameters
        # (Q_sel, slot_keys), causing permanent routing degeneracy.
        self.slot_output_gate = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        # P1 (2026-05-31): content-conditioned per-token injection gate.
        # Replaces the content-independent scalar alpha=tanh(slot_output_gate)
        # in forward() with g=sigmoid(inject_gate(hidden_states)) so the model
        # can learn to suppress injection when retrieval is irrelevant (e.g.
        # LongBench top1_sim≈0.015). See ops/research_notes/
        # 20260531_compression_memory_training_methods.md Part D 方案1.
        #
        # Initialization for smooth transition (no behavior break at step 0):
        #   weight = 0  -> gate output is purely the bias at init
        #   bias = logit(tanh(0.5)) = logit(0.462) ≈ -0.152
        #   so sigmoid(bias) ≈ 0.462 == the prior scalar alpha.
        self.inject_gate = nn.Linear(self.d_model, 1)
        # P1-v3: init weight with non-zero values so gate has per-token
        # variation from step 0. With weight=0 the gate is constant (std=0)
        # and receives zero gradient — a deadlock. Scale 0.1/sqrt(d_model)
        # gives initial gate output std ≈ 0.017 in bf16 (enough to break symmetry
        # while keeping initial behavior close to the constant-gate baseline).
        nn.init.normal_(self.inject_gate.weight, std=10.0 / (self.d_model ** 0.5))
        nn.init.constant_(self.inject_gate.bias, config.inject_gate_bias_init)

        # P2 (2026-06-03): decoupled cross-attention READ module.
        # When config.use_decoupled_read=True, the memory READ contribution to
        # hidden states is produced by this standalone CrossAttentionMemoryV2
        # (slots get their OWN softmax, out_proj zero-initialised) instead of
        # diluting the live-token softmax via KV-prepend. We use ONLY its
        # .read() (Q=hidden, K/V=slot projections) — writeback stays on the
        # existing top-k path. out_proj=0 → step-0 read output = 0 so init
        # behaviour is identical to "no memory injection". n_heads/n_kv_heads
        # default to a single full-head config when not supplied (e.g. toy /
        # tests that don't thread the model's GQA head counts).
        if config.use_decoupled_read:
            from .selector import CrossAttentionMemoryV2
            _nh = n_heads if n_heads is not None else max(1, d_model // 128)
            _nkv = n_kv_heads if n_kv_heads is not None else _nh
            self.decoupled_read = CrossAttentionMemoryV2(
                d_model=d_model,
                n_heads=_nh,
                n_kv_heads=_nkv,
                num_slots=config.num_slots,
                dropout=0.0,
            )
        else:
            self.decoupled_read = None

        # P8 (2026-06-05): dedicated memory cross-attention READ with its OWN
        # softmax and a per-head content-dependent gate that is ACTIVE at init.
        # When config.use_memory_xattn=True the H->L1 prepend is masked off (same
        # mask_h_to_l1 plumbing as P2) and the memory READ contribution is
        # produced here instead of via KV-prepend. Unlike P2's decoupled_read
        # (zero-init out_proj + tiny shared inject_gate ≈ dead at start), this
        # module's out_proj is small-random and the per-head gate inits to
        # ~memory_xattn_gate_init (0.4) so real gradient flows through memory
        # from step 0. Writeback + routing are untouched. n_heads/n_kv_heads
        # default to a sane single-/full-head config for toy tests that don't
        # thread the model's GQA head counts.
        if config.use_memory_xattn:
            from .selector import MemoryCrossAttentionRead
            _nh = n_heads if n_heads is not None else max(1, d_model // 128)
            _nkv = n_kv_heads if n_kv_heads is not None else _nh
            self.memory_xattn = MemoryCrossAttentionRead(
                d_model=d_model,
                n_heads=_nh,
                n_kv_heads=_nkv,
                gate_init=config.memory_xattn_gate_init,
                dropout=0.0,
            )
        else:
            self.memory_xattn = None

        # hidden_to_slot: freeze gate controlled by ``config.hidden_to_slot_frozen``.
        #
        # Historical note (pre-Fix-J, 2026-04-29): the comment here used to claim
        # that hidden_to_slot "participates in NO operation whose output influences
        # the loss" because:
        #   1. O_mem_slot = hidden_to_slot(O_mem_hidden)
        #   2. memory_bank.write(idx, O_mem_slot, beta) — was DETACHED on write
        #   3. _reset_banks discards the bank every chunk
        # That claim was made TRUE in two places:
        #   - memory_bank.write detached (removed in Branch-3, 2026-04-26)
        #   - layer.py:499 soft-proxy einsum detached `slots` (removed in Fix J-A,
        #     2026-04-29)
        # With both detaches removed AND hidden_to_slot in the optimizer via
        # --unfreeze_hidden_to_slot (Fix I, 2026-04-29), gradient now flows:
        #   loss → next_hidden → M_sel_hidden → M_sel_slot_soft → slots
        #        → (scatter from prior write) → O_mem_slot → hidden_to_slot.weight
        #
        # Stage-2a (2026-04-26): the freeze is now gated on
        # `config.hidden_to_slot_frozen`. Set False (via --unfreeze_hidden_to_slot)
        # to let the projection train; this is the default in the mem_space
        # ablation path after Fix J-A.
        if config.hidden_to_slot_frozen:
            for p in self.hidden_to_slot.parameters():
                p.requires_grad = False

        # Writeback gate — single learnable scalar, init ``writeback_gate_init``.
        # Effective β = sigmoid(gate_param) · warmup_frac · gate_max.
        self.gate_param = nn.Parameter(
            torch.tensor(float(config.writeback_gate_init))
        )

        # H6 (LM2-inspired): dual-gate projections (input + forget).
        # Each projection maps slot_dim → 2 * slot_dim, the two halves are
        # interpreted as (g_in_logit, g_forget_logit) before sigmoid.
        # Two separate projections (one for new_repr, one for current slot)
        # match LM2's design: gates condition on BOTH new content and prior
        # memory state. forget_bias is added to the second half so g_forget
        # starts ≈ sigmoid(forget_bias_init).
        # Writeback-mode resolution (2026-06-04). `writeback_mode` is the
        # authoritative selector for the gate parameterisation. For backward
        # compatibility a plain `use_dual_gate=True` with the DEFAULT mode
        # ("dual_gate") still builds the full LM2 dual-gate projections, so all
        # pre-existing checkpoints (which only ever set use_dual_gate) construct
        # byte-identically. When the user explicitly selects a cheaper mode
        # (lowrank_gate / diag_gate / scalar_beta) the dual-gate projections are
        # NOT built — this is what avoids the slot_dim=16384 OOM (two
        # Linear(16384, 32768) ≈ 34B params/model).
        _dual_gate_active = (config.writeback_mode == "dual_gate") and config.use_dual_gate
        if _dual_gate_active:
            self.gate_proj_new = nn.Linear(self.slot_dim, 2 * self.slot_dim, bias=False)
            self.gate_proj_mem = nn.Linear(self.slot_dim, 2 * self.slot_dim, bias=False)
            # Bias vector: zeros for input half, forget_bias_init for forget half
            bias_init = torch.cat([
                torch.full((self.slot_dim,), float(config.input_bias_init)),
                torch.full((self.slot_dim,), float(config.forget_bias_init)),
            ])
            self.gate_bias = nn.Parameter(bias_init)
            # Init projections small (xavier with fan-in) so initial gates ≈ sigmoid(bias)
            nn.init.xavier_uniform_(self.gate_proj_new.weight, gain=0.5)
            nn.init.xavier_uniform_(self.gate_proj_mem.weight, gain=0.5)
        else:
            self.gate_proj_new = None
            self.gate_proj_mem = None
            self.gate_bias = None

        # lowrank_gate (A, 2026-06-04): two-stage low-rank projection of
        # (s_new, M_prev) → rank r → 2*slot_dim gate logits.
        #   gate_logits = U( V_new(s_new) + V_mem(M_prev) ) + lr_gate_bias
        # Cost ≈ (2*slot_dim*r + r*2*slot_dim) = 4*slot_dim*r per layer (vs
        # 4*slot_dim^2 for dual_gate). bias halves seeded input/forget like
        # dual_gate so g_forget starts ≈ sigmoid(forget_bias_init).
        if config.writeback_mode == "lowrank_gate":
            r = config.lowrank_gate_rank
            self.lr_V_new = nn.Linear(self.slot_dim, r, bias=False)
            self.lr_V_mem = nn.Linear(self.slot_dim, r, bias=False)
            self.lr_U = nn.Linear(r, 2 * self.slot_dim, bias=False)
            lr_bias_init = torch.cat([
                torch.full((self.slot_dim,), float(config.input_bias_init)),
                torch.full((self.slot_dim,), float(config.forget_bias_init)),
            ])
            self.lr_gate_bias = nn.Parameter(lr_bias_init)
            nn.init.xavier_uniform_(self.lr_V_new.weight, gain=0.5)
            nn.init.xavier_uniform_(self.lr_V_mem.weight, gain=0.5)
            nn.init.xavier_uniform_(self.lr_U.weight, gain=0.5)
        else:
            self.lr_V_new = None
            self.lr_V_mem = None
            self.lr_U = None
            self.lr_gate_bias = None

        # diag_gate (B, 2026-06-04): per-feature diagonal (element-wise) gate.
        #   g_in_logit     = a_in * s_new + c_in * M_prev + b_in
        #   g_forget_logit = a_f  * s_new + c_f  * M_prev + b_f
        # Cost = 6*slot_dim params/layer (no full matrix). a/c init N(0,0.02);
        # bias halves seeded like dual_gate (b_in=input_bias_init, b_f=forget_bias_init).
        if config.writeback_mode == "diag_gate":
            self.diag_a_in = nn.Parameter(torch.empty(self.slot_dim).normal_(0.0, 0.02))
            self.diag_c_in = nn.Parameter(torch.empty(self.slot_dim).normal_(0.0, 0.02))
            self.diag_a_f = nn.Parameter(torch.empty(self.slot_dim).normal_(0.0, 0.02))
            self.diag_c_f = nn.Parameter(torch.empty(self.slot_dim).normal_(0.0, 0.02))
            self.diag_b_in = nn.Parameter(
                torch.full((self.slot_dim,), float(config.input_bias_init))
            )
            self.diag_b_f = nn.Parameter(
                torch.full((self.slot_dim,), float(config.forget_bias_init))
            )
        else:
            self.diag_a_in = None
            self.diag_c_in = None
            self.diag_a_f = None
            self.diag_c_f = None
            self.diag_b_in = None
            self.diag_b_f = None

        # Step counter (incremented by the outer training loop).
        self.step_counter: int = 0

        # FastMem (Gated Delta Rule, 2026-05-21): per-layer continuous memory.
        if config.use_fast_mem:
            self.fast_mem = FastMemModule(
                d_model=d_model,
                num_heads=config.fast_mem_num_heads,
                d_state=config.fast_mem_d_state,
                chunk_size=config.fast_mem_chunk_size,
                fusion_init=config.fast_mem_fusion_init,
            )
        self._fast_mem_state: Optional[torch.Tensor] = None

        # Internal forward-call counter (independent of step_counter;
        # incremented every forward() call; used for diagnostic log scheduling
        # when global_step is not passed into forward()).
        self._fwd_count: int = 0

        # Instance index: 0 = first layer constructed, 1 = second, etc.
        # Diagnostic logs are emitted only from instance 0 (layer 0).
        self._layer_idx: int = MemorySpaceLayer._instance_counter
        MemorySpaceLayer._instance_counter += 1

        # Side-channel state (populated on each forward).
        self.last_aux_losses: Dict[str, torch.Tensor] = {}
        self.last_idx: Optional[torch.Tensor] = None
        self.last_scores: Optional[torch.Tensor] = None

        # Cross-chunk slot-usage histogram (diagnostic only, layer-0 use).
        # Accumulates how often each of the N slots is picked by the routed
        # top-k across the chunks BETWEEN two QUERY_DIAG emissions. Reset after
        # each emission so the reported coverage/entropy reflect a fresh window.
        # This answers "are slots being USED in a varied way across chunks?"
        # (a routing-distribution signal), complementing key_max_cos which
        # answers "are slot KEYS separable?" (a representation signal).
        self._slot_usage_hist: Optional[torch.Tensor] = None
        self._slot_usage_chunks: int = 0

        # Cross-chunk Jaccard accumulator (2026-06-04): tracks how much the
        # routed slot SET changes from one chunk to the next. Distinguishes
        # "every chunk picks a different 16 slots" (true content addressing,
        # Jaccard->0) from "every chunk picks the SAME 16 slots" (degenerate
        # shortcut, Jaccard->1). route_aux (uniform top-k supervision) cannot
        # tell these apart, so this is the key signal for whether routing
        # actually learned content-specific addressing.
        self._prev_chunk_idx_set: Optional[torch.Tensor] = None
        self._jaccard_sum: float = 0.0
        self._jaccard_count: int = 0

        # Latest layer-0 diagnostic scalars, refreshed inside the QUERY_DIAG
        # block (every 50 fwd). The training loop reads these at log_interval
        # to push to wandb. Default 0.0 until the first diag emission.
        self._last_key_max_cos: float = 0.0
        self._last_usage_cov: float = 0.0
        self._last_usage_ent: float = 0.0

    # --------------------------------------------------------------------- #
    # Gate
    # --------------------------------------------------------------------- #

    def _current_beta(self) -> torch.Tensor:
        """Return the effective writeback gate as a scalar tensor."""
        cfg = self.config
        warmup = cfg.writeback_gate_warmup_steps
        if warmup <= 0:
            warmup_frac = 1.0
        else:
            warmup_frac = min(float(self.step_counter) / float(warmup), 1.0)
        # σ in (0, 1), times warmup in [0, 1], times cap.
        return torch.sigmoid(self.gate_param) * warmup_frac * cfg.writeback_gate_max

    # --------------------------------------------------------------------- #
    # Ablation bypass
    # --------------------------------------------------------------------- #

    def _maybe_ckpt_wrapped_layer(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Any,
    ) -> Any:
        """Call ``self.wrapped_layer(hidden_states, **kwargs)``, optionally under
        ``torch.utils.checkpoint`` for activation-memory reduction.

        Phase 11 (2026-05-16): the L1+L2+L3 stack at chunk_size=1024 + 4k
        context = 4 chunks/sample BPTT pushes peak past H20's 97 GB. Wrapping
        the wrapped LlamaDecoderLayer forward in checkpoint cuts activation
        memory ~50% at ~2x compute. Only enabled when training (avoids the
        compute hit + the no-grad / inference incompatibility) and when the
        config flag is set.

        We use a closure so kwargs (which include non-Tensor values like None
        and tuples of Tensors) pass through cleanly — torch.utils.checkpoint
        unrolls only positional Tensor args by default, but accepts kwargs via
        a Python closure that captures them.

        FSDP-mode (2026-05-16): when this MemorySpaceLayer is wrapped inside an
        FSDP unit (see ``_wrap_model_fsdp`` in
        ``scripts/train_mem_space_babilong.py``), manual ``torch.utils.checkpoint``
        is incompatible with FSDP's reshard_after_forward (would recompute
        without resharded params). FSDP-native activation checkpointing must be
        applied at wrap time. The training script sets
        ``self._inside_fsdp_unit = True`` after FSDP wrapping; when set, we
        skip the manual checkpoint here and just call wrapped_layer directly
        — FSDP / its checkpoint_wrapper (if applied) handles activation memory.
        """
        if getattr(self, "_inside_fsdp_unit", False):
            return self.wrapped_layer(hidden_states, **kwargs)
        if not (self.config.gradient_checkpointing and self.training):
            return self.wrapped_layer(hidden_states, **kwargs)

        def _ckpt_fn(h: torch.Tensor) -> Any:
            return self.wrapped_layer(h, **kwargs)

        return _ckpt.checkpoint(_ckpt_fn, hidden_states, use_reentrant=False)

    # --------------------------------------------------------------------- #
    # Ablation bypass
    # --------------------------------------------------------------------- #

    def forward_no_memory(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Skip memory entirely and call the wrapped decoder layer directly.

        Intended for ablation / numerical parity checks against a vanilla
        baseline.  Guaranteed to produce *identical* outputs to the original
        LlamaDecoderLayer within fp32 tolerance — nothing in the memory path
        is touched.
        """
        return self.wrapped_layer(hidden_states, **kwargs)

    # --------------------------------------------------------------------- #
    # Main forward
    # --------------------------------------------------------------------- #

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        l3_summaries: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # Teacher-mode bypass: when _memory_disabled is set (e.g. for KD
        # teacher pass), skip all memory logic and run the wrapped layer
        # directly. This produces identical output to vanilla Llama.
        if self._memory_disabled:
            return self.forward_no_memory(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        # v0 does not integrate with HF's DynamicCache (Stage 2 work).  HF's
        # outer LlamaModel forward flips ``use_cache`` on by default for
        # inference; we silently downgrade to ``False`` here so the smoke-test
        # / warmup path works.  ``past_key_values`` is also ignored — the
        # joint-attn extended sequence cannot be incrementally decoded yet.
        use_cache = False
        past_key_values = None
        if position_embeddings is None:
            raise RuntimeError(
                "MemorySpaceLayer requires `position_embeddings` (cos, sin) from "
                "the outer LlamaModel forward. This is always provided by HF "
                "transformers >= 4.45."
            )

        # L3: if not explicitly provided, compute fresh from the shared l3_pool
        # using the PREVIOUS chunk's detached H (stashed by post-forward hook).
        # Calling pool(_prev_chunk_h) here, inside the current chunk's forward,
        # gives the pool's parameters a clean gradient path through this chunk's
        # loss.backward(). _prev_chunk_h is detached so we never reach into the
        # previous chunk's already-freed graph.
        # First chunk: _prev_chunk_h is None → l3_summaries stays None → no L3
        # prepend (cold start).
        # Per-chunk cache: pool() is expensive (~50ms × 32 layers = 1.6s overhead).
        # Cache on the pool itself; cleared by patch.py hook at end of chunk.
        if l3_summaries is None and self.l3_pool is not None:
            cached = getattr(self.l3_pool, "_chunk_summary_cache", None)
            if cached is not None:
                l3_summaries = cached
            else:
                prev_h = getattr(self.l3_pool, "_prev_chunk_h", None)
                if prev_h is not None:
                    prev_summary = getattr(self.l3_pool, "_prev_summary", None)
                    l3_summaries = self.l3_pool(prev_h, prev_summary=prev_summary)
                    object.__setattr__(self.l3_pool, "_chunk_summary_cache", l3_summaries)

        B, T, d = hidden_states.shape
        if d != self.d_model:
            raise RuntimeError(
                f"hidden_states last-dim {d} != d_model {self.d_model}"
            )

        # Increment internal forward counter (cheap; done before any heavy work).
        self._fwd_count += 1

        cfg = self.config

        # Effective k for L1: 0 when disable_l1_inject is set (pure-L3 ablation).
        k_slots_effective = 0 if cfg.disable_l1_inject else cfg.top_k

        # 1. Lazy-init / re-init on batch-size change.
        if not cfg.disable_l1_inject:
            cold_start_this_call = not self.memory_bank.is_initialized(B)
            if cold_start_this_call:
                # Slot dim may differ from d_model; project first if needed.
                H_for_init = hidden_states
                if self.slot_dim != self.d_model:
                    H_for_init = self.hidden_to_slot(hidden_states)
                self.memory_bank.init_from_hidden(H_for_init, batch_size=B)

            slots = self.memory_bank.get()                         # [B, N, slot_dim]

            # 2. Top-k select over hidden states (Fix Z.2: per-token routing).
            # Pass full [B, T, d_model] instead of mean-pooled [B, d_model].
            # v8 (2026-06-01): also pass the L3 summary tokens as multi-query
            # sub-queries (used only when routing_pool_mode=="multi_query"; the
            # selector falls back to max_pool when l3_summaries is None).
            idx, scores, ste_weights = self.selector(
                hidden_states, slots, query_tokens=l3_summaries
            )  # idx:[B,k], scores:[B,N]
            self._last_top1_sim = scores.max(dim=-1).values.float().mean().item()
            k_slots = idx.shape[-1]

            # v7 (2026-05-18): Always-on global slots — append the last
            # num_global_slots indices unconditionally (these bypass top-k routing
            # so they are updated on every forward call regardless of relevance
            # score; intended to provide EMA-free accumulation registers).
            if cfg.num_global_slots > 0:
                _g = cfg.num_global_slots
                _N = cfg.num_slots
                _glob_idx = torch.arange(_N - _g, _N, device=idx.device, dtype=idx.dtype)
                _glob_idx = _glob_idx.unsqueeze(0).expand(B, -1)   # [B, g]
                # Concatenate; keep duplicates (scatter will just overwrite twice
                # which is fine — the replacement write for globals wins).
                idx = torch.cat([idx, _glob_idx], dim=1)            # [B, k+g]
                k_slots = idx.shape[-1]

            # ---- Cross-chunk slot-usage accumulation (layer-0, no-op compute) ----
            # Tally the top-k slot picks for batch[0] into a persistent
            # histogram so the next QUERY_DIAG can report how the routing
            # distributes load over the N slots across MANY chunks (not just
            # within one). idx[0] holds the routed indices (k or k+g).
            if self._layer_idx == 0:
                with torch.no_grad():
                    if (self._slot_usage_hist is None
                            or self._slot_usage_hist.numel() != cfg.num_slots):
                        self._slot_usage_hist = torch.zeros(
                            cfg.num_slots, device=idx.device, dtype=torch.long
                        )
                        self._slot_usage_chunks = 0
                    self._slot_usage_hist.scatter_add_(
                        0, idx[0].long(),
                        torch.ones_like(idx[0], dtype=torch.long),
                    )
                    self._slot_usage_chunks += 1

                    # ---- Cross-chunk Jaccard of the routed slot SET ----
                    # Jaccard(prev_set, curr_set) = |∩| / |∪| over the unique
                    # slot indices picked for batch[0]. Accumulate the mean
                    # across chunks; reset together with usage at emission.
                    # First chunk (no prev) just stores the set.
                    _cur_set = idx[0].long().unique()  # dedup (globals may dup)
                    if self._prev_chunk_idx_set is not None:
                        _prev = self._prev_chunk_idx_set.to(_cur_set.device)
                        _inter = torch.isin(_cur_set, _prev).sum().item()
                        _union = _cur_set.numel() + _prev.numel() - _inter
                        if _union > 0:
                            self._jaccard_sum += _inter / _union
                            self._jaccard_count += 1
                    self._prev_chunk_idx_set = _cur_set

            # ---- QUERY_DIAG (diagnostic log, no-op on computation) ----
            # Emit every 200 forward calls, rank-0 / layer-0 only.
            _should_log_diag = (
                self._layer_idx == 0
                and self._fwd_count % 50 == 0  # Fix L-3 (2026-04-29): 200→50 for earlier norm explosion detection
            )
            try:
                import torch.distributed as _dist_diag
                if _dist_diag.is_available() and _dist_diag.is_initialized():
                    _should_log_diag = _should_log_diag and (_dist_diag.get_rank() == 0)
            except Exception:
                pass
            if _should_log_diag:
                with torch.no_grad():
                    # top-1 similarity scores (highest score per batch item)
                    _top1_sim = scores.max(dim=-1).values          # [B]
                    _top1_sim_mean = _top1_sim.float().mean().item()
                    # topk_mass (2026-06-04): total softmax mass captured by the
                    # k (or k+g) selected slots. ->1.0 = mass concentrated on the
                    # chosen set (hard routing is real); ->k/N (=16/128=0.125) =
                    # mass smeared over all slots, top-k selection is meaningless.
                    _topk_mass = scores.gather(1, idx).sum(dim=-1).float().mean().item()
                    # norm of the currently-selected slots (before projection)
                    _idx_exp_diag = idx.unsqueeze(-1).expand(-1, -1, self.slot_dim)
                    _M_sel_diag = slots.gather(1, _idx_exp_diag)   # [B, k, slot_dim]
                    _retrieved_norm = _M_sel_diag.float().norm(dim=-1).mean().item()
                    # Fix Z.2: per-token logit variance diagnostic
                    _pt_std = getattr(self.selector, '_last_per_token_logit_std', 0.0)
                    # Fix Z.2f: content-based key diversity (includes slot_key_bias)
                    _K_content = F.normalize(
                        self.selector.K_sel(slots) + self.selector.slot_key_bias.unsqueeze(0),
                        dim=-1,
                    )  # [B, N, S]
                    _K0 = _K_content[0]  # [N, S]
                    _key_sim = torch.mm(_K0, _K0.t()).fill_diagonal_(0.0)
                    _key_max_cos = _key_sim.abs().max().item()
                    # v8 multi-query diagnostics (default 0.0 when not multi_query)
                    _sq_max_cos = getattr(self.selector, "_last_summary_query_max_cos", 0.0)
                    _sq_mean_cos = getattr(self.selector, "_last_summary_query_mean_cos", 0.0)
                    _uniq_sel = getattr(self.selector, "_last_unique_selected_slots", 0)
                    # v10: pre-projection S diversity (raw L3 summary tokens,
                    # before Q_sel). Low S_max_cos + high summary_q_max_cos
                    # ⇒ Q_sel projection collapse confirmed.
                    _S_max_cos = getattr(self.selector, "_last_S_max_cos", 0.0)
                    # v11: slot_query attention entropy (default 0.0 when not
                    # slot_query). High ⇒ slots smear attention over all tokens.
                    _slot_attn_entropy = getattr(self.selector, "_last_slot_attn_entropy", 0.0)
                    # v11 fix (2026-06-04): cross-chunk slot-USAGE distribution.
                    # The previous `sel_slots = idx[0].unique().numel()` was a
                    # useless constant: top-k returns distinct indices by
                    # construction, so with B=1 it ALWAYS equals k_slots and
                    # tells us nothing about whether slots are "separable".
                    # Replace it with two signals computed over the accumulated
                    # multi-chunk histogram:
                    #   usage_cov  = #slots picked at least once / N   (coverage)
                    #   usage_ent  = normalized entropy of the usage distribution
                    #                in [0,1]; 1.0 = perfectly uniform load,
                    #                ->0 = routing collapsed onto a few slots.
                    # Together these answer "is the router USING varied slots
                    # across chunks?" — the real question, distinct from
                    # key_max_cos (key-space separability).
                    _usage_cov = 0.0
                    _usage_ent = 0.0
                    _usage_var = 0.0
                    _usage_chunks = self._slot_usage_chunks
                    if self._slot_usage_hist is not None and _usage_chunks > 0:
                        _h = self._slot_usage_hist.float()
                        _N_slots = _h.numel()
                        _usage_cov = (_h > 0).float().mean().item()
                        _tot = _h.sum()
                        if _tot > 0:
                            _p = _h / _tot
                            _nz = _p[_p > 0]
                            _ent = -(_nz * _nz.log()).sum()
                            import math as _math
                            _usage_ent = (_ent / _math.log(_N_slots)).item() if _N_slots > 1 else 0.0
                            # usage_var (2026-06-04): population variance of the
                            # per-slot selection-probability distribution over the
                            # diag window. Uniform load (p_i=1/N) -> 0; routing
                            # collapsed onto a few slots -> large. Complements
                            # usage_ent (entropy) with a direct dispersion measure.
                            _usage_var = _p.var(unbiased=False).item()
                    # reset window so each emission reflects fresh chunks
                    if self._slot_usage_hist is not None:
                        self._slot_usage_hist.zero_()
                        self._slot_usage_chunks = 0
                    # cross-chunk Jaccard: mean over the accumulated window.
                    # ->0 = each chunk routes to a different slot set (ideal
                    # content addressing); ->1 = every chunk picks the same set
                    # (degenerate). Reset window like usage; keep _prev set so
                    # the chain stays continuous across emissions.
                    _chunk_idx_jaccard = self._jaccard_sum / max(1, self._jaccard_count)
                    self._jaccard_sum = 0.0
                    self._jaccard_count = 0
                    # Stash scalars so the training loop can push them to wandb
                    # at log_interval (these are layer-0 only; the diag block
                    # only runs every 50 fwd, so the loop reads the latest).
                    self._last_key_max_cos = _key_max_cos
                    self._last_usage_cov = _usage_cov
                    self._last_usage_ent = _usage_ent
                    self._last_usage_var = _usage_var
                print(
                    f"[QUERY_DIAG step={self.step_counter} fwd={self._fwd_count}]"
                    f" top1_sim_mean={_top1_sim_mean:.6f}"
                    f" topk_mass={_topk_mass:.6f}"
                    f" retrieved_norm_mean={_retrieved_norm:.6f}"
                    f" per_tok_logit_std={_pt_std:.6f}"
                    f" key_max_cos={_key_max_cos:.4f}"
                    f" S_max_cos={_S_max_cos:.4f}"
                    f" summary_q_max_cos={_sq_max_cos:.4f}"
                    f" summary_q_mean_cos={_sq_mean_cos:.4f}"
                    f" uniq_sel_slots={_uniq_sel}"
                    f" usage_cov={_usage_cov:.4f}"
                    f" usage_ent={_usage_ent:.4f}"
                    f" usage_var={_usage_var:.6f}"
                    f" usage_chunks={_usage_chunks}"
                    f" chunk_idx_jaccard={_chunk_idx_jaccard:.4f}"
                    f" slot_attn_entropy={_slot_attn_entropy:.4f}",
                    flush=True,
                )
            # -----------------------------------------------------------

            # 2b. (Optional) slot dropout at train time.
            if self.training and cfg.slot_dropout > 0.0:
                drop_mask = (torch.rand_like(scores) < cfg.slot_dropout)  # True → drop
                scores = scores.masked_fill(drop_mask, 0.0)
                # Note: we do NOT re-pick idx — dropout affects the load-balance
                # loss path only.  Slots themselves continue to be selected via the
                # original idx to preserve hard top-k semantics.

            # 3. Gather selected slots, project to hidden dim if needed.
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, self.slot_dim)
            M_sel_slot = slots.gather(1, idx_exp)                   # [B, k, slot_dim]
            # Fix E (2026-04-28): Do NOT attenuate M_sel_slot by w_gathered before projecting
            # to hidden dim. The original code (M_sel_slot * w_gathered → slot_to_hidden) made
            # M_sel_hidden ~40,000× smaller than H at init (w≈1/512 uniform × std=0.02 projection),
            # rendering slot tokens invisible in cross-attention → slot_delta≈0 → zero gradient
            # to slot_output_gate, Q_sel, and slot_keys → permanent routing degeneracy.
            #
            # Fix: project at full scale. STE gradient preserved via an additive zero-forward
            # correction: (w_gathered - w_gathered.detach()) = 0 in forward, but contributes
            # M_sel_hidden.detach() to d/d(w_gathered) backward → gradient flows to Q_sel/slot_keys.
            # FIX H (2026-04-29): Differentiable soft routing proxy
            # Fix F's STE had near-zero gradient: M_sel_centered≈0 because all slots init
            # from same hidden_pool_mean → centering cancels the signal. Fix H replaces this
            # with a soft weighted-sum proxy that has O(1) non-zero gradient regardless of
            # slot content diversity.
            #
            # Forward:  uses M_sel_hard (exact hard-selected slot content, same as before)
            # Backward: gradient flows through M_sel_soft (differentiable in scores)
            #           d(loss)/d(scores[b,i]) = d(loss)/d(M_sel_soft) · slot_to_hidden(slots[b,i])
            #           This is O(1) and non-zero as long as slots have non-zero norm.

            # Hard path: exact selected slot content (no gradient through selection)
            M_sel_hidden_hard = self.slot_to_hidden(M_sel_slot)            # [B, k, d]

            # Soft proxy: differentiable weighted sum over ALL slots using softmax scores
            # scores: [B, N]  (softmax probabilities from selector)
            # Fix J-A (2026-04-29): REMOVED slots.detach(). The prior detach was from the
            # old design when hidden_to_slot was permanently frozen and couldn't receive
            # gradient. After Fix I (hidden_to_slot added to the optimizer via
            # --unfreeze_hidden_to_slot), the ONLY differentiable path from the loss back
            # to hidden_to_slot.weight goes through this einsum:
            #   loss → next_hidden → M_sel_hidden → M_sel_slot_soft → slots → O_mem_slot
            #                                                              ↑
            #                                            hidden_to_slot(O_mem_hidden)
            # Detaching `slots` severs this chain and keeps hidden_to_slot.weight.grad=None
            # even when the param is registered in the optimizer (Fix I failure mode,
            # trainable_with_grad=128/224). Do NOT reintroduce .detach() on `slots` here.
            M_sel_slot_soft = torch.einsum(
                "bn,bnd->bd",
                scores,
                slots
            )                                                               # [B, slot_dim]
            M_sel_hidden_soft = self.slot_to_hidden(
                M_sel_slot_soft.unsqueeze(1).expand(-1, k_slots, -1)
            )                                                               # [B, k(+g), d]

            # STE: forward=hard (correct slot content), backward=soft (non-zero gradient to Q_sel)
            M_sel_hidden = M_sel_hidden_hard.detach() + (M_sel_hidden_soft - M_sel_hidden_soft.detach())

            # Fix L-1 (2026-04-29): Adaptive norm clip — prevents slot_to_hidden weight growth
            # from generating M_sel_hidden vectors 20-44× above hidden_states scale, which
            # overwhelms joint attention and causes NaN spirals (root cause of fix_j_ablation
            # PPL explosion at step ~100). One-directional: only shrinks, never expands.
            # Uses hidden_states.detach() so the reference does not create extra gradient paths.
            _h_norm_ref = hidden_states.detach().norm(dim=-1).mean().clamp(min=1.0)
            _m_norms = M_sel_hidden.norm(dim=-1, keepdim=True)
            M_sel_hidden = M_sel_hidden * (_h_norm_ref / _m_norms.clamp(min=1e-6)).clamp(max=1.0)
        else:
            # disable_l1_inject=True: skip selector, slot gather, projection
            cold_start_this_call = False
            k_slots = 0
            idx = None
            scores = None
            _should_log_diag = False

        # 4. Build extended sequence + masks for the joint softmax.
        # Extended sequence layout: [L3(k_l3) | L2(k_l2) | L1(k_slots) | H(T)]
        # When L2 is None or has no prev_latents, k_l2=0 and the layout
        # collapses to the legacy [L3 | L1 | H] form.
        k_l3 = 0
        if l3_summaries is not None:
            k_l3 = l3_summaries.shape[1]

        # L2: read prev chunk's compressed latents (computed by the post-forward
        # hook in patch.py at the LAST mem layer). Reconstruct K, V into
        # model space via kv_b, then average them as a "pseudo-token" that the
        # wrapped layer's K/V projections will re-project. Double-projection is
        # wasteful but keeps the wrapped attention unchanged (Stage-2 cleanup).
        l2_tokens = None
        k_l2 = 0
        if self.l2 is not None and self.l2.prev_latents.numel() > 0:
            pl = self.l2.prev_latents  # [B, n_l2, d_c + d_h_R]
            # If L2 was reset between forwards but a stale tensor with B==0 lingers,
            # the numel() check above already guards. Now ensure batch matches.
            if pl.shape[0] == B:
                pl_content = pl[..., : self.l2.d_c]
                kv_recon = self.l2.kv_b(pl_content)             # [B, n_l2, 2*n_kv*d_head]
                K_recon, V_recon = kv_recon.chunk(2, dim=-1)
                l2_tokens = 0.5 * (K_recon + V_recon)           # [B, n_l2, d_model]
                # Cast to hidden_states dtype for downstream attention.
                l2_tokens = l2_tokens.to(hidden_states.dtype)
                k_l2 = l2_tokens.shape[1]

        # Build extended_hidden.
        parts = []
        if k_l3 > 0:
            parts.append(l3_summaries)
        if k_l2 > 0:
            parts.append(l2_tokens)
        if k_slots > 0:
            parts.append(M_sel_hidden)
        parts.append(hidden_states)
        if len(parts) == 1:
            extended_hidden = hidden_states                      # pure bypass
        else:
            extended_hidden = torch.cat(parts, dim=1)            # [B, k_l3+k_l2+k_slots+T, d]

        # Position embeddings: L3, L2, L1 all use position 0 (memory tokens are
        # position-less by design; v0 keeps L2 at position 0 — see L2 research §4.5).
        ext_pos_emb = _extend_position_embeddings(
            position_embeddings, k_l3 + k_l2 + k_slots,
        )

        # Always construct an explicit additive 4-D mask — attention_mask from
        # the outer model may be None (SDPA path's implicit causal).  Our
        # extended sequence is NOT a plain causal sequence, so we cannot rely
        # on the implicit path.
        if k_l2 > 0:
            ext_attn_mask = _build_extended_attn_mask_l2(
                k_l3=k_l3,
                k_l2=k_l2,
                k_l1=k_slots,
                T=T,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
                batch_size=B,
                swa_window=cfg.swa_window,
                mask_h_to_l1=cfg.use_decoupled_read or cfg.use_memory_xattn,
            )
        else:
            ext_attn_mask = _build_extended_attn_mask(
                k=k_slots,
                T=T,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
                batch_size=B,
                swa_window=cfg.swa_window,  # 0 = full causal (default, backward compat)
                k_l3=k_l3,
                mask_h_to_l1=cfg.use_decoupled_read or cfg.use_memory_xattn,
            )

        # H2 FIX REVERTED (2026-04-26 22:30): the earlier H2 fix
        # pre-computed a 4-D causal mask for the bypass call. Kwargs-level
        # diagnostic (`tests/test_bypass_kwargs_diagnostic.py`, ran on
        # b200-2 GPU 0) proved this *introduced* the 1.56e-02 bypass drift
        # rather than fixing it: HF 5.6.2's `LlamaModel.forward` passes
        # `attention_mask=None` to each decoder layer under SDPA
        # (via `sdpa_mask(..., allow_is_causal_skip=True)` which returns
        # None for plain causal prompts). Verdict: PRE_FIX_BYPASS_WAS_FINE
        # at both L0 and L8 → pre-fix form `attention_mask=None` matched
        # vanilla bit-exactly (err_A_max = 0.0); the 4-D mask form was the
        # regression (err_B_max = 5.86e-3 at L0, 1.56e-2 at L8).
        # References:
        #   * tests/test_bypass_kwargs_diagnostic.py (4-config unit test)
        #   * outputs/diag_kwargs/verdict_L{0,8}.json
        #   * tests/test_bypass_call_dispatch.py  (original H2 pairwise test —
        #     proved A≠B, but did NOT check which side matches vanilla)
        # The residual §5.4 probe err(L0)=1.56e-02 must come from elsewhere
        # in the wrapper (RoPE cos/sin aliasing between bypass and extended
        # forwards, or something in the probe's capture logic itself).

        # 5. Run the wrapped decoder layer twice: once on the bare hidden
        #    states (pure bypass — guaranteed to reproduce the vanilla Llama
        #    forward bit-for-bit) and once on the extended sequence with
        #    slot K/V prepended. We then combine as Flamingo-style
        #    bypass + tanh(alpha) · (extended - bypass).
        #
        #    Why two forwards: the single-concat approach suffers from
        #    phantom-logit softmax-denominator pollution — k zero slot-K
        #    vectors each contribute exp(0)=1 to the softmax denominator
        #    without contributing to the numerator, attenuating H-query
        #    attention by α(t) = S_H(t)/(k+S_H(t)) per layer and
        #    compounding 32× to 60-90 % signal loss. The output-side
        #    tanh(alpha) gate (alpha init = 0) structurally guarantees
        #    bypass parity regardless of any phantom-logit leakage.
        #    Reference: ops/research_notes/20260426_mem_space_v0_tier3_fix3_fail.md §5.
        bypass_out = self._maybe_ckpt_wrapped_layer(
            hidden_states,
            attention_mask=None,  # vanilla dispatch: HF installs SDPA is_causal=True
            position_ids=None,
            past_key_values=None,
            use_cache=False,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if isinstance(bypass_out, tuple):
            bypass_h = bypass_out[0]
        else:
            bypass_h = bypass_out

        ext_out = self._maybe_ckpt_wrapped_layer(
            extended_hidden,
            attention_mask=ext_attn_mask,
            position_ids=None,        # RoPE now driven by ext_pos_emb.
            past_key_values=None,
            use_cache=False,
            position_embeddings=ext_pos_emb,
            **kwargs,
        )
        # LlamaDecoderLayer in transformers ≥ 5.0 returns a bare Tensor.
        # Older versions returned a tuple; keep a safe unpack path.
        if isinstance(ext_out, tuple):
            ext_h = ext_out[0]
            extra = ext_out[1:]
        else:
            ext_h = ext_out
            extra = ()
        if ext_h.shape[1] != k_l3 + k_l2 + k_slots + T:
            raise RuntimeError(
                f"expected wrapped layer output length {k_l3+k_l2+k_slots+T}, "
                f"got {ext_h.shape[1]}"
            )

        # 6. Split into L3+L2 (discard), L1 memory-head (O_mem), and body.
        #    Layout: [L3(k_l3) | L2(k_l2) | L1(k_slots) | H(T)]
        #    L3 outputs are spurious (already computed externally) — discard.
        #    L2 outputs are also discarded — L2 is read-only at this layer;
        #    the post-forward hook on the LAST mem layer recomputes
        #    prev_latents from the post-stack hidden states for the next chunk.
        # P1 (2026-05-31): content-conditioned per-token injection gate.
        # g = sigmoid(inject_gate(hidden_states)), shape [B, T, 1]. Uses the
        # layer INPUT hidden_states (pre-attention representation entering this
        # memory layer) so the gate is content-dependent and causal-safe. At
        # init (weight=0, bias=-0.1523) g ≈ 0.462 == prior scalar alpha.
        # P1-v3 fix: compute gate in float32 to avoid bf16 precision loss.
        # Root cause: hidden_states norm ≈ 0.45 (post-RMSNorm), weight_max ≈ 0.006
        # → logit perturbation ≈ 0.003, below bf16 step size (0.0078) at bias=-2.0.
        # Cast both input and weight to float32 for the gate computation.
        _hs_f32 = hidden_states.float()
        _gate_logit = torch.nn.functional.linear(_hs_f32, self.inject_gate.weight.float(), self.inject_gate.bias.float())
        g = torch.sigmoid(_gate_logit).to(hidden_states.dtype)  # [B, T, 1]
        # DEBUG: print gate diagnostics on early forward calls (layer 0 only)
        if _should_log_diag and self._fwd_count <= 100:
            _gw_norm = self.inject_gate.weight.float().norm().item()
            _logit_std = _gate_logit.std().item()
            print(f"[GATE_DEBUG fwd={self._fwd_count}] weight_norm={_gw_norm:.6f} logit_std={_logit_std:.6f} g_std={g.float().std().item():.6f}")
        # v5 cold-start gating: on cold start, zero the gate so the noisy
        # initial slot content does not pollute hidden states; writeback still
        # proceeds normally below.
        if cold_start_this_call and cfg.zero_alpha_on_cold_start:
            g = torch.zeros_like(g)
        l1_start = k_l3 + k_l2
        O_mem_hidden = ext_h[:, l1_start:l1_start + k_slots, :]   # [B, k_slots, d]
        slot_delta = ext_h[:, l1_start + k_slots:, :] - bypass_h  # [B, T, d]
        # Fix M-1 (2026-04-29): clip slot_delta per-token norm to bypass_h norm scale.
        # Root cause: slot_delta_max=7.97 × alpha=0.462 × 32 layers → 117 effective residual shift.
        # Fix L-1 guards the INPUT side (M_sel_hidden). Fix M-1 guards the OUTPUT side (slot_delta).
        # One-directional: only clips DOWN (never amplifies). Same pattern as Fix L-1.
        # P1-v2: skip clipping when no_slot_delta_clip is set (stronger gate gradient).
        if not cfg.no_slot_delta_clip:
            _bypass_norms = bypass_h.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            _sd_norms = slot_delta.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            slot_delta = slot_delta * (_bypass_norms / _sd_norms).clamp(max=1.0)

        # P2 (2026-06-03): decoupled cross-attention READ contribution.
        # When use_decoupled_read=True the H→L1 prepend attention is masked off
        # above, so `slot_delta` carries only the L3/L2 contributions (≈0 when
        # neither is present). The memory READ-to-hidden signal is instead
        # produced here by a standalone cross-attention: Q=hidden_states,
        # K/V=ALL slots (decoupled from the top-k write routing), with the slots
        # getting their OWN softmax (no live-token dilution) and out_proj
        # zero-init (step-0 output = 0). Gated by the same content-conditioned
        # inject_gate g so the model can learn to suppress irrelevant reads.
        decoupled_read_out = None
        if cfg.use_decoupled_read and self.decoupled_read is not None and not cfg.disable_l1_inject:
            # slots: [B, N, slot_dim]; project to d_model for the read if needed.
            if self.slot_dim != self.d_model:
                read_slots = self.slot_to_hidden(slots)            # [B, N, d]
            else:
                read_slots = slots
            read_out, _ = self.decoupled_read.read(
                hidden_states, read_slots, read_slots,
            )                                                       # [B, T, d]
            decoupled_read_out = read_out

        # P8 (2026-06-05): dedicated memory cross-attention READ contribution.
        # When use_memory_xattn=True the H->L1 prepend is masked off above (so
        # `slot_delta` carries only L3/L2, ≈0 when neither present). The memory
        # READ-to-hidden signal is produced here by MemoryCrossAttentionRead:
        # Q=hidden, K/V=ALL slots, with its OWN softmax (no live-token dilution).
        # The blend gate is PER-HEAD + content-dependent and lives INSIDE the
        # module (init ~memory_xattn_gate_init), so — unlike the P2 decoupled
        # read — we do NOT multiply by the shared scalar inject_gate g here; the
        # read output is already gated and active at init. out_proj small-random
        # (not zero) → real gradient flows through memory from step 0.
        memory_xattn_out = None
        if cfg.use_memory_xattn and self.memory_xattn is not None and not cfg.disable_l1_inject:
            if self.slot_dim != self.d_model:
                xattn_slots = self.slot_to_hidden(slots)           # [B, N, d]
            else:
                xattn_slots = slots
            memory_xattn_out = self.memory_xattn.read(
                hidden_states, xattn_slots, xattn_slots,
            )                                                       # [B, T, d]

        # FastMem (Gated Delta Rule, 2026-05-21): continuous fast-weight
        # contribution from ALL tokens.  Runs on post-attention bypass_h so it
        # can learn what attention missed (complementary to attention, not parallel).
        # The fusion_gate inside fast_mem starts near 0, so at init this adds ≈0.
        fast_mem_out = torch.zeros_like(bypass_h)
        if self.config.use_fast_mem and hasattr(self, 'fast_mem'):
            fast_mem_out, self._fast_mem_state = self.fast_mem(
                bypass_h, self._fast_mem_state
            )

        next_hidden = bypass_h + g * slot_delta + fast_mem_out
        if decoupled_read_out is not None:
            next_hidden = next_hidden + g * decoupled_read_out
        if memory_xattn_out is not None:
            # P8: the read output is already per-head gated inside the module, so
            # it is added directly (NOT multiplied by the shared inject_gate g).
            # On cold start the slots are noisy uninitialised content; honour the
            # same zero_alpha_on_cold_start guard so we don't pollute hidden.
            if cold_start_this_call and cfg.zero_alpha_on_cold_start:
                memory_xattn_out = torch.zeros_like(memory_xattn_out)
            next_hidden = next_hidden + memory_xattn_out

        # 7. Writeback (if enabled). Branch-3 (2026-04-26): gradient-bearing
        # writeback — O_mem_slot stays attached to the autograd graph and β is
        # passed as a tensor so ``gate_param`` picks up a gradient. Combined
        # with ``config.shared_memory_bank=True`` (patch.py wires one shared
        # MemoryBank across all 32 decoder layers) this threads intra-chunk
        # BPTT through depth: layer i's write produces a new ``slots`` tensor
        # that layer i+1 reads, so "writing a good slot helps the next layer's
        # LM loss" becomes an end-to-end gradient signal.
        # Inter-chunk graph break is preserved by ``_reset_banks`` at chunk
        # boundary + init-time ``.detach()`` in ``MemoryBank.init_from_hidden``.
        # See ops/research_notes/20260426_mem_space_v0_branch3_writeback_bptt.md §3 (Option A.2).
        beta_t = self._current_beta()
        # v3 short-fix (2026-05-16): allow the trainer to suppress writeback
        # for one forward call by setting ``self._skip_writeback_this_call=True``
        # before invoking the layer.  Used when the entire sample fits in the
        # chunk window (no streaming required) so the model learns to ignore
        # the memory bank at short range.  Auto-cleared after the call below.
        _skip_wb = bool(getattr(self, "_skip_writeback_this_call", False))
        # P1 / v12 (2026-06-01): capture the gradient-bearing written slot
        # VALUES (M_write) for the summary-reconstruction loss. We take the
        # return value of the REGULAR top-k write (not the always-on global
        # slots) so the recon target reflects content-routed writes. Stays None
        # when writeback is skipped or l_recon is disabled.
        M_write: Optional[torch.Tensor] = None
        # WRITEBACK_DIAG (2026-06-04): capture the per-feature gate means for
        # the dual_gate / lowrank_gate / diag_gate modes so the diagnostic can
        # report g_in_mean / g_forget_mean. Stays None for scalar_beta (legacy
        # single-β EMA path, which has no per-feature gates).
        _wb_g_in_mean: Optional[float] = None
        _wb_g_forget_mean: Optional[float] = None
        _want_recon = (
            self._layer_idx == 0
            and cfg.l_recon_weight > 0.0
            and self.recon_decoder is not None
            and l3_summaries is not None
        )
        if cfg.enable_writeback and not cfg.disable_l1_inject and not _skip_wb:
            O_mem_slot = self.hidden_to_slot(O_mem_hidden)      # [B, k, slot_dim]
            if cfg.use_dual_gate and self.gate_proj_new is not None:
                # H6 dual-gate (LM2-inspired). Both gates are content-conditioned
                # on (new_repr, current_slot_value).
                idx_exp = idx.unsqueeze(-1).expand(-1, -1, self.slot_dim)
                # Read current slot values at selected positions for the gate
                # projection input. Detach NOT applied — gates need grad through
                # current memory to learn "this slot is stale, replace it".
                if self.memory_bank.slots is None:
                    # Should not happen post-init, but fall back to zeros if so.
                    M_prev = torch.zeros_like(O_mem_slot)
                else:
                    M_prev = self.memory_bank.slots.gather(1, idx_exp)  # [B, k, d]
                gate_logits = (
                    self.gate_proj_new(O_mem_slot)
                    + self.gate_proj_mem(M_prev)
                    + self.gate_bias  # broadcast [2d] across [B, k, 2d]
                )                                                     # [B, k, 2d]
                g_in_logit, g_forget_logit = gate_logits.chunk(2, dim=-1)
                g_in = torch.sigmoid(g_in_logit)                       # [B, k, d]
                g_forget = torch.sigmoid(g_forget_logit)               # [B, k, d]
                if _should_log_diag:
                    with torch.no_grad():
                        _wb_g_in_mean = g_in.float().mean().item()
                        _wb_g_forget_mean = g_forget.float().mean().item()
                if cfg.num_global_slots > 0 and cfg.global_slot_forget_bias != cfg.forget_bias_init:
                    _k_reg = idx.shape[1] - cfg.num_global_slots
                    _bias_delta = cfg.global_slot_forget_bias - cfg.forget_bias_init
                    g_forget_glob = torch.sigmoid(g_forget_logit[:, _k_reg:, :] + _bias_delta)
                    g_forget = torch.cat([g_forget[:, :_k_reg, :], g_forget_glob], dim=1)
                if cfg.num_global_slots > 0:
                    _k_reg = idx.shape[1] - cfg.num_global_slots
                    _idx_reg  = idx[:, :_k_reg]
                    _idx_glob = idx[:, _k_reg:]
                    M_write = self.memory_bank.write(
                        _idx_reg,
                        O_mem_slot[:, :_k_reg, :],
                        gate=g_in[:, :_k_reg, :],
                        forget_gate=g_forget[:, :_k_reg, :],
                        tanh_new=cfg.dual_gate_tanh_new,
                    )
                    if cfg.global_slot_input_gate_only:
                        # v8-C: slot ← g_in · tanh(s_new), no forget
                        _g_forget_zero = torch.zeros_like(g_in[:, _k_reg:, :])
                        self.memory_bank.write(
                            _idx_glob,
                            O_mem_slot[:, _k_reg:, :],
                            gate=g_in[:, _k_reg:, :],
                            forget_gate=_g_forget_zero,
                            tanh_new=True,
                        )
                    else:
                        # v8-A (or v7 when global_slot_forget_bias==forget_bias_init): dual gate
                        self.memory_bank.write(
                            _idx_glob,
                            O_mem_slot[:, _k_reg:, :],
                            gate=g_in[:, _k_reg:, :],
                            forget_gate=g_forget[:, _k_reg:, :],
                            tanh_new=cfg.dual_gate_tanh_new,
                        )
                else:
                    M_write = self.memory_bank.write(
                        idx,
                        O_mem_slot,
                        gate=g_in,
                        forget_gate=g_forget,
                        tanh_new=cfg.dual_gate_tanh_new,
                    )
            elif cfg.writeback_mode == "lowrank_gate" and self.lr_U is not None:
                # lowrank_gate (A): U(V_new(s_new)+V_mem(M_prev)) + bias → 2*slot_dim.
                # Same content-conditioned dual-gate semantics as dual_gate, only
                # the logit computation is low-rank. dolmino uses num_global_slots==0.
                idx_exp = idx.unsqueeze(-1).expand(-1, -1, self.slot_dim)
                if self.memory_bank.slots is None:
                    M_prev = torch.zeros_like(O_mem_slot)
                else:
                    M_prev = self.memory_bank.slots.gather(1, idx_exp)  # [B, k, d]
                _z = self.lr_V_new(O_mem_slot) + self.lr_V_mem(M_prev)  # [B, k, r]
                gate_logits = self.lr_U(_z) + self.lr_gate_bias          # [B, k, 2d]
                g_in_logit, g_forget_logit = gate_logits.chunk(2, dim=-1)
                g_in = torch.sigmoid(g_in_logit)
                g_forget = torch.sigmoid(g_forget_logit)
                if _should_log_diag:
                    with torch.no_grad():
                        _wb_g_in_mean = g_in.float().mean().item()
                        _wb_g_forget_mean = g_forget.float().mean().item()
                M_write = self.memory_bank.write(
                    idx,
                    O_mem_slot,
                    gate=g_in,
                    forget_gate=g_forget,
                    tanh_new=cfg.dual_gate_tanh_new,
                )
            elif cfg.writeback_mode == "diag_gate" and self.diag_a_in is not None:
                # diag_gate (B): per-feature diagonal gate.
                #   g_in_logit     = a_in*s_new + c_in*M_prev + b_in
                #   g_forget_logit = a_f *s_new + c_f *M_prev + b_f
                # dolmino uses num_global_slots==0.
                idx_exp = idx.unsqueeze(-1).expand(-1, -1, self.slot_dim)
                if self.memory_bank.slots is None:
                    M_prev = torch.zeros_like(O_mem_slot)
                else:
                    M_prev = self.memory_bank.slots.gather(1, idx_exp)  # [B, k, d]
                g_in_logit = self.diag_a_in * O_mem_slot + self.diag_c_in * M_prev + self.diag_b_in
                g_forget_logit = self.diag_a_f * O_mem_slot + self.diag_c_f * M_prev + self.diag_b_f
                g_in = torch.sigmoid(g_in_logit)
                g_forget = torch.sigmoid(g_forget_logit)
                if _should_log_diag:
                    with torch.no_grad():
                        _wb_g_in_mean = g_in.float().mean().item()
                        _wb_g_forget_mean = g_forget.float().mean().item()
                M_write = self.memory_bank.write(
                    idx,
                    O_mem_slot,
                    gate=g_in,
                    forget_gate=g_forget,
                    tanh_new=cfg.dual_gate_tanh_new,
                )
            else:
                # Legacy single-gate path (H/H5/H3).
                # v6/v7 (2026-05-18): choose writeback mode based on config.
                if cfg.num_global_slots > 0:
                    # v7: split idx into regular top-k slots (EMA) and global
                    # always-on slots (replacement).
                    _k_reg = idx.shape[1] - cfg.num_global_slots
                    _idx_reg  = idx[:, :_k_reg]
                    _idx_glob = idx[:, _k_reg:]
                    _O_reg  = O_mem_slot[:, :_k_reg, :]
                    _O_glob = O_mem_slot[:, _k_reg:, :]
                    M_write = self.memory_bank.write(_idx_reg, _O_reg, beta_t)
                    self.memory_bank.write(_idx_glob, _O_glob, beta_t, replace=True)
                elif cfg.use_replace_writeback:
                    # v6: direct replacement for ALL selected slots.
                    M_write = self.memory_bank.write(idx, O_mem_slot, beta_t, replace=True)
                else:
                    # Default EMA path.
                    M_write = self.memory_bank.write(idx, O_mem_slot, beta_t)

        # ---- WRITEBACK_DIAG (diagnostic log, no-op on computation) ----
        # Emit every 200 forward calls, rank-0 / layer-0 only.
        if _should_log_diag:
            with torch.no_grad():
                _gate_val = beta_t.float().item() if isinstance(beta_t, torch.Tensor) else float(beta_t)
                _g_mean = g.float().mean().item()
                _g_std = g.float().std().item()
                _sd_abs_mean = slot_delta.float().abs().mean().item()
                _sd_abs_max  = slot_delta.float().abs().max().item()
                _msh_norm_mean = M_sel_hidden.float().norm(dim=-1).mean().item()
            print(
                f"[WRITEBACK_DIAG step={self.step_counter} fwd={self._fwd_count}]"
                f" gate_val(beta)={_gate_val:.6f}"
                f" alpha(inject_gate_mean)={_g_mean:.6f}"
                f" inject_gate_std={_g_std:.6f}"
                f" slot_delta_abs_mean={_sd_abs_mean:.6f}"
                f" slot_delta_max={_sd_abs_max:.6f}"
                f" M_sel_hidden_norm_mean={_msh_norm_mean:.6f}"
                f" wb_mode={cfg.writeback_mode}"
                f" g_in_mean={_wb_g_in_mean if _wb_g_in_mean is not None else float('nan'):.6f}"
                f" g_forget_mean={_wb_g_forget_mean if _wb_g_forget_mean is not None else float('nan'):.6f}"
                f" step_counter={self.step_counter}",
                flush=True,
            )
        # -----------------------------------------------------------

        # ---- SKRL_DIAG removed in Fix Z.1 ----
        # slot_keys are frozen, no SKRL to monitor

        # 8. Stash side-channel outputs.
        aux: Dict[str, torch.Tensor] = {}
        if cfg.return_aux_losses and not cfg.disable_l1_inject:
            lb = self.selector.load_balance_loss(scores, idx)
            aux["load_balance"] = lb * cfg.load_balance_weight
            ent = self.selector.entropy_aux_loss(scores)
            aux["entropy"] = ent * cfg.entropy_aux_weight
            # Fix Z.2g: key repulsion prevents collapse
            kr = self.selector.key_repulsion_loss(threshold=cfg.key_repulsion_threshold, slots=slots)
            aux["key_repulsion"] = kr * cfg.key_repulsion_weight
            # K_sel weight orthogonality: prevents weight matrix rank collapse
            # which causes all projected keys to be identical (key_max_cos=1.0).
            wo = self.selector.weight_ortho_loss()
            aux["weight_ortho"] = wo * cfg.key_repulsion_weight
            # v9 (2026-06-01): L3 summary-token diversity. l3_pool is a shared
            # singleton across all 32 layers, so collect this ONCE (layer_idx==0)
            # to avoid counting it 32×. Only when L3 summaries actually exist
            # (None on the cold-start first chunk).
            #
            # v10 (2026-06-01): the v9 l3_diversity loss acts on S (pre-Q_sel,
            # 4096-dim), but routing actually uses q_multi (post-Q_sel + LN,
            # 128-dim). Empirically S could be diverse (S_max_cos≈0.6) while
            # q_multi was collapsed (summary_q_max_cos=1.0) → the S-space loss
            # was blind to the real collapse. We therefore add a SECOND diversity
            # term acting directly on q_multi (computed in selector.forward,
            # gradient flows to Q_sel + q_sel_ln). DECISION: keep BOTH —
            #   * S-loss (v9): harmless, keeps the L3 Q-Former output diverse so
            #     the projection has diverse inputs to work with;
            #   * q_multi-loss (v10): the load-bearing term, since it constrains
            #     the exact space routing uses.
            # Both share l3_diversity_weight. The q_multi term is collected on
            # layer_idx==0 too (the selector is per-layer, but only layer 0 is
            # the canonical diagnostic/aux layer here, matching the v9 guard;
            # this keeps the loss magnitude comparable to v9 and avoids 32×).
            if (
                self._layer_idx == 0
                and self.l3_pool is not None
                and l3_summaries is not None
            ):
                l3_div = self.l3_pool.query_diversity_loss(
                    l3_summaries, threshold=cfg.l3_diversity_threshold
                )
                aux["l3_diversity"] = l3_div * cfg.l3_diversity_weight
                # v10 post-projection diversity on the actual routing query.
                _q_div = getattr(self.selector, "_last_q_multi_diversity_loss", None)
                if _q_div is not None:
                    aux["q_multi_diversity"] = _q_div * cfg.l3_diversity_weight
            # P1 / v12 (2026-06-01): summary-reconstruction auxiliary loss.
            # Reconstruct this chunk's L3 summary tokens from the slot VALUES
            # written this chunk (M_write); MSE against stopgrad(S_L3) gives the
            # write path a near-distance "store decodable content" objective.
            # Computed only on layer_idx==0 (recon_decoder is a shared singleton,
            # avoid 32×). M_write is None on cold-start / writeback-skip / β≈0.
            if _want_recon and M_write is not None:
                S_hat = self.recon_decoder(M_write)              # [B, num_summary, d]
                l_recon = self.recon_decoder.recon_loss(S_hat, l3_summaries)
                aux["recon"] = l_recon * cfg.l_recon_weight
            aux["beta"] = beta_t
            # Per-slot usage (fraction of batch that selected each slot).
            one_hot = torch.zeros_like(scores).scatter_(-1, idx, 1.0)
            aux["slot_usage"] = one_hot.float().mean(dim=0).detach()
        self.last_aux_losses = aux
        self.last_idx = idx.detach() if idx is not None else None
        self.last_scores = scores.detach() if scores is not None else None

        if extra:
            return (next_hidden, *extra)
        return next_hidden
