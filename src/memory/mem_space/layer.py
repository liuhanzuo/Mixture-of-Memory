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
import os
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
    k_ev: int = 0,
    isolate_ev: bool = False,
) -> torch.Tensor:
    """Return [B, 1, L, L] additive mask for the joint-attn extended seq.

    The extended sequence layout is: [L3(k_l3) | L1(k) | EV(k_ev) | H(T)].
    Total length L = k_l3 + k + k_ev + T.

    Convention: 0 means "allowed", ``-inf`` means "masked out".

    Attention pattern:
        * L3 rows (0..k_l3-1): attend to everything (full row of zeros).
        * L1 rows: attend to everything (full row of zeros).
        * EV (evidence) rows: attend to everything (full row of zeros).
        * H rows:
          - L3 / L1 / EV keys: always allowed.
          - H keys: causal (or SWA-causal if swa_window>0).

    When k_l3 == 0 and k_ev == 0: behaviour is IDENTICAL to the pre-L3/-EV
    implementation.
    """
    prefix = k_l3 + k + k_ev
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
        # L3/L1/EV-queries and H-queries→L3/L1/EV keys: already 0 (allowed).

    # P2 decoupled-read (2026-06-03): when mask_h_to_l1=True, BLOCK H-queries
    # from attending to the L1 slot block (cols k_l3..k_l3+k-1). This removes
    # the "injection dilution" — live tokens no longer share their softmax with
    # the prepended slot KV. L1-queries→H attention is KEPT (so the writeback's
    # O_mem_hidden is computed exactly as before); only the H→L1 read direction
    # is severed. The memory READ contribution is instead produced by the
    # standalone CrossAttentionMemoryV2.read path in forward().
    # NOTE (evidence): we sever ONLY the L1 block; the evidence block (cols
    # k_l3+k .. k_l3+k+k_ev-1) stays allowed so H can read the precise tokens.
    if mask_h_to_l1 and k > 0 and T > 0:
        mask[prefix:, k_l3:k_l3 + k] = neg_inf

    # Landmark EV-isolation (2026-06-17): restrict H's prefix softmax to the EV
    # block only by severing H→{L3, L1}. EV cols are [k_l3+k .. k_l3+k+k_ev-1];
    # everything before EV in the prefix (L3 cols 0..k_l3-1, L1 cols k_l3..k_l3+k-1)
    # is masked for H rows, so the precise evidence tokens own the prefix softmax
    # denominator (vs the compressed L3/L1 prefix). No-op when k_ev==0.
    if isolate_ev and k_ev > 0 and T > 0:
        ev_start = k_l3 + k
        mask[prefix:, 0:ev_start] = neg_inf

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
    k_ev: int = 0,
    isolate_ev: bool = False,
) -> torch.Tensor:
    """Return [B, 1, L, L] additive mask for the L2-extended joint-attn seq.

    Extended layout: [L3(k_l3) | L2(k_l2) | L1(k_l1) | EV(k_ev) | H(T)].
    Total length L = k_l3 + k_l2 + k_l1 + k_ev + T.

    Attention pattern:
        * L3 / L2 / L1 / EV rows: attend to everything.
        * H rows:
          - cols 0..prefix-1 (L3, L2, L1, EV keys): always allowed.
          - cols prefix..L-1 (H keys): causal (or SWA-causal if swa_window>0).

    When k_l2 == 0 and k_ev == 0: collapses to the [L3 | L1 | H] layout (same as
    ``_build_extended_attn_mask`` with the same k_l3/k_l1).
    """
    prefix = k_l3 + k_l2 + k_l1 + k_ev
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

    # P2 decoupled-read: sever H→L1 attention (cols k_l3+k_l2 .. k_l3+k_l2+k_l1-1).
    # The evidence block (immediately after L1) is NOT severed — H must read it.
    if mask_h_to_l1 and k_l1 > 0 and T > 0:
        mask[prefix:, k_l3 + k_l2:k_l3 + k_l2 + k_l1] = neg_inf

    # Landmark EV-isolation: restrict H's prefix softmax to the EV block only by
    # severing H→{L3, L2, L1}. EV cols start at k_l3+k_l2+k_l1. No-op when k_ev==0.
    if isolate_ev and k_ev > 0 and T > 0:
        ev_start = k_l3 + k_l2 + k_l1
        mask[prefix:, 0:ev_start] = neg_inf

    return mask.view(1, 1, L, L).expand(batch_size, 1, L, L).contiguous()


def _extend_position_embeddings(
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    k: int,
    ev_pos: Optional[torch.Tensor] = None,
    k_pos0: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepend k prefix entries to the (cos, sin) rotary tables.

    Each of cos, sin has shape ``[1 or B, T, head_dim]``.  Result has shape
    ``[*, k+T, head_dim]``.

    Legacy behaviour (``ev_pos is None``): all k prefix entries reuse the
    position-0 rotary phase (memory tokens treated as position-less).

    Landmark position fix (2026-06-17, ``ev_pos`` given): the EV (evidence)
    sub-block of the prefix is injected at its REAL source RoPE phase instead
    of position 0, so the frozen decoder re-projects the evidence K/V at the
    position it actually saw the token (each streaming chunk resets positions
    to 0, so the source position == the token's in-chunk offset). Layout of the
    k prefix entries: ``[ k_pos0 position-0 entries (L3|L2|L1) | EV at ev_pos ]``.

    Args:
        ev_pos: ``[B, k_ev]`` or ``[k_ev]`` long — source position of each EV
            token. Clamped to ``[0, T-1]`` and used to index the current chunk's
            cos/sin table (positions beyond the current window fall back to the
            last available phase).
        k_pos0: number of leading prefix entries that stay at position 0
            (k_l3 + k_l2 + k_slots). ``k_ev = k - k_pos0``.
    """
    cos, sin = position_embeddings
    # Handle both [1, T, D] and [B, T, D] layouts.
    if cos.dim() != 3 or sin.dim() != 3:
        raise ValueError(
            f"position_embeddings must be 3-D tensors; got cos={tuple(cos.shape)}, "
            f"sin={tuple(sin.shape)}"
        )
    if ev_pos is None or k_pos0 is None:
        # Legacy pos-0 prefix (default path; byte-identical to pre-fix).
        cos0 = cos[:, :1, :]                                       # [*, 1, D]
        sin0 = sin[:, :1, :]
        cos_ext = torch.cat([cos0.expand(cos.shape[0], k, cos.shape[-1]), cos], dim=1)
        sin_ext = torch.cat([sin0.expand(sin.shape[0], k, sin.shape[-1]), sin], dim=1)
        return cos_ext, sin_ext

    # Real-position EV injection.
    Bc, T, D = cos.shape
    k_ev = k - k_pos0
    # ev_pos -> [Bc, k_ev] long, clamped into the available rotary window.
    _ep = ev_pos
    if _ep.dim() == 1:
        _ep = _ep.unsqueeze(0)
    _ep = _ep.to(device=cos.device, dtype=torch.long).clamp_(0, T - 1)
    if _ep.shape[0] != Bc:
        # Broadcast a single-row ev_pos across the cos batch (or vice versa).
        if _ep.shape[0] == 1:
            _ep = _ep.expand(Bc, -1)
        elif Bc == 1:
            cos = cos.expand(_ep.shape[0], T, D)
            sin = sin.expand(_ep.shape[0], T, D)
            Bc = _ep.shape[0]
        else:
            # Shape mismatch we can't reconcile — fall back to pos-0 for EV.
            _ep = torch.zeros(Bc, k_ev, device=cos.device, dtype=torch.long)
    _gather_idx = _ep.unsqueeze(-1).expand(Bc, k_ev, D)            # [Bc, k_ev, D]
    cos_ev = cos.gather(1, _gather_idx)                            # [Bc, k_ev, D]
    sin_ev = sin.gather(1, _gather_idx)
    cos0 = cos[:, :1, :]
    sin0 = sin[:, :1, :]
    cos_ext = torch.cat(
        [cos0.expand(Bc, k_pos0, D), cos_ev, cos], dim=1
    )                                                              # [Bc, k+T, D]
    sin_ext = torch.cat(
        [sin0.expand(Bc, k_pos0, D), sin_ev, sin], dim=1
    )
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
        gist_readout: Optional[nn.Module] = None,
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
                evidence_buffer_size=(
                    config.evidence_buffer_size if config.use_slot_evidence else 0
                ),
                evidence_dim=d_model if config.use_slot_evidence else None,
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

        # Raw-KV READOUT — Method A (2026-06-19). The trainable gist scorer is a
        # shared singleton (registered once on the model root by patch.py, like
        # l3_pool). Registered via object.__setattr__ so its params are NOT
        # duplicated in every layer's state_dict. None when use_rawkv_readout
        # is off. The per-sequence raw-KV readout STORE is created/owned on the
        # shared MemoryBank (reuses the bank's reset/detach lifecycle).
        if gist_readout is not None:
            object.__setattr__(self, "gist_readout", gist_readout)
        else:
            object.__setattr__(self, "gist_readout", None)
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
        # Dead-slot-binding fix: independent learnable slot key. When True, the
        # selector uses a pure learnable per-slot key (slot_key_param) instead of
        # the content-derived K_sel(slots)+slot_key_bias. Default False →
        # byte-identical to content-based routing.
        self.selector._independent_slot_key = getattr(
            config, "independent_slot_key", False
        )
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
        # P11 (2026-06-06): delta-rule writeback + normalized readout. Both
        # default off → byte-identical to pre-P11. delta_rule switches the gated
        # writeback to a residual (slot + g·(new−slot)) update; normalize_readout
        # rescales M_sel_hidden to the local hidden-state magnitude before
        # injection. Read in forward() / passed to memory_bank.write().
        self._use_delta_rule_writeback = getattr(
            config, "use_delta_rule_writeback", False
        )
        self._normalize_readout = getattr(config, "normalize_readout", False)
        self._readout_norm_scale = getattr(config, "readout_norm_scale", 1.0)
        # DeltaNet-style erase-then-write. When True, the gated writeback applies
        # the associative erase-then-write update instead of EMA / delta-rule.
        # Default False → byte-identical to the existing write paths.
        self._delta_erase_write = getattr(config, "delta_erase_write", False)

        # EXP-R1 (2026-06-11): dead-slot recycling knobs. All default off
        # (interval<=0) → byte-identical to P11.
        self._dead_slot_reset_interval = int(
            getattr(config, "dead_slot_reset_interval", 0)
        )
        self._dead_slot_reset_mode = getattr(
            config, "dead_slot_reset_mode", "strided_current"
        )
        self._dead_slot_grace_chunks = int(
            getattr(config, "dead_slot_grace_chunks", 1)
        )
        # EXP-R1c (2026-06-11): dead-slot judge. "window" (default) = R1
        # window-scoped `_recycle_usage==0`; "cumulative" = sample-scoped
        # `_cum_usage==0` (never erases a long-range memory slot).
        self._dead_slot_criterion = getattr(
            config, "dead_slot_criterion", "window"
        )

        # v20 (2026-06-12): read-based slot lifecycle knobs. All default OFF
        # (rate<=0 / mode=="off") → byte-identical to P11.
        #   Arm A — soft read-decay: every _slot_read_decay_interval chunks,
        #     multiply each slot's value by a factor in [min_keep, 1] driven by
        #     its recent read fraction (unread slots fade, never hard-deleted).
        #   Arm B — readmass eviction: at each recycle boundary, evict the
        #     coldest (lowest _cum_read_mass) slots that have stayed cold for
        #     >= _slot_evict_protect_chunks boundaries, capped at max_frac·N.
        self._slot_read_decay_rate = float(
            getattr(config, "slot_read_decay_rate", 0.0)
        )
        self._slot_read_decay_interval = int(
            getattr(config, "slot_read_decay_interval", 8)
        )
        self._slot_read_decay_min_keep = float(
            getattr(config, "slot_read_decay_min_keep", 0.5)
        )
        self._slot_evict_mode = getattr(config, "slot_evict_mode", "off")
        self._slot_evict_max_frac = float(
            getattr(config, "slot_evict_max_frac", 0.1)
        )
        self._slot_evict_floor = float(getattr(config, "slot_evict_floor", 0.0))
        self._slot_evict_protect_chunks = int(
            getattr(config, "slot_evict_protect_chunks", 2)
        )

        # EXP-W2 (2026-06-11): dense all-slot soft delta-write knobs. Default
        # off (weight<=0) → byte-identical to P11. When >0, every chunk applies
        # a weak per-slot-gated delta nudge to ALL N slots (see config.py +
        # MemoryBank.soft_write). The content module is built below (it needs
        # d_model / n_heads). Orthogonal to EXP-R1; the soft-write block in
        # forward() is gated by soft_write_weight>0 and lives OUTSIDE the
        # _do_recycle block, so the two switches compose independently.
        self._soft_write_weight = float(
            getattr(config, "soft_write_weight", 0.0)
        )
        self._soft_write_content = getattr(
            config, "soft_write_content", "slot_query"
        )

        # FIFO hidden-state memory (方案B, 2026-06-24).
        # When use_fifo_memory=True, ALL slot routing is bypassed and the layer
        # maintains a rolling FIFO buffer of past chunk hidden states. The
        # current chunk attends these via full causal attention (no routing, no
        # slot projection) — MemoryLLM write-direct / read-full style.
        self._use_fifo_memory = getattr(config, "use_fifo_memory", False)
        self._fifo_buffer_chunks = int(getattr(config, "fifo_buffer_chunks", 50))
        self._fifo_detach = bool(getattr(config, "fifo_detach", True))
        # The FIFO buffer is a list of [B, T_c, d] tensors (one per past chunk).
        # Managed as a plain attribute (not registered as a parameter/buffer) so
        # it does not appear in state_dict or interfere with DDP sync.
        self._fifo_buf: list = []

        # FIFO eval-time probes (2026-06-25, H_POS + H_DIL falsification).
        # All default to None / False → _forward_fifo path is byte-identical to
        # pre-probe behaviour. Set via run_babilong_mem_space.py CLI helpers.
        #   _fifo_pos_mode: None | "packed" | "real"
        #     None  -> legacy all-pos-0 prefix (current behaviour)
        #     "packed" -> in-distribution positions = keep_idx_in_set * chunk + i
        #     "real"   -> original chunk_idx * chunk + i (may be OOD; relies on
        #                 Llama-3 RoPE theta=500000 extrapolation)
        #   _fifo_keep_set_mode: None | "flat_readerattn"
        #     None -> attend ALL buffered chunks (legacy)
        #     "flat_readerattn" -> reader q.k top-K_keep + last R recency floor
        #   _fifo_keep_topk:    int top-K for keep-set selection (default 25)
        #   _fifo_keep_recency: int last-R recency floor (default 2)
        #   _fifo_keep_all_buffer: bool — when True, eviction is suppressed so
        #     buffer keeps the full history (only useful with keep_set_mode set).
        self._fifo_pos_mode: Optional[str] = None
        self._fifo_keep_set_mode: Optional[str] = None
        self._fifo_keep_topk: int = 25
        self._fifo_keep_recency: int = 2
        self._fifo_keep_all_buffer: bool = False
        # ORACLE keep-set probe (2026-06-25, decisive perfect-isolation test).
        # When _fifo_keep_set_mode == "oracle", keep ONLY the buffered chunk(s)
        # whose ORIGINAL (document-absolute) chunk index is in
        # ``_fifo_oracle_needle_chunks`` (a set of 0-based absolute chunk indices
        # set per-sample by the eval harness, mirroring the rawkv oracle's
        # cfg.rawkv_oracle_needle_chunk channel) plus the recency floor. To map a
        # buffer entry → its document-absolute chunk index under eviction we keep
        # a parallel list ``_fifo_buf_abs_idx`` (entry i was the abs-idx-th chunk
        # appended). The bookkeeping is maintained ONLY in oracle mode, so the
        # None / "flat_readerattn" paths are byte-identical. None / empty needle
        # set, or a needle that has been EVICTED from the buffer → fall back to
        # keep-all (counted in _fifo_oracle_fallback_count) rather than crash.
        self._fifo_oracle_needle_chunks: Optional[set] = None
        self._fifo_buf_abs_idx: list = []          # parallel to _fifo_buf (oracle only)
        self._fifo_write_seq: int = 0              # monotonic abs-chunk counter (reset per doc)
        self._fifo_oracle_fallback_count: int = 0  # samples/forwards that kept-all
        self._fifo_oracle_evicted_count: int = 0   # subset: needle was evicted

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
                disable_null_sink=config.memory_xattn_disable_null_sink,
                learnable_mass_bias=config.use_learnable_mass_bias,
                learnable_mass_bias_init=config.learnable_mass_bias_init,
                use_shared_addressing=config.use_shared_addressing,
                selector_dim=config.selector_dim,
                address_temperature=config.selector_temperature,
            )
        else:
            self.memory_xattn = None

        # EXP-W2 (2026-06-11): dense soft-write content module. When
        # soft_write_weight>0 (and soft_write_content=="slot_query"), build a
        # slots-as-query write-attention that produces per-slot DISTINCT content
        # from the current chunk tokens (anti-homogenisation; see
        # selector.SoftWriteContent). Default off → no module, no params, no
        # behaviour change (byte-identical to P11). n_heads defaults like the
        # other xattn modules when not threaded by the model's GQA head counts.
        if self._soft_write_weight > 0.0 and self._soft_write_content == "slot_query":
            from .selector import SoftWriteContent
            _swc_heads = n_heads if n_heads is not None else 8
            self.soft_write_content_mod = SoftWriteContent(
                d_model=d_model,
                slot_dim=slot_dim,
                n_heads=_swc_heads,
                out_proj_std=0.02,
                dropout=0.0,
            )
        else:
            self.soft_write_content_mod = None
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

        # Slot-Routed Evidence Memory (2026-06-17): resolve evidence_topr (0 →
        # full buffer) and clamp to the buffer size. This layer owns the
        # evidence read/write only when its index == config.evidence_layer.
        _ev_topr = config.evidence_topr if config.evidence_topr > 0 else config.evidence_buffer_size
        self._evidence_topr = min(int(_ev_topr), int(config.evidence_buffer_size))
        self._is_evidence_layer = (
            config.use_slot_evidence and self._layer_idx == config.evidence_layer
        )

        # Parallel raw-KV retrieval channel (2026-06-18). This layer owns the
        # raw-KV store's append + retrieve only when its index == rawkv_layer
        # (single shared bank → one owner, mirroring the evidence layer). Default
        # off → _is_rawkv_layer is False on every layer (byte-identical to pre).
        self._is_rawkv_layer = (
            config.use_rawkv_retrieval and self._layer_idx == config.rawkv_layer
        )

        # TRUE in-attention K/V concat channel (2026-06-18). This layer owns the
        # raw-KV store write + the in-attention injection only when its index ==
        # inattn_kv_layer. Install the injection-aware wrapper on the wrapped
        # decoder layer's self-attention NOW (idempotent; defaults to a no-op
        # byte-identical pass until the layer stashes retrieved KV per-forward).
        # Default off → never installed → byte-identical to pre.
        #
        # Multi-layer injection (v2, 2026-06-19): when config.inattn_kv_layers is
        # a non-empty list, the READ/INJECT is owned by EVERY layer in the list
        # (each re-projects retrieved hidden through its own k/v_proj). The store
        # WRITE stays owned by exactly ONE layer (smallest index in the list) so
        # the shared raw-KV store is appended once per chunk. Empty/None → the
        # single inattn_kv_layer owns both write + read (byte-identical to v1).
        _inattn_layers = list(getattr(config, "inattn_kv_layers", None) or [])
        if config.use_inattn_kv and _inattn_layers:
            self._is_inattn_kv_layer = self._layer_idx in _inattn_layers
            self._is_inattn_write_owner = (
                self._layer_idx == min(_inattn_layers)
            )
        else:
            self._is_inattn_kv_layer = (
                config.use_inattn_kv and self._layer_idx == config.inattn_kv_layer
            )
            self._is_inattn_write_owner = self._is_inattn_kv_layer
        if self._is_inattn_kv_layer:
            from .inattn_kv import install_inattn_wrapper
            _attn = getattr(self.wrapped_layer, "self_attn", None)
            if _attn is not None:
                install_inattn_wrapper(_attn)

        # Raw-KV READOUT — Method A (2026-06-19). Layer ownership mirrors the
        # inattn_kv channel: the READ/INJECT is owned by every layer in
        # config.rawkv_readout_layers (or the single rawkv_readout_layer); the
        # per-chunk STORE WRITE (raw tokens + gist source) is owned by exactly
        # ONE layer (the smallest read index) so the shared store is populated
        # once per chunk. We reuse the SAME inattn self-attn wrapper (it accepts
        # the 3-tuple stash with a per-column gist log-bias). Default off →
        # _is_rawkv_readout_layer is False everywhere (byte-identical to pre).
        _ro_layers = list(getattr(config, "rawkv_readout_layers", None) or [])
        if config.use_rawkv_readout and _ro_layers:
            self._is_rawkv_readout_layer = self._layer_idx in _ro_layers
            self._is_rawkv_readout_write_owner = (
                self._layer_idx == min(_ro_layers)
            )
        else:
            self._is_rawkv_readout_layer = (
                config.use_rawkv_readout
                and self._layer_idx == config.rawkv_readout_layer
            )
            self._is_rawkv_readout_write_owner = self._is_rawkv_readout_layer
        if self._is_rawkv_readout_layer:
            from .inattn_kv import install_inattn_wrapper
            _attn = getattr(self.wrapped_layer, "self_attn", None)
            if _attn is not None:
                install_inattn_wrapper(_attn)
                # (B) two-stage grouped-softmax readout flags (2026-06-20). When
                # rawkv_grouped_readout is on, inattn_kv replaces the flat softmax
                # over [native ; retrieved] with a hierarchical block-select x
                # within-block softmax (sub-block size = rawkv_subblock_size).
                _attn._rawkv_grouped_readout = bool(
                    getattr(config, "rawkv_grouped_readout", False)
                )
                _attn._rawkv_subblock_size = int(
                    getattr(config, "rawkv_subblock_size", 64)
                )
                _attn._rawkv_stage1_select = bool(
                    getattr(config, "rawkv_stage1_select", False)
                )
                # (B in-window summary, 2026-06-20) selection-side: when on,
                # the current chunk's self-attn becomes an in-window bottleneck
                # (later tokens reach earlier sub-blocks only via summary key).
                _attn._rawkv_inwindow_summary = bool(
                    getattr(config, "rawkv_inwindow_summary", False)
                )

        # Per-slot raw-KV cache (2026-06-22). This is NOT Method A: retrieval is
        # coupled to the normal slot selector. A single owner layer appends every
        # chunk's raw hidden states under the selected slot ids, and the same layer
        # injects all raw hidden cached under the CURRENT selected slots through
        # the existing in-attention K/V concat wrapper. No capacity / eviction in
        # v1; this is an upper-bound test.
        self._is_slot_kv_cache_layer = (
            bool(getattr(config, "use_slot_kv_cache", False))
            and self._layer_idx == int(getattr(config, "slot_kv_cache_layer", 16))
        )
        if self._is_slot_kv_cache_layer:
            from .inattn_kv import install_inattn_wrapper
            _attn = getattr(self.wrapped_layer, "self_attn", None)
            if _attn is not None:
                install_inattn_wrapper(_attn)

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
        # EXP-D3 (2026-06-11): mean pairwise cosine of slot CONTENT (off-diag),
        # the direct homogenisation signal for any broad-write method (W1/W2/R1).
        # Refreshed in QUERY_DIAG; ->1 == all slots' content collapsed onto one
        # direction (the failure mode a dense soft-write risks). Default 0.0.
        self._last_slot_content_cos: float = 0.0

        # EXP-R1 / EXP-D2 (2026-06-11): per-sample dead-slot recycling state.
        # All layer-0 only (the recycler drives the SHARED bank from one layer
        # so 32 layers don't fight; matches the layer-0 usage-histogram pattern).
        # Lazily materialised [B, N] on device at first use; re-allocated on
        # batch-size change; reset at sample/document cold start.
        #   _recycle_usage : selections per slot since the last reset window
        #                    began (window-scoped; zeroed after each reset event)
        #                    → dead = (_recycle_usage == 0) at a reset boundary.
        #   _cum_usage     : selections per slot over the WHOLE sample (never
        #                    window-reset) → EXP-D2 cumulative dead-slot frac.
        #   _recycle_chunk_count : chunks processed since cold start (layer-0
        #                    forward count within the current sample).
        #   _recycle_grace_mask  : [B, N] bool — slots in the grace window that
        #                    must be force-written this/next chunk(s).
        #   _recycle_grace_remaining : int — grace chunks left.
        # Defaults make every path a no-op when dead_slot_reset_interval <= 0.
        self._recycle_usage: Optional[torch.Tensor] = None
        self._cum_usage: Optional[torch.Tensor] = None
        self._recycle_chunk_count: int = 0
        self._recycle_grace_mask: Optional[torch.Tensor] = None
        self._recycle_grace_remaining: int = 0
        # v20 (2026-06-12): read-based slot lifecycle accumulators (layer-0,
        # no_grad telemetry; same alloc/cold-start lifecycle as _cum_usage).
        #   _cum_read_mass    : [B, N] float — read-path softmax mass each slot
        #                       has EVER drawn over the whole sample (the
        #                       correct liveness measure: read≠write). Drives
        #                       Arm B eviction (lowest cum read-mass evicted).
        #   _recent_read_mass : [B, N] float — read mass over the CURRENT window
        #                       (zeroed at each Arm A decay / Arm B reset
        #                       boundary, mirroring _recycle_usage). Drives Arm A
        #                       decay (recent-read fraction) + Arm B cold judge.
        #   _evict_cold_streak: [B, N] long — consecutive reset boundaries a slot
        #                       has stayed cold (Arm B protection window).
        # All default to a no-op when both arms are off → telemetry only.
        self._cum_read_mass: Optional[torch.Tensor] = None
        self._recent_read_mass: Optional[torch.Tensor] = None
        self._evict_cold_streak: Optional[torch.Tensor] = None
        self._read_decay_chunk_count: int = 0
        # Arm A/B latest scalars (layer-0; pushed to QUERY_DIAG).
        self._last_cum_read_mass_cov: float = -1.0
        self._last_n_evicted_readmass: int = -1
        # EXP-D2 latest scalars (layer-0; pushed to wandb by the loop).
        self._last_dead_slot_frac: float = 0.0
        self._last_max_slot_select_count: float = 0.0
        # Count of recycle reset events this sample (diagnostic).
        self._last_recycle_resets: int = 0
        # EXP-R1c (2026-06-11): #slots judged dead at the LAST reset boundary
        # (sums over batch). Lets us compare window vs cumulative judges.
        self._last_n_recycled: int = 0

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
    # EXP-R1 (2026-06-11): dead-slot recycling helpers
    # --------------------------------------------------------------------- #

    def _select_diverse_strided_content(
        self,
        H_slot: torch.Tensor,
        slots: torch.Tensor,
        dead_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build replacement content for dead slots from CURRENT-chunk tokens.

        For each sample we take a strided pool of candidate tokens from the
        current chunk (in slot_dim space, `H_slot = hidden_to_slot(hidden)` or
        hidden when slot_dim==d_model), score each candidate by its MAXIMUM
        cosine similarity to the LIVE slots, and greedily assign the
        LEAST-similar (most diverse) candidates to the dead slots. This occupies
        the content region the live slots do NOT cover, instead of duplicating
        them — the explicit "diversity maximisation" the design asks for, and
        the protection against the pooled-mean homogenisation that would
        collapse top1_sim.

        Args:
            H_slot:    [B, T, slot_dim] current chunk hidden states in slot space.
            slots:     [B, N, slot_dim] current slot content.
            dead_mask: [B, N] bool, True for dead slots.

        Returns:
            new_content: [B, N, slot_dim] — content for EVERY row (only the dead
            rows are consumed by recycle_reset; live rows are placeholder copies
            of the current slots so the tensor shape matches).
        """
        B, T, d = H_slot.shape
        N = slots.shape[1]
        device = H_slot.device
        # Candidate pool: strided tokens from the current chunk. Use up to 4N
        # candidates (cap by T) so there is room to pick diverse ones.
        n_cand = min(T, max(N, 4 * N))
        stride = max(1, T // n_cand)
        cand_idx = (torch.arange(n_cand, device=device) * stride) % T       # [n_cand]
        cand = H_slot[:, cand_idx, :]                                       # [B, n_cand, d]

        new_content = slots.detach().clone()                               # [B, N, d]
        cand_n = F.normalize(cand.float(), dim=-1)                         # [B, n_cand, d]
        slots_n = F.normalize(slots.float(), dim=-1)                       # [B, N, d]

        for b in range(B):
            dead_b = torch.nonzero(dead_mask[b], as_tuple=False).flatten()
            if dead_b.numel() == 0:
                continue
            live_b = torch.nonzero(~dead_mask[b], as_tuple=False).flatten()
            if live_b.numel() > 0:
                live_keys = slots_n[b, live_b]                             # [L, d]
                # max cosine of each candidate to ANY live slot → lower = more
                # diverse / less covered by the live set.
                sim = cand_n[b] @ live_keys.t()                           # [n_cand, L]
                cov = sim.max(dim=-1).values                              # [n_cand]
            else:
                # No live slots (degenerate) → all candidates equally fine.
                cov = torch.zeros(cand_n.shape[1], device=device)
            order = torch.argsort(cov)                                     # ascending: most diverse first
            n_fill = int(dead_b.numel())
            chosen = order[:n_fill]
            if chosen.numel() < n_fill:
                # Fewer candidates than dead slots — repeat (wrap) the order.
                reps = (n_fill + order.numel() - 1) // order.numel()
                chosen = order.repeat(reps)[:n_fill]
            new_content[b, dead_b] = cand[b, chosen].to(new_content.dtype)
        return new_content

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

        # FIX (2026-06-24): torch.utils.checkpoint(use_reentrant=False) skips
        # grad tracking when hidden_states.requires_grad is False (frozen embed
        # path), making the checkpoint output a leaf with no grad_fn.  This
        # silently breaks the inject_gate gradient path even though the gate
        # parameter has requires_grad=True.  We ensure the input carries a
        # grad by requiring grad here — the backbone is frozen so no extra
        # memory is committed for backbone weights; only the activation itself
        # needs to be retained for inject_gate backprop.
        _hs = hidden_states if hidden_states.requires_grad else hidden_states.requires_grad_(True)
        return _ckpt.checkpoint(_ckpt_fn, _hs, use_reentrant=False)

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

    def _fifo_write_to_buffer(self, h_stored: torch.Tensor) -> None:
        """Append ``h_stored`` to the FIFO buffer + apply eviction.

        For the None / "flat_readerattn" paths this is byte-identical to the
        legacy inline append+pop (the abs-idx bookkeeping below is GATED on
        oracle mode, so it is completely inert — and allocates nothing — off the
        oracle path; non-oracle behaviour and state are unchanged).

        ORACLE bookkeeping (only when ``_fifo_keep_set_mode == "oracle"``): keep
        a parallel ``_fifo_buf_abs_idx`` list naming each buffer entry's
        DOCUMENT-ABSOLUTE chunk index, and a monotonic ``_fifo_write_seq``
        counter (the absolute index of this write). Eviction pops the oldest
        entry from BOTH lists so ``_fifo_buf_abs_idx[j]`` always names
        ``_fifo_buf[j]``. ``_fifo_write_seq`` is reset to 0 at every document
        boundary by the eval harness (``_set_fifo_oracle_needle``) alongside
        ``_fifo_buf`` so ``needle_token_pos // chunk_size`` is meaningful.
        """
        _oracle = getattr(self, "_fifo_keep_set_mode", None) == "oracle"
        self._fifo_buf.append(h_stored)
        if _oracle:
            self._fifo_buf_abs_idx.append(int(self._fifo_write_seq))
            self._fifo_write_seq += 1
        # Eviction (skip when _fifo_keep_all_buffer is set → keep full history).
        if (
            not getattr(self, "_fifo_keep_all_buffer", False)
            and len(self._fifo_buf) > self._fifo_buffer_chunks
        ):
            self._fifo_buf.pop(0)
            if _oracle and self._fifo_buf_abs_idx:
                self._fifo_buf_abs_idx.pop(0)

    def _forward_fifo(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        position_embeddings=None,
        **kwargs,
    ) -> torch.Tensor:
        """FIFO hidden-state memory forward (方案B, MemoryLLM-style, 2026-06-24).

        Maintains a rolling FIFO buffer of the past `fifo_buffer_chunks` chunks'
        hidden states (detached). The current chunk H attends all buffered past
        hiddens as a read-only causal prefix via the wrapped decoder layer's own
        full attention — no routing, no slot projections.

        Write: store current hidden_states (detached if fifo_detach=True) into
               buffer, evicting the oldest entry when full.
        Read:  concat [past_hiddens_prefix | current_H], build a causal-looking
               extended attention mask (prefix tokens see each other causally,
               H-tokens see all prefix + earlier H), run wrapped layer, slice
               out H-portion, apply inject gate, return next_hidden.

        The inject gate is shared with the slot path (self.inject_gate) so the
        model can learn to suppress irrelevant prefix content.
        """
        B, T, d = hidden_states.shape

        # ---- Step 1: Read — build prefix from FIFO buffer ----
        # We track, for each prefix token, BOTH:
        #   - its in-keep-set index ("keep_idx_in_set", 0..K-1 over the kept
        #     chunks only; used by pos_mode='packed' for in-distribution RoPE)
        #   - its original chunk index in self._fifo_buf (0..len(_fifo_buf)-1;
        #     used by pos_mode='real' and by the keep-set mask)
        # Both are LongTensors of shape [P].
        valid: list = []
        valid_orig_idx: list = []      # chunk idx in self._fifo_buf for each valid entry
        if self._fifo_buf:
            for _ci, _h in enumerate(self._fifo_buf):
                if _h.shape[0] == B:
                    valid.append(_h)
                    valid_orig_idx.append(_ci)
        if valid:
            # Decide which chunks to KEEP (Step 1b: keep-set probe). The default
            # (mode=None) keeps ALL valid chunks → byte-identical to legacy.
            kept_local_idx = list(range(len(valid)))   # indices into `valid`
            _ks_mode = getattr(self, "_fifo_keep_set_mode", None)
            if _ks_mode == "flat_readerattn":
                _kept = self._fifo_select_keep_set_reader_attn(
                    hidden_states=hidden_states,
                    valid_chunks=valid,
                    position_embeddings=position_embeddings,
                    topk=int(getattr(self, "_fifo_keep_topk", 25)),
                    recency=int(getattr(self, "_fifo_keep_recency", 2)),
                )
                if _kept is not None:
                    kept_local_idx = _kept    # sorted list of local indices
            elif _ks_mode == "oracle":
                # ORACLE perfect-isolation probe (2026-06-25). Keep ONLY the
                # buffered chunk(s) whose document-absolute index is the needle's
                # (set per-sample on self._fifo_oracle_needle_chunks, mirroring
                # the rawkv oracle's cfg.rawkv_oracle_needle_chunk channel at
                # layer.py:3005), plus the recency floor (last R chunks — the
                # question lives in the last chunk). valid_orig_idx[i] is the
                # local-buffer index; map it to the document-absolute chunk index
                # via _fifo_buf_abs_idx. Falls back to keep-all (logged) when the
                # needle position is unknown or the needle chunk has been evicted.
                _kept = self._fifo_select_keep_set_oracle(
                    valid_orig_idx=valid_orig_idx,
                    n_valid=len(valid),
                    recency=int(getattr(self, "_fifo_keep_recency", 2)),
                )
                if _kept is not None:
                    kept_local_idx = _kept    # sorted list of local indices

            # Concatenate the KEPT chunks in their original (causal) order.
            kept_chunks = [valid[i] for i in kept_local_idx]
            prefix = torch.cat(kept_chunks, dim=1)        # [B, P, d]
            P = prefix.shape[1]
            extended_hidden = torch.cat([prefix, hidden_states], dim=1)  # [B, P+T, d]

            # Build per-prefix-token tags (LongTensors, length P).
            # `keep_pos_per_tok[i]` = position of token i in the KEPT sequence
            #     (== local kept-set index → used by pos_mode='packed')
            # `orig_chunk_per_tok[i]` = original index into self._fifo_buf
            #     (used by pos_mode='real' and by the keep-set mask)
            dev = hidden_states.device
            _keep_pos_pieces = []
            _orig_pos_pieces = []
            for _packed_idx, _local_idx in enumerate(kept_local_idx):
                _tc = valid[_local_idx].shape[1]
                _keep_pos_pieces.append(
                    torch.full((_tc,), _packed_idx, dtype=torch.long, device=dev)
                )
                _orig_pos_pieces.append(
                    torch.full(
                        (_tc,), valid_orig_idx[_local_idx],
                        dtype=torch.long, device=dev,
                    )
                )
            keep_pos_per_tok = torch.cat(_keep_pos_pieces, dim=0)    # [P]
            orig_chunk_per_tok = torch.cat(_orig_pos_pieces, dim=0)  # [P]
        else:
            # No valid past entries — cold start, run bypass.
            P = 0
            extended_hidden = hidden_states
            keep_pos_per_tok = None
            orig_chunk_per_tok = None

        # ---- Step 2: Build extended position embeddings ----
        # Three modes (all eval-only, default None == legacy pos-0):
        #   None     -> all prefix tokens at RoPE pos-0 (legacy, current).
        #   'packed' -> in-distribution: token in kept-chunk k @ in-chunk-offset i
        #               gets RoPE pos = k * chunk_size + i (using KEPT index,
        #               so positions stay within the trained window when
        #               chunk_size*K_keep < trained context).
        #   'real'   -> original sparse positions: token in chunk c @ offset i
        #               gets RoPE pos = c * chunk_size + i (may exceed trained
        #               window; relies on Llama-3 theta=500000 extrapolation).
        # NOTE: cos/sin tables provided via position_embeddings are the cos/sin
        # FOR THE CURRENT CHUNK ONLY (positions 0..T-1 of this chunk). To get
        # cos/sin for arbitrary positions we re-compute via the model-level
        # rotary_emb. We resolve the rotary_emb module lazily via the wrapped
        # decoder layer's self_attn or the outer model handle stashed on the
        # config; fall back to LEGACY pos-0 if we can't find it.
        _pos_mode = getattr(self, "_fifo_pos_mode", None)
        if P > 0 and position_embeddings is not None:
            cos, sin = position_embeddings              # each [B or 1, T, hd]
            # Legacy path (default): all prefix tokens at pos-0.
            if _pos_mode is None or _pos_mode == "pos0":
                if cos.dim() == 3:
                    pos0_cos = cos[:, :1, :].expand(cos.shape[0], P, -1)  # [*, P, hd]
                    pos0_sin = sin[:, :1, :].expand(sin.shape[0], P, -1)
                else:
                    pos0_cos = cos[:1, :].expand(P, -1).unsqueeze(0).expand(B, -1, -1)
                    pos0_sin = sin[:1, :].expand(P, -1).unsqueeze(0).expand(B, -1, -1)
                ext_cos = torch.cat([pos0_cos, cos], dim=1)
                ext_sin = torch.cat([pos0_sin, sin], dim=1)
                ext_pos_emb = (ext_cos, ext_sin)
            else:
                # Build prefix RoPE positions per-token.
                # chunk_size used here is the FIFO write chunk size — i.e. the
                # length of each entry in self._fifo_buf (which may vary across
                # entries; we use each entry's actual length). For 'packed' we
                # pack by kept-index; for 'real' by original chunk index.
                # In-chunk offset is the token's index within its source chunk.
                dev = hidden_states.device
                kept_chunks_local = [valid[i] for i in kept_local_idx]
                if _pos_mode == "real":
                    base_index_per_tok = orig_chunk_per_tok
                elif _pos_mode in ("packed", "hierarchical"):
                    base_index_per_tok = keep_pos_per_tok
                else:
                    base_index_per_tok = keep_pos_per_tok  # safe default
                # Per-token in-chunk offset (0..T_c-1).
                _offsets_pieces = []
                for _kc in kept_chunks_local:
                    _tc = _kc.shape[1]
                    _offsets_pieces.append(
                        torch.arange(_tc, dtype=torch.long, device=dev)
                    )
                _offsets = torch.cat(_offsets_pieces, dim=0)         # [P]
                # We use the CURRENT-chunk length T as the stride (chunk_size)
                # so the position arithmetic is uniform. This matches the
                # write-time assumption that each chunk is T tokens wide.
                pos_pre = base_index_per_tok * T + _offsets          # [P]
                pos_pre = pos_pre.clamp_min_(0)
                # Resolve rotary_emb to compute cos/sin at arbitrary positions.
                _rot = self._fifo_resolve_rotary_emb()
                if _rot is None:
                    # No rotary_emb handle → fall back to legacy pos-0 prefix.
                    if cos.dim() == 3:
                        pos0_cos = cos[:, :1, :].expand(cos.shape[0], P, -1)
                        pos0_sin = sin[:, :1, :].expand(sin.shape[0], P, -1)
                    else:
                        pos0_cos = cos[:1, :].expand(P, -1).unsqueeze(0).expand(B, -1, -1)
                        pos0_sin = sin[:1, :].expand(P, -1).unsqueeze(0).expand(B, -1, -1)
                    ext_cos = torch.cat([pos0_cos, cos], dim=1)
                    ext_sin = torch.cat([pos0_sin, sin], dim=1)
                else:
                    # Call model rotary_emb(x, position_ids) → (cos, sin).
                    # x is only used for dtype/device; pass extended_hidden.
                    pos_ids_pre = pos_pre.unsqueeze(0)                # [1, P]
                    _pcos, _psin = _rot(extended_hidden, pos_ids_pre) # each [1, P, hd]
                    # Broadcast to match cos batch dim.
                    if cos.dim() == 3:
                        if cos.shape[0] != _pcos.shape[0]:
                            _pcos = _pcos.expand(cos.shape[0], -1, -1)
                            _psin = _psin.expand(sin.shape[0], -1, -1)
                        ext_cos = torch.cat([_pcos.to(cos.dtype), cos], dim=1)
                        ext_sin = torch.cat([_psin.to(sin.dtype), sin], dim=1)
                    else:
                        # cos is [T, hd] (rare path); upcast to [1, T, hd].
                        cos2 = cos.unsqueeze(0)
                        sin2 = sin.unsqueeze(0)
                        ext_cos = torch.cat([_pcos.to(cos2.dtype), cos2], dim=1)
                        ext_sin = torch.cat([_psin.to(sin2.dtype), sin2], dim=1)
                ext_pos_emb = (ext_cos, ext_sin)
        else:
            ext_pos_emb = position_embeddings

        # ---- Step 3: Build causal attention mask ----
        # Layout: [prefix P tokens | current H T tokens]
        # mask[i,j] = True means token i CAN attend token j (True = attend).
        # Prefix tokens: causal among themselves (prefix[i] sees prefix[0..i]).
        # H tokens: attend all prefix + causal H.
        # (Keep-set mode pre-filters which chunks are present at all in the
        # prefix; once selected, the prefix attends as before — no extra
        # column-mask needed because non-kept chunks are physically absent.)
        S = P + T
        if P > 0:
            # Build [B, 1, S, S] additive mask (0 = attend, -inf = mask out).
            # Using additive mask convention matching HF SDPA.
            dev = hidden_states.device
            dtype = hidden_states.dtype
            mask_2d = torch.triu(
                torch.full((S, S), float("-inf"), device=dev, dtype=dtype),
                diagonal=1,
            )
            ext_attn_mask = mask_2d.unsqueeze(0).unsqueeze(0)   # [1, 1, S, S]
        else:
            # No prefix: use the original attention_mask unchanged.
            ext_attn_mask = attention_mask

        # ---- Step 4: Run bypass (current H only, for inject gate) ----
        bypass_out = self._maybe_ckpt_wrapped_layer(
            hidden_states,
            attention_mask=None,
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

        if P == 0:
            # Cold start — nothing in buffer yet; just use bypass result.
            # Write current hiddens to buffer and return.
            h_stored = hidden_states.detach() if self._fifo_detach else hidden_states
            self._fifo_write_to_buffer(h_stored)
            return bypass_h

        # ---- Step 5: Run extended attention (prefix + current H) ----
        ext_out = self._maybe_ckpt_wrapped_layer(
            extended_hidden,
            attention_mask=ext_attn_mask,
            position_ids=None,
            past_key_values=None,
            use_cache=False,
            position_embeddings=ext_pos_emb,
            **kwargs,
        )
        if isinstance(ext_out, tuple):
            ext_h = ext_out[0]
            extra = ext_out[1:]
        else:
            ext_h = ext_out
            extra = ()

        # Slice out the H-portion (last T tokens).
        h_out = ext_h[:, P:, :]      # [B, T, d]

        # ---- Step 6: Compute slot_delta + inject gate (same as slot path) ----
        slot_delta = h_out - bypass_h

        # Clip slot_delta per-token norm (same as slot path Fix M-1).
        cfg = self.config
        if not cfg.no_slot_delta_clip:
            _bypass_norms = bypass_h.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            _sd_norms = slot_delta.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            slot_delta = slot_delta * (_bypass_norms / _sd_norms).clamp(max=1.0)

        # Inject gate (same learned gate as the slot path).
        # NOTE: do NOT detach hidden_states here — inject_gate is the only
        # trainable parameter in the FIFO path, and its gradient must flow
        # through hidden_states → gate logit → g → next_hidden → loss.
        _hs_f32 = hidden_states.float()
        _gate_logit = torch.nn.functional.linear(
            _hs_f32,
            self.inject_gate.weight.float(),
            self.inject_gate.bias.float(),
        )
        g = torch.sigmoid(_gate_logit).to(hidden_states.dtype)  # [B, T, 1]

        next_hidden = bypass_h + g * slot_delta

        # ---- Step 7: Write current hidden_states to FIFO buffer ----
        h_stored = hidden_states.detach() if self._fifo_detach else hidden_states
        self._fifo_write_to_buffer(h_stored)

        # Telemetry (re-use last_aux_losses slot so training loop can log it).
        self.last_aux_losses = {}
        self.last_idx = None
        self.last_scores = None

        if extra:
            return (next_hidden, *extra)
        return next_hidden

    # --------------------------------------------------------------------- #
    # FIFO eval-time probes (2026-06-25): position-fix + keep-set helpers.
    # All used ONLY by _forward_fifo when the corresponding probe flag is set.
    # --------------------------------------------------------------------- #

    def _fifo_resolve_rotary_emb(self):
        """Return a callable rotary_emb(x, position_ids) -> (cos, sin) for
        recomputing RoPE at arbitrary positions, or None if we can't find it.

        Search order:
          1. self.wrapped_layer.self_attn.rotary_emb (older HF Llama).
          2. Outer model handle stashed on self.config._model_root (set by
             the eval helper at load time, if available).
          3. Walk parents via self._mem_space_model_root (set by patch.py)
             — currently not guaranteed; treated as best-effort.
        """
        try:
            _attn = getattr(self.wrapped_layer, "self_attn", None)
            if _attn is not None:
                _rot = getattr(_attn, "rotary_emb", None)
                if _rot is not None:
                    return _rot
        except Exception:
            pass
        # Stashed root handle. Stored list-wrapped (`[root]`) so that nn.Module
        # does NOT register the outer model as a child submodule of this layer
        # — a bare nn.Module attribute would create a model<->layer cycle and
        # make model.train() recurse infinitely. Fall back to the legacy bare
        # attr only if some old caller still set it.
        _ref = getattr(self, "_fifo_rotary_root_ref", None)
        if _ref:
            _root = _ref[0]
        else:
            _root = getattr(self, "_fifo_rotary_root", None)
        if _root is not None:
            _rot = getattr(_root, "rotary_emb", None)
            if _rot is not None:
                return _rot
            # transformers >= 4.45 stashes rotary_emb on model.model.
            _inner = getattr(_root, "model", None)
            if _inner is not None:
                _rot = getattr(_inner, "rotary_emb", None)
                if _rot is not None:
                    return _rot
        return None

    def _fifo_select_keep_set_reader_attn(
        self,
        hidden_states: torch.Tensor,
        valid_chunks: list,
        position_embeddings,
        topk: int,
        recency: int,
    ):
        """Pick which chunks of the FIFO buffer to KEEP using this layer's
        native q.k salience (reader-attn). Returns a sorted list of local
        indices into `valid_chunks` (the kept set), or None on failure.

        Scoring per chunk c:
            score = mean over query tokens q of (max over chunk tokens t of
                    q_q · k_t / sqrt(hd))
            pooled over heads via amax(head).
        Then keep top-K_keep chunks by score, plus the last `recency` chunks
        unconditionally (recency floor). Result is sorted to preserve causal
        order in the prefix.
        """
        try:
            import torch as _t
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
            C = len(valid_chunks)
            if C == 0:
                return None
            # If everything would be kept anyway, skip the work.
            keep_n = max(1, min(int(topk), C))
            recn = max(0, min(int(recency), C))
            if keep_n + recn >= C:
                return list(range(C))
            _attn = getattr(self.wrapped_layer, "self_attn", None)
            if _attn is None:
                return None
            _pre_norm = getattr(self.wrapped_layer, "input_layernorm", None)
            with _t.no_grad():
                _hs = hidden_states
                B, Tq, d = _hs.shape
                hd = _attn.head_dim
                if _pre_norm is not None:
                    _hs_q = _pre_norm(_hs)
                else:
                    _hs_q = _hs
                q = _attn.q_proj(_hs_q).view(B, Tq, -1, hd).transpose(1, 2)  # [B,nh,Tq,hd]
                cos, sin = position_embeddings
                q_r, _ = apply_rotary_pos_emb(q, q, cos, sin)
                # Use the LAST query token's hidden as the salience probe
                # (matches _reader_attn_keep_set's convention).
                qv = q_r[:, :, -1, :]                                      # [B, nh, hd]
                nh = qv.shape[1]
                sal = _t.empty(C, device=_hs.device, dtype=_t.float32)
                for c, _kh in enumerate(valid_chunks):
                    _kh_in = _kh.to(_hs.device, dtype=_hs.dtype)
                    if _pre_norm is not None:
                        _kh_in = _pre_norm(_kh_in)
                    M = _kh_in.shape[1]
                    kk = _attn.k_proj(_kh_in).view(B, M, -1, hd).transpose(1, 2)  # [B,nkv,M,hd]
                    nkv = kk.shape[1]
                    if nh != nkv:
                        kk = kk.repeat_interleave(nh // nkv, dim=1)               # [B,nh,M,hd]
                    aw = _t.einsum("bhd,bhmd->bhm", qv.float(), kk.float()) * (hd ** -0.5)
                    aw = aw.amax(dim=1)                                          # [B, M]
                    sal[c] = aw.amax(dim=-1).mean().float()                      # mean over batch
                # Top-K by score.
                top_idx = _t.topk(sal, k=keep_n, dim=0).indices.tolist()
                kept = set(int(i) for i in top_idx)
                # Recency floor: always keep last `recn` chunks.
                for c in range(max(0, C - recn), C):
                    kept.add(c)
                return sorted(kept)
        except Exception:
            return None

    def _fifo_reader_attn_salience(
        self,
        hidden_states: torch.Tensor,
        chunk_hiddens: list,
        position_embeddings,
    ):
        """GRAD-BEARING per-chunk reader-attn salience (2026-06-27, learn-to-select).

        This is the differentiable twin of the scoring block inside
        ``_fifo_select_keep_set_reader_attn`` (layer.py:1647-1674). That method
        wraps the einsum in ``torch.no_grad()`` because it is only used to pick a
        keep-set; here we deliberately KEEP the graph so a supervised selection
        loss can push gradient into this layer's ``q_proj`` / ``k_proj``.

        Returns a ``[C]`` float32 tensor ``sal`` (C == len(chunk_hiddens)) where
        ``sal[c]`` is the SAME quantity the eval selector ranks:
            score(c) = mean_batch( max_token( max_head( q_last . k_t / sqrt(hd) ) ) )
        ``q_last`` is the last query token's RoPE-rotated projection of
        ``hidden_states``; ``k`` is ``k_proj(chunk_c)``. No top-k, no recency floor
        — the caller (train step) applies a CE over the full ``sal`` vector with
        the known needle-chunk index as the target. Returns ``None`` on any
        structural failure (caller then skips the selection loss for that step).

        CRITICAL: do NOT call this inside ``torch.no_grad()`` and make sure this
        layer's ``q_proj`` / ``k_proj`` are trainable (``--unfreeze_layers_from``
        must be <= this layer's index), otherwise the returned tensor carries no
        grad and the selection loss is a silent no-op (the #1 death trap flagged
        in status/LEARN_TO_SELECT_DESIGN §4-D).
        """
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        C = len(chunk_hiddens)
        if C == 0:
            return None
        _attn = getattr(self.wrapped_layer, "self_attn", None)
        if _attn is None:
            return None
        _pre_norm = getattr(self.wrapped_layer, "input_layernorm", None)
        _hs = hidden_states
        B, Tq, d = _hs.shape
        hd = _attn.head_dim
        _hs_q = _pre_norm(_hs) if _pre_norm is not None else _hs
        q = _attn.q_proj(_hs_q).view(B, Tq, -1, hd).transpose(1, 2)   # [B,nh,Tq,hd]
        cos, sin = position_embeddings
        q_r, _ = apply_rotary_pos_emb(q, q, cos, sin)
        # Last query token's hidden as the salience probe (matches the no_grad
        # scorer and the eval convention).
        qv = q_r[:, :, -1, :]                                         # [B, nh, hd]
        nh = qv.shape[1]
        sal_list = []
        for _kh in chunk_hiddens:
            _kh_in = _kh.to(_hs.device, dtype=_hs.dtype)
            if _pre_norm is not None:
                _kh_in = _pre_norm(_kh_in)
            M = _kh_in.shape[1]
            kk = _attn.k_proj(_kh_in).view(B, M, -1, hd).transpose(1, 2)  # [B,nkv,M,hd]
            nkv = kk.shape[1]
            if nh != nkv:
                kk = kk.repeat_interleave(nh // nkv, dim=1)               # [B,nh,M,hd]
            aw = torch.einsum("bhd,bhmd->bhm", qv.float(), kk.float()) * (hd ** -0.5)
            aw = aw.amax(dim=1)                                           # [B, M]
            sal_list.append(aw.amax(dim=-1).mean().float())              # scalar
        sal = torch.stack(sal_list, dim=0)                               # [C]
        return sal

    def _fifo_select_keep_set_oracle(self, valid_orig_idx, n_valid, recency):
        """ORACLE keep-set (2026-06-25): keep ONLY the buffered chunk(s) holding
        the needle (perfect isolation) plus the recency floor. Returns a sorted
        list of LOCAL indices into the `valid` list (the kept set), or None to
        signal "keep all" fallback (so the caller leaves kept_local_idx == all).

        Needle channel: ``self._fifo_oracle_needle_chunks`` is a set of 0-based
        DOCUMENT-ABSOLUTE chunk indices, set per-sample by the eval harness — the
        FIFO analogue of the rawkv oracle's ``cfg.rawkv_oracle_needle_chunk``
        (read at layer.py ~3005). Empty/None → unknown → keep all (logged).

        Chunk-index mapping (CRITICAL):
          * The harness derives the needle's absolute chunk index as
            ``needle_token_pos // chunk_size`` (token p lives in chunk p//cs).
          * Each FIFO buffer entry's document-absolute chunk index is recorded in
            ``self._fifo_buf_abs_idx`` at write time (entry j was the
            abs_idx-th chunk appended). ``valid_orig_idx[i]`` maps the i-th
            *valid* entry → its index into ``self._fifo_buf`` /
            ``_fifo_buf_abs_idx``; we then look up its absolute index. This makes
            the keep decision correct even after eviction (the buffer holds only
            the most recent ``fifo_buffer_chunks`` entries unless
            ``_fifo_keep_all_buffer`` is set).
          * If NO valid buffer entry carries a needle absolute index (the needle
            chunk has been evicted), we fall back to keep-all and bump the
            evicted counter so the run log shows how many samples were affected.

        The recency floor (last ``recency`` LOCAL entries) is always kept because
        the question is in the most recent chunk(s); without it the reader would
        have no query context.
        """
        needle_abs = getattr(self, "_fifo_oracle_needle_chunks", None)
        if not needle_abs:
            # Needle position unknown for this sample → keep all (logged).
            self._fifo_oracle_fallback_count += 1
            return None
        abs_list = getattr(self, "_fifo_buf_abs_idx", None) or []
        kept = set()
        for _local_i, _buf_j in enumerate(valid_orig_idx):
            if 0 <= _buf_j < len(abs_list) and abs_list[_buf_j] in needle_abs:
                kept.add(_local_i)
        if not kept:
            # Needle chunk evicted (or never written) → keep all (logged).
            self._fifo_oracle_fallback_count += 1
            self._fifo_oracle_evicted_count += 1
            return None
        # Recency floor: always keep the last `recency` LOCAL entries (question).
        recn = max(0, min(int(recency), n_valid))
        for c in range(max(0, n_valid - recn), n_valid):
            kept.add(c)
        return sorted(kept)

    # --------------------------------------------------------------------- #
    # Main forward
    # --------------------------------------------------------------------- #

    def _reader_attn_keep_set(self, hidden_states, store, self_attn, pre_norm,
                              position_embeddings, k):
        """Pick the top-k kept chunks by the READER's OWN native q.k salience
        (2026-06-20 dilution fix, keep_set_mode='reader_attn'). No trained
        scorer -> sidesteps H2. Returns a 1-D LongTensor of chunk indices.

        Per chunk c: salience = max over (query token q, chunk token t) of
        q_q . k_t / sqrt(hd), using THIS layer's native q_proj/k_proj + RoPE
        (the same projections the reader attends with). The query is the LAST
        real token's hidden (the readout position). Pure inference (no_grad).
        """
        try:
            import torch as _t
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
            with _t.no_grad():
                B, M, d = store.token_hidden.shape
                C = store.gist_src.shape[1]
                dev = hidden_states.device
                hd = self_attn.head_dim
                # query = last query-token hidden, native q_proj + RoPE.
                _hs = hidden_states
                if pre_norm is not None:
                    _hs = pre_norm(_hs)
                q = self_attn.q_proj(_hs).view(B, _hs.shape[1], -1, hd).transpose(1, 2)
                cos, sin = position_embeddings
                q_r, _ = apply_rotary_pos_emb(q, q, cos, sin)
                qv = q_r[:, :, -1, :]                                  # [B, nh, hd]
                # chunk-token keys: native k_proj on stored hidden (no RoPE here;
                # salience is a content match, position-agnostic for selection).
                _kh = store.token_hidden.to(dev, dtype=_hs.dtype)
                if pre_norm is not None:
                    _kh = pre_norm(_kh)
                kk = self_attn.k_proj(_kh).view(B, M, -1, hd).transpose(1, 2)  # [B,nkv,M,hd]
                nh = qv.shape[1]
                nkv = kk.shape[1]
                kk = kk.repeat_interleave(nh // nkv, dim=1)            # [B,nh,M,hd]
                aw = _t.einsum("bhd,bhmd->bhm", qv.float(), kk.float()) * (hd ** -0.5)
                aw = aw.amax(dim=1)                                    # [B, M] max over heads
                tok_chunk = store.token_chunk.to(dev)[0]              # [M]
                sal = _t.full((C,), float("-inf"), device=dev)
                # per-chunk max salience over its tokens (batch row 0; shared store)
                for c in range(C):
                    m = (tok_chunk == c)
                    if m.any():
                        sal[c] = aw[0][m].max()
                kk_top = min(k, C)
                idx = _t.topk(sal, k=kk_top, dim=0).indices
                return idx
        except Exception:
            return None

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
                    # Batched-eval padding (2026-06-09): the L3 pool reduces over
                    # the PREVIOUS chunk's tokens; if that chunk was right-padded
                    # (the growing last/generation chunk), mask the pads so the
                    # summary matches the unpadded single-sample path. None for
                    # full (streaming) chunks → byte-identical to pre-2026-06-09.
                    _prev_tok_mask = getattr(self.l3_pool, "_prev_chunk_token_mask", None)
                    l3_summaries = self.l3_pool(
                        prev_h, prev_summary=prev_summary, chunk_mask=_prev_tok_mask
                    )
                    object.__setattr__(self.l3_pool, "_chunk_summary_cache", l3_summaries)

        B, T, d = hidden_states.shape
        if d != self.d_model:
            raise RuntimeError(
                f"hidden_states last-dim {d} != d_model {self.d_model}"
            )

        # Increment internal forward counter (cheap; done before any heavy work).
        self._fwd_count += 1

        cfg = self.config

        # ------------------------------------------------------------------
        # FIFO hidden-state memory (方案B, 2026-06-24).
        # When use_fifo_memory=True, bypass ALL slot routing and use a
        # MemoryLLM-style FIFO rolling buffer of raw past-chunk hiddens as
        # a read-only prefix for the wrapped decoder layer.
        # ------------------------------------------------------------------
        if self._use_fifo_memory:
            return self._forward_fifo(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        # Batched-eval padding mask (2026-06-09). The eval driver stashes the
        # current chunk's [B, T] token mask on the shared memory bank as
        # ``_active_token_mask`` (1=real token, 0=pad). It is consumed ONLY by
        # the non-causal pooling reductions (selector routing here, and the L3
        # pool over the *previous* chunk above). None during training / bsz=1
        # eval → all reductions are byte-identical to pre-2026-06-09.
        _active_token_mask = getattr(self.memory_bank, "_active_token_mask", None)
        if _active_token_mask is not None and (
            _active_token_mask.shape[0] != B or _active_token_mask.shape[1] != T
        ):
            # Stale mask (e.g. left over from a differently-shaped chunk) — ignore.
            _active_token_mask = None

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
                hidden_states, slots, query_tokens=l3_summaries,
                token_mask=_active_token_mask,
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

            # ---- Slot-Routed Evidence Memory: WRITE (2026-06-17) ----
            # Only the designated evidence layer caches evidence (single shared
            # bank → one owner). For each selected slot, score the chunk tokens
            # by their routing affinity to that slot (q·kᵀ, reusing the EXISTING
            # normalized routing projections stashed by the selector), take the
            # top-Bcnt most-salient tokens, and store their UNCOMPRESSED hidden
            # states. No-op while frozen (eval question-time) or when disabled.
            if (
                self._is_evidence_layer
                and not self.memory_bank.frozen
                and idx is not None
            ):
                _q = getattr(self.selector, "_last_routing_q", None)   # [B, T, S]
                _k = getattr(self.selector, "_last_routing_k", None)   # [B, N, S]
                if _q is not None and _k is not None and _q.shape[1] == T:
                    with torch.no_grad():
                        Bcnt = cfg.evidence_buffer_size
                        _idx_l = idx.long()                            # [B, k]
                        _S = _k.shape[-1]
                        # Routing key of each selected slot.
                        _k_sel = _k.gather(
                            1, _idx_l.unsqueeze(-1).expand(-1, -1, _S)
                        )                                              # [B, k, S]
                        # Affinity of every token to each selected slot.
                        _aff = torch.einsum("bts,bks->bkt", _q.float(), _k_sel.float())  # [B,k,T]
                        if _active_token_mask is not None:
                            _aff = _aff.masked_fill(
                                (~_active_token_mask.bool()).unsqueeze(1),
                                float("-inf"),
                            )
                        _C = min(Bcnt, T)
                        _top_s, _top_t = torch.topk(_aff, k=_C, dim=2)  # [B,k,C]
                        # Gather the chunk-token hidden states for those indices.
                        _hd = hidden_states.shape[-1]
                        _tok_h = hidden_states.detach().gather(
                            1,
                            _top_t.reshape(B, -1).unsqueeze(-1).expand(-1, -1, _hd),
                        ).reshape(B, k_slots, _C, _hd)                 # [B,k,C,d]
                        self.memory_bank.write_evidence(
                            _idx_l, _tok_h, _top_s, token_pos=_top_t
                        )
            # ---- Parallel raw-KV retrieval channel: WRITE (2026-06-18) ----
            # SLOT-INDEPENDENT. On the designated rawkv_layer, append EVERY real
            # token of this chunk's uncompressed hidden states to the per-sequence
            # raw-KV store, keyed by its routing query (reused from the selector's
            # stashed _last_routing_q so writer + retriever share one projection).
            # No top-k, no slot coupling: precise facts are never compressed away.
            # No-op while frozen (eval question-time) or when disabled.
            if self._is_rawkv_layer and not self.memory_bank.frozen:
                _rq = getattr(self.selector, "_last_routing_q", None)   # [B, T, S]
                if _rq is not None and _rq.shape[0] == B and _rq.shape[1] == T:
                    # Source positions = in-chunk offsets (each streaming chunk
                    # resets RoPE to 0, so offset == the token's real phase).
                    _rk_pos = torch.arange(
                        T, device=hidden_states.device, dtype=torch.long
                    ).unsqueeze(0).expand(B, -1)
                    self.memory_bank.append_rawkv(
                        hidden_states.detach(), _rq, token_pos=_rk_pos
                    )
            # ---- TRUE in-attention K/V concat channel: WRITE (2026-06-18) ----
            # Same per-sequence raw-KV store as the rawkv channel, but written by
            # the inattn_kv_layer (independent owner). Appends EVERY real token's
            # uncompressed hidden + routing-query key so the in-attention read can
            # retrieve them and inject as native K/V. No-op while frozen / off.
            # Multi-layer (v2): only the single write owner appends, so the shared
            # store is populated once per chunk (all inject layers read from it).
            if self._is_inattn_write_owner and not self.memory_bank.frozen:
                _iq = getattr(self.selector, "_last_routing_q", None)   # [B, T, S]
                if _iq is not None and _iq.shape[0] == B and _iq.shape[1] == T:
                    _ik_pos = torch.arange(
                        T, device=hidden_states.device, dtype=torch.long
                    ).unsqueeze(0).expand(B, -1)
                    self.memory_bank.append_rawkv(
                        hidden_states.detach(), _iq, token_pos=_ik_pos
                    )
            # ---- Raw-KV READOUT (Method A): WRITE (2026-06-19) ----
            # The write owner appends this chunk's UNCOMPRESSED token hidden
            # (pre-LN layer input) + source positions + a per-chunk gist source
            # (mean-pooled chunk hidden) to the per-sequence readout store. NO
            # compression, NO slot coupling, NO TopKSelector. The store lives on
            # the shared bank (lazy-created) so it follows the bank's reset/detach
            # lifecycle. No-op while frozen (eval question-time) or when off.
            if self._is_rawkv_readout_write_owner and not self.memory_bank.frozen:
                from .rawkv_readout import RawKVReadoutStore
                _ro_store = getattr(self.memory_bank, "_rawkv_readout_store", None)
                if _ro_store is None:
                    _ro_store = RawKVReadoutStore()
                    object.__setattr__(
                        self.memory_bank, "_rawkv_readout_store", _ro_store
                    )
                _ro_pos = torch.arange(
                    T, device=hidden_states.device, dtype=torch.long
                ).unsqueeze(0).expand(B, -1)
                _ro_store.append_chunk(
                    hidden_states.detach(), token_pos=_ro_pos,
                    pool=getattr(cfg, "rawkv_gist_pool", "mean"),
                )
            # Driven from layer-0 only (the bank is SHARED across all 32 layers,
            # so a single driver avoids 32 layers fighting — mirrors the layer-0
            # usage-histogram pattern). Runs whenever the bank is writable (NOT
            # frozen): like normalize_readout, it changes STORED state so it must
            # be consistent across train AND eval-haystack ingestion (eval
            # question-time generation freezes the bank → skipped automatically).
            # Every step is a no-op when dead_slot_reset_interval <= 0 → P11.
            _do_recycle = (
                self._layer_idx == 0
                and self._dead_slot_reset_interval > 0
                and not self.memory_bank.frozen
                and not cfg.disable_l1_inject
            )
            _track_cum = (
                self._layer_idx == 0
                and not cfg.disable_l1_inject
            )
            if _track_cum:
                with torch.no_grad():
                    _Ncfg = cfg.num_slots
                    # (Re)allocate per-sample counters on cold start / shape change.
                    _need_alloc = (
                        self._cum_usage is None
                        or self._cum_usage.shape[0] != B
                        or self._cum_usage.shape[1] != _Ncfg
                    )
                    if cold_start_this_call or _need_alloc:
                        self._cum_usage = torch.zeros(
                            B, _Ncfg, device=idx.device, dtype=torch.long
                        )
                        self._recycle_usage = torch.zeros(
                            B, _Ncfg, device=idx.device, dtype=torch.long
                        )
                        self._recycle_chunk_count = 0
                        self._recycle_grace_mask = None
                        self._recycle_grace_remaining = 0
                        self._last_recycle_resets = 0
                        # v20: read-mass accumulators share the same lifecycle.
                        self._cum_read_mass = torch.zeros(
                            B, _Ncfg, device=idx.device, dtype=torch.float32
                        )
                        self._recent_read_mass = torch.zeros(
                            B, _Ncfg, device=idx.device, dtype=torch.float32
                        )
                        self._evict_cold_streak = torch.zeros(
                            B, _Ncfg, device=idx.device, dtype=torch.long
                        )
                        self._read_decay_chunk_count = 0
                    # Per-sample selection indicator for THIS chunk (dedup the
                    # global-slot duplicates via clamp to 1).
                    _sel = torch.zeros(
                        B, _Ncfg, device=idx.device, dtype=torch.long
                    ).scatter_(1, idx.long(), 1)               # [B, N], 0/1
                    self._cum_usage += _sel
                    if self._recycle_usage is not None:
                        self._recycle_usage += _sel
                    # Per-slot token-mass accumulation (2026-06-15). Routing is
                    # PER-CHUNK: a slot that wins this chunk's top-k (_sel==1)
                    # absorbs a write aggregated over ALL the chunk's real tokens,
                    # so it gains exactly `tok_count[b]` mass this chunk. tok_count
                    # = #real (non-pad) tokens per sample: T when no pad mask, else
                    # the row-sum of _active_token_mask. Accumulated layer-0-only
                    # on the shared bank (this block is layer-0 gated), so it is
                    # not 32x over-counted, and stays in the SAME slot index order
                    # the read aligns to. Gated on the flag so it is zero-overhead
                    # when off. add_token_mass is a no_grad no-op while frozen.
                    if cfg.use_readout_mass_bias or cfg.use_learnable_mass_bias:
                        if _active_token_mask is not None:
                            _tok_count = (
                                _active_token_mask.to(torch.float32)
                                .sum(dim=1)
                            )                                      # [B]
                        else:
                            _tok_count = torch.full(
                                (B,), float(T), device=idx.device, dtype=torch.float32
                            )
                        self.memory_bank.add_token_mass(_sel, _tok_count)
                    # EXP-D2 scalars (cumulative, sample-scoped). Mean over batch.
                    self._last_dead_slot_frac = (
                        (self._cum_usage == 0).float().mean().item()
                    )
                    self._last_max_slot_select_count = (
                        self._cum_usage.max().float().item()
                    )

            if _do_recycle and self._recycle_usage is not None:
                self._recycle_chunk_count += 1
                # H in slot space, shared by force-write + reset (detached: the
                # recycled content is a re-initialisation, like init_from_hidden,
                # and must not build a giant graph).
                _need_content = (
                    self._recycle_grace_remaining > 0
                    or (self._recycle_chunk_count % self._dead_slot_reset_interval == 0)
                )
                if _need_content:
                    with torch.no_grad():
                        if self.slot_dim != self.d_model:
                            _H_slot = self.hidden_to_slot(hidden_states).detach()
                        else:
                            _H_slot = hidden_states.detach()
                # ① grace force-write FIRST (applies to slots reset on a PRIOR
                # chunk, so a brand-new reset below grants its grace to the NEXT
                # chunks, giving exactly G post-reset writes). WRITE-ONLY: nudges
                # only the recycled rows via delta-rule toward fresh, diverse,
                # on-manifold current-chunk content — the selector + read path
                # are untouched (key discriminator vs ROUTE-A arm4).
                if (
                    self._recycle_grace_remaining > 0
                    and self._recycle_grace_mask is not None
                ):
                    with torch.no_grad():
                        _gmask = self._recycle_grace_mask.to(idx.device)
                        _gcontent = self._select_diverse_strided_content(
                            _H_slot, self.memory_bank.get(), _gmask
                        )
                    self.memory_bank.force_write(_gmask, _gcontent, beta=self._current_beta())
                    self._recycle_grace_remaining -= 1
                    if self._recycle_grace_remaining <= 0:
                        self._recycle_grace_mask = None
                # ② periodic RESET at the window boundary: dead = zero selections
                # over the last `interval` chunks. Overwrite ONLY dead rows with
                # diverse strided current-chunk content, then open a grace window.
                if self._recycle_chunk_count % self._dead_slot_reset_interval == 0:
                    _n_evicted_readmass = -1
                    with torch.no_grad():
                        if (
                            self._slot_evict_mode == "readmass"
                            and self._cum_read_mass is not None
                            and self._recent_read_mass is not None
                            and self._evict_cold_streak is not None
                        ):
                            # ---- v20 Arm B: read-mass eviction + protection ----
                            # A slot is "cold" this boundary if its RECENT read
                            # mass is <= floor AND it had zero recent WRITES
                            # (selections). The cold-streak counter increments
                            # while cold and resets the moment a slot is read or
                            # written. Only slots cold for >= protect_chunks
                            # consecutive boundaries are eligible; from those,
                            # evict at most ceil(max_frac·N) with the LOWEST
                            # cumulative read-mass. read≠write: this is the CORRECT
                            # liveness judge (vs the write-based _cum_usage path).
                            _Nb = cfg.num_slots
                            _cold = (
                                (self._recent_read_mass <= self._slot_evict_floor)
                                & (self._recycle_usage == 0)
                            )                                       # [B, N] bool
                            self._evict_cold_streak = torch.where(
                                _cold,
                                self._evict_cold_streak + 1,
                                torch.zeros_like(self._evict_cold_streak),
                            )
                            _eligible = _cold & (
                                self._evict_cold_streak
                                >= self._slot_evict_protect_chunks
                            )                                       # [B, N] bool
                            import math as _math_evict
                            _max_evict = int(
                                _math_evict.ceil(
                                    self._slot_evict_max_frac * _Nb
                                )
                            )
                            _dead = torch.zeros_like(_eligible)     # [B, N] bool
                            if _max_evict > 0 and bool(_eligible.any()):
                                # Rank by cumulative read-mass ascending; mask
                                # ineligible to +inf so they never get picked.
                                _score = self._cum_read_mass.clone()
                                _score = _score.masked_fill(
                                    ~_eligible, float("inf")
                                )
                                _k = min(_max_evict, _Nb)
                                _vals, _topidx = torch.topk(
                                    _score, _k, dim=1, largest=False
                                )                                   # [B, _k]
                                _finite = torch.isfinite(_vals)     # eligible only
                                _dead.scatter_(1, _topidx, _finite)
                            _n_dead = int(_dead.sum().item())
                            _n_evicted_readmass = _n_dead
                            # Evicted slots restart their cold streak (fresh).
                            self._evict_cold_streak = torch.where(
                                _dead,
                                torch.zeros_like(self._evict_cold_streak),
                                self._evict_cold_streak,
                            )
                        else:
                            # EXP-R1c: existing dead-slot judge — byte-for-byte
                            # unchanged when slot_evict_mode == "off".
                            #   "window"     → R1: zero selections in the last
                            #                  `interval` chunks (window-scoped).
                            #   "cumulative" → R1c: zero selections over the WHOLE
                            #                  sample so far (sample-scoped, never
                            #                  window-zeroed) — only ever recycle a
                            #                  slot that has NEVER been selected,
                            #                  so a long-range memory slot that is
                            #                  merely temporarily silent is spared.
                            if (
                                self._dead_slot_criterion == "cumulative"
                                and self._cum_usage is not None
                            ):
                                _dead = (self._cum_usage == 0)     # [B, N] bool
                            else:
                                _dead = (self._recycle_usage == 0)  # [B, N] bool
                            _n_dead = int(_dead.sum().item())
                    if _n_dead > 0:
                        if self._dead_slot_reset_mode == "zero":
                            _content = torch.zeros_like(self.memory_bank.get())
                        else:
                            with torch.no_grad():
                                _content = self._select_diverse_strided_content(
                                    _H_slot, self.memory_bank.get(), _dead
                                )
                        self.memory_bank.recycle_reset(_dead, _content)
                        self._recycle_grace_mask = _dead.clone()
                        self._recycle_grace_remaining = self._dead_slot_grace_chunks
                        self._last_recycle_resets += 1
                    self._last_n_recycled = _n_dead
                    self._last_n_evicted_readmass = _n_evicted_readmass
                    # Open a fresh window.
                    self._recycle_usage.zero_()
                    # Arm B owns the recent-read window at the recycle boundary;
                    # only zero it when Arm B is active so Arm A's independent
                    # decay window is not perturbed by R1 recycling running
                    # alongside (each arm manages its own window otherwise).
                    if (
                        self._slot_evict_mode == "readmass"
                        and self._recent_read_mass is not None
                    ):
                        self._recent_read_mass.zero_()


            # ---- v20 Arm A: soft read-decay (layer-0, no_grad) ----
            # Independent of EXP-R1 recycling: guarded ONLY by
            # slot_read_decay_rate>0 + writable bank. Every
            # slot_read_decay_interval chunks, multiply each slot's VALUE by a
            # factor in [min_keep, 1] driven by its recent read fraction — slots
            # read recently keep their norm, unread slots fade SLOWLY (never hard
            # deleted). Uses _recent_read_mass accumulated over the window so far
            # (this chunk's read mass is added AFTER, at the read call), then
            # zeroes the window. No-op when rate<=0 → P11 byte-identical.
            _do_read_decay = (
                self._layer_idx == 0
                and self._slot_read_decay_rate > 0.0
                and not self.memory_bank.frozen
                and not cfg.disable_l1_inject
                and self._recent_read_mass is not None
            )
            if _do_read_decay:
                self._read_decay_chunk_count += 1
                if (
                    self._read_decay_chunk_count
                    % self._slot_read_decay_interval == 0
                ):
                    with torch.no_grad():
                        _rrm = self._recent_read_mass               # [B, N] float
                        _rmax = _rrm.max(dim=1, keepdim=True).values.clamp(min=1e-8)
                        _recent_read_frac = _rrm / _rmax            # [B, N] in [0,1]
                        _decay = (
                            1.0
                            - self._slot_read_decay_rate
                            * (1.0 - _recent_read_frac)
                        ).clamp(
                            min=self._slot_read_decay_min_keep, max=1.0
                        )                                           # [B, N]
                    self.memory_bank.decay_(_decay.unsqueeze(-1))
                    with torch.no_grad():
                        self._recent_read_mass.zero_()

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
                    # EXP-D3 (2026-06-11): mean pairwise cosine of slot CONTENT
                    # (off-diagonal), the direct homogenisation signal for any
                    # broad-write method (W1/W2/R1). key_max_cos above measures
                    # key-space separability; this measures the slot VALUE space
                    # the read softmax actually attends over. ->1 == all slots'
                    # content collapsed onto one direction → the all-N read can
                    # no longer discriminate → top1_sim flattens. We use the
                    # MEAN (not max) off-diagonal cosine of slots[0] so a slow
                    # drift toward homogenisation is visible early.
                    _slot_content_cos = 0.0
                    _Nsl = slots.shape[1]
                    if _Nsl > 1:
                        _Sc = F.normalize(slots[0].float(), dim=-1)   # [N, slot_dim]
                        _Scs = torch.mm(_Sc, _Sc.t())                 # [N, N]
                        _slot_content_cos = (
                            (_Scs.sum() - _Nsl) / (_Nsl * (_Nsl - 1))
                        ).item()
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
                    # ---- EXP-D (2026-06-11): L3 telemetry (no_grad, layer-0) ----
                    # L3 is the long-range主力 but ran as a black box (no read-mass
                    # or redundancy signal). Two scalars make EXP-1/EXP-2/EXP-3
                    # interpretable instead of blind. Fully guarded: any failure
                    # falls back to 0.0 and can never perturb training.
                    #   l3_attn_mass : mean fraction of the H-query joint-attention
                    #                  softmax mass that lands on the L3 summary key
                    #                  block (i.e. "does H actually USE L3?"). Faithfully
                    #                  replicates the wrapped layer's attention
                    #                  (input_layernorm -> q/k proj -> RoPE -> GQA) over
                    #                  keys [L3(pos0) | H(causal)] — matching the real
                    #                  joint softmax (H->L1 is masked under use_memory_xattn).
                    #   l3_tok_cos   : mean pairwise cosine of the K L3 summary tokens
                    #                  (1.0 => collapsed to one direction; the exact
                    #                  redundancy EXP-1's --l3_diversity_weight treats).
                    _l3_attn_mass = 0.0
                    _l3_tok_cos = 0.0
                    if l3_summaries is not None and l3_summaries.shape[1] > 0:
                        try:
                            _k_l3 = l3_summaries.shape[1]
                            # (ii) L3 summary-token pairwise cosine redundancy
                            _l3b0 = F.normalize(l3_summaries[0].float(), dim=-1)  # [k_l3, d]
                            _l3sim = torch.mm(_l3b0, _l3b0.t())                   # [k_l3, k_l3]
                            if _k_l3 > 1:
                                _l3_tok_cos = (
                                    (_l3sim.sum() - _k_l3) / (_k_l3 * (_k_l3 - 1))
                                ).item()
                            # (i) H->L3 joint-attention mass
                            _attn = self.wrapped_layer.self_attn
                            _ln = self.wrapped_layer.input_layernorm
                            _hd = _attn.head_dim
                            _nh = _attn.q_proj.out_features // _hd
                            _nkv = _attn.k_proj.out_features // _hd
                            _grp = _nh // _nkv
                            _scl = _attn.scaling
                            _cos_e, _sin_e = position_embeddings  # [*, T, head_dim]
                            _Bd = hidden_states.shape[0]
                            _h_ln = _ln(hidden_states)            # [B, T, d]
                            _l3_ln = _ln(l3_summaries)            # [B, k_l3, d]
                            _q = _attn.q_proj(_h_ln).view(_Bd, T, _nh, _hd).transpose(1, 2)
                            _kH = _attn.k_proj(_h_ln).view(_Bd, T, _nkv, _hd).transpose(1, 2)
                            _kL = _attn.k_proj(_l3_ln).view(_Bd, _k_l3, _nkv, _hd).transpose(1, 2)

                            def _rope(_x, _c, _s):
                                _c = _c.unsqueeze(1)  # [*,1,S,hd]
                                _s = _s.unsqueeze(1)
                                _half = _x.shape[-1] // 2
                                _x1 = _x[..., :_half]
                                _x2 = _x[..., _half:]
                                _rot = torch.cat((-_x2, _x1), dim=-1)
                                return _x * _c + _rot * _s

                            # H uses its actual positions; L3 sits at RoPE position 0.
                            _q = _rope(_q.float(), _cos_e.float(), _sin_e.float())
                            _kH = _rope(_kH.float(), _cos_e.float(), _sin_e.float())
                            _cos0 = _cos_e[:, :1, :].float().expand(-1, _k_l3, -1)
                            _sin0 = _sin_e[:, :1, :].float().expand(-1, _k_l3, -1)
                            _kL = _rope(_kL.float(), _cos0, _sin0)
                            # GQA: expand kv heads to query heads
                            if _grp > 1:
                                _kH = _kH.repeat_interleave(_grp, dim=1)
                                _kL = _kL.repeat_interleave(_grp, dim=1)
                            _lg_L3 = torch.matmul(_q, _kL.transpose(2, 3)) * _scl  # [B,nh,T,k_l3]
                            _lg_H = torch.matmul(_q, _kH.transpose(2, 3)) * _scl   # [B,nh,T,T]
                            _causal = torch.triu(
                                torch.ones(T, T, dtype=torch.bool, device=_lg_H.device),
                                diagonal=1,
                            )
                            _lg_H = _lg_H.masked_fill(_causal, float("-inf"))
                            _lg = torch.cat([_lg_L3, _lg_H], dim=-1)              # [B,nh,T,k_l3+T]
                            _aw = torch.softmax(_lg, dim=-1)
                            _l3_attn_mass = _aw[..., :_k_l3].sum(dim=-1).mean().item()
                        except Exception:
                            _l3_attn_mass = 0.0
                            _l3_tok_cos = 0.0
                    self._last_l3_attn_mass = _l3_attn_mass
                    self._last_l3_tok_cos = _l3_tok_cos
                    # Stash scalars so the training loop can push them to wandb
                    # at log_interval (these are layer-0 only; the diag block
                    # only runs every 50 fwd, so the loop reads the latest).
                    self._last_key_max_cos = _key_max_cos
                    self._last_usage_cov = _usage_cov
                    self._last_usage_ent = _usage_ent
                    self._last_usage_var = _usage_var
                    self._last_slot_content_cos = _slot_content_cos
                    # v20 (2026-06-12): fraction of slots EVER read (cum read-mass
                    # > 0). -1 == not measured (no read accumulator allocated).
                    _cum_read_mass_cov = -1.0
                    if self._cum_read_mass is not None:
                        _cum_read_mass_cov = (
                            (self._cum_read_mass > 0).float().mean().item()
                        )
                    self._last_cum_read_mass_cov = _cum_read_mass_cov
                print(
                    f"[QUERY_DIAG step={self.step_counter} fwd={self._fwd_count}]"
                    f" top1_sim_mean={_top1_sim_mean:.6f}"
                    f" topk_mass={_topk_mass:.6f}"
                    f" retrieved_norm_mean={_retrieved_norm:.6f}"
                    f" per_tok_logit_std={_pt_std:.6f}"
                    f" key_max_cos={_key_max_cos:.4f}"
                    f" slot_content_cos={_slot_content_cos:.4f}"
                    f" S_max_cos={_S_max_cos:.4f}"
                    f" summary_q_max_cos={_sq_max_cos:.4f}"
                    f" summary_q_mean_cos={_sq_mean_cos:.4f}"
                    f" uniq_sel_slots={_uniq_sel}"
                    f" usage_cov={_usage_cov:.4f}"
                    f" usage_ent={_usage_ent:.4f}"
                    f" usage_var={_usage_var:.6f}"
                    f" usage_chunks={_usage_chunks}"
                    f" chunk_idx_jaccard={_chunk_idx_jaccard:.4f}"
                    f" slot_attn_entropy={_slot_attn_entropy:.4f}"
                    f" l3_attn_mass={_l3_attn_mass:.4f}"
                    f" l3_tok_cos={_l3_tok_cos:.4f}"
                    # EXP-D2 (2026-06-11): cumulative (sample-scoped) dead-slot
                    # telemetry. dead_slot_frac = frac of slots NEVER selected
                    # over the whole sample (the deadlock metric usage_cov can't
                    # see because it window-resets); max_slot_select_count = how
                    # many times the "richest" slot was picked (rich-get-richer).
                    # recycle_resets = #reset events this sample (0 when EXP-R1
                    # off → no behaviour change to the line when disabled).
                    f" dead_slot_frac={self._last_dead_slot_frac:.4f}"
                    f" max_slot_select_count={self._last_max_slot_select_count:.1f}"
                    # EXP-D (2026-06-12): read-path softmax mass landing on
                    # never-written (dead) vs ever-written (live) slots, from the
                    # MemoryCrossAttentionRead no-grad telemetry. -1 == not
                    # measured (no memory_xattn, or dead_mask absent). Answers
                    # "do the ~330 frozen slots still passively get read?".
                    f" dead_slot_read_mass={(self.memory_xattn._last_dead_slot_read_mass if self.memory_xattn is not None else -1.0):.4f}"
                    f" live_slot_read_mass={(self.memory_xattn._last_live_slot_read_mass if self.memory_xattn is not None else -1.0):.4f}"
                    f" recycle_resets={self._last_recycle_resets}"
                    f" n_recycled={self._last_n_recycled}"
                    # v20 (2026-06-12): read-based slot lifecycle telemetry.
                    # cum_read_mass_cov = frac of slots EVER read (correct
                    # liveness; -1 == not measured). n_evicted_readmass = #slots
                    # Arm B evicted at the last boundary (-1 == Arm B off / no
                    # boundary this emission).
                    f" cum_read_mass_cov={self._last_cum_read_mass_cov:.4f}"
                    f" n_evicted_readmass={self._last_n_evicted_readmass}",
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
            if self._normalize_readout:
                # P11 (2026-06-06): normalized readout. Rescale M_sel_hidden so
                # its per-token magnitude MATCHES the local hidden-state scale
                # (× readout_norm_scale) — both shrinking AND amplifying — so the
                # downstream gate sees a memory signal comparable to the
                # local-attention output. Differs from the default shrink-only
                # clamp below (which only attenuates). The target reference uses
                # hidden_states.detach() so no extra gradient path is created.
                _target = _h_norm_ref * float(self._readout_norm_scale)
                M_sel_hidden = M_sel_hidden * (_target / _m_norms.clamp(min=1e-6))
            else:
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
        # ---- Slot-Routed Evidence Memory: READ (2026-06-17) ----
        # On the evidence layer, gather the evidence of the top-k selected slots
        # (idx) and insert it as a 4th prefix segment: [L3 | L2 | L1 | EV | H].
        # The frozen decoder re-projects these uncompressed token hidden states
        # through its own K/V proj, so H can recall the precise original tokens.
        ev_tokens = None
        k_ev = 0
        _ev_pos = None
        _ev_parts = []
        _ev_pos_parts = []   # parallel source-position lists for the EV tokens
        if (
            self._is_evidence_layer
            and k_slots > 0
            and idx is not None
            and getattr(self.memory_bank, "slot_evidence", None) is not None
        ):
            _se = self.memory_bank.slot_evidence                    # [B, N, Bcnt, d_ev]
            if _se.shape[0] == B:
                _topr = max(1, min(self._evidence_topr, _se.shape[2]))
                _d_ev = _se.shape[-1]
                _idx_g = idx.long().unsqueeze(-1).unsqueeze(-1).expand(
                    -1, -1, _se.shape[2], _d_ev
                )                                                   # [B, k, Bcnt, d_ev]
                _gathered = _se.gather(1, _idx_g)                   # [B, k, Bcnt, d_ev]
                _gathered = _gathered[:, :, :_topr, :]              # [B, k, topr, d_ev]
                _heur = _gathered.reshape(B, k_slots * _topr, _d_ev).to(
                    hidden_states.dtype
                )                                                   # [B, k*topr, d_ev]
                _ev_parts.append(_heur)
                # Landmark fix (2026-06-17): gather the SOURCE positions of those
                # evidence tokens so they inject at their real RoPE phase, not 0.
                _sep = getattr(self.memory_bank, "slot_evidence_pos", None)
                if _sep is not None and _sep.shape[0] == B:
                    _idx_p = idx.long().unsqueeze(-1).expand(-1, -1, _sep.shape[2])
                    _gpos = _sep.gather(1, _idx_p)[:, :, :_topr]    # [B, k, topr]
                    _ev_pos_parts.append(_gpos.reshape(B, k_slots * _topr))
                else:
                    _ev_pos_parts.append(
                        torch.zeros(B, k_slots * _topr, device=hidden_states.device,
                                    dtype=torch.long)
                    )

        # ---- Parallel raw-KV retrieval channel: READ (2026-06-18) ----
        # SLOT-INDEPENDENT. On the rawkv_layer, score the per-sequence raw-KV
        # store with the CURRENT query's routing key (selector._last_routing_q,
        # the same projection used at write), take the top-rawkv_topk original
        # tokens, and append them to the SAME EV prefix block the slot-routed
        # evidence uses → [... | EV(slot-ev + rawkv) | H], injected at their real
        # source RoPE positions. Reuses the existing k_ev / ext-mask / ext-pos
        # plumbing, so no new attention machinery. No-op when the store is empty.
        if self._is_rawkv_layer:
            _rq_read = getattr(self.selector, "_last_routing_q", None)  # [B, T, S]
            if _rq_read is not None and _rq_read.shape[0] == B:
                _ret = self.memory_bank.retrieve_rawkv(
                    _rq_read, cfg.rawkv_topk
                )
                if _ret is not None:
                    _rk_h, _rk_pos = _ret                  # [B,R,d], [B,R]
                    _ev_parts.append(_rk_h.to(hidden_states.dtype))
                    _ev_pos_parts.append(_rk_pos.to(hidden_states.device))
                    # Validity diagnostic (eval-only, env-gated): confirm the
                    # raw-KV store is non-empty and retrieval actually injects
                    # tokens — proves the channel is NOT a silent no-op.
                    if os.environ.get("RAWKV_DEBUG") == "1":
                        print(
                            f"[rawkv] layer={self._layer_idx} store_M="
                            f"{self.memory_bank.rawkv_size()} retrieved="
                            f"{_rk_h.shape[1]} -> EV prefix",
                            flush=True,
                        )

        # ---- ORACLE evidence injection (eval-only, 2026-06-17) ----
        # Bypass routing entirely: if this layer is an oracle injection layer,
        # prepend the GOLD needle span's pre-captured hidden states as an extra
        # evidence prefix block, regardless of what the selector routed. This is
        # the decisive go/no-go probe: it isolates "can the frozen reader USE
        # evidence?" from "did write/retrieval capture the right span?". Set by
        # the eval harness via set_oracle_evidence(); None on the default path.
        # Skipped on the in-attn layer when in-attn is active: that layer injects
        # the oracle through the TRUE in-attention K/V path instead of this EV
        # prefix, so we must NOT also prepend it here (would double-inject).
        _oracle_layers = getattr(self.memory_bank, "_oracle_layers", None)
        _inattn_owns_oracle = self._is_inattn_kv_layer
        if _oracle_layers and self._layer_idx in _oracle_layers and not _inattn_owns_oracle:
            _ohbl = getattr(self.memory_bank, "_oracle_hidden_by_layer", None)
            _oh = _ohbl.get(self._layer_idx) if _ohbl else None
            if _oh is not None and _oh.shape[0] == B:
                _oh = _oh.to(device=hidden_states.device, dtype=hidden_states.dtype)
                _ev_parts.append(_oh)
                # Oracle source positions: the needle span's REAL in-chunk offsets
                # (Landmark fix). Stashed by the harness as _oracle_pos_by_layer;
                # fall back to a contiguous 0..S-1 run if absent.
                _opbl = getattr(self.memory_bank, "_oracle_pos_by_layer", None)
                _op = _opbl.get(self._layer_idx) if _opbl else None
                _S = _oh.shape[1]
                if _op is not None:
                    _op = _op.to(device=hidden_states.device, dtype=torch.long)
                    if _op.dim() == 1:
                        _op = _op.unsqueeze(0)
                    if _op.shape[0] != B:
                        _op = _op[:1].expand(B, -1)
                    _ev_pos_parts.append(_op[:, :_S])
                else:
                    _ev_pos_parts.append(
                        torch.arange(_S, device=hidden_states.device,
                                     dtype=torch.long).unsqueeze(0).expand(B, -1)
                    )

        if _ev_parts:
            ev_tokens = torch.cat(_ev_parts, dim=1) if len(_ev_parts) > 1 else _ev_parts[0]
            k_ev = ev_tokens.shape[1]
            _ev_pos = (
                torch.cat(_ev_pos_parts, dim=1) if len(_ev_pos_parts) > 1
                else _ev_pos_parts[0]
            ) if _ev_pos_parts else None

        parts = []
        if k_l3 > 0:
            parts.append(l3_summaries)
        if k_l2 > 0:
            parts.append(l2_tokens)
        if k_slots > 0:
            parts.append(M_sel_hidden)
        if k_ev > 0:
            parts.append(ev_tokens)
        parts.append(hidden_states)
        if len(parts) == 1:
            extended_hidden = hidden_states                      # pure bypass
        else:
            extended_hidden = torch.cat(parts, dim=1)            # [B, k_l3+k_l2+k_slots+k_ev+T, d]

        # Position embeddings: L3, L2, L1 use position 0 (memory tokens are
        # position-less by design). EV (evidence) is injected at its REAL source
        # RoPE phase (Landmark fix, 2026-06-17) instead of position 0, so the
        # frozen decoder re-projects evidence K/V at the position it actually saw
        # the token — and the EV tokens no longer collide their RoPE phases with
        # the L3/L2/L1 prefix at position 0. Gated on use_slot_evidence: when EV
        # is absent (k_ev==0) the call is byte-identical to the legacy pos-0 path.
        # cfg.evidence_real_positions=False forces the legacy pos-0 EV injection
        # (kept as the A/B control arm against the real-position fix).
        if k_ev > 0 and _ev_pos is not None and cfg.evidence_real_positions:
            ext_pos_emb = _extend_position_embeddings(
                position_embeddings, k_l3 + k_l2 + k_slots + k_ev,
                ev_pos=_ev_pos, k_pos0=k_l3 + k_l2 + k_slots,
            )
        else:
            ext_pos_emb = _extend_position_embeddings(
                position_embeddings, k_l3 + k_l2 + k_slots + k_ev,
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
                k_ev=k_ev,
                isolate_ev=cfg.evidence_isolate_softmax,
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
                k_ev=k_ev,
                isolate_ev=cfg.evidence_isolate_softmax,
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

        # ---- TRUE in-attention K/V concat: READ + stash (2026-06-18) ----
        # Retrieve raw tokens for the current chunk's query, project them through
        # the wrapped layer's NATIVE k/v_proj (after input_layernorm so they live
        # in the native K/V distribution), RoPE the keys at their REAL source
        # positions, and stash on self_attn so the installed wrapper concatenates
        # them onto the native K/V of BOTH the bypass and extended calls below.
        # Because the SAME K_raw/V_raw is added to both, its contribution cancels
        # in slot_delta (= ext_h - bypass_h) and survives in bypass_h → it enters
        # next_hidden at FULL strength (weight 1.0), the clean training-free read.
        #
        # Two retrieval sources (independent, may compose):
        #   * RETRIEVED: top-inattn_kv_topk from the raw-KV store, scored by the
        #     query routing key (the realistic channel).
        #   * ORACLE (eval-only): the gold needle span's pre-captured layer-input
        #     hidden, stashed on the bank as _oracle_hidden_by_layer (bypasses
        #     the scorer). Used to cleanly isolate the READOUT mechanism from the
        #     known-0% retrieval-quality failure — mirrors the oracle-evidence
        #     control, but injected through the TRUE in-attention path this time.
        # Diagnostic counters stashed for the smoke. No-op when both empty / off.
        self._last_inattn_R = 0
        self._last_inattn_pos = None
        _inattn_attn = None
        if self._is_inattn_kv_layer:
            _inattn_attn = getattr(self.wrapped_layer, "self_attn", None)
            # Reset the stash at the START of this forward (gradient-checkpoint
            # safe). Under --gradient_checkpointing the wrapped_layer forward is
            # recomputed during backward; if we cleared the stash at the END of
            # the forward (old behaviour) the recompute would see native-only K/V
            # (R=0) while the saved forward saw native+R injected keys → a
            # CheckpointError metadata mismatch. By resetting here and NOT
            # clearing at the end, the stash stays valid through the recompute,
            # and a forward with no retrieval correctly falls back to None.
            if _inattn_attn is not None:
                _inattn_attn._inattn_kv = None
            _pre_norm = getattr(self.wrapped_layer, "input_layernorm", None)
            _src_h_parts = []
            _src_pos_parts = []
            # ORACLE source (eval-only; bypasses the scorer).
            _orc_layers = getattr(self.memory_bank, "_oracle_layers", None)
            if _orc_layers and self._layer_idx in _orc_layers:
                _ohbl = getattr(self.memory_bank, "_oracle_hidden_by_layer", None)
                _oh = _ohbl.get(self._layer_idx) if _ohbl else None
                if _oh is not None and _oh.shape[0] == B:
                    _src_h_parts.append(_oh.to(hidden_states.dtype))
                    _opbl = getattr(self.memory_bank, "_oracle_pos_by_layer", None)
                    _op = _opbl.get(self._layer_idx) if _opbl else None
                    _So = _oh.shape[1]
                    if _op is not None:
                        _op = _op.to(device=hidden_states.device, dtype=torch.long)
                        if _op.dim() == 1:
                            _op = _op.unsqueeze(0)
                        if _op.shape[0] != B:
                            _op = _op[:1].expand(B, -1)
                        _src_pos_parts.append(_op[:, :_So])
                    else:
                        _src_pos_parts.append(
                            torch.arange(_So, device=hidden_states.device,
                                         dtype=torch.long).unsqueeze(0).expand(B, -1)
                        )
            # RETRIEVED source (realistic channel) — skipped when oracle-only is
            # requested via _inattn_oracle_only on the bank.
            _oracle_only = bool(getattr(self.memory_bank, "_inattn_oracle_only", False))
            _iq_read = getattr(self.selector, "_last_routing_q", None)  # [B,T,S]
            if (
                not (_oracle_only and _src_h_parts)
                and _inattn_attn is not None
                and _iq_read is not None
                and _iq_read.shape[0] == B
            ):
                _iret = self.memory_bank.retrieve_rawkv(_iq_read, cfg.inattn_kv_topk)
                if _iret is not None:
                    _ik_h, _ik_pos = _iret                  # [B,R,d], [B,R]
                    _src_h_parts.append(_ik_h.to(hidden_states.dtype))
                    _src_pos_parts.append(_ik_pos.to(hidden_states.device))
            if _inattn_attn is not None and _src_h_parts:
                from .inattn_kv import build_retrieved_kv
                _src_h = (
                    torch.cat(_src_h_parts, dim=1) if len(_src_h_parts) > 1
                    else _src_h_parts[0]
                )
                _src_pos = (
                    torch.cat(_src_pos_parts, dim=1) if len(_src_pos_parts) > 1
                    else _src_pos_parts[0]
                )
                _K_raw, _V_raw = build_retrieved_kv(
                    _inattn_attn, _src_h, _src_pos, position_embeddings,
                    pre_norm=_pre_norm,
                )
                # Grad-flow diagnostic (2026-06-18): when the layer is asked to
                # probe the injection's in-graph status (set by the training
                # smoke via _inattn_grad_probe), retain grad on the injected K_raw
                # so the smoke can assert K_raw.grad is non-None after backward —
                # proving the injection path is differentiable (NOT detached) and
                # the unfrozen backbone learns to consume it. Zero cost otherwise.
                if getattr(self, "_inattn_grad_probe", False) and _K_raw.requires_grad:
                    _K_raw.retain_grad()
                    self._last_inattn_K_raw = _K_raw
                _inattn_attn._inattn_kv = (_K_raw, _V_raw)
                self._last_inattn_R = int(_K_raw.shape[2])
                self._last_inattn_pos = _src_pos

        # ---- Per-slot raw-KV cache: READ / INJECT (2026-06-22) ----
        # Different from Method A: there is no independent gist scorer and no
        # chunk-level retrieval. We reuse the selector's CURRENT slot ids (idx),
        # gather ALL raw hidden cached under those slots, and inject them through
        # the same native in-attention K/V concat path. This tests the upper bound
        # of "slot picked correctly + raw KV is available" with no capacity cap.
        self._last_slot_kv_cache_R = 0
        if self._is_slot_kv_cache_layer and idx is not None:
            _sk_attn = getattr(self.wrapped_layer, "self_attn", None)
            if _sk_attn is not None:
                _sk_attn._inattn_kv = None
            _sk_mode = getattr(cfg, "slot_kv_select_mode", "router")
            if _sk_mode == "all":
                _sk_idx = torch.arange(
                    int(cfg.num_slots), device=idx.device, dtype=idx.dtype
                ).unsqueeze(0).expand(idx.shape[0], -1)
            elif _sk_mode == "recency":
                _sk_idx = self.memory_bank.recent_slot_kv_slots(int(cfg.top_k))
                if _sk_idx is not None:
                    _sk_idx = _sk_idx.to(device=idx.device, dtype=idx.dtype)
            else:
                _sk_idx = idx
            _sk_ret = None if _sk_idx is None else self.memory_bank.retrieve_slot_kv_cache(_sk_idx)
            if _sk_attn is not None and _sk_ret is not None:
                _sk_h, _sk_pos = _sk_ret                    # [B,R,d], [B,R]
                from .inattn_kv import build_retrieved_kv
                _sk_pre_norm = getattr(self.wrapped_layer, "input_layernorm", None)
                _skK, _skV = build_retrieved_kv(
                    _sk_attn, _sk_h.to(hidden_states.dtype), _sk_pos,
                    position_embeddings, pre_norm=_sk_pre_norm,
                )
                _sk_attn._inattn_kv = (_skK, _skV)
                self._last_slot_kv_cache_R = int(_skK.shape[2])
                if os.environ.get("SLOT_KV_DEBUG") == "1":
                    print(
                        f"[slot_kv_cache] layer={self._layer_idx} mode={_sk_mode} "
                        f"select_k={int(_sk_idx.shape[1]) if _sk_idx is not None else 0} "
                        f"store_M={self.memory_bank.slot_kv_cache_size()} "
                        f"retrieved={self._last_slot_kv_cache_R}",
                        flush=True,
                    )

        # ---- Raw-KV READOUT (Method A): READ / INJECT (2026-06-19) ----
        # Differentiable, TRAINABLE gist-key soft attention (replaces the
        # non-differentiable TopKSelector hard top-k of the inattn probe). The
        # current chunk's query (the pre-LN layer input `hidden_states`,
        # grad-bearing) scores every stored chunk's gist key via the shared
        # trainable GistReadout; the soft-top-k chunks' raw token hidden are
        # re-projected through THIS layer's native k/v_proj + RoPE
        # (build_retrieved_kv) and concatenated in-attention, with the per-chunk
        # gist weight injected as an additive log-bias on the retrieved columns
        # (the 3-tuple stash the inattn wrapper consumes). gradient flows:
        # loss → native softmax → col_bias → GistReadout.query/key_proj, so the
        # scorer is trained on the read path (Landmark mechanism). No-op when the
        # store is empty (e.g. the FIRST chunk, before any write).
        self._last_rawkv_readout_R = 0
        self._last_rawkv_readout_fired = False
        if self._is_rawkv_readout_layer and self.gist_readout is not None:
            _ro_attn = getattr(self.wrapped_layer, "self_attn", None)
            if _ro_attn is not None:
                # Reset the stash at the START (grad-ckpt safe; mirrors inattn).
                _ro_attn._inattn_kv = None
                # (B4) expose the shared GistReadout (holds summary_proj) so
                # build_retrieved_kv can summarize sub-blocks when inwindow_summary
                # is on. Cheap attr set; None-safe when feature off.
                _ro_attn._gist_readout_ref = self.gist_readout
            _ro_store = getattr(self.memory_bank, "_rawkv_readout_store", None)
            if (
                _ro_attn is not None
                and _ro_store is not None
                and _ro_store.size() > 0
            ):
                # Kept-chunk SELECTION mode (2026-06-20 dilution fix). Default
                # "gist" keeps the trained-scorer top-k. "reader_attn" picks the
                # kept chunks by the reader's OWN native q.k salience (no trained
                # scorer); "oracle" forces the needle chunk. The chosen indices
                # are passed to retrieve() which HARD-isolates them (gathers only
                # those chunks into attention, excluding the rest -> no dilution).
                _keep_mode = getattr(cfg, "rawkv_keep_set_mode", "gist")
                _keep_override = None
                _topk_for_retrieve = cfg.rawkv_readout_topk_chunks
                if _keep_mode in ("reader_attn", "oracle"):
                    _kk = max(int(cfg.rawkv_readout_topk_chunks), 1)
                    _Cc = _ro_store.gist_src.shape[1] if _ro_store.gist_src is not None else 0
                    if _keep_mode == "oracle":
                        _oc = int(getattr(cfg, "rawkv_oracle_needle_chunk", -1))
                        if 0 <= _oc < _Cc:
                            _keep_override = torch.tensor([_oc], device=hidden_states.device)
                    else:  # reader_attn: score each chunk by native q.k
                        _keep_override = self._reader_attn_keep_set(
                            hidden_states, _ro_store, _ro_attn,
                            getattr(self.wrapped_layer, "input_layernorm", None),
                            position_embeddings, _kk,
                        )
                _ro_ret = self.gist_readout.retrieve(
                    hidden_states, _ro_store,
                    topk_chunks=_topk_for_retrieve,
                    temperature=cfg.rawkv_readout_temp,
                    disable_col_bias=getattr(cfg, "rawkv_disable_col_bias", False),
                    keep_set_override=_keep_override,
                )
                if _ro_ret is not None:
                    _ro_h, _ro_pos, _ro_bias = _ro_ret      # [B,R,d],[B,R],[B,Tq,R]
                    # Eval ablation: zero the trained gist col_bias so the reader
                    # attends raw-KV via its own native q·k only (pure emergent
                    # selection, no trained-scorer log-weight). See config flag.
                    if getattr(cfg, "rawkv_readout_zero_col_bias", False):
                        _ro_bias = torch.zeros_like(_ro_bias)
                    from .inattn_kv import build_retrieved_kv
                    _ro_pre_norm = getattr(
                        self.wrapped_layer, "input_layernorm", None
                    )
                    _roK, _roV = build_retrieved_kv(
                        _ro_attn, _ro_h.to(hidden_states.dtype), _ro_pos,
                        position_embeddings, pre_norm=_ro_pre_norm,
                    )
                    # (B4 in-window summary) When the retrieved KV are summarized
                    # into per-sub-block tokens, R changes (R_raw -> n_sub) so the
                    # per-raw-token col_bias [B,Tq,R_raw] no longer aligns with the
                    # summary columns. Selection now flows through summary_proj (a
                    # trainable bottleneck), not the col_bias log-weight, so drop
                    # the bias for the summary path (stash 2-tuple = no col_bias).
                    _summary_on = (
                        bool(getattr(_ro_attn, "_rawkv_inwindow_summary", False))
                        and getattr(self.gist_readout, "summary_proj", None) is not None
                        and _roK.shape[2] != _ro_h.shape[1]   # confirm summarization happened
                    )
                    if _summary_on:
                        _ro_bias = None
                    # Grad-flow probe (mirrors inattn): retain grad on the bias so
                    # the smoke can assert the gist scorer's path is in-graph.
                    if (
                        getattr(self, "_inattn_grad_probe", False)
                        and _ro_bias is not None
                        and _ro_bias.requires_grad
                    ):
                        _ro_bias.retain_grad()
                        self._last_rawkv_readout_bias = _ro_bias
                    if _ro_bias is None:
                        # Summary path: 2-tuple (no col_bias) — selection is via
                        # the trainable summary_proj bottleneck, not col_bias.
                        _ro_attn._inattn_kv = (_roK, _roV)
                    else:
                        _ro_attn._inattn_kv = (_roK, _roV, _ro_bias)
                    self._last_rawkv_readout_R = int(_roK.shape[2])
                    self._last_rawkv_readout_fired = True

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
        # NOTE (2026-06-18): the in-attention KV stash is NOT cleared here. It is
        # reset to None at the START of the next forward's inattn block (see
        # above) — clearing it here breaks gradient-checkpoint recompute (the
        # recomputed wrapped_layer forward would see R=0 vs. the saved R>0).
        if ext_h.shape[1] != k_l3 + k_l2 + k_slots + k_ev + T:
            raise RuntimeError(
                f"expected wrapped layer output length {k_l3+k_l2+k_slots+k_ev+T}, "
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
        # Skip the evidence block (k_ev) between L1 and H when slicing the body.
        slot_delta = ext_h[:, l1_start + k_slots + k_ev:, :] - bypass_h  # [B, T, d]
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
            # EXP-D (2026-06-12): dead-slot read-mass diagnostic. _cum_usage is
            # [B, cfg.num_slots] and only layer-0 tracks it; xattn_slots is the
            # SAME cfg.num_slots in identical index order (memory_bank.get() →
            # [B, N, slot_dim], global slots are the last indices WITHIN
            # num_slots). So (_cum_usage == 0) aligns with the first-N read
            # columns. Non-owner layers (_cum_usage is None) pass None → no-op.
            _dead_mask = (
                (self._cum_usage == 0)
                if self._cum_usage is not None
                else None
            )
            # Per-slot token-mass readout bias (2026-06-15). slot_token_mass is
            # [B, cfg.num_slots] in the SAME index order as xattn_slots (it is
            # accumulated layer-0-only on the shared bank, mirroring _cum_usage),
            # so it aligns with the first-N read columns. Passed when EITHER the
            # fixed (use_readout_mass_bias) OR the learnable per-head written-ness
            # bias (use_learnable_mass_bias, 2026-06-23) is on AND the mass has
            # been materialised; otherwise None → read() adds no bias and the
            # softmax is byte-identical to P8/P11.
            _read_mass = (
                getattr(self.memory_bank, "slot_token_mass", None)
                if (cfg.use_readout_mass_bias or cfg.use_learnable_mass_bias)
                else None
            )
            _routing_keys = None
            if cfg.use_shared_addressing:
                _slots_for_key = (
                    slots
                    if getattr(self.selector, "_no_detach_slots", False)
                    else slots.detach()
                )
                if getattr(self.selector, "_independent_slot_key", False):
                    _routing_keys = F.normalize(
                        self.selector.slot_key_param.unsqueeze(0), dim=-1
                    ).expand(B, -1, -1)
                else:
                    _routing_keys = F.normalize(
                        self.selector.K_sel(_slots_for_key)
                        + self.selector.slot_key_bias.unsqueeze(0),
                        dim=-1,
                    )
                if _routing_keys is None:
                    _routing_keys = getattr(self.selector, "_last_routing_k", None)
            memory_xattn_out = self.memory_xattn.read(
                hidden_states, xattn_slots, xattn_slots,
                dead_mask=_dead_mask,
                mass=_read_mass,
                routing_keys=_routing_keys,
                # The fixed log1p(mass) bias in read() fires whenever mass is not
                # None. When ONLY the learnable per-head bias is on (fixed off) we
                # still pass mass (the learnable path needs it) but must neutralise
                # the fixed path → mass_coef=0.0 makes its added bias exactly 0,
                # leaving only the learnable per-head term. When the fixed flag is
                # on, use its configured coef.
                mass_coef=(cfg.readout_mass_coef if cfg.use_readout_mass_bias else 0.0),
            )                                                       # [B, T, d]
            # v20 (2026-06-12): accumulate per-slot read-mass into the layer-0
            # cumulative + windowed accumulators (no_grad telemetry; index order
            # matches _cum_usage since xattn_slots is the SAME cfg.num_slots in
            # identical order). Guarded for shape / None — never raises.
            if (
                self._layer_idx == 0
                and self._cum_read_mass is not None
                and getattr(self.memory_xattn, "_last_read_mass_per_slot", None)
                is not None
            ):
                with torch.no_grad():
                    _rm = self.memory_xattn._last_read_mass_per_slot
                    if (
                        _rm.dim() == 2
                        and _rm.shape[0] == self._cum_read_mass.shape[0]
                        and _rm.shape[1] == self._cum_read_mass.shape[1]
                    ):
                        _rm = _rm.to(
                            device=self._cum_read_mass.device,
                            dtype=self._cum_read_mass.dtype,
                        )
                        self._cum_read_mass += _rm
                        if self._recent_read_mass is not None:
                            self._recent_read_mass += _rm

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
                        delta_rule=self._use_delta_rule_writeback,
                        delta_erase_write=self._delta_erase_write,
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
                            delta_rule=self._use_delta_rule_writeback,
                            delta_erase_write=self._delta_erase_write,
                        )
                    else:
                        # v8-A (or v7 when global_slot_forget_bias==forget_bias_init): dual gate
                        self.memory_bank.write(
                            _idx_glob,
                            O_mem_slot[:, _k_reg:, :],
                            gate=g_in[:, _k_reg:, :],
                            forget_gate=g_forget[:, _k_reg:, :],
                            tanh_new=cfg.dual_gate_tanh_new,
                            delta_rule=self._use_delta_rule_writeback,
                            delta_erase_write=self._delta_erase_write,
                        )
                else:
                    M_write = self.memory_bank.write(
                        idx,
                        O_mem_slot,
                        gate=g_in,
                        forget_gate=g_forget,
                        tanh_new=cfg.dual_gate_tanh_new,
                        delta_rule=self._use_delta_rule_writeback,
                        delta_erase_write=self._delta_erase_write,
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
                    delta_rule=self._use_delta_rule_writeback,
                    delta_erase_write=self._delta_erase_write,
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
                    delta_rule=self._use_delta_rule_writeback,
                    delta_erase_write=self._delta_erase_write,
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

            # ---- EXP-W2 (2026-06-11): DENSE all-slot soft delta-write ----
            # AFTER (in addition to) the top-k hard write above, apply a weak
            # delta-rule nudge to ALL N slots, each toward its OWN per-slot
            # DISTINCT content (slots-as-query attention over the chunk tokens):
            #     slot_n ← slot_n + λ·g_n·(content_n − slot_n)   (g_n ≡ 1 here)
            # This is the "native" dense fast-weight write (DeltaNet/Titans):
            # dead slots that never enter the top-k idx still receive a slow
            # trickle of their own content, breaking the rich-get-richer
            # deadlock. Per-slot DISTINCT content (NOT a pooled/broadcast vector)
            # avoids the arm4 homogenisation trap. λ small (≤0.05) so live
            # slots' precise content drifts negligibly → the all-N read peak is
            # preserved. ISOLATION: gated by soft_write_weight>0, lives OUTSIDE
            # the _do_recycle block, and uses a NEW MemoryBank.soft_write (does
            # NOT touch write()/force_write()/recycle_reset()), so EXP-R1 and
            # EXP-W2 compose as independent on/off switches. When weight<=0 (or
            # the module is absent) this block is a no-op → byte-identical to P11.
            # Skipped automatically at eval question-time (bank frozen →
            # soft_write returns early), matching the force_write semantics.
            if (
                self._soft_write_weight > 0.0
                and self.soft_write_content_mod is not None
                and not self.memory_bank.frozen
                and self.memory_bank.slots is not None
            ):
                # Per-slot distinct content from the CURRENT chunk tokens
                # (hidden_states, [B, T, d_model]); slots act as queries.
                _sw_content = self.soft_write_content_mod(
                    self.memory_bank.slots, hidden_states
                )                                                  # [B, N, slot_dim]
                # g_n ≡ 1.0 (gate=None): a clean single-variable λ-only arm. A
                # learnable per-slot gate (Titans-style forget) is the reserved
                # candidate-4 refinement if homogenisation appears.
                self.memory_bank.soft_write(
                    _sw_content, weight=self._soft_write_weight, gate=None
                )

        # ---- Per-slot raw-KV cache: WRITE (2026-06-22) ----
        # Append AFTER the current layer read/write so a target chunk cannot read
        # its own just-cached raw tokens in the same forward. Context chunks then
        # become available to later chunks via their selected slot ids. No capacity
        # limit / replacement: this deliberately measures the upper bound.
        if (
            self._is_slot_kv_cache_layer
            and cfg.enable_writeback
            and not cfg.disable_l1_inject
            and not _skip_wb
            and idx is not None
            and not self.memory_bank.frozen
        ):
            _sk_pos = torch.arange(
                T, device=hidden_states.device, dtype=torch.long
            ).unsqueeze(0).expand(B, -1)
            self.memory_bank.append_slot_kv_cache(
                idx.long(), hidden_states.detach(), token_pos=_sk_pos,
                token_mask=_active_token_mask,
            )

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
