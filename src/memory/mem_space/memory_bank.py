"""Memory-Space v0 — per-layer, per-sample slot bank.

See `ops/research_notes/20260426_memory_space_design_direction.md` §2.1 / §2.3
for the slot-identity-stability argument: selected slots are updated
*in place*, unselected slots are untouched, so slot position == slot identity
across time.

Design decisions
----------------
* ``slots`` is **not** a ``nn.Parameter``.  We want the caller (a segmented
  training loop) to decide when to reset / detach, so the tensor lives as
  "state" on the module and is updated functionally per forward.
* Lazy initialisation: on the first forward after construction (or after a
  manual ``reset``), we initialise from a pooled copy of the current hidden
  states + small gaussian noise, which keeps the slots on the target layer's
  hidden-state manifold and sidesteps the cold-start problem (see design
  doc §2.2).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Memory bank
# --------------------------------------------------------------------------- #


class MemoryBank(nn.Module):
    """Stateful slot bank for a single transformer layer.

    Shape: ``slots: Tensor[B, N, slot_dim]``.  ``None`` until the first
    ``init_from_hidden`` or ``reset`` call.

    The bank is *not* a Parameter — it is per-forward state.  Callers are
    expected to:

        * call ``reset(batch_size)`` at the start of a segmented rollout (this
          forces re-init on the next forward), OR
        * call ``detach_()`` between segments they don't want to backprop
          through.

    Writeback is in-place on the selected positions only (EMA), matching the
    "slot identity" property from §2.3 of the design note.
    """

    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        *,
        init_noise: float = 0.02,
        slot_init: str = "hidden_pool",
        slot_value_norm_cap: float = 0.0,
        evidence_buffer_size: int = 0,
        evidence_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if num_slots <= 0:
            raise ValueError(f"num_slots must be > 0, got {num_slots}")
        if slot_dim <= 0:
            raise ValueError(f"slot_dim must be > 0, got {slot_dim}")
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.init_noise = float(init_noise)
        self.slot_init = slot_init
        self._slot_value_norm_cap = float(slot_value_norm_cap)

        # Slot-Routed Evidence Memory (2026-06-17). Per-slot buffer of UNCOMPRESSED
        # original-token hidden states ([d_model]) routed into the slot, so the
        # readout can recall precise facts the compressed latent loses. Bcnt =
        # evidence_buffer_size entries per slot; evidence_dim = the hidden dim the
        # evidence tokens live in (== d_model, NOT slot_dim, since they are灌进
        # the wrapped frozen decoder as a joint-attention prefix). 0 / None =
        # disabled → every evidence code path is a no-op (byte-identical to pre).
        self.evidence_buffer_size = int(evidence_buffer_size)
        self.evidence_dim = int(evidence_dim) if evidence_dim is not None else None
        # Lazily materialised on the first write_evidence call (like slot_token_mass).
        #   slot_evidence       : [B, N, Bcnt, evidence_dim]  bf16 token hidden states
        #   slot_evidence_score : [B, N, Bcnt]   salience of each stored entry
        #                         (init -inf so empty slots lose to any candidate)
        #   slot_evidence_count : [B, N]   int   #valid entries per slot (telemetry)
        self.slot_evidence: Optional[torch.Tensor] = None
        self.slot_evidence_score: Optional[torch.Tensor] = None
        self.slot_evidence_count: Optional[torch.Tensor] = None

        # Stateful buffer, lazily materialised on first forward.
        # We deliberately keep this as a plain attribute (not register_buffer)
        # so ``.state_dict()`` does not try to checkpoint/transport it — slots
        # are runtime state, not weights.
        self.slots: Optional[torch.Tensor] = None
        # Tracks the batch size we're currently initialised for; used to
        # auto-reinit when the caller changes batch size.
        self._batch_size: Optional[int] = None
        # Per-slot token-mass (2026-06-15): [B, num_slots] float, no_grad. Tracks
        # the cumulative number of REAL tokens each slot has absorbed over the
        # whole sample. Lazily materialised on the first add_token_mass call
        # (or stays None when use_readout_mass_bias is off → zero overhead).
        # Semantics: routing is PER-CHUNK (the selector pools the whole chunk
        # into [B,N] logits → top-k → idx:[B,k]; every selected slot receives a
        # write aggregated over the chunk's tokens). So "tokens slot n absorbed
        # this chunk" = (real-token count of the chunk) iff slot n was selected,
        # else 0. Accumulated layer-0-only on the SHARED bank (mirrors the
        # _cum_usage / dead_mask / _cum_read_mass layer-0 telemetry the read
        # already aligns to), so it is NOT 32x over-counted by the per-layer
        # write calls. Resets to None with the bank; recycled rows are zeroed.
        self.slot_token_mass: Optional[torch.Tensor] = None
        # When True, write() is a no-op — used during greedy generation to
        # prevent question tokens from overwriting haystack-accumulated slots.
        self.frozen: bool = False

    # --------------------------------------------------------------------- #
    # Lifecycle
    # --------------------------------------------------------------------- #

    def reset(self, batch_size: Optional[int] = None) -> None:
        """Force a re-init on the next ``init_from_hidden`` call."""
        self.slots = None
        self._batch_size = batch_size
        # Token-mass is per-sample state, like slots — drop it at the document /
        # rollout boundary so a fresh sample starts every slot at zero mass.
        self.slot_token_mass = None
        # Evidence buffers are per-sample state too — drop at the rollout boundary.
        self.slot_evidence = None
        self.slot_evidence_score = None
        self.slot_evidence_count = None

    def detach_(self) -> None:
        """Break the autograd graph across a segment boundary.

        Use this between segments you don't want to backprop through (the
        common case for a long-rollout training loop).
        """
        if self.slots is not None:
            self.slots = self.slots.detach()

    def is_initialized(self, batch_size: int) -> bool:
        return self.slots is not None and self._batch_size == batch_size

    # --------------------------------------------------------------------- #
    # Init
    # --------------------------------------------------------------------- #

    def init_from_hidden(
        self,
        H_l: torch.Tensor,
        batch_size: int,
    ) -> None:
        """Initialise ``slots`` from a pooled copy of ``H_l`` + noise.

        Args:
            H_l: [B, T, d] current-layer hidden states.  Only used when
                ``slot_init == "hidden_pool"``; for zero/random init we just
                take dtype/device from this tensor.
            batch_size: explicit batch size (matches ``H_l.shape[0]``).

        The tensor is created with the same dtype / device as ``H_l`` but is
        not attached to its graph (slots are state, not a differentiable
        function of the first-batch hidden states).
        """
        if H_l.dim() != 3:
            raise ValueError(f"H_l must be [B, T, d], got {tuple(H_l.shape)}")
        if H_l.shape[0] != batch_size:
            raise ValueError(
                f"batch_size={batch_size} does not match H_l.shape[0]={H_l.shape[0]}"
            )
        if H_l.shape[-1] != self.slot_dim:
            raise ValueError(
                f"H_l hidden dim {H_l.shape[-1]} != slot_dim {self.slot_dim}. "
                "Slot projections must be handled outside MemoryBank."
            )

        B, _, d = H_l.shape
        N = self.num_slots
        device, dtype = H_l.device, H_l.dtype

        if self.slot_init == "zero":
            slots = torch.zeros(B, N, d, device=device, dtype=dtype)
        elif self.slot_init == "random":
            slots = torch.randn(B, N, d, device=device, dtype=dtype) * self.init_noise
        elif self.slot_init == "hidden_pool":
            # Mean-pool over tokens, broadcast to N slots, add small gaussian
            # noise so slots are not initially identical (otherwise the
            # selector has nothing to pick from).
            pooled = H_l.detach().mean(dim=1, keepdim=True)           # [B, 1, d]
            slots = pooled.expand(B, N, d).contiguous().clone()
            if self.init_noise > 0.0:
                slots = slots + torch.randn_like(slots) * self.init_noise
        elif self.slot_init == "strided_token":
            # Fix K (2026-04-29): each slot gets a distinct evenly-spaced token.
            # For N=512, T=4096: stride=8, slot i = token (i*8) % T.
            # Handles T < N by repeating modulo. Diversity ~256x better than hidden_pool.
            T_len = H_l.shape[1]
            indices = torch.arange(N, device=device) * max(1, T_len // N)
            indices = indices % T_len                             # [N]
            slots = H_l.detach()[:, indices, :].clone()          # [B, N, d]
            if self.init_noise > 0.0:
                slots = slots + torch.randn_like(slots) * self.init_noise
            # Slot 0 = last-token summary (full-chunk receptive field under SWA)
            slots[:, 0, :] = H_l.detach()[:, -1, :]
        else:  # pragma: no cover — validated in config
            raise ValueError(f"unknown slot_init: {self.slot_init!r}")

        self.slots = slots.detach()
        self._batch_size = batch_size

    # --------------------------------------------------------------------- #
    # Token-mass (2026-06-15): per-slot cumulative real-token count.
    # --------------------------------------------------------------------- #

    def add_token_mass(
        self,
        sel: torch.Tensor,
        token_counts: torch.Tensor,
    ) -> None:
        """Accumulate per-slot token mass for the current chunk (no_grad).

        Args:
            sel: [B, N] tensor (0/1 or bool) — which slots were selected this
                chunk (deduped; global-slot duplicates clamped to 1). Index
                order MUST match ``slots`` (the same N-slot order the selector
                and read path use).
            token_counts: [B] (or [B, 1]) tensor — the number of REAL (non-pad)
                tokens in this chunk, per sample. A selected slot gains exactly
                ``token_counts[b]`` mass this chunk (routing is per-chunk: a slot
                that wins the chunk's top-k absorbs a write aggregated over the
                whole chunk's tokens — see class/__init__ note).

        Lazily materialises ``slot_token_mass`` [B, N] on first call (after the
        bank is initialised) and reallocates on a batch-size / N change. Pure
        no_grad state mutation — never enters the autograd graph (the readout
        bias consumes a detached copy). No-op while ``frozen`` (e.g. eval
        question-time generation) so question tokens don't inflate mass, exactly
        like ``write`` short-circuits when frozen.
        """
        if self.frozen or self.slots is None:
            return
        with torch.no_grad():
            B, N = self.slots.shape[0], self.slots.shape[1]
            if sel.shape[0] != B or sel.shape[1] != N:
                # Shape mismatch (e.g. stale counter) — skip silently rather
                # than corrupt the accumulator or raise in the hot path.
                return
            if (
                self.slot_token_mass is None
                or self.slot_token_mass.shape[0] != B
                or self.slot_token_mass.shape[1] != N
            ):
                self.slot_token_mass = torch.zeros(
                    B, N, device=self.slots.device, dtype=torch.float32
                )
            _sel = sel.to(device=self.slot_token_mass.device, dtype=torch.float32)
            _tc = token_counts.to(
                device=self.slot_token_mass.device, dtype=torch.float32
            )
            if _tc.dim() == 1:
                _tc = _tc.unsqueeze(-1)                       # [B, 1]
            self.slot_token_mass = self.slot_token_mass + _sel * _tc

    def zero_token_mass(self, mask: torch.Tensor) -> None:
        """Zero the token mass of the masked (recycled/evicted) slots (no_grad).

        ``mask``: [B, N] bool — True where the slot was just recycled / evicted
        and its accumulated history should be discarded (the slot now holds
        brand-new content, so its old token mass no longer describes it).
        No-op when mass is uninitialised or the shape mismatches.
        """
        if self.slot_token_mass is None:
            return
        with torch.no_grad():
            m = mask.to(device=self.slot_token_mass.device)
            if m.dim() != 2 or m.shape != self.slot_token_mass.shape:
                return
            self.slot_token_mass = self.slot_token_mass.masked_fill(m.bool(), 0.0)

    # --------------------------------------------------------------------- #
    # Slot-Routed Evidence Memory (2026-06-17).
    # --------------------------------------------------------------------- #

    def write_evidence(
        self,
        slot_idx: torch.Tensor,
        token_hidden: torch.Tensor,
        token_score: torch.Tensor,
    ) -> None:
        """Append the highest-salience routed tokens into the per-slot evidence
        buffer, keeping the top ``Bcnt`` by salience (priority replacement).

        Args:
            slot_idx: [B, k] long — the slots that won this chunk's top-k (the
                slots whose evidence buffer should receive candidates).
            token_hidden: [B, k, C, evidence_dim] — for each selected slot, C
                candidate token hidden states routed into it this chunk.
            token_score: [B, k, C] — salience (routing weight) of each candidate.

        Merge policy (MVP): for each selected slot, concatenate the C candidates
        with the slot's existing ``Bcnt`` stored entries along the entry axis,
        then keep the top-``Bcnt`` by score. Empty entries carry score -inf so
        any real candidate wins over them. Pure no_grad state mutation (evidence
        is FROZEN hidden states — it never enters the autograd graph). No-op when
        evidence is disabled, the bank is frozen, or uninitialised.
        """
        if self.evidence_buffer_size <= 0 or self.frozen or self.slots is None:
            return
        if slot_idx.dim() != 2 or token_hidden.dim() != 4 or token_score.dim() != 3:
            return
        with torch.no_grad():
            B, N = self.slots.shape[0], self.slots.shape[1]
            Bcnt = self.evidence_buffer_size
            d_ev = token_hidden.shape[-1]
            if self.evidence_dim is None:
                self.evidence_dim = int(d_ev)
            if slot_idx.shape[0] != B:
                return
            k = slot_idx.shape[1]
            if token_hidden.shape[0] != B or token_hidden.shape[1] != k:
                return
            C = token_hidden.shape[2]
            dev = self.slots.device
            # Lazy / reshape-safe (re)allocation of the evidence buffers.
            if (
                self.slot_evidence is None
                or self.slot_evidence.shape[0] != B
                or self.slot_evidence.shape[1] != N
                or self.slot_evidence.shape[2] != Bcnt
                or self.slot_evidence.shape[3] != self.evidence_dim
            ):
                self.slot_evidence = torch.zeros(
                    B, N, Bcnt, self.evidence_dim, device=dev, dtype=self.slots.dtype
                )
                self.slot_evidence_score = torch.full(
                    (B, N, Bcnt), float("-inf"), device=dev, dtype=torch.float32
                )
                self.slot_evidence_count = torch.zeros(
                    B, N, device=dev, dtype=torch.long
                )
            _idx = slot_idx.to(device=dev, dtype=torch.long)            # [B, k]
            _cand_h = token_hidden.to(device=dev, dtype=self.slot_evidence.dtype)
            _cand_s = token_score.to(device=dev, dtype=torch.float32)   # [B, k, C]
            # Gather the existing entries for the selected slots.
            _idx_e = _idx.unsqueeze(-1).unsqueeze(-1).expand(B, k, Bcnt, self.evidence_dim)
            existing_h = self.slot_evidence.gather(1, _idx_e)           # [B, k, Bcnt, d]
            _idx_s = _idx.unsqueeze(-1).expand(B, k, Bcnt)
            existing_s = self.slot_evidence_score.gather(1, _idx_s)     # [B, k, Bcnt]
            # Concatenate existing + candidate, then keep top-Bcnt by score.
            merged_h = torch.cat([existing_h, _cand_h], dim=2)         # [B, k, Bcnt+C, d]
            merged_s = torch.cat([existing_s, _cand_s], dim=2)         # [B, k, Bcnt+C]
            top_s, top_i = torch.topk(merged_s, k=Bcnt, dim=2)        # [B, k, Bcnt]
            top_h = merged_h.gather(
                2, top_i.unsqueeze(-1).expand(B, k, Bcnt, self.evidence_dim)
            )                                                          # [B, k, Bcnt, d]
            # Scatter the kept entries back into the global buffer at the
            # selected slots. NOTE: with duplicate slot indices (e.g. global
            # slots) scatter keeps the last write — acceptable for an MVP.
            self.slot_evidence.scatter_(1, _idx_e, top_h)
            self.slot_evidence_score.scatter_(1, _idx_s, top_s)
            # Update valid-entry count (telemetry): #finite scores per slot.
            new_count = (top_s > float("-inf")).sum(dim=2).to(torch.long)  # [B, k]
            self.slot_evidence_count.scatter_(1, _idx, new_count)

    def zero_evidence(self, mask: torch.Tensor) -> None:
        """Invalidate the evidence of the masked (recycled / evicted) slots.

        ``mask``: [B, N] bool — True where the slot was just recycled and its
        cached evidence no longer describes its content. Resets those rows'
        entries to zero hidden / -inf score / 0 count. No-op when evidence is
        uninitialised or the shape mismatches.
        """
        if self.slot_evidence is None:
            return
        with torch.no_grad():
            m = mask.to(device=self.slot_evidence.device)
            if m.dim() != 2 or m.shape != self.slot_evidence.shape[:2]:
                return
            mb = m.bool()
            self.slot_evidence = self.slot_evidence.masked_fill(
                mb.unsqueeze(-1).unsqueeze(-1), 0.0
            )
            self.slot_evidence_score = self.slot_evidence_score.masked_fill(
                mb.unsqueeze(-1), float("-inf")
            )
            self.slot_evidence_count = self.slot_evidence_count.masked_fill(mb, 0)

    # --------------------------------------------------------------------- #
    # Access
    # --------------------------------------------------------------------- #

    def get(self) -> torch.Tensor:
        """Return the current slot tensor.  Raises if uninitialised."""
        if self.slots is None:
            raise RuntimeError(
                "MemoryBank.get() called before init_from_hidden(); "
                "the wrapping MemorySpaceLayer should lazy-init on first forward."
            )
        return self.slots

    # --------------------------------------------------------------------- #
    # EXP-R1 (2026-06-11): dead-slot recycling — grad-preserving row ops.
    # --------------------------------------------------------------------- #

    def recycle_reset(
        self,
        dead_mask: torch.Tensor,
        new_content: torch.Tensor,
    ) -> None:
        """Stage-① RESET: overwrite ONLY the dead-slot rows with fresh content.

        ``dead_mask``  : [B, N] bool — True where the slot is "dead" (never
                         selected in the last reset window) and should be
                         re-initialised.
        ``new_content``: [B, N, slot_dim] — the replacement content for every
                         row (only the dead rows are actually used). Caller is
                         responsible for choosing diverse, on-manifold content
                         (strided current-chunk tokens, NOT pooled-mean).

        Implementation note (graph safety): we use ``torch.where`` instead of an
        in-place / no_grad scatter so the LIVE rows keep their autograd graph
        (cross-chunk BPTT within a bptt_window must survive). The dead rows take
        ``new_content.detach()`` — a fresh re-initialisation, like
        ``init_from_hidden`` which is also detached. A no_grad reassignment of
        ``self.slots`` would detach the WHOLE tensor and sever the live slots'
        cross-chunk gradient, which is why we route through ``torch.where``.
        """
        if self.frozen or self.slots is None:
            return
        if dead_mask.dim() != 2 or new_content.dim() != 3:
            raise ValueError(
                f"dead_mask must be [B,N], new_content [B,N,d]; got "
                f"dead_mask={tuple(dead_mask.shape)} new_content={tuple(new_content.shape)}"
            )
        _m = dead_mask.to(device=self.slots.device).unsqueeze(-1)          # [B,N,1]
        _new = new_content.detach().to(device=self.slots.device, dtype=self.slots.dtype)
        self.slots = torch.where(_m, _new, self.slots)
        # A recycled slot now holds brand-new content, so its accumulated token
        # mass no longer describes it — zero the masked rows (no-op if mass off).
        self.zero_token_mass(dead_mask)
        # Its cached evidence is likewise stale — invalidate it (no-op if off).
        self.zero_evidence(dead_mask)
        # Keep the global norm cap consistent with the gated write paths
        # (gradient-preserving fixed-ratio clamp; only shrinks).
        if self._slot_value_norm_cap > 0.0:
            slot_norms_all = self.slots.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scale_all = (slot_norms_all / self._slot_value_norm_cap).clamp(min=1.0).detach()
            self.slots = self.slots / scale_all

    def force_write(
        self,
        force_mask: torch.Tensor,
        content: torch.Tensor,
        beta,
    ) -> None:
        """Stage-② GRACE forced delta-write on the recycled (masked) rows.

        Delta-rule nudge of ONLY the masked rows toward ``content``:

            slot_row ← slot_row + beta · (content_row − slot_row)

        ``force_mask``: [B, N] bool — the slots just recycled (in the grace
                        window) that must receive a forced write this chunk.
        ``content``   : [B, N, slot_dim] — per-slot target content (strided
                        current-chunk tokens; per-slot distinct, on-manifold).
        ``beta``      : float / 0-d tensor blend rate.

        This is WRITE-ONLY: it mutates stored content but does NOT touch the
        selector or the read path. Under ``use_memory_xattn`` the read attends
        ALL slots through a separate softmax, so refreshing a recycled slot's
        content lets it later be READ purely by natural content match — never
        forced — which is the key discriminator vs. the rejected ROUTE-A arm4
        (which forced WHICH slots are read and collapsed top1_sim).

        Graph safety: same ``torch.where`` rationale as ``recycle_reset`` — the
        blended value is a function of ``self.slots`` so every row (live AND
        recycled) keeps its graph; only the recycled rows are nudged.
        """
        if self.frozen or self.slots is None:
            return
        if force_mask.dim() != 2 or content.dim() != 3:
            raise ValueError(
                f"force_mask must be [B,N], content [B,N,d]; got "
                f"force_mask={tuple(force_mask.shape)} content={tuple(content.shape)}"
            )
        if isinstance(beta, torch.Tensor):
            _b = beta.to(device=self.slots.device, dtype=self.slots.dtype)
        else:
            _b = float(beta)
        _c = content.to(device=self.slots.device, dtype=self.slots.dtype)
        blended = self.slots + _b * (_c - self.slots)
        _m = force_mask.to(device=self.slots.device).unsqueeze(-1)         # [B,N,1]
        self.slots = torch.where(_m, blended, self.slots)
        if self._slot_value_norm_cap > 0.0:
            slot_norms_all = self.slots.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scale_all = (slot_norms_all / self._slot_value_norm_cap).clamp(min=1.0).detach()
            self.slots = self.slots / scale_all

    def soft_write(
        self,
        content: torch.Tensor,
        weight,
        gate=None,
    ) -> None:
        """EXP-W2 (2026-06-11): DENSE all-slot soft delta-write.

        Applies a *weak* delta-rule nudge to **every** slot row (no mask /
        no top-k):

            slot_n ← slot_n + weight · g_n · (content_n − slot_n)

        This is the "native" dense fast-weight write (DeltaNet / Titans style):
        every memory cell is touched each chunk, weighted by a small global λ
        (``weight``) and an optional per-slot gate ``g_n``. It is meant to run
        IN ADDITION to the existing top-k hard ``write`` (which is left
        completely untouched) so that dead slots receive a slow trickle of their
        OWN per-slot-distinct content and can re-enter the routing distribution,
        breaking the cold-start "rich-get-richer" deadlock.

        Isolation guarantee: this method is INDEPENDENT of ``write`` (top-k hard
        write), ``force_write`` (EXP-R1 grace write), and ``recycle_reset``
        (EXP-R1 reset). It mirrors ``force_write``'s graph-safe blend but with
        ``mask = ALL N rows`` and ``weight = λ``. Callers gate it on
        ``soft_write_weight > 0``; when off, it is never invoked → P11 behaviour
        is byte-identical.

        Args:
            content: [B, N, slot_dim] per-slot target content (per-slot DISTINCT,
                e.g. slots-as-query write-attention over the chunk tokens). Must
                NOT be a broadcast/pooled vector — undifferentiated content
                homogenises the bank (the arm4 failure).
            weight:  λ — float / 0-d tensor global soft-write rate (small, e.g.
                0.02-0.05).
            gate:    optional per-slot gate. None → 1.0 (pure λ delta). A scalar,
                [B, N, 1], or [B, N, slot_dim] tensor is broadcast against the
                residual.

        Graph safety: same rationale as ``force_write`` / ``recycle_reset`` —
        the blended value is a function of ``self.slots`` so every row keeps its
        autograd graph (cross-chunk BPTT within a bptt_window survives). The
        norm cap (if any) is applied gradient-preservingly (detached scale).
        """
        if self.frozen or self.slots is None:
            return
        if content.dim() != 3:
            raise ValueError(
                f"content must be [B,N,d]; got {tuple(content.shape)}"
            )
        if content.shape[1] != self.num_slots or content.shape[-1] != self.slot_dim:
            raise ValueError(
                f"content shape {tuple(content.shape)} incompatible with "
                f"(num_slots={self.num_slots}, slot_dim={self.slot_dim})"
            )
        if isinstance(weight, torch.Tensor):
            _w = weight.to(device=self.slots.device, dtype=self.slots.dtype)
        else:
            _w = float(weight)
        _c = content.to(device=self.slots.device, dtype=self.slots.dtype)
        if gate is None:
            _eff = _w
        elif isinstance(gate, torch.Tensor):
            _eff = _w * gate.to(device=self.slots.device, dtype=self.slots.dtype)
        else:
            _eff = _w * float(gate)
        # Dense delta-rule update over ALL rows (no scatter/mask): every slot is
        # nudged toward its own content_n by the (per-slot-gated) global rate.
        self.slots = self.slots + _eff * (_c - self.slots)
        if self._slot_value_norm_cap > 0.0:
            slot_norms_all = self.slots.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scale_all = (slot_norms_all / self._slot_value_norm_cap).clamp(min=1.0).detach()
            self.slots = self.slots / scale_all

    def decay_(self, factor) -> None:
        """v20 (2026-06-12): Arm A soft read-decay — in-place value scaling.

        Multiplies every slot's VALUE by a per-slot ``factor`` (detached,
        no_grad), shrinking the norm of slots that have not been read recently:

            slot_n ← slot_n · factor_n

        ``factor``: float / 0-d tensor (uniform), or [B, N] / [B, N, 1] tensor
                    (per-slot). Broadcast against ``slots`` [B, N, slot_dim].

        This is NOT a hard delete — a value scaled by e.g. 0.5 still carries its
        direction and can be read or revived by a later write. Guarded on
        ``frozen`` / ``None`` like ``recycle_reset`` / ``force_write``; uses a
        no_grad reassignment (the factor is detached telemetry-derived state, so
        there is no gradient to preserve through it — the decay is a state
        mutation analogous to the recycle ops, not a differentiable transform).
        """
        if self.frozen or self.slots is None:
            return
        with torch.no_grad():
            if isinstance(factor, torch.Tensor):
                _f = factor.to(device=self.slots.device, dtype=self.slots.dtype)
                if _f.dim() == 2:                      # [B, N] -> [B, N, 1]
                    _f = _f.unsqueeze(-1)
            else:
                _f = float(factor)
            self.slots = (self.slots * _f).detach()
            if self._slot_value_norm_cap > 0.0:
                slot_norms_all = self.slots.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                scale_all = (slot_norms_all / self._slot_value_norm_cap).clamp(min=1.0).detach()
                self.slots = self.slots / scale_all

    def write(
        self,
        idx: torch.Tensor,
        new_repr: torch.Tensor,
        gate,
        *,
        forget_gate: Optional[torch.Tensor] = None,
        tanh_new: bool = False,
        replace: bool = False,
        delta_rule: bool = False,
        delta_erase_write: bool = False,
    ):
        """In-place EMA writeback on the selected slot positions.

        Three modes:

        **Replacement (v6/v7)** when ``replace=True``:

            slots[idx] = new_repr  (no EMA, direct overwrite)

        **Single-gate (legacy, H/H5/H3)** when ``forget_gate is None``:

            slots[idx] = (1 - gate) · slots[idx] + gate · new_repr

        ``gate`` is a Python float or 0-dim Tensor (broadcast scalar).

        **Dual-gate (H6, LM2-inspired)** when ``forget_gate`` is supplied:

            new = tanh(new_repr) if tanh_new else new_repr
            slots[idx] = gate · new + forget_gate · slots[idx]

        ``gate`` and ``forget_gate`` are full ``[B, k, slot_dim]`` tensors (one
        sigmoid value per feature, per selected slot). They are independent
        — the dual-gate is NOT a (1-β, β) split; LM2 lets both gates move
        freely so the network can choose to "fully remember + fully overwrite"
        or "fully forget + write nothing".

        Args:
            idx: [B, k] long tensor of slot indices to update.
            new_repr: [B, k, slot_dim] new content for each selected slot.
            gate: scalar β (legacy) OR per-feature input gate [B, k, slot_dim] (H6).
            forget_gate: per-feature forget gate [B, k, slot_dim] (H6 only).
            tanh_new: bound new content with tanh before mixing (LM2 default;
                replaces the manual ``max_norm`` clamp when active).
            replace: v6/v7 direct-replacement mode — skip EMA and write new_repr
                directly into the selected slots. Default False = EMA behaviour.

        Returns:
            The gradient-bearing written VALUES for the selected slots
            ``[B, k, slot_dim]`` (the post-write content, BEFORE the global
            ``slot_value_norm_cap`` rebind of ``self.slots``), or
            ``None`` on a no-op (frozen / β≈0). P1/v12 uses this as ``M_write``
            for the summary-reconstruction loss. As of FIX v13 (2026-06-02) the
            norm-cap rebind is gradient-PRESERVING (scaled outside ``no_grad``
            with a detached scale), so both this ``updated`` return AND the
            ``self.slots`` read by the next chunk stay attached to the graph —
            cross-chunk gradient now flows.
        """
        if self.frozen:
            return None
        if self.slots is None:
            raise RuntimeError("MemoryBank.write() called before initialisation.")
        if idx.dim() != 2 or new_repr.dim() != 3:
            raise ValueError(
                f"idx must be [B,k] and new_repr [B,k,d]; got "
                f"idx={tuple(idx.shape)}, new_repr={tuple(new_repr.shape)}"
            )
        B, k = idx.shape
        if new_repr.shape[0] != B or new_repr.shape[1] != k:
            raise ValueError(
                f"new_repr shape {tuple(new_repr.shape)} incompatible with idx {tuple(idx.shape)}"
            )
        if new_repr.shape[-1] != self.slot_dim:
            raise ValueError(
                f"new_repr dim {new_repr.shape[-1]} != slot_dim {self.slot_dim}"
            )

        # ---- REPLACEMENT PATH (v6/v7) ----
        # Direct overwrite: slot ← s_new (no EMA, no gate).
        # Applied BEFORE the dual-gate check so that global-slot replacement in
        # v7 (called with replace=True) also bypasses dual-gate logic cleanly.
        if replace:
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, self.slot_dim)
            updated = new_repr.to(self.slots.dtype)
            # Apply norm cap before scatter to keep slots stable.
            if self._slot_value_norm_cap > 0.0:
                slot_norms = updated.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                scale = (slot_norms / self._slot_value_norm_cap).clamp(min=1.0)
                updated = updated / scale
            self.slots = self.slots.scatter(1, idx_exp, updated)
            return updated

        # ---- DUAL-GATE PATH (H6) ----
        if forget_gate is not None:
            if forget_gate.shape != new_repr.shape:
                raise ValueError(
                    f"forget_gate shape {tuple(forget_gate.shape)} must equal "
                    f"new_repr shape {tuple(new_repr.shape)}"
                )
            if not isinstance(gate, torch.Tensor) or gate.shape != new_repr.shape:
                raise ValueError(
                    "Dual-gate write requires `gate` (the input gate) to also be a "
                    f"[B,k,d] tensor; got {type(gate).__name__} "
                    f"shape={getattr(gate, 'shape', None)}"
                )
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, self.slot_dim)
            current = self.slots.gather(1, idx_exp)                          # [B, k, d]
            new_content = (
                torch.tanh(new_repr) if tanh_new else new_repr
            ).to(self.slots.dtype)
            g_in = gate.to(device=self.slots.device, dtype=self.slots.dtype)
            g_forget = forget_gate.to(device=self.slots.device, dtype=self.slots.dtype)
            if delta_erase_write:
                # DeltaNet-style erase-then-write (associative update). For each
                # selected slot, first ERASE the component of the current slot
                # aligned with the new-content direction k_hat, then WRITE the
                # new value, both scaled by the input gate beta = g_in:
                #   k_hat   = new_content / (||new_content|| + eps)
                #   updated = current - beta·(current·k_hat)·k_hat + beta·new
                # The independent g_forget is intentionally ignored on this path
                # (precedence over the delta_rule / two-gate forms). k_hat is
                # normalized per-slot over the feature dim with an eps for
                # numerical safety. Default off → never entered, byte-identical.
                _eps = 1e-6
                k_hat = new_content / (
                    new_content.norm(dim=-1, keepdim=True) + _eps
                )                                                       # [B, k, d]
                proj = (current * k_hat).sum(dim=-1, keepdim=True)      # [B, k, 1]
                updated = current - g_in * (proj * k_hat) + g_in * new_content
                # FIX (2026-06-13): force back to slots dtype — k_hat division /
                # eps can upcast to fp32, leaving `updated` fp32 while self.slots
                # is bf16 → scatter() dtype-mismatch RuntimeError. Cast guards it.
                updated = updated.to(self.slots.dtype)
            elif delta_rule:
                # P11 (2026-06-06) delta-rule writeback. Instead of the LM2
                # two-independent-gate form (g_in·new + g_forget·current), write
                # the RESIDUAL between the new value and what is already stored,
                # scaled by the input gate only (forget tied to 1 − g_in):
                #   slot_new = current + g_in · (new_content − current)
                # The independent `g_forget` is intentionally ignored here.
                # Default off → this branch is never entered and the dual-gate
                # update below is byte-identical to pre-P11.
                updated = current + g_in * (new_content - current)
            else:
                updated = g_in * new_content + g_forget * current

            # tanh already bounds new_content to [-1, 1] per dim; we no longer
            # need the manual max_norm clamp from H/H5 (that was a band-aid for
            # 32-layer compounding writes blowing up bf16). Keep the global
            # slot_value_norm_cap below as a safety net only.
            self.slots = self.slots.scatter(1, idx_exp, updated)
            # FIX v13 (2026-06-02): gradient-PRESERVING norm cap. The previous
            # `with torch.no_grad(): self.slots = self.slots / scale_all` rebound
            # self.slots to a tensor DETACHED from the graph, so the slots read by
            # the NEXT chunk carried no gradient → cross-chunk grad was severed
            # (the whole "teach the writer to store readable content" signal was
            # killed). We now scale WITHOUT no_grad, detaching only `scale_all`
            # (a fixed-ratio clamp: limits magnitude without altering grad
            # direction), so self.slots stays in the autograd graph.
            if self._slot_value_norm_cap > 0.0:
                slot_norms_all = self.slots.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                scale_all = (slot_norms_all / self._slot_value_norm_cap).clamp(min=1.0).detach()
                self.slots = self.slots / scale_all
            return updated

        # ---- SINGLE-GATE LEGACY PATH (H/H5/H3) ----
        # Tensor-or-float gate (Branch-3 2026-04-26). A tensor gate threads
        # gradient back into gate_param; a float gate preserves the Tier-3
        # zero-BPTT path and still short-circuits when β is exactly 0 for
        # perf parity.
        is_tensor_gate = isinstance(gate, torch.Tensor)
        if not is_tensor_gate and gate <= 0.0:
            return None  # β ≈ 0 → no-op (common during the early warmup phase).

        # Expand idx so we can gather along the N axis: [B, k, 1] -> [B, k, d].
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, self.slot_dim)
        # Read current values at the selected positions (to keep it an EMA
        # rather than an overwrite).  Cast new_repr to the slot dtype so we
        # don't silently upcast the bank (this matters for bf16 finetuning).
        current = self.slots.gather(1, idx_exp)                    # [B, k, d]
        if is_tensor_gate:
            # Keep β on the same device/dtype as slots so the EMA stays in
            # the bank's compute dtype (bf16 in practice). The tensor keeps
            # its grad_fn so gate_param picks up a gradient via β · new_repr.
            gate_t = gate.to(device=self.slots.device, dtype=self.slots.dtype)
            new_contrib = new_repr.to(self.slots.dtype)
            updated = (1.0 - gate_t) * current + gate_t * new_contrib
        else:
            updated = (1.0 - gate) * current + gate * new_repr.to(self.slots.dtype)

        # FIX H (2026-04-29): Slot norm clipping — prevents 32-layer shared_memory_bank
        # compounding writes from causing bf16 overflow. Target norm: sqrt(slot_dim)*2 ≈
        # typical Llama hidden state norm scaled to bf16 safe range.
        slot_dim_f = float(updated.shape[-1])
        max_norm = math.sqrt(slot_dim_f) * 2.0          # e.g. ~90.5 for slot_dim=2048
        slot_norms = updated.norm(dim=-1, keepdim=True)  # [B, k, 1]
        scale = (slot_norms.clamp(max=max_norm) / slot_norms.clamp(min=1e-6))
        # FIX (2026-05-15): keep bf16 dtype after norm-clip multiply.
        # `slot_norms` defaults to fp32 from .norm(), and `updated * scale`
        # promotes back to fp32, breaking the subsequent scatter into bf16
        # `self.slots`. Streaming chunked inference (16k+ context) is the
        # first path that triggers a frozen-bank write outside autocast,
        # which is when the dtype mismatch surfaces. Cast back explicitly.
        updated = (updated * scale).to(self.slots.dtype)

        # scatter writes `updated` into `self.slots` at positions idx_exp
        # along dim=1.  We re-bind ``self.slots`` because ``Tensor.scatter``
        # (non-in-place) preserves the autograd graph; ``scatter_`` would
        # mutate a tensor that is still referenced by an upstream layer's
        # read and leak the graph across layers unsafely. Inter-chunk graph
        # break is handled by ``reset`` and ``detach_`` at chunk boundary.
        self.slots = self.slots.scatter(1, idx_exp, updated)

        # FIX X.1 (2026-04-30): slot value norm cap — applied globally to ALL slots
        # after the scatter, so even unselected slots that accumulated high norms
        # from prior steps are kept in check. Only active when slot_value_norm_cap > 0.
        # FIX v13 (2026-06-02): made gradient-PRESERVING (see dual-gate path above).
        # The old `with torch.no_grad(): self.slots = self.slots / scale_all`
        # detached self.slots, cutting cross-chunk gradient. Now scale outside
        # no_grad with a detached `scale_all` (fixed-ratio clamp).
        if self._slot_value_norm_cap > 0.0:
            slot_norms_all = self.slots.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scale_all = (slot_norms_all / self._slot_value_norm_cap).clamp(min=1.0).detach()
            self.slots = self.slots / scale_all
        return updated
