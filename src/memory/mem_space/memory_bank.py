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

        # Stateful buffer, lazily materialised on first forward.
        # We deliberately keep this as a plain attribute (not register_buffer)
        # so ``.state_dict()`` does not try to checkpoint/transport it — slots
        # are runtime state, not weights.
        self.slots: Optional[torch.Tensor] = None
        # Tracks the batch size we're currently initialised for; used to
        # auto-reinit when the caller changes batch size.
        self._batch_size: Optional[int] = None
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

    def write(
        self,
        idx: torch.Tensor,
        new_repr: torch.Tensor,
        gate,
        *,
        forget_gate: Optional[torch.Tensor] = None,
        tanh_new: bool = False,
        replace: bool = False,
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
            ``slot_value_norm_cap`` no-grad rebind of ``self.slots``), or
            ``None`` on a no-op (frozen / β≈0). P1/v12 uses this as ``M_write``
            for the summary-reconstruction loss: it must stay attached to the
            autograd graph so gradient flows back into the write path. (Reading
            ``self.slots`` after the call would be detached because the norm-cap
            rebinds it under ``torch.no_grad()``.)
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
            updated = g_in * new_content + g_forget * current

            # tanh already bounds new_content to [-1, 1] per dim; we no longer
            # need the manual max_norm clamp from H/H5 (that was a band-aid for
            # 32-layer compounding writes blowing up bf16). Keep the global
            # slot_value_norm_cap below as a safety net only.
            self.slots = self.slots.scatter(1, idx_exp, updated)
            if self._slot_value_norm_cap > 0.0:
                with torch.no_grad():
                    slot_norms_all = self.slots.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    scale_all = (slot_norms_all / self._slot_value_norm_cap).clamp(min=1.0)
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
        if self._slot_value_norm_cap > 0.0:
            with torch.no_grad():
                slot_norms_all = self.slots.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                scale_all = (slot_norms_all / self._slot_value_norm_cap).clamp(min=1.0)
                self.slots = self.slots / scale_all
        return updated
