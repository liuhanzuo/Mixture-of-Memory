"""Memory-Space v0 — top-k selector with straight-through gradient.

Design reference:
    ops/research_notes/20260426_memory_space_design_direction.md §2.1 (step 1)
    and §2.2 (top-k selector recommendation: hard top-k + load-balance loss).

The selector scores each slot against a pooled summary of the current-layer
hidden states via a scaled-dot-product softmax on two small projections
(``Q_sel`` on hidden, ``K_sel`` on slots), then picks the k highest-scoring
slots.  Gradients flow to both projections via a straight-through estimator
(``one_hot_topk`` values forward, softmax scores backward), so the selector
stays differentiable end-to-end even though ``torch.topk`` is not.

MoE-style load-balance auxiliary loss (Switch Transformer, Fedus et al. 2021):

    aux = N · Σ_i  importance_i · load_i
        importance_i = mean over (B, T) of scores[:, :, i]
        load_i       = mean over (B, T) of one_hot_topk[:, :, i]

This pushes both *softmax mass* and *dispatch frequency* to be uniform across
slots, mitigating the slot-collapse failure mode flagged as R1 in the design
doc.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Top-k selector
# --------------------------------------------------------------------------- #


class TopKSelector(nn.Module):
    """Hard top-k slot selector with straight-through soft-weight gradients.

    Args:
        d_model: input hidden size (pooled query dimension).
        slot_dim: slot vector dimension.
        selector_dim: projection dim shared by ``Q_sel`` / ``K_sel``.
        top_k: number of slots selected per forward.
        num_slots: N (for the load-balance loss normalisation).

    Forward:
        pool_of_H: [B, d_model]
        slots:     [B, N, slot_dim]
        →
        idx:         [B, top_k]      long, selected slot indices (hard)
        scores:      [B, N]          softmax scores (for load-balance loss)
        ste_weights: [B, N]          straight-through weights — scores forward,
                                     one-hot-top-k values on the backward path

    ``ste_weights`` is intended for any downstream op that wants a
    *differentiable* gate on slots (e.g. scaling the gathered memory tokens
    before prepending them) — the research doc keeps this option open.

    Fix B (2026-04-28): ``slot_keys = nn.Parameter([N, selector_dim])`` replaces
    the ``K_sel(slots)`` key computation. Each slot has a direct learnable key
    vector, decoupling slot addressing from slot content. ``K_sel`` is retained
    but frozen for checkpoint backward compatibility with pre-Fix-B checkpoints.
    """

    def __init__(
        self,
        d_model: int,
        slot_dim: int,
        *,
        selector_dim: int = 128,
        top_k: int = 16,
        num_slots: int = 128,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if top_k <= 0 or top_k > num_slots:
            raise ValueError(
                f"top_k must be in (0, num_slots], got top_k={top_k} num_slots={num_slots}"
            )
        self.d_model = d_model
        self.slot_dim = slot_dim
        self.selector_dim = selector_dim
        self.top_k = top_k
        self.num_slots = num_slots
        self.temperature = temperature  # Fix O (2026-04-29): configurable; was hardcoded 10.0

        self.Q_sel = nn.Linear(d_model, selector_dim, bias=False)
        self.K_sel = nn.Linear(slot_dim, selector_dim, bias=False)

        # Small init to avoid biasing the early softmax toward any slot —
        # important because the slot bank is initialised from a pooled copy of
        # the same hidden states, which makes all slots near-identical at the
        # start.  We want the gaussian noise in ``slot_init`` to be the only
        # tiebreaker on the very first step.
        nn.init.normal_(self.Q_sel.weight, std=0.02)
        nn.init.normal_(self.K_sel.weight, std=0.02)

        # Fix Z.2g: Learnable slot_keys with peak routing loss.
        # Keys are learnable so they can specialize.
        self.slot_keys = nn.Parameter(
            F.normalize(torch.randn(num_slots, selector_dim), dim=-1),
        )

        # K_sel frozen — not used in forward (keep for checkpoint compat).
        for p in self.K_sel.parameters():
            p.requires_grad = False

    # --------------------------------------------------------------------- #
    # Forward
    # --------------------------------------------------------------------- #

    def forward(
        self,
        pool_of_H: torch.Tensor,
        slots: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with per-token routing (Fix Z.2).

        Accepts either [B, d_model] (legacy mean-pool) or [B, T, d_model]
        (full hidden states).  When T > 1, each token independently votes
        for its preferred slots; softmax is applied per-token, then averaged
        across tokens to produce the aggregate routing distribution.
        """
        if pool_of_H.dim() not in (2, 3):
            raise ValueError(
                f"pool_of_H must be [B, d_model] or [B, T, d_model]; "
                f"got {pool_of_H.dim()}D shape {tuple(pool_of_H.shape)}"
            )
        if slots.dim() != 3:
            raise ValueError(
                f"slots must be [B, N, slot_dim]; got {tuple(slots.shape)}"
            )
        if pool_of_H.dim() == 2:
            pool_of_H = pool_of_H.unsqueeze(1)  # [B, 1, d_model]

        B, T, d_in = pool_of_H.shape
        if d_in != self.d_model:
            raise ValueError(f"pool_of_H last-dim {d_in} != d_model {self.d_model}")
        if slots.shape[0] != B:
            raise ValueError(
                f"batch mismatch: pool={B}, slots={slots.shape[0]}"
            )
        if slots.shape[1] != self.num_slots:
            raise ValueError(
                f"slots.shape[1] {slots.shape[1]} != num_slots {self.num_slots}"
            )

        # Fix Z.2 (2026-04-30): per-token routing.
        # Fix Z.2f: Use learnable slot_keys for routing (reverted from content-based
        # routing which caused DDP issues and K_sel collapse). Keys are learnable.
        q = F.normalize(self.Q_sel(pool_of_H), dim=-1)  # [B, T, S]
        k = F.normalize(
            self.slot_keys.unsqueeze(0).expand(B, -1, -1),
            dim=-1,
        )                                                # [B, N, S], unit vectors
        # Per-token logits: [B, T, N]
        per_token_logits = torch.einsum("bts,bns->btn", q, k) * self.temperature
        # Fix Z.2b: max-pool across tokens instead of softmax-then-mean.
        # For each slot, take the highest logit across all T tokens. This means
        # "this chunk has at least one token that strongly prefers this slot".
        # Max-pool preserves diversity because different slots are championed
        # by different tokens (unlike mean which averages to uniformity).
        logits = per_token_logits.max(dim=1).values               # [B, N]
        scores = F.softmax(logits, dim=-1)                        # [B, N]

        # Hard top-k indices (no gradient through this op).
        _, idx = torch.topk(scores, k=self.top_k, dim=-1, largest=True, sorted=False)
        idx = idx.detach()                               # [B, top_k]

        # Build the one-hot mask: [B, N].  Used both for the STE weights
        # (forward value) and for the load-balance aux loss.
        one_hot = torch.zeros_like(scores).scatter_(
            dim=-1, index=idx, value=1.0
        )                                                # [B, N]

        # Straight-through: forward = one_hot * scores, backward = scores.
        # We pick ``scores + (one_hot_scores - scores).detach()`` so that on
        # the backward pass d/dscores = 1 (i.e. gradient behaves as if we had
        # just used the raw softmax).  On the forward pass the value equals
        # ``one_hot * scores`` — zero for unselected slots, softmax score for
        # selected ones.  This matches the expression from the design note.
        one_hot_scores = one_hot * scores
        ste_weights = scores + (one_hot_scores - scores).detach()  # [B, N]

        # Fix Z.1 (2026-04-30): Remove VQ-EMA and InfoNCE storage
        # slot_keys are now frozen random orthogonal vectors, no need for EMA or alignment losses
        self.last_idx = idx  # [B, top_k], hard selection indices (kept for load balance loss)

        # Fix Z.2 diagnostic: store per-token logit variance for monitoring
        with torch.no_grad():
            self._last_per_token_logit_std = per_token_logits.std(dim=1).mean().item()

        return idx, scores, ste_weights

    # Fix Z.1 (2026-04-30): VQ-EMA and dead slot revival removed

    # --------------------------------------------------------------------- #
    # Key diversity loss (Fix Z.2c)
    # --------------------------------------------------------------------- #

    def key_repulsion_loss(self, threshold: float = 0.3) -> torch.Tensor:
        """Penalise high cosine similarity between slot key pairs.

        Prevents key collapse where all slot_keys converge to the same
        direction under LM loss gradient.  Only penalises cosines above
        `threshold` so keys can be somewhat similar (near-orthogonal is fine)
        but not identical.

        Returns:
            scalar: mean of max(0, cos(K_i, K_j) - threshold) for i < j.
        """
        # Normalize keys (they may have drifted from unit norm)
        K = F.normalize(self.slot_keys, dim=-1)          # [N, S]
        # Pairwise cosine similarity matrix
        sim = torch.mm(K, K.t())                         # [N, N]
        # Zero diagonal (self-similarity = 1)
        sim.fill_diagonal_(0.0)
        # Penalise similarities above threshold
        penalty = F.relu(sim - threshold)
        return penalty.mean()

    def peak_routing_loss(self, scores: torch.Tensor) -> torch.Tensor:
        """Loss that pushes per-chunk routing to be peaked (low conditional entropy).

        Maximises the top-1 score per batch item, which encourages the model to
        commit to specific slots for each chunk rather than spreading mass uniformly.

        Returns:
            scalar: negative mean top-1 score (minimising = maximising peak).
        """
        top1 = scores.max(dim=-1).values  # [B]
        return -top1.mean()

    # --------------------------------------------------------------------- #
    # MoE-style aux loss
    # --------------------------------------------------------------------- #

    def load_balance_loss(
        self,
        scores: torch.Tensor,
        idx: torch.Tensor,
    ) -> torch.Tensor:
        """Switch-Transformer style aux loss.

        Args:
            scores: [B, N]    softmax scores as returned by ``forward``.
            idx:    [B, top_k] hard-selected indices.

        Returns:
            scalar tensor, finite and differentiable in ``scores`` (through
            ``Q_sel``/``K_sel``).  The ``load`` term is not differentiable
            (it's a hard indicator) — that matches Fedus et al.'s formulation.
        """
        if scores.dim() != 2:
            raise ValueError(f"scores must be [B, N]; got {tuple(scores.shape)}")
        B, N = scores.shape
        if N != self.num_slots:
            raise ValueError(
                f"scores.shape[1] {N} != num_slots {self.num_slots}"
            )

        # importance_i: average softmax mass routed to slot i.
        importance = scores.mean(dim=0)                   # [N]
        # load_i: average number of times slot i was chosen in the hard top-k.
        one_hot = torch.zeros_like(scores).scatter_(
            dim=-1, index=idx, value=1.0
        )
        load = one_hot.float().mean(dim=0)                # [N]

        # Scale by N so the loss is O(1) when perfectly uniform
        # (importance = load = top_k / N  →  aux = top_k^2 / N; Switch uses
        # `N * Σ importance · load` as the canonical form).
        aux = float(N) * torch.sum(importance * load)
        return aux

    # --------------------------------------------------------------------- #
    # Entropy auxiliary loss (Fix D.2)
    # --------------------------------------------------------------------- #

    def entropy_aux_loss(
        self,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        """Entropy maximisation auxiliary loss for routing diversity.

        Maximises Shannon entropy of the mean routing distribution:
            H = -Σ_i  p_i * log(p_i)    where p_i = mean_over_batch(scores[:, i])

        We return  -H  (minimising negative entropy = maximising entropy).

        Unlike the Switch-Transformer load-balance loss, this has NON-ZERO
        gradient at the uniform fixed point (where all slots get equal score):
            d(-H)/d(logits_i) ∝ -(log(p_i) + 1) / N
        At the uniform point, log(p_i) = log(1/N), so the gradient is
            -(log(1/N) + 1) / N = (log(N) - 1) / N  ≠ 0  for N > e.
        For N=512: (log(512) - 1) / 512 = (6.24 - 1) / 512 ≈ 0.010 — a healthy
        non-zero push away from uniformity.

        Fix D.2 (2026-04-28): Provides gradient at the uniform fixed point where
        the Switch-style load_balance_loss has zero gradient (because the load
        indicator function is not differentiable and importance × load has zero
        first-order gradient when the optimum is uniform).

        Args:
            scores: [B, N]  softmax scores as returned by ``forward``.

        Returns:
            scalar — negative entropy, to be minimised (weighted by a small
            coefficient, e.g. 0.001, added to the total loss).
        """
        if scores.dim() != 2:
            raise ValueError(f"scores must be [B, N]; got {tuple(scores.shape)}")
        B, N = scores.shape
        if N != self.num_slots:
            raise ValueError(
                f"scores.shape[1] {N} != num_slots {self.num_slots}"
            )
        # Mean routing distribution over the batch.
        p = scores.mean(dim=0)                                    # [N], sums to 1
        # Clamp to avoid log(0); 1e-8 is far below any real softmax value for N=512
        entropy = -(p * torch.log(p.clamp(min=1e-8))).sum()      # scalar ≥ 0
        return -entropy                                           # negate: minimise → maximise H
