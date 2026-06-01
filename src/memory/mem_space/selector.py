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
from typing import Optional, Tuple

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

        # v10 (2026-06-01): LayerNorm applied to the Q_sel projection output in
        # the multi_query branch ONLY. Root cause of multi_query degeneration:
        # the unconstrained linear Q_sel [d_model→selector_dim] collapses the
        # diverse L3 summary tokens S (4096-dim, max_cos≈0.6) into nearly the
        # SAME 128-dim direction+magnitude → after F.normalize the 64 routing
        # queries become identical (summary_q_max_cos=1.0) → multi_query falls
        # back to single_query → uniform routing. LayerNorm breaks the "all
        # outputs share one direction + magnitude" collapse: it re-centers and
        # re-scales each query independently before normalize, so two inputs
        # that Q_sel mapped close together are pushed apart by their per-feature
        # deviations. Scoped to multi_query so max_pool / chunk_query behaviour
        # is unchanged.
        self.q_sel_ln = nn.LayerNorm(selector_dim)

        # Small init to avoid biasing the early softmax toward any slot —
        # important because the slot bank is initialised from a pooled copy of
        # the same hidden states, which makes all slots near-identical at the
        # start.  We want the gaussian noise in ``slot_init`` to be the only
        # tiebreaker on the very first step.
        nn.init.normal_(self.Q_sel.weight, std=0.02)
        nn.init.normal_(self.K_sel.weight, std=0.02)

        # Content-based routing (2026-05-25): K_sel projects slot content to
        # selector_dim for content-based addressing. Previous static slot_keys
        # approach caused routing collapse (top1_sim stuck at 1/N uniform).
        # K_sel is now TRAINABLE — no VQ-EMA, no InfoNCE, just LM loss +
        # load_balance loss.
        #
        # Per-slot key bias (2026-05-25): With shared_memory_bank, writeback
        # causes all slots to converge to similar values (32 layers write
        # similar content with random routing). This makes K_sel(slots) collapse
        # even when K_sel weights are healthy. The per-slot bias ensures key
        # diversity regardless of slot content — it acts as a "topic prior"
        # that content-based signal can override as training progresses.
        self.slot_key_bias = nn.Parameter(
            F.normalize(torch.randn(num_slots, selector_dim), dim=-1) * 2.0
        )

        # Keep slot_keys for checkpoint backward compat (not used in forward).
        self.slot_keys = nn.Parameter(
            F.normalize(torch.randn(num_slots, selector_dim), dim=-1),
            requires_grad=False,
        )

        # P1-v2 (2026-05-31): configurable detach behavior for slots in routing.
        self._no_detach_slots = False
        # P1-v3 (2026-05-31): routing pool mode — "max_pool" / "chunk_query" /
        # "multi_query".
        self._routing_pool_mode = "max_pool"
        # v8 multi-query routing (2026-06-01): logsumexp aggregation temperature
        # over the M sub-query dimension (L3 summary tokens). Layer may override.
        self._multi_query_tau = 1.0
        # v10 (2026-06-01): cosine threshold for the post-projection q_multi
        # diversity loss. Penalise pairs with cos(q_i, q_j) > threshold. Reuses
        # the L3 diversity threshold semantics (layer wires it from config).
        self._q_multi_diversity_threshold = 0.5

        # v8 multi-query diagnostics (set during multi_query forward; layer
        # reads them via getattr with defaults so non-multi_query modes are safe).
        self._last_summary_query_max_cos = 0.0
        self._last_summary_query_mean_cos = 0.0
        self._last_unique_selected_slots = 0
        # v10 (2026-06-01): pre-projection summary-token (S) diversity, measured
        # on query_tokens[0] BEFORE Q_sel. Compared against
        # _last_summary_query_max_cos (post-projection): if S_max_cos is low
        # (e.g. 0.6) but summary_q_max_cos≈1.0 → Q_sel projection collapse is
        # confirmed. Default 0.0 / safe for non-multi_query modes.
        self._last_S_max_cos = 0.0
        # v10: differentiable diversity loss computed on the post-projection,
        # post-LayerNorm routing query q_multi (the space routing actually uses).
        # Gradient flows back to Q_sel and q_sel_ln. Layer reads this via getattr
        # and adds it to aux (replacing/augmenting the v9 S-space l3_diversity).
        self._last_q_multi_diversity_loss = None

    # --------------------------------------------------------------------- #
    # Forward
    # --------------------------------------------------------------------- #

    def forward(
        self,
        pool_of_H: torch.Tensor,
        slots: torch.Tensor,
        query_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with per-token routing (Fix Z.2).

        Accepts either [B, d_model] (legacy mean-pool) or [B, T, d_model]
        (full hidden states).  When T > 1, each token independently votes
        for its preferred slots; softmax is applied per-token, then averaged
        across tokens to produce the aggregate routing distribution.

        v8 multi-query (2026-06-01): when ``_routing_pool_mode == "multi_query"``
        and ``query_tokens`` ([B, M, d_model], the L3 summary tokens) is given,
        each of the M summary tokens acts as an independent sub-query; per-slot
        relevance is aggregated across the M queries via logsumexp (temperature
        ``_multi_query_tau``), then global top-k. If ``query_tokens is None``
        (cold start / L3 disabled), falls back to the max_pool branch.
        """
        # v10: reset the per-forward multi_query diversity loss so a non-
        # multi_query call (or cold-start fallback) does not leak a stale graph.
        self._last_q_multi_diversity_loss = None
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

        # Content-based routing (2026-05-25): per-token routing using K_sel(slots).
        q = F.normalize(self.Q_sel(pool_of_H), dim=-1)  # [B, T, S]
        # Detach slots before K_sel: gradient trains K_sel weights only, not slot
        # content. Slots get gradient from the read path (slot_to_hidden → output).
        # Without detach, TBPTT gradient through routing destroys slot norms.
        # P1-v2: config.no_detach_slots_in_selector skips detach to give K_sel
        # direct gradient from the routing path.
        _slots_for_key = slots if self._no_detach_slots else slots.detach()
        k = F.normalize(
            self.K_sel(_slots_for_key) + self.slot_key_bias.unsqueeze(0),
            dim=-1,
        )                                                # [B, N, S]

        if self._routing_pool_mode == "multi_query" and query_tokens is not None:
            # v8 multi-query routing (2026-06-01): use the M L3 summary tokens
            # as M independent sub-queries instead of a single chunk query.
            # Each sub-query scores all N slots; per-slot relevance is aggregated
            # across the M queries via logsumexp (a soft-max between max and mean),
            # preserving intra-chunk semantic heterogeneity → avoids the
            # single-query "万金油" collapse.
            if query_tokens.dim() != 3:
                raise ValueError(
                    f"query_tokens must be [B, M, d_model]; got "
                    f"{tuple(query_tokens.shape)}"
                )
            if query_tokens.shape[0] != B:
                raise ValueError(
                    f"query_tokens batch {query_tokens.shape[0]} != pool batch {B}"
                )
            if query_tokens.shape[-1] != self.d_model:
                raise ValueError(
                    f"query_tokens last-dim {query_tokens.shape[-1]} != "
                    f"d_model {self.d_model}"
                )
            q_multi = F.normalize(self.q_sel_ln(self.Q_sel(query_tokens)), dim=-1)  # [B, M, S]
            # v10: LayerNorm (q_sel_ln) sits between Q_sel and normalize. Without
            # it the unconstrained Q_sel collapses diverse S into one direction
            # (summary_q_max_cos=1.0); LayerNorm re-centers/re-scales per feature
            # so distinct S stay distinct after normalize.
            score = torch.einsum("bms,bns->bmn", q_multi, k) * self.temperature  # [B, M, N]
            tau_q = self._multi_query_tau
            logits = torch.logsumexp(score / tau_q, dim=1) * tau_q   # [B, N]

            # ---- v10 q_multi diversity loss (DIFFERENTIABLE — NOT in no_grad) ----
            # The v9 l3_diversity loss acts on S (pre-projection, 4096-dim). But
            # routing actually uses q_multi (post-Q_sel, post-LN, 128-dim). If
            # Q_sel collapses S, an S-space loss can be satisfied while q_multi
            # is still collapsed. So we add diversity pressure directly on the
            # space routing uses: penalise pairs of routing queries with
            # cos > threshold. Gradient flows back through F.normalize → q_sel_ln
            # → Q_sel (q_multi is NOT detached). Computed per batch item, mean
            # over b and over off-diagonal i<j pairs.
            B_q, M_q, _ = q_multi.shape
            if M_q >= 2:
                qsim_loss = torch.bmm(q_multi, q_multi.transpose(1, 2))  # [B, M, M]
                _iu = torch.triu(
                    torch.ones(M_q, M_q, device=q_multi.device, dtype=torch.bool),
                    diagonal=1,
                )
                _pair = F.relu(qsim_loss - self._q_multi_diversity_threshold)[:, _iu]
                self._last_q_multi_diversity_loss = _pair.mean()
            else:
                self._last_q_multi_diversity_loss = q_multi.new_zeros(())

            # ---- multi-query diagnostics (no grad, cheap for M~64) ----
            with torch.no_grad():
                q0 = q_multi[0]                                  # [M, S], normalized
                if q0.shape[0] > 1:
                    qsim = torch.mm(q0, q0.t())                  # [M, M]
                    qsim.fill_diagonal_(0.0)
                    M_q = q0.shape[0]
                    self._last_summary_query_max_cos = qsim.abs().max().item()
                    # mean over off-diagonal entries
                    self._last_summary_query_mean_cos = (
                        qsim.sum() / (M_q * (M_q - 1))
                    ).item()
                else:
                    self._last_summary_query_max_cos = 0.0
                    self._last_summary_query_mean_cos = 0.0
                # v10: pre-projection diversity of the raw L3 summary tokens S
                # (query_tokens[0], 4096-dim) — compare against the post-
                # projection summary_q_max_cos above to confirm/deny Q_sel
                # projection collapse.
                S0 = F.normalize(query_tokens[0], dim=-1)        # [M, d_model]
                if S0.shape[0] > 1:
                    Ssim = torch.mm(S0, S0.t())                  # [M, M]
                    Ssim.fill_diagonal_(0.0)
                    self._last_S_max_cos = Ssim.abs().max().item()
                else:
                    self._last_S_max_cos = 0.0
                # coverage: if each sub-query picked its single favourite slot,
                # how many distinct slots would that cover (batch[0])?
                per_q_argmax = score[0].argmax(dim=-1)           # [M]
                self._last_unique_selected_slots = int(
                    per_q_argmax.unique().numel()
                )
        elif self._routing_pool_mode == "chunk_query":
            # P1-v3 fix (2026-05-31): chunk-level single query routing.
            # Mean-pool hidden states → single query per chunk → one logit per slot.
            # This eliminates the max-pool uniformity problem where T>>N causes
            # every slot to find a "champion token" → all max-logits similar → uniform.
            q_chunk = q.mean(dim=1, keepdim=False)       # [B, S]
            logits = torch.einsum("bs,bns->bn", q_chunk, k) * self.temperature  # [B, N]
        else:
            # Legacy max-pool: per-token logits then max over T.
            per_token_logits = torch.einsum("bts,bns->btn", q, k) * self.temperature
            logits = per_token_logits.max(dim=1).values  # [B, N]

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

        # Diagnostic: store logit std for monitoring
        with torch.no_grad():
            self._last_per_token_logit_std = logits.std(dim=-1).mean().item()

        return idx, scores, ste_weights

    # Fix Z.1 (2026-04-30): VQ-EMA and dead slot revival removed

    # --------------------------------------------------------------------- #
    # Key diversity loss (Fix Z.2c) + K_sel weight orthogonality (2026-05-25)
    # --------------------------------------------------------------------- #

    def key_repulsion_loss(self, threshold: float = 0.3, slots: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Penalise high cosine similarity between projected slot key pairs.

        With content-based routing, keys are K_sel(slots) + slot_key_bias.
        Falls back to static slot_keys if slots not provided (legacy compat).

        Returns:
            scalar: mean of max(0, cos(K_i, K_j) - threshold) for i < j.
        """
        if slots is not None:
            K = F.normalize(
                self.K_sel(slots[0].detach()) + self.slot_key_bias, dim=-1
            )  # [N, S] — detach slots so gradient only trains K_sel weights
        else:
            K = F.normalize(self.slot_keys, dim=-1)         # [N, S] fallback
        sim = torch.mm(K, K.t())                            # [N, N]
        sim.fill_diagonal_(0.0)
        penalty = F.relu(sim - threshold)
        return penalty.mean()

    def weight_ortho_loss(self) -> torch.Tensor:
        """Prevent K_sel weight collapse by penalizing deviation from orthogonality.

        Computes ||W^T W - I||_F^2 on the row-normalized K_sel weight.
        When K_sel.weight rows are orthogonal, this is 0. When they collapse
        to rank-1, this is maximally large. This directly prevents the failure
        mode where load_balance_loss drives all projected keys to be identical.
        """
        W = F.normalize(self.K_sel.weight, dim=1)  # [selector_dim, slot_dim] row-normalized
        WtW = torch.mm(W, W.t())                   # [S, S]
        eye = torch.eye(WtW.shape[0], device=WtW.device, dtype=WtW.dtype)
        return ((WtW - eye) ** 2).mean()

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


# --------------------------------------------------------------------------- #
# Cross-attention memory (Perceiver-IO / Block Recurrent Transformer style)
# --------------------------------------------------------------------------- #


class CrossAttentionMemory(nn.Module):
    """Cross-attention based memory (Perceiver-IO / Block Recurrent Transformer style).

    Replaces the top-k selector + KV-prepend pipeline with:
    - Read: tokens cross-attend to slots (tokens=Q, slots=K/V)
    - Write: slots cross-attend to tokens (slots=Q, tokens=K/V)

    No routing degeneracy because attention softmax naturally peaks per-token.

    Unlike TopKSelector, this module handles both the read and write phases.
    The MemorySpaceLayer just needs to call forward() for read and
    writeback() for write.

    Args:
        d_model: backbone hidden size.
        n_slots: number of memory slots (N).
        n_heads: number of attention heads for cross-attention.
        slot_dim: slot vector dimensionality. None means d_model.
    """

    def __init__(
        self,
        d_model: int,
        n_slots: int = 128,
        n_heads: int = 8,
        slot_dim: Optional[int] = None,
        *,
        use_dual_gate: bool = False,
        forget_bias_init: float = 1.0,
        input_bias_init: float = 0.0,
        dual_gate_tanh_new: bool = True,
        read_topk: int = 0,
    ) -> None:
        super().__init__()
        slot_dim = slot_dim or d_model
        self.d_model = d_model
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.n_heads = n_heads
        self.use_dual_gate = use_dual_gate
        self.dual_gate_tanh_new = dual_gate_tanh_new
        self.read_topk = read_topk

        # Projections if slot_dim != d_model
        self.need_proj = slot_dim != d_model
        if self.need_proj:
            self.slot_to_hidden = nn.Linear(slot_dim, d_model, bias=False)
            self.hidden_to_slot = nn.Linear(d_model, slot_dim, bias=False)
            nn.init.normal_(self.slot_to_hidden.weight, std=0.02)
            nn.init.normal_(self.hidden_to_slot.weight, std=0.02)

        # Read cross-attention: tokens attend to slots
        self.read_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, bias=False,
        )
        # LoRA-style zero init on output projection: out_proj=0 means entire
        # cross-attn path outputs exactly 0 at init → no disruption to pretrained LM.
        # Gradient d(loss)/d(out_proj) = (d(loss)/d(hidden))^T @ attn_out ≠ 0 from step 1.
        # After step 1, out_proj ≠ 0 → cascading activation: V → K → Q.
        nn.init.zeros_(self.read_attn.out_proj.weight)

        # LayerNorm on read attention output (before gating)
        self.read_ln = nn.LayerNorm(d_model)

        # Write cross-attention: slots attend to tokens
        # Also zero-init out_proj so writeback output starts at 0 (same LoRA-B logic).
        # Slots maintain their initial values until write_attn learns useful patterns.
        self.write_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, bias=False,
        )
        nn.init.zeros_(self.write_attn.out_proj.weight)

        # LayerNorm on write attention output
        self.write_ln = nn.LayerNorm(d_model)

        # No gate — out_proj=0 handles zero-start (LoRA-B style).
        # out_proj grows naturally via gradient descent, controlling output magnitude.
        # No tanh wrapper needed — removes the amplification problem.
        self.output_gate = None  # unused, kept for checkpoint compat

        # Writeback gate (init 0, sigmoid(0)=0.5 moderate writeback)
        self.write_gate = nn.Parameter(torch.tensor(0.0))

        # H6 (LM2-inspired): dual-gate (input + forget) writeback projections.
        # When use_dual_gate=True, replace the single sigmoid(write_gate) blend
        # with content-conditioned per-feature gates:
        #     g_in, g_forget = sigmoid_split(W_n·new_repr + W_m·M_prev + bias)
        #     slots = g_in * tanh(new_repr) + g_forget * M_prev
        # Two separate projections (LM2 design) so gates condition on BOTH new
        # content and prior slot state. forget_bias_init=1.0 makes g_forget
        # start at sigmoid(1)≈0.73 ("remember by default", LSTM heuristic).
        if use_dual_gate:
            self.gate_proj_new = nn.Linear(slot_dim, 2 * slot_dim, bias=False)
            self.gate_proj_mem = nn.Linear(slot_dim, 2 * slot_dim, bias=False)
            bias_init = torch.cat([
                torch.full((slot_dim,), float(input_bias_init)),
                torch.full((slot_dim,), float(forget_bias_init)),
            ])
            self.gate_bias = nn.Parameter(bias_init)
            # Small init on gate projections so initial gates ≈ sigmoid(bias):
            # input ≈ sigmoid(0)=0.5, forget ≈ sigmoid(1)=0.73
            nn.init.xavier_uniform_(self.gate_proj_new.weight, gain=0.5)
            nn.init.xavier_uniform_(self.gate_proj_mem.weight, gain=0.5)
        else:
            self.gate_proj_new = None
            self.gate_proj_mem = None
            self.gate_bias = None

    def _project_slots_to_hidden(self, slots: torch.Tensor) -> torch.Tensor:
        """Project slots from slot_dim to d_model if needed."""
        if self.need_proj:
            return self.slot_to_hidden(slots)
        return slots

    def _project_hidden_to_slot(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden states from d_model to slot_dim if needed."""
        if self.need_proj:
            return self.hidden_to_slot(hidden)
        return hidden

    def forward(
        self,
        hidden_states: torch.Tensor,
        slots: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read phase: tokens cross-attend to memory slots.

        Args:
            hidden_states: [B, T, d_model] current layer input.
            slots: [B, N, slot_dim] current slot content (from MemoryBank).

        Returns:
            gated_hidden: [B, T, d_model] memory-augmented hidden states.
            attn_weights: [B, n_heads, T, N] attention weights for diagnostics.
        """
        # Project slots to d_model space
        slots_d = self._project_slots_to_hidden(slots)  # [B, N, d_model]

        # Read: tokens (Q) attend to slots (K, V)
        attn_out, attn_weights = self.read_attn(
            query=hidden_states,
            key=slots_d,
            value=slots_d,
            need_weights=True,
            average_attn_weights=False,  # [B, n_heads, T, N]
        )

        # No gate — out_proj=0 ensures zero contribution at init.
        # No LayerNorm — it would amplify the small out_proj output to unit scale,
        # disrupting the pretrained model. out_proj controls magnitude directly.
        gated = hidden_states + attn_out

        return gated, attn_weights

    def writeback(
        self,
        hidden_states: torch.Tensor,
        slots: torch.Tensor,
        beta: float = 0.1,
    ) -> torch.Tensor:
        """Write phase: slots cross-attend to hidden states, then EMA update.

        Args:
            hidden_states: [B, T, d_model] decoder output (post self-attention).
            slots: [B, N, slot_dim] current slot content.
            beta: EMA blend rate (how much new content to mix in). Ignored when
                use_dual_gate=True (dual gate is fully content-conditioned).

        Returns:
            updated_slots: [B, N, slot_dim] new slot content.
        """
        # Project slots to d_model space
        slots_d = self._project_slots_to_hidden(slots)  # [B, N, d_model]

        # Write: slots (Q) attend to hidden states (K, V)
        new_slot_content, _ = self.write_attn(
            query=slots_d,
            key=hidden_states,
            value=hidden_states,
            need_weights=False,
        )
        new_slot_content = self.write_ln(new_slot_content)  # [B, N, d_model]

        # Project back to slot_dim
        new_slot_projected = self._project_hidden_to_slot(new_slot_content)  # [B, N, slot_dim]

        # ---- H6 dual-gate path (LM2-inspired) ----
        if self.use_dual_gate and self.gate_proj_new is not None:
            # Both gates condition on (new_repr, prior memory). Per-feature.
            gate_logits = (
                self.gate_proj_new(new_slot_projected)
                + self.gate_proj_mem(slots)
                + self.gate_bias  # broadcast [2d] -> [B, N, 2d]
            )
            g_in_logit, g_forget_logit = gate_logits.chunk(2, dim=-1)
            g_in = torch.sigmoid(g_in_logit)         # [B, N, slot_dim]
            g_forget = torch.sigmoid(g_forget_logit) # [B, N, slot_dim]
            new_content = (
                torch.tanh(new_slot_projected)
                if self.dual_gate_tanh_new
                else new_slot_projected
            )
            updated_slots = g_in * new_content + g_forget * slots
            return updated_slots

        # ---- Legacy single-gate path (H/H5/H3) ----
        # Writeback gate (learnable, init 0)
        gamma = torch.sigmoid(self.write_gate)  # [0, 1]

        # EMA update with learnable gate + fixed beta
        effective_beta = gamma * beta
        updated_slots = (1.0 - effective_beta) * slots + effective_beta * new_slot_projected

        return updated_slots

    def compute_attn_entropy(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """Compute mean attention entropy across heads (diagnostic).

        High entropy = spread attention (degenerate/uniform).
        Low entropy = peaked attention (healthy, content-addressed).

        Args:
            attn_weights: [B, n_heads, T, N] from forward().

        Returns:
            scalar tensor: mean entropy (lower is better).
        """
        # Clamp to avoid log(0)
        p = attn_weights.clamp(min=1e-8)
        entropy = -(p * p.log()).sum(dim=-1)  # [B, n_heads, T]
        return entropy.mean()  # scalar


# --------------------------------------------------------------------------- #
# Infini-attention compressive memory (Munkhdalai et al., 2024)
# --------------------------------------------------------------------------- #


class InfiniAttentionMemory(nn.Module):
    """Infini-attention style compressive memory (Munkhdalai et al., 2024).

    Uses linear attention (ELU+1 kernel) over an associative matrix M.
    No softmax -> no routing degeneracy or uniform-attention problem.
    Reuses pretrained Q, K, V projections from the wrapped decoder layer.

    Trainable params: only beta (per-head gate scalar, ~32 per layer).
    Memory state: M (associative matrix) and z (normalizer), updated in-place.
    """

    def __init__(self, d_model: int, n_heads: int = 32, n_kv_heads: Optional[int] = None, beta_init: float = -1.0, memory_scale_init: float = 1.0, trainable_proj: bool = False) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.d_head = d_model // n_heads
        self.trainable_proj = trainable_proj

        # Per-head gate: init from beta_init (default -1.0 -> sigmoid(-1) ~ 0.27)
        self.beta = nn.Parameter(torch.full((self.n_kv_heads,), beta_init))

        # Per-head learnable scalar to amplify memory retrieval output.
        # Shape [n_kv_heads, 1, 1] broadcasts over [B, H, T, d].
        self.memory_scale = nn.Parameter(torch.full((self.n_kv_heads, 1, 1), memory_scale_init))

        # v5: Trainable output projection to replace frozen o_proj for memory path.
        # Zero-init: no disruption at start, identical to pretrained backbone.
        # Gradient flows from step 1: d(loss)/d(weight) = d(loss)/d(hidden)^T @ input.
        if self.trainable_proj:
            self.mem_o_proj = nn.Linear(d_model, d_model, bias=False)
            nn.init.zeros_(self.mem_o_proj.weight)

        # Memory state (per-sample, lazily initialized to [B, ...] at runtime)
        # Uses n_kv_heads for M/z to match K/V projection dimensions (GQA support).
        self.M = None  # [B, n_kv_heads, d_head, d_head]
        self.z = None  # [B, n_kv_heads, d_head]

    def _sigma(self, x: torch.Tensor) -> torch.Tensor:
        """Linear attention kernel: sigma(x) = ELU(x) + 1"""
        return F.elu(x) + 1.0

    def is_initialized(self, batch_size: int) -> bool:
        return self.M is not None and self.M.shape[0] == batch_size

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        """Initialize memory state for given batch size."""
        self.M = torch.zeros(batch_size, self.n_kv_heads, self.d_head, self.d_head,
                             device=device, dtype=dtype)
        self.z = torch.zeros(batch_size, self.n_kv_heads, self.d_head,
                             device=device, dtype=dtype)

    def reset(self) -> None:
        """Reset memory state to zero (called at document boundaries)."""
        if self.M is not None:
            self.M.zero_()
            self.z.zero_()

    def detach_(self) -> None:
        """Detach memory state (break autograd graph across chunks)."""
        if self.M is not None:
            self.M = self.M.detach()
            self.z = self.z.detach()

    def retrieve(self, hidden_states: torch.Tensor,
                 q_proj: nn.Linear, o_proj: nn.Linear) -> torch.Tensor:
        """Linear attention memory retrieval.

        Args:
            hidden_states: [B, T, d_model]
            q_proj: frozen backbone Q projection
            o_proj: frozen backbone output projection

        Returns:
            gated_mem_output: [B, T, d_model] -- memory contribution, gated by beta
        """
        B, T, D = hidden_states.shape

        # Compute Q using backbone's frozen projection (NO RoPE)
        Q = q_proj(hidden_states)  # [B, T, n_heads * d_head]
        Q_heads = Q.view(B, T, self.n_heads, self.d_head).permute(0, 2, 1, 3)  # [B, H, T, d]
        sigma_Q = self._sigma(Q_heads)  # [B, H, T, d]

        # GQA: Q has n_heads, M has n_kv_heads. Repeat Q heads to match KV groups.
        n_rep = self.n_heads // self.n_kv_heads
        if n_rep > 1:
            # Average sigma_Q across heads in each KV group for memory retrieval
            sigma_Q_kv = sigma_Q.view(B, self.n_kv_heads, n_rep, T, self.d_head).mean(dim=2)
        else:
            sigma_Q_kv = sigma_Q

        # A_mem = sigma(Q) @ M / (sigma(Q) @ z + eps)
        # M: [B, KV, d, d], sigma_Q_kv: [B, KV, T, d]
        numer = torch.matmul(sigma_Q_kv, self.M)  # [B, KV, T, d]
        denom = torch.matmul(sigma_Q_kv, self.z.unsqueeze(-1))  # [B, KV, T, 1]
        A_mem_kv = numer / denom.clamp(min=1e-6)  # [B, KV, T, d]

        # Expand back to n_heads for output
        if n_rep > 1:
            A_mem = A_mem_kv.unsqueeze(2).expand(B, self.n_kv_heads, n_rep, T, self.d_head).reshape(B, self.n_heads, T, self.d_head)
        else:
            A_mem = A_mem_kv

        # Per-head gate (one per KV head, broadcast to all Q heads in group)
        gate = torch.sigmoid(self.beta).view(1, self.n_kv_heads, 1, 1)
        if n_rep > 1:
            gate = gate.unsqueeze(2).expand(B, self.n_kv_heads, n_rep, 1, 1).reshape(B, self.n_heads, 1, 1)
        A_mem_gated = gate * A_mem  # [B, H, T, d]
        # Expand memory_scale from [n_kv_heads, 1, 1] to [1, n_heads, 1, 1]
        mem_sc = self.memory_scale.repeat_interleave(n_rep, dim=0).unsqueeze(0)
        A_mem_gated = A_mem_gated * mem_sc  # per-head amplification

        # Reshape and project through output projection
        A_flat = A_mem_gated.permute(0, 2, 1, 3).reshape(B, T, D)  # [B, T, D]
        # v5: use trainable mem_o_proj when enabled, otherwise frozen backbone o_proj
        proj = self.mem_o_proj if self.trainable_proj else o_proj
        mem_output = proj(A_flat)  # [B, T, D]

        return mem_output

    def update(self, hidden_states: torch.Tensor,
               k_proj: nn.Linear, v_proj: nn.Linear) -> None:
        """Delta rule memory update (self-correcting).

        Only stores the residual (new info - already stored), preventing
        redundant overwriting and providing self-correction.
        """
        B, T, D = hidden_states.shape

        K = k_proj(hidden_states)  # [B, T, n_kv_heads * d_head]
        V = v_proj(hidden_states)  # [B, T, n_kv_heads * d_head]
        K_heads = K.view(B, T, self.n_kv_heads, self.d_head).permute(0, 2, 1, 3)  # [B, KV, T, d]
        V_heads = V.view(B, T, self.n_kv_heads, self.d_head).permute(0, 2, 1, 3)  # [B, KV, T, d]
        sigma_K = self._sigma(K_heads)  # [B, KV, T, d]

        # Delta rule: compute what memory already stores for these keys
        existing_V = torch.matmul(sigma_K, self.M) / \
                     torch.matmul(sigma_K, self.z.unsqueeze(-1)).clamp(min=1e-6)  # [B, H, T, d]
        delta_V = V_heads - existing_V  # residual: only new information

        # Update M and z (in-place)
        # M += sigma_K^T @ delta_V (summed over batch and time)
        delta_M = torch.einsum('bhti,bhtj->bhij', sigma_K, delta_V)  # [B, H, d, d]
        delta_z = sigma_K.sum(dim=2)  # [B, H, d] -- sum over time

        self.M = self.M + delta_M
        self.z = self.z + delta_z

    def compute_diagnostics(self, hidden_states: torch.Tensor,
                            q_proj: nn.Linear) -> dict:
        """Compute diagnostic metrics (for logging)."""
        with torch.no_grad():
            B, T, D = hidden_states.shape
            Q = q_proj(hidden_states)
            Q_heads = Q.view(B, T, self.n_heads, self.d_head).permute(0, 2, 1, 3)
            sigma_Q = self._sigma(Q_heads)

            # GQA: average across heads in each KV group
            n_rep = self.n_heads // self.n_kv_heads
            if n_rep > 1:
                sigma_Q_kv = sigma_Q.view(B, self.n_kv_heads, n_rep, T, self.d_head).mean(dim=2)
            else:
                sigma_Q_kv = sigma_Q

            # Memory utilization: how much of M is being used
            M_norm = self.M.float().norm(dim=(-2, -1)).mean().item()  # scalar
            z_norm = self.z.float().norm(dim=-1).mean().item()  # scalar

            # Retrieval magnitude
            numer = torch.matmul(sigma_Q_kv, self.M)
            denom = torch.matmul(sigma_Q_kv, self.z.unsqueeze(-1)).clamp(min=1e-6)
            A_mem = numer / denom
            mem_norm = A_mem.float().norm(dim=-1).mean().item()

            # Gate values
            beta_vals = torch.sigmoid(self.beta)
            beta_mean = beta_vals.mean().item()
            beta_min = beta_vals.min().item()
            beta_max = beta_vals.max().item()

        return {
            'M_norm': M_norm,
            'z_norm': z_norm,
            'mem_retrieval_norm': mem_norm,
            'beta_mean': beta_mean,
            'beta_min': beta_min,
            'beta_max': beta_max,
            'memory_scale_mean': self.memory_scale.mean().item(),
            'memory_scale_min': self.memory_scale.min().item(),
            'memory_scale_max': self.memory_scale.max().item(),
        }


# --------------------------------------------------------------------------- #
# Cross-Attention Memory V2 (Scheme A: zero-init, no LayerNorm, no gate)
# --------------------------------------------------------------------------- #


class CrossAttentionMemoryV2(nn.Module):
    """Independent cross-attention for memory read/write.

    Scheme A implementation: replaces the ChunkMemoryBank prepend approach
    with a standalone cross-attention module.

    Key design choices (vs existing CrossAttentionMemory):
    - NO LayerNorm (amplifies small outputs, disrupting pretrained model)
    - NO gate (out_proj=0 handles zero-start, LoRA-B style)
    - NO write cross-attention (uses simpler delta-rule from read attention weights)
    - Supports GQA: n_kv_heads may differ from n_heads (Llama-3 compat)

    Forward flow:
        1. read(): Q=hidden_states, K/V=slot projections -> memory_output
           out_proj is zero-initialized, so at step 0 output = 0
        2. write(): delta-rule update using attention weights from read()
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        num_slots: int,
        dropout: float = 0.0,
        write_lr: float = 1.0,
        *,
        use_dual_gate: bool = False,
        forget_bias_init: float = 1.0,
        input_bias_init: float = 0.0,
        dual_gate_tanh_new: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.num_slots = num_slots
        self.head_dim = d_model // n_heads
        self.write_lr = write_lr
        self.use_dual_gate = use_dual_gate
        self.dual_gate_tanh_new = dual_gate_tanh_new

        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"
        assert n_heads % n_kv_heads == 0, f"n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}"

        # GQA replication factor
        self.n_rep = n_heads // n_kv_heads

        # Q projection: from content hidden states
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        # K projection: from memory slots
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        # V projection: from memory slots
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        # Output projection: ZERO INITIALIZED (key requirement!)
        self.out_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=True)

        # Zero initialization: ensures step 0 output = 0, equivalent to vanilla model
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Initialize Q/K/V with small random values (standard init)
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # H6 (LM2-inspired): dual-gate (input + forget) writeback projections.
        # When use_dual_gate=True, replace the single-rate delta-rule with
        # content-conditioned per-feature gates:
        #     g_in, g_forget = sigmoid_split(W_n·content + W_m·slot_prev + bias)
        #     slots = g_in * tanh(content) + g_forget * slot_prev
        # Both projections use slot dim = d_model (V2 has no separate slot_dim).
        # forget_bias_init=1.0 → g_forget starts ≈ sigmoid(1)≈0.73 (LSTM heuristic
        # "remember by default"). Per-feature gates → 2 * d_model dim per slot.
        # Reference: LM2 (arXiv:2502.06049), src/memory.py:259-263 + create_gates.
        if use_dual_gate:
            self.gate_proj_new = nn.Linear(d_model, 2 * d_model, bias=False)
            self.gate_proj_mem = nn.Linear(d_model, 2 * d_model, bias=False)
            bias_init = torch.cat([
                torch.full((d_model,), float(input_bias_init)),
                torch.full((d_model,), float(forget_bias_init)),
            ])
            self.gate_bias = nn.Parameter(bias_init)
            # Small init so initial gates ≈ sigmoid(bias) at step 0:
            # input ≈ sigmoid(0)=0.5, forget ≈ sigmoid(1)≈0.73
            nn.init.xavier_uniform_(self.gate_proj_new.weight, gain=0.5)
            nn.init.xavier_uniform_(self.gate_proj_mem.weight, gain=0.5)
        else:
            self.gate_proj_new = None
            self.gate_proj_mem = None
            self.gate_bias = None

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat K/V heads to match Q heads for GQA. x: [B, n_kv_heads, S, D]"""
        if self.n_rep == 1:
            return x
        B, H, S, D = x.shape
        return x[:, :, None, :, :].expand(B, H, self.n_rep, S, D).reshape(B, H * self.n_rep, S, D)

    def read(
        self,
        hidden_states: torch.Tensor,
        slot_keys: torch.Tensor,
        slot_values: torch.Tensor,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cross-attention read: Q=hidden, K=slot_keys, V=slot_values.

        Args:
            hidden_states: [B, T, d_model] content hidden states.
            slot_keys: [B, num_slots, d_model] memory slot key projections.
            slot_values: [B, num_slots, d_model] memory slot value content.
            return_logits: if True, also return pre-softmax attention logits.

        Returns:
            memory_output: [B, T, d_model] cross-attention output (via zero-init out_proj).
            attention_weights: [B, n_heads, T, num_slots] for delta-rule write-back.
            attn_logits (only when return_logits=True): [B, n_heads, T, num_slots]
                pre-softmax scaled dot-product logits, for contrastive loss.
        """
        B, T, _ = hidden_states.shape
        N = slot_keys.shape[1]

        # Project Q from hidden states, K/V from slots
        Q = self.q_proj(hidden_states)  # [B, T, n_heads * head_dim]
        K = self.k_proj(slot_keys)      # [B, N, n_kv_heads * head_dim]
        V = self.v_proj(slot_values)    # [B, N, n_kv_heads * head_dim]

        # Reshape to multi-head: [B, S, H, D] -> [B, H, S, D]
        Q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)        # [B, n_heads, T, D]
        K = K.view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)     # [B, n_kv_heads, N, D]
        V = V.view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)     # [B, n_kv_heads, N, D]

        # GQA: repeat K/V to match Q heads
        K = self._repeat_kv(K)  # [B, n_heads, N, D]
        V = self._repeat_kv(V)  # [B, n_heads, N, D]

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, n_heads, T, N]
        attn_weights = F.softmax(attn_logits, dim=-1, dtype=torch.float32).to(V.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum
        attn_output = torch.matmul(attn_weights, V)  # [B, n_heads, T, D]

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, -1)  # [B, T, d_model]
        memory_output = self.out_proj(attn_output)  # [B, T, d_model]

        if return_logits:
            return memory_output, attn_weights, attn_logits
        return memory_output, attn_weights

    def write(
        self,
        hidden_states: torch.Tensor,
        slot_values: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Delta-rule write: error-corrective update of slot values.

        For each slot j, compute:
            content_j = sum_i(attention_weights[:, :, i, j] * hidden_states[:, i, :])
            error_j = content_j - slot_values[j]
            slot_values[j] = slot_values[j] + write_lr * error_j

        This is equivalent to:
            updated_slots = (1 - write_lr) * slot_values + write_lr * content

        With write_lr=1.0, slots are completely replaced by the new content.
        With write_lr < 1.0, slots do a weighted average of old and new content.

        H6 (LM2-inspired) — when use_dual_gate=True, replace the single write_lr
        with content-conditioned per-feature gates:
            g_in, g_forget = sigmoid_split(W_n·content + W_m·slot_prev + bias)
            slots = g_in * tanh(content) + g_forget * slot_prev
        Both gates float independently (NOT a 1-β split), so the network can
        choose to fully remember + fully overwrite, or fully forget + write
        nothing, on a per-slot per-feature basis.

        Args:
            hidden_states: [B, T, d_model] content hidden states.
            slot_values: [B, num_slots, d_model] current slot values.
            attention_weights: [B, n_heads, T, num_slots] from read().

        Returns:
            updated_slots: [B, num_slots, d_model] new slot values.
        """
        # Average attention weights across heads for write: [B, T, num_slots]
        avg_weights = attention_weights.mean(dim=1)  # [B, T, N]

        # Compute attention-weighted content: what each slot "sees"
        # avg_weights: [B, T, N] -> [B, N, T]
        # hidden_states: [B, T, d_model]
        # Result: [B, N, d_model]
        content = torch.bmm(avg_weights.transpose(1, 2), hidden_states)  # [B, N, d_model]

        # ---- H6 dual-gate path (LM2-inspired) ----
        if self.use_dual_gate and self.gate_proj_new is not None:
            gate_logits = (
                self.gate_proj_new(content)
                + self.gate_proj_mem(slot_values)
                + self.gate_bias  # broadcast [2d] -> [B, N, 2d]
            )
            g_in_logit, g_forget_logit = gate_logits.chunk(2, dim=-1)
            g_in = torch.sigmoid(g_in_logit)         # [B, N, d_model]
            g_forget = torch.sigmoid(g_forget_logit) # [B, N, d_model]
            new_content = (
                torch.tanh(content) if self.dual_gate_tanh_new else content
            )
            updated_slots = g_in * new_content + g_forget * slot_values
            return updated_slots

        # ---- Legacy delta-rule path (H/H5/H3) ----
        # Delta-rule: error = desired content - current slot value
        error = content - slot_values  # [B, N, d_model]

        # Apply learning rate to error
        updated_slots = slot_values + self.write_lr * error

        return updated_slots

    @staticmethod
    def compute_attn_entropy(attn_weights: torch.Tensor) -> float:
        """Mean attention entropy across heads (diagnostic for peaked vs uniform)."""
        p = attn_weights.clamp(min=1e-8)
        entropy = -(p * p.log()).sum(dim=-1)  # [B, n_heads, T]
        return entropy.mean().item()
