"""Q-Filters KV-cache compression.

Pure-tensor operations. Given the per-(kv)-head filters (top-`rank` right-singular
vectors of the calibration Q matrix), score each cached key by cosine similarity
in filter space, keep the top (budget - recent_window) plus the most recent
`recent_window` tokens.

No side effects, no learned parameters. All tensors remain on their input device
and retain their input dtype (except scoring which is done in float32 for
numerical safety).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


_EPS = 1e-6


def score_keys(
    filters: torch.Tensor,
    keys: torch.Tensor,
    eps: float = _EPS,
) -> torch.Tensor:
    """Score every cached key by Sum_r |cos(filter_r, key)|.

    GQA note: H here is num_key_value_heads (NOT num_attention_heads).
    Keys in HF's DynamicCache are stored per-KV-head, and the filters are
    reduced to KV-head granularity by `calibration.compute_filters` (via the
    `num_kv_heads` arg, which averages Q within each GQA group before SVD).
    For Llama-2-7B the two counts are equal (32==32) so this is a no-op; for
    Llama-3.0-8B (32 Q heads, 8 KV heads, group=4) the filter tensor has
    H=8 — matching `keys.shape[1]`.

    Args:
        filters: [H, D, R]  (H = num_kv_heads, D = head_dim, R = filter_rank)
        keys:    [B, H, T, D]  (H = num_kv_heads)
        eps:     numerical-safety floor for zero-norm rows

    Returns:
        scores: [B, H, T] in float32
    """
    # Cast to float32 for stability; fp16/bf16 cosine dies on large norms.
    f = filters.to(torch.float32)                          # [H, D, R]
    k = keys.to(torch.float32)                             # [B, H, T, D]

    f_norm = torch.linalg.vector_norm(f, dim=1, keepdim=True)
    f_norm = f_norm.clamp_min(eps)
    f_hat = f / f_norm                                     # [H, D, R]

    k_norm = torch.linalg.vector_norm(k, dim=-1, keepdim=True)
    k_norm = k_norm.clamp_min(eps)
    k_hat = k / k_norm                                     # [B, H, T, D]

    # cos_{b,h,t,r} = <k_hat[b,h,t,:], f_hat[h,:,r]>
    cos = torch.einsum("bhtd,hdr->bhtr", k_hat, f_hat)     # [B, H, T, R]
    scores = cos.abs().sum(dim=-1)                         # [B, H, T]
    # nan/inf guard
    scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores


def compress_kv(
    queries_proj: Optional[torch.Tensor],
    filters: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    budget: int,
    recent_window: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prune a KV cache to `budget` tokens per head.

    Args:
        queries_proj: unused here (kept in signature for parity with the
            reference impl, which projects queries into filter space for
            attention-score estimation). The filter-based scoring used here
            operates on keys only, which is sufficient because the
            top right-singular vectors of Q identify the dominant query
            directions -- projecting keys onto those directions estimates
            the column-sum of future attention scores.
        filters: [H, D, R]           (H = num_kv_heads)
        keys:    [B, H, T, D]
        values:  [B, H, T, D]
        budget:  target #tokens kept per head
        recent_window: N most-recent tokens that are always kept

    Returns:
        (kept_keys, kept_values, kept_indices)
            kept_keys:    [B, H, budget_eff, D]
            kept_values:  [B, H, budget_eff, D]
            kept_indices: [B, H, budget_eff]  (positions into the original T axis,
                                              sorted in ascending order)
        where budget_eff = min(T, budget).
    """
    assert keys.shape == values.shape, "keys/values shape mismatch"
    assert keys.dim() == 4, f"expected 4-D keys, got {keys.shape}"
    assert budget > 0, f"budget must be > 0, got {budget}"
    assert recent_window >= 0, f"recent_window must be >= 0, got {recent_window}"

    B, H, T, D = keys.shape
    device = keys.device

    # Nothing to prune.
    if T <= budget:
        idx = torch.arange(T, device=device).view(1, 1, T).expand(B, H, T)
        return keys, values, idx

    # 2026-04-26: soft floor (budget-1) guarantees at least one filter-scored
    # slot survives even if the config guard in QFiltersConfig is somehow
    # bypassed. See ops/research_notes/20260426_qfilters_recent_eq_kv_edge_case.md
    r = min(recent_window, budget - 1, T)
    keep_old = budget - r

    # Indices of the "recent" block (always kept).
    if r > 0:
        recent_idx = torch.arange(T - r, T, device=device).view(1, 1, r).expand(B, H, r)
    else:
        recent_idx = torch.empty(B, H, 0, dtype=torch.long, device=device)

    if keep_old <= 0:
        # Only recent window survives.
        gather_idx = recent_idx
    else:
        old_keys = keys[:, :, : T - r, :]                  # [B, H, T-r, D]
        scores = score_keys(filters, old_keys)             # [B, H, T-r]
        # top-k by score among the "old" block
        _, topk_idx = torch.topk(scores, keep_old, dim=-1, largest=True, sorted=False)
        # sort to preserve positional order (nicer for debugging; attention
        # itself does not require ordered keys, but keeping order makes RoPE
        # relative-offset math easier to reason about).
        topk_idx, _ = topk_idx.sort(dim=-1)
        gather_idx = torch.cat([topk_idx, recent_idx], dim=-1)   # [B, H, budget]

    # Gather keys/values.
    gather_expand = gather_idx.unsqueeze(-1).expand(-1, -1, -1, D)
    kept_keys = keys.gather(2, gather_expand)
    kept_values = values.gather(2, gather_expand)
    return kept_keys, kept_values, gather_idx
