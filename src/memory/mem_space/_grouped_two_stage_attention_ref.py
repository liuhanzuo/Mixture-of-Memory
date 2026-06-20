"""Reference implementation: two-stage (Landmark grouped-softmax) attention for the
raw-KV readout path (gap B). 2026-06-20, landmark-repro (Landmark-side logic) for
methodA-eval to splice into src/memory/mem_space/inattn_kv.py.

WHY: methodA-eval's current inattn_kv concats retrieved kept-KV onto native K/V and
runs ONE flat softmax. A correctly-selected 512-token chunk still drowns a ~25-token
needle in a flat softmax (within-block dilution): chunk-oracle=57.5% vs token-oracle
=90% — the 32.5pt gap is exactly this. Landmark avoids it with a TWO-STAGE softmax:
  stage 1 (block selection): each 64-token block competes as ONE unit (its landmark
           token's softmax weight among all blocks).
  stage 2 (within-block):     tokens inside the selected block get their own softmax,
           normalized ONLY against their block-mates (not against other blocks /
           native columns) → no cross-block dilution, and a 64-token block keeps the
           needle at ~40% mass.

This is mathematically: P(col) = P(block of col) * P(col | its block).
Equivalent to a hierarchical softmax. CANNOT be done with a fused SDPA kernel because
the normalization is grouped — must hand-roll QK^T -> grouped softmax -> @V.

Drop-in: replace the `attention_interface(...)` call (inattn_kv.py ~:204-213) with
`grouped_two_stage_attention(...)` WHEN the (B) flag is on AND injection is present;
otherwise keep the native SDPA path (byte-identical).
"""
from __future__ import annotations
import math
import torch


def grouped_two_stage_attention(
    q: torch.Tensor,            # [B, H, Lq, hd]   (RoPE'd query)
    k: torch.Tensor,            # [B, H, Lk_native + R, hd]  (native ; retrieved), RoPE'd
    v: torch.Tensor,            # [B, H, Lk_native + R, hd]
    base_mask: torch.Tensor,    # [B, 1, Lq, Lk_native]  additive causal mask for native part
    Lk_native: int,             # number of native key columns
    R: int,                     # number of retrieved key columns (MUST be multiple of group_size)
    group_size: int = 64,       # sub-block size (= Landmark mem_freq+1)
    scaling: float | None = None,
    block_logbias: torch.Tensor | None = None,  # [B, Lq, n_sub] optional stage-1 log-weight
                                                 # (e.g. reader-attn sub-block selection score);
                                                 # added to each group's top-level logit.
):
    """Hierarchical (two-stage) attention. Native columns each compete as individual
    columns; every `group_size`-token retrieved sub-block competes as a SINGLE unit at
    the top level, with an internal softmax among its own tokens.

    Returns (out [B,H,Lq,hd], attn [B,H,Lq,Lk_native+R]).
    """
    B, H, Lq, hd = q.shape
    scaling = scaling if scaling is not None else (1.0 / math.sqrt(hd))
    assert R % group_size == 0, f"R={R} must be a multiple of group_size={group_size}"
    n_sub = R // group_size

    logits = torch.matmul(q, k.transpose(-1, -2)) * scaling     # [B,H,Lq,Lk_native+R]
    nat = logits[..., :Lk_native] + base_mask                   # native: + causal mask
    ret = logits[..., Lk_native:]                               # [B,H,Lq,R] retrieved

    # ---- stage 2: within-sub-block softmax (each sub-block normalized internally) ----
    sub_id = (torch.arange(R, device=q.device) // group_size)   # [R] -> 0..n_sub-1
    sub_idx = sub_id.view(1, 1, 1, R).expand(B, H, Lq, R)
    gmax = ret.new_full((B, H, Lq, n_sub), float("-inf"))
    gmax.scatter_reduce_(-1, sub_idx, ret, reduce="amax", include_self=False)
    gmax_per_col = gmax.gather(-1, sub_idx)
    e_ret = torch.exp((ret - gmax_per_col).float())             # fp32 for stability
    gden = torch.zeros(B, H, Lq, n_sub, dtype=e_ret.dtype, device=q.device)
    gden.scatter_add_(-1, sub_idx, e_ret)
    within = (e_ret / gden.gather(-1, sub_idx)).to(logits.dtype)  # [B,H,Lq,R] sum=1 per sub-block

    # ---- stage 1: top-level softmax over (native columns) vs (n_sub block-units) ----
    # Each sub-block's representative logit = its log-sum-exp (the "soft-max" energy of
    # the block), so a block that contains a strongly-matching token gets high mass.
    group_lse = gmax + torch.log(gden.clamp_min(1e-20)).to(gmax.dtype)   # [B,H,Lq,n_sub]
    if block_logbias is not None:
        # add an explicit stage-1 selection score (e.g. reader-attn sub-block score)
        group_lse = group_lse + block_logbias.unsqueeze(1).to(group_lse.dtype)
    top_logits = torch.cat([nat, group_lse], dim=-1)            # [B,H,Lq,Lk_native+n_sub]
    top_p = torch.softmax(top_logits.float(), dim=-1).to(logits.dtype)
    p_nat = top_p[..., :Lk_native]                              # [B,H,Lq,Lk_native]
    p_grp = top_p[..., Lk_native:]                              # [B,H,Lq,n_sub]

    # ---- combine: P(retrieved col) = within(col) * P(its block) ----
    ret_col_p = within * p_grp.gather(-1, sub_idx)              # [B,H,Lq,R]
    attn = torch.cat([p_nat, ret_col_p], dim=-1)               # [B,H,Lq,Lk_native+R], rows sum~1
    out = torch.matmul(attn.to(v.dtype), v)                    # [B,H,Lq,hd]
    return out, attn


# ---------------------------------------------------------------------------
# Integration sketch (replaces inattn_kv.py ~:196-213 attention_interface call):
#
#   if getattr(self, "_rawkv_grouped_readout", False) and R > 0:
#       attn_output, attn_weights = grouped_two_stage_attention(
#           q, k, v, base_mask=base, Lk_native=Lk_native, R=R,
#           group_size=getattr(self.config, "rawkv_subblock_size", 64),
#           scaling=self.scaling,
#           block_logbias=_stage1_logbias,   # or None for equal-weight (variant A)
#       )
#   else:
#       attn_output, attn_weights = attention_interface(self, q, k, v, full_mask, ...)
#   attn_output = attn_output.reshape(*input_shape, -1).contiguous()
#   attn_output = self.o_proj(attn_output)
#
# NOTES:
#  * R MUST be a multiple of group_size(64). If kept = topk chunks of 512 tokens each,
#    each 512-chunk = 8 sub-blocks; if already chunk_size=64, each kept chunk = 1 group.
#  * GQA: if n_kv_heads < n_heads, expand k/v heads BEFORE calling (repeat_kv), same as
#    the native path does, so q,k,v all have H heads here.
#  * block_logbias=None => variant A (pure grouping, equal block weight) — fastest test.
#    block_logbias = reader-attn per-sub-block score (log) => variant B (full two-stage).
#  * Pure-eval: gate behind the flag, run existing h1fix ckpt on real BABILong — no
#    retrain. If it lifts W0 from 4k7 toward 57.5/90, stage-2 works at inference time.
# ---------------------------------------------------------------------------
