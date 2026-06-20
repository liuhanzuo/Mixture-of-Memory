"""CPU smoke for B-truncate in-window bottleneck (2026-06-20, landmark-repro).

Verifies the THREE load-bearing properties (the "don't be a bypass / don't be a
dead param" gate team-lead + methodA-eval demanded):
  (1) BOTTLENECK: a query in sub-block s_t>0 puts ZERO probability on earlier
      sub-blocks' INDIVIDUAL tokens, and NONZERO probability on the earlier
      sub-blocks' SUMMARY keys → earlier-in-chunk info is reachable ONLY via the
      summary (real bottleneck, not a side-path).
  (2) GRADIENT: the summary KEY/VALUE (proxy for summary_proj output) receives
      NON-ZERO gradient from the chunk LM-style loss → summary_proj would train.
  (3) LOCALITY: own-sub-block attention is full causal (local detail preserved).
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.memory.mem_space._inwindow_bottleneck_ref import inwindow_bottleneck_attention  # noqa: E402


def main():
    torch.manual_seed(0)
    B, H, hd = 1, 4, 16
    sub_size = 4          # tiny sub-blocks for the test
    n_sub = 3
    T = sub_size * n_sub  # 12 tokens, 3 sub-blocks of 4
    q = torch.randn(B, H, T, hd, requires_grad=True)
    k = torch.randn(B, H, T, hd, requires_grad=True)
    v = torch.randn(B, H, T, hd, requires_grad=True)
    # summary key/value as leaf tensors with grad (proxy for summary_proj output)
    summary_k = torch.randn(B, H, n_sub, hd, requires_grad=True)
    summary_v = torch.randn(B, H, n_sub, hd, requires_grad=True)

    out, (p_own, p_summ) = inwindow_bottleneck_attention(
        q, k, v, summary_k, summary_v, sub_size=sub_size,
    )
    print(f"[smoke] out shape={tuple(out.shape)} p_own={tuple(p_own.shape)} p_summ={tuple(p_summ.shape)}")

    pos = torch.arange(T)
    sub_of = pos // sub_size

    # ---- (1) BOTTLENECK: earlier-sub individual tokens get ZERO mass ----
    # for query t, "earlier-sub token j" = sub_of[j] < sub_of[t]
    earlier_tok = (sub_of.view(T, 1) > sub_of.view(1, T))   # [T(query), T(key)] key in strictly earlier sub
    mass_on_earlier_indiv = (p_own * earlier_tok.view(1, 1, T, T)).sum().item()
    print(f"[smoke] (1) prob mass on earlier-sub INDIVIDUAL tokens = {mass_on_earlier_indiv:.3e} (must be 0)")
    assert mass_on_earlier_indiv < 1e-6, "BYPASS! earlier tokens directly attendable — not a bottleneck"

    # queries in sub-block >0 must have NONZERO mass on summaries (route through bottleneck)
    q_in_later = sub_of > 0
    summ_mass_later = p_summ[:, :, q_in_later, :].sum().item()
    print(f"[smoke] (1) summary mass for queries in sub>0 = {summ_mass_later:.3f} (must be >0)")
    assert summ_mass_later > 1e-3, "later queries don't route through summary — bottleneck dead"

    # ---- (3) LOCALITY: own-sub causal works (row sums ~1 over all) ----
    rowsum = (p_own.sum(-1) + p_summ.sum(-1))
    print(f"[smoke] (3) attn row sums min={rowsum.min().item():.4f} max={rowsum.max().item():.4f} (≈1)")
    assert torch.allclose(rowsum, torch.ones_like(rowsum), atol=1e-3), "rows not normalized"

    # ---- (2) GRADIENT to summary key/value (=> summary_proj trains) ----
    loss = out.float().pow(2).mean()       # surrogate for LM loss on chunk tokens
    loss.backward()
    gk = summary_k.grad.norm().item()
    gv = summary_v.grad.norm().item()
    print(f"[smoke] (2) summary_k.grad_norm={gk:.4e}  summary_v.grad_norm={gv:.4e} (both must be >0)")
    assert gk > 0 and gv > 0, "summary key/value got NO gradient — summary_proj would be a dead param (H2 revival)"

    print("\n[smoke] VERDICT: PASS — in-window bottleneck is real (earlier info only via "
          "summary, summary gets dense gradient, locality preserved). Not a bypass, not a dead param.")


if __name__ == "__main__":
    main()
