"""Reference: in-window summary-key BOTTLENECK for the selection-side fix (gap B).
2026-06-20, landmark-repro (Landmark-side logic) for methodA-eval to wire into
mem_space retrieve/layer.py.

★ THE CORE PROBLEM THIS SOLVES
A (chunk512 + grouped readout) trains the INNER readout (consumption). But the
OUTER selection key is the reader's native q·k over a DETACHED content snapshot —
a WEAK signal (re-projects frozen content, can't make the representation itself
more selectable). To beat the 73% outer-hit ceiling we need a selection key that
is TRAINED to summarize each block. H2 says we can't train it via cross-block
selection (no gradient). Landmark's trick: train the summary key via an IN-WINDOW
objective where it is a *bottleneck* the later tokens MUST route through.

★ HOW LANDMARK MAKES IT A REAL BOTTLENECK (NOT A BYPASS) — verified llama_mem.py:921-925
Landmark inserts a physical <landmark> token every mem_freq tokens. Its grouped
attention mask does TWO things in-window:
  mem_ids = cumsum(is_mem) - is_mem          # each token's block id
  last_section_mask = (max(mem_ids)==mem_ids) # tokens in the CURRENT (last) section
  -> a token in the CURRENT section attends:
       * its own section: FULL per-token (normal causal)        [local detail]
       * EARLIER sections: only via each section's LANDMARK token  [bottleneck!]
The earlier sections' individual tokens are GROUPED under their landmark in the
grouped-softmax (llama_mem.py:223-241): a current-section query attends "the
landmark of block k" (one weight) × "tokens within block k" (group-internal). So
to pull info from an earlier block, the query MUST go through that block's landmark
-> the landmark token receives DENSE in-window gradient (every later token that
needs earlier info routes through it) -> it learns to summarize its block.
**This is a STRUCTURAL bottleneck (the direct query->earlier-token path is grouped
under the landmark), not an added side-path the token can ignore.**

★ WHY A NAIVE summary_proj SIDE-PATH FAILS (methodA-eval's H2-revival warning)
If we just ADD an attention path to a summary key while leaving native within-chunk
direct attention intact, later tokens reach earlier tokens DIRECTLY (native) and
ignore the summary key -> summary_proj gets ~no gradient -> H2 revival. The summary
key MUST sit on the ONLY path to earlier-block info.

================================================================================
TWO FAITHFUL IMPLEMENTATIONS (both real bottlenecks; pick per cost/risk)
================================================================================

## B-insert  (closest to Landmark; recommended first per methodA-eval)
Insert a trainable summary token at each 64-block boundary in the *training* stream
(like <landmark>). Within the window, later tokens reach earlier blocks only via
these summary tokens (grouped-softmax truncation as above). Cost: touches tokenize
+ label shift + the per-doc/T2 chunk layout. But it is the proven mechanism.

## B-truncate  (no token insert; mask-based bottleneck)
Keep token stream unchanged. In the CURRENT chunk's self-attention, REPLACE the
direct "later token -> earlier token (same chunk, across 64-sub-block boundary)"
edges with a routed path: later token attends earlier 64-sub-blocks ONLY via each
sub-block's summary key (summary_proj of that sub-block's hidden). Implemented as a
grouped-softmax over [own-subblock tokens (full) ; earlier-subblock SUMMARY KEYS].
This is the function below. It touches the wrapped (frozen) self-attn forward and
needs an ablation that short-range LM doesn't regress.

The function below implements B-truncate's in-window grouped attention. summary_keys
must come from a TRAINABLE summary_proj (so gradient flows to it); the per-subblock
summary key is the bottleneck for cross-subblock in-window reads.
"""
from __future__ import annotations
import math
import torch


def inwindow_bottleneck_attention(
    q: torch.Tensor,            # [B, H, T, hd]  current-chunk queries (RoPE'd)
    k: torch.Tensor,            # [B, H, T, hd]  current-chunk keys (RoPE'd)
    v: torch.Tensor,            # [B, H, T, hd]
    summary_k: torch.Tensor,    # [B, H, n_sub, hd]  per-sub-block summary KEY (from summary_proj, RoPE'd at block end)
    summary_v: torch.Tensor,    # [B, H, n_sub, hd]  per-sub-block summary VALUE
    sub_size: int = 64,
    scaling: float | None = None,
):
    """In-window bottleneck self-attention for the CURRENT chunk.

    Each query at position t (in sub-block s_t = t // sub_size) attends:
      * tokens in its OWN sub-block, causally (full per-token, local detail);
      * EARLIER sub-blocks (s < s_t) ONLY via their SUMMARY key (bottleneck) —
        the individual earlier tokens are NOT directly attendable.
    => to read earlier-in-chunk info, the query must go through summary_k[s] →
       summary_proj receives dense in-window gradient (it is the only path).

    Returns (out [B,H,T,hd], attn_diag). Pure function of q/k/v/summary — the LM
    loss on the chunk's tokens flows back through summary_k → summary_proj.
    """
    B, H, T, hd = q.shape
    scaling = scaling if scaling is not None else (1.0 / math.sqrt(hd))
    n_sub = summary_k.shape[2]
    dev = q.device
    pos = torch.arange(T, device=dev)
    sub_of = pos // sub_size                                  # [T] each token's sub-block

    # ---- own-sub-block causal logits (full per-token, local) ----
    own = torch.matmul(q, k.transpose(-1, -2)) * scaling      # [B,H,T,T]
    # allow only: same sub-block AND causal (j<=t)
    same_sub = sub_of.view(T, 1) == sub_of.view(1, T)         # [T,T]
    causal = pos.view(T, 1) >= pos.view(1, T)
    own_mask = (same_sub & causal).view(1, 1, T, T)
    own = own.masked_fill(~own_mask, float("-inf"))

    # ---- earlier-sub-block SUMMARY logits (bottleneck; one column per earlier sub) ----
    summ = torch.matmul(q, summary_k.transpose(-1, -2)) * scaling   # [B,H,T,n_sub]
    # query in sub s_t may attend summary of sub s ONLY if s < s_t (strictly earlier)
    earlier = sub_of.view(T, 1) > torch.arange(n_sub, device=dev).view(1, n_sub)  # [T,n_sub]
    summ = summ.masked_fill(~earlier.view(1, 1, T, n_sub), float("-inf"))

    # ---- ONE softmax over [own tokens ; earlier-sub summaries] (joint normalization) ----
    cat = torch.cat([own, summ], dim=-1)                      # [B,H,T,T+n_sub]
    # rows where everything is -inf (t in sub0, no earlier, but own causal always has self) → safe
    p = torch.softmax(cat.float(), dim=-1).to(q.dtype)
    p_own = p[..., :T]
    p_summ = p[..., T:]

    out = torch.matmul(p_own, v) + torch.matmul(p_summ, summary_v)  # [B,H,T,hd]
    return out, (p_own, p_summ)


# ---------------------------------------------------------------------------
# WIRING NOTES for methodA-eval (mem_space layer.py / inattn_kv):
#  * summary_k/summary_v = k_proj/v_proj( summary_proj( pool(sub-block hidden) ) ).
#    summary_proj is the NEW trainable param. pool over the sub-block's token hidden
#    (mean or attn-pool). RoPE summary_k at the sub-block's END position (so it sits
#    "after" its tokens, like a landmark).
#  * This REPLACES the current chunk's native self-attn (in readout layers 16-31)
#    when --rawkv_inwindow_bottleneck is on. Default off => native SDPA (byte-identical).
#  * Gradient: chunk LM loss → p_summ → summ = q·summary_k → summary_k = k_proj(summary_proj(pool))
#    → summary_proj gets DENSE gradient (every later token that reads earlier info routes
#    through it). This is the bottleneck guarantee. ASSERT in smoke: summary_proj.grad nonzero.
#  * BOTTLENECK ASSERT (smoke, critical — methodA-eval "real bottleneck not bypass"):
#    verify a query in sub-block s_t>0 has ZERO probability mass on earlier-sub INDIVIDUAL
#    tokens (p_own[..., earlier-sub tokens] must be 0 by construction) and NONZERO mass on
#    p_summ — i.e. earlier info ONLY reachable via summary. Counter: assert
#    (p_own * earlier_token_mask).sum() == 0.
#  * Inference: the trained summary_k doubles as the cross-block selection key
#    (outer topk: query·summary_k over all stored blocks) — same key, in-window-trained.
#  * Short-range regression risk: a token in sub-block 0 has NO earlier subs → behaves
#    like normal causal (only own sub). Tokens in later subs lose direct access to early
#    tokens — ablate that perplexity/short-NIAH doesn't regress vs A.
# ---------------------------------------------------------------------------
