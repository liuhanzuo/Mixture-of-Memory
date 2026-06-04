# v16 — Loss-Free Balancing for Slot Routing (P7)

**Date**: 2026-06-05
**Reference**: "Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts" (DeepSeek, arXiv:2408.15664)
**Researcher provenance**: ops/research_notes/benchmark_survey_and_improvements_20260604.md BLOCK 3b #1 (confidence: high)

## Motivation

The Switch-Transformer style `load_balance_loss` (selector.py `load_balance_loss`,
weighted by `config.load_balance_weight`) pushes routing toward **uniform** by
adding `N · Σ importance_i · load_i` to the total loss. Diagnosed root cause of
routing collapse: this aux **injects a task-interfering gradient** into `Q_sel`/
`K_sel` that fights the LM objective and, far from preventing collapse, can
*worsen* it by dragging every slot toward the same uniform score.

Loss-free balancing replaces the uniform-pushing gradient with an **online
per-slot routing-logit bias** that equalises slot usage **without producing any
gradient** — the bias only influences *which* slots win top-k, never the values
that flow backward.

## Architecture (forward pseudocode)

```
logits      = score(Q_sel(query), K_sel(slots))        # [B, N]   (unchanged)
scores      = softmax(logits)                          # [B, N]   ← GRADIENT PATH (bias-free)

if use_loss_free_balance:
    sel_scores = softmax(logits + routing_bias)        # [B, N]   biased — selection only
else:
    sel_scores = scores

idx = topk(sel_scores, k=top_k).indices.detach()       # which slots win

# returned scores / ste_weights are built from `scores` (UNBIASED logits):
one_hot     = scatter(idx -> 1)
ste_weights = scores + (one_hot*scores - scores).detach()
return idx, scores, ste_weights

# --- online bias update (no_grad, training only) ---
if use_loss_free_balance and self.training:
    with torch.no_grad():
        load        = one_hot(idx).mean(dim=0)         # [N] observed usage in [0,1]
        target_load = top_k / N                        # each slot's fair share
        err         = target_load - load               # under-used → positive
        routing_bias += loss_free_update_rate * sign(err)
```

**Key invariant**: `routing_bias` enters **only** `sel_scores` (the top-k index
choice). The returned `scores` and `ste_weights` are functions of the *unbiased*
`logits`, so the gradient that reaches `Q_sel`/`K_sel`/slots is byte-identical to
what it would be with the bias absent. The bias steers selection; it never
perturbs the LM/task gradient.

## Initialization

- `routing_bias`: `torch.zeros(num_slots)` — registered as a **buffer** (NOT a
  Parameter → no gradient, no optimizer state; moves with `.to(device)` and is
  saved/loaded in state_dict so balancing state survives checkpointing).
- `loss_free_update_rate`: `0.001` (sign update; DeepSeek uses sign for
  robustness to load scale).
- `use_loss_free_balance`: `False` by default.

When `use_loss_free_balance=False` the selector is **byte-identical to pre-P7**
(`sel_scores is scores`, no bias update). Unit-verified.

## Relationship to prior work

- **vs Switch-Transformer `load_balance_loss`** (selector.py, Fedus et al. 2021):
  Switch adds an aux *loss term* whose gradient flows into the routing
  projections and competes with the LM objective. P7 produces **no gradient** —
  balancing is a discrete bias update outside the autograd graph. Recommended to
  set `load_balance_weight=0.0` when enabling P7 (config docstring + argparse
  help warn about this; the two should not both be non-zero).
- **vs DeepSeek loss-free (arXiv:2408.15664)**: same idea — per-expert bias added
  to selection logits only, sign-updated from observed load. We apply it to
  fixed memory slots (experts ↔ slots) with `target_load = top_k/N`.

## Config / CLI surface

- `config.use_loss_free_balance: bool = False`
- `config.loss_free_update_rate: float = 0.001`
- layer.py wires both into `selector.use_loss_free_balance` /
  `selector.loss_free_update_rate` (only the selector-construction block; read/
  write paths untouched).
- `scripts/train_mem_space_babilong.py`: `--use_loss_free_balance` (store_true),
  `--loss_free_update_rate` (float, default 0.001), threaded into the
  `MemorySpaceConfig(...)` build and the run-config log dict.

## Known issues

- The bias is global per-slot, not per-token; it balances aggregate batch usage,
  not per-query fairness (matches DeepSeek's expert-level formulation).
- `sign`-based update gives a fixed step regardless of imbalance magnitude — very
  large initial imbalance takes many steps to correct. `loss_free_update_rate`
  may need tuning if balancing is too slow/oscillatory.
- Buffer is shared per-layer (each layer's selector has its own `routing_bias`),
  consistent with the per-layer slot bank.
- Not yet validated in a full training run — only unit-tested for correctness,
  back-compat, and gradient-path cleanliness.
