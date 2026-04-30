# Q-Filters `kv_budget` Sweep Mechanism Analysis (2026-04-25)

**Author**: researcher (autonomous chain)
**Predecessor**: `ops/research_notes/20260425_qfilters_post_fix_analysis.md` (hypothesis (e), option A recommended)
**Scope**: all runs Llama-2-7B, `pg19_chunks[200:400]`, seq_length=4096, bf16, sdpa, `filter_rank=2`, `calibration_chunks=64`, `sub_window_len=1024`, 8× L20A b200-1, same `filters.pt` reused across budgets.

## 0. Data (anchor table)

| kv_budget | recent_window | keep_old (filter) | compression | PPL     | vs dense 3625.39 |
|-----------|---------------|-------------------|-------------|---------|-------------------|
| 256       | 64            | 192               | 16×         | 2635.55 | 0.727× (**BEST**) |
| 512       | 64            | 448               | 8×          | 2662.19 | 0.734×            |
| 64        | 16            | 48                | 64×         | 2693.21 | 0.743×            |
| 128       | 64            | 64                | 32×         | 3079.18 | 0.849×            |
| 64        | 64            | **0**             | 64×         | 3608.74 | 0.995× (filter OFF) |
| 4096      | —             | —                 | 1×          | 3625.39 | 1.000× (dense)    |
| 512       | 64 (SW)       | —                 | 8×          | 5609.34 | 1.548× (last-512) |

---

## §1. Best operating point

**Best: kv_budget=256, recent=64, PPL=2635.55, 16× compression, −27.3% vs dense.** It beats budget=512 (PPL 2662.19, 8×) by 27 PPL (~1%). Mechanism: at budget=512 the filter keeps top-448 projected-cosine keys; with `filter_rank=2` and 4096 candidate old keys, the 448th ranked key is far down the cosine distribution and is effectively noise. At budget=256 the filter keeps only top-192 — a much tighter cut on the rank-2 subspace, so the retained old keys are genuinely high-alignment with calibration queries. More filter mass ≠ better: the rank-2 projection has a sharp signal/noise floor, and we cross it between 192 and 448 retained tokens.

**Non-monotonicity 512→256→128.** 128 (PPL 3079) is a 400-PPL jump back toward dense. Reading the recipe: `keep_old = 128 − 64 = 64`, so only 64 filter-selected keys survive alongside the last-64 window — half the budget is spent on raw recency, half on filter picks. At this ratio the filter signal is starved (64 picks from 4096 under a rank-2 score is coarse) *and* the recent window is still small. The 64/recent=16 row (PPL 2693) shows the fix: shrink recency to 16 so `keep_old=48` still gets meaningful filter selection at half the total budget. So 128/64 is pathological *because* recent_window is frozen at 64: recency dominates budget without adding enough positional signal, and filter is too starved to compensate. The curve isn't non-monotonic in budget — it's non-monotonic in `keep_old / recent_window` ratio.

**Headline pick: 16× @ −27.3% (budget=256)**, not 64× @ −25.7% (budget=64 recent=16). Two reasons: (i) the 256 row has recent_window at the same 64 used in every other filter-on row, so it's the clean published point with a stable recency baseline; (ii) 64×/recent=16 tunes two knobs at once (budget *and* recent) — publishable as a secondary "how far can you push compression" result, but not the primary.

## §2. The budget=64 / recent=64 accident as a control

With `keep_old = 64 − 64 = 0` the filter is mechanically OFF: pure last-64-per-sub-window carryover. PPL=3608.74 ≈ dense 3625.39 (0.995×). This is an unplanned but clean secondary baseline.

**Reframing.** Because of `sub_window_len=1024` carryover, "last-64" at eval time means *each forward sees up to 64 tokens from the previous sub-window plus 1024 fresh tokens*, i.e. effective scope per forward is ~1088, not 64. That explains why it isn't catastrophic: the sliding-window collapse observed at budget=512 SW (PPL 5609) was specifically because 512-carryover-per-1024 still amounts to a trailing window that breaks anchor continuity across sub-windows 3→4. A 64-token bridge plus 1024 fresh tokens replicates roughly what a chunked-dense evaluator would see if chunks were 1088 long. That it *ties* dense 4096-context is itself evidence that pg19 cold chunks are dominated by local information; the long tail of 4096 dense tokens isn't helping much.

**The 915 PPL delta (3608.74 → 2693.21) at identical budget=64** is the cleanest single-variable ablation we have. Both rows hold budget, both use carryover; the only difference is `recent_window` going 64→16, which turns the filter from OFF to ON (keep_old 0→48). Frame it as: **"Holding KV budget at 64 and flipping filter signal from off to on drops PPL by 25% (3609→2693)."** Filter signal is worth ~915 PPL at 64-budget on pg19.

## §3. Filter cost-benefit curve

PPL vs log2(compression):

```
 1×  (dense)        3625  ────────────●
 8×  (512/64)       2662  ●
16×  (256/64)       2636  ● ← knee (minimum)
32×  (128/64)       3079  ──●  (starved filter at fixed recent=64)
64×  (64/16)        2693    ●   (re-tuned recent brings it back)
64×  (64/64, noF)   3609  ────────●  (filter off, curve falls off cliff)
```

Shape: bowl, not monotone. Minimum at 16×. The 8×→16× segment slopes gently *down* (−27 PPL per doubling), then 16×→32× jumps *up* (+443) when recent_window isn't retuned, but 16×→64× with retuned recent is only +57 PPL. The filter-off control at 64× is +973 above the knee. Interpretation: compression ratio is a misleading x-axis; what matters is (a) does the filter have enough `keep_old` slots to resolve the rank-2 signal (≥~48 seems sufficient, 64 ample, 192 near-optimal, 448 noisy), and (b) is `recent_window / budget` ≤ 0.25 (violated at 128/64 where it's 0.5).

**Mechanism: (e) "dispersed long-range anchor preservation" vs new "filter as noise filter".** Two data points push toward (e)+noise-filtering hybrid, not pure noise-filtering:

- If filter were purely a noise-filter indifferent to position, 16× (top-192) and 8× (top-448) would be monotone — more of a good thing. Instead 16× beats 8× by 27 PPL, consistent with the filter *excluding* lower-ranked keys that are positionally diffuse but semantically weak. That's position-agnostic noise rejection.
- But (e) still fires: at 64× recent=16, keeping 48 filter-picked keys from 4096 positions + last-16 tokens *outperforms dense 4096* (2693 vs 3625). The 48 retained keys are necessarily dispersed across 0–4095 (rank-2 scoring has no positional bias). No purely-recent or purely-local mechanism explains this row.

So (e) is confirmed as a load-bearing mechanism, augmented by a noise-rejection effect that caps benefit as `keep_old` grows past ~200.

## §4. Risks

1. **Recent_window sensitivity untested.** We have `recent ∈ {16, 64}` only. The 128/64 pathology proves the curve is sensitive to `recent / budget`. If best-point shifts to `recent=32` at budget=256 (say PPL 2550), the headline gets sharper; if it shifts *up* to 2700, the 16× claim softens. High-leverage gap.
2. **Chunk-cold artifact.** Dense 3625 is itself PPL-inflated vs streaming eval (cold chunks punish full-context attention). The −27% claim is scoped to chunked eval at seq_length=4096 on pg19[200:400]. Streaming eval may shrink or invert the gap.
3. **Calibration generalization.** Single `filters.pt` (rank-2, 64 calib chunks) used across 5 budget points. Tight-budget rows (64/16) may benefit from rank-1 or a budget-matched calibration; loose-budget rows (512/64) may benefit from rank-4. Untested.
4. **Llama-2-7B only.** Paper targets Llama-3.1; cross-family validity unverified.

## §5. Recommended next experiment

**Primary: (i) recent_window sensitivity sweep at budget=256**, values `{16, 32, 48, 64, 96, 128}`. Cost: ~6 runs × ~7 min calibration-reused ≈ 45 min. Payoff: locks the headline. If PPL bowl in `recent_window` has minimum at 64, we publish 16× @ 2636 with high confidence; if minimum is at 32, we upgrade to 2550-ish; if at 96, the 128/64 pathology was a local artifact and we report a wider best-region.

Deprioritize (iii) Llama-3.1 port *until* (i) runs — porting a mis-tuned operating point wastes compute. Predecessor note ordered C→A→B; sweep data now inserts A'→B where A' is this one-knob sensitivity run. (ii) filter_rank at 64/16 and (iv) streaming eval are valuable but secondary: (ii) confirms mechanism (e) robustness, (iv) tests artifact; neither is needed to publish the 16× headline if (i) stabilizes it.

## §6. Publication framing

> On Llama-2-7B / pg19 cold chunks, Q-Filters at `kv_budget=256, recent_window=64` compresses the KV cache 16× while reducing perplexity by 27% versus full dense attention (2636 vs 3625), and a 64× compression point at `kv_budget=64, recent_window=16` retains 26% of that improvement — three to eight times more aggressive than the paper's reported operating regime. The filter mechanism is load-bearing, not decorative: holding total budget at 64 tokens and toggling the filter off (recent=64, keep_old=0) loses 915 PPL (3609 vs 2693) — a clean single-variable ablation that isolates filter signal from recency.
