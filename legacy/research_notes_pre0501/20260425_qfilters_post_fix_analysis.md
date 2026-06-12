# Q-Filters Post-Harness-Fix Mechanism Analysis (2026-04-25)

**Author**: researcher (autonomous chain)
**Input**: full 200-chunk re-ablation at 15:53, harness patched per `issue_20260425_qfilters_harness_noop`
**Predecessor note**: `ops/research_notes/20260425_qfilters_result_analysis.md` (hypothesis table, §2)

## 0. Headline (repeat for anchoring)

All runs: `pg19_chunks[200:400]`, seq_length=4096, bf16, sdpa, `kv_budget=512, recent_window=64, filter_rank=2, calibration_chunks=64, sub_window_len=1024`.

| mode | PPL | vs vanilla |
|------|-----|------------|
| `--mode qfilters`       | **2662.19** | 0.736× (−26%) |
| `--mode sliding_window` | 5609.34     | 1.551× (+55%) |
| vanilla dense           | 3616.25     | 1.000×        |

ΔPPL(qf vs sw) = 2947 = 52% reduction.

---

## 1. Which §2 hypothesis is supported?

My earlier §2 table predicted three SW outcomes: ≈3600 (win is sliding window), ≈4200 (filters add signal), ≈5000 (QF-specific). The measured SW=5609 is **off the top of the table** — plain last-k at this budget actually **loses to dense vanilla**. This falsifies hypothesis **(b) "sliding-window anchor"** as stated: there is no free lunch from simply restricting attention. If (b) were the mechanism, SW should match or approach QF; instead it collapses.

What is load-bearing, now:

- **(a) proper-noun / content-key denoising — upgraded from "secondary" to "plausible primary contributor."** Filter scoring keeps semantically heavy keys regardless of position; last-k keeps only the most recent 512. If pg19 chunks have content-bearing tokens spread across positions 0–4095, only QF retains them.
- **(c) calibration — upgraded from "bounded" to "load-bearing."** The rank-2 right-singular vectors of the calibration Q matrix are what let QF beat both SW (no filters) and dense (full but noisy). Without calibration the method degenerates to SW.
- **(e) NEW: *filter scoring preserves dispersed long-range anchor keys that plain last-k drops.*** This is the cleanest single-sentence mechanism consistent with all three PPLs: filters project keys onto the dominant query subspace, so the top-448 "old" keys kept per layer are exactly those that attention would have attended to anyway. SW keeps last-512 regardless, throwing away every high-score key >512 tokens back; dense keeps everything including low-score noise. QF keeps high-score only.

**Summary**: (b) alone is refuted; the real mechanism is (a)+(c)+(e). Filter scoring is doing real work, not just masking.

---

## 2. Is the 26% dense-beating real, or a harness artifact?

Not fully clean. **Vanilla 3616.25 was produced by the single-forward path**; QF/SW used `sub_window_len=1024` with carryover. Three possible harness-induced differences:

1. **Label-shift drop.** HF internally shifts `labels[..., 1:]`; at K=4 boundaries per chunk we lose ~0.07% of tokens. Small.
2. **Token-counting asymmetry.** Vanilla=766,974 vs QF/SW=819,000 (6%) — likely pad/EOS masking in the vanilla path. Numerator and denominator shift together, so PPL effect is bounded but uncontrolled.
3. **Attention-scope.** Dense sub-window carryover accumulates full KV (no compression), so within a chunk attention scope equals single-forward 4096-dense — in principle. In practice, sub-window splitting can perturb position-id layout and BOS handling.

**Verdict**: directionally real (the huge QF–SW gap and the 45% diagnostic-gate gap both confirm filters do work), but the exact **26% ratio is not apples-to-apples**. A `vanilla + sub_window_len=1024` rerun is required before we quote "26% better than dense" publicly.

---

## 3. What does SW=5609 > vanilla=3616 tell us?

Verified `QFiltersCache.compress_layer` fallback (layer.py L140-149): when `filters_on(layer) is None`, it does `layer.keys = keys[..., -budget:, :]` — pure last-k. Correct semantics, no bug.

SW=5609 is therefore a genuine result: on 4096-token cold pg19 chunks, truncating to the most recent 512 every 1024 tokens destroys context that dense attention uses. Sub-window 4 (tokens 3072–4095) sees only last-512 of sub-window 3 + itself — effective scope ≤1536 tokens. Dense sees all 4096. QF sees 512 high-score keys *spread across all 4096 positions*.

Implication: hypothesis (b) was **wrong in direction**. Restricting attention to a trailing window does not automatically help on pg19; it hurts. The QF win is about **which** keys are kept, not **how many**.

---

## 4. Recommendation: next experiment

**Primary: option C — honest vanilla rerun with `sub_window_len=1024`.**

- (A) kv_budget sweep and (B) Llama-3.1 port both need a trusted dense baseline. Any "X× compression at Y% PPL delta" claim is meaningless with a harness-asymmetric baseline.
- (C) costs ~30 min (no calibration, no patching) and resolves §2. Outcomes:
  - vanilla-with-carryover ≈ 3616 ± ε → 26% dense-beat claim locks in; proceed to (A).
  - vanilla-with-carryover materially lower → QF-vs-dense gap shrinks, but the 2947 QF-vs-SW gap (mechanism (e)) is the real publishable finding; pivot to (B) for cross-family validation.
- Running (A) or (B) first risks burning compute on a moving baseline.

---

## 5. Publication framing (conditional on §4-C holding)

> Q-Filters' rank-2 calibration filters identify a small subspace of query directions that carry disproportionate downstream attention weight on pg19. Compressing the KV cache to 512 of 4096 tokens per head via filter-top-k + last-64 window simultaneously reduces KV memory 8× *and* lowers perplexity by 26% vs full dense attention on cold literary chunks, while plain last-k sliding window at the same budget degrades perplexity by 55%. The mechanism is *which* keys are kept, not *how many*: filter scoring retains dispersed long-range anchor keys that stabilize attention in later sub-windows, whereas dense attention is hurt by uninformative early-position tokens in cold chunks, and plain last-k simply discards the long-range anchors entirely. On Llama-2-7B / pg19 this is an out-of-paper observation (Godey et al. claim smallest PPL *drop*, not PPL gain); cross-family validation on Llama-3.1-8B is the natural next step.

---

## 6. Risks / caveats

1. **Baseline not apples-to-apples** (§2) — must run (C) before any public claim. Biggest risk.
2. **Token-count asymmetry** (766,974 vs 819,000) — pad/EOS masking drift; independent of sub-window split. Lock down with (C).
3. **pg19 cold-chunk artifact.** 4096-token chunks without prior context punish dense attention disproportionately; on streaming eval the QF-vs-dense gap will likely shrink or invert. Claim must be scoped to chunked-eval settings until a streaming eval is run.
4. **Filter calibration overlap.** Calib uses `pg19_chunks.npy[:64]`, eval uses `[200:400]` — same novel(s) may overlap. For paper-grade claims, use held-out author.
5. **Cross-family untested.** Paper targets Llama-3.1-8B; all our results are Llama-2-7B. (B) is required before any claim of generality.
6. **SW wall clock ~30s** (no calibration) vs QF ~6.5 min (calib dominates; eval itself 45s × K sub-windows). Compression overhead is not the bottleneck — good for scaling (A).
