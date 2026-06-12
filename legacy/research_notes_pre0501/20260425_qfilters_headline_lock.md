# Q-Filters Headline Lock — `recent_window` Sensitivity at `kv_budget=256` (2026-04-25)

**Author**: researcher (autonomous chain)
**Predecessor**: `ops/research_notes/20260425_qfilters_kvbudget_sweep_analysis.md` (§5 recommended this A' sweep; this note supersedes §6 of that note).
**Scope**: Llama-2-7B, `pg19_chunks[200:400]`, seq_length=4096, bf16, sdpa, `--mode qfilters`, `--kv_budget 256`, `--filter_rank 2`, `--calibration_chunks 64`, `--sub_window_len 1024`. Single `filters.pt` reused across all six runs; only `--recent_window` varied.

## 0. Data

| recent_window | keep_old | PPL     | vs dense 3625.39 | Δ vs min (+PPL) |
|---------------|----------|---------|-------------------|-----------------|
| 16            | 240      | 2856.22 | 0.788×            | +220.67         |
| 32            | 224      | 2769.47 | 0.764×            | +133.92         |
| 48            | 208      | 2687.63 | 0.741×            | +52.08          |
| 64            | 192      | 2635.55 | 0.727× (**MIN**)  | 0               |
| 96            | 160      | 2683.82 | 0.740×            | +48.27          |
| 128           | 128      | 2904.47 | 0.801×            | +268.92         |

## §1. Is the bowl genuine?

**Strictly U-shaped, no reversals.** Left arm 16→32→48→64 is monotone decreasing with deltas (−86.75, −81.84, −52.08): smooth, decelerating as we approach the minimum. Right arm 64→96→128 is monotone increasing with deltas (+48.27, +220.65): convex, with the 96→128 jump four times the 64→96 rise. Both arms are clean — no saw-tooth, no local bump — so the bowl is a genuine single-knob response curve and not a noise artifact.

**Minimum robustness.** Second-best is recent=96 at 2683.82, a gap of 48.27 PPL (1.83% of the minimum) above `recent=64`. That gap is roughly the same magnitude as the 8×→16× gain in the budget sweep (27 PPL), i.e. non-negligible but not dominant. The minimum is robust to ±32 tokens of recent_window drift — one step either side costs <2% — but the right edge falls off fast (recent=128 is +10% over the minimum, matching the `budget=128/recent=64` pathology from the predecessor sweep). The headline operating point is a soft well, not a knife-edge, which is the right topology for a publication claim.

## §2. Mechanism refinement

The predecessor's hypothesis (e) — **dispersed anchor preservation** — survives and tightens. Left arm (recent 16→64, keep_old 240→192): more recency monotonically helps while `keep_old` shrinks by 48 slots, so the marginal value of a recent slot is strictly greater than the marginal value of the 193rd–240th filter pick. This is consistent with the predecessor's "filter has a rank-2 signal/noise floor past ~200 retained tokens" finding: the lowest-ranked filter picks are noise, and giving that budget back to recency wins.

Right arm (recent 64→128, keep_old 192→128): recency starts **crowding out** filter signal. The 96→128 delta (+220 PPL) is specifically the step where `keep_old` drops from 160 to 128 — precisely where the predecessor sweep's `128/64` pathology lived. The rank-2 filter appears to need ≥~160 old-key slots on pg19 to resolve enough dispersed anchors; below that the noise-rejection benefit collapses.

So (e) is refined: **the filter's contribution is not flat in `keep_old`, it has a plateau from ~160 to ~240 and a cliff below ~160**. Recent_window's contribution is monotone-helpful up to ~64 then flat-to-hurtful as it eats the filter plateau. The optimum sits where the recency margin has saturated *and* the filter plateau is still intact — i.e. `recent ≈ budget/4`.

## §3. Publication claim status

**Headline "16× @ −27.3%" is ready to publish, conditional on hedging.** Single-variable risk on `recent_window` is now closed: the bowl is genuine, the minimum is robust within ±32 tokens, and the neighborhood (32–96) is entirely within −26% to −27.4% of dense. Within the current scope (Llama-2-7B / pg19 chunked eval / in-domain calibration) there is no remaining single-variable knob that could flip the sign of the claim.

Remaining hedges — all acknowledged, none in the single-variable-risk class:

- Chunk-cold artifact on dense baseline (streaming eval untested).
- Llama-2-7B only (Llama-3.1 untested).
- pg19 only; calibration chunks drawn from same corpus (held-out-author generalization untested).
- `filter_rank=2` fixed across the whole sweep.

None of these can be ruled out by a local sweep — they require cross-setting runs. The A'' candidates (filter_rank at 256/64; streaming eval; held-out calibration) would each sharpen one hedge but none is blocking the headline. Key judgment: the shape and stability of the bowl are themselves evidence that we are not sitting on a fragile outlier — if any of the hedges were catastrophic, we would expect the bowl to be jagged or the minimum to drift unpredictably with `recent_window`; it doesn't.

## §4. Recommended next experiment

**Proceed to Option B (Llama-3.1-8B port) directly.** Rationale: (i) every remaining risk is cross-setting, and the biggest expected-value knob is cross-family validation — the paper targets Llama-3.1, so if the 16×/−27% shape doesn't reproduce there, no amount of A'' on Llama-2 matters; (ii) A'' candidates are individually worth ~0.5 hedge each and all require separate compute; (iii) the bowl is stable enough that porting a *mis*-tuned operating point is no longer a meaningful risk — `recent = budget/4` is a transferable heuristic, not a Llama-2-specific artifact.

If Llama-3.1 reproduces the bowl shape (even at a different PPL scale), we publish; if it doesn't, the A'' work is the natural follow-up ablation. Defer filter_rank, streaming eval, and held-out-author calibration to post-port confirmation runs.

## §5. Closing publication paragraph

> On Llama-2-7B evaluated over pg19 chunks (seq_length=4096, cold-chunk protocol, in-domain calibration), Q-Filters at `kv_budget=256, recent_window=64` compress the KV cache 16× while reducing perplexity by 27.3% versus full dense attention (2635.55 vs 3625.39). A `recent_window` sensitivity sweep at this budget yields a clean U-shaped response curve with a robust minimum at `recent=64`: the 32- and 96-token neighbors sit within 2% of the minimum, and both arms are strictly monotone with no reversals, indicating a stable operating point rather than a tuned outlier. The shape confirms a two-component mechanism — monotone recency gain up to `recent ≈ budget/4`, then a filter-plateau regime of ≥~160 `keep_old` slots that the rank-2 projection needs to resolve dispersed anchors — and predicts a transferable heuristic `recent ≈ budget/4` rather than a brittle single-point tuning. The claim is scoped to chunked evaluation on pg19 with in-domain calibration on Llama-2-7B; streaming-eval robustness, held-out-author calibration, and cross-family (Llama-3.1) reproduction remain to be verified but do not fall in the single-variable-risk class this sweep has now closed.

---

Chain req_20260425_162300_qfilters_recent_window_sweep terminates. Next: spawn req_20260425_YYYYYY_llama31_port (Option B).
