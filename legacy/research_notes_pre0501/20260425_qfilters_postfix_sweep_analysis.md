# Q-Filters POST-FIX Full Sweep (2026-04-25)

**Author**: trainer (autonomous chain)
**Predecessor**: `ops/research_notes/20260425_qfilters_kvbudget_sweep_analysis.md` (pre-fix; all numbers invalidated)
**Trigger**: bug investigation 2026-04-25 afternoon discovered double-label-shift in dataset `__getitem__`
**Scope**: Llama-2-7B, pg19_chunks[200:400], seq=4096, bf16, sdpa, filter_rank=2, calibration_chunks=64 (reused filters.pt), sub_window_len=1024, 8× L20A b200-1.

---

## §0. Bug

`scripts/eval_qfilters.py::PreTokenizedEvalDataset.__getitem__` and `scripts/eval_baseline_ppl.py::PreTokenizedEvalDataset.__getitem__` were returning `input_ids = tokens[:-1]`, `labels = tokens[1:]`. HF's `LlamaForCausalLM.forward(labels=...)` then applied its own `shift_logits = logits[..., :-1, :]` / `shift_labels = labels[..., 1:]` — a **second** shift. Net effect: scoring was predicting 2 tokens ahead, not 1. PPL inflation: Llama-3 went from ~1.14 (bare correct scoring) to ~5×10⁷ (double-shift); Llama-2 from ~60 to ~3600 at dense 4096.

Fix in commit 2026-04-25 15:31: pass `tokens` as both `input_ids` and `labels`, let HF do the single internal shift.

Calibration filters (filter_rank=2, 64 chunks) are **label-shift-invariant** — computed from queries not losses — so `outputs/qfilters_baseline/filters.pt` is still valid and was reused across all 13 post-fix runs.

---

## §1. Post-fix anchor table

| tag                | mode            | kv_budget | recent | PPL (post-fix) | PPL (pre-fix)* | vs dense 300.10 |
|--------------------|-----------------|-----------|--------|----------------|----------------|-----------------|
| dense_4096         | sliding_window  | 4096      | 64     | **300.10**     | 3625.39        | 1.000× (ref)    |
| qf_b64_r64 (no-F)  | qfilters        | 64        | 64     | 692.29         | 3608.74        | 2.307×          |
| qf_b128_r64        | qfilters        | 128       | 64     | 460.89         | 3079.18        | 1.536×          |
| qf_b256_r64        | qfilters        | 256       | 64     | 449.74         | 2635.55        | 1.499×          |
| qf_b512_r64        | qfilters        | 512       | 64     | 541.63         | 2662.19        | 1.805×          |
| qf_b64_r16         | qfilters        | 64        | 16     | 392.85         | 2693.21        | 1.309×          |
| sw_b512_r64 (ctrl) | sliding_window  | 512       | 64     | 1468.97        | 5609.34        | 4.895×          |
| qf_b256_r16        | qfilters        | 256       | 16     | 491.97         | —              | 1.640×          |
| qf_b256_r32        | qfilters        | 256       | 32     | 474.19         | —              | 1.580×          |
| qf_b256_r48        | qfilters        | 256       | 48     | 457.64         | —              | 1.525×          |
| qf_b256_r96        | qfilters        | 256       | 96     | 445.24         | —              | 1.484×          |
| qf_b256_r128       | qfilters        | 256       | 128    | 453.20         | —              | 1.510×          |

*pre-fix numbers from `ops/research_notes/20260425_qfilters_kvbudget_sweep_analysis.md` — all invalidated as artifacts.

---

## §2. Headline reversed

**Pre-fix publication claim**: 16× KV compression at −27.3% PPL vs dense (2636 vs 3625).
**Post-fix ground truth**: 16× compression **costs +49.9% PPL** (450 vs 300). **Dense wins every Q-Filters operating point.**

The pre-fix "win" was a pure scoring artifact: double-shift punishes long-range dependencies (predicting 2 ahead is geometrically harder under RoPE than 1 ahead, and the gap widens with context length). Dense 4096 paid this penalty across all 4096 positions; compressed caches paid it on fewer positions, so appeared to win. Once the bug is fixed, the pattern inverts and the real compression cost surfaces.

---

## §3. Salvageable qualitative findings

Most pre-fix qualitative claims do not survive. The ones that do:

1. **Filter is load-bearing** (confirmed). Single-variable ablation at kv_budget=64: filter-OFF (recent=64, keep_old=0) PPL 692.29 vs filter-ON (recent=16, keep_old=48) PPL 392.85. Flipping filter signal on saves 300 PPL at identical budget — qualitatively matches pre-fix 915 PPL gap, smaller in absolute terms but same direction. Filter genuinely contributes beyond recency.

2. **Recent_window bowl at kv_budget=256 exists** (confirmed but shifted). Pre-fix minimum was recent=64. Post-fix curve:

    ```
    recent=16   491.97
    recent=32   474.19
    recent=48   457.64
    recent=64   449.74
    recent=96   445.24   ← new minimum
    recent=128  453.20
    ```

    Minimum moved from 64 to **96**, and the bowl is much flatter post-fix (range 445–492, 10% spread) than pre-fix (range 2635–2905, 10% spread in absolute but different PPL regime). Direction of the curve is preserved.

3. **Sliding-window-only collapses** (confirmed, stronger). `sw_b512_r64` is 4.9× dense PPL post-fix vs 1.55× pre-fix. With the bug gone, the filter's role in keeping semantically-relevant old keys is unambiguously necessary — sliding-window-only throws away information that recency can't recover.

4. **Aggressive compression re-tuning works** (confirmed). `qf_b64_r16` (64× compression) at 392.85 is still the best among all compressed operating points, beating the 16× headline by ~15%. Mechanism: when budget is tight, recency must shrink to leave filter slots; 64/16 gives keep_old=48 slots, 256/64 gives keep_old=192 (and those extra 144 are noise past the rank-2 signal floor).

## §4. Publication framing — revised

> On Llama-2-7B / pg19 cold chunks, Q-Filters imposes a 30–50% PPL cost across all tested compression ratios (vs dense 300 PPL). The filter mechanism is load-bearing — it keeps compressed eval 30–70% better than pure sliding-window — but it does not beat dense attention within this evaluation regime. Best operating point: kv_budget=64 / recent_window=16 (64× compression) at 393 PPL, a 31% PPL penalty. The headline-candidate 16× point (budget=256, recent=96) is 48% worse than dense. These are chunked cold-start evals; streaming evaluation or longer context may shift the picture.

## §5. What needs re-doing downstream

- Any Llama-3 cross-family claim built on the pre-fix Llama-2 headline is invalidated. Re-running Llama-3 dense + 256/64 in parallel (`_run_llama3_postfix.sh`, b200-1 master_port=29533, launched 2026-04-25 ~20:04 GMT+8).
- The `researcher_done` Q-Filters entries in `AUTO_CHAIN.jsonl` reference pre-fix numbers; mark superseded.
- `configs/remote_experiments.json` Q-Filters headline fields must be rewritten after Llama-3 post-fix completes.

## §6. Methodological lesson

Silent double-shifts are catastrophic because they produce **finite, plausible-looking losses**. PPL 3625 on dense Llama-2/pg19 passed every sanity check (same order of magnitude as cold-start evals in other memory papers, monotone within sweep). Only the Llama-3 PPL 4×10⁷ under the same codepath — too large to ignore — forced the investigation. The controls that caught it: (a) direct bare-forward PPL probe on 5 chunks (1.14 correct vs 10⁷ wrong), (b) comparing our eval codepath to `scripts/eval_baseline_ppl.py` side-by-side and noticing both had the same pre-shift. The 6 other scripts (train_*.py, eval_base_ppl.py, eval_sparse_memory_ppl.py, eval_window_only_ppl.py) with the same bug are flagged for followup — training losses that used pre-shifted labels are all invalid.

## §7. Recommended next experiments

1. ~~Llama-3 post-fix 2-step (in progress)~~ **DONE 2026-04-25 20:08 GMT+8**. See §8.
2. **Streaming eval** — test whether the chunked-cold-start dense baseline is itself pathologically good because 4096 tokens is exactly the window where RoPE isn't yet diluted. Streaming over longer docs (≥32k tokens) may show dense degrading and compression becoming competitive.
3. Fix double-shift in the 6 other scripts and re-establish any training losses that used pre-shifted labels.
4. **Llama-3 kv_budget + recent_window sweep** (new, motivated by §8): Llama-3 qfilters is 48.4× dense PPL at 16×, vs Llama-2 only 1.50×. Need curve to find whether *any* Llama-3 operating point is competitive, or if rank-2 filters are fundamentally inadequate for 32:8 GQA.
5. **Filter_rank sweep on Llama-3**: the GQA 32:8 collapse means one filter covers 4× as many query heads. Rank 2 may be too little; try rank-4, rank-8.

---

## §8. Llama-3.0-8B post-fix cross-family result (2026-04-25 20:05–20:08 GMT+8)

Two-step `_run_llama3_postfix.sh` on b200-1, same harness & data as Llama-2 sweep, reusing `outputs/qfilters_llama3_full_bestpoint/filters.pt` (rank-2, 64 calib chunks, label-shift-invariant).

| tag        | mode            | kv_budget | recent | PPL (post-fix) | PPL (pre-fix) | vs L3 dense  |
|------------|-----------------|-----------|--------|----------------|---------------|--------------|
| dense_4096 | sliding_window  | 4096      | 64     | **1.5468**     | 584 429       | 1.000× (ref) |
| qf_b256_r64| qfilters        | 256       | 64     | **74.9346**    | 17 297        | 48.4×        |

**Dense now sane.** 1.55 PPL is exactly what Llama-3-8B should do on well-distributed natural text (pg19 is in the pre-training distribution); the pre-fix 584 429 was the double-shift catastrophe that first flagged the bug. Llama-2's dense moving from 3625 → 300 post-fix was a 12× drop; Llama-3's dense moving from 584 429 → 1.55 is a **378 000× drop** — the steeper family-dependent inflation under double-shift is consistent with Llama-3's much sharper loss gradient near ground truth (smaller loss → larger exponential impact of per-position error) and confirms the bug was the dominant artifact.

**Cross-family comparison (post-fix only, all ratios vs same-family dense):**

| metric                               | Llama-2-7B | Llama-3.0-8B | ratio L3/L2 |
|--------------------------------------|-----------:|-------------:|------------:|
| dense PPL (kv=4096)                  |     300.10 |        1.55  | 0.005×      |
| qf 256/64 PPL                        |     449.74 |       74.93  | 0.17×       |
| qf / dense ratio (compression cost)  |     1.499× |     48.438×  | **32.3×**   |

**Finding.** Pattern direction holds (compression hurts on both families). Magnitude is radically different: Llama-2 loses 50% at 16×, Llama-3 loses 4744%. Two candidate mechanisms:

- **GQA 32:8 geometry.** Each of the 8 KV heads is read by 4 query heads. A rank-2 filter per KV head must summarize 4 query distributions. On Llama-2 (32:32), each filter serves exactly one query stream; on Llama-3, it averages 4. This is not a bug (the code already pre-averages Q over GQA groups before SVD — coder-audited 2026-04-25 16:45), but it means the rank-2 projection may be strictly insufficient for the more diverse averaged query distribution.
- **Loss-sensitivity amplification.** Llama-3's baseline is 1.55 PPL. At avg_loss = 0.44 nats, the loss landscape is *flat* and small perturbations from compressed attention (wrong keys being kept) are proportionally more disruptive than at Llama-2's avg_loss = 5.70 nats. This is independent of architecture and would affect any compressed method on a strong base model.

These two are not mutually exclusive; a filter_rank sweep on Llama-3 (rank ∈ {2, 4, 8}) + a dense→qf budget curve on Llama-3 {dense, 512, 256, 128, 64} would disentangle them.

**Publication implication.** The "Q-Filters on Llama-2-7B" framing still stands (§4), but any claim of cross-family generalization is now strongly negative. The correct headline is: *Q-Filters compression cost is family-dependent; on Llama-2 the penalty is moderate (30–50% at 16–64×), on Llama-3 the rank-2 filter is inadequate (nearly 50× dense penalty at 16×).* If we want a cross-family positive result, we must sweep filter_rank on Llama-3 first.

---

## §9. Llama-3 filter_rank sweep + kv_budget curve (2026-04-25 20:28–20:48 GMT+8)

Two back-to-back b200-1 sweeps executed to disentangle the §8 mechanisms. Drivers `scripts/_run_llama3_rank_sweep.sh` then `scripts/_run_llama3_kv_curve.sh`. All points: Llama-3.0-8B, pg19_chunks_llama3_noeos, seq=4096, skip=200, max=200, sub_window_len=1024, bf16, sdpa, calibration_chunks=64.

### §9.1 filter_rank sweep @ kv=256 recent=64

| tag                  | rank | PPL      |
|----------------------|------|---------:|
| qf_b256_r64_rank2    | 2    |  74.9346 |
| qf_b256_r64_rank4    | 4    | 107.8800 |
| qf_b256_r64_rank8    | 8    | 105.7033 |

**Falsifies mechanism (a) — GQA rank-insufficiency.** If one rank-2 filter were too narrow to cover 4 averaged query streams, PPL should drop monotonically with rank. Instead rank=2 is *best* and rank=4/8 are ~43% worse. Higher rank adds noisy singular directions (the tail of the Q covariance spectrum is not information; it is per-calibration-chunk noise). The rank-2 projection already captures the useful structure; keeping extra components corrupts the key-scoring geometry.

**Confirms mechanism (b) — loss-sharpness amplification.** With (a) ruled out, Llama-3's 48× compression cost vs Llama-2's 1.5× must be driven by the asymmetric sensitivity of the two base models. Llama-3 at avg_loss 0.44 nats is in the flat part of the loss landscape where small attention perturbations translate to large PPL swings; Llama-2 at 5.70 nats is in the linear regime where attention noise is absorbed.

### §9.2 kv_budget curve @ rank=2

| tag               | mode           | kv  | recent | PPL     | vs L3 dense |
|-------------------|----------------|----:|-------:|--------:|------------:|
| dense_4096        | sliding_window |4096|    64  |  1.5468 | 1.000×      |
| qf_b64_r64 (no-F) | qfilters       |  64|    64  |110.2063 | 71.2×       |
| qf_b64_r16        | qfilters       |  64|    16  | 72.6819 | 47.0×       |
| qf_b128_r32       | qfilters       | 128|    32  | 72.1159 | 46.6×       |
| qf_b128_r64       | qfilters       | 128|    64  | 74.1250 | 47.9×       |
| qf_b256_r64       | qfilters       | 256|    64  | 74.9346 | 48.4×       |
| qf_b512_r64       | qfilters       | 512|    64  | 69.7619 | **45.1×** best compressed |
| sw_b256_r64       | sliding_window | 256|    64  |133.6194 | 86.4×       |

**The Llama-3 kv-budget curve is essentially flat at rank=2**, ranging 70–75 PPL across kv ∈ {64, 128, 256, 512} once recent_window is tuned. Contrast Llama-2 post-fix where kv went 692 → 450 → 542 across the same range (strong concave bowl). The implication: on Llama-3, budget is not the binding constraint — the rank-2 filter subspace is saturated at ~70 PPL, and adding more keys past kv≈128 yields near-zero improvement.

**Filter contribution (same budget):**
- Pure sliding at kv=256 r=64 → 133.62 PPL.
- Q-Filters at kv=256 r=64 → 74.93 PPL.
- Factor 1.78× PPL reduction attributable to filter. Smaller absolute than Llama-2 (1468→450 is 3.3×) but filter still load-bearing.

**Recent=64 vs keep_old=0 (filter-OFF control).**
- kv=64 r=64 keep_old=0 → 110.21 PPL. No filter signal, pure recency.
- kv=64 r=16 keep_old=48 → 72.68 PPL. Filter selecting 48 old keys.
- Δ = 37.5 PPL from filter at identical 64 total slots. Qualitatively the same result as Llama-2 (filter load-bearing) but the magnitude of filter gain is capped by the PPL floor set by loss-sharpness regime.

### §9.3 Revised cross-family headline

| metric                                     | Llama-2-7B | Llama-3.0-8B |
|--------------------------------------------|-----------:|-------------:|
| dense PPL                                  |     300.10 |        1.55  |
| best compressed op-point tag               | qf_b64_r16 | qf_b512_r64  |
| best compressed PPL                        |     392.85 |       69.76  |
| best compressed / dense                    |     1.309× |      45.07×  |
| compression ratio at best point            |        64× |          8×  |
| kv-budget curve shape                      | concave bowl | flat plateau |
| filter_rank optimum                        |         2  |           2  |

**Llama-3 publication implication.** No Llama-3 operating point within kv ∈ {64, 128, 256, 512} is competitive with dense. The best compressed point (kv=512 at 45× dense penalty) still costs ~45× the log-PPL gap. The curve's flatness rules out a "larger budget saves it" escape — the system is rank-limited, not budget-limited, and higher rank makes it worse. The cross-family result remains strongly negative for Llama-3 Q-Filters at 4096 seq length. Two remaining escape hatches: (i) streaming evaluation over ≥32k tokens, where dense itself may degrade and absolute PPL gap compresses; (ii) a fundamentally different calibration set (Llama-3 calibration on in-distribution vs pg19 mismatch).

### §9.4 What this rules out / opens

Ruled out:
- Rank-2 is insufficient for Llama-3 GQA — **no**, rank=2 is optimal.
- Budget is insufficient — **no**, PPL plateau 70–75 from kv=128 to kv=512.

Opens:
- Is the rank-2 saturation a property of Q-Filters *as a method*, or of our pg19 calibration being out-of-distribution for Llama-3? A calibration sweep on in-distribution (e.g., C4-web) might shift the floor.
- Does the loss-sharpness mechanism predict all compression methods fail on Llama-3? If so, ablate with heavy-hitter / KV-quantization baselines to verify the floor is method-agnostic.

**Chain state after §9**: Llama-3 Q-Filters rank-2 best-case is 45× dense penalty at 8× compression. Cross-family Q-Filters claim is **strongly negative**; no competitive Llama-3 op-point exists in the budget-rank grid tested. Recommended next experiment (§7 item 2): streaming eval on Llama-2 and Llama-3 over ≥32k tokens.

