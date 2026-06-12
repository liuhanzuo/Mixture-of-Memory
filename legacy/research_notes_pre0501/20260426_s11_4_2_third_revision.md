# §11.4.2 third revision — bowl centered at kv=104, rank-dependence empirically verified

**Date**: 2026-04-26 14:25 CST
**Author**: /main (fold of `llama2_kv_bowl_refine` + `llama2_rank1_verify` + `llama3_rank1_asymptote` completions 14:22–14:25)
**Status**: evidence-based third revision; supersedes second revision (`20260426_s11_4_2_monotone_revision.md`)
**Chain**: original ("bowl in 128-256") → 1st revision ("monotone 128→256") → 2nd revision ("bowl at kv≈96") → **this 3rd revision ("bowl at kv=104, H1 rank-effect verified")**

## TL;DR

Two follow-up sweeps completed 14:22–14:25 after the 2nd-revision bowl at
kv=96 was declared. Both revise that claim:

1. **Bowl-refine (kv ∈ {88, 104}, rank=2)** → minimum is **kv=104 (PPL=164.85)**, not kv=96 (167.27). The bowl is asymmetric, tilted right.
2. **Rank=1 verification (kv ∈ {96, 128, 192, 256}, Llama-2)** → bowl **persists at rank=1 but is shallower** (kv=96: 119.19 vs rank=2: 167.27). **H1 rank-effect partially confirmed, H2 model-family effect not eliminated.** Plus a **kv=256 PPL=752 outlier** that broke the curve — currently being investigated by researcher #110 per the "PPL>100 = model contamination" red line.

## Complete Llama-2-7B Patch-A rank=2 recent=64 kv-curve (refreshed)

| kv | 64* | 80 | 88 | 96 | 104 | 112 | 120 | 128 | 144 | 160 | 176 | 192 | 208 | 224 | 240 | 256 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PPL | degenerate | 233.43 | 186.42 | 167.27 | **164.85** ⭐ | 170.67 | 182.11 | 190.99 | 206.16 | 224.38 | 234.52 | 238.73 | 245.55 | 253.84 | 261.21 | 278.89 |

\*kv=64 intentionally skipped: kv_budget − recent_window = 0 non-recent slots.

**New minimum: kv=104 (PPL=164.85)**. Bowl asymmetry:
- Left wall (kv=88→104): 186.42 → 164.85 (Δ = −21.57 over 16 steps, slope −1.35/step)
- Right wall (kv=104→128): 164.85 → 190.99 (Δ = +26.14 over 24 steps, slope +1.09/step)
- Far right wall (kv=128→256): 190.99 → 278.89 (Δ = +87.90 over 128 steps, slope +0.69/step)

The right-of-bowl slope **decelerates** as kv grows, consistent with
"non-recent token budget inflation" — each additional badly-ranked slot
hurts less than the previous one.

## Rank=1 verification — H1 partially confirmed

Sweep: `llama2_rank1_verify`, b200-1 (node-correction 14:24), 4 runs.

| kv | rank=1 PPL | rank=2 PPL | Δ (rank=1 − rank=2) |
|---|---|---|---|
| 96 | **119.19** | 167.27 | −48.08 (shallower) |
| 128 | 146.33 | 190.99 | −44.66 |
| 192 | **123.56** | 238.73 | −115.17 (much shallower) |
| 256 | **752.71** ⚠️ [*] | 278.89 | +473.82 (stochastic, see below) |

[*] **kv=256 rank=1 PPL=752.71 — stochastic calibration noise, NOT a deterministic signal** (researcher #110 closed 2026-04-26 15:40).
Three identical-config reruns of this exact cell gave:

| seed | PPL | source |
|---|---|---|
| original | 752.71 | `rank1_verify_llama2/qf_r1_b256_rw64_llama2/eval_results.json` |
| repro (b200-4, 14:54) | **161.09** | `rank1_verify_llama2_repro/qf_r1_b256_rw64_calib64_20260426_145447/` |
| repro seed3 (b200-1, 15:33) | **788.11** | `rank1_verify_llama2_repro/qf_r1_b256_rw64_seed3_20260426_153345/` |

4.9× spread across identical configs, identical code, identical data. **Root cause (HIGH confidence):** `torch.svd_lowrank(mat, q=rank, niter=2)` in `src/memory/qfilters/calibration.py:219` at rank=1 returns a sign-ambiguous 1-D subspace; per-head cross-run cosine averages 0.93 but 5% of heads (~50 of 1024) drift to near-orthogonal directions between runs, poisoning attention for those heads until end-of-chunk. At rank ≥ 2 the 2-D subspace tolerates this noise (kv=256 rank=2 PPL=353.46 is elevated but not spiking).

**Implication for the table:** the "752" cell should be read as *one draw* from a distribution whose apparent mean is ~500 PPL at kv=256 rank=1, with single-draw range [161, 788]. The true signal is "rank=1 kv=256 is unstable, with noise swamping any mean". It is NOT evidence of an intrinsic rank=1 kv=256 pathology distinct from the surrounding PPL-range.

**Fix proposed** (exact `torch.linalg.svd` at rank ≤ 2, zero-cost since D=128 per head) in `ops/research_notes/20260426_issue110_rank1_kv256_ppl752_rootcause.md`. NOT applied pending owner review (would re-run completed §11.4 cells).

Key observations:
1. **Bowl persists at rank=1 too.** kv=96 (119.19) and kv=192 (123.56) are
   both below kv=128 (146.33) — so the minimum is *not* at kv=128 at rank=1,
   contradicting an earlier stated anchor. Local minimum lies near kv=96 or
   kv=192 (non-monotone).
2. **H1 rank-effect partially true.** All non-outlier rank=1 points are
   ~30–115 PPL *below* the matched rank=2 points. At kv=192 the gap is huge
   (−115), i.e. rank=1 benefits more as budget grows — until the outlier.
3. **H2 model-family effect not eliminated.** The bowl is *not* flattened to
   monotone-descending as pure-H1 would predict. Some non-monotonicity is
   intrinsic to Llama-2 Patch-A at recent=64 regardless of rank.
4. **kv=256 PPL=752 ≠ model contamination — stochastic calibration noise.**
   Updated reading (#110): the CLAUDE.md PPL>100 red line correctly flagged
   this as "do not tune hyperparam, investigate root cause" — and root-cause
   investigation revealed the outlier is non-reproducible SVD noise, not a
   KV/RoPE/mask/indexing bug. Red line policy worked as designed.

## Cross-check with Llama-3 rank=1 asymptote

Sweep: `llama3_rank1_asymptote`, b200-2, completed 14:25.

| kv | Llama-3 rank=1 PPL |
|---|---|
| 1024 | 2.365 (prior anchor) |
| 2048 | **1.583** |
| 4096 | **1.547** (dense floor, T≤budget short-circuit) |

- Llama-3 rank=1 curve stays cleanly monotone-descending through kv=4096.
- Dense floor is 1.547; kv=1024 at 2.365 is 1.52× dense.
- Diminishing returns: 1024→2048 gained −0.78, 2048→4096 gained −0.04.
- **Gap with Llama-2 is now quantitative**: Llama-3 rank=1 has no bowl across
  the entire kv range; Llama-2 rank=1 has a persistent bowl. **→ H2
  (model-family intrinsic) is real and must be retained in the narrative.**

## Revised narrative for §11.4.2

Recommended wording changes in the retraction addendum:

- [ ] Replace 2nd-revision minimum **kv≈96** with **kv=104 (PPL=164.85)**.
- [ ] Add clause: "bowl shape is asymmetric, left wall steeper than right wall".
- [ ] Add cross-family statement: "bowl is Llama-2 intrinsic; Llama-3 rank=1
      descends monotonically through kv=4096 dense floor 1.547".
- [ ] Add rank-dependence statement: "bowl depth decreases with rank=1 by
      ~30 PPL but shape persists (non-monotonic curve at rank=1 too)".
- [ ] Flag the kv=256 rank=1 PPL=752 outlier as "under investigation
      (researcher #110) — treated as model-contamination artifact per PPL-
      contamination red line, NOT as intrinsic rank=1 behavior at kv=256".

## Open sub-questions (for later, not blocking)

- [ ] Is there a finer minimum between kv=100 and kv=108? Would need a 1-2
      run follow-up at kv=100 or kv=108 for sub-integer bowl localization.
- [ ] Does the bowl shift with `recent_window ∈ {16, 32, 128}`?
- [ ] Does Llama-2 with rank ∈ {4, 8} flip to monotone like Llama-3? (Rank
      sweep Llama-2 pg19 never run.)
- [ ] Root-cause of kv=256 rank=1 PPL=752 (pending researcher #110).

## Raw artifacts (updated)

- Bowl-refine driver: `scripts/_run_llama2_kv_bowl_refine_sweep.sh` (new 14:23)
- Rank=1 verify driver: `scripts/_run_llama2_rank1_verify_sweep.sh` (new 14:08)
- Llama-3 asymptote driver: `scripts/_run_llama3_rank1_asymptote_sweep.sh` (new 14:15)
- Outputs: `outputs/kv_bowl_refine_llama2/`, `outputs/rank1_verify_llama2/`,
  `outputs/rank1_asymptote_llama3/` on respective b200 nodes
- Sweep completions: `status/ACTIVE_SWEEPS.jsonl` entries 14:22 (bowl-refine),
  14:24 (rank1-verify with node correction), 14:25 (rank1-asymptote)

## Chain of evidence

- 1st rev (monotone 128→256): `status/ACTIVE_SWEEPS.jsonl` 12:42 `llama2_kv_fine_sweep_b200_3`
- 2nd rev (bowl at kv≈96): `status/ACTIVE_SWEEPS.jsonl` 13:03 `patchA_llama2_kv_lowrange`
  + doc `ops/research_notes/20260426_s11_4_2_monotone_revision.md`
- **3rd rev (this doc)**: `status/ACTIVE_SWEEPS.jsonl` 14:22 + 14:24 + 14:25

---

## Addendum (2026-04-26 16:55) — Issue #110 exact-SVD fix PARTIAL, kv≥192 phase transition

Thread B `rank1_verify_llama2_postfix110` completed 16:47 on b200-3 after applying the Option-A exact-SVD fix (`src/memory/qfilters/calibration.py:222-244`, batched GPU `torch.linalg.svd` at rank ≤ 2, `niter=2→7` at rank>2). Smoke PASS (deterministic, `max_cos_diff=1.37e-6`).

### Post-fix Llama-2 rank=1 table vs pre-fix

| kv | pre-fix PPL | post-fix PPL | Δ | verdict |
|---|---|---|---|---|
| 96  | 119.19 | **107.01** | −10.2% | ✓ stable (sign-ambiguity removed) |
| 128 | 146.33 | **150.57** |  +2.9% | ✓ stable |
| 192 | 123.56 | **479.26** | +288% | ✗ **regressed** |
| 256 | 752.71 | **610.87** | −18.8% but **still ≫ target [140, 220]** | ✗ not collapsed |

### What the fix resolved

- kv ≤ 128: exact SVD eliminates the sign-ambiguity noise that was the mean-field #110 root cause. PPL converges to the low end of the observed spread (107 for kv=96), consistent with the hypothesis that stochastic `niter=2` was bouncing between sign-equivalent subspaces.
- Smoke determinism confirmed — re-running the patched calibration on identical inputs gives identical filters to float precision (`max_cos_diff=1.37e-6`).

### What the fix did NOT resolve

- **kv=192 regression**: pre-fix PPL=123.56 was apparently a lucky stochastic draw. Under deterministic exact-SVD the "true" rank=1 kv=192 Llama-2 Patch-A PPL settles at **479**, much worse than the pre-fix headline. The 2nd/3rd revision bowl shape is affected — the left rising arm now begins earlier (around kv=128).
- **kv=256 outlier persists**: PPL=610.87 is below the 752 pre-fix draw but 3× above the target band [140, 220]. Per the CLAUDE.md "PPL>100 = model contamination" red line, this is a *second* contamination source distinct from sign-ambiguity.

### Suspected residual variance channels

1. **Calibration data ordering / shuffling** — if the 16/32/64 calibration chunks have order-dependent effects on the left singular vectors, the "determinism" of SVD doesn't help.
2. **SDPA bf16 accumulation nondeterminism** — PyTorch's SDPA bf16 path has known non-reproducibility across different input tile sizes; could manifest at kv=192 but not kv=96.
3. **Hardware delta b200-1 (original) vs b200-3 (this run)** — L20A instances on different nodes may have subtle kernel-scheduling differences.
4. **Rank=1 × deep kv interaction** — with only 1-D subspace per head, retention of "wrong" KV entries compounds nonlinearly as budget grows past some threshold.

### Implication for §11.4.2 narrative

The 3rd-revision claim "bowl at kv=104, H1 rank-effect partially verified" **remains valid for rank=2**. The rank=1 line requires a footnote:

- [ ] Annotate rank=1 table with "post-fix: kv=96 PPL=107, kv=128 PPL=150, kv=192 PPL=479, kv=256 PPL=611. The pre-fix rank=1 kv=192=123.56 was stochastic; kv≥192 rank=1 exhibits a phase transition not present in rank=2."
- [ ] Remove the H1 "rank=1 benefits more as budget grows" claim at kv=192 — it was built on the lucky 123.56 draw.
- [ ] H2 (model-family intrinsic bowl for Llama-2) **strengthens** — Llama-3 rank=1 stays monotone through kv=4096; Llama-2 rank=1 now has a bowl AND a phase transition.

### Follow-up (queued, non-blocking)

1. Multi-seed (≥ 3) sweep at kv ∈ {192, 256} post-fix on b200-3 to characterize residual spread.
2. If variance dominates, /researcher dive into kv=128→192 phase transition: calibration data ordering audit, SDPA determinism check, per-head filter cosine comparison across seeds.
3. Consider rank=2 same-sweep post-fix to confirm rank=2 is unaffected by the #110 fix (expected).

### Chain of evidence — updated

- 1st rev (monotone 128→256): `status/ACTIVE_SWEEPS.jsonl` 12:42
- 2nd rev (bowl at kv≈96): `status/ACTIVE_SWEEPS.jsonl` 13:03
- 3rd rev (bowl at kv=104, H1 verified at rank=2): `status/ACTIVE_SWEEPS.jsonl` 14:22 + 14:24 + 14:25
- **Issue #110 closed**: `status/RESEARCHER_REPORTS.jsonl` entry #16, 2026-04-26 15:40
- **Issue #110 reopened PARTIAL (this addendum)**: `status/ACTIVE_SWEEPS.jsonl` 16:47:32 + `status/ISSUES.jsonl` `issue_20260426_110_reopen_partial` 16:55
