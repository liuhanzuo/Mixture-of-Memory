# Post-ROUTE-A Redirect — Re-judging the Long-Context Root Cause

**Date:** 2026-06-11
**Author:** general-purpose-53 (research, no code changes, no training started)
**Trigger:** ROUTE-A routing-balance sweep (RUN_REGISTRY §3d) REJECTED all 3 single-knob arms; arm4 (ST-Gumbel, usage_cov→1.0) eval still in flight. The project's standing hypothesis — "long-context decay = selector routing concentration (usage_cov~0.25)" — must be revised.
**Scope:** pure log/registry/code audit of the P11 chunk512 delta-rule+normreadout line and its ablations (§3c/§3d). Builds on gp-44 (`20260610_late_chunk_memory_collapse_rootcause.md`) and gp-46 (ROUTE-A direction check).

---

## TL;DR

1. **ROUTE-A證伪了"路由集中=病根"。** Three single-knob arms (loss_free 0.01 / entropy_aux 0.01 / temp 40→20) all REJECTED on the long-range cell; even the best (arm1 step1000) gives qa5 8k/16k/32k = 36/42/31 vs P11 base 48/45/44. arm4 lifted training usage_cov 0.25→1.0 but its eval is pending — **prediction: arm4 also fails the long-range cell** (see §3). So routing concentration is a **symptom, not the病根**.

2. **The "病根 ∈ {(a) read-collapse, (b) write-saturation, (c) routing}" framing is itself wrong.** gp-44 already showed (a) and (b) are **rare transients, not chronic** (sink_mass flat 0.0077; bulk slot_delta~0.007 healthy; both collapse signatures <1% and self-clearing). ROUTE-A now removes (c). **All three are L1-side mechanisms — and the L1-only ablation proves L1 barely does long-range retrieval at all.** The root cause is at a different level than a/b/c.

3. **The real, cross-validated picture:** **L3 is the long-range主力; L1 has a near-zero long-range ceiling regardless of routing health; and the whole adapter monotonically over-trains (step500≫step5000 on BOTH retrieval and semantics).** Fixing L1 routing was always low-ceiling. The frontier is the **L3 summary channel** (which is unregularized, untelemetered, and read through a diluted shared softmax) plus the **overtraining** phenomenon — NOT L1 routing aux.

4. **Highest-ROI immediate experiments (both pure-hyperparameter, single-variable, runnable now on .196 + .249, 1000 steps):**
   - **EXP-1 (PRIMARY, confidence medium):** turn ON L3 summary-token diversity reg (`--l3_diversity_weight 0.1`, currently 0.0). The v9/v10 machinery built to fight L3 redundancy is **dormant in every canonical run**.
   - **EXP-2 (SECONDARY, confidence medium):** widen the L3 bottleneck (`--l3_n_summary 64→128`). Add capacity to the channel that is actually used (unlike num_slots on L1, which is證伪).
   - Both evaluated WITH eval-time cross-chunk SWA W2 (the single biggest proven long-range lever, +28/+24/+22 at 4k/8k/16k).

---

## 1. Negative result digestion — what ROUTE-A actually rules out

ROUTE-A底座 = P11 chunk512 delta-rule+normreadout, single routing knob each, 1000 steps. Result (RUN_REGISTRY §3d), long-range cell qa5 8k/16k/32k vs P11 base 48/45/44:

| arm | knob | qa5 8k/16k/32k (best ckpt) | verdict |
|-----|------|----------------------------|---------|
| arm1 | loss_free_update_rate 0.001→0.01 | 36/42/31 (step1000) | REJECTED (relatively harmless, no gain) |
| arm2 | entropy_aux 0→0.01 | 4/3/1 | REJECTED (crushes even 0k: 12 vs 74) |
| arm3 | selector_temperature 40→20 | 10/1/4 | REJECTED |
| arm4 | ST-Gumbel-topk (usage_cov→1.0) | **eval pending (gp-51, B200)** | prediction below |

**Interpretation.** The three completed arms span three independent dispersion mechanisms (selection-bias / gradient-entropy / softmax-sharpness, per gp-46). None improves long-range; entropy_aux actively destroys it. The standing hypothesis was that spreading routing onto all 128 slots would push the "saturation length" out ~4×. **It did not — because the binding constraint is not L1 slot count or L1 routing diversity.**

**arm4 prediction (confidence medium-high):** arm4 forces usage_cov 0.25→1.0 (all 128 slots used) but should still **fail the long-range cell**, because §3c L1-only shows L1's long-range ceiling is ~0 *with or without* full slot utilization. If arm4 instead improves long-range, my redirect is wrong and routing-diversity re-opens — so arm4 is a clean falsifier. Watch its qa5 8k-32k vs 48/45/44.

---

## 2. Re-judging gp-44's three mechanisms after ROUTE-A

gp-44 attributed the apparent "late-chunk collapse" to three candidate mechanisms. Post-ROUTE-A status:

- **(a) read-collapse via null-sink — still REFUTED, but note the gap.** gp-44 measured `XATTN_DIAG sink_mass` flat at 0.0077 across 1000 training steps → the **L1** cross-attention read is healthy. BUT this only covers the L1 `MemoryCrossAttentionRead` path. **The L3 read path (H→L3 inside the shared joint-attention softmax) is NOT instrumented anywhere** (gp-44 §5.3-5.5; grep of layer.py/selector.py finds no H→L3 mass field). So "read collapse" is refuted *for L1*, **unknown for L3**.

- **(b) write-saturation — not chronic, lowest-likelihood root.** Bulk `slot_delta≈0.007` healthy; the 4e-5 freezes are isolated sample-boundary transients at g_in=0.5 (open). No chronic EMA freeze. (b) is essentially eliminated as a chronic root cause.

- **(c) routing concentration — now證伪 as病根 by ROUTE-A.** Confirmed *present* (usage_cov~0.29, jaccard~0.91) but证伪 as *causal* for long-range — fixing it (arm1-3) does not help; maximizing usage (arm4) predicted not to help.

**Verdict on "a vs b":** if forced to choose between the two surviving candidates, **(a) read-side is the more likely locus than (b) write-side — but specifically the *L3* read dilution (unmeasured), NOT the L1 null-sink (measured healthy).** (b) write-saturation is effectively ruled out as chronic. **However, the honest answer is that the a/b/c trichotomy mislocates the problem** — see §3.

---

## 3. The cross-validated root cause: L3 is the主力, L1 has a low ceiling, and everything over-trains

Three independent registry results triangulate to a single conclusion that supersedes the a/b/c debate:

### 3.1 L1-only ablation (§3c) — the decisive cross-check
`--no_l3_summary`, single variable, canonical base:
```
                qa5: 0k  1k  2k  4k  8k  16k 32k
noL3 step500         75   4  13  10   2   5   2
P11 base step500     74  89  81  60  48  45  44
```
**L1 memory slots alone deliver qa5 1k=4 vs 89.** L1's entire ≥1k retrieval contribution is near-zero; **L3 carries essentially all long-range retrieval.** This is the single most important fact for re-judging the root cause:

> **All of mechanisms (a)/(b)/(c) live on the L1 path (selector routing, delta-rule write, L1 null-sink read). But L1's long-range ceiling is ~0. So no L1-side fix — routing, write-gate, or null-sink — could ever have moved the long-range cell much.** Routing concentration on L1 is a symptom of the LM learning to *ignore* L1 (it leans on L3 and lets L1 atrophy → few slots ever win → low usage_cov). This explains why ROUTE-A failed: it was tuning a channel that doesn't do the work.

### 3.2 L3 is the主力 but runs blind and unregularized
- All L3 regularizers are **0.0 in every canonical run**: `l3_diversity_weight=0.0`, `l_recon_weight=0.0`, `l3_recon_token_weight=0.0` (verified in `launch_mem_space_p11_chunk512_remote196.sh:41-42`). L3 = a plain, unregularized recurrent Q-Former.
- The v9/v10 diversity machinery (`l3_summary.py:96-105,156-194`, wired at `layer.py:1629-1641`) was built **specifically** to fight L3 summary-token redundancy (64 tokens collapsing to near-identical directions → effective L3 capacity ≪ 64). It is **dormant**. (Verified the S-space diversity term fires under `slot_query` mode — it acts on `l3_summaries` directly, independent of routing mode; only the q_multi term is multi_query-gated.)
- **No telemetry on L3 at all** in slot_query runs (the `summary_q_*_cos` fields are inert defaults — gp-44 §5.1). We cannot currently see whether L3's 64 tokens are diverse or collapsed, nor how much H attention mass lands on L3.

### 3.3 D6 xattn_off (§3c) — own-softmax read helps; L3 lacks one
D6 proved that L1's **dedicated own-softmax read** (`MemoryCrossAttentionRead`) is valuable (xattn_off degrades across the board). **L3 has no such dedicated read** — H reads L3 inside the *shared* joint-attention softmax, competing with (and diluted by) the growing live-token context (gp-44 §5.3). By symmetry with D6, **giving the主力 channel (L3) its own read path is the most mechanistically promising bigger bet** (EXP-3, needs code).

### 3.4 Overtraining is a separate, dominant axis (NOT the same as capacity)
step500≫step5000 holds on BOTH BABILong (qa5 step500 82/86/83/64/50 vs step5000 54/62/51/30/28) AND LongBench (AVG 8.87 vs 6.06, all 6 tasks worse incl. global-semantic ones — §3b). EVAL-2 refuted "capability migration"; it is **monotone pollution of the whole adapter**. This is a **distinct problem from capacity** — it is a schedule/optimization problem (the adapter keeps over-writing the backbone). Early-stop (step500) is the current mitigation. Note: L1-only inverts (step5000>step500) → the degradation is tied to the L1/L3 interaction over training, worth a dedicated probe later.

### 3.5 nullsink_off (D6 arm C, gp-52, eval pending)
Prediction (confidence medium-high, per gp-44): given sink_mass flat 0.0077, removing the null-sink should change the long-range cell **little**. If it changes a lot, the L1 read mechanism re-enters the picture. Use as falsifier, not fix.

---

## 4. Ranked next experiments

Validation口径 for ALL: same-protocol BABILong n=100, qa1/qa2/qa5 × 0k-32k, chunk512 bf16 sdpa, `scripts/run_babilong_mem_space.py` + `scripts/score_nested_babilong.py`. **Evaluate every arm WITH eval-time cross-chunk SWA W2** (`--swa_eval_chunks 2`) — it is the single biggest proven long-range lever (qa5 4k/8k/16k W2 vs W0 = 88/72/67 vs 60/48/45, §"step500×SWA"). Compare against P11 base step500 W0 = qa5 74/89/81/60/48/45/44 (and W2 = 79/89/86/88/72/67/49). Train `--total_steps 1000 --save_interval 500`;续训只在 step1000>step500 时。

### EXP-1 (PRIMARY) — turn ON L3 diversity regularization. confidence: medium. NO code change.
- **What:** P11 chunk512 base + `--l3_diversity_weight 0.1` (currently 0.0 in `launch_mem_space_p11_chunk512_remote196.sh:41`). Single variable. Node: **.196**.
- **Mechanism:** L3 = confirmed long-range主力 (§3.1) but runs fully unregularized (§3.2). Its 64 summary tokens plausibly collapse toward redundant directions — the exact failure the dormant v9/v10 loss was written for. Diversifying them raises *effective* L3 capacity → more distinct facts retrievable at long range. Pure hyperparameter, zero extra memory, leverages already-implemented-and-wired machinery.
- **Why first:** highest ROI/risk ratio — targets the channel that demonstrably does the work, costs nothing, and the 1000-step BABILong result *is* the redundancy test (if it helps, redundancy was real; if neutral, L3 was already diverse — cheap to learn).
- **Judge:** qa5/qa1 8k-32k (W2) vs P11 base. Risk: if L3 already diverse, reg is neutral-to-slightly-harmful → revert.

### EXP-2 (SECONDARY, parallel) — widen the L3 bottleneck. confidence: medium. NO code change.
- **What:** P11 chunk512 base + `--l3_n_summary 128` (currently 64). Single variable. Node: **.249**. (Shape change → trains from scratch, not warm-start; fine at 1000 steps.)
- **Mechanism:** L3 is the *used* channel (unlike L1, where adding num_slots is證伪 by usage_cov). If 64 summary tokens is the long-range bottleneck, doubling the bottleneck width directly increases storable facts. This is "add capacity where it is actually used" — the on-mechanism version of the capacity lever.
- **Judge:** same cell W2 vs P11 base. Note slightly higher L3 compute (64→128 keys in joint attn × 32 layers) — measure wall-clock.

### EXP-3 (BIGGER BET, needs coder) — give L3 its own dedicated read path. confidence: medium.
- **What:** route H→L3 read through a dedicated own-softmax cross-attention (mirror of `MemoryCrossAttentionRead`), instead of the shared diluted joint-attention softmax. Code change in `layer.py` (selector/read path), isolated behind a flag.
- **Mechanism:** D6 proved own-softmax read helps L1 (§3.3); L3 (the主力) currently lacks one and is diluted against growing local context. Highest mechanistic upside but requires implementation + a from-scratch train. Queue after EXP-1/2 read out.
- **Judge:** same cell; add an H→L3 attention-mass diagnostic at the same time (see EXP-D).

### EXP-D (DIAGNOSTIC, low cost, parallelizable, needs coder) — instrument L3.
- **What:** add to QUERY_DIAG: (i) H→L3 joint-attention mass fraction, (ii) L3 summary-token pairwise cosine (redundancy). Currently L3 is a black box (§3.2).
- **Why:** L3 is the主力 yet completely unobservable; this makes EXP-1/2/3 interpretable instead of blind. No training-value risk (no-grad diagnostics).

### DEPRIORITIZE
- **ROUTE-B (periodic reset of inactive L1 slots):** same L1 routing axis ROUTE-A just證伪, on the low-ceiling L1 channel. **Likely also ineffective** (confidence medium-high). Do NOT run before L3 experiments read out.
- **num_slots↑, capacity-length scaling sweep on L1, further routing-aux knobs:** all L1-side, all low-ceiling. Skip.
- **A new clean L3 ablation:** NOT needed — §3c L1-only already is the clean `--no_l3_summary` single-variable ablation gp-44 asked for; it answers "is L3 essential" = YES.

---

## 5. Confidence & open falsifiers
- Root-cause redirect (L3-is-主力 / L1-low-ceiling / overtraining-is-separate): **confidence high** (three independent registry results: L1-only, ROUTE-A, EVAL-2/§3b).
- "病根 is L3 redundancy / dilution specifically": **confidence medium** — L3 is untelemetered, so this is mechanism-grounded inference, not measured. EXP-1 + EXP-D resolve it.
- **Falsifiers in flight:** arm4 (usage_cov→1.0) improving long-range would re-open routing; nullsink_off changing the cell a lot would re-open L1 read. Both predicted negative.
- **No code changed, no training started.** EXP-1/EXP-2 are pure-hyperparameter and ready for coder+launch on .196/.249.
