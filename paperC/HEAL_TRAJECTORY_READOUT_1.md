# Heal-trajectory read-out #1 — MMLU-Pro on the first five milestones

**Date** 2026-08-13, 01:40-02:08 +08:00. **Cost 5.57 GPU-h** of a 120 GPU-h budget (4.6%).
Nodes `.73` + `.82` (8×H20 each, zwfy6). Training on `.104` was **not touched**.

This discharges `HEAL_CONFOUND_PREREGISTRATION.md` §10, which reserved 16 cards for
"the offline MMLU-Pro scoring of milestones (which is 8-GPU sharded and **is the
actual next bottleneck**)". That reservation had been idle since the 08-12 launch.

> ⚠️ **The pre-registered read-out is step 121000 and this is NOT it.** Every healed
> cell below is a mid-run milestone at step ≤ 7000, i.e. **4-6%** of the read-out
> budget. Nothing here is the P1/P2 verdict, and §8's outcome labels are deliberately
> **not** applied. Per the pre-registration's own instruction — "Do not re-choose the
> read-out step after seeing intermediate numbers" — these numbers must not be quoted
> as "the healed Qwen3 result".

---

## 1. RAN (this session, first-hand)

| what | where | result |
|---|---|---|
| `nvidia-smi` self-check before launching | `.73`, `.82` | both 8×`0 MiB`, `0 %`, **0** compute apps |
| Pin 4 milestones by hardlink | `.73` (zwfy6) | 4 links, `df` unchanged (0 extra bytes) |
| MMLU-Pro score `step5000`, `step6000` | `.73`, 8 shards | 409 s, 408 s — `ALL ARMS DONE`, 2/2 MERGE OK |
| MMLU-Pro score `step5500`, `step6500` | `.82`, 8 shards | 434 s, 437 s — `ALL ARMS DONE`, 2/2 MERGE OK |
| MMLU-Pro score `step7000` | `.73` (refilled at 01:55) | 410 s, MERGE OK |
| MMLU-Pro score OLMo-2 `keep8@step45000` | `.82` (refilled at 02:01) | 407 s, MERGE OK |
| Trajectory nulls + integrity gate | `.73` CPU | 10/10 cells pass |
| Degeneracy decomposition | `.73` CPU | see §4 |
| Re-measure training rate from log timestamps | `.104` (read-only) | **5.847 s/step** over 6240 steps |

**Not run, deliberately:** the step-121000 read-out (the arm is at step ~7240 of it),
and the P1/P2 verdicts that depend on it.

## 2. READ (pre-existing, not produced here)

| source | what was taken from it |
|---|---|
| `HEAL_CONFOUND_PREREGISTRATION.md` §8 | floor `always-A 0.116606`; `MAXLEN=2048`, `add_bos 0`, `desc_style none`, `chat_template=False`; read-out step 121000 |
| `HEAL_CONFOUND_PREREGISTRATION.md` §10 | the 16-card reservation this session spends; the 8-vs-16-card measurement (1.10×) |
| `HEAL_CONFOUND_LAUNCH_RECORD.md` | launch config, the 5.716 s/step launch figure, the family-dispatch closure |
| `mmlu_pro_lc_crossfamily_results_fix/qwen3_8b_base_k8/` | the **P1 control**, −0.881 pp, p=0.0362, BELOW floor (archived 08-12) |
| `mmlu_pro_letter_content_results/7B_keep8_step121000/` | the **P2 comparator**, −0.116 pp, p=0.7118, AT floor (archived) |
| `paperC/code/mmlu_pro_power_nulls.py` | `best_constant_letter`, `paired_boot`, `two_sided_boot_p`, `load_records` — **imported, not copied** |

Both archived cells were **re-derived here through my own code path and reproduce
exactly** (−0.881/p=0.0362 and −0.116/p=0.7118), which is the check that my
analysis layer is not shifting numbers.

---

## 3. The trajectory

`n = 12032` per cell, floor `always-A = 0.116606` asserted bit-identical in all 10 cells.
`hw` = CI95 half-width; all cells are **powered** against MMLU's own 1.389 pp effect.

| cell | heal steps | letter acc | Δ vs floor | hw | boot p | verdict |
|---|---:|---:|---:|---:|---:|---|
| qwen3 k8+fresh2 | 5000 | 0.115276 | −0.133 | 0.241 | 0.2834 | AT floor |
| qwen3 k8+fresh2 | 5500 | 0.115691 | −0.091 | 0.353 | 0.6076 | AT floor |
| qwen3 k8+fresh2 | 6000 | 0.114860 | −0.175 | 0.303 | 0.2638 | AT floor |
| qwen3 k8+fresh2 | 6500 | 0.114943 | −0.166 | 0.312 | 0.2836 | AT floor |
| qwen3 k8+fresh2 | 7000 | 0.115775 | −0.083 | 0.253 | 0.5206 | AT floor |
| **olmo2 keep8** (new) | 45000 | 0.117271 | +0.066 | 0.445 | 0.7526 | AT floor |
| **olmo2 keep8** (P2, archived) | 121000 | 0.115442 | −0.116 | 0.582 | 0.7118 | AT floor |
| **qwen3 k8 UN-healed** (P1, archived) | 0 | 0.107796 | **−0.881** | 0.819 | **0.0362** | **BELOW floor** |
| qwen3 intact (ref) | — | 0.461104 | +34.450 | 0.968 | 0.0001 | ABOVE floor |
| olmo2 intact (ref) | — | 0.271858 | +15.525 | 0.906 | 0.0001 | ABOVE floor |

**Shape: flat.** Across 5000→7000 the healed Qwen3 moves within [−0.175, −0.083] pp of
floor, a span of 0.09 pp — smaller than every cell's own CI half-width. It is at floor
from the first scored milestone and stays there. It is **already off** the un-healed
twin's −0.881 pp, so whatever separates them happened before step 5000.

---

## 4. The finding that actually matters: "AT floor" and "BELOW floor" are the same phenomenon

All five healed cells are at floor. So is OLMo-2 at 45k and 121k. The un-healed twin
is below it. **This is fully explained without any competence term.**

`always-<L>` accuracy is a **dataset property**, and on MMLU-Pro it is not flat:

```
A .1166   B .1124   D .1110   C .1092   G .0982   E .0955   F .0938   H .0927   I .0921   J .0785
```

Spread A→J = **3.81 pp**; `always-E` is **−2.11 pp** below `always-A`. A model that emits
one letter almost always scores that letter's marginal. So the **identity of the collapse
letter**, not the presence of competence, decides the verdict:

- healed Qwen3 collapses onto **A** (82-91% of items) → **A is the argmax, i.e. the floor by
  construction** → "AT floor";
- un-healed Qwen3 collapses onto **E** (94.5%, and emits only **5** distinct letters) →
  `always-E` = 0.0955 → "significantly BELOW floor".

Fitting each cell with an **independence model** that has no competence term at all,
`acc_hat = Σ_L P(pred=L)·P(gold=L)`:

| cell | modal pred | share | distinct letters | acc_hat | **residual** |
|---|---|---:|---:|---:|---:|
| qwen3 heal@5000 | A | 0.912 | 10 | 0.114617 | **+0.066 pp** |
| qwen3 heal@5500 | A | 0.819 | 10 | 0.112311 | **+0.338 pp** |
| qwen3 heal@6000 | A | 0.856 | 10 | 0.113312 | **+0.155 pp** |
| qwen3 heal@6500 | A | 0.862 | 10 | 0.113345 | **+0.160 pp** |
| qwen3 heal@7000 | A | 0.905 | 10 | 0.114431 | **+0.134 pp** |
| olmo2 heal@45000 | A | 0.767 | 8 | 0.115068 | +0.220 pp |
| olmo2 heal@121000 | A | 0.575 | 10 | 0.113263 | +0.218 pp |
| qwen3 k8 UN-healed | E | 0.945 | 5 | 0.096476 | +1.132 pp |
| qwen3 intact | A | 0.263 | 10 | 0.106609 | **+35.449 pp** |
| olmo2 intact | A | 0.301 | 10 | 0.108997 | **+16.286 pp** |

Every damaged cell — healed or not, either family — is explained to within **+0.07 to
+1.13 pp** by "degenerate emitter × dataset marginal". The intact models are **not**
(+35.4, +16.3 pp). The independence model is a sufficient description of every damaged
cell and a hopeless one for undamaged models, which is exactly the signature of
degeneracy rather than partial competence.

### Consequence for the pre-registered outcome labels

§8 committed: **H_heal supported** if the healed arm "moves UP to AT-floor" while its
un-healed twin sits below. On the letter interface that criterion is **satisfiable by
changing which letter the model degenerates onto**, and it would be satisfied by a
model with no MMLU-Pro competence whatsoever. The current milestones satisfy it
(−0.083…−0.175 pp, all p > 0.26) while emitting `A` for 82-91% of 12032 items.

So the H_heal / H_family dichotomy, **as operationalised on letter accuracy vs floor**,
is not yet identifying what it was meant to identify. This is a defect in the read-out
statistic, not in the arm, and it is better found now at step 7000 than on day 8.

**Recommended amendment, to be decided BEFORE step 121000 is scored** (so it is still
pre-hoc): report `modal_pred_share` and the independence residual **alongside** every
letter-vs-floor cell, and treat "AT floor with modal share > ~0.6 and residual < ~0.5 pp"
as **degenerate-at-floor**, explicitly distinguished from **competent-at-floor**. Both
already exist in the evidence JSONs, so this costs no GPU. Note this does not rescue
`content_norm` either — it is *below* letter in all five healed cells (0.075-0.078).

---

## 5. Design decisions, and what each bought / gave up

**Which milestones.** All four that existed, plus step7000 as it appeared: 5000, 5500,
6000, 6500, 7000. Not a subsample — at 0.93 GPU-h per arm the whole set cost less than
the 120 GPU-h budget's 5%, so cadence was not the binding constraint. **Bought:** 5
points, enough to establish a shape (flat) rather than a single reading, and the
5500/6500 points are precisely the ones rotation would have destroyed. **Gave up:**
nothing at this budget. Adding OLMo-2 `keep8@45000` was the highest-value extra: it
tests whether the degeneracy decays with heal budget within the family that has the
long run, and it does not (modal share still 0.767 at 45k, 0.575 at 121k).

**Two nodes.** Two **independent 8-card shards** on different milestones, not one 16-card
job. Scoring is embarrassingly parallel across arms (one ckpt per arm, 8 shards inside
an arm, no cross-arm communication), so two nodes on different milestones is a **linear
2×**, whereas §10's own measurement puts 16-card DDP at **1.10×** on this trainer with an
added TCPStore failure mode. Milestones were **interleaved** across nodes (`.73` = 5000/6000,
`.82` = 5500/6500) so losing a node costs alternating points, not a contiguous half of the
trajectory. Both nodes were refilled the moment they drained (`.73` at 01:55, `.82` at 02:01),
so neither idled.

**Was it even the bottleneck?** Partly — and the answer is more interesting than yes/no.

- Scoring is **not** the compute bottleneck: 0.93 GPU-h per milestone against a
  multi-day training. §10's "actual next bottleneck" is not true in GPU-hours.
- But it *was* a real bottleneck in the sense that mattered: it is the only thing that
  could have surfaced the §4 read-out defect, and **it was time-critical for a reason
  nobody had noticed** (§6). Deferring it to day 8 would have cost 3 of the 5 milestones
  permanently.
- **Is scoring at step ~7000 informative?** For the P1/P2 verdicts, **no** — and this
  document does not report them. For the read-out statistic's validity, **yes, decisively.**

---

## 6. A live data-loss race, found and closed

The live output dir is under rotation with `keep_last_n=3 milestone_every=5000
keep_milestones=8`. Only multiples of **5000** are retained long-term; **5500/6000/6500
are transient** and are deleted a few saves later.

**Observed, not predicted:** `step5500.pt` was present at 01:34 and **gone from the live
dir by 01:46**, deleted when `step7000` was written. It survived only because it had been
hardlinked at 01:34.

Before scoring, 5000/5500/6000/6500 were hardlinked into
`outputs/paperC_qwen3base_heal_k8f2_pinned/` — same inodes, `df` unchanged at 3.4 TB,
**0 extra bytes** — and the rotator cannot touch them because it only globs its own
`output_dir`. `step7000` was pinned the same way when it appeared.

**Rule for whoever scores the next milestones: pin first, then score.** Reading a
non-multiple-of-5000 milestone straight out of the live dir races the rotator, and the
`--ckpt` path can vanish between enumeration and load.

---

## 7. Integrity assertions (all hard — they raise, they do not warn)

Shown as executed, for all 10 cells:

```
INTEGRITY OK for all 10 cells: shard set {0..7}, n==12032, 0 dup, 0 nan, 0 trunc,
chat_template is False
floor: always-A 0.116606 (asserted bit-identical in all 10 cells)
estimators: imported from a01_gate3_fp32_vs_bf16.py; local verbatim copies asserted identical
```

1. **Shard index set == {0..7}**, not a count of 8 — a duplicated shard 3 with shard 5
   missing also counts to 8. Driver-side `merge_task` independently refuses a partial merge.
2. **`n_scored == 12032` exactly**, 0 duplicate `item_id`, `n_nan == 0`.
3. **`n_trunc == 0`**, re-derived from the per-shard json rather than trusted from the log.
   `MAXLEN=2048` is the task-#252 fix and is present in the zwfy6 driver — verified by
   diffing both disks' copies: the only differences are four `paperG`→`paperC` comment
   renames (the md5s differ, which is why the diff was necessary rather than the hash).
4. **`chat_template is False`** — asserted with `is False`. Negative-tested: `None` **passes**
   `is not True` and is correctly **rejected** by `is False`.
   Also asserted `add_bos is False`, `desc_style == "none"`.
5. **Floor bit-identical** across all cells (a pure dataset property; drift ⇒ different item
   sets ⇒ cells not comparable). Gold marginal separately asserted identical in §4's script.
6. **Architecture matches the label**: each cell's `keep_front_layers`/`n_fresh_layers` is
   asserted against what its arm name claims (`8`/`2`), so a mislabelled dir cannot enter
   the table.
7. **Estimators imported**, not copied, from `mmlu_pro_power_nulls.py`, which itself asserts
   bit-identity against A01's.
8. **The gate demonstrably fires**: the first run aborted on
   `7B_keep8_step45000: shard index set [] != [0..7]` because that job was still in flight.
   It was not relaxed — a "0 shards" dir is now skipped as *not yet analysable*, while a
   1-7 shard dir still reaches the assertion and raises.

**Not asserted / caveats.** `qwen3_8b_base_k8` is eval-time truncation with `n_fresh=0`,
so its `keep/fresh` check is `8`/`0`, not `8`/`2` — it is the un-healed control, and that
architectural difference (no fresh layers to train) is part of what "un-healed" means here.
Per the launch record, bf16 numerics in this harness are **batch-size dependent**; every
cell here is `BS=48`, the archive's own value.

## 8. Corpus caveat (unchanged, still applies)

Pre-registration §6/§9.2 stand: the Qwen3 arm heals on **5.541B SlimPajama tokens
(5.72 epochs over 121k steps)** against OLMo-2's **1.0 epoch of 31.7B Dolmino**. Nothing in
this read-out narrows that asymmetry.

## 9. Training status (read-only; `.104` untouched)

`pid 3343471`, step **7240/200000** at 02:09, `maxmem 77.5 GB`, loss ~3.07 / ppl ~21.
Measured **elapsed/iter = 5.847 s/step** over steps 1000→7240 (6240 steps, 36483 s,
313 log points; per-interval median 5.750, max 8.300 at checkpoint flushes). The launch
record's 5.716 s/step is the flush-free rate; **5.847 is the realised rate** and the one to
plan with — the difference is checkpoint-write overhead, which is exactly why the
instantaneous `s/it` field must not be quoted.

Remaining from step 7240: **184.8 h ≈ 7.70 d** to step 121000 (≈ **2026-08-20**),
313.1 h ≈ 13.0 d to 200000. No §13 kill condition is met.

## 10. Next actions

1. **Decide the §4 amendment before step 121000 is scored**, so it stays pre-hoc. Zero GPU.
2. **Pin then score** later milestones (10000, 15000, …) to extend the trajectory; ~0.93 GPU-h
   each. Multiples of 5000 are rotation-safe, but pinning is free insurance.
3. At **step 121000**, run the pre-registered P1/P2 read-out, reporting modal share and
   independence residual next to every cell.
4. Consider scoring an intact-but-`n_fresh`-matched control to separate "damage" from
   "2 randomly-initialised layers on top of 8 inherited ones".

## Provenance

| artefact | path |
|---|---|
| trajectory + integrity | `paperC/evidence/heal_trajectory_mmlu_pro.json` (md5 `2b947e21…`) |
| degeneracy decomposition | `paperC/evidence/heal_degeneracy_decomposition.json` (md5 `139f54a6…`) |
| trajectory read-out code | `paperC/code/heal_trajectory_nulls.py` |
| decomposition code | `paperC/code/heal_degeneracy_decomposition.py` |
| per-example records (10 cells) | zwfy6 `mmlu_pro_lc_paperC_heal_results/`, `mmlu_pro_lc_crossfamily_results_fix/`, `mmlu_pro_letter_content_results/` |
| pinned checkpoints | zwfy6 `outputs/paperC_qwen3base_heal_k8f2_pinned/` (hardlinks) |
| driver logs | zwfy6 `logs/paperC_heal_traj_node73.out`, `…node73b.out`, `…node82.out`, `logs/paperC_heal_olmo45k_node82.out` |

Evidence JSONs are on **both** disks and md5-verified identical. The per-example records
and pinned checkpoints are **zwfy6-only** (they are 38 GB each; `.73`/`.82`/`.104` share
zwfy6, so the `.104` checkpoints were read directly with **no transfer** — verified by
`ls` and by loading them from `.73`/`.82`, which is what made this job cost 5.57 GPU-h).
