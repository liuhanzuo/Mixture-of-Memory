# A04 — the missing in-domain PPL at step 150 000: the PLATEAU grid asymmetry is closed

**Date**: 2026-08-12. **GPU cost, measured not extrapolated**: 8×H20 on `.73`, driver wall
14:53:17 → 14:54:50 = **93 s total** for two configs (regression re-measure of step 100 000: 43 s;
new step 150 000: 44 s; two merges + assertions: 6 s) = **0.207 GPU-h**. Per-shard scoring times
were 18.94–19.17 s across all 8 shards, i.e. the fan-out is balanced and the 93 s is a measured
end-to-end wall, not the first shard scaled up. Analysis is CPU-only, seconds.

This closes `A04_STEP100K_PLATEAU_VS_NI_VERDICT.md` §8 item 1 verbatim: *"Closing the 150k side
would need an in-domain PPL run at step 150 000 (cheap, ~8 shards), which is not done here."*

| item | status before | status now |
|---|---|---|
| in-domain PPL at step 150 000 | **DOES NOT EXIST** → PLATEAU UNDEFINED there | **MEASURED: 15.607276788333472** → PLATEAU **ACCEPTS** (`rate_5k = 0.22602 %/5k`) |
| checkpoints where **both** rules have a verdict | `{100 000, 200 000}` — 2 of 4 | **`{100 000, 150 000, 200 000}` — 3 of 4** |
| the capability/PPL grid asymmetry | capability at 4 steps, PLATEAU evaluable at 3 | **both at 4 steps for every step that has a preceding interval** |
| earliest disagreement | step 100 000 | **step 100 000 — UNCHANGED** |

---

## 1. Scope: this adds one evaluable checkpoint. It changes nothing else.

Stated first because the temptation runs the other way. **One PPL point is one cell, not a trend.**
Every standing A04 finding is unchanged, and the new cell reinforces the one that kills the rung:

* `keep7+fresh2` at 1B remains a **CONSTANT-REJECT rung**. NI rejects **4/4 axes at all four
  checkpoints** (50k/100k/150k/200k), **6.64–9.88× Δ**, recovery **4.78–39.00 %** of intact
  residual, **zero NI accepts in 16 cells**. The step-150 000 NI cells were already measured and
  are **bit-identical** here (§4); nothing about capability moved.
* **K1 stays INDETERMINATE.** Its ≥24-cell precondition is *not* helped in the way that matters:
  there are still **12 NI decision cells**, not 24. What did change is a *different* count — the
  decision cells at which PLATEAU is *also* defined, so a disagreement could even be formed, rose
  **6 → 9**. That is not K1's denominator and must not be quoted as if it were.
* **K2 does not fire; still necessary-not-sufficient.** Untouched.
* **keep12 remains a constant-reject rung.** Untouched.
* The **earliest disagreement is still step 100 000.** The new checkpoint is *later* than the
  existing earliest accept, so it cannot move that claim earlier, and it did not.

**Net effect: exactly one thing changes — step 150 000 goes from "PLATEAU UNDEFINED, cannot form a
disagreement" to "PLATEAU ACCEPTS, and it disagrees with NI".** No verdict, no kill condition and
no gate status moves.

---

## 2. Why step 150 000 and not 147 000 — the cheaper pairing is *impossible*, not merely worse

There is already a PPL point at **147 000**, only 3 000 steps from 150 000, and pairing capability
to *it* would have cost **zero GPU** (PPL already on disk). I checked that first. It cannot be done:

**`step147000.pt` no longer exists on either disk.**

| where I looked | result |
|---|---|
| `zwfy6:outputs/olmo2_probe2_1B_keep7fresh2_16card/` | holds only `step{50000,100000,150000,200000}.pt` + `final.pt` |
| `find /apdcephfs_zwfy6/share_304376610/pighzliu_code -maxdepth 4 -name 'step147000*'` | **no hits** |
| `find /apdcephfs_wzc1/... -maxdepth 3 -type d -name '*keep7fresh2*'` (LOCAL wzc1) | **no such directory at all** — the arm is zwfy6-resident |
| `/apdcephfs_wzc1` as seen from `.73` | `lrwxrwxrwx -> /apdcephfs_zwfy6` — a symlink, so a wzc1-looking path there is the *same* physical file set, not a second copy |

The 147 000 PPL number is a **fossil**: `olmo2_ppl_results/1B_keep7_step147000/` is dated
2026-07-19, when the ckpt still existed; the ckpt was pruned later. So capability at 147 000 is
permanently unobtainable short of retraining, while `step150000.pt` (12 181 310 078 B) is on disk.
**Step 150 000 is therefore not merely the better-matched target — it is the only target that can
ever close this side of the bracket.** Both disks were checked because this repo's standing rule is
that "the file does not exist" is not established until both wzc1 and zwfy6 have been searched.

The driver also emits a loud note if `step147000.pt` ever reappears, so the rationale cannot be
silently outrun by the filesystem.

---

## 3. Recovered protocol — established from two independent sources, then matched

The new point had to be measured **the same way** as the archived four, or PLATEAU's rate
computation mixes protocols and the exercise is pointless. The settings were recovered from the
launcher **and** cross-confirmed against what the archived summaries themselves recorded:

| parameter | value | how established |
|---|---|---|
| harness | `scripts/eval_olmo2_probe2_ppl.py` | the existing one; **no new evaluator written** |
| launcher of record | `scripts/_run_olmo2_probe2_ppl_8gpu.sh` (git `89d5f15`, 2026-07-19) | it is the only script naming `1B_keep7_step{50000,100000,147000}` |
| `--val_path` | `data/dolmino_now_val.npy`, shape **(4096, 2048)** uint32 | launcher line 8 **and** `val_path` in all four archived `summary.json` |
| `--batch_size` | **4** | launcher line 9 (`BS=4`) |
| `--num_shards` / `--shard_index` | **8** / 0..7, one process per GPU, `windows[g::8]` | launcher lines 30-36 |
| `--limit` | **not passed** (default 0 = no cap) | absent from the launcher; confirmed by the token count below |
| `--base_model` | `../models/OLMo-2-0425-1B` → `/apdcephfs_zwfy6/.../models/OLMo-2-0425-1B` | launcher lines 17-19; **identical to the ckpt's own recorded `base_model_path`** |
| `keep_front_layers` / `n_fresh_layers` | **7 / 2** → `num_hidden_layers = 9` | passed explicitly; `load_pruned_model` reads them from ckpt meta and **raises if the CLI disagrees**, so passing them is a free assertion, not a guess |
| dtype | fp32 master weights, **bf16 autocast** forward, fp32 `reduction='sum'` CE | fixed in the harness, not a CLI knob |
| merge | `--merge`, token-weighted `exp(Σnll/Σtok)` | never a mean of per-shard ppl |

**Measurement-basis invariants**, satisfied by all four archived points and asserted by the driver
for the new one (it exits non-zero otherwise): `n_shards = 8`, `n_tokens = 8 384 512`,
`n_windows = 4 096`. 4096 windows × 2047 predicted positions = 8 384 512 — i.e. the *whole* val
set, which is what rules out a silent `--limit`. This assertion exists because a partial merge is a
known failure mode in this repo (`d380bbc`: a 5/8 merge once produced a plausible PPL over 2560
windows that was not comparable to the 8-shard points it was plotted against).

**Harness identity.** `eval_olmo2_probe2_ppl.py` is byte-identical across wzc1 and zwfy6
(md5 `12b2dede720410c861eee78fc91e012a`). Three commits touched it after the archived run:

| commit | change | on this driver's live path? |
|---|---|---|
| `d380bbc` | `merge_shards` refuses a silent partial-shard merge | **no** — guard only; on a complete 8/8 set the arithmetic is unchanged |
| `36ddb1e` | adds `load_base_model_any_family` (new function, non-OLMo) | **no** |
| `7ac9653` | adds `load_truncated_any_family` (new function, non-OLMo) | **no** |

None modifies `load_pruned_model` or `score_windows`. That distinction is load-bearing: the repo's
standing rule is that same-arch/same-harness re-runs are **byte-identical**, so a code delta on the
live path would have made the new point non-comparable rather than merely noisy.

---

## 4. Regression check — the known point reproduces *byte-for-byte*

Step **100 000** — an already-published point — was re-measured under the recovered protocol
**first**, into a separate dir `A04_regress_1B_keep7_step100000` so the archived dir was never
touched:

| quantity | archived (2026-07-19) | re-measured (2026-08-12) | agreement |
|---|---|---|---|
| `ppl` | 16.161295049729876 | **16.161295049729876** | **0.000e+00** |
| `sum_nll` | 23330903.982421875 | **23330903.982421875** | **exactly equal** |
| `avg_nll` | 2.7826191890979315 | 2.7826191890979315 | exact |
| `n_tokens` / `n_windows` / `n_shards` | 8384512 / 4096 / 8 | 8384512 / 4096 / 8 | exact |

Not merely "reproduces to 1e-9" — the **`sum_nll` float is bit-identical across 26 days and a
different launcher**, which is the strongest available evidence that the protocol was recovered
correctly rather than approximately. So the step-150 000 number is on the same footing as the
trajectory it joins.

**The capability side reproduces too.** Re-running the committed analysis on the unchanged 4-point
trajectory (control) reproduced the committed verdict exactly, and in the 5-point run **every** NI
quantity across all 12 decision cells — `diff_mean_pp`, `diff_lower95_one_sided_pp`, `delta_pp`,
`residual_fraction_recovered` — is identical to the committed values at **max |Δ| = 0.000e+00 pp,
with 0 `ni_accept` flips**. The script's own regression guard on the archived step-200000 cells also
passed at 0.000e+00 pp in both runs. **The new PPL point moves the PLATEAU side only; it cannot and
did not touch NI.**

---

## 5. The new point, and the two things it forced

**`ppl(step 150 000) = 15.607276788333472`** (`sum_nll = 23038436.08203125`, `avg_nll =
2.747737266287084`, 8/8 shards, 4096 windows, 8 384 512 tokens). The trajectory is monotone
decreasing at all five points.

Inserting it is **not** a matter of appending to the ppl json, and both obstacles are structural:

1. **The committed analysis hard-fails on an unknown step.** `a04_step100k_plateau_vs_ni.py`
   cross-checks *every* step's `rate_5k` against `a04_plateau_rule_repair.json` at tol 1e-9 and
   exits with `FATAL: step 150000 absent from repair JSON`. Fixed the only legitimate way: by
   **re-running the committed `a04_plateau_rule_repair.py`** on the 5-point trajectory to produce
   `a04_plateau_rule_repair_5pt.json`. The rule was **not** reimplemented anywhere.
2. **Adding a point *changes* step-200 000's `rate_5k`** — and this is a real property of the rule,
   not a bug. R3 measures the rate over the *preceding interval*, so 200 000's predecessor moves
   from 147 000 (d = 53 000) to 150 000 (d = 50 000), and its rate goes
   **0.13172918300312642 → 0.12606867323571302**. The verdict is unchanged (both ≪ T = 2.0), but the
   published number does move, which is why the repair JSON had to be regenerated rather than
   patched.

### 5.1 R3 on the 5-point grid

| step | ppl | d | `rate_5k` (%/5k) | R3 (T = 2.0) | change vs published |
|---:|---:|---:|---:|:--|:--|
| 50 000 | 17.619441896079884 | — | — | **UNDEFINED** (first point, no preceding interval) | unchanged |
| 100 000 | 16.161295049729876 | 50 000 | 0.8601172082 | ACCEPT | unchanged |
| 147 000 | 15.628480830626273 | 47 000 | 0.3560056973 | ACCEPT | unchanged |
| **150 000** | **15.607276788333472** | **3 000** | **0.2260237949** | **ACCEPT** | **NEW — was UNDEFINED** |
| 200 000 | 15.411630407090653 | 50 000 | **0.1260686732** | ACCEPT | rate changed from 0.1317291830 (predecessor moved); verdict same |

`rate_5k` is still **monotone decreasing** across the four defined points, so R3's accept region is
still an interval and the earliest accept is still 100 000 — the script's own invariant
(`FATAL` if R3's earliest accept ≠ 100 000) passed.

### 5.2 Side effect I must report: **R1's** earliest accept moves 200 000 → 150 000

R1 is the pre-registered *unscaled* reading, which compares an interval-length-dependent quantity to
a per-5k threshold. The new interval is short (d = 3 000), so `rel_improve = 0.1357 % < T = 2.0` and
**R1 accepts at step 150 000** — earlier than before, **with no change whatsoever to the underlying
run**. This is precisely R1's grid-dependence defect (the one R3 was built to repair) manifesting as
a relocation of its own first accept. It is a *demonstration* of the documented defect, not new
evidence about the model.

Consequently a **hardcoded prose string in the committed script was now factually false**: it
asserted "R1's earliest accept is step 200 000". I changed it to **derive** the value from the
trajectory (`min(r1_accepts)`) plus an explicit grid-dependence caveat. Verified both ways: on the
4-point trajectory it still emits **200000**; on the 5-point it emits **150000**. The 4-point
control output is otherwise unchanged.

---

## 6. Recomputed PLATEAU-vs-NI table

Pre-registered `split` convention; `T = 2.0 %/5k`, `Δ = 0.10 · residual(intact)`, one-sided lower
95 % bound of the paired item bootstrap. **Bold** = newly evaluable.

| step | R3 | `rate_5k` (%/5k) | NI reject / decision axes | rules disagree? |
|---:|:--|---:|:--|:--|
| 50 000 | UNDEFINED (first trajectory point) | — | 3/3 | not evaluable |
| 100 000 | ACCEPT | 0.86012 | 3/3 | **YES** ← still the earliest |
| 147 000 | ACCEPT | 0.35601 | *no capability scored* (ckpt deleted, §2) | — |
| **150 000** | **ACCEPT** | **0.22602** | **3/3** | **YES — was "not evaluable"** |
| 200 000 | ACCEPT | 0.12607 | 3/3 | YES |

Per-axis at step 150 000 (all values identical to the already-committed cells):

| axis | n | reported | residual (pp) | frac of intact residual | lower-95 % of diff (pp) | Δ (pp) | NI |
|---|---:|---:|---:|---:|---:|---:|:--|
| TriviaQA EM | 17 944 | 0.0931788 | 9.0615 | 22.41 % | −31.9773 | 4.0431 | **REJECT** (7.91× Δ) |
| PopQA EM | 14 267 | 0.0396019 | 1.6682 | 12.63 % | −12.0137 | 1.3205 | **REJECT** (9.10× Δ) |
| MMLU-content | 14 042 | 0.3226748 | 3.8225 | 37.33 % | −6.9791 | 1.0239 | **REJECT** (6.82× Δ) |

**Convention-invariant**: the step-150 000 disagreement holds under all five null conventions
(`split`, `first`, `last`, `credit`, `wrong`) — 3/3 NI rejects and `rules_disagree = True` in every
one. Integrity re-asserted per cell: 8/8 shards, shard indices exactly 0..7, no duplicate
`item_id`, exact item counts (14 042 / 17 944 / 14 267 / 3 610), **zero** `nan` rows, and `item_id`
sequences identical across arms so the paired difference is genuinely item-paired.

**What may now be said**: "At step 150 000 the repaired rule PLATEAU accepts (`rate_5k =
0.22602 %/5k`) while NI rejects on 3/3 decision axes by 6.8–9.1× Δ under all five null conventions
— a third checkpoint at which the two rules disagree."

**What must NOT be said**: that the earliest disagreement moved (it did not — still 100 000); that
K1's ≥24-cell clause is closer to satisfied (12 decision cells, unchanged); that anything is known
about the *shape* of the PPL trajectory from one added point; or that R3's accept at 150 000 makes
`keep7+fresh2` a usable rung (it is still constant-reject, §1).

---

## 7. Reproduce

```bash
# 1. GPU (8×H20, zwfy6 node; 93 s measured). Config 1 is the regression
#    re-measure of step 100 000; config 2 is the new point. Asserts the
#    measurement basis and exits non-zero on any mismatch.
PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
setsid nohup bash proposal/active/A04-recovery-certification/code/a04_ppl_step150k_driver.sh \
  > logs/a04_ppl_step150k.out 2>&1 &

# 2. CPU: regenerate the repair JSON on the 5-point trajectory with the
#    COMMITTED repair script (the rule is never reimplemented).
cd proposal/active/A04-recovery-certification/code
python a04_plateau_rule_repair.py \
  --ppl_json ../evidence/a04_1b_keep7f2_ppl_trajectory_5pt.json \
  --out_json ../evidence/a04_plateau_rule_repair_5pt.json

# 3. CPU: the PLATEAU-vs-NI analysis. Must run with code/ as cwd (it imports
#    pilot_zero_rule_disagreement as a module). Hard-fails unless the archived
#    step-200000 cells reproduce to 1e-9 pp.
python a04_step100k_plateau_vs_ni.py \
  --raw_root        /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
  --ppl_json        ../evidence/a04_1b_keep7f2_ppl_trajectory_5pt.json \
  --repair_json     ../evidence/a04_plateau_rule_repair_5pt.json \
  --pilot_zero_json ../evidence/pilot_zero_rule_disagreement.json \
  --out_json        ../evidence/a04_step150k_ppl_closes_plateau_grid.json
```

Machine-readable: `evidence/a04_step150k_ppl_closes_plateau_grid.json`.
Raw PPL shards: `zwfy6:olmo2_ppl_results/1B_keep7_step150000/` (new) and
`zwfy6:olmo2_ppl_results/A04_regress_1B_keep7_step100000/` (regression).
Driver log: `zwfy6:logs/a04_ppl_step150k_progress.log`.

## 8. What this document does NOT establish

1. **Anything about the shape of the PPL trajectory.** One point is one cell. The 5-point
   trajectory happens to be monotone, but nothing here is evidence about *how* it decays.
2. **A PLATEAU verdict at step 50 000.** It is the trajectory's first point and has no preceding
   interval; no measurement can fix that, only an earlier checkpoint could.
3. **Capability at step 147 000.** Permanently unobtainable — the ckpt is deleted from both disks
   (§2). The 147 000 row of the table will stay empty.
4. **K1's ≥24-cell clause or K2.** Both unchanged. 12 decision cells, still short of 24.
5. **That a rung exists where NI can accept.** Still unmeasured at 1B. This pass makes the keep7
   rung's unsuitability more certain, not less.
6. **Causality of the D5 48-item drift.** Untouched; still needs a same-code control.
