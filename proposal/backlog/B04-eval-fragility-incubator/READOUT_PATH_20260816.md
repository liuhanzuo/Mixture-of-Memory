# B04 clause 5 — the revision-3 READ-OUT PATH now exists and has been executed. 2026-08-16, 0 GPU

> **Headline.** φ can now be computed from disk. Before today it could not be: `phi_budget()` had
> exactly three call sites and all three were inside `selftest_phi()` on hand-written y-vectors, so
> **nothing the analyzer read from disk ever reached the decision statistic** — and the grid
> constants were still revision 2's, with `GRID_I` / `153500` / `S=100000` appearing nowhere in the
> file. Both gaps are closed.
>
> **Run today, against the real disk state, it returns `READOUT_ABSENT` and exits 3** — naming the 4
> missing `GRID_I` rungs and the 4 missing `GRID_W` rungs. That abort is the deliverable: the path
> has been executed, not merely written. Real output pasted verbatim in §3.
>
> **Both selftests pass.** The pre-existing arithmetic selftest is untouched and still passes; a new
> **21-check fixture selftest** drives the real disk-reading code end to end against a synthetic
> tree, so "selftest passes" now says something about the pipeline and not only about the arithmetic.
>
> **Scope.** This document IMPLEMENTS the pre-registration. It does **not** amend it. No threshold,
> no grid, no abort name was chosen here — every one is copied from `DECIDABILITY_FIX_20260816.md`
> §3. `lifecycle` stays `ready_cpu`; the verdict stays `READOUT_ABSENT`; revision 2's constants and
> code path are preserved unmodified for provenance. Four ambiguities/inconsistencies I found in the
> prereg are **reported in §6, not resolved.**

---

## 1. What was implemented, with file:line

All in `proposal/backlog/B04-eval-fragility-incubator/code/analyze_b04_wzc1_floor.py`
(1415 lines; was 546). Revision 2 occupies `:159-160` and `:655-...` **unchanged**.

### 1.1 The revision-3 constants — `:181-247`

| symbol | line | value |
|---|---|---|
| `COMPARATOR_INTERVAL_I` | `:186` | `(83500, 200000)`, `|I| = 116500` — **DESCRIPTIVE ONLY**, barred from multiplying any slope |
| `GRID_I` / `SPAN_I` | `:188-189` | `[100000, 128000, 153500, 175000, 200000]` / `100000` — **PRIMARY** |
| `GRID_W` / `SPAN_W` | `:190-191` | `list(G1_READOUT_STEPS)` / `175000` — **SECONDARY**, revision 2 verbatim |
| `SUP_RATIO_I` / `SUP_RATIO_W` | `:197-198` | `1.220390` / `1.173627` |
| `VERDICT_SEVERITY` | `:201` | `{"PASS":0, "NARROWED":1, "KILL":2}` |
| `READOUT_UNION_STEPS` | `:207` | `[25000, 50000, 100000, 128000, 153500, 175000, 200000]` |
| `SEARCH_ROOTS` | `:243` | wzc1 + zwfy6, both named |

`GRID_W` is **defined as** `list(G1_READOUT_STEPS)` rather than re-typed, and `:275-277` asserts
`GRID_W == G1_READOUT_STEPS and SPAN_W == READOUT_SPAN` at import. So revision 2's grid cannot drift
away from revision 3's secondary grid.

**Import-time sup-ratio guard, `:269-274`.** Each grid's sup ratio is recomputed from
`_sup_ratio()` (`:262`, the closed form `S · Σ_{w_i>0} w_i`) and compared to the pinned literal;
a mismatch > 5e-7 exits. A grid edit therefore cannot silently leave a stale sup ratio behind — the
exact failure mode that let revision 1 disable its own max-guard (sup 0.781300 < 1).

Independently recomputed this session, `python3` on wzc1:

```
GRID_I sup(S=100000) = 1.2203900013087292
GRID_W sup(S=175000) = 1.1736269780824236
GRID_W sup(S=116500) = 0.7813002454091563     <- revision 1: sup<1, max-guard dead
points of GRID_I inside I = 5/5 ; GRID_W = 3/5
S_I/|I| = 0.8583690987124464 ; uncovered = 14.163090128755366 %
PASS_MIN GRID_I = 0.103137 ; PASS_MIN GRID_W = 0.102923 ; KILL_MIN = 0.095408
```

All six figures match `DECIDABILITY_FIX_20260816.md` §4 Part A/B to the digits it printed.

### 1.2 The statistic and the combine rule — `:281-338`

- `phi_budget_grid(y, D, grid_name, steps, span, evaluated_steps=None)` at **`:281`** —
  `phi_G = max(max(y_G)-min(y_G), |OLS slope of y_G on heal_step| * S_G) / D`, verdict at
  `KILL >= 0.60` / `PASS <= 0.30`. Emits grid, span, both terms, binding term, `D`, the grid's own
  sup ratio, and `points_inside_comparator_I`.
- `combine_verdicts()` at **`:331`** — `max` over `VERDICT_SEVERITY`, i.e. the **MORE SEVERE** of
  `verdict_I` and `verdict_W`.

### 1.3 The read-out path — `:341-594`

| function | line | what it does |
|---|---|---|
| `readout_margins(dirpath)` | `:341` | pooled `|score(gold) − max(other)|` on **`norm_scores`** for one arm |
| `inspect_readout_dir(dirpath, want_step)` | `:391` | is this dir the named arm at the named step, and is it intact? |
| `find_readout_arms(steps, roots)` | `:461` | scan both disks; resolve step → arm |
| `clause5_revision3(D, sigma_hat, rho)` | `:512` | **THE GATE**: y from disk → φ per grid → combine |
| `print_clause5_revision3(rep)` | `:596` | MANDATORY REPORTING |
| `ABORT_EXIT_CODES` | `:650` | `READOUT_ABSENT 3`, `PROTOCOL_VIOLATION`/`FIELD_ASYMMETRY 4`, `DENOMINATOR_UNRESOLVED`/`FLOOR_UNMEASURABLE 5` |

**The margin definition is the analyzer's own, confirmed at the cited location.** The task asked me
to confirm `:172-200`; that was `margins()`, which the revision-3 insert pushed down to **`:699-729`**
(unmodified). The read-out's live copy of that arithmetic is `readout_margins():341-389`. The
operative lines:

```python
analyze_b04_wzc1_floor.py:372-376
            sc, g = o["norm_scores"], o["gold_letter"]
            oth = [v for k, v in sc.items() if k != g and v is not None]
            if sc.get(g) is None or not oth:
                continue
            out.append(abs(sc[g] - max(oth)))
```

identical to `margins():723` (`out.append(abs(sc[g] - max(oth)))`), and the median is
`statistics.median` as in `metrics_of():732`. **One deliberate difference, and it is a tightening not
a loosening:**
`readout_margins()` does **not** offer `margins()`' `norm_lens`-transplant fallback. Every read-out
arm is produced by the `a163a89` harness, which writes `norm_scores`/`norm_lens` natively, so a
missing field means the arm is not the protocol of record. The prereg has a `FIELD_ASYMMETRY` abort
for exactly this, and a silent transplant would defeat it. Documented at `:341-355`.

### 1.4 Where it is called from — the gap that closed

`phi_budget` call sites are **unchanged** (`:1355`, `:1378`, `:1383` in the new numbering; still all
three inside `selftest_phi()`). What is new is that `clause5_revision3()` — which *does* read disk —
is now called from `main()`:

```
:962   clause5_r3 = clause5_revision3(dam_range, sigma["median_margin"], ...)   <- in main()
:1022  print_clause5_revision3(clause5_r3)                                      <- in main()
```

and `phi_budget_grid()` is called from **`:585`, inside `clause5_revision3()`**, on a `y` assembled
from `census["resolved"][st]["median_margin"]` — values loaded from `per_example_*.jsonl` on disk.
That is the path that did not exist. Enumerated by script (each occurrence with its enclosing
`def`), not by eye.

### 1.5 New CLI surface

- `--readout-only` (`:1402`) — computes ONLY clause 5 revision 3, from the constants already banked
  in the evidence JSON. Does not rewrite the evidence JSON. **Exits non-zero on any abort.**
- `--selftest` (`:1394`) — now runs **both** selftests; both must pass.
- default (full run) — unchanged behaviour plus the revision-3 block, and now **exits 3** rather than
  0, because today's verdict is a hard abort and an abort must never look like success to a shell.

### 1.6 Provenance preserved, nothing overwritten

- Revision 2's `G1_READOUT_STEPS`, `READOUT_SPAN`, `SLOPE_TERM_SUP_RATIO`, `phi_budget()` and
  `selftest_phi()` are **byte-unchanged**.
- The evidence JSON gains **one new key**, `clause5_budget_discrimination_revision3`. Machine-checked
  against a pre-session copy:

```
added  : ['clause5_budget_discrimination_revision3']
removed: []
changed: (none)
```

  i.e. every revision-2 value is preserved byte-for-byte.
- `scripts/_run_b04_readout_evalfill.sh:490-500` — its closing note said the analyzer "does NOT yet
  read these dirs". That is now false, so it was replaced with the actual next command and its exit
  codes. `bash -n` clean; `--dry-run` still exits 0.

---

## 2. A real bug the fixture selftest caught immediately

Worth recording, because it is the entire argument for check 5 of the task.

The first draft of `find_readout_arms()` returned `census["resolved"]` **re-keyed to `str`** for JSON
friendliness, while `clause5_revision3()` indexes it with the grid's **`int`** steps. Every lookup
missed. The consequence was not a crash — it was a *plausible* wrong answer: the census correctly
reported `step200000 median_margin=0.108500`, and `missing_per_grid` simultaneously listed
`200000` as missing. Both grids reported 5/5 absent instead of 4/5.

Fixture check **F1** asserts the exact missing lists and failed on the first run. Fixed at `:504-508`
(int-keyed for lookups; the str-keyed copy is built separately at `:565-570` for the report only),
with the reason recorded in the comment so it cannot regress silently.

An arithmetic-only selftest could not have caught this: the arithmetic was always right.

---

## 3. THE DELIVERABLE — real output against today's actual disk state

```
$ python3 proposal/backlog/B04-eval-fragility-incubator/code/analyze_b04_wzc1_floor.py --readout-only
```

```
=== CLAUSE 5, REVISION 3 (two-grid, interval-matched) ===
  D (damaged median_margin range) = 0.021820   sigma_hat = 0.000541   6*sigma guard = 0.0032435
  Spearman(core6, heal_steps) = +0.6669 (wzc1 ladder; +0.8721 on zwfy6 -- naming the ladder is mandatory)
  comparator interval I = [83500, 200000] |I| = 116500 (DESCRIPTIVE ONLY -- BARRED from multiplying any slope (the revision-1 defect); used only to test grid support)
  GRID_I  PRIMARY   steps=[100000, 128000, 153500, 175000, 200000] span=100000 sup=1.220390 inside_I=5/5
  GRID_W  SECONDARY steps=[25000, 50000, 100000, 128000, 200000] span=175000 sup=1.173627 inside_I=3/5
  NOT VERIFIABLE FROM DISK (disclosed prereg gap): batch_size (prereg 8), max_len (prereg 1024, harness default), harness git commit (prereg a163a89), one-driver-invocation
    enforced instead by: scripts/_run_b04_readout_evalfill
  scanned wzc1   olmo2_downstream_results  (45 dirs)
  UNSEARCHABLE zwfy6  /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/olmo2_downstream_results  -> not mounted from this node -> absence on this disk is NOT established here
    step25000   ABSENT
    step50000   ABSENT
    step100000  ABSENT
    step128000  ABSENT
    step153500  ABSENT
    step175000  ABSENT
    step200000  median_margin=0.108500 n=17195 <- wzc1:keep14_s1234_step200000_sv181
  ==> GATE VERDICT = READOUT_ABSENT   [HARD ABORT, NON-PASS]
      why: a named arm of a grid lacks a margin-computable eval dir. phi is UNDEFINED -- not small, not large. NON-PASS: an undefined ratio cannot license 244-2560 GPU-h.
      GRID_I missing 4 of 5 arms: [100000, 128000, 153500, 175000]
      GRID_W missing 4 of 5 arms: [25000, 50000, 100000, 128000]
      phi is UNDEFINED -- not small, not large. No interpolation, no shortened grid, no NaN.
=== EXIT CODE: 3 ===
```

This is exactly the behaviour the task specified: a clean `READOUT_ABSENT` naming 4 missing `GRID_I`
rungs and 4 missing `GRID_W` rungs. `step200000` is found, identified by its **own**
`summary.json.meta.ckpt` = `outputs/olmo2_probe2_7B_keep14fresh2_seed1234/step200000.pt`, and its
`median_margin` **0.108500 at n = 17195** is recomputed from the six `per_example_*.jsonl` — it
reproduces the archived figure.

The full run (`analyze_b04_wzc1_floor.py`, no flags) prints the same block after the revision-2
block and also **exits 3**. Nothing was written to `olmo2_downstream_results/`
(`find olmo2_downstream_results -newermt '-70 minutes'` → empty).

> **The `UNSEARCHABLE zwfy6` line is a feature, not a defect.** `/apdcephfs_zwfy6` is not mounted on
> LOCAL (`ls` → `No such file or directory`; the only ceph mount is `/apdcephfs_wzc1/share_304376610`).
> Per `memory/two-disk-rule-applies-to-main-too.md`, the path reports `searched=False` with a reason
> rather than treating an unmounted root as an empty one. **Absence on zwfy6 is inherited from
> `DECIDABILITY_FIX_20260816.md` §4 Part D (165 dirs scanned from `.73`), not re-established here** —
> and it does not matter for the fill, which must land on sm_100/wzc1 anyway.

---

## 4. Both selftests, pasted

```
$ python3 proposal/backlog/B04-eval-fragility-incubator/code/analyze_b04_wzc1_floor.py --selftest
```

```
phi = max(range, |beta|*175000) / 0.021820;  KILL>=0.6  PASS<=0.3
  KILL   monotone compression        range=0.018300 slope_term=0.018035 bind=range phi=0.8387 -> KILL
  KILL   non-monotone V (rev-1 hole) range=0.020500 slope_term=0.006212 bind=range phi=0.9395 -> KILL
  NARROW mid excursion               range=0.008500 slope_term=0.008160 bind=range phi=0.3896 -> NARROWED
  PASS   early convergence           range=0.002300 slope_term=0.002168 bind=range phi=0.1054 -> PASS
  PASS   pure-noise-scale wobble     range=0.001080 slope_term=0.000154 bind=range phi=0.0495 -> PASS
[ok] all three verdicts are reachable -> the gate is falsifiable both ways
[ok] single-number boundaries shape-safe: KILL min<=0.095408 (phi=0.704176), PASS min>=0.102923; the range-only 0.101954 and the 6dp truncation 0.102922 are both correctly rejected as NARROWED

--- fixture selftest: the REAL disk-reading path, on a synthetic tree ---
  [ok] F1 one-arm-only        -> READOUT_ABSENT, GRID_I missing 4, GRID_W missing 4, no phi computed
  [ok] F2 full 7-arm ramp     -> phi_I=0.2979 PASS @S=100000, phi_W=0.3896 NARROWED @S=175000, combined=NARROWED (the SECONDARY grid carries it -- the combine rule is load-bearing)
  [ok] F3 steep early         -> phi_I=0.5729 NARROWED, phi_W=0.8478 KILL, combined=KILL
  [ok] F4 flat read-out       -> phi_I=0.0183, phi_W=0.0183, combined=PASS (all three verdicts reachable THROUGH DISK, not only in arithmetic)
  [ok] F5 5/8 shards          -> PROTOCOL_VIOLATION (arm_step153500 has 5/8 shard files -- refusing partial mer...)
  [ok] F6 n_shards=5 in summary -> PROTOCOL_VIOLATION (PARTIAL MERGE)
  [ok] F7 n_scored short 100  -> PROTOCOL_VIOLATION (arm_step128000/hellaswag n_scored=9942 != 10042...)
  [ok] F8 per_example truncated, summary clean -> PROTOCOL_VIOLATION (arm_step100000/per_example_hellaswag.jsonl has 9942 ...)
  [ok] F9 norm_scores stripped -> FIELD_ASYMMETRY
  [ok] F10 wrong ckpt, right name -> READOUT_ABSENT (GRID_I missing [153500]; the dir is NOT silently accepted)
  [ok] F11 ckpt_step != dir name -> READOUT_ABSENT (GRID_I missing [175000]; identification is by meta, not by name)
  [ok] F12 keep/fresh drift   -> PROTOCOL_VIOLATION (damage not held fixed)
  [ok] F13 add_bos=True       -> PROTOCOL_VIOLATION
  [ok] F14 n_nan=3            -> PROTOCOL_VIOLATION
  [ok] F15 know5-shaped decoy -> ignored, core6 dir used, combined=PASS
  [ok] F16 duplicate arms disagree -> PROTOCOL_VIOLATION (refuses to pick)
  [ok] F17 wrong base_model   -> PROTOCOL_VIOLATION
  [ok] F18 unmounted root     -> reported searched=False with a reason, not treated as empty
  [ok] F19 D<=0 / D<6sigma / sigma==0 -> DENOMINATOR_UNRESOLVED x2, FLOOR_UNMEASURABLE
  [ok] F20 k=4 / k=6 / None / NaN -> all abort; no shortened grid, no NaN flows onward
  [ok] F21 combine monotone   -> 0/20000 random shapes where combined was less severe than either grid alone
[ok] fixture selftest: 21/21 checks passed -- the DISK-READING path is exercised end to end, not only the arithmetic
```

`EXIT CODE: 0`. Runtime 10.4 s (the fixture writes ~17 195 rows × 6 tasks × 7 arms per scenario).

**Why the fixture is safe.** Every fixture dir is *freshly written* into a `tempfile.mkdtemp()` tree
and `shutil.rmtree`'d in a `finally`. `assert not tmp.is_symlink()` at `:1091`. **No fixture ever
points at `olmo2_downstream_results/`, and nothing is ever symlinked into a scratch area** — per
`memory/repo-checkers-are-writers-not-probes.md`, a checker in this project is a *writer*, so a
symlinked live evidence dir could be mutated in place. The one place the live dir is touched is the
§5 rehearsal, which passes it as a **read-only search root**.

F2 is the check that most earns its keep: it is the case where the two grids **disagree**
(`phi_I` PASS, `phi_W` NARROWED) and the combined verdict is NARROWED. If the combine rule were
"primary wins", the gate would return PASS there. The rule is load-bearing and now tested.

---

## 5. Firability precheck, re-stated — and what φ returns the moment the fill lands

### Can φ fire today? **No.** Honest answer, plainly:

```
$ python3 .../analyze_b04_wzc1_floor.py --readout-only ; echo $?
... ==> GATE VERDICT = READOUT_ABSENT   [HARD ABORT, NON-PASS]
3
```

**φ still cannot fire until the eval fill completes.** But the failure has changed *category*, and
that is the whole point of today's work:

| | before 2026-08-16 | now |
|---|---|---|
| grid constants | revision 2 only; `GRID_I`, `153500`, `S=100000` absent from the file | present, guarded at import |
| does any disk value reach φ? | **no** — 3/3 `phi_budget` call sites in `selftest_phi()` | **yes** — `:585` inside `clause5_revision3()` |
| behaviour with 6 holes | `phi_budget()` would raise before its own guards; no code path even attempted it | clean `READOUT_ABSENT`, 4+4 named, exit 3 |
| what "selftest passes" means | the arithmetic is falsifiable | the arithmetic **and** the disk pipeline |

So: **the fill is now the only remaining blocker.** It was necessary but not sufficient before; it is
now necessary and sufficient.

### What φ will return the moment the 6 evals land

**Mechanically** — verified by a rehearsal, 0 GPU. I combined the **real** `step200000` arm read
from the live wzc1 root (read-only) with 6 **placeholder** stand-ins in a throwaway scratch root, and
the gate fired:

```
  scanned wzc1_LIVE_readonly olmo2_downstream_results  (45 dirs)
  scanned scratch_standins /tmp/.../b04_firability_...  (6 dirs)
    step25000   median_margin=0.100000 n=17195 <- scratch_standins:keep14_s1234_step25000_b04fill
    ... (50000 0.101000, 100000 0.102000, 128000 0.103000, 153500 0.104000, 175000 0.105000)
    step200000  median_margin=0.108500 n=17195 <- wzc1_LIVE_readonly:keep14_s1234_step200000_sv181
  phi_GRID_I = 0.2979 -> PASS   [span=100000 binding=range range_term=0.006500 slope_term=0.006032 D=0.021820]
  phi_GRID_W = 0.3896 -> NARROWED [span=175000 binding=range range_term=0.008500 slope_term=0.008102 D=0.021820]
  COMBINE (FINAL = the MORE SEVERE of verdict_I and verdict_W (KILL > NARROWED > PASS))
  ==> GATE VERDICT = NARROWED  [NON-PASS -- blocks the 244-2560 GPU-h ladder]
REHEARSAL: gate FIRED. verdict=NARROWED
```

> **The six y-values above are PLACEHOLDERS, not predictions.** They exist to prove the code path
> fires and combines. The only measured number in that block is `step200000 = 0.108500`.

**What it will return is therefore determined by exactly one thing**: the six measured
`median_margin` values. The gate is fully pinned around them:

| | value | provenance |
|---|---|---|
| `D` | `0.021820` (`0.02181999999999995`) | `evidence/B04_wzc1_floor_analysis.json → clause5_budget_discrimination.damaged_range_median_margin` |
| `σ̂` | `0.000541` (`0.0005405884438142497`) | `per_metric_floor_analysis.median_margin.sigma_hat` |
| `6σ̂` guard | `0.0032435306628854983`; `D` = **6.73×** → admissible | recomputed this session |
| `y[200000]` | **`0.108500`, n = 17195** | recomputed from disk, `keep14_s1234_step200000_sv181` |
| KILL if | `max(range, |β|·S) ≥ 0.60·D = 0.013092` (24.2 σ̂) | |
| PASS if | `max(range, |β|·S) ≤ 0.30·D = 0.006546` (12.1 σ̂) | |
| **if max stays 0.108500**: KILL when | `min(y) ≤ 0.095408` (both grids) | |
| **if max stays 0.108500**: PASS when | `min(y) ≥ 0.103137` on `GRID_I`, `≥ 0.102923` on `GRID_W` | rounded UP per the 2026-08-15 shape-safety rule |

**Handoff to the fill driver is verified, not assumed.** `scripts/_run_b04_readout_evalfill.sh`
writes `keep14_s1234_step<STEP>_b04fill`. I built exactly those six dir names in scratch and
confirmed all seven arms resolve. Identification is by `summary.json.meta.ckpt` + `.ckpt_step`, never
by name, so **`SUFFIX` is free**: rename the dirs and the read-out still finds them, while a dir with
a *right name and wrong checkpoint* is rejected (F10/F11).

**If only 4 rungs land** (`STEPS="100000 128000 153500 175000"`, `GRID_I` only): the combined verdict
is still `READOUT_ABSENT`, because `GRID_W` needs 25000/50000. `EVAL_FILL_READY_20260816.md` §5.3 is
right that `GRID_I` alone becomes *computable*, but the prereg's combine rule needs both, and a
partial run is **not** a verdict. Run all 6.

---

## 6. Prereg ambiguities and inconsistencies — **reported, not resolved**

Per instruction. I did not silently fix any of these.

### 6.1 The prereg names fields that `summary.json` does not record — **material**

`DECIDABILITY_FIX_20260816.md` §3's READ-OUT block fixes `batch_size 8`, `--max_len` 1024 (default),
"harness `scripts/eval_olmo2_probe2_downstream.py`", "one driver invocation"; the field table in
`EVAL_FILL_READY_20260816.md` §1 adds the harness commit `a163a89` as *"the one confound with
measured teeth"*. **None of these is written into `summary.json`.** Measured on the archived rung:

```
top keys : ['output_name', 'n_shards', 'add_bos', 'meta', 'tasks']
meta keys: ['mode','keep_front_layers','n_fresh_layers','num_hidden_layers',
            'ckpt_step','ckpt','base_model','add_bos']
  batch_size present=False   max_len present=False   harness/commit/git present=False
```

So a read-out arm's **own artifacts cannot prove it is same-harness.** The gate's
`PROTOCOL_VIOLATION` abort lists shard count and `n_scored` — which *are* on disk — but the prereg's
own most-load-bearing invariant is unverifiable from the artifact.

**Not resolved.** I did not weaken the prereg, and I did not invent a check it does not authorise.
The path **discloses** it: `UNVERIFIABLE_FROM_DISK` at `:228-241`, printed in every report
(see §3) and emitted in the JSON. Enforcement stays where it actually lives —
`scripts/_run_b04_readout_evalfill.sh`, which pins the flags and refuses to run against an
uncommitted harness (`note_fail "harness has UNCOMMITTED changes"`). **Consequence for the next
agent: a rung produced by any route other than that driver is not provably same-harness.** The clean
fix — have the harness stamp `batch_size`/`max_len`/commit into `summary.json` — is a *prereg
amendment* and a *harness change* (itself a driver boundary), so it is not mine to make.

### 6.2 `DECIDABILITY_FIX` §4 Part A prints `GRID_I`'s sup ratio as `1.220390`; the exact value is `1.2203900013087292`

Immaterial (10th significant figure) but it means the pinned literal cannot be compared with `==`.
The import guard uses a `5e-7` tolerance. Flagged because the 2026-08-15 shape-safety fix turned on
precisely this kind of rounding: two lenses proposed `0.102922` vs `0.102923` for the same threshold
and the truncated one **failed**. Direction of rounding matters for a `min(y) >= T` rule; it does not
for a sup ratio used only as an audit figure. **Not resolved** — I kept the prereg's printed value as
the literal and tolerated the tail rather than silently re-pinning it.

### 6.3 The prereg does not say what to do if **two** margin-computable dirs exist for one step

`READOUT_ABSENT` covers "no dir". `PROTOCOL_VIOLATION` covers a wrong *step set*. Neither covers
"the same step has two eval dirs". This is not hypothetical: `step200000` already has **two**
`seed1234` dirs on wzc1 — `keep14_s1234_step200000_sv181` (core6) and `..._know` (know5). Today they
are disambiguated cleanly because the `_know` dir's task set is not core6 (fixture F15 pins this).
But two *core6* dirs for one step — e.g. re-running the fill with a different `SUFFIX`, or
`INCLUDE_200K=1` — is **reachable and unspecified**.

**Not resolved by choosing.** `find_readout_arms():493-502` **refuses**: identical `median_margin`
(to 12 dp) is accepted as the same measurement; any disagreement raises `PROTOCOL_VIOLATION` naming
both dirs. Fixture F16 pins it. Picking one silently would be exactly the "quietly fixing a
pre-registration" failure the task warns about. **The next agent should note that
`INCLUDE_200K=1` will create a second core6 `step200000` dir** — harmless if it reproduces
`0.108500` (which is the point of that flag), a hard abort if it does not, which is the correct
outcome because it would mean the archived point is *not* same-harness.

### 6.4 `READOUT_ABSENT`'s "on EITHER disk" is not checkable from one node

The abort text says a rung counts absent only if it lacks an eval dir *"on EITHER disk"*, but
`/apdcephfs_zwfy6` is not mounted on LOCAL and `/apdcephfs_wzc1` does not exist on `.82`. **No single
node can evaluate that clause.** Today it is moot in the strong direction — wzc1 has 6 of 7 absent
and zwfy6 was found (08-16, from `.73`) to reference no `seed1234` checkpoint at all — but the clause
as written is not locally decidable. **Not resolved:** the path reports the zwfy6 root as
`searched: False` with the reason "absence on this disk is NOT established here" and never treats it
as empty. A reviewer who needs the clause discharged must re-run the zwfy6 scan from `.73/.82/.104`.

### 6.5 Not a defect, recorded for completeness

- `SHARD_SAMPLES_ARE_NOT_A_READOUT` needs no code: the read-out path only ever opens
  `per_example_{task}.jsonl`, never `shard*of8.json`'s `samples[]`. The shard files are counted, not
  parsed for scores. The abort is satisfied by construction.
- The `--max_steps`-style extension risk is enforced in code: `phi_budget_grid()`'s
  `evaluated_steps != steps` → `PROTOCOL_VIOLATION` (F20 pins k=4 and k=6).

---

## 7. Order of operations for the next agent — **fill first**

**Fill first, then read out.** The read-out is 0 GPU, ~1 s, idempotent, and re-runnable at will; the
fill needs 8 idle sm_100 cards. There is nothing the read-out can learn from an empty disk that it
has not already reported (§3), and nothing about the fill depends on the read-out. So the read-out is
the *verification* step, not a prerequisite. It was worth building **before** the fill only because
it establishes that the fill is *sufficient* — which, before today, it was not.

```bash
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory

# 0. (0 GPU, ~4 s) validate. Warns if the cards are busy; a real launch exits 5 instead.
bash scripts/_run_b04_readout_evalfill.sh --dry-run

# 1. (8 GPUs, ONE sm_100 node = LOCAL or .21, ~36 min wall / ~4.83 occupancy GPU-h)
#    NEVER .73/.82/.104: sigma_hat and the whole comparator ladder are sm_100-only, so an
#    H20 arm confounds run-to-run with hardware. PREFETCH=1 moves ~1.7 GPU-h of cold
#    CephFS read off the GPU clock without changing the wall time.
setsid nohup bash scripts/_run_b04_readout_evalfill.sh \
  > logs/b04_evalfill_$(date +%Y%m%d_%H%M%S).out 2>&1 &

# 2. (0 GPU, ~1 s) THE GATE. Exit 0 = a computed verdict; 3 = READOUT_ABSENT,
#    4 = PROTOCOL_VIOLATION / FIELD_ASYMMETRY, 5 = DENOMINATOR_UNRESOLVED / FLOOR_UNMEASURABLE.
python3 proposal/backlog/B04-eval-fragility-incubator/code/analyze_b04_wzc1_floor.py --readout-only
echo "exit=$?"          # <- CHECK THIS. Do not read the verdict off stdout alone.

# 3. (0 GPU) bank it into the evidence JSON (adds clause5_budget_discrimination_revision3;
#    revision 2's keys are never touched). Also exits 3 while the read-out is absent.
python3 proposal/backlog/B04-eval-fragility-incubator/code/analyze_b04_wzc1_floor.py

# regression, any time (0 GPU, ~10 s): both selftests must pass
python3 proposal/backlog/B04-eval-fragility-incubator/code/analyze_b04_wzc1_floor.py --selftest
```

**Run all 6 rungs, not 4.** `GRID_I`-only is computable but the prereg's verdict is the *combine* of
both grids, so a 4-rung fill still yields `READOUT_ABSENT` (§5).

**Do not promote on the strength of this document.** `lifecycle` stays `ready_cpu`; the verdict stays
`READOUT_ABSENT`; promotion is the next independent adversarial pass's call. What changed is only
that the gate is now *capable of firing* — which is a precondition for a verdict, not a verdict.

---

## 8. Provenance

Every number here is either reproduced this session on wzc1 (0 GPU) or quoted with its file:line.
Repo HEAD at time of writing: `a8cd0a5`. Harness: `a163a89` (2026-08-08), clean vs git.

- `proposal/backlog/B04-eval-fragility-incubator/code/analyze_b04_wzc1_floor.py` — all line refs in
  §1; `:159-160` + `:655-...` are revision 2, unchanged; `:181-650` is revision 3;
  `:1030-1335` is `selftest_readout_fixture()`; `:1337-...` is `selftest_phi()`, unchanged.
- `proposal/backlog/B04-eval-fragility-incubator/DECIDABILITY_FIX_20260816.md` §3 (verbatim gate),
  §4 Part A/B/D (spans, constants, census) — the authority this code implements.
- `proposal/backlog/B04-eval-fragility-incubator/EVAL_FILL_READY_20260816.md` §1 (protocol of
  record + field table), §4 (4.83 occupancy GPU-h), §5.2 (the gap this document closes).
- `proposal/backlog/B04-eval-fragility-incubator/evidence/B04_wzc1_floor_analysis.json` — `D`, `σ̂`,
  `ρ`; **only new key added**, no existing key changed (machine-diffed against a pre-session copy).
- `olmo2_downstream_results/keep14_s1234_step200000_sv181/` — `summary.json.meta` read; 45 dirs
  enumerated; `median_margin 0.108500` at `n = 17195` recomputed from the 6 `per_example_*.jsonl`.
- `olmo2_downstream_results/keep14_s1234_step200000_sv181_know/` — the second `seed1234` dir;
  task set `['boolq','commonsense_qa','lambada_openai','mmlu','social_iqa']`, hence not a core6
  candidate (§6.3).
- `scripts/eval_olmo2_probe2_downstream.py:555-557,725-726` — what the harness writes into
  `summary.json`; the basis for §6.1.
- `scripts/_run_b04_readout_evalfill.sh:490-500` — closing note corrected; `bash -n` clean,
  `--dry-run` exit 0 re-verified after the edit.

**GPU used: none.** Nothing written to `olmo2_downstream_results/` (verified by `find -newermt`).
Fixtures were freshly-written throwaway `mkdtemp` trees, never symlinks, removed in a `finally`.
zwfy6 was not touched at all: it is not mounted on this node, which the report states rather than
assuming.
