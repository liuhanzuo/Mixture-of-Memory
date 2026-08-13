# A04 — PRE-REGISTRATION: 4-axis NI on the two never-tested control arms

**Committed BEFORE the first margin was computed.** Nothing below is written
after seeing a number. The arms' *accuracies* were unavoidably visible while
verifying protocol provenance (they sit in `summary.json` next to the `meta`
fields this document had to read), so this prereg states its predictions in
terms of **margins and orderings**, and §6 records exactly which raw numbers
were seen in advance and why that does not launder the predictions.

**Date:** 2026-08-13 · **GPU budget:** 0 planned (all four axes × both arms
already have 8/8-shard per-example predictions on zwfy6). Analysis is CPU-only.
**Nodes:** `.73` / `.82` only, and only if a scoring re-run turns out to be
needed. **Not touched:** `LOCAL` / `.21` (SparseForge #246), `.104` (paperC
Qwen3 heal).

---

## 1. The two arms, and why they are the right control

Neither has ever entered any A04 evidence file. Verified mechanically:

```
grep -rilE "freezefront|fromscratch|frozen_front|scratch16" \
     proposal/active/A04-recovery-certification/{evidence,code}/ *.md
→ 0 hits
```

| arm | construction | trainer flag | `arch_meta.json` | trainable |
|---|---|---|---|---|
| **train-all** (`keep14fresh2`) | front 14 transplanted + 2 fresh, **all 16 layers trained** | (default) | `healing_front14+fresh2` | 4060.4 M / 4060.4 M |
| **FF** (`keep14fresh2_freezefront`) | front 14 transplanted **and frozen**, only fresh tail + embed + norm + lm_head train | `--freeze_front` | `frozen_front14+fresh2` | **1226.9 M** / 4060.4 M |
| **FS** (`keep14fresh2_fromscratch`) | base weights **ignored**, all 16 layers random-init, all trained | `--from_scratch` | `scratch16L` | 4060.4 M / 4060.4 M |

`apply_freeze_front` (`scripts/train_olmo2_arch_probe2.py:397`) sets
`requires_grad_(False)` on exactly `model.layers.{0..13}.*` — read, not assumed.
So FF is **the same damage as train-all with strictly less repair capacity**, and
FS is **the same architecture with zero inheritance**.

---

## 2. The three pre-registered predictions

### P1 — FF ≤ train-all on every axis

> For each of the 3 decision axes, `margin_pp(FF) ≤ margin_pp(train-all)` at
> step 200 000.

**Rationale.** Identical injury (front 14 kept, 2 fresh), identical corpus,
identical step count, identical `eff_bs`; the *only* difference is that FF
cannot adapt the 2833.5 M inherited parameters. A strictly smaller trainable
set cannot recover more, unless "train-all is the stronger repair" is false.

**Judged on:** all 3 decision axes, `split` convention, canonical `ni_rule`.
**Verdict strings:** `P1_HOLDS` (3/3), `P1_HOLDS_WEAK` (2/3, and the violation
is inside item noise), `P1_VIOLATED` (≥1 axis where FF > train-all by more than
its own bootstrap SE).

### P2 — FS is the worst arm, and is a floor anchor

> FS has the lowest margin of the three arms on a majority (≥2/3) of decision
> axes, and rejects on 3/3.

**Rationale.** FS threw away a 100 352-vocab embedding and readout head that
200 k steps at `eff_bs=128 × 2048` (≈52 B token presentations) is not obviously
enough to relearn. It exists in `A04_GATE_DESIGN.md` §3.2 as arm **A4**, "the
'did inheritance matter at all' floor".

**Verdict strings:** `P2_HOLDS`, `P2_VIOLATED`.

### P3 — the falsifier (the reason this is worth running)

> **If P1 is violated — FF's margin is HIGHER than train-all's on ≥1 decision
> axis by more than that cell's bootstrap SE — then "train-all is the stronger
> repair" is false**, and A04's implicit ordering of repair strength is wrong.

Consequences that are **pre-committed**, so they cannot be renegotiated after
the fact:

1. A04 may no longer describe train-all as "the strongest available repair at
   fixed damage". The `A04_GATE_DESIGN.md` §3.2 arm ordering
   (A1 canonical > A2 keep-only > A3 random-trunk > A4 scratch) becomes an
   **untested assumption** about repair strength, not a design fact.
2. The `PROPOSAL.md` §8 statement *"whether any damage depth admits an accept at
   all is unknown"* must be widened: it is not only depth that is
   unexplored, it is **repair mode**. A rung is `(depth, repair mode)`, not
   `depth`.
3. It becomes a **positive, cheap, testable direction**: if freezing the
   inherited trunk protects the axes that train-all erodes, then the route to an
   accept may be *less* training of inherited weights, not more — the exact
   opposite of "heal longer", which `PROPOSAL.md` §4.1/§4.3 already falsified.

If P1 holds, that is also informative and is **not** to be reported as a null:
it confirms the arm ordering the gate design assumes but had never measured, and
it means P3's alternative route does not exist at this damage depth.

---

## 3. What this experiment CANNOT answer, stated now

- ⛔ **This is not a `σ_run` measurement.** Three arms, one seed each, no seed
  variance at 7B (`PROPOSAL.md §7.2`, `must_not_claim[23]`). Differences between
  arms are **not** tested against run-to-run variance because no such estimate
  exists or is reconstructible at 7B.
- ⛔ **This is not a rung of the keepN ladder.** All three are `keep_front=14,
  n_fresh=2, 16 layers`. They differ in **repair mode**, not depth. They may
  never be tabulated as a depth series.
- ⛔ **Q3's "current stopping rules accept what we reject" half cannot be
  answered on these arms**, and this is registered as a **design limitation
  before the fact**: `PLATEAU(T)` needs in-domain validation PPL on a frozen
  grid, and `olmo2_ppl_results/` contains **no** `freezefront` or
  `fromscratch` directory (checked: only `7B_keep14_step{0,128000,153500,200000}`
  and an unrelated `7B_scratch16L_lr2e5_*` LR-control run). So the ordering can
  be compared against `RATIO(0.85)` only. **`RATIO` is not the plateau rule**,
  and a `RATIO`-only comparison answers at most half of Q3.
- ⛔ **No claim that FF/FS "confirm" or "disconfirm" `safe_residual_claim`.**
  They can only populate the arm ordering that claim's arms table presumes.

---

## 4. Protocol口径 — asserted, not assumed

All to be re-verified mechanically by the analysis script before any margin is
emitted (fail-closed, no output file on deviation):

| requirement | how it is established |
|---|---|
| `cb_bs = 32` | `logs/cb_driver_104.out` echoes `START freezefront_step200k … bs=32` and `START fromscratch_step200k … bs=32`; `logs/nqopen_driver_104.log` and `logs/nqopen_scratch.log` echo `bs=32` for both `_nqopen` dirs |
| `mmlu_bs = 16` | `scripts/p06_run_104_transferred.sh` (FF) and `scripts/p06_run_transferred.sh` (FS) both leave `BS` unset → `_run_olmo2_mmlu_content.sh:43` `BS="${BS:-16}"` |
| `add_bos = False` | `summary.json:meta.add_bos`, asserted with `is False`, **never** `is not True` |
| `chat_template` | **structural**: neither `eval_olmo2_closedbook_qa.py` nor `eval_olmo2_mmlu_content.py` has a chat-template code path. Asserted `is not False`-safe by construction; recorded as `no_code_path` |
| same harness as the anchor and the endpoint | md5 `2ed41993241226c795a3ca38375933f7` (closedbook) / `fe4a62dbdf884a1e2aedc6ed26887b4e` (mmlu_content) — **identical** to the values pinned in `A04_KEEP14_TRAJECTORY_NI_VERDICT.md` §5.1 item 5 |
| shard integrity | shard index set **exactly {0..7}** (not "8 files"), merged `n` exactly `EXPECTED_N`, **0** duplicate `item_id`, **0** `nan`, item_id sequence identical to the anchor's |
| anchor | `ANCHOR` **imported** from `a04_shallow_rung_ni_7b` (`7B_base` / `base_full` / `base_full_nqopen`), never redeclared, never substituted (G0/G2) |
| nulls | `build_nulls` **imported** from `pilot_zero_rule_disagreement`. **No margin is ever obtained by subtracting a recorded null** — the error mode that produced four wrong numbers on 2026-08-13 |
| Δ | `PREREG["delta_fraction"] = 0.10` imported; `Δ` never substituted |
| endpoint reproduction | re-deriving `keep14fresh2@200k` under its archived offset (`arm_index = 201`) must return −28.4624 / −15.0810 / −7.4749 pp to < 5e-5 pp, else hard-fail |
| bootstrap seeds | `arm_index` **800 (FF@200k) / 801 (FS@200k) / 802 (FF@23500)**; disjoint from every archived block (0-1, 100-102, 200-203, 300-301, 400-408, 500-503, 700-702), checked mechanically |
| node + numpy | one node, recorded in-band. `.73` = numpy 2.5.1, `.82` = 2.4.6; same-seed `multinomial` drifts ≤ 0.005294 pp between them. **No margin quoted finer than 0.01 pp** (`must_not_claim[24]`) |

---

## 5. ⚠️ Provenance facts established BEFORE scoring, and their consequences

### 5.1 The zwfy6 training log for FF is a **different, abandoned run** — the wzc1 log is authoritative

This is the single most important thing found while verifying. `logs/olmo2_7B_keep14fresh2_freezefront.log` exists on **both** disks with **different
content**:

| | zwfy6 copy (162 067 B) | **wzc1 copy (1 368 257 B) ← authoritative** |
|---|---|---|
| first banner | 2026-07-21 02:02:20 | **2026-07-25 12:15:48** |
| geometry | `bs=4 gaccum=4` | **`bs=16 gaccum=1`** |
| `dataset rows=` | **15 491 607** | **7 570 911** |
| last step reached | **23 640** (dies) | **200 000 + `final.pt`** |
| last save | `step23500.pt` @ 07-23 13:45 | `step200000.pt` @ 07-28 21:24 |

So the FF run that reached 200 k is the **wzc1** run on the **7 570 911-row**
corpus at `bs=16 gaccum=1`; the zwfy6 log documents an **earlier, abandoned**
attempt on the **other** corpus at a different micro-batch geometry that died at
step 23 640.

**Binding consequences, registered before any number:**

1. ⛔ **`outputs/…_freezefront/step23500.pt` on zwfy6 is NOT a neighbour of
   `step200000.pt`, and is NOT on the same trajectory.** Its mtime
   (2026-07-23 13:45:20.774755372) matches the *abandoned* run's save line
   exactly, and its size (26 056 482 807 B, with optimizer state) belongs to that
   run. The wzc1 run *also* wrote a `step23500.pt` (07-25 22:40:21) but rotated
   it away. **`step23500` is therefore dropped from this analysis entirely** —
   not demoted to "far neighbour", **dropped**. Scoring it would compare two
   different corpora and two different micro-batch geometries. Bootstrap offset
   802 is reserved and left unused.
2. ⚠️ The **only** available neighbour statement for these arms is
   *"none exists"*, which is exactly what `A04_GATE_DESIGN.md` §2.0.2 permits
   ("or a statement that none exist"). It will be reported that way.
3. ⚠️ The two disks' same-named training logs are **not** copies. Any future
   agent reading a `logs/*.log` for these arms must state which disk.

### 5.2 The three arms ARE corpus- and budget-matched — verified, and this is unusual

From the **wzc1** (authoritative) logs, all three:

| arm | first banner | `dataset rows=` | geometry | `max_steps` | reached | resumes |
|---|---|---|---|---|---|---|
| train-all | 2026-07-16 21:36:20 | **7 570 911** | `bs=16 gaccum=1 eff_bs=128` | 200 000 | 200 000 | 4 |
| FF | 2026-07-25 12:15:48 | **7 570 911** | `bs=16 gaccum=1 eff_bs=128` | 200 000 | 200 000 | 1 |
| FS | 2026-07-21 02:00:06 | **7 570 911** | `bs=16 gaccum=1 eff_bs=128` | 200 000 | 200 000 | **0** |

Same corpus, same `eff_bs`, same `seq_len=2048`, same step count, same disk, same
`fp32` AdamW, same node class. **The two-corpora confound of
`STATUS.json:warning` does NOT apply to this comparison** — it applies across
the keepN *depth* ladder, and these three are one depth. That is precisely what
makes the repair-mode contrast cleaner than any depth contrast in the repo, and
it is registered here so it cannot be claimed as a post-hoc discovery.

### 5.3 The LR grouping differs across the arms — a real, registered confound

`_classify_param` (`scripts/train_olmo2_arch_probe2.py:420`) returns `"fresh"`
**first** when `from_scratch`. From the authoritative logs:

| arm | `[optim] group` lines | effective LR |
|---|---|---|
| train-all | `inh_decay 4060.1M @2e-5` + `inh_nodecay 0.3M @2e-5` | **uniform 2e-5** |
| FF | `inh_decay 1226.8M @2e-5` + `inh_nodecay 0.0M @2e-5` | **uniform 2e-5** |
| **FS** | **`fresh_decay 4060.1M @1e-4`** + `fresh_nodecay 0.3M @1e-4` | **uniform 1e-4** |

- train-all vs FF are **LR-matched at 2e-5**. The P1 contrast is therefore clean
  on LR: it isolates *which parameters move*, not *how fast*.
- **FS trained at 5× the LR** because `--from_scratch` routes everything to the
  `fresh` group. So P2's floor is **confounded by LR**, and this is registered
  **before** the number: if FS is worst, that is jointly attributable to "no
  inheritance" **and** "5× LR". FS is a **floor anchor with a caveat**, never a
  clean isolation of inheritance.
- ✅ Note the `fresh`-group bug of `PROPOSAL.md §7.2` / the distill trainer does
  **not** bite here: this trainer has the `module.`-strip fix, and FF's log
  proves it (only 1226.8 M in `inh_decay` = exactly the unfrozen set, so
  classification worked). But the *observed* consequence is that no arm ever ran
  differential LR — all three are uniform, at two different values.

### 5.4 `step200000.pt` on zwfy6 is a slimmed copy, not the training artefact

| arm | wzc1 `step200000.pt` | zwfy6 `step200000.pt` |
|---|---|---|
| train-all | 48 724 467 827 B (07-21 01:58) | 16 241 486 089 B (08-02 20:58) |
| FF | 26 056 479 363 B (07-28 21:25) | 16 241 487 014 B (08-02 21:12) |
| FS | 48 724 467 699 B (07-25 06:05) | 16 241 486 829 B (08-02 20:58) |

The zwfy6 files are **model-only** copies staged for eval (the launchers
`p06_run_transferred.sh` / `p06_run_104_transferred.sh` hard-assert those exact
byte counts, and their headers say "ckpts scp'd from LOCAL … slim model-only").
All three land at ~16.24 GB = 4 060 352 512 params × 4 B fp32 + zip overhead.
**The size ordering on wzc1 (FF 26 GB < the other two 48.7 GB) is an
optimizer-state artefact** — FF's optimizer covers only 1 226.9 M trainable
params — **not** a different architecture. Both zwfy6 copies load `179 tensors,
strict, num_hidden_layers=16` per the eval logs.

### 5.5 Pre-fix / post-fix sampler status

All three runs launched **2026-07-16 … 2026-07-25**, i.e. **before `ce5c298`
(2026-08-09 23:21:09)**. Per `PROPOSAL.md §7.2` they are all **pre-fix**:
`--seed` moved only fresh-tail init, data order was byte-identical. They are
therefore mutually **口径-consistent** (all three on the same side of the
break), and **none may enter any `σ_run` estimate**, nor be pooled with any
post-fix run. Recorded per arm in the output JSON as
`sampler_regime: "pre_ce5c298"`.

---

## 6. What was seen before this document was committed

Honesty requirement, since `summary.json` interleaves protocol `meta` with
scores. While establishing §4's口径 I saw the **raw accuracies** in the four
`summary.json` files and the `[merge]` lines of `cb_driver_104.out`:

- FF: popqa EM 0.09743, triviaqa EM 0.24766, nq_open EM 0.04958, mmlu
  content_norm 0.36042
- FS: popqa EM 0.04892, triviaqa EM 0.20859, nq_open EM 0.06316, mmlu
  content_norm 0.35978

I did **not** compute any residual, any Δ-relative margin, any lo95 bound, or
any NI verdict before committing this file. The predictions P1/P2/P3 are stated
over **margins**, which depend on the imported null, the imported anchor and a
one-sided bootstrap bound — none of which was evaluated.

⚠️ Because raw EM was visible, P1 and P2 are **weakly** blinded on the two
generative axes (train-all@200k reports triviaqa 29.403 % / popqa 7.976 %, and
the raw ordering FF < train-all on triviaqa is already visible there). This is
declared rather than hidden. **P3, the falsifier, is NOT weakened by this**: the
raw MMLU content_norm numbers (FF 0.36042, FS 0.35978, train-all 0.38321) do not
determine any NI verdict, and P1's per-axis SE-gated test is not decidable from
raw EM alone. Any reader may discount P1 accordingly; the analysis reports
per-axis SE so they can.

---

## 7. Analysis plan (frozen)

1. Import `ni_rule`, `ratio_rule`, `build_nulls`, `load_shards`,
   `mmlu_content_norm_vec`, `qa_metric_vec`, `EXPECTED_N`, `AXES`,
   `DEMOTED_AXES`, `PREREG` from `pilot_zero_rule_disagreement`;
   `paired_bootstrap`, `TIE_CONVS`, `N_BOOT`, `SEED` from A03's
   `analyze_1b_knowledge_floor`; `ANCHOR`, `_load_arm`, `assert_aligned`,
   `d4_interface_degenerate`, `D2_RESIDUAL_FLOOR_PP`, `Z95_TWO_SIDED` from
   `a04_shallow_rung_ni_7b`. **Reimplement nothing.**
2. Guard D1–D6 **before** NI (G1), all five `TIE_CONVS`.
3. Reproduce the archived train-all endpoint under offset 201 → hard-fail if
   > 5e-5 pp off.
4. NI + `RATIO(0.85)` for FF and FS on all 4 axes × 5 conventions.
5. P1 / P2 / P3 verdicts, per-axis, with each cell's `bootstrap_se_pp` so the
   reader can see flip distance.
6. Q2's floor question: FS's `residual_arm_pp` vs 0 (the null itself), with a
   paired bootstrap on `FS − null`, i.e. is FS **above its own best-constant
   floor** — and if so by how much of the intact residual.
7. Q3: the 3-arm margin ordering, plus the honest statement of §3's limitation
   that `PLATEAU(T)` is not computable for these arms.
8. Emit `evidence/a04_control_arms_ni.json`; write
   `A04_CONTROL_ARMS_NI_VERDICT.md`; append **one** new `STATUS.json` key,
   modifying **no** existing key (currently 42 — verified by
   `json.load` + `len()`, and each pre-existing key byte-compared after).

> **Note on the key count.** The dispatch said 41 keys. `json.load` on
> `STATUS.json` at commit `f79efd4` returns **42**. The count is verified
> mechanically, and the assertion enforced by the writer is "every pre-existing
> key is byte-identical and the new count is old+1", which is correct under
> either reading.

---

## 8. Cost

**0 GPU-h planned.** Every input is a per-example shard set already on zwfy6.
If any integrity assert fails, the affected axis is **reported as unavailable**
rather than re-scored, unless re-scoring is cheap and a node in the budget is
idle — in which case the re-run's node, numpy and wall time are recorded and
`gpu_h_spent` is updated to `wall × 8`.
