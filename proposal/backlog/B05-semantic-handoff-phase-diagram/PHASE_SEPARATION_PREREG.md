# B05 — PHASE SEPARATION PREREG (PRE-DATA)

**Written 2026-08-14. 0 GPU spent. No B05 cell has been run.**
This file exists so that the separation criterion is fixed **before** any B05 cell is
inspected. `proposal/README.md` rule 1: 新方向先写 PROPOSAL.md 和 kill gate，再启动 GPU.

The thing this file fixes is the one thing `STATUS.json` admitted was unfixed:
its `exit_clause.unspecified` field says «'不清晰' is not operationalised». Below it is.

---

## 0. What is already measured, and what B05 is actually adding

A02 already ran **one row** of B05's grid: it varied **split depth j** at a **fixed
readout** (distilled LoRA), on the **retrieval-closed** RULER cells where recall@12 = 99–100 %
so retrieval cannot confound depth.

| A02 arm | j | readout | RULER mean over the 4 retrieval-closed cells |
|---|---|---|---|
| A0 | 0 | native suffix (no adapter) | **99.75** ← anchor |
| A1 | 0 | LoRA r32, 87.29 M | 99.75 (0 flips / 400 vs A0) |
| A2 | 6 | LoRA r32, 72.74 M | 99.25 |
| A3 | 9 | LoRA r32, 65.47 M | 98.25 |
| A4 | 12 | LoRA r32, 58.20 M | 90.75 |
| A5 | 18 | LoRA r32, 43.65 M | 20.75 |
| A6 | 12 | LoRA r40, 72.74 M (cap-matched to A2) | 90.50 |

Source: `proposal/backlog/A02-comem-write-read-repair/evidence/read_tax_ruler/a02_read_tax_ruler.json`
(md5 `fcb32b6dddb90cc5ca64925b5740462b`), all fail-closed gates PASS, n=100/cell, 8/8 shards.

**The column A02 did NOT run is the native-suffix readout at j > 0.** That column is
precisely B05's `readout capacity` axis. At j = 0 the axis is measured and is
**degenerate by construction** — GATE 0 of A02's prereg establishes that at `resume_j=0`
teacher == student, so the "optimal j=0 adapter" is the identity; A1 vs A0 is 0 flips
out of 400 paired items. **j = 0 therefore carries no information about whether readout
capacity is a real axis, and is excluded from every phase-non-emptiness test below.**

### Free prior evidence, and why it is NOT reusable as a paired cell

`ruler_results/qcmem_8b_zeroshot_j9_chatFALSE/` on wzc1 holds an adapter-free
(`lora_adapter: null`, `zero_training_no_adapter: true`) j=9 run at
`selector=iter_bm25 / topk=12 / hop=4 / chunk=512 / sink=bos / chat_template=false / seed=42`
— i.e. the canonical protocol. Merging its 8/8 shards under the canonical
`recall >= 1.0` rule (`scripts/eval_ruler_qcmem.py:720`) gives:

| cell | native j=9 (this dir) | A02 LoRA j=9 (A3) |
|---|---|---|
| niah_multikey_1 16k | 44.00 | 99.0 |
| niah_multikey_1 32k | 59.00 | 95.0 |
| variable_tracking 16k | 18.00 | 99.0 |
| variable_tracking 32k | 16.00 | 100.0 |
| **mean** | **34.25** | **98.25** |

**This is recorded as MOTIVATION, not as a B05 cell**, for two fail-closed reasons:

1. **It emits no `records.json`.** `ls ruler_results/qcmem_8b_zeroshot_j9_chatFALSE/*.records.json`
   → no such file (the dir predates the per-item/`input_ids_sha256` emission at
   `scripts/eval_ruler_qcmem.py:710-754`, mtime 2026-07-24 vs script 2026-08-04). Without
   `input_ids_sha256` there is no GATE C2 pairing, so a paired delta against A02's arms
   would be an **assumed** pairing, not a constructed one. A02's own verdict records that
   its first analyzer version would have *silently passed* on a `None` config read; the
   same class of defect is not being re-admitted here.
2. It does not record `max_new_tokens`, so the `variable_tracking` 60-token override
   (`eval_ruler_qcmem.py:602-603`) cannot be confirmed identical to A02's arms.

**Consequence, stated pre-data:** the native column must be **re-run** with the current
harness. The 34.25-vs-98.25 numbers above are declared here so that nobody can later
claim the gate was designed after seeing them — they are visible, and §3 explains exactly
why they do **not** make the gate un-failable.

---

## 1. The grid B05 will actually run (single variable: readout capacity)

Fixed at the A02 values, byte-for-byte, so that pairing is by construction:

```
model            ../models/Qwen--Qwen3-8b   (L = 36; identical string to the A02 anchors)
selector         iter_bm25    topk 12    iter_hop_topk 4    sink_tokens bos
chunk_size       512          limit 100  num_shards 8       seed 42
chat_template    False        enable_thinking False         bf16 + sdpa + greedy
PRIMARY cells    niah_multikey_1 x {16k, 32k},  variable_tracking x {16k, 32k}
```

Four new arms, **no training, no adapter**:

| B05 arm | flags | paired against (A02, on disk) |
|---|---|---|
| `N6`  | `--resume_j 6`  (no `--lora_adapter`) | `A2` `a02_rtax_ruler_A2_j6` |
| `N9`  | `--resume_j 9`  (no `--lora_adapter`) | `A3` `a02_rtax_ruler_A3_j9` |
| `N12` | `--resume_j 12` (no `--lora_adapter`) | `A4` `a02_ruler_c2_j12_readlora` |
| `N18` | `--resume_j 18` (no `--lora_adapter`) | `A5` `a02_rtax_ruler_A5_j18` |

Ceiling anchor for both readouts: **`A0`** = `ruler_results/a02_dvr_ruler_j0_top12` (j=0, no adapter).

4 arms × 4 cells = **16 cells**. Zero training steps: the native readout has no
parameters to fit. This is the whole reason B05's first GPU spend is ~3 GPU-h and not ~192.

---

## 2. Floor, and the Δ guard (both fixed pre-data)

### 2.1 Noise floor

Per `memory/same-harness-runs-bit-identical`, a re-run at **same arch + same disk + same
harness** in this repo is **byte-identical (0 flips)**. The "15–20 flip within-disk floor"
that circulated earlier was code-version drift between two harness versions, not runtime
jitter, and is **not** imported here. With arms paired on identical `input_ids_sha256`,
the only remaining variance is sampling error, i.e. the paired bootstrap CI.

**Resolution floor, taken from the comparator's own measured behaviour** rather than
invented: on the same 4 cells at n=100, A02's paired bootstrap (n_boot=5000, seed=42)
called **−4.0 pp [−8, −1] SIGNIFICANT** and **−3.0 pp [−7, 0] NOT significant**. So the
empirically demonstrated detection boundary at n=100/cell lies between 3 and 4 pp.

> **`FLOOR = 4.0 pp` per cell, plus paired-bootstrap CI95 excluding 0.** Both conditions,
> not either. A delta smaller than 4.0 pp is declared *indistinguishable* regardless of
> its CI.

### 2.2 Δ guard — the denominator can be zero or negative

The anchor `A0` sits at **100.0** on 3 of the 4 retrieval-closed cells (Wilson lo 96.3),
and at 99.0 on the fourth. Therefore:

* **All primary quantities are ABSOLUTE paired percentage points.** No headline number is
  a ratio.
* A recovery fraction `R_j = (LoRA_j − native_j) / (A0 − native_j)` is computed **only if**
  `(A0 − native_j) >= 10.0 pp` **and** that denominator's paired bootstrap CI95 lower
  bound `> 0`. Otherwise the analyzer must emit
  `{"recovery_fraction": null, "reason": "denominator <= 10.0pp or CI95_lo <= 0 -> Δ ill-defined"}`
  and the phase assignment falls back to the absolute-pp rule of §3. This is the case that
  actually bites at `j = 6`, where native may be at the ceiling and the denominator → 0.
* **Ceiling censoring is declared, not discovered:** an arm scoring 100.0 on a cell whose
  anchor is also 100.0 has an unmeasurable positive Δ. Such cells are reported as
  `"at_ceiling, positive Δ censored"` and may support *"indistinguishable"* but may never
  support *"better than"*.

### 2.3 Aggregation hygiene, inherited verbatim

`a02_read_tax_ruler.json`'s own `aggregation_hygiene` field: **"PER-CELL ONLY."** The only
permitted cross-cell number is the mean over the 4 retrieval-closed RULER cells, labelled
as such.

**BABILong / LongBench / LoCoMo are CONTRAST ONLY and may never enter a phase assignment.**
`A02_DEPTH_VS_RETRIEVAL_VERDICT.md` measured recall@12 = 22.9–63.2 % on those cells, so
54.9–78.6 % of the change there is a *retrieval* effect, not a read effect. B05's
"semantic QA" family is therefore **demoted to a contrast arm** by prior measurement, not
by preference. Likewise the "format/verbalization" family sits on a scorer already shown
(`code/analyze_a02_format_mechanism.py`) to order arms by first-period truncation; it is
out of the primary grid.

### 2.4 Two prohibitions carried over from dead work

* **Do not re-run the forward-probe-predicts-graft-depth regression.**
  `proposal/archive/paperC-v1-frozen-cap/POSTMORTEM.md` §4.1 killed it: four forward probes
  on one model span the entire depth domain (0.000L → 1.000L), and the knowledge logit-lens
  that P-C2 was built on regresses at only r=+0.7347 and is systematically too shallow.
  B05's `必须避免` item 3 is that epitaph. Violating it collapses B05 into the archive.
* **Do not report the bm25-era bracket/gap ladder as a B05 result.**
  `status/QCMEM_J_DETERMINATION.md` is `selector=bm25`; the standing rule
  (`memory/qcmem-eval-selector-iterbm25`, user 2026-07-17) voids old bm25 numbers.
  It motivates the grid; it is not evidence in it.

---

## 3. The phase definition, and the KILL GATE

### 3.1 Phases (assigned per cell, then required to agree across cells)

At each ladder rung `j ∈ {6, 9, 12, 18}`, using absolute paired pp vs the `A0` anchor:

| phase | condition |
|---|---|
| **I — joint readability** | native_j *and* LoRA_j are both within FLOOR of `A0` (i.e. `|Δ| < 4.0 pp` or CI95 includes 0), **and** native_j vs LoRA_j is itself within FLOOR |
| **II — handoff band** | LoRA_j within FLOOR of `A0`, **and** native_j below `A0` by `>= 4.0 pp` with CI95 excluding 0 → readout capacity is load-bearing at this depth |
| **III — joint failure** | *both* native_j and LoRA_j below `A0` by `>= 4.0 pp` with CI95 excluding 0 |

`j = 0` is **excluded** (degenerate: A1 vs A0 = 0 flips / 400, teacher==student at
`resume_j=0` per A02 GATE 0). A phase is **non-empty** only if at least one rung in
`{6, 9, 12, 18}` is assigned to it.

### 3.2 KILL GATE (pre-registered, fires on the 16-cell read-out)

> **B05 dies as a standalone paper if, on the 4 retrieval-closed RULER cells at
> Qwen3-8B, fewer than 3 of the phases {I, II, III} are non-empty over the rungs
> `j ∈ {6, 9, 12, 18}`; or if the Phase I/II boundary rung differs by `>= 2` ladder
> rungs between `niah_multikey_1` and `variable_tracking`.**
>
> On firing, B05's own pre-existing exit clause executes: fold into a Paper A/B
> mechanism subsection, do not make it standalone. It does **not** mean the direction
> was wrong — it means readout capacity and split depth are not separable into named
> phases at this resolution, which is itself a reportable negative for Paper A.

### 3.3 Where the gate's discriminating power actually sits — stated honestly

I am required to construct a failing counterexample before believing this is a gate.
Taking each phase in turn:

* **Phase III is already non-empty and will not fail.** A02's `A5` (j=18, LoRA) is
  20.75 vs anchor 99.75, i.e. −79.0 pp with CI95 [−99, −47] per cell. If LoRA is dead at
  j=18, native cannot be alive there. **This clause carries no power and I am not
  pretending it does.**
* **Phase II is very likely non-empty**, given the motivation numbers in §0
  (native j=9 ≈ 34.25 vs LoRA j=9 = 98.25). Low power. Also declared.
* **Phase I is the coin flip, and it is where this gate can actually kill.**
  Phase I requires a rung `j >= 6` at which native readout is *still intact*. The
  evidence genuinely does not settle it:
  - `status/QCMEM_J_DETERMINATION.md` places the 8B zero-shot readout cliff at
    j9 = 100 / j10 = 81 / j12 = 9 on **niah_single 16k** → suggests native survives j=6, so
    Phase I would be non-empty.
  - But the §0 merge shows native j=9 on the **harder retrieval-closed** cells is already
    44 / 59 / 18 / 16 — far below `niah_single`'s 100. So the cliff on *these* cells is
    **shallower than j=9**, and whether it is shallower than **j=6** is unmeasured.

**The concrete failing result:** if `N6` comes back at, say, 78 / 85 / 61 / 58 (mean ≈ 70.5),
then every rung in `{6, 9, 12, 18}` has native below `A0` by ≥ 4.0 pp with CI95 excluding 0.
Phase I is then **empty on the ladder**, only Phases II and III are populated, `2 < 3`,
and **the kill gate fires.** The read tax would be a two-regime cliff — which is what
Paper A already reports — and B05 would have no third phase to name. That outcome is
entirely plausible on the numbers above, so the gate is falsifiable.

**Second, independent failure route:** suppose `N6` = 96 / 97 / 62 / 59. Then
`niah_multikey_1` puts the I/II boundary at j = 9 while `variable_tracking` puts it at
j < 6 — a gap of ≥ 2 rungs — and the cross-task clause fires even though three phases
exist on the pooled view. This is the clause that stops "phase diagram" from being
retrofitted onto two unrelated per-task curves.

---

## 4. Read-out point (pre-registered; this is the decision point, not a step count)

B05 is offline eval, so there is no training step. The read-out is a **completeness
condition**, and it is enforced fail-closed by the driver, not by inspection:

1. **All 16 cells at 8/8 shards and n == 100 each.** Fork of
   `proposal/backlog/A02-comem-write-read-repair/code/run_a02_read_tax_eval.sh:174` —
   `if [ "$have" -ne "$want" ]; then note "  ABORT ruler $NAME: only $have/$want records"; return 9; fi`
   — which aborts the arm rather than merging a partial cell. This gate was
   **negative-tested in A02**, not merely asserted (deleting a shard produced
   `G1_SHARD_INCOMPLETE 7/8`), and it fired in production on unfinished cells.
2. **GATE C2 `input_ids_sha256` pairing PASS across all 4 native arms and their 4 LoRA
   comparators and `A0`** (9 arms), joined by `sample_index`. Analyzer:
   `code/analyze_a02_read_tax.py` (asserts sha equality, fail-closed;
   `N_EXPECT = 100` at line 90).
3. **GATE D config identity PASS**: `resume_j` as expected per arm, `lora_adapter is None`
   for all four native arms, `selector=iter_bm25`, `topk=12`, `iter_hop_topk=4`,
   `chunk_size=512`, `chat_template=False`. Read from the level each field actually lives
   at (A02's recorded bug: RULER stores config flat and carries no `chat_template`; that
   lives in the sibling summary — comparing `is not False`, never `is not True`).
4. **GATE E**: scorers **imported** from `analyze_a02_depth_vs_retrieval`, never
   reimplemented.

**The phase assignment of §3 is read exactly once, on the analyzer's emitted evidence
JSON, after conditions 1–4 all PASS.** No intermediate peeking at partial cells decides
anything. This file must be committed with a hash whose timestamp precedes the mtime of
the first B05 result file; if it is not, the prereg is void and the run is descriptive only.

---

## 5. Node / architecture requirement (not a preference)

**sm_90 (H20: `.73` / `.82` / `.104`, zwfy6 disk). Not B200.** Three independent reasons:

1. **The comparator was measured on sm_90.** `A02_READ_TAX_RULER_VERDICT.md:3` —
   "78 min wall, `.82` only (8× H20, zwfy6)". B05's native column is paired *against those
   exact arms*. Same-harness paired comparison must not straddle architectures, or stack
   drift confounds with hardware drift.
2. **The comparator's raw cells live on zwfy6.** `ruler_results/a02_rtax_ruler_*` and
   `ruler_results/a02_dvr_ruler_j0_top12` are recorded as zwfy6-resident
   (`A02_READ_TAX_RULER_VERDICT.md:324`) and are **absent on wzc1** (verified from LOCAL:
   `ls -d ruler_results/a02_*` → no match).
3. **The driver and analyzer are sha256-verified byte-identical on zwfy6** after `scp -O`
   (verdict lines 14–18), so no code transfer step is needed there.

No adapters are needed for B05's own arms (native readout = no LoRA), so the fact that 5
of A02's 6 adapters are wzc1-absent does not block B05 — but the **comparator result dirs
do** need to be present, which is why the node must be zwfy6.

⚠️ **Verify-before-dispatch (I could not check this from LOCAL):** `/apdcephfs_zwfy6` is
not mounted on this node (`ls: cannot access '/apdcephfs_zwfy6'`). Every zwfy6 path above
is **recorded-only from here**. Per `memory/two-disk-rule-applies-to-main-too`, the
dispatching agent must `ls` on `.73`/`.82`/`.104` and confirm all five comparator dirs
(`a02_dvr_ruler_j0_top12`, `a02_rtax_ruler_A2_j6`, `a02_rtax_ruler_A3_j9`,
`a02_ruler_c2_j12_readlora`, `a02_rtax_ruler_A5_j18`) each carry 8/8 `*.records.json`
**before** launching. If any comparator dir lacks `records.json`, that rung's pairing is
impossible and the rung must be dropped from the ladder *and the drop recorded here*.

---

## 6. Cost, from measured anchors only

**Anchor 1 (direct, same 4 cells, same native readout, 8 shards):**
`ruler_results/qcmem_8b_zeroshot_j9_chatFALSE/{cell}_shard*of8.json` → `elapsed_seconds`,
max over the 8 shards × 8 GPUs:

| cell | max shard s | GPU-h / cell |
|---|---|---|
| niah_multikey_1 16k | 37.0 | 0.0822 |
| niah_multikey_1 32k | 70.2 | 0.1561 |
| variable_tracking 16k | 67.6 | 0.1502 |
| variable_tracking 32k | 191.6 | 0.4259 |
| **per arm (4 cells)** | | **0.8144** |

→ 4 arms × 0.8144 = **3.26 GPU-h**.

**Anchor 2 (independent cross-check, sm_90):** `A02_READ_TAX_RULER_VERDICT.md:3` — 78 min
wall × 8 H20 = 10.4 GPU-h for 5 arms × 10 cells = 50 cells → **0.208 GPU-h/cell**;
16 cells → **3.33 GPU-h**. The two anchors agree to ~2 %.

**Budget: 3.3 GPU-h, ≤ 5 GPU-h with 50 % headroom for queueing and the VT 60-token
override.** That is ~25 min wall on one 8×H20 node. Compare `N1:6`'s pre-data guess of
"1 node ~1 day" ≈ 192 GPU-h — a **~58×** overestimate, entirely because that estimate
assumed the 4-readout × 4-task 64-cell grid rather than the one column that is actually
decisive and requires no training.

---

## 7. What is explicitly NOT in this gate

* No `small decoder` readout column — it has no measured rate anywhere in this repo, so
  costing it would be an invented number. Dropped as a legitimate narrowing.
* No `affine/tuned lens` column in the *primary* gate. It is nearly free (`extract_sec: 5.9`
  in `results/knowledge_logit_lens_Qwen3-8b-local.json`) but it is a **descriptive band**,
  never a predictor of graft depth (§2.4).
* No second model family. Cross-family replication is a *post-gate* extension and would
  cost 2.91 GPU-h per LoRA comparator adapter (measured: `outputs/qcmem_distill_qwen_j12_r32_4k`,
  4000 steps, `world_size=8`, ~24.5 samp/s per `logs/qcmem_distill_qwen_j12_r32_4k.log:421`;
  mtimes 00:04:02 → 00:25:50 = 1308 s wall × 8 = 2.91 GPU-h). It is not in the kill gate.
* No `parametric knowledge` phase cell (dead-probe prohibition, §2.4).
