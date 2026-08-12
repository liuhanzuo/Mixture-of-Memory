# A02 — READ TAX vs DEPTH on RULER (retrieval-closed): VERDICT

**Executed**: 2026-08-12, 21:06:59 → 22:25:10 CST (78 min wall), `.82` only (8× H20, zwfy6).
**This is the step A02's own training log named as next**, verbatim:

```
[08-12 17:38:06] ALL ARMS DONE
[08-12 17:38:06] next: offline eval of A0/A1/A2/A3/A4/A5/A6 on RULER (retrieval-closed) per PREREG 2.6
```

**Pre-registration**: `A02_J0_DEPTH_CONTROL_PREREG.md` §2.6 (read-out) / §2.7 (predictions).
The four §2.7 predictions were read **before** any result was inspected; each is
adjudicated in §3 below.
**Driver**: `code/run_a02_read_tax_eval.sh`, sha256 `295f6f56ac42b2cb594e2dee7801c25f9e93d1429c0a81318aa5fa153cbaa912`
(**identical on wzc1 and zwfy6**, verified after `scp -O`).
**Analyzer**: `code/analyze_a02_read_tax.py`, sha256 `3123319d7cc8bea17ab6a0365aec58112f0d131234eb5009116db63f6a4a8730`
(**identical on wzc1 and zwfy6**); it **imports** `analyze_a02_depth_vs_retrieval.py`'s
loaders/scorers/CI rather than reimplementing them (PREREG GATE E).
**Evidence**: `evidence/read_tax_ruler/a02_read_tax_ruler.json` (md5 `fcb32b6dddb90cc5ca64925b5740462b`)
+ `a02_read_tax_per_item_vectors.json` (md5 `3044dbf9f9e9929921c359b1dffe1ced`) — **md5-identical on both disks**.

---

## 0. RAN vs READ — what is new here and what is reused

**RAN (new GPU spent in this gate)** — 5 arms × (4 RULER cells + 6 BABILong cells),
all 8 shards, `limit 100`, on `.82`:

| arm | j | adapter | trainable | RULER | BABILong |
|---|---|---|---|---|---|
| A1 | 0 | `a02_j0control_lora_r32/final` | 87.29 M | **RAN** | **RAN** |
| A2 | 6 | `qcmem_distill_qwen_j6_r32_4k/final` | 72.74 M | **RAN** (never before) | **RAN** |
| A3 | 9 | `qcmem_distill_qwen_j9_r32_4k/final` | 65.47 M | **RAN** (never before) | **RAN** |
| A5 | 18 | `qcmem_distill_qwen_j18_r32_4k/final` | 43.65 M | **RAN** (never before) | **RAN** |
| A6 | 12 | `a02_j12_capmatch_r40/final` | 72.74 M | **RAN** | **RAN** |

**READ (reused from disk, NOT re-run)** — the two anchors from the dvr gate. Their
cell configs were re-verified field-by-field before use (`resume_j`, `selector=iter_bm25`,
`topk=12`, `iter_hop_topk=4`, `chunk_size=512`, `baseline=none`, `chat_template=false`,
`seed=42`), so the pairing is by construction, not by assumption:

| arm | j | adapter | source on disk |
|---|---|---|---|
| **A0** | 0 | **none** (= the optimal j=0 adapter, per GATE 0) | `ruler_results/a02_dvr_ruler_j0_top12`, `babilong_results/a02_dvr_babilong_j0_top12` |
| **A4** | 12 | flagship `qcmem_distill_qwen_j12_r32_4k/final`, sha `dd09cd17…` | `ruler_results/a02_ruler_c2_j12_readlora`, `babilong_results/a02_babilong_c2_j12_readlora` |

Re-running A0/A4 would have changed nothing but risked overwriting the very anchors
the paired deltas are measured against, so they were deliberately left untouched.

**Also READ (not re-measured here)**: the recall@12 numbers that license calling
RULER "retrieval-closed" come from the dvr gate (`A02_DEPTH_VS_RETRIEVAL_VERDICT.md` §4):
niah_mk1 **100.0 %** @16k / **99.0 %** @32k. See §5 for the honest caveat on
`variable_tracking`, whose recall was **never directly measured**.

### Gate status — all fail-closed gates PASSED

```
GATE A  flagship Read-LoRA sha dd09cd17457c63578c0f            PASS
GATE B  adapter spans/ranks + exact capacity arithmetic         PASS
        A1 r32 [0..35] n=36 87.29M | A2 r32 [6..35] n=30 72.74M
        A3 r32 [9..35] n=27 65.47M | A5 r32 [18..35] n=18 43.65M
        A6 r40 [12..35] n=24 72.74M
        capacity: A2 == A6 == 72,744,960  (exact)
GATE C  shard completeness + per-cell n==100 + 0 dup ids + 0 NaN PASS   (0 refused cells)
GATE C2 RULER input_ids_sha256 pairing across all 7 arms         PASS   (0 failures)
GATE D  chat_template=False + iter_bm25 + topk12 + chunk512
        + expected resume_j + expected adapter, all 7 arms       PASS   (0 config errors)
GATE E  canonical scorers imported, never reimplemented          PASS
```

**GATE C was negative-tested**, not merely asserted: deleting one shard from a scratch
copy makes it refuse the cell (`G1_SHARD_INCOMPLETE 7/8`) instead of merging 7/8.
Two intermediate analyzer runs during the eval genuinely refused the not-yet-finished
BABILong cells — the gate fired in production, not just in the self-test.

> ⚠️ **A GATE D bug in my first analyzer version was found and fixed, and it is worth
> recording because it would have silently passed.** I initially read `chat_template`
> from the RULER `records.json` top level and the QCMem block from a `qcmem` sub-dict.
> On disk the RULER `records.json` stores config **flat** at top level and carries **no**
> `chat_template` (that lives in the sibling summary `*.json`), while BABILong nests
> `chat_template` under **`prompt`**. My first version therefore reported 192 config
> "errors" (all spurious `None` reads). Had the comparison been `is not True` instead of
> `is not False`, it would have **silently passed on a `None`** and I would have claimed
> chat_template was verified when nothing was checked. The fixed gate reads each field
> from the level it actually lives at and now returns 0 errors across 7 arms × 10 cells.

---

## 1. PRIMARY — RULER, where retrieval is closed

Per-cell accuracy (n=100/cell, 8 shards merged, Wilson 95 % CI) and the **paired
per-cell delta vs A0** (paired bootstrap, n_boot=5000, seed=42; `*` = CI95 excludes 0).
This delta is *the read tax at depth j*.

| arm | j | params | niah_mk1 16k | niah_mk1 32k | var_track 16k | var_track 32k | **mean** |
|---|---|---|---|---|---|---|---|
| **A0** | 0 | — | **100.0** | **99.0** | **100.0** | **100.0** | **99.75** |
| **A1** | 0 | 87.29 M | 100.0 | 99.0 | 100.0 | 100.0 | 99.75 |
| **A2** | 6 | 72.74 M | 99.0 | 99.0 | 99.0 | 100.0 | 99.25 |
| **A3** | 9 | 65.47 M | 99.0 | 95.0 | 99.0 | 100.0 | 98.25 |
| **A4** | 12 | 58.20 M | 90.0 | 96.0 | 88.0 | 89.0 | 90.75 |
| **A5** | 18 | 43.65 M | 32.0 | 42.0 | 4.0 | 5.0 | 20.75 |
| **A6** | 12 | 72.74 M | 90.0 | 96.0 | 88.0 | 88.0 | 90.50 |

**Paired read tax vs A0** (pp, CI95):

| arm | j | niah_mk1 16k | niah_mk1 32k | var_track 16k | var_track 32k | mean |
|---|---|---|---|---|---|---|
| A1 | 0 | +0.00 [+0.00,+0.00] | +0.00 [+0.00,+0.00] | +0.00 [+0.00,+0.00] | +0.00 [+0.00,+0.00] | **0.00** |
| A2 | 6 | −1.00 [−3.00,+0.00] | +0.00 [+0.00,+0.00] | −1.00 [−3.00,+0.00] | +0.00 [+0.00,+0.00] | **−0.50** |
| A3 | 9 | −1.00 [−3.00,+0.00] | −4.00 [−8.00,−1.00]\* | −1.00 [−3.00,+0.00] | +0.00 [+0.00,+0.00] | **−1.50** |
| A4 | 12 | −10.00 [−16.00,−5.00]\* | −3.00 [−7.00,+0.00] | −12.00 [−19.00,−6.00]\* | −11.00 [−18.00,−5.00]\* | **−9.00** |
| A5 | 18 | −68.00 [−77.00,−58.00]\* | −57.00 [−66.00,−47.00]\* | −96.00 [−99.00,−92.00]\* | −95.00 [−99.00,−90.00]\* | **−79.00** |
| A6 | 12 | −10.00 [−16.00,−5.00]\* | −3.00 [−7.00,+0.00] | −12.00 [−19.00,−6.00]\* | −12.00 [−19.00,−6.00]\* | **−9.25** |

**The read tax is not linear in j — it is a cliff.** j=0→9 costs ≤1.5 pp; j=12 costs
9 pp; j=18 costs **79 pp**, i.e. the model stops functioning. The knob is nearly free
for the first quarter of the network and catastrophic past a third of it.

**A5 (j=18) is qualitatively broken, not merely worse.** Its `variable_tracking`
outputs are unrelated token salad (`'EDI RPY DTM QRF ZIF'` against target
`'WNHQX | ZGRCN | LMKVK | ECIWK | LJKMR'`), sometimes with the chain partially
right but duplicated (`'OQONN RNPVV ZRRCO OQONN RNPVV'`). At 4–5 % on VT this is
the "untrained-depth-resume is non-functional" regime, now shown to arrive **even
with a fully distilled adapter** at 43.65 M params.

---

## 2. The matched-capacity contrast (what Arm 2 exists for)

Capacity is structurally tied to j (`layers[j:36]`), so the ladder is capacity-**un**matched
by construction. A6 was trained to break that tie **exactly**:
`75776 × 40 × 24 = 75776 × 32 × 30 = 72,744,960` params.

**A2 (j=6, r32) vs A6 (j=12, r40) — identical trainable params, only depth differs:**

| cell | A2 (j=6) | A6 (j=12) | delta | CI95 | |
|---|---|---|---|---|---|
| niah_mk1 16k | 99.0 | 90.0 | −9.00 | [−15.00, −4.00] | SIG |
| niah_mk1 32k | 99.0 | 96.0 | −3.00 | [−7.00, +0.00] | ns |
| var_track 16k | 99.0 | 88.0 | −11.00 | [−17.00, −5.00] | SIG |
| var_track 32k | 100.0 | 88.0 | −12.00 | [−19.00, −6.00] | SIG |
| **mean** | **99.25** | **90.50** | **−8.75** | | 3/4 SIG |

**So the depth effect survives exact capacity matching.** The j=6→j=12 gap is
−8.75 pp at *identical* parameter count, versus −8.75 pp between the same two arms
on the unmatched ladder (A2 99.25 → A4 90.75 = −8.50 pp). The curve is a **depth**
curve, not a capacity artefact.

---

## 3. The four pre-registered predictions, adjudicated

### Prediction 1 — "A1 ≈ A0, or A1 < A0." → **CONFIRMED (in its strongest form)**

A1 = A0 on **all four** RULER cells, delta **+0.00 pp with CI95 [+0.00, +0.00]**.
Stronger than aggregate equality: **0 correctness flips out of 400 paired items**.
Not a single sample changed verdict. On the secondary BABILong cells A1 is
−1/−1/−1/−1/+0/+2 pp vs A0 (22 flips / 600, all n.s.), i.e. noise around identity.

A1 is **not** significantly better than A0 anywhere, so GATE 0's vacuity argument
stands and **§1 of the PREREG does not need retracting** (its §6 failure mode
"if A1 evaluates better than A0, 'optimal j=0 adapter == identity' is wrong" did
not trigger).

### Prediction 2 — "A1's training loss stays ~1e-3 for all 4000 steps." → **CONFIRMED**

Machine-checked in A02's own training log before this eval began:

| | flagship j=12 | **A1 j=0** |
|---|---|---|
| loss @ step 10 | 0.2991 | **0.0011** |
| loss @ step 2010 | — | **0.0015** |
| loss @ step 4000 | 0.0555 | **0.0012** |

400 logged points, no descent, ~250× below the flagship's *starting* loss. The j=0
objective was at its optimum at initialisation.

### Prediction 3 — "Read tax is monotone-ish in j across A0/A2/A3/A4/A5." → **CONFIRMED**

On the r=32 ladder means (99.75 → 99.25 → 98.25 → 90.75 → 20.75):
**Spearman ρ = −1.000 (p = 1.4e-24)** — perfectly monotone in rank.
Successive diffs: −0.5, −1.0, −7.5, −70.0 pp; **4/4 steps non-increasing**.
Pearson r = −0.800 (p = 0.104) is *weaker only because the relationship is a cliff,
not a line* — which is itself the finding. Per-cell: 4/4 monotone steps in 3 of 4
cells; niah_mk1×32k has one +1.0 pp blip (A3 95 → A4 96), well inside noise.

### Prediction 4 — "A6 ≈ A4." → **CONFIRMED**

Direct paired contrast at j=12, r40 (72.74 M) vs r32 (58.20 M):

| cell | A4 | A6 | delta | CI95 | |
|---|---|---|---|---|---|
| niah_mk1 16k | 90.0 | 90.0 | +0.00 | [−3.00, +3.00] | ns |
| niah_mk1 32k | 96.0 | 96.0 | +0.00 | [+0.00, +0.00] | ns |
| var_track 16k | 88.0 | 88.0 | +0.00 | [−3.00, +3.00] | ns |
| var_track 32k | 89.0 | 88.0 | −1.00 | [−5.02, +3.00] | ns |

363/400 vs 362/400 correct. **All four n.s.** A 25 % capacity increase (+14.5 M params)
buys **nothing** at fixed depth. So the PREREG §6 failure mode "if A6 ≠ A4 materially,
every depth number must be re-stated at matched capacity" did **not** trigger; the
dvr verdict's depth numbers stand as written.

**4 / 4 predictions confirmed.** No retraction required.

---

## 4. SECONDARY — BABILong, contrast only

**These cells cannot support depth inference** (dvr: recall@12 = 22.9–63.2 %, i.e.
retrieval-dominated). Reported per-cell **only**, to show *why* the primary read-out
had to be retrieval-closed.

| arm | j | qa1 16k | qa1 32k | qa2 16k | qa2 32k | qa5 16k | qa5 32k |
|---|---|---|---|---|---|---|---|
| A0 | 0 | 33.0 | 35.0 | 17.0 | 11.0 | 53.0 | 61.0 |
| A1 | 0 | 32.0 | 34.0 | 16.0 | 10.0 | 53.0 | 63.0 |
| A2 | 6 | 23.0 | 21.0 | 13.0 | 3.0 | 62.0 | 62.0 |
| A3 | 9 | 21.0 | 14.0 | 8.0 | 2.0 | 63.0 | 57.0 |
| A4 | 12 | 17.0 | 12.0 | 8.0 | 1.0 | 58.0 | 58.0 |
| A5 | 18 | 19.0 | 12.0 | 9.0 | 7.0 | 49.0 | 48.0 |
| A6 | 12 | 19.0 | 12.0 | 8.0 | 1.0 | 61.0 | 59.0 |

**This table is the argument for the protocol, and it is decisive.** Compare the
two read-outs for the same arm A5 (j=18):

* **RULER (retrieval closed)**: −79.00 pp. Model is destroyed.
* **BABILong qa1×16k**: −14.00 pp — and qa5×16k is only **−4.0 pp, n.s.**

A retrieval-dominated benchmark **masks a catastrophic read failure**, reporting a
mid-teens penalty for an arm that has actually collapsed. Worse, BABILong **misorders**
the arms: it ranks A5 (j=18, RULER 20.75) as *better than or equal to* A4 (j=12,
RULER 90.75) on 4 of 6 cells (qa1×16k 19 vs 17, qa2×16k 9 vs 8, qa2×32k 7 vs 1,
qa1×32k 12 vs 12). Any depth conclusion drawn from these cells would have been not
merely attenuated but **wrong in sign**. The floor effect (qa2×32k spans 1–11 %) is
what does it.

**Aggregation hygiene**: per-cell only. No pooled BABILong/LongEval figure was computed.
The banned pooled numbers (−17.89 pp / +2.00 pp) appear **nowhere** in the evidence JSON
except inside the note that names them as banned (verified by grep: one occurrence, in
`aggregation_hygiene`). The one cross-cell mean present is over the **4 RULER
retrieval-closed cells only** and is labelled as such in the JSON key itself.

---

## 5. What this licenses — and what it does NOT

**LICENSED**

1. **`next_gate[3]` is RETIRED, by execution rather than assertion.** The requested
   "matched-quality depth control (a LoRA distilled for j=0)" was trained to
   completion (4000 steps, 87.29 M params, 73 min) and evaluated. It is a **null
   adapter**: 0/400 correctness flips vs the base model on RULER. The defect
   *dissolves* rather than being repaired.
2. **`A02_DEPTH_VS_RETRIEVAL_VERDICT.md` §7.1 should be restated.** Its
   "there is NO arm here that isolates depth with a matched-quality adapter" is
   **over-pessimistic**: since the optimal j=0 adapter *is* the identity (now measured,
   not argued), the dvr `read_deployed` step already was the matched-quality contrast.
3. **A 5-point depth-tax curve on a retrieval-closed benchmark** (j = 0/6/9/12/18),
   with retrieval byte-identical and the distillation recipe matched on 22 fields —
   the first such curve in this project. Shape: **free to j≈9, −9 pp at j=12,
   −79 pp at j=18.** A cliff, not a slope.
4. **The curve is depth, not capacity** (A2 vs A6 at exactly 72,744,960 params:
   −8.75 pp; A6 vs A4 at fixed depth: 0.00 pp n.s.).
5. **A methodological result with reach beyond A02**: on the *same* arms and the *same*
   protocol, a retrieval-dominated benchmark under-reports a read failure by ~65 pp
   and inverts the arm ordering. Benchmark choice is not cosmetic for depth claims.

**NOT LICENSED**

* **No revival of A02's thesis.** Storage form stays **DEAD** (h12 = 2048× raw text).
  Read-compute stays a **1.03–1.37×** micro-optimisation. Every number in this document
  is a **tax**; the best possible outcome here was "the knob is cheap", and even that
  only holds for j ≤ 9.
* **No quality win.** A0 (no adapter at all) is the **best** arm on RULER at 99.75.
  Nothing beat doing nothing.
* **No claim that j≤9 is "free" in general** — it is free *on these two RULER tasks at
  16k/32k*, where the base model is already at ceiling (99–100 %). A ceiling cannot
  show an improvement and can hide a small regression.
* **No cross-model / cross-family claim**: Qwen3-8B only, one seed (42), H20/bf16/sdpa.
* **No BABILong-based depth claim** (§4), and no pooled figure of any kind.
* **No claim about differential LR or capacity matching across the ladder** — the
  ladder is capacity-unmatched by construction; only A2-vs-A6 and A4-vs-A6 are matched.

### Honest caveats (carried forward, not hidden)

1. **`variable_tracking` recall@12 was never directly measured** (dvr §7.3: `n/a`,
   0 gold-locatable chunks — `_locate_needle_chunks` cannot localise a VT chain).
   Two of my four primary cells are VT, and their "retrieval-closed" status rests on
   the dvr *accuracy* step (j0_top12 100.0 vs pack-all 100.0/99.0), **not** on a recall
   number. The inference is sound for niah_mk1 (recall 99–100 % measured directly) and
   *inherited* for VT. Notably the VT cells show the **largest** j=18 collapse (4–5 %),
   so if VT recall were secretly poor the depth tax would be *over*-stated there — the
   niah_mk1 cells alone still give −68/−57 pp at j=18, so the cliff does not depend on VT.
2. **Ceiling at j ≤ 9.** A0/A1/A2/A3 sit at 95–100 % on all four cells. "Tax ≈ 0" at
   shallow j is therefore a statement about a saturated benchmark; a harder
   retrieval-closed task could resolve differences these cells cannot.
3. **A1's adapter did move — it just did not learn.** `||lora_B||₂` = 15.20 (flagship:
   15.47), 100 % of entries non-zero, i.e. 4000 steps of AdamW did write a full-norm
   update. But `max|B|` = 1.38e-2 vs the flagship's 6.52e-2 (**4.7× smaller**): a
   diffuse isotropic perturbation rather than concentrated learned structure. This is
   exactly PREREG §1.2's predicted "Adam-amplified random walk on bf16 noise" — and it
   is *visible in the outputs*: A1's generations differ from A0's on 30/32/12/49 of 100
   samples per cell, yet **zero** of those differences change correctness. The
   divergence is confined to post-answer continuation text (`'…as "One of the special
   magic numbers…'` vs `'…as: "One of the special…'`); the answer span is untouched.
   So "null adapter" means **null in effect**, not null in weights — a sharper statement
   than vacuity, and the reason the PREREG insisted on running 4000 steps instead of
   asserting from the 20-step probe.
4. **One seed**, n=100/cell. The ±1 pp differences among A0/A1/A2/A3 are not resolvable;
   only the j=12 (−9 pp) and j=18 (−79 pp) steps are.

---

## 6. Provenance

| artefact | location (both disks unless noted) |
|---|---|
| driver | `code/run_a02_read_tax_eval.sh` sha256 `295f6f56ac42b2cb…` |
| analyzer | `code/analyze_a02_read_tax.py` sha256 `3123319d7cc8bea1…` (imports the dvr analyzer verbatim) |
| per-cell + paired deltas | `evidence/read_tax_ruler/a02_read_tax_ruler.json` md5 `fcb32b6dddb90cc5ca64925b5740462b` |
| per-item vectors | `evidence/read_tax_ruler/a02_read_tax_per_item_vectors.json` md5 `3044dbf9f9e9929921c359b1dffe1ced` |
| eval progress log | `logs/a02_read_tax_eval_progress.log` (**zwfy6 / `.82`**) |
| new RULER cells | `ruler_results/a02_rtax_ruler_{A1_j0control,A2_j6,A3_j9,A5_j18,A6_j12_r40}/` (**zwfy6**) |
| new BABILong cells | `babilong_results/a02_rtax_babilong_{A1_j0control,A2_j6,A3_j9,A5_j18,A6_j12_r40}/` (**zwfy6**) |
| A0 / A4 anchors | `ruler_results/a02_dvr_ruler_j0_top12`, `ruler_results/a02_ruler_c2_j12_readlora` (**zwfy6**) |
| trained adapters | `outputs/a02_j0control_lora_r32/final`, `outputs/a02_j12_capmatch_r40/final` (**zwfy6**) |
| flagship recipe | `outputs/qcmem_distill_qwen_j12_r32_4k/distill_args.json` — **wzc1 only** (it trained on L20A; its absence on zwfy6 is expected, not missing) |

**Node discipline**: only `.82` was used. `LOCAL`/`.21` (SparseForge #246), `.104`
(paperC Qwen3 heal) and `.73` were never contacted — no ssh, no process, no file write.
`.82` verified 8/8 cards at 0 MiB / 0 % before launch and released to 0 MiB / 0 % after.
