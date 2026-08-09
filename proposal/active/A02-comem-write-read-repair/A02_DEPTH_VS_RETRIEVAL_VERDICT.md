# A02 — Depth vs Retrieval Quality Gate: VERDICT

**Gate run**: 2026-08-10, node `.82` (8× H20, zwfy6 disk), bounded 4-GPU pool
(GPUs 0–3; GPUs 4–7 left free for the live A03 eval watcher, which was never
touched and was still running at exit).
**Driver**: `proposal/active/A02-comem-write-read-repair/code/run_a02_depth_vs_retrieval.sh`
**Analyzers**: `.../code/analyze_a02_depth_vs_retrieval.py`,
`.../code/analyze_a02_conditional_recall.py`
**Evidence**: `evidence/a02_depth_vs_retrieval_{ci,per_item,conditional}.json`
(byte-identical on wzc1 and zwfy6; md5 verified both sides)

---

## 0. Bottom line

**Both causes are real, and which one dominates is decided by the benchmark —
so the single-sentence answer phase-1 implied does not exist.**

* On **BABILong qa1/qa2** (the four decisive phase-1 cells): **RETRIEVAL
  dominates**, 54.9–78.6 % of the change, on all four (though at qa1×32k it only
  leads −28 vs −23, so that cell is "both contribute").
* On **RULER** (both tasks, both lengths): **retrieval is exactly ~0** — recall@12
  is 99–100 % — and **the entire loss is the mid-depth read**.
* Therefore **phase-1's verdict needs restating for BABILong but stands for
  RULER.** Its "C1 beats C2" reading conflated a retrieval effect (BABILong) with
  a read effect (RULER) and reported them as one phenomenon.

**A02's mid-layer-read thesis was only ever properly tested by RULER**, where the
deployed CoMem read costs a real, significant **−3 to −12 pp** with retrieval held
identical. That is a genuine falsification-relevant number, and it is **far
smaller than the −35 to −55 pp phase-1 headline**, which was mostly retrieval.

---

## 1. What was wrong with phase 1 (verified on disk, not from prose)

Read from the actual `*_shard0of8.json` cell configs on zwfy6:

| | `no_retrieval` | `selector` | `topk` | `resume_j` | LoRA |
|---|---|---|---|---|---|
| **C1** | `True` | `None` | `None` | 0 | `None` |
| **C2** | `False` | `iter_bm25` | 12 | 12 | flagship |

So `{read depth} × {LoRA} × {retrieval-vs-pack-all} × {selector}` all moved at
once. A −55 pp C1→C2 gap was therefore uninterpretable: it could be a mid-layer
read failure (A02's thesis) or a top-12 recall failure (a property of
`iter_bm25`, nothing to do with CoMem's memory).

## 2. The arms

Four arms; **each adjacent pair differs in exactly one variable.** The two middle
arms are new; the outer two are phase-1's existing on-disk runs.

| arm | depth | LoRA | pack | selector | status |
|---|---|---|---|---|---|
| `c1_pack_all` | j=0 | no | ALL chunks | (none) | phase-1 C1, on disk |
| `j0_top12` | j=0 | no | top-12 | `iter_bm25` | **NEW** |
| `j12_frozen` | j=12 | **no** | top-12 | `iter_bm25` | **NEW** |
| `c2_comem` | j=12 | yes | top-12 | `iter_bm25` | phase-1 C2, on disk |

Steps: `retrieval` = c1→j0_top12 · `read_depth` = j0_top12→j12_frozen ·
`read_lora` = j12_frozen→c2_comem · **`read_deployed` = j0_top12→c2_comem**
(the mid-depth read *as actually shipped*, retrieval identical — the
like-for-like number).

`retrieval + read_deployed` **exactly** partitions the total c1→c2 change, and
`read_depth + read_lora` **exactly** equals `read_deployed`. Both identities are
asserted arithmetically in the evidence and hold to 1e-6 on all 10 cells, so no
effect is double-counted.

### The confound the cost gate called irreducible is now broken

The cost gate had only j=0 (no LoRA) and j=12 (+LoRA), so it declared
depth↔LoRA inseparable. **`j12_frozen` separates them**: j=12 with retrieval
identical and the adapter *dropped*. It is a functional arm (this repo has
shipped `*_j12_frozen_iterbm25_chatFALSE` runs before), verified on disk as
`resume_j=12, selector=iter_bm25, topk=12, lora_adapter=None`.

## 3. Per-cell result (never pooled)

pp change, paired bootstrap n_boot=5000 seed=42, CI95; `*` = CI excludes 0.

| cell | c1 | j0_top12 | j12_frozen | c2_comem | **RETR** | **DEPLOY** | depth | lora | rec@12 | dominant |
|---|---|---|---|---|---|---|---|---|---|---|
| babilong qa1×16k | 72.0 | 33.0 | 3.0 | 17.0 | **−39.0\*** | −16.0\* | −30.0\* | +14.0\* | 63.2 | **retrieval (70.9 %)** |
| babilong qa1×32k | 63.0 | 35.0 | 3.0 | 12.0 | **−28.0\*** | −23.0\* | −32.0\* | +9.0\* | 57.0 | retrieval (54.9 %) — **not decisive** |
| babilong qa2×16k | 50.0 | 17.0 | 4.0 | 8.0 | **−33.0\*** | −9.0 ns | −13.0\* | +4.0 ns | 49.5 | **retrieval (78.6 %)** |
| babilong qa2×32k | 36.0 | 11.0 | 2.0 | 1.0 | **−25.0\*** | −10.0\* | −9.0\* | −1.0 ns | 22.9 | **retrieval (71.4 %)** |
| babilong qa5×16k | 44.0 | 53.0 | 45.0 | 58.0 | +9.0 ns | +5.0 ns | −8.0 ns | +13.0\* | 64.1 | (both n.s. — unattributable) |
| babilong qa5×32k | 59.0 | 61.0 | 41.0 | 58.0 | +2.0 ns | −3.0 ns | −20.0\* | +17.0\* | 57.9 | (both n.s. — unattributable) |
| ruler niah_mk1×16k | 100.0 | 100.0 | 3.0 | 90.0 | **+0.0 ns** | **−10.0\*** | −97.0\* | +87.0\* | **100.0** | **read (100 %)** |
| ruler niah_mk1×32k | 98.0 | 99.0 | 1.0 | 96.0 | +1.0 ns | −3.0 ns | −98.0\* | +95.0\* | **99.0** | read (75.0 %) |
| ruler var_track×16k | 100.0 | 100.0 | 0.0 | 88.0 | **+0.0 ns** | **−12.0\*** | −100.0\* | +88.0\* | n/a | **read (100 %)** |
| ruler var_track×32k | 99.0 | 100.0 | 0.0 | 89.0 | +1.0 ns | **−11.0\*** | −100.0\* | +89.0\* | n/a | **read (91.7 %)** |

"Dominant" = the larger of |RETR| and |DEPLOY| (the two-way partition).
`dominant_is_decisive` in the JSON marks whether the larger is ≥2× the smaller;
it is **False** for qa1×32k (−28 vs −23, i.e. retrieval leads but the read is
nearly as large) and for both qa5 cells (where neither step is significant).
Those three cells are honestly **"both contribute"**, not clean wins.

## 4. Retrieval recall of the top-12 `iter_bm25` pack

Measured directly, **independently of answer accuracy**: a HIT iff every
gold-support chunk index lands in the pack. Wilson 95 % CI.

| cell | recall@12 | CI | n gold-locatable | mean n_ctx chunks |
|---|---|---|---|---|
| babilong qa1×16k | **63.2 %** | [53.1, 72.2] | 95 | 29.6 |
| babilong qa1×32k | **57.0 %** | [47.2, 66.3] | 100 | 60.0 |
| babilong qa2×16k | **49.5 %** | [39.6, 59.4] | 95 | 29.6 |
| babilong qa2×32k | **22.9 %** | [15.7, 32.3] | 96 | 59.9 |
| babilong qa5×16k | 64.1 % | [54.0, 73.2] | 92 | 29.2 |
| babilong qa5×32k | 57.9 % | [47.9, 67.3] | 95 | 59.8 |
| ruler niah_mk1×16k | **100.0 %** | [96.2, 100.0] | 97 | 30.0 |
| ruler niah_mk1×32k | **99.0 %** | [94.6, 99.8] | 100 | 62.0 |
| ruler var_track×{16k,32k} | n/a | — | 0 | 31.0 / 63.0 |

**This is the number that settles it.** Top-12 `iter_bm25` **fails to retrieve
the gold support on 37–77 % of BABILong qa1/qa2 samples** — worst at qa2×32k
where it misses **77 %**. On RULER it retrieves the needle essentially always
(99–100 %). So the retrieval channel is wide open on BABILong and closed on
RULER, exactly matching where each cause dominates. Recall also **degrades with
length** (qa2: 49.5 % → 22.9 % from 16k→32k), which is why phase-1's C1 wins grew
with context: pack-all has recall ≡ 100 % by construction.

## 5. Conditional-on-recall check (internal validity)

Accuracy split by retrieval HIT/MISS (`evidence/..._conditional.json`). If
retrieval were the whole story, the retrieval step would be ~0 on HIT rows.

| cell | subset | n | c1 | j0_top12 | j12_frozen | c2_comem | RETR step |
|---|---|---|---|---|---|---|---|
| qa1×16k | hit | 60 | 76.67 | 40.00 | 1.67 | 15.00 | −36.67 SIG |
| qa1×16k | miss | 35 | 65.71 | 20.00 | 2.86 | 17.14 | −45.71 SIG |
| qa2×16k | hit | 47 | 48.94 | 27.66 | 4.26 | 12.77 | −21.28 SIG |
| qa2×16k | miss | 48 | 52.08 | **8.33** | 4.17 | 2.08 | −43.75 SIG |
| qa2×32k | hit | 22 | 40.91 | 31.82 | 0.00 | 0.00 | −9.09 **ns** |
| qa2×32k | miss | 74 | 33.78 | **5.41** | 2.70 | 0.00 | −28.38 SIG |
| niah_mk1×32k | hit | 99 | 97.98 | 100.00 | 1.01 | 96.97 | +2.02 **ns** |
| niah_mk1×32k | miss | 1 | 100.00 | 0.00 | 0.00 | 0.00 | −100.00 SIG |

(Full table incl. qa1×32k and qa5 in `evidence/a02_depth_vs_retrieval_conditional.json`.)

Two things follow, and the second is an honest qualification of §4:

1. **The retrieval mechanism is confirmed causal.** The retrieval penalty is
   always larger on MISS than on HIT (qa1×16k −45.7 vs −36.7; qa2×16k −43.8 vs
   −21.3), and on MISS rows `j0_top12` collapses to 5–20 % — the gold chunk is
   not in the pack, so no read can recover it. On RULER (recall ≈ 100 %) the
   retrieval step is **+2.0 ns** on the 99-sample HIT subset and −100 on the
   single MISS sample.
2. **Recall does not explain the BABILong gap by itself.** The retrieval step
   stays significantly negative on the *HIT* subset too (qa1×16k −36.7,
   qa2×16k −21.3). So "gold chunk ∈ pack" is **not** sufficient: dropping from
   ~30–60 context chunks to 12 also removes *distractor/aggregation* context that
   BABILong's multi-fact tasks use, and my strict all-in-pack gold locator is an
   **upper bound** on true support recall for qa2/qa5 (see §7.2). Attributing the
   BABILong change to "retrieval" therefore means **"the retrieval/pack-narrowing
   axis"**, not narrowly "the gold chunk was missing".

## 6. What this means for A02's thesis

* **The mid-layer read is real but small, and only RULER ever measured it.**
  With retrieval held byte-identical, the deployed CoMem read costs **−10.0 pp
  (CI [−16, −5]) at niah_mk1×16k, −12.0 (CI [−19, −6]) and −11.0 (CI [−18, −5])
  on variable_tracking**, and −3.0 (ns) at niah_mk1×32k. So the read is **not
  free**, and A02's thesis is **not rescued** — but the honest magnitude is
  ~3–12 pp, not the 35–55 pp phase-1 advertised.
* **The mid-depth read without its adapter is catastrophic, and the adapter
  recovers almost all of it.** `j12_frozen` scores **0.0–3.0 %** on RULER while
  `c2_comem` scores **88–96 %**: `read_depth` −97 to −100 pp, `read_lora` +87 to
  +95 pp. Resuming at layer 12 on chunk-local hiddens is non-functional untrained;
  essentially the entire capability is supplied by the distilled Read-LoRA. This
  is a **new, clean, previously unmeasured** result and the strongest positive
  finding in this gate — but it is a statement about *the adapter*, not about
  memory being cheap.
* **The 66.5× / N\*≈0.13 cost story is now fully explained as a baseline
  artifact.** The cost gate found 93 % of the apparent speedup was retrieval;
  this gate finds 54.9–78.6 % of the apparent *quality loss* on BABILong is the
  same axis. Both directions of phase-1's headline were dominated by
  `c1_pack_all` being an unusually strong-on-quality / catastrophic-on-cost
  baseline.
* **Consequence**: A02 still does **not** clear promotion. Nothing here revives
  the storage form (dead: 2048× raw text) or makes the read-compute win large
  (1.03–1.37×). The gate's value is diagnostic: **phase-1's BABILong evidence
  should not be cited as evidence about mid-layer reading.**

## 7. Confounds and UNVERIFIED items

1. **CONFOUND I CHOSE TO KEEP — and it is the one I could not remove:**
   `j12_frozen` runs the j=12 read **without** the adapter that was distilled
   *for* j=12, so it is a **lower bound** on depth-12 capability, not a fair
   depth arm. Conversely `read_deployed` (j0_top12→c2_comem) bundles **depth +
   adapter** together. I report **both**, and I state plainly: **there is no arm
   here that isolates depth with a matched-quality adapter**, because training a
   j=0 "control LoRA" was out of scope for an eval-only gate. The `read_depth`
   −97 pp on RULER must therefore **not** be read as "depth costs 97 pp"; it means
   "untrained depth-12 resume is non-functional". The defensible depth-effect
   numbers are the `read_deployed` ones (−3 to −12 pp).
2. **UNVERIFIED (upper bound): the gold-support locator for qa2/qa5.** It matches
   the *answer* string, which for qa1 faithfully marks the supporting fact, but
   for two-fact qa2 / three-argument qa5 marks only the **answer-bearing** fact,
   not the full support chain. **True recall on qa2/qa5 is therefore ≤ the
   reported value**, which makes the retrieval attribution on those cells
   *conservative* (real recall is worse) but means the absolute qa2/qa5 recall
   percentages should not be quoted as exact.
3. **UNVERIFIED: variable_tracking recall is not measured at all** (`n/a`,
   0 gold-locatable). `_locate_needle_chunks` could not localise a VT chain, so
   its "retrieval ≈ 0" claim rests only on the *accuracy* step (`j0_top12` 100.0
   vs `c1_pack_all` 100.0/99.0), not on a direct recall number.
4. **Scope**: 10 cells only (BABILong qa1/qa2/qa5 × {16k,32k}; RULER
   niah_multikey_1/variable_tracking × {16k,32k}), n=100 per cell, one seed
   (42), single model (Qwen3-8B), H20/bf16/sdpa. **4k was not re-run** — phase-1
   showed those cells n.s., so they are uninformative for this question, but that
   means the length trend rests on two points per task.
5. **Not re-run**: LongBench / LoCoMo / LongEval. Phase-1 called them TIEs, so
   they cannot discriminate depth from retrieval. LongEval's 8k→128k sign flip
   remains unexplained by this gate.
6. **Not tested**: whether a better selector (higher topk, oracle, or a semantic
   retriever) would close the BABILong gap. That is the obvious follow-up and it
   is **not CoMem-specific** — it would help text-RAG identically, exactly as the
   cost gate found for sublinear retrieval.
7. **`read_lora` on BABILong qa2 is n.s.** (+4.0 and −1.0), unlike RULER's +87 to
   +95. The adapter's benefit is thus **task-dependent** and not established on
   multi-fact bAbI tasks.

## 8. Integrity gates (all enforced fail-closed, all passed)

| gate | result |
|---|---|
| G0 Read-LoRA sha == flagship `dd09cd17…` | PASS (driver aborts otherwise) |
| G1 shard completeness, every cell every arm | PASS, 8/8 all cells; **0 refused** |
| G1 negative test (delete a shard) | **FIRES** → `G1_SHARD_INCOMPLETE 7/8` |
| G2 BABILong pairing (question,target) identical across arms | PASS |
| G2 RULER `input_ids_sha256` identical across all 4 arms + regeneration | PASS, **0 failures** |
| G3 config identity (selector/topk/chunk/depth/LoRA per arm) | PASS, **0 config errors** |
| `chat_template=False` asserted in every cell config | PASS |
| partition identity `RETR+DEPLOY == total` | PASS to 1e-6, 10/10 cells |
| partition identity `depth+lora == DEPLOY` | PASS to 1e-6, 10/10 cells |
| canonical scorers imported, never reimplemented | BABILong `compare_answers`; RULER harness per-item `correct` |
| evidence identical on both disks | md5 verified wzc1 == zwfy6 |

G1 was **negative-tested**, per the A04 precedent: with 8/8 shards the loader
returns 100 items and no error; after deleting one shard it refuses with
`G1_SHARD_INCOMPLETE 7/8` instead of silently merging 7/8.

**Aggregation hygiene**: no pooled BABILong or pooled LongEval figure is computed
or quoted anywhere in this gate. The evidence JSON contains per-cell blocks only.
The banned −17.89 pp / +2.00 pp pooled values are absent by construction.

**Operational note**: the run used a bounded 4-GPU pool and left GPUs 4–7 free
throughout; the A03 watcher (PID 2165190) was alive before, during and after, and
all 8 GPUs were released on completion.

---

## 9. Verdict

**Which cause dominates: it depends on the benchmark, and that is the finding.**

* **BABILong qa1/qa2 → RETRIEVAL dominates** (54.9–78.6 % of the change; recall@12
  only 22.9–63.2 %). Phase-1's four "decisive C1 wins" are **mostly a top-12
  `iter_bm25` recall/pack-narrowing artifact**, not evidence about mid-layer
  reading. **Phase-1's verdict must be restated for these cells.** (qa1×32k is
  the least clean: retrieval leads but only −28 vs −23.)
* **RULER → READ dominates** (retrieval ≈ 0, recall 99–100 %). Here A02's thesis
  *was* genuinely tested, and the deployed read costs a significant but modest
  **−3 to −12 pp**.
* **BABILong qa5 → neither** (both steps n.s.), so the phase-1 "C2 wins qa5"
  sign-flip is not attributable by this decomposition.

**Net effect on A02**: the kill clause's quality finding **survives in weakened,
narrowed form** — CoMem's read does cost accuracy where retrieval is not the
bottleneck (RULER, −3 to −12 pp) — but the **headline magnitude was inflated
2–5× by a retrieval confound**. Combined with the cost gate (storage DEAD;
read-compute 1.03–1.37×), A02 remains **not promotable**, and the recommended
positioning is unchanged. The one genuinely new positive result is that
**mid-depth resume is entirely adapter-dependent** (`j12_frozen` 0–3 % → `c2_comem`
88–96 % on RULER), which is a fact about distillation, not about memory
efficiency.
