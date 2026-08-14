# B08 leg-1 gate — PRE-REGISTRATION (written 2026-08-14, 0 GPU, PRE-DATA)

This document is written **before any B08 arm has been run**. Nothing in the repo
contains a B08 answer-quality number. Every threshold below is fixed here.

---

## 0. What this document does

`STATUS.json` previously carried `next_gate = "NOT_SPECIFIED"` because B08 is a
three-leg portfolio and nobody had picked which leg to gate. This picks one, and
turns the proposal's prose kill clause into an arithmetic rule.

It also records the **portfolio-level** rule that was missing: as written, B08's
three kill clauses were disjoint, so no single experiment could ever kill B08 —
only narrow it. §6 fixes that.

---

## 1. Leg selection, and why the other two are not gated first

| leg | decision | reason |
|---|---|---|
| **1. query-conditioned notes + raw evidence** | **GATED FIRST** | Its premise is already measured true (§2), it has a retrieval-closed cell (§3), and it costs ~1.1 GPU-h (§7). |
| 2. typed personal-memory ledger | **folded into leg 1's cell, gated only if leg 1 passes** | Its whole task list (overwrite / stale / contradiction / LongMemEval update) *is* the `knowledge-update` cell. Gating it separately would re-run the same 78 items. |
| 3. multi-tier pyramid memory | **NOT gated; premise already measured against** | Its kill clause is "far-memory read cost swallows the fixed Read advantage". A02 already measured that advantage at **1.03–1.37x**, and break-even reuse `N*` growing **8 → 25 → 186** as the corpus goes 32k → 128k → 1M, i.e. the economics get *worse* exactly where a pyramid is pitched. Resurrecting leg 3 requires new evidence about *that* premise, not a new arm. |

Leg 3 is **not** killed by this document — no B08 experiment has run. It is
*deprioritised on measured adjacent evidence*, which is a scheduling statement.

---

## 2. The premise is measured, not assumed (0 GPU, already on disk)

Leg 1's motivation is "first-stage retrieval is already near-saturated, so stop
optimising the reranker". That is **confirmed** on LongMemEval-S:

`evidence/b08_lme_bm25_recall_topk{5,10,20,50}.json` (BM25, `--reader stub`,
`degraded: false`, all 500 questions):

| top_k | overall any-hit | mean gold coverage |
|---|---|---|
| 5 | 94.6% | 84.9% |
| **10** | **96.8%** | **91.4%** |
| 20 | 98.2% | 94.7% |
| 50 | 99.6% | 98.2% |

**Consequence, and it is a constraint not a win:** with ~3.2pp of retrieval
headroom left at top_k=10, leg 1 can no longer be framed as a retrieval
contribution. Any surviving claim must be **reader-side**. A reranker/RRF
improvement is explicitly *not* a B08 result (it would help plain text-RAG
identically — A02's selector-ceiling finding).

---

## 3. The cell: a retrieval-CLOSED stratum, so recall is a constant

From the same measured table, at `top_k=10` exactly two question types have
`any_hit_recall == 1.000`:

| type | n | any_hit @k=10 | coverage @k=10 |
|---|---|---|---|
| `knowledge-update` | 78 | 1.0000 | 0.99359 |
| `single-session-assistant` | 56 | 1.0000 | 1.00000 |
| **stratum** | **134** | **1.0000** | — |

`knowledge-update` is already 1.000 at `top_k=5` too.

**Why this matters (the A02 re-attribution guard).** A02's phase-1 BABILong
quality evidence was re-attributed: 54.9–78.6% of the C1-vs-C2 change turned out
to be the *retrieval* axis, not the mid-layer read. Any "notes helped" claim is
exposed to exactly that. Here retrieval is **frozen at BM25 top_k=10, identical
inputs for every arm**, on a stratum where any-hit is 1.000 — so recall is a
**constant, not a covariate**, and a difference between arms cannot be
re-attributed to retrieval.

The de-saturated cell (`single-session-preference`, 76.7% @k=10) is the
*complement* and is deliberately **excluded** from the gate: it is where
retrieval still moves, so it cannot isolate a reader-side effect.

---

## 4. Arms — single variable is context composition

Same reader, same weights, same retrieved evidence set, same notes text, same
decode settings. **The only thing that varies is what occupies the reader's
context.**

| arm | context | status |
|---|---|---|
| `A-raw` | top-10 raw evidence | runnable today (`--compressor none`) |
| `A-notes+raw` | notes block **prepended**, raw kept | runnable today (harness does exactly this, `longmemeval/run_baseline.py:152-162`) |
| `A-notes-only` | notes block **only**, raw withheld | **NOT RUNNABLE — must be coded** (§8) |

Fixed for all arms: `--retriever bm25 --reranker none --top_k 10
--evidence_token_budget 4000 --reader local_hf`, `max_new_tokens=64`, greedy
(`do_sample=False, num_beams=1`), `chat_template=False` (satisfied as-is:
`LocalHFReader._build_prompt` builds a direct-completion base-model prompt,
`longmemeval/reader.py:229-254`).

### 4.1 The notes generator must be the reader's own model, NOT mem_space

`--compressor mom_notes` **cannot be run.** `MoMNotesCompressor` requires a
mem_space `MemorySpaceConfig` adapter config + a `.pt` checkpoint
(`longmemeval/compressor.py:150-167`). Verified on wzc1: **45** `adapter_config.json`
files exist under `outputs/`, and **zero** contain `num_slots` or `slot_dim` —
they are all PEFT-LoRA configs (e.g.
`outputs/qcmem_distill_llama3_j12_r32_4k/final/adapter_config.json` is
`peft_type: LORA`). There is no mem_space adapter config anywhere on wzc1.

Independently of availability, mem_space would be the **wrong** generator: it
introduces a second model, so "notes help" would be confounded with "the
mem_space model is good". The gate therefore uses **self-notes**: the *same*
`Meta-Llama-3-8B` that reads also writes the notes, from the *same* retrieved
evidence. Then the single variable really is context composition.

---

## 5. Read-out, metrics, and the KILL rule

### 5.1 Pre-registered read-out point

There is no training and no step axis. The read-out point is:

> **all 134 items × 3 arms complete, and `n_scored == 134` asserted per arm
> before any metric is computed.**

No metric may be computed on a partial arm. (Precedent: a silent 5/8-shard merge
once corrupted a whole protocol; the fix is asserting `n_scored == expected` per
cell, not just checking for NaNs.)

### 5.2 Metrics

- **ACC** — judge-correct rate via `scripts/a02_judge_openweight.py`
  (Qwen3-8B, `temperature=0`, `enable_thinking=False`, `max_new_tokens=8`,
  `judge_cache.jsonl` + `judge_meta.json`). NOT the official GPT-4o judge —
  `longmemeval/scoring.py:6-9` deliberately does not bundle it.
  Denominator: **128** non-abstention items (134 − 6 `_abs`).
- **Abstention items** (6 in `knowledge-update`, 0 in `single-session-assistant`)
  are scored by the abstention rule (correct iff the model refuses), never by
  substring/F1. Mis-scoring these corrupts both ACC and U.
- **U — unsupported-claim rate.** Fraction of non-abstention answers containing a
  factual claim not present in **that arm's own context**. Denominator **128**.
  This operationalises the proposal's non-negotiable invariant
  ("必须测 notes faithfulness"). **This scorer does not exist yet** (§8).

Statistics: paired bootstrap, 10,000 resamples, seed 42, reusing the
`scripts/analyze_p019_recall_readout.py` scaffold. ⚠️ Bootstrap must run on **one
node**: the five nodes carry three numpy versions (LOCAL 2.3.5 / .82 2.4.6 /
rest 2.5.1) and same-seed `multinomial` differs, so cross-node resampling is not
reproducible.

### 5.3 Reporting rule (A02's pooling lesson)

`knowledge-update` and `single-session-assistant` are reported **separately,
always**. The stratum-level n=134 number is reportable **only if the two
per-type point estimates have the same sign**. A02's -17.89pp pooled BABILong
headline was a mean over 9 cells with opposite true signs (4 significantly
negative, 2 significantly **positive**) — "pooling summarises nothing".

### 5.4 Threshold derivation (why 10.82 pp, not a round number)

For a paired binary comparison, minimum detectable effect at 80% power,
two-sided α=0.05:

```
MDE = (z_0.975 + z_0.80) * sqrt(disc / n) * 100     # disc = P(discordant pair)
    = (1.95996 + 0.84162) * sqrt(0.20 / 134) * 100
    = 10.82 pp                                       # n = 134, disc = 0.20
```

Measured MDE grid (computed this session):

| n | disc=0.10 | disc=0.20 | disc=0.30 |
|---|---|---|---|
| 78 (KU only) | 10.03 | 14.19 | 17.37 |
| **134 (stratum)** | 7.65 | **10.82** | 13.26 |
| 500 (all types — NOT retrieval-closed) | 3.96 | 5.60 | 6.86 |

This is why the stratum (n=134), not KU alone (n=78), is leg 1's primary cell:
KU alone cannot resolve anything under ~14pp. **The observed discordance will be
reported and the MDE recomputed from it**; the *formula* is pre-registered here,
not just the single number.

### 5.5 KILL rule

Define, on the retrieval-closed stratum, all paired, all in absolute pp:

- `Δ_aug  = ACC(A-notes+raw)  − ACC(A-raw)`
- `Δ_sub  = ACC(A-notes-only) − ACC(A-raw)`
- `ΔU     = U(A-notes-only)   − U(A-raw)`

> **KILL leg 1 iff ALL THREE hold:**
> - **K1** `Δ_aug` 95% paired-bootstrap CI **contains 0**
>   → adding notes to raw gives no benefit this design can resolve (< 10.82 pp).
> - **K2** `ΔU` 95% CI **upper bound < +5.0 pp**
>   → notes-only does **not** hallucinate measurably more, so the proposal's
>     faithfulness invariant has nothing to measure.
> - **K3** `Δ_sub` 95% CI is **NOT entirely above −2.0 pp**
>   → notes-only is also not near-parity, so there is no compression story either.

All three ⇒ notes merely *omit* facts without inventing them, and adding them
changes nothing resolvable. There is no paper in that. Archive leg 1.

### 5.6 Survival branches (each is a different, narrower claim)

| branch | surviving claim |
|---|---|
| `Δ_aug` CI entirely **above 0** | Notes+raw beats raw **with retrieval pinned at 1.000** — the architecture is prior art (`longmemeval/compressor.py:8` credits it as the LongMemEval-V2 winning pattern), so the contribution is the retrieval-closed *isolation*, never the architecture. |
| `ΔU` CI entirely **above +5.0 pp** | "Notes are an adjunct, never a substitute" — **measured**, not asserted. This is leg 1's actual novelty. |
| `Δ_sub` CI entirely **above −2.0 pp** | ~30x context reduction at no accuracy cost on a retrieval-closed cell. **Do not fold into B08** — route to a new proposal with its own gate. |

### 5.7 Falsifiability check — a concrete result that KILLS

Fully plausible, and it fires the kill:

```
ACC:  A-raw 61.9% (83/134)   A-notes+raw 64.9% (87/134)   A-notes-only 53.7% (72/134)
U  :  A-raw  8.2%            A-notes-only 8.9%

Δ_aug = +3.0 pp, CI [-4.5, +10.4]  -> contains 0            -> K1 fires
ΔU    = +0.7 pp, CI [-2.6,  +4.1]  -> UB 4.1 < +5.0         -> K2 fires
Δ_sub = -8.2 pp, CI [-15.1, -1.8]  -> not entirely > -2.0   -> K3 fires
=> KILL
```

And a result that passes: `Δ_aug = +12.0 pp, CI [+3.1, +20.6]` (K1 does not
fire) — or `ΔU = +11.4 pp, CI [+6.2, +16.9]` (K2 does not fire, faithfulness
claim survives). The gate is two-sided.

### 5.8 Denominator guard (ratios are ill-defined here)

All primary read-outs are **absolute pp differences**, so no denominator exists
and no guard is needed for them.

If anyone later reports a *fraction of headroom recovered*, the denominator is
`(ceiling − ACC(A-raw))` with `ceiling = 100%` (justified: measured retrieval
`any_hit = 1.000` on this stratum). That ratio is **FORBIDDEN unless
`ACC(A-raw) ≤ 95.0%`, i.e. denominator ≥ 5.0 pp**, and the guard must be
committed before scoring. Rationale: as the denominator → 0 the ratio explodes
and flips sign on noise; a range is not a measurement until it clears its floor.

---

## 6. Portfolio-level kill rule (the missing structural piece)

The three original clauses were disjoint, so B08 could only ever be narrowed.
Fix:

> **B08 is ARCHIVED as a portfolio iff the §5.5 gate kills, in the same run.**

That single gate reaches all three legs:

- **leg 1** — killed directly by §5.5.
- **leg 2** — its entire task surface (overwrite / stale / contradiction /
  supersession) **is** the `knowledge-update` cell. If context composition does
  not move ACC or U on the cell where facts are *superseded by construction*,
  then a typed ledger — whose only lever is *marking* that supersession in the
  context — has no surface left. Leg 2's own clause ("typed ledger does not
  reduce stale/conflict error") is evaluated on that same n=78.
- **leg 3** — premise already measured against (§1); nothing in this run
  resurrects it.

So one ~1.1 GPU-h experiment can kill B08 outright. It is no longer unkillable.

---

## 7. Cost — measured anchor, no 1-cell timing run needed

**Anchor (measured, someone else's real run):**
`proposal/backlog/A02-comem-write-read-repair/evidence/a02_storage_readcompute_verdict.json`,
cell `cost/32k|gpu → per_G/32 → latency_decomposition/comem`:

```
read_s   = 0.6774476431310177 s  for read_len = 6177 tokens  -> 0.1097 ms / prefill-token
decode_s = 1.2638465277850628 s  for 32 new tokens           -> 39.50 ms / decode-token
```

Provenance: raw per-proc file
`evidence/a02_storage_readcompute_serve/a02_serve_32k_gpu_niah_multikey_1_proc0.json`
(`dtype: bfloat16`, `attn_impl: sdpa`, `num_layers: 36`, `hidden: 4096`,
`n_repeat: 5`, `warmup: 2`, `use_chat_template: False`). Launcher
`code/run_a02_storage_readcompute.sh` defaults
`PROJECT_ROOT=/apdcephfs_zwfy6/...` + `/opt/conda/envs/torch-base/bin/python`
and its header says "run on .82" ⇒ **H20 / sm_90**.

**Measured token counts (mine, this session, 0 GPU — real tokenizer, the exact
`LocalHFReader` prompt string, real BM25 retrieval, real 4000-token budget):**

| stratum | n | mean prefill | p90 | max | **Σ prefill tokens** |
|---|---|---|---|---|---|
| KU + SSA, budget=4000 | 134 | 3952.4 | 4387 | 4609 | **529,620** |
| KU + SSA, budget=0 | 134 | 4960.5 | 6249 | 8429 | 664,711 |
| KU only, budget=4000 | 78 | 4086.0 | 4386 | 4609 | 318,706 |

**Arithmetic:**

```
per answer arm : 529620 * 0.1097ms            =  58.1 s prefill
                 134 * 64 new * 39.50ms       = 338.7 s decode
                                              = 396.8 s = 0.110 GPU-h  (2.96 s/question)
notes-gen pass : 58.1 s + 134*128*39.50ms     = 735.5 s = 0.204 GPU-h
3 answer arms + 1 notes pass                  = 1925.9 s = 0.535 GPU-h
x2.0 slack (model load, tokenizer, p90 tail)  = 1.07 GPU-h
+ judge (134*3 items, ~600-tok prompt, 8 new) = 0.043 GPU-h
--------------------------------------------------------------------
TOTAL ~= 1.1 GPU-h on ONE card   (budget ceiling: 2 GPU-h)
```

No 1-cell timing run is required: the s/token anchor is measured, and the token
counts are measured.

**Node requirement — sm_90 (.73 / .82 / .104) preferred, and why:**

1. The anchor was measured on sm_90. Running on sm_100 would still give valid
   *quality* numbers but would make the 1.1 GPU-h figure an extrapolation across
   architectures rather than a validated one.
2. The judge protocol (`scripts/a02_judge_openweight.py`) was established on .82.
3. This is a **first** run — there is no archived B08 quality result to be
   same-arch comparable to. Within-run comparability is what matters, and all 3
   arms run sequentially in one process on one card, so it holds by construction.

Not an sm_100 task: nothing here needs B200 memory or Blackwell. One card,
~1 hour.

---

## 8. What still blocks GPU (honest list)

Writing this gate does **not** make B08 `ready_gpu`. Remaining, in dependency
order — all CPU:

1. **`RELATED_WORK.md` does not exist.** `proposal/README.md`: Related Work must
   be written before new GPU. `novelty_checked: false` + no file pins B08 at
   `ready_cpu` by construction. Must be leg-1-only, and must concede upfront that
   notes+raw is credited as the LongMemEval-V2 winning pattern in our own
   `longmemeval/compressor.py:8` — so the differentiator is the **faithfulness
   measurement**, not the architecture. (`RELATED_WORK_GAP_AUDIT_20260808.md:98`
   rates B08 `严重不足`, the worst of any proposal, and requires splitting the
   three legs; :146-152 warns a feature-list claim will not survive. That is a
   **writing** constraint — under project rule, only B08's own experiment gate
   may kill it, and the bar is "completely identical", not "overlap".)
2. **`A-notes-only` arm does not exist.** `run_baseline.py:162` hardcodes
   `reader_evidence = [notes_block] + list(evidence)` — no way to withhold raw.
3. **The `U` unsupported-claim scorer does not exist.** `longmemeval/scoring.py`
   emits only a submission JSONL + recall@k. This is the primary novelty metric.
4. **A self-notes compressor does not exist.** `--compressor mom_notes` is
   unrunnable (no mem_space adapter config on wzc1, §4.1); needs a
   `SelfNotesCompressor` using the reader's own Llama-3-8B.
5. **Judge input adapter.** `a02_judge_openweight.py` takes `--result_dirs` of
   LoCoMo-shaped dirs; longmemeval emits `{question_id, hypothesis}` JSONL.
6. **zwfy6 asset verification.** `/apdcephfs_zwfy6` is not mounted on this node
   and ssh was barred, so every presence claim here is **wzc1-scoped**. Before
   booking an sm_90 card, confirm on the target node: `models/Meta-Llama-3-8B`
   (15G on wzc1), `data/longmemeval/longmemeval_s.json` (278 MB on wzc1), the
   Qwen3-8B judge weights, and whether any longmemeval results already exist.
   Cross-disk moves are `scp -O` + checksum.

**Also on the record (not a blocker, a provenance warning):** three of the six
code assets in `SOURCES.md` — `src/memory/l2/`, `src/memory/l3/`,
`src/agents/memory_agent.py` — are byte-identical to copies archived as **dead**
in `legacy/src_dead_subsystems/` (commit `b63b5a1`) and are **untracked** in git.
The typed-ledger / pyramid code B08 cites is a restored working-tree copy of code
this project already abandoned. The one exception is `src/eval/update_eval.py`
(git-tracked; its overwrite / stale / contradiction / temporary metric set is
genuinely reusable for leg 2).

---

## 9. Deviations from this pre-registration

Any change to §3 (cell), §4 (arms), §5.2 (metrics), §5.5 (kill rule), or §5.1
(read-out point) after the first arm runs must be appended below with a
timestamp and a reason. Silent edits invalidate the gate.

*(none yet — written pre-data)*
