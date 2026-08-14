# B02 — KILL GATE (v1.0) and its verdict

**Status: the kill gate below is DEFINED, and it has ALREADY FIRED on both lengths.**
Written 2026-08-14, 0 GPU. The GPU that produced the inputs was spent by the confirmatory
run pre-registered in `FIXED_SAMPLE_PROTOCOL.md` §6b (6.438 GPU-h measured, see §4).

---

## 1. Why the kill gate needed writing at all

`STATUS.json` had **no `kill_gate` field**, which is why `proposal/ready_queue.py` classified B02
`ready_cpu` ("kill gate undefined → writing it is 0 GPU and blocking"). `PROPOSAL.md` Stage 0 does
contain a stopping intent — *"若 oracle 相对最佳 fixed config 的收益不足，方向关闭"* — but as
`FIXED_SAMPLE_PROTOCOL.md` §5 already established, **that sentence is not decidable as written**: with
`C` configs and noisy per-item outcomes, `oracle − best_fixed` is positive by construction even with
zero exploitable structure, so "收益不足" has no threshold that isn't arbitrary.

The v0.1 protocol fixed that by gating on excess-over-a-null. Then the pilot **falsified the protocol's
own primary null** (§5b): for a **binary** oracle, `max_j M[i,j] = 1[rowsum_i ≥ 1]`, so the oracle is a
function of the row margins alone, and curveball preserves row margins exactly ⇒ Null B is **provably
invariant** (`sd = 2.22e-16`, `p = 1.0` on every dataset, forever).

So this file's job is to state a kill gate that (a) is decidable, (b) **cannot** degenerate the way
Null B did, and (c) does not let a max-over-noise artefact be reported as headroom.

---

## 2. The kill gate

> **B02-KILL-1 (adopted 2026-08-14, `n = 200`, VT × {16k, 32k}, Qwen3-32B, `chat=False`,
> `selector=iter_bm25`).** B02's premise is that *the best configuration varies per query*, so that a
> query-adaptive controller can beat the best single fixed configuration. The direction is **killed**
> unless **both** clauses pass, each on **each length independently** (no pooling across lengths):
>
> **Clause 1 — complementarity must clear its own independence floor.**
> On the **fractional-recall** scale, with `Δ_excess = oracle_obs − mean(oracle under Null A)`
> where **Null A** independently permutes each config's outcome column across items
> (`B = 10000` draws, seeded `PCG64`, single node, `numpy.__version__` recorded):
> `Δ_excess > 0` **and** its 95 % permutation interval **excludes 0**.
> *Interpretation of failure:* `Δ_excess ≤ 0` means configs are **positively coupled** — hard items are
> hard at every depth — which is strictly worse for a router than no signal at all.
>
> **Clause 2 — the headroom must be REACHABLE, not merely present.**
> A router restricted to **pre-answer** features (i.e. features observable before the answer is
> scored: input length, evidence count, retrieval-score statistics) must beat the best fixed
> configuration **out-of-sample**, where the comparator's fixed configuration is **also** selected on
> the training half. Pass requires the 95 % interval of the held-out gain to **exclude 0**.
> *Rationale:* the oracle is an upper bound that no deployable router observes. Clause 1 alone can pass
> while every realizable controller loses; certifying opportunity is not certifying realizability.
>
> **Both clauses fail ⇒ B02 is killed as a router direction.** A negative Clause 1 is a
> *stronger* kill than an inconclusive one: it says the action space is redundant, not just noisy.
>
> **Mandatory wording on failure (pre-committed).** `n = 200` resolves `|Δ_excess| ≥ 0.033` at 80 %
> power. If an interval contains 0 the write-up must say **"no effect larger than 3.3 pp"**, never
> "no effect". If an interval excludes 0 in the **negative** direction — what actually happened — the
> finding is positive and may be stated as such.

### 2.1 Why this gate cannot degenerate the way Null B did

Null B's failure was **not** a sampling accident; it was an invariance. So the replacement must be
checked against that same failure mode structurally, not empirically. Two independent arguments:

**(a) The statistic is not a function of the preserved margins.** Null A preserves **column** margins
(each `p_j`) and destroys the item×config coupling. The oracle `mean_i max_j M[i,j]` is a function of
the **rows**. Null A does not preserve rows: permuting each column independently changes which values
co-occur in a row, hence changes `max_j` for individual rows. There is therefore no invariance
argument available, and the null has genuine variance. Measured on the actual confirmatory matrices:

| | Null A `sd` | Null B `sd` |
|---|---|---|
| VT 16k fractional | **0.01039** | 5.55e-17 (float zero) |
| VT 32k fractional | **0.00897** | 5.55e-17 (float zero) |

**(b) Moving to the fractional scale removes the binary identity that caused the degeneracy.** The
degeneracy proof needed `max_j` to be determined by the row sum. That holds for `{0,1}` entries only.
Counterexample on `[0,1]`-valued recall, two configs:

```
row (0.50, 0.00): rowsum = 0.50, oracle = 0.50
row (0.25, 0.25): rowsum = 0.50, oracle = 0.25     <- same row margin, different oracle
```

So on the fractional scale the oracle is **not** a function of row margins, and *even a
row-margin-preserving null would not be invariant*. The gate does not depend on this — Null A is
primary and is column-preserving — but it means the fractional scale is robust to the entire class of
failure that killed Null B, rather than merely avoiding one instance of it. This is also why the binary
scale is reported as **secondary**: at `n = 200`, 3 of 8 binary marginals are exactly 0
(`[.035, .005, .000, .050, .275, .085, .000, .000]`), i.e. binarising discards most of what VT's
`string_match_all` actually measured.

**(c) Clause 2 cannot degenerate to a tautology either.** The comparator is re-selected on each
training split, so the gate is not "router beats a comparator tuned on the test set". This is exactly
the selection-validity flaw named by arXiv:2608.08265 (§5), and avoiding it is why Clause 2 is stated
with train-selected comparators rather than the global best fixed `j`.

---

## 3. Verdict: FIRED, on both lengths, both clauses

### Clause 1 — failed, and failed *negatively* (`p ≤ 0.0008`)

`evidence/b02_confirmatory_vt16k_n200.json`, `evidence/b02_confirmatory_vt32k_n200.json`
(produced by the pre-registered `analyze_b02_oracle.py`; independently recomputed from the raw
`*.records.json` by a separate script, agreeing to all printed digits).

| length | scale | `oracle_obs` | `best_fixed` | raw headroom | Null-A mean | **`Δ_excess`** | CI95 | `p` |
|---|---|---:|---:|---:|---:|---:|---|---:|
| **16k** | fractional | 0.5910 | 0.3420 | +0.2490 | 0.6300 | **−0.0390** | [−0.0590, −0.0190] | **0.0008** |
| **16k** | binary | 0.3600 | 0.2750 | +0.0850 | 0.3949 | −0.0349 | [−0.0600, −0.0100] | 0.0152 |
| **32k** | fractional | 0.5430 | 0.3310 | +0.2120 | 0.5745 | **−0.0315** | [−0.0480, −0.0140] | **0.0008** |
| **32k** | binary | 0.3000 | 0.2350 | +0.0650 | 0.3263 | −0.0263 | [−0.0450, −0.0050] | 0.0224 |

All four intervals exclude 0 **on the negative side**. The raw headroom looks large (+21 to +25 pp) and
is **entirely below its own independence floor**. Reporting that +24.9 pp as router headroom would have
been precisely the error the protocol was written to prevent.

**32k is an independent replication.** It is a different length, scored after the 16k leg, with the
same sign, the same magnitude and the same `p`. The kill does not rest on one cell.

### Clause 2 — failed: no realizable router beats a train-selected fixed `j`

`evidence/b02_realizability_leg.json` (0 GPU). 300 random 50/50 splits; per-bucket best `j` learned on
train, applied to held-out test; comparator fixed `j` also train-selected.

| length | router | held-out gain | CI95 | beats fixed? |
|---|---|---:|---|---|
| 16k | length, 2 buckets | −0.0143 | [−0.0720, +0.0000] | **no** |
| 16k | length, 4 buckets | −0.0250 | [−0.1121, +0.0020] | **no** |
| 32k | length, 2 buckets | −0.0075 | [−0.0840, +0.0000] | **no** |
| 32k | length, 4 buckets | −0.0140 | [−0.0860, +0.0080] | **no** |

**And the input features themselves are degenerate on this task** — a finding that is arguably more
decisive than the intervals, because it is structural rather than statistical:

- **`document length`**: RULER pads every item to the target length. Measured span across 200 items is
  **10 tokens out of 32 713–32 723** (`sd = 1.96`). A `PROPOSAL.md` input feature with no variance.
- **`estimated evidence count`**: **constant at 9** for every item in `variable_tracking`. A router
  keyed on it is *identically* the best fixed config.

`PROPOSAL.md` lists seven input features; on the task B02's own sweep uses, two are constants and the
retrieval-side ones (`BM25/dense score gap`, `retrieval entropy`) are not emitted per item by the
harness, so they were not testable without new instrumentation. This means Clause 2 could not have been
passed on this task by *any* amount of GPU, and B02's Stage 0 as written could never have been decided
on `variable_tracking` alone.

---

## 4. What survives, and the cost that bought it

**Surviving positive result (the reason this is a finding and not just a dead end):**
the depth curve's **shape and peak are stable across lengths** — Spearman over the 8 `j` values between
16k and 32k mean recall is **+0.976**, and `j = 27` is the best arm at **both** lengths. Combined with
Clause 1's negative `Δ_excess`, the reading is:

> On this task, **one well-chosen fixed split depth is the right answer**, and it transfers across
> context length. Per-query depth adaptation has no complementarity left to exploit — configs are
> positively coupled, so items that fail at the best depth fail at all depths.

**Measured cost (this is the `gpu_cost_estimate` basis).** From the 16 cell JSONs' `elapsed_seconds` on
`.73` (8×H20 sm_90, one cell per GPU, `oom_count = 0` in all 16):

| leg | cells | mean/cell | s/item | GPU-h |
|---|---:|---:|---:|---:|
| 16k, n=200 | 8 | 1340.0 s | 6.70 | 2.978 |
| 32k, n=200 | 8 | 1557.3 s | 7.79 | 3.461 |
| **total** | **16** | | | **6.438** |

Wall clock on 8 GPUs (2 cells/GPU) = **48.3 min**, against 48 min projected in §6b of the protocol —
the projection was accurate to ~1 %. Pilot adds 0.32 GPU-h. **Total B02 spend: 6.76 GPU-h.**

## 5. Prior art that this gate's design is answerable to

Full adjudication in `NOVELTY_B02.md`. The two that bear directly on the gate's *form*:

- **arXiv:2608.08265**, *Opportunity Is Not Realizability* (2026-08-08, **arXiv only**; DBLP total = 0,
  no OpenReview record) — names both flaws this gate guards against: testing against a best fixed model
  selected on the same examples invalidates paired inference, and a full-information oracle sees
  outcomes no deployable router observes. It reports oracle gaps of 9.7–30.7 points of which the best
  deployable router recovers only 7.5–14.4 %. **Concurrent (6 days)**, so per project rule it does not
  preempt; it is nonetheless the correct citation for Clause 2's construction, and B02's `n_tok`/`n_ref`
  degeneracy is a *sharper* negative than theirs (their realizable share is small; ours is zero because
  the features have no variance).
- **arXiv:2607.03436**, *How Much of the Routing Gap Is Real?* (2026-07-03, **DBLP: CoRR 2026,
  Informal and Other Publications**) — decomposes the router-to-oracle gap into reproducible headroom
  plus a single-draw selection floor, and shows 12–36 % of reported gap is label noise. Same disease,
  different medicine: their floor comes from **stochastic decoding** (they re-sample at `k ≥ 20`);
  B02's floor comes from **max-over-columns** at `k = 1` greedy decoding, so Null A is the right
  correction here and multi-sample oracles are not required.

Neither addresses **within-model split-depth** selection, which is B02's actual action space; the
oracle-routing literature is uniformly **between-model**. That gap is real but, given Clause 1 and 2
both failing negatively, it is a gap B02 no longer has a positive result to fill.
