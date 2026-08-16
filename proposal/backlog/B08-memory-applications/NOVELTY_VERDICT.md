# B08 — NOVELTY GATE ADJUDICATION (verdict document)

**STATUS: IN PROGRESS (being written 2026-08-16). 0 GPU, 0 ssh.**

This file is the *adjudication* that `proposal/ready_queue.py` looks for. `RELATED_WORK.md`
(written 2026-08-15, committed `463dca4`, re-verified in its own §11) is the *evidence*; this file
is the *verdict*. They were separated deliberately: the 08-15 agent explicitly declined to write a
clearance token into `STATUS.json` on the grounds that "an agent writing its own clearance field is
not the clearance being reviewed" (`RELATED_WORK.md` §10.1). This file is that review.

Sections are appended as each check completes. See §5 for the verdict line.

---

## 1. What this adjudication had to decide (and why it is not a re-run of 2026-08-15)

`RELATED_WORK.md` §9.6 named, in writing, the **one** question that could still collapse the narrow
claim, and left it open:

> ⚠️ **Every 2026 paper here was adjudicated from its ABSTRACT ONLY.** … The unresolved question
> that can still collapse leg 1 is whether **"Retain or Consolidate?"** (`arXiv:2607.17545`)
> **PINS RETRIEVAL**. If it does, leg 1's residual claim — which is precisely the retrieval-closed
> isolation — goes with it. Read its method + evaluation sections **BEFORE** spending the
> 1.1 GPU-h, not after.

`STATUS.json.related_work_20260815.highest_value_remaining_literature_task_0_GPU` repeats it as the
highest-value 0-GPU task outstanding. **This session read that full text.** The answer changes the
verdict, so the adjudication is a substantive act and not bookkeeping.

## 2. RESOLVED: `arXiv:2607.17545` DOES pin retrieval — differentiator (a) is FALSIFIED

Source: `https://arxiv.org/html/2607.17545v1` (HTTP 200, 347,897 B, fetched 2026-08-16 via
`hy-proxy.woa.com:3128`; 73,821 chars of extracted text). Verbatim, from *Full-History Retrieval
Stress Test → Controlled setting*:

> "Throughout, "retention," each operator, and the oracle gate receive the **same gold evidence per
> question**; the study measures the consolidation decision **in isolation**."

and from *Experiments → Setup*:

> "In the controlled evaluation, **every action receives the same gold evidence** and differs only
> in its budgeted representation. … **This pairing isolates the when–which decision from evidence
> discovery.**"

**ROC's headline protocol does not merely pin retrieval, it removes it**: gold evidence is supplied
directly, so recall is 1.000 *by construction* — a **stronger** closure than B08's, which is
1.000 *by measurement* on a BM25 top_k=10 stratum. ROC additionally runs a *second*, non-closed
protocol (BM25 top-20 under a 2,048-token cap) as an explicit "retrieval stress test", i.e. it
knowingly separates the closed and open regimes. It also reports the **knowledge-update question
type separately** (n=40) — the same LongMemEval type as 78 of B08's 134 items.

**Consequence, stated against our own record.** `RELATED_WORK.md` §3.3 and §8 rest leg 1's
differentiation partly on:

> "(a) it never pins retrieval at a **measured** `any_hit = 1.000`, so its deltas mix retrieval with
> composition — the exact confound B08's stratum was chosen to remove"

**That sentence is now false and must be struck.** ROC's deltas are *not* retrieval-confounded in
its controlled protocol. B08 may **no longer claim retrieval-closed isolation of the
raw-vs-consolidated comparison on LongMemEval as its contribution.** This is the collapse
`RELATED_WORK.md` §9.6 warned about, and it happened.

## 3. FOUR papers §1-§11 missed, two of them structurally closer than anything in §3

Details, verbatim quotes and venue evidence are in `RELATED_WORK.md` §12.2, §12.4, §12.5. Summary of
what each does to the claim:

| # | work | venue (verified) | what it takes from B08 | what it leaves |
|---|---|---|---|---|
| 1 | **WhenLoss** `arXiv:2605.24579` (2026-05-23) | DBLP **CoRR 2026** → arXiv-only | **The whole methodological framing.** Fixed reader + four controlled input conditions on LongMemEval, *built to separate the compression axis from the retrieval axis*, incl. an `OE (Oracle Evidence)` retrieval-closed condition. "We isolate the composition axis by pinning retrieval" is no longer available. | Its reader-independent metrics are **omission-only** (turn recall / span recall of *gold* content). `hallucinat`/`unsupported`/`fabricat`/`entail` = **0 hits in 58,176 chars**. No notes+raw arm (`alongside`/`hybrid`/`adjunct`/`supplement` = 0). |
| 2 | **Supersede** `arXiv:2606.27472` (2026-06-25) | DBLP **CoRR 2026** → arXiv-only | **The substitution-accuracy result on B08's own cell.** Bounded generated notes vs full context on the **knowledge-update subset of LongMemEval** — 78 of B08's 134 items — **92%→77%, paired McNemar p=0.0033**, plus scale and memory-size controls, plus an RL fix. B08's `Δ_sub` (notes-only vs raw accuracy) is essentially already answered, in the same direction, with a paired test. | `retriev` = **0 hits** — no retrieval stage at all; the contrast is notes vs *full context*, not notes vs a *retrieved* set. No faithfulness metric (`unsupported`/`fabricat`/`entail` = 0). |
| 3 | **LazyMem** `arXiv:2607.22690` (2026-07-17, "under review") | DBLP **CoRR 2026** → arXiv-only | **The mechanism**: query-time construction of query-conditioned notes from a broad retrieved pool, with a reward that explicitly rewards compressions "**faithful to the source**". | Faithfulness is a *training objective*, never a *measured read-out*; and it substitutes for raw rather than accompanying it. |
| 4 | **Fixed RAG Compression Collapses Measured Reader Scaling** `arXiv:2606.21807` (2026-06-20) | DBLP **CoRR 2026** → arXiv-only | Not novelty — **a design threat**: "generic summarization flips **31% of pairwise model rankings on LongMemEval-S**", and compression gain shrinks with reader strength (9/10 settings p<0.05, 20 readers). | Nothing. It must be *obeyed*, not differentiated: B08's gate uses **one** reader on **exactly this benchmark**. |

Also newly logged (`RELATED_WORK.md` §12.5): **`arXiv:2505.00019`** (CoRR 2025, arXiv-only) whose
stated analysis axes already include *"model hallucinations"* and *"word omission analysis"* across
six compressors × 13 datasets; and **MemFail `arXiv:2605.26667`** (CoRR 2026, arXiv-only), which
decomposes memory into summarization/storage/retrieval and attributes errors per operation.

**All four are CONCURRENT with the 2026-08-14 prereg (2026-05 … 2026-07) and none is peer-reviewed.**
Per `memory/prior-work-differentiate-dont-abandon.md` they **cannot preempt**, and none is
完全相同/抄袭. The bar is not met for FULLY_PREEMPTED — but two of the three differentiators the
proposal actually wrote down are gone.

## 4. Differentiator ledger — what is left, item by item

`RELATED_WORK.md` §8's residual sentence rests on three differentiators. Status after this round:

| differentiator (as written) | status | why |
|---|---|---|
| **(a) retrieval-closed isolation** — "no verified work holds retrieval at a measured `any_hit = 1.000` and varies only context composition" | ❌ **FALSIFIED** | ROC supplies **gold evidence** to every arm (closure by construction, stronger than ours). WhenLoss's `OE` condition is the same idea, and axis-separation *is its contribution*. §2 above. |
| **(b) unsupported-claim rate `U` on a notes-only arm** | ✅ **SURVIVES, and is now the ONLY load-bearing one** | Term census over full texts: ROC 73,821 chars → `faithful` 0, `hallucinat` 0, `fabricat` 0, `entail` 0, `unsupported` 1 (a motivating clause, not a metric). WhenLoss 58,176 chars → 0/0/0/0, `faithful` 1 = a reference-list hit. Supersede 40,915 chars → same. All three measure **omission or accuracy**; none measures **what generation ADDS**. Five targeted queries returned **total 0** (`RELATED_WORK.md` §12.3). |
| **(c) `notes+raw` as a distinct arm from `notes-only`** | ✅ **SURVIVES** | ROC: operators "**replace** the packed constituents in the answer context (freeing budget)". LazyMem, Supersede, WhenLoss `CSM`: all substitution. `alongside`/`hybrid`/`adjunct`/`supplement` = **0 hits in all three full texts**. The adjunct-vs-substitute *contrast itself* is unoccupied. |

**The claim does not die — it loses its first leg and must be re-anchored on (b)+(c).** That is a
narrowing, and it is exactly the outcome the standing rule prescribes: differentiate, do not abandon.

## 5. VERDICT

```
verdict: NEEDS_NARROWING
scope:   LEG 1 ONLY. Leg 2 stays FOLDED. Leg 3 recommended CUT (RELATED_WORK.md §6, now
         strengthened by Supersede occupying leg 2's read-out and MM-Mem holding leg 3's name
         at ACL 2026 Main, 2026.acl-long.533).
already_dead_should_archive: NO -- nothing is 完全相同 / 抄袭; all four new hits are CONCURRENT
         (2026-05 .. 2026-07 vs prereg 2026-08-14) and all are arXiv-only.
gpu:     NOT AUTHORISED by this verdict. A novelty verdict says "not preempted", never
         "worth a card now" (ready_queue.py:806-814). The four CPU blockers in
         STATUS.json.prior_gate are untouched by this file and two of them ARE the novelty.
```

### 5.1 The REQUIRED narrowing (mandatory, not advisory)

**Strike** from `RELATED_WORK.md` §3.3, §6 and §8 every formulation in which *retrieval-closed
isolation* is a contribution. Concretely:

1. ❌ **DELETE the differentiator** "(a) it never pins retrieval at a measured `any_hit = 1.000`, so
   its deltas mix retrieval with composition". It is factually false of `arXiv:2607.17545`.
2. ✅ The retrieval-closed stratum is **DEMOTED from contribution to precondition**. It stays in the
   design — it is still the right cell, and the measurement at the gate's own `budget=4000` is on
   disk (`evidence/b08_prereg_corrections_20260814.json`: `knowledge-update` n=78 `any_hit=1.0000`,
   `single-session-assistant` n=56 `any_hit=1.0000`, while overall drops to 0.9600 and
   `single-session-preference` to 0.7000). It may be described as *"a stratum where closure is
   measured on the deployed retriever rather than supplied by an oracle"* — a **provenance**
   difference from ROC/WhenLoss, worth one sentence, never a headline.
3. ✅ **The surviving claim must be re-anchored on `ΔU` (fabrication), with `Δ_sub` demoted.** Per
   `B08_LEG1_GATE_PREREG.md` §5.6 the three survival branches are separable; after this round only
   the **`ΔU` branch** ("notes are an adjunct, never a substitute" — MEASURED) is unoccupied.
   ⚠️ The `Δ_sub` branch ("~30× context reduction at no accuracy cost") is now **largely answered in
   the opposite direction by Supersede** (92%→77% on the knowledge-update subset). B08 may still
   report `Δ_sub`, but **only as a replication with a different closure regime**, never as a finding.
4. ✅ **`Δ_aug` must carry a single-reader scope clause.** `arXiv:2606.21807` measures a **31%
   pairwise ranking flip from generic summarisation on LongMemEval-S** — B08's exact benchmark — and
   a gain that shrinks with reader strength across 20 readers. A one-reader `Δ_aug` at
   Meta-Llama-3-8B is therefore **not** a claim about notes. Either add the clause verbatim to the
   gate, or add a second reader.

### 5.2 What did NOT change

- **The kill gate is untouched.** `kill_gate.conditions_KILL_iff_ALL_THREE` (K1+K2+K3) and the
  `disc_U ≤ 0.0872` evaluability precondition stand as pre-registered. **A literature count may not
  fire a kill gate** — only the proposal's own experiment may (`proposal/README.md`;
  `memory/prior-work-differentiate-dont-abandon.md`). K2, the `ΔU` clause, is now the *decisive*
  clause rather than one of three.
- **The four CPU blockers stand** (`STATUS.json.prior_gate`): the `A-notes-only` arm
  (`longmemeval/run_baseline.py:162` hardcodes `[notes_block] + list(evidence)`, no withhold path),
  the `U` scorer, the `SelfNotesCompressor`, the judge input adapter. **After this round, blockers 2
  and 3 are no longer merely blocking — they are the ENTIRE remaining contribution.** Building them
  is the only work that can still produce a finding.
- **`novelty_checked` and the cross-disk asset check.** See §6.

## 6. Why this file does not flip `novelty_checked`, and what it does instead

`RELATED_WORK.md` §10.1 measured that appending a clearance token flips the reader's
`novelty_checked` to `True` while B08 **stays `ready_cpu`** (held elsewhere), and declined to write it
on the principle that "an agent writing its own clearance field is not the clearance being reviewed"
(`memory/a-declared-lifecycle-is-not-an-adjudicated-one.md`). That principle is right about
**self**-clearance. This file is a **different session adjudicating that file's open question and
reaching a different conclusion than it did** — so the adjudication is recorded under
`novelty_verdict` (a `NOVELTY_VERDICT_KEYS` member, `ready_queue.py:288-289`) with verdict
`NEEDS_NARROWING`, which the reader maps to **`VERDICT_PENDING`** (`ready_queue.py:293`), i.e.
`novelty_checked` stays **False** and B08 stays `ready_cpu`.

**That is the correct and intended outcome, and it is not a failure to close the gate.** The gate
asked "is this preempted?" and the answer is a *conditional* no: not preempted, but two of three
differentiators are gone and the narrowing in §5.1 has not yet been applied to the documents it
governs. A verdict of `NEEDS_NARROWING` is `ready_queue.py`'s own vocabulary for exactly that state.
It will become `NOVEL_ENOUGH` when §5.1's four edits are applied to `RELATED_WORK.md` §3/§6/§8 and to
`B08_LEG1_GATE_PREREG.md` §5.6 — **0 GPU, and it is now the top of B08's queue.**

⚠️ **Still not verified by anyone, and it is not a literature question**: `remaining_blockers_all_CPU[6]`
— the zwfy6 asset check. Every presence claim for `models/Meta-Llama-3-8B`,
`data/longmemeval/longmemeval_s.json` and the Qwen3-8B judge weights is **wzc1-scoped**
(`memory/two-disk-rule-applies-to-main-too.md`). This round did **0 ssh** and did not close it either.

## 7. Provenance of this adjudication

- **0 GPU, 0 ssh, 0 training.** All 40 cards across 5 nodes were in use by other work; nothing here
  touched them.
- Network via `hy-proxy.woa.com:3128`. **Positive control first**: `arXiv:2310.08560` → HTTP 200,
  3,010 B, before any negative was recorded, so "no results" is distinguishable from "proxy broken".
- Full texts read (not abstracts): `arXiv:2607.17545v1` (347,897 B), `arXiv:2605.24579v1`
  (303,373 B), `arXiv:2606.27472v1` (163,841 B).
- Venues: DBLP `publ/api` for all six new rows (`total=1`, `CoRR`, `Informal and Other
  Publications`), with a **positive control that DBLP indexes ACL 2026** (MM-Mem returns both
  `ACL 2026 | 10.18653/V1/2026.ACL-LONG.533` and `CoRR 2026`), plus direct Anthology volume fetches.
  **Known gap: the `2026.findings-acl` index fetch truncated at 1,343,488 / 6,675,019 B (120 s
  timeout), so the Findings-ACL-2026 negative rests on DBLP alone.** Recorded rather than glossed.
- On-disk premise re-checked directly, not inherited:
  `evidence/b08_lme_bm25_recall_topk10.json` and `evidence/b08_prereg_corrections_20260814.json`.

## 8. Self-check: what the reader ACTUALLY does with this file (measured, not assumed)

**Before** (`python3 proposal/ready_queue.py`, whole-queue capture):

```
B08-memory-applications
   why: novelty gate not adjudicated (absent) -> the actionable task is 0 GPU: run it
```

**After**:

```
B08-memory-applications
   why: novelty gate not adjudicated (novelty_verdict.verdict=NEEDS_NARROWING -- not preempted
        (nothing is 完全相同/抄袭; all four newly found hits… (PENDING)) -> the actionable task is
        0 GPU: run it
```

A full `diff` of the before/after queue output is **exactly one line long** — line 33, the B08 `why`.
**No proposal changed section membership** (`ready_gpu` stays at 1 item, `B12`), so nothing was
promoted as a side effect. `read_one()` on the file returns `related_work_md: true`,
`problems: []`, `live_blockers: []`, `lifecycle: ready_cpu`.

**B08 is still `ready_cpu`, and the reason string is still the `novelty gate not adjudicated`
branch.** Being honest about that: the *evidence substring* changed from `absent` to the parsed
verdict, which is what proves the file is being read, but `NEEDS_NARROWING` is in
`ready_queue.py:293 VERDICT_PENDING`, so `novelty_checked` stays `False` and the reader keeps routing
B08 down the same branch. **A `NEEDS_NARROWING` verdict cannot and should not flip that branch** —
§6 explains why this is the intended terminal state of *this* gate rather than a shortfall.

### 8.1 Measured on a `/tmp` copy: B08 cannot be mis-dispatched to a GPU by a novelty flip

Because the previous section could be read as "so just write NOVEL_ENOUGH next time and it goes to
`ready_gpu`", that was **measured** rather than guessed (simulation only, in `/tmp/b08sim/`, **no
repo file touched**):

```
IF verdict became NOVEL_ENOUGH:
  lifecycle      : ready_cpu          <-- NOT ready_gpu
  novelty_checked: True
  reason         : DECLARED lifecycle=needs_prior_gate (honoured over an inferred ready_gpu;
                   prior_gate_needs_gpu=False -> schema sec 1.1 folds a 0-GPU prior gate into
                   ready_cpu): 4 CPU items, none needs a card: ...
```

So B08 is held by `lifecycle: "needs_prior_gate"` + `prior_gate_needs_gpu: false`
(`ready_queue.py:815-833`), **not** by the novelty axis. **Good news for GPU safety** — unlike B06,
no amount of novelty bookkeeping can spend a card here. It also means the four `prior_gate` CPU items
are the real queue, and after this round items **(2)** the `A-notes-only` arm and **(3)** the `U`
scorer are the entire remaining contribution.

⚠️ **One stale string worth flagging for whoever does the next B08 pass** (not editable — append-only,
`LIFECYCLE_SCHEMA.md` sec 0): `prior_gate` item **(1)** still reads *"leg-1-only RELATED_WORK.md"* as
an open CPU item. `RELATED_WORK.md` has existed since 2026-08-15 (commit `463dca4`) and is now 59,799 B
with two independent verification rounds appended. Item (1) is **discharged in fact**; only items
(2)-(4) plus `remaining_blockers_all_CPU[5]`/`[6]` remain. Recorded here because the frozen string will
otherwise keep being re-reported as outstanding — the same failure mode as
`memory/a-gate-that-says-never-run-may-already-have-run.md`.
