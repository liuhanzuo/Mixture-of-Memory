# B09 — NOVELTY ADJUDICATION, 2026-08-17

**0 GPU, 0 ssh. This file adjudicates; it runs nothing and authorises nothing.**

> ## ⚠️ FIRST, THE CORRECTION THAT MOTIVATES THIS FILE
>
> I was dispatched to *write* `proposal/backlog/B09-trajectory-aware-sft-data-selection/RELATED_WORK.md`
> on the premise that it does **not exist on disk**.
>
> **It exists.** 33,942 bytes, written 2026-08-15. It is a full venue-verified adjudication with a
> threat-ordered collision table, six added MUST-NOT-CLAIM items (15-20), a narrowed residual
> claim, and an eight-item "honest gaps" section. There is also a separate `NOVELTY.md`
> (16,669 B, 2026-08-08) with 14 further prohibitions. Overwriting either would have destroyed
> real artefacts.
>
> The premise traces to `ready_queue.py`'s line
> `why: novelty gate not adjudicated (absent) -> the actionable task is 0 GPU: run it`.
> `(absent)` is `rec["novelty_evidence"]` and reports **no verdict KEY in STATUS.json** — a
> different fact from file absence, which the reader prints on its own separate line and does
> **not** print for B09. Same class as `memory/append-only-records-outlive-their-own-truth.md`
> and `memory/read-what-the-consumer-reads-not-the-bare-key.md`.
>
> **So this file is the adjudication of the existing write-up**, not a replacement: an independent
> re-verification of the venues the 08-15 file left open, a check of its sharpest claim, and the
> STATUS.json verdict key the queue actually reads.

---

## 1. The claim under test

From `RELATED_WORK.md` §5, which already narrowed `NOVELTY.md` §4.1 on contact with Weasel.
Verbatim, and note that **it is stated as a null B09 intends to reject**:

> When a parent agent trajectory is expanded into many dependent SFT child rows, a selector that
> is given a hard parent-multiplicity bound `|S ∩ U_g| ≤ m_g` and a fixed assistant-target-token
> budget `Σ c_i ≤ B` does *not* outperform a random subset drawn under those *same* constraints,
> matched on benchmark distribution, decision-type mix and length bin.

Two properties make this the right object to adjudicate, and I verified both against
`PROPOSAL.md` rather than accepting the summary:

1. **The primary comparator is a null, not a method** (`PROPOSAL.md` §8: 「主 null baseline 不是
   row-random」, followed by the four matching requirements). Confirmed at `PROPOSAL.md:493-500`.
2. **The unit of analysis is a paired end-to-end SFT seed, not a row** (`PROPOSAL.md` §11:
   「不能把 100K 派生 rows 当成独立样本」). Confirmed at `PROPOSAL.md:664-677`.

⚠️ **And the load-bearing context, which no novelty verdict can discharge:** B09's `next_gate[0]`
is **GATE -1**, and as of 2026-08-16 it resolved to *"a 1.0 GiB download + CPU derive task"*
(`STATUS.json.gate_minus1_feasibility_2026_08_16`), **which has not been executed.** The pool
still does not exist on either disk. Novelty is therefore **not** what blocks B09, and clearing
it must not be allowed to imply otherwise — see §6.

## 2. Prior-art surfaces this claim must clear

| # | Surface | Where adjudicated |
|---|---|---|
| a | trajectory-aware / whole-trajectory agent data selection | `RELATED_WORK.md` §3.1 (Weasel), §3.2 (TopoCurate) |
| b | multi-turn dialogue rows are not independent | §3.3 (MDS) |
| c | within-trajectory critical-step supervision, and *verified* credit | §3.4 (ATLAS, CSO) |
| d | target-aware selection | §3.5 (RDS+) |
| e | submodular / budgeted / diminishing-returns coverage | §3.6 (DELIFT, SMART, MIG) |
| f | offline-RL / imitation demonstration coreset | §3.8 (CUPID, DataMIL) |
| g | **hard partition (parent-multiplicity) cap + row-multiplicity stress test** | §5/§5.1 — B09's residual; **re-tested here, §4** |

Surfaces (a)-(f) were searched on 2026-08-15. **I re-verified every venue claim** (§3) rather than
re-running the searches, because the 08-15 file's own §7 lists venue status as its main open item
and because venue errors are the failure mode this task specifically warns about.

---

## 3. Venue re-verification — per family, first-hand, 2026-08-17

Proxy positive control first, so an empty or failed lookup is never read as a finding:
`https://export.arxiv.org/api/query?search_query=all:electron&max_results=1` → HTTP **200**,
2938 B, `totalResults=184943`.

### 3.1 ACL family — authority is ACL Anthology (NOT OpenReview)

This is the rule the task flags, and `memory/venue-verify-acl-family-needs-anthology.md` exists
because applying the OpenReview rule here once misreported a Findings-NAACL-2025 paper as a
preprint. All five fetched directly from `aclanthology.org`, HTTP 200, `<title>` read:

| Anthology ID | title returned by the Anthology page | claim in `RELATED_WORK.md` | status |
|---|---|---|---|
| `2025.findings-acl.1299` | *ATLAS: Agent Tuning via Learning Critical Steps* | Findings of ACL 2025 | ✅ **CONFIRMED** |
| `2026.findings-acl.130` | *Data Selection for Multi-turn Dialogue Instruction Tuning* | Findings of ACL 2026 (MDS) | ✅ **CONFIRMED** |
| `2026.findings-acl.1974` | *Verified Critical Step Optimization for LLM Agents* | Findings of ACL 2026 (CSO) | ✅ **CONFIRMED** |
| `2025.findings-acl.515` | *MIG: Automatic Data Selection for Instruction Tuning by Maximizing Information G…* | Findings of ACL 2025 | ✅ **CONFIRMED** |
| `2024.findings-acl.766` | *SMART: Submodular Data Mixture Strategy for Instruction Tuning* | Findings of ACL 2024 | ✅ **CONFIRMED** |

**All five are Findings**, and all five are the ones DBLP/arXiv return as `journals/corr/` CoRR
records. The 08-15 file's warning — that auditing B09 through the CoRR record alone would
under-rate its three sharpest collisions (ATLAS, MDS, CSO) — is **correct and now independently
reproduced.** Note the title check also matters: the Anthology page reads **ATLAS**, while arXiv
`2503.02197` capitalises **ATLaS**; cite the Anthology spelling.

### 3.2 OpenReview family — authority is `venueid`

Queried `api2.openreview.net/notes/search`, HTTP 200:

| work | `venueid` returned | forum | status |
|---|---|---|---|
| **Weasel** — *Out-of-Domain Generalization for Web Agents via Importance-Diversity Data Selection* | **`ICML.cc/2026/Conference`**, `venue = "ICML 2026 regular"` | `EXCUyr6hhZ` | ✅ **CONFIRMED — published, main conference** |
| Weasel (second record) | `ICLR.cc/2026/Workshop/LLA`, `venue = "LLA 2026 Poster"` | `ixNDssFCkd` | workshop record; cite the ICML one |

This is the single most consequential row in B09's file, because Weasel is what forced the
narrowing. **It is confirmed at ICML 2026 main conference, and it is 2026-05 — three months before
this pass, so it is NOT concurrent** and cannot be set aside under the 2-3 month rule. DBLP still
carries it only as `journals/corr/abs-2605-20291`, exactly the 2026-conference lag
`memory/venue-verify-must-use-openreview-2026.md` documents.

### 3.3 The three rows that were `arXiv-only` — still `arXiv-only`

`RELATED_WORK.md` §7 item 1 lists TopoCurate, RDS+ and SWE-TRACE as unresolved and asks for
re-verification. Done, and **the answer has not changed**, which is itself worth recording so the
next agent does not re-run it a third time:

| work | DBLP (HTTP 200) | OpenReview (HTTP 200) | verdict |
|---|---|---|---|
| **TopoCurate** `2603.01714` | `total="1"`, `journals/corr/abs-2603-01714`, `venue=CoRR 2026` | search returned **0 titled notes** | **arXiv-only** |
| **RDS+** *Large-Scale Data Selection for Instruction Tuning* `2503.01807` | CoRR 2025 | note exists but `venueid = dblp.org/journals/CORR/2025`, `venue = "CoRR 2025"`, forum `Widxp7XGYm` — i.e. OpenReview is **mirroring the CoRR record**, not reporting an acceptance | **arXiv-only** |
| **SWE-TRACE** `2604.14820` | found among 23 hits: *SWE-TRACE: Optimizing Long-Horizon SWE Agents Through Rubric Process R…*, `journals/corr/abs-2604-14820`, `venue=CoRR 2026` | not queried | **arXiv-only** |

⚠️ **A distinction worth stating because it is exactly the trap:** RDS+ *does* have an OpenReview
note, and a careless reader would count that as an acceptance. Its `venueid` is
`dblp.org/journals/CORR/2025` — an **archive mirror**, not a conference venueid. An OpenReview
*hit* is not an OpenReview *acceptance*; only `venueid` naming a conference is.

Per `memory/venue-verify-must-use-openreview-2026.md`, `arXiv-only` means **"no peer-reviewed
venue verifiable from this node"**, never "no venue exists".

---

## 4. Surface (g) re-tested: is B09's residual claim still unoccupied?

This is the part of B09 that actually carries the direction after the narrowing, and it is the
part the 08-15 file could only support with *empty* searches. Empty searches are weak evidence,
so I ran five **new** queries it did not run, all HTTP 200:

| query | totalResults | disposition |
|---|---|---|
| `abs:"trajectory-level" AND abs:"data selection" AND abs:"agent"` | **0** | — |
| `abs:"constraint-matched" AND abs:"random baseline"` | **0** | — |
| `abs:"group constraint" AND abs:"data selection" AND abs:"language model"` | **0** | — |
| `abs:"per-trajectory" AND abs:"budget" AND abs:"selection"` | 3 | none relevant: VLA RL chunk masking `2605.16154`, Scratch repair `2603.29624`, CodeMonkeys `2501.14723` |
| `abs:"redundancy" AND abs:"trajectories" AND abs:"fine-tuning" AND abs:"agent"` | 27 | 1 new candidate (§4.1); the rest are RL credit assignment / GUI agents / benchmarks, and Weasel + TopoCurate reappear (already adjudicated) |

Combined with the 08-15 file's ten empty searches (`"sibling" AND "redundancy" AND "training
data"` → 0; `"parent" AND "child" AND "coreset"` → 0; `"hierarchical" AND "coreset" AND
"instruction"` → 0; …), **surface (g) remains unoccupied**, and the residual claim in
`RELATED_WORK.md` §5 stands as written.

⚠️ **Same caveat as the 08-15 file, and it does not go away by being repeated:** these are
`abs:`/`all:` field queries. A paper doing group-constrained coreset selection in different
vocabulary (active learning, dataset distillation, matroid-constrained submodular maximisation)
would be missed. **Searched and not found ≠ does not exist.**

### 4.1 NEW CANDIDATE — CurateEvo (not in the 08-15 file)

* **Cite**: *CurateEvo: Data-Curation Evolving for Agentic Post-Training*, `arXiv:2607.06140`,
  published **2026-07-07**.
* **Venue, authority used**: **arXiv-only.** No `<arxiv:comment>` at all (so no self-reported
  acceptance to weigh), and the paper is 2026-07 — squarely inside the window where DBLP/S2 lag,
  so absence of a DBLP conference record would not be evidence either way. Not ACL family.
* **Relation: ADJACENT.**
* **Why it matters to B09 specifically.** It is the closest thing found to B09's *problem framing*
  rather than its mechanism: it attacks agentic post-training curation and explicitly
  **"prun[es] redundant or low-utility training turns under a cost-aware objective"** — which is
  B09's `FORMAT_ONLY`/sibling-redundancy concern plus a token-cost objective, at **turn**
  granularity, on agent data. Evaluated on ACEBench-Agent, BFCL-V4, τ²-Bench.
* **Why it does not preempt, checked against its own abstract:**
  1. **Its selector is an evolving LLM-written program**, rewritten each epoch from *failed
     trajectories on a held-out dev set*. That is **meta-selection**, the same category as
     `RELATED_WORK.md` §3.9's DataMaster — and it makes the selection rule a moving target,
     which is the opposite of B09's pre-registered fixed objective with frozen weights
     (`PROPOSAL.md` §10 Phase 3 freezes selector weights before test).
  2. **No hard parent-multiplicity bound.** Turn pruning under a cost objective is a soft penalty;
     nothing enforces `|S ∩ U_g| ≤ m_g`, and B09's H1 is precisely soft-penalty vs hard-cap.
  3. **No constraint-matched null.** Its comparators are "prior curation methods"; it reports
     +3.2/+2.7 average points against them. It does not ask whether the gain survives a random
     subset drawn under the same cap/quota/length/token constraints — the question RDS+'s own
     result says is where selectors die.
  4. **It also produces RL data and an inference-time memory bank**, so it is a pipeline paper,
     not a selection-mechanism study.
* **Consequence — one addition to MUST-NOT-CLAIM, and it is a real tightening:**
  ❌ **"First to prune redundant/low-utility agent training turns under a cost-aware objective."**
  CurateEvo (2026-07) does that. B09's token-budget track survives as the axis on which
  *"the win is just more tokens"* is **falsifiable** (its own kill criterion #5), but the *idea*
  of cost-aware turn pruning on agent data is no longer available as a novelty claim.

---

## 5. Per-candidate summary (this file's additions and re-verifications)

| cite | venue | authority used + how | relation | why |
|---|---|---|---|---|
| **Weasel** `2605.20291` | **ICML 2026 (main)** | **OpenReview `venueid=ICML.cc/2026/Conference`**, `venue="ICML 2026 regular"`, forum `EXCUyr6hhZ` — used because DBLP has CoRR only | **ADJACENT, closest** | occupies importance+diversity over trajectory steps at fixed budget with OOD read-out; **lacks** hard parent cap, token knapsack, decision taxonomy, and the constraint-matched null. 2026-05 → **not concurrent** |
| **ATLAS** `2503.02197` | **Findings of ACL 2025** | **ACL Anthology `2025.findings-acl.1299`** (HTTP 200, title read) | **ADJACENT** (mandatory baseline) | critical-step loss masking vs full/random/PPL |
| **CSO** `2602.03412` | **Findings of ACL 2026** | **ACL Anthology `2026.findings-acl.1974`** | **ADJACENT** | owns outcome-flip *verified* credit → B09's `k_i` is only *proxy* criticality |
| **MDS** `2604.07892` | **Findings of ACL 2026** | **ACL Anthology `2026.findings-acl.130`** | **ADJACENT** | owns "multi-turn rows are not independent" |
| **MIG** `2504.13835` | **Findings of ACL 2025** | **ACL Anthology `2025.findings-acl.515`** | **ADJACENT** | owns diminishing-returns coverage = B09's `F_meta` form |
| **SMART** `2403.08370` | **Findings of ACL 2024** | **ACL Anthology `2024.findings-acl.766`** | **ADJACENT** | owns task selection + budget allocation + within-task selection |
| **CurateEvo** `2607.06140` | arXiv-only | arXiv API; no `arxiv:comment`; 2026-07 inside the lag window | **ADJACENT (new)** | cost-aware redundant-turn pruning on agent data via an evolving LLM-written curator; no hard cap, no matched null |
| **TopoCurate** `2603.01714` | **arXiv-only** | DBLP `journals/corr/abs-2603-01714` + OpenReview 0 titled notes | **ADJACENT** | whole-trajectory tool-use selection; cite as preprint |
| **RDS+** `2503.01807` | **arXiv-only** | DBLP CoRR 2025 + OpenReview note whose `venueid` is the **CoRR mirror**, not a conference | **ADJACENT, dual-signed** | removes "target-query relevance is new"; its "selectors lose to random at scale" finding is B09's best justification |
| **SWE-TRACE** `2604.14820` | **arXiv-only** | DBLP `journals/corr/abs-2604-14820` | anchor | token-efficient trajectory synthesis |

**This file: n_candidates = 10 re-adjudicated (1 new), n_preempt = 0, n_adjacent = 10.**
Combined with the 08-15 pass (≈28 named works) and `NOVELTY.md` (2026-08-08):
**n_candidates = 29, n_preempt = 0, n_adjacent = 11** (Weasel, TopoCurate, MDS, ATLAS, CSO, RDS+,
DELIFT, SMART, MIG, CurateEvo, plus the CUPID/DataMIL demonstration-coreset pair as one family).

---

## 6. Verdict

```
verdict: hold_in_backlog -- novelty gate CLEARED for the NARROWED claim of RELATED_WORK.md
         section 5; NOT preempted by any candidate
n_candidates: 29   n_preempt: 0   n_adjacent: 11
venue_authority: ACL Anthology for the 5 Findings rows (ATLAS/MDS/CSO/MIG/SMART);
                 OpenReview venueid for Weasel (ICML.cc/2026/Conference)
new_must_not_claim: "first to prune redundant/low-utility agent training turns under a
                     cost-aware objective" -- foreclosed by CurateEvo arXiv:2607.06140 (2026-07)
gpu: NONE authorised. Blocked on GATE -1 (data acquisition), which is 0 GPU and NOT DONE.
already_dead_should_archive: NO
```

**No candidate is 完全相同 / 抄袭.** Weasel is closest and is genuinely published at ICML 2026, but
it lacks all four of B09's distinguishing constraints, and per
`memory/prior-work-differentiate-dont-abandon.md` overlap is not preemption. The correct response
was narrowing, and the 08-15 file already performed it.

### 6.1 ⚠️ CLEARING NOVELTY MUST NOT MAKE B09 `ready_gpu` — I MEASURED THIS

I ran `ready_queue.read_one()` on temp copies in `/tmp` (**no repo file touched**) to see what a
cleared novelty verdict does to B09's inferred lifecycle:

| STATUS.json | inferred lifecycle |
|---|---|
| as-is today | `ready_cpu` |
| + a cleared novelty verdict the reader can see | ⚠️ **`ready_gpu`** |

**That would be wrong**, and dangerously so: B09's actual next step is a **1.0 GiB download plus a
CPU derive** (`gate_minus1_feasibility_2026_08_16.next_action`), the candidate pool **does not
exist on either disk**, and `PROPOSAL.md`'s entire experimental program is downstream of it. The
inference reaches `ready_gpu` because B09 has no `gpu_policy`/`blocked_by` key and — decisively —
**no `next_gate_gpu` key at all**, so `_next_gate_is_free` finds nothing to read and cannot hold it.
B06 is held in `ready_cpu` by exactly that mechanism; B09 has no equivalent.

**So my append declares `lifecycle_20260817: "ready_cpu"` explicitly.** Per `ready_queue.py:805-822`
a declared `ready_cpu` is authoritative and is honoured in the *safe* direction ("down-grading
GPU→CPU on the owner's say-so can only ever free a card, never waste one"). I also append a
`next_gate_gpu`-family cost key stating the next step is 0 GPU, so that the *inference* path would
also reach the right answer if the declaration were ever removed — belt and braces, because a
single field that can flip a proposal onto a GPU queue is exactly the
`memory/a-declared-lifecycle-is-not-an-adjudicated-one.md` failure mode.

---

## 7. What I appended to STATUS.json, and the reader bug found doing it

`ready_queue.py:320-321`:

```python
NOVELTY_VERDICT_KEYS = ["novelty_check_2026_08_09", "novelty_verdict",
                        "k1_novelty", "related_work_status"]
```

**Hardcoded, with no dated-priority slot** — unlike `NEXT_GATE_KEYS`, `KILL_KEYS`,
`NEXT_GATE_COST_KEYS`, `_DATED_LIFECYCLE`, `_DATED_LIFECYCLE_WHY`, each of which was given one
*after* this defect bit. Measured on temp copies:

| appended key | reader sees |
|---|---|
| `novelty_verdict_20260817` | `novelty_checked=False`, evidence `absent` — **INVISIBLE** |
| `novelty_verdict` (undated) | `novelty_checked=True`, evidence `novelty_verdict.verdict=hold_in_backlog` |

So the dated key alone would have been a **silent no-op** — the gate would still read
"not adjudicated" while a verdict sat in the file, which is the six-day
`kill_gate_executable_20260814` stall repeated on the novelty axis. I appended **both**: the dated
key for provenance and the undated `novelty_verdict` the reader resolves, cross-referencing each
other. **I did not edit `ready_queue.py`** — a `NOVELTY_VERDICT_KEYS` dated slot is the right fix
but it re-scores all 15 proposals and should not land as a side effect of a B06/B09 task.

## 8. Honest gaps in THIS file

1. **No full text was read.** Every characterisation is from abstracts plus venue metadata. The
   08-15 file's §7 item 6 already flags that the Weasel/TopoCurate/MDS/ATLAS/CSO differences are
   design details an abstract can hide (is the cap hard or soft? is there a token budget?), and
   **that requirement is not discharged by this file.** It remains the largest risk.
2. **I did not re-run surfaces (a)-(f)'s searches**, only their venue claims. If the 08-15
   searches missed something, this pass would not catch it.
3. **`SOURCES.md` item 2 (official implementations + licence compatibility) is still NOT DONE.**
   No repository cloned, no `LICENSE` read. Downstream of GATE -1 anyway.
4. **Zero cross-disk verification.** `/apdcephfs_zwfy6` is not mounted here and ssh was forbidden
   for this task. Every disk fact is carried from `DATA_AUDIT_VERDICT_20260810.md` and
   `gate_minus1_feasibility_2026_08_16`, which did search both disks. Per
   `memory/two-disk-rule-applies-to-main-too.md` I am not re-deriving it and not claiming to.
5. **`promotion_criteria[4]` still needs amending** (`RELATED_WORK.md` §5.3): it names "ATLAS,
   RDS+, and facility location" and must add Weasel and CSO. STATUS.json is append-only and this
   file is not authorised to rewrite promotion criteria; recorded again so it is not lost.
6. **`abs:` field scoping** — see §4's caveat. Nine of the fifteen negative queries across both
   passes returned exactly 0, and a 0 from a field query is the weakest evidence in this file.

---

## 9. ⚠️ SELF-CORRECTION — a claim in §6.1 was wrong, and I only found it by testing it

§6.1 above ends: *"I also append a `next_gate_gpu`-family cost key stating the next step is 0 GPU,
so that the **inference** path would also reach the right answer if the declaration were ever
removed — belt and braces."*

**I appended `next_gate_gpu_20260817` and then tested that sentence. It was FALSE.**

`ready_queue.py:249-251` — `NEXT_GATE_COST_KEYS` is hardcoded as
`('next_gate_gpu_20260816', 'next_gate_gpu', 'next_gate_cost', 'gate_gpu', 'next_gate_gpu_cost')`
and `_DATED_COST_KEYS = ('next_gate_gpu_20260816',)`. **A 2026-08-17 date is not in either.**
Measured:

| state | `_next_gate_is_free()` | inferred lifecycle (declaration stripped) |
|---|---|---|
| with `next_gate_gpu_20260817` only | `''` | ⚠️ **`ready_gpu`** |
| after also appending undated `next_gate_gpu` | `'next_gate_gpu: 0 GPU. GATE -1 is a 1.0 GiB HTTP download…'` | ✅ `ready_cpu` |

**This is `memory/fix-the-class-not-the-instance.md` in the space of a single session.** §7 of this
very file documents the dated-slot defect on the **novelty** key family — and I then walked
straight into the identical defect on the **cost** key family, while writing the sentence claiming
I had guarded against it. Finding it required running the check instead of asserting the outcome;
had I not, B09 would have been one field-deletion away from a GPU queue with no candidate pool
on disk.

**Repaired** by appending the undated `next_gate_gpu` sibling plus
`reader_dated_slot_defect_20260817`, which records the whole family census:

| key family in `ready_queue.py` | resolution | future dates work? |
|---|---|---|
| `NEXT_GATE_KEYS` | enumerated, dated slot `…_20260814` | ❌ pinned |
| `KILL_KEYS` | enumerated, dated slot `…_20260814` | ❌ pinned |
| `NEXT_GATE_COST_KEYS` | enumerated, dated slot `…_20260816` | ❌ pinned — **bit me today** |
| `NOVELTY_VERDICT_KEYS` | enumerated, **no dated slot at all** | ❌ — **bit me today** |
| `_DATED_LIFECYCLE` | **regex** `^lifecycle_(20\d{6})…` | ✅ |
| `_DATED_LIFECYCLE_WHY` | **regex** `^lifecycle_why_(20\d{6})…` | ✅ |

Two families were converted to regex on 2026-08-17 (per the comments at `ready_queue.py:398-406`
and `:434-449`); **the two that were left enumerated are exactly the two that failed.** The fix is
to give `NOVELTY_VERDICT_KEYS` and `NEXT_GATE_COST_KEYS` the same regex treatment. **I did not
apply it** — it re-scores all 15 proposals and must not land as a side effect of a novelty task.

**Standing instruction for the next agent:** appending a dated key to a STATUS.json is **not**
sufficient to change what the scheduler sees. Append the undated sibling too, and **verify with
`python3 proposal/ready_queue.py`** — do not assume the append landed.
