# B06 — NOVELTY ADJUDICATION, 2026-08-17

**0 GPU, 0 ssh. This file adjudicates; it runs nothing and authorises nothing.**

> ## ⚠️ FIRST, THE CORRECTION THAT MOTIVATES THIS FILE
>
> I was dispatched to *write* `proposal/backlog/B06-portable-decompression-adapter/RELATED_WORK.md`
> on the stated premise that it does **not exist on disk** and that B06's novelty gate is therefore
> "unadjudicated with no RELATED_WORK.md".
>
> **The file exists.** 28,761 bytes, written 2026-08-15, and it is a full venue-verified
> adjudication with a verdict block, a 11-item MUST-NOT-CLAIM list, and a §5 "honest gaps"
> section. Overwriting it would have destroyed a real artefact and re-run ~8 hours of literature work.
>
> `STATUS.json` already says so: `related_work_presence_correction_20260816` records
> `"exists": true, "bytes": 28761, "verified_by": "os.stat on wzc1 at 2026-08-16"`. So does
> `ready_queue.py`'s own docstring (lines 55-66, "CORRECTED 2026-08-16 … among the proposals this
> tool actually reports on, the count of missing RELATED_WORK.md is ZERO").
>
> **The premise came from `ready_queue.py`'s report line, which is accurate but easy to misread:**
> `why: novelty gate not adjudicated (absent) -> the actionable task is 0 GPU: run it`.
> The word `(absent)` is `rec["novelty_evidence"]`, and it describes **the absence of a verdict KEY
> in STATUS.json**, not the absence of the file. The file-presence problem is a *separate* line
> (`! RELATED_WORK.md absent`), and for B06 that line is **not printed** — precisely because the
> file is there. This is the same class as
> `memory/append-only-records-outlive-their-own-truth.md` and
> `memory/read-what-the-consumer-reads-not-the-bare-key.md`.
>
> **So this file is not the write-up.** It is the *adjudication* of the existing write-up: an
> independent verification pass over the parts `RELATED_WORK.md` §5 itself flagged as unfinished,
> plus the STATUS.json verdict key that the queue actually reads.

---

## 1. The claim under test

From `STATUS.json.claim_scope_discipline.may_claim`, verbatim:

> On a retrieval-free HCache read path at 8B j=12, a self-distilled LoRA recovers +23.12 pp
> Judge_1:4 (paired, p=2.6e-67), so the adapter is not a CoMem-retrieval-pack specialisation.

The **contested word is "portable"**, and the audit bar
(`proposal/shared/literature/RELATED_WORK_GAP_AUDIT_20260808.md:96`) requires multi-task,
multi-compressor, multi-model, or an explicit layer/module transfer. The existing
`RELATED_WORK.md` §1 scores B06 at **1 of 4 disjuncts met, and the weakest reading of that one**.
I re-verified that scoring against `STATUS.json` and it is correct:
`kill_gate.condition_3_status = "UNTESTED -- no second compressor has been run"`.

## 2. Prior-art surfaces this claim must clear

The five families named by the gap audit, plus one the audit did not name:

| # | Surface | Adjudicated in |
|---|---|---|
| a | activation decompression adapter | `RELATED_WORK.md` §2.1 |
| b | split-compute reconstruction | §2.2 (delegated to B05's file) |
| c | adapter transfer (cross-model / cross-task) | §2.3 |
| d | intermediate self-distillation | §2.4 |
| e | **cross-codec portability** | §2.5 — **INCOMPLETE, see §3 below** |
| f | recomputation-based repair as the systems foil | §2.5 table (Cache-Craft, CacheBlend) |

Surfaces (a)-(d) and (f) were searched to completion on 2026-08-15 with venues verified per
family. **I did not re-litigate them** — re-running a completed search is not verification, and
the four families' dispositions are internally checkable from the Anthology/OpenReview IDs
already recorded. What was *not* complete is (e), and that is what this file adds.

---

## 3. Closing surface (e): the three searches that died on HTTP 429

`RELATED_WORK.md` §5 item 2 flags, in its own words:

> ⚠️ **Three cross-codec-portability queries died on arXiv 429 / read-timeout and were NOT
> retried to completion** … **§2.5's "no direct hit" is therefore the weakest finding in this
> file** and must be re-run before any submission.

That is the single highest-value 0-GPU item in the file, because §2.5 is the family B06's opening
rests on. Re-run 2026-08-17 through `hy-proxy.woa.com:3128`.

**Proxy positive control BEFORE any not-found claim** (per
`memory/absence-on-path-is-not-absence-on-disk.md` — an empty result must be distinguished from a
dead endpoint): `https://export.arxiv.org/api/query?search_query=all:electron&max_results=1`
→ HTTP **200**, 2938 B, `<opensearch:totalResults>184943</opensearch:totalResults>`.

(results appended below as each query returns)

### 3.1 The three previously-dead queries, re-run to completion

| query (verbatim) | 2026-08-15 | 2026-08-17 | totalResults |
|---|---|---|---|
| `all:"KV cache" AND all:"repair" AND all:"LoRA"` | 429, NOT RUN | **HTTP 200** | **0** |
| `all:"CacheBlend" AND all:"adapter"` | 429, NOT RUN | **HTTP 200** | **1** → §3.2 |
| `all:"cross-method" AND all:"cache reuse" AND all:"generalization"` | 429, NOT RUN | **HTTP 200** | **0** |

### 3.2 NEW CANDIDATE — KV Packet (the hit the 429 was hiding)

* **Cite**: *KV Packet: Recomputation-Free Context-Independent KV Caching for LLMs*,
  `arXiv:2604.13226` v2, published **2026-04-14**.
* **Venue, and the authority used**: **arXiv-only.**
  * DBLP (`dblp.org/search/publ/api`, HTTP 200): `total="1"`, key
    `journals/corr/abs-2604-13226`, `venue=CoRR`, `year=2026`,
    `type="Informal and Other Publications"`.
  * OpenReview (`api2.openreview.net/notes/search`, HTTP 200) — checked **because the repo rule
    says DBLP lags 2026 conferences** (`memory/venue-verify-must-use-openreview-2026.md`).
    Returned 8 notes, **none of which is a paper record**: they are anonymous *review* notes
    (fields `summary`/`soundness`/`confidence`/`questions`) whose content is about a
    different system, `CrossKV` segment-level KV sharing. **No `venueid` on any note.**
    So OpenReview does **not** upgrade this to a published venue, and the fact that
    a text search returns reviewer notes for an unrelated submission is itself a reminder that
    an OpenReview *search hit* is not an OpenReview *acceptance record*.
  * Not ACL family → Anthology not applicable.
* **Relation: ADJACENT (and it is now the closest work on B06's own mechanism).**
* **Why.** KV Packet wraps cached documents in **light-weight trainable soft-token adapters
  trained via self-supervised distillation to bridge context discontinuities**, and positions
  itself explicitly against **CacheBlend / EPIC / SAM-KV**, i.e. against the recomputation-based
  repair line that `RELATED_WORK.md` §2.5 already names as B06's systems foil. Structurally this
  is *the same idea shape as B06*: a **trained, distilled module** that repairs a **reused cache**
  so a frozen model can consume it, instead of recomputing.
  Three differences, each load-bearing and each checkable from the abstract:
  1. **Cache level.** KV Packet operates on the **KV cache** (per-layer K/V tensors);
     B06 operates on **mid-layer hidden states** at a fixed depth `j=12` and repairs the
     **readout**, not the cache. The HCache line (`eval_qcmem_locomo.py:56`, EuroSys 2025)
     exists precisely because those are different objects.
  2. **What is trained.** KV Packet trains **soft tokens wrapping the cache** — the cache side.
     B06 trains a **rank-32 LoRA on the model's upper layers** — the reader side. This is the
     same dichotomy `RELATED_WORK.md` §2.4 already draws against **Cartridges** (ICLR 2026),
     and KV Packet lands on the *Cartridges* side of it.
  3. **Portability is not tested.** Its evaluation is Llama-3.1 + Qwen2.5 on its **own**
     packet format. It does **not** ask whether one adapter survives a change of cache
     production rule — which is the *only* thing B06's residual claim asserts.
* **Consequence — this tightens B06's MUST-NOT-CLAIM list by one item, and it is not cosmetic.**
  `RELATED_WORK.md` §3 item 1 forecloses "we introduce a decompression adapter for cached
  intermediate states" (citing SeDeM + ICAE). KV Packet additionally forecloses the *serving*
  framing: **"a trained adapter is a cheaper alternative to recomputation-based cache repair"
  is now prior art at 2026-04**, four months before B06's measurement, so it is **not**
  concurrent and cannot be waved off on the 2-3 month rule.
  B06 may still claim the **transfer/portability** property; it may no longer claim the
  **adapter-instead-of-recompute** framing as its own.

### 3.3 NEW CANDIDATE — LongAttnComp (cross-family compression)

* **Cite**: *LongAttnComp: Cross-Family Context Compression for Long-Context Reasoning*,
  `arXiv:2606.01336` v2 (v1 2026-05-31, v2 2026-06-19).
* **Venue, and the authority used**: **arXiv-only.** The paper's own
  `<arxiv:comment>` reads verbatim **"Under review"** — i.e. the authors themselves state it is
  unpublished, which is the strongest available signal and needs no third-party lookup.
* **Relation: ADJACENT.**
* **Why.** Its headline includes "**transfers across four target models from three families**",
  so it is the nearest thing found to a *portability* claim in context compression, and its
  Stage-1/Stage-2 recipe fine-tunes **the compressor's scoring layer**. But: (a) it compresses at
  the **token** level (selects which tokens survive), not the activation level — it is a
  write-side selection policy, the same disposition `RELATED_WORK.md` §2.1 gives PromptDistill;
  (b) the thing that transfers is **the compressor across reader models**, whereas B06 asks
  whether **one reader-side adapter survives across compressors** — the axes are transposed;
  (c) no adapter on the reader at all. **Does not preempt; must be cited** as the closest
  existing use of the word "transfers" in this literature.

### 3.4 Five further cross-codec queries, all HTTP 200

| query | totalResults | disposition |
|---|---|---|
| `all:"soft-token adapter" AND all:"KV cache"` | 1 | KV Packet again (§3.2) |
| `abs:"same adapter" AND abs:"different compressor"` | **0** | — |
| `abs:"compression-agnostic" AND abs:"language model"` | **0** | — |
| `abs:"codec-agnostic" AND abs:"large language model"` | **0** | — |
| `abs:"reusable" AND abs:"adapter" AND abs:"cache reuse"` | 1 | *Adaptive KV Cache Reuse for Fast Long-Context LLM Serving* `2605.24022` — training-free non-prefix reuse, no trained module. Not a collision. |
| `all:"intermediate layer" AND all:"LoRA" AND all:"cached activations"` | **0** | — |
| `abs:"context distillation" AND abs:"transfer" AND abs:"compression method"` | **0** | — |
| `abs:"decompression" AND abs:"hidden states" AND abs:"adapter"` | **0** | — |
| `abs:"cache format" AND abs:"generalize"` | 1 | RAP runtime pruning `2505.17138` — unrelated |
| `all:"adapter" AND all:"transfers across" AND all:"compression"` | 9 | 1 relevant (LongAttnComp, §3.3); 8 unrelated (video SR, LiDAR world models, graph adaptation, dataset distillation, prompt optimisation, concept models, Gaussian vision) |

**Verdict on surface (e):** it is now **searched to completion** and the honest finding has
*changed direction slightly*. `RELATED_WORK.md` §2.5 called it "the family where I could not find
a direct hit, and that is B06's opening". After closing the 429s, the accurate statement is:

> **No work performs B06's measurement** (one trained reader-side repair, held fixed, evaluated
> across ≥2 cache production rules). **But the adapter-instead-of-recompute *idea* is occupied**
> by KV Packet (2026-04, arXiv-only) at the KV level, and the word "transfers" is occupied by
> LongAttnComp at the compressor level. The opening is **narrower than §2.5 implied** — it is the
> *transfer experiment*, not the *mechanism* and not the *framing*.

This is a **narrowing, not a kill**: per `memory/prior-work-differentiate-dont-abandon.md` the bar
is 完全相同/抄袭, and neither candidate runs B06's contrast.

---

## 4. Per-candidate summary table (this file's additions only)

| cite | venue | authority used | relation | why |
|---|---|---|---|---|
| KV Packet `2604.13226` | arXiv-only, CoRR 2026 | DBLP `journals/corr/abs-2604-13226` (HTTP 200) **+ OpenReview searched, no `venueid`** | **ADJACENT** | trained distilled adapter repairs a reused **KV** cache vs recomputation; B06 repairs a **hidden-state readout**. Forecloses B06's *adapter-instead-of-recompute framing*; does not test portability. **2026-04 = NOT concurrent.** |
| LongAttnComp `2606.01336` | arXiv-only | **the paper's own `arxiv:comment` = "Under review"** | **ADJACENT** | token-level compressor transferring **across reader models**; B06 transfers a **reader adapter across compressors** — transposed axes |
| Adaptive KV Cache Reuse `2605.24022` | arXiv-only | arXiv API only | not a collision | training-free non-prefix reuse; no trained module |
| RAP `2505.17138` | arXiv-only | arXiv API only | not a collision | runtime pruning, unrelated |

**n_candidates adjudicated in this file: 4. n_preempt: 0. n_adjacent: 2.**
Combined with the 2026-08-15 pass (18 candidates across surfaces a-d and f, 0 preempting):
**n_candidates = 22, n_preempt = 0, n_adjacent = 4** (SeDeM, RAC, KV Packet, LongAttnComp as the
concurrent/near set; LLoCO + Embedding Recycling remain the two peer-reviewed *setup* anchors).

---

## 5. Verdict

```
verdict: hold_in_backlog -- novelty gate CLEARED for the narrowed claim; NOT preempted
n_candidates: 22   n_preempt: 0   n_adjacent: 4
surface_e_status: CLOSED (was the 2026-08-15 file's weakest finding; 11 queries, all HTTP 200)
new_must_not_claim: "a trained adapter is a cheaper alternative to recomputation-based cache
                     repair" -- foreclosed by KV Packet arXiv:2604.13226 (2026-04, NOT concurrent)
already_dead_should_archive: NO
```

**No candidate is 完全相同 / 抄袭.** The closest two on the mechanism (SeDeM `2608.00311`,
RAC `2608.04991`) are concurrent; the closest on the framing (KV Packet, 2026-04) is **not**
concurrent but does not run B06's transfer contrast and trains the cache side rather than the
reader side.

### 5.1 ⚠️ THE GATE IS NOT WHAT BLOCKS B06, AND I AM NOT RECOMMENDING A CARD

Adjudicating novelty does **not** make B06 `ready_gpu`, and I verified this rather than assuming it.
`STATUS.json.drift_resolution_leg_20260815.kill_gate_status_after_this_leg` records that kill
conditions 1 and 2 **do not fire** and condition 3 (second compressor) is untested. The remaining
0-GPU items — already named in `RELATED_WORK.md` §4 and in
`drift_resolution_leg_20260815.highest_value_next_0_gpu_followup` — are:

1. **`wc -l` on the canonical judge cache on zwfy6** (one ssh, 0 GPU). If it holds ~1439 records
   instead of 1540, silent judge-API failures are the *proven* mechanism for the canonical 8.11
   outlier. **This task forbade ssh, so I did not run it** — it stays open.
2. **A full-text differential read of SeDeM / RAC / KV Packet.** All three were adjudicated from
   abstracts only. §3.2's three differences are abstract-level and must be checked against method
   sections before any write-up.

`ready_queue.py` already holds B06 in `ready_cpu` via `_next_gate_is_free`
(`next_gate_gpu` = "The drift leg is 0 GPU…"), and **that is the correct disposition.** My
STATUS.json append is deliberately written so it cannot flip that: see §6.

---

## 6. What I appended to STATUS.json, and the reader bug I found doing it

**A dated key alone would have been a silent no-op.** `ready_queue.py:320-321`:

```python
NOVELTY_VERDICT_KEYS = ["novelty_check_2026_08_09", "novelty_verdict",
                        "k1_novelty", "related_work_status"]
```

That list is **hardcoded with no dated-priority slot**, unlike `NEXT_GATE_KEYS`, `KILL_KEYS`,
`NEXT_GATE_COST_KEYS`, `_DATED_LIFECYCLE` and `_DATED_LIFECYCLE_WHY`, all five of which were given
one after this exact bug bit. Measured on temp copies in `/tmp` (no repo file touched):

| appended key | reader sees |
|---|---|
| `novelty_verdict_20260817` | `novelty_checked=False`, evidence `absent` — **INVISIBLE** |
| `novelty_verdict` (undated) | `novelty_checked=True`, evidence `novelty_verdict.verdict=hold_in_backlog` |

So the instruction to "use these key names so `proposal/ready_queue.py` can read them" was
**not satisfiable as literally specified** — `novelty_verdict_20260817` is exactly the shape of
key the reader cannot see. This is the sixth instance of the same defect class the file's own
docstring memorialises (`kill_gate_executable_20260814` unread for six days;
`lifecycle_20260817` unread; `lifecycle_why_20260817` unread). I therefore appended **both**:
the dated key for provenance **and** the undated `novelty_verdict` the reader resolves, with the
dated one naming the other so they cannot drift apart.

**I did not edit `ready_queue.py`.** Adding a `NOVELTY_VERDICT_KEYS` dated slot is the right fix
and is one line, but it changes how **all 15** proposals are scored, and B06/B09 are not the place
to land a repo-wide scheduler change without an adversarial pass. Recorded here as the
highest-value 0-GPU follow-up on the tooling axis.

### 6.1 The defect is a FAMILY, not one key (verified 2026-08-17)

While appending B09's equivalent record I asserted that a dated *cost* key
(`next_gate_gpu_20260817`) would hold that proposal on the inference path, then **tested it and
found it false**: `NEXT_GATE_COST_KEYS` is likewise hardcoded with its dated slot pinned to
`next_gate_gpu_20260816`. Census of all six key families:

| family | resolution | future dates work? |
|---|---|---|
| `NEXT_GATE_KEYS` | enumerated, dated slot `…_20260814` | ❌ pinned |
| `KILL_KEYS` | enumerated, dated slot `…_20260814` | ❌ pinned |
| `NEXT_GATE_COST_KEYS` | enumerated, dated slot `…_20260816` | ❌ pinned |
| `NOVELTY_VERDICT_KEYS` | enumerated, **no dated slot** | ❌ |
| `_DATED_LIFECYCLE` | **regex** | ✅ |
| `_DATED_LIFECYCLE_WHY` | **regex** | ✅ |

The two families converted to regex on 2026-08-17 work; **all four still enumerated are pinned to
past dates.** B06 is unaffected in practice — it already has an undated `next_gate_gpu`
("The drift leg is 0 GPU…") which `_next_gate_is_free` reads, and I verified that with B06's
lifecycle declaration stripped the inference path still returns `ready_cpu`. Recorded here because
the *class* matters more than B06's instance: **a dated key alone is a silent no-op, and the only
way to know is to run `python3 proposal/ready_queue.py`.**
