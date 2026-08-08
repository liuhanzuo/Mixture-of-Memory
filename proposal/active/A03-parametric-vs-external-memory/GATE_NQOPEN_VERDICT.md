---
gate: A03 second knowledge-axis floor gate (NQ-open)
date: 2026-08-09
node: .82 (8x H20 zwfy6)
wall: 5 min for 3 arms
verdict: AXIS_MEASURABLE -- NQ-open joins PopQA/TriviaQA/MMLU-content as A03's certified axes
also: three other axes (conflicting-knowledge, multi-evidence, new-injected-facts) are FORMALLY OUT OF SCOPE for A03; scope narrowed to closed-book parametric knowledge
---

# A03 second-axis floor gate — NQ-open

## 1. Result

Ran `_run_a03_axes_floor_82.sh` on `.82`, 8-shard × 3 arms, n=3610 per arm (Natural
Questions open-domain, closed-book):

| arm | model | em | contains | maj_em null | above floor? |
|---|---|---:|---:|---:|---|
| A03_1B_base | OLMo-2-0425-1B intact 16L | **0.1025** | 0.1429 | 0.0053 | YES (residual 0.948) |
| A03_1B_keep7_step200k | pruned+healed 9L @200k | **0.0285** | 0.0787 | 0.0053 | YES (residual 0.814) |
| A03_1B_keep7_step500 | barely-healed 9L @500 | **0.0017** | 0.0097 | 0.0053 | **NO** (below floor) |

Same three-arm pattern the PopQA/TriviaQA gate produced: intact well above floor,
healed above floor, barely-healed at/below floor. So the axis is measurable, and
'at floor' is empirically detectable rather than an abstract claim.

## 2. What this settles

**A03 now has FOUR floor-certified closed-book parametric-knowledge axes at 1B:**

* MMLU-content (content-interface, letter-interface retired at 1B)
* PopQA EM (with length-matched contains as secondary)
* TriviaQA EM (primary axis, largest floor-free dynamic range)
* **NQ-open EM (this gate)**

The kill condition A03 originally listed ("run the same floor gate on the three
remaining knowledge axes") is now **partially settled**: only one of those three
(NQ-open, which sits in the same closed-book parametric-knowledge family as PopQA/
TriviaQA) can actually be certified against the same protocol. The other two are
formally out of scope; see below.

## 3. Three axes A03 must formally drop, and why

The A03 PROPOSAL originally named three remaining axes: "new injected facts",
"updated/conflicting knowledge", "multi-evidence". A datasets audit on **both**
disks (wzc1 and zwfy6, verified this session) found none of the standard
instantiations available, AND identified a deeper protocol mismatch. Reporting
here so this does not get relitigated later:

| axis | canonical datasets | on disk? | protocol-compatible with A03? |
|---|---|---|---|
| conflicting knowledge | CounterFact, MQuAKE, zsRE, KnowEdit | NONE on either disk | NO — all require *context injection* (edit-then-query), not closed-book |
| multi-evidence | HotpotQA, 2WikiMultihopQA, MuSiQue | present only in LongBench WITH-CONTEXT form | NO — multi-hop with context is open-book, orthogonal to A03's closed-book question |
| new injected facts | (no canonical dataset) | N/A | NO — requires a CPT phase that injects facts and a held-out probe; A03 has neither yet |

**Reframe**: A03 is a closed-book parametric-knowledge study. Conflicting
knowledge and multi-hop retrieval are different papers. New-fact injection
requires a CPT arm A03 has not yet defined and is a follow-up. The paper's
6-arm study should explicitly limit the outcome variable to closed-book
parametric-knowledge accuracy on {MMLU-content, PopQA, TriviaQA, NQ-open}, four
axes now certified.

## 4. What's next for A03

1. **CPU: extend the analyzer.** `proposal/active/A03-parametric-vs-external-memory/code/analyze_1b_knowledge_floor.py` hardcodes popqa+triviaqa at lines 427-428. Adding one tuple `("nq_open", 3610, "em")` to that loop gives paired bootstrap CI, exact McNemar and BH-corrected floor calibration for NQ-open, matching the format used for PopQA/TriviaQA. ~15 min coder work.
2. **Design the 6-arm study on the four certified axes.** The 6 arms are the ones A03's PROPOSAL already names: intact, pruned+heal, pruned+heal+CPT-on-corpus, pruned+heal+raw-text RAG, pruned+heal+residual memory, pruned+heal+CPT+memory. Each arm is measured on {MMLU-content, PopQA EM, TriviaQA EM, NQ-open EM} against its own construct-appropriate null.
3. **NOT next**: do not spend cycles on CounterFact/MQuAKE/HotpotQA — those are separate papers.

## 5. Provenance

* Driver: `scripts/_run_a03_axes_floor_82.sh`
* Harness: `scripts/eval_olmo2_closedbook_qa.py --tasks nq_open` (native, item n=3610 asserted before merge)
* Dataset cache (zwfy6): `data/hf_datasets_cache/google-research-datasets___nq_open/` (6.7 MB)
* Output dirs (zwfy6): `olmo2_qa_results/A03_1B_{base,keep7_step200k,keep7_step500}/`
* Log: `.82:/apdcephfs_zwfy6/.../logs/a03_axes_floor.out`
* Wall: 5 min on 8× H20
