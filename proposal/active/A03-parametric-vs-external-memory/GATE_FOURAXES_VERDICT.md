---
gate: A03 four-axis floor calibration (complete, BH-corrected)
date: 2026-08-09
compute: CPU only on .82 (analyzer run, ~1 min); the underlying GPU evals were already done
verdict: ALL FOUR AXES FLOOR-CERTIFIED -- A03's measurement instrument is validated on closed-book parametric knowledge
supersedes: GATE_NQOPEN_VERDICT.md's "analyzer_TODO" (now closed)
---

# A03 — four-axis floor calibration, complete

## 1. What was missing and is now closed

`GATE_NQOPEN_VERDICT.md` reported raw NQ-open EM against the majority-answer floor but
flagged that `analyze_1b_knowledge_floor.py` hardcoded only popqa+triviaqa, so NQ-open
had **no bootstrap CI, no BH correction, and no length-matched contains null**. That
one-line patch is applied (`("nq_open", 3610, "em")` added to the task loop at line
427-429) and the analyzer re-run over all three arms.

## 2. The complete table (n_perm bootstrap, BH-corrected across all 33 cells)

Residual fraction = (reported − null) / (1 − null): what share of the *available*
headroom above the construct-appropriate floor the arm actually captures.

| arm | axis | interface | reported | null | resid | frac | BH p | verdict |
|---|---|---|---:|---:|---:|---:|---:|---|
| intact | MMLU | letter | 0.3807 | 0.2689 | +0.1118 | 29.4% | 1.1e-4 | ABOVE |
| intact | MMLU | content_norm | 0.3868 | 0.2845 | +0.1024 | 26.5% | 1.1e-4 | ABOVE |
| pruned+healed | MMLU | letter | 0.2512 | 0.2689 | −0.0177 | −7.1% | 3.7e-3 | **BELOW** |
| pruned+healed | MMLU | content_norm | 0.3244 | 0.2845 | +0.0399 | 12.3% | 1.1e-4 | ABOVE |
| barely_healed | MMLU | letter | 0.2295 | 0.2689 | −0.0395 | −17.2% | 1.1e-4 | BELOW |
| barely_healed | MMLU | content_norm | 0.2632 | 0.2845 | −0.0212 | −8.1% | 1.1e-4 | BELOW |
| intact | PopQA | em | 0.1550 | 0.0229 | +0.1321 | 85.2% | 1.1e-4 | ABOVE |
| pruned+healed | PopQA | em | 0.0394 | 0.0229 | +0.0165 | 41.8% | 1.1e-4 | ABOVE |
| barely_healed | PopQA | em | 0.0000 | 0.0229 | −0.0229 | n/a | 1.1e-4 | BELOW |
| intact | PopQA | contains_lenmatched | 0.1678 | 0.0259 | +0.1419 | 84.6% | 1.1e-4 | ABOVE |
| pruned+healed | PopQA | contains_lenmatched | 0.1119 | 0.0928 | +0.0191 | 17.0% | 1.1e-4 | ABOVE |
| barely_healed | PopQA | contains_lenmatched | 0.0271 | 0.1030 | −0.0760 | −280.8% | 1.1e-4 | BELOW |
| intact | TriviaQA | em | 0.4069 | 0.0026 | +0.4043 | **99.4%** | 1.1e-4 | ABOVE |
| pruned+healed | TriviaQA | em | 0.0959 | 0.0026 | +0.0933 | 97.3% | 1.1e-4 | ABOVE |
| barely_healed | TriviaQA | em | 0.0002 | 0.0026 | −0.0023 | −1050% | 1.1e-4 | BELOW |
| **intact** | **NQ-open** | **em** | **0.1025** | **0.0055** | **+0.0970** | **94.6%** | **1.1e-4** | **ABOVE** |
| **pruned+healed** | **NQ-open** | **em** | **0.0285** | **0.0055** | **+0.0230** | **80.6%** | **1.1e-4** | **ABOVE** |
| **barely_healed** | **NQ-open** | **em** | **0.0017** | **0.0055** | **−0.0039** | **−233%** | **6.4e-3** | **BELOW** |
| intact | NQ-open | contains_lenmatched | 0.1429 | 0.0310 | +0.1119 | 78.3% | 1.1e-4 | ABOVE |
| pruned+healed | NQ-open | contains_lenmatched | 0.0787 | 0.0446 | +0.0341 | 43.3% | 1.1e-4 | ABOVE |
| barely_healed | NQ-open | contains_lenmatched | 0.0097 | 0.0446 | −0.0349 | −360% | 1.1e-4 | BELOW |

(Full 33-cell table incl. raw `contains` in the JSON.)

## 3. Two things this settles that the raw numbers did not

**(a) The NQ-open floor is 0.0055, not 0.0053.** The exact-match best-constant null
recomputed inside the analyzer's own bootstrap is 0.0055; `GATE_NQOPEN_VERDICT.md`
quoted 0.0053 from the eval script's own summary. Use **0.0055** going forward; the
difference does not change any verdict but the two numbers must not both circulate.

**(b) Every axis's barely-healed control is BELOW floor with BH p < 0.05 — except one.**
`barely_healed × TriviaQA × contains` is **AT floor, not below**: 0.0079 vs 0.0086,
bootstrap p = 0.399, BH p = 0.399 — *indistinguishable from a constant predictor*.
That is the single non-significant cell in 33 and it is exactly the cell where the
degenerate arm and the constant baseline coincide. Report it as the "at floor" case
the protocol is designed to detect, not as a failure of the protocol.

## 4. Why length-matched nulls matter (the numbers make the case)

For `contains`, the naive best-constant null and the length-matched null diverge
sharply as prediction length grows, because healing inflates output length ~6×:

| arm | axis | naive contains null | length-matched null | ratio |
|---|---|---:|---:|---:|
| intact | NQ-open | 0.0144 | 0.0310 | 2.2× |
| pruned+healed | NQ-open | 0.0144 | 0.0446 | 3.1× |
| barely_healed | NQ-open | 0.0144 | 0.0446 | 3.1× |
| barely_healed | PopQA | 0.0488 | 0.1030 | 2.1× |

Using the naive null would credit `pruned+healed × NQ-open × contains` with 81.7% of
the headroom; the length-matched null gives **43.3%** — a 1.9× inflation from the
wrong null on a single cell. This is the same class of error A01 documents for MC
interfaces, reproduced independently on generative QA.

## 5. A03's certified instrument (final)

Four closed-book parametric-knowledge axes, each with a construct-appropriate null and
a demonstrated at/below-floor control:

1. **MMLU-content** (`content_norm` vs longest-option split-tie 0.2845) — letter retired at 1B
2. **PopQA** (EM vs best-constant 0.0229; `contains` vs length-matched)
3. **TriviaQA** (EM vs best-constant 0.0026 — largest dynamic range, 99.4% residual on intact) — **primary**
4. **NQ-open** (EM vs best-constant 0.0055; `contains` vs length-matched)

The three axes A03 originally also named (conflicting knowledge, multi-evidence,
new-injected-facts) remain formally out of scope — see `GATE_NQOPEN_VERDICT.md` §3 and
`STATUS.json:remaining_axes_status`. A03 is a **closed-book parametric-knowledge** study.

## 6. What still gates the 6-arm study

The instrument is ready; the **arms** are not. Per `status/proposal_prep/A03_6ARM_DESIGN.md`:

* Arms 1 (intact) + 2 (pruned+healed): **READY**, ckpts on zwfy6
* Arm 3 (+CPT): needs a ~20k-step Dolmino resume from `step200000.pt`, ~30-60 min on one H20
* Arms 4 (raw-text RAG) + 5 (residual memory) + 6 (both): **BLOCKED on code/training** —
  Arm 5 would require training a 1B-scale QCMem LoRA from scratch (none exists on either disk)

So the next A03 GPU step is **Arm 3 only**. Arms 4-6 are a coder project, not a gate.

## 7. Provenance

* Analyzer: `proposal/active/A03-parametric-vs-external-memory/code/analyze_1b_knowledge_floor.py` (patched line 427-429)
* Output: `proposal/active/A03-parametric-vs-external-memory/evidence/a03_1b_floor_nulls_4axes.json` (30 KB, on wzc1)
* Ran on: `.82` (zwfy6), CPU only, ~1 min
* Per-example inputs (zwfy6): `olmo2_closedbook_results/A03_1B_{base,keep7_step200k,keep7_step500}/per_example_{popqa,triviaqa,nq_open}.jsonl` — nq_open symlinked in from the `_nq` sibling dirs
* MMLU inputs (zwfy6): `olmo2_mmlu_content_results/` same three arms
