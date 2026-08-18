---
id: B04
check: novelty
date: 2026-08-09
outcome: hold_in_backlog (needs 2nd family + mechanism-level hypothesis; narrow OLMo-2 finding is not preempted but not stand-alone-paper strength)
---

# B04 Novelty Check — OLMo-2-narrowed "damage × acc_norm margin fingerprint"

> ## ⚠️ 2026-08-14 CROSS-REFERENCE BANNER — the Qwen "kill" cited below was downgraded on 2026-08-10
>
> This document was written **2026-08-09**, one day before the Qwen leg was re-analysed.
> Its *verdict* (`hold_in_backlog`) is still current and `STATUS.json` says so explicitly
> ("The hold_in_backlog novelty verdict of 2026-08-09 is unaffected"). But two statements
> in the body are stale, and one of them is **actionable**:
>
> 1. Where the body says the general claim was **"killed"** / the "Qwen replication
>    failed" (§Claim under test, and the Model-scope row of the differentiation table):
>    `STATUS.json.kill_history[1]` downgraded that verdict `GENERAL_CLAIM_KILLED` →
>    **`NON_MATCHED_INCONCLUSIVE`** on 2026-08-10, because the Qwen ladder confounds
>    damage with training budget. Cross-family generality is **untested, not refuted**.
>    (This is not a promotion — the narrowing to OLMo-2-only stands either way.)
> 2. **Actionable:** the resurrection gate below asks for a second family **"NOT Qwen"**.
>    That clause was **WITHDRAWN** on 2026-08-10 (`STATUS.json.resurrection_conditions[0]`,
>    `status_note_2026_08_10.qwen_readmitted_as_candidate_family`) on the grounds that
>    Qwen was never fairly tested. Qwen **is** re-admitted, but only via a
>    budget-matched ladder meeting the six conditions in
>    `status_note_2026_08_10.matched_ladder_requirements`. Do not exclude Qwen on the
>    strength of the text below.
>
> Also disclosed by the same 2026-08-10 note and **not** reflected below: the OLMo ladder
> is itself budget-varying, Spearman(core6, heal_steps) = **+0.8721** across its 5 damaged
> rungs. It is defensible only because damage depth spans five values there and is a
> perfect rank predictor. **Quote the +0.8721 wherever the +1.00 is quoted.**
>
> Nothing in the body has been edited — the scientific text is left as the dated record of
> what was concluded on 2026-08-09. `STATUS.json` is the authority on current standing.

## Claim under test

On a 6-rung damage ladder (`base + shortgpt16 + keep{14,12,10,8}`) of OLMo-2-7B, aggregate core6 degradation is accompanied by per-item acc_norm margin compression: **Spearman(core6, median_margin) = +1.00** and **Spearman(core6, frac<0.005) = −1.00**, both at n=6 exact-permutation lower bound p = 0.0028. ~~General claim killed by Qwen3-8B replication (ρ=+0.43, p=0.42; ρ=−0.49, p=0.36).~~ Remaining claim is OLMo-2-family-specific.

> **CORRECTION 2026-08-17 (0 GPU) — the struck sentence above.** The Qwen leg is
> **`NON_MATCHED_INCONCLUSIVE`**, not a kill. `STATUS.json.kill_history[1]` downgraded
> `GENERAL_CLAIM_KILLED` → `NON_MATCHED_INCONCLUSIVE` on **2026-08-10** because the Qwen
> ladder confounds damage depth with training budget
> (`DIRECTION_A_QWEN_LADDER_CONFOUND_ADDENDUM.md`). The two ρ values themselves are not
> retracted — what is retracted is reading them as a *refutation*. Cross-family generality
> is **UNTESTED, not refuted**, and Qwen is re-admitted as a candidate family via a
> budget-matched ladder (`resurrection_conditions[0]`). The narrowing to OLMo-2-only stands
> either way, so the `hold_in_backlog` verdict is unaffected. This discharges the first of
> the two stale sentences the 2026-08-14 banner above flagged as actionable.
>
> **Also required whenever the +1.00 is quoted** (`kill_gate.mandatory_disclosure_on_any_report`):
> Spearman(core6, heal_steps) on the SAME ladder, **with the ladder named** — **+0.6669 (wzc1)**
> / **+0.8721 (zwfy6)** — plus σ̂ = 0.000541, R = 68.26, and the clause-5 φ. **φ is currently
> UNDEFINED** (the G1 read-out has not been filled; `--readout-only` returns rc=3
> `READOUT_ABSENT`). φ is *not* small and *not* large — it does not yet exist.

## Top 5 nearest candidates (venue-verified)

> ⚠️ **AMENDED 2026-08-15 (MAIN): this table was INCOMPLETE and its ranking was wrong.**
> The 2026-08-09 sweep and the 2026-08-14 gate design both missed an **ICML 2026 regular**
> paper with a **same-named metric on the same task family under the same damage operation**.
> It is added as row 0 below because it, not the TMLR paper, is the nearest candidate.
> The `hold_in_backlog` verdict at the top of this file **survives** (see row 0's last column),
> but any text that cited "top 5" as evidence of the residual must be re-read against row 0.
> Surfaced by the B02/B04 RELATED_WORK pass; venue re-verified by MAIN at
> `api2.openreview.net/notes/search` — `venueid=ICML.cc/2026/Conference`, `venue="ICML 2026 regular"`,
> `invitations` includes `Submission31462/-/Camera_Ready_Revision`, `pdate=1777576923498`.
> **DBLP still shows only `journals/corr/abs-2605-07271` (CoRR 2026), i.e. DBLP lags — OpenReview
> is the authority here**, exactly as [[venue-verify-must-use-openreview-2026]] says.

| # | Paper | Venue (verified) | What it covers | What B04 covers that it does NOT |
|---|---|---|---|---|
| **0** | **Understanding Performance Collapse in Layer-Pruned Large Language Models via Decision Representation Transitions.** arXiv:2605.07271, 2026-05-08 | **ICML 2026 regular** (OpenReview `venueid=ICML.cc/2026/Conference` + `Camera_Ready_Revision`; **DBLP lags at CoRR 2026**) | Defines **"Decision Margin"** verbatim as the probability gap between gold and the best alternative, **on MC tasks, under layer pruning**, on Llama3-8B / Llama2-7B / Qwen3-4B, and reports it collapsing. **2026-05-08 → NOT concurrent; real prior art.** | Its DM is a **layer-wise trajectory within one network**; B04's is a **cross-checkpoint distribution over 17,195 items**. Full text greps `heal` / `retrain` / `continued-pretrain` = **0** (bounded SFT only), `spearman` = **0**, `per-item` = **0**. It satisfies none of B04's three `kill_definition` conditions jointly. ⚠️ **But it costs B04 three must-not-claim items** — B04 may no longer claim "Decision Margin", nor margin collapse under pruning on MC tasks, as its own. What remains is the prune-**then-heal** ladder + the joint Spearman fingerprint. |
| 1 | Tropeano, Maistro, Ruotsalo, Lioma. "Don't Go Breaking My LLM: The Impact of Pruning Attention Layers on Explanation Faithfulness and Confidence Calibration." arXiv:2606.24970 | **TMLR 2026** (OpenReview venueid `TMLR`, `Camera_Ready_Revision` present) | 5 LLMs × 8 datasets; attention-layer pruning degrades ECE/Brier calibration and faithfulness even when accuracy stays stable | Aggregate ECE/Brier only; no per-item acc_norm margin field, no joint Spearman(core6, margin-density) fingerprint, no depth-prune-then-heal ladder |
| 2 | Madaan, Yuret, Hupkes et al. "Quantifying Variance in Evaluation Benchmarks." arXiv:2406.10229 | **NeurIPS 2024 RegML Workshop** (OpenReview venueid `NeurIPS.cc/2024/Workshop/RegML`) | Seed variance and monotonicity of benchmark scores across training; explicitly notes IRT and item analysis fail to reduce MMLU variance; continuous-vs-discrete framing | No structural damage axis (only seed / training-step variance); no acc_norm margin decomposition; no prune-heal ladder |
| 3 | Men, Xu, Zhang et al. "ShortGPT: Layers in Large Language Models are More Redundant Than You Expect." arXiv:2403.03853 | **Findings of ACL 2025** (DBLP `conf/acl/MenXZYWL0HC25`, `booktitle=ACL (Findings)`, `10.18653/v1/2025.findings-acl.1035`) | Block Influence layer-drop pruning method; reports aggregate MMLU / MC accuracy vs #layers removed | Aggregate acc only; no per-item margin distribution; no joint fingerprint |
| 4 | Kim, Kim, Kim et al. "Shortened LLaMA: A Simple Depth Pruning for Large Language Models." arXiv:2402.02834 | **arXiv preprint, venue unverified** (DBLP `journals/corr/abs-2402-02834`, `Informal and Other Publications`) | Depth-prune LLaMA + continued-pretraining recovery, compares retraining methods; aggregate MC / reasoning benchmarks | The core prune-then-heal setup B04 uses, but again aggregate score only; no per-item margin fingerprint |
| 5 | Hong, Bhagia, Sun et al. "Fluid Language Model Benchmarking." arXiv:2509.11106 | **COLM 2025** (OpenReview venueid `colmweb.org/COLM/2025/Conference`) | Adaptive / IRT-based benchmarking of LLMs; per-item difficulty and information for adaptive picks | No damage axis at all; per-item lens is IRT difficulty, not acc_norm margin under structural damage |

## Also-noted (less adjacent, checked to be safe)

- **Fu et al., "Multiple Choice Questions: Reasoning Makes LLMs More Self-Confident, Especially When Wrong."** arXiv:2501.09775, IEEE Intelligent Systems 2025. Per-item MCQ confidence, but the perturbation axis is CoT-vs-direct prompting, not structural damage of the model.
- **Su et al., "A Single Character Can Make or Break your LLM Evals."** arXiv:2510.05152. Delimiter fragility of MMLU. ICLR 2026 Rejected Submission (OpenReview venueid `ICLR.cc/2026/Conference/Rejected_Submission`) — arXiv preprint, venue unverified. Same "fragility" umbrella but on prompt-format axis, not damage.
- **Kim, Yun, Kim et al., "Rethinking Layer Redundancy."** arXiv:2604.24938. Calibration-data > search-algorithm for depth pruning. arXiv preprint, no venue metadata yet. Aggregate score only; no per-item margin.
- **Ghosted Layers.** arXiv:2605.15491, preprint. Recovery method for layer-pruned LLMs. Different problem (fix-up), not diagnostic.

## Differentiation table

> ⚠️ **REGENERATED 2026-08-17 (MAIN, 0 GPU).** The 2026-08-09 version of this table had
> **no column for arXiv:2605.07271** (Shi et al., ICML 2026 regular), which row 0 of the
> top-5 table identifies as the *nearest* candidate. A differentiation table missing its
> nearest comparator cannot support the `hold_in_backlog` verdict it is cited for. The
> 2605.07271 column is now added **as the leftmost comparator** (nearest first), and the
> `Model scope` row's stale "Qwen replication failed" parenthetical is corrected.
> This discharges `STATUS.json:318 related_work.actionable_0_gpu_followup`, whose other
> half (the top-5 table) was already done on 2026-08-15.
> The original 6-column table is preserved verbatim in git history; **the verdict does not
> change**, but the residual it protects is now visibly narrower — see the ⚠️ row.

| Axis | **Shi 2026 (ICML, 2605.07271) ← NEAREST** | Tropeano 2026 (TMLR) | Madaan 2024 (RegML-WS) | ShortGPT 2025 (Findings-ACL) | Shortened LLaMA (arXiv) | Fluid Bench (COLM 2025) | **B04 (OLMo-2-narrowed)** |
|---|---|---|---|---|---|---|---|
| Damage axis | layer prune, **no continued-pretrain heal** (greps: `heal`/`retrain`/`continued-pretrain` = 0; bounded SFT only) | attention-layer prune, no heal | seed / step | depth-prune, minimal heal | depth-prune + heal | none | depth-prune + heal, 6-rung |
| Lens | **per-item MC "Decision Margin"** — gold minus best alternative, i.e. **the same lens** | ECE / Brier (aggregate calibration) | seed variance, monotonicity | aggregate accuracy | aggregate accuracy | IRT difficulty | **per-item acc_norm margin distribution** |
| Metric | Decision Margin as a **layer-wise trajectory within one network** | ECE, Brier, faithfulness | seed-variance, IRT-fail rate | MMLU %, other MC % | MMLU / commonsense % | IRT-adaptive score | **Spearman(core6, median_margin) & Spearman(core6, frac<threshold)** |
| Item pool | MC tasks on Llama3-8B / Llama2-7B / Qwen3-4B | 8 datasets (unspecified per-item lens) | MMLU inter alia | MMLU + reasoning | MMLU + commonsense | LM-eval-harness tasks | core6 (H+ARC-E+ARC-C+PIQA+OBQA+WG), N=17,195 pooled |
| Statistic | descriptive collapse curves (greps: `spearman` = 0, `per-item` = 0) | descriptive shifts | descriptive | descriptive | descriptive | IRT posterior | **exact-permutation p at n=6 lower bound** |
| Cross-checkpoint? | **No** — within-network across depth | no | across seeds/steps | no | no | no | **Yes** — across 6 healed checkpoints |
| Noise floor measured? | not reported | no | seed variance (its own subject) | no | no | no | **σ̂ = 0.000541 from a held-fixed seed pair; R = 68.26** |
| Model scope | Llama3-8B, Llama2-7B, Qwen3-4B | 5 LLMs, per-family aggregate | multiple LMs | LLaMA family | LLaMA | many | **OLMo-2-7B only (cross-family UNTESTED — the Qwen leg is `NON_MATCHED_INCONCLUSIVE`, not a failed replication; see the 2026-08-17 correction under "Claim under test")** |
| ⚠️ Effect on B04's residual | **Costs B04 three must-not-claim items** (`STATUS.json:313`): (8) the per-item MC decision margin as a pruning lens, or that pruning compresses it; (9) damage/phase-dependent perturbation sensitivity as novel; (10) recovery-bounded-by-remaining-depth. **NOT a kill** — fails `kill_definition` clauses 1 and 3 (no heal ladder; no co-variation with aggregate score across checkpoints). | not a kill | not a kill | not a kill | not a kill | not a kill | **Residual after row 0: the margin as a cross-checkpoint RANK fingerprint on a prune-THEN-heal ladder against a measured noise floor, plus clause 5's heal-budget discrimination. Narrower than this file recorded on 2026-08-09.** |

The joint fingerprint **`{Spearman(core6, median_margin), Spearman(core6, frac<0.005)}` at exact-permutation lower bound p** on a `keepN + shortgpt16` prune-heal ladder is not covered by any of the above. ~~Tropeano's ECE/Brier lens is the closest but is measurement-family disjoint~~ — **corrected 2026-08-17: Tropeano is no longer the closest.** Shi 2026 (2605.07271) is measurement-family *identical* (same per-item MC margin definition, same layer-pruning damage operation), and the differentiation now rests entirely on the axes marked in bold in its column: **no heal ladder, no cross-checkpoint rank statistic, no measured noise floor.** That is a narrower and more fragile residual than a disjoint-lens argument, and it is the honest one. Tropeano remains the closest *peer-reviewed-with-a-disjoint-lens* comparator, which is a weaker claim than the one struck here.

## MAIN independent verification (2026-08-09, after subagent delivery)

MAIN re-verified all five venue claims from scratch rather than accepting them:

| claim | MAIN's independent check | result |
|---|---|---|
| Tropeano = TMLR 2026 | OpenReview `/notes/search` → `venue: "Accepted by TMLR"`, `venueid: TMLR` | ✅ CONFIRMED |
| ShortGPT = **Findings**-ACL 2025 (not main) | DBLP bibtex `conf/acl/MenXZYWL0HC25` → `booktitle = {Findings of the ACL, ACL 2025}`, `series = {Findings of ACL}`, pages 20192–20204, doi `10.18653/v1/2025.findings-acl.1035` | ✅ CONFIRMED, and the Findings-vs-main distinction is correct |
| Fluid Bench = COLM 2025 | arXiv `2509.11106` comment field = `COLM 2025` | ✅ CONFIRMED |
| Shortened LLaMA = arXiv, venue unverified | arXiv `2402.02834` comment: v1 accepted at **ICLR 2024 Workshop ME-FoMo** (`openreview.net/forum?id=18VGxuOdpu`); v2 (the version with continued-pretraining, which is the part relevant to B04) has no venue | ✅ subagent's "unverified" is right, **with a refinement** — see below |
| Madaan = NeurIPS 2024 RegML Workshop | OpenReview search returns THREE records for this title: `RegML 2024` (`NeurIPS.cc/2024/Workshop/RegML`), CoRR 2024, and **`ICLR.cc/2025/Conference/Rejected_Submission`** | ✅ workshop claim CONFIRMED; ⚠️ note the ICLR-2025 rejection below |

Two refinements MAIN adds:

1. **Shortened LLaMA's venue is version-dependent.** arXiv-v1 was accepted at the ICLR 2024 ME-FoMo *workshop*; arXiv-v2 added the continued-pretraining-for-severe-pruning-ratios results — and v2 is precisely the part that overlaps B04's prune-then-heal setup. So the overlapping content is **not** the peer-reviewed content. This makes it a *weaker* preemption threat than "arXiv preprint" alone conveys, and it should be cited as `arXiv:2402.02834v2 (workshop acceptance applies to v1 only)`.

2. **Madaan was rejected at ICLR 2025 after the RegML workshop.** The workshop version is legitimately citable, but if B04 ever leans on Madaan's "IRT/item analysis fails to reduce MMLU variance" claim as *support*, that claim did not survive full peer review at a main venue. Cite it as a workshop paper and do not treat its negative result about item analysis as settled — which, if anything, leaves more room for B04's per-item margin lens.

Neither refinement changes the HOLD verdict; both make the novelty case marginally stronger.

## What would kill the direction

A paper is a kill iff **all** hold:
1. Uses depth-prune-then-heal (not just prune-only or attention-only), on ≥3 rungs;
2. Reports per-item acc_norm (or equivalent length-normalized) margin distribution on MC tasks; and
3. Reports either median_margin or `frac(margin<threshold)` co-varying with aggregate score on that ladder.

None of the top-5 or the also-noted set satisfies all three. Concurrent (≤3 months, arXiv only) same-lens work on a **different** family is not a kill — differentiation, not preemption.

## Bottom-line recommendation: **HOLD IN BACKLOG**

Rationale, in order of weight:
- Novelty is intact: the specific per-item acc_norm margin fingerprint on a prune-heal damage ladder is not covered by any peer-reviewed prior work I can verify. The closest peer-reviewed paper (Tropeano 2026 TMLR) uses a different measurement family (ECE/Brier vs per-item margin) and no heal step.
- **But** the OLMo-2-only scope after the Qwen kill is too narrow for stand-alone promotion. Per the internal resurrection gate (2nd confirming family AND mechanism-level hypothesis), the current result is a legitimate methods appendix but not a paper.
- Recommended next moves (not part of this novelty check): (a) attempt a 2nd non-Qwen family — e.g. Llama-3 or Mistral prune-heal ladder — and (b) propose a mechanism-level hypothesis (e.g. fresh-layer-init variance ↔ margin dispersion) that can be tested a priori. If (a) confirms and (b) is falsifiable, then promote to `paper<X>`. If (a) also fails, archive with `POSTMORTEM.md`.

## Confidence: **medium-high**

- Venue verification is solid: TMLR / Findings-ACL / RegML-WS / COLM all verified via authoritative sources (OpenReview venueid + Camera_Ready_Revision for OpenReview family, DBLP + Anthology for ACL family).
- I have not exhaustively swept knowledge-decay / catastrophic-forgetting literature at the per-item lens; a diligent adversary could still surface a niche paper. But the closest peer-reviewed adjacent work (Tropeano) has a genuinely disjoint measurement family, which is the strongest evidence of non-preemption.
- Search limitation: Semantic Scholar API hit heavy rate-limiting; arXiv API + OpenReview API + DBLP were the workhorses. Any candidate that lives only on S2's private index and neither arXiv nor OpenReview could have been missed — low probability for a topic this instrumentation-heavy.
