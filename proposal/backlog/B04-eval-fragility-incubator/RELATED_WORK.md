# B04 — RELATED WORK / NOVELTY BOUNDARY (narrowed claim, undecided gate)

**Written 2026-08-15. 0 GPU, 0 SSH. Adjudication + venue verification only.**

This closes the blocker `proposal/ready_queue.py:542-554` trips on
(`RELATED_WORK.md absent (blocks PROMOTION; 0-GPU task)`) and discharges the Related Work
item `proposal/shared/literature/RELATED_WORK_GAP_AUDIT_20260808.md:94` assigns B04
(rating **不足** / insufficient; required families: *numerical/hardware/batch
nondeterminism; benchmark ranking instability; margin/calibration; damage robustness;
mediation*; and the binding boundary: **"新意只能是 model damage 是否系统性放大 nuisance
sensitivity"** — the only admissible novelty is whether model damage *systematically
amplifies* nuisance sensitivity).

---

## ⚠️ 0. READ THIS FIRST — TWO THINGS THIS FILE IS NOT ALLOWED TO PRETEND

### 0.1 The general claim is dead. Only an OLMo-2-only claim survives.

`STATUS.json:3` — `"status": "NARROWED_TO_OLMO_2_ONLY"`.
`memory/direction-a-eval-fragility-established.md` records the correction bluntly: the
**general claim was killed by the Qwen cross-family replication** (ρ = +0.43, p = 0.42;
ρ = −0.49, p = 0.36 vs OLMo's ±1.00 at exact p = 0.0028). The novelty check is
**`hold_in_backlog`** (2026-08-09), not cleared-for-promotion.

**Nuance that must travel with that sentence** (`STATUS.json:14-24`, `status_note_2026_08_10`):
the Qwen *kill* was itself **downgraded** on 2026-08-10 from `GENERAL_CLAIM_KILLED` to
**`NON_MATCHED_INCONCLUSIVE`**, because the Qwen ladder confounds damage depth with training
budget (among its 5 damaged rungs, Spearman(core6, heal_steps) = **+0.8944** while
Spearman(core6, layers_kept) = **−0.3536** — budget out-predicts damage *and* has the better
sign; `layers_kept` takes only **two** values there). So cross-family generality is now
**UNTESTED, not refuted.**

**That is not a promotion, and this file does not treat it as one.** `STATUS.json:26`
verbatim: *"removing a failed kill does not manufacture a successful replication. Before the
downgrade cross-family generality was refuted; after it, untested. Both are equally far from
established, and 'untested' is not a promotion argument."*

Everything below is therefore written against the **OLMo-2-7B-only** claim:

> On the OLMo-2-7B `keepN + shortgpt16` prune-then-heal ladder, aggregate `core6`
> degradation is accompanied by **per-item `acc_norm` decision-margin compression**.
> `Spearman(core6, median_margin) = +1.0000`, exact two-sided `p = 0.0028` (the n=6
> permutation floor, 2/720); `Spearman(core6, frac(margin<0.005)) = −1.0000`, same `p`.

**Mandatory co-disclosure on every quotation of that ρ** (`STATUS.json:132`,
`DIRECTION_A_VERDICT.md:39`): print beside it **(i)** `Spearman(core6, heal_steps)` for the
*same* ladder **with the ladder named** — **+0.6669** on the wzc1 sm_100 ladder,
**+0.8721** on the zwfy6 `_bs16` ladder (they differ because keep12 is step111500 there vs
step124000 here); **(ii)** `σ̂ = 0.000541` and `R = 68.26`; **(iii)** the `φ` from clause 5.
The OLMo leg is *less confounded*, **not unconfounded** — its licence to claim rests on
damage depth spanning **five** distinct values (8/10/12/14/16) and being a **perfect** rank
predictor (ρ = +1.0000), which is exactly what fails on Qwen.

### 0.2 The gate is NOT decidable yet. This file does not assume it will pass.

`STATUS.json:253` (`lifecycle_reason`), verbatim in substance:

> **3/3 adversarial lenses returned `NEEDS_REVISION`** (`wf_c2a3b490-e40`, 2026-08-14 23:0x).
> The **decidability** lens found the gate **NOT decidable**: `φ` used a rescale span of
> **116500** (the *damaged ladder's* heal-step span) while the read-out's own span is
> **175000**, so *the statistic as written was not the measured quantity* — it understated
> the excursion by exactly `175000/116500 = 1.5021×`, **in the direction that favours B04's
> own hypothesis.** The **falsifiability** and **affordability** lenses each demanded a
> different decision statistic.

Revision 2 (2026-08-14, 0 GPU, PRE-DATA) applied all three fixes on disk, and
`lifecycle` **deliberately remains `ready_cpu`** — promotion is the *next independent
adversarial pass's* call, not the reviser's. `STATUS.json:247`: *"a fix is not a verdict."*

**Consequence for this document:** the only clause that can still kill B04 —
**clause 5, budget discrimination at fixed damage** — has **never been run**. Every claim
below is therefore conditional, and §4's safe residual is written as a claim about a
*measured association* plus an *explicitly open* attribution, never as an established causal
statement about damage.

---

## 1. Verification discipline, and what was reachable from this node

Family-split venue verification; the wrong authority produces false calls both ways:

* ICLR / NeurIPS / ICML / TMLR / COLM → **OpenReview `venueid`** (+ `Camera_Ready_Revision`)
  — `memory/venue-verify-must-use-openreview-2026.md`
* ACL / EMNLP / NAACL / EACL **including Findings** → **ACL Anthology + DBLP**
  — `memory/venue-verify-acl-family-needs-anthology.md`
* Non-CS journals → **Crossref DOI**
* **`arXiv-only` means "no peer-reviewed venue verifiable from this node", NOT "no venue
  exists".**

### 1.1 Endpoint status, measured 2026-08-15

| Endpoint | Status | Consequence |
|---|---|---|
| **OpenReview v2** `api2.openreview.net/notes/search` | ✅ **works, and returns `invitations`** | `Camera_Ready_Revision` **was** verifiable this session — see §2.3 (TMLR ×2, ICML 2026, COLM 2025, ICML 2024). Contrast with the A01 pass, which recorded api2 as 403 everywhere. |
| **OpenReview v2** `notes?id=` / `?forum=` | ❌ HTTP 403 `ChallengeRequiredError` (`reqId 2026-08-14-9997156`) | no forum enumeration; a title-match miss cannot be recovered by forum lookup |
| **OpenReview v1** `api.openreview.net/notes/search` | ✅ works but **index is stale for 2026** | not used as an authority where v2 answered |
| **ACL Anthology** `.bib` | ✅ works | authority for Findings-vs-main |
| **DBLP** `search/publ/api` + `rec/<key>.bib` | ✅ works, **intermittent HTTP 500** under rapid queries | all `total = 0` were re-run ≥1× with a 4 s delay |
| **arXiv API** (https) + `arxiv.org/html/<id>` | ✅ works | metadata, abstracts, and **one full-text HTML pass** (§2.1) |
| **Semantic Scholar** | ❌ HTTP 429 | **never used as an authority** |
| **Crossref** | ✅ works | not needed (no non-CS venues in B04's scope) |

---

## 2. Named closest collisions, by the audit's five families

### 2.1 ⚠️ THE COLLISION THE PRIOR NOVELTY CHECK DID NOT HAVE — and it is peer-reviewed

**Shi, Liu, Gao, Yang, Geng — *Understanding Performance Collapse in Layer-Pruned Large
Language Models via Decision Representation Transitions* — arXiv:2605.07271v1, 2026-05-08.**

* **Venue: ICML 2026 (regular).** Verified this session via **OpenReview v2
  `notes/search` → `venue = "ICML 2026 regular"`, `venueid = ICML.cc/2026/Conference`, and
  `Camera_Ready_Revision` present in the `invitations` list.** DBLP still shows only
  `journals/corr/abs-2605-07271` (`CoRR 2026`) — **DBLP lags; OpenReview is the authority
  for this family**, exactly as `memory/venue-verify-must-use-openreview-2026.md` says.
* **Full text read this session** (`arxiv.org/html/2605.07271v1`, 312 KB, tag-stripped).
* **What it does** — and this is uncomfortably close to B04's lens:
  * Introduces **“Decision Margin” (DM)**, defined verbatim as *"the probability gap between
    the ground-truth option and the most likely alternative"*, **on multiple-choice tasks**.
    That is, up to `logprob` vs `probability` and length normalisation, **the same
    quantity B04 calls `median_margin`.**
  * Introduces **Option Frequency (OF)** and an **Iterative Pruning (IP)** greedy
    layer-by-layer procedure, explicitly to avoid the block-pruning confound in ShortGPT
    (it cites Men et al. 2025).
  * Finds a **sharp decision transition** partitioning the stack into a **Silent Phase** and
    a **Decisive Phase**; pruning the Decisive Phase is near-harmless, pruning the Silent
    Phase triggers immediate collapse.
  * §4.3.1 **“Phase-Dependent Sensitivity to Stochastic Perturbations”**: injects noise
    `h'_l = h_l + ε·σ(h_l)·N(0,I)` and monitors **DM**, finding *"a striking asymmetric
    vulnerability"* — the Silent Phase degrades at `σ² = 0.02` while the Decisive Phase holds
    at `σ² = 0.1`.
  * §4.3.3 **“Bounded Recovery via Supervised Fine-Tuning”**: SFT improves DM everywhere but
    recovery is *"strictly bounded by the remaining structural depth"*; beyond ~40.6 %
    pruning the improvement plateaus. Models: Llama3-8B, Llama2-7B, Qwen3-4B; tasks include
    ARC-Easy, ARC-Challenge, HellaSwag.
* **Why this is the strongest collision in B04's history, stated plainly:** it is
  **peer-reviewed at a top venue**, it uses **a per-item MC decision margin**, it varies
  **layer-pruning depth**, it reports the margin **co-varying with pruning**, and its
  §4.3.1 is a **damage × perturbation-sensitivity interaction** — the audit's designated
  "only admissible novelty" family. A reviewer will find this paper. **B04 must cite it in
  its first paragraph.**
* **Why it is nevertheless NOT preemption** — four differences, each checked against the
  full text, not guessed:
  1. **No heal step.** Full-text counts: `heal` **0**, `retrain` **0**,
     `continued pretrain`/`continued pre-train` **0**. Its only recovery mechanism is
     **SFT** (§4.3.3), and it is presented as a *bounded remedy*, not as the axis of the
     ladder. B04's kill definition (`NOVELTY_CHECK.md:95-99`) requires
     **depth-prune-then-heal on ≥3 rungs**; this paper's rungs are **prune-ratio rungs on an
     unhealed model**. Its own ladder is therefore **not budget-confounded** — and equally,
     it cannot speak to B04's live question at all.
  2. **Its margin is the *layer-wise* trajectory of one model; B04's is the
     *cross-checkpoint* distribution over items.** DM in 2605.07271 is read at **every layer
     `l` of a single network** to locate a transition point. B04's `median_margin` is read
     **once per healed checkpoint**, over `n = 17,195` pooled `core6` items, and correlated
     **across rungs**. Different index, different statistic, different question: theirs is
     *"where in depth does the decision form?"*, B04's is *"does damage compress the margin
     distribution of the finished model?"*.
  3. **No rank statistic and no noise floor.** Full-text `spearman` → **0**, `per-item` →
     **0**. It reports curves and figures descriptively. B04's claim is an **exact-permutation
     Spearman at the n=6 floor** with a **σ̂ from a same-damage same-step seed pair** and a
     `range/σ̂ = 68.26` admissibility check. `NOVELTY_CHECK.md`'s kill clause (3) requires
     the margin to be *reported co-varying with aggregate score on that ladder* **as a
     statistic**; a figure is not that.
  4. **Its nuisance is injected Gaussian noise in the hidden state; B04's is an
     evaluation-harness nuisance** (batch size / arch / seed). These are different objects:
     theirs is a **model-internal robustness probe**, B04's is **measurement fragility of a
     reported benchmark number**. The audit's boundary — *does damage amplify **nuisance**
     sensitivity* — is about the second.
* ⚠️ **What this collision DOES cost B04**, and this is the honest part:
  * **B04 may no longer claim the per-item MC decision margin as its own lens under
    layer pruning.** ICML 2026 owns "Decision Margin", by that name, on MC tasks, under
    layer pruning, three months before this pass. **This is not concurrent** (2026-05-08 →
    2026-08-15 is >3 months).
  * **B04 may no longer claim first observation that pruning compresses MC decision
    margins.** Its §4.2 does that.
  * **B04 may no longer claim first observation of phase/damage-dependent perturbation
    sensitivity.** Its §4.3.1 does that, for injected noise.
  * **What is left**: the margin as a **cross-checkpoint fingerprint on a prune-*then-heal*
    ladder**, as a **rank statistic against a measured noise floor**, and the **heal-budget
    discrimination** clause 5 exists to test. **That residual is narrower than
    `NOVELTY_CHECK.md` (2026-08-09) believed**, because that pass never located this paper.
  * ✅ **What this collision GIVES B04** — per `memory/prior-work-differentiate-dont-abandon.md`
    (2026-08-12 强化: *"印证+组合本身就是贡献"*): 2605.07271 is **independent, peer-reviewed
    corroboration on three other model families (Llama3-8B, Llama2-7B, Qwen3-4B) that the MC
    decision margin is the right lens on layer damage.** B04's Qwen leg was
    `NON_MATCHED_INCONCLUSIVE`; this paper supplies external cross-family support for the
    *lens*, while leaving the *heal-budget* question — the one B04 is actually gated on —
    entirely open, since it has no heal axis. **Cite it as corroboration, not as a threat.**

### 2.2 Family A — numerical / hardware / batch nondeterminism

| # | Work | Venue (authority) | What it does | Precise difference from B04 |
|---|---|---|---|---|
| A1 | **Introducing Background Temperature to Characterise Hidden Randomness in LLMs**, arXiv:2604.22411v1, 2026-04-24 | **TMLR (Feb 2026).** OpenReview v2 `notes/search` → `venue = "Accepted by TMLR"`, `venueid = TMLR`, forum `bz0he4bARF`, **`TMLR/Paper6133/-/Camera_Ready_Revision` present**; arXiv `journal_ref` agrees. | Formalises `T=0` nondeterminism as an **effective "background temperature" `T_bg`** induced by an implementation-dependent perturbation process (batch-size variation, kernel non-invariance, FP non-associativity); gives a protocol to estimate it. | **The peer-reviewed anchor for the whole family.** Establishes that the nuisance B04 measures **is a real, quantified phenomenon** — B04 may not claim it. Difference: `T_bg` is a property of the **inference environment**, measured on **intact** provider models; B04's question is whether **damage to the weights** changes an *evaluation verdict's* sensitivity. **No damage axis whatsoever.** |
| A2 | **Beyond Reproducibility: Token Probabilities Expose LLM Nondeterminism**, arXiv:2601.06118v1, 2026-01-03 | **arXiv-only.** DBLP `journals/corr/abs-2601-06118`, `CoRR 2026`. | Analyses GPU nondeterminism at the level of **token probabilities** rather than generated text; finds effects are **significant for probabilities in 0.1–0.9 and much smaller near 0 or 1**. | ⚠️ **Structurally the most dangerous alternative explanation for B04's finding, and B04 must address it head-on.** If nondeterminism bites hardest at mid-range probabilities, then a model whose margins are *compressed* has more items in the sensitive band **by arithmetic**, with no need for any damage-specific mechanism. **B04's "damage amplifies nuisance sensitivity" claim must be stated as an increment over this baseline, not as a discovery.** Difference: A2 has no damage axis and no MC-accuracy verdict; it never asks whether the *number a paper reports* moves. |
| A3 | **MarginGate: Sparse Margin-Triggered Verification for Batch-Invariant LLM Inference**, arXiv:2605.30218v1, 2026-05-28 | **arXiv-only.** DBLP `journals/corr/abs-2605-30218`, `CoRR 2026`. | Measures batch-induced token flips (`0.3–1.3 %` on MATH500/GSM8K/HumanEval; Llama-3.1-8B `0.48 %`) and finds **K/V perturbations stay flat before flips while low top-1/top-2 logit margin exposes much of the flip risk**; uses that to verify only low-margin steps. | ⚠️ **This is B04's mediation hypothesis, already published, on intact models.** "Low margin ⇒ flip risk" is exactly the mechanism B04's `frac(margin<τ)` metric encodes. **B04 may not claim the margin→flip link.** Differences: (i) generative decoding steps, not MC `acc_norm` verdicts; (ii) intact models only — no damage ladder; (iii) it is a *systems remedy*, B04 is a *measurement claim*. **And it sharpens a live problem for B04**: on B04's own harness, same-driver same-arch re-runs are **bit-deterministic (0 flips)** (`status/PAPERB_WITHIN_DISK_FLOOR_V3.md`), which is precisely why clause 1 of the original gate was retired as `UNTESTABLE` (`GATE_DESIGN.md` §0). MarginGate gets a non-zero flip denominator because it decodes generatively; B04's does not. |
| A4 | **LLM-42: Enabling Determinism in LLM Inference with Verified Speculation**, arXiv:2601.17768v2, 2026-01-25 | **arXiv-only.** DBLP `journals/corr/abs-2601-17768`, `CoRR 2026`. | Per-token verification to restore deterministic decoding. | Remedy, not diagnostic. No damage axis. Cited by A3 as the baseline it makes sparse. |
| A5 | **Deterministic Inference across Tensor Parallel Sizes…**, arXiv:2511.17826v2, 2025-11-21 | **arXiv-only.** | Batch/TP-invariant kernels. | Systems remedy; no evaluation-verdict or damage claim. |

**Family A verdict: the nuisance is established, quantified, and peer-reviewed (A1), and the
margin→flip link is published (A3).** B04's admissible residual is **only** the
`damage × nuisance` **interaction** — precisely as the audit ruled. And A2 supplies a
**null-mechanism** for that interaction which B04 has not yet excluded.

### 2.3 Family B — benchmark ranking instability

| # | Work | Venue (authority) | What it does | Precise difference from B04 |
|---|---|---|---|---|
| B1 | **Madaan, Yuret, Hupkes et al. — Quantifying Variance in Evaluation Benchmarks**, arXiv:2406.10229, 2024-06-14 | **NeurIPS 2024 RegML Workshop.** OpenReview v2 → `venue = "RegML 2024"`, `venueid = NeurIPS.cc/2024/Workshop/RegML`. ⚠️ **Also returns `ICLR.cc/2025/Conference/Rejected_Submission`** — it was **rejected at ICLR 2025** after the workshop. **Cite as a workshop paper; do not treat its negative result about IRT/item analysis as settled.** | Seed variance and **monotonicity during training** across many LMs; explicitly reports that IRT and item analysis **fail** to reduce MMLU variance. | The closest work on "benchmark numbers have variance you should quantify". Its axes are **seed and training step** — i.e. **B04's own confound**, not B04's factor of interest. No structural-damage axis, no per-item margin decomposition. Its ICLR rejection is why B04 must not lean on its IRT-fails claim as *support*. |
| B2 | **Hong, Bhagia, Sun et al. — Fluid Language Model Benchmarking**, arXiv:2509.11106 | **COLM 2025.** OpenReview v2 → `venue = "COLM 2025"`, `venueid = colmweb.org/COLM/2025/Conference`, **`Camera_Ready_Revision` present**; arXiv comment agrees. | IRT-based adaptive benchmarking; per-item difficulty and information used to pick items. | The per-item lens exists in the literature — but it is **IRT difficulty**, an *item* property estimated across models, not a *model* property (margin) estimated across items. **No damage axis at all.** |
| B3 | **Instance-level Randomization: Toward More Stable LLM Evaluations**, arXiv:2509.12678v1, 2025-09-16 | **Findings of EMNLP 2025.** DBLP `conf/emnlp/LiWLSQQCC25` → `booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025"`, pp. 3411–3425, DOI `10.18653/v1/2025.findings-emnlp.182`. ⚠️ **Findings, not EMNLP main.** | Theoretically decomposes variance from random factors (e.g. few-shot examples), then **randomises every factor per instance** and averages, to reduce variance and unfair comparison. | **Peer-reviewed and it owns "fixed nuisance settings make cross-model comparison unfair"** — which is B04's motivating premise. B04 may not claim it. Differences: its nuisance is **prompt-side randomness** (few-shot choice, ordering), not numerical/batch; its remedy is a **protocol**; and crucially **it does not ask whether the sensitivity differs by model condition** — it treats variance as a nuisance to average away, whereas B04's admissible claim is that the nuisance *magnitude itself* is informative about damage. |
| B4 | **Do Repetitions Matter? Strengthening Reliability in LLM Evaluations**, arXiv:2509.24086v1, 2025-09-28 | **arXiv-only.** DBLP `journals/corr/abs-2509-24086`, `CoRR 2025`. | Three independent runs on AI4Math; **mixed-effects logistic regression, rank-instability analysis, ICC**; finds **10/12 slices (83 %) invert ≥1 pairwise rank** vs the three-run majority, and two runs remove ~83 % of single-run inversions. | Closest on *rank instability as a measured quantity*, and it uses the same statistical furniture as B04's `status/ICC_DESIGN_EFFECT.md`. Differences: **stochastic decoding** is the variance source (B04's harness is bit-deterministic within arch), 8 intact frontier models, no damage axis, no per-item margin. |

### 2.4 Family C — margin / calibration under damage

| # | Work | Venue (authority) | What it does | Precise difference from B04 |
|---|---|---|---|---|
| C1 | **Tropeano, Maistro, Ruotsalo, Lioma — Don't Go Breaking My LLM: The Impact of Pruning Attention Layers on Explanation Faithfulness and Confidence Calibration**, arXiv:2606.24970v1, 2026-06-23 | **TMLR 2026.** **Re-verified this session** via OpenReview v2 → `venue = "Accepted by TMLR"`, `venueid = TMLR`, forum `VxZd6HfMOo`, **`TMLR/Paper7760/-/Camera_Ready_Revision` present**. (A second record, forum `HYemFPbg8k`, is the DBLP mirror `CoRR 2026` — do not mistake it for the venue.) arXiv comment: *"Accepted at TMLR"*. | 5 LLMs × 8 datasets: **attention-layer** pruning degrades **ECE / Brier / faithfulness even when accuracy stays stable**, and *"faithfulness and calibration can fluctuate significantly, even when accuracy remains stable"*. | Previously B04's closest peer-reviewed neighbour, and still the closest on the *calibration* axis. Differences (unchanged from `NOVELTY_CHECK.md` §1): **attention-only** pruning, **no heal step**, and **aggregate** calibration (ECE/Brier) rather than a **per-item margin distribution**. Measurement family disjoint. **But note the overlap in spirit is exactly B04's headline** — "the aggregate number hides a micro-structural shift" — so **B04 may not claim that framing as new.** |
| C2 | **Compressed Models are NOT Trust-equivalent to Their Large Counterparts**, arXiv:2508.13533v1, 2025-08-19 | **arXiv-only.** DBLP `journals/corr/abs-2508-13533`, `CoRR 2025`. | Two-dimensional trust-equivalence framework: **interpretability alignment** (LIME/SHAP) + **calibration similarity** (ECE, MCE, Brier, reliability diagrams). Explicit thesis: *"performance parity does not guarantee trust-equivalence."* BERT-base + compressed variants. | The sharpest available statement of B04's motivating premise, **already in the literature** and framed as a general principle. B04 may not claim "equal accuracy ≠ equal behaviour". Differences: encoder classification, BERT-scale, aggregate calibration metrics, **no depth-prune-then-heal ladder, no per-item margin, no rank statistic**. |
| C3 | **Decoding Compressed Trust: Scrutinizing the Trustworthiness of Efficient LLMs Under Compression**, arXiv:2403.15447v3, 2024-03-18 | **ICML 2024 Poster.** OpenReview v2 → `venue = "ICML 2024 Poster"`, `venueid = ICML.cc/2024/Conference`, **`Camera_Ready_Revision` present**. (Also indexed as a SeT-LLM@ICLR-2024 workshop record — cite the **ICML 2024** one.) | 3 LLMs × 5 compression methods × **8 trustworthiness dimensions**; finds **4-bit quantization retains trustworthiness while pruning degrades it even at 50 % sparsity**. | **The peer-reviewed anchor that damage type differentially affects non-accuracy behaviour.** B04 may not claim that. Differences: trustworthiness dimensions (safety, fairness, robustness), not decision margins; no heal ladder; no per-item statistic; no nuisance-sensitivity interaction. |
| C4 | **Men, Xu, Zhang et al. — ShortGPT: Layers in LLMs are More Redundant Than You Expect**, arXiv:2403.03853 | **Findings of ACL 2025.** **Double-verified this session**: Anthology `2025.findings-acl.1035.bib` → `booktitle = "Findings of the Association for Computational Linguistics: ACL 2025"`, pp. 20192–20204, DOI `10.18653/v1/2025.findings-acl.1035`; DBLP `conf/acl/MenXZYWL0HC25` → `series = {Findings of ACL}`, same pages/DOI. ⚠️ **Findings, NOT ACL main** — the exact trap `memory/venue-verify-acl-family-needs-anthology.md` warns about. | Block-Influence layer-drop pruning; aggregate MMLU / MC accuracy vs #layers removed. | **Supplies B04's own `shortgpt16` rung**, so it is a dependency as much as a collision. Aggregate accuracy only; no per-item margin; no fingerprint. (Also: 2605.07271 §1 criticises exactly this paper's *block* pruning for conflating phases.) |
| C5 | **Kim, Kim, Kim et al. — Shortened LLaMA: A Simple Depth Pruning for LLMs**, arXiv:2402.02834 | **Version-dependent.** v1 → **ICLR 2024 ME-FoMo *Workshop*** (`openreview.net/forum?id=18VGxuOdpu`); **v2 has no venue.** DBLP `journals/corr/abs-2402-02834`, `Informal`. ⚠️ DBLP returned HTTP 500 twice this session, so this row is **carried from `NOVELTY_CHECK.md`'s 2026-08-09 MAIN re-verification, not re-verified here.** | Depth-prune + **continued-pretraining recovery**, comparing retraining methods. | **The closest match to B04's prune-then-heal *setup*** — and `NOVELTY_CHECK.md` §"refinement 1" makes the load-bearing observation: **the continued-pretraining results (the part overlapping B04) are in v2, which is NOT the peer-reviewed version.** Aggregate scores only; no per-item margin. |
| C6 | **Small LLMs: Pruning vs. Training from Scratch**, arXiv:2606.14150v3, 2026-06-12 | **arXiv-only.** DBLP `journals/corr/abs-2606-14150`, `CoRR 2026`; OpenReview v2 `notes/search` → **0 matching notes**. | Prune-then-retrain vs from-scratch at matched small scale. | Named in A04's related-work pass as its closest methodological collision; for B04 it matters because it is the family that **must** control training budget. Aggregate scores; no margin lens. |

### 2.5 Family D — damage robustness

Covered by C1/C2/C3 above (calibration/trust under compression) and by §2.1 §4.3.1 (noise
sensitivity by phase). Two additional negative-search results are recorded rather than
rows, because they are what B04's residual rests on:

* `abs:"pruned" AND abs:"prompt sensitivity"` → **0 hits.**
* `abs:"quantized" AND abs:"prompt format" AND abs:"sensitivity"` → **0 hits.**
* `abs:"amplif*" AND abs:"sensitivity" AND abs:"pruning" AND abs:"evaluation"` → **0 hits.**
* `abs:"seed" AND abs:"variance" AND abs:"pruning" AND abs:"language model"` → **0 hits.**
* `abs:"interaction" AND abs:"damage" AND abs:"nuisance"` → **0 hits.**
* `abs:"training budget" AND abs:"confound" AND abs:"pruning"` → **0 hits.**
* `abs:"layer pruning" AND abs:"healing" AND abs:"per-item"` → **0 hits.**
* `abs:"margin" AND abs:"pruning" AND abs:"per-item"` → **0 hits.**
* `abs:"decision margin" AND abs:"pruning"` → **2 hits**, one being §2.1, one unrelated
  (PathMark watermarking).

**So the audit's designated novelty — *does damage systematically amplify nuisance
sensitivity* — returned no direct hit.** §4.1 states why that is nevertheless not enough.

### 2.6 Family E — mediation

The audit named mediation; `DIRECTION_A_VERDICT.md:92` already declares B04 is **NOT** a
mediation claim (*"that would need LOO across items, not addressed here"*), and
`GATE_DESIGN.md` §0 records that the LOO/constant-rate clause was **replaced by a floor
test** because it needs a flip endpoint that is untestable on a bit-deterministic harness.

Searches: `abs:"mediation" AND abs:"language model" AND abs:"evaluation"` → 8 hits, **all**
unrelated (clinical trials, flame-war moderation, sycophancy, activation steering). The
causal-mediation-in-LMs literature that *is* relevant is the causal-tracing family
(e.g. arXiv:2106.06087), which mediates *internal mechanisms*, not
*damage → benchmark verdict*.

**Verdict: the mediation family is a citation obligation only, and B04 must keep its
disclaimer.** The nearest published statement of the mediating variable B04 would use is
**A3/MarginGate** ("low margin exposes flip risk"), on intact models.

---

## 3. What the audit's boundary permits, given §2

Audit boundary, verbatim: *"新意只能是 model damage 是否系统性放大 nuisance sensitivity."*

| Component | Owned by | B04's status |
|---|---|---|
| The nuisance exists and is quantifiable at `T=0` | **A1, TMLR Feb 2026** | ❌ may not claim |
| Nondeterminism concentrates at mid-range probabilities | **A2, arXiv:2601.06118** | ❌ may not claim — **and this is a null mechanism B04 has not excluded** |
| Low margin ⇒ flip risk | **A3, arXiv:2605.30218** | ❌ may not claim |
| Benchmark ranks are unstable across runs; report uncertainty | **B4** (+ **B1** workshop, **B3** Findings-EMNLP) | ❌ may not claim |
| Fixed nuisance settings make cross-model comparison unfair | **B3, Findings of EMNLP 2025** | ❌ may not claim |
| Equal accuracy ≠ equal behaviour under compression | **C2**, **C3 (ICML 2024)**, **C1 (TMLR 2026)** | ❌ may not claim |
| **Per-item MC decision margin compresses under layer pruning** | **§2.1, ICML 2026** | ❌ **may no longer claim — NEW this pass** |
| **Damage-phase-dependent sensitivity to injected perturbation** | **§2.1 §4.3.1, ICML 2026** | ❌ **may no longer claim — NEW this pass** |
| Recovery from deep pruning is bounded | **§2.1 §4.3.3**; also A04's C2/N1 family | ❌ may not claim |
| **The margin as a cross-checkpoint rank fingerprint on a prune-THEN-HEAL ladder, tested as an exact-permutation Spearman against a σ̂ measured from a same-damage same-step seed pair, with heal budget discriminated at fixed damage** | — no located work | ✅ **the entire residual** |

**That last row is one sentence long, and it is conditional on a gate that has not run.**
This is the honest sizing, and it is narrower than `NOVELTY_CHECK.md` (2026-08-09) recorded
because that pass did not have §2.1.

---

## 4. Safe residual claim — one falsifiable sentence

> **On the OLMo-2-7B `keepN`+`shortgpt16` prune-then-heal ladder evaluated under one fixed
> harness on one architecture, per-item `acc_norm` margin compression is a *perfect rank
> co-variate* of aggregate `core6` degradation (`ρ = +1.0000`, exact two-sided
> `p = 0.0028`, `n = 6`), and the ladder's `median_margin` range (`0.021820`) exceeds the
> run-to-run floor measured from a same-damage same-step seed pair (`σ̂ = 0.000541`) by
> `40.3×` — a resolvable, reproducible micro-structural signature that aggregate accuracy
> alone does not expose. Whether that signature is attributable to DAMAGE DEPTH rather than
> to HEAL BUDGET is NOT YET DETERMINED and is exactly what clause 5 tests.**

**How to falsify it, clause by clause** (each has an instrument already on disk):

* **The rank claim** fails if `ρ ≠ +1.0000` or `p > 0.0028`. Measured 2026-08-14 on the wzc1
  sm_100 ladder: `ρ = +1.0000`, `p = 0.0028`. Banked
  (`STATUS.json.kill_gate.clause_2_exact_test`).
* **The resolvability claim** fails if the damaged-ladder range is below `6·σ̂ = 0.0032435`.
  Measured `0.021820 = 6.73×` the guard, i.e. `40.3·σ̂`. Banked (clause 3). Note the same
  guard **fails** for `frac(margin<0.005)` (`R = 3.88`, 0/5 adjacent gaps clearing `2σ̂`),
  which is why the primary was moved to `median_margin` **pre-data** (`GATE_DESIGN.md` §1)
  and why `frac<τ` results may not overturn the primary.
* ⚠️ **The attribution clause is the one that can still kill it, and it is UNRUN.** Clause 5
  evaluates `median_margin` at heal steps `{25000, 50000, 100000, 128000, 200000}` of
  `olmo2_probe2_7B_keep14fresh2_seed1234` — damage held **exactly** at `keep_front=14`,
  `n_fresh=2`, `seed=1234` — and computes
  `φ = max( range(y), |OLS slope on heal_step| × 175000 ) / 0.021820`.
  **`φ ≥ 0.60` ⇒ KILL** (budget alone reproduces ≥60 % of what B04 attributes to damage; B04
  folds into the Paper B methods appendix and the 244–2560 GPU-h family ladder is never
  funded). **`φ ≤ 0.30` ⇒ PASS. `0.30 < φ < 0.60` ⇒ NARROWED** to "damage and heal budget
  *jointly* compress margins". Cost: **1.08 GPU-h** (measured anchor:
  `logs/sv181_main.log:5-6`, 121 s wall × 8 GPU = 0.269 GPU-h/rung × 4 rungs).
* **And the gate is pre-registered as reachable in both directions**, which is the only
  reason a PASS would mean anything: `code/analyze_b04_wzc1_floor.py --selftest` `sys.exit`s
  if any of KILL/NARROWED/PASS is unreachable (constructed: KILL 0.8387 monotone and 0.9395
  non-monotone-V; NARROWED 0.3896; PASS 0.1054 / 0.0495). **The single most relevant
  empirical precedent predicts KILL**: the only fixed-damage budget ladder that exists
  anywhere in this project — the Qwen `f12k2`/14L cell at steps 2000/20000/200000 — scores
  `φ = 1.7760` at its own span.

**What must NOT be written even if clause 5 passes:**
1. Not *"eval fragility"* as a general phenomenon — **OLMo-2-7B only**, and cross-family
   generality is **UNTESTED** (`STATUS.json:43`).
2. Not *"margin mediates the damage→flip drop"* — that needs LOO across items and a
   non-degenerate flip denominator, and the harness is **bit-deterministic within arch**
   (`status/PAPERB_WITHIN_DISK_FLOOR_V3.md`; `memory/same-harness-runs-bit-identical.md`).
3. Not the `ρ = +1.00` without `Spearman(core6, heal_steps)` beside it, **with the ladder
   named** (+0.6669 wzc1 / +0.8721 zwfy6), plus `σ̂`, `R`, and `φ`.
4. Not a stand-alone paper. `STATUS.json:45` — recommended home is the **Paper B methods
   appendix or the A01 null-cal spin-out**. Promotion needs a second **budget-matched**
   family ladder *and* an a-priori mechanism-level hypothesis
   (`STATUS.json.resurrection_conditions`).

### 4.1 ⚠️ Why "0 hits on the interaction" is not a green light

The audit's designated novelty returned no direct hit (§2.5). Three reasons that is **not**
sufficient:

1. **A2 (arXiv:2601.06118) supplies a null mechanism.** If nondeterminism is largest at
   mid-range probabilities, a compressed-margin model has more items in the sensitive band
   *by arithmetic*. B04's interaction claim must be an **increment over that**, and no arm
   currently on disk isolates it.
2. **B04's harness cannot see the interaction it wants to claim.** Same-driver same-arch
   re-runs are **bit-identical, 0 flips** — so the "nuisance sensitivity" denominator is
   structurally ~zero, which is why clause 1 was retired as `UNTESTABLE` and clause 4 (second
   nuisance = torch/GPU arch) was **withdrawn** because crossing arch is exactly what
   `LIFECYCLE_SCHEMA.md` §3 forbids. **B04 currently has no admissible nuisance axis on
   which to measure amplification at all.** That is a bigger gap than any citation.
3. **§2.1 shows the neighbourhood was mis-mapped once.** An ICML 2026 paper with a
   same-named metric on the same task family went unfound by the 2026-08-09 novelty pass and
   by the 2026-08-14 gate design. `memory/reporting-a-gap-is-not-closing-it.md` applies: the
   right response is a re-run of the novelty sweep with the queries in §2.5 and §6, not a
   promotion.

---

## 5. MUST-NOT-CLAIM list (binding on any B04 / Paper B appendix / A01 spin-out text)

1. ❌ **General "evaluation fragility under model damage".** OLMo-2-7B only; the Qwen leg is
   `NON_MATCHED_INCONCLUSIVE` (untested, not refuted) — `STATUS.json:14-24`.
2. ❌ **First to quantify `T=0` nondeterminism / batch-size sensitivity.**
   **A1, TMLR Feb 2026** (`venueid = TMLR`, `Camera_Ready_Revision` verified) owns it.
3. ❌ **First to observe that low logit margins predict flips under batch nondeterminism.**
   **A3/MarginGate, arXiv:2605.30218** owns it (0.3–1.3 % flip rates, margin-triggered).
4. ❌ **First to observe benchmark rank instability across runs, or to recommend repetitions
   and uncertainty reporting.** **B4, arXiv:2509.24086** (83 % of slices invert) and
   **B1, NeurIPS 2024 RegML Workshop** own it. **Do not cite B1 as a main-venue paper — it
   was rejected at ICLR 2025.**
5. ❌ **First to observe that fixed nuisance settings make cross-model comparison unfair, or
   to propose per-instance randomisation.** **B3, Findings of EMNLP 2025**
   (`2025.findings-emnlp.182`) owns it. **Findings, not main.**
6. ❌ **"Equal accuracy does not imply equal behaviour under compression."**
   **C2 (arXiv:2508.13533)**, **C3 (ICML 2024 Poster, `Camera_Ready_Revision` verified)**,
   **C1 (TMLR 2026, `Camera_Ready_Revision` verified)** jointly own it.
7. ❌ **First to report calibration/faithfulness degrading while accuracy holds under layer
   pruning.** **C1/Tropeano, TMLR 2026** owns it.
8. ❌ **NEW — the per-item multiple-choice DECISION MARGIN as a lens on layer pruning, or
   the observation that pruning compresses it.** **arXiv:2605.07271, ICML 2026 regular**
   (`venueid = ICML.cc/2026/Conference`, `Camera_Ready_Revision` verified) defines
   *"Decision Margin"* as *"the probability gap between the ground-truth option and the most
   likely alternative"* on MC tasks and reports it collapsing under pruning across
   Llama3-8B / Llama2-7B / Qwen3-4B. **2026-05-08 is >3 months before this pass → NOT
   concurrent → real prior art.**
9. ❌ **NEW — damage-dependent sensitivity to perturbation as a novel observation.**
   Same paper, §4.3.1: injected-noise sensitivity is **asymmetric between the Silent and
   Decisive phases** (`σ²=0.02` vs `σ²=0.1`).
10. ❌ **NEW — that recovery/heal from deep pruning is bounded by remaining depth.**
    Same paper, §4.3.3 (SFT plateaus beyond ~40.6 % pruning).
11. ❌ **First to use layer-drop pruning on OLMo-scale LMs.** **C4/ShortGPT, Findings of
    ACL 2025** (`2025.findings-acl.1035`) — the `shortgpt16` rung *is* their method.
    **Findings, not ACL main.**
12. ❌ **A mediation claim** ("near-ties cause the aggregate drop"). `DIRECTION_A_VERDICT.md:92`
    disclaims it; the harness has no flip denominator.
13. ❌ **That the gate has been passed, or that `lifecycle` is anything but `ready_cpu`.**
    3/3 adversarial lenses returned `NEEDS_REVISION`; revision 2 is applied but
    **un-adjudicated** (`STATUS.json:247`, `:253`).
14. ❌ **Any number from `PROPOSAL.md:9-15`.** It disagrees with
    `evidence/B04_6rung_bs16_analysis.json` on **every rung** (base `0.124594` vs `0.131806`,
    keep8 `0.075801` vs `0.094933`) and on the `p` (`0.0167` vs `0.0028`). The JSON side
    reproduces; `PROPOSAL.md`'s table must be marked **superseded** before any threshold is
    quoted against it (`STATUS.json.next_gate.prereg_G0_first_0_GPU` step (c);
    `remaining_blockers_after_this_design[2]`).
15. ❌ **`frac(margin<0.005)` (or `<0.001`, `<0.01`) as the primary.** It fails its own floor
    (`R = 3.88 / 1.83 / 7.93`; adjacent gaps clearing `2σ̂` = 0/5, 0/5, 1/5). Reported for
    completeness only; **a `φ` computed on any of them cannot overturn the primary.**
16. ❌ **A run-to-run standard deviation from `σ̂`.** `σ̂` rests on `n = 2` (df = 1) and is
    admissible **only as a floor with a 6× safety factor** —
    `paperB/SEEDVAR_KEEP14_VERDICT.md` §5 explicitly forbids the other reading.

---

## 6. Honest gaps in this adjudication

1. ⚠️ **The single biggest finding of this pass is a gap in the previous pass.**
   **arXiv:2605.07271 (ICML 2026)** was missed by `NOVELTY_CHECK.md` (2026-08-09), by MAIN's
   independent re-verification the same day, and by the 2026-08-14 gate design — despite
   using B04's own metric name on B04's own task family. It surfaced only on the query
   `abs:"decision margin" AND abs:"language model"`. **`NOVELTY_CHECK.md`'s
   `hold_in_backlog` verdict survives, but its "top-5 nearest candidates" table and its
   differentiation table are now incomplete and should be regenerated.** That is the
   actionable 0-GPU follow-up, and it is *not* done by this file.
2. ⚠️ **DBLP lags OpenReview for 2026 conference papers, again.** 2605.07271 is `CoRR 2026 /
   Informal` on DBLP and **ICML 2026 regular with a `Camera_Ready_Revision`** on OpenReview.
   A DBLP-only sweep would have called it a preprint and mis-sized the threat, exactly as
   `memory/venue-verify-must-use-openreview-2026.md` predicts.
3. ⚠️ **`api2.openreview.net/notes?id=` and `?forum=` are HTTP 403 `ChallengeRequiredError`.**
   `notes/search` worked and returned `invitations`, so `Camera_Ready_Revision` was verified
   for A1, C1, C3, B2 and §2.1. **No forum-level enumeration was performed**, so a paper
   whose title-match fails cannot be recovered.
4. ⚠️ **Semantic Scholar HTTP 429 all session.** Never used as an authority. A paper indexed
   only on S2 could have been missed.
5. ⚠️ **DBLP intermittent HTTP 500.** `Shortened LLaMA` (C5) never resolved this session, so
   **C5's row is carried second-hand from `NOVELTY_CHECK.md`'s 2026-08-09 MAIN
   verification** and is not re-verified here.
6. ⚠️ **Two rows rest on version-dependent or workshop-vs-main distinctions that are easy to
   miscite**: C5 (**workshop acceptance applies to v1 only**; the prune-then-heal content is
   in v2, which has no venue) and B1 (**RegML workshop, rejected at ICLR 2025**).
7. **Only ONE full text was read** (§2.1, `arxiv.org/html/2605.07271v1`). Every other
   overlap judgement is from **abstract + venue metadata + `comment`/`journal_ref`**. In
   particular **A2's** claim that nondeterminism concentrates in the 0.1–0.9 probability
   band — which §4.1 treats as a null mechanism for B04's interaction — is taken **from its
   abstract**, and the quantitative form of that band-dependence was not read. **Before B04
   claims an increment over it, A2 must be read in full.**
8. **`arXiv-only` rows**: A2, A3, A4, A5, B4, C2, C6 (7 of 16). Means "no peer-reviewed
   venue verifiable from this node", not "no venue exists".
9. **No `.bib` entries emitted.** Safe to add after this pass: **§2.1 (ICML 2026)**,
   **A1 (TMLR)**, **B1 (NeurIPS 2024 RegML Workshop)**, **B2 (COLM 2025)**,
   **B3 (Findings EMNLP 2025)**, **C1 (TMLR 2026)**, **C3 (ICML 2024)**,
   **C4 (Findings ACL 2025)** — all `venueid`- or Anthology-verified. **Not safe**: every
   `arXiv-only` row, and **C5** (version-dependent, unverified this session).
10. **Zero cross-disk verification.** Every path cited is on **wzc1**. The zwfy6 `_bs16`
    ladder dirs that `DIRECTION_A_VERDICT.md:97` cites (and whose `+0.8721` this file quotes)
    were **not** `ls`-confirmed — per `STATUS.json:244` that is *"unverified from here, NOT
    gone"* (`memory/two-disk-rule-applies-to-main-too.md`). **Nothing above claims any such
    absence.** Note also that those zwfy6/sm_90 dirs are **not admissible as G1 comparators**
    (arch confound, `STATUS.json:91`).
11. **`PROPOSAL.md` was not edited.** Its numbers are still the superseded set (gap 14
    above). Left as the dated record of what was proposed; `STATUS.json` +
    `GATE_DESIGN.md` are the authority.

---

## 7. Verdict

```
related_work_status:  audited
lifecycle:            ready_cpu  (UNCHANGED -- 3/3 adversarial lenses NEEDS_REVISION;
                      revision 2 applied but NOT adjudicated. This file does not promote.)
novelty_status:       hold_in_backlog (2026-08-09 verdict SURVIVES) -- but its
                      candidate table is now INCOMPLETE, see below
claim_scope:          OLMo-2-7B ONLY. Cross-family generality UNTESTED (the Qwen leg's
                      kill was downgraded to NON_MATCHED_INCONCLUSIVE on 2026-08-10;
                      untested is not a promotion argument).
gate_status:          NOT DECIDABLE YET -- clause 5 (budget discrimination at fixed
                      damage, phi vs the 175000-step read-out span) has NEVER RUN.
                      1.08 GPU-h. The only existing empirical precedent (Qwen f12k2/14L)
                      scores phi = 1.7760 => predicts KILL.

strongest_collision:  Shi et al., "Understanding Performance Collapse in Layer-Pruned
                      LLMs via Decision Representation Transitions", arXiv:2605.07271,
                      **ICML 2026 regular** (OpenReview venueid=ICML.cc/2026/Conference,
                      Camera_Ready_Revision verified 2026-08-15).
why_not_preemption:   Full text read. It has NO HEAL STEP (grep: heal 0, retrain 0,
                      continued-pretrain 0; only bounded SFT in 4.3.3), its Decision
                      Margin is a LAYER-WISE trajectory within one network rather than a
                      CROSS-CHECKPOINT distribution over 17,195 items, and it reports NO
                      rank statistic and NO noise floor (grep: spearman 0, per-item 0).
                      B04's kill_definition requires depth-prune-THEN-HEAL on >=3 rungs
                      plus a per-item margin distribution co-varying with aggregate score
                      AS A STATISTIC; this paper meets none of the three jointly.
                      => differentiation + external corroboration of the LENS on three
                      other families, NOT preemption.
BUT_it_costs_B04:     3 new must-not-claim items (#8/#9/#10). It is >3 months old, so it
                      is NOT concurrent. NOVELTY_CHECK.md's top-5 table is now incomplete
                      and should be regenerated (0 GPU).
runner_up_threat:     arXiv:2601.06118 (arXiv-only) -- nondeterminism concentrates at
                      mid-range token probabilities, which is a NULL MECHANISM for
                      "damage amplifies nuisance sensitivity": compressed margins put
                      more items in the sensitive band by arithmetic. B04 has no arm
                      that isolates an increment over this.
promotion:            NO. Needs (a) an independent adversarial pass returning SOUND on
                      the revised gate, (b) clause 5 run and not firing, (c) a
                      budget-matched second family, (d) an a-priori mechanism hypothesis,
                      and now (e) a regenerated novelty table including 2605.07271.
```

No candidate is 完全相同/抄袭. Every collision differs on at least one load-bearing axis:
**heal step present vs absent**, **cross-checkpoint distribution vs layer-wise trajectory**,
**rank statistic against a measured floor vs descriptive curves**, **harness nuisance vs
injected hidden-state noise**, **per-item margin vs aggregate ECE/Brier**, or **intact vs
damaged regime**. Per `memory/prior-work-differentiate-dont-abandon.md` the correct output of
this pass is a **citation-obligation list plus a differentiation map plus one corroboration
opportunity** — and **not** a scope reduction beyond what B04's own measurements already
forced. B04 dies, if it dies, from **clause 5**, not from this table.
