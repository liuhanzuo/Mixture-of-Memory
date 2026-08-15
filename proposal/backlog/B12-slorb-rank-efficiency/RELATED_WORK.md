# B12 — Related Work / novelty check (G0 LEG 1)

**Written**: 2026-08-16. **GPU spent**: ZERO (all authorities are web APIs; no model loaded).
**Verdict**: **NOTHING PREEMPTS.** 30 candidates checked, **0 preempt**, **30 adjacent**.
The specific delta B12 retains is stated in §6.

**Load-bearing question** (B12's own framing, `STATUS.json.next_gate`):

> Has anyone already done **block-basis coarsening (or an equivalent post-hoc rank/basis reduction)
> of a low-rank branch bolted onto a 2:4 semi-structured-sparse base model**?

Preemption bar (per `memory/prior-work-differentiate-dont-abandon.md`): **"already shows the same
thing"**, not "same area". Work within 2–3 months is **concurrent** and does not preempt.

---

## 0. Authorities used, and what each refused

| authority | endpoint | used for | status |
|---|---|---|---|
| arXiv API | `export.arxiv.org/api/query` | id metadata + full-text search | **200** on every call. Positive control (`2502.09992` → "Large Language Diffusion Models") **PASSED** before any negative was trusted. |
| OpenReview | `api2.openreview.net/notes/search?term=…` | ICLR/NeurIPS/ICML `venueid` | **200**. Used for SLoPe, SLiM, Wanda, VeRA, NOLA, DoRA, RoSA, SLTrain, ALPS, MaskLLM, LoSparse, OWL. |
| ACL Anthology | `aclanthology.org/<id>/` | ACL-family **incl. Findings** | **200** + title match on all 5 IDs fetched. |
| DBLP | `dblp.org/search/publ/api` | ACL-family + AAAI/HPCA/NeurIPS keys, DOIs | **200** except two `500`s on long queries (worked when shortened). |
| Semantic Scholar | — | **NOT USED** | Known to return 429 persistently in this environment; its silence is not evidence. |

**Refusals I did NOT convert into claims:**
- `api.openreview.net` (v1) `/notes?content.venueid=…` → **403 `ChallengeRequiredError`**. This is the
  documented challenge gate, **not** "OpenReview unreachable". `/notes/search` on api2 works.
- **AdaLoRA on OpenReview**: `/notes/search` for both the arXiv title ("AdaLoRA: Adaptive Budget
  Allocation…") and the camera-ready title returned **no matching note in 20/25 results**. Its venue is
  established instead by **DBLP `conf/iclr/ZhangCBH0CZ23`, venue "ICLR 2023"**, which is a
  legitimate authority for the fact; it just is not the OpenReview `venueid` this repo prefers for the
  OR family. Flagged rather than papered over. Note the **camera-ready title drops the "AdaLoRA:"
  prefix** — that is why the title-based OR search missed, and it is exactly the arXiv-vs-camera-ready
  drift `memory/venue-verify-acl-family-needs-anthology.md` warns about.
- **CAST (arXiv:2509.25996)**: OpenReview `/notes/search` **no hit**; DBLP has **only** a CoRR record
  (`journals/corr/abs-2509-25996`). Its own arXiv comment says **"Submitted to IEEE TPAMI"**. So
  `venue = PREPRINT / under submission`. Per this repo's own rule, "DBLP says CoRR" = **NOT-FOUND for a
  conference record**, but here the paper's *own* comment corroborates "submitted", so preprint is the
  right label. **Do not cite CAST as a published venue.**

---

## 1. SLoRB's own lineage — the one place preemption was most likely

SLoRB is **not B12's invention and not SparseForge's**. It is an existing component whose code this
repo vendors verbatim.

| # | work | id | venue (authority) | mechanism | preempts? |
|---|---|---|---|---|---|
| 1 | **AST** — Pruning LLMs with Semi-Structural Adaptive Sparse Training. Huang, Hu, Jian, Zhu, Chen (2024) | arXiv:2407.20584 | **AAAI 2025** (DBLP `conf/aaai/HuangHJZC25`, DOI `10.1609/AAAI.V39I23.34592`) | Retrains 2:4 semi-structured sparse LLMs with learnable masks + KD, **plus "a supplementary set of well-initialized parameters"** — that supplementary set **IS SLoRB**. `baselines/ast_official_clean/sparse_modeling.py:107-131` is the origin of `init_SLoRB`, `SLoRB_k`, `SLoRB_init_type ∈ {mean,sum,xavier}`, and the block-indicator `x_proj`; `README.md:90-92` shows `--SLoRB_k 16`. | **NO.** AST *introduces* the branch and trains it. It never reduces an already-trained branch post hoc, and it does not titrate a density/quality curve over branch size. It is the work B12 **operates on**, and B12 must cite it as the source of the mechanism, never re-derive it. |
| 2 | **CAST** — Continuous Adaptive Sparse Trainer. Huang, Hu, Zhu, Chen (2025) | arXiv:2509.25996 | **PREPRINT** ("Submitted to IEEE TPAMI", per the paper's own arXiv comment; DBLP CoRR-only; no OpenReview note) | Same group's successor: fully continuous/differentiable N:M sparsity-aware training. The repo's own `baselines/cast_repro/SPEC.md:3` names it as the reproduction target and `CAST-repro` is the 62.0919 @ 50.0% row B12 is barred from claiming Pareto-dominance over. | **NO.** Training-time; no post-hoc branch surgery. |

**This is the important structural fact for the write-up.** B12's operator acts on a mechanism
introduced by AST (AAAI 2025). Any B12 text must therefore say *"we post-hoc compress AST's SLoRB
branch"* — the branch is a **cited object**, not a contribution. `PROPOSAL.md` already gets the code
provenance right (`sparse_modeling.py:415-427`, `:883-886`) but does **not** name AST or CAST
anywhere; **that is a citation gap this file closes.**

---

## 2. Surface (a) — post-hoc SVD / low-rank truncation of **trained adapters**

This is the surface where a preemption would be most damaging, because B12's control **Dctl is
literally density-matched SVD of a trained branch**. Six works do post-hoc spectral surgery on
adapters. **None touches a sparse base.**

| # | work | id | venue (authority) | mechanism | preempts? |
|---|---|---|---|---|---|
| 3 | **PARA** — Post-Optimization Adaptive Rank Allocation for LoRA. Kumaravelu, Gupta, Srijith (2026-04-30) | arXiv:2604.27796 | **UNVERIFIED** — no OpenReview note, no DBLP record found; arXiv comment empty. Treat as preprint. | Data-free post-hoc SVD of a *trained* LoRA with a **global singular-value threshold across layers** → non-uniform per-layer rank. | **NO, and it is CONCURRENT** (2026-04, ~4 months before B12; borderline but on the safe side of the 2–3-month rule). Closest hit on surface (a): same "trained adapter → SVD → non-uniform rank" skeleton. **But**: dense base, no N:M, no basis coarsening, and its non-uniformity is *emergent from a global threshold* whereas B12's asymmetry is a **pre-registered per-family map** chosen by an exhaustive 5⁷ frontier scan. **B12 must cite PARA as the closest prior art on the SVD control** and must not present density-matched SVD-of-adapter as new. |
| 4 | **LoRA-Squeeze**. Vulić, Grycner, de Laroussilhe, Pfeiffer (2026-02-11) | arXiv:2602.10993 | **PREPRINT** (arXiv comment: "Preprint"); no OpenReview/DBLP conference record found | "Learn a higher-rank solution then compress it", post-hoc **or** in-training. | **NO.** Explicitly the thesis "train big, squeeze after" — the same *philosophy* as B12's "the trained branch is over-provisioned", but on dense-base LoRA, SVD-family operator, no sparsity, no density axis. **Cite as the strongest statement of the shared premise.** |
| 5 | **Spectral Surgery** — Training-Free Refinement of LoRA via Gradient-Guided Singular Value Reweighting. Tian, Chen, Han, Liao (2026-03-04) | arXiv:2603.03995 | **UNVERIFIED** — no OpenReview/DBLP record found; treat as preprint | SVD-decompose a trained LoRA, score components by gradient sensitivity, **reweight** (not just truncate). | **NO.** Reweighting ≠ parameter deletion; no density claim; dense base. |
| 6 | **PHLoRA** — data-free Post-hoc Low-Rank Adapter extraction from full-rank checkpoint. Vasani, FitzGerald, Fang, Vaish (2025-09-13) | arXiv:2509.10971 | **UNVERIFIED** — no OpenReview/DBLP record found; treat as preprint | SVD of `W_ft − W_base` to *manufacture* an adapter that never existed. | **NO — and note the direction is inverted.** PHLoRA *creates* a low-rank branch from a weight difference; B12 *shrinks* a branch that was genuinely trained. Useful as the "SVD-of-a-weight-delta is standard practice" citation. |
| 7 | **SpectralLoRA**. Singh (2026-04-12) | arXiv:2604.10649 | **UNVERIFIED** — no OpenReview/DBLP record; single-author IIT Roorkee preprint | 2-D DCT of trained LoRA updates; **33% of DCT coefficients hold 90% of energy**; 10% retention → 10× storage cut. | **NO.** BERT/RoBERTa + GLUE, dense base. **Scientifically relevant as a CONTRAST**: it reports LoRA updates are spectrally *concentrated*, whereas B12 measured `E`'s spectrum to be **flat vs a Gaussian-product null in 32/35 projections** (`slorb_rank_5b_headline.json`). Different regime, opposite finding — cite it precisely so the flat-spectrum claim is not read as contradicting the literature. |
| 8 | **LoRA-drop**. Zhou, Lu, Xu, Zhu, et al. (2024) | arXiv:2402.07721 | **COLING 2025**, `2025.coling-main.371`, pp. 5530-5543 (Anthology **200** + title match; DBLP `conf/coling/ZhouLXZZY25`) | Prunes **whole LoRA modules** by output magnitude. | **NO.** Module-granularity drop, not intra-branch rank/basis reduction; dense base. |

---

## 3. Surface (b) — adapter-rank over-provisioning / rank selection ablations

| # | work | id | venue (authority) | mechanism | preempts? |
|---|---|---|---|---|---|
| 9 | **AdaLoRA** — Adaptive Budget Allocation for PEFT. Zhang, Chen, Bukharin, Karampatziakis, et al. (2023) | arXiv:2303.10512 | **ICLR 2023** (DBLP `conf/iclr/ZhangCBH0CZ23`). ⚠ OpenReview `/notes/search` returned no matching note under either title — see §0. | SVD-parameterised adapter; prunes unimportant singular triplets **during** training. | **NO.** Training-time budget reallocation. Cannot be reproduced by post-hoc surgery, and `must_not_claim[0]` already forbids B12 from claiming anything about training-time design. |
| 10 | **DyLoRA**. Valipour, Rezagholizadeh, Kobyzev, Ghodsi (2022) | arXiv:2210.07558 | **EACL 2023**, DOI `10.18653/V1/2023.EACL-MAIN.239` (DBLP `conf/eacl/ValipourRKG23`) | Trains nested ranks so one adapter serves a **range** of ranks at inference — search-free rank selection. | **NO.** Requires training with the nesting objective. B12's rungs are surgery on a `k=16`-trained tensor; the ladder is *not* a nested-rank family. |
| 11 | **SoRA** — Sparse Low-rank Adaptation. Ding, Lv, Wang, Chen, et al. (2023) | arXiv:2311.11696 | **EMNLP 2023 Main**, `2023.emnlp-main.252` (Anthology **200**; DBLP `conf/emnlp/DingLWCZL023`) | Gated singular values with proximal L1 → adaptive rank during training. | **NO.** Training-time. |
| 12 | **VeRA**. Kopiczko, Blankevoort, Asano (2023) | arXiv:2310.11454 | **ICLR 2024 poster** (OpenReview `venueid=ICLR.cc/2024/Conference`, id `NjNfLdxr3A`) | **Frozen shared random** matrices + trained scaling vectors. | **NO** — but architecturally the closest analogue of *"the basis need not be stored"*: VeRA's basis is a PRNG seed, B12's coarse basis is a **segment-sum with zero stored parameters**. Different reason for the same effect; cite as prior art on parameter-free bases. |
| 13 | **NOLA**. Koohpayegani, Navaneet, Nooralinejad, Kolouri, et al. (2023) | arXiv:2310.02556 | **ICLR 2024 poster** (OpenReview `venueid=ICLR.cc/2024/Conference`, id `TjfXcDgvzk`) | Adapter = linear combination of **random basis** matrices; only the coefficients are stored. | **NO.** Same "store coefficients, not basis" idea, trained that way from the start. |
| 14 | **Tied-LoRA**. Renduchintala, Konuk, Kuchaiev (2023) | arXiv:2311.09578 | **NAACL-HLT 2024**, DOI `10.18653/V1/2024.NAACL-LONG.481` (DBLP `conf/naacl/RenduchintalaKK24`) | Weight **tying** across layers + selective freezing. | **NO.** Cross-layer sharing, training-time. |
| 15 | **LoRA-FA**. Zhang, Zhang, Shi, Chu, et al. (2023) | arXiv:2308.03303 | **UNVERIFIED** — arXiv comment empty; no OpenReview/DBLP conference record located | Freezes the down-projection `A`, trains only `B`. | **NO.** Freezing one factor at train time. |
| 16 | **DoRA**. Liu, Wang, Yin, Molchanov, et al. (2024) | arXiv:2402.09353 | **ICML 2024 Oral** (OpenReview `venueid=ICML.cc/2024/Conference`, id `3d5CIRG1n2`; corroborated `dblp.org/conf/ICML/2024`) | Magnitude/direction decomposition of the update. | **NO.** Reparameterisation, training-time. |
| 17 | **Shears** — Unstructured Sparsity with Neural Low-rank Adapter Search. Muñoz, Yuan, Jain (2024) | arXiv:2404.10934 | **NAACL 2024 Industry**, `2024.naacl-industry.34`, DOI `10.18653/V1/2024.NAACL-INDUSTRY.34` (Anthology **200**) | **Elastic-rank LoRA + NAS on a sparse base model.** | **NO — but this is the closest hit on surface (d)+(b) jointly.** It genuinely puts adapters on a *sparse* base and searches over adapter rank. **Three decisive differences**: (i) its sparsity is **unstructured**, not 2:4/N:M, so it has no exact-N:M invariant to preserve and no two-matmul deployment-density accounting; (ii) the rank search is **a training-time NAS**, so its "smaller adapter" is *trained* at that rank — precisely the comparison `must_not_claim[0]` says B12 cannot make; (iii) no post-hoc surgery on a fixed trained branch. **B12 must cite Shears as the nearest neighbour and state these three deltas explicitly.** |
| 18 | Low-Rank Adapters Meet Neural Architecture Search for LLM Compression | arXiv:2501.16372 | **UNVERIFIED** — not chased past arXiv (survey/position-style companion to Shears) | Survey-ish framing of elastic-adapter NAS. | **NO.** Same family as #17. |

---

## 4. Surfaces (c)+(d) — sparse + low-rank hybrids, and low-rank branches on N:M-sparse weights

This is the narrow surface where B12 could have been preempted outright. **It was not.**

| # | work | id | venue (authority) | mechanism | preempts? |
|---|---|---|---|---|---|
| 19 | **SLoPe** — Double-Pruned Sparse Plus **Lazy** Low-Rank Adapter Pretraining. Mozaffari, Yazdanbakhsh, Zhang, Mehri Dehnavi (2024) | arXiv:2405.16325 | **ICLR 2025 Poster** (OpenReview `venueid=ICLR.cc/2025/Conference`, id `lqHv6dxBkj`) | Sparse **pretraining** + low-rank adapters added in the **final 1%** of iterations; double-pruned backward pass. | **NO, and this is the single most important non-preemption to state.** SLoPe is the archetype of "N:M-sparse base + added low-rank adapter" and B12 must cite it as such. But SLoPe's question is **when to ADD** the adapter (lazily, cheaply) — B12's is **how much of an already-trained adapter can be REMOVED**. SLoPe never shrinks its adapter post hoc, never reports a density↔quality curve over adapter size, and never coarsens a basis. Opposite direction on the same axis. |
| 20 | **SLiM** — One-shot Quantization and Sparsity with Low-rank Approximation. Mozaffari, Yazdanbakhsh, Mehri Dehnavi (2024) | arXiv:2410.09615 | **ICML 2025 poster** (OpenReview `venueid=ICML.cc/2025/Conference`, id `4UfRP8MopP`) | One-shot: quantisation + sparsity + a low-rank corrector, no retraining. | **NO — closest on the "no-retraining" axis.** SLiM *constructs* a low-rank corrector one-shot to compensate sparsification error; B12 *reduces* a corrector that was genuinely trained for 17,900 iterations. SLiM never asks whether its own corrector is over-provisioned. **Must cite: it is the strongest prior art for "low-rank correction of a sparse base without retraining".** |
| 21 | **LoSparse**. Li, Yu, Zhang, Liang, et al. (2023) | arXiv:2306.11222 | **ICML 2023 Poster** (OpenReview `venueid=ICML.cc/2023/Conference`; DBLP `conf/icml/LiYZLHCZ23`) | `W ≈ low-rank + sparse`, jointly *trained* as a compression parameterisation. | **NO.** Trains the decomposition; not N:M; no post-hoc reduction. |
| 22 | **RoSA**. Nikdan, Tabesh, Crnčević, Alistarh (2024) | arXiv:2401.04679 | **ICML 2024 Poster** (OpenReview `venueid=ICML.cc/2024/Conference`) | Robust-PCA-inspired: jointly trains low-rank **and** highly-sparse components over frozen weights. | **NO.** Training-time PEFT; the *sparse* part is the adapter, not the base. |
| 23 | **SLTrain**. Han, Li, Huang, Hong, et al. (2024) | arXiv:2406.02214 | **NeurIPS 2024 poster** (OpenReview `venueid=NeurIPS.cc/2024/Conference`) | Pretrain with `W = low-rank + sparse` for memory efficiency. | **NO.** Pretraining parameterisation. |
| 24 | **FlexiGPT** — Pruning and Extending LLMs with Low-Rank **Weight Sharing**. Smith et al. (2025) | arXiv:2501.14713 | **NAACL 2025 Long**, `2025.naacl-long.31`, DOI `10.18653/V1/2025.NAACL-LONG.31` (Anthology **200**) | Replaces pruned **blocks** by a weight-sharing scheme + block-specific low-rank adapters. | **NO.** Block/layer granularity replacement of *removed* blocks; the sharing is *across blocks*, not *within a projection's basis*. Not N:M. |
| 25 | **MaskLLM**. Fang, Yin, Muralidharan, Heinrich, et al. (2024) | arXiv:2409.17481 | **NeurIPS 2024 Spotlight** (OpenReview `venueid=NeurIPS.cc/2024/Conference`) | Learnable N:M mask distribution via Gumbel-softmax. | **NO.** Mask learning only; no low-rank branch at all. |
| 26 | **Wanda**. Sun, Liu, Bair, Kolter (2023) | arXiv:2306.11695 | **ICLR 2024 poster** (OpenReview `venueid=ICLR.cc/2024/Conference`, id `PxoFut3dWW`) | One-shot magnitude×activation pruning, incl. 2:4. | **NO.** Pruning criterion; no branch. Cited as a base-model baseline only. |
| 27 | **SparseGPT**. Frantar, Alistarh (2023) | arXiv:2301.00774 | **ICML 2023** (DBLP `conf/icml/FrantarA23`) | One-shot Hessian-based pruning with weight update, incl. 2:4. | **NO.** Same. |
| 28 | **ALPS**. Meng, Behdin, Wang, Mazumder (2024) | arXiv:2406.07831 | **NeurIPS 2024 poster** (OpenReview `venueid=NeurIPS.cc/2024/Conference`) | Operator-splitting optimisation for one-shot sparse pruning. | **NO.** Pruning optimiser; no low-rank branch. |
| 29 | **OWL**. Yin, Wu, Zhang, Hsieh, et al. (2023) | arXiv:2310.05175 | **ICML 2024** (DBLP `conf/icml/0006W0HWJLJPLBW24`). ⚠ OpenReview shows `ICLR.cc/2024/Conference/Rejected_Submission` **and** a PML4LRS workshop poster — per this repo's rule, a Rejected_Submission record is **NOT-FOUND, not NOT-PUBLISHED**; the ICML 2024 DBLP record is the live one. | Outlier-weighted per-layer sparsity ratios. | **NO.** Per-layer *sparsity* budget, not adapter rank. **Structurally analogous to B12's per-family asymmetry** (non-uniform budget allocation guided by measured per-layer sensitivity) — worth citing as the precedent for the asymmetry *idea*, on the sparsity axis rather than the basis axis. |
| 30 | **DSnoT** — Dynamic Sparse No Training. Zhang, Zhao, Lin, Sun, et al. (2023) | arXiv:2310.08915 | **ICLR 2024** (DBLP `conf/iclr/0002ZLSYHT0J24`) | Training-free iterative weight *growing/pruning* to fine-tune sparse LLMs. | **NO.** Training-free like B12, but it edits the **sparse mask**, never a low-rank branch. Cite as precedent for "training-free post-hoc surgery on a sparse LLM is a legitimate genre". |

### Block coarsening / averaging as a compression operator, named in other literatures

The task asked whether the specific operator has a name elsewhere. **It does, and none of the hits
is about adapters.** `abs:"weight clustering" AND abs:"large language model"` returned **3 results**
total; `abs:"group-wise quantization" AND abs:"block"` returned **1**.

| work | id | venue | why not preemptive |
|---|---|---|---|
| **eDKM** — train-time weight clustering for LLMs. Cho et al. | arXiv:2309.00964 | **HPCA 2025**, DOI `10.1109/HPCA61900.2025.00133` (also *IEEE CAL* 2024, DOI `10.1109/LCA.2024.3363492`) | Clusters **base weights** into a palette for compression. B12 averages **columns of a low-rank basis**, and the coarse basis is a *constant-1 block indicator*, which makes the LS solution a closed-form block **mean** — a different object with a different optimality argument. |
| Only relative ranks matter in weight-clustered LLMs | arXiv:2603.17917 | **PREPRINT** (DBLP CoRR-only, `journals/corr/abs-2603-17917`) | Weight clustering of base weights again; concurrent (2026-03). |

**Also checked and empty** (arXiv full-text, `totalResults` verbatim):
- `abs:"coarsening" AND abs:"low-rank" AND abs:"adapter"` → **3 total**, all irrelevant (PDE
  coarsening, APOLLO optimiser, matrix-sketching).
- `abs:"2:4" AND abs:"post-hoc"` → **1 total**, a brain-tumour segmentation paper.
- `abs:"adapter" AND abs:"rank reduction" AND abs:"without retraining"` → **0 total**.
- `abs:"N:M sparsity" AND abs:"low-rank"` → **1 total**, and it is SLoPe (#19).
- `all:"SLoRB"` → **0 total**. **The term does not appear anywhere on arXiv**, including in AST's own
  abstract (AST calls it "a supplementary set of well-initialized parameters"). This is a real
  finding: nobody has written about this branch by name, so there is no literature to preempt B12 on
  *this specific branch*.

---

## 5. What this changes about B12's own framing (evidence over narrative)

Two things the survey forces, both of which sharpen `PROPOSAL.md` rather than contradict it.

**5.1 The honest framing is "which operator wins where, at matched density" — confirmed by the
literature, not just by our file.** `op_matched_density_sample.json` already shows SVD winning the
density-matched `W_eff` comparison in **1/35 at c=1, 2/35 at c=2, 12/35 at c=4, 15/35 at c=8, 24/35
at c=16** — coarsening saturates into the delete-branch ceiling (median 0.1841→0.2055 against 0.2000)
while SVD degrades gracefully. The literature independently favours the SVD family: **every** post-hoc
adapter-compression work found here (#3–#7) is SVD/spectral, and **zero** use block coarsening. So
`Dctl` is the operator with the *prior*, and coarsening is the challenger. `kill_gate.clause_5` is
therefore correctly pre-registered, and **nothing in B12 may describe Dctl as included-to-be-beaten**
(`must_not_claim[6]`).

**5.2 One genuine tension with the literature, which B12 should state rather than hide.**
SpectralLoRA (#7) reports LoRA updates are spectrally **concentrated** (33% of DCT coefficients →
90% of energy); B12 measured `E`'s spectrum **flat** vs a Gaussian-product null in 32/35 projections.
These are compatible — different backbones (BERT/RoBERTa vs Llama-2-7B), different objects (a
fine-tuning delta vs a sparsity-correction branch trained alongside a mask), different transforms
(DCT vs svdvals) — but **B12 must not state "low-rank branches have flat spectra" as a general fact.**
The measured claim is about **this branch on this checkpoint**.

---

## 6. Verdict and the delta

**0 of 30 preempt. B12 survives.** The delta is the **conjunction**, and every conjunct is load-bearing:

1. **The base is exactly 2:4 semi-structured**, and this is now *measured over all 224 in-scope
   tensors on CPU*, not assumed: `mask_ones_fraction = 0.500000`, **`mask_exact_2of4_violations = 0`**,
   `columns_outside_any_group_of_4 = 0`
   (`evidence/g0_leg2_rungA_selfcheck_20260816.json`). Every surface-(a)/(b) work (#3–#18) has a
   **dense** base. Shears (#17) is sparse but **unstructured**.
2. **The branch is genuinely trained** (17,900 iterations, `SLoRB_k=16`, `trainable_projection=true`),
   then **reduced post hoc with no retraining**. SLoPe (#19) adds lazily; SLiM (#20) constructs
   one-shot; Shears (#17) trains at the searched rank. None reduces a trained branch.
3. **The operator is block-basis coarsening with a closed-form LS refit**, not SVD.
   `argmin_S ‖S·B_coarse − E‖_F = (E·B_coarseᵀ)/k_eff` = the block **mean**, exact because the coarse
   basis columns are disjoint and constant-1 (verified on-block entries **exactly 1.0** in 12/12
   sampled projections). The literature's post-hoc adapter operator is uniformly SVD/spectral;
   `all:"SLoRB"` → 0 hits and no adapter-coarsening hits exist.
4. **The axis reported is deployment density (two-matmul live-parameter count), not adapter storage.**
   #3–#8 report storage/param-count reductions of an adapter in isolation. B12's ψ is defined against
   a **hard floor of 50.0%** (pure 2:4, zero branch) and a folded-dense ceiling of 63.1011%. That axis
   only exists because the base is N:M.
5. **The asymmetry is pre-registered per projection family** from an exhaustive 5⁷ = 78,125-map
   frontier scan, with the frontier shown to contain **no uniform map above ψ≈0.52**. PARA (#3) gets
   non-uniformity emergently from a global threshold; OWL (#29) does non-uniform budgets on the
   *sparsity* axis. Nobody does a pre-registered per-family **basis-width** map.

### Where B12 must cite rather than claim

| claim B12 might be tempted to make | must instead cite |
|---|---|
| the SLoRB branch / `x_proj` block-indicator init / `SLoRB_k` | **AST, AAAI 2025** (arXiv:2407.20584). The branch is AST's; `PROPOSAL.md` currently cites only file:line and **names neither AST nor CAST** — fix this. |
| "a low-rank branch on N:M-sparse weights" | **SLoPe, ICLR 2025** (arXiv:2405.16325) |
| "low-rank correction of a sparse base without retraining" | **SLiM, ICML 2025** (arXiv:2410.09615) |
| "adapter rank is over-provisioned; train big then compress" | **LoRA-Squeeze** (arXiv:2602.10993, preprint); **AdaLoRA, ICLR 2023**; **DyLoRA, EACL 2023** |
| "post-hoc SVD of a trained adapter with non-uniform per-layer rank" (= what Dctl is) | **PARA** (arXiv:2604.27796, preprint, CONCURRENT) — Dctl is **not** a novel control |
| "the basis need not be stored" | **VeRA, ICLR 2024**; **NOLA, ICLR 2024** |
| "elastic adapter rank on a sparse base" | **Shears, NAACL 2024 Industry** (`2024.naacl-industry.34`) |
| "non-uniform per-layer budget from measured sensitivity" | **OWL, ICML 2024** |
| "training-free post-hoc surgery on a sparse LLM" | **DSnoT, ICLR 2024** |
| the CAST-repro comparison row | **CAST** (arXiv:2509.25996) — **PREPRINT, "Submitted to IEEE TPAMI"; do not assign it a venue** |

### Honest limits of this check

- **9 entries carry a venue I could not raise to a conference record** (#3 PARA, #4 LoRA-Squeeze,
  #5 Spectral Surgery, #6 PHLoRA, #7 SpectralLoRA, #15 LoRA-FA, #18 NAS-companion, CAST,
  `2603.17917`). Each is marked **PREPRINT** or **UNVERIFIED with the authority tried**. None of them
  changes the verdict: they are all dense-base or base-weight work.
- **AdaLoRA's OpenReview note was not located** by title search; its ICLR 2023 venue rests on DBLP
  `conf/iclr/ZhangCBH0CZ23`. Flagged in §0.
- **Semantic Scholar was not consulted** (429 in this environment). Its absence is not evidence.
- Bibliographic fields here are copied from API responses; **nothing is reconstructed from memory**.
  Raw responses used are reproducible by re-issuing the queries recorded in §0 and §4.
