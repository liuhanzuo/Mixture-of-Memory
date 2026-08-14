# B03 — RELATED WORK / NOVELTY ADJUDICATION

**Written 2026-08-15. 0 GPU, 0 SSH. Adjudication of an existing 176 KB corpus, not a new search.**

This closes the blocker `proposal/ready_queue.py:504-515` trips on
(`RELATED_WORK.md absent (blocks PROMOTION; 0-GPU task)`) and answers the five collision
families named at `proposal/shared/literature/RELATED_WORK_GAP_AUDIT_20260808.md:93`
(rating **部分充分** / partially sufficient; families: **LLF / layer reinitialization;
plasticity loss; prune-regrow; optimizer-state reset; single-pass vs repeated-data**).

## 0. Two constraints this file is bound by BEFORE any literature is discussed

**(A) The gap audit's safe-boundary clause, verbatim:**

> 「已正确降级为 regime-boundary gate，**不得包装成新 reset 方法**。」
> (*Already correctly demoted to a regime-boundary gate; must NOT be repackaged as a new
> reset method.*)

`PROPOSAL.md` §「不是新方法」 says the same thing at source, and lists three forbidden claims
verbatim: `cyclic prune-regrow 新方法`, `depth cycling`, `新 plasticity 机制`.
**Everything below preserves that demotion. This file does not, at any point, argue that B03
proposes a reset method.** §3 explains why the literature makes that repackaging not merely
disallowed but hopeless.

**(B) B03 has NO GATE. This file must not pretend otherwise.**
`STATUS.json:next_gate` is the sentinel `NOT_SPECIFIED`, and its own text says why: the
`PROPOSAL.md` §「1B 核心 gate」 section specifies a **2×3 design** (data regime
{single-pass, repeated-data} × reset count N∈{0,1,3}, at 1B) but **no decision rule — no n,
no statistic, no alpha, no effect-size threshold, no null-floor magnitude, no comparator.**
Its survival clauses (「显著 reset × data-regime interaction」,「PPL 与知识恢复曲线显著分离」)
name a statistic *family* without an n or a threshold, so **no outcome of the 2×3 as written
would decide the question.** `gpu_cost_estimate` is `UNKNOWN`.

> **Writing the read-out is a separate 0-GPU task and is NOT this file's job.** Clearing
> novelty does **not** make B03 `ready_gpu`, and `STATUS.json` deliberately keeps
> `next_gate: NOT_SPECIFIED` so that clearing novelty alone cannot make `ready_gpu`
> inferable. Per `memory/a-declared-lifecycle-is-not-an-adjudicated-one.md`, a declared
> lifecycle is not an adjudicated one. **This file changes `novelty_checked` only.**

**(C) Standing rule on preemption** — `memory/prior-work-differentiate-dont-abandon.md`
(user, 2026-08-07): the bar is **完全相同 / 抄袭**, not overlap; work within 2–3 months is
**concurrent**; a direction dies from its **own kill gate**, never from a literature count.
B03's kill gate is `PROPOSAL.md` §「关闭条件」 (three qualitative clauses) and **has never been
run**. So nothing below can kill B03 — but §3 shows that the *method-shaped* version of B03 is
already dead on the operator, which is exactly why the demotion in (A) is correct rather than
merely cautious.

---

## 1. What was adjudicated, and the discipline used

The corpus was **already built** on 2026-08-06 and is 176 KB across five files in
`literature/`. `STATUS.json:novelty_status_detail_20260815` is explicit that the 0-GPU task is
to **ADJUDICATE these, not to re-search**:

| File | Size | Its own verdict |
|---|---|---|
| `AUDIT0_venue_verification.md` | 31.9 KB | 49 arXiv IDs, five-channel venue verification, **0 UNRESOLVED**; 32 peer-reviewed / 17 preprint |
| `KILLCHECK_forward_citations.md` | 21.1 KB | **SURVIVES** — 434 forward citations across 6 seeds + 10 tech-report full-text greps + OpenReview sweep; no published paper hits all five criteria |
| `SKEPTIC1_vs_monotonic.md` | 36.3 KB | **WEAKENED** (heavily) — LLF (ICLR 2022) is mathematically the same operator |
| `SKEPTIC2_vs_cyclic.md` | 45.3 KB | **WEAKENED** — the benefit is confined by the original authors to small-data/overfitting regimes |
| `SKEPTIC3_vs_plasticity.md` | 42.2 KB | **WEAKENED** — 2 of AUDIT3's 4 surviving "doors" are refuted outright |

**Verification discipline for this pass.** Every venue below was **independently re-checked
this session** (2026-08-15) — the 2026-08-06 corpus used S2 as a primary channel, which
`memory/venue-verify-must-use-openreview-2026.md` forbids as an authority, so the calls needed
redoing with family-correct authorities:

| Family | Authority | Status from this node |
|---|---|---|
| ICLR / NeurIPS / ICML / TMLR | OpenReview `venueid` (+ `Camera_Ready_Revision`) | ⚠️ **`api2.openreview.net` returns HTTP 403 `ChallengeRequiredError` on every path.** Only **API v1** works, which gives `venue`/`venueid` but **not** the invitation list → **no `Camera_Ready_Revision` check was possible this session.** |
| ACL / EMNLP / NAACL / EACL incl. Findings | ACL Anthology + DBLP | ✅ both work |
| everything, as cross-check | **DBLP `search/publ/api`** | ✅ works and carried most of the load this session |
| arXiv metadata | `arxiv.org/abs/<id>` HTML `citation_*` + comment/jref cells | ✅ works (the **API** is 429-limited) |
| Semantic Scholar | — | ❌ HTTP 429. **Not used as an authority anywhere.** |

`arXiv-only` below means **"no peer-reviewed venue verifiable from this node"**, not
"no venue exists".

---

## 2. Named closest collisions, by family

### 2.1 Family: LLF / layer reinitialization — ★ THE OPERATOR COLLISION

| Paper | Year | Venue (authority) | What it does |
|---|---|---|---|
| **Zhou, Vani, Larochelle, Courville — *Fortuitous Forgetting in Connectionist Networks* (LLF)** | 2022 | **ICLR 2022 Poster.** DBLP `conf/iclr/ZhouVLC22` (venue `ICLR`, year 2022) **+** OpenReview **v1** note `ei3SY1_zYsE`, `venue = "ICLR 2022 Poster"`, `venueid = ICLR.cc/2022/Conference`. **Both re-verified this session.** ⚠️ `Camera_Ready_Revision` not checkable (api2 blocked); the arXiv comment says `ICLR Camera Ready` and the jref says `ICLR 2022`. | Proposes **later-layer forgetting**: mask $M^l_{\mathrm{LLF}} = 1$ if $l<L$, $0$ if $l\ge L$ → the layers at and above threshold $L$ are given **a new random initialization** (its footnote 3 distinguishes "reset/reinitialize = a new initialization" from "rewind = back to original init"). Run for **N=3/8/10 generations**. Ablates *fixed* vs *fresh* reinit per generation and finds **fresh is necessary**. Also ablates **early-layer instead of later-layer** reset and finds it worse. |
| Sarfi, Karimpour, Chaudhary, Khalid, Ravanelli, Mohammadi, Bacon — *Simulated Annealing in Early Layers Leads to Better Generalization* (SEAL) | 2023 | **CVPR 2023.** DBLP `conf/cvpr/SarfiKCKRMB23` — re-verified this session. | Direct LLF follow-up. Reports that **LLF features degrade transfer learning across all datasets explored**. |
| Alabdulmohsin, Maennel, Keysers — *The Impact of Reinitialization on Generalization in Convolutional Neural Networks* | 2021 | **arXiv-only.** DBLP `journals/corr/abs-2109-00267` (CoRR 2021) — no `conf/` record. Re-verified this session. | Bottom-up **layerwise** reinitialization across many architectures. §5: benefit is *"particularly for small datasets"* and **"For large datasets, however, reinitialization does not seem to offer a benefit."** Its decision tree splits near the root on **"Training Set Size < 35K?"**. |
| Frati, Traub, Cianci, Cattaneo — *Reset It and Forget It: Relearning Last-Layer Weights…* | 2024 | **ECAI 2024.** DBLP `conf/ecai/FratiTCC24` + arXiv comment `Published in ECAI 2024`. Re-verified this session. | Periodic last-layer reset for continual/transfer learning. |

**⚠️ THE HARD FACT, and it is our own code that establishes it.** `SKEPTIC1` §1.3 and
`SKEPTIC2` §1.4 both checked `scripts/train_olmo2_arch_probe2.py` (docstring lines 9–11 +
`transplant_front()`, which asserts `missing_layer_ids == range(keep_front, keep_front+n_fresh)`).
With total depth held constant, **"keep the front $K_f$ layers + append $K$ fresh
randomly-initialised layers" and "reinitialize every layer at or above index $K_f$ in place"
produce the same function class, the same parameter shapes, and the same init distribution** —
they differ only in implementation path (new module object vs in-place re-init). That is
**literally $M^l_{\mathrm{LLF}}$ with $L=K_f$.**

**Consequence, stated without hedging: at the operator level a cyclic version of B03's
construction, built from the existing `--keep_front_layers` / `--n_fresh_layers` flags, IS LLF.**
A reviewer needs one sentence — *"this is LLF (Zhou et al., ICLR 2022) applied to LM
pretraining"* — and a method claim is at zero. `SKEPTIC1` §9 says it cannot rebut that
sentence, and neither can this file. **This is why (A)'s demotion is the correct reading of the
evidence, not a defensive posture.**

**What LLF/2109.00267/SEAL do NOT do**, and it is a real gap: all three are **CNN image
classification** (ResNet18/50, WideResNet-28-10, DenseNet-BC; CIFAR-10/100, Flower-1020,
CUB-5994, Aircraft-3334, MIT67-5360, Dogs-12000). **None is a decoder-only transformer, none
is LM pretraining, none measures parametric/factual knowledge.** LLF's own effective regime is
**thousand-sample, "prone to overfitting"** tasks (§4.1), and its Table A8 reports LLF
**losing** to the baseline once the data gets larger and the baseline stronger
(WRN-28-10 CIFAR-10 96.32 → 95.91; CIFAR-100 81.29 → 80.95).

### 2.2 Family: plasticity loss (incl. the LLM-scale leg)

| Paper | Year | Venue (authority) | What it does / why it collides |
|---|---|---|---|
| **Chen, Marchisio, Raileanu, Adelani, Stenetorp, Riedel, Artetxe — *Improving Language Plasticity via Pretraining with Active Forgetting*** | 2023 | **NeurIPS 2023.** DBLP `conf/nips/ChenMRAS0A23` — re-verified this session. | **★ The framing collision.** Abstract: *"by **resetting the embedding layer every K updates during pretraining**, we encourage the PLM to improve its ability of learning new embeddings"*; §3 also **resets the optimizer states and LR scheduler together with the embedding layer**; Figure 4 describes the *"episodic pattern … every embedding forgetting produces a **loss spike, from which the model learns to recover. Through such repeats of forget-relearn**"*. **The motivation paragraph B03 would want to write was published in 2023.** |
| **Springer, Goyal, Wen, Kumar, Yue, Malladi, Neubig, Raghunathan — *Overtrained Language Models Are Harder to Fine-Tune*** | 2025 | **ICML 2025.** DBLP `conf/icml/SpringerGWKYMNR25` — re-verified this session. (arXiv comment carries no venue → S2/comment alone would have missed it.) | **★ The timing-axis collision.** Names **catastrophic overtraining** and **progressive sensitivity**: *"For a fixed magnitude of perturbation, the change in perplexity between the base model and the perturbed model **increases monotonically with the number of pre-training tokens**."* Measured on **OLMo-1B (3T), OLMo-2-7B, LLM360-Amber-7B** — **our own model family and scale**. So "cost of structural damage vs *when* in pretraining it happens" is an **already-published monotone curve**. |
| Shin, Oh, Lee, Yun — *DASH: Warm-Starting Neural Network Training in Stationary Settings without Loss of Plasticity* | 2024 | **NeurIPS 2024.** DBLP `conf/nips/ShinO0Y24` + arXiv comment `Published at NeurIPS 2024` — re-verified this session. | Beats Shrink-&-Perturb; and **Appendix C.1 already reports that RL-style `Reset` and L2-INIT "cannot be a solution" under a stationary data distribution**, attributing plasticity loss to **noise memorization** instead. LM pretraining is (near-)stationary → **a published negative prior for reset-type interventions in our regime.** |
| Anon. — *FIRE: Frobenius-Isometry Reinitialization for Balancing the Stability-Plasticity Tradeoff* | 2026 | ⚠️ **arXiv comment says `ICLR'26 (oral)`; DBLP has only `journals/corr/abs-2602-08040` (CoRR 2026); `venueid` NOT verifiable this session (api2 403).** `KILLCHECK` reported OpenReview forum `CfZLxT3zIZ` under `ICLR.cc/2026/Conference` on 2026-08-06; **I could not reproduce that today.** → **Record as `ICLR 2026 oral (self-reported; venueid unverified this session)`.** | Its abstract's *second sentence* treats **"standard reinitialization methods … are widely used"** as background, and frames the trade-off as *"conservative reinitializations fail to restore plasticity, while aggressive ones **erase useful knowledge**"* — i.e. B03's intended trade-off, as a 2026 background sentence. Evaluates **language modeling (OpenWebText, GPT-0.1B)** → "reinit-for-plasticity has never been done on LMs" is **false**. |
| Hernandez-Garcia, Figliolia, Millidge — *Can Scale Save Us From Plasticity Loss in Large Language Models?* | 2026 | **arXiv-only.** DBLP `journals/corr/abs-2606-24752` (CoRR 2026); no comment/jref — re-verified this session. | Plasticity loss in GPT-style transformers 5M–314M, sublinear scaling-law onset; and its §II lists *"periodically reinitializes units"* (Continual Backprop, ReDo, SNR, GraMa) as **a standard mitigation family** — so "periodic reinitialisation for plasticity" is **survey furniture in the LLM literature, not a blank cell**. |
| Han, Bordt, Zhang, Kakade — *Weight Decay Improves Language Model Plasticity* | 2026 | ⚠️ **arXiv-only from this node.** DBLP `journals/corr/abs-2602-11137` (CoRR 2026); no comment/jref. `KILLCHECK` says OpenReview `ICML.cc/2026/Conference` (2026-08-06) and `AUDIT0` row 2602.11137 says `venueid` + `Submission26903/-/Camera_Ready_Revision` — **neither reproducible today (api2 403)**. → **`ICML 2026 (reported by two prior in-repo passes; unverified this session)`.** | Owns the research question **"pretraining choice → downstream adaptability, and validation loss is not enough."** Its independent variable is weight decay; B03's would be structural reset. **Same story skeleton, different knob.** |

### 2.3 Family: prune-regrow (dense→sparse→dense and its LLM descendants)

| Paper | Year | Venue (authority) | Collision and its limit |
|---|---|---|---|
| Han, Pool, Narang, Mao, Gong, Tang, Elsen, Vajda, Paluri, Tran, Catanzaro, Dally — *DSD: Dense-Sparse-Dense Training* | 2017 | **ICLR 2017 Poster.** DBLP `conf/iclr/HanPNMGTEVPTCD17` + OpenReview **v1** `HyoST_9xl` (`venue = "ICLR 2017 Poster"`, `venueid = ICLR.cc/2017/conference`) — re-verified this session. | The canonical destroy-then-restore-capacity cycle. **`SKEPTIC1` §9 corrects AUDIT1: DSD is *iterative*** (Algorithm 1's `goto Sparse Phase`; §4.3 *"A second DSD iteration can further improve the accuracy"*), so "multi-round cycling" is not a moat. **Limit**: weight-level, not layer-level. |
| Peste, Iofinova, Vladu, Alistarh — *AC/DC: Alternating Compressed/DeCompressed Training* | 2021 | **NeurIPS 2021 Poster.** DBLP `conf/nips/PesteIVA21` + OpenReview **v1** `T3_AJr9-R5g` (`venue = "NeurIPS 2021 Poster"`, `venueid = NeurIPS.cc/2021/Conference`) — re-verified this session. | Many alternating compress/decompress phases with a **deliverable same-size endpoint** → kills "multi-round + size-preserving" as a distinguishing pair. **Limit**: unstructured sparsity, CNNs. |
| Thangarasa, Gupta, Marshall, Li, Leong, DeCoste, Lie, Saxena — *SPDF: Sparse Pre-training and Dense Fine-tuning for LLMs* | 2023 | **UAI 2023** (per `SKEPTIC2` §4: S2 `type=conference` + arXiv comment self-report). ⚠️ **Not re-verified this session** — UAI is neither an Anthology nor a reliably-OpenReview venue, and api2 was blocked. Carry as **UAI 2023 (in-repo prior pass; not re-verified)**. | 75% sparsity into **1.3B GPT-3 XL**, then dense recovery. **Kills the "no prune-regrow work above 314M" scale-vacuum argument.** **Limit**: one-shot (not cyclic), unstructured weights, and its motive is training-FLOPs efficiency. |
| Anon. — *Sample-efficient LLM Optimization with Reset Replay* (LoRR) | 2025 | **arXiv-only.** DBLP `journals/corr/abs-2508-06412` (CoRR 2025) — re-verified this session. | Periodic reset on **Qwen2.5-7B-class** models. ⚠️ **Its ablation reports that resetting `full_layers` "proves detrimental, likely due to the destruction of learned features essential for reasoning"** → a **7B-scale published negative** for full-layer reset. **Limit**: post-training (DPO/RLHF), not pretraining. |
| Wang, Shen, Ding, Xue, Liu, Ding — *Layer as Puzzle Pieces* (CoMe) | 2025 | **NeurIPS 2025 Poster** (verified in `proposal/active/A04-recovery-certification/RELATED_WORK.md` §C3 via OpenReview `venueid` + `Camera_Ready_Revision`). Not re-verified here. | Layer merging + hierarchical distillation recovery — the layer-granularity prune-and-recover state of the art B03 would be compared against. |

### 2.4 Family: optimizer-state reset

**This is the family with the least room, and the audit was right to name it.**
**Active Forgetting (NeurIPS 2023) §3 already resets the optimizer states and the LR scheduler
together with the reinitialised layer**, and gives the reason (*"pretraining involves advanced
training strategies, like optimizers with states and learning rate schedulers"*).
`PROPOSAL.md`'s shared condition 「reset layers 的 optimizer moments 同时重置」 is therefore
**a correctness requirement inherited from published practice, not a contribution.**
LLF's per-generation retrain and DSD/AC-DC's phase restarts carry the same implicit
requirement. **B03 must present optimizer-moment reset as protocol hygiene it copies, and must
cite Active Forgetting §3 for it.**

### 2.5 Family: single-pass vs repeated-data — ★ THE AXIS THAT ACTUALLY SURVIVES, PARTLY

| Paper | Year | Venue (authority) | Relation to B03's axis 1 |
|---|---|---|---|
| **Muennighoff, Rush, Barak, Scao, Tazi, Piktus, Pyysalo, Wolf, Raffel — *Scaling Data-Constrained Language Models*** | 2023 / 2025 | **NeurIPS 2023** (DBLP `conf/nips/MuennighoffRBST23`) **and** a **JMLR 2025** version (DBLP `journals/jmlr/MuennighoffRBSP25`) — **both re-verified this session.** ⚠️ **Two records exist; pick one deliberately and do not cite "NeurIPS/JMLR" ambiguously.** | **Owns the single-pass-vs-repeated-data axis for LM pretraining**: up to **4 epochs of repeated data ≈ negligible loss change**, beyond which the value of added compute decays to zero; supplies a validated scaling law. Up to 900B tokens / 9B params, ~400 runs. **B03's axis 1 is this paper's independent variable. B03 may not claim it — it must adopt it and cite it.** |
| Allen-Zhu & Li — *Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws* | 2025 | **ICLR 2025.** DBLP `conf/iclr/Allen-ZhuL25` — re-verified this session. | The reference for **how many bits of factual knowledge** a given parameter budget stores and how **exposure/repetition** governs it. B03's outcome axis is exactly "parametric knowledge", so this is the yardstick a reviewer will demand. |
| Anon. — *Data-Constrained Language Model Pretraining: Improved Regularization and Scaling Laws* | 2026 | **arXiv-only.** DBLP `journals/corr/abs-2606-06888` (CoRR 2026); no comment/jref — re-verified this session. | 2026 continuation: proposes **SoftQ**, arguing the additive Chinchilla-style form is **misspecified under repeated data**. **Relevant and concurrent**: if B03's 2×3 crosses data regime with anything, the repeated-data cell needs a scaling form that is not misspecified. |

**Why this family is the least occupied *in combination*.** Data-constrained pretraining work
measures **loss/scaling** under repetition; the reset literature measures **plasticity/
generalization** under intervention. **Nothing found crosses them** — i.e. nobody asks whether
a destructive layer-level forget-and-relearn intervention behaves *differently* under
single-pass versus repeated data. That crossing is B03's `PROPOSAL.md` survival clause 1
(「显著 reset × data-regime interaction」). It is a **real, unoccupied cell** — but it is an
**interaction effect**, and an interaction is the *hardest* thing to power. **With no n, no
statistic and no alpha (§0(B)), B03 currently cannot say whether it could detect one.**

### 2.6 Nearest misses recorded by `KILLCHECK` (re-verified where possible)

| Paper | Venue (re-verified this session) | Which single criterion it misses |
|---|---|---|
| *Exploring Pretraining via Active Forgetting … for Decoder Language Models* (2410.16168) | **arXiv-only.** DBLP `journals/corr/abs-2410-16168` (CoRR 2024) | **Granularity.** Decoder-only ✓, cyclic ✓ (every ~10k steps), pretraining ✓, size-preserving ✓ — but resets **token embeddings**, not decoder blocks. |
| *Forget to Generalize: Iterative Adaptation …* (2602.04536, IFA) | **arXiv-only** (`KILLCHECK`: no comment/jref) | **Model type.** Later-layer reinit ✓, cyclic ✓, size-preserving ✓ — but CIFAR-10 / MIT-Indoors / Stanford Dogs under federated learning. |
| *FIRE* (2602.08040) | See §2.2 — **ICLR 2026 oral self-reported, `venueid` unverified today** | **Granularity.** Reinitialises individual weight matrices (Newton–Schulz on Q/K projections), not whole blocks. |
| *LoRR* (2508.06412) | **arXiv-only**, DBLP CoRR 2025 | **Training phase.** Post-training, not pretraining — and reports full-layer reset is harmful. |
| *LLF* (2202.00155) | **ICLR 2022** (§2.1) | **Model type / domain.** CNN image classification. |

`KILLCHECK` also full-text grepped **10 LLM tech reports** (OLMo 2, Qwen 2.5, Qwen3, Llama 3,
DeepSeek V3, Gemma 2, Gemma 3, MiniCPM, Falcon 180B, Mistral 7B) and found **no mid-training
layer discard/reset** — only optimizer warm-up, SGDR warm-restarts, and job restarts. **That
negative is the strongest single piece of evidence that the exact cell is empty in published
practice.**

---

## 3. MUST-NOT-CLAIM list (binding on any B03 writeup)

1. ❌ **A new cyclic prune-regrow method / depth cycling / a new plasticity mechanism.**
   Forbidden by `PROPOSAL.md` §「不是新方法」 **and** by the gap audit's boundary clause. At the
   operator level the construction **is** LLF's mask (§2.1).
2. ❌ **Any priority for "reinitialise the upper layers and retrain, repeatedly, at fixed total
   size."** **LLF (ICLR 2022)** owns it, including the N=3/8/10 cycling, the fresh-vs-fixed
   reinit ablation, and the early-vs-later direction ablation. **A B03 run built from
   `keep_front + n_fresh` would re-run published ICLR 2022 ablations.**
3. ❌ **"Periodically resetting a module during LM pretraining to buy plasticity" as a framing.**
   **Active Forgetting (NeurIPS 2023)** owns it, with the loss-spike / forget-relearn / episodic
   narrative.
4. ❌ **Resetting optimizer moments along with the reset layers, as a contribution.**
   **Active Forgetting §3** already does it and states the reason (§2.4).
5. ❌ **"Cost of structural damage versus *when* in pretraining it is applied" as a new curve.**
   **Springer et al. (ICML 2025)** owns it under the name **progressive sensitivity**, measured
   on **OLMo-1B / OLMo-2-7B**. A B03 timing sweep would be *"Springer et al. with a different
   perturbation operator"* unless it shows the discrete/structural operator behaves
   **qualitatively** differently from their Gaussian and fine-tuning perturbations.
6. ❌ **"Reinit-for-plasticity has never been tried on language models."** False:
   **FIRE** evaluates GPT-0.1B on OpenWebText; **Active Forgetting** is RoBERTa-base;
   **2606.24752** covers 5M–314M GPT-style; **LoRR** is Qwen2.5-7B-class post-training.
7. ❌ **"Nothing in this line reaches ≥1B."** False: **SPDF** is 1.3B; **Springer** is 1B–7B;
   **LoRR** is 7B-class.
8. ❌ **"Structural granularity beats weight granularity"** on the strength of RePr's Table 3.
   `SKEPTIC2` §2 caught this misreading: **RePr's own text says weight-level DSD (7.8) and
   filter-level RePr-Weights (7.7) *"perform roughly the same function … similar
   performance"***, and the 6.9 improvement comes from **changing the metric** (inter-filter
   orthogonality), **not** the granularity. **Copying that sentence into a paper would
   misrepresent a cited work.**
9. ❌ **Any claim of a first single-pass-vs-repeated-data study.**
   **Muennighoff et al. (NeurIPS 2023 / JMLR 2025)** owns that axis for LM pretraining.
10. ❌ **Any claim that a knowledge-vs-PPL separation is a *new phenomenon*.** It is
    **our own Paper B's existing asset**, not a B03 discovery: keep14 at step 200k reaches
    **PPL tax 1.428×** while recovering only **19.5%** of base above-chance MMLU
    (`status/PAPERB_KEEP14_200K_EVAL.md` lines 11/13, verified this session).
    `SKEPTIC3` §4 makes this point explicitly. B03 may **use** it as a prior; it may not
    **claim** it.
11. ❌ **Presenting the reset-cost asymmetry as a *reason to expect B03 to work*.**
    `SKEPTIC2` §3.2 flags this as a self-contradiction: the same Paper B fact read as a prior
    predicts that **each additional reset cycle pays another near-irreversible knowledge tax**,
    i.e. **monotone worsening in N**. Combined with `2109.00267` §5 (no benefit on large data),
    **LLF Table A8** (loses to baseline once data/baseline strengthen), **SEAL** (LLF degrades
    transfer), **DASH App. C.1** (reset ineffective under stationary data), and **LoRR**
    (full-layer reset detrimental at 7B) — **there are five independent published negative
    priors and one internal one.** Any B03 write-up must state them **before** its results.

---

## 4. Safe residual claim — one falsifiable sentence

> **The forget-and-relearn benefit that layerwise reinitialisation delivers in
> small-data/multi-epoch vision has a regime boundary, and single-pass knowledge-dense
> decoder-only LM pretraining lies on the far side of it: destructive top-$K$ layer reset
> there produces recoverable perplexity alongside parametric-knowledge loss that does not
> recover, and the deficit grows with reset count $N$ — with the reset $\times$ data-regime
> (single-pass vs repeated) interaction as the discriminating measurement.**

This is a **regime-boundary / negative-result claim**, which is what (A) permits. It is
testable and it is genuinely two-sided:
* **If the interaction is null and degradation is uniformly monotone in $N$** → that is
  `PROPOSAL.md` §「关闭条件」 clauses 1 and 2 and the direction **closes**. Death by its own
  kill gate, which is the only admissible death.
* **If the interaction is significant** → it is the one cell §2.5 shows is unoccupied: the
  crossing of the data-constrained-pretraining axis (Muennighoff et al.) with the
  reset/plasticity axis (LLF, Active Forgetting, DASH), on a model class and outcome variable
  (parametric knowledge) that neither literature measures.

**⚠️ This sentence is NOT yet a gate.** It has no $n$, no statistic, no $\alpha$, no
effect-size threshold and no null-floor construction — exactly the gaps
`STATUS.json:next_gate_design_extracted_from_PROPOSAL_20260815.what_is_missing_before_this_is_executable`
enumerates. **Writing that read-out is the blocking 0-GPU prerequisite and it is not
discharged by this file.** And per §2.5, the surviving claim is an **interaction**, which is
the least powerful thing to detect — so the read-out has to justify its $n$ against the
interaction, not against a main effect.

**Two design constraints the literature forces on whatever read-out gets written:**
1. **The reset operator must not be `keep_front + n_fresh` if any structural distinction from
   LLF is wanted.** `SKEPTIC2` §1.4: top-$K$ via those flags is **exactly** LLF's mask. A
   mid-stack excision (non-top-$K$, non-contiguous-to-top) would differ, but **the current
   trainer does not support it** — that is new code, not glue.
2. **The comparator is no longer Shrink-&-Perturb.** `SKEPTIC3` §3 raises the bar: **DASH
   (NeurIPS 2024) already beats S&P**, so S&P is not the thing to beat. And DASH's own protocol
   (incrementally growing dataset, trained to 99.9% train accuracy) is **not** single-epoch
   streaming pretraining — so its **numbers are not directly comparable** and must be used as a
   prior and as reviewer ammunition, never as a cross-tabulated baseline.

---

## 5. Honest gaps in this adjudication

1. ⚠️ **The repo's mandated OpenReview route is DOWN.** `api2.openreview.net` → HTTP 403
   `ChallengeRequiredError` on every path tried (`notes/search`, `notes?forum=`,
   `notes?content.venueid=`). API **v1** works but exposes no invitation list, so **zero
   `Camera_Ready_Revision` checks were possible.** Where a call rests on v1 or DBLP alone I said
   so. **Three calls are weaker than the repo standard requires and must be redone before any
   B03 write-up**: **FIRE** (2602.08040 — arXiv comment `ICLR'26 (oral)`, DBLP CoRR only;
   `KILLCHECK`'s forum `CfZLxT3zIZ` not reproducible today), **Weight-Decay-Plasticity**
   (2602.11137 — DBLP CoRR only; ICML 2026 asserted by two earlier in-repo passes), and
   **SPDF** (UAI 2023, from an S2-based in-repo pass, and UAI is covered by neither
   family authority).
2. ⚠️ **Semantic Scholar 429 all session, arXiv API 429 all session.** The 2026-08-06 corpus
   used S2 as a *primary* channel — which `memory/venue-verify-must-use-openreview-2026.md`
   forbids — so its S2-only calls (notably **SPDF/UAI 2023**, **`2502.07274` Cho et al. ICLR
   2026 from a third-party citation**) are **inherited, not re-verified**. `AUDIT0`'s own
   §5 also records `2602.11137` as *"疑 ICML 2026，需 MAIN 复核"* in `SKEPTIC1` before
   `KILLCHECK` upgraded it — that upgrade is what I could not reproduce.
3. ⚠️ **`Muennighoff et al.` has TWO DBLP records** (`conf/nips/MuennighoffRBST23` **and**
   `journals/jmlr/MuennighoffRBSP25`, with different author-initial strings). Cite one
   deliberately; do not write "NeurIPS/JMLR".
4. **No full-text PDF was read this session.** Every "what it does" above is from
   (a) the abstract fetched this session, or (b) a **verbatim quotation already recorded in the
   `literature/` corpus** with its source section named. The corpus's quotations were taken from
   `ar5iv`/`arxiv.org/html` full texts that were stored in **`/tmp`** on 2026-08-06 — and
   per `memory/persist-artifacts-on-wzc1-or-diskb.md` **`/tmp` does not survive a restart, so
   those extraction artifacts are gone.** The quotations are therefore **not independently
   re-checkable from this node**; they are cited as the corpus's records. `SKEPTIC1`'s own
   footer warns of exactly this.
5. **`KILLCHECK`'s declared blind spots stand unchanged**: some S2 `paper/search` queries 429'd;
   no Chinese/Japanese/Korean-language venues searched; industrial internal tech notes and
   in-review ICLR 2027 / NeurIPS 2026 submissions are undetectable; S2 lags 2026-06→08 arXiv.
6. **Recency sweep this session was thin and returned nothing.** Six arXiv full-text/abstract
   queries — `"reinitializ" "layers" "pretraining" "language model"`, `"forget-and-relearn"`
   (3 hits, all already in the corpus: 2310.07996, 2202.00155, plus an unrelated neuromorphic
   paper), `"layer reset" "pretraining"`, `"reinitializing" "transformer layers"
   "pretraining"`, `"plasticity" "pretraining" "reset"`, `"repeated data" "epochs"
   "pretraining" "language model"` — surfaced **one** new item (2606.06888, §2.5). **This is
   not a substitute for a forward-citation re-scan**, which is what would actually catch a
   2026-06→08 collision; the corpus's 434-citation scan is 9 days old and was not redone.
7. **`PROPOSAL.md` and `SOURCES.md` unmodified.** The gap audit's remedy for B03 is scoped to
   the boundary clause, which is already in `PROPOSAL.md` §「不是新方法」. `SOURCES.md` still
   lists only `literature/KILLCHECK…` + `literature/AUDIT0…` and omits the three SKEPTIC files
   that carry the load here — whoever next edits it should add them.
8. **Zero cross-disk verification.** Everything cited is on **wzc1**. Nothing here asserts a
   file is absent, so `memory/two-disk-rule-applies-to-main-too.md` is not violated — but the
   trainer facts in §2.1/§4 were read from the **wzc1 copy** of
   `scripts/train_olmo2_arch_probe2.py`, and the zwfy6 checkout is a separate, often-lagging
   copy.

---

## 6. Verdict

```
related_work_status: audited
novelty_status: SURVIVES as a REGIME-BOUNDARY / NEGATIVE-RESULT question ONLY.
                DEAD as a method claim -- LLF (ICLR 2022) owns the operator.
                No paper is 完全相同/抄袭 of the surviving question.
lifecycle:      UNCHANGED (ready_cpu). This file does NOT make B03 ready_gpu:
                next_gate is still NOT_SPECIFIED and gpu_cost_estimate is still UNKNOWN.
strongest_collision: Zhou/Vani/Larochelle/Courville, LLF, ICLR 2022
                     (DBLP conf/iclr/ZhouVLC22 + OpenReview v1 ei3SY1_zYsE) -- its mask
                     M^l = 1[l<L] IS our keep_front/n_fresh construction. Not preemption of
                     the SURVIVING question because it is CNN image classification with no
                     LM pretraining, no decoder-only transformer, no parametric-knowledge
                     axis, and its own Table A8 reports it LOSING once data/baseline
                     strengthen -- which is the regime boundary B03 proposes to locate.
runner_up:           Springer et al., ICML 2025 (DBLP conf/icml/SpringerGWKYMNR25) -- owns
                     damage-cost-vs-pretraining-time on OLMo-1B/OLMo-2-7B, i.e. our own
                     family and scale. Kills the timing axis as a contribution.
priority:            UNCHANGED low / hold_gate_only (PROPOSAL.md: 「HOLD / GATE-ONLY。最低優先級。」)
```

**Why this is not an abandonment**, per `memory/prior-work-differentiate-dont-abandon.md`:
the audit's output is a **citation-obligation list** (§3, eleven items) plus **one unoccupied
cell** (§2.5: reset × data-regime interaction, measured on parametric knowledge in a
decoder-only LM) plus **six published negative priors** that make the question worth answering
in *either* direction — if the interaction is null, the direction closes by its own kill gate;
if it is real, it is the first crossing of two literatures that have never been crossed.
**B03 dies only from `PROPOSAL.md` §「关闭条件」, and that gate has not been run — nor can it
be, until the read-out in §0(B) is written.**
