# Paper D Post-Mortem: Independent Direction Review
**Reviewer**: Claude Sonnet 4.6 (independent, not MAIN)  
**Date**: 2026-08-06  
**Scope**: What to do after Paper D (model stitching) is dead; literature independently verified via arxiv API and WebFetch

---

## Verification Protocol

All citations below were fetched independently via `http://export.arxiv.org/api/query` and/or `https://arxiv.org/abs/<id>` with proxy `hy-proxy.woa.com:3128`. Papers confirmed are marked **[VERIFIED]** with the method used. Papers that could not be fetched are marked **[UNVERIFIED]**. No citations are copied from MAIN's reports.

---

## Q1: Five Independent, Novel Candidate Directions

### Direction D1: Layer-PPL-Knowledge Dissociation as a Measurement Science Paper

**One-sentence pitch**: Systematically document, across multiple model families, the three-way dissociation between (a) token-level perplexity, (b) factual knowledge benchmarks (MMLU/TriviaQA/PopQA), and (c) structural integrity (intact-depth layer count) after depth pruning followed by continued pretraining — and establish which of these axes best predicts downstream usability.

**Existing assets that make this cheap**:
- Paper B's entire experimental apparatus (8 model checkpoints, 5 eval benchmarks, matched-PPL re-heal, 3 training seeds planned)
- OLMo-2 7B + Qwen3-8B cross-family point already done (P2.3)
- The content-MMLU vs letter-MMLU split (P0.6) already shows a 3-way split: readout-binding / competence / fluency
- 40 cards of compute available

**Prior work to beat or dialogue with**:

1. **"The Unreasonable Ineffectiveness of the Deeper Layers"** — Gromov et al., 2024 (preprint, later appeared as ICML 2024 spotlight)  
   ArXiv: `2403.17887` (2024-03-26) [VERIFIED via arxiv API title search]  
   Method: Prune a contiguous block of layers chosen by layer-similarity, then QLoRA heal for 100–200 steps. Key claim: even after removing many deep layers, performance on QA benchmarks recovers quickly.  
   Critical gap: They only measure aggregate QA accuracy, not the closed-book factual recall vs. letter-selection split. They do not study the healing trajectory length or what fails to recover.

2. **"ShortGPT: Layers in Large Language Models are More Redundant Than You Expect"** — Men et al., 2024 (preprint)  
   ArXiv: `2403.03853` (2024-03-06) [VERIFIED via arxiv API title search]  
   Method: Block Influence (BI) metric; drops layers with low BI; finds 25% layer removal causes minimal PPL increase.  
   Critical gap: No fine-grained knowledge eval; no trajectory of recovery; no dissociation finding.

3. **"Shortened LLaMA: Depth Pruning for LLMs with Comparison of Retraining Methods"** — Kim et al., EMNLP 2024  
   ArXiv: `2402.02834` (2024-02-05) [VERIFIED]  
   Method: Systematic comparison of depth vs. width pruning; different retraining recipes on Llama-2.  
   Critical gap: Focuses on efficiency trade-offs, not the epistemological question of which capabilities recover at different rates.

4. **"The Curse of Depth in Large Language Models"** — Sun et al., 2025 (preprint)  
   ArXiv: `2502.05795` (2025-02-09) [VERIFIED]  
   Method: Theoretically and empirically shows that deep layers in modern LLMs (Llama, Mistral, DeepSeek, Qwen) are systematically less effective.  
   Critical gap: Does not study factual knowledge recovery after pruning+finetuning.

5. **"Scaling Laws for Downstream Task Performance of Large Language Models"** — Isik et al., 2024 (preprint)  
   ArXiv: `2402.04177` (2024-02-06) [VERIFIED]  
   Method: Scaling law framing for pretrain-then-finetune; models pretraining loss vs. downstream task performance.  
   Critical gap: Studies upstream (pretrained) models, not post-pruning recovery dynamics.

**Maximum kill risk**: If another group publishes the 3-way (PPL / factual-recall / layer-count) dissociation finding with multiple model families before submission, the novelty is gone. As of 2026-08, no paper in the arxiv corpus I found explicitly reports "perplexity heals while factual knowledge lags" as the main finding with quantitative dissociation curves.

**Needed benchmark**: TriviaQA + NQ-open + PopQA (already have), plus MMLU with the letter/content split (already have). All chat_template=False. The key metric is the **recovery lag** — step at which PPL returns to baseline vs. step at which factual-recall returns. No new benchmark needed; existing data is sufficient.

---

### Direction D2: Post-Norm vs Pre-Norm Representations as a Natural Intervention for CKA Geometry

**One-sentence pitch**: OLMo-2-7B is the single hardest-to-align model in the 14-model CKA corpus (midband 0.329 vs. next-lowest ~0.38) specifically because it uses post-layer-norm; use this as a natural experiment to test whether normalizer design — not family, not depth — is the dominant driver of cross-model representation compatibility.

**Existing assets**:
- 91-pair CKA matrix already computed; OLMo-2's outlier status established (R4 finding)
- 14 models × 7 families already run; can add 1–2 post-norm ablations cheaply (e.g., convert OLMo-2's pre-norm to post-norm via weight surgery, or compare against a custom-trained post-norm vs. pre-norm twin)
- `repr_alignment_multimodel.py` is fully instrumented; adding a new model requires one GPU job

**Prior work**:

1. **"The Platonic Representation Hypothesis"** — Huh, Cheung, Wang, Isola, NeurIPS 2024  
   ArXiv: `2405.07987` (2024-05-13) [VERIFIED via arxiv API]  
   Method: Shows that larger models tend to align more across modalities/architectures; proposes that models converge to a common statistical model of reality.  
   Critical gap: Does not study what architectural choices (pre-norm vs. post-norm) break or preserve this convergence.

2. **"Revisiting the Platonic Representation Hypothesis: An Aristotelian View"** — Groger, Wen, Brbic, 2026 (preprint)  
   ArXiv: `2602.14486` (2026-02-16) [VERIFIED]  
   Method: Shows existing metrics (including CKA) are confounded by network scale; introduces permutation-based null calibration.  
   Critical connection: Our R4 finding independently rediscovered the same null-calibration concern (shuffle-null = 0.453 vs. observed 0.491). This paper appears 6 months before our R4 work, so **it is a direct prior that occupies the same methodological niche**. Our R4 work only adds scale (91 pairs vs. fewer) and the normalization-architecture confound — but the core null-calibration critique is already made.  
   **Kill risk for Direction D2**: If the normalization effect is simply the "scale confound" Groger et al. already attribute to depth/width, then our OLMo-2 outlier is not novel.

3. **"Reliability of CKA as a Similarity Measure in Deep Learning"** — Davari et al., NeurIPS 2022  
   ArXiv: `2210.16156` (2022-10-28) [VERIFIED]  
   Method: Critiques linear CKA and proposes improvements; shows CKA can be misleading when representations have different norms.  
   Key finding: Our R4 z-scoring is partly a response to this known problem.

4. **"Contrastive-Difference CKA Reveals Concept-Specific Structural Alignment Across Language Model Architectures"** — Gao, 2026 (preprint)  
   ArXiv: `2606.16897` (2026-06-15) [VERIFIED via arxiv fetch]  
   Method: CKA_Delta — computes kernel alignment on per-sample contrastive differences to isolate concept-specific convergence from global geometry similarity. Finds "moderate geometric convergence coexists with near-perfect functional transfer" across multiple LLM architecture families.  
   **Direct collision**: This paper already does cross-architecture CKA and finds the key finding we were building toward. Published June 2026.

**Maximum kill risk**: Direction D2 is substantially occupied by arXiv:2606.16897 (Gao 2026), which is the most direct competitor. That paper explicitly studies cross-architecture CKA for LLMs. Our OLMo-2 post-norm angle is narrower and not directly addressed there, but "normalizer design shapes cross-model alignment" is a small enough claim that it probably cannot sustain a full paper alone.

**Benchmark**: WikiText-103 (already have). No new benchmark needed — the question is architectural, not about a new capability.

---

### Direction D3: Factual Knowledge Onset Depth as a Structured Probe

**One-sentence pitch**: Use logit-lens and per-layer edge-probe evidence across multiple model families to characterize *where* factual recall first "crystallizes" in depth, whether this onset is consistent across model families, and whether this onset depth correctly predicts which layers are dispensable for both PPL and factual tasks.

**Existing assets**:
- Paper B already has logit-lens knowledge onset for OLMo-2 (layers 18–19) and Qwen3-8B (layer 24)
- Edge-probe harness (POS/DEPREL/CoLA/WiC/SST2/RTE) already implemented
- 14 models already extracted; no new model runs needed for analysis
- Connection to Paper B's healing curves: onset depth (18/32 = 56%) correlates with the "safe keep-front" depth

**Prior work**:

1. **"Neuron-Level Knowledge Attribution in Large Language Models"** — Yu and Ananiadou, AAAI 2024  
   ArXiv: `2312.12141` (2023-12-19) [VERIFIED]  
   Method: Static neuron-level attribution for knowledge. Finds specific neurons encode factual associations.  
   Key gap: Neuron-level, not layer-level; does not study onset depth or its relationship to structural pruning.

2. **"How do language models learn facts? Dynamics, curricula and hallucinations"** — Zucchet et al., 2025 (preprint)  
   ArXiv: `2503.21676` (2025-03-27) [VERIFIED]  
   Method: Studies the three-phase dynamics of factual knowledge acquisition during training; attention circuits form in the plateau before factual recall crystallizes.  
   Key gap: Training dynamics, not layer depth; does not study which depth onset should be kept for structural pruning.

3. **"The Unreasonable Ineffectiveness of the Deeper Layers"** — Gromov et al., 2024 [VERIFIED above]  
   Key gap: Their "optimal block to prune" is chosen by CKA layer similarity, not by factual onset.

4. **"ShortGPT"** — Men et al. [VERIFIED above]  
   Key gap: Uses Block Influence (BI), not knowledge-onset or logit-lens.

**Maximum kill risk**: If logit-lens knowledge onset depth is already systematically studied as a pruning criterion in a paper I missed, this direction is dead. Given what I found, that work does not exist yet as of mid-2026. The key claim — "onset depth (the layer where factual recall first activates) is the principled criterion for determining keep-front depth, and this criterion generalizes across families" — is novel.

**Benchmark**: MMLU + TriviaQA (already have for OLMo-2; extend to Qwen3-8B with P2.5 harness). chat_template=False mandatory.

---

### Direction D4: Cross-Family PPL-Knowledge Dissociation Scaling: When Does Healing Close the Gap?

**One-sentence pitch**: Characterize the token budget (not just step budget) at which PPL recovery and factual-knowledge recovery converge or diverge across model sizes and families, and establish whether there is a power-law relationship between inherited depth fraction and required healing budget.

**Existing assets**:
- OLMo-2 keep8/10/12/14 full 200k-step trajectories; Qwen3-8B f12k2 200k trajectory
- 200k token budget = 409.6B tokens (OLMo-2 effective batch 128 × 2048 × 200k)
- Paper B's healing trajectory curves by step; conversion to token budget is trivial
- Pythia suite (publicly available checkpoints every 512 steps) could be used for external validation

**Prior work**:

1. **"Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling"** — Biderman et al., ICML 2023  
   ArXiv: `2304.01373` (2023-04-03) [VERIFIED]  
   Method: 16 LLMs from 70M to 12B, all on The Pile, 154 checkpoints each. Enables training-dynamics analysis.  
   Key gap: Studies the original pretraining dynamics, not post-pruning recovery. Our question is about how much continued-pretraining token budget is needed to restore factual knowledge after structural damage.

2. **"Shortened LLaMA"** — Kim et al. [VERIFIED above]  
   Key gap: Compares retraining methods by final performance, not by trajectory length to recovery.

3. **"ShortGPT"** — Men et al. [VERIFIED above]; no trajectory analysis.

4. **"The Unreasonable Ineffectiveness"** — Gromov et al. [VERIFIED above]; heals with tiny QLoRA budget (100–200 steps), which is far shorter than our 200k step full-param healing.

**Maximum kill risk**: If someone publishes a "token-budget-to-recovery" scaling law for depth-pruned LLMs before our submission, this direction dies. Current coverage is sparse, so this is low risk in the short term.

**Benchmark**: Same as Paper B (in-domain PPL + TriviaQA + MMLU) — keep consistent for direct comparison. No new benchmark needed; the claim is about the trajectory, not a new evaluation protocol.

---

### Direction D5: CKA U-Shape as a Measurement Study (Workshop/Short Paper)

**One-sentence pitch**: Across 91 model pairs spanning 7 architecture families, the mid-network residual stream is systematically the least geometrically alignable region — the layer band that has traditionally been the target of model stitching and capability transfer is the one where cross-model representations collapse furthest from each other.

**Existing assets**:
- 91-pair CKA matrix already computed (R4, verified by MAIN)
- All statistical tests run; distance-control argument verified
- Result is clean: 72/91 U-shaped, binom p=2e-8, confirmed by residual-quadratic test
- Manuscript-ready: methodology section, all tables, hardcoded assertions passing

**Prior work** (most relevant; this is the key question for Q3):

1. **"Reliability of CKA as a Similarity Measure in Deep Learning"** — Davari et al., NeurIPS 2022 [VERIFIED above]  
   They study CKA as a layer-comparison metric within models and between similar-architecture models. They do NOT report the cross-family U-shape as a cross-model depth pattern.

2. **"Revisiting the Platonic Representation Hypothesis"** — Groger et al. [VERIFIED above]  
   They study scale confounds in representational similarity. Closest to our null-calibration concern. They do NOT report a U-shaped depth profile across 91 cross-family pairs.

3. **"Contrastive-Difference CKA"** — Gao 2026 [VERIFIED above]  
   Studies cross-architecture concept-specific alignment. Does NOT report U-shape along the depth diagonal.

4. **"The Platonic Representation Hypothesis"** — Huh et al. [VERIFIED above]  
   Argues models converge globally; uses mean CKA, not depth-resolved profile. Does NOT report U-shape.

**Assessment for Q3**: None of the papers I could independently verify reports the cross-model depth-resolved U-shaped CKA profile (mid-depth collapse) across 91 pairs with the distance-controlled statistical design. This result appears to be novel as of mid-2026. However, as MAIN already found, the shuffle-layer-order null (0.453) is only 0.038 below the observed mean (0.491), which substantially weakens the "meaningful layer correspondence" interpretation. A short paper could publish the U-shape measurement under the honest framing: "the mid-depth band has the worst layer-alignment, and most of that absolute alignment signal is not layer-correspondence information — this is a fundamental obstacle for model stitching."

**Maximum kill risk**: A paper showing that CKA depth profiles are U-shaped in transformers (intra-model or cross-model) would kill this. I did not find such a paper in 2023–2026. The specific contribution is the **cross-model** U-shape across 91 pairs with distance control and shuffle-null calibration.

---

## Q2: Scoring Matrix

| Direction | Novelty Risk (low=good) | Feasibility w/ Assets | Effect-Size Prior | Top-Venue Probability |
|---|:---:|:---:|:---:|:---:|
| D1: PPL-knowledge 3-way dissociation | **2** | **5** | **5** | **4** |
| D2: Post-norm CKA geometry | **4** | 4 | 3 | 2 |
| D3: Knowledge onset depth as pruning criterion | **2** | **5** | 4 | 3 |
| D4: Token-budget-to-recovery scaling | **2** | 4 | 3 | 3 |
| D5: CKA U-shape measurement (short) | **3** | **5** | 3 | 2 |

Scores: 0=worst, 5=best. Novelty risk: 5 = highly novel (low risk), 0 = already claimed.

**Top-1: Direction D1 (PPL-Knowledge Dissociation)**

D1 has the highest combination of novelty (no paper I found makes the 3-way dissociation its headline), feasibility (Paper B provides 95% of the experimental apparatus), and effect size (keep14@200k MMLU only 19.5% recovery while PPL is at 1.428× baseline is a striking, real gap). The finding also has genuine theoretical interest: it challenges the assumption that perplexity is a sufficient proxy for model capability.

The key claim — "PPL recovers within ~50k steps but factual knowledge remains persistently degraded even at 200k steps across multiple depth variants and model families" — is falsifiable, clean, and has concrete policy implications for practitioners who use PPL to decide when a fine-tuned/pruned model is "ready."

Caution: The current Paper B framing already leans this direction. If Paper B is accepted at ACL main track, D1 becomes a natural follow-up / extension, not a separate first submission. D1 as a standalone paper would need to establish the scaling/cross-family aspect more systematically (multiple model sizes, multiple families, quantitative token-budget-to-recovery curves).

**Top-2: Direction D3 (Knowledge Onset Depth as Pruning Criterion)**

D3 bridges our CKA work (R4 finding: H1 shows mid-depth collapse) with Paper B's healing work. The specific claim is mechanistic: the logit-lens onset depth (where factual recall first crystallizes) is the principled "safe" depth to prune down to. Paper B's success at keep14 (18th layer out of 32, or 56%) may not be coincidental — the OLMo-2 logit-lens shows knowledge onset at layers 18-19.

This direction is novel: none of the prior work I verified proposes logit-lens onset as a principled pruning criterion, and no paper systematically compares onset-based vs. CKA-similarity-based vs. layer-importance-based criteria on the same factual benchmarks. The cost is low (logit-lens already instrumented), and the effect size is high (onset-guided keep-front should outperform BI/CKA-guided keep-front on factual tasks).

---

## Q3: Can R4 H1 (U-Shape CKA) Stand Alone as a Measurement Paper?

**Short answer: Marginal. It could sustain a findings note or workshop paper, but not a stand-alone long paper at a top venue.**

**The three most relevant prior works on CKA depth profiles / cross-model alignment**:

1. **"Reliability of CKA as a Similarity Measure in Deep Learning"** — Davari et al., NeurIPS 2022  
   ArXiv: `2210.16156` (2022-10-28) [VERIFIED via arxiv API search and fetch]  
   They study within-model layer comparison and cross-model comparison for architecturally similar networks. They do not report cross-family depth-resolved profiles, and they do not report a U-shape. Their finding is that CKA can be misleading when representation norms differ — which is exactly why R4 uses z-scoring.  
   Our marginal contribution over them: 91 cross-family pairs with null calibration, documenting the mid-depth collapse.

2. **"Revisiting the Platonic Representation Hypothesis: An Aristotelian View"** — Groger, Wen, Brbic, 2026 (preprint)  
   ArXiv: `2602.14486` (2026-02-16) [VERIFIED]  
   They show that CKA and related metrics are inflated by network scale, and propose permutation-based null calibration. This is nearly identical in spirit to R4's layer-order-shuffle null (0.453 vs. 0.491), which shows most midband CKA is not layer-correspondence signal.  
   **This is the closest competitor for H1's null-calibration finding.** Our shuffle-null is operationally equivalent to their permutation-based null. They predate R4 by ~6 months. Our marginal contribution: the null-calibration applied specifically to the cross-model depth diagonal (not just block-mean CKA), and the U-shape observation as the consequence.

3. **"Contrastive-Difference CKA Reveals Concept-Specific Structural Alignment Across Language Model Architectures"** — Gao, 2026 (preprint)  
   ArXiv: `2606.16897` (2026-06-15) [VERIFIED]  
   Studies cross-architecture CKA for LLMs using a concept-contrastive variant. Finds moderate geometric convergence but near-perfect functional transfer. The "geometric convergence coexists with functional transfer" finding is weaker (higher level) than our specific depth-resolved U-shape, but is directly in the same subfield.  
   Our marginal contribution: the specific depth profile along the relative-depth diagonal (not block-mean), and the statistical argument against layer-correspondence interpretation.

**Assessment**: The U-shape finding is novel in its specificity (91 cross-family pairs, depth-diagonal, distance-controlled, shuffle-null calibrated) and has a clear take-home message for model stitching. It could appear as a **findings note (4 pages) at ACL/EMNLP/NAACL 2026** or as a **workshop paper** (e.g., BlackboxNLP or Representation Learning for NLP). A 8-page long paper at ICLR/NeurIPS would require a stronger theoretical or practical contribution — for example, showing that the U-shape predicts which stitch points succeed or fail, which would require running actual model stitching experiments.

**The minimum it needs to be publishable as a standalone**: (a) Confirm the U-shape with a simple, cheap functional experiment — e.g., train a 1-layer linear bridge at each depth point and show the reconstruction loss follows the U-shape exactly. If the functional bridge cost tracks the CKA U-shape, the paper has a clean claim: "geometric alignment predicts functional bridging cost, and both are U-shaped across depth." (b) A clear negative implication: "attempts to stitch at mid-depth will require the largest bridge."

---

## Q4: Paper B Prune-Heal as Adaptation, Not Compression

**Can it leverage into a new direction?**

Yes. The framing shift is: Paper B documents that **structural pruning is a controlled damage**, and the healing process is a natural experiment in **capability re-learning under architectural constraint**. The novel framing for a new direction is:

> "What does a language model re-learn first, and in what order, after architectural injury? And does this order depend on the nature of the injury (which layers removed) or the nature of the target capability (factual knowledge vs. syntactic vs. reasoning)?"

This is a measurement science paper about **learning dynamics under reconstruction constraint**, not about compression efficiency.

**How to distinguish from Paper B**:
- Paper B: "PPL heals but knowledge lags — is this a problem for practitioners who trust PPL?"
- New direction: "The lag pattern reveals the order in which capabilities are re-acquired. Factual recall lags behind syntactic/surface capabilities. Is this order fixed across architectures and layer-removal policies?"
- The new direction requires: multiple healing trajectories across different removal patterns (keep-front vs. keep-back vs. ShortGPT vs. random selection), with per-capability (not per-benchmark) trajectory curves.

**Most novel extension axes** (ranked by cost/novelty ratio):

1. **Multi-capability trajectory**: Add POS-tagging probe + CoLA acceptability probe as fine-grained trajectory markers alongside factual recall. Already have edge-probe harness. Claims: "Syntactic competence heals within 10k steps; factual recall needs 50k–100k; multi-step reasoning never recovers within 200k." This is directly testable with existing infrastructure.

2. **Cross-family comparison**: Qwen3-8B f12k2 is already at 200k. Does the PPL-knowledge gap close at the same token budget in Qwen as in OLMo-2? If the gap is fixed in token-budget across families, that is a universal law. If it scales with model size, that is a different kind of law. Either is publishable.

3. **Severity gradient**: Compare keep8 / keep10 / keep12 / keep14 trajectories as a severity gradient. The current data shows keep8 never recovers MMLU to above-chance at 200k, while keep14 reaches 19.5% recovery. Where is the phase transition? This is a clean scaling-law question with existing data.

**Adjacent papers to dialogue with**:

- **"Shortened LLaMA"** [VERIFIED above]: They compare depth vs. width pruning by final performance. We would add trajectory analysis and capability-specific recovery.
- **"How do language models learn facts?"** — Zucchet et al. 2025 [VERIFIED above]: Studies training-dynamics of factual recall from scratch. We would study re-learning after structural damage — a different but closely related question.
- **"Pythia"** [VERIFIED above]: Provides training-dynamics checkpoints for reference. We can compare our healing trajectories against Pythia's original pretraining trajectories.

**Separation from Paper B** in venue framing: Paper B is an empirical observation paper ("here is the dissociation"). The new direction is a mechanistic/measurement paper ("here is the order in which capabilities re-emerge, and this order is predictable from architecture"). They are complementary but distinct enough to not self-plagiarize if the new paper explicitly compares across removal policies and capabilities, not just across depth rungs.

---

## Summary Recommendation

The two strongest candidate directions coming out of this analysis are:

**Immediate (D1 extension of Paper B)**: Expand the PPL-knowledge-dissociation finding into a cross-family, multi-capability, token-budget-normalized measurement paper. The existing data (OLMo-2 + Qwen3-8B, 4 depth rungs, 3 structural policies, 5 benchmarks) supports the first version of this paper with no new GPU runs beyond what Paper B has already planned (Qwen P2.5, OLMo-2 multi-seed). The key claim to add: "The dissociation is universal across model families and follows a power law in inherited depth fraction."

**Medium-term (D3, logit-lens onset as pruning criterion)**: The mechanistic link between knowledge onset depth and safe pruning depth is novel, directly testable with existing infrastructure, and would unite the CKA findings (Paper D asset) with Paper B's healing trajectories. A 6-month project.

**For the CKA U-shape finding (H1 only)**: Target a workshop or findings note, not a long paper. Add one functional experiment (linear bridge reconstruction cost vs. depth) to turn the geometric observation into a practical cost estimate for model stitching.

**Direction D2 is effectively blocked** by Gao 2026 (arXiv:2606.16897) on the functional side, and by Groger et al. 2026 (arXiv:2602.14486) on the methodological side. Pursuing D2 would require a non-trivial new angle (e.g., a causal intervention — train OLMo-2 variants with vs. without post-norm and measure CKA — which requires GPU budget we may not want to spend on a potentially blocked direction).

---

*All arxiv fetch timestamps: 2026-08-06 UTC. Proxy: hy-proxy.woa.com:3128. Citations verified independently by fetching arxiv.org/abs/<id> or using the export.arxiv.org API query endpoint.*
